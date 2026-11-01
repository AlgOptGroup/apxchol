#include "crc32c.h"
#include <array>
#include <bit>
#include <cstring>
#include <span>
#include <vector>

#if defined(__SSE4_2__)
#include <nmmintrin.h>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <wmmintrin.h>
#define APXCHOL_CRC32C_FOLD 1
// PCLMULQDQ is not part of the x86-64-v2 baseline that portable builds target,
// so the folding kernel is compiled for it and chosen at run time.
#define APXCHOL_CRC32C_FOLD_TARGET __attribute__((target("sse4.2,pclmul")))
#endif
#elif defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#if defined(__ARM_FEATURE_AES) && defined(__aarch64__)
#include <arm_neon.h>
#define APXCHOL_CRC32C_FOLD 1
#define APXCHOL_CRC32C_FOLD_TARGET
#endif
#endif

namespace apxchol::detail {
namespace {

constexpr std::uint32_t polynomial = 0x82F63B78u;  // reflected

// tables[s][b]: the register update for byte b followed by s zero bytes.
using byte_tables = std::array<std::array<std::uint32_t, 256>, 8>;
constexpr byte_tables make_byte_tables() {
    byte_tables t{};
    for (std::uint32_t b = 0; b < 256; ++b) {
        std::uint32_t c = b;
        for (int k = 0; k < 8; ++k) c = c & 1 ? (c >> 1) ^ polynomial : c >> 1;
        t[0][b] = c;
    }
    for (std::size_t s = 1; s < 8; ++s)
        for (std::size_t b = 0; b < 256; ++b)
            t[s][b] = (t[s - 1][b] >> 8) ^ t[0][t[s - 1][b] & 0xff];
    return t;
}
constexpr byte_tables tables = make_byte_tables();

// Advances the raw register (no pre- or post-inversion) over `size` bytes.
std::uint32_t update_portable(std::uint32_t c, const unsigned char* p, std::size_t size) noexcept {
    if constexpr (std::endian::native == std::endian::little) {
        for (; size >= 8; p += 8, size -= 8) {
            std::uint64_t w;
            std::memcpy(&w, p, 8);
            w ^= c;
            c = tables[7][w & 0xff] ^ tables[6][(w >> 8) & 0xff] ^ tables[5][(w >> 16) & 0xff] ^
                tables[4][(w >> 24) & 0xff] ^ tables[3][(w >> 32) & 0xff] ^
                tables[2][(w >> 40) & 0xff] ^ tables[1][(w >> 48) & 0xff] ^ tables[0][w >> 56];
        }
    }
    for (; size; ++p, --size) c = tables[0][(c ^ *p) & 0xff] ^ (c >> 8);
    return c;
}

// a(x) * b(x) mod P(x) in the reflected representation, where 1u << 31 is x^0.
constexpr std::uint32_t multiply(std::uint32_t a, std::uint32_t b) noexcept {
    std::uint32_t product = 0;
    for (std::uint32_t m = 1u << 31; m; m >>= 1) {
        if (a & m) product ^= b;
        b = b & 1 ? (b >> 1) ^ polynomial : b >> 1;
    }
    return product;
}

// x^n mod P(x), by repeated squaring.
constexpr std::uint32_t power(std::uint64_t n) noexcept {
    std::uint32_t result = 1u << 31, square = 1u << 30;  // x^0 and x^1
    for (; n; n >>= 1) {
        if (n & 1) result = multiply(square, result);
        square = multiply(square, square);
    }
    return result;
}

#if defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
#if defined(__SSE4_2__)
inline std::uint32_t step8(std::uint32_t c, std::uint64_t w) noexcept {
    return static_cast<std::uint32_t>(_mm_crc32_u64(c, w));
}
inline std::uint32_t step1(std::uint32_t c, unsigned char b) noexcept { return _mm_crc32_u8(c, b); }
#else
inline std::uint32_t step8(std::uint32_t c, std::uint64_t w) noexcept { return __crc32cd(c, w); }
inline std::uint32_t step1(std::uint32_t c, unsigned char b) noexcept { return __crc32cb(c, b); }
#endif

inline std::uint64_t load8(const unsigned char* p) noexcept {
    std::uint64_t w;
    std::memcpy(&w, p, 8);
    return w;
}

// One crc instruction has a latency of about three cycles but can issue every
// cycle, so three independent lanes over adjacent ranges keep it busy. Their
// registers merge by shifting the earlier ones past the later bytes. Long lanes
// cover most of a buffer; short ones leave at most 767 bytes to a single lane.
constexpr std::size_t long_lane = 8192, short_lane = 256;

// shift[j][b] = (b << 8j) * x^(8 * bytes): the shift as four byte lookups.
using shift_tables = std::array<std::array<std::uint32_t, 256>, 4>;
constexpr shift_tables make_shift_tables(std::uint64_t bytes) {
    const std::uint32_t op = power(8 * bytes);
    shift_tables t{};
    for (std::size_t j = 0; j < 4; ++j)
        for (std::uint32_t b = 0; b < 256; ++b) t[j][b] = multiply(op, b << (8 * j));
    return t;
}
constexpr shift_tables long_one = make_shift_tables(long_lane), long_two = make_shift_tables(2 * long_lane);
constexpr shift_tables short_one = make_shift_tables(short_lane), short_two = make_shift_tables(2 * short_lane);

inline std::uint32_t shift(const shift_tables& t, std::uint32_t c) noexcept {
    return t[0][c & 0xff] ^ t[1][(c >> 8) & 0xff] ^ t[2][(c >> 16) & 0xff] ^ t[3][c >> 24];
}

// Runs three lanes of `lane` bytes while 3 * lane bytes remain.
std::uint32_t three_lanes(std::uint32_t c, const unsigned char*& p, std::size_t& size, std::size_t lane,
                          const shift_tables& one, const shift_tables& two) noexcept {
    for (; size >= 3 * lane; p += 3 * lane, size -= 3 * lane) {
        std::uint32_t a = c, b = 0, d = 0;
        for (std::size_t i = 0; i < lane; i += 8) {
            a = step8(a, load8(p + i));
            b = step8(b, load8(p + lane + i));
            d = step8(d, load8(p + 2 * lane + i));
        }
        c = shift(two, a) ^ shift(one, b) ^ d;
    }
    return c;
}

std::uint32_t update_lanes(std::uint32_t c, const unsigned char* p, std::size_t size) noexcept {
    c = three_lanes(c, p, size, long_lane, long_one, long_two);
    c = three_lanes(c, p, size, short_lane, short_one, short_two);
    for (; size >= 8; p += 8, size -= 8) c = step8(c, load8(p));
    for (; size; ++p, --size) c = step1(c, *p);
    return c;
}
#endif

#if defined(APXCHOL_CRC32C_FOLD)
// The hybrid kernel: carry-less multiplies fold 64 bytes per step in four
// 128-bit accumulators while three crc instruction lanes take 24 bytes each,
// so both units work at once. A 128-bit lane loaded from memory holds bytes in
// reflected order, so a 64-bit half h stands for h(x) and the multiply of h by
// a constant r placed in the low 32 bits of a half yields h(x) r(x) x^33.
// Folding a block by d bytes therefore uses r = x^(8d + 31) for the low half
// and x^(8d - 33) for the high half, and shifting a register c by n bytes is
// one multiply by x^(8n - 33) followed by one crc step.
#if defined(__SSE4_2__)
using v128 = __m128i;
APXCHOL_CRC32C_FOLD_TARGET inline v128 load16(const unsigned char* p) noexcept {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
}
APXCHOL_CRC32C_FOLD_TARGET inline v128 constants(std::uint32_t low, std::uint32_t high) noexcept {
    return _mm_setr_epi32(static_cast<int>(low), 0, static_cast<int>(high), 0);
}
// a.low * k.low ^ a.high * k.high
APXCHOL_CRC32C_FOLD_TARGET inline v128 fold(v128 a, v128 k) noexcept {
    return _mm_xor_si128(_mm_clmulepi64_si128(a, k, 0x00), _mm_clmulepi64_si128(a, k, 0x11));
}
APXCHOL_CRC32C_FOLD_TARGET inline v128 vxor(v128 a, v128 b) noexcept { return _mm_xor_si128(a, b); }
APXCHOL_CRC32C_FOLD_TARGET inline std::uint64_t half(v128 a, int i) noexcept {
    return i ? static_cast<std::uint64_t>(_mm_extract_epi64(a, 1)) : static_cast<std::uint64_t>(_mm_cvtsi128_si64(a));
}
APXCHOL_CRC32C_FOLD_TARGET inline std::uint64_t multiply64(std::uint32_t a, std::uint32_t b) noexcept {
    return static_cast<std::uint64_t>(_mm_cvtsi128_si64(
        _mm_clmulepi64_si128(_mm_cvtsi32_si128(static_cast<int>(a)), _mm_cvtsi32_si128(static_cast<int>(b)), 0)));
}
APXCHOL_CRC32C_FOLD_TARGET inline std::uint32_t fold_step8(std::uint32_t c, std::uint64_t w) noexcept {
    return static_cast<std::uint32_t>(_mm_crc32_u64(c, w));
}
#else
using v128 = uint64x2_t;
inline v128 load16(const unsigned char* p) noexcept { return vreinterpretq_u64_u8(vld1q_u8(p)); }
inline v128 constants(std::uint32_t low, std::uint32_t high) noexcept {
    return vcombine_u64(vcreate_u64(low), vcreate_u64(high));
}
inline v128 fold(v128 a, v128 k) noexcept {
    const v128 low = vreinterpretq_u64_p128(vmull_p64(vgetq_lane_u64(a, 0), vgetq_lane_u64(k, 0)));
    const v128 high = vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(a), vreinterpretq_p64_u64(k)));
    return veorq_u64(low, high);
}
inline v128 vxor(v128 a, v128 b) noexcept { return veorq_u64(a, b); }
inline std::uint64_t half(v128 a, int i) noexcept { return i ? vgetq_lane_u64(a, 1) : vgetq_lane_u64(a, 0); }
inline std::uint64_t multiply64(std::uint32_t a, std::uint32_t b) noexcept {
    return vgetq_lane_u64(vreinterpretq_u64_p128(vmull_p64(a, b)), 0);
}
inline std::uint32_t fold_step8(std::uint32_t c, std::uint64_t w) noexcept { return __crc32cd(c, w); }
#endif

constexpr std::size_t vector_bytes = 64, lane_bytes = 24, step_bytes = vector_bytes + 3 * lane_bytes;
constexpr std::uint32_t fold64_low = power(8 * 64 + 31), fold64_high = power(8 * 64 - 33);
constexpr std::uint32_t fold16_low = power(8 * 16 + 31), fold16_high = power(8 * 16 - 33);

APXCHOL_CRC32C_FOLD_TARGET inline std::uint32_t shift_by(std::uint32_t c, std::uint32_t k) noexcept {
    return fold_step8(0, multiply64(c, k));
}

APXCHOL_CRC32C_FOLD_TARGET std::uint32_t update_fold(std::uint32_t c, const unsigned char* p,
                                                     std::size_t size) noexcept {
    const std::size_t steps = size / step_bytes;
    if (steps >= 4) {
        // Layout: lanes a, b, d of 24 * steps bytes each, then the vector part.
        const std::size_t lane = lane_bytes * steps, vector = vector_bytes * steps;
        // The shifts depend only on `steps`; the archive reader calls with one size.
        thread_local std::size_t cached_steps = 0;
        thread_local std::uint32_t shift_a, shift_b, shift_d;
        if (cached_steps != steps) {
            shift_a = power(8 * (2 * lane + vector) - 33);
            shift_b = power(8 * (lane + vector) - 33);
            shift_d = power(8 * vector - 33);
            cached_steps = steps;
        }
        const unsigned char *a = p, *b = p + lane, *d = p + 2 * lane, *v = p + 3 * lane;
        std::uint32_t ca = c, cb = 0, cd = 0;
        v128 x0 = load16(v), x1 = load16(v + 16), x2 = load16(v + 32), x3 = load16(v + 48);
        const v128 k64 = constants(fold64_low, fold64_high);
        for (std::size_t s = 0; s < steps; ++s) {
            for (std::size_t i = 0; i < lane_bytes; i += 8) {
                ca = fold_step8(ca, load8(a + i));
                cb = fold_step8(cb, load8(b + i));
                cd = fold_step8(cd, load8(d + i));
            }
            a += lane_bytes; b += lane_bytes; d += lane_bytes;
            if (s + 1 < steps) {
                v += vector_bytes;
                x0 = vxor(fold(x0, k64), load16(v));
                x1 = vxor(fold(x1, k64), load16(v + 16));
                x2 = vxor(fold(x2, k64), load16(v + 32));
                x3 = vxor(fold(x3, k64), load16(v + 48));
            }
        }
        const v128 k16 = constants(fold16_low, fold16_high);
        v128 x = vxor(fold(x0, k16), x1);
        x = vxor(fold(x, k16), x2);
        x = vxor(fold(x, k16), x3);
        const std::uint32_t cv = fold_step8(fold_step8(0, half(x, 0)), half(x, 1));
        c = shift_by(ca, shift_a) ^ shift_by(cb, shift_b) ^ shift_by(cd, shift_d) ^ cv;
        p += steps * step_bytes;
        size -= steps * step_bytes;
    }
    return update_lanes(c, p, size);
}

bool fold_supported() noexcept {
#if defined(__SSE4_2__)
    static const bool supported = [] {
        __builtin_cpu_init();
        return __builtin_cpu_supports("pclmul") != 0;
    }();
    return supported;
#else
    return true;
#endif
}
#endif

} // namespace

std::uint32_t crc32c_portable(std::uint32_t crc, const void* data, std::size_t size) noexcept {
    return ~update_portable(~crc, static_cast<const unsigned char*>(data), size);
}

std::uint32_t crc32c(std::uint32_t crc, const void* data, std::size_t size) noexcept {
    const auto* p = static_cast<const unsigned char*>(data);
#if defined(APXCHOL_CRC32C_FOLD)
    if (fold_supported()) return ~update_fold(~crc, p, size);
#endif
#if defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
    return ~update_lanes(~crc, p, size);
#else
    return ~update_portable(~crc, p, size);
#endif
}

std::span<const crc32c_implementation> crc32c_implementations() noexcept {
    static const auto list = [] {
        std::vector<crc32c_implementation> all{{"portable", crc32c_portable}};
#if defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
        all.push_back({"crc instruction lanes", [](std::uint32_t crc, const void* data, std::size_t size) noexcept {
            return ~update_lanes(~crc, static_cast<const unsigned char*>(data), size);
        }});
#endif
#if defined(APXCHOL_CRC32C_FOLD)
        if (fold_supported())
            all.push_back({"crc lanes with carry-less folding", [](std::uint32_t crc, const void* data, std::size_t size) noexcept {
                return ~update_fold(~crc, static_cast<const unsigned char*>(data), size);
            }});
#endif
        return all;
    }();
    return list;
}

std::uint32_t crc32c_combine(std::uint32_t crc_a, std::uint32_t crc_b, std::uint64_t size_b) noexcept {
    return multiply(power(8 * size_b), crc_a) ^ crc_b;
}

} // namespace apxchol::detail
