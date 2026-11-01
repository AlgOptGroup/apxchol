#include "crc32c.h"
#include <array>
#include <bit>
#include <cstring>

#if defined(__SSE4_2__)
#include <nmmintrin.h>
#elif defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
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

// x^(8 * bytes) mod P(x), by repeated squaring.
constexpr std::uint32_t shift_operator(std::uint64_t bytes) noexcept {
    std::uint32_t result = 1u << 31, square = 1u << 23;  // x^0 and x^8
    for (; bytes; bytes >>= 1) {
        if (bytes & 1) result = multiply(square, result);
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

// One crc instruction has a latency of about three cycles but can issue every
// cycle, so three independent lanes over adjacent `lane` byte ranges keep it
// busy. Their registers merge by shifting the earlier ones past the later bytes.
constexpr std::size_t lane = 8192;

// shift[j][b] = (b << 8j) * x^(8 * bytes): the shift as four byte lookups.
using shift_tables = std::array<std::array<std::uint32_t, 256>, 4>;
constexpr shift_tables make_shift_tables(std::uint64_t bytes) {
    const std::uint32_t op = shift_operator(bytes);
    shift_tables t{};
    for (std::size_t j = 0; j < 4; ++j)
        for (std::uint32_t b = 0; b < 256; ++b) t[j][b] = multiply(op, b << (8 * j));
    return t;
}
constexpr shift_tables shift_one = make_shift_tables(lane), shift_two = make_shift_tables(2 * lane);

inline std::uint32_t shift(const shift_tables& t, std::uint32_t c) noexcept {
    return t[0][c & 0xff] ^ t[1][(c >> 8) & 0xff] ^ t[2][(c >> 16) & 0xff] ^ t[3][c >> 24];
}

std::uint32_t update_hardware(std::uint32_t c, const unsigned char* p, std::size_t size) noexcept {
    for (; size >= 3 * lane; p += 3 * lane, size -= 3 * lane) {
        std::uint32_t a = c, b = 0, d = 0;
        for (std::size_t i = 0; i < lane; i += 8) {
            std::uint64_t wa, wb, wd;
            std::memcpy(&wa, p + i, 8);
            std::memcpy(&wb, p + lane + i, 8);
            std::memcpy(&wd, p + 2 * lane + i, 8);
            a = step8(a, wa);
            b = step8(b, wb);
            d = step8(d, wd);
        }
        c = shift(shift_two, a) ^ shift(shift_one, b) ^ d;
    }
    for (; size >= 8; p += 8, size -= 8) {
        std::uint64_t w;
        std::memcpy(&w, p, 8);
        c = step8(c, w);
    }
    for (; size; ++p, --size) c = step1(c, *p);
    return c;
}
#endif

} // namespace

std::uint32_t crc32c_portable(std::uint32_t crc, const void* data, std::size_t size) noexcept {
    return ~update_portable(~crc, static_cast<const unsigned char*>(data), size);
}

std::uint32_t crc32c(std::uint32_t crc, const void* data, std::size_t size) noexcept {
#if defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
    return ~update_hardware(~crc, static_cast<const unsigned char*>(data), size);
#else
    return crc32c_portable(crc, data, size);
#endif
}

std::uint32_t crc32c_combine(std::uint32_t crc_a, std::uint32_t crc_b, std::uint64_t size_b) noexcept {
    return multiply(shift_operator(size_b), crc_a) ^ crc_b;
}

} // namespace apxchol::detail
