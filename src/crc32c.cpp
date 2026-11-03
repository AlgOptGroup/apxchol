#include "crc32c.h"
#include <array>
#include <cstring>
#include <vector>

#if defined(__SSE4_2__) && defined(__x86_64__)
#include <nmmintrin.h>
#define APXCHOL_CRC32C_INSTRUCTION 1
#elif defined(__ARM_FEATURE_CRC32) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#include <arm_acle.h>
#define APXCHOL_CRC32C_INSTRUCTION 1
#endif

namespace apxchol::detail {
namespace {

constexpr std::uint32_t polynomial = 0x82F63B78u;  // reflected

constexpr std::array<std::uint32_t, 256> make_table() {
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t b = 0; b < 256; ++b) {
        std::uint32_t c = b;
        for (int k = 0; k < 8; ++k) c = c & 1 ? (c >> 1) ^ polynomial : c >> 1;
        t[b] = c;
    }
    return t;
}
constexpr auto table = make_table();

// Advances the raw register (no pre- or post-inversion) over `size` bytes.
std::uint32_t update_portable(std::uint32_t c, const unsigned char* p, std::size_t size) noexcept {
    for (; size; ++p, --size) c = table[(c ^ *p) & 0xff] ^ (c >> 8);
    return c;
}

#if defined(APXCHOL_CRC32C_INSTRUCTION)
// The crc instruction takes 8 bytes at a time in little-endian order.
std::uint32_t update_instruction(std::uint32_t c, const unsigned char* p, std::size_t size) noexcept {
    for (; size >= 8; p += 8, size -= 8) {
        std::uint64_t w;
        std::memcpy(&w, p, 8);
#if defined(__x86_64__)
        c = static_cast<std::uint32_t>(_mm_crc32_u64(c, w));
#else
        c = __crc32cd(c, w);
#endif
    }
    return update_portable(c, p, size);
}
#endif

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

} // namespace

std::uint32_t crc32c_portable(std::uint32_t crc, const void* data, std::size_t size) noexcept {
    return ~update_portable(~crc, static_cast<const unsigned char*>(data), size);
}

std::uint32_t crc32c(std::uint32_t crc, const void* data, std::size_t size) noexcept {
#if defined(APXCHOL_CRC32C_INSTRUCTION)
    return ~update_instruction(~crc, static_cast<const unsigned char*>(data), size);
#else
    return crc32c_portable(crc, data, size);
#endif
}

std::span<const crc32c_implementation> crc32c_implementations() noexcept {
    static const std::vector<crc32c_implementation> list{
        {"portable", crc32c_portable},
#if defined(APXCHOL_CRC32C_INSTRUCTION)
        {"crc instruction", crc32c},
#endif
    };
    return list;
}

std::uint32_t crc32c_combine(std::uint32_t crc_a, std::uint32_t crc_b, std::uint64_t size_b) noexcept {
    return multiply(power(8 * size_b), crc_a) ^ crc_b;
}

} // namespace apxchol::detail
