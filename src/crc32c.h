#pragma once
#include <cstddef>
#include <cstdint>
#include <span>

namespace apxchol::detail {

// CRC-32C (Castagnoli): reflected polynomial 0x82F63B78, initial value and
// final XOR 0xFFFFFFFF; "123456789" gives 0xE3069283. `crc` is the checksum of
// the preceding bytes (0 for none), and the result covers them and `data`.

/// The fastest implementation this build can run (see below).
std::uint32_t crc32c(std::uint32_t crc, const void* data, std::size_t size) noexcept;
/// Table-driven version, used where no crc instruction is available.
std::uint32_t crc32c_portable(std::uint32_t crc, const void* data, std::size_t size) noexcept;
/// Checksum of A followed by B from crc32c(A), crc32c(B) and B's length.
std::uint32_t crc32c_combine(std::uint32_t crc_a, std::uint32_t crc_b, std::uint64_t size_b) noexcept;

/// Every implementation that runs here: the table and, where available, the
/// crc instruction (SSE4.2 or the ARMv8 CRC extension), which crc32c uses.
/// Tests compare them all.
struct crc32c_implementation {
    const char* name;
    std::uint32_t (*checksum)(std::uint32_t crc, const void* data, std::size_t size) noexcept;
};
std::span<const crc32c_implementation> crc32c_implementations() noexcept;

} // namespace apxchol::detail
