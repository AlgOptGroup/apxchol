#pragma once
// bfloat16 storage type for the SpTRSV factor values (-DAPXCHOL_SPTRSV_BF16).
//
// bf16 is the top 16 bits of an IEEE-754 binary32: 1 sign, 8 exponent, 7
// explicit mantissa bits (8 with the hidden bit -> relative precision 2^-8
// per round-to-nearest, i.e. |x - bf16(x)| <= 2^-8 |x|; same dynamic range
// as fp32). It is a STORAGE format only: every read widens to fp32 -- a pure
// integer shift, exact -- and the arithmetic stays in double.
//
// Rounding on store is round-to-nearest-even on the fp32 bit pattern; the
// carry out of the mantissa into the exponent (and, for values near FLT_MAX,
// into infinity) is handled by plain integer addition. NaN/inf are not
// expected in a factor, but a NaN is kept a NaN (never rounded into inf).
// double narrows through fp32 first (double rounding -- immaterial at 8-bit
// precision, and it keeps "RNE on the fp32 pattern" the single definition).
//
// No intrinsics: the shift/memcpy form vectorizes fine and is portable
// (AVX512_BF16 would only matter for the store, which is setup-only).
#include <bit>
#include <cstdint>
#include <type_traits>

namespace apxchol {

struct bf16_t {
    // No default member initializer, deliberately: bf16_t stays trivially
    // default-constructible like float, so `new bf16_t[n]` is uninitialized
    // (the transpose's transient bucket relies on that) while
    // std::vector::resize still value-initializes to zero bits (== 0.0f).
    std::uint16_t bits;

    bf16_t() = default;

    // Implicit narrowing from any arithmetic type (float / double / ints):
    // this is what makes `sptrsv_value_t v = some_double;` -- the assembler's
    // factor writes -- Just Work when sptrsv_value_t is bf16_t, exactly like
    // the implicit double -> float narrowing of the fp32 build.
    template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
    constexpr bf16_t(T v) : bits(round_bits(std::bit_cast<std::uint32_t>(static_cast<float>(v)))) {}

    // Widening is EXPLICIT on purpose: it forces every consumer of the stored
    // factor value (in particular the SpTRSV inner loops) to route through
    // to_float()/widen(), so no read can silently do bf16 arithmetic.
    constexpr float  to_float() const { return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16); }
    explicit constexpr operator float()  const { return to_float(); }
    explicit constexpr operator double() const { return static_cast<double>(to_float()); }

    static constexpr bf16_t from_bits(std::uint16_t b) { bf16_t r; r.bits = b; return r; }

    // RNE on the fp32 bit pattern. `0x7fff + lsb` is the classic
    // half-way-tie-to-even bias: adding it to the low 16 bits rounds up iff
    // the discarded half is > 0x8000, or == 0x8000 and the kept lsb is odd.
    // The addition carries into the exponent when the mantissa overflows
    // (e.g. 0x3f7fffff -> 0x3f80 == 1.0f), which is the correct RNE result.
    static constexpr std::uint16_t round_bits(std::uint32_t u) {
        if ((u & 0x7fffffffu) > 0x7f800000u)              // NaN: keep it a NaN
            return static_cast<std::uint16_t>((u >> 16) | 0x0040u);
        const std::uint32_t lsb  = (u >> 16) & 1u;
        const std::uint32_t bias = 0x7fffu + lsb;
        return static_cast<std::uint16_t>((u + bias) >> 16);
    }

    friend constexpr bool operator==(bf16_t a, bf16_t b) { return a.bits == b.bits; }
    friend constexpr bool operator!=(bf16_t a, bf16_t b) { return a.bits != b.bits; }
};
static_assert(sizeof(bf16_t) == 2, "bf16_t must be exactly 16 bits");
static_assert(std::is_trivially_copyable_v<bf16_t>);

/// Round an fp32 to bf16 (RNE).
inline constexpr bf16_t from_float(float f) { return bf16_t(f); }
/// Widen a bf16 to fp32 (exact).
inline constexpr float  to_float(bf16_t h)  { return h.to_float(); }

/// widen(): read a stored factor value into a double for compute. Identity
/// (modulo the promotion the arithmetic would do anyway) for the fp32/fp64
/// builds, bf16 -> fp32 -> double for the bf16 build. The SpTRSV kernels
/// route every factor read through this.
inline constexpr double widen(double v) { return v; }
inline constexpr double widen(float v)  { return static_cast<double>(v); }
inline constexpr double widen(bf16_t v) { return static_cast<double>(v.to_float()); }

} // namespace apxchol
