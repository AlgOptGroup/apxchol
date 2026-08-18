#pragma once
// bfloat16 storage type for the SpTRSV factor values (APXCHOL_SPTRSV_LOWPREC =
// BF16 or BF16_SCALED; the legacy -DAPXCHOL_SPTRSV_BF16 is an alias of BF16 --
// see lowprec.h, which also holds the fp16_t / fp24_t siblings).
//
// bf16 is the top 16 bits of an IEEE-754 binary32: 1 sign, 8 exponent, 7
// explicit mantissa bits (8 with the hidden bit -> relative precision 2^-8
// per round-to-nearest, i.e. |x - bf16(x)| <= 2^-8 |x|; same dynamic range
// as fp32). It is a STORAGE format only: every read widens to fp32 -- a pure
// integer shift, exact -- and the arithmetic stays in double.
//
// Rounding on store is round-to-nearest-even on the fp32 bit pattern by
// default (the converting constructor); the carry out of the mantissa into
// the exponent (and, for values near FLT_MAX, into infinity) is handled by
// plain integer addition. NaN/inf are not expected in a factor, but a NaN is
// kept a NaN (never rounded into inf). double narrows through fp32 first
// (double rounding -- immaterial at 8-bit precision, and it keeps "RNE on the
// fp32 pattern" the single definition). from_float_stochastic() is the
// opt-in UNBIASED alternative (stochastic rounding, deterministic per-entry
// threshold) that the SpTRSV setup uses under APXCHOL_BF16_STOCHASTIC=1.
//
// Where the narrowing happens: NOT in the factor assembler. The factor
// (sparse_csc::vals_, factor_value_t) stays fp32 under the bf16 builds so the
// exact fp32 DIAGONAL (and, for BF16_SCALED, the per-column scale) is still
// available when omp_sptrsv::setup narrows the off-diagonals into its own bf16
// CSR/CSC copies (see omp.h: narrow_value, diag_, scale_).
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
    // this is what makes `sptrsv_value_t v = some_float;` -- the SpTRSV
    // setup's copy of the fp32 factor into its bf16 CSR/CSC -- Just Work when
    // sptrsv_value_t is bf16_t, exactly like the implicit double -> float
    // narrowing of the fp32 build. RNE (round_bits); the stochastic variant
    // is a separate, explicit call (from_float_stochastic).
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

    // STOCHASTIC rounding on the fp32 bit pattern: `t` is a uniform 16-bit
    // threshold. Let r = the discarded low 16 bits of u (0 <= r < 2^16). The
    // truncated value rounds UP iff r + t >= 2^16, i.e. with probability
    // r / 2^16 -- exactly the fraction of a bf16 ulp being dropped -- so the
    // rounding is UNBIASED in expectation (E[bf16(x)] == x, unlike RNE, whose
    // per-entry error is deterministic and can be systematically signed on a
    // structured input). Same carry-into-exponent handling as RNE (plain
    // integer addition), same NaN guard. r == 0 (exact) never rounds up.
    static constexpr std::uint16_t round_bits_stochastic(std::uint32_t u, std::uint16_t t) {
        if ((u & 0x7fffffffu) > 0x7f800000u)              // NaN: keep it a NaN
            return static_cast<std::uint16_t>((u >> 16) | 0x0040u);
        return static_cast<std::uint16_t>((u + t) >> 16);
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

/// Deterministic per-entry 16-bit threshold for stochastic rounding: a
/// splitmix64 finalizer of the entry's index (its position in the factor's
/// CSC value array), top 16 bits. Pure function of the index -> the same
/// entry gets the same rounding in every array that stores it (CSC and its
/// CSR transpose MUST agree, or the preconditioner L1 L2^T is not SPD) and on
/// every run / thread count (run-to-run deterministic given the factor).
inline constexpr std::uint16_t bf16_stochastic_threshold(std::uint64_t idx) {
    std::uint64_t z = idx + 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    return static_cast<std::uint16_t>(z >> 48);
}
/// Round an fp32 to bf16 with stochastic rounding, threshold from `idx`.
inline constexpr bf16_t from_float_stochastic(float f, std::uint64_t idx) {
    return bf16_t::from_bits(bf16_t::round_bits_stochastic(
        std::bit_cast<std::uint32_t>(f), bf16_stochastic_threshold(idx)));
}

/// widen(): read a stored factor value into a double for compute. Identity
/// (modulo the promotion the arithmetic would do anyway) for the fp32/fp64
/// builds, bf16 -> fp32 -> double for the bf16 build. The SpTRSV kernels
/// route every factor read through this.
inline constexpr double widen(double v) { return v; }
inline constexpr double widen(float v)  { return static_cast<double>(v); }
inline constexpr double widen(bf16_t v) { return static_cast<double>(v.to_float()); }

} // namespace apxchol
