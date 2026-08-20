#pragma once
// fp16 STORAGE for the SpTRSV factor values -- the type, the converters, and
// THE runtime switch both SpTRSV backends read.
//
// Selected at RUNTIME, per setup, by the single environment variable
//
//     APXCHOL_SPTRSV_FP16 = 0 | 1
//
// (sptrsv_fp16_env_tristate() below is the one reader; unset resolves per
// DEVICE -- OFF on the CPU, ON on the GPU -- because the two have different
// measured verdicts, see the backend headers). Before 2026-08-20 the CPU side
// was a CMake cache variable, APXCHOL_SPTRSV_LOWPREC=OFF|FP16_SCALED, and the
// GPU side a separate env, APXCHOL_GPU_SPTRSV_FP16; both are gone. The old
// GPU name is still READ, as a deprecated alias with a one-shot stderr note.
//
// The shape of the storage:
//
//   * ONLY the off-diagonals of the SpTRSV's CSR/CSC value arrays are stored
//     narrow; the DIAGONAL is kept fp32 in a separate array (omp_sptrsv::diag_
//     on the CPU, d_diag_ on the GPU) -- a narrow diagonal was the dominant
//     iteration-count damage of the first all-bf16 variant: the scaled
//     L_jj / s_j (omp_sptrsv::stored_diag) plus the column's rounding residual.
//   * The factor itself (sparse_csc::vals_, factor_value_t) is fp32; the
//     narrowing happens once, at SpTRSV setup, through
//     omp_sptrsv::narrow_value<fp16_t> (a pure function of the entry, so the
//     CSR transpose and the CSC copy agree bit-for-bit) / the GPU's
//     cuda_host::narrow_fp16_scaled_value.
//   * Every read in the solve kernels widens to fp64 (CPU) / fp32 (GPU) in
//     registers via widen(); the arithmetic is unchanged and the kernels are
//     one source for every storage type (the CPU's fat-level kernels of the
//     16-bit storage are SIMD: _mm256_cvtph_ps + a 4-way FMA chain). This is a
//     preconditioner-QUALITY knob (PCG iteration count), never a
//     residual-floor one.
//
//   The format: IEEE binary16 (11 significant bits, 2^-11 relative in the
//   normal range) of L_ij / s_j with a per-COLUMN scale s_j = max_i |L_ij|
//   over column j's off-diagonals (stored fp32; 1.0f if the column has no
//   nonzero off-diagonal). The scaling maps every column's largest
//   off-diagonal to +-1.0 exactly, so no entry overflows; entries below
//   2^-14 of their column max would fall into fp16's SUBNORMAL range and are
//   FLUSHED to signed zero at storage time. setup() counts the flushed and
//   the subnormal and reports them under APXCHOL_VERBOSE. The scale is NOT
//   multiplied back by the kernels: it is folded into the vectors
//   (forward_solve returns D y, transpose_solve takes it and scales its input
//   by D^-2 -- omp.h "FOLDED INTO THE VECTORS").
//
// (The bf16 / bf16-scaled / fp24 siblings that were measured against it --
// 8-bit mantissa 3-6x the PCG iterations on IPM, fp24 marginal -- were removed
// 2026-08-18; only fp16 with per-column folded scaling + fp32 diagonal +
// column-sum compensation stayed.)
//
// This header holds fp16_t, the widen() overloads (float / double / fp16_t)
// and the env reader. Both SpTRSV backends are its consumers.
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#if defined(__F16C__)
#include <immintrin.h>   // _cvtsh_ss: the one-instruction fp16 -> fp32 widen (vcvtph2ps)
#endif

namespace apxchol {

/// THE fp16-storage switch, read by BOTH SpTRSV backends at every setup:
/// APXCHOL_SPTRSV_FP16=0|1. Tri-state: -1 unset (each backend applies its own
/// default -- OFF on the CPU, ON on the GPU), else 0/1.
///
/// APXCHOL_GPU_SPTRSV_FP16, the GPU-only name this replaced on 2026-08-20, is
/// still read as a DEPRECATED alias when the new name is unset; setting it
/// prints a one-shot stderr note. Note that, being an alias of the UNIFIED
/// variable, it now governs the CPU backend too.
inline int sptrsv_fp16_env_tristate() {
    if (const char* e = std::getenv("APXCHOL_SPTRSV_FP16"); e && *e)
        return std::atoi(e) != 0 ? 1 : 0;
    if (const char* e = std::getenv("APXCHOL_GPU_SPTRSV_FP16"); e && *e) {
        static const bool warned = [] {
            std::fprintf(stderr,
                "[apxchol] APXCHOL_GPU_SPTRSV_FP16 is DEPRECATED: the fp16 factor storage is one knob for both"
                " devices now, APXCHOL_SPTRSV_FP16=0|1 (unset = on for the GPU, off for the CPU). Reading the"
                " old name as an alias -- which means it governs the CPU SpTRSV as well.\n");
            return true;
        }();
        (void)warned;
        return std::atoi(e) != 0 ? 1 : 0;
    }
    return -1;
}

// ── fp16_t: IEEE-754 binary16 storage ───────────────────────────────────
// 1 sign, 5 exponent (bias 15), 10 explicit mantissa bits (11 significant ->
// RNE relative error <= 2^-11 in the normal range [2^-14, 65504]). Values
// below 2^-14 are SUBNORMAL (m * 2^-24, m = 1..1023: absolute error <= 2^-25,
// relative precision degrades towards 1 bit at 2^-24); |x| < 2^-25 rounds to
// (signed) ZERO under RNE, |x| == 2^-25 exactly ties to even -> zero; |x| >=
// 65520 (the midpoint above 65504) rounds to +-inf; NaN stays NaN.
//
// Storage type only: the converting constructor narrows fp32 -> fp16 with a
// bit-level RNE (round_bits: same result as the compiler's _Float16 cast, which
// a unit test cross-checks exhaustively where _Float16 exists), and to_float()
// widens exactly -- via the F16C intrinsic _cvtsh_ss (one vcvtph2ps) where the
// target has it (-march=native on any x86 since Ivy Bridge / Zen), else the
// compiler-native _Float16, else the bit-level widen_bits(). The intrinsic is
// used deliberately instead of `(float)(_Float16)`: GCC folds the subsequent
// float -> double promotion into a direct half -> double conversion, which has
// no hardware instruction and becomes a libgcc __extendhfdf2 CALL in the SpTRSV
// inner loop (measured 3x solve slowdown before the switch). Widening is
// EXPLICIT so no read can silently do fp16 arithmetic; the SpTRSV kernels
// route through widen().
struct fp16_t {
    std::uint16_t bits;

    // No default member initializer, deliberately: fp16_t stays trivially
    // default-constructible like float, so `new fp16_t[n]` is uninitialized
    // (the transpose's transient bucket relies on that) while
    // std::vector::resize still value-initializes to zero bits (== 0.0f).
    fp16_t() = default;

    template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
    constexpr fp16_t(T v) : bits(round_bits(std::bit_cast<std::uint32_t>(static_cast<float>(v)))) {}

    float to_float() const {
#if defined(__F16C__)
        return _cvtsh_ss(bits);
#elif defined(__FLT16_MANT_DIG__)
        return static_cast<float>(std::bit_cast<_Float16>(bits));
#else
        return widen_bits(bits);
#endif
    }
    explicit operator float()  const { return to_float(); }
    explicit operator double() const { return static_cast<double>(to_float()); }

    static constexpr fp16_t from_bits(std::uint16_t b) { fp16_t r; r.bits = b; return r; }

    // Classification on the bit pattern (what setup()'s flush statistics use).
    static constexpr bool is_zero(std::uint16_t h)      { return (h & 0x7fffu) == 0; }
    static constexpr bool is_subnormal(std::uint16_t h) { return (h & 0x7c00u) == 0 && (h & 0x03ffu) != 0; }
    static constexpr bool is_inf_or_nan(std::uint16_t h){ return (h & 0x7c00u) == 0x7c00u; }

    // fp32 bit pattern -> fp16 bit pattern, round-to-nearest-even, IEEE
    // semantics for subnormals / overflow / NaN. Pure integer arithmetic,
    // constexpr.
    static constexpr std::uint16_t round_bits(std::uint32_t u) {
        const std::uint32_t sign = (u >> 16) & 0x8000u;
        const std::uint32_t a    = u & 0x7fffffffu;            // |x| pattern
        if (a >= 0x7f800000u)                                  // inf or NaN
            return static_cast<std::uint16_t>(sign | 0x7c00u | (a > 0x7f800000u ? 0x0200u : 0u));
        if (a >= 0x477ff000u)                                  // >= 65520: RNE overflows to inf
            return static_cast<std::uint16_t>(sign | 0x7c00u);
        if (a >= 0x38800000u) {                                // normal fp16 range: |x| >= 2^-14
            // Rebias the exponent (127 -> 15 == subtract 112 << 23), then drop
            // the low 13 mantissa bits with the half-way-tie-to-even bias; a
            // mantissa carry propagates into the exponent by plain addition.
            const std::uint32_t m   = a - (112u << 23);
            const std::uint32_t lsb = (m >> 13) & 1u;
            return static_cast<std::uint16_t>(sign | ((m + 0xfffu + lsb) >> 13));
        }
        if (a >= 0x33000000u) {                                // subnormal fp16 range: 2^-25 <= |x| < 2^-14
            // fp16 subnormal value = r * 2^-24. With the hidden bit restored the
            // fp32 significand M (24 bits) represents M * 2^(e-150) (e = biased
            // exponent), so r = round(M * 2^(e-126)) = RNE(M >> (126 - e)),
            // shift in [14, 24]. r may round up to 0x400 == the smallest normal,
            // which is the correct encoding.
            const std::uint32_t M     = (a & 0x7fffffu) | 0x800000u;
            const int           shift = 126 - static_cast<int>(a >> 23);
            const std::uint32_t half  = 1u << (shift - 1);
            const std::uint32_t lsb   = (M >> shift) & 1u;
            return static_cast<std::uint16_t>(sign | ((M + half - 1u + lsb) >> shift));
        }
        return static_cast<std::uint16_t>(sign);               // |x| < 2^-25: flush to signed zero
    }

    // fp16 bit pattern -> fp32 (exact), bit-level reference implementation.
    static constexpr float widen_bits(std::uint16_t h) {
        const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
        const std::uint32_t e    = (h >> 10) & 0x1fu;
        const std::uint32_t m    = h & 0x3ffu;
        if (e == 0) {
            if (m == 0) return std::bit_cast<float>(sign);                    // +-0
            const float f = static_cast<float>(m) * 0x1p-24f;                 // subnormal: exact
            return std::bit_cast<float>(sign | std::bit_cast<std::uint32_t>(f));
        }
        if (e == 31) return std::bit_cast<float>(sign | 0x7f800000u | (m << 13)); // inf / NaN
        return std::bit_cast<float>(sign | ((e + 112u) << 23) | (m << 13));
    }

    friend constexpr bool operator==(fp16_t a, fp16_t b) { return a.bits == b.bits; }
    friend constexpr bool operator!=(fp16_t a, fp16_t b) { return a.bits != b.bits; }
};
static_assert(sizeof(fp16_t) == 2, "fp16_t must be exactly 16 bits");
static_assert(std::is_trivially_copyable_v<fp16_t>);

/// widen(): read a stored factor value into a double for compute. Identity
/// (modulo the promotion the arithmetic would do anyway) for the fp32/fp64
/// storage, fp16 -> fp32 -> double for the fp16 one. The SpTRSV kernels route
/// every factor read through this.
inline constexpr double widen(double v) { return v; }
inline constexpr double widen(float v)  { return static_cast<double>(v); }
inline double           widen(fp16_t v) { return static_cast<double>(v.to_float()); }

/// Subnormal-ness of a STORED value, per storage type (setup()'s flush
/// statistics; only fp16 can realistically be subnormal for a factor entry).
inline bool is_stored_subnormal(float v)  { return std::fpclassify(v) == FP_SUBNORMAL; }
inline bool is_stored_subnormal(double v) { return std::fpclassify(v) == FP_SUBNORMAL; }
inline constexpr bool is_stored_subnormal(fp16_t v) { return fp16_t::is_subnormal(v.bits); }

} // namespace apxchol
