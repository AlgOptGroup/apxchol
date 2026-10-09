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
// GPU side a separate env, APXCHOL_GPU_SPTRSV_FP16; both are gone.
//
// The shape of the storage:
//
//   * ONLY the off-diagonals of the SpTRSV's CSR/CSC value arrays are stored
//     narrow; the DIAGONAL is kept fp32 in a separate array (omp_sptrsv::diag_
//     on the CPU, d_diag_ on the GPU) -- a narrow diagonal was the dominant
//     iteration-count damage of the first all-bf16 variant: the scaled
//     L_jj / s_j (omp_sptrsv::stored_diag) plus the column's rounding residual.
//   * The factor itself (sparse_csc::vals_, factor_value_t) is fp32; the
//     narrowing happens once at SpTRSV setup. CPU and GPU-host preparation
//     share detail::narrow_scaled_fp16, a pure per-entry conversion, so
//     the CSR transpose and CSC copy use the same bits. Device finalization
//     implements this contract separately in CUDA.
//   * Every read in the solve kernels widens to fp64 (CPU) / fp32 (GPU) in
//     registers via widen(); the arithmetic is unchanged and the kernels are
//     one source for every storage type (the CPU's fat-level kernels of the
//     16-bit storage are SIMD: packed widening and FP64 arithmetic). This is a
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
//   by D^-2; see docs/precision.md).
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
#include <cstdlib>
#include <type_traits>

namespace apxchol {

/// THE fp16-storage switch, read by BOTH SpTRSV backends at every setup:
/// APXCHOL_SPTRSV_FP16=0|1. Tri-state: -1 unset (each backend applies its own
/// default -- OFF on the CPU, ON on the GPU), else 0/1.
inline int sptrsv_fp16_env_tristate() {
    if (const char* e = std::getenv("APXCHOL_SPTRSV_FP16"); e && *e)
        return std::atoi(e) != 0 ? 1 : 0;
    return -1;
}

// Native IEEE binary16 storage; arithmetic in the solver still uses double.
using fp16_t = _Float16;
static_assert(sizeof(fp16_t) == 2);
static_assert(std::is_trivially_copyable_v<fp16_t>);

namespace detail {

inline constexpr bool fp16_is_subnormal(std::uint16_t bits) {
    return (bits & 0x7c00u) == 0 && (bits & 0x03ffu) != 0;
}

// Keep setup's fixed round-to-nearest-even contract even if the caller changes
// the floating-point rounding mode. Solve-time widening uses native casts.
inline constexpr std::uint16_t round_fp16_bits(std::uint32_t u) {
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

inline fp16_t narrow_scaled_fp16(float value, float scale) {
    auto bits = round_fp16_bits(std::bit_cast<std::uint32_t>(value / scale));
    if (fp16_is_subnormal(bits)) bits &= 0x8000u;
    return std::bit_cast<fp16_t>(bits);
}

inline bool fp16_flushes(float value) {
    return (round_fp16_bits(std::bit_cast<std::uint32_t>(value)) & 0x7c00u) == 0;
}

} // namespace detail

inline constexpr double widen(double v) { return v; }
inline constexpr double widen(float v) { return static_cast<double>(v); }
inline double widen(fp16_t v) {
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__) && __GNUC__ < 17
    // Avoid software half-to-double conversion until GCC 17 is our minimum.
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=127720
    return double(__builtin_assoc_barrier(float(v)));
#else
    return double(v);
#endif
}

inline bool is_stored_subnormal(float v) { return std::fpclassify(v) == FP_SUBNORMAL; }
inline bool is_stored_subnormal(double v) { return std::fpclassify(v) == FP_SUBNORMAL; }
inline constexpr bool is_stored_subnormal(fp16_t v) {
    return detail::fp16_is_subnormal(std::bit_cast<std::uint16_t>(v));
}

} // namespace apxchol
