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
// Both SpTRSV backends share these storage conversions and the env reader.
#include <cmath>
#include <cstdlib>

// TODO: use std::float16_t once Clang defines __STDCPP_FLOAT16_T__
// and libc++ provides <stdfloat> in our supported toolchains.
// https://github.com/llvm/llvm-project/issues/105196
// https://github.com/llvm/llvm-project/pull/78503
namespace apxchol {

/// THE fp16-storage switch, read by BOTH SpTRSV backends at every setup:
/// APXCHOL_SPTRSV_FP16=0|1. Tri-state: -1 unset (each backend applies its own
/// default -- OFF on the CPU, ON on the GPU), else 0/1.
inline int sptrsv_fp16_env_tristate() {
    if (const char* e = std::getenv("APXCHOL_SPTRSV_FP16"); e && *e)
        return std::atoi(e) != 0 ? 1 : 0;
    return -1;
}

namespace detail {

inline bool fp16_flushes(_Float16 value) {
    // numeric_limits<_Float16> is unavailable with Clang and in CUDA's C++20 mode.
    // https://github.com/llvm/llvm-project/issues/105196
    return std::abs(float(value)) < __FLT16_MIN__;
}

inline _Float16 narrow_scaled_fp16(float value, float scale) {
    const auto stored = _Float16(value / scale);
    return fp16_flushes(stored) ? _Float16(std::copysign(0.0f, float(stored))) : stored;
}

} // namespace detail

inline constexpr double widen(double v) { return v; }
inline constexpr double widen(float v) { return static_cast<double>(v); }
inline double widen(_Float16 v) {
#if defined(__x86_64__) && __GNUC__ < 17 && !defined(__clang__) && !defined(__CUDACC__)
    // Avoid software half-to-double conversion until GCC 17 is our minimum.
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=127720
    return double(__builtin_assoc_barrier(float(v)));
#else
    return double(v);
#endif
}

// The builtin also handles _Float16 before C++23 library support is available:
// https://github.com/llvm/llvm-project/issues/105196
template<class T>
inline bool is_stored_subnormal(T v) {
    return __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, v) == FP_SUBNORMAL;
}

} // namespace apxchol
