#pragma once
// FP16 triangular-solve storage and per-solver selection.
//
// Off-diagonals store L_ij / s_j in binary16, where s_j is the column's
// maximum absolute off-diagonal (one for an empty/zero column). With the normal
// floating-point environment, conversion rounds to nearest, ties to even;
// subnormals flush to signed zero.
// Scales, compensated diagonals, constructed factors and exports remain FP32.
// CPU solve arithmetic widens to FP64; GPU triangular solves use FP32.
// CPU/CUDA fold the scales into the solve vectors, while Metal reconstructs
// FP32 coefficients. See docs/precision.md for the storage contract.
#include <cmath>
#include <stdexcept>
#include <string_view>

// TODO: use std::float16_t once Clang defines __STDCPP_FLOAT16_T__
// and libc++ provides <stdfloat> in our supported toolchains.
// https://github.com/llvm/llvm-project/issues/105196
// https://github.com/llvm/llvm-project/pull/78503
namespace apxchol {

enum class factor_storage_type { automatic, fp16, fp32 };

namespace detail {

inline factor_storage_type parse_factor_storage(std::string_view value) {
    if (value == "auto") return factor_storage_type::automatic;
    if (value == "float16") return factor_storage_type::fp16;
    if (value == "float32") return factor_storage_type::fp32;
    throw std::invalid_argument("factor storage must be auto, float16 or float32");
}

inline bool resolve_fp16_storage(factor_storage_type storage, bool default_fp16) {
    switch (storage) {
    case factor_storage_type::automatic: return default_fp16;
    case factor_storage_type::fp16: return true;
    case factor_storage_type::fp32: return false;
    }
    throw std::invalid_argument("invalid factor storage type");
}

// Whether a normalized FP32 value rounds to zero or a subnormal in FP16.
inline bool fp16_flushes(float value) {
    // numeric_limits<_Float16> is unavailable with our Clang toolchains.
    // https://github.com/llvm/llvm-project/issues/105196
    // The midpoint itself rounds up to the smallest normal FP16 value.
    constexpr float flush_threshold = __FLT16_MIN__ - 0.5f * __FLT16_DENORM_MIN__;
    return std::abs(value) < flush_threshold;
}

inline _Float16 narrow_scaled_fp16(float value, float scale) {
    value /= scale;
    return _Float16(fp16_flushes(value) ? std::copysign(0.0f, value) : value);
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
