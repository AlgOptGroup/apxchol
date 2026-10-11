#pragma once
// Host arithmetic for Metal PCG. Error-free transforms require explicit FMA
// and no implicit floating-point contraction or reassociation.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <omp.h>

namespace apxchol::detail::metal_host { using std::fma; }
#define APXCHOL_GPU_CODE(...) __VA_ARGS__
#include "double_float.inc"
#undef APXCHOL_GPU_CODE

namespace apxchol::detail::metal_host {

// Use the stable reduction when the sum of squares is outside the normal range.
template <class Vector>
auto norm(const Vector& v) {
    const auto squared = v.squaredNorm();
    return std::isnormal(squared) ? std::sqrt(squared) : v.stableNorm();
}

/// fp64 -> double-float: hi = fp32(v), lo = fp32(v - hi). Exact for any v
/// with at most 48 significant bits whose parts stay in the fp32 normal range.
inline df split(double v) {
    const float hi = static_cast<float>(v);
    const float lo = static_cast<float>(v - static_cast<double>(hi));
    return {hi, lo};
}

/// hi + lo in fp64: exact for a normalized double-float.
inline double join(df v) { return static_cast<double>(v.hi) + static_cast<double>(v.lo); }

/// A finite power-of-two scale that brings max_abs into [1, 2), except for
/// the smallest inputs, where the scale's exponent is capped. Scaling preserves
/// significands unless a value underflows or overflows. max_abs is finite and positive.
inline double pow2_scale(double max_abs) {
    const int exponent = std::min(-std::ilogb(max_abs), std::numeric_limits<double>::max_exponent - 1);
    return std::ldexp(1.0, exponent);
}

/// fp64 value as a double-float, saturated to the largest finite fp32 (used
/// for the stop threshold, whose overflow would mean "stop").
inline df split_saturated(double v) {
    constexpr double kMax = static_cast<double>(std::numeric_limits<float>::max());
    return split(std::min(v, kMax));
}

// Permutation and vector packing.

/// To the factor's permuted space: out[perm[v]] = v_in[v].
template <class Index>
void scatter(const double* v_in, const Index* perm, std::size_t n, double* out) {
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t v = 0; v < static_cast<std::ptrdiff_t>(n); ++v) out[perm[v]] = v_in[v];
}

/// Back from the permuted space: out[v] = v_perm[perm[v]].
template <class Index>
void gather(const double* v_perm, const Index* perm, std::size_t n, double* out) {
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t v = 0; v < static_cast<std::ptrdiff_t>(n); ++v) out[v] = v_perm[perm[v]];
}

inline void pack(const double* w, std::size_t n, double scale, df* out) {
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        out[i] = split(scale * w[i]);
    }
}

inline void add_unpacked(const df* v, std::size_t n, double scale, double* out) {
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
        out[i] += join(v[i]) / scale;
}

}  // namespace apxchol::detail::metal_host
