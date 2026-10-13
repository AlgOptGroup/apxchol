#pragma once
// Host arithmetic for Metal PCG. Error-free transforms require explicit FMA
// and no implicit floating-point contraction or reassociation.
#include "gpu_scaling.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace apxchol::detail::metal_host { using std::fma; }
#define APXCHOL_GPU_CODE(...) __VA_ARGS__
#include "double_float.inc"
#include "metal_factor_storage.inc"
#undef APXCHOL_GPU_CODE

namespace apxchol::detail::metal_host {

using gpu_scaling::norm;
using gpu_scaling::pow2_scale;

/// fp64 -> double-float: hi = fp32(v), lo = fp32(v - hi). Exact for any v
/// with at most 48 significant bits whose parts stay in the fp32 normal range.
inline df split(double v) {
    const float hi = static_cast<float>(v);
    const float lo = static_cast<float>(v - static_cast<double>(hi));
    return {hi, lo};
}

/// hi + lo in fp64: exact for a normalized double-float.
inline double join(df v) { return static_cast<double>(v.hi) + static_cast<double>(v.lo); }

inline factor_metadata pack_factor_metadata(float scale, float scaled_diagonal) {
    const float diagonal = static_cast<float>(double(scaled_diagonal) * scale);
    if (!(scale > 0) || !std::isfinite(scale) || diagonal == 0 || !std::isfinite(diagonal))
        throw std::domain_error("Metal FP16 factor has an invalid scale or diagonal");
    return {scale, diagonal};
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
void scatter(std::span<const double> v_in, const std::vector<Index>& perm, std::span<double> out) {
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t v = 0; v < static_cast<std::ptrdiff_t>(v_in.size()); ++v) out[perm[v]] = v_in[v];
}

/// Back from the permuted space: out[v] = v_perm[perm[v]].
template <class Index>
void gather(std::span<const double> v_perm, const std::vector<Index>& perm, std::span<double> out) {
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t v = 0; v < static_cast<std::ptrdiff_t>(v_perm.size()); ++v) out[v] = v_perm[perm[v]];
}

inline void pack(std::span<const double> w, double scale, std::span<df> out) {
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(w.size()); ++i) {
        out[i] = split(scale * w[i]);
    }
}

inline void add_unpacked(std::span<const df> v, double scale, std::span<double> out) {
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(v.size()); ++i)
        out[i] += join(v[i]) / scale;
}

}  // namespace apxchol::detail::metal_host
