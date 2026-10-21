#pragma once
// Host arithmetic for Metal PCG. Error-free transforms require explicit FMA
// and no implicit floating-point contraction or reassociation.
#include "gpu_scaling.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace apxchol::detail::metal_host { using std::fma; }
#define APXCHOL_GPU_CODE(...) __VA_ARGS__
#include "double_float.inc"
#include "metal_factor_storage.inc"
#undef APXCHOL_GPU_CODE

namespace apxchol::detail::metal_host {

using gpu_scaling::norm;
using gpu_scaling::pow2_scale;

// Keep ratios and scaled norms representable even when the true norm overflows.
struct norm_reference {
    double value = 0, scale = 1;

    norm_reference() = default;
    template<class Vector>
    explicit norm_reference(const Vector& v) : value(norm(v)) {
        if (std::isinf(value)) {
            scale = pow2_scale(v.cwiseAbs().maxCoeff());
            value = norm(v * scale);
        }
    }
    template<class Vector>
    double relative(const Vector& v) const {
        return scale == 1 ? norm(v) / value : norm(v * scale) / value;
    }
    double scaled(double factor) const {
        return scale == 1 ? value * factor : value * (factor / scale);
    }
};

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

}  // namespace apxchol::detail::metal_host
