#pragma once
#include "apxchol/detail/dense.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace apxchol::detail::gpu_scaling {

// Bring a finite, positive maximum into [1, 2), capping the scale for the
// smallest inputs. Power-of-two scaling preserves representable significands.
inline double pow2_scale(double max_abs) {
    const int exponent = std::min(-std::ilogb(max_abs), std::numeric_limits<double>::max_exponent - 1);
    return std::ldexp(1.0, exponent);
}

// Norm of scale * v without overflow or underflow in the sum of squares: the
// ordinary sum when it is normal, otherwise the sum of squares of v scaled
// into [1, 2) by a power of two, unscaled after the square root.
inline double scaled_norm(const_matrix_ref v, double scale) {
    const double squared = scale == 1 ? vec::squared_norm(v) : vec::scaled_squared_norm(v, scale);
    if (std::isnormal(squared)) return std::sqrt(squared);
    const double maximum = vec::max_abs(v) * scale;
    if (maximum == 0 || !std::isfinite(maximum)) return maximum;
    const double stable = pow2_scale(maximum);
    double sum = 0;
    for (std::int64_t i = 0; i < v.rows(); ++i) {
        const double x = (scale * v[i]) * stable;
        sum += x * x;
    }
    return std::sqrt(sum) / stable;
}

// Use the stable reduction when the sum of squares is outside the normal range.
inline double norm(const_matrix_ref v) { return scaled_norm(v, 1.0); }

// Keep ratios and scaled norms representable even when the true norm overflows.
struct norm_reference {
    double value = 0, scale = 1;

    norm_reference() = default;
    explicit norm_reference(const_matrix_ref v) : value(norm(v)) {
        if (std::isinf(value)) {
            scale = pow2_scale(vec::max_abs(v));
            value = scaled_norm(v, scale);
        }
    }
    double relative(const_matrix_ref v) const {
        const double length = norm(v);
        if (!std::isinf(length)) return (length / value) * scale;
        const double maximum = vec::max_abs(v);
        if (!std::isfinite(maximum)) return length;
        const double reduction = pow2_scale(maximum);
        return scaled_norm(v, reduction) / (value * (reduction / scale));
    }
    double scaled(double factor) const {
        return scale == 1 ? value * factor : value * (factor / scale);
    }
};

// Leave ordinary CUDA inputs unchanged, with room for FP32 intermediate values.
inline double fp32_scale(double max_abs) {
    return max_abs == 0 || (max_abs >= 0x1p-32 && max_abs <= 0x1p32)
        ? 1.0 : pow2_scale(max_abs);
}

} // namespace apxchol::detail::gpu_scaling
