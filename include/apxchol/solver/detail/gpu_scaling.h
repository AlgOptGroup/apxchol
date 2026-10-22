#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

namespace apxchol::detail::gpu_scaling {

// Use the stable reduction when the sum of squares is outside the normal range.
template <class Vector>
auto norm(const Vector& v) {
    const auto squared = v.squaredNorm();
    return std::isnormal(squared) ? std::sqrt(squared) : v.stableNorm();
}

// Bring a finite, positive maximum into [1, 2), capping the scale for the
// smallest inputs. Power-of-two scaling preserves representable significands.
inline double pow2_scale(double max_abs) {
    const int exponent = std::min(-std::ilogb(max_abs), std::numeric_limits<double>::max_exponent - 1);
    return std::ldexp(1.0, exponent);
}

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
        const double length = norm(v);
        if (!std::isinf(length)) return (length / value) * scale;
        const double maximum = v.cwiseAbs().maxCoeff();
        if (!std::isfinite(maximum)) return length;
        const double reduction = pow2_scale(maximum);
        return norm(v * reduction) / (value * (reduction / scale));
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
