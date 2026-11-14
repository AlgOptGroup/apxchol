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

// Reference magnitude for the input of FP32 triangular sweeps: the power of
// two at the square root of the preconditioner's diagonal scale d, taken as
// the geometric mean of the smallest and largest positive diagonal entries of
// L L^T, that is of L_jj^2. The sweeps turn an input r into about r / sqrt(d)
// and then r / d, so an input near sqrt(d) centres the three around 1 and
// leaves each the same room inside FP32. A factor that stays on the device
// was built from the operator, whose diagonal stands in for it. Metal applies
// the same power of two to the preconditioned residual. 1 when there is no
// positive diagonal.
inline double reference_scale(double min_diagonal, double max_diagonal) {
    if (!(min_diagonal > 0) || !(max_diagonal >= min_diagonal) || !std::isfinite(max_diagonal))
        return 1.0;
    const int exponent_sum = std::ilogb(min_diagonal) + std::ilogb(max_diagonal);
    return std::ldexp(1.0, static_cast<int>(std::floor(exponent_sum / 4.0)));
}

// reference_scale of a host factor (factorization.h): the diagonal of column
// j is L_jj, times the column scale s_j under FP16 storage. Columns [0, m).
template<class Factor, class Index>
double factor_reference_scale(const Factor& F, Index m) {
    const auto ptr = F.L.col_ptrs();
    const auto idx = F.L.row_indices();
    const auto val = F.L.values();
    const bool scaled = !F.scale.empty();
    double min_diagonal = std::numeric_limits<double>::infinity(), max_diagonal = 0;
    #pragma omp parallel for schedule(static) reduction(min : min_diagonal) reduction(max : max_diagonal) if (m > 4096)
    for (Index col = 0; col < m; ++col) {
        const auto p = ptr[col];
        if (p == ptr[col + 1] || static_cast<Index>(idx[p]) != col) continue;
        const double entry = static_cast<double>(val[p]) * (scaled ? static_cast<double>(F.scale[col]) : 1.0);
        const double diagonal = entry * entry;
        if (diagonal > 0 && std::isfinite(diagonal)) {
            min_diagonal = std::min(min_diagonal, diagonal);
            max_diagonal = std::max(max_diagonal, diagonal);
        }
    }
    return reference_scale(min_diagonal, max_diagonal);
}

// reference_scale of a compressed symmetric operator (n, ptr, idx, val).
template<class Operator>
double operator_reference_scale(const Operator& A) {
    using index = decltype(A.n);
    double min_diagonal = std::numeric_limits<double>::infinity(), max_diagonal = 0;
    #pragma omp parallel for schedule(static) reduction(min : min_diagonal) reduction(max : max_diagonal) if (A.n > 4096)
    for (index col = 0; col < A.n; ++col) {
        double diagonal = 0;
        for (auto p = A.ptr[col]; p < A.ptr[col + 1]; ++p)
            if (static_cast<index>(A.idx[p]) == col) diagonal += static_cast<double>(A.val[p]);
        if (diagonal > 0) {
            min_diagonal = std::min(min_diagonal, diagonal);
            max_diagonal = std::max(max_diagonal, diagonal);
        }
    }
    return reference_scale(min_diagonal, max_diagonal);
}

// Power of two that brings a finite, positive maximum into
// [reference, 2 reference), capped to the finite powers of two.
inline double pow2_scale(double max_abs, double reference) {
    const int exponent = std::clamp(std::ilogb(reference) - std::ilogb(max_abs),
                                    std::numeric_limits<double>::min_exponent
                                        - std::numeric_limits<double>::digits,
                                    std::numeric_limits<double>::max_exponent - 1);
    return std::ldexp(1.0, exponent);
}

// Scale of one right-hand-side column ahead of FP32 triangular sweeps: every
// column is brought into [reference, 2 reference), so the sweeps always see
// an input of the same size whatever the caller passed. Powers of two change
// no significand.
inline double fp32_scale(double max_abs, double reference = 1.0) {
    return max_abs == 0 ? 1.0 : pow2_scale(max_abs, reference);
}

} // namespace apxchol::detail::gpu_scaling
