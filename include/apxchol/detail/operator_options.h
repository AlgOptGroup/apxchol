#pragma once
#include <cmath>
#include <stdexcept>

namespace apxchol {

inline constexpr double kDefaultLumpMassCeiling = 0.25;

namespace detail {
inline void validate_regularization(double value) {
    if (!std::isfinite(value) || value < 0)
        throw std::invalid_argument("regularization must be finite and nonnegative");
}

inline void validate_lump_max_mass(double value) {
    if (!std::isfinite(value) || value < 0 || value > 1)
        throw std::invalid_argument("lump_max_mass must be in [0, 1]");
}
} // namespace detail
} // namespace apxchol
