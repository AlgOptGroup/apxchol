#pragma once
#include <Eigen/Core>
#include <cstdint>
#include <random>

namespace apxchol::test {

// Zero-mean, unit-norm right-hand side from a fixed seed.
inline Eigen::VectorXd random_rhs(Eigen::Index n, std::uint64_t seed = 1) {
    std::mt19937_64 rng(seed);
    Eigen::VectorXd b(n);
    for (Eigen::Index i = 0; i < n; ++i)
        b[i] = static_cast<double>(rng() >> 11) * 0x1p-52 - 1.0;
    b.array() -= b.mean();
    return b.normalized();
}

} // namespace apxchol::test
