#include <apxchol.h>
#include <Eigen/Sparse>
#include <cmath>
#include <iostream>
#include <vector>

static_assert(__cplusplus > 202002L, "apxchol must propagate C++23");
int apxchol_consumer_core_alignment();
Eigen::VectorXd apxchol_consumer_core_vector(Eigen::Index n);
double apxchol_consumer_core_take(Eigen::VectorXd v);

int main() {
    if (apxchol_consumer_core_alignment() != EIGEN_MAX_ALIGN_BYTES) {
        std::cerr << "core and consumer Eigen alignment differ\n";
        return 1;
    }
    if (apxchol_consumer_core_vector(1000).sum() != 1000) return 1;
    if (apxchol_consumer_core_take(Eigen::VectorXd::Ones(1001)) != 1001) return 1;
    for (int n : {17, 33, 65}) {
        Eigen::SparseMatrix<double> matrix(n, n);
        std::vector<Eigen::Triplet<double>> entries;
        for (int i = 0; i < n; ++i) {
            entries.emplace_back(i, i, 3.0);
            if (i + 1 < n) {
                entries.emplace_back(i, i + 1, -1.0);
                entries.emplace_back(i + 1, i, -1.0);
            }
        }
        matrix.setFromTriplets(entries.begin(), entries.end());
        Eigen::VectorXd expected(n);
        for (int i = 0; i < n; ++i) expected[i] = std::sin(0.13 * i);
        const Eigen::VectorXd rhs = matrix * expected;
        const apxchol::csc_view view{n,
            {matrix.outerIndexPtr(), static_cast<std::size_t>(n) + 1},
            {matrix.innerIndexPtr(), static_cast<std::size_t>(matrix.nonZeros())},
            {matrix.valuePtr(), static_cast<std::size_t>(matrix.nonZeros())}};
        const auto result = apxchol::solve(view, {rhs.data(), static_cast<std::size_t>(n)});
        const Eigen::Map<const Eigen::VectorXd> x(result.x.data(), n);
        const double residual = (matrix * x - rhs).norm() / rhs.norm();
        if (!std::isfinite(residual) || residual > 1e-8) {
            std::cerr << "original-system residual failed at size " << n << '\n';
            return 1;
        }
    }
    std::cout << "C++23, matching Eigen alignment " << EIGEN_MAX_ALIGN_BYTES
              << ", three solves passed\n";
}
