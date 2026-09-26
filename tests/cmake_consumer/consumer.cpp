#include <apxchol.h>
#include <cmath>
#include <iostream>
#include <vector>

static_assert(__cplusplus > 202002L, "apxchol must propagate C++23");
int apxchol_consumer_core_alignment();

int main() {
    if (apxchol_consumer_core_alignment() != EIGEN_DEFAULT_ALIGN_BYTES) {
        std::cerr << "core and consumer Eigen alignment differ\n";
        return 1;
    }
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
        const auto result = apxchol::solve(matrix, rhs);
        const double residual = (matrix * result.x - rhs).norm() / rhs.norm();
        if (!std::isfinite(residual) || residual > 1e-8) {
            std::cerr << "original-system residual failed at size " << n << '\n';
            return 1;
        }
    }
    std::cout << "C++23, matching Eigen alignment " << EIGEN_DEFAULT_ALIGN_BYTES
              << ", three solves passed\n";
}
