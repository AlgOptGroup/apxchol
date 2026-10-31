/// Eigen::SparseMatrix<double>: column-major with int indices by default.
#include "apxchol.h"
#include <Eigen/SparseCore>
#include <cstdio>
#include <vector>

int main() {
    // Laplacian of the path 0-1-2-3 with edge weights 1, 2, 1, from edge triplets.
    std::vector<Eigen::Triplet<double>> entries;
    const double weight[] = {1, 2, 1};
    for (int u = 0; u < 3; ++u) {
        const int v = u + 1;
        entries.insert(entries.end(), {{u, u, weight[u]}, {v, v, weight[u]},
                                       {u, v, -weight[u]}, {v, u, -weight[u]}});
    }
    Eigen::SparseMatrix<double> A(4, 4);
    A.setFromTriplets(entries.begin(), entries.end());  // sorted, duplicates summed
    A.makeCompressed();  // already compressed here; required after insert()

    const auto n = static_cast<std::size_t>(A.cols());
    const auto nnz = static_cast<std::size_t>(A.nonZeros());
    const apxchol::csc_view view{A.cols(), {A.outerIndexPtr(), n + 1},
                                 {A.innerIndexPtr(), nnz}, {A.valuePtr(), nnz}};

    Eigen::VectorXd b(4);
    b << 1, 0, 0, -1;
    const auto result = apxchol::solve(view, {b.data(), n}, 1e-10);
    const Eigen::Map<const Eigen::VectorXd> x(result.x.data(), A.cols());

    std::printf("Eigen: residual %.1e, |Ax - b| = %.1e\n", result.residual, (A * x - b).norm());
    return result.converged ? 0 : 1;
}
