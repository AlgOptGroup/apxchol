#include "apxchol/graph.h"
#include <Eigen/Sparse>

namespace apxchol {

double graph::weighted_degree(node_index v) const {
    double deg = 0.0;
    for_neighbors(v, [&](node_index, double w) { deg += w; });
    return deg;
}

Eigen::SparseMatrix<double> graph::laplacian() const {
    using T = Eigen::Triplet<double>;
    const int N = n();
    std::vector<T> trips;

    for (int v = 0; v < N; ++v) {
        double deg = 0.0;
        for_neighbors(v, [&](node_index u, double w) {
            trips.emplace_back(v, u, -w);
            deg += w;
        });
        trips.emplace_back(v, v, deg);
    }

    Eigen::SparseMatrix<double> L(N, N);
    L.setFromTriplets(trips.begin(), trips.end());
    return L;
}

} // namespace apxchol
