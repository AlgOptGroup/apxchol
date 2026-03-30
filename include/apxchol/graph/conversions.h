#pragma once
/// Conversions between Eigen sparse matrices and graphs.
///
///   laplacian(g)     — graph → Eigen Laplacian
///   make_graph<G>(L) — Eigen Laplacian → G

#include "apxchol/graph/graph.h"
#include <Eigen/Sparse>
#include <vector>

namespace apxchol {

/// Build the graph Laplacian L = D - A as a sparse matrix.
template<typename G>
Eigen::SparseMatrix<double> laplacian(const G& g) {
    using T = Eigen::Triplet<double>;
    const index_t N = g.n();
    std::vector<T> trips;

    for (index_t v = 0; v < N; ++v) {
        double deg = 0.0;
        for (const auto& e : g.neighbors(v)) {
            trips.emplace_back(v, e.to, -e.w);
            deg += e.w;
        }
        trips.emplace_back(v, v, deg);
    }

    Eigen::SparseMatrix<double> L(N, N);
    L.setFromTriplets(trips.begin(), trips.end());
    return L;
}

/// Build a graph from a Laplacian matrix.
template<typename G = graph<>>
G make_graph(const Eigen::SparseMatrix<double>& L) {
    G g(static_cast<index_t>(L.rows()));
    for (index_t k = 0; k < static_cast<index_t>(L.outerSize()); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
            if (it.row() > it.col())
                g.add_edge(static_cast<node_index>(it.row()),
                           static_cast<node_index>(it.col()),
                           -it.value());
    return g;
}

} // namespace apxchol
