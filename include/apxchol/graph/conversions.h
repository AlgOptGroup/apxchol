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

/// Build a graph from a Laplacian or SDDM matrix.
///
/// Off-diagonal entries become weighted edges (negated).
/// Any excess diagonal (row sum > 0, i.e. SDDM) is stored
/// per-vertex in graph::excess() for use during factorization.
template<typename G = graph<>>
G make_graph(const Eigen::SparseMatrix<double>& L) {
    const auto n = static_cast<index_t>(L.rows());
    G g(n);

    // Phase 1: extract edges and diagonal.
    std::vector<double> diag(n, 0.0);
    for (index_t k = 0; k < static_cast<index_t>(L.outerSize()); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
            if (it.row() > it.col())
                g.add_edge(static_cast<node_index>(it.row()),
                           static_cast<node_index>(it.col()),
                           -it.value());
            else if (it.row() == it.col())
                diag[k] = it.value();
        }

    // Phase 2: excess = diag − weighted degree.  Zero for pure Laplacians.
    // Use relative tolerance to avoid false SDDM detection from
    // floating-point accumulation differences across storage backends.
    for (index_t v = 0; v < n; ++v) {
        double wdeg = 0.0;
        for (auto [u, w] : g.neighbors(static_cast<node_index>(v)))
            wdeg += w;
        double excess = diag[v] - wdeg;
        g.excess(static_cast<node_index>(v)) =
            excess > diag[v] * 1e-12 ? excess : 0.0;
    }
    return g;
}

} // namespace apxchol
