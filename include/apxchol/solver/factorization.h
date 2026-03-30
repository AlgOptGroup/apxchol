#pragma once
#include "apxchol/graph/conversions.h"
#include <Eigen/Sparse>
#include <cstddef>

namespace apxchol {

struct checkpoint;

/// Result of the approximate Cholesky factorization.
///
/// The factorization produces a sparse lower-triangular matrix L and a
/// permutation P such that P^T L L^T P ≈ original Laplacian.
/// Used as a preconditioner for PCG.
struct factorization {
    Eigen::SparseMatrix<double> L;                          // lower-triangular factor
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, index_t> perm;

    // Per-round statistics (populated when checkpoint is provided)
    struct round_stats {
        int active;     // active vertices at start of round
        int is_size;    // IS size chosen
        double avg_deg; // average degree of active vertices
    };
    std::vector<round_stats> rounds;
};

struct factor_options {
    unsigned seed = 42;
    double degree_multiplier = 3.0;  // IS degree threshold = multiplier × avg_degree
    size_t omp_threshold = 2000;     // min active/IS vertices before engaging OpenMP
};

/// Compute approximate Cholesky factorization of a graph Laplacian.
template<incidence_storage Incidence>
factorization factorize(const graph<Incidence>& G,
                        const factor_options& opts = {},
                        checkpoint* cp = nullptr);

/// Factorize from a Laplacian matrix, building a Graph internally.
template<typename Graph = graph<>>
factorization factorize(const Eigen::SparseMatrix<double>& L,
                        const factor_options& opts = {},
                        checkpoint* cp = nullptr) {
    auto G = make_graph<Graph>(L);
    return factorize(G, opts, cp);
}

/// Runtime-dispatch overload: picks the graph backend from graph_storage enum.
factorization factorize(const Eigen::SparseMatrix<double>& L,
                        graph_storage storage,
                        const factor_options& opts = {},
                        checkpoint* cp = nullptr);

} // namespace apxchol

#include "apxchol/solver/factorization_impl.h"
