#pragma once
#include "apxchol/graph/conversions.h"
#include "apxchol/solver/factor_options.h"
#include <Eigen/Sparse>
#include <cstddef>

namespace apxchol {

struct checkpoint;

// Forward-declare the default IS selector (defined in is_block_greedy.h).
struct block_greedy_is;

/// Result of the approximate Cholesky factorization.
///
/// The factorization produces a sparse lower-triangular matrix L and a
/// permutation P such that P^T L L^T P ≈ original Laplacian.
/// Used as a preconditioner for PCG.
struct factorization {
    Eigen::SparseMatrix<double> L;                          // lower-triangular factor
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, index_t> perm;

    // Peak graph memory during factorization (bytes, heap only).
    std::size_t peak_graph_bytes = 0;

    // Per-round statistics (populated when checkpoint is provided)
    struct round_stats {
        size_t active;  // active vertices at start of round
        size_t is_size; // IS size chosen
        double avg_deg; // average degree of active vertices
    };
    std::vector<round_stats> rounds;
};

/// Compute approximate Cholesky factorization of a graph Laplacian.
///
/// ISSelector controls the independent-set selection strategy.
/// Defaults to block_greedy_is; callers can substitute luby_is or
/// baumann_kyng_is (see is_*.h headers) at compile time:
///   auto F = factorize<luby_is>(G, opts);
template<typename ISSelector = block_greedy_is, incidence_storage Incidence>
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

/// Runtime-dispatch overload: picks graph backend from graph_storage enum
/// and IS selector from factor_options::is_select.
factorization factorize(const Eigen::SparseMatrix<double>& L,
                        graph_storage storage,
                        const factor_options& opts = {},
                        checkpoint* cp = nullptr);

namespace detail {

struct factor_col {
    node_index vertex;
    std::vector<std::pair<node_index, double>> entries; // (neighbor, L_value)
};

// Build elimination-order permutation and assemble L in CSC format.
// Precondition: factor_cols contains exactly one entry per vertex (all
// n vertices have been eliminated before this call).
void build_csc(factorization& result,
               const std::vector<factor_col>& factor_cols,
               index_t n,
               checkpoint* cp);

} // namespace detail

} // namespace apxchol

#include "apxchol/solver/factorization_impl.h"
