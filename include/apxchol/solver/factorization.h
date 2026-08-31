#pragma once
#include "apxchol/graph/conversions.h"
#include "apxchol/operator_class.h"
#include "apxchol/solver/elimination/elimination.h"
#include "apxchol/solver/elimination/volume_tree.h"
#include "apxchol/solver/factor_options.h"
#include "apxchol/solver/partitioner_list.h"
#include "apxchol/sparse_csc.h"
#include <Eigen/Sparse>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace apxchol {

struct checkpoint;

/// Result of the approximate Cholesky factorization.
///
/// The factorization produces a sparse lower-triangular matrix L and a
/// permutation P such that P^T L L^T P ≈ original Laplacian.
/// Used as a preconditioner for PCG.
struct factorization {
    sparse_csc L;                                           // lower-triangular factor (owned CSC)
    // Elimination-order permutation: perm[original_vertex] = new_position.
    //   to permuted space   : x_perm[perm[v]] = b[v]   (scatter)
    //   from permuted space : x[v] = x_perm[perm[v]]    (gather)
    // Replaces Eigen::PermutationMatrix so the factor owns no signed-int Eigen state.
    std::vector<node_index> perm;

    // True when the input was SDDM (positive row sums) rather than
    // a pure Laplacian.  Affects preconditioner rank and centering.
    bool sddm = false;

    // Peak graph memory during factorization (bytes, heap only).
    std::size_t peak_graph_bytes = 0;

    // Stored positive off-diagonal entries the M-matrix lumping moved onto the
    // diagonal before this factor was built (0 = the input was already an
    // M-matrix, the overwhelmingly common case). Two entries per pair; the
    // rank of the perturbation is half this. See operator_class.h. The
    // OPERATOR is untouched — only the matrix the preconditioner was built
    // from carries the lumping.
    Eigen::Index lumped_offdiag = 0;

    // Per-round statistics. active/is_size/avg_deg are always populated;
    // nnz_added/nnz_total only when a checkpoint is provided (they need an
    // extra factor_cols scan). is_size drives the SpTRSV round-as-level path.
    struct round_stats {
        size_t active;     // active vertices at start of round
        size_t is_size;    // IS size chosen (== num_regions for singleton partitions)
        double avg_deg;    // average degree of active vertices
        size_t nnz_added;  // nnz added to factor L this round (sum of column sizes incl. diag)
        size_t nnz_total;  // cumulative nnz(L) at end of round
    };
    std::vector<round_stats> rounds;
};

/// Compute approximate Cholesky factorization of a graph Laplacian.
///
/// Partitioner controls the per-round IS / region selection strategy.
/// Defaults to block_greedy_partitioner; callers can substitute any
/// partitioner from partitioner_list.h at compile time:
///   auto F = factorize<priority_greedy_partitioner>(G, opts);
template<typename Partitioner = block_greedy_partitioner, incidence_storage Incidence>
factorization factorize(graph<Incidence> G,
                        const factor_options& opts = {},
                        checkpoint* cp = nullptr);

namespace detail {

/// The prologue EVERY matrix-level factorize() shares, so that no surface can
/// drift from another: assert the operator contract (operator_class.h), apply
/// M-matrix lumping if the matrix needs and permits it, build the graph from
/// the result, and record what was lumped on the returned factorization.
///
/// `L` itself is never modified — a lumped input is factorized from a private
/// copy — which is what keeps the operator PCG applies bit-identical to the
/// one the caller passed.
template<typename Graph, typename Build>
factorization factorize_operator(const Eigen::SparseMatrix<double>& L,
                                 Build&& build) {
    const operator_view op(L);
    factorization F = build(make_graph<Graph>(op.matrix()));
    F.lumped_offdiag = op.lumped();
    return F;
}

} // namespace detail

/// Factorize from a Laplacian matrix, building a Graph internally.
template<typename Graph = graph<>>
factorization factorize(const Eigen::SparseMatrix<double>& L,
                        const factor_options& opts = {},
                        checkpoint* cp = nullptr) {
    return detail::factorize_operator<Graph>(L, [&](auto&& G) {
        return factorize(std::move(G), opts, cp);
    });
}

/// Custom-eliminator overload: substitute your own star-vertex elimination
/// rule (see elimination/elimination.h for the `eliminator` contract).
/// The default overloads dispatch the built-in sampler named by
/// factor_options::clique_sampler (GKS unless explicitly changed).
template<typename Partitioner = block_greedy_partitioner, eliminator E,
         incidence_storage Incidence>
factorization factorize(graph<Incidence> G, const E& elim,
                        const factor_options& opts = {},
                        checkpoint* cp = nullptr);

/// Partitioner-instance overloads: pass a configured (possibly stateful)
/// partitioner — the seam for custom independent-set selection and custom
/// elimination order (see solver/partitioner.h for the concept and contract).
/// The instance is taken by value; state mutated during the run is not
/// observable afterwards — reference it through a member pointer if you need
/// to inspect it post-run.
template<partitioner P, incidence_storage Incidence>
factorization factorize(graph<Incidence> G, P part,
                        const factor_options& opts = {},
                        checkpoint* cp = nullptr);

template<partitioner P, eliminator E, incidence_storage Incidence>
factorization factorize(graph<Incidence> G, P part, const E& elim,
                        const factor_options& opts = {},
                        checkpoint* cp = nullptr);

/// Matrix-level conveniences for the custom seams: build the graph internally
/// (default storage) and forward. Equivalent to
/// factorize(make_graph<Graph>(L), ...).
template<typename Partitioner = block_greedy_partitioner, eliminator E,
         typename Graph = graph<>>
factorization factorize(const Eigen::SparseMatrix<double>& L, const E& elim,
                        const factor_options& opts = {},
                        checkpoint* cp = nullptr) {
    return detail::factorize_operator<Graph>(L, [&](auto&& G) {
        return factorize<Partitioner>(std::move(G), elim, opts, cp);
    });
}

template<partitioner P, typename Graph = graph<>>
factorization factorize(const Eigen::SparseMatrix<double>& L, P part,
                        const factor_options& opts = {},
                        checkpoint* cp = nullptr) {
    return detail::factorize_operator<Graph>(L, [&](auto&& G) {
        return factorize(std::move(G), std::move(part), opts, cp);
    });
}

template<partitioner P, eliminator E, typename Graph = graph<>>
factorization factorize(const Eigen::SparseMatrix<double>& L, P part,
                        const E& elim,
                        const factor_options& opts = {},
                        checkpoint* cp = nullptr) {
    return detail::factorize_operator<Graph>(L, [&](auto&& G) {
        return factorize(std::move(G), std::move(part), elim, opts, cp);
    });
}

/// Runtime-dispatch overload for graphs: picks partitioner by name from
/// factor_options::is_select at runtime via dispatch_partitioner.
template<incidence_storage Incidence>
factorization factorize_with_strategy(graph<Incidence> G,
                                      const factor_options& opts = {},
                                      checkpoint* cp = nullptr);

/// Runtime partitioner dispatch with a custom eliminator.
template<eliminator E, incidence_storage Incidence>
factorization factorize_with_strategy(graph<Incidence> G, const E& elim,
                                      const factor_options& opts = {},
                                      checkpoint* cp = nullptr);

/// Runtime-dispatch overload: picks graph backend from graph_storage enum
/// and partitioner by name from factor_options::is_select.
factorization factorize(const Eigen::SparseMatrix<double>& L,
                        graph_storage storage,
                        const factor_options& opts = {},
                        checkpoint* cp = nullptr);

namespace detail {

struct factor_entry {
    node_index neighbor;
    factor_value_t value;
};
static_assert(std::is_trivially_destructible_v<factor_entry>);

struct factor_col {
    node_index vertex;
    // L diagonal: sqrt(total_deg) including SDDM excess. Stored at the factor's
    // precision (factor_value_t: fp32),
    // Like the arena entries below, it is computed in fp64 and only copied into
    // sparse_csc::vals_ of that same type, so narrowing here is bit-identical
    // (and keeps the per-column header compact; there are n of them).
    factor_value_t diag = 0;
    // Exact-size off-diagonal ranges come from one monotonic resource per
    // elimination thread. They remain valid until assembly and avoid both one
    // heap allocation per vertex and geometric append-buffer growth.
    factor_entry* entries = nullptr;
    node_index entry_count = 0;
};

// Build elimination-order permutation and assemble L in CSC format.
// Precondition: factor_cols contains exactly one entry per vertex (all
// n vertices have been eliminated before this call).
void build_csc(factorization& result,
               const std::vector<factor_col>& factor_cols,
               node_index n,
               checkpoint* cp);

} // namespace detail

} // namespace apxchol

#include "apxchol/solver/factorization_impl.h"
