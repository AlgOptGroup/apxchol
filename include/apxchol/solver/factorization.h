#pragma once
#include "apxchol/graph/conversions.h"
#include "apxchol/operator_class.h"
#include "apxchol/solver/elimination/elimination.h"
#include "apxchol/solver/factor_options.h"
#include "apxchol/solver/detail/setup_route.h"
#include "apxchol/solver/detail/factor_entry.h"
#include "apxchol/solver/partitioner_list.h"
#include "apxchol/sparse_csc.h"
#include <Eigen/Sparse>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace apxchol {

struct checkpoint;
#if defined(APXCHOL_USE_CUDA)
class cuda_sptrsv_device_factor;
#endif

/// Approximate Cholesky preconditioner: P^T L L^T P approximates the operator
/// used for construction, including any lumping or regularization.
struct factorization {
#if defined(APXCHOL_USE_CUDA)
    // Internal CUDA ownership capsule. Exported factors retain host L values;
    // consuming solves may retain only L's dimensions and column pointers.
    std::shared_ptr<cuda_sptrsv_device_factor> research_device_factor;
#endif
    sparse_csc L;                                           // lower-triangular factor (owned CSC)
    // Elimination-order permutation: perm[original_vertex] = new_position.
    //   to permuted space   : x_perm[perm[v]] = b[v]   (scatter)
    //   from permuted space : x[v] = x_perm[perm[v]]    (gather)
    std::vector<node_index> perm;

    // True if any vertex has positive diagonal excess; false for a pure Laplacian.
    // Affects preconditioner rank and centering.
    bool sddm = false;

    // Peak graph memory during factorization (bytes, heap only).
    std::size_t peak_graph_bytes = 0;

    // Stored positive off-diagonal entries removed by M-matrix lumping (two
    // per pair). Only the preconditioner changes; the solve operator is untouched.
    Eigen::Index lumped_offdiag = 0;

    // Per-round statistics. active/is_size are always populated; Metal leaves
    // avg_deg as NaN when unavailable. nnz_added/nnz_total may be zero without
    // a checkpoint. is_size drives the SpTRSV round-as-level path.
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
template<typename Partitioner = block_greedy_partitioner>
factorization factorize(graph G,
                        const factor_options& opts = {},
                        checkpoint* cp = nullptr);

namespace detail {

/// Shared matrix-input path: validate, lump if needed, and build the graph with
/// regularization. The caller's operator remains unchanged for PCG.
template<typename Build>
factorization factorize_operator(const Eigen::SparseMatrix<double>& L,
                                 const factor_options& options, Build&& build) {
    validate_factor_options(options);
    const operator_view op(L, options.lump_max_mass);
    // The graph builder applies these matrix-only options.
    auto graph_options = options;
    graph_options.regularization = 0;
    graph_options.lump_max_mass = kDefaultLumpMassCeiling;
    factorization F = build(make_graph_from_operator(op.matrix(),
        &op.matrix() == &L && op.scan().triangles_bit_identical, options.regularization), graph_options);
    F.lumped_offdiag = op.lumped();
    return F;
}

} // namespace detail

/// Custom-eliminator overload: substitute your own star-vertex elimination
/// rule (see elimination/elimination.h for the `eliminator` contract).
/// The default overloads build the tree_elimination rule from `opts`.
template<typename Partitioner = block_greedy_partitioner, eliminator E>
factorization factorize(graph G, const E& elim,
                        const factor_options& opts = {},
                        checkpoint* cp = nullptr);

/// Partitioner-instance overloads: pass a configured (possibly stateful)
/// partitioner — the seam for custom independent-set selection and custom
/// elimination order (see solver/partitioner.h for the concept and contract).
/// The instance is taken by value; state mutated during the run is not
/// observable afterwards — reference it through a member pointer if you need
/// to inspect it post-run.
template<partitioner P>
factorization factorize(graph G, P part,
                        const factor_options& opts = {},
                        checkpoint* cp = nullptr);

template<partitioner P, eliminator E>
factorization factorize(graph G, P part, const E& elim,
                        const factor_options& opts = {},
                        checkpoint* cp = nullptr);

/// Matrix overloads validate the operator, apply permitted lumping and
/// regularization, then forward the graph to the chosen partitioner/eliminator.
template<typename Partitioner = block_greedy_partitioner, eliminator E>
factorization factorize(const Eigen::SparseMatrix<double>& L, const E& elim,
                        const factor_options& opts = {},
                        checkpoint* cp = nullptr) {
    return detail::factorize_operator(L, opts, [&](auto&& G, const auto& graph_options) {
        return factorize<Partitioner>(std::move(G), elim, graph_options, cp);
    });
}

template<partitioner P>
factorization factorize(const Eigen::SparseMatrix<double>& L, P part,
                        const factor_options& opts = {},
                        checkpoint* cp = nullptr) {
    return detail::factorize_operator(L, opts, [&](auto&& G, const auto& graph_options) {
        return factorize(std::move(G), std::move(part), graph_options, cp);
    });
}

template<partitioner P, eliminator E>
factorization factorize(const Eigen::SparseMatrix<double>& L, P part,
                        const E& elim,
                        const factor_options& opts = {},
                        checkpoint* cp = nullptr) {
    return detail::factorize_operator(L, opts, [&](auto&& G, const auto& graph_options) {
        return factorize(std::move(G), std::move(part), elim, graph_options, cp);
    });
}

/// Matrix-level factorization using the built-in AoS graph and named partitioner.
factorization factorize(const Eigen::SparseMatrix<double>& L,
                        const factor_options& opts = {},
                        checkpoint* cp = nullptr);

namespace detail {

// Internal consuming route used only by the preconditioner. Public factorize()
// always retains exportable host arrays; the resident solve may omit them.
factorization factorize_for_solver(const Eigen::SparseMatrix<double>& L,
                                  const factor_options& opts,
                                  checkpoint* cp,
                                  setup_route route = setup_route::cpu,
                                  bool keep_factor = false);

struct factor_col {
    node_index vertex;
    // sqrt(total degree including SDDM excess), computed in FP64 and stored
    // at the factor's precision.
    factor_value_t diag = 0;
    // Off-diagonal entries borrow per-thread arena storage until CSC assembly.
    // Device-resident consuming factors may omit them while retaining entry_count.
    factor_entry* entries = nullptr;
    node_index entry_count = 0;
};

// Build elimination-order permutation and assemble L in CSC format.
// Precondition: factor_cols contains exactly one entry per vertex (all
// n vertices have been eliminated before this call).
void build_csc(factorization& result,
               const std::vector<factor_col>& factor_cols,
               node_index n,
               checkpoint* cp, bool build_values = true);

} // namespace detail

} // namespace apxchol

#include "apxchol/solver/factorization_impl.h"
