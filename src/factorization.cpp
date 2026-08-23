#include "apxchol/solver/factorization.h"
#include "apxchol/solver/elimination/elimination.h"
#include "apxchol/graph/graph.h"
#include "apxchol/checkpoint.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace apxchol {

namespace detail {

// Fill the CSC arrays for the lower-triangular factor L (our sparse_csc).
// Each factor_col maps to a permuted column; entries are sorted by permuted
// row index to keep each column in ascending-row order. Each factor_col writes
// to a distinct permuted column (perm is a true permutation), so the
// per-column writes are embarrassingly parallel.
//
// Offsets are edge_index (can exceed 2^31 on dense factors, e.g. com-Orkut);
// row indices are node_index. The cumulative nnz is overflow-checked.
static void assemble_csc(sparse_csc& L,
                         const std::vector<factor_col>& factor_cols,
                         const std::vector<node_index>& perm,
                         node_index n) {
    edge_index total_offdiag = 0;
    for (const auto& col : factor_cols) {
        const edge_index add = static_cast<edge_index>(col.entry_count);
        if (total_offdiag > sparse_csc::kEdgeMax - add)
            edge_index_overflow("assemble_csc(off-diagonal)");
        total_offdiag += add;
    }
    if (total_offdiag > sparse_csc::kEdgeMax - n)
        edge_index_overflow("assemble_csc(nnz)");
    const edge_index nnz = total_offdiag + n; // off-diagonal + one diagonal per column

    L.resize(n, n);
    L.reserve(nnz);

    // Count entries per column (off-diag + 1 diagonal each). Per-column counts
    // fit node_index (<= degree < n); the cumulative outer pointers are edge_index.
    std::vector<node_index> col_counts(n, 1);
    for (const auto& col : factor_cols)
        col_counts[perm[col.vertex]] += col.entry_count;

    // Build outer pointer array (cumulative sum). Bounded by nnz (checked above).
    auto* outerPtr = L.outerIndexPtr();
    outerPtr[0] = 0;
    for (node_index c = 0; c < n; ++c)
        outerPtr[c + 1] = outerPtr[c] + col_counts[c];

    // Allocate inner indices + values arrays.
    L.resizeNonZeros(nnz);
    auto* innerIdx = L.innerIndexPtr();
    auto* values   = L.valuePtr();

    // Fill each column in parallel: each factor_col writes to its own
    // distinct permuted-column slot, so there are no cross-thread conflicts.
    const std::ptrdiff_t nc = static_cast<std::ptrdiff_t>(factor_cols.size());
    #pragma omp parallel
    {
        std::vector<std::pair<node_index, factor_value_t>> col_entries;
        #pragma omp for schedule(dynamic, 256)
        for (std::ptrdiff_t i = 0; i < nc; ++i) {
            const auto& col = factor_cols[i];
            node_index perm_col = perm[col.vertex];

            col_entries.clear();
            col_entries.reserve(col.entry_count);
            for (node_index j = 0; j < col.entry_count; ++j) {
                const auto& [nbr, val] = col.entries[j];
                col_entries.emplace_back(perm[nbr], static_cast<factor_value_t>(-val));
            }
            std::sort(col_entries.begin(), col_entries.end());

            edge_index pos = outerPtr[perm_col];
            innerIdx[pos] = perm_col;
            values[pos]   = col.diag;     // diagonal; narrows to fp32 under the flag
            ++pos;
            for (auto [row, val] : col_entries) {
                innerIdx[pos] = row;
                values[pos]   = val;
                ++pos;
            }
        }
    }

    L.makeCompressed();
}

void build_csc(factorization& result,
               const std::vector<factor_col>& factor_cols,
               node_index n,
               checkpoint* cp) {
    // Permutation: perm[original_vertex] = new (elimination-order) index.
    // factor_cols is already in elimination order, so position i holds the
    // i-th eliminated vertex.
    result.perm.assign(n, 0);
    for (node_index i = 0; i < n; ++i)
        result.perm[factor_cols[i].vertex] = i;

    if (cp) (*cp)("permutation");

    assemble_csc(result.L, factor_cols, result.perm, n);
    if (cp) (*cp)("assembly");
}

} // namespace detail

// Explicit instantiations for factorize_with_strategy across all storage backends.
// Per-partitioner instantiation is no longer needed: dispatch_partitioner is a
// function template that is inlined into factorize_with_strategy's body.
// Definition lives in factorization_impl.h for on-demand instantiation of
// other (including user-provided) backends.
template factorization factorize_with_strategy<vec_incidence>(
    graph<vec_incidence>, const factor_options&, checkpoint*);
template factorization factorize_with_strategy<forward_star_incidence>(
    graph<forward_star_incidence>, const factor_options&, checkpoint*);
template factorization factorize_with_strategy<bstr_incidence>(
    graph<bstr_incidence>, const factor_options&, checkpoint*);
template factorization factorize_with_strategy<vec_pool_incidence>(
    graph<vec_pool_incidence>, const factor_options&, checkpoint*);
template factorization factorize_with_strategy<directed_vec_pool_incidence>(
    graph<directed_vec_pool_incidence>, const factor_options&, checkpoint*);

factorization factorize(const Eigen::SparseMatrix<double>& L,
                        graph_storage storage,
                        const factor_options& opts_in,
                        checkpoint* cp) {
    // Assert the operator contract and lump positive off-diagonals if the
    // matrix needs it. Same `operator_view` the header's factorize() overloads
    // use — one implementation, so the CLI, the C++ API and both bindings
    // cannot diverge. Throws (naming the failed condition) on a violation;
    // `op.matrix()` is L itself whenever nothing had to be lumped.
    //
    // Deliberately OUTSIDE the checkpoint bracket below: this is input
    // validation, not factorization work, and folding it into "make_graph"
    // would silently move the bench's setup_time.
    const operator_view op(L);

    // make_graph runs BEFORE factorize_with_strategy's own descend("setup") —
    // without its own bracket, its cost (~700 ms on IPM iter40: full
    // Eigen-sparse → graph conversion) shows up as un-accounted wall time
    // outside the profile. Wrap it here so the bench's setup_time captures it.
    if (cp) { cp->descend("setup"); cp->tick(); }
    factor_options opts = opts_in;

    // Move the throwaway make_graph result into factorize_with_strategy's
    // by-value graph parameter — no defensive deep-copy on the dispatch path.
    auto do_factorize = [&](auto&& G) {
        if (cp) { (*cp)("make_graph"); cp->ascend(); }
        factorization F = factorize_with_strategy(std::move(G), opts, cp);
        F.lumped_offdiag = op.lumped();
        return F;
    };

    const Eigen::SparseMatrix<double>& A = op.matrix();
    switch (storage) {
    case graph_storage::forward_star: {
        auto G = make_graph<graph<forward_star_incidence>>(A);
        return do_factorize(std::move(G));
    }
    case graph_storage::bstr: {
        auto G = make_graph<graph<bstr_incidence>>(A);
        return do_factorize(std::move(G));
    }
    case graph_storage::vec_pool: {
        auto G = make_graph<graph<vec_pool_incidence>>(A);
        return do_factorize(std::move(G));
    }
    case graph_storage::vec_pool_aos: {
        auto G = make_graph<graph<directed_vec_pool_incidence>>(A);
        return do_factorize(std::move(G));
    }
    default: {
        auto G = make_graph<graph<vec_incidence>>(A);
        return do_factorize(std::move(G));
    }
    }
}

} // namespace apxchol
