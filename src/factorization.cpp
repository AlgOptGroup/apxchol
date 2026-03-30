#include "apxchol/solver/factorization.h"
#include "apxchol/graph/graph.h"
#include "apxchol/checkpoint.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace apxchol {

namespace detail {

// Fill the Eigen CSC arrays for the lower-triangular factor L.
// Each factor_col maps to a permuted column; entries are sorted by
// permuted row index to satisfy Eigen's compressed-column format.
static void assemble_csc(Eigen::SparseMatrix<double>& L,
                         const std::vector<factor_col>& factor_cols,
                         const std::vector<index_t>& perm,
                         index_t n) {
    index_t total_offdiag = 0;
    for (const auto& col : factor_cols)
        total_offdiag += static_cast<index_t>(col.entries.size());

    const index_t nnz = total_offdiag + n; // off-diagonal + one diagonal per column

    L.resize(n, n);
    L.reserve(nnz);

    // Count entries per column (off-diag + 1 diagonal each).
    std::vector<index_t> col_counts(n, 1);
    for (const auto& col : factor_cols)
        col_counts[perm[col.vertex]] += static_cast<index_t>(col.entries.size());

    // Build outer pointer array (cumulative sum).
    auto* outerPtr = L.outerIndexPtr();
    outerPtr[0] = 0;
    for (index_t c = 0; c < n; ++c)
        outerPtr[c + 1] = outerPtr[c] + col_counts[c];

    // Allocate inner indices + values arrays.
    L.resizeNonZeros(nnz);
    auto* innerIdx = L.innerIndexPtr();
    auto* values   = L.valuePtr();

    // Fill each column (diagonal + off-diag entries sorted by row).
    std::vector<index_t> write_pos(n);
    for (index_t c = 0; c < n; ++c)
        write_pos[c] = outerPtr[c];

    std::vector<double> diag(n, 0.0);
    std::vector<std::pair<index_t, double>> col_entries;

    for (const auto& col : factor_cols) {
        index_t perm_col = perm[col.vertex];

        col_entries.clear();
        for (const auto& [nbr, val] : col.entries) {
            col_entries.emplace_back(perm[nbr], -val);
            diag[perm_col] += val;
        }
        std::sort(col_entries.begin(), col_entries.end());

        // Diagonal first (smallest row index for lower-triangular format).
        auto pos = write_pos[perm_col];
        innerIdx[pos] = perm_col;
        values[pos]   = diag[perm_col];
        ++pos;

        // Off-diagonal entries in sorted row order.
        for (auto [row, val] : col_entries) {
            innerIdx[pos] = row;
            values[pos]   = val;
            ++pos;
        }
        write_pos[perm_col] = pos;
    }

    L.makeCompressed();
}

void build_csc(factorization& result,
               const std::vector<factor_col>& factor_cols,
               index_t n,
               checkpoint* cp) {
    // Build elimination order from factor_cols (all n vertices).
    std::vector<node_index> order;
    order.reserve(n);
    for (const auto& col : factor_cols)
        order.push_back(col.vertex);

    // Permutation: map[original_vertex] = new_index.
    std::vector<index_t> perm(n);
    for (index_t i = 0; i < n; ++i)
        perm[order[i]] = i;

    if (cp) (*cp)("permutation");

    assemble_csc(result.L, factor_cols, perm, n);

    result.perm.resize(n);
    result.perm.indices() = Eigen::Map<Eigen::Matrix<index_t, Eigen::Dynamic, 1>>(perm.data(), n);
    if (cp) (*cp)("assembly");
}

} // namespace detail

// Pre-compiled instantiations for the three built-in backends.
// Other backends (custom containers, different SSO sizes, etc.) are
// instantiated on demand via the template definitions in factorization_impl.h.
template factorization factorize<vec_incidence>(
    const graph<vec_incidence>&, const factor_options&, checkpoint*);
template factorization factorize<forward_star_incidence>(
    const graph<forward_star_incidence>&, const factor_options&, checkpoint*);
template factorization factorize<small_vec_incidence>(
    const graph<small_vec_incidence>&, const factor_options&, checkpoint*);

factorization factorize(const Eigen::SparseMatrix<double>& L,
                        graph_storage storage,
                        const factor_options& opts,
                        checkpoint* cp) {
    if (L.rows() != L.cols())
        throw std::invalid_argument("factorize: matrix must be square");

    switch (storage) {
    case graph_storage::forward_star:
        return factorize<graph<forward_star_incidence>>(L, opts, cp);
    case graph_storage::small_vec:
        return factorize<graph<small_vec_incidence>>(L, opts, cp);
    default:
        return factorize<graph<vec_incidence>>(L, opts, cp);
    }
}

} // namespace apxchol
