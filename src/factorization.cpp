#include "apxchol/solver/factorization.h"
#include "apxchol/solver/elimination/elimination.h"
#include "apxchol/graph/graph.h"
#include "apxchol/checkpoint.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace apxchol {

namespace detail {

// Threshold-free production form. CSC assembly already needs a parallel team
// for its independent column writes; use that same team for every invariant
// prepass instead of paying extra launches and then guessing an amortization
// cutoff. At one OpenMP thread this is the serial algorithm naturally.
static void assemble_csc(
        factorization& result,
        const std::vector<factor_col>& factor_cols,
        node_index n,
        checkpoint* cp) {
    auto& L = result.L;
    auto& perm = result.perm;
    assert(factor_cols.size() == static_cast<std::size_t>(n));

    const std::uint64_t edge_max =
        static_cast<std::uint64_t>(sparse_csc::kEdgeMax);
    const std::uint64_t nc = factor_cols.size();
#ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
#else
    const int max_threads = 1;
#endif
    std::vector<std::uint64_t> block_total(
        static_cast<std::size_t>(max_threads), 0);
    std::vector<unsigned char> block_overflow(
        static_cast<std::size_t>(max_threads), 0);
    std::vector<edge_index> block_base(
        static_cast<std::size_t>(max_threads) + 1, 0);

    bool overflow = false;
    edge_index nnz = 0;
    edge_index* outer_ptr = nullptr;
    node_index* inner_idx = nullptr;
    factor_value_t* values = nullptr;
    std::exception_ptr error;

    #pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
        const int team = omp_get_num_threads();
#else
        const int tid = 0;
        const int team = 1;
#endif
        const std::uint64_t begin =
            nc * static_cast<std::uint64_t>(tid) /
            static_cast<std::uint64_t>(team);
        const std::uint64_t end =
            nc * static_cast<std::uint64_t>(tid + 1) /
            static_cast<std::uint64_t>(team);

        // One contiguous pass builds the inverse permutation and this
        // worker's checked nnz subtotal (including one diagonal per column).
        std::uint64_t local_total = 0;
        bool local_overflow = false;
        for (std::uint64_t c = begin; c < end; ++c) {
            const auto& col = factor_cols[static_cast<std::size_t>(c)];
            perm[col.vertex] = static_cast<node_index>(c);
            const std::uint64_t count =
                static_cast<std::uint64_t>(col.entry_count);
            if (count == UINT64_MAX ||
                local_total > UINT64_MAX - (count + 1)) {
                local_overflow = true;
            } else {
                local_total += count + 1;
            }
        }
        block_total[static_cast<std::size_t>(tid)] = local_total;
        block_overflow[static_cast<std::size_t>(tid)] = local_overflow;

        #pragma omp barrier
        #pragma omp master
        {
            try {
                if (cp) (*cp)("permutation");
                std::uint64_t total = 0;
                for (int t = 0; t < team; ++t) {
                    const std::uint64_t add =
                        block_total[static_cast<std::size_t>(t)];
                    if (block_overflow[static_cast<std::size_t>(t)] ||
                        total > edge_max || add > edge_max - total) {
                        overflow = true;
                        break;
                    }
                    block_base[static_cast<std::size_t>(t)] =
                        static_cast<edge_index>(total);
                    total += add;
                }
                if (!overflow) {
                    block_base[static_cast<std::size_t>(team)] =
                        static_cast<edge_index>(total);
                    nnz = static_cast<edge_index>(total);
                    L.resize(n, n);
                    L.reserve(nnz);
                    outer_ptr = L.outerIndexPtr();
                    outer_ptr[n] = nnz;
                    // sparse_csc owns independent vectors for the outer,
                    // inner, and value arrays, so all storage can be acquired
                    // in this one handoff before any worker writes it.
                    L.resizeNonZeros(nnz);
                    inner_idx = L.innerIndexPtr();
                    values = L.valuePtr();
                }
            } catch (...) {
                error = std::current_exception();
            }
        }
        #pragma omp barrier

        // The barrier publishes both the primary thread's overflow decision
        // and allocation. Every worker therefore takes the same branch.
        if (!overflow && !error) {
            edge_index pos = block_base[static_cast<std::size_t>(tid)];
            for (std::uint64_t c = begin; c < end; ++c) {
                outer_ptr[c] = pos;
                pos += static_cast<edge_index>(
                    factor_cols[static_cast<std::size_t>(c)].entry_count) + 1;
            }
            assert(pos == block_base[static_cast<std::size_t>(tid) + 1]);

            #pragma omp barrier

            // Each factor_col writes to one distinct CSC column. Catch a
            // worker allocation failure, finish the workshare collectively,
            // and rethrow only after leaving OpenMP.
            std::vector<std::pair<node_index, factor_value_t>> col_entries;
            #pragma omp for schedule(dynamic, 256)
            for (std::ptrdiff_t i = 0;
                 i < static_cast<std::ptrdiff_t>(nc); ++i) {
                try {
                    const auto& col =
                        factor_cols[static_cast<std::size_t>(i)];
                    const node_index perm_col =
                        static_cast<node_index>(i);
                    assert(perm[col.vertex] == perm_col);

                    col_entries.clear();
                    col_entries.reserve(col.entry_count);
                    for (node_index j = 0; j < col.entry_count; ++j) {
                        const auto& [nbr, val] = col.entries[j];
                        col_entries.emplace_back(
                            perm[nbr], static_cast<factor_value_t>(-val));
                    }
                    std::sort(col_entries.begin(), col_entries.end());

                    edge_index write = outer_ptr[perm_col];
                    inner_idx[write] = perm_col;
                    values[write] = col.diag;
                    ++write;
                    for (auto [row, val] : col_entries) {
                        inner_idx[write] = row;
                        values[write] = val;
                        ++write;
                    }
                    assert(write == outer_ptr[perm_col + 1]);
                } catch (...) {
                    #pragma omp critical(apxchol_csc_assembly_exception)
                    {
                        if (!error)
                            error = std::current_exception();
                    }
                }
            }
        }
    }

    if (error)
        std::rethrow_exception(error);
    if (overflow)
        edge_index_overflow("assemble_csc(nnz)");

    // Keep the historical checkpoint boundary: all worker-local scratch has
    // been destroyed and the OpenMP team has joined before assembly ends.
    // These calls are outside an OpenMP structured block, so exceptions retain
    // their ordinary propagation semantics.
    L.makeCompressed();
    if (cp) (*cp)("assembly");
}

void build_csc(factorization& result,
               const std::vector<factor_col>& factor_cols,
               node_index n,
               checkpoint* cp) {
    // Permutation: perm[original_vertex] = new (elimination-order) index.
    // factor_cols is already in elimination order, so position i holds the
    // i-th eliminated vertex.
    result.perm.assign(n, 0);
    if (std::getenv("APXCHOL_VERBOSE"))
        std::fprintf(stderr,
                     "[apxchol] factor CSC assembly: fused invariant team\n");
    assemble_csc(result, factor_cols, n, cp);
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
    // The scan and optional M-matrix copy/lumping are required by apxchol's
    // preconditioner (not common benchmark parsing), so they belong to setup.
    // Keep the phase visible rather than hiding it inside make_graph.
    if (cp) { cp->descend("setup"); cp->tick(); }
    const operator_view op(L);
    if (cp) (*cp)("operator_prepare");

    // make_graph runs BEFORE factorize_with_strategy's own descend("setup") —
    // without its own bracket, its cost (~700 ms on IPM iter40: full
    // Eigen-sparse → graph conversion) shows up as un-accounted wall time
    // outside the profile. Wrap it here so the bench's setup_time captures it.
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
