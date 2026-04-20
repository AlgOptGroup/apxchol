#pragma once
/// Template implementation of the Kyng-Sachdeva approximate Cholesky factorization.
///
/// Included from factorization.h so that third-party code can use custom
/// incidence_storage backends without modifying our explicit-instantiation list.
/// The four built-in backends (vec, forward_star, small_vec, bstr) are
/// pre-instantiated in factorization.cpp; any other backend will be
/// instantiated on demand when the user includes <apxchol/solver/factorization.h>.

#include "apxchol/solver/factorization.h"
#include "apxchol/solver/elimination/elimination.h"
#include "apxchol/solver/is/block_greedy.h"
#include "apxchol/solver/is/luby.h"
#include "apxchol/solver/is/baumann_kyng.h"
#include "apxchol/solver/is/rootset.h"
#include "apxchol/graph/graph.h"
#include "apxchol/checkpoint.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <span>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace apxchol {

namespace detail {
// Eliminate vertices in the IS: record L-columns, add clique edges.
// IS vertices are pairwise non-adjacent, enabling parallel processing.
// The computation phase (gather neighbors, build factor column, sample
// clique edges) runs in parallel with per-thread RNGs; graph mutation
// (add_edge, deactivate) is deferred and applied sequentially because
// the forward_star node pool is shared.

// Deferred clique edge type is defined in elimination.h.

// Core per-vertex elimination: gather neighbors, build L-column, sample
// clique edges via the given Eliminator strategy.
// Caller must merge parallel edges for v before calling this.
//
// For SDDM matrices, the excess diagonal (self-loop to ground) is
// included in the degree used for the L-column entries.  Excess is
// propagated to each neighbor proportionally: Δexcess[u] = w_u · e_v / d_total.
// The caller must apply the returned excess updates (deferred for thread safety).
template<typename Eliminator, typename Incidence>
void process_vertex(const Eliminator& elim,
                    graph<Incidence>& G,
                    node_index v,
                    factor_col& col,
                    std::mt19937& gen,
                    std::vector<deferred_edge>& edges_out,
                    std::vector<std::pair<node_index, double>>& valid,
                    std::vector<double>& prefix,
                    std::vector<std::pair<node_index, double>>& excess_out) {
    valid.clear();
    double edge_deg = 0.0;
    for (auto [u, w] : G.neighbors(v)) {
        valid.emplace_back(u, w);
        edge_deg += w;
    }

    if (valid.empty()) {
        // Isolated vertex: use excess diagonal, or regularize to 1.0
        // to prevent division-by-zero in the triangular solve.
        // (Can happen with aggressive clique sparsifiers.)
        double d = G.excess(v);
        col = {v, d > 0.0 ? std::sqrt(d) : 1.0, {}};
        return;
    }

    std::sort(valid.begin(), valid.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    double total_deg = edge_deg + G.excess(v);
    if (total_deg <= 0.0) total_deg = 1.0; // guard against numerically zero degree
    double sqrt_deg = std::sqrt(total_deg);
    col.vertex = v;
    col.diag = sqrt_deg;
    col.entries.reserve(valid.size());
    for (const auto& [u, w] : valid)
        col.entries.emplace_back(u, w / sqrt_deg);

    elim.sample_clique(valid, total_deg, gen, edges_out, prefix);

    // Propagate excess to neighbors (deferred for thread safety).
    double ev = G.excess(v);
    if (ev > 0.0) {
        for (const auto& [u, w] : valid)
            excess_out.emplace_back(u, w * ev / total_deg);
    }
}

template<typename Eliminator, typename Incidence>
void eliminate_set(const Eliminator& elim,
                   graph<Incidence>& G,
                   const std::vector<node_index>& is,
                   std::vector<factor_col>& factor_cols,
                   std::mt19937& rng,
                   const factor_options& opts,
                   checkpoint* cp = nullptr) {
    if (cp) { cp->descend("eliminate"); cp->tick(); }

    // Merge parallel edges on IS vertices for consistent weights.
    // IS vertices are pairwise non-adjacent → disjoint edge sets → safe to parallelize.
    #pragma omp parallel for schedule(static) if(is.size() > opts.omp_threshold)
    for (size_t i = 0; i < is.size(); ++i)
        G.merge_parallel_edges(is[i]);
    if (cp) (*cp)("merge_is");

    const size_t n_is = is.size();
    std::vector<factor_col> local_cols(n_is);

    #ifdef _OPENMP
    if (n_is > opts.omp_threshold) {
        // Parallel path: per-thread RNGs seeded deterministically from main rng.
        int num_threads = omp_get_max_threads();
        std::vector<unsigned> seeds(num_threads);
        for (int t = 0; t < num_threads; ++t)
            seeds[t] = rng();
        std::vector<std::vector<deferred_edge>> edge_buffers(num_threads);
        std::vector<std::vector<std::pair<node_index, double>>> excess_buffers(num_threads);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::mt19937 local_rng(seeds[tid]);
            std::vector<std::pair<node_index, double>> valid;
            std::vector<double> prefix;

            #pragma omp for schedule(static)
            for (size_t k = 0; k < n_is; ++k)
                process_vertex(elim, G, is[k], local_cols[k], local_rng,
                               edge_buffers[tid], valid, prefix,
                               excess_buffers[tid]);
        }
        if (cp) (*cp)("compute");

        for (auto& col : local_cols)
            factor_cols.push_back(std::move(col));
        for (auto& buf : edge_buffers)
            for (auto [u, v, w] : buf)
                G.add_edge(u, v, w);
        for (auto& buf : excess_buffers)
            for (auto [v, delta] : buf)
                G.excess(v) += delta;
        for (auto u : is)
            G.deactivate(u);
        if (cp) (*cp)("apply");
    } else
    #endif
    {
        // Serial path: per-vertex inline application for cache locality.
        std::vector<std::pair<node_index, double>> valid;
        std::vector<double> prefix;
        std::vector<deferred_edge> vertex_edges;
        std::vector<std::pair<node_index, double>> excess_updates;
        for (size_t k = 0; k < n_is; ++k) {
            vertex_edges.clear();
            excess_updates.clear();
            process_vertex(elim, G, is[k], local_cols[k], rng,
                           vertex_edges, valid, prefix, excess_updates);
            factor_cols.push_back(std::move(local_cols[k]));
            for (auto [u, v, w] : vertex_edges)
                G.add_edge(u, v, w);
            for (auto [v, delta] : excess_updates)
                G.excess(v) += delta;
            G.deactivate(is[k]);
        }
        if (cp) (*cp)("compute");
    }

    if (cp) cp->ascend();
}

// Sequential fallback: eliminate all remaining vertices one by one.
// Used when IS fraction drops below threshold — avoids the O(|active|)
// IS-finding scan when only a few vertices can be chosen anyway.
template<typename Eliminator, typename Incidence>
void eliminate_remaining(const Eliminator& elim,
                         graph<Incidence>& G,
                         std::span<const node_index> active,
                         std::vector<factor_col>& factor_cols,
                         std::mt19937& rng) {
    std::vector<std::pair<node_index, double>> valid;
    std::vector<double> prefix;
    std::vector<deferred_edge> edges;
    std::vector<std::pair<node_index, double>> excess_updates;
    factor_col col;
    for (auto v : active) {
        G.merge_parallel_edges(v);
        edges.clear();
        excess_updates.clear();
        col.entries.clear();
        process_vertex(elim, G, v, col, rng, edges, valid, prefix, excess_updates);
        factor_cols.push_back(std::move(col));
        for (auto [a, b, w] : edges)
            G.add_edge(a, b, w);
        for (auto [u, delta] : excess_updates)
            G.excess(u) += delta;
        G.deactivate(v);
    }
}

} // namespace detail

template<typename ISSelector, typename Eliminator, incidence_storage Incidence>
factorization factorize_impl(const Eliminator& elim,
                             const graph<Incidence>& G,
                             const factor_options& opts,
                             checkpoint* cp) {
    const index_t n = G.n();
    if (n == 0)
        return {};

    if (cp) cp->descend("setup");

    factorization result;
    graph<Incidence> work(G);

    // Detect SDDM: any vertex with positive excess means the matrix
    // is positive definite (not just semidefinite like a Laplacian).
    // Note: make_graph already filters out FP noise (excess < diag * 1e-12),
    // so any remaining positive excess is genuine.
    for (index_t v = 0; v < n; ++v) {
        if (work.excess(v) > 0.0) { result.sddm = true; break; }
    }
    if (cp) (*cp)("build_adj");

    std::vector<detail::factor_col> factor_cols;
    factor_cols.reserve(n);

    std::mt19937 rng(opts.seed);

    ISSelector selector;

    // Active vertex list — filtered in-place after each round.
    std::vector<node_index> active(n);
    std::ranges::iota(active, node_index{0});

    while (active.size() > 1) {
        auto [is, avg_deg] = detail::find_independent_set(selector, work, active, opts, cp);

        result.rounds.push_back({active.size(), is.size(), avg_deg});

        // If IS is too small, the IS-finding overhead exceeds the benefit
        // of batch elimination.  Fall back to sequential elimination of all
        // remaining vertices (handled after the loop).
        if (is.size() < active.size() * opts.min_is_fraction)
            break;

        detail::eliminate_set(elim, work, is, factor_cols, rng, opts, cp);

        result.peak_graph_bytes = std::max(result.peak_graph_bytes,
                                           work.memory_bytes());

        std::erase_if(active, [&](node_index v) { return !work.is_active(v); });
    }

    // Eliminate any remaining vertices (0 or 1 after the while loop,
    // or all remaining when the IS-fraction fallback triggered above).
    if (!active.empty())
        detail::eliminate_remaining(elim, work, active, factor_cols, rng);
    if (cp) (*cp)("elim_remaining");

    detail::build_csc(result, factor_cols, n, cp);

    if (cp) cp->ascend();

    return result;
}

template<typename ISSelector, incidence_storage Incidence>
factorization factorize(const graph<Incidence>& G,
                        const factor_options& opts,
                        checkpoint* cp) {
    switch (opts.elim) {
    case elimination_strategy::star:
        return factorize_impl<ISSelector>(detail::star_elimination{}, G, opts, cp);
    case elimination_strategy::clique:
        return factorize_impl<ISSelector>(detail::clique_elimination{}, G, opts, cp);

    default:
        return factorize_impl<ISSelector>(detail::tree_elimination{}, G, opts, cp);
    }
}

template<incidence_storage Incidence>
factorization factorize_with_strategy(const graph<Incidence>& G,
                                      const factor_options& opts,
                                      checkpoint* cp) {
    switch (opts.is_select) {
    case is_strategy::luby:
        return factorize<luby_is>(G, opts, cp);
    case is_strategy::baumann_kyng:
        return factorize<baumann_kyng_is>(G, opts, cp);
    case is_strategy::rootset:
        return factorize<rootset_is>(G, opts, cp);
    default:
        return factorize<block_greedy_is>(G, opts, cp);
    }
}

} // namespace apxchol
