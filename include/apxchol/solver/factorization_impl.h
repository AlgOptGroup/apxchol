#pragma once
/// Template implementation of the Kyng-Sachdeva approximate Cholesky factorization.
///
/// Included from factorization.h so that third-party code can use custom
/// incidence_storage backends without modifying our explicit-instantiation list.
/// The four built-in backends (vec, forward_star, small_vec, bstr) are
/// pre-instantiated in factorization.cpp; any other backend will be
/// instantiated on demand when the user includes <apxchol/solver/factorization.h>.

#include "apxchol/solver/factorization.h"
#include "apxchol/solver/is_block_greedy.h"
#include "apxchol/solver/is_luby.h"
#include "apxchol/solver/is_baumann_kyng.h"
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

// Deferred clique edge for batched application.
struct deferred_edge { node_index u, v; double w; };

// Core per-vertex elimination: gather neighbors, build L-column, sample
// clique edges via Kyng-Sachdeva random sparsification.
// Caller must merge parallel edges for v before calling this.
template<typename Incidence>
void process_vertex(graph<Incidence>& G,
                    node_index v,
                    factor_col& col,
                    std::mt19937& gen,
                    std::vector<deferred_edge>& edges_out,
                    std::vector<std::pair<node_index, double>>& valid,
                    std::vector<double>& prefix) {
    valid.clear();
    double deg = 0.0;
    for (auto [u, w] : G.neighbors(v)) {
        valid.emplace_back(u, w);
        deg += w;
    }

    if (valid.empty()) {
        col = {v, {}};
        return;
    }

    std::sort(valid.begin(), valid.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    double sqrt_deg = std::sqrt(deg);
    col.vertex = v;
    col.entries.reserve(valid.size());
    for (const auto& [u, w] : valid)
        col.entries.emplace_back(u, w / sqrt_deg);

    prefix.resize(valid.size());
    prefix[0] = valid[0].second;
    for (size_t i = 1; i < valid.size(); ++i)
        prefix[i] = prefix[i - 1] + valid[i].second;

    for (size_t i = 0; i + 1 < valid.size(); ++i) {
        double suffix_sum = prefix.back() - prefix[i];
        if (suffix_sum <= 0.0) continue;

        std::uniform_real_distribution<double> U(0.0, suffix_sum);
        double r = U(gen);

        auto it = std::upper_bound(
            prefix.begin() + ptrdiff_t(i) + 1,
            prefix.end(),
            prefix[i] + r);
        size_t j = it - prefix.begin();
        if (j >= valid.size()) j = valid.size() - 1;

        auto [va, wa] = valid[i];
        auto [vb, wb] = valid[j];
        double w = wa * wb / (wa + wb); // harmonic-mean weight
        edges_out.push_back({va, vb, w});
    }
}

template<typename Incidence>
void eliminate_set(graph<Incidence>& G,
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

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::mt19937 local_rng(seeds[tid]);
            std::vector<std::pair<node_index, double>> valid;
            std::vector<double> prefix;

            #pragma omp for schedule(static)
            for (size_t k = 0; k < n_is; ++k)
                process_vertex(G, is[k], local_cols[k], local_rng,
                               edge_buffers[tid], valid, prefix);
        }
        if (cp) (*cp)("compute");

        for (auto& col : local_cols)
            factor_cols.push_back(std::move(col));
        for (auto& buf : edge_buffers)
            for (auto [u, v, w] : buf)
                G.add_edge(u, v, w);
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
        for (size_t k = 0; k < n_is; ++k) {
            vertex_edges.clear();
            process_vertex(G, is[k], local_cols[k], rng,
                           vertex_edges, valid, prefix);
            factor_cols.push_back(std::move(local_cols[k]));
            for (auto [u, v, w] : vertex_edges)
                G.add_edge(u, v, w);
            G.deactivate(is[k]);
        }
        if (cp) (*cp)("compute");
    }

    if (cp) cp->ascend();
}

// Sequential fallback: eliminate all remaining vertices one by one.
// Used when IS fraction drops below threshold — avoids the O(|active|)
// IS-finding scan when only a few vertices can be chosen anyway.
template<typename Incidence>
void eliminate_remaining(graph<Incidence>& G,
                         std::span<const node_index> active,
                         std::vector<factor_col>& factor_cols,
                         std::mt19937& rng) {
    std::vector<std::pair<node_index, double>> valid;
    std::vector<double> prefix;
    std::vector<deferred_edge> edges;
    factor_col col;
    for (auto v : active) {
        G.merge_parallel_edges(v);
        edges.clear();
        col.entries.clear();
        process_vertex(G, v, col, rng, edges, valid, prefix);
        factor_cols.push_back(std::move(col));
        for (auto [a, b, w] : edges)
            G.add_edge(a, b, w);
        G.deactivate(v);
    }
}

} // namespace detail

template<typename ISSelector, incidence_storage Incidence>
factorization factorize(const graph<Incidence>& G,
                        const factor_options& opts,
                        checkpoint* cp) {
    const index_t n = G.n();
    if (n == 0)
        return {};

    if (cp) cp->descend("setup");

    factorization result;
    graph<Incidence> work(G);
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

        detail::eliminate_set(work, is, factor_cols, rng, opts, cp);

        result.peak_graph_bytes = std::max(result.peak_graph_bytes,
                                           work.memory_bytes());

        std::erase_if(active, [&](node_index v) { return !work.is_active(v); });
    }

    // Eliminate any remaining vertices (0 or 1 after the while loop,
    // or all remaining when the IS-fraction fallback triggered above).
    if (!active.empty())
        detail::eliminate_remaining(work, active, factor_cols, rng);

    detail::build_csc(result, factor_cols, n, cp);

    if (cp) cp->ascend();

    return result;
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
    default:
        return factorize<block_greedy_is>(G, opts, cp);
    }
}

} // namespace apxchol
