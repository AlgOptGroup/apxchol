#pragma once
/// Template implementation of the Kyng-Sachdeva approximate Cholesky factorization.
///
/// Included from factorization.h so that third-party code can use custom
/// incidence_storage backends without modifying our explicit-instantiation list.
/// The four built-in backends (vec, forward_star, small_vec, bstr) are
/// pre-instantiated in factorization.cpp; any other backend will be
/// instantiated on demand when the user includes <apxchol/solver/factorization.h>.

#include "apxchol/solver/factorization.h"
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

struct factor_col {
    node_index vertex;
    std::vector<std::pair<node_index, double>> entries; // (neighbor, L_value)
};

struct find_is_result {
    std::vector<node_index> is;
    double avg_degree;
};

// Select low-degree active vertices for elimination.
//
// Phase 1: Prune dead edges and compute degrees (parallel).
// Multi-threaded path (block-greedy):
//   Partition active vertices into per-thread blocks.  Each thread runs
//   serial greedy IS on its block, checking only within-block chosen
//   state (no cross-thread data races).  A fast serial pass then removes
//   the small number of cross-block boundary conflicts.
//   IS size ≈ serial greedy → same round count; IS selection parallelized.
// Single-threaded path:
//   Serial greedy scan — no overhead.
template<incidence_storage Incidence>
find_is_result find_independent_set(graph<Incidence>& G,
                                    std::span<const node_index> active,
                                    const factor_options& opts) {
    if (active.empty()) return {{}, 0.0};

    // Scratch arrays: persist across calls (static lifetime).
    // Only the main thread calls this; OMP regions inside access by index.
    static std::vector<char> chosen;
    static std::vector<int> block_of;
    static std::vector<char> near_boundary;
    static std::vector<int> degrees;

    if (chosen.size() < static_cast<size_t>(G.n())) {
        auto n = static_cast<size_t>(G.n());
        chosen.resize(n, 0);
        block_of.resize(n);
        near_boundary.resize(n, 0);
        degrees.resize(n);
    }

    // Phase 1: Prune dead edges and compute degrees (parallel).
    double total_degree = 0;
    #pragma omp parallel reduction(+:total_degree) if(active.size() > opts.omp_threshold)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < active.size(); ++i) {
            degrees[i] = G.prune_and_degree(active[i]);
            total_degree += degrees[i];
        }
    }
    double avg_degree = total_degree / static_cast<double>(active.size());
    double degree_threshold = opts.degree_multiplier * avg_degree;

    #ifdef _OPENMP
    if (omp_get_max_threads() > 1 && active.size() > opts.omp_threshold) {
        // ── Block-greedy parallel IS ──
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            auto bs = active.size() * static_cast<size_t>(tid) / nthreads;
            auto be = active.size() * static_cast<size_t>(tid + 1) / nthreads;

            for (size_t i = bs; i < be; ++i)
                block_of[active[i]] = tid;

            #pragma omp barrier

            for (size_t i = bs; i < be; ++i) {
                if (degrees[i] > degree_threshold) continue;
                auto v = active[i];
                bool ok = true;
                bool boundary = false;
                for (auto idx : G.adj(v)) {
                    auto u = G.edge_target(idx, v);
                    if (!G.is_active(u)) continue;
                    if (block_of[u] == tid) {
                        if (ok && chosen[u]) ok = false;
                    } else {
                        boundary = true;
                    }
                    if (!ok && boundary) break;
                }
                if (ok) {
                    chosen[v] = 1;
                    if (boundary) near_boundary[v] = 1;
                }
            }
        }

        // Fix cross-block conflicts (parallel — each vertex independent).
        // Flag removals in near_boundary (reuse: 1=boundary, 2=remove).
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < active.size(); ++i) {
            auto v = active[i];
            if (!chosen[v] || !near_boundary[v]) continue;
            int my_block = block_of[v];
            for (auto idx : G.adj(v)) {
                auto u = G.edge_target(idx, v);
                if (G.is_active(u) && chosen[u] && u < v && block_of[u] != my_block) {
                    near_boundary[v] = 2;
                    break;
                }
            }
        }

        // Apply removals.
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < active.size(); ++i) {
            auto v = active[i];
            if (near_boundary[v] == 2) chosen[v] = 0;
        }
    } else
    #endif
    {
        // ── Serial greedy ──
        for (size_t i = 0; i < active.size(); ++i) {
            if (degrees[i] > degree_threshold) continue;
            auto v = active[i];
            bool ok = true;
            for (auto idx : G.adj(v)) {
                auto u = G.edge_target(idx, v);
                if (G.is_active(u) && chosen[u]) { ok = false; break; }
            }
            if (ok) chosen[v] = 1;
        }
    }

    std::vector<node_index> is;

    // Collect IS vertices from chosen[] into is[].
    // Thread-local gather preserves ordering (active[] is sorted →
    // per-thread ranges are contiguous).
    {
        int nt_collect = 1;
        #ifdef _OPENMP
        nt_collect = omp_get_max_threads();
        #endif
        std::vector<std::vector<node_index>> local_is(nt_collect);
        #pragma omp parallel
        {
            int tid = 0, nthreads = 1;
            #ifdef _OPENMP
            tid = omp_get_thread_num();
            nthreads = omp_get_num_threads();
            #endif
            auto bs = active.size() * static_cast<size_t>(tid) / nthreads;
            auto be = active.size() * static_cast<size_t>(tid + 1) / nthreads;
            for (size_t i = bs; i < be; ++i)
                if (chosen[active[i]]) local_is[tid].push_back(active[i]);
        }
        for (auto& v : local_is)
            is.insert(is.end(), v.begin(), v.end());
    }

    // Reset scratch for next round (only touched entries, parallel).
    #pragma omp parallel for schedule(static) if(active.size() > opts.omp_threshold)
    for (size_t i = 0; i < active.size(); ++i) {
        chosen[active[i]] = 0;
        near_boundary[active[i]] = 0;
    }

    return {std::move(is), avg_degree};
}

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
            prefix.begin() + static_cast<ptrdiff_t>(i) + 1,
            prefix.end(),
            prefix[i] + r);
        size_t j = static_cast<size_t>(it - prefix.begin());
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
        if (cp) { (*cp)("compute"); cp->tick(); }
    }
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

// Build elimination-order permutation and assemble L in CSC format.
//
// factor_cols: one entry per eliminated vertex (in elimination order).
// active: remaining un-eliminated vertices (appended to order).
// n: total number of vertices in the original graph.
inline void build_csc(factorization& result,
                      const std::vector<factor_col>& factor_cols,
                      std::span<const node_index> active,
                      index_t n,
                      checkpoint* cp) {
    // Build elimination order: eliminated vertices first, then remaining.
    std::vector<node_index> order;
    order.reserve(n);
    for (const auto& col : factor_cols)
        order.push_back(col.vertex);
    for (auto v : active)
        order.push_back(v);

    // Permutation: map[original_vertex] = new_index.
    std::vector<index_t> perm(n);
    for (index_t i = 0; i < n; ++i)
        perm[order[i]] = i;

    if (cp) (*cp)("permutation");

    // Count exact number of off-diagonal entries for precise reservation.
    index_t total_offdiag = 0;
    for (const auto& col : factor_cols)
        total_offdiag += static_cast<index_t>(col.entries.size());

    const index_t nnz = total_offdiag + n; // off-diagonal + one diagonal per column

    // Build sparse L in CSC format directly — no triplets, no setFromTriplets.
    // Each factor column maps to a permuted column; entries are sorted by
    // permuted row index to satisfy Eigen's compressed format.
    result.L.resize(n, n);
    result.L.reserve(nnz);

    // Step 1: Count entries per column (off-diag + 1 diagonal each).
    std::vector<index_t> col_counts(n, 1); // every column has a diagonal
    for (const auto& col : factor_cols)
        col_counts[perm[col.vertex]] += static_cast<index_t>(col.entries.size());

    // Step 2: Build outer pointer array (cumulative sum).
    auto* outerPtr = result.L.outerIndexPtr();
    outerPtr[0] = 0;
    for (index_t c = 0; c < n; ++c)
        outerPtr[c + 1] = outerPtr[c] + col_counts[c];

    // Step 3: Allocate inner indices + values arrays.
    result.L.resizeNonZeros(nnz);
    auto* innerIdx = result.L.innerIndexPtr();
    auto* values   = result.L.valuePtr();

    // Step 4: Fill each column (diagonal + off-diag entries sorted by row).
    std::vector<index_t> write_pos(n);
    for (index_t c = 0; c < n; ++c)
        write_pos[c] = outerPtr[c];

    std::vector<double> diag(n, 0.0);
    std::vector<std::pair<index_t, double>> col_entries;

    for (const auto& col : factor_cols) {
        index_t perm_col = perm[col.vertex];

        // Accumulate diagonal and prepare sorted off-diagonal entries.
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

    // Remaining columns (un-eliminated vertices) — diagonal only.
    for (index_t c = 0; c < n; ++c) {
        if (write_pos[c] == outerPtr[c]) {
            innerIdx[write_pos[c]] = c;
            values[write_pos[c]]   = 0.0;
            ++write_pos[c];
        }
    }

    result.L.makeCompressed();

    result.perm.resize(n);
    result.perm.indices() = Eigen::Map<Eigen::Matrix<index_t, Eigen::Dynamic, 1>>(perm.data(), n);
    if (cp) (*cp)("assembly");
}

} // namespace detail

template<incidence_storage Incidence>
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

    // Active vertex list — filtered in-place after each round.
    std::vector<node_index> active(n);
    std::ranges::iota(active, node_index{0});

    while (active.size() > 1) {
        auto [is, avg_deg] = detail::find_independent_set(work, active, opts);
        if (cp) (*cp)("find_is");

        // If IS is empty or too small, the IS-finding overhead exceeds the
        // benefit of batch elimination.  Fall back to sequential elimination
        // of all remaining vertices.
        if (is.empty() || is.size() < active.size() * opts.min_is_fraction) {
            detail::eliminate_remaining(work, active, factor_cols, rng);
            active.clear();
            break;
        }

        result.rounds.push_back({
            static_cast<int>(active.size()),
            static_cast<int>(is.size()),
            avg_deg
        });

        checkpoint* elim_cp = nullptr;
        if (cp) { cp->descend("eliminate"); cp->tick(); elim_cp = cp; }
        detail::eliminate_set(work, is, factor_cols, rng, opts, elim_cp);
        if (cp) cp->ascend();

        result.peak_graph_bytes = std::max(result.peak_graph_bytes,
                                           work.memory_bytes());

        std::erase_if(active, [&](node_index v) { return !work.is_active(v); });
    }

    detail::build_csc(result, factor_cols, active, n, cp);

    if (cp) cp->ascend();

    return result;
}

} // namespace apxchol
