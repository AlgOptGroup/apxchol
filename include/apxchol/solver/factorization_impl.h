#pragma once
/// Template implementation of the Kyng-Sachdeva approximate Cholesky factorization.
///
/// Included from factorization.h so that third-party code can use custom
/// incidence_storage backends without modifying our explicit-instantiation list.
/// The three built-in backends (vec, forward_star, small_vec) are pre-instantiated
/// in factorization.cpp; any other backend will be instantiated on demand when
/// the user includes <apxchol/solver/factorization.h>.

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

// ── Algorithm constants (defaults; overridden by factor_options) ──

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
// Prunes dead edges and computes degrees (parallel), then uses a serial
// greedy scan: each eligible vertex (degree ≤ threshold) joins the IS
// if none of its active neighbors are already chosen.
//
// The greedy scan is inherently sequential but produces a larger IS than
// fully-parallel methods (Luby's gives ≈1/(d+1) fraction vs ≈1/2 for
// greedy on the eligible set), reducing overall round count.  Since the
// dominant parallel work (prune+degree, merge, elimination) is elsewhere,
// the serial scan cost is acceptable.
template<incidence_storage Incidence>
find_is_result find_independent_set(graph<Incidence>& G,
                                    std::span<const node_index> active,
                                    const factor_options& opts) {
    if (active.empty()) return {{}, 0.0};

    // Prune dead edges and compute degrees in a single pass (parallel).
    std::vector<int> degrees(active.size());
    double total_degree = 0;
    #pragma omp parallel reduction(+:total_degree) if(active.size() > opts.omp_threshold)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < active.size(); ++i) {
            degrees[i] = G.prune_and_degree(active[i]);
            total_degree += degrees[i];
        }
    }
    double degree_threshold = opts.degree_multiplier * total_degree
                            / static_cast<double>(active.size());

    std::vector<char> chosen(G.n(), 0);
    for (size_t i = 0; i < active.size(); ++i) {
        if (degrees[i] > degree_threshold)
            continue;
        auto v = active[i];
        bool ok = true;
        G.for_active_neighbors(v, [&](node_index u, double) {
            if (chosen[u]) ok = false;
        });
        if (ok) chosen[v] = 1;
    }

    std::vector<node_index> is;
    for (auto v : active)
        if (chosen[v]) is.push_back(v);
    return {std::move(is), total_degree / static_cast<double>(active.size())};
}

// Eliminate vertices in the IS: record L-columns, add clique edges.
// IS vertices are pairwise non-adjacent, enabling parallel processing.
// The computation phase (gather neighbors, build factor column, sample
// clique edges) runs in parallel with per-thread RNGs; graph mutation
// (add_edge, deactivate) is deferred and applied sequentially because
// the forward_star node pool is shared.
template<typename Incidence>
void eliminate_set(graph<Incidence>& G,
                   const std::vector<node_index>& is,
                   std::vector<factor_col>& factor_cols,
                   std::mt19937& rng,
                   const factor_options& opts,
                   checkpoint* cp = nullptr) {
    // Merge parallel edges on IS vertices for consistent weights.
    // IS vertices are pairwise non-adjacent → disjoint edge sets → safe to parallelize.
    // merge_parallel_edges uses filter (no pool growth), so forward_star is safe.
    #pragma omp parallel for schedule(static) if(is.size() > opts.omp_threshold)
    for (size_t i = 0; i < is.size(); ++i)
        G.merge_parallel_edges(is[i]);
    if (cp) (*cp)("merge_is");

    const size_t n_is = is.size();

    // Per-IS-vertex factor columns, indexed by position in IS.
    std::vector<factor_col> local_cols(n_is);

    // Deferred clique edges: (endpoint_a, endpoint_b, weight).
    struct deferred_edge { node_index a, b; double w; };

    // Core per-vertex logic: gather neighbors, build L-column, sample clique.
    auto process_vertex = [&](size_t k, std::mt19937& gen,
                              std::vector<deferred_edge>& edges_out,
                              std::vector<std::pair<node_index, double>>& valid,
                              std::vector<double>& prefix) {
        auto u = is[k];
        valid.clear();
        double deg = 0.0;
        G.for_active_neighbors(u, [&](node_index v, double w) {
            valid.emplace_back(v, w);
            deg += w;
        });

        if (valid.empty()) {
            local_cols[k] = {u, {}};
            return;
        }

        std::sort(valid.begin(), valid.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        double sqrt_deg = std::sqrt(deg);
        local_cols[k].vertex = u;
        local_cols[k].entries.reserve(valid.size());
        for (const auto& [v, w] : valid)
            local_cols[k].entries.emplace_back(v, w / sqrt_deg);

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
    };

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
                process_vertex(k, local_rng, edge_buffers[tid], valid, prefix);
        }
        if (cp) (*cp)("compute");

        // Apply deferred results sequentially.
        for (auto& col : local_cols)
            factor_cols.push_back(std::move(col));
        for (auto& buf : edge_buffers)
            for (auto [a, b, w] : buf)
                G.add_edge(a, b, w);
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
            process_vertex(k, rng, vertex_edges, valid, prefix);
            factor_cols.push_back(std::move(local_cols[k]));
            for (auto [a, b, w] : vertex_edges)
                G.add_edge(a, b, w);
            G.deactivate(is[k]);
        }
        if (cp) { (*cp)("compute"); cp->tick(); }
    }
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
    std::iota(active.begin(), active.end(), node_index{0});

    while (active.size() > 1) {
        auto [is, avg_deg] = detail::find_independent_set(work, active, opts);
        if (cp) (*cp)("find_is");

        // Stop if IS is empty or too small to make meaningful progress.
        // A tiny IS (< 0.1% of active) indicates the degree threshold is
        // too restrictive for this graph structure, and continuing would
        // produce O(n) rounds of O(1) eliminations.
        if (is.empty() || is.size() * 1000 < active.size()) break;

        result.rounds.push_back({
            static_cast<int>(active.size()),
            static_cast<int>(is.size()),
            avg_deg
        });

        checkpoint* elim_cp = nullptr;
        if (cp) { cp->descend("eliminate"); cp->tick(); elim_cp = cp; }
        detail::eliminate_set(work, is, factor_cols, rng, opts, elim_cp);
        if (cp) cp->ascend();

        std::erase_if(active, [&](node_index v) { return !work.is_active(v); });
    }

    // Build elimination order + append remaining active vertex
    std::vector<node_index> order;
    order.reserve(n);
    for (const auto& col : factor_cols)
        order.push_back(col.vertex);
    for (auto v : active)
        order.push_back(v);

    // Build permutation: map[original_vertex] = new_index
    std::vector<index_t> map(n);
    for (index_t i = 0; i < n; ++i)
        map[order[i]] = i;

    if (cp) (*cp)("permutation");

    // Count exact number of off-diagonal entries for precise reservation.
    index_t total_offdiag = 0;
    for (const auto& col : factor_cols)
        total_offdiag += static_cast<index_t>(col.entries.size());

    // Build sparse L directly from factor_cols — no intermediate copy.
    // Each factor column maps to a permuted column of L; neighbors
    // (still active when the vertex was eliminated) always have higher
    // permuted indices, producing a lower-triangular matrix.
    using T = Eigen::Triplet<double>;
    std::vector<T> trips;
    trips.reserve(total_offdiag + n); // off-diagonal + one diagonal per column

    std::vector<double> diag(n, 0.0);
    for (const auto& col : factor_cols) {
        index_t perm_col = map[col.vertex];
        for (const auto& [nbr, val] : col.entries) {
            trips.emplace_back(map[nbr], perm_col, -val);
            diag[perm_col] += val;
        }
    }
    for (index_t i = 0; i < n; ++i)
        trips.emplace_back(i, i, diag[i]);

    result.L.resize(n, n);
    result.L.setFromTriplets(trips.begin(), trips.end());
    result.L.makeCompressed();

    result.perm.resize(n);
    result.perm.indices() = Eigen::Map<Eigen::Matrix<index_t, Eigen::Dynamic, 1>>(map.data(), n);
    if (cp) (*cp)("assembly");

    if (cp) cp->ascend();

    return result;
}

} // namespace apxchol
