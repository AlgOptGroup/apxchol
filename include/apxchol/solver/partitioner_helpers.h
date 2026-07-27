#pragma once
/// Shared building blocks for partitioners that opt into the standard
/// prune-and-collect pipeline: degree-threshold computation, the parallel
/// prune-dead-edges pass, and chosen-mask collection — public, and custom
/// partitioners are welcome to them.  Sampling-based partitioners that
/// estimate degrees (baumann_kyng) skip the O(active + edges) pre-pass on
/// purpose; that choice is why it is not hoisted into the orchestrator.
/// (detail:: below additionally holds the internal bulk-apply of deferred
/// edge/excess updates used by the elimination phase.)

#include "apxchol/checkpoint.h"
#include "apxchol/graph/graph.h"
#include "apxchol/solver/factor_options.h"
#include "apxchol/solver/factorize_workspace.h"
#include "apxchol/solver/partition.h"
#include <algorithm>
#include <span>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace apxchol {

/// IS degree cap threshold. When popts.degree_quantile is in (0,1), returns the
/// degree at that quantile of the CURRENT active-degree distribution (found via
/// nth_element on a scratch copy, O(active)); else the legacy
/// degree_multiplier × avg_degree. A quantile cap admits a fixed low-degree
/// fraction regardless of the distribution, so it is safe on near-uniform-degree
/// graphs where a sub-1 multiplier would exclude every vertex (IS collapse).
inline double is_degree_threshold(std::span<const node_index> degrees,
                                  size_t n_active, double avg_deg,
                                  const partition_options& popts,
                                  std::vector<node_index>& scratch) {
    const double q = popts.degree_quantile;
    if (q > 0.0 && q < 1.0 && n_active > 0) {
        scratch.assign(degrees.begin(), degrees.begin() + n_active);
        size_t k = static_cast<size_t>(q * n_active);
        if (k >= scratch.size()) k = scratch.size() - 1;
        std::nth_element(scratch.begin(), scratch.begin() + k, scratch.end());
        return static_cast<double>(scratch[k]);
    }
    return popts.degree_multiplier * avg_deg;
}

/// Standard pre-pass: prune edges to eliminated vertices and fill
/// degrees[i] for i in [0, active.size()) — indexed by *position in
/// `active`*, not by vertex id.  Returns the average degree.
template<incidence_storage Incidence>
double prune_and_degrees(graph<Incidence>& G,
                         std::span<const node_index> active,
                         std::vector<node_index>& degrees,
                         std::size_t omp_threshold) {
    degrees.resize(active.size());
    double total_degree = 0;

    if constexpr (std::is_same_v<Incidence, forward_star_incidence>) {
        if (G.adj_filter_append_enabled()
            && active.size() > omp_threshold) {
            G.begin_parallel_append_adj();
        }
    }

    #pragma omp parallel for reduction(+:total_degree) schedule(dynamic, 256) \
        if(active.size() > omp_threshold)
    for (size_t i = 0; i < active.size(); ++i) {
        degrees[i] = G.prune_and_degree(active[i]);
        total_degree += degrees[i];
    }

    if constexpr (std::is_same_v<Incidence, forward_star_incidence>) {
        if (G.adj_filter_append_enabled()
            && active.size() > omp_threshold) {
            G.end_parallel_append_adj();
        }
    }
    return total_degree / double(active.size());
}

namespace detail {

/// Bulk-apply deferred edges from ws.threads[*].edge_buffer to G.
///
/// For forward_star: parallel prefix-sum across threads + single reservation
/// of edge/adj pool slots + per-thread atomic_push to each endpoint head.
///
/// For other backends: serial iteration calling G.add_edge for each entry.
///
/// All per-thread edge_buffers are cleared after apply.
template<incidence_storage Incidence>
void apply_deferred_edges(graph<Incidence>& G,
                          factorize_workspace& ws,
                          checkpoint* cp = nullptr) {
    const int num_threads = static_cast<int>(ws.threads.size());

    if constexpr (std::is_same_v<Incidence, forward_star_incidence>) {
        // Forward_star: parallel atomic apply via pre-reserved pool slots.
        std::vector<size_t> e_offsets(num_threads + 1, 0);
        for (int t = 0; t < num_threads; ++t)
            e_offsets[t + 1] = e_offsets[t] + ws.threads[t].edge_buffer.size();
        const size_t N_edges = e_offsets[num_threads];
        if (N_edges == 0) {
            if (cp) (*cp)("apply_edges");
            return;
        }

        const edge_index e_start = G.reserve_edge_pool(static_cast<edge_index>(N_edges));
        const edge_index a_start = G.reserve_adj_pool(static_cast<edge_index>(2 * N_edges));

        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            if (tid < num_threads) {
                const size_t base = e_offsets[tid];
                const auto& ebuf = ws.threads[tid].edge_buffer;
                for (size_t i = 0; i < ebuf.size(); ++i) {
                    auto [u, v, w] = ebuf[i];
                    const edge_index es = e_start + static_cast<edge_index>(base + i);
                    const edge_index as_u = a_start + static_cast<edge_index>(2 * (base + i));
                    const edge_index as_v = as_u + 1;
                    G.write_edge_at(es, u, v, w);
                    G.adj_push_atomic(u, as_u, es);
                    G.adj_push_atomic(v, as_v, es);
                }
            }
        }
        // Clear per-thread buffers (consumed).
        for (auto& t : ws.threads) t.edge_buffer.clear();
    } else if constexpr (std::is_same_v<Incidence, vec_pool_incidence>) {
        // vec_pool: serial pre-pass counts per-vertex incoming edges and
        // grows each slab to fit; parallel apply then uses atomic_push_reserved
        // for lock-free slot claim into the pre-sized slabs.
        std::vector<size_t> e_offsets(num_threads + 1, 0);
        for (int t = 0; t < num_threads; ++t)
            e_offsets[t + 1] = e_offsets[t] + ws.threads[t].edge_buffer.size();
        const size_t N_edges = e_offsets[num_threads];
        if (N_edges == 0) {
            if (cp) (*cp)("apply_edges");
            return;
        }

        // Step 1: count per-vertex incoming edges (each edge contributes to both endpoints).
        std::vector<node_index> incoming(static_cast<size_t>(G.n()), 0);
        for (int t = 0; t < num_threads; ++t)
            for (const auto& e : ws.threads[t].edge_buffer) {
                incoming[e.u]++;
                incoming[e.v]++;
            }

        // Step 2: pre-grow each slab so cap >= current_count + incoming.
        for (node_index v = 0; v < G.n(); ++v)
            if (incoming[v] > 0)
                G.adj_reserve_for(v, G.adj_count(v) + incoming[v]);

        // Step 3: reserve edge pool for all (u,v,w) tuples in one block, then
        // atomic-claim adj slots in parallel.
        const edge_index e_start = G.reserve_edge_pool(static_cast<node_index>(N_edges));

        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            if (tid < num_threads) {
                const size_t base = e_offsets[tid];
                const auto& ebuf = ws.threads[tid].edge_buffer;
                for (size_t i = 0; i < ebuf.size(); ++i) {
                    auto [u, v, w] = ebuf[i];
                    const edge_index es = e_start + static_cast<edge_index>(base + i);
                    G.write_edge_at(es, u, v, w);
                    G.adj_atomic_push_reserved(u, es);
                    G.adj_atomic_push_reserved(v, es);
                }
            }
        }
        for (auto& t : ws.threads) t.edge_buffer.clear();
    } else {
        // Legacy backends (vec/bstr): serial G.add_edge. Tried two
        // parallel schemes (per-(src,dst) shard relay + vertex-range scan);
        // both regress 30-60% on grid + IPM at T=8/16 vs serial. The bottleneck
        // is adj_[v].push_back() — per-vertex std::vector reallocs/memcpy
        // serialize on heap-allocator state under multi-thread pressure.
        // Forward_star's pre-allocated pool sidesteps this; vec cannot.
        for (auto& t : ws.threads) {
            for (auto [u, v, w] : t.edge_buffer)
                G.add_edge(u, v, w);
            t.edge_buffer.clear();
        }
    }
    if (cp) (*cp)("apply_edges");
}

/// Bulk-apply deferred excess updates from ws.threads[*].excess_buffer to G.
template<incidence_storage Incidence>
void apply_deferred_excess(graph<Incidence>& G,
                           factorize_workspace& ws,
                           checkpoint* cp = nullptr) {
    if constexpr (std::is_same_v<Incidence, forward_star_incidence>) {
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            if (tid < static_cast<int>(ws.threads.size())) {
                for (auto [u, delta] : ws.threads[tid].excess_buffer)
                    G.atomic_add_excess(u, delta);
                ws.threads[tid].excess_buffer.clear();
            }
        }
    } else {
        for (auto& t : ws.threads) {
            for (auto [u, delta] : t.excess_buffer)
                G.excess(u) += delta;
            t.excess_buffer.clear();
        }
    }
    if (cp) (*cp)("apply_excess");
}

} // namespace detail

} // namespace apxchol
