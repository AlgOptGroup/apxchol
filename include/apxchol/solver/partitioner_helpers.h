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
#include <array>
#include <span>
#include <type_traits>
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

/// Exact kth-order statistic for unsigned degree values, using one radix
/// selection pass per byte. Unlike a histogram approximation this returns the
/// same threshold value as nth_element, including duplicate degrees. Scratch
/// is caller-owned so repeated factorization rounds do not allocate it again.
inline node_index parallel_degree_quantile(
        std::span<const node_index> degrees, size_t n_active, double q,
        std::vector<std::array<size_t, 256>>& histograms) {
    static_assert(std::is_unsigned_v<node_index>);
    size_t rank = static_cast<size_t>(q * n_active);
    if (rank >= n_active) rank = n_active - 1;

#ifndef _OPENMP
    std::vector<node_index> scratch(degrees.begin(), degrees.begin() + n_active);
    std::nth_element(scratch.begin(), scratch.begin() + rank, scratch.end());
    return scratch[rank];
#else
    histograms.resize(static_cast<size_t>(omp_get_max_threads()));
    std::array<size_t, 256> totals{};
    node_index prefix = 0;
    node_index mask = 0;

    #pragma omp parallel shared(rank, prefix, mask, totals, histograms)
    {
        const int tid = omp_get_thread_num();
        auto& local = histograms[static_cast<size_t>(tid)];
        for (int shift = int(sizeof(node_index) * 8) - 8;
             shift >= 0; shift -= 8) {
            local.fill(0);
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n_active; ++i) {
                const node_index x = degrees[i];
                if ((x & mask) == prefix)
                    ++local[static_cast<unsigned>(
                        (x >> shift) & node_index{255})];
            }

            #pragma omp single
            {
                totals.fill(0);
                const int team = omp_get_num_threads();
                for (int t = 0; t < team; ++t)
                    for (size_t b = 0; b < totals.size(); ++b)
                        totals[b] += histograms[static_cast<size_t>(t)][b];

                size_t below = 0;
                size_t bucket = 0;
                while (bucket + 1 < totals.size()
                       && rank >= below + totals[bucket]) {
                    below += totals[bucket];
                    ++bucket;
                }
                rank -= below;
                prefix |= static_cast<node_index>(bucket) << shift;
                mask |= node_index{255} << shift;
            }
            // The implicit barrier after single publishes the chosen prefix
            // and residual rank before the next byte pass.
        }
    }
    return prefix;
#endif
}

/// Fill vertex-indexed degrees and compact eligible vertices in the original
/// candidate order. The output vector stays at its high-water size; callers
/// use the returned prefix length. This preserves every partitioner's exact
/// candidate sequence.
inline size_t parallel_ordered_degree_filter(
        std::span<const node_index> active,
        std::span<const node_index> degrees,
        double threshold,
        std::vector<node_index>& live_degrees,
        std::vector<node_index>& eligible_storage,
        std::vector<size_t>& offsets) {
#ifndef _OPENMP
    if (eligible_storage.size() < active.size())
        eligible_storage.resize(active.size());
    size_t out = 0;
    for (size_t i = 0; i < active.size(); ++i) {
        live_degrees[active[i]] = degrees[i];
        if (degrees[i] <= threshold) eligible_storage[out++] = active[i];
    }
    return out;
#else
    const int max_threads = omp_get_max_threads();
    offsets.resize(static_cast<size_t>(max_threads) + 1);
    if (eligible_storage.size() < active.size())
        eligible_storage.resize(active.size());
    int used_threads = 1;

    #pragma omp parallel shared(offsets, eligible_storage, live_degrees, used_threads)
    {
        const int tid = omp_get_thread_num();
        const int team = omp_get_num_threads();
        const size_t begin = active.size() * static_cast<size_t>(tid)
                           / static_cast<size_t>(team);
        const size_t end = active.size() * static_cast<size_t>(tid + 1)
                         / static_cast<size_t>(team);
        size_t count = 0;
        for (size_t i = begin; i < end; ++i) {
            live_degrees[active[i]] = degrees[i];
            count += degrees[i] <= threshold;
        }
        offsets[static_cast<size_t>(tid) + 1] = count;

        #pragma omp barrier
        #pragma omp single
        {
            used_threads = team;
            offsets[0] = 0;
            for (int t = 0; t < team; ++t)
                offsets[static_cast<size_t>(t) + 1] +=
                    offsets[static_cast<size_t>(t)];
        }

        size_t out = offsets[static_cast<size_t>(tid)];
        for (size_t i = begin; i < end; ++i)
            if (degrees[i] <= threshold) eligible_storage[out++] = active[i];
    }
    return offsets[static_cast<size_t>(used_threads)];
#endif
}

/// Stable parallel replacement for erase_if(active, !is_live). Scratch and
/// active alternate storage, avoiding another O(|active|) allocation after the
/// first high-water mark.
template<typename IsLive>
void parallel_stable_active_filter(std::vector<node_index>& active,
                                   std::vector<node_index>& scratch,
                                   std::vector<size_t>& offsets,
                                   IsLive&& is_live) {
#ifndef _OPENMP
    std::erase_if(active, [&](node_index v) { return !is_live(v); });
#else
    const int max_threads = omp_get_max_threads();
    offsets.resize(static_cast<size_t>(max_threads) + 1);
    if (scratch.size() < active.size()) scratch.resize(active.size());
    int used_threads = 1;

    #pragma omp parallel shared(active, scratch, offsets, used_threads)
    {
        const int tid = omp_get_thread_num();
        const int team = omp_get_num_threads();
        const size_t begin = active.size() * static_cast<size_t>(tid)
                           / static_cast<size_t>(team);
        const size_t end = active.size() * static_cast<size_t>(tid + 1)
                         / static_cast<size_t>(team);
        size_t count = 0;
        for (size_t i = begin; i < end; ++i) count += is_live(active[i]);
        offsets[static_cast<size_t>(tid) + 1] = count;

        #pragma omp barrier
        #pragma omp single
        {
            used_threads = team;
            offsets[0] = 0;
            for (int t = 0; t < team; ++t)
                offsets[static_cast<size_t>(t) + 1] +=
                    offsets[static_cast<size_t>(t)];
        }

        size_t out = offsets[static_cast<size_t>(tid)];
        for (size_t i = begin; i < end; ++i)
            if (is_live(active[i])) scratch[out++] = active[i];
    }
    scratch.resize(offsets[static_cast<size_t>(used_threads)]);
    active.swap(scratch);
#endif
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

inline int factorize_thread_num() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

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
            const int tid = factorize_thread_num();
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
    } else if constexpr (is_vec_pool_incidence_v<Incidence>) {
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

        // Step 3: reserve the indexed edge pool when present, then
        // atomic-claim endpoint-incidence slots in parallel.
        edge_index e_start = 0;
        if constexpr (graph<Incidence>::stores_directed_incidence)
            G.record_edges_added(static_cast<edge_index>(N_edges));
        else
            e_start = G.reserve_edge_pool(static_cast<edge_index>(N_edges));

        #pragma omp parallel
        {
            const int tid = factorize_thread_num();
            if (tid < num_threads) {
                const size_t base = e_offsets[tid];
                const auto& ebuf = ws.threads[tid].edge_buffer;
                for (size_t i = 0; i < ebuf.size(); ++i) {
                    auto [u, v, w] = ebuf[i];
                    const edge_index es = e_start + static_cast<edge_index>(base + i);
                    if constexpr (graph<Incidence>::stores_directed_incidence) {
                        G.adj_atomic_push_directed(u, v, w);
                        G.adj_atomic_push_directed(v, u, w);
                    } else {
                        G.write_edge_at(es, u, v, w);
                        G.adj_atomic_push_reserved(u, es);
                        G.adj_atomic_push_reserved(v, es);
                    }
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
            const int tid = factorize_thread_num();
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
