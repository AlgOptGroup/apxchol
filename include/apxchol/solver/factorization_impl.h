#pragma once
/// Template implementation of the Kyng-Sachdeva approximate Cholesky factorization.
///
/// Included from factorization.h for custom partitioners and eliminators.

#include "apxchol/solver/factorization.h"
#include "apxchol/detail/parallel_exception.h"
#include "apxchol/solver/elimination/elimination.h"
#include "apxchol/solver/elimination/gpu_round_shadow.h"
#include "apxchol/solver/partitioner_helpers.h"
#include "apxchol/solver/partitioner_list.h"
#include "apxchol/solver/factorize_workspace.h"
#include "apxchol/graph/graph.h"
#include "apxchol/checkpoint.h"
#include "apxchol/solver/detail/residual_policy.h"
#if defined(APXCHOL_USE_CUDA)
#include "apxchol/solver/gpu_block_frontend.h"
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

namespace apxchol {

namespace detail {

// process_vertex is the indivisible unit of elimination work. Size each small
// round's team from both quantities that can keep workers useful: independent
// pivots and their already-measured live adjacency work. Each additional
// worker must be backed by at least 4096 adjacency slots. A singleton therefore
// stays serial regardless of degree; a large independent set keeps the old
// full-team policy. This is a resource-allocation rule, not a matrix heuristic.
inline constexpr size_t kEliminationWorkPerThread = 4096;
inline constexpr long double kIncrementalDegreeBreakEvenRatio = 10.0L;
inline constexpr long double kIncrementalDegreeWorkRatio = 16.0L;
// The exact endpoint reduction below is a four-pass byte radix for the
// default 32-bit node_index.  Each worker contributes one 256-bin histogram
// and participates in three barriers on every pass.  Each endpoint is also
// scanned on every pass, so the pass count multiplies both the useful and
// fixed work; the amortization floor is one endpoint per histogram bin.
inline constexpr size_t kIncrementalDegreeRadixBins = 256;
inline constexpr size_t kIncrementalDegreeMinWorkPerWorker =
    kIncrementalDegreeRadixBins;

inline bool incremental_degree_traffic_worthwhile(std::size_t active,
                                                   double average_degree,
                                                   std::size_t selected_work) {
    if (selected_work == 0) return false;
    const long double prune_work =
        static_cast<long double>(active) * average_degree;
    return prune_work >=
        kIncrementalDegreeWorkRatio *
            static_cast<long double>(selected_work);
}

inline bool incremental_degree_traffic_can_improve(std::size_t active,
                                                    double average_degree,
                                                    std::size_t selected_work) {
    if (selected_work == 0) return false;
    const long double prune_work =
        static_cast<long double>(active) * average_degree;
    return prune_work >=
        kIncrementalDegreeBreakEvenRatio *
            static_cast<long double>(selected_work);
}

inline bool incremental_degree_parallel_work_worthwhile(
        std::size_t selected_work, std::size_t workers) {
    return workers != 0 &&
        selected_work / workers >= kIncrementalDegreeMinWorkPerWorker;
}

inline bool incremental_degree_worthwhile(std::size_t active,
                                          double average_degree,
                                          std::size_t selected_work,
                                          std::size_t workers) {
    return incremental_degree_traffic_worthwhile(
               active, average_degree, selected_work) &&
        incremental_degree_parallel_work_worthwhile(
               selected_work, workers);
}

inline constexpr size_t elimination_round_team_size(
        size_t vertices, size_t adjacency_work, size_t vertex_threshold,
        size_t available_threads) {
    if (available_threads <= 1 || vertices <= 1) return 1;
    if (vertices > vertex_threshold)
        return std::min(vertices, available_threads);
    const size_t workers_from_work =
        adjacency_work / kEliminationWorkPerThread;
    return std::max<size_t>(1, std::min(
        {vertices, available_threads, workers_from_work}));
}

inline constexpr int elimination_compute_chunk(
        size_t vertices, size_t vertex_threshold) {
    return vertices > vertex_threshold ? 64 : 1;
}

// Collective byte-radix over one exact-size endpoint stream. Every member of
// the current OpenMP team must call this function. Fixed contiguous input
// ranges and bucket-major/thread-major output ranges make the result
// deterministic while avoiding both per-thread vector growth and atomics.
inline void parallel_radix_sort_node_indices(
        std::vector<node_index>& values,
        std::vector<node_index>& scratch,
        std::vector<std::array<std::size_t, 256>>& histograms,
        int tid, int num_threads) {
    const std::size_t begin = values.size() * static_cast<std::size_t>(tid) /
                              static_cast<std::size_t>(num_threads);
    const std::size_t end = values.size() * static_cast<std::size_t>(tid + 1) /
                            static_cast<std::size_t>(num_threads);
    node_index* source = values.data();
    node_index* target = scratch.data();
    for (unsigned byte = 0; byte < sizeof(node_index); ++byte) {
        auto& local = histograms[static_cast<std::size_t>(tid)];
        local.fill(0);
        const unsigned shift = 8 * byte;
        for (std::size_t i = begin; i < end; ++i)
            ++local[(source[i] >> shift) & node_index{255}];
#pragma omp barrier
#pragma omp single
        {
            std::size_t prefix = 0;
            for (std::size_t bucket = 0; bucket < 256; ++bucket) {
                for (int worker = 0; worker < num_threads; ++worker) {
                    auto& slot = histograms[static_cast<std::size_t>(worker)]
                                           [bucket];
                    const std::size_t next = prefix + slot;
                    slot = prefix;
                    prefix = next;
                }
            }
        }
#pragma omp barrier
        for (std::size_t i = begin; i < end; ++i) {
            const node_index value = source[i];
            target[local[(value >> shift) & node_index{255}]++] = value;
        }
#pragma omp barrier
        std::swap(source, target);
    }
    assert(source == values.data());
}

// Eliminate vertices in the partition: record L-columns, add clique edges.
// Partition vertices across regions are pairwise non-adjacent (for singleton
// regions, every vertex is its own region — same constraint as the old IS).
// The computation phase (gather neighbors, build factor column, sample
// clique edges) runs in parallel with per-thread RNGs; graph mutation
// is deferred, then applied through the backend's bulk-insertion path.

// Deferred clique edge type is defined in elimination.h.

#ifndef NDEBUG
// Debug-only: verify the partitioner's contract — selected vertices must be
// pairwise non-adjacent (they are eliminated independently in parallel).
inline void assert_partition_independent(graph& G,
                                  const partition_result& part) {
    static thread_local std::vector<char> in_part;
    if (in_part.size() < size_t(G.n())) in_part.assign(G.n(), 0);
    for (auto v : part.data) in_part[v] = 1;
    for (auto v : part.data)
        for (auto idx : G.adj(v)) {
            auto u = idx.to;
            assert(!(G.is_active(u) && in_part[u]) &&
                   "partitioner contract violation: selected vertices are "
                   "adjacent (the partition must be an independent set)");
        }
    for (auto v : part.data) in_part[v] = 0;
}
#endif

// Core per-vertex elimination: gather neighbors, build L-column, sample
// clique edges via the given Eliminator strategy.
// Neighbors are deduplicated while gathering, preserving first-seen order.
//
// For SDDM matrices, the excess diagonal (self-loop to ground) is
// included in the degree used for the L-column entries.  Excess is
// propagated to each neighbor proportionally: Δexcess[u] = w_u · e_v / d_total.
// The caller must apply the returned excess updates (deferred for thread safety).
template<typename Eliminator>
void process_vertex(const Eliminator& elim,
                    graph& G,
                    node_index v,
                    std::uint64_t run_seed,
                    factor_col& col,
                    factorize_workspace::per_thread& ws,
                    std::span<node_index> degree_decrements = {}) {
    auto& nbrs       = ws.neighbors;
    auto& excess_out = ws.excess_buffer;
    col.entries = nullptr;
    col.entry_count = 0;
    nbrs.clear();
    std::size_t decrement_pos = 0;
    double edge_deg = 0.0;
    // Pooled slabs expose their raw size in O(1), so deduplicate in a
    // pivot-sized contiguous hash table. The old vertex-indexed table
    // was one random access into an n-entry array per incidence and
    // one such array per worker; it became a cache/TLB and RSS cost on
    // large graphs. Preserve first-seen neighbor order exactly.
    const size_t raw_count = G.adj(v).size();
    const size_t table_size = std::bit_ceil(
        std::max<size_t>(8, 2 * raw_count));
    if (ws.dedup_hash.size() < table_size) {
        ws.dedup_hash.assign(table_size, {});
        ws.dedup_hash_epoch = 1;
    } else if (++ws.dedup_hash_epoch == 0) {
        for (auto& entry : ws.dedup_hash) entry.stamp = 0;
        ws.dedup_hash_epoch = 1;
    }
    const size_t mask = table_size - 1;
    const std::uint32_t epoch = ws.dedup_hash_epoch;
    for (auto [u, w] : G.neighbors(v)) {
        if (!G.is_active(u)) continue;
        if (!degree_decrements.empty()) {
            assert(decrement_pos < degree_decrements.size());
            degree_decrements[decrement_pos++] = u;
        }
        size_t slot = (static_cast<std::uint64_t>(u) *
                       0x9e3779b97f4a7c15ULL) & mask;
        while (ws.dedup_hash[slot].stamp == epoch &&
               ws.dedup_hash[slot].key != u)
            slot = (slot + 1) & mask;
        auto& entry = ws.dedup_hash[slot];
        if (entry.stamp != epoch) {
            entry.stamp = epoch;
            entry.key = u;
            entry.value = static_cast<node_index>(nbrs.size());
            nbrs.emplace_back(u, w);
        } else {
            nbrs[entry.value].weight += w;
        }
        edge_deg += w;
    }

    assert(degree_decrements.empty() ||
           decrement_pos == degree_decrements.size());
    if (nbrs.empty()) {
        double d = G.excess(v);
        col.vertex = v;
        col.diag   = static_cast<factor_value_t>(d > 0.0 ? std::sqrt(d) : 1.0);
        return;
    }

    double total_deg = edge_deg + G.excess(v);
    if (total_deg <= 0.0) total_deg = 1.0;
    double sqrt_deg = std::sqrt(total_deg);
    col.vertex = v;
    col.diag = static_cast<factor_value_t>(sqrt_deg);
    col.entries = static_cast<factor_entry*>(ws.factor_entries->allocate(
        nbrs.size() * sizeof(factor_entry), alignof(factor_entry)));
    for (const auto& [u, w] : nbrs) {
        std::construct_at(col.entries + col.entry_count,
            factor_entry{u, static_cast<factor_value_t>(w / sqrt_deg)});
        ++col.entry_count;
    }

    // Propagate excess to neighbors (deferred for thread safety). Runs BEFORE
    // sample_clique so the eliminator receives the neighbor span as dead
    // scratch — free to permute or overwrite entirely.
    double ev = G.excess(v);
    if (ev > 0.0) {
        for (const auto& [u, w] : nbrs)
            excess_out.emplace_back(u, w * ev / total_deg);
    }

    // Sample the clique edges into ws.edge_buffer (consumed by
    // endpoint publication) through the append-only emitter. The seed is a
    // pure function of (run seed, vertex): draws are identical at any thread
    // count and schedule.
    const std::uint64_t vseed =
        gpu_round_shadow_pivot_seed(run_seed, v);
    elim.sample_clique(std::span<weighted_neighbor>(nbrs), total_deg, vseed,
                       edge_emitter(ws.edge_buffer));
}

template<typename Eliminator>
void eliminate_partition_singleton(const Eliminator& elim,
                                   graph& G,
                                   const partition_result& part,
                                   std::vector<factor_col>& factor_cols,
                                   factorize_workspace& ws,
                                   const factor_options& opts,
                                   checkpoint* cp = nullptr,
                                   size_t work_hint = 0,
                                   std::vector<node_index>* live_degrees = nullptr) {
    const size_t n_verts = part.num_regions();
    // n_verts > 0 guaranteed by the dispatcher; early-exit not needed here.
    // The final column array is reserved to n before any round. Grow it once
    // here, then let each iteration write its own column directly. This avoids
    // constructing a temporary vector-of-vectors and serially moving every
    // column after the parallel region.
    const size_t factor_base = factor_cols.size();
    factor_cols.resize(factor_base + n_verts);
    auto output_col = [&](size_t k) -> factor_col& {
        return factor_cols[factor_base + k];
    };
    if (live_degrees) {
        auto& offsets = ws.degree_decrement_offsets;
        offsets.resize(n_verts + 1);
        offsets[0] = 0;
        for (std::size_t k = 0; k < n_verts; ++k) {
            const std::size_t degree = (*live_degrees)[part.data[k]];
            {
                const std::size_t physical = G.adj_count(part.data[k]);
                assert(physical >= degree);
                ws.degree_retired_dead_incidence += physical - degree;
            }
            if (offsets[k] > std::numeric_limits<std::size_t>::max() - degree)
                throw std::overflow_error("incremental degree stream overflow");
            offsets[k + 1] = offsets[k] + degree;
        }
        const std::size_t total = offsets.back();
        if (ws.degree_decrements.capacity() < total) {
            std::vector<node_index> exact(total);
            ws.degree_decrements.swap(exact);
        } else {
            ws.degree_decrements.resize(total);
        }
        ws.degree_removed_incidence = total;
        assert(work_hint == 0 || total == work_hint);
    }
    auto decrement_slice = [&](std::size_t k) -> std::span<node_index> {
        if (!live_degrees) return {};
        const auto& offsets = ws.degree_decrement_offsets;
        const std::size_t count = offsets[k + 1] - offsets[k];
        if (count == 0) return {};
        return {ws.degree_decrements.data() + offsets[k], count};
    };

    // Size a small round's team by both independent pivots and selected live
    // adjacency work. This differs from the retired APXCHOL_TAIL_THREADS
    // experiment: that rule gave every small tail the full team by vertex
    // count and regressed sparse IPM / grid rounds. Here cheap rounds stay
    // serial and increasingly dense rounds acquire workers continuously.
    const int team_threads = static_cast<int>(elimination_round_team_size(
        n_verts, work_hint, opts.omp_threshold, ws.threads.size()));
    const bool parallel_round = team_threads > 1;
    if (parallel_round) {
        // Large independent sets amortize dynamic scheduling with coarse
        // chunks. Dense-small rounds enter this path because their adjacency
        // work is high despite having few vertices; a 64-vertex chunk can then
        // leave almost the whole team idle (or even give the entire round to
        // one thread). Give those rounds one pivot per claim so the work gate
        // actually exposes their available inter-vertex parallelism.
        const int compute_chunk =
            elimination_compute_chunk(n_verts, opts.omp_threshold);
        // The fused paths use exactly the work-sized team selected above.
        {
            // Compute, reserve, publish edges and deactivate pivots in one
            // parallel region. Histogram claims assign disjoint endpoint slots;
            // the single section accounts for edges and starts bulk reserve.
            const int num_threads = team_threads;
            size_t edge_count = 0;
            // Persistent all-zero histogram (see factorize_workspace::incoming);
            // sized once, reset per round over the touched entries only.
            if (ws.incoming.size() < static_cast<size_t>(G.n()))
                ws.incoming.assign(static_cast<size_t>(G.n()), 0);
            std::vector<node_index>& incoming = ws.incoming;
            if (live_degrees) {
                const std::size_t total = ws.degree_decrements.size();
                if (ws.degree_decrement_scratch.capacity() < total) {
                    std::vector<node_index> exact(total);
                    ws.degree_decrement_scratch.swap(exact);
                } else {
                    ws.degree_decrement_scratch.resize(total);
                }
                ws.degree_decrement_histograms.resize(
                    static_cast<std::size_t>(num_threads));
            }

            parallel_exception error;
            #pragma omp parallel num_threads(num_threads)
            {
                int tid = omp_get_thread_num();

                ws.threads[tid].edge_buffer.clear();
                ws.threads[tid].excess_buffer.clear();

                #pragma omp for schedule(dynamic, compute_chunk)
                for (size_t k = 0; k < n_verts; ++k)
                    error.run([&] {
                        process_vertex(elim, G, part.data[k], opts.seed, output_col(k),
                                       ws.threads[tid], decrement_slice(k));
                    });
                // implicit barrier — edge_buffers populated before pre-pass

                if (!error.failed() && live_degrees) {
                    auto& decrements = ws.degree_decrements;
                    // Radix histogram rows belong to workers that actually
                    // joined this region. Unused requested-worker rows can
                    // contain offsets from an earlier round.
                    const int degree_workers = omp_get_num_threads();
                    const std::size_t begin = decrements.size() *
                        static_cast<std::size_t>(tid) /
                        static_cast<std::size_t>(degree_workers);
                    const std::size_t end = decrements.size() *
                        static_cast<std::size_t>(tid + 1) /
                        static_cast<std::size_t>(degree_workers);
                    const double decrement_start = omp_get_wtime();
                    parallel_radix_sort_node_indices(
                        decrements, ws.degree_decrement_scratch,
                        ws.degree_decrement_histograms, tid, degree_workers);
                    std::size_t unique = 0;
                    std::size_t i = begin;
                    if (i > 0 && i < end) {
                        const node_index continuation = decrements[i - 1];
                        while (i < end && decrements[i] == continuation) ++i;
                    }
                    while (i < end) {
                        const node_index u = decrements[i];
                        std::size_t j = i + 1;
                        while (j < decrements.size() && decrements[j] == u) ++j;
                        assert(j - i <=
                               std::numeric_limits<node_index>::max());
                        const node_index delta = static_cast<node_index>(j - i);
                        const node_index before = std::exchange(
                            (*live_degrees)[u],
                            (*live_degrees)[u] - delta);
                        assert(before >= delta);
                        (void)before;
                        ++unique;
                        i = j;
                    }
                    ws.threads[tid].degree_decrement_unique = unique;
                    ws.threads[tid].degree_decrement_ms =
                        1000.0 * (omp_get_wtime() - decrement_start);
                }

                // Parallel atomic histogram of incoming edges per vertex.
                // Each thread sweeps its own edge_buffer and atomic-increments
                // incoming[u] and incoming[v]. Directed AoS also retains each
                // returned offset, so the apply pass can write that reserved
                // slot without a second pair of atomics. On first bump from
                // 0→1 we push the vertex onto touched_buffer, so reserve_for
                // skips the full O(G.n()) scan.
                if (!error.failed()) error.run([&] {
                    auto& my_touched = ws.threads[tid].touched_buffer;
                    my_touched.clear();
                    const auto& my_buf = ws.threads[tid].edge_buffer;
                    {
                        auto& offsets = ws.threads[tid].endpoint_offsets;
                        offsets.resize(2 * my_buf.size());
                        for (size_t i = 0; i < my_buf.size(); ++i) {
                            const auto& e = my_buf[i];
                            offsets[2 * i] = std::atomic_ref(incoming[e.u]).fetch_add(
                                1, std::memory_order_relaxed);
                            offsets[2 * i + 1] = std::atomic_ref(incoming[e.v]).fetch_add(
                                1, std::memory_order_relaxed);
                            if (offsets[2 * i] == 0)
                                my_touched.push_back(e.u);
                            if (offsets[2 * i + 1] == 0)
                                my_touched.push_back(e.v);
                        }
                    }
                });
                #pragma omp barrier  // incoming[] complete before reserve_for

                // Account for appended edges and concatenate touched buffers for
                // the adjacency bulk reserve below.
                #pragma omp single
                {
                    if (!error.failed()) error.run([&] {
                        const int actual_threads = omp_get_num_threads();
                        edge_count = 0;
                        for (int t = 0; t < actual_threads; ++t)
                            edge_count += ws.threads[t].edge_buffer.size();
                        const size_t N_edges = edge_count;
                        if (live_degrees) ws.degree_fill_edges = N_edges;
                        if (N_edges > 0) {
                            G.record_edges_added(static_cast<edge_index>(N_edges));
                        }
                        ws.touched_concat.clear();
                        for (int t = 0; t < actual_threads; ++t)
                            ws.touched_concat.insert(
                                ws.touched_concat.end(),
                                ws.threads[t].touched_buffer.begin(),
                                ws.threads[t].touched_buffer.end());
                        if (N_edges > 0)
                            G.adj_compact_for_round_if_needed(incoming);
                    });
                }
                // The implicit barrier after single publishes the edge count,
                // touched_concat, and any rebuilt adjacency pool. Keeping the
                // compaction predicate inside single prevents late workers
                // from observing its reset state and skipping a team barrier.

                // Reserve every touched vertex's slab in parallel: the
                // per-vertex slab copies dominate a serial reserve loop on
                // dense rounds.
                if (edge_count > 0) {
                    if (!error.failed()) error.run([&] {
                        G.adj_bulk_reserve_parallel(
                            ws.touched_concat.begin(), ws.touched_concat.end(), incoming);
                    });
                    #pragma omp barrier  // all reserve attempts finish before publication
                }

                if (!error.failed()) {
                    // Apply edges into histogram-assigned endpoint slots.
                    {
                        const auto& ebuf = ws.threads[tid].edge_buffer;
                        {
                            const auto& offsets = ws.threads[tid].endpoint_offsets;
                            for (size_t i = 0; i < ebuf.size(); ++i) {
                                const auto& [u, v, w] = ebuf[i];
                                G.adj_write_reserved_directed_at(
                                    u, offsets[2 * i], v, w);
                                G.adj_write_reserved_directed_at(
                                    v, offsets[2 * i + 1], u, w);
                            }
                        }
                        ws.threads[tid].edge_buffer.clear();
                        // Apply excess atomically (each thread's own buffer).
                        for (auto [u, delta] : ws.threads[tid].excess_buffer)
                            G.atomic_add_excess(u, delta);
                        ws.threads[tid].excess_buffer.clear();
                    }

                    {
                        // Direct writers left count_[v] unchanged. Publish each
                        // completed suffix once, after every writer is done.
                        #pragma omp barrier
                        #pragma omp for schedule(static) nowait
                        for (size_t i = 0; i < ws.touched_concat.size(); ++i) {
                            const node_index v = ws.touched_concat[i];
                            if (live_degrees)
                                (*live_degrees)[v] += incoming[v];
                            G.adj_commit_reserved_directed(v, incoming[v]);
                            incoming[v] = 0;
                        }
                    }

                    #pragma omp for schedule(static) nowait
                    for (size_t k = 0; k < n_verts; ++k)
                        G.set_inactive_unchecked(part.data[k]);
                }
            }
            error.rethrow();
            G.bulk_decrement_active(static_cast<node_index>(n_verts));

            if (cp) (*cp)("compute+apply_fused");
        }
    } else
    {
        // Serial path: per-vertex inline application for cache locality.
        // Inline dedup in process_vertex replaces merge_parallel_edges.
        auto& wt = ws.threads[0];
        for (size_t k = 0; k < n_verts; ++k) {
            wt.edge_buffer.clear();
            wt.excess_buffer.clear();
            process_vertex(elim, G, part.data[k], opts.seed, output_col(k), wt,
                           decrement_slice(k));
            if (live_degrees) {
                for (node_index u : decrement_slice(k)) {
                    assert((*live_degrees)[u] > 0);
                    --(*live_degrees)[u];
                }
            }
            for (auto [u, v, w] : wt.edge_buffer) {
                G.add_edge(u, v, w);
                if (live_degrees) {
                    ++ws.degree_fill_edges;
                    ++(*live_degrees)[u];
                    ++(*live_degrees)[v];
                }
            }
            for (auto [v, delta] : wt.excess_buffer)
                G.excess(v) += delta;
            G.deactivate(part.data[k]);
        }
        if (cp) { (*cp)("merge_is"); (*cp)("compute"); }
    }
    // NOTE: no cp->ascend() here — the dispatcher (eliminate_partition) owns the
    // descend/ascend bracket around both the singleton and multi paths.
}

// All surviving partitioners emit singleton regions (one vertex each), so
// elimination always takes the singleton path.
template<typename Eliminator>
void eliminate_partition(const Eliminator& elim,
                         graph& G,
                         const partition_result& part,
                         std::vector<factor_col>& factor_cols,
                         factorize_workspace& ws,
                         const factor_options& opts,
                         checkpoint* cp = nullptr,
                         size_t work_hint = 0,
                         std::vector<node_index>* live_degrees = nullptr) {
    if (cp) { cp->descend("eliminate"); cp->tick(); }
    const size_t n_verts = part.num_regions();
    if (n_verts == 0) {
        if (cp) cp->ascend();
        return;
    }
    eliminate_partition_singleton(elim, G, part, factor_cols, ws, opts, cp,
                                  work_hint, live_degrees);
    if (live_degrees && opts.verbose) {
        const std::size_t raw = ws.degree_decrements.size();
        std::size_t unique = 0;
        double maximum_ms = 0.0;
        for (const auto& thread : ws.threads) {
            unique += thread.degree_decrement_unique;
            maximum_ms = std::max(maximum_ms, thread.degree_decrement_ms);
        }
        std::fprintf(stderr,
            "[incremental-degree] round=%llu raw=%zu "
            "unique_worker_endpoints=%zu max_reduce_ms=%.6f\n",
            static_cast<unsigned long long>(ws.round_index), raw, unique,
            maximum_ms);
    }
    if (cp) cp->ascend();
}

// Eliminate remaining vertices after the main IS-elimination loop.
// Used when IS fraction drops below threshold — avoids the O(|active|)
// IS-finding scan when only a few vertices can be chosen anyway.
//
// It does NOT see the whole residual. The caller runs BK rounds first, down to
// the partitioner's `residual_handoff_threshold` (500 for every shipped rule
// that can bail), so this peels a bounded tail; see the residual-loop comment
// in factorize_impl.
template<typename Eliminator>
void eliminate_remaining(const Eliminator& elim,
                         graph& G,
                         std::vector<node_index>& active,
                         std::vector<factor_col>& factor_cols,
                         factorize_workspace& ws,
                         const factor_options& opts) {
    auto& wt = ws.threads[0];
    factor_col col;

    auto peel_one = [&](node_index v) {
        wt.edge_buffer.clear();
        wt.excess_buffer.clear();
        process_vertex(elim, G, v, opts.seed, col, wt);
        factor_cols.push_back(std::move(col));
        for (auto [a, b, w] : wt.edge_buffer)
            G.add_edge(a, b, w);
        for (auto [u, delta] : wt.excess_buffer)
            G.excess(u) += delta;
        G.deactivate(v);
    };

    if (opts.residual_peel == residual_peel_strategy::min_degree) {
        // Historical lazy degree-heap heuristic, retained under min_degree.
        // Revalidate the popped key, but leave other keys unchanged. Degree
        // decreases can therefore hide a better pivot under a stale larger
        // key: this is not exact current-minimum-degree selection. Changing
        // the update policy changes pivot order, factor quality and cost.
        using entry = std::pair<node_index, node_index>;  // (degree, vertex)
        std::vector<entry> heap;
        heap.reserve(active.size());
        for (auto v : active) {
            if (!G.is_active(v)) continue;
            heap.emplace_back(G.prune_and_degree(v), v);
        }
        std::make_heap(heap.begin(), heap.end(), std::greater<entry>{});

        while (!heap.empty()) {
            std::pop_heap(heap.begin(), heap.end(), std::greater<entry>{});
            auto [stored_d, v] = heap.back();
            heap.pop_back();
            if (!G.is_active(v)) continue;
            node_index cur_d = G.prune_and_degree(v);
            if (cur_d != stored_d) {
                heap.emplace_back(cur_d, v);
                std::push_heap(heap.begin(), heap.end(), std::greater<entry>{});
                continue;
            }
            peel_one(v);
        }
        active.clear();
    } else if (opts.residual_peel == residual_peel_strategy::bk_serial) {
        // BK-style sampling: pick ~√|active| candidates, peel min-degree one.
        // Serial path — a local generator (seeded from the run seed) suffices.
        std::mt19937 peel_rng(opts.seed ^ 0x1CE4E5B9U);
        std::uniform_int_distribution<size_t> pick(0, 0);
        while (!active.empty()) {
            // Compact dead entries lazily.
            while (!active.empty() && !G.is_active(active.back()))
                active.pop_back();
            if (active.empty()) break;

            size_t k = std::max<size_t>(1, static_cast<size_t>(std::sqrt(double(active.size()))));
            k = std::min(k, active.size());
            size_t best_idx = 0;
            node_index best_deg = std::numeric_limits<node_index>::max();
            pick.param(std::uniform_int_distribution<size_t>::param_type(0, active.size() - 1));
            for (size_t s = 0; s < k; ++s) {
                size_t i = pick(peel_rng);
                if (!G.is_active(active[i])) continue;
                node_index d = G.prune_and_degree(active[i]);
                if (d < best_deg) { best_deg = d; best_idx = i; }
            }
            node_index v = active[best_idx];
            active[best_idx] = active.back();
            active.pop_back();
            if (!G.is_active(v)) continue;
            peel_one(v);
        }
    } else {
        // natural order — fastest, no extra scan.
        for (auto v : active)
            peel_one(v);
        active.clear();
    }
    active.clear();
}

// Internal reference controls for testing the adaptive production policy.
struct factorization_controls {
    enum class degree_cache { automatic, disabled, forced };
    degree_cache incremental_degrees = degree_cache::automatic;
    double prune_skip_factor = 4.0;
    bool residual_sparsify = true;
};

} // namespace detail

// Takes the graph BY VALUE (a sink). Elimination mutates the working graph, so
// it needs its own copy. Callers that pass an rvalue (the runtime-dispatch path,
// which builds a throwaway via make_graph) move into the parameter for free;
// callers that pass an lvalue (and want to keep their graph) copy once here —
// same cost as the old defensive `graph work(G)`.
template<typename Partitioner, typename Eliminator>
factorization factorize_impl(const Eliminator& elim,
                             Partitioner& partitioner,
                             graph G,
                             const factor_options& opts_in,
                             checkpoint* cp,
                             const Eigen::SparseMatrix<double>* initial_csc = nullptr,
                             detail::setup_route route = detail::setup_route::cpu,
                             [[maybe_unused]] bool keep_factor = false,
                             const detail::factorization_controls& controls = {}) {
    detail::validate_factor_options(opts_in);
    const node_index n = initial_csc ? static_cast<node_index>(initial_csc->rows()) : G.n();
    if (n == 0)
        return {};

    const bool gpu_owned_setup_route = route == detail::setup_route::gpu;

    // Resolve the by-route sentinels for this factorization.
    const factor_options opts = [&] {
        factor_options o = opts_in;
        if (o.partition.degree_quantile < 0.0)
            o.partition.degree_quantile = gpu_owned_setup_route
                                              ? degree_quantile_device_default
                                              : degree_quantile_host_default;
        // The device route does not implement the host trace-cycle core rules.
        if (o.exact_core_max_h == exact_core_by_route)
            o.exact_core_max_h = gpu_owned_setup_route
                                     ? 0u : exact_core_host_default;
        return o;
    }();

    if (cp) cp->descend("setup");

    factorization result;
    if (cp) cp->tick();
    graph work(std::move(G));
    if (cp) (*cp)("graph_copy");
    auto gpu_round_shadow =
        detail::make_gpu_round_shadow_session<Eliminator>(elim,
            route, opts.verbose);

    // Detect SDDM: any vertex with positive excess means the matrix
    // is positive definite (not just semidefinite like a Laplacian).
    // Note: make_graph already filters out FP noise (excess < diag * 1e-12),
    // so any remaining positive excess is genuine.
    if (!initial_csc) for (node_index v = 0; v < n; ++v) {
        if (work.excess(v) > 0.0) { result.sddm = true; break; }
    }
    if (cp) (*cp)("sddm_scan");

    const int num_threads_factorize = omp_get_max_threads();
    factorize_workspace ws;
    {
        ws.threads.resize(num_threads_factorize);
        for (auto& t : ws.threads) {
            t.factor_entries =
                std::make_unique<std::pmr::monotonic_buffer_resource>();
        }
    }
    std::vector<detail::factor_col> factor_cols;
    constexpr bool sample_bounded = partitioner_sample_bounded_v<Partitioner>;
    constexpr size_t residual_handoff_default =
        partitioner_residual_handoff_v<Partitioner>;
    size_t residual_thresh = opts.parallel_residual_threshold;
    if (residual_thresh == std::numeric_limits<size_t>::max())
        residual_thresh = residual_handoff_default;

    // Active vertex list (natural index order) — filtered in-place each round.
    std::vector<node_index> active(n);
    // TODO: use ranges::iota once we stop supporting macos-15 CI
    // with its default Xcode 16.4 toolchain, whose libc++ lacks it.
    std::iota(active.begin(), active.end(), node_index{0});
    std::vector<node_index> active_scratch;

#if defined(APXCHOL_USE_CUDA)
    if (gpu_owned_setup_route) {
        constexpr bool eligible =
            std::is_same_v<Partitioner, block_greedy_partitioner> &&
            std::is_same_v<std::remove_cvref_t<Eliminator>, tree_elimination>;
        if constexpr (!eligible) {
            throw std::invalid_argument("GPU setup requires block_greedy and the built-in sampler");
        } else {
            if (!initial_csc)
                throw std::invalid_argument("GPU setup requires a supported CSC operator");
            std::unique_ptr<detail::gpu_block_frontend> gpu_frontend;
            result.sddm = gpu_round_shadow.initialize_owned_csc(
                detail::gpu_owned_csc_host_buffers(*initial_csc), gpu_frontend,
                opts.regularization);
            if (cp) (*cp)("gpu_csc_initialize");
            std::size_t previous_nnz = 0;
            bool sparsification_attempted = false;
            result.peak_graph_bytes = std::max(result.peak_graph_bytes, work.memory_bytes());
            while (!active.empty()) {
                const auto prep = gpu_frontend->prepare(active, opts.partition
                    , opts.seed, ws.round_index
                );
                auto part = gpu_frontend->select_block_greedy();
                if (part.data.empty())
                    throw std::runtime_error("GPU-owned factorization selection made no progress");
                const double min_yield = detail::adaptive_is_yield_fraction(
                    opts.min_is_fraction, active.size(), prep.average_degree, residual_thresh);
                if (controls.residual_sparsify && !sparsification_attempted &&
                    detail::selection_should_handoff(part.num_regions(),
                        prep.candidate_count, active.size(), min_yield, residual_thresh,
                        opts.omp_threshold)) {
                    // This is a preview, not an eliminated/countable round.
                    // Capture its gate inputs before changing the residual.
                    sparsification_attempted = true;
                    const auto handoff = part.num_regions();
                    part.clear();
                    double distinct_degree = 0.0;
                    bool worthwhile = false;
                    if (active.size() > residual_thresh && handoff) {
                        distinct_degree = gpu_round_shadow.probe_owned_distinct_degree(active, *gpu_frontend);
                        worthwhile = detail::residual_sparsify_worthwhile(active.size(),
                            residual_thresh, handoff, distinct_degree);
                    }
                    if (opts.verbose)
                        std::fprintf(stderr, "[gpu-owned-sparsify-gate] round=%zu active=%zu "
                            "handoff=%zu avg_distinct_degree=%.17g attempted=1 worthwhile=%d\n",
                            ws.round_index, active.size(), handoff, distinct_degree, int(worthwhile));
                    if (worthwhile) {
                        const auto sparse_seed = opts.seed ^ ws.round_index;
                        const auto stats = gpu_round_shadow.sparsify_owned_residual(*gpu_frontend, sparse_seed);
                        if (opts.verbose)
                            std::fprintf(stderr, "[gpu-owned-sparsify] implementation=device round=%zu "
                                "seed=%llu physical_before=%zu distinct_before=%zu kept=%zu backbone=%zu "
                                "expected=%.17g min_p=%.17g max_inv_p=%.17g "
                                "scalar_download_bytes=%zu peak_device_bytes=%zu\n",
                                ws.round_index, static_cast<unsigned long long>(sparse_seed),
                                stats.physical_before, stats.distinct_before, stats.kept_edges,
                                stats.backbone_edges, stats.expected_kept_edges,
                                stats.min_offtree_probability, stats.max_inverse_probability,
                                stats.scalar_download_bytes, stats.peak_device_bytes);
                        if (cp) (*cp)("gpu_owned_sparsify");
                    }
                    // Reprepare even when the one-shot traffic test says keep.
                    // No RNG phase, append position or numerical generation advances.
                    continue;
                }
                const auto report = gpu_round_shadow.run_owned_prefix_round(
                    gpu_frontend->device_selection(), opts.seed);
                const std::size_t nnz = report.factor_log_columns + report.factor_log_entries;
                result.rounds.push_back({active.size(), part.num_regions(), prep.average_degree,
                    cp ? nnz - previous_nnz : 0, cp ? nnz : 0});
                previous_nnz = nnz;
                // The device selector returns natural active-id order. This
                // is host metadata maintenance, not a read of the stale graph.
                std::size_t selected = 0;
                std::erase_if(active, [&](node_index v) {
                    if (selected < part.data.size() && part.data[selected] == v) {
                        ++selected; return true;
                    }
                    return false;
                });
                if (selected != part.data.size())
                    throw std::logic_error("GPU-owned prefix selection lost natural active order");
                if (report.active_count != active.size())
                    throw std::logic_error("GPU-owned factorization active count diverged");
                if (!active.empty()) gpu_round_shadow.advance_selector(*gpu_frontend);
                ++ws.round_index;
            }
            std::vector<detail::factor_entry> entries;
            const auto headers = gpu_round_shadow.complete_owned_factorization(
                result.perm, result.L, keep_factor ? &entries : nullptr);
            if (keep_factor) {
                std::vector<detail::factor_col> columns;
                columns.reserve(n);
                for (const auto& c : headers)
                    columns.push_back({c.vertex, c.diag,
                        c.entry_count ? entries.data() + c.entry_begin : nullptr, c.entry_count});
                detail::build_csc(result, columns, n, cp);
            }
            gpu_frontend.reset();
            if (cp) (*cp)("gpu_owned_factorization");
        }
    }
#endif

    // Complete device ownership has no CPU factor columns to materialize.
    // Other routes retain the same reservation before their first elimination.
    if (!active.empty()) factor_cols.reserve(n);

    // The partitioner's view of the run-constant services, the shared
    // selection structure, and the degree-prepass scratch (all owned here).
    selection sel;
    std::vector<node_index> pre_degrees, pre_scratch, pre_eligible;
    std::vector<std::array<size_t, 256>> pre_histograms;
    std::vector<size_t> pre_filter_offsets;
    std::vector<node_index> live_degrees;   // vertex-indexed, for ctx.degrees
    // Keep exact degrees when their sparse updates cost less than a full walk.
    const bool incremental_degree_capable =
        std::is_same_v<Partitioner, block_greedy_partitioner> &&
        controls.incremental_degrees != detail::factorization_controls::degree_cache::disabled;
    const bool incremental_degree_auto = incremental_degree_capable &&
        controls.incremental_degrees == detail::factorization_controls::degree_cache::automatic;
    bool incremental_degree_active = incremental_degree_capable && !incremental_degree_auto;
    bool incremental_degree_ready = false;
    bool incremental_degree_auto_decided = !incremental_degree_auto;
    std::size_t incremental_live_incidence = 0;
    std::size_t incremental_dead_incidence = 0;
    constexpr double incremental_refresh_ratio = 0.10;

    // Prune-walk skip. Only where AUTO has permanently DECLINED the exact
    // incremental cache above (block-greedy on directed vec_pool): that path's
    // selector and eliminator already filter dead adjacency entries, because
    // an active cache never prunes, and from then on every round re-walks
    // every active vertex. In a skipping round a vertex whose raw adjacency
    // count exceeds prune_skip_factor x the previous round's eligibility
    // threshold is reported at that count instead of being walked
    // (prune_and_degrees). Never while the cache is active or undecided (its
    // sample and its seed are these walks), and never under the forced exact
    // reference policies.
    //
    // A raw count only grows until the vertex is walked, so a hub whose
    // neighbours die hides behind it and is eliminated late. Full
    // walks therefore alternate with the skipping rounds and AUDIT the skip
    // they replace: the number of eligible vertices it would have hidden. A
    // clean audit doubles the skipping rounds before the next one (1..8), a
    // hidden vertex resets them to 1.
    const double prune_skip_factor = controls.prune_skip_factor;
    unsigned prune_skip_gap = 1, prune_skip_streak = 0;
    double previous_degree_threshold = 0.0;   // 0 = no previous round yet

    // Shared round front-end: prepass (when the partitioner's trait asks for
    // it) + find_partition + finalize, with uniform profiling labels.
    // last_avg_degree feeds the per-round stats; it is the prepass's exact
    // average (0 = not measured, for partitioners without the prepass).
    double last_avg_degree = 0.0;
    size_t last_candidate_count = 0;
    auto run_partitioner = [&](auto& p, graph& g,
                               std::span<const node_index> act)
        -> const partition_result& {
        using P = std::remove_reference_t<decltype(p)>;
        sel.reset(g.n());
        last_avg_degree = 0.0;
        last_candidate_count = act.size();
        partition_context pctx{
            .options = opts.partition,
            .seed = opts.seed,
            .omp_threshold = opts.omp_threshold,
            .cp = cp,
            .degrees = {},
        };
        if (cp) { cp->descend("find_partition"); cp->tick(); }
        if constexpr (partitioner_degree_prepass_v<P>) {
            constexpr bool incremental_for_this_partitioner =
                std::is_same_v<P, block_greedy_partitioner> &&
                std::is_same_v<P, Partitioner>;
            double avg_deg = 0.0;
            const bool refresh_incremental =
                incremental_for_this_partitioner &&
                incremental_degree_active && incremental_degree_ready &&
                incremental_dead_incidence > 0 &&
                static_cast<long double>(incremental_dead_incidence) >=
                    incremental_refresh_ratio *
                    static_cast<long double>(incremental_live_incidence);
            if (incremental_for_this_partitioner &&
                incremental_degree_active && incremental_degree_ready &&
                !refresh_incremental) {
                pre_degrees.resize(act.size());
                double total_degree = 0.0;
#pragma omp parallel for reduction(+:total_degree) schedule(static) \
    if(act.size() > opts.omp_threshold)
                for (std::size_t i = 0; i < act.size(); ++i) {
                    pre_degrees[i] = live_degrees[act[i]];
                    total_degree += pre_degrees[i];
                }
                avg_deg = total_degree / static_cast<double>(act.size());
            } else {
                if (refresh_incremental &&
                    opts.verbose)
                    std::fprintf(stderr,
                        "[incremental-refresh] round=%llu dead=%zu live=%zu "
                        "ratio=%.6f threshold=%.6f\n",
                        static_cast<unsigned long long>(ws.round_index),
                        incremental_dead_incidence,
                        incremental_live_incidence,
                        incremental_live_incidence == 0 ? 0.0 :
                            static_cast<double>(incremental_dead_incidence) /
                            static_cast<double>(incremental_live_incidence),
                        incremental_refresh_ratio);
                node_index skip_above = 0, audit_above = 0;
                if (incremental_degree_auto &&
                    incremental_degree_auto_decided &&
                    !incremental_degree_active &&
                    prune_skip_factor > 0.0 &&
                    previous_degree_threshold > 0.0) {
                    const auto cutoff = static_cast<node_index>(std::min(
                        prune_skip_factor * previous_degree_threshold,
                        static_cast<double>(
                            std::numeric_limits<node_index>::max())));
                    if (prune_skip_streak < prune_skip_gap) {
                        skip_above = cutoff;
                        ++prune_skip_streak;
                    } else {
                        audit_above = cutoff;
                    }
                }
                std::size_t hidden = 0;
                avg_deg = prune_and_degrees(
                    g, act, pre_degrees, opts.omp_threshold, skip_above,
                    audit_above,
                    static_cast<node_index>(std::min(
                        previous_degree_threshold,
                        static_cast<double>(
                            std::numeric_limits<node_index>::max()))),
                    &hidden);
                if (audit_above > 0) {
                    prune_skip_gap = hidden == 0
                        ? std::min(2u * prune_skip_gap, 8u) : 1u;
                    prune_skip_streak = 0;
                    if (opts.verbose)
                        std::fprintf(stderr,
                            "[prune-skip] round=%llu active=%zu hidden=%zu next_gap=%u\n",
                            static_cast<unsigned long long>(ws.round_index),
                            act.size(), hidden, prune_skip_gap);
                }
                if (incremental_for_this_partitioner &&
                    incremental_degree_capable) {
                    incremental_live_incidence = static_cast<std::size_t>(
                        std::llround(avg_deg * static_cast<double>(act.size())));
                    incremental_dead_incidence = 0;
                }
            }
            const double q = degree_quantile_or_host_default(
                opts.partition.degree_quantile);
            const double thr = q > 0.0 && q < 1.0
                                   && act.size() > opts.omp_threshold
                ? static_cast<double>(parallel_degree_quantile(
                      pre_degrees, act.size(), q, pre_histograms))
                : is_degree_threshold(pre_degrees, act.size(), avg_deg,
                                      opts.partition, pre_scratch);
            previous_degree_threshold = thr;
            if (live_degrees.size() < static_cast<size_t>(g.n()))
                live_degrees.resize(g.n());
            size_t eligible_count;
            if (act.size() > opts.omp_threshold) {
                eligible_count = parallel_ordered_degree_filter(
                    act, pre_degrees, thr, live_degrees, pre_eligible,
                    pre_filter_offsets);
            } else {
                pre_eligible.clear();
                for (size_t i = 0; i < act.size(); ++i) {
                    live_degrees[act[i]] = pre_degrees[i];
                    if (pre_degrees[i] <= thr) pre_eligible.push_back(act[i]);
                }
                eligible_count = pre_eligible.size();
            }
            last_candidate_count = eligible_count;
            last_avg_degree = avg_deg;
            if (incremental_for_this_partitioner &&
                incremental_degree_capable)
                incremental_degree_ready = true;
            pctx.degrees = live_degrees;
            if (cp) (*cp)("prune");
            p.find_partition(g, std::span<const node_index>(
                                 pre_eligible.data(), eligible_count),
                             pctx, sel);
        } else {
            p.find_partition(g, act, pctx, sel);
        }
        if (cp) (*cp)("select");
        const partition_result& part = sel.finalize();
        if (cp) { (*cp)("collect"); cp->ascend(); }
        return part;
    };

    // Sampling-based (sample_bounded) partitioners may legitimately return an
    // empty round and succeed on retry; cap the retries so a partitioner that
    // stops making progress hands the residual to the serial peel instead of
    // spinning forever.
    size_t consecutive_empty = 0;
    size_t handoff_vertices = 0;
    constexpr size_t kMaxEmptyRounds = 64;
    // An empty sampled round: true when the retry budget is spent (hand the
    // residual to the peel), else advance the round index for the retry.
    auto retry_budget_spent = [&](size_t& empty_rounds) {
        if (++empty_rounds >= kMaxEmptyRounds) return true;
        ++ws.round_index;              // keep retry rounds distinguishable
        return false;
    };
    // Shared round epilogue: per-round nnz tally (only when the caller wants
    // the stats), peak graph memory, the stable active filter and the round
    // index.
    auto finish_round = [&](size_t cols_before) {
        if (cp) {
            size_t round_nnz = 0;
            for (size_t ci = cols_before; ci < factor_cols.size(); ++ci)
                round_nnz += factor_cols[ci].entry_count + 1;
            result.rounds.back().nnz_added = round_nnz;
            result.rounds.back().nnz_total =
                (result.rounds.size() >= 2
                    ? result.rounds[result.rounds.size() - 2].nnz_total : 0)
                + round_nnz;
        }
        result.peak_graph_bytes = std::max(result.peak_graph_bytes,
                                           work.memory_bytes());
        if (active.size() > opts.omp_threshold) {
            parallel_stable_active_filter(
                active, active_scratch, pre_filter_offsets,
                [&](node_index v) { return work.is_active(v); });
        } else {
            std::erase_if(active,
                          [&](node_index v) { return !work.is_active(v); });
        }
        ++ws.round_index;
    };

    while (active.size() > 1) {
        ws.reset_for_round();
        const auto& part = run_partitioner(partitioner, work, active);
#ifndef NDEBUG
        detail::assert_partition_independent(work, part);
#endif

        // If partition is too small, the partitioning overhead exceeds the
        // benefit of batch elimination.  Fall back to sequential elimination
        // of all remaining vertices (handled after the loop).  Skipped for
        // sample-bounded partitioners — their per-round cost does not scale
        // with |active|, so the BG-style fallback is strictly worse.
        //
        // The break must come BEFORE recording the round: this partition is
        // discarded (the residual is peeled instead), so counting it would add a
        // phantom round whose is_size eliminates no factor columns. round-as-level
        // builds its level boundaries from cumulative is_size, so a phantom round
        // desyncs the boundaries and mislabels the sequential peel tail as one
        // independent level -> incorrect triangular solve.
        if constexpr (!sample_bounded) {
            const double min_yield = detail::adaptive_is_yield_fraction(
                opts.min_is_fraction, active.size(), last_avg_degree,
                residual_thresh);
            // A low relative yield can still be a large, profitable round.
            // Keep any non-empty selection that is already large enough for
            // the parallel elimination path; the yield rule is for small
            // selections whose scan cost is no longer amortized.
            if (detail::selection_should_handoff(
                    part.num_regions(),
                    last_candidate_count, active.size(), min_yield,
                    residual_thresh, opts.omp_threshold)) {
                handoff_vertices = part.num_regions();
                if (opts.verbose)
                    std::fprintf(stderr,
                        "[apxchol] selector handoff: active=%zu candidates=%zu "
                        "selected=%zu yield=%.6f base=%.6f effective=%.6f "
                        "avg_degree=%.3f residual_threshold=%zu\n",
                        active.size(), last_candidate_count,
                        part.num_regions(), last_candidate_count
                            ? static_cast<double>(part.num_regions()) /
                                  static_cast<double>(last_candidate_count)
                            : 0.0,
                        opts.min_is_fraction, min_yield, last_avg_degree,
                        residual_thresh);
                break;
            }
        } else {
            if (part.num_regions() == 0) {
                if (retry_budget_spent(consecutive_empty))
                    break;                     // hand the residual to the peel
                continue;                      // whiffed sample round: retry
            }
            consecutive_empty = 0;
        }
        result.rounds.push_back({active.size(), part.num_regions(), last_avg_degree, 0, 0});

        const size_t cols_before = cp ? factor_cols.size() : 0;
        size_t elimination_work_hint = 0;
        if constexpr (requires(const Partitioner& p) {
                          { p.selected_degree_work() } ->
                              std::convertible_to<size_t>;
                      }) {
            elimination_work_hint =
                static_cast<size_t>(partitioner.selected_degree_work());
        } else if constexpr (partitioner_degree_prepass_v<Partitioner>) {
            for (node_index v : part.data) {
                const size_t degree = static_cast<size_t>(live_degrees[v]);
                if (elimination_work_hint >
                    std::numeric_limits<size_t>::max() - degree) {
                    elimination_work_hint =
                        std::numeric_limits<size_t>::max();
                    break;
                }
                elimination_work_hint += degree;
            }
        }
        const long double prune_work =
            static_cast<long double>(active.size()) * last_avg_degree;
        const long double prune_to_update = elimination_work_hint == 0
            ? 0.0L : prune_work /
                static_cast<long double>(elimination_work_hint);
        const size_t incremental_workers = detail::elimination_round_team_size(
            part.num_regions(), elimination_work_hint, opts.omp_threshold,
            static_cast<size_t>(num_threads_factorize));
        // Every ordinary prepass is an exact decision sample. Across the
        // nine-graph screen the measured
        // crossover lies near 10 adjacency visits per pivot update; require 16
        // for margin. A sample below the measured break-even ratio 10 is a
        // permanent OFF decision. Ratios in [10,16), and traffic-favourable
        // rounds that are merely too small for the radix collective, remain
        // pending while the residual evolves. Once both activation gates pass,
        // updates keep the cache exact and the choice is permanently ON for
        // the rest of the main-selector phase.
        if (incremental_degree_auto && !incremental_degree_auto_decided &&
            incremental_degree_ready) {
            const bool traffic_worthwhile =
                detail::incremental_degree_traffic_worthwhile(
                    active.size(), last_avg_degree, elimination_work_hint);
            const bool traffic_can_improve =
                detail::incremental_degree_traffic_can_improve(
                    active.size(), last_avg_degree, elimination_work_hint);
            if (!traffic_can_improve) {
                incremental_degree_active = false;
                incremental_degree_auto_decided = true;
            } else if (traffic_worthwhile &&
                       detail::incremental_degree_parallel_work_worthwhile(
                           elimination_work_hint, incremental_workers)) {
                incremental_degree_active = true;
                incremental_degree_auto_decided = true;
            }
        }
        if (incremental_degree_capable &&
            opts.verbose) {
            std::fprintf(stderr,
                "[incremental-gate] round=%llu active=%zu candidates=%zu "
                "selected=%zu avg_degree=%.6f selected_work=%zu "
                "workers=%zu work_per_worker=%zu "
                "prune_to_update=%.6Lf active=%d decided=%d\n",
                static_cast<unsigned long long>(ws.round_index), active.size(),
                last_candidate_count, part.num_regions(), last_avg_degree,
                elimination_work_hint, incremental_workers,
                incremental_workers == 0
                    ? 0 : elimination_work_hint / incremental_workers,
                prune_to_update,
                incremental_degree_active ? 1 : 0,
                incremental_degree_auto_decided ? 1 : 0);
        }
        detail::eliminate_partition(elim, work, part, factor_cols, ws, opts, cp,
                                    elimination_work_hint,
                                    incremental_degree_active &&
                                            incremental_degree_ready
                                        ? &live_degrees : nullptr);
        if (incremental_degree_active && incremental_degree_ready) {
            const std::size_t removed = ws.degree_removed_incidence;
            const std::size_t retired_dead =
                ws.degree_retired_dead_incidence;
            const std::size_t fill = ws.degree_fill_edges;
            assert(incremental_live_incidence >= 2 * removed);
            assert(incremental_dead_incidence >= retired_dead);
            incremental_live_incidence -= 2 * removed;
            incremental_live_incidence += 2 * fill;
            incremental_dead_incidence -= retired_dead;
            incremental_dead_incidence += removed;
        }
        finish_round(cols_before);

        // Pool defragmentation happens inside bulk_reserve_parallel.
        // Record the live fraction after each round.
        work.adj_note_live_fraction();
    }

    // Before the BK residual loop, sparsify once only when the observed
    // handoff yield predicts enough future edge traffic to repay the rebuild.
    {
        if (controls.residual_sparsify && active.size() > residual_thresh &&
            handoff_vertices != 0) {
            const auto sample =
                detail::residual_coalescer::sample(work, active);
            const bool worthwhile = detail::residual_sparsify_worthwhile(
                active.size(), residual_thresh, handoff_vertices,
                sample.avg_distinct_degree);
            if (opts.verbose)
                std::fprintf(stderr,
                    "[apxchol] residual sparsify gate: active=%zu "
                    "handoff_vertices=%zu avg_distinct_degree=%.6f "
                    "duplicate_ratio=%.6f strongest_weight_share=%.9f "
                    "decision=%s\n",
                    active.size(), handoff_vertices,
                    sample.avg_distinct_degree, sample.duplicate_ratio,
                    sample.strongest_weight_share,
                    worthwhile ? "sparsify" : "keep");
            if (worthwhile) {
                const auto stats =
                    detail::residual_coalescer::sparsify(
                        work, active,
                        detail::kResidualSparsifyKeepProbability,
                        opts.seed ^ ws.round_index);
                if (opts.verbose)
                    std::fprintf(stderr,
                        "[apxchol] residual sparsify: active=%zu p=%.2f "
                        "edges=%zu/%zu->%zu expected=%.1f backbone=%zu "
                        "forest_candidates=%zu shards=%zu rebuild_threads=%zu "
                        "min_p=%.9g max_inv_p=%.3g "
                        "graph=%.1f->%.1f MiB stages_ms="
                        "%.3f/%.3f/%.3f/%.3f/%.3f\n",
                        active.size(),
                        detail::kResidualSparsifyKeepProbability,
                        stats.physical_before, stats.distinct_before,
                        stats.kept_edges, stats.expected_kept_edges,
                        stats.backbone_edges,
                        stats.forest_candidates, stats.forest_shards,
                        stats.rebuild_threads,
                        stats.min_offtree_probability,
                        stats.max_inverse_probability,
                        stats.bytes_before / 1048576.0,
                        stats.bytes_after / 1048576.0,
                        stats.coalesce_ms, stats.collect_ms, stats.order_ms,
                        stats.forest_ms, stats.rebuild_ms);
                if (cp) (*cp)("sparsify_residual");
            }
        }
    }

    // BK residual loop: when the main loop bailed with a large residual, run BK
    // rounds down to `residual_thresh` and let the serial peel have only the
    // tail. BK is sample-bounded, so it does not hit the main loop's
    // min_is_fraction fallback the way the bailed-out partitioner just did.
    //
    // What it buys is the elimination ORDER, not parallelism: BK's
    // degree-quantile cap picks low-degree pivots where the peel's `natural`
    // order takes whatever comes next, so the resulting cliques are smaller and
    // the factor has less fill. Most residual rounds are below `omp_threshold`
    // and run the peel's per-vertex code plus one O(|active|) BK scan.
    //
    // `parallel_residual_threshold` = SIZE_MAX means "defer to the
    // partitioner", NOT "off": block_greedy / priority_greedy declare
    // `residual_handoff_threshold` = 500, so this loop runs by default and the
    // peel sees at most 500 columns. Only a partitioner that declares no
    // threshold skips the loop. It fires only where the main loop bails with a
    // large residual (typically social graphs).
    if (active.size() > residual_thresh) {
        baumann_kyng_partitioner bk;
        // BK is sample-bounded here too, so an empty round means what it means
        // in the main loop: THIS round's hash sample whiffed, not that the
        // residual has stopped shrinking. Retry on the next round's seed under
        // the same budget; bailing on the first whiff would make the handoff
        // point an artefact of where that whiff landed.
        //
        // There is deliberately no yield-based early stop. BK's per-round yield
        // decays smoothly near the threshold, whiffed rounds are nearly free,
        // and every column moved to the peel costs both fill and time.
        size_t bk_consecutive_empty = 0;
        // BK's own round-0 seed is 2*m/|active|, and graph::m() is monotone —
        // it still counts every edge the main loop's eliminations consumed — so
        // starting BK here reads a badly inflated average degree, samples at
        // 1/(c*d) against it and under-fills its first round. Hand it the main
        // loop's last measured average instead (every partitioner that declares
        // residual_handoff_threshold also declares degree_prepass, so this is
        // populated whenever the loop can be reached; the 2*m fallback stays for
        // a hand-set parallel_residual_threshold under a prepass-less rule).
        if (last_avg_degree > 0.0)
            bk.est_avg_degree = last_avg_degree;
        while (active.size() > residual_thresh) {
            ws.reset_for_round();
            const auto& bk_part = run_partitioner(bk, work, active);
            if (bk_part.num_regions() == 0) {
                if (retry_budget_spent(bk_consecutive_empty))
                    break;                     // hand the residual to the peel
                continue;                      // whiffed sample round: retry
            }
            bk_consecutive_empty = 0;
            result.rounds.push_back({active.size(), bk_part.num_regions(),
                                     last_avg_degree, 0, 0});
            const size_t cols_before_bk =
                cp ? factor_cols.size() : 0;
            detail::eliminate_partition(
                elim, work, bk_part, factor_cols, ws, opts, cp,
                bk.selected_degree_work());
            finish_round(cols_before_bk);
        }
    }
    if (!active.empty()) {
        detail::eliminate_remaining(elim, work, active, factor_cols, ws, opts);
    }
    if (cp) (*cp)("elim_remaining");
    gpu_round_shadow.finish();

    // Quantify incidence-pool over-allocation: peak working-graph bytes vs the
    // factor's "useful" size, and the live fraction (1 - abandoned-slab share).
    // live_frac well below 1 => abandoned slabs dominate -> compaction would help.
    // Read off the working graph HERE, before it is freed below (nnz(L) is the
    // off-diagonal count + one diagonal per column -- what assemble_csc builds).
    if (opts.verbose) {
        const double MB = 1.0 / (1024.0 * 1024.0);
        size_t nnz = gpu_owned_setup_route ? result.L.nonZeros() : factor_cols.size();
        for (const auto& c : factor_cols) nnz += c.entry_count;
        const double live_frac = work.adj_live_fraction();
        std::fprintf(stderr,
            "[mem] peak_graph=%.0f MB  factor_nnz=%zu (inner=%.0f MB)  "
            "pool_live_frac=%.3f  (1/live_frac=%.2fx)\n",
            result.peak_graph_bytes * MB, nnz, nnz * sizeof(node_index) * MB,
            live_frac, live_frac > 0 ? 1.0 / live_frac : 0.0);
        {
            const size_t tot_edges = static_cast<size_t>(work.m());
            {
                std::fprintf(stderr,
                    "[mem] vec_pool_aos: grows=%zu  compactions=%zu  "
                    "min_live_frac=%.3f  undirected_edges_added=%zu  "
                    "logical_directed_payload=%.0f MB\n",
                    work.adj_grow_count(), work.adj_compact_count(),
                    work.adj_min_live_fraction(), tot_edges,
                    2.0 * tot_edges * sizeof(directed_pool_edge) * MB);
            }
        }
    }

    // ── Elimination is over: drop the elimination state BEFORE assembly ──
    // Assembly reads only factor_cols (+ n). The residual graph (now at its
    // largest: every clique edge ever added), the per-thread workspace (T
    // vertex-indexed dedup buckets + edge buffers) and the round scratch are all
    // dead here, yet they used to stay alive while assemble_csc allocated the
    // factor's CSC on top of them -- and that overlap was the process peak
    // (grid_2000: ~960 MB of dead state under a 200 MB assembly; iter0040:
    // ~300 MB under 70 MB). Freeing them first moves the peak back to the
    // elimination phase itself. Pure lifetime change: bit-identical output.
    std::vector<std::unique_ptr<std::pmr::monotonic_buffer_resource>>
        factor_entry_resources;
    factor_entry_resources.reserve(ws.threads.size());
    for (auto& t : ws.threads)
        factor_entry_resources.push_back(std::move(t.factor_entries));
    work = graph();
    ws   = factorize_workspace();
    sel  = selection();
    std::vector<node_index>().swap(active);
    std::vector<node_index>().swap(active_scratch);
    std::vector<node_index>().swap(live_degrees);
    std::vector<node_index>().swap(pre_degrees);
    std::vector<node_index>().swap(pre_scratch);
    std::vector<node_index>().swap(pre_eligible);
    std::vector<std::array<size_t, 256>>().swap(pre_histograms);
    std::vector<size_t>().swap(pre_filter_offsets);

    if (!gpu_owned_setup_route)
        detail::build_csc(result, factor_cols, n, cp);
#if defined(APXCHOL_USE_CUDA)
    if (gpu_owned_setup_route) {
        result.research_device_factor = gpu_round_shadow.finalize_device_factor(
            result.perm, result.sddm ? n : n - 1, opts.factor_storage, opts.drop_tolerance);
        if (cp) (*cp)("gpu_factor_finalize");
        if (opts.verbose) {
            const std::size_t omitted_payload_bytes =
                (static_cast<std::size_t>(result.L.nonZeros()) - n) * sizeof(detail::factor_entry);
            std::fprintf(stderr,
                "[gpu-factor-assembly] host_csc=%s host_csc_array_bytes=%zu "
                "host_metadata_bytes=%zu raw_factor_nnz=%zu "
                "host_factor_entry_download_bytes=%zu host_factor_entry_omitted_bytes=%zu\n",
                keep_factor ? "retained" : "omitted",
                result.L.inner_.capacity() * sizeof(node_index) +
                    result.L.vals_.capacity() * sizeof(factor_value_t),
                result.perm.size() * sizeof(node_index) +
                    result.L.outer_.size() * sizeof(edge_index),
                static_cast<std::size_t>(result.L.nonZeros()),
                keep_factor ? omitted_payload_bytes : 0,
                keep_factor ? 0 : omitted_payload_bytes);
        }
    }
#endif
    // The per-column ranges and their monotonic resources are consumed: free
    // both here, not at return.
    std::vector<detail::factor_col>().swap(factor_cols);
    std::vector<std::unique_ptr<std::pmr::monotonic_buffer_resource>>().swap(
        factor_entry_resources);

    if (cp) cp->ascend();

    return result;
}

// Build the tree eliminator from options. Single source of truth so the
// fixed-partitioner and runtime-dispatch paths stay in sync -- and therefore
// the place the by-route exact-core sentinel has to be resolved, since the
// eliminator is built here, before factorize_impl sees the options, and the
// GPU round shadow inspects the ELIMINATOR's copy of the knob.
inline tree_elimination make_tree_elim(const factor_options& opts,
        detail::setup_route route = detail::setup_route::cpu) {
    const std::size_t exact_core =
        opts.exact_core_max_h == exact_core_by_route
            ? (route == detail::setup_route::gpu ? 0u : exact_core_host_default)
            : opts.exact_core_max_h;
    return tree_elimination{
        .exact_clique_max_degree = opts.exact_clique_max_degree,
        .exact_core_max_h = exact_core,
        .double_cycle_min_h = opts.double_cycle_min_h,
        .sampler = opts.sampler};
}

// Default-construct-the-partitioner convenience layer.
template<typename Partitioner, typename Eliminator>
factorization factorize_impl(const Eliminator& elim,
                             graph G,
                             const factor_options& opts,
                             checkpoint* cp) {
    Partitioner partitioner;
    return factorize_impl(elim, partitioner, std::move(G), opts, cp);
}

// All factorize entry points take the graph by value (sink) and move it down
// the chain into factorize_impl's working copy — a throwaway caller pays no copy.
template<typename Partitioner>
factorization factorize(graph G,
                        const factor_options& opts,
                        checkpoint* cp) {
    detail::validate_graph_options(opts);
    return factorize_impl<Partitioner>(make_tree_elim(opts), std::move(G), opts, cp);
}

template<typename Partitioner, eliminator E>
factorization factorize(graph G, const E& elim,
                        const factor_options& opts,
                        checkpoint* cp) {
    detail::validate_graph_options(opts);
    return factorize_impl<Partitioner>(elim, std::move(G), opts, cp);
}

template<partitioner P>
factorization factorize(graph G, P part,
                        const factor_options& opts,
                        checkpoint* cp) {
    detail::validate_graph_options(opts);
    return factorize_impl(make_tree_elim(opts), part, std::move(G), opts, cp);
}

template<partitioner P, eliminator E>
factorization factorize(graph G, P part, const E& elim,
                        const factor_options& opts,
                        checkpoint* cp) {
    detail::validate_graph_options(opts);
    return factorize_impl(elim, part, std::move(G), opts, cp);
}

} // namespace apxchol
