#pragma once
/// Template implementation of the Kyng-Sachdeva approximate Cholesky factorization.
///
/// Included from factorization.h so that third-party code can use custom
/// incidence_storage backends without modifying our explicit-instantiation list.
/// The four built-in backends (vec, forward_star, bstr, vec_pool) are
/// pre-instantiated in factorization.cpp; any other backend will be
/// instantiated on demand when the user includes <apxchol/solver/factorization.h>.

#include "apxchol/solver/factorization.h"
#include "apxchol/solver/elimination/elimination.h"
#include "apxchol/solver/partitioner_helpers.h"
#include "apxchol/solver/partitioner_list.h"
#include "apxchol/solver/factorize_workspace.h"
#include "apxchol/graph/graph.h"
#include "apxchol/checkpoint.h"
#include "apxchol/env_knobs.h"
#if defined(APXCHOL_USE_CUDA)
#include "apxchol/solver/gpu_priority_frontend.h"
#endif
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace apxchol {

namespace detail {

// The IS-yield bailout measures the selector against the vertices it was
// actually allowed to choose, not against all active vertices.  With the
// default degree_quantile=0.2, using |active| as the denominator silently made
// min_is_fraction=0.05 mean "select at least 25% of the candidate pool".
//
// Near the partitioner's handoff threshold there is no profitable fallback:
// BK stops at the threshold and a bail just above it sends almost the whole
// threshold-sized residual to the singleton peel. Protect twice the handoff
// size so integer/seed noise cannot turn a productive main selector into that
// dependency-level cliff. selected==0 remains the unconditional escape from a
// stuck custom selector.
inline bool is_yield_too_small(size_t selected, size_t candidates,
                               size_t active, double min_fraction,
                               size_t residual_handoff_threshold) {
    if (selected == 0) return true;
    if (residual_handoff_threshold != std::numeric_limits<size_t>::max()) {
        const size_t protected_tail = residual_handoff_threshold >
                std::numeric_limits<size_t>::max() / 2
            ? std::numeric_limits<size_t>::max()
            : 2 * residual_handoff_threshold;
        if (active <= protected_tail) return false;
    }
    return selected < candidates * min_fraction;
}

inline bool selection_should_handoff(
        size_t selected_regions, size_t selected_vertices, size_t candidates,
        size_t active, double min_fraction,
        size_t residual_handoff_threshold, size_t omp_threshold) {
    if (selected_regions == 0) return true;
    if (omp_threshold != 0 && selected_vertices >= omp_threshold) return false;
    return is_yield_too_small(selected_regions, candidates, active,
                              min_fraction, residual_handoff_threshold);
}

// The relative yield at which the main selector stops should reflect both the
// size and the density of the residual. BG/priority-greedy rescan it; BK samples
// bounded edge work after an O(active) hash pass. Keep min_is_fraction as the
// sparse/small-residual base and preserve zero as "disable yield bailout".
//
// Adapt only while at least four handoff-sized chunks remain, so BK has enough
// runway to amortize taking over and small residuals do not fall off a level
// cliff. A residual whose average degree is itself at least the final-tail size
// triples the base. Sparse residuals keep the user's base: switching them early
// creates thousands of thin BK levels and can make the solve dominate the setup
// saving. The 0.15 ceiling bounds the quality trade even if a caller chose a
// larger handoff threshold.
inline double adaptive_is_yield_fraction(
        double base, size_t active, double average_degree,
        size_t residual_handoff_threshold) {
    if (!(base > 0.0) ||
        residual_handoff_threshold == std::numeric_limits<size_t>::max() ||
        residual_handoff_threshold > std::numeric_limits<size_t>::max() / 4 ||
        active <= 4 * residual_handoff_threshold ||
        average_degree < static_cast<double>(residual_handoff_threshold))
        return base;
    return std::max(base, std::min(0.15, 3.0 * base));
}

// The old elimination gate used only the number of independent vertices.
// That misses small dense rounds: processing 500 columns with 200 live edges
// each is much more work than processing 2000 mesh columns. Reuse the degree
// prepass as a zero-traversal-cost work hint and compare it with the old vertex
// threshold expressed as adjacency slots. 24 slots/vertex is conservative:
// the representative sparse controls peak below this boundary, while the
// social-graph rounds it admits carry 48k-200k slots each.
inline constexpr size_t kEliminationWorkPerVertex = 24;

inline bool elimination_parallel_worthwhile(
        size_t vertices, size_t adjacency_work, size_t vertex_threshold,
        size_t team_threads) {
    if (team_threads <= 1) return false;
    if (vertices > vertex_threshold) return true;
    const size_t work_threshold = vertex_threshold >
            std::numeric_limits<size_t>::max() / kEliminationWorkPerVertex
        ? std::numeric_limits<size_t>::max()
        : vertex_threshold * kEliminationWorkPerVertex;
    return adjacency_work > work_threshold;
}

// Eliminate vertices in the partition: record L-columns, add clique edges.
// Partition vertices across regions are pairwise non-adjacent (for singleton
// regions, every vertex is its own region — same constraint as the old IS).
// The computation phase (gather neighbors, build factor column, sample
// clique edges) runs in parallel with per-thread RNGs; graph mutation
// (add_edge, deactivate) is deferred and applied sequentially because
// the forward_star node pool is shared.

// Deferred clique edge type is defined in elimination.h.

#ifndef NDEBUG
// Debug-only: verify the partitioner's contract — selected vertices must be
// pairwise non-adjacent (they are eliminated independently in parallel).
template<typename Incidence>
void assert_partition_independent(graph<Incidence>& G,
                                  const partition_result& part) {
    static thread_local std::vector<char> in_part;
    if (in_part.size() < size_t(G.n())) in_part.assign(G.n(), 0);
    for (auto v : part.data) in_part[v] = 1;
    for (auto v : part.data)
        for (auto idx : G.adj(v)) {
            auto u = G.edge_target(idx, v);
            assert(!(G.is_active(u) && in_part[u]) &&
                   "partitioner contract violation: selected vertices are "
                   "adjacent (the partition must be an independent set)");
        }
    for (auto v : part.data) in_part[v] = 0;
}
#endif

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
                    std::uint64_t run_seed,
                    factor_col& col,
                    factorize_workspace::per_thread& ws,
                    bool dedup_inline = false) {
    auto& nbrs       = ws.neighbors;
    auto& excess_out = ws.excess_buffer;
    col.entries = nullptr;
    col.entry_count = 0;
    nbrs.clear();
    double edge_deg = 0.0;
    if (dedup_inline) {
        static constexpr node_index npos = node_index(-1);
        auto& first   = ws.dedup_bucket;
        auto& touched = ws.dedup_touched;
        if (first.size() < size_t(G.n())) first.assign(G.n(), npos);
        touched.clear();
        for (auto [u, w] : G.neighbors(v)) {
            if (!G.is_active(u)) continue;
            if (first[u] == npos) {
                first[u] = static_cast<node_index>(nbrs.size());
                touched.push_back(u);
                nbrs.emplace_back(u, w);
            } else {
                nbrs[first[u]].weight += w;
            }
            edge_deg += w;
        }
        for (auto t : touched) first[t] = npos;
    } else {
        for (auto [u, w] : G.neighbors(v)) {
            nbrs.emplace_back(u, w);
            edge_deg += w;
        }
    }

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
    // apply_deferred_edges) through the append-only emitter. The seed is a
    // pure function of (run seed, vertex): draws are identical at any thread
    // count and schedule.
    const std::uint64_t vseed =
        run_seed ^ ((std::uint64_t(v) + 1) * 0x9E3779B97F4A7C15ULL);
    elim.sample_clique(std::span<weighted_neighbor>(nbrs), total_deg, vseed,
                       edge_emitter(ws.edge_buffer));
}

template<typename Eliminator, incidence_storage Incidence>
void eliminate_partition_singleton(const Eliminator& elim,
                                   graph<Incidence>& G,
                                   const partition_result& part,
                                   std::vector<factor_col>& factor_cols,
                                   factorize_workspace& ws,
                                   const factor_options& opts,
                                   checkpoint* cp = nullptr,
                                   bool capture_gpu_topology = false,
                                   size_t work_hint = 0) {
    const size_t n_verts = part.num_vertices();
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
    if (capture_gpu_topology) {
        ws.gpu_topology_updates.clear();
        ws.gpu_topology_batches.clear();
    }

    #ifdef _OPENMP
    // Rounds below omp_threshold take the serial path unless the degree
    // prepass says their selected columns still carry enough adjacency work.
    // This differs from the retired APXCHOL_TAIL_THREADS experiment: that rule
    // parallelized every small tail by vertex count and regressed sparse IPM /
    // grid rounds; this gate leaves those rounds serial and admits only dense
    // work. The parallel path applies clique edges in thread-arrival order, so
    // the conservative work boundary also limits structural exposure.
    const int team_threads = static_cast<int>(ws.threads.size());
    const bool parallel_round = elimination_parallel_worthwhile(
        n_verts, work_hint, opts.omp_threshold,
        static_cast<size_t>(team_threads));
    if (parallel_round) {
        // Team size for the fused paths: the full workspace team.
        if constexpr (std::is_same_v<Incidence, forward_star_incidence>) {
            // Mega-fused parallel region: compute + deactivate + apply all under
            // one fork-join.  On BG/BK with ~1k+ rounds this saves ~2 extra
            // fork-joins/round vs the post-38ffdbc split.
            // process_vertex does inline dedup + dead-edge filter so we
            // skip the explicit merge_parallel_edges pass.
            const int num_threads = team_threads;
            std::vector<size_t> e_offsets(num_threads + 1, 0);
            edge_index e_start = 0;
            edge_index a_start = 0;   // adj-pool base offset

            #pragma omp parallel num_threads(num_threads)
            {
                int tid = omp_get_thread_num();

                // Clear per-thread buffers for this round.
                ws.threads[tid].edge_buffer.clear();
                ws.threads[tid].excess_buffer.clear();

                #pragma omp for schedule(dynamic, 64)
                for (size_t k = 0; k < n_verts; ++k)
                    process_vertex(elim, G, part.data[k], opts.seed, output_col(k),
                                   ws.threads[tid],
                                   /*dedup_inline=*/true);
                // implicit barrier — edge_buffers populated before prefix-sum

                // Single thread does prefix-sum + pool reservation.
                #pragma omp single
                {
                    for (int t = 0; t < num_threads; ++t)
                        e_offsets[t + 1] = e_offsets[t] + ws.threads[t].edge_buffer.size();
                    const size_t N_edges = e_offsets[num_threads];
                    if (N_edges > 0) {
                        e_start = G.reserve_edge_pool(static_cast<edge_index>(N_edges));
                        a_start = G.reserve_adj_pool(static_cast<edge_index>(2 * N_edges));
                    }
                }
                // implicit barrier after single — offsets/pools visible

                // Apply phase: each thread writes to its slot range and
                // pushes onto adj chains via CAS.
                {
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
                    ws.threads[tid].edge_buffer.clear();
                    // Apply excess atomically (each thread owns its own buffer).
                    for (auto [u, delta] : ws.threads[tid].excess_buffer)
                        G.atomic_add_excess(u, delta);
                    ws.threads[tid].excess_buffer.clear();
                }

                // Deactivate partition vertices in parallel (disjoint vertices,
                // distinct head_[] entries from those touched by atomic pushes).
                #pragma omp for schedule(static) nowait
                for (size_t k = 0; k < n_verts; ++k)
                    G.set_inactive_unchecked(part.data[k]);
            }
            G.bulk_decrement_active(static_cast<node_index>(n_verts));

            // NOTE: the mega-fused parallel region runs compute + apply + apply_excess
            // back-to-back inside one fork-join, so we cannot attribute these
            // sub-phases separately without breaking the fusion. Emit one honest
            // label that reflects what was actually measured.
            if (cp) (*cp)("compute+apply_fused");
        } else if constexpr (is_vec_pool_incidence_v<Incidence>) {
            // Mega-fused for vec_pool: compute + apply pre-pass + parallel
            // atomic push + deactivate all in ONE parallel region. Saves
            // the fork-join overhead of the legacy 2-region pattern
            // (~10us × num_threads per round). The omp single section does
            // the vec_pool-specific work: prefix-sum of thread edge counts,
            // edge-pool reservation, per-vertex incoming count, and
            // serial reserve_for grow (reserve_for is NOT thread-safe).
            // The actual atomic_push_reserved phase is parallel and lock-free.
            const int num_threads = team_threads;
            std::vector<size_t> e_offsets(num_threads + 1, 0);
            // Persistent all-zero histogram (see factorize_workspace::incoming);
            // sized once, reset per round over the touched entries only.
            if (ws.incoming.size() < static_cast<size_t>(G.n()))
                ws.incoming.assign(static_cast<size_t>(G.n()), 0);
            std::vector<node_index>& incoming = ws.incoming;
            edge_index e_start = 0;

            #pragma omp parallel num_threads(num_threads)
            {
                int tid = omp_get_thread_num();

                ws.threads[tid].edge_buffer.clear();
                ws.threads[tid].excess_buffer.clear();

                #pragma omp for schedule(dynamic, 64)
                for (size_t k = 0; k < n_verts; ++k)
                    process_vertex(elim, G, part.data[k], opts.seed, output_col(k),
                                   ws.threads[tid],
                                   /*dedup_inline=*/true);
                // implicit barrier — edge_buffers populated before pre-pass

                // Parallel atomic histogram of incoming edges per vertex.
                // Each thread sweeps its own edge_buffer and atomic-increments
                // incoming[u] and incoming[v]. On first bump from 0→1 we push
                // the vertex onto the per-thread `touched_buffer` so the
                // serial reserve_for loop below can skip the full O(G.n())
                // scan and iterate only the touched vertices.
                {
                    auto& my_touched = ws.threads[tid].touched_buffer;
                    my_touched.clear();
                    const auto& my_buf = ws.threads[tid].edge_buffer;
                    for (const auto& e : my_buf) {
                        if (__atomic_fetch_add(&incoming[e.u], 1, __ATOMIC_RELAXED) == 0)
                            my_touched.push_back(e.u);
                        if (__atomic_fetch_add(&incoming[e.v], 1, __ATOMIC_RELAXED) == 0)
                            my_touched.push_back(e.v);
                    }
                }
                #pragma omp barrier  // incoming[] complete before reserve_for

                // Single: prefix-sum thread edge offsets, reserve the indexed
                // edge pool when present, and concatenate touched buffers for
                // the adjacency bulk reserve below.
                #pragma omp single
                {
                    for (int t = 0; t < num_threads; ++t)
                        e_offsets[t + 1] = e_offsets[t] + ws.threads[t].edge_buffer.size();
                    const size_t N_edges = e_offsets[num_threads];
                    if (N_edges > 0) {
                        if constexpr (graph<Incidence>::stores_directed_incidence)
                            G.record_edges_added(static_cast<edge_index>(N_edges));
                        else
                            e_start = G.reserve_edge_pool(static_cast<edge_index>(N_edges));
                    }
                    ws.touched_concat.clear();
                    for (int t = 0; t < num_threads; ++t)
                        ws.touched_concat.insert(
                            ws.touched_concat.end(),
                            ws.threads[t].touched_buffer.begin(),
                            ws.threads[t].touched_buffer.end());
                }
                // implicit barrier after single — e_start + touched_concat visible.

                // Bulk parallel reserve_for: replaces the per-touched serial
                // reserve_for loop. Per round trace on IPM iter40 (16T) shows
                // the serial reserve was 61% of round time (mostly std::copy_n
                // inside grow); bulk drops it to 22% by parallelizing the
                // per-vertex slab copies under one pool_.resize. 10-rep
                // paired-A/B confirms ~5% total bench win on IPM iter40,
                // neutral on grid_2000 (small-slab workloads see the parallel-
                // for overhead match the saved copy work).
                if (e_offsets[num_threads] > 0) {
                    G.adj_bulk_reserve_parallel(
                        ws.touched_concat.begin(),
                        ws.touched_concat.end(),
                        incoming);
                }

                // Apply phase: parallel atomic push into pre-reserved slabs.
                {
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
                    if (!capture_gpu_topology)
                        ws.threads[tid].edge_buffer.clear();
                    // Apply excess atomically (each thread's own buffer).
                    for (auto [u, delta] : ws.threads[tid].excess_buffer)
                        G.atomic_add_excess(u, delta);
                    ws.threads[tid].excess_buffer.clear();
                }

                // Restore the all-zero invariant of ws.incoming over exactly
                // the touched vertices (every vertex with incoming > 0 was
                // pushed once, on its 0->1 bump). Safe here: the only reader
                // of incoming[] is bulk_reserve_parallel's single section,
                // which ended at that call's trailing omp-for barrier.
                #pragma omp for schedule(static) nowait
                for (size_t i = 0; i < ws.touched_concat.size(); ++i)
                    incoming[ws.touched_concat[i]] = 0;

                // Deactivate partition vertices in parallel (disjoint).
                #pragma omp for schedule(static) nowait
                for (size_t k = 0; k < n_verts; ++k)
                    G.set_inactive_unchecked(part.data[k]);
            }
            G.bulk_decrement_active(static_cast<node_index>(n_verts));

            if (capture_gpu_topology) {
                ws.gpu_topology_batches.reserve(ws.threads.size());
                for (const auto& thread : ws.threads) {
                    if (!thread.edge_buffer.empty()) {
                        ws.gpu_topology_batches.push_back(
                            {thread.edge_buffer.data(),
                             thread.edge_buffer.size()});
                    }
                }
            }

            if (cp) (*cp)("compute+apply_fused");
        } else {
            // Legacy backends (vec/bstr): separate parallel compute +
            // serial apply (apply_deferred_edges has no fast path for these).
            // Skip the explicit merge_parallel_edges pass: process_vertex
            // does inline dedup + dead-edge filter, saving a full
            // adjacency traversal per IS vertex per round.
            // Team: the OpenMP default (as before).
            #pragma omp parallel num_threads(omp_get_max_threads())
            {
                int tid = omp_get_thread_num();

                // Clear per-thread buffers for this round.
                ws.threads[tid].edge_buffer.clear();
                ws.threads[tid].excess_buffer.clear();

                #pragma omp for schedule(dynamic, 64) nowait
                for (size_t k = 0; k < n_verts; ++k)
                    process_vertex(elim, G, part.data[k], opts.seed, output_col(k),
                                   ws.threads[tid],
                                   /*dedup_inline=*/true);
            }
            if (cp) { (*cp)("merge_is"); (*cp)("compute"); }

            // Apply phase via helpers (replaces previous inline serial apply).
            detail::apply_deferred_edges(G, ws, cp);
            detail::apply_deferred_excess(G, ws, cp);

            for (size_t k = 0; k < n_verts; ++k)
                G.deactivate(part.data[k]);
        }
    } else
    #endif
    {
        // Serial path: per-vertex inline application for cache locality.
        // Inline dedup in process_vertex replaces merge_parallel_edges.
        auto& wt = ws.threads[0];
        for (size_t k = 0; k < n_verts; ++k) {
            wt.edge_buffer.clear();
            wt.excess_buffer.clear();
            process_vertex(elim, G, part.data[k], opts.seed, output_col(k), wt,
                           /*dedup_inline=*/true);
            for (auto [u, v, w] : wt.edge_buffer) {
                if (capture_gpu_topology)
                    ws.gpu_topology_updates.push_back({u, v});
                G.add_edge(u, v, w);
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
template<typename Eliminator, incidence_storage Incidence>
void eliminate_partition(const Eliminator& elim,
                         graph<Incidence>& G,
                         const partition_result& part,
                         std::vector<factor_col>& factor_cols,
                         factorize_workspace& ws,
                         const factor_options& opts,
                         checkpoint* cp = nullptr,
                         bool capture_gpu_topology = false,
                         size_t work_hint = 0) {
    if (cp) { cp->descend("eliminate"); cp->tick(); }
    const size_t n_verts = part.num_vertices();
    // Diagnostic only: work_hint is the selected vertices' live-degree sum,
    // already computed by the caller for the elimination gate. Keeping it on
    // the existing opt-in round trace makes work-gated rounds auditable without
    // another graph traversal or any default-path output.
    if (std::getenv("APXCHOL_ROUND_TRACE"))
        std::fprintf(stderr, "[round] n_verts=%zu adjacency_work=%zu\n",
                     n_verts, work_hint);
    if (n_verts == 0) {
        if (cp) cp->ascend();
        return;
    }
    eliminate_partition_singleton(elim, G, part, factor_cols, ws, opts, cp,
                                  capture_gpu_topology, work_hint);
    if (cp) cp->ascend();
}

// Eliminate remaining vertices after the main IS-elimination loop.
// Used when IS fraction drops below threshold — avoids the O(|active|)
// IS-finding scan when only a few vertices can be chosen anyway.
//
// It does NOT see the whole residual. The caller runs BK rounds first, down to
// the partitioner's `residual_handoff_threshold` (500 for every shipped rule
// that can bail), so this peels a bounded tail — 500 columns on the social
// graphs, 76 on iter0040, 1 on grid_2000. An older note here claimed the
// opposite ("an earlier version tried parallel BK rounds ... serial peel was
// strictly faster"); the BK residual loop has in fact been on by default all
// along, and stopping it early to feed this function more columns is a measured
// loss on both fill and setup — see the residual-loop comment in
// factorize_impl.
template<typename Eliminator, typename Incidence>
void eliminate_remaining(const Eliminator& elim,
                         graph<Incidence>& G,
                         std::vector<node_index>& active,
                         std::vector<factor_col>& factor_cols,
                         factorize_workspace& ws,
                         const factor_options& opts) {
    auto& wt = ws.threads[0];
    factor_col col;

    auto peel_one = [&](node_index v) {
        wt.edge_buffer.clear();
        wt.excess_buffer.clear();
        process_vertex(elim, G, v, opts.seed, col, wt,
                       /*dedup_inline=*/true);
        factor_cols.push_back(std::move(col));
        for (auto [a, b, w] : wt.edge_buffer)
            G.add_edge(a, b, w);
        for (auto [u, delta] : wt.excess_buffer)
            G.excess(u) += delta;
        G.deactivate(v);
    };

    if (opts.residual_peel == residual_peel_strategy::min_degree) {
        // Greedy min-degree using a binary heap with lazy revalidation.
        // Initial pass: compute degree for every active vertex once.
        // Each pop revalidates by recomputing degree via prune_and_degree;
        // if the stored degree no longer matches the live degree (a neighbour
        // was eliminated or a clique edge raised it), re-push with the fresh
        // value and continue popping.  Amortised cost: O((|active| + Δ) log)
        // where Δ is the total degree growth caused by clique fill-in.
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

} // namespace detail

// Takes the graph BY VALUE (a sink). Elimination mutates the working graph, so
// it needs its own copy. Callers that pass an rvalue (the runtime-dispatch path,
// which builds a throwaway via make_graph) move into the parameter for free;
// callers that pass an lvalue (and want to keep their graph) copy once here —
// same cost as the old defensive `graph work(G)`.
template<typename Partitioner, typename Eliminator, incidence_storage Incidence>
factorization factorize_impl(const Eliminator& elim,
                             Partitioner& partitioner,
                             graph<Incidence> G,
                             const factor_options& opts_in,
                             checkpoint* cp) {
    const node_index n = G.n();
    if (n == 0)
        return {};

    // APXCHOL_OMP_THRESHOLD (experiment knob, see env_knobs.h) overrides
    // factor_options::omp_threshold for this factorization -- it flows from
    // here into the partitioner context, the degree prepass and the per-round
    // serial/parallel elimination gate. Unset = opts_in unchanged.
    const factor_options opts = [&] {
        factor_options o = opts_in;
        const long ov = detail::env_knobs::get().omp_threshold;
        if (ov >= 0) o.omp_threshold = static_cast<size_t>(ov);
        return o;
    }();

    if (cp) cp->descend("setup");

    factorization result;
    if (cp) cp->tick();
    graph<Incidence> work(std::move(G));
    if (cp) (*cp)("graph_copy");

    if constexpr (std::is_same_v<Incidence, forward_star_incidence>) {
        if (opts.fs_filter_append) work.set_adj_filter_append(true);
    }

    // Detect SDDM: any vertex with positive excess means the matrix
    // is positive definite (not just semidefinite like a Laplacian).
    // Note: make_graph already filters out FP noise (excess < diag * 1e-12),
    // so any remaining positive excess is genuine.
    for (node_index v = 0; v < n; ++v) {
        if (work.excess(v) > 0.0) { result.sddm = true; break; }
    }
    if (cp) (*cp)("sddm_scan");

    factorize_workspace ws;
    {
        int num_threads_factorize = 1;
#ifdef _OPENMP
        num_threads_factorize = omp_get_max_threads();
#endif
        ws.threads.resize(num_threads_factorize);
        for (auto& t : ws.threads)
            t.factor_entries =
                std::make_unique<std::pmr::monotonic_buffer_resource>();
    }
    std::vector<detail::factor_col> factor_cols;
    factor_cols.reserve(n);

    constexpr bool sample_bounded = partitioner_sample_bounded_v<Partitioner>;
    constexpr size_t residual_handoff_default =
        partitioner_residual_handoff_v<Partitioner>;
    size_t residual_thresh = opts.parallel_residual_threshold;
    if (residual_thresh == std::numeric_limits<size_t>::max())
        residual_thresh = residual_handoff_default;

    // Active vertex list (natural index order) — filtered in-place each round.
    std::vector<node_index> active(n);
    std::ranges::iota(active, node_index{0});
    std::vector<node_index> active_scratch;

#if defined(APXCHOL_USE_CUDA)
    constexpr bool gpu_priority_frontend_eligible =
        std::is_same_v<Partitioner, priority_greedy_partitioner> &&
        std::is_same_v<Incidence, vec_pool_incidence> &&
        std::is_same_v<std::remove_cvref_t<Eliminator>, detail::tree_elimination>;
    constexpr bool gpu_block_frontend_eligible =
        std::is_same_v<Partitioner, block_greedy_partitioner> &&
        std::is_same_v<Incidence, vec_pool_incidence> &&
        std::is_same_v<std::remove_cvref_t<Eliminator>, detail::tree_elimination>;
    constexpr bool gpu_frontend_eligible =
        gpu_priority_frontend_eligible || gpu_block_frontend_eligible;
    std::unique_ptr<detail::gpu_priority_frontend> gpu_frontend;
    detail::gpu_priority_frontend::mode gpu_frontend_mode =
        detail::gpu_priority_frontend::mode::disabled;
    if constexpr (gpu_frontend_eligible) {
        if constexpr (gpu_block_frontend_eligible)
            gpu_frontend_mode =
                detail::gpu_priority_frontend::configured_block_mode();
        else
            gpu_frontend_mode =
                detail::gpu_priority_frontend::configured_mode();
        if (gpu_frontend_mode != detail::gpu_priority_frontend::mode::disabled &&
            opts.exact_clique_max_degree != 0) {
            throw std::invalid_argument(
                "the forced GPU setup front-end requires the default "
                "d-1-edge tree sampler (exact clique mode can grow topology)");
        }
        if (gpu_frontend_mode != detail::gpu_priority_frontend::mode::disabled) {
            const auto runtime = detail::gpu_priority_frontend::probe_runtime(
                n, static_cast<std::size_t>(work.m()),
                gpu_block_frontend_eligible);
            if (!runtime.cooperative_launch || !runtime.memory_fits) {
                throw std::runtime_error(
                    !runtime.cooperative_launch
                        ? "the forced GPU setup front-end requires a CUDA device "
                          "with cooperative-kernel launch support"
                        : "the forced GPU setup front-end does not fit in currently "
                          "free device memory");
            }
            if (cp) (*cp)("gpu_frontend_probe");
            std::vector<detail::gpu_topology_edge> initial_topology;
            initial_topology.reserve(static_cast<std::size_t>(work.m()));
            for (node_index v = 0; v < n; ++v) {
                for (auto idx : work.adj(v)) {
                    const node_index u = work.edge_target(idx, v);
                    if (v < u) initial_topology.push_back({v, u});
                }
            }
            gpu_frontend = std::make_unique<detail::gpu_priority_frontend>(
                n, initial_topology);
            if (cp) (*cp)("gpu_frontend_init");
        }
    }
#endif

    // The partitioner's view of the run-constant services, the shared
    // selection structure, and the degree-prepass scratch (all owned here).
    selection sel;
    std::vector<node_index> pre_degrees, pre_scratch, pre_eligible;
    std::vector<std::array<size_t, 256>> pre_histograms;
    std::vector<size_t> pre_filter_offsets;
    std::vector<node_index> live_degrees;   // vertex-indexed, for ctx.degrees

    // Shared round front-end: prepass (when the partitioner's trait asks for
    // it) + find_partition + finalize, with uniform profiling labels.
    // last_avg_degree feeds the per-round stats; it is the prepass's exact
    // average (0 = not measured, for partitioners without the prepass).
    double last_avg_degree = 0.0;
    size_t last_candidate_count = 0;
#if defined(APXCHOL_USE_CUDA)
    size_t gpu_elimination_work_hint = 0;
#endif
    auto run_partitioner = [&](auto& p, graph<Incidence>& g,
                               std::span<const node_index> act)
        -> const partition_result& {
        using P = std::remove_reference_t<decltype(p)>;
#if defined(APXCHOL_USE_CUDA)
        if constexpr (gpu_frontend_eligible &&
                      (std::is_same_v<P, priority_greedy_partitioner> ||
                       std::is_same_v<P, block_greedy_partitioner>)) {
            if (gpu_frontend) {
                if (cp) { cp->descend("find_partition"); cp->tick(); }
                const auto prep = gpu_frontend->prepare(act, opts.partition);
                last_candidate_count = prep.candidate_count;
                last_avg_degree = prep.average_degree;
                if (cp) (*cp)("prune");
                const partition_result* part = nullptr;
                if constexpr (std::is_same_v<P,
                                             priority_greedy_partitioner>) {
                    part = &gpu_frontend->select(opts.seed, p.round);
                    ++p.round;
                } else {
                    part = &gpu_frontend->select_block_greedy();
                }
                gpu_elimination_work_hint =
                    gpu_frontend->selected_degree_work();
                if (cp) {
                    (*cp)("select");
                    (*cp)("collect");
                    cp->ascend();
                }
                return *part;
            }
        }
#endif
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
            const double avg_deg =
                prune_and_degrees(g, act, pre_degrees, opts.omp_threshold);
            const double q = opts.partition.degree_quantile;
            const double thr = q > 0.0 && q < 1.0
                                   && act.size() > opts.omp_threshold
                ? static_cast<double>(parallel_degree_quantile(
                      pre_degrees, act.size(), q, pre_histograms))
                : is_degree_threshold(pre_degrees, act.size(), avg_deg,
                                      opts.partition, pre_scratch);
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
    constexpr size_t kMaxEmptyRounds = 64;

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
                    part.num_regions(), part.num_vertices(),
                    last_candidate_count, active.size(), min_yield,
                    residual_thresh, opts.omp_threshold)) {
                if (std::getenv("APXCHOL_VERBOSE"))
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
                if (++consecutive_empty >= kMaxEmptyRounds)
                    break;                     // hand the residual to the peel
                ++ws.round_index;              // keep retry rounds distinguishable
                continue;                      // whiffed sample round: retry
            }
            consecutive_empty = 0;
        }
        result.rounds.push_back({active.size(), part.num_regions(), last_avg_degree, 0, 0});

        const size_t cols_before = cp ? factor_cols.size() : 0;
        size_t elimination_work_hint = 0;
        if constexpr (partitioner_degree_prepass_v<Partitioner>) {
#if defined(APXCHOL_USE_CUDA)
            if (gpu_frontend) {
                elimination_work_hint = gpu_elimination_work_hint;
            } else
#endif
            {
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
        }
        const bool capture_gpu_topology =
#if defined(APXCHOL_USE_CUDA)
            static_cast<bool>(gpu_frontend);
#else
            false;
#endif
        detail::eliminate_partition(elim, work, part, factor_cols, ws, opts, cp,
                                    capture_gpu_topology,
                                    elimination_work_hint);
#if defined(APXCHOL_USE_CUDA)
        if (gpu_frontend) {
            gpu_frontend->advance(part.data, ws.gpu_topology_updates,
                                  ws.gpu_topology_batches);
            if (cp) (*cp)("gpu_frontend_advance");
        }
#endif
        // Tally nnz added this round (only when caller wants the stats).
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

        // Auto-compact forward_star adjacency pool when fragmentation
        // crosses the configured threshold.  Avoids unbounded pointer-
        // chase as filter() leaves orphan nodes in nodes_.
        //
        // Default is off (fs_compact_threshold == 0): on sparse graphs
        // (uniform / weighted grids) compaction is pure overhead and
        // adds 70-100% to setup time.  It pays off only on dense fill
        // workloads (LP-IPM Schur complements, ~30% setup gain at
        // thresh=0.75) but even there the [vec] storage backend is
        // ~2x faster than [fwd_star]+compact, so the right answer for
        // dense matrices is to switch storage rather than turn this on.
        // Pass --fs-compact 0.75 explicitly if you do want it.
        if constexpr (std::is_same_v<Incidence, forward_star_incidence>) {
            if (opts.fs_compact_threshold > 0.0 &&
                work.adj_live_fraction() < opts.fs_compact_threshold) {
                work.compact_adj();
                if (cp) (*cp)("compact_adj");
            }
        }
        // vec_pool: pool defrag happens inside vec_pool's bulk_reserve_parallel
        // (built in, always on). Here we only sample the worst fragmentation
        // seen, for the APXCHOL_MEM_DUMP diagnostic.
        if constexpr (is_vec_pool_incidence_v<Incidence>)
            work.adj_note_live_fraction();
    }

    // Eliminate any remaining vertices (0 or 1 after the while loop,
    // or all remaining when the candidate-yield fallback triggered above).
    //
    // BK residual loop: when the main loop bailed with a large residual, run BK
    // rounds down to `residual_thresh` and let the serial peel have only the
    // tail.  BK is sample-bounded, so it does not hit the main loop's
    // min_is_fraction fallback the way the bailed-out partitioner just did.
    //
    // What it buys is the elimination ORDER, not parallelism: BK's
    // degree-quantile cap picks low-degree pivots where the peel's `natural`
    // order takes whatever comes next, and the resulting cliques are much
    // smaller.  The rounds themselves are mostly NOT parallel — their IS is far
    // below `omp_threshold`, so eliminate_partition takes its serial branch,
    // which is the peel's per-vertex code (as-Skitter / coAuthorsDBLP /
    // com-Amazon: max IS 231 / 160 / 171 over every residual round, i.e. never;
    // com-LiveJournal: 112 rounds of 13336 above the gate).  Each round is
    // therefore one O(|active|) BK scan on top of elimination work the peel
    // would have done anyway, and it still wins because the pivot that scan
    // buys makes the work cheaper.  Over the 4825 as-Skitter vertices between
    // the two handoff points (8 paired reps, medians): the rounds cost 202 ms of
    // `find_partition` + 145 ms of `eliminate` = ~72 us/vertex, against 397 ms
    // of `eliminate_remaining` = ~82 us/column for the same vertices in the
    // peel — and they leave 4.9% less fill behind.
    //
    // Default: `parallel_residual_threshold` = SIZE_MAX means "defer to the
    // partitioner", NOT "off" — block_greedy / priority_greedy both declare
    // `residual_handoff_threshold` = 500, so this loop runs by default and the
    // peel sees at most 500 columns.  Only a partitioner that declares no
    // threshold leaves it at SIZE_MAX and skips the loop entirely.  It fires
    // only where the main loop bails with a large residual, i.e. on social
    // graphs (under the candidate-relative yield rule, as-Skitter enters at
    // 1696-2122 active across the measured seeds and com-LiveJournal at 30882;
    // iter0040 and grids finish the main loop below the handoff, so they never
    // enter this path).
    //
    // A heavily duplicated residual is rebuilt once before BK. This is exact
    // with respect to the partitioner's multigraph degree: coalesce_active()
    // stores each endpoint pair once but carries the represented edge count in
    // a sidecar consumed by prune_and_degree(). The numerical pivot already
    // aggregated those parallel weights in fp64, so only one fp32 store of the
    // sum can perturb the later sampler. The conservative 256-vertex probe and
    // ratio >= 4 gate are load-bearing: as-Skitter estimates 4.96-6.18 and its
    // residual phase falls 185-192 -> 145-152 ms while graph heap falls about
    // 720 -> 46-49 MB; com-LiveJournal estimates 1.20 and rebuilding it is a
    // large regression; kron seed 42 estimates 3.91 and sits at break-even.
    // APXCHOL_RESIDUAL_COALESCE=0 is the rollback.
    if constexpr (std::is_same_v<Incidence, vec_pool_incidence>) {
        const char* enabled_env =
            std::getenv("APXCHOL_RESIDUAL_COALESCE");
        const bool enabled = !enabled_env || !*enabled_env ||
                             std::strcmp(enabled_env, "0") != 0;
        constexpr size_t kMinActive = 1024;
        constexpr double kMinDuplicateRatio = 4.0;
        if (enabled && active.size() > residual_thresh &&
            active.size() >= kMinActive) {
            const double estimated_ratio =
                detail::residual_coalescer<Incidence>::estimate(work, active);
            if (estimated_ratio >= kMinDuplicateRatio) {
                const auto stats =
                    detail::residual_coalescer<Incidence>::rebuild(work, active);
                if (std::getenv("APXCHOL_VERBOSE")) {
                    std::fprintf(stderr,
                                 "[apxchol] residual coalesce: active=%zu "
                                 "estimate=%.3f edges=%zu->%zu graph=%.1f->%.1f MiB\n",
                                 active.size(), estimated_ratio,
                                 stats.multi_edges, stats.distinct_edges,
                                 stats.bytes_before / 1048576.0,
                                 stats.bytes_after / 1048576.0);
                }
                if (cp) (*cp)("coalesce_residual");
            }
        }
    }

    if (active.size() > residual_thresh) {
        baumann_kyng_partitioner bk;
        // BK is sample-bounded here too, so an empty round means what it means
        // in the main loop: THIS round's hash sample whiffed, not that the
        // residual has stopped shrinking.  Retry on the next round's seed under
        // the same budget.  Bailing on the first empty round (the behaviour
        // until 2026-08-20) made the handoff point an artefact of where the
        // first whiff happened to land — coAuthorsDBLP stopped at 507 active
        // (harmless, the threshold is 500) but as-Skitter stopped at 5325 and
        // handed all of it to the serial peel, while BK was still eliminating
        // ~25 vertices per round.
        //
        // NO YIELD-BASED EARLY STOP. BK's per-round yield does decay to well
        // under one vertex by the time `active` reaches the handoff threshold
        // (as-Skitter ~80 vertices/round on entry, ~0.4/round at active = 600 —
        // a smooth power-law decay, no cliff to aim a rule at), and a
        // window-average "stop when the last 32 rounds averaged < y" rule was
        // built and measured against exactly that. REJECTED: it buys no
        // measurable setup and gives up fill.
        //
        // Same-binary A/B, arms by env, interleaved, 8 paired reps, T=16,
        // vec_pool, bg+tree, seed 42. Every counter is deterministic (identical
        // in all 8 reps of each arm); the timings are medians:
        //
        //   as-Skitter        stop at   nnz(L)       setup    find_part  elim_rem
        //   no rule (ship)        500   18 479 113   0.981x    1171 ms     26 ms
        //   yield < 1.0          2069   18 648 027   0.973x    1084 ms    168 ms
        //   (baseline: bail on the first whiff, 5325)  19 430 230  1.000x
        //                                                       969 ms    423 ms
        //
        // The setup column is a wash — the honest noise floor here is iter0040,
        // which is BIT-IDENTICAL across all arms (it never enters this loop) and
        // still swings 0.82-1.24x per rep. What is not a wash is `nnz(L)`: the
        // rule gives back 0.91% of the 4.90% fill win for nothing. Pushing the
        // handoff further out only makes that worse and monotonically so —
        // as-Skitter nnz(L) at a 500/1000/2000/3000/5325 handoff is
        // 18.48/18.51/18.63/18.83/19.43 M against `elim_remaining`
        // 26/65/172/285/423 ms and only 0/31/92/136 ms of `find_partition`
        // saved; com-LiveJournal at 500/2000/5000/10000/20000 is
        // 120.6/120.7/121.3/123.4/128.8 M against `elim_remaining`
        // 49/362/1326/4829/5302 ms; on coAuthorsDBLP and com-Amazon setup is
        // *lowest* at the 500 handoff and nnz(L) is +18% / +5% at 15000.
        //
        // Two things make the tail cheap enough not to be worth cutting.
        // (1) Whiffed rounds — the thing the rule was aimed at — are nearly
        // free: a whiff means the hash sample caught nothing, so there is no
        // edge-visit work behind it. as-Skitter's 864 whiffs cost 2.5 ms TOTAL
        // (2.9 us each) against a ~2.9 s setup; com-LiveJournal's 1583 cost
        // 8.3 ms. (2) A round is not a fork-join tax on top of the peel — it IS
        // the peel's per-vertex code plus one scan, and the pivot that scan buys
        // makes the elimination cheaper (~72 vs ~82 us per tail vertex; see the
        // block above the loop).
        //
        // Do not re-add the rule without a workload where `eliminate_remaining`
        // is cheaper per column than a BK round is per vertex.
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
                if (++bk_consecutive_empty >= kMaxEmptyRounds)
                    break;                     // hand the residual to the peel
                ++ws.round_index;              // keep retry rounds distinguishable
                continue;                      // whiffed sample round: retry
            }
            bk_consecutive_empty = 0;
            result.rounds.push_back({active.size(), bk_part.num_regions(),
                                     last_avg_degree, 0, 0});
            const size_t cols_before_bk = cp ? factor_cols.size() : 0;
            detail::eliminate_partition(elim, work, bk_part, factor_cols, ws, opts, cp);
            if (cp) {
                size_t bk_round_nnz = 0;
                for (size_t ci = cols_before_bk; ci < factor_cols.size(); ++ci)
                    bk_round_nnz += factor_cols[ci].entry_count + 1;
                result.rounds.back().nnz_added = bk_round_nnz;
                result.rounds.back().nnz_total =
                    (result.rounds.size() >= 2
                        ? result.rounds[result.rounds.size() - 2].nnz_total : 0)
                    + bk_round_nnz;
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
        }
    }
    if (!active.empty()) {
        detail::eliminate_remaining(elim, work, active, factor_cols, ws, opts);
    }
    if (cp) (*cp)("elim_remaining");

    // Quantify incidence-pool over-allocation: peak working-graph bytes vs the
    // factor's "useful" size, and the live fraction (1 - abandoned-slab share).
    // live_frac well below 1 => abandoned slabs dominate -> compaction would help.
    // Read off the working graph HERE, before it is freed below (nnz(L) is the
    // off-diagonal count + one diagonal per column -- what assemble_csc builds).
    if (const char* e = std::getenv("APXCHOL_MEM_DUMP"); e && *e) {
        const double MB = 1.0 / (1024.0 * 1024.0);
        size_t nnz = factor_cols.size();
        for (const auto& c : factor_cols) nnz += c.entry_count;
        double live_frac = -1.0;
        if constexpr (is_vec_pool_incidence_v<Incidence> ||
                      std::is_same_v<Incidence, forward_star_incidence>)
            live_frac = work.adj_live_fraction();
        std::fprintf(stderr,
            "[mem] peak_graph=%.0f MB  factor_nnz=%zu (inner=%.0f MB)  "
            "pool_live_frac=%.3f  (1/live_frac=%.2fx)\n",
            result.peak_graph_bytes * MB, nnz, nnz * sizeof(node_index) * MB,
            live_frac, live_frac > 0 ? 1.0 / live_frac : 0.0);
        if constexpr (is_vec_pool_incidence_v<Incidence>) {
            const size_t tot_edges = static_cast<size_t>(work.m());
            if constexpr (graph<Incidence>::stores_directed_incidence) {
                std::fprintf(stderr,
                    "[mem] vec_pool_aos: grows=%zu  compactions=%zu  "
                    "min_live_frac=%.3f  undirected_edges_added=%zu  "
                    "logical_directed_payload=%.0f MB\n",
                    work.adj_grow_count(), work.adj_compact_count(),
                    work.adj_min_live_fraction(), tot_edges,
                    2.0 * tot_edges * sizeof(directed_pool_edge) * MB);
            } else {
                std::fprintf(stderr,
                    "[mem] vec_pool: grows=%zu  compactions=%zu  "
                    "min_live_frac=%.3f  edge_pool_entries=%zu (~%.0f MB)\n",
                    work.adj_grow_count(), work.adj_compact_count(),
                    work.adj_min_live_fraction(), tot_edges,
                    tot_edges * sizeof(
                        typename std::remove_cvref_t<decltype(work)>::edge) * MB);
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
#if defined(APXCHOL_USE_CUDA)
    gpu_frontend.reset();
#endif
    std::vector<std::unique_ptr<std::pmr::monotonic_buffer_resource>>
        factor_entry_resources;
    factor_entry_resources.reserve(ws.threads.size());
    for (auto& t : ws.threads)
        factor_entry_resources.push_back(std::move(t.factor_entries));
    work = graph<Incidence>();
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

    detail::build_csc(result, factor_cols, n, cp);
    // The per-column ranges and their monotonic resources are consumed: free
    // both here, not at return.
    std::vector<detail::factor_col>().swap(factor_cols);
    std::vector<std::unique_ptr<std::pmr::monotonic_buffer_resource>>().swap(
        factor_entry_resources);

    // Optional debugging hook: print final nnz(L) when env var is set.
    // Useful for comparing factor fill across option settings without touching
    // the caller.
    if (const char* e = std::getenv("APXCHOL_DUMP_NNZ"); e && *e) {
        std::fprintf(stderr, "[apxchol] nnz(L) = %zu\n",
                     static_cast<size_t>(result.L.nonZeros()));
    }

    if (cp) cp->ascend();

    return result;
}

// Build the tree eliminator from options. Single source of truth so the
// fixed-partitioner and runtime-dispatch paths stay in sync.
inline detail::tree_elimination make_tree_elim(const factor_options& opts) {
    return detail::tree_elimination{
        .exact_clique_max_degree = opts.exact_clique_max_degree};
}

// Default-construct-the-partitioner convenience layer.
template<typename Partitioner, typename Eliminator, incidence_storage Incidence>
factorization factorize_impl(const Eliminator& elim,
                             graph<Incidence> G,
                             const factor_options& opts,
                             checkpoint* cp) {
    Partitioner partitioner;
    return factorize_impl(elim, partitioner, std::move(G), opts, cp);
}

// All factorize entry points take the graph by value (sink) and move it down
// the chain into factorize_impl's working copy — a throwaway caller pays no copy.
template<typename Partitioner, incidence_storage Incidence>
factorization factorize(graph<Incidence> G,
                        const factor_options& opts,
                        checkpoint* cp) {
    return factorize_impl<Partitioner>(make_tree_elim(opts), std::move(G), opts, cp);
}

template<typename Partitioner, eliminator E, incidence_storage Incidence>
factorization factorize(graph<Incidence> G, const E& elim,
                        const factor_options& opts,
                        checkpoint* cp) {
    return factorize_impl<Partitioner>(elim, std::move(G), opts, cp);
}

template<partitioner P, incidence_storage Incidence>
factorization factorize(graph<Incidence> G, P part,
                        const factor_options& opts,
                        checkpoint* cp) {
    return factorize_impl(make_tree_elim(opts), part, std::move(G), opts, cp);
}

template<partitioner P, eliminator E, incidence_storage Incidence>
factorization factorize(graph<Incidence> G, P part, const E& elim,
                        const factor_options& opts,
                        checkpoint* cp) {
    return factorize_impl(elim, part, std::move(G), opts, cp);
}

template<incidence_storage Incidence>
factorization factorize_with_strategy(graph<Incidence> G,
                                      const factor_options& opts,
                                      checkpoint* cp) {
    return dispatch_partitioner<factorization>(opts.is_select,
        [&]<typename P>() -> factorization {
            return factorize_impl<P>(make_tree_elim(opts), std::move(G), opts, cp);
        });
}

template<eliminator E, incidence_storage Incidence>
factorization factorize_with_strategy(graph<Incidence> G, const E& elim,
                                      const factor_options& opts,
                                      checkpoint* cp) {
    return dispatch_partitioner<factorization>(opts.is_select,
        [&]<typename P>() -> factorization {
            return factorize_impl<P>(elim, std::move(G), opts, cp);
        });
}

} // namespace apxchol
