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
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
        col = {v, d > 0.0 ? std::sqrt(d) : 1.0, {}};
        return;
    }

    double total_deg = edge_deg + G.excess(v);
    if (total_deg <= 0.0) total_deg = 1.0;
    double sqrt_deg = std::sqrt(total_deg);
    col.vertex = v;
    col.diag = sqrt_deg;
    col.entries.reserve(nbrs.size());
    for (const auto& [u, w] : nbrs)
        col.entries.emplace_back(u, w / sqrt_deg);

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
                                   checkpoint* cp = nullptr) {
    const size_t n_verts = part.num_vertices();
    // n_verts > 0 guaranteed by the dispatcher; early-exit not needed here.
    std::vector<factor_col> local_cols(n_verts);

    #ifdef _OPENMP
    // APXCHOL_TAIL_THREADS (experiment knob, see env_knobs.h): rounds whose IS
    // is at or below omp_threshold normally take the fully serial path at the
    // bottom of this function. When the knob is set (> 0), such "tail" rounds
    // run the fused PARALLEL path below instead, on a small pinned team of
    // min(APXCHOL_TAIL_THREADS, ws.threads.size()) threads. Unset (default) =
    // serial tail, unchanged. The parallel path applies clique edges in
    // thread-arrival order, so a tail-parallel factor may differ from the
    // serial-tail factor by fp merge-order ulps (same class of difference as
    // the main path at T > 1).
    const int tail_threads = detail::env_knobs::get().tail_threads;
    const bool tail_parallel = tail_threads > 0 && n_verts <= opts.omp_threshold;
    if (n_verts > opts.omp_threshold || tail_parallel) {
        // Team size for the fused paths: the full workspace team, or the pinned
        // tail team on a tail round.
        const int team_threads = tail_parallel
            ? std::min<int>(tail_threads, static_cast<int>(ws.threads.size()))
            : static_cast<int>(ws.threads.size());
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
                    process_vertex(elim, G, part.data[k], opts.seed, local_cols[k],
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

            for (auto& col : local_cols)
                factor_cols.push_back(std::move(col));

            // NOTE: the mega-fused parallel region runs compute + apply + apply_excess
            // back-to-back inside one fork-join, so we cannot attribute these
            // sub-phases separately without breaking the fusion. Emit one honest
            // label that reflects what was actually measured.
            if (cp) (*cp)("compute+apply_fused");
        } else if constexpr (std::is_same_v<Incidence, vec_pool_incidence>) {
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
            std::vector<node_index> incoming(static_cast<size_t>(G.n()), 0);
            edge_index e_start = 0;

            #pragma omp parallel num_threads(num_threads)
            {
                int tid = omp_get_thread_num();

                ws.threads[tid].edge_buffer.clear();
                ws.threads[tid].excess_buffer.clear();

                #pragma omp for schedule(dynamic, 64)
                for (size_t k = 0; k < n_verts; ++k)
                    process_vertex(elim, G, part.data[k], opts.seed, local_cols[k],
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

                // Single: prefix-sum thread edge offsets + edge_pool reserve
                // + concatenate touched_buffers for the bulk reserve below.
                #pragma omp single
                {
                    for (int t = 0; t < num_threads; ++t)
                        e_offsets[t + 1] = e_offsets[t] + ws.threads[t].edge_buffer.size();
                    const size_t N_edges = e_offsets[num_threads];
                    if (N_edges > 0)
                        e_start = G.reserve_edge_pool(static_cast<edge_index>(N_edges));
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
                        G.write_edge_at(es, u, v, w);
                        G.adj_atomic_push_reserved(u, es);
                        G.adj_atomic_push_reserved(v, es);
                    }
                    ws.threads[tid].edge_buffer.clear();
                    // Apply excess atomically (each thread's own buffer).
                    for (auto [u, delta] : ws.threads[tid].excess_buffer)
                        G.atomic_add_excess(u, delta);
                    ws.threads[tid].excess_buffer.clear();
                }

                // Deactivate partition vertices in parallel (disjoint).
                #pragma omp for schedule(static) nowait
                for (size_t k = 0; k < n_verts; ++k)
                    G.set_inactive_unchecked(part.data[k]);
            }
            G.bulk_decrement_active(static_cast<node_index>(n_verts));

            for (auto& col : local_cols)
                factor_cols.push_back(std::move(col));

            if (cp) (*cp)("compute+apply_fused");
        } else {
            // Legacy backends (vec/bstr): separate parallel compute +
            // serial apply (apply_deferred_edges has no fast path for these).
            // Skip the explicit merge_parallel_edges pass: process_vertex
            // does inline dedup + dead-edge filter, saving a full
            // adjacency traversal per IS vertex per round.
            // Team: the OpenMP default (as before) unless this is a
            // tail-parallel round, which pins the small tail team.
            const int legacy_threads = tail_parallel ? team_threads
                                                     : omp_get_max_threads();
            #pragma omp parallel num_threads(legacy_threads)
            {
                int tid = omp_get_thread_num();

                // Clear per-thread buffers for this round.
                ws.threads[tid].edge_buffer.clear();
                ws.threads[tid].excess_buffer.clear();

                #pragma omp for schedule(dynamic, 64) nowait
                for (size_t k = 0; k < n_verts; ++k)
                    process_vertex(elim, G, part.data[k], opts.seed, local_cols[k],
                                   ws.threads[tid],
                                   /*dedup_inline=*/true);
            }
            if (cp) { (*cp)("merge_is"); (*cp)("compute"); }

            for (auto& col : local_cols)
                factor_cols.push_back(std::move(col));

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
            process_vertex(elim, G, part.data[k], opts.seed, local_cols[k], wt,
                           /*dedup_inline=*/true);
            factor_cols.push_back(std::move(local_cols[k]));
            for (auto [u, v, w] : wt.edge_buffer)
                G.add_edge(u, v, w);
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
                         checkpoint* cp = nullptr) {
    if (cp) { cp->descend("eliminate"); cp->tick(); }
    const size_t n_verts = part.num_vertices();
    if (std::getenv("APXCHOL_ROUND_TRACE"))  // one line per round: IS size this round
        std::fprintf(stderr, "[round] n_verts=%zu\n", n_verts);
    if (n_verts == 0) {
        if (cp) cp->ascend();
        return;
    }
    eliminate_partition_singleton(elim, G, part, factor_cols, ws, opts, cp);
    if (cp) cp->ascend();
}

// Eliminate remaining vertices after the main IS-elimination loop.
// Used when IS fraction drops below threshold — avoids the O(|active|)
// IS-finding scan when only a few vertices can be chosen anyway.
//
// (An earlier version tried parallel BK rounds on the residual, but the
// per-round fork-join cost outweighed the gain on the large, high-degree
// residuals BG hands us; serial peel was strictly faster.)
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
        col.entries.clear();
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
    }
    std::vector<detail::factor_col> factor_cols;
    factor_cols.reserve(n);

    constexpr bool sample_bounded = partitioner_sample_bounded_v<Partitioner>;
    constexpr size_t residual_handoff_default =
        partitioner_residual_handoff_v<Partitioner>;

    // Active vertex list (natural index order) — filtered in-place each round.
    std::vector<node_index> active(n);
    std::ranges::iota(active, node_index{0});

    // The partitioner's view of the run-constant services, the shared
    // selection structure, and the degree-prepass scratch (all owned here).
    selection sel;
    std::vector<node_index> pre_degrees, pre_scratch, pre_eligible;
    std::vector<node_index> live_degrees;   // vertex-indexed, for ctx.degrees

    // Shared round front-end: prepass (when the partitioner's trait asks for
    // it) + find_partition + finalize, with uniform profiling labels.
    // last_avg_degree feeds the per-round stats; it is the prepass's exact
    // average (0 = not measured, for partitioners without the prepass).
    double last_avg_degree = 0.0;
    auto run_partitioner = [&](auto& p, graph<Incidence>& g,
                               std::span<const node_index> act)
        -> const partition_result& {
        using P = std::remove_reference_t<decltype(p)>;
        sel.reset(g.n());
        last_avg_degree = 0.0;
        partition_context pctx{opts.partition, opts.seed, opts.omp_threshold, cp};
        if (cp) { cp->descend("find_partition"); cp->tick(); }
        if constexpr (partitioner_degree_prepass_v<P>) {
            const double avg_deg =
                prune_and_degrees(g, act, pre_degrees, opts.omp_threshold);
            const double thr = is_degree_threshold(
                pre_degrees, act.size(), avg_deg, opts.partition, pre_scratch);
            if (live_degrees.size() < static_cast<size_t>(g.n()))
                live_degrees.resize(g.n());
            pre_eligible.clear();
            for (size_t i = 0; i < act.size(); ++i) {
                live_degrees[act[i]] = pre_degrees[i];
                if (pre_degrees[i] <= thr) pre_eligible.push_back(act[i]);
            }
            last_avg_degree = avg_deg;
            pctx.degrees = live_degrees;
            if (cp) (*cp)("prune");
            p.find_partition(g, std::span<const node_index>(pre_eligible),
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
            if (part.num_regions() < active.size() * opts.min_is_fraction)
                break;
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
        detail::eliminate_partition(elim, work, part, factor_cols, ws, opts, cp);
        // Tally nnz added this round (only when caller wants the stats).
        if (cp) {
            size_t round_nnz = 0;
            for (size_t ci = cols_before; ci < factor_cols.size(); ++ci)
                round_nnz += factor_cols[ci].entries.size() + 1;
            result.rounds.back().nnz_added = round_nnz;
            result.rounds.back().nnz_total =
                (result.rounds.size() >= 2
                    ? result.rounds[result.rounds.size() - 2].nnz_total : 0)
                + round_nnz;
        }

        result.peak_graph_bytes = std::max(result.peak_graph_bytes,
                                           work.memory_bytes());

        std::erase_if(active, [&](node_index v) { return !work.is_active(v); });

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
        if constexpr (std::is_same_v<Incidence, vec_pool_incidence>)
            work.adj_note_live_fraction();
    }

    // Eliminate any remaining vertices (0 or 1 after the while loop,
    // or all remaining when the IS-fraction fallback triggered above).
    //
    // Parallel residual peel: when the main loop bailed with a large
    // residual, switch to BK rounds.  BK is sample-bounded so it does
    // not hit the same fallback, and each round eliminates a large
    // batch in parallel via the same eliminate_partition machinery — far
    // cheaper than the serial process_vertex peel for the high-degree
    // residuals of LP-IPM Schur complements.
    //
    // Smart default: if user left parallel_residual_threshold at
    // SIZE_MAX (disabled), use the partitioner's `residual_handoff_threshold`
    // trait (defaults to SIZE_MAX = disabled).  Empirically measured
    // on iter0010 (524k LP factor, 16t): reduces setup 8% (bg+fs,
    // luby+fs), 2-6% (vec variants), also drops nnz(L) by ~5% via
    // the cheaper IS-ordered peel.
    size_t residual_thresh = opts.parallel_residual_threshold;
    if (residual_thresh == std::numeric_limits<size_t>::max())
        residual_thresh = residual_handoff_default;
    if (active.size() > residual_thresh) {
        baumann_kyng_partitioner bk;
        while (active.size() > residual_thresh) {
            ws.reset_for_round();
            const auto& bk_part = run_partitioner(bk, work, active);
            if (bk_part.num_regions() == 0) break;
            result.rounds.push_back({active.size(), bk_part.num_regions(),
                                     last_avg_degree, 0, 0});
            const size_t cols_before_bk = cp ? factor_cols.size() : 0;
            detail::eliminate_partition(elim, work, bk_part, factor_cols, ws, opts, cp);
            if (cp) {
                size_t bk_round_nnz = 0;
                for (size_t ci = cols_before_bk; ci < factor_cols.size(); ++ci)
                    bk_round_nnz += factor_cols[ci].entries.size() + 1;
                result.rounds.back().nnz_added = bk_round_nnz;
                result.rounds.back().nnz_total =
                    (result.rounds.size() >= 2
                        ? result.rounds[result.rounds.size() - 2].nnz_total : 0)
                    + bk_round_nnz;
            }
            result.peak_graph_bytes = std::max(result.peak_graph_bytes,
                                               work.memory_bytes());
            std::erase_if(active, [&](node_index v) { return !work.is_active(v); });
            ++ws.round_index;
        }
    }
    if (!active.empty()) {
        detail::eliminate_remaining(elim, work, active, factor_cols, ws, opts);
    }
    if (cp) (*cp)("elim_remaining");

    detail::build_csc(result, factor_cols, n, cp);

    // Optional debugging hook: print final nnz(L) when env var is set.
    // Useful for comparing factor fill across option settings without touching
    // the caller.
    if (const char* e = std::getenv("APXCHOL_DUMP_NNZ"); e && *e) {
        std::fprintf(stderr, "[apxchol] nnz(L) = %zu\n",
                     static_cast<size_t>(result.L.nonZeros()));
    }

    // Quantify incidence-pool over-allocation: peak working-graph bytes vs the
    // factor's "useful" size, and the live fraction (1 - abandoned-slab share).
    // live_frac well below 1 => abandoned slabs dominate -> compaction would help.
    if (const char* e = std::getenv("APXCHOL_MEM_DUMP"); e && *e) {
        const double MB = 1.0 / (1024.0 * 1024.0);
        const size_t nnz = static_cast<size_t>(result.L.nonZeros());
        double live_frac = -1.0;
        if constexpr (std::is_same_v<Incidence, vec_pool_incidence> ||
                      std::is_same_v<Incidence, forward_star_incidence>)
            live_frac = work.adj_live_fraction();
        std::fprintf(stderr,
            "[mem] peak_graph=%.0f MB  factor_nnz=%zu (inner=%.0f MB)  "
            "pool_live_frac=%.3f  (1/live_frac=%.2fx)\n",
            result.peak_graph_bytes * MB, nnz, nnz * sizeof(node_index) * MB,
            live_frac, live_frac > 0 ? 1.0 / live_frac : 0.0);
        if constexpr (std::is_same_v<Incidence, vec_pool_incidence>) {
            const size_t tot_edges = static_cast<size_t>(work.m());
            std::fprintf(stderr,
                "[mem] vec_pool: grows=%zu  compactions=%zu  min_live_frac=%.3f  "
                "edges_=%zu (~%.0f MB)\n",
                work.adj_grow_count(), work.adj_compact_count(),
                work.adj_min_live_fraction(), tot_edges,
                tot_edges * sizeof(typename std::remove_cvref_t<decltype(work)>::edge) * MB);
        }
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
