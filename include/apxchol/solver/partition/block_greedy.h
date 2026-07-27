#pragma once
/// Block-greedy parallel independent set selection (partitioner form).
///
/// Algorithm: the active vertex list is split into one contiguous block per
/// thread.  Each thread runs a serial greedy IS scan over its own block —
/// a vertex is picked if no already-chosen neighbor lies in the SAME block —
/// so intra-block picks are exact greedy with no synchronization.  Picks with
/// a neighbor in another block are marked "boundary" and re-checked in a
/// cross-block conflict-resolution pass (a boundary vertex drops itself when a
/// still-chosen neighbor in another block beats it by (degree, index)).  A
/// degree threshold (quantile or multiplier cap, see factor_options) gates
/// which candidates are eligible at all, biasing elimination toward low-degree
/// candidates.  Produces a flat list of selected singleton-region candidates.

#include "apxchol/solver/partitioner.h"
#include "apxchol/solver/partitioner_helpers.h"
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace apxchol {

struct block_greedy_partitioner {
    static constexpr std::string_view name = "block_greedy";
    static constexpr size_t residual_handoff_threshold = 500;
    static constexpr bool degree_prepass = true;

    template<incidence_storage Incidence>
    void find_partition(graph<Incidence>& G, std::span<const node_index> candidates,
                        const partition_context& ctx, selection& out) {
        select_into_chosen(G, candidates, out, ctx);
    }

private:
    // Per-call scratch.  Lives on the partitioner because it's BG-specific
    // (other partitioners don't use blocks).
    std::vector<int>  block_of;
    // Per-thread list of boundary candidates (a cross-block neighbor exists)
    // collected during the last select_into_chosen call. Drives the
    // conflict-resolution pass over O(boundary) instead of O(active).
    // Sized lazily at first parallel call.
    std::vector<std::vector<node_index>> per_thread_boundary_;

    template<incidence_storage Incidence>
    void select_into_chosen(const graph<Incidence>& G,
                            std::span<const node_index> candidates,
                            selection& out,
                            const partition_context& ctx) {
        size_t nn = G.n();
        block_of.resize(nn);
        const bool tiebreak  = ctx.options.degree_tiebreak;
        const auto degrees   = ctx.degrees;
        // u beats v (u survives the conflict, v drops) — low-degree wins, index
        // breaks ties. Falls back to plain index order when tiebreak is off.
        auto u_beats_v = [&](node_index u, node_index v) {
            if (!tiebreak) return u < v;
            const node_index du = degrees[u], dv = degrees[v];
            return du < dv || (du == dv && u < v);
        };

        #ifdef _OPENMP
        if (omp_get_max_threads() > 1 && candidates.size() > ctx.omp_threshold) {
            if (per_thread_boundary_.size() < static_cast<size_t>(omp_get_max_threads()))
                per_thread_boundary_.resize(omp_get_max_threads());
            // ONE parallel region for: block-assignment, main greedy scan,
            // cross-block conflict resolution.  One contiguous block per thread.
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int nthreads = omp_get_num_threads();
                auto bs = candidates.size() * size_t(tid) / nthreads;
                auto be = candidates.size() * size_t(tid + 1) / nthreads;

                for (size_t i = bs; i < be; ++i)
                    block_of[candidates[i]] = tid;

                #pragma omp barrier

                // Per-thread list of boundary picks. Used to drive the
                // conflict-resolution pass below over O(boundary) instead of
                // O(active). On typical workloads the boundary fraction is
                // small (5-20%), so this saves a full active-scan per round.
                auto& my_boundary = per_thread_boundary_[tid];
                my_boundary.clear();
                // Greedy-pick candidate i (pick if no chosen same-block
                // neighbor). Mark cross-block picks as boundary for the
                // conflict-resolution pass below.
                auto greedy_body = [&](size_t i) {
                    auto v = candidates[i];
                    bool ok = true;
                    bool boundary = false;
                    for (auto idx : G.adj(v)) {
                        auto u = G.edge_target(idx, v);
                        if (!G.is_active(u)) continue;
                        if (block_of[u] == tid) {
                            if (ok && out.contains(u)) ok = false;
                        } else {
                            boundary = true;
                        }
                        if (!ok && boundary) break;
                    }
                    if (ok) {
                        out.add(v);
                        if (boundary) my_boundary.push_back(v);
                    }
                };
                for (size_t i = bs; i < be; ++i) greedy_body(i);

                #pragma omp barrier  // all chosen[] writes visible

                // Parallel "shared-flag" conflict resolution: each thread drops
                // its own boundary candidates that have a still-chosen cross-block
                // neighbor that beats them, reading the shared chosen[] directly.
                // Racy (reads chosen[u] while u may be getting dropped) -> can
                // over-drop chains (a<b<c can drop both b AND c), giving a
                // slightly smaller IS. The over-drop is empirically negligible
                // (PCG iters and solve-per-iter unchanged across IPM/grid/
                // SuiteSparse workloads) while keeping the resolution fully
                // parallel — an exact serial index-ordered resolution was ~6x
                // slower on dense residuals with no quality gain.
                for (auto v : my_boundary) {
                    int mb = block_of[v];
                    for (auto idx : G.adj(v)) {
                        auto u = G.edge_target(idx, v);
                        if (G.is_active(u) && out.contains(u) && u_beats_v(u, v) && block_of[u] != mb) {
                            out.remove(v); break;
                        }
                    }
                }
            }
        } else
        #endif
        {
            // Serial greedy.
            for (auto v : candidates) {
                bool ok = true;
                for (auto idx : G.adj(v)) {
                    auto u = G.edge_target(idx, v);
                    if (G.is_active(u) && out.contains(u)) { ok = false; break; }
                }
                if (ok) out.add(v);
            }
        }
    }
};

} // namespace apxchol
