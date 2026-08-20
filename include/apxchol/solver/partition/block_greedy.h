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

                // Parallel "shared-flag" conflict resolution, DECIDE-THEN-APPLY:
                // each thread first works out which of its own boundary picks
                // lose to a cross-block neighbor (reads only), then a barrier,
                // then the removals land. Every read therefore sees the SAME
                // post-greedy snapshot of chosen[] in every thread, whatever the
                // scheduling — which is what makes the round's selection, hence
                // the elimination order, hence the factor's STRUCTURE,
                // reproducible run to run at a fixed thread count (see the
                // determinism contract in the class comment below).
                //
                // Resolving against the snapshot always takes the over-drop of a
                // chain (a beats b beats c across three blocks: b loses to a and
                // c loses to b, so both go even though b itself is leaving).
                // That is precisely the worst case the pre-2026-08-20 racy
                // single-phase pass already produced whenever a thread happened
                // to read chosen[b] before b's owner cleared it, so it cannot
                // cost more IS than that pass already did on a bad interleaving:
                // measured on a 120x120 grid at T=32, per-round IS sizes move by
                // <= 0.5% (round 2: 1400-1405 racy across runs -> 1398) and the
                // round count stays inside the racy spread (44, vs 44-47). PCG
                // iteration counts do not get worse -- on grid 300x300 and
                // 600x600 the deterministic factor lands at or below EVERY count
                // the racy build produced over 3 runs (300 at T=32: 43 vs
                // 43/45/46; 600 at T=32: 46 vs 48/48/50).
                // An exact index-ordered resolution — v survives iff no neighbor
                // that beats it SURVIVES — is a recursive definition along those
                // chains; the serial form of it was ~6x slower on dense residuals
                // with no quality gain.
                //
                // Correctness is unaffected by the over-drop: for any edge (u,v)
                // with both endpoints picked and in different blocks the loser
                // under (degree, index) sees the winner in the snapshot and
                // drops, so no adjacent pair survives — the output is still an
                // independent set (debug builds verify this after every round).
                size_t ndrop = 0;
                for (size_t bi = 0; bi < my_boundary.size(); ++bi) {
                    const node_index v = my_boundary[bi];
                    const int mb = block_of[v];
                    for (auto idx : G.adj(v)) {
                        auto u = G.edge_target(idx, v);
                        if (G.is_active(u) && out.contains(u) && u_beats_v(u, v) && block_of[u] != mb) {
                            my_boundary[ndrop++] = v;   // ndrop <= bi: never clobbers unread entries
                            break;
                        }
                    }
                }

                #pragma omp barrier  // decisions are all taken against the snapshot

                for (size_t i = 0; i < ndrop; ++i) out.remove(my_boundary[i]);
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
