#pragma once
/// Block-greedy parallel independent set selection (partitioner form).
///
/// Algorithm: the active vertex list is split into one contiguous block per
/// thread.  Each thread runs a serial greedy IS scan over its own block —
/// a vertex is picked if no already-chosen neighbor lies in the SAME block —
/// so intra-block picks are exact greedy with no synchronization.  Picks with
/// a neighbor in another block are marked "boundary" and re-checked in a
/// cross-block conflict-resolution pass (a boundary vertex drops itself when a
/// still-chosen neighbor in another block beats it by (degree, index)).  Every
/// drop can un-block neighbors, so a third pass repairs maximality over the
/// REPAIR FRONTIER — the dropped picks and their candidate neighbors, the only
/// vertices a drop can have freed.  A degree threshold (quantile or multiplier
/// cap, see factor_options) gates which candidates are eligible at all, biasing
/// elimination toward low-degree candidates.  Produces a flat list of selected
/// singleton-region candidates.

#include "apxchol/solver/partitioner.h"
#include "apxchol/solver/partitioner_helpers.h"
#include <algorithm>
#include <cstdint>
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

    // ── Repair-pass scratch ──────────────────────────────────────────────
    // Round stamp answering "is u one of THIS round's candidates?".  block_of
    // cannot: it keeps a stale block id for vertices that were candidates in an
    // earlier round, and the repair must never admit a vertex the degree cap
    // left out of `candidates`.
    std::vector<uint32_t> cand_stamp_;
    uint32_t round_stamp_ = 0;
    // Frontier state.  mark_[v]: 0 = not on any list, 1 = on its owner's
    // frontier list, 2 = pending (free, contesting this pass).  It is all-zero
    // between rounds: a vertex leaves a frontier list only by being cleared.
    std::vector<uint8_t> mark_;
    std::vector<std::vector<node_index>> per_thread_front_, per_thread_next_;
    // mailbox_[src * team_stride_ + owner]: frontier vertices thread `src`
    // discovered but does not own.  Drained by the owner in thread order, so
    // which thread admits a vertex — hence where it lands in the round's
    // selection list — does not depend on scheduling.
    std::vector<std::vector<node_index>> mailbox_;
    size_t team_stride_ = 0;
    // Per-thread pending counts, one 64-byte line apart, for the fixpoint test.
    static constexpr size_t kCountStride = 8;
    std::vector<size_t> pend_count_;
    // Drops this round, summed across threads: zero means nothing can have been
    // freed, so the repair is skipped without an extra barrier.
    size_t total_drop_ = 0;

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
        // Either way it is a STRICT TOTAL ORDER on the candidates, which is what
        // makes the repair pass below terminate.
        auto u_beats_v = [&](node_index u, node_index v) {
            if (!tiebreak) return u < v;
            const node_index du = degrees[u], dv = degrees[v];
            return du < dv || (du == dv && u < v);
        };

        #ifdef _OPENMP
        if (omp_get_max_threads() > 1 && candidates.size() > ctx.omp_threshold) {
            const size_t nt = static_cast<size_t>(omp_get_max_threads());
            if (per_thread_boundary_.size() < nt)
                per_thread_boundary_.resize(nt);
            if (per_thread_front_.size() < nt) {
                per_thread_front_.resize(nt);
                per_thread_next_.resize(nt);
            }
            if (team_stride_ != nt) {
                team_stride_ = nt;
                mailbox_.clear();
                mailbox_.resize(nt * nt);
            }
            if (pend_count_.size() < nt * kCountStride)
                pend_count_.assign(nt * kCountStride, 0);
            if (mark_.size() < nn) mark_.assign(nn, 0);
            if (cand_stamp_.size() < nn) cand_stamp_.assign(nn, 0);
            if (++round_stamp_ == 0) {          // wrap: retire every stale stamp
                std::fill(cand_stamp_.begin(), cand_stamp_.end(), uint32_t{0});
                round_stamp_ = 1;
            }
            total_drop_ = 0;
            // ONE parallel region for: block-assignment, main greedy scan,
            // cross-block conflict resolution, maximality repair.  One
            // contiguous block per thread.
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int nthreads = omp_get_num_threads();
                auto bs = candidates.size() * size_t(tid) / nthreads;
                auto be = candidates.size() * size_t(tid + 1) / nthreads;

                for (size_t i = bs; i < be; ++i) {
                    block_of[candidates[i]] = tid;
                    cand_stamp_[candidates[i]] = round_stamp_;
                }

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
                // with no quality gain. The over-drop is not left standing
                // either: whatever it frees, the repair pass below takes back.
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

                #pragma omp atomic
                total_drop_ += ndrop;

                #pragma omp barrier  // decisions are all taken against the snapshot

                for (size_t i = 0; i < ndrop; ++i) out.remove(my_boundary[i]);

                // ── Maximality repair over the frontier ──────────────────
                // What the greedy pass leaves is MAXIMAL over the candidate
                // set: a candidate goes unpicked only when a same-block
                // neighbor is already chosen, and chosen[] only grows while
                // that pass runs.  The conflict pass then REMOVES picks, and
                // every removal can un-block neighbors — so what survives is
                // independent but no longer maximal, and it is the missing
                // picks, not the drops themselves, that cost IS size.
                //
                // The vertices a drop can have freed are a small, exactly
                // known set — the REPAIR FRONTIER: the dropped picks, plus
                // their candidate neighbors.  (If an unpicked candidate v is
                // free now, some neighbor u blocked it at the end of the
                // greedy pass and is unchosen now, so u was dropped and v is a
                // neighbor of a dropped vertex.  A dropped vertex may be free
                // itself.)  Nothing outside the frontier changed, so the
                // repair costs O(sum of degrees over the frontier) instead of
                // another pass over every candidate.  That is the whole point.
                // On a well-numbered mesh the drop set is a thin block
                // boundary, so the repair stays a rounding error on the scan
                // it is attached to — measured in adjacency-slot visits at
                // T=16, 1.6% of the greedy+conflict scan on grid_2000 and 2.2%
                // on ecology1, against 33% and 28% for the same repair done by
                // rescanning all candidates.  On a graph whose numbering
                // carries no locality (com-Youtube, as-Skitter) the frontier
                // IS most of the boundary and the two cost the same — but that
                // is exactly where the repair pays: bail |active| 27211 ->
                // 9257 and 69598 -> 22584, forward levels 3430 -> 3047 and
                // 5772 -> 5162.
                //
                // Admission, repeated to fixpoint: a frontier vertex is
                // PENDING when no chosen active neighbor blocks it, and is
                // admitted when no pending neighbor beats it under
                // (degree,index).  The two roles are exactly what separates a
                // committed pick from a competitor, and mark_ carries the
                // distinction: a chosen neighbor blocks absolutely, a pending
                // one is merely contested.  Admitted vertices are pairwise
                // non-adjacent and have no chosen neighbor, so `out` stays an
                // independent set.  Since (degree,index) is a strict total
                // order, the minimum of a non-empty pending set is always
                // admitted — so every pass admits at least one vertex, and the
                // loop terminates in at most |pending| passes (in practice 1-3).
                //
                // Determinism: a frontier vertex is owned by the thread whose
                // BLOCK contains it, never by whichever thread reached it
                // first; discovered neighbors are handed to their owner through
                // per-(source, owner) mailboxes drained in thread order; and
                // every decision reads a barrier-separated snapshot.  So the
                // selection AND the order it is added in stay a pure function
                // of (graph, candidates, ctx, team size), as the partitioner
                // determinism contract requires.
                if (total_drop_ != 0) {
                    auto& front = per_thread_front_[tid];
                    auto& next  = per_thread_next_[tid];
                    front.clear();
                    for (size_t s = 0; s < team_stride_; ++s)
                        mailbox_[size_t(tid) * team_stride_ + s].clear();

                    for (size_t i = 0; i < ndrop; ++i) {
                        const node_index d = my_boundary[i];
                        if (!mark_[d]) { mark_[d] = 1; front.push_back(d); }
                        for (auto idx : G.adj(d)) {
                            const auto u = G.edge_target(idx, d);
                            if (!G.is_active(u) || cand_stamp_[u] != round_stamp_) continue;
                            const int ow = block_of[u];
                            if (ow == tid) {
                                if (!mark_[u]) { mark_[u] = 1; front.push_back(u); }
                            } else {
                                mailbox_[size_t(tid) * team_stride_ + size_t(ow)].push_back(u);
                            }
                        }
                    }

                    #pragma omp barrier  // mailboxes filled; every removal applied

                    for (int s = 0; s < nthreads; ++s)
                        for (const node_index u : mailbox_[size_t(s) * team_stride_ + size_t(tid)])
                            if (!mark_[u]) { mark_[u] = 1; front.push_back(u); }

                    for (;;) {
                        // Pending = free against the CURRENT chosen set. A
                        // vertex that is blocked here can never come back
                        // (chosen[] only grows from now on), so it leaves the
                        // list for good.
                        size_t k = 0;
                        for (const node_index v : front) {
                            if (out.contains(v)) { mark_[v] = 0; continue; }
                            bool free_v = true;
                            for (auto idx : G.adj(v)) {
                                const auto u = G.edge_target(idx, v);
                                if (G.is_active(u) && out.contains(u)) { free_v = false; break; }
                            }
                            if (free_v) { mark_[v] = 2; front[k++] = v; }
                            else        { mark_[v] = 0; }
                        }
                        front.resize(k);

                        #pragma omp barrier  // every mark_ write visible

                        next.clear();
                        size_t nadd = 0;
                        for (const node_index v : front) {
                            bool win = true;
                            for (auto idx : G.adj(v)) {
                                const auto u = G.edge_target(idx, v);
                                if (G.is_active(u) && mark_[u] == 2 && u_beats_v(u, v)) {
                                    win = false; break;
                                }
                            }
                            if (win) front[nadd++] = v;   // nadd <= read cursor
                            else     next.push_back(v);
                        }

                        #pragma omp barrier  // nobody is reading mark_ any more

                        // Emit in candidate order (the frontier list is built
                        // drop-first, then mailbox): the round's selection is
                        // the elimination order, so keeping it in the same
                        // ascending order the greedy pass emits keeps the
                        // elimination sweep monotone in vertex id.
                        std::sort(front.begin(), front.begin() + nadd);
                        for (size_t j = 0; j < nadd; ++j) {
                            out.add(front[j]);
                            mark_[front[j]] = 0;
                        }
                        front.swap(next);
                        pend_count_[size_t(tid) * kCountStride] = front.size();

                        #pragma omp barrier  // counts and out.add()s visible

                        size_t remaining = 0;
                        for (int s = 0; s < nthreads; ++s)
                            remaining += pend_count_[size_t(s) * kCountStride];
                        if (remaining == 0) break;   // same sum in every thread
                    }
                }
            }
        } else
        #endif
        {
            // Serial greedy. No blocks, so no cross-block drops and nothing to
            // repair: a candidate goes unpicked only when a neighbor is chosen,
            // and no pick is ever taken back, so this IS is already maximal.
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
