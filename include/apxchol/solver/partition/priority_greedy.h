#pragma once
/// Fixed-priority greedy independent-set selection (partitioner form).
///
/// Each elimination round assigns every eligible active vertex one immutable
/// priority (optionally degree-major, so low-degree candidates win). Repeated
/// parallel local-minimum passes produce exactly the maximal independent set
/// of a sequential greedy scan in that priority order. A serial tail completes
/// the same order if the parallel-pass budget is exhausted; the budget affects
/// performance only, never the selected set.

#include "apxchol/solver/partitioner.h"
#include "apxchol/solver/partitioner_helpers.h"
#include "apxchol/solver/partition/priority_greedy_config.h"
#include <algorithm>

namespace apxchol {

struct priority_greedy_partitioner {
    static constexpr std::string_view name = "priority_greedy";
    static constexpr size_t residual_handoff_threshold = 500;
    static constexpr bool degree_prepass = true;

    uint64_t round = 0;
    int parallel_passes = detail::priority_greedy_parallel_passes();

    template<incidence_storage Incidence>
    void find_partition(graph<Incidence>& G, std::span<const node_index> candidates,
                        const partition_context& ctx, selection& out) {
        select_into_chosen_k0(G, candidates, out, ctx);
        ++round;
    }

private:
    // Per-round scratch reused across calls.
    // Fixed-priority greedy MIS scratch:
    std::vector<node_index> cand_;     // undecided-candidate working list
    std::vector<char>       status_;   // 0=undecided, 1=chosen, 2=excluded
    std::vector<char>       pick_;     // pass-1 output (separate buffer; avoids the
                                       // read/write race on status_ within a pass)

    template<incidence_storage Incidence>
    void select_into_chosen_k0(const graph<Incidence>& G,
                               std::span<const node_index> candidates,
                               selection& out,
                               const partition_context& ctx) {
        // Pick all local minima, exclude their neighbors, and repeat. Immutable
        // priorities make this a parallel execution of one greedy order, not
        // textbook Luby (which redraws priorities between passes).
        uint64_t round_seed = ctx.seed
            ^ (round * 6364136223846793005ULL + 1442695040888963407ULL);
        // Degree-aware priority (degree_tiebreak): compare degree first, then
        // the full hash and vertex id. The local-min IS prefers low-degree
        // candidates (parallel min-degree) without the overflow/collision risk
        // of packing a truncated hash into one integer. Plain hash otherwise.
        const bool tiebreak = ctx.options.degree_tiebreak;
        const auto degrees  = ctx.degrees;
        auto hash = [&](node_index v) -> uint64_t {
            return (uint64_t(v) ^ round_seed) * 11400714819323198485ULL;
        };
        auto precedes = [&](node_index a, node_index b) {
            if (tiebreak && degrees[a] != degrees[b])
                return degrees[a] < degrees[b];
            const uint64_t ha = hash(a), hb = hash(b);
            return ha != hb ? ha < hb : a < b;
        };
        // status_[v]: 0 undecided candidate, 1 chosen, 2 excluded.  Invariant:
        // 2 everywhere outside this call (reset over candidates at the end),
        // so non-candidates never read as undecided.
        if (status_.size() < G.n()) status_.assign(G.n(), 2);
        if (pick_.size() < G.n()) pick_.assign(G.n(), 0);
        cand_.clear();
        for (auto v : candidates) {
            status_[v] = 0;
            cand_.push_back(v);
        }

        for (int iter = 0; iter < parallel_passes && !cand_.empty(); ++iter) {
            // Pass 1: read the STABLE status_ (unchanged within this pass), write
            // pick_ only. Local-min uses immutable priorities, so no race. cand_
            // holds only in-play (status_==0) candidates after compaction.
            #pragma omp parallel for schedule(static) if(cand_.size() > ctx.omp_threshold)
            for (size_t i = 0; i < cand_.size(); ++i) {
                auto v = cand_[i];
                bool lmin = true;
                for (auto idx : G.adj(v)) {
                    auto u = G.edge_target(idx, v);
                    if (G.is_active(u) && status_[u] == 0 && precedes(u, v)) {
                        lmin = false;
                        break;
                    }
                }
                pick_[v] = lmin ? 1 : 0;
            }
            // Pass 2: commit picks. chosen candidates are mutually non-adjacent (the
            // lower-priority endpoint of any edge wins), so writing status_ here is
            // safe; neighbor exclusions all write the same value (2) -> benign.
            #pragma omp parallel for schedule(static) if(cand_.size() > ctx.omp_threshold)
            for (size_t i = 0; i < cand_.size(); ++i) {
                auto v = cand_[i];
                if (!pick_[v]) continue;
                status_[v] = 1;
                for (auto idx : G.adj(v)) {
                    auto u = G.edge_target(idx, v);
                    if (G.is_active(u) && status_[u] == 0) status_[u] = 2;
                }
            }
            // Compact the working list to the still-undecided survivors.
            size_t w = 0;
            for (size_t i = 0; i < cand_.size(); ++i)
                if (status_[cand_[i]] == 0) cand_[w++] = cand_[i];
            cand_.resize(w);
        }

        // A fixed pass cap used to leave survivors unclassified, making the
        // result non-maximal on a sufficiently deep priority DAG. Complete the
        // exact same greedy order serially. Normally cand_ is empty after a few
        // passes, so this is only a bounded-path correctness fallback.
        if (!cand_.empty()) {
            std::sort(cand_.begin(), cand_.end(), precedes);
            for (auto v : cand_) {
                if (status_[v] != 0) continue;
                status_[v] = 1;
                for (auto idx : G.adj(v)) {
                    const auto u = G.edge_target(idx, v);
                    if (G.is_active(u) && status_[u] == 0) status_[u] = 2;
                }
            }
        }

        // Commit in candidate order (the historical within-round elimination
        // order), then restore the status invariant (2 everywhere).
        for (auto v : candidates) {
            if (status_[v] == 1) out.add(v);
            status_[v] = 2;
        }
    }
};

} // namespace apxchol
