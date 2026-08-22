#pragma once
/// Luby-style independent set selection (partitioner form).
///
/// Algorithm: each round assigns every eligible active vertex a random
/// priority (optionally degree-major, so low-degree candidates win), picks all
/// local priority minima simultaneously, excludes their neighbors, and
/// iterates on the survivors until the IS is maximal (or an iteration cap is
/// hit).  Fully parallel — no blocks and no cross-thread conflict resolution,
/// so the IS size cannot collapse at high thread counts.

#include "apxchol/solver/partitioner.h"
#include "apxchol/solver/partitioner_helpers.h"
#include "apxchol/solver/partition/luby_config.h"

namespace apxchol {

struct luby_partitioner {
    static constexpr std::string_view name = "luby";
    static constexpr size_t residual_handoff_threshold = 500;
    static constexpr bool degree_prepass = true;

    uint64_t round = 0;

    template<incidence_storage Incidence>
    void find_partition(graph<Incidence>& G, std::span<const node_index> candidates,
                        const partition_context& ctx, selection& out) {
        select_into_chosen_k0(G, candidates, out, ctx);
        ++round;
    }

private:
    // Per-round scratch reused across calls.
    // iterated-Luby (maximal IS) scratch:
    std::vector<node_index> cand_;     // undecided-candidate working list
    std::vector<char>       status_;   // 0=undecided, 1=chosen, 2=excluded
    std::vector<char>       pick_;     // pass-1 output (separate buffer; avoids the
                                       // read/write race on status_ within a pass)

    template<incidence_storage Incidence>
    void select_into_chosen_k0(const graph<Incidence>& G,
                               std::span<const node_index> candidates,
                               selection& out,
                               const partition_context& ctx) {
        // k=0 path. Default: ITERATE the local-min IS to a MAXIMAL one (pick all
        // local minima, exclude their neighbors, repeat on the survivors). This
        // gives a maximal IS like greedy bg, but FULLY parallel — no blocks, no
        // cross-block conflict resolution, so it can't collapse at high threads.
        // APXCHOL_LUBY_ITERS=1 reproduces the old single-pass (non-maximal) Luby.
        // (Re-randomizing priorities each inner iter — faithful textbook Luby —
        // measured neutral-to-slightly-worse vs these fixed per-round priorities,
        // so we keep the fixed-priority iterated random-greedy MIS.)
        uint64_t round_seed = ctx.seed
            ^ (round * 6364136223846793005ULL + 1442695040888963407ULL);
        // Degree-aware priority (degree_tiebreak): degree in the high bits,
        // hash in the low bits — the local-min IS then prefers low-degree
        // candidates (parallel min-degree), with the hash breaking ties. Plain
        // hash priority otherwise.
        const bool tiebreak = ctx.options.degree_tiebreak;
        const auto degrees  = ctx.degrees;
        auto prio = [&](node_index v) -> uint64_t {
            const uint64_t h = (uint64_t(v) ^ round_seed) * 11400714819323198485ULL;
            if (!tiebreak) return h;
            return (uint64_t(degrees[v]) << 40) | (h >> 24);
        };
        const int kMaxIters = detail::luby_iteration_limit();

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

        for (int iter = 0; iter < kMaxIters && !cand_.empty(); ++iter) {
            // Pass 1: read the STABLE status_ (unchanged within this pass), write
            // pick_ only. Local-min uses immutable priorities, so no race. cand_
            // holds only in-play (status_==0) candidates after compaction.
            #pragma omp parallel for schedule(static) if(cand_.size() > ctx.omp_threshold)
            for (size_t i = 0; i < cand_.size(); ++i) {
                auto v = cand_[i];
                auto pv = prio(v);
                bool lmin = true;
                for (auto idx : G.adj(v)) {
                    auto u = G.edge_target(idx, v);
                    if (G.is_active(u) && status_[u] == 0 && prio(u) < pv) { lmin = false; break; }
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

        // Commit, then restore the status invariant (2 everywhere).  Any
        // still-undecided survivors after kMaxIters are simply not taken this
        // round — they get another shot next elimination round.
        for (auto v : candidates) {
            if (status_[v] == 1) out.add(v);
            status_[v] = 2;
        }
    }
};

} // namespace apxchol
