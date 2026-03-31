#pragma once
/// Luby-style (priority-based) independent set selection.
///
/// ─── Algorithm overview ───
///
/// Each vertex is assigned a fixed hash-priority.  A low-degree vertex
/// joins the IS if it has the smallest priority among all its active
/// neighbors.  This is embarrassingly parallel — every vertex decides
/// independently with no cross-vertex data dependencies.
///
/// ─── The Luby framework ───
///
/// Luby (1986), "A Simple Parallel Algorithm for the Maximal
/// Independent Set Problem," introduced the idea of using random
/// priorities for parallel MIS computation.  The classic Luby MIS
/// algorithm:
///   1. Each vertex picks a random priority r(v).
///   2. v joins the IS if r(v) < r(u) for all neighbors u.
///   3. Remove IS vertices and their neighbors; repeat.
///
/// In expectation, each round removes a constant fraction of edges,
/// so O(log n) rounds suffice for MIS.
///
/// ─── Our variant ───
///
/// We use a deterministic hash (Fibonacci hashing on vertex index) as
/// the priority function, applied to the subset of low-degree active
/// vertices.  This gives a fixed permutation — no per-round randomness
/// needed.  The Fibonacci constant 2^64 / φ ensures good dispersion
/// even for sequential vertex indices (grid graphs, meshes).
///
/// A vertex v is added to the IS if:
///   (a) degree(v) ≤ degree_threshold, and
///   (b) hash(v) < hash(u) for every active neighbor u.
///
/// ─── IS size and trade-offs ───
///
/// For a graph with maximum degree Δ, the expected IS size from a
/// random-priority local minimum is n/(Δ+1).  On low-degree graphs
/// (grids with Δ ≈ 4), this gives roughly n/5 per round, which is
/// smaller than what greedy achieves (roughly n/3 to n/4).  The
/// difference means more elimination rounds and more total prune work.
///
/// On irregular graphs or when simplicity and perfect parallelism
/// matter more than round count, this strategy may be preferable to
/// block-greedy.
///
/// ─── References ───
///
///   • Luby M. (1986), "A Simple Parallel Algorithm for the Maximal
///     Independent Set Problem," SIAM J. Comput. 15(4):1036–1053.
///   • Alon, Babai, Itai (1986), "A fast and simple randomized
///     parallel algorithm for the maximal independent set problem,"
///     J. Algorithms 7(4):567–583.

#include "apxchol/solver/independent_set.h"

namespace apxchol {

struct luby_is {
    template<incidence_storage Incidence>
    void select(const graph<Incidence>& G,
                std::span<const node_index> active,
                std::span<const index_t> degrees,
                double degree_threshold,
                std::span<char> chosen,
                const factor_options& opts) {
        // Fibonacci hashing: maps sequential indices to well-spread priorities.
        // Constant = 2^64 / φ (golden ratio), ensuring good dispersion for
        // sequential vertex indices common in grid/mesh graphs.
        auto prio = [](node_index v) -> uint64_t {
            return uint64_t(v) * 11400714819323198485ULL;
        };

        #pragma omp parallel for schedule(static) if(active.size() > opts.omp_threshold)
        for (size_t i = 0; i < active.size(); ++i) {
            if (degrees[i] > degree_threshold) continue;
            auto v = active[i];
            auto pv = prio(v);
            bool is_local_min = true;
            for (auto idx : G.adj(v)) {
                auto u = G.edge_target(idx, v);
                if (G.is_active(u) && prio(u) < pv) {
                    is_local_min = false;
                    break;
                }
            }
            if (is_local_min) chosen[v] = 1;
        }
    }

    void cleanup(std::span<const node_index> /*active*/,
                 const factor_options& /*opts*/) {
        // No selector-specific scratch to reset.
    }
};

} // namespace apxchol
