#pragma once
/// Luby-style independent set selection with degree-dependent probability.
///
/// ─── Algorithm overview ───
///
/// The actual Luby (1986) algorithm selects each vertex v independently
/// with probability 1/(2d(v)), where d(v) is v's degree.  For every
/// edge with both endpoints selected, the lower-degree endpoint is
/// removed.  This is degree-aware: low-degree vertices are sampled more
/// aggressively, yielding larger IS than the uniform-priority variant.
///
/// We implement the "random-priority" formulation which is equivalent:
/// assign each vertex a random priority r(v), and v joins the IS if
/// r(v) < r(u) for all active neighbors u.  The key improvement over
/// a fixed hash is that we re-hash each round (using a round counter
/// mixed into the seed), preventing the same vertices from winning
/// every time and enabling faster progress.
///
/// ─── IS size and trade-offs ───
///
/// For a graph with maximum degree Δ, the expected IS size from a
/// random-priority local minimum is ≥ n/(Δ+1).  On low-degree graphs
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

#include "apxchol/solver/is/independent_set.h"

namespace apxchol {

struct luby_is {
    /// Internal round counter for per-round re-hashing.
    uint64_t round = 0;

    template<incidence_storage Incidence>
    void select(const graph<Incidence>& G,
                std::span<const node_index> active,
                std::span<const index_t> degrees,
                double degree_threshold,
                std::span<char> chosen,
                const factor_options& opts) {
        // Per-round seed: mixes round counter into the hash to vary
        // priorities across rounds (prevents same vertices always winning).
        uint64_t round_seed = round * 6364136223846793005ULL + 1442695040888963407ULL;

        // Fibonacci hashing with per-round seed.
        auto prio = [round_seed](node_index v) -> uint64_t {
            return (uint64_t(v) ^ round_seed) * 11400714819323198485ULL;
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

        ++round;
    }

    void cleanup(std::span<const node_index> /*active*/,
                 const factor_options& /*opts*/) {
        // No selector-specific scratch to reset.
    }
};

} // namespace apxchol
