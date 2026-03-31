#pragma once
/// Rootset (Blelloch) independent set selection.
///
/// ─── Algorithm overview ───
///
/// Implements the "rootset peeling" approach from Blelloch, Fineman &
/// Shun (2012).  The key insight: fix a random priority assignment
/// (hash) on vertices.  A vertex v can join the IS if ALL its eligible
/// neighbors with lower priority have already been decided (either
/// joined the IS or are blocked).  The priority DAG encodes these
/// dependencies; its depth is O(log²n) WHP for any graph, meaning
/// O(log²n) synchronous peeling rounds suffice to expand the IS to
/// the exact sequential greedy result under the random ordering.
///
/// Algorithm:
///   1. Assign each eligible vertex a random priority (hash).
///   2. For each eligible vertex v, compute priority[v] = count of
///      eligible active neighbors with lower hash.
///   3. Roots (priority 0) join the IS — they have no undecided
///      earlier-priority neighbor.
///   4. Neighbors of roots are "blocked" (cannot join IS).
///   5. For each blocked vertex u, decrement priority of u's later
///      neighbors (since u is now decided).  Vertices reaching
///      priority 0 become new roots.
///   6. Repeat from step 3 until no more roots.
///
/// ─── IS quality ───
///
/// This produces the exact sequential greedy IS under the random hash
/// ordering, which is optimal for our factorization (≈ n/(d+1) IS per
/// round for average degree d).  The quality matches block_greedy's
/// serial fallback and exceeds Luby's random-priority local minimum.
///
/// ─── Trade-offs ───
///
/// The number of peeling rounds per IS computation can be large
/// (O(log²n) ≈ 500 for n=4M), each requiring a frontier-expansion
/// synchronization.  On low-degree PDE graphs, block_greedy achieves
/// similar IS quality in a single parallel pass.  The rootset approach
/// is primarily useful for verifying IS quality or for graphs where
/// block partitioning fares poorly (e.g. highly irregular degree).
///
/// ─── Reference ───
///
///   • Blelloch, Fineman & Shun (2012), "Greedy Sequential Maximal
///     Independent Set and Matching are Parallel on Average,"
///     arXiv:1202.3205 / SPAA '12.

#include "apxchol/solver/is/independent_set.h"

namespace apxchol {

struct rootset_is {
    std::vector<int32_t> pri;      // priority = # of undecided earlier eligible neighbors
    std::vector<char> eligible;    // 1 if degree <= threshold, 0 otherwise
    std::vector<char> dead;        // 1 if in IS or blocked (neighbor of IS)
    uint64_t round = 0;

    template<incidence_storage Incidence>
    void select(const graph<Incidence>& G,
                std::span<const node_index> active,
                std::span<const index_t> degrees,
                double degree_threshold,
                std::span<char> chosen,
                const factor_options& opts) {
        size_t nn = G.n();
        pri.resize(nn, 0);
        eligible.resize(nn, 0);
        dead.resize(nn, 0);

        uint64_t seed = round * 6364136223846793005ULL + 1442695040888963407ULL;
        auto hash_fn = [seed](node_index v) -> uint64_t {
            return (uint64_t(v) ^ seed) * 11400714819323198485ULL;
        };

        // 1. Mark eligible vertices (degree ≤ threshold).
        #pragma omp parallel for schedule(static) if(active.size() > opts.omp_threshold)
        for (size_t i = 0; i < active.size(); ++i)
            eligible[active[i]] = (degrees[i] <= degree_threshold) ? 1 : 0;

        // 2. Compute initial priorities: count eligible active neighbors
        //    with lower hash.
        #pragma omp parallel for schedule(static) if(active.size() > opts.omp_threshold)
        for (size_t i = 0; i < active.size(); ++i) {
            auto v = active[i];
            if (!eligible[v]) { pri[v] = -1; continue; }
            auto hv = hash_fn(v);
            int32_t cnt = 0;
            for (auto idx : G.adj(v)) {
                auto u = G.edge_target(idx, v);
                if (G.is_active(u) && eligible[u] && hash_fn(u) < hv)
                    ++cnt;
            }
            pri[v] = cnt;
        }

        // 3. Collect initial roots (priority 0, eligible).
        std::vector<node_index> frontier;
        for (size_t i = 0; i < active.size(); ++i) {
            auto v = active[i];
            if (eligible[v] && pri[v] == 0)
                frontier.push_back(v);
        }

        // 4. Peel the priority DAG level by level.
        std::vector<node_index> removed;
        std::vector<node_index> next_frontier;

        while (!frontier.empty()) {
            // Mark frontier vertices as IS.
            for (auto v : frontier) {
                chosen[v] = 1;
                dead[v] = 1;
            }

            // Find blocked vertices: eligible active neighbors of IS
            // vertices that are still undecided.
            removed.clear();
            for (auto r : frontier) {
                for (auto idx : G.adj(r)) {
                    auto u = G.edge_target(idx, r);
                    if (G.is_active(u) && eligible[u] && !dead[u]) {
                        dead[u] = 1;
                        removed.push_back(u);
                    }
                }
            }

            // Decrement priorities: for each blocked vertex u, its later
            // eligible neighbors w (hash(u) < hash(w)) were counting u as
            // an undecided earlier neighbor.  Now u is decided → decrement.
            next_frontier.clear();
            for (auto u : removed) {
                auto hu = hash_fn(u);
                for (auto idx : G.adj(u)) {
                    auto w = G.edge_target(idx, u);
                    if (G.is_active(w) && eligible[w] && !dead[w]
                        && hu < hash_fn(w)) {
                        if (--pri[w] == 0)
                            next_frontier.push_back(w);
                    }
                }
            }

            std::swap(frontier, next_frontier);
        }

        ++round;
    }

    void cleanup(std::span<const node_index> active,
                 const factor_options& opts) {
        #pragma omp parallel for schedule(static) if(active.size() > opts.omp_threshold)
        for (size_t i = 0; i < active.size(); ++i) {
            auto v = active[i];
            pri[v] = 0;
            eligible[v] = 0;
            dead[v] = 0;
        }
    }
};

} // namespace apxchol
