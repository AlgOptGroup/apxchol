#pragma once
/// Baumann-Kyng independent set selection via random sampling.
///
/// ─── Algorithm overview ───
///
/// This implements the IS selection framework from:
///
///   Baumann Y. & Kyng R. (2024), "A Framework for Parallelizing
///   Approximate Gaussian Elimination," SPAA '24, pp. 195–206.
///   DOI: 10.1145/3626183.3659987
///
/// The key insight: for a graph with m edges where all candidate
/// vertices have degree ≤ √m, a uniform random sample of ≈ n / √m
/// vertices contains a constant fraction of isolated vertices (in the
/// induced subgraph of the sample).  These isolated vertices form an
/// independent set in the original graph.
///
/// ─── Algorithm (simplified) ───
///
/// The paper's Algorithm 3 "ExtractIS":
///   1. Fix a random permutation π over low-degree active vertices.
///   2. In iteration k, take the k-th prefix of size ≈ |active| / √m.
///   3. Build the induced subgraph on the prefix.
///   4. Return isolated vertices (degree 0 in induced subgraph) as IS.
///
/// Our implementation simplifies this to a single round: sample each
/// low-degree vertex independently with probability p = c / √m (for a
/// tunable constant c), then check which sampled vertices have no
/// sampled neighbors.
///
/// ─── IS size ───
///
/// For maximum degree Δ ≤ √m, each sampled vertex is isolated in the
/// induced subgraph with probability ≥ (1 − p)^Δ ≈ e^{−Δp}.  With
/// p = c/√m and Δ ≤ √m, this gives isolation probability ≥ e^{−c}.
///
/// Expected IS size per round ≈ |active| · p · e^{−c} = |active| · c · e^{−c} / √m.
///
/// This is asymptotically optimal for sub-linear depth (O(√m) rounds
/// suffice to eliminate all vertices), but produces smaller IS per
/// round than greedy or block-greedy for practical graphs.
///
/// ─── Implementation notes ───
///
/// We use a per-round deterministic hash (seeded by round counter) to
/// decide which vertices are "sampled," avoiding explicit permutation
/// storage.  The sampling constant c is exposed as a tunable parameter.
///
/// ─── When to use ───
///
/// This strategy is primarily of theoretical interest — it matches the
/// SPAA '24 paper's analysis and achieves O(√m log³ n) depth with
/// O(m log³ n) work.  For practical performance on moderate-size PDE
/// graphs, block_greedy_is is faster due to larger IS per round.

#include "apxchol/solver/is/independent_set.h"
#include <cmath>

namespace apxchol {

struct baumann_kyng_is {
    /// Internal round counter (incremented each call to select).
    uint64_t round = 0;

    template<incidence_storage Incidence>
    void select(const graph<Incidence>& G,
                std::span<const node_index> active,
                std::span<const index_t> degrees,
                double degree_threshold,
                std::span<char> chosen,
                const factor_options& opts) {
        // Sampling probability: p = 1 / (c · d_max), clamped to [0, 1].
        // Yves Baumann suggested using 1/d, 1/(2d), 1/(3d) to adapt to
        // actual graph structure rather than the paper's 1/√m.
        double c = opts.bk_sampling_constant;
        double d_max = std::max(degree_threshold, 1.0);
        double p = std::min(1.0 / (c * d_max), 1.0);

        // Hash threshold: sample vertex if hash(v, round) < p * 2^64.
        // Use a per-round seed mixed into the hash to vary sampling across rounds.
        uint64_t threshold = (p >= 1.0) ? UINT64_MAX
                           : uint64_t(p * double(UINT64_MAX));
        uint64_t round_seed = round * 6364136223846793005ULL + 1442695040888963407ULL;

        // Phase 1: mark sampled vertices.
        // We store sampling state in chosen[] temporarily:
        //   chosen[v] = 1 means "sampled" (candidate for IS).
        #pragma omp parallel for schedule(static) if(active.size() > opts.omp_threshold)
        for (size_t i = 0; i < active.size(); ++i) {
            if (degrees[i] > degree_threshold) continue;
            auto v = active[i];
            uint64_t h = (uint64_t(v) ^ round_seed) * 11400714819323198485ULL;
            if (h <= threshold) chosen[v] = 1;
        }

        // Phase 2: un-mark sampled vertices that have a sampled neighbor
        // (not isolated in the induced subgraph of the sample).
        //
        // The paper's ExtractIS uses strict isolation (remove both endpoints),
        // but this gives IS ≈ n·p·(1-p)^d — on regular grids with p≈0.4,
        // this is only ~0.6% of vertices, triggering immediate sequential
        // fallback.  Using index tie-breaking keeps the lower-indexed
        // endpoint (equivalent to random-priority within the sample),
        // giving IS ≈ |S|/(d_S+1) ≈ 16%, which sustains the parallel loop.
        #pragma omp parallel for schedule(static) if(active.size() > opts.omp_threshold)
        for (size_t i = 0; i < active.size(); ++i) {
            auto v = active[i];
            if (chosen[v] != 1) continue;
            for (auto idx : G.adj(v)) {
                auto u = G.edge_target(idx, v);
                if (G.is_active(u) && chosen[u] >= 1) {
                    // Both endpoints sampled — not isolated.
                    // Use index tie-break: higher index loses.
                    if (u < v) { chosen[v] = 2; break; }
                }
            }
        }

        // Phase 3: clear non-isolated markers.
        #pragma omp parallel for schedule(static) if(active.size() > opts.omp_threshold)
        for (size_t i = 0; i < active.size(); ++i) {
            auto v = active[i];
            if (chosen[v] == 2) chosen[v] = 0;
        }

        ++round;
    }

    void cleanup(std::span<const node_index> /*active*/,
                 const factor_options& /*opts*/) {
        // No selector-specific scratch to reset.
    }
};

} // namespace apxchol
