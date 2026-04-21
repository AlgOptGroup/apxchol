#pragma once
/// Configurable options for the approximate Cholesky factorization.

#include <cstddef>
#include <limits>

namespace apxchol {

/// Enumerate available IS selection strategies for runtime dispatch.
enum class is_strategy { block_greedy, luby, baumann_kyng, rootset, hybrid };

/// Enumerate available elimination (clique sampling) strategies for runtime dispatch.
enum class elimination_strategy { tree, star, clique };

/// Enumerate orderings applied to the initial active vertex list.
/// Affects which vertices the greedy IS scan visits first; downstream
/// effects: IS size per round, total round count, and (most importantly
/// for sparse triangular solve) the level count of the resulting factor.
enum class vertex_order {
    natural,      // vertex index order (input order)
    random,       // Fisher-Yates shuffle, rng-seeded
    random_hash,  // sort by splitmix64(id ^ seed)
    degree_asc,   // ascending initial degree (low degree first)
    degree_desc   // descending initial degree
};

struct factor_options {
    unsigned seed = 42;
    double degree_multiplier = 2.0;  // IS degree threshold = multiplier × avg_degree
    double min_is_fraction = 0.05;   // fall back to sequential when IS < 5% of active
    size_t omp_threshold = 2000;     // min active/IS vertices before engaging OpenMP
    is_strategy is_select = is_strategy::block_greedy;  // IS selection strategy (runtime dispatch)
    elimination_strategy elim = elimination_strategy::tree;  // Elimination strategy (runtime dispatch)
    vertex_order order = vertex_order::natural;          // Initial active-list ordering
    double bk_sampling_constant = 0.3;  // BK: sample prob = 1/(c·d_max); lower c → larger IS

    // When the main IS-finding loop bails out (IS < min_is_fraction · active),
    // optionally parallelize the residual peel by switching to BK rounds until
    // active shrinks below this threshold, then serial peel the tail.
    //
    // Default SIZE_MAX = disabled (always serial peel).  Empirically the
    // serial peel wins on the dense, high-degree residuals BG produces:
    // BK rounds on the residual eliminate only ~100 vertices each, so the
    // 100s of fork-join overheads add up to more than the serial peel cost.
    // Set to a small value (e.g. 256) to enable for graphs with low-degree
    // or near-IS-shaped residuals.
    size_t parallel_residual_threshold = std::numeric_limits<size_t>::max();
};

} // namespace apxchol
