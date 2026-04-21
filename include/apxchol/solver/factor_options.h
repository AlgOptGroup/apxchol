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

/// Strategy for the serial residual peel that runs after the main parallel
/// IS-finding loop bails out.  All strategies are serial; they differ only
/// in pivot order.
enum class residual_peel_strategy {
    natural,     // peel in original active-list order (cheapest, current default)
    min_degree,  // pick min current-degree vertex each step (best fill-in)
    bk_serial    // sample √|active| vertices and peel the min-degree one
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

    // Pivot ordering strategy for the serial residual peel (eliminate_remaining).
    // - natural    (default): O(|active|) per step, no extra work.
    // - min_degree: O(|active| log |active|) total, picks lowest-degree pivot
    //   each step.  Tends to reduce final nnz(L) on dense residuals.
    // - bk_serial: sample √|active| vertices, pivot on the lowest-degree
    //   sample.  Cheap heuristic that approximates min_degree.
    residual_peel_strategy residual_peel = residual_peel_strategy::natural;
    // forward_star adjacency-pool compaction trigger.  After every round,
    // if the live fraction of nodes_ falls below this threshold, rebuild
    // the pool with each chain laid out contiguously.  Eliminates the
    // pointer-chase fragmentation that accumulates over many filter() calls.
    //
    // 0.0 disables; default 0.5 = compact when only 50% of slots are live.
    // (Empirically 0.5–0.75 wins ~25-30% setup time on Yves IPM matrices.)
    // forward_star storage only; vec/small_vec ignore this option.
    double fs_compact_threshold = 0.5;

    // forward_star: when true, filter() writes survivors contiguously at the
    // end of the node pool instead of doing in-place re-link.  Trades pool
    // growth (reclaimed by compact_adj()) for per-chain locality without
    // waiting for a global compact pass.  forward_star only.
    bool fs_filter_append = false;
};

} // namespace apxchol
