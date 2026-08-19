#pragma once
/// Configurable options for the approximate Cholesky factorization.

#include <cstddef>
#include <limits>
#include <string>

#include "apxchol/types.h"

namespace apxchol {

/// Strategy for the serial residual peel that runs after the main parallel
/// IS-finding loop bails out.  All strategies are serial; they differ only
/// in pivot order.
enum class residual_peel_strategy {
    natural,     // peel in original active-list order (cheapest, current default)
    min_degree,  // pick min current-degree vertex each step (best fill-in)
    bk_serial    // sample √|active| vertices and peel the min-degree one
};

/// Selection knobs consulted by the built-in partitioners — the only part of
/// the options a partitioner should normally read (as opts.partition).
struct partition_options {
    /// Quantile-based IS degree cap. When in (0,1), each round admits only the
    /// lowest `degree_quantile` fraction of active vertices by CURRENT degree
    /// (threshold found via nth_element, O(active)), instead of the
    /// `degree_multiplier × avg_degree` cap. Robust to the degree distribution:
    /// a multiplier cap below 1 can exclude *every* vertex on near-uniform-degree
    /// graphs (→ empty IS, round explosion), whereas a quantile always admits a
    /// fixed fraction. Eliminating the low-degree quantile first approximates
    /// adaptive min-degree in parallel. 0 = disabled (use the multiplier).
    /// DEFAULT 0.2: chosen by a q∈{0.1..0.5} sweep over grids/FEM/IPM/social
    /// — best all-around on iters AND total time. Fewer PCG iters on
    /// every tested matrix vs the multiplier cap, no convergence regressions;
    /// closes the adaptive-min-degree iteration gap on hub graphs (com-Amazon
    /// 58→35, coAuthors 81→25, kron 27→15) while staying parallel. Lower q (0.1)
    /// trims a touch more on uniform grids but adds rounds (worse total); higher
    /// q (0.3) helps com-Amazon slightly but is worse on the grid/FEM/IPM bulk.
    double degree_quantile = 0.2;

    /// Fallback IS degree cap when degree_quantile == 0:
    /// threshold = degree_multiplier × avg_degree.
    double degree_multiplier = 2.0;

    /// Conflict resolution / greedy tie-break by (degree, index) instead of index
    /// alone: when two adjacent candidates compete, the LOWER-degree one wins
    /// (index breaks ties). Biases the IS toward low-degree vertices with no
    /// per-block sort. block_greedy uses the (degree,index) resolution; luby and
    /// rootset fold degree into their selection priority (degree-major, hash
    /// tie-break). DEFAULT on — pairs with degree_quantile; small consistent
    /// iteration win, O(1) per conflict (no sort).
    bool degree_tiebreak = true;
};

struct factor_options {
    unsigned seed = 42;
    partition_options partition;     // selection knobs (see above)
    double min_is_fraction = 0.05;   // fall back to sequential when IS < 5% of active
    size_t omp_threshold = 2000;     // min active/IS vertices before engaging OpenMP.
                                     // Also gates the partitioners' parallel paths;
                                     // lowering it below typical round sizes makes the
                                     // factor structure nondeterministic run to run
                                     // (racy block_greedy conflicts -- see env_knobs.h)
    std::string is_select = "block_greedy";  // Partitioner name (runtime dispatch via dispatch_partitioner)

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
    // 0.0 (default) disables compaction entirely.
    //
    // Empirical findings on grid + IPM Laplacians (16T, see fs-compact sweep):
    //   * grid_2000:  every threshold > 0 makes setup slower by 30-100% vs off,
    //                 and total time is also worse.
    //   * LP-IPM Schur complements: helps root+tree (~25% total), neutral on
    //                 bg+tree, hurts bk+tree.
    // So we ship it OFF by default and let callers opt in per matrix class.
    // forward_star storage only; other backends ignore this option.
    double fs_compact_threshold = 0.0;

    // forward_star: when true, filter() writes survivors contiguously at the
    // end of the node pool instead of doing in-place re-link.  Trades pool
    // growth (reclaimed by compact_adj()) for per-chain locality without
    // waiting for a global compact pass.  forward_star only.
    bool fs_filter_append = false;

    /// Exact-clique-at-low-degree: when the eliminated vertex's degree is at
    /// most this value, emit the FULL exact Schur-complement clique
    /// (all d(d-1)/2 edges, weight w_i·w_j/deg) instead of the d-1 sampled
    /// edges. This is zero-variance (exact) elimination where it is cheap
    /// (quadratic in d, so only worth it for small d). Sampling resumes above
    /// the threshold. 0 = disabled (always sample). Targets preconditioner
    /// quality: most eliminated vertices are low-degree, so exact handling of
    /// them removes the bulk of the sampling variance at modest fill cost.
    size_t exact_clique_max_degree = 0;


};

} // namespace apxchol
