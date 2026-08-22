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
    /// Minimum selector yield before handing a large residual to the BK path:
    /// selected regions / vertices ELIGIBLE under the degree cap. Relative to
    /// the candidate pool, not to all active vertices. A non-empty selection is
    /// always accepted once active <= 2 × parallel_residual_threshold (or the
    /// partitioner's default threshold): near that boundary a bail leaves too
    /// little BK runway and turns almost the whole tail into singleton peel
    /// levels. Zero progress always bails. This is an algorithmic yield knob;
    /// omp_threshold is also the absolute safeguard: a selection already that
    /// large is retained regardless of relative yield. On a residual with more
    /// than four handoff-sized chunks and average degree at least the handoff
    /// threshold, the effective yield is max(base, min(0.15, 3*base)); sparse,
    /// small and no-handoff partitioners use the base exactly. Set zero to
    /// disable yield-based handoff, including the dense-residual adaptation.
    double min_is_fraction = 0.05;
    size_t omp_threshold = 2000;     // min active/IS vertices before engaging OpenMP;
                                     // elimination also engages when selected-degree
                                     // work exceeds 24 times this value.
                                     // Also gates the partitioners' parallel paths;
                                     // lowering it below typical round sizes makes the
                                     // factor structure nondeterministic run to run
                                     // (racy block_greedy conflicts -- see env_knobs.h)
    std::string is_select = "block_greedy";  // Partitioner name (runtime dispatch via dispatch_partitioner)

    // When the main IS-finding loop bails out
    // (IS < a residual-size/density-adapted min_is_fraction · degree-eligible
    // candidates; exactly min_is_fraction near the handoff and on small
    // residuals),
    // the residual is worked down by BK rounds until active falls below this
    // threshold; the serial peel then takes the tail.
    //
    // SIZE_MAX here does NOT mean "serial peel": it means "defer to the
    // partitioner", and every shipped partitioner that can bail out
    // (block_greedy, luby, rootset) declares residual_handoff_threshold = 500,
    // so the BK residual loop IS on by default.  Set an explicit value to
    // override the trait; the serial peel is only reached for the last
    // `residual_handoff_threshold` vertices.
    //
    // Do NOT raise it to hand the peel more of the residual.  BK's per-round
    // yield does collapse on a dense residual (as-Skitter, T=16: ~80
    // vertices/round at the handoff, ~0.4/round by the time active reaches 600),
    // but the rounds still win: they buy the elimination ORDER, and the peel's
    // `natural` order is dearer per column than a BK round is per vertex.
    // Measured on as-Skitter (T=16, vec_pool, bg+tree, deterministic counters)
    // by forcing the handoff at a larger active and reading nnz(L) and
    // `elim_remaining` off the checkpoint:
    //     handoff at   500    1000    2000    3000    5325 (a first-whiff bail)
    //     nnz(L) M   18.48   18.51   18.63   18.83   19.43   (+5.1% at 5325)
    //     elim_rem ms   26      65     172     285     421
    //     BK find_partition saved by stopping there: 0 / 31 / 92 / 136 ms
    // i.e. every column moved to the peel costs both fill AND time.  The same
    // shape on com-LiveJournal (handoff at 500 / 2000 / 5000 / 10000 / 20000:
    // nnz 120.6 / 120.7 / 121.3 / 123.4 / 128.8 M, elim_rem 49 / 362 / 1326 /
    // 4829 / 5302 ms), coAuthorsDBLP and com-Amazon (setup lowest at 500 on
    // both, nnz +18% / +5% at a 15000 handoff).  A yield-triggered early stop
    // was implemented and measured against the same frontier and is dominated —
    // see the residual-loop comment in factorization_impl.h.
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
