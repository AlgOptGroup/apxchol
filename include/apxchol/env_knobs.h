#pragma once
/// Experiment-only environment knobs (from branch exp/grounding-and-tails).
///
/// All values are read ONCE (first use) and cached for the process lifetime,
/// so flipping an env var mid-run has no effect. Nothing here is public API:
/// it exists so the bench harness can A/B the experiments below without a
/// rebuild. The tail knobs default to "unset" = prior behaviour; the grounding
/// and fused-parallel knobs have real defaults (center-k K = 10; 2000).
///
///   APXCHOL_GROUND = center-k | reg              (default: center-k)
///     How a pure Laplacian (rank n-1) is grounded in the preconditioner.
///       center-k  rank-(n-1) factor; the two mean-subtraction passes of an
///                 application (centre the input, re-centre the output) run
///                 only on every K-th application (K = APXCHOL_CENTER_K,
///                 default 10; applications K, 2K, ... of a 1-based per-solve
///                 count -- cpu_solver::solve restarts the count, so every
///                 solve on a factor sees the same schedule). The other
///                 applications skip both passes. PCG keeps the residual in
///                 the centred subspace in exact arithmetic, so the periodic
///                 centring is an fp-drift guard only; x may drift along the
///                 null space (1), which the residual does not see --
///                 cpu_solver::solve therefore subtracts mean(x) ONCE from
///                 the returned solution (min-norm solution, as with K = 1).
///                 K = 1 centres every application (the previous unconditional
///                 behaviour); its canonical spelling is
///                 APXCHOL_GROUND=center-k APXCHOL_CENTER_K=1.
///       center    accepted for compatibility as an alias of center-k with
///                 K = 1 (it overrides APXCHOL_CENTER_K).
///       reg       do NOT centre: at make_graph time every vertex gets an
///                 explicit self-loop excess[v] = max(exact_excess, eps*diag[v]),
///                 eps = APXCHOL_REG_EPS (default 1e-8). The matrix is then
///                 classified SDDM (full-rank n factor, no centring passes) --
///                 i.e. the previous accidental fp32-phantom-excess behaviour,
///                 made explicit and deterministic.
///
///   APXCHOL_OMP_THRESHOLD = <int>
///     Overrides factor_options::omp_threshold (the IS-size gate below which an
///     elimination round runs the fully serial path). NOTE it also gates the
///     PARTITIONER's parallel paths (partition_context.omp_threshold):
///     block_greedy's cross-block conflict resolution is racy, so lowering
///     this below typical round sizes makes the FACTOR STRUCTURE (nnz)
///     nondeterministic run to run -- measured 12/50 failures of
///     SpTRSVSetupMemory.SetupConsumingReleasesTheFactorAndSolvesIdentically
///     at 256 vs 0/50 at the 2000 default (2026-08-19, T=4). Promoting 256 to
///     the default was also perf-refuted: iter0040 10-rep paired medians show
///     setup 0.867 s (2000/serial tail) vs 0.869 s (256/tail 4) -- the
///     eliminate-stage win (~-22%) is offset elsewhere.
///
///   APXCHOL_TAIL_THREADS = <int>
///     When set (> 0) and a round's IS is <= omp_threshold, run the PARALLEL
///     fused elimination path with num_threads(APXCHOL_TAIL_THREADS) (clamped
///     to the workspace team size) instead of the serial path. Unset/0 =
///     serial tail as before. NOTE: the parallel path applies edges in
///     thread-arrival order, so the factor may differ from the serial-tail
///     factor by fp merge-order ulps (rounds whose IS fits in one
///     schedule(dynamic,64) chunk are processed by a single thread and stay
///     bit-deterministic). Tail 4 at threshold 2000 measured NEUTRAL-to-worse
///     on iter0040/grid_2000 (2026-08-19), so the default stays serial.
///
///   APXCHOL_FUSED_PARALLEL_MIN = <int>
///     Vector length above which the fused PCG / preconditioner vector passes
///     (src/solve.cpp + apply_core; see detail::fused_omp_min() in
///     preconditioner.h) engage their OpenMP team. At or below it the SAME
///     loops run on the encountering thread (the `if` clause disables the
///     team), bit-identical to a T=1 run. Default 2000, the value the passes
///     shipped with. Raising it was REFUTED by measurement (2026-08-19,
///     T=16, Zen 4): at n = 299k/335k (coAuthorsDBLP / com-Amazon -- vectors
///     are NOT cache-resident, 6 fp64 n-vectors alone are ~16 MB) every
///     gated pass is faster parallel (update_xr 0.086 vs 0.396 ms/iter,
///     permute 0.089 vs 0.171, unpermute+rz 0.096 vs 0.236, spmv+pAp 0.62
///     vs 4.32), and serializing them also starves the level-set SpTRSV
///     in between (sleeping team, forward 130 -> 226 ms). Knob kept for A/B.

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace apxchol::detail {

enum class grounding_kind { center_k, reg };

struct env_knobs {
    grounding_kind ground   = grounding_kind::center_k;
    int            center_k = 10;      // APXCHOL_CENTER_K (center-k mode only)
    double         reg_eps  = 1e-8;    // APXCHOL_REG_EPS  (reg mode only)
    long           omp_threshold = -1; // APXCHOL_OMP_THRESHOLD; < 0 = unset
    int            tail_threads  = 0;  // APXCHOL_TAIL_THREADS;  <= 0 = serial tail
    long           fused_parallel_min = 2000;   // APXCHOL_FUSED_PARALLEL_MIN

    static const env_knobs& get() {
        static const env_knobs k = parse();
        return k;
    }

private:
    static env_knobs parse() {
        env_knobs k;
        if (const char* e = std::getenv("APXCHOL_CENTER_K"); e && *e) {
            const int v = std::atoi(e);
            if (v >= 1) k.center_k = v;
        }
        // Parsed after APXCHOL_CENTER_K so the "center" alias can pin K = 1.
        if (const char* e = std::getenv("APXCHOL_GROUND"); e && *e) {
            if (std::strcmp(e, "center-k") == 0)      k.ground = grounding_kind::center_k;
            else if (std::strcmp(e, "center") == 0) { k.ground = grounding_kind::center_k; k.center_k = 1; }
            else if (std::strcmp(e, "reg") == 0)      k.ground = grounding_kind::reg;
            else {
                std::fprintf(stderr,
                    "[apxchol] APXCHOL_GROUND='%s' unknown (center-k|reg; center = center-k with K=1); "
                    "using center-k\n", e);
            }
        }
        if (const char* e = std::getenv("APXCHOL_REG_EPS"); e && *e) {
            const double v = std::atof(e);
            if (v > 0.0) k.reg_eps = v;
        }
        if (const char* e = std::getenv("APXCHOL_OMP_THRESHOLD"); e && *e) {
            const long v = std::atol(e);
            if (v >= 0) k.omp_threshold = v;
        }
        if (const char* e = std::getenv("APXCHOL_TAIL_THREADS"); e && *e) {
            const int v = std::atoi(e);
            if (v >= 0) k.tail_threads = v;
        }
        if (const char* e = std::getenv("APXCHOL_FUSED_PARALLEL_MIN"); e && *e) {
            const long v = std::atol(e);
            if (v >= 0) k.fused_parallel_min = v;
        }
        return k;
    }
};

} // namespace apxchol::detail
