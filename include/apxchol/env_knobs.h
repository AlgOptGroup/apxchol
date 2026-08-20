#pragma once
/// Experiment-only environment knobs (from branch exp/grounding-and-tails).
///
/// All values are read ONCE (first use) and cached for the process lifetime,
/// so flipping an env var mid-run has no effect. Nothing here is public API:
/// it exists so the bench harness can A/B the experiments below without a
/// rebuild. The grounding knob has a real default (center-k, K = 10).
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
        return k;
    }
};

} // namespace apxchol::detail
