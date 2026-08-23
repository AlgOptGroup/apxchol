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
///     PARTITIONER's parallel paths (partition_context.omp_threshold), so
///     lowering it changes the factor by moving more rounds from serial to
///     parallel selection. The parallel path is deterministic at a fixed seed
///     and thread count; the old cross-block race was removed 2026-08-20.
///     Promoting 256 to the default was perf-refuted: iter0040 10-rep paired
///     medians show setup 0.867 s (2000/serial tail) vs 0.869 s (256/tail 4) --
///     the eliminate-stage win (~-22%) is offset elsewhere.
///
///   APXCHOL_LUMP = 0 | 1                          (default: 1, ON)
///     M-matrix lumping of positive off-diagonals when the PRECONDITIONER is
///     built (operator_class.h). Default ON because it cannot produce a wrong
///     answer -- PCG keeps applying the operator the caller passed -- only a
///     worse preconditioner. APXCHOL_LUMP=0 turns it off for an A/B: the run
///     PROCEEDS (both arms of an A/B must produce a number) but prints a
///     stderr warning, since the factorization is then handed negative edge
///     weights and the only symptom is a bad residual at the end.
///
///   APXCHOL_LUMP_MAX = <fraction in (0, 1]>       (default: 0.25)
///     Ceiling on the fraction of off-diagonal |mass| lumping may delete
///     before the matrix is refused instead. See kDefaultLumpMassCeiling in
///     operator_class.h for the measured basis.


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
    bool           lump = true;        // APXCHOL_LUMP
    // APXCHOL_LUMP_MAX; <= 0 = unset, resolved against
    // apxchol::kDefaultLumpMassCeiling (defined in operator_class.h, which
    // owns the documented default -- this header must not restate it).
    double         lump_max_mass = -1.0;

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
        if (const char* e = std::getenv("APXCHOL_LUMP"); e && *e)
            k.lump = std::strcmp(e, "0") != 0;
        if (const char* e = std::getenv("APXCHOL_LUMP_MAX"); e && *e) {
            const double v = std::atof(e);
            if (v > 0.0 && v <= 1.0) k.lump_max_mass = v;
            else
                std::fprintf(stderr,
                    "[apxchol] APXCHOL_LUMP_MAX='%s' is not a fraction in "
                    "(0, 1]; using the default ceiling\n", e);
        }
        return k;
    }
};

} // namespace apxchol::detail
