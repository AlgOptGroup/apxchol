#pragma once
/// Experiment-only environment knobs (branch exp/grounding-and-tails).
///
/// Every knob defaults to "unset", and unset means behaviour identical to the
/// code before this header existed. All values are read ONCE (first use) and
/// cached for the process lifetime, so flipping an env var mid-run has no
/// effect. Nothing here is public API: it exists so the bench harness can A/B
/// the two experiments below without a rebuild.
///
///   APXCHOL_GROUND = center | center-k | reg     (default: center)
///     How a pure Laplacian (rank n-1) is grounded in the preconditioner.
///       center    rank-(n-1) factor; every application subtracts the mean of
///                 the input and re-centres the output (the correct path that
///                 the SDDM-classification fix re-activated).
///       center-k  same factor, but the two mean-subtraction passes run only on
///                 every K-th application (K = APXCHOL_CENTER_K, default 10);
///                 the other applications skip both passes. PCG keeps the
///                 residual in the centred subspace in exact arithmetic, so
///                 the periodic centring is an fp-drift guard only; x may drift
///                 along the null space (1), which the residual does not see.
///       reg       do NOT centre: at make_graph time every vertex gets an
///                 explicit self-loop excess[v] = max(exact_excess, eps*diag[v]),
///                 eps = APXCHOL_REG_EPS (default 1e-8). The matrix is then
///                 classified SDDM (full-rank n factor, no centring passes) --
///                 i.e. the previous accidental fp32-phantom-excess behaviour,
///                 made explicit and deterministic.
///
///   APXCHOL_OMP_THRESHOLD = <int>
///     Overrides factor_options::omp_threshold (the IS-size gate below which an
///     elimination round runs the fully serial path).
///
///   APXCHOL_TAIL_THREADS = <int>
///     When set (> 0) and a round's IS is <= omp_threshold, run the PARALLEL
///     fused elimination path with num_threads(APXCHOL_TAIL_THREADS) (clamped
///     to the workspace team size) instead of the serial path. Unset = serial
///     tail as before. NOTE: the parallel path applies edges in thread-arrival
///     order, so the factor may differ from the serial-tail factor by fp merge-
///     order ulps.

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace apxchol::detail {

enum class grounding_kind { center, center_k, reg };

struct env_knobs {
    grounding_kind ground   = grounding_kind::center;
    int            center_k = 10;      // APXCHOL_CENTER_K (center-k mode only)
    double         reg_eps  = 1e-8;    // APXCHOL_REG_EPS  (reg mode only)
    long           omp_threshold = -1; // APXCHOL_OMP_THRESHOLD; < 0 = unset
    int            tail_threads  = 0;  // APXCHOL_TAIL_THREADS;  <= 0 = unset

    static const env_knobs& get() {
        static const env_knobs k = parse();
        return k;
    }

private:
    static env_knobs parse() {
        env_knobs k;
        if (const char* e = std::getenv("APXCHOL_GROUND"); e && *e) {
            if (std::strcmp(e, "center") == 0)        k.ground = grounding_kind::center;
            else if (std::strcmp(e, "center-k") == 0) k.ground = grounding_kind::center_k;
            else if (std::strcmp(e, "reg") == 0)      k.ground = grounding_kind::reg;
            else {
                std::fprintf(stderr,
                    "[apxchol] APXCHOL_GROUND='%s' unknown (center|center-k|reg); "
                    "using center\n", e);
            }
        }
        if (const char* e = std::getenv("APXCHOL_CENTER_K"); e && *e) {
            const int v = std::atoi(e);
            if (v >= 1) k.center_k = v;
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
            if (v > 0) k.tail_threads = v;
        }
        return k;
    }
};

} // namespace apxchol::detail
