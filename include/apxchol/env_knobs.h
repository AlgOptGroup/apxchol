#pragma once
/// Environment knobs read by the CPU and GPU setup routes.
///
/// All values are read ONCE (first use) and cached for the process lifetime,
/// so flipping an env var mid-run has no effect. Nothing here is public API.
///
///   APXCHOL_GROUND = center-k | center | reg       (default: center-k)
///     How a pure Laplacian (rank n-1) is grounded in the preconditioner.
///       center-k  rank-(n-1) factor; the two mean-subtraction passes of an
///                 application (centre the input, re-centre the output) run
///                 only on every K-th application (K = 10) of a 1-based
///                 per-solve count -- cpu_solver::solve restarts the count, so
///                 every solve on a factor sees the same schedule. The other
///                 applications skip both passes. PCG keeps the residual in
///                 the centred subspace in exact arithmetic, so the periodic
///                 centring is an fp-drift guard only; x may drift along the
///                 null space (1), which the residual does not see --
///                 cpu_solver::solve therefore subtracts mean(x) ONCE from
///                 the returned solution (min-norm solution, as with K = 1).
///       center    center-k with K = 1: centre every application.
///       reg       do NOT centre: at make_graph time every vertex gets an
///                 explicit self-loop excess[v] = max(exact_excess, eps*diag[v]),
///                 eps = APXCHOL_REG_EPS (default 1e-8). The matrix is then
///                 classified SDDM (full-rank n factor, no centring passes).
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
    int            center_k = 10;      // centring period (center-k mode only)
    double         reg_eps  = 1e-8;    // APXCHOL_REG_EPS  (reg mode only)
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
