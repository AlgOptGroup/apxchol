#pragma once
#include "apxchol/checkpoint.h"
#include "apxchol/solver/preconditioner.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>

namespace apxchol {

inline constexpr double default_tol      = 1e-8;
inline constexpr int    default_max_iter = 200;

struct solve_options {
    double tol       = default_tol;
    int    max_iter  = default_max_iter;
    /// Check every N iterations whether residual improved by ≥50%.
    /// If not, declare stagnation and stop early.  0 = disabled.
    int    stagnation_window = 50;
    graph_storage storage = graph_storage::vec_pool;  // benchmark-best default
    /// Keep the factor's row/value arrays alive after SpTRSV setup so
    /// cpu_solver::preconditioner().factor().L stays readable (costs one extra
    /// factor-sized copy in memory; the bindings' L/D/P export needs it).
    bool keep_factor_values = false;
    factor_options factor_opts;
};

struct solve_result {
    Eigen::VectorXd x;
    Eigen::Index iterations = 0;
    double residual      = 0.0;
    // Positive off-diagonal entries the M-matrix lumping moved onto the
    // diagonal while building the PRECONDITIONER (0 = the operator was already
    // an M-matrix). The operator this residual is measured against is the one
    // the caller passed, lumping or not. See operator_class.h.
    Eigen::Index lumped_offdiag = 0;
    checkpoint timings;
    // Device VRAM (MB) held at solve end (operator + factor + PCG vectors still
    // resident), via cudaMemGetInfo. -1 = unmeasured (CPU build / host PCG).
    // Sampled inside solve() because the GPU-resident PCG frees all device state
    // before returning -- callers can't measure it post-hoc.
    double solve_vram_mb = -1.0;
};

/// Generate a random zero-mean unit RHS vector.
Eigen::VectorXd generate_test_rhs(Eigen::Index n);

/// Reusable CPU solver: factor + SpMV operator built ONCE, then PCG-solve any
/// number of right-hand sides. This is the machinery of the one-shot solve()
/// (which delegates to it), exposed for repeated-solve callers — the Python /
/// Octave bindings hold one of these per matrix.
///
///   cpu_solver slv(L);                       // factorize + build parallel-SpMV operator
///   auto r1 = slv.solve(b1);                 // tol/max_iter from opts
///   auto r2 = slv.solve(b2, 1e-10, 1000);    // per-call override
///   auto z  = slv.apply(r);                  // one preconditioner application, M^{-1} r
///
/// `cp` (optional) receives the setup-phase checkpoint records (factorize +
/// sptrsv_setup + spmv_lrm_build); the per-solve PCG records go into the
/// returned solve_result::timings. The one-shot solve() passes the SAME
/// checkpoint for both so its result carries the full setup+pcg tree.
/// NOTE: the preconditioner's internal forward/back records during PCG also
/// target `cp` — pass the result's checkpoint (as solve() does) or nullptr
/// (as the bindings do), not an unrelated one.
class cpu_solver {
public:
    explicit cpu_solver(const Eigen::SparseMatrix<double>& L,
                        const solve_options& opts = {},
                        checkpoint* cp = nullptr);

    /// Adopt an externally computed factorization (e.g. from a factorize()
    /// call with a custom eliminator or partitioner) instead of factorizing
    /// internally.  L is still required — it is the PCG operator.
    cpu_solver(const Eigen::SparseMatrix<double>& L,
               factorization F,
               const solve_options& opts = {},
               checkpoint* cp = nullptr);

    /// PCG against the held factor. tol < 0 / max_iter < 0 = use opts values.
    /// `x0` (optional) is the initial guess; nullptr = start from zero; a
    /// wrong-length x0 throws std::invalid_argument.
    /// For a pure Laplacian (rank n-1) the returned x is the min-norm
    /// solution (mean(x) = 0), whatever x0's constant component was.
    solve_result solve(const Eigen::VectorXd& b,
                       double tol = -1.0, int max_iter = -1,
                       const Eigen::VectorXd* x0 = nullptr) const;
    /// In-place variant: appends x/iterations/residual + the PCG checkpoint
    /// records into an existing result (used by the one-shot solve() so setup
    /// and pcg land in one timings tree).
    void solve(const Eigen::VectorXd& b, solve_result& res,
               double tol = -1.0, int max_iter = -1,
               const Eigen::VectorXd* x0 = nullptr) const;
    /// Caller-provided-memory variant: writes the solution into `x` (length n,
    /// contents overwritten — pass a warm start via `x0`) and allocates
    /// nothing per call beyond the first (the PCG workspace is reused across
    /// calls). The returned result's `x` member is left empty.
    /// Note: like all solve/apply paths, not safe for concurrent calls on the
    /// same cpu_solver (the reused workspace is shared state).
    solve_result solve(const Eigen::VectorXd& b, Eigen::Ref<Eigen::VectorXd> x,
                       double tol = -1.0, int max_iter = -1,
                       const Eigen::VectorXd* x0 = nullptr) const;

    /// One preconditioner application: z = M^{-1} r (forward+back SpTRSV).
    Eigen::VectorXd apply(const Eigen::VectorXd& r) const { return precond_.solve(r); }

    const apx_cholesky& preconditioner() const { return precond_; }
    Eigen::Index rows() const { return n_; }

private:
    /// Shared ctor tail: builds the row-major full-symmetric SpMV operator
    /// (fp32 when lossless) once the preconditioner is ready.
    void build_operator(const Eigen::SparseMatrix<double>& L, checkpoint* cp);
    /// PCG core: solution written into `x`; res.x is not touched.
    void solve_impl(const Eigen::VectorXd& b, Eigen::Ref<Eigen::VectorXd> x,
                    solve_result& res, double tol, int max_iter,
                    const Eigen::VectorXd* x0) const;

    solve_options opts_;
    apx_cholesky precond_;
    // Reused PCG workspace (lazily sized). Makes repeated solves allocation-
    // free; shared state — concurrent solves on one cpu_solver are a race.
    mutable Eigen::VectorXd r_, z_, p_, Ap_;
    // Per-thread partial sums of the fused PCG kernels' deterministic
    // reductions (one cache line per thread; sized to the max team on use).
    mutable std::vector<double> part_;
    // Row-major FULL symmetric operator for the parallel SpMV; exactly one of
    // the two is populated (fp32 when every value round-trips float losslessly).
    Eigen::SparseMatrix<double, Eigen::RowMajor> Lrm_;
    Eigen::SparseMatrix<float, Eigen::RowMajor>  Lrm_f_;
    bool op_fp32_ = false;
    Eigen::Index n_ = 0;
};

/// One-shot solve: factorize L, then solve Lx = b via PCG.
/// (Constructs a cpu_solver internally; on CUDA builds the GPU-resident PCG
/// path is taken instead.)
solve_result solve(const Eigen::SparseMatrix<double>& L,
                   const Eigen::VectorXd& b,
                   const solve_options& opts = {});



} // namespace apxchol
