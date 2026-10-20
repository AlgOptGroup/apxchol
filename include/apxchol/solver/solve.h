#pragma once
#include "apxchol/checkpoint.h"
#include "apxchol/solver/preconditioner.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>
#include <memory>

namespace apxchol {

inline constexpr double default_tol      = 1e-8;
inline constexpr int    default_max_iter = 200;

enum class solve_backend { automatic, cpu, gpu };

struct solve_options {
    double tol       = default_tol;
    int    max_iter  = default_max_iter;
    /// Check every N iterations whether residual improved by ≥50%.
    /// If not, declare stagnation and stop early.  0 = disabled.
    int    stagnation_window = 50;
    /// Keep the factor's row/value arrays alive after SpTRSV setup so
    /// solver::factor().L stays readable (costs one extra host factor copy;
    /// the bindings' L/D/P export needs it, including in CUDA builds).
    bool keep_factor_values = false;
    factor_options factor_opts{};
    /// Select a complete setup/solve route; execution errors never change it.
    solve_backend backend = solve_backend::automatic;
    /// CPU thread ceiling for setup and every solve/apply. Positive values
    /// override the calling OpenMP task's setting; 0 inherits it.
    /// Small workloads may use fewer threads.
    int num_threads = 0;
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
    // Device-wide VRAM usage (MB), sampled at solve end via cudaMemGetInfo.
    // -1 = unmeasured (CPU build / host PCG).
    double solve_vram_mb = -1.0;
    solve_backend backend = solve_backend::cpu;
};

/// Generate a random zero-mean unit RHS vector.
Eigen::VectorXd generate_test_rhs(Eigen::Index n);

/// Reusable CPU solver: factor + SpMV operator built ONCE, then PCG-solve any
/// number of right-hand sides. Setup, triangular solves and PCG always run on
/// the CPU, including in CUDA builds.
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
/// The constructor does not retain `cp`; each solve records into its result.
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
    /// contents overwritten — pass a warm start via `x0`). Ordinary repeated
    /// solves reuse the PCG workspace; extreme inputs need scaled copies.
    /// The returned result's `x` member is left empty.
    /// Note: like all solve/apply paths, not safe for concurrent calls on the
    /// same cpu_solver (the reused workspace is shared state).
    solve_result solve(const Eigen::VectorXd& b, Eigen::Ref<Eigen::VectorXd> x,
                       double tol = -1.0, int max_iter = -1,
                       const Eigen::VectorXd* x0 = nullptr) const;

    /// One preconditioner application: z = M^{-1} r (forward+back SpTRSV).
    Eigen::VectorXd apply(const Eigen::VectorXd& r) const;
    /// The same application into caller storage, allocating nothing. r and z
    /// must not overlap: the permutation scatter reads r while it writes z.
    void apply(Eigen::Ref<const Eigen::VectorXd> r, Eigen::Ref<Eigen::VectorXd> z) const;
    /// ||b - A x|| / ||b|| in fp64 against the operator this solver's PCG
    /// applies (the caller's matrix; fp32 only when lossless unless
    /// APXCHOL_FP32_OPERATOR forces it), with power-of-two scaled norms that
    /// neither underflow nor overflow. 0 when b - A x = 0; infinity for a
    /// non-finite input or residual, or a nonzero residual of b = 0. Uses the
    /// PCG workspace, so like solve it is not safe concurrently.
    double relative_residual(Eigen::Ref<const Eigen::VectorXd> b,
                             Eigen::Ref<const Eigen::VectorXd> x) const;

    const apx_cholesky& preconditioner() const { return precond_; }
    Eigen::Index rows() const { return n_; }

private:
    /// Shared ctor tail: builds the owned row-major full-symmetric SpMV
    /// operator (fp32 when lossless) once the preconditioner is ready. A full
    /// symmetric CSC is copied in its already-compatible CSR array layout;
    /// one-triangle custom inputs use the general lower-reflection fallback.
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
    // Owned row-major FULL symmetric operator for the parallel SpMV; exactly
    // one is populated (fp32 when every value round-trips float losslessly).
    // These cannot map constructor input: a reusable solver may outlive it.
    Eigen::SparseMatrix<double, Eigen::RowMajor> Lrm_;
    Eigen::SparseMatrix<float, Eigen::RowMajor>  Lrm_f_;
    bool op_fp32_ = false;
    Eigen::Index n_ = 0;
};

/// Reusable solver owning one CPU or GPU backend, selected once at setup.
/// Owns its input data; movable, noncopyable, and not safe for concurrent calls.
/// CUDA uses device-owned setup and does not accept supplied host factors.
class solver {
public:
    explicit solver(const Eigen::SparseMatrix<double>& A,
                    const solve_options& opts = {}, checkpoint* cp = nullptr);
    solver(const Eigen::SparseMatrix<double>& A, factorization F,
           const solve_options& opts = {}, checkpoint* cp = nullptr);
    ~solver();
    solver(solver&&) noexcept;
    solver& operator=(solver&&) noexcept;
    solver(const solver&) = delete;
    solver& operator=(const solver&) = delete;

    solve_result solve(const Eigen::VectorXd& b, double tol = -1.0,
                       int max_iter = -1, const Eigen::VectorXd* x0 = nullptr) const;
    void solve(const Eigen::VectorXd& b, solve_result& result, double tol = -1.0,
               int max_iter = -1, const Eigen::VectorXd* x0 = nullptr) const;
    /// Write into caller storage; the returned result leaves x empty.
    solve_result solve(const Eigen::VectorXd& b, Eigen::Ref<Eigen::VectorXd> x,
                       double tol = -1.0, int max_iter = -1,
                       const Eigen::VectorXd* x0 = nullptr) const;
    Eigen::VectorXd apply(const Eigen::VectorXd& r) const;
    const factorization& factor() const;
    Eigen::Index rows() const;
    solve_backend backend() const;
    factor_storage_type factor_storage() const;
    /// The CPU owner, or nullptr on a GPU route, for the CPU-only extensions
    /// (caller-owned apply, original-system residual).
    const cpu_solver* cpu() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

/// One-shot solve: factorize L, then solve Lx = b via PCG.
/// Automatic selects GPU when the compiled backend is available and compatible,
/// otherwise CPU. Explicit GPU selects CUDA's device-owned
/// route or Metal's host-factor/GPU-solve route, according to the build.
/// Unsupported inputs and execution failures are errors, never a CPU retry.
solve_result solve(const Eigen::SparseMatrix<double>& L,
                   const Eigen::VectorXd& b,
                   const solve_options& opts = {});



} // namespace apxchol
