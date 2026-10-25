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

/// CPU/automatic are always available without querying a device. GPU requires
/// a usable device and compatible compiled kernels; options may still be unsupported.
bool backend_available(solve_backend backend) noexcept;

struct solve_options {
    double tol       = default_tol;
    int    max_iter  = default_max_iter;
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
    bool collect_timings = true;
};

namespace detail {
void validate_solve_parameters(double tol, int max_iter);
void validate_solve_options(const solve_options& opts);
void resolve_solve_parameters(double& tol, int& max_iter, double default_tolerance,
                              int default_iterations);
}

struct solve_result {
    Eigen::VectorXd x;
    Eigen::Index iterations = 0;
    /// Recomputed relative residual of the returned solution against the original operator/RHS.
    double residual      = 0.0;
    bool converged = false; // residual < the effective tolerance, including zero-iteration exits.
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

using matrix_view = Eigen::Ref<Eigen::MatrixXd, 0,
    Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
using const_matrix_view = Eigen::Ref<const Eigen::MatrixXd, 0,
    Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

namespace detail {
void check_matrix_views(const_matrix_view B, matrix_view X, Eigen::Index rows,
                        const const_matrix_view* X0 = nullptr);
}

struct batch_solve_result {
    // Solutions are written to caller storage; each column's x remains empty.
    std::vector<solve_result> columns;
    checkpoint timings;
};

/// Generate a random zero-mean unit RHS vector using Eigen's global std::rand state.
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
    /// Construction options (including regularization/lumping) are not reapplied;
    /// factor storage, dropping, scheduling and solve options still apply.
    /// L is interpreted through its lower triangle. The caller must ensure it
    /// is a valid operator compatible with F; use solver(L, F) for validation.
    cpu_solver(const Eigen::SparseMatrix<double>& L,
               factorization F,
               const solve_options& opts = {},
               checkpoint* cp = nullptr);

    /// PCG against the held factor. Finite tol < 0 / max_iter < 0 = use opts values.
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
    /// contents overwritten — pass a warm start via `x0`). Reuses the PCG
    /// workspace, which can resize when panel width changes. Extreme inputs
    /// need scaled copies.
    /// The returned result's `x` member is left empty.
    /// Note: like all solve/apply paths, not safe for concurrent calls on the
    /// same cpu_solver (the reused workspace is shared state).
    solve_result solve(const Eigen::VectorXd& b, Eigen::Ref<Eigen::VectorXd> x,
                       double tol = -1.0, int max_iter = -1,
                       const Eigen::VectorXd* x0 = nullptr) const;

    /// One preconditioner application: z = M^{-1} r (forward+back SpTRSV).
    /// Inputs must be finite, for both scalar and matrix application.
    Eigen::VectorXd apply(const Eigen::VectorXd& r) const;

    const apx_cholesky& preconditioner() const { return precond_; }
    /// Independent RHS columns, with bounded panel workspace. Output must not
    /// overlap the RHS or warm start. tol/max_iter apply to each column.
    template<class Input, class Output>
        requires (Input::ColsAtCompileTime != 1
            && (Input::Flags & Eigen::DirectAccessBit) != 0
            && (Output::Flags & Eigen::DirectAccessBit) != 0)
    batch_solve_result solve(const Eigen::MatrixBase<Input>& B, Eigen::MatrixBase<Output>& X,
                             double tol = -1, int max_iter = -1,
                             const const_matrix_view* X0 = nullptr) const {
        using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
        const Eigen::Map<const Eigen::MatrixXd, 0, Stride> rhs(B.derived().data(),
            B.rows(), B.cols(), Stride(B.colStride(), B.rowStride()));
        Eigen::Map<Eigen::MatrixXd, 0, Stride> out(X.derived().data(),
            X.rows(), X.cols(), Stride(X.colStride(), X.rowStride()));
        return solve_batch(const_matrix_view(rhs), matrix_view(out), tol, max_iter, X0);
    }
    void apply(const_matrix_view R, matrix_view Z) const;

    Eigen::Index rows() const { return n_; }

private:
    batch_solve_result solve_batch(const_matrix_view B, matrix_view X, double tol,
                                   int max_iter, const const_matrix_view* X0) const;
    /// Shared ctor tail: builds the owned row-major full-symmetric SpMV
    /// operator (fp32 when lossless) once the preconditioner is ready. A full
    /// symmetric CSC is copied in its already-compatible CSR array layout;
    /// one-triangle custom inputs use the general lower-reflection fallback.
    void build_operator(const Eigen::SparseMatrix<double>& L, checkpoint* cp);
    /// PCG core: solution written into `x`; res.x is not touched.
    void solve_impl(const Eigen::VectorXd& b, Eigen::Ref<Eigen::VectorXd> x,
                    solve_result& res, double tol, int max_iter,
                    const Eigen::VectorXd* x0) const;

    template<int Columns>
    void solve_panel(const double* b, double* x, solve_result* results, checkpoint& timings,
                     double tol, int max_iter, const double* x0) const;
    template<int Columns>
    void check_solution(const_matrix_view B, const double* x, solve_result* results,
                        checkpoint& timings, double tol) const;

    solve_options opts_;
    apx_cholesky precond_;
    // Reused PCG workspace, resized when the panel width changes.
    // Concurrent solves on one cpu_solver are a race.
    mutable Eigen::VectorXd r_, z_, p_, Ap_, panel_x_;
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
/// Supplied factors use CPU in automatic mode; an explicit GPU request uploads them.
class solver {
public:
    explicit solver(const Eigen::SparseMatrix<double>& A,
                    const solve_options& opts = {}, checkpoint* cp = nullptr);
    /// Adopt F without rebuilding or regularizing it. lump_max_mass still bounds
    /// operator validation; factor storage, dropping and solve options apply.
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
    /// Apply the preconditioner to finite scalar or matrix input.
    Eigen::VectorXd apply(const Eigen::VectorXd& r) const;
    /// Independent RHS columns, with bounded panel workspace. Output must not
    /// overlap the RHS or warm start. tol/max_iter apply to each column.
    template<class Input, class Output>
        requires (Input::ColsAtCompileTime != 1
            && (Input::Flags & Eigen::DirectAccessBit) != 0
            && (Output::Flags & Eigen::DirectAccessBit) != 0)
    batch_solve_result solve(const Eigen::MatrixBase<Input>& B, Eigen::MatrixBase<Output>& X,
                             double tol = -1, int max_iter = -1,
                             const const_matrix_view* X0 = nullptr) const {
        using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
        const Eigen::Map<const Eigen::MatrixXd, 0, Stride> rhs(B.derived().data(),
            B.rows(), B.cols(), Stride(B.colStride(), B.rowStride()));
        Eigen::Map<Eigen::MatrixXd, 0, Stride> out(X.derived().data(),
            X.rows(), X.cols(), Stride(X.colStride(), X.rowStride()));
        return solve_batch(const_matrix_view(rhs), matrix_view(out), tol, max_iter, X0);
    }
    void apply(const_matrix_view R, matrix_view Z) const;

    const factorization& factor() const;
    Eigen::Index rows() const;
    solve_backend backend() const;
    factor_storage_type factor_storage() const;

private:
    batch_solve_result solve_batch(const_matrix_view B, matrix_view X, double tol,
                                   int max_iter, const const_matrix_view* X0) const;
    struct impl;
    std::unique_ptr<impl> impl_;
};

/// One-shot solve: factorize L, then solve Lx = b via PCG.
/// Automatic selects GPU when the compiled backend is available and compatible,
/// otherwise CPU. Explicit GPU selects device-owned CUDA or Metal setup and solve,
/// according to the build.
/// Unsupported inputs and execution failures are errors, never a CPU retry.
solve_result solve(const Eigen::SparseMatrix<double>& L,
                   const Eigen::VectorXd& b,
                   const solve_options& opts = {});



} // namespace apxchol
