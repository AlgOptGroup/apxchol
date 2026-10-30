#pragma once
#include "apxchol/checkpoint.h"
#include "apxchol/solver/preconditioner.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <array>
#include <vector>
#include <memory>

namespace apxchol {
namespace detail { struct solver_archive_access; }

inline constexpr double default_tol      = 1e-8;
inline constexpr int    default_max_iter = 200;

enum class solve_backend { automatic, cpu, gpu };

/// CPU/automatic are always available without querying a device. GPU requires
/// a usable device and compatible compiled kernels; options may still be unsupported.
bool backend_available(solve_backend backend) noexcept;

struct solve_options {
    /// Default relative residual tolerance; must be finite and positive.
    double tol       = default_tol;
    /// Nonnegative default iteration budget; zero only checks the initial guess.
    int    max_iter  = default_max_iter;
    /// Retain FP32 host factor values for export or adoption by another solver.
    /// Costs an additional host copy; false keeps the permutation but releases
    /// row/value arrays. Python factorize() enables retention by default.
    bool keep_factor_values = false;
    /// Factor construction and triangular-solve storage controls.
    factor_options factor_opts{};
    /// Select a complete setup/solve route; execution errors never change it.
    solve_backend backend = solve_backend::automatic;
    /// CPU thread ceiling for setup and every solve/apply. Positive values
    /// override the calling OpenMP task's setting; 0 inherits it.
    /// Small workloads may use fewer threads.
    int num_threads = 0;
    /// Record timing checkpoints. Final residual verification always runs.
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
    /// Relative residual checked against the original operator and RHS.
    double residual      = 0.0;
    /// Whether the checked residual is below the requested tolerance.
    bool converged = false;
    /// Positive off-diagonal entries moved onto the preconditioner's diagonal.
    /// The original operator is unchanged, including for residual verification.
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

/// CPU-specific reusable solver, also used by the generic solver's CPU route.
/// Prefer solver with solve_backend::cpu unless direct preconditioner access is
/// needed. Setup and solves stay on CPU, including in GPU-enabled builds.
/// The optional checkpoint receives setup timings and is not retained.
/// A call made while another thread is using the same instance throws; see
/// docs/usage.md#thread-safety.
class cpu_solver {
    friend struct detail::solver_archive_access;
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

    /// Copies share the factor, triangular-solve structures and operator, and
    /// get their own workspace, so each thread can solve with its own copy.
    cpu_solver(const cpu_solver& other);
    cpu_solver& operator=(const cpu_solver& other);

    /// Solve using the retained factor. Negative tol/max_iter use constructor
    /// defaults. x0 is an optional initial guess; nullptr starts from zero.
    /// Laplacian solutions are centered to zero mean.
    solve_result solve(const Eigen::VectorXd& b,
                       double tol = -1.0, int max_iter = -1,
                       const Eigen::VectorXd* x0 = nullptr) const;
    /// Write into an existing result, preserving its setup timing records.
    void solve(const Eigen::VectorXd& b, solve_result& res,
               double tol = -1.0, int max_iter = -1,
               const Eigen::VectorXd* x0 = nullptr) const;
    /// Overwrite caller-provided x; supply an initial guess separately in x0.
    /// The returned result leaves x empty.
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
    void finish_solution(const_matrix_view B, double* x, solve_result* results,
                         checkpoint& timings, double tol, int max_iter) const;
    template<int Columns>
    void check_solution(const_matrix_view B, const double* x, solve_result* results,
                        checkpoint& timings, double tol) const;

    // Owned row-major FULL symmetric operator for the parallel SpMV; exactly
    // one matrix is populated (fp32 when every value round-trips float
    // losslessly). It cannot map constructor input: a reusable solver may
    // outlive it. Read-only after setup; copies share it.
    struct pcg_operator {
        Eigen::SparseMatrix<double, Eigen::RowMajor> Lrm;
        Eigen::SparseMatrix<float, Eigen::RowMajor>  Lrm_f;
        bool fp32 = false;
    };

    solve_options opts_;
    apx_cholesky precond_;
    // Reused PCG workspace, resized when the panel width changes. Guarded by
    // the preconditioner's call guard, which rejects overlapping calls.
    mutable Eigen::VectorXd r_, z_, p_, Ap_, panel_x_;
    // Per-thread partial sums of the fused PCG kernels' deterministic
    // reductions (one cache line per thread; sized to the max team on use).
    mutable std::vector<double> part_;
    std::shared_ptr<const pcg_operator> op_;
    Eigen::Index n_ = 0;
};

/// Reusable solver owning one CPU or GPU backend, selected once at setup.
/// Owns its input data. solve() and apply() use this object's workspace, so
/// they are non-const. Concurrent non-const calls on one object are not allowed:
/// use a copy per thread, or batch the right-hand sides
/// (docs/usage.md#thread-safety). An overlapping call throws.
/// Copies of a CPU solver share its factor and operator; GPU solvers cannot be copied.
/// Every solve verifies the returned residual against the original operator/RHS.
/// Supplied factors use CPU in automatic mode; an explicit GPU request uploads them.
class solver {
    friend struct detail::solver_archive_access;
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
    solver(const solver& other);
    solver& operator=(const solver& other);

    /// Solve Ax=b. Negative tol/max_iter use constructor defaults. x0 supplies
    /// an optional initial guess; nullptr starts from zero. Inspect converged
    /// in the result even when the iteration budget is exhausted.
    solve_result solve(const Eigen::VectorXd& b, double tol = -1.0,
                       int max_iter = -1, const Eigen::VectorXd* x0 = nullptr);
    /// Write into an existing result, preserving its setup timing records.
    void solve(const Eigen::VectorXd& b, solve_result& result, double tol = -1.0,
               int max_iter = -1, const Eigen::VectorXd* x0 = nullptr);
    /// Write into caller storage; the returned result leaves x empty.
    solve_result solve(const Eigen::VectorXd& b, Eigen::Ref<Eigen::VectorXd> x,
                       double tol = -1.0, int max_iter = -1,
                       const Eigen::VectorXd* x0 = nullptr);
    /// Apply the preconditioner to finite scalar or matrix input.
    Eigen::VectorXd apply(const Eigen::VectorXd& r);
    /// Independent RHS columns, with bounded panel workspace. Output must not
    /// overlap the RHS or warm start. tol/max_iter apply to each column.
    template<class Input, class Output>
        requires (Input::ColsAtCompileTime != 1
            && (Input::Flags & Eigen::DirectAccessBit) != 0
            && (Output::Flags & Eigen::DirectAccessBit) != 0)
    batch_solve_result solve(const Eigen::MatrixBase<Input>& B, Eigen::MatrixBase<Output>& X,
                             double tol = -1, int max_iter = -1,
                             const const_matrix_view* X0 = nullptr) {
        using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
        const Eigen::Map<const Eigen::MatrixXd, 0, Stride> rhs(B.derived().data(),
            B.rows(), B.cols(), Stride(B.colStride(), B.rowStride()));
        Eigen::Map<Eigen::MatrixXd, 0, Stride> out(X.derived().data(),
            X.rows(), X.cols(), Stride(X.colStride(), X.rowStride()));
        return solve_batch(const_matrix_view(rhs), matrix_view(out), tol, max_iter, X0);
    }
    void apply(const_matrix_view R, matrix_view Z);

    /// Factor metadata; L's row/value arrays require keep_factor_values=true.
    /// The returned reference is owned by this solver.
    const factorization& factor() const;
    /// Operator dimension.
    Eigen::Index rows() const;
    /// Stored entries in the original operator, after input canonicalization
    /// and before preconditioner lumping or dropping.
    Eigen::Index operator_nnz() const;
    /// Selected route: cpu or gpu, never automatic.
    solve_backend backend() const;
    /// Installed triangular-solve storage: fp16 or fp32, never automatic.
    factor_storage_type factor_storage() const;

private:
    batch_solve_result solve_batch(const_matrix_view B, matrix_view X, double tol,
                                   int max_iter, const const_matrix_view* X0);
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
