#pragma once
#include "apxchol/solver/solve.h"
#include <memory>
#include <vector>

namespace apxchol::detail {
// CPU alternative of solver's backend variant; the benchmark also uses it for
// the triangular-solve statistics of its preconditioner.
// Setup and solves stay on CPU, including in GPU-enabled builds.
// The optional checkpoint receives setup timings and is not retained.
// A call made while another thread is using the same instance throws.
class cpu_route {
    friend struct solver_archive_access;
public:
    explicit cpu_route(const Eigen::SparseMatrix<double>& L,
                       const solve_options& opts = {},
                       checkpoint* cp = nullptr);

    /// Adopt an externally computed factorization (e.g. from a factorize()
    /// call with a custom eliminator or partitioner) instead of factorizing
    /// internally.  L is still required — it is the PCG operator.
    /// Construction options (including regularization/lumping) are not reapplied;
    /// factor storage, dropping, scheduling and solve options still apply.
    /// L is interpreted through its lower triangle. The caller must ensure it
    /// is a valid operator compatible with F; solver(L, F) validates it.
    cpu_route(const Eigen::SparseMatrix<double>& L,
              factorization F,
              const solve_options& opts = {},
              checkpoint* cp = nullptr);

    /// Copies share the factor, triangular-solve structures and operator, and
    /// get their own workspace, so each thread can solve with its own copy.
    cpu_route(const cpu_route& other);
    cpu_route& operator=(const cpu_route& other);

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

    const cpu_preconditioner& preconditioner() const { return precond_; }
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
    /// other stored patterns use the general lower-reflection fallback.
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
    cpu_preconditioner precond_;
    // Reused PCG workspace, resized when the panel width changes. Guarded by
    // the preconditioner's call guard, which rejects overlapping calls.
    mutable Eigen::VectorXd r_, z_, p_, Ap_, panel_x_;
    // Per-thread partial sums of the fused PCG kernels' deterministic
    // reductions (one cache line per thread; sized to the max team on use).
    mutable std::vector<double> part_;
    std::shared_ptr<const pcg_operator> op_;
    Eigen::Index n_ = 0;
};

} // namespace apxchol::detail
