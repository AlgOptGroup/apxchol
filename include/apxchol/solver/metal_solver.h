#pragma once
/// Apple GPU block PCG (build with -DAPXCHOL_USE_METAL=ON; macOS only).
///
/// metal_solver is an explicit opt-in: apxchol::solve() and cpu_solver are
/// unchanged by this backend. It factorizes on the host exactly as cpu_solver
/// does, applies the same dropped fp32 factor the CPU triangular solves store,
/// and runs preconditioned CG on up to 64 right-hand sides in lockstep on the
/// GPU (wider blocks run as sequential batches).
///
/// Precision (docs/precision.md, "Apple Metal block PCG"): fp32 factor values
/// with fp32 reciprocal diagonals; x, r, A p and (unless every value is
/// fp32-exact) the operator in double-float (hi + lo fp32, about 48 bits); p
/// and z in fp32; every reduction on one fixed tree that depends only on n.
/// The reported residual is recomputed on the host in fp64 against the
/// caller's operator. A column's bits are reproducible run to run and do not
/// depend on the block width, the other columns, the column's position or the
/// host thread count; they are not promised across devices, OS or Metal
/// compiler versions, nor equal to the CPU or CUDA solves.
///
/// Laplacians (factor().sddm == false): every preconditioner application is
/// centred, the right-hand side must be compatible with every connected
/// component (it is not projected), and the returned x is the min-norm
/// solution (mean(x) = 0) whatever x0's constant component was.
///
/// Calls on one object are serialized by an internal mutex.
#include "apxchol/checkpoint.h"
#include "apxchol/solver/factorization.h"
#include "apxchol/solver/solve.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace apxchol {

/// Why a column stopped.
enum class metal_stop {
    zero_rhs,             // b = 0: x = 0 without iterating
    initial_guess,        // x0 already met tol on the original operator
    recursive_tolerance,  // the double-float recursive residual met tol
    max_iterations,       // the iteration budget ran out
    breakdown,            // p.Ap <= 0 or non-finite (that iteration is not counted)
    stagnation,           // no 2x reduction within stagnation_window iterations
    nonfinite,            // a non-finite residual or PCG scalar
};

const char* to_string(metal_stop stop) noexcept;

struct metal_column_result {
    Eigen::Index iterations = 0;
    double residual = 0.0;            // ||b - A x|| / ||b||, host fp64, the caller's operator
    double recursive_residual = 0.0;  // the device recursion's ||r|| / ||b|| at exit
    bool converged = false;           // residual < tol (strict)
    metal_stop stop = metal_stop::max_iterations;
};

struct metal_block_result {
    Eigen::MatrixXd X;  // empty for the caller-memory overload
    std::vector<metal_column_result> columns;
    Eigen::Index lumped_offdiag = 0;  // see solve_result::lumped_offdiag
    checkpoint timings;
};

class metal_solver {
public:
    using block_cref = Eigen::Ref<const Eigen::MatrixXd>;

    struct statistics {
        Eigen::Index n = 0;
        std::size_t levels_forward = 0;
        std::size_t levels_backward = 0;
        std::size_t steps_forward = 0;   // dispatch steps per sweep at the full block width
        std::size_t steps_backward = 0;
        int block_columns = 0;           // right-hand sides per device batch
        bool operator_double_float = false;  // some operator value is not fp32-exact
        std::string device;
    };

    /// Factorizes A (operator contract of operator_class.h) and prepares the
    /// device. Throws std::runtime_error if available() is false (before
    /// factorizing), std::invalid_argument on an operator-contract violation,
    /// std::domain_error on operator or factor magnitudes outside
    /// [2^-100, 2^100] (or a zero factor diagonal).
    explicit metal_solver(const Eigen::SparseMatrix<double>& A,
                          const solve_options& opts = {}, checkpoint* cp = nullptr);
    /// Adopts an externally computed factorization of A (values retained).
    metal_solver(const Eigen::SparseMatrix<double>& A, factorization F,
                 const solve_options& opts = {}, checkpoint* cp = nullptr);
    ~metal_solver();
    metal_solver(metal_solver&&) noexcept;
    metal_solver& operator=(metal_solver&&) noexcept;
    metal_solver(const metal_solver&) = delete;
    metal_solver& operator=(const metal_solver&) = delete;

    /// Built with Metal, a usable device, the kernels compiled at run time and
    /// the device's double-float self-test passed. Checked once per process.
    static bool available() noexcept;

    /// One right-hand side. tol < 0 / max_iter < 0 select the options' values;
    /// x0 (nullptr = zero) must have length n. b must be finite.
    solve_result solve(const Eigen::VectorXd& b, double tol = -1.0, int max_iter = -1,
                       const Eigen::VectorXd* x0 = nullptr) const;

    /// k right-hand sides (n x k, any k >= 0) in lockstep batches.
    metal_block_result solve(block_cref B, double tol = -1.0, int max_iter = -1,
                             const block_cref* X0 = nullptr) const;
    /// The same into caller memory X (n x k); the result's X is left empty.
    metal_block_result solve(block_cref B, Eigen::Ref<Eigen::MatrixXd> X, double tol = -1.0,
                             int max_iter = -1, const block_cref* X0 = nullptr) const;
    /// Eigen expressions convert to both the vector and the block parameter
    /// types; these route a compile-time vector to the one-RHS solve and
    /// anything else (e.g. a MatrixXd) to the block solve.
    template <class Derived>
        requires(Derived::ColsAtCompileTime == 1)
    solve_result solve(const Eigen::MatrixBase<Derived>& b, double tol = -1.0, int max_iter = -1,
                       const Eigen::VectorXd* x0 = nullptr) const {
        return solve(Eigen::VectorXd(b), tol, max_iter, x0);
    }
    template <class Derived>
        requires(Derived::ColsAtCompileTime != 1)
    metal_block_result solve(const Eigen::MatrixBase<Derived>& B, double tol = -1.0,
                             int max_iter = -1, const block_cref* X0 = nullptr) const {
        return solve(block_cref(B), tol, max_iter, X0);
    }

    /// One device preconditioner application z = M^{-1} r (centred for a
    /// Laplacian).
    Eigen::VectorXd apply(const Eigen::VectorXd& r) const;

    /// The factorization (values released after setup unless
    /// solve_options::keep_factor_values).
    const factorization& factor() const;
    statistics stats() const;
    Eigen::Index rows() const;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}  // namespace apxchol
