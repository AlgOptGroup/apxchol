#pragma once
// Metal PCG with reusable factor, operator and workspace.
// Concurrent calls on one instance are unsupported. apxchol::solver rejects
// them, and a Metal-backed solver cannot be copied (docs/usage.md#thread-safety).
#include "apxchol/checkpoint.h"
#include "apxchol/solver/factorization.h"
#include "apxchol/solver/solve.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cstddef>
#include <memory>
#include <string>

namespace apxchol {

class metal_solver {
    friend struct detail::solver_archive_access;
public:
    struct statistics {
        Eigen::Index n = 0;
        std::size_t levels_forward = 0;
        std::size_t levels_backward = 0;
        std::size_t steps_forward = 0;   // dispatch steps per sweep
        std::size_t steps_backward = 0;
        bool operator_double_float = false;  // some operator value is not fp32-exact
        std::string device;
    };

    /// Factorizes A on the device. Requires available(), supported GKS options
    /// and compressed, sorted, paired CSC input satisfying operator_class.h.
    /// Numerical limits are in docs/precision.md.
    explicit metal_solver(const Eigen::SparseMatrix<double>& A,
                          const solve_options& opts = {}, checkpoint* cp = nullptr);
    /// Adopts an externally computed factorization of A (values retained).
    /// Construction options (including regularization/lumping) are not reapplied.
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
    /// Append solve timings to an existing result, as cpu_solver does.
    void solve(const Eigen::VectorXd& b, solve_result& result, double tol = -1.0,
               int max_iter = -1, const Eigen::VectorXd* x0 = nullptr) const;

    /// One device preconditioner application z = M^{-1} r (centred for a
    /// Laplacian).
    Eigen::VectorXd apply(const Eigen::VectorXd& r) const;

    /// The factorization (values released after setup unless
    /// solve_options::keep_factor_values).
    const factorization& factor() const;
    factor_storage_type factor_storage() const;
    statistics stats() const;
    Eigen::Index rows() const;

private:
    friend class solver;
    // Automatic routing has already checked this same input when true.
    metal_solver(const Eigen::SparseMatrix<double>& A, const solve_options& opts,
                 checkpoint* cp, bool input_prechecked);
    batch_solve_result solve_panel(const_matrix_view B, matrix_view X, double tol,
                                  int max_iter, const const_matrix_view* X0 = nullptr) const;
    struct impl;
    struct impl_deleter {
        void operator()(impl*) const;
    };
    std::unique_ptr<impl, impl_deleter> impl_;
};

}  // namespace apxchol
