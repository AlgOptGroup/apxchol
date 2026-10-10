#pragma once
// Reusable single-RHS Metal PCG. Host factorization and FP32 factor storage;
// double-float GPU recurrences, with the reported residual checked in host FP64.
// Repeated calls reuse the factor, operator and device workspace. Like cpu_solver,
// concurrent calls on one instance are unsupported.
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

class metal_solver {
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

    /// Factorizes A (operator contract of operator_class.h) and prepares the
    /// device. Throws std::runtime_error if available() is false (before
    /// factorizing), std::invalid_argument on an operator-contract violation,
    /// std::domain_error on operator or factor magnitudes outside
    /// [2^-100, 2^100] (or a zero factor diagonal), std::length_error when the
    /// factor or the operator (above 2^30 stored entries) exceeds the device's
    /// 32-bit offsets, and a std::bad_alloc (what() names the buffer) when the
    /// device cannot hold the system.
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
    /// Append solve timings to an existing result, as cpu_solver does.
    void solve(const Eigen::VectorXd& b, solve_result& result, double tol = -1.0,
               int max_iter = -1, const Eigen::VectorXd* x0 = nullptr) const;

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
