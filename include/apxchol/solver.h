#pragma once
#include "factorization.h"
#include "preconditioner.h"
#include <Eigen/Core>
#include <Eigen/Sparse>

namespace apxchol {

struct solve_options {
    double tol       = 1e-8;
    int    max_iter  = 200;
    factor_options factor_opts;
};

struct solve_result {
    Eigen::VectorXd x;
    int    iterations    = 0;
    double residual      = 0.0;
    double setup_seconds = 0.0;
    double solve_seconds = 0.0;
};

/// One-shot solve: factorize L, then solve Lx = b via PCG.
solve_result solve(const Eigen::SparseMatrix<double>& L,
                   const Eigen::VectorXd& b,
                   const solve_options& opts = {});

/// Reusable solver: factorize once, solve for multiple right-hand sides.
class solver {
public:
    solver() = default;
    explicit solver(const Eigen::SparseMatrix<double>& L,
                    const solve_options& opts = {});

    solve_result solve(const Eigen::VectorXd& b) const;

    const factorization& factor() const { return F_; }
    const Eigen::SparseMatrix<double>& matrix() const { return L_; }

private:
    Eigen::SparseMatrix<double> L_;
    factorization F_;
    solve_options opts_;
    double setup_seconds_ = 0.0;
};

} // namespace apxchol
