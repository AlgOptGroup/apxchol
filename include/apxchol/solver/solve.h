#pragma once
#include "apxchol/checkpoint.h"
#include "apxchol/solver/preconditioner.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

namespace apxchol {

inline constexpr double default_tol      = 1e-8;
inline constexpr int    default_max_iter = 200;

struct solve_options {
    double tol       = default_tol;
    int    max_iter  = default_max_iter;
    graph_storage storage = graph_storage::forward_star;
    factor_options factor_opts;
};

struct solve_result {
    Eigen::VectorXd x;
    Eigen::Index iterations = 0;
    double residual      = 0.0;
    checkpoint timings;
};

/// Generate a random zero-mean unit RHS vector.
Eigen::VectorXd generate_test_rhs(Eigen::Index n);

/// One-shot solve: factorize L, then solve Lx = b via PCG.
solve_result solve(const Eigen::SparseMatrix<double>& L,
                   const Eigen::VectorXd& b,
                   const solve_options& opts = {});



} // namespace apxchol
