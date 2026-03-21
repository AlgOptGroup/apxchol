#include "apxchol/solver.h"
#include <Eigen/IterativeLinearSolvers>
#include <chrono>

namespace apxchol {

using Clock = std::chrono::high_resolution_clock;

solver::solver(const Eigen::SparseMatrix<double>& L,
               const solve_options& opts)
    : L_(L), opts_(opts) {
    auto t0 = Clock::now();
    F_ = apxchol::factorize(L_, opts_.factor_opts);
    auto t1 = Clock::now();
    setup_seconds_ = std::chrono::duration<double>(t1 - t0).count();
}

solve_result solver::solve(const Eigen::VectorXd& b) const {
    solve_result res;
    res.setup_seconds = setup_seconds_;

    preconditioner M(F_);

    Eigen::ConjugateGradient<
        Eigen::SparseMatrix<double>,
        Eigen::Lower | Eigen::Upper,
        preconditioner
    > cg;
    cg.setMaxIterations(opts_.max_iter);
    cg.setTolerance(opts_.tol);
    cg.preconditioner() = M;

    auto t0 = Clock::now();
    cg.compute(L_);

    Eigen::VectorXd rhs = b;
    rhs.array() -= rhs.mean();
    res.x = cg.solve(rhs);
    auto t1 = Clock::now();

    res.x.array() -= res.x.mean();
    res.solve_seconds = std::chrono::duration<double>(t1 - t0).count();
    res.iterations = static_cast<int>(cg.iterations());

    Eigen::VectorXd r = rhs - L_ * res.x;
    r.array() -= r.mean();
    double bnorm = rhs.norm();
    res.residual = (bnorm > 0) ? r.norm() / bnorm : r.norm();

    return res;
}

solve_result solve(const Eigen::SparseMatrix<double>& L,
                   const Eigen::VectorXd& b,
                   const solve_options& opts) {
    solver S(L, opts);
    return S.solve(b);
}

} // namespace apxchol
