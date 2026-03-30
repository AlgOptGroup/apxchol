#include "apxchol/solver/solve.h"
#include <Eigen/IterativeLinearSolvers>

namespace apxchol {

solve_result solve(const Eigen::SparseMatrix<double>& L,
                   const Eigen::VectorXd& b,
                   const solve_options& opts) {
    Eigen::ConjugateGradient<
        Eigen::SparseMatrix<double>, Eigen::Lower, apx_cholesky> cg;
    cg.setTolerance(opts.tol);
    cg.setMaxIterations(opts.max_iter);
    cg.preconditioner().set_options(opts.factor_opts);
    cg.preconditioner().set_storage(opts.storage);

    solve_result res;
    cg.preconditioner().set_checkpoint(&res.timings);

    cg.compute(L);
    res.x = cg.solve(b);

    res.iterations = cg.iterations();
    res.residual = cg.error();
    return res;
}

Eigen::VectorXd generate_test_rhs(Eigen::Index n) {
    Eigen::VectorXd b = Eigen::VectorXd::Random(n);
    b.array() -= b.mean();
    return b.normalized();
}

} // namespace apxchol
