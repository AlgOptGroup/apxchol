#include "apxchol/solver/solve.h"
#include <cmath>

namespace apxchol {

namespace {
// Eigen's parallel SpMV requires Eigen::initParallel() to be called
// once per process before any threaded operation.  Calling it from a
// function-local static keeps it lazy and thread-safe.
inline void ensure_eigen_parallel() {
    static const bool dummy = []{ Eigen::initParallel(); return true; }();
    (void)dummy;
}
} // namespace

solve_result solve(const Eigen::SparseMatrix<double>& L,
                   const Eigen::VectorXd& b,
                   const solve_options& opts) {
    ensure_eigen_parallel();
    // Build preconditioner.
    apx_cholesky precond;
    solve_result res;
    precond.set_options(opts.factor_opts);
    precond.set_storage(opts.storage);
    precond.set_checkpoint(&res.timings);
    precond.compute(L);

    // Preconditioned CG with stagnation detection.
    const Eigen::Index n = L.rows();
    const double bnorm = b.norm();
    if (bnorm == 0.0) { res.x = Eigen::VectorXd::Zero(n); return res; }

    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd r = b;                         // r = b - L*0 = b
    Eigen::VectorXd z(n), p(n), Ap(n);

    z = precond.solve(r);
    p = z;
    double rz = r.dot(z);

    double prev_check_residual = 1.0;
    const int check_interval = opts.stagnation_window;

    for (int i = 0; i < opts.max_iter; ++i) {
        Ap.noalias() = L.selfadjointView<Eigen::Lower>() * p;
        double pAp = p.dot(Ap);
        if (pAp <= 0.0) break;
        double alpha = rz / pAp;

        x += alpha * p;
        r -= alpha * Ap;

        double rnorm = r.norm() / bnorm;
        res.iterations = i + 1;
        res.residual = rnorm;

        if (rnorm < opts.tol) break;

        // Stagnation detection: if residual hasn't improved sufficiently
        // over the last check_interval iterations, stop early.
        if (check_interval > 0 && (i + 1) % check_interval == 0) {
            if (rnorm > prev_check_residual * 0.5) break;
            prev_check_residual = rnorm;
        }

        z = precond.solve(r);
        double rz_new = r.dot(z);
        double beta = rz_new / rz;
        p = z + beta * p;
        rz = rz_new;
    }

    res.x = std::move(x);
    return res;
}

Eigen::VectorXd generate_test_rhs(Eigen::Index n) {
    Eigen::VectorXd b = Eigen::VectorXd::Random(n);
    b.array() -= b.mean();
    return b.normalized();
}

} // namespace apxchol
