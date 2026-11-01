/// Basic usage: one-shot, reusable and batched solves.
#include "apxchol.h"
#include "example_grid.h"
#include <cstdio>

int main() {
    auto L = grid_laplacian(100);                       // 10,000 × 10,000
    Eigen::VectorXd b = Eigen::VectorXd::Zero(L.rows());   // unit source and sink
    b(0) = 1.0;
    b(L.rows() - 1) = -1.0;

    // One-shot: factorize + PCG in one call.
    auto res = apxchol::solve(L, b, /*tol=*/1e-8, /*max_iter=*/500);
    std::printf("one-shot : %ld iterations, residual %.2e\n",
                long(res.iterations), res.residual);

    // Reusable: factor once, then solve any number of right-hand sides.
    apxchol::solver slv(L);
    auto r1 = slv.solve(b);                             // default tol and max_iter
    auto r2 = slv.solve(b, /*tol=*/1e-10, /*max_iter=*/1000);
    std::printf("reusable : %ld iterations @1e-8, %ld @1e-10\n",
                long(r1.iterations), long(r2.iterations));

    // Matrix columns are independent right-hand sides sharing the same factor.
    Eigen::MatrixXd B(L.rows(), 2), X(L.rows(), 2);
    B.col(0) = b;
    B.col(1) = b.reverse();
    auto batch = slv.solve(B, X);
    std::printf("batched  : residuals %.2e, %.2e\n",
                batch.columns[0].residual, batch.columns[1].residual);

    // One preconditioner application z = M^{-1} r (e.g. for your own Krylov loop).
    Eigen::VectorXd z = slv.apply(b);
    std::printf("apply    : |M^-1 b| = %.3e\n", z.norm());

    bool converged = res.converged && r1.converged && r2.converged;
    for (const auto& column : batch.columns)
        converged = converged && column.converged;
    return converged ? 0 : 1;
}
