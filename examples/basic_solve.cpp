/// Basic usage: the one-shot solve() and the reusable cpu_solver.
#include "apxchol.h"
#include "example_grid.h"
#include <cstdio>

int main() {
    auto L = grid_laplacian(100);                       // 10,000 × 10,000
    Eigen::VectorXd b = apxchol::generate_test_rhs(L.rows());

    // One-shot: factorize + PCG in one call.
    auto res = apxchol::solve(L, b, {.tol = 1e-8, .max_iter = 500});
    std::printf("one-shot : %ld iterations, residual %.2e\n",
                long(res.iterations), res.residual);

    // Reusable: factor once, then solve any number of right-hand sides.
    apxchol::cpu_solver slv(L);
    auto r1 = slv.solve(b);                             // opts defaults
    auto r2 = slv.solve(b, /*tol=*/1e-10, /*max_iter=*/1000);
    std::printf("reusable : %ld iterations @1e-8, %ld @1e-10\n",
                long(r1.iterations), long(r2.iterations));

    // One preconditioner application z = M^{-1} r (e.g. for your own Krylov loop).
    Eigen::VectorXd z = slv.apply(b);
    std::printf("apply    : |M^-1 b| = %.3e\n", z.norm());

    return (res.residual < 1e-8 && r2.residual < 1e-10) ? 0 : 1;
}
