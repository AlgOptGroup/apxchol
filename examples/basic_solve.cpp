/// Basic usage: one-shot, reusable and batched solves.
#include "apxchol.h"
#include "example_grid.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    const auto grid = grid_laplacian(100);              // 10,000 × 10,000
    const apxchol::csc_view L = grid.view();
    const auto n = static_cast<std::size_t>(L.n);
    std::vector<double> b(n, 0.0);                      // unit source and sink
    b.front() = 1.0;
    b.back() = -1.0;

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
    // Column-major storage: column j starts at j * n.
    std::vector<double> B(2 * n), X(2 * n);
    std::copy(b.begin(), b.end(), B.begin());
    std::reverse_copy(b.begin(), b.end(), B.begin() + n);
    const auto rows = static_cast<std::int64_t>(n);
    auto batch = slv.solve(apxchol::const_matrix_view{B.data(), rows, 2, rows},
                           apxchol::matrix_view{X.data(), rows, 2, rows});
    std::printf("batched  : residuals %.2e, %.2e\n",
                batch.columns[0].residual, batch.columns[1].residual);

    // One preconditioner application z = M^{-1} r (e.g. for your own Krylov loop).
    const std::vector<double> z = slv.apply(b);
    double norm = 0.0;
    for (double v : z) norm += v * v;
    std::printf("apply    : |M^-1 b| = %.3e\n", std::sqrt(norm));

    bool converged = res.converged && r1.converged && r2.converged;
    for (const auto& column : batch.columns)
        converged = converged && column.converged;
    return converged ? 0 : 1;
}
