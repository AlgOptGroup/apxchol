#include <cstdio>
#include <iostream>
#include <Eigen/Sparse>

#include "apxchol/graph/conversions.h"
#include "apxchol/graph/graph.h"
#include "apxchol/solver/factorization.h"
#include "apxchol/solver/solve.h"

static Eigen::SparseMatrix<double> grid_laplacian(int rows, int cols) {
    apxchol::graph<> G(rows * cols);
    auto id = [cols](int r, int c) { return r * cols + c; };
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            if (r + 1 < rows) G.add_edge(id(r, c), id(r + 1, c), 1.0);
            if (c + 1 < cols) G.add_edge(id(r, c), id(r, c + 1), 1.0);
        }
    return apxchol::laplacian(G);
}

int main() {
    std::printf("%-8s %8s %10s %10s %8s %8s %6s\n",
                "grid", "n", "nnz(L)", "fillin",
                "fact(ms)", "solve(ms)", "iters");
    std::printf("--------------------------------------------------------------\n");

    for (int side : {10, 20, 50, 100, 200, 500, 1000}) {
        int n = side * side;
        auto L = grid_laplacian(side, side);
        auto b = apxchol::generate_test_rhs(n);

        apxchol::solve_options opts;
        auto res = apxchol::solve(L, b, opts);

        // Factorize separately to get fill-in stats
        auto F = apxchol::factorize(L);
        double fillin = (L.nonZeros() > 0)
            ? static_cast<double>(F.L.nonZeros()) / L.nonZeros() : 0.0;

        char label[16];
        std::snprintf(label, sizeof(label), "%dx%d", side, side);
        std::printf("%-8s %8d %10lld %10.2f %8.1f %8.1f %6lld\n",
                    label, n, static_cast<long long>(F.L.nonZeros()), fillin,
                    res.timings.total("setup") * 1000,
                    res.timings.total("solve") * 1000,
                    static_cast<long long>(res.iterations));
        res.timings.report(std::cout);
    }
}
