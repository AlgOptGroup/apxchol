#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>
#include <Eigen/Sparse>

#include "apxchol/graph/conversions.h"
#include "apxchol/graph/graph.h"
#include "apxchol/solver/factorization.h"

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
    for (int side : {100, 200, 500, 1000}) {
        int n = side * side;
        auto L = grid_laplacian(side, side);
        auto F = apxchol::factorize(L);

        int m = n - 1;
        Eigen::SparseMatrix<double> L11 = F.L.topLeftCorner(m, m);
        L11.makeCompressed();

        // Compute depth of each row in the dependency DAG of L (lower triangular).
        // depth[i] = max(depth[j] for j in L's non-zero entries in row i below diagonal) + 1
        std::vector<int> depth(m, 0);
        for (int j = 0; j < m; ++j) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(L11, j); it; ++it) {
                int i = it.row();
                if (i > j)
                    depth[i] = std::max(depth[i], depth[j] + 1);
            }
        }

        int num_levels = *std::max_element(depth.begin(), depth.end()) + 1;

        // Count rows per level
        std::vector<int> level_size(num_levels, 0);
        for (int i = 0; i < m; ++i)
            level_size[depth[i]]++;

        // NNZ per level (for work estimation)
        std::vector<int> level_nnz(num_levels, 0);
        for (int j = 0; j < m; ++j) {
            int nnz_col = 0;
            for (Eigen::SparseMatrix<double>::InnerIterator it(L11, j); it; ++it)
                nnz_col++;
            level_nnz[depth[j]] += nnz_col;
        }

        std::printf("\n=== %dx%d grid (n=%d, m=%d, nnz(L11)=%lld) ===\n",
                    side, side, n, m, (long long)L11.nonZeros());
        std::printf("  Levels: %d\n", num_levels);
        std::printf("  %-8s %10s %10s %8s\n", "Level", "Rows", "NNZ", "Avg NNZ");
        std::printf("  -------------------------------------------\n");

        int shown = 0;
        for (int l = 0; l < num_levels && shown < 15; ++l) {
            if (level_size[l] > 0) {
                std::printf("  %-8d %10d %10d %8.1f\n",
                            l, level_size[l], level_nnz[l],
                            (double)level_nnz[l] / level_size[l]);
                shown++;
            }
        }
        if (num_levels > 15) {
            std::printf("  ... (%d more levels)\n", num_levels - 15);
            // Show last 3 levels
            for (int l = std::max(15, num_levels - 3); l < num_levels; ++l)
                std::printf("  %-8d %10d %10d %8.1f\n",
                            l, level_size[l], level_nnz[l],
                            (double)level_nnz[l] / level_size[l]);
        }

        // Summary stats
        int max_level_size = *std::max_element(level_size.begin(), level_size.end());
        double avg_level_size = (double)m / num_levels;
        int single_row_levels = std::count(level_size.begin(), level_size.end(), 1);

        std::printf("  Max level size: %d (%.1f%% of m)\n",
                    max_level_size, 100.0 * max_level_size / m);
        std::printf("  Avg level size: %.1f\n", avg_level_size);
        std::printf("  Single-row levels: %d (%.1f%%)\n",
                    single_row_levels, 100.0 * single_row_levels / num_levels);
    }
    return 0;
}
