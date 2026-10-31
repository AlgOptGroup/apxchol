#pragma once
#include "apxchol/graph/conversions.h"
#include "csc_test_util.h"

namespace apxchol::test {

inline Eigen::SparseMatrix<double> grid_laplacian(int rows, int cols) {
    apxchol::detail::graph G(rows * cols);
    auto id = [cols](int r, int c) { return r * cols + c; };
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            if (r + 1 < rows) G.add_edge(id(r, c), id(r + 1, c), 1.0);
            if (c + 1 < cols) G.add_edge(id(r, c), id(r, c + 1), 1.0);
        }
    return to_eigen(apxchol::detail::laplacian(G).ref());
}

} // namespace apxchol::test
