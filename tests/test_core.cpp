#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "apxchol/graph.h"
#include "apxchol/adj_list_graph.h"

// ── Helpers ──────────────────────────────────────────

static apxchol::adj_list_graph make_path(int n) {
    apxchol::adj_list_graph G(n);
    for (int i = 0; i + 1 < n; ++i)
        G.add_edge(i, i + 1, 1.0);
    return G;
}

static apxchol::adj_list_graph make_grid(int rows, int cols) {
    apxchol::adj_list_graph G(rows * cols);
    auto id = [cols](int r, int c) { return r * cols + c; };
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            if (r + 1 < rows) G.add_edge(id(r, c), id(r + 1, c), 1.0);
            if (c + 1 < cols) G.add_edge(id(r, c), id(r, c + 1), 1.0);
        }
    return G;
}

// ── Graph tests ──────────────────────────────────────

TEST(AdjListGraph, BasicConstruction) {
    apxchol::adj_list_graph G(5);
    G.add_edge(0, 1, 2.0);
    G.add_edge(1, 2, 3.0);
    EXPECT_EQ(G.n(), 5);
    EXPECT_EQ(G.m(), 2);  // 2 undirected edges
}

TEST(AdjListGraph, TraverseXor) {
    apxchol::adj_list_graph G(3);
    G.add_edge(0, 2, 1.0);
    auto& e = G.get_edge(0);
    EXPECT_EQ(e.traverse(0), 2);
    EXPECT_EQ(e.traverse(2), 0);
}

TEST(AdjListGraph, ForNeighbors) {
    auto G = make_path(4);  // 0-1-2-3
    std::vector<std::pair<int, double>> nbrs;
    G.for_neighbors(1, [&](int u, double w) { nbrs.push_back({u, w}); });
    EXPECT_EQ(nbrs.size(), 2u);
}

TEST(AdjListGraph, WeightedDegree) {
    apxchol::adj_list_graph G(3);
    G.add_edge(0, 1, 2.0);
    G.add_edge(0, 2, 5.0);
    EXPECT_DOUBLE_EQ(G.weighted_degree(0), 7.0);
    EXPECT_DOUBLE_EQ(G.weighted_degree(1), 2.0);
}

// ── Laplacian tests ──────────────────────────────────

TEST(Laplacian, RowSumsZero) {
    auto G = make_grid(10, 10);
    auto L = G.laplacian();
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(L.rows());
    Eigen::VectorXd res = L * ones;
    EXPECT_LT(res.norm(), 1e-12);
}

TEST(Laplacian, Symmetric) {
    auto G = make_grid(8, 8);
    auto L = G.laplacian();
    Eigen::SparseMatrix<double> diff = L - Eigen::SparseMatrix<double>(L.transpose());
    EXPECT_LT(diff.norm(), 1e-12);
}

TEST(Laplacian, PositiveDiagonal) {
    auto G = make_grid(5, 5);
    auto L = G.laplacian();
    for (int k = 0; k < L.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
            if (it.row() == it.col())
                EXPECT_GE(it.value(), 0.0);
        }
}

TEST(Laplacian, PathGraph) {
    auto G = make_path(3);  // 0-1-2, weights 1.0
    auto L = G.laplacian();
    // L should be [[1,-1,0],[-1,2,-1],[0,-1,1]]
    EXPECT_EQ(L.rows(), 3);
    EXPECT_NEAR(L.coeff(0, 0), 1.0, 1e-15);
    EXPECT_NEAR(L.coeff(1, 1), 2.0, 1e-15);
    EXPECT_NEAR(L.coeff(0, 1), -1.0, 1e-15);
    EXPECT_NEAR(L.coeff(0, 2), 0.0, 1e-15);
}
