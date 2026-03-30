#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "apxchol/graph/conversions.h"
#include "apxchol/graph/graph.h"

// ── Typed test fixture ───────────────────────────────

using AllStorages = ::testing::Types<
    apxchol::vec_incidence,
    apxchol::forward_star_incidence,
    apxchol::small_vec_incidence
>;

template<typename Incidence>
class GraphTest : public ::testing::Test {
protected:
    using Graph = apxchol::graph<Incidence>;

    static Graph make_path(int n) {
        Graph G(n);
        for (int i = 0; i + 1 < n; ++i)
            G.add_edge(i, i + 1, 1.0);
        return G;
    }

    static Graph make_grid(int rows, int cols) {
        Graph G(rows * cols);
        auto id = [cols](int r, int c) { return r * cols + c; };
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                if (r + 1 < rows) G.add_edge(id(r, c), id(r + 1, c), 1.0);
                if (c + 1 < cols) G.add_edge(id(r, c), id(r, c + 1), 1.0);
            }
        return G;
    }
};

TYPED_TEST_SUITE(GraphTest, AllStorages);

// ── Graph tests ──────────────────────────────────────

TYPED_TEST(GraphTest, BasicConstruction) {
    using Graph = typename TestFixture::Graph;
    Graph G(5);
    G.add_edge(0, 1, 2.0);
    G.add_edge(1, 2, 3.0);
    EXPECT_EQ(G.n(), 5);
    EXPECT_EQ(G.m(), 2);
}

TYPED_TEST(GraphTest, DeactivateAndMerge) {
    using Graph = typename TestFixture::Graph;
    Graph G(4);
    G.add_edge(0, 1, 1.0);
    G.add_edge(0, 2, 2.0);
    G.add_edge(1, 3, 3.0);
    G.add_edge(2, 3, 4.0);
    EXPECT_EQ(G.num_active(), 4);

    G.deactivate(0);
    EXPECT_FALSE(G.is_active(0));
    EXPECT_EQ(G.num_active(), 3);

    // Vertex 1's neighbor list still contains vertex 0 (deactivated but not pruned).
    // After prune_and_degree, only active neighbors remain.
    G.prune_and_degree(1);
    int count = 0;
    for (auto [target, w] : G.neighbors(1)) ++count;
    EXPECT_EQ(count, 1);  // only vertex 3

    // Add a parallel edge and merge
    G.add_edge(1, 3, 1.0);
    G.merge_parallel_edges(1);
    double total_w = 0;
    for (const auto& e : G.neighbors(1)) total_w += e.w;
    EXPECT_DOUBLE_EQ(total_w, 4.0);  // 3.0 + 1.0 merged
}

TYPED_TEST(GraphTest, ForNeighbors) {
    auto G = TestFixture::make_path(4);  // 0-1-2-3
    std::vector<std::pair<int, double>> nbrs;
    for (const auto& e : G.neighbors(1))
        nbrs.push_back({e.to, e.w});
    EXPECT_EQ(nbrs.size(), 2u);
}

TYPED_TEST(GraphTest, WeightedDegree) {
    using Graph = typename TestFixture::Graph;
    Graph G(3);
    G.add_edge(0, 1, 2.0);
    G.add_edge(0, 2, 5.0);
    auto wdeg = [&](apxchol::node_index v) {
        double d = 0;
        for (const auto& e : G.neighbors(v)) d += e.w;
        return d;
    };
    EXPECT_DOUBLE_EQ(wdeg(0), 7.0);
    EXPECT_DOUBLE_EQ(wdeg(1), 2.0);
}

// ── Laplacian tests (storage-independent) ────────────

static apxchol::graph<> make_path_vec(int n) {
    apxchol::graph<> G(n);
    for (int i = 0; i + 1 < n; ++i)
        G.add_edge(i, i + 1, 1.0);
    return G;
}

static apxchol::graph<> make_grid_vec(int rows, int cols) {
    apxchol::graph<> G(rows * cols);
    auto id = [cols](int r, int c) { return r * cols + c; };
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            if (r + 1 < rows) G.add_edge(id(r, c), id(r + 1, c), 1.0);
            if (c + 1 < cols) G.add_edge(id(r, c), id(r, c + 1), 1.0);
        }
    return G;
}

TEST(Laplacian, RowSumsZero) {
    auto G = make_grid_vec(10, 10);
    auto L = apxchol::laplacian(G);
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(L.rows());
    Eigen::VectorXd res = L * ones;
    EXPECT_LT(res.norm(), 1e-12);
}

TEST(Laplacian, Symmetric) {
    auto G = make_grid_vec(8, 8);
    auto L = apxchol::laplacian(G);
    Eigen::SparseMatrix<double> diff = L - Eigen::SparseMatrix<double>(L.transpose());
    EXPECT_LT(diff.norm(), 1e-12);
}

TEST(Laplacian, PositiveDiagonal) {
    auto G = make_grid_vec(5, 5);
    auto L = apxchol::laplacian(G);
    for (int k = 0; k < L.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
            if (it.row() == it.col()) {
                EXPECT_GE(it.value(), 0.0);
            }
        }
}

TEST(Laplacian, PathGraph) {
    auto G = make_path_vec(3);  // 0-1-2, weights 1.0
    auto L = apxchol::laplacian(G);
    // L should be [[1,-1,0],[-1,2,-1],[0,-1,1]]
    EXPECT_EQ(L.rows(), 3);
    EXPECT_NEAR(L.coeff(0, 0), 1.0, 1e-15);
    EXPECT_NEAR(L.coeff(1, 1), 2.0, 1e-15);
    EXPECT_NEAR(L.coeff(0, 1), -1.0, 1e-15);
    EXPECT_NEAR(L.coeff(0, 2), 0.0, 1e-15);
}
