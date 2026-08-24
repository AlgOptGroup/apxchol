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
    apxchol::bstr_incidence,
    apxchol::vec_pool_incidence,
    apxchol::directed_vec_pool_incidence
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

TEST(ActiveState, ParallelClearsSharingWordsDoNotLoseUpdates) {
    constexpr apxchol::node_index n = 4096;
    apxchol::graph<apxchol::directed_vec_pool_incidence> G(n);

    #pragma omp parallel for schedule(static, 1)
    for (apxchol::node_index v = 0; v < n; ++v)
        G.set_inactive_unchecked(v);
    G.bulk_decrement_active(n);

    EXPECT_EQ(G.num_active(), 0u);
    for (apxchol::node_index v = 0; v < n; ++v)
        EXPECT_FALSE(G.is_active(v));
}

TEST(ActiveState, RepeatedDeactivateDoesNotDoubleCount) {
    apxchol::graph<apxchol::directed_vec_pool_incidence> G(65);
    G.deactivate(64);
    G.deactivate(64);
    EXPECT_EQ(G.num_active(), 64u);
    EXPECT_FALSE(G.is_active(64));
    EXPECT_TRUE(G.is_active(63));
}

TYPED_TEST(GraphTest, BasicConstruction) {
    using Graph = typename TestFixture::Graph;
    Graph G(5);
    G.add_edge(0, 1, 2.0);
    G.add_edge(1, 2, 3.0);
    EXPECT_EQ(G.n(), 5);
    EXPECT_EQ(G.m(), 2);
}

TYPED_TEST(GraphTest, DeactivateAndPrune) {
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
    for (auto _ : G.neighbors(1)) ++count;
    EXPECT_EQ(count, 1);  // only vertex 3
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

TEST(VecPoolGraph, CoalescePreservesWeightsMultiplicityAndActiveState) {
    using Graph = apxchol::graph<apxchol::vec_pool_incidence>;
    Graph G(6);
    G.add_edge(0, 1, 1.0);
    G.add_edge(0, 1, 2.0);
    G.add_edge(0, 2, 4.0);
    G.add_edge(1, 2, 3.0);
    G.add_edge(1, 2, 5.0);
    G.add_edge(1, 2, 7.0);
    G.add_edge(2, 3, 1.0);
    G.add_edge(3, 4, 9.0);  // dropped: vertex 4 is inactive
    G.add_edge(0, 5, 8.0);  // dropped: vertex 5 is inactive
    G.excess(0) = 0.25;
    G.excess(2) = 0.75;
    G.deactivate(4);
    G.deactivate(5);

    const std::vector<apxchol::node_index> active = {0, 1, 2, 3};
    EXPECT_EQ(G.prune_and_degree(0), 3u);
    EXPECT_EQ(G.prune_and_degree(1), 5u);
    EXPECT_EQ(G.prune_and_degree(2), 5u);
    EXPECT_EQ(G.prune_and_degree(3), 1u);
    EXPECT_DOUBLE_EQ(
        apxchol::detail::residual_coalescer<apxchol::vec_pool_incidence>::
            estimate(G, active, active.size()),
        14.0 / 8.0);

    const auto stats =
        apxchol::detail::residual_coalescer<apxchol::vec_pool_incidence>::
            rebuild(G, active);
    EXPECT_EQ(stats.multi_edges, 7u);
    EXPECT_EQ(stats.distinct_edges, 4u);
    EXPECT_EQ(G.m(), 4u);
    EXPECT_EQ(G.num_active(), 4u);
    EXPECT_FALSE(G.is_active(4));
    EXPECT_FALSE(G.is_active(5));
    EXPECT_DOUBLE_EQ(G.excess(0), 0.25);
    EXPECT_DOUBLE_EQ(G.excess(2), 0.75);

    // Physical adjacency is simple, but the partitioner's degree remains the
    // old multigraph degree through the multiplicity sidecar.
    EXPECT_EQ(G.adj(0).size(), 2u);
    EXPECT_EQ(G.adj(1).size(), 2u);
    EXPECT_EQ(G.prune_and_degree(0), 3u);
    EXPECT_EQ(G.prune_and_degree(1), 5u);
    EXPECT_EQ(G.prune_and_degree(2), 5u);
    EXPECT_EQ(G.prune_and_degree(3), 1u);

    auto find_edge = [&](apxchol::node_index u, apxchol::node_index v) {
        for (const auto idx : G.adj(u)) {
            if (G.edge_target(idx, u) == v)
                return std::pair{G.edge_weight(idx),
                                 G.edge_multiplicity(idx)};
        }
        return std::pair{0.0, apxchol::node_index{0}};
    };
    EXPECT_EQ(find_edge(0, 1), (std::pair{3.0, apxchol::node_index{2}}));
    EXPECT_EQ(find_edge(1, 2), (std::pair{15.0, apxchol::node_index{3}}));
    EXPECT_DOUBLE_EQ(
        apxchol::detail::residual_coalescer<apxchol::vec_pool_incidence>::
            estimate(G, active, active.size()),
        14.0 / 8.0);

    // Future sampled edges append with multiplicity one; the sidecar remains
    // aligned with the edge pool after reserve/write and adjacency insertion.
    G.adj_reserve_for(0, G.adj_count(0) + 1);
    G.adj_reserve_for(3, G.adj_count(3) + 1);
    const auto slot = G.reserve_edge_pool(1);
    G.write_edge_at(slot, 0, 3, 6.0);
    G.adj_atomic_push_reserved(0, slot);
    G.adj_atomic_push_reserved(3, slot);
    EXPECT_EQ(G.edge_multiplicity(slot), 1u);
    EXPECT_EQ(G.prune_and_degree(0), 4u);
    EXPECT_EQ(G.prune_and_degree(3), 2u);
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
