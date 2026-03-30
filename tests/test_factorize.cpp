#include <gtest/gtest.h>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include "apxchol/solver/factorization.h"
#include "apxchol/solver/preconditioner.h"
#include "apxchol/graph/conversions.h"
#include "apxchol/graph/graph.h"
#include "apxchol/solver/solve.h"

// ── Helpers ──────────────────────────────────────────

static Eigen::SparseMatrix<double> path_laplacian(int n) {
    apxchol::graph<> G(n);
    for (int i = 0; i + 1 < n; ++i)
        G.add_edge(i, i + 1, 1.0);
    return apxchol::laplacian(G);
}

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

static Eigen::SparseMatrix<double> star_laplacian(int n) {
    apxchol::graph<> G(n);
    for (int i = 1; i < n; ++i)
        G.add_edge(0, i, 1.0);
    return apxchol::laplacian(G);
}

static Eigen::SparseMatrix<double> cycle_laplacian(int n) {
    apxchol::graph<> G(n);
    for (int i = 0; i < n; ++i)
        G.add_edge(i, (i + 1) % n, 1.0);
    return apxchol::laplacian(G);
}

static Eigen::SparseMatrix<double> weighted_grid_laplacian(int rows, int cols,
                                                           double w_horiz,
                                                           double w_vert) {
    apxchol::graph<> G(rows * cols);
    auto id = [cols](int r, int c) { return r * cols + c; };
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            if (r + 1 < rows) G.add_edge(id(r, c), id(r + 1, c), w_vert);
            if (c + 1 < cols) G.add_edge(id(r, c), id(r, c + 1), w_horiz);
        }
    return apxchol::laplacian(G);
}

// ── Typed test infrastructure ────────────────────────

using AllStorages = ::testing::Types<
    apxchol::vec_incidence,
    apxchol::forward_star_incidence,
    apxchol::small_vec_incidence
>;

// ── Factorization structure tests ────────────────────

template<typename Incidence>
class FactorizeTest : public ::testing::Test {
protected:
    apxchol::factorization factorize_with(
        const Eigen::SparseMatrix<double>& L, unsigned seed = 42) {
        return apxchol::factorize(L, Incidence::tag, {.seed = seed});
    }
};

TYPED_TEST_SUITE(FactorizeTest, AllStorages);

TYPED_TEST(FactorizeTest, PathGraphDimensions) {
    auto L = path_laplacian(10);
    auto F = this->factorize_with(L);
    EXPECT_EQ(F.L.rows(), 10);
    EXPECT_EQ(F.L.cols(), 10);
    EXPECT_EQ(F.perm.size(), 10);
}

TYPED_TEST(FactorizeTest, GridGraphDimensions) {
    auto L = grid_laplacian(5, 5);
    auto F = this->factorize_with(L);
    EXPECT_EQ(F.L.rows(), 25);
    EXPECT_EQ(F.L.cols(), 25);
}

TYPED_TEST(FactorizeTest, PermutationIsValid) {
    auto L = grid_laplacian(4, 4);
    auto F = this->factorize_with(L);

    const int n = F.L.rows();
    std::vector<int> seen(n, 0);
    for (int i = 0; i < n; ++i) {
        int idx = F.perm.indices()(i);
        ASSERT_GE(idx, 0);
        ASSERT_LT(idx, n);
        seen[idx]++;
    }
    for (int i = 0; i < n; ++i)
        EXPECT_EQ(seen[i], 1) << "index " << i << " appears " << seen[i] << " times";
}

TYPED_TEST(FactorizeTest, LowerTriangular) {
    auto L = grid_laplacian(5, 5);
    auto F = this->factorize_with(L);

    const int m = F.L.rows() - 1;
    for (int k = 0; k < F.L.outerSize() && k < m; ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(F.L, k); it; ++it)
            if (it.row() < m && it.col() < m) {
                EXPECT_GE(it.row(), it.col())
                    << "upper-triangle entry at (" << it.row() << "," << it.col() << ")";
            }
}

TYPED_TEST(FactorizeTest, PositiveDiagonal) {
    auto L = grid_laplacian(5, 5);
    auto F = this->factorize_with(L);

    const int m = F.L.rows() - 1;
    for (int i = 0; i < m; ++i)
        EXPECT_GT(F.L.coeff(i, i), 0.0) << "zero/negative diagonal at " << i;
}

TYPED_TEST(FactorizeTest, Deterministic) {
    auto L = grid_laplacian(5, 5);
    auto F1 = this->factorize_with(L, 42);
    auto F2 = this->factorize_with(L, 42);

    EXPECT_EQ(F1.L.nonZeros(), F2.L.nonZeros());
    EXPECT_EQ(F1.perm.indices(), F2.perm.indices());
    Eigen::SparseMatrix<double> diff = F1.L - F2.L;
    EXPECT_LT(diff.norm(), 1e-14);
}

TYPED_TEST(FactorizeTest, DifferentSeeds) {
    auto L = grid_laplacian(8, 8);
    auto F1 = this->factorize_with(L, 1);
    auto F2 = this->factorize_with(L, 999);
    EXPECT_EQ(F1.L.rows(), F2.L.rows());
}

TYPED_TEST(FactorizeTest, SmallPath) {
    auto L = path_laplacian(3);
    auto F = this->factorize_with(L);
    EXPECT_EQ(F.L.rows(), 3);
    EXPECT_GT(F.L.nonZeros(), 0);
}

TYPED_TEST(FactorizeTest, StarGraph) {
    auto L = star_laplacian(10);
    auto F = this->factorize_with(L);
    EXPECT_EQ(F.L.rows(), 10);
}

TYPED_TEST(FactorizeTest, CycleGraph) {
    auto L = cycle_laplacian(20);
    auto F = this->factorize_with(L);
    EXPECT_EQ(F.L.rows(), 20);
}

TYPED_TEST(FactorizeTest, TwoVertices) {
    auto L = path_laplacian(2);
    auto F = this->factorize_with(L);
    EXPECT_EQ(F.L.rows(), 2);
}

TYPED_TEST(FactorizeTest, NnzPositive) {
    auto L = grid_laplacian(10, 10);
    auto F = this->factorize_with(L);
    EXPECT_GT(F.L.nonZeros(), 0);
}

// ── Preconditioner quality tests (PCG convergence) ───

template<typename Incidence>
class SolveTest : public ::testing::Test {
protected:
    apxchol::solve_result solve_with(
        const Eigen::SparseMatrix<double>& L,
        const Eigen::VectorXd& b,
        double tol = 1e-6, int max_iter = 500) {
        return apxchol::solve(L, b,
            {.tol = tol, .max_iter = max_iter,
             .storage = Incidence::tag,
             .factor_opts = {.seed = 42}});
    }
};

TYPED_TEST_SUITE(SolveTest, AllStorages);

TYPED_TEST(SolveTest, PathGraphConverges) {
    auto L = path_laplacian(50);
    auto b = apxchol::generate_test_rhs(50);
    auto res = this->solve_with(L, b);
    EXPECT_LT(res.residual, 1e-5);
}

TYPED_TEST(SolveTest, GridGraphConverges) {
    auto L = grid_laplacian(10, 10);
    auto b = apxchol::generate_test_rhs(100);
    auto res = this->solve_with(L, b);
    EXPECT_LT(res.residual, 1e-5);
}

TYPED_TEST(SolveTest, StarGraphConverges) {
    auto L = star_laplacian(30);
    auto b = apxchol::generate_test_rhs(30);
    auto res = this->solve_with(L, b);
    EXPECT_LT(res.residual, 1e-5);
}

TYPED_TEST(SolveTest, CycleGraphConverges) {
    auto L = cycle_laplacian(50);
    auto b = apxchol::generate_test_rhs(50);
    auto res = this->solve_with(L, b);
    EXPECT_LT(res.residual, 1e-5);
}

TYPED_TEST(SolveTest, WeightedGridConverges) {
    auto L = weighted_grid_laplacian(10, 10, 100.0, 1.0);
    auto b = apxchol::generate_test_rhs(100);
    auto res = this->solve_with(L, b, 1e-6, 1000);
    EXPECT_LT(res.residual, 1e-4);
}

TYPED_TEST(SolveTest, LargerGridConverges) {
    auto L = grid_laplacian(20, 20);
    auto b = apxchol::generate_test_rhs(400);
    auto res = this->solve_with(L, b, 1e-6, 1000);
    EXPECT_LT(res.residual, 1e-5);
}

TYPED_TEST(SolveTest, SolutionSatisfiesSystem) {
    auto L = grid_laplacian(10, 10);
    auto b = apxchol::generate_test_rhs(100);
    auto res = this->solve_with(L, b, 1e-8);

    Eigen::VectorXd Lx = L * res.x;
    Lx.array() -= Lx.mean();
    Eigen::VectorXd bc = b;
    bc.array() -= bc.mean();
    double rel_err = (Lx - bc).norm() / bc.norm();
    EXPECT_LT(rel_err, 1e-5);
}

TYPED_TEST(SolveTest, DeterministicWithSameSeed) {
    auto L = grid_laplacian(8, 8);
    auto b = apxchol::generate_test_rhs(64);

    auto r1 = this->solve_with(L, b);
    auto r2 = this->solve_with(L, b);
    EXPECT_EQ(r1.iterations, r2.iterations);
#ifdef APXCHOL_USE_CUDA
    EXPECT_NEAR(r1.residual, r2.residual, 1e-12);
#else
    EXPECT_DOUBLE_EQ(r1.residual, r2.residual);
#endif
}

// ── Performance / timing sanity (storage-independent) ─

TEST(Solve, TimingsReported) {
    auto L = grid_laplacian(10, 10);
    auto b = apxchol::generate_test_rhs(100);
    auto res = apxchol::solve(L, b);
    EXPECT_GT(res.timings.total("setup"), 0.0);
    EXPECT_GT(res.timings.total("solve"), 0.0);
}
