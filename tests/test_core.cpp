#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include "graphs.h"
#include "laplacian_preconditioner.h"

// ── Helpers ──────────────────────────────────────────

static Eigen::SparseMatrix<double>
build_laplacian(const std::vector<std::vector<Edge>>& adj) {
    using T = Eigen::Triplet<double>;
    int n = static_cast<int>(adj.size());
    std::vector<T> trips;
    for (int i = 0; i < n; ++i) {
        double deg = 0;
        for (auto& e : adj[i]) {
            deg += e.w;
            trips.emplace_back(i, e.to, -e.w / 2);
            trips.emplace_back(e.to, i, -e.w / 2);
        }
        trips.emplace_back(i, i, deg);
    }
    Eigen::SparseMatrix<double> L(n, n);
    L.setFromTriplets(trips.begin(), trips.end());
    return L;
}

// ── Trivial preconditioner for testing template ──────

class diagonal_preconditioner
    : public laplacian_preconditioner<diagonal_preconditioner> {
public:
    diagonal_preconditioner() = default;

    explicit diagonal_preconditioner(const Eigen::SparseMatrix<double>& L)
        : diag_inv_(L.rows()) {
        for (int k = 0; k < L.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
                if (it.row() == it.col())
                    diag_inv_[k] = (it.value() > 0) ? (1.0 / it.value()) : 1.0;
        n_ = L.rows();
    }

    Eigen::VectorXd apply(const Eigen::VectorXd& rhs) const {
        return diag_inv_.asDiagonal() * rhs;
    }

private:
    Eigen::VectorXd diag_inv_;
};

// ── Graph generator tests ────────────────────────────

TEST(Graphs, GridGraphSymmetric) {
    auto adj = grid_graph(10, 10);
    EXPECT_EQ(adj.size(), 100u);
    for (int u = 0; u < (int)adj.size(); ++u)
        for (auto& e : adj[u]) {
            bool found = false;
            for (auto& re : adj[e.to])
                if (re.to == u) { found = true; break; }
            EXPECT_TRUE(found) << "Missing reverse edge " << e.to << " -> " << u;
        }
}

TEST(Graphs, CheckerboardWeights) {
    auto adj = grid_graph_checkerboard(4, 4, 100.0, 1.0, 2);
    for (auto& row : adj)
        for (auto& e : row)
            EXPECT_GT(e.w, 0.0);
}

TEST(Graphs, ErdosRenyiBasic) {
    auto adj = erdos_renyi_graph(100, 0.1, 42);
    EXPECT_EQ(adj.size(), 100u);
    int total_edges = 0;
    for (auto& row : adj) total_edges += row.size();
    EXPECT_GT(total_edges, 0);
}

TEST(Graphs, ErdosRenyiEdgeCount) {
    // E[m] = n*(n-1)/2 * p; for n=1000, p=0.01 => ~5000 edges
    auto adj = erdos_renyi_graph(1000, 0.01, 99);
    int total = 0;
    for (auto& row : adj) total += row.size();
    int m = total / 2;
    EXPECT_GT(m, 2000);
    EXPECT_LT(m, 8000);
}

// ── Laplacian matrix properties ──────────────────────

TEST(Laplacian, RowSumsZero) {
    auto adj = grid_graph(10, 10);
    auto L = build_laplacian(adj);
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(L.rows());
    Eigen::VectorXd result = L * ones;
    EXPECT_LT(result.norm(), 1e-12);
}

TEST(Laplacian, Symmetric) {
    auto adj = grid_graph_checkerboard(10, 10, 100.0, 1.0, 2);
    auto L = build_laplacian(adj);
    Eigen::SparseMatrix<double> diff = L - Eigen::SparseMatrix<double>(L.transpose());
    EXPECT_LT(diff.norm(), 1e-12);
}

TEST(Laplacian, PositiveDiagonal) {
    auto adj = grid_graph(10, 10);
    auto L = build_laplacian(adj);
    for (int k = 0; k < L.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
            if (it.row() == it.col())
                EXPECT_GE(it.value(), 0.0);
}

// ── Preconditioner template tests ────────────────────

TEST(Preconditioner, TemplateCompiles) {
    auto adj = grid_graph(10, 10);
    auto L = build_laplacian(adj);
    diagonal_preconditioner P(L);
    Eigen::VectorXd b = Eigen::VectorXd::Random(L.rows());
    b.array() -= b.mean();
    Eigen::VectorXd x = P.solve(b);
    EXPECT_EQ(x.size(), b.size());
    // Output should be zero-mean
    EXPECT_NEAR(x.mean(), 0.0, 1e-12);
}

TEST(Preconditioner, DiagonalPCGConverges) {
    auto adj = grid_graph(20, 20);
    auto L = build_laplacian(adj);
    int n = L.rows();

    // Generate RHS from known solution
    std::mt19937_64 rng(42);
    std::normal_distribution<double> N(0, 1);
    Eigen::VectorXd g(n);
    for (int i = 0; i < n; ++i) g[i] = N(rng);
    Eigen::VectorXd b = L * g;
    b.array() -= b.mean();
    b /= b.norm();

    // Use template-based preconditioner with Eigen's PCG
    diagonal_preconditioner P(L);
    Eigen::ConjugateGradient<
        Eigen::SparseMatrix<double>,
        Eigen::Lower | Eigen::Upper,
        diagonal_preconditioner> cg;
    cg.setMaxIterations(500);
    cg.setTolerance(1e-8);
    cg.preconditioner() = P;
    cg.compute(L);
    Eigen::VectorXd x = cg.solve(b);

    x.array() -= x.mean();
    Eigen::VectorXd r = b - L * x;
    r.array() -= r.mean();
    double rel_res = r.norm() / b.norm();

    EXPECT_LT(rel_res, 1e-6);
    EXPECT_LT(cg.iterations(), 400);
}
