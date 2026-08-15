#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "apxchol/solver/factorization.h"
#include "apxchol/solver/preconditioner.h"
#include "apxchol/graph/conversions.h"
#include "apxchol/graph/graph.h"
#include "apxchol/solver/solve.h"

// ── Helpers ──────────────────────────────────────────

// Convert the owned sparse_csc factor to an Eigen::SparseMatrix for tests
// (the library no longer stores Eigen factors).
static Eigen::SparseMatrix<double> factor_to_eigen(const apxchol::sparse_csc& L) {
    const auto* outer = L.outerIndexPtr();
    const auto* inner = L.innerIndexPtr();
    const auto* vals  = L.valuePtr();
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<size_t>(L.nonZeros()));
    for (apxchol::node_index c = 0; c < L.cols(); ++c)
        for (apxchol::edge_index p = outer[c]; p < outer[c + 1]; ++p)
            trips.emplace_back(static_cast<int>(inner[p]), static_cast<int>(c), vals[p]);
    Eigen::SparseMatrix<double> M(static_cast<int>(L.rows()), static_cast<int>(L.cols()));
    M.setFromTriplets(trips.begin(), trips.end());
    return M;
}

// Build an Eigen permutation from perm[v] = new position of original vertex v.
static Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>
perm_to_eigen(const std::vector<apxchol::node_index>& perm) {
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P(
        static_cast<Eigen::Index>(perm.size()));
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(perm.size()); ++i)
        P.indices()[i] = static_cast<int>(perm[i]);
    return P;
}

// L(i,i): scan column i of the CSC factor for the diagonal entry.
static double factor_diag(const apxchol::sparse_csc& L, int i) {
    const auto* outer = L.outerIndexPtr();
    const auto* inner = L.innerIndexPtr();
    const auto* vals  = L.valuePtr();
    for (apxchol::edge_index p = outer[i]; p < outer[i + 1]; ++p)
        if (static_cast<int>(inner[p]) == i) return vals[p];
    return 0.0;
}

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
    apxchol::bstr_incidence,
    apxchol::vec_pool_incidence>;

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
        int idx = static_cast<int>(F.perm[i]);
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
    const auto* outer = F.L.outerIndexPtr();
    const auto* inner = F.L.innerIndexPtr();
    for (int k = 0; k < static_cast<int>(F.L.outerSize()) && k < m; ++k)
        for (apxchol::edge_index p = outer[k]; p < outer[k + 1]; ++p) {
            const int row = static_cast<int>(inner[p]);
            const int col = k;
            if (row < m && col < m)
                EXPECT_GE(row, col)
                    << "upper-triangle entry at (" << row << "," << col << ")";
        }
}

TYPED_TEST(FactorizeTest, PositiveDiagonal) {
    auto L = grid_laplacian(5, 5);
    auto F = this->factorize_with(L);

    const int m = F.L.rows() - 1;
    for (int i = 0; i < m; ++i)
        EXPECT_GT(factor_diag(F.L, i), 0.0) << "zero/negative diagonal at " << i;
}

TYPED_TEST(FactorizeTest, Deterministic) {
    auto L = grid_laplacian(5, 5);
    auto F1 = this->factorize_with(L, 42);
    auto F2 = this->factorize_with(L, 42);

    EXPECT_EQ(F1.L.nonZeros(), F2.L.nonZeros());
    EXPECT_EQ(F1.perm, F2.perm);
    Eigen::SparseMatrix<double> diff = factor_to_eigen(F1.L) - factor_to_eigen(F2.L);
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
    // The GPU-resident PCG is not bit-deterministic run-to-run (cuSPARSE SpSV
    // atomics + cuBLAS reduction order); observed wobble ~3e-5 relative. Gate
    // at 1e-3 relative — still catches any real seed/algorithm change.
    EXPECT_NEAR(r1.residual, r2.residual, 1e-3 * r1.residual);
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
    // After 53e1d1d the PCG-loop ops are grouped under "pcg"; on the CPU path
    // the precond.solve() triangular-solve subtree nests as pcg.solve, while
    // the GPU-resident PCG records one pcg.gpu_pcg_loop leaf instead.
    EXPECT_GT(res.timings.total("pcg"), 0.0);
#ifdef APXCHOL_USE_CUDA
    EXPECT_GT(res.timings.total("pcg.gpu_pcg_loop"), 0.0);
#else
    EXPECT_GT(res.timings.total("pcg.solve"), 0.0);
#endif
}

// ── SDDM support tests ────────────────────────────────

/// Build an SDDM matrix: grid Laplacian + positive diagonal perturbation.
static Eigen::SparseMatrix<double> sddm_grid(int rows, int cols, double excess) {
    auto L = grid_laplacian(rows, cols);
    const int n = rows * cols;
    // Add excess to diagonal (makes it strictly diagonally dominant).
    for (int i = 0; i < n; ++i)
        L.coeffRef(i, i) += excess;
    return L;
}

TYPED_TEST(SolveTest, SDDMGridConverges) {
    auto M = sddm_grid(10, 10, 2.0);
    Eigen::VectorXd b = Eigen::VectorXd::Random(100);
    auto res = this->solve_with(M, b);
    EXPECT_LT(res.residual, 1e-5);
}

TYPED_TEST(SolveTest, SDDMLargeExcessConverges) {
    auto M = sddm_grid(10, 10, 100.0);
    Eigen::VectorXd b = Eigen::VectorXd::Random(100);
    auto res = this->solve_with(M, b);
    EXPECT_LT(res.residual, 1e-5);
}

TYPED_TEST(FactorizeTest, SDDMFlagSet) {
    auto M = sddm_grid(5, 5, 1.0);
    auto F = this->factorize_with(M);
    EXPECT_TRUE(F.sddm);
}

TYPED_TEST(FactorizeTest, LaplacianFlagNotSet) {
    auto L = grid_laplacian(5, 5);
    auto F = this->factorize_with(L);
    EXPECT_FALSE(F.sddm);
}

// Build a weighted grid Laplacian DIRECTLY from fp64 triplets. Deliberately
// bypasses weighted_grid_laplacian/graph<>: under APXCHOL_POOL_FP32 the graph
// edge pool stores weights in fp32, so routing the weights through graph<>
// would silently quantize them and mask the regression tested below.
static Eigen::SparseMatrix<double> fp64_weighted_grid_laplacian(
        int rows, int cols, double w_horiz, double w_vert) {
    const int n = rows * cols;
    auto id = [cols](int r, int c) { return r * cols + c; };
    std::vector<Eigen::Triplet<double>> trips;
    std::vector<double> deg(n, 0.0);
    auto edge = [&](int u, int v, double w) {
        trips.emplace_back(u, v, -w);
        trips.emplace_back(v, u, -w);
        deg[u] += w;
        deg[v] += w;
    };
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            if (r + 1 < rows) edge(id(r, c), id(r + 1, c), w_vert);
            if (c + 1 < cols) edge(id(r, c), id(r, c + 1), w_horiz);
        }
    for (int i = 0; i < n; ++i) trips.emplace_back(i, i, deg[i]);
    Eigen::SparseMatrix<double> L(n, n);
    L.setFromTriplets(trips.begin(), trips.end());
    return L;
}

// Regression (fp32-quantized wdeg vs exact diag): make_graph used to compute
// the excess test's weighted degree by re-walking the stored edge weights,
// which are fp32-quantized under APXCHOL_POOL_FP32. Exact fp64 diag minus a
// quantized wdeg left phantom excess ≈1e-8·diag — above the 1e-12 gate — so a
// pure Laplacian whose weights are not fp32-representable (0.01, 2/101) was
// misclassified as SDDM. The excess must come from the exact fp64 input.
TYPED_TEST(FactorizeTest, NonFp32ExactWeightsLaplacianNotSDDM) {
    auto L = fp64_weighted_grid_laplacian(10, 10, 0.01, 2.0 / 101.0);
    auto F = this->factorize_with(L);
    EXPECT_FALSE(F.sddm);
}

// Positive control for the regression above: with the same non-fp32-exact
// weights, a genuinely SDDM matrix must still be detected.
TYPED_TEST(FactorizeTest, NonFp32ExactWeightsSDDMFlagStillSet) {
    auto L = fp64_weighted_grid_laplacian(10, 10, 0.01, 2.0 / 101.0);
    L.coeffRef(0, 0) += 0.5;
    auto F = this->factorize_with(L);
    EXPECT_TRUE(F.sddm);
}

TYPED_TEST(FactorizeTest, SDDMFullRankDiagonal) {
    auto M = sddm_grid(5, 5, 1.0);
    auto F = this->factorize_with(M);
    // All n diagonal entries should be positive (including the last).
    const int n = F.L.rows();
    for (int i = 0; i < n; ++i)
        EXPECT_GT(factor_diag(F.L, i), 0.0) << "zero diagonal at " << i;
}

// ── Comprehensive IS × elimination convergence tests ──

struct StrategyCombo {
    std::string is;
    const char* name;
};

// Stable value printer. Without it gtest appends a raw byte dump (containing
// ASLR-varying pointers) to --gtest_list_tests, which gtest_discover_tests
// folds into the registered ctest name and filter — the filter then matches
// nothing and the test "passes" in 0.00s without running.
static void PrintTo(const StrategyCombo& c, std::ostream* os) { *os << c.name; }

class StrategyConvergenceTest
    : public ::testing::TestWithParam<StrategyCombo> {};

static const StrategyCombo all_combos[] = {
    {"block_greedy", "bg_tree"},
    {"luby",         "luby_tree"},
    {"baumann_kyng", "bk_tree"},
    {"rootset",      "root_tree"},
};

TEST_P(StrategyConvergenceTest, GridConverges) {
    auto [is, name] = GetParam();
    auto L = grid_laplacian(15, 15);
    auto b = apxchol::generate_test_rhs(225);
    auto res = apxchol::solve(L, b,
        {.tol = 1e-6, .max_iter = 1000,
         .storage = apxchol::graph_storage::forward_star,
         .factor_opts = {.seed = 42, .is_select = is}});
    EXPECT_LT(res.residual, 1e-4)
        << "strategy " << name << " failed: iters=" << res.iterations
        << " residual=" << res.residual;
}

TEST_P(StrategyConvergenceTest, SDDMConverges) {
    auto [is, name] = GetParam();
    auto M = sddm_grid(10, 10, 2.0);
    Eigen::VectorXd b = Eigen::VectorXd::Random(100);
    auto res = apxchol::solve(M, b,
        {.tol = 1e-6, .max_iter = 1000,
         .storage = apxchol::graph_storage::forward_star,
         .factor_opts = {.seed = 42, .is_select = is}});
    EXPECT_LT(res.residual, 1e-4)
        << "strategy " << name << " failed on SDDM: iters=" << res.iterations
        << " residual=" << res.residual;
}

INSTANTIATE_TEST_SUITE_P(AllStrategies, StrategyConvergenceTest,
    ::testing::ValuesIn(all_combos),
    [](const auto& info) { return info.param.name; });

// ─── F1: factorization quality regression tests ──────────────────────
//
// Bounds calibrated empirically: max residual observed across all
// partitioners × thread counts {1, 8, 16}, multiplied by 1.5 for headroom.
//
// Update bounds only when a deliberate algorithmic change shifts the
// achievable residual.  A regression in unbiasedness shows up as
// residuals exceeding the bound on the tree strategy.

namespace f1_bounds {
    // ── grid20 (pure Laplacian, n = 400) ──
    inline constexpr double grid20_tree   = 0.311241;
    // ── sddm_grid20 (SDDM, n = 400, excess = 2 on diagonal) ──
    inline constexpr double sddm_grid20_tree   = 0.13196;
}

// Reconstruct A from F and compute ‖A − Pᵀ L Lᵀ P‖_F / ‖A‖_F.
static double factor_residual(const apxchol::factorization& F,
                              const Eigen::SparseMatrix<double>& A) {
    Eigen::SparseMatrix<double> Lmat = factor_to_eigen(F.L);
    auto P = perm_to_eigen(F.perm);
    Eigen::SparseMatrix<double> LLt = Lmat * Lmat.transpose();
    Eigen::SparseMatrix<double> reconstructed = P.transpose() * LLt * P;
    Eigen::SparseMatrix<double> diff = A - reconstructed;
    double a_norm = A.norm();
    return a_norm == 0.0 ? 0.0 : diff.norm() / a_norm;
}

static apxchol::factorization factorize_tree(
        const Eigen::SparseMatrix<double>& A, unsigned seed = 42) {
    apxchol::factor_options opts{.seed = seed};
    return apxchol::factorize(A, apxchol::graph_storage::forward_star, opts);
}

TEST(FactorQualityTest, Grid20Tree) {
    auto L = grid_laplacian(20, 20);
    auto F = factorize_tree(L);
    EXPECT_LT(factor_residual(F, L), f1_bounds::grid20_tree);
}

TEST(FactorQualityTest, SddmGrid20Tree) {
    auto M = sddm_grid(20, 20, 2.0);
    auto F = factorize_tree(M);
    EXPECT_LT(factor_residual(F, M), f1_bounds::sddm_grid20_tree);
}

// ── AtomicEdgePool tests ──────────────────────────────

TEST(AtomicEdgePool, ReserveClaimFinalize) {
    apxchol::graph<apxchol::forward_star_incidence> G(10);
    const apxchol::edge_index pre_size = static_cast<apxchol::edge_index>(G.m());
    constexpr int N = 200;
    G.reserve_edge_pool_atomic(N);

    std::vector<apxchol::edge_index> claimed(N);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        claimed[i] = G.claim_edge_slot();

    std::sort(claimed.begin(), claimed.end());
    for (int i = 0; i < N; ++i)
        EXPECT_EQ(claimed[i], pre_size + i) << "slot " << i;

    G.finalize_edge_pool();
    EXPECT_EQ(static_cast<apxchol::edge_index>(G.m()), pre_size + N);
}

TEST(AtomicEdgePool, UnderClaimTrims) {
    apxchol::graph<apxchol::forward_star_incidence> G(5);
    const apxchol::edge_index pre = static_cast<apxchol::edge_index>(G.m());
    G.reserve_edge_pool_atomic(100);
    apxchol::edge_index s1 = G.claim_edge_slot();
    apxchol::edge_index s2 = G.claim_edge_slot();
    G.write_edge_at(s1, 0, 1, 1.0);
    G.write_edge_at(s2, 2, 3, 2.0);
    G.finalize_edge_pool();
    EXPECT_EQ(static_cast<apxchol::edge_index>(G.m()), pre + 2);
}

// ── AtomicAdjPool tests ───────────────────────────────

TEST(AtomicAdjPool, ForwardStarInlinePushParallel) {
    apxchol::graph<apxchol::forward_star_incidence> G(6);
    G.reserve_edge_pool_atomic(20);
    G.reserve_adj_pool_atomic(40);   // 2 pushes per edge

    constexpr int N = 20;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        apxchol::edge_index slot = G.claim_edge_slot();
        apxchol::node_index a = i % 6;
        apxchol::node_index b = (i + 1) % 6;
        G.write_edge_at(slot, a, b, 1.0);
        G.adj_push_inline(a, slot);
        G.adj_push_inline(b, slot);
    }
    G.finalize_edge_pool();

    size_t total = 0;
    for (apxchol::node_index v = 0; v < 6; ++v)
        for ([[maybe_unused]] auto e : G.neighbors(v)) ++total;
    EXPECT_EQ(total, size_t(2 * N));
}

TEST(AtomicAdjPool, VecInlinePushSerial) {
    apxchol::graph<apxchol::vec_incidence> G(4);
    G.reserve_edge_pool_atomic(3);
    apxchol::edge_index s0 = G.claim_edge_slot();
    apxchol::edge_index s1 = G.claim_edge_slot();
    apxchol::edge_index s2 = G.claim_edge_slot();
    G.write_edge_at(s0, 0, 1, 1.0);
    G.write_edge_at(s1, 1, 2, 2.0);
    G.write_edge_at(s2, 2, 3, 3.0);
    G.adj_push_inline(0, s0); G.adj_push_inline(1, s0);
    G.adj_push_inline(1, s1); G.adj_push_inline(2, s1);
    G.adj_push_inline(2, s2); G.adj_push_inline(3, s2);
    G.finalize_edge_pool();

    size_t total = 0;
    for (apxchol::node_index v = 0; v < 4; ++v)
        for ([[maybe_unused]] auto e : G.neighbors(v)) ++total;
    EXPECT_EQ(total, 6u);
}


// ─── Exact-clique-at-low-degree option ──────────────────────────────────────
TEST(ExactCliqueOption, Grid40_FewerItersThanSampled) {
    auto L = grid_laplacian(40, 40);
    auto b = apxchol::generate_test_rhs(L.rows());
    apxchol::factor_options opts;
    opts.seed = 42;
    opts.is_select = "block_greedy";
    auto sampled = apxchol::solve(L, b,
        {.tol = 1e-8, .max_iter = 200,
         .storage = apxchol::graph_storage::vec,
         .factor_opts = opts});
    opts.exact_clique_max_degree = 8;
    auto exact = apxchol::solve(L, b,
        {.tol = 1e-8, .max_iter = 200,
         .storage = apxchol::graph_storage::vec,
         .factor_opts = opts});
    EXPECT_LT(exact.residual, 1e-6);
    EXPECT_LE(exact.iterations, sampled.iterations)
        << "exact low-degree cliques should not need more PCG iterations: "
        << "exact=" << exact.iterations << ", sampled=" << sampled.iterations;
}
