#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <span>
#include <string>
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
#include "apxchol/solver/partition/baumann_kyng.h"
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
            trips.emplace_back(static_cast<int>(inner[p]), static_cast<int>(c),
                               apxchol::widen(vals[p]));
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
        if (static_cast<int>(inner[p]) == i) return apxchol::widen(vals[p]);
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

// ── Determinism of the PARALLEL selection path ───────
//
// FactorizeTest.Deterministic above factorizes a 5x5 grid: 25 candidates,
// far below factor_options::omp_threshold, so every round takes the SERIAL
// branch of the partitioner and the parallel one is never executed. This test
// executes it, and is the regression guard for two schedule-dependent outputs
// fixed 2026-08-20:
//
//   * block_greedy's cross-block conflict resolution read the shared chosen[]
//     mask while other threads were clearing it, so whether a boundary pick
//     dropped depended on who got there first. That changed the round's
//     independent set, hence the elimination order, hence the factor's
//     STRUCTURE and its nnz, run to run at one seed and one thread count.
//     Measured at the parent commit on a 32-core box, standalone (i.e. not
//     under ctest's OMP_NUM_THREADS=4 cap), SpTRSVSetupMemory.SetupConsuming-
//     ReleasesTheFactorAndSolvesIdentically — which factorizes a 120x120 grid
//     twice and compares — failed 19/50 at T=8, 16/50 at T=16, 19/50 at T=32,
//     and 0/50 at T=1 and T=4.
//   * rootset's peel passes are `schedule(dynamic, 64)` into per-thread
//     buffers, so the frontier ORDER (not the set) followed chunk arrival:
//     same independent set, different labels, different stored factor.
//
// Both are selection-side, so one storage suffices here; the storage x
// partitioner matrix was checked separately.
namespace {
// Raise the OpenMP team size for one test and put it back (gtest runs the
// whole binary in one process; ctest pins OMP_NUM_THREADS=4 for it).
struct scoped_threads {
    int saved = 1;
    explicit scoped_threads([[maybe_unused]] int n) {
#ifdef _OPENMP
        saved = omp_get_max_threads();
        omp_set_num_threads(n);
#endif
    }
    ~scoped_threads() {
#ifdef _OPENMP
        omp_set_num_threads(saved);
#endif
    }
};

// Byte-for-byte equality of two factors: same column pointers, same row
// indices, same values. Structure AND values, not a norm.
void expect_same_factor(const apxchol::factorization& a,
                        const apxchol::factorization& b,
                        const std::string& what) {
    ASSERT_EQ(a.perm, b.perm) << what << ": elimination order differs";
    ASSERT_EQ(a.L.nonZeros(), b.L.nonZeros()) << what << ": factor nnz differs";
    const std::size_t nc = static_cast<std::size_t>(a.L.cols()) + 1;
    const std::size_t nz = static_cast<std::size_t>(a.L.nonZeros());
    EXPECT_EQ(std::memcmp(a.L.outerIndexPtr(), b.L.outerIndexPtr(),
                          nc * sizeof(apxchol::edge_index)), 0)
        << what << ": column pointers differ";
    EXPECT_EQ(std::memcmp(a.L.innerIndexPtr(), b.L.innerIndexPtr(),
                          nz * sizeof(apxchol::node_index)), 0)
        << what << ": row indices differ (factor STRUCTURE is not reproducible)";
    EXPECT_EQ(std::memcmp(a.L.valuePtr(), b.L.valuePtr(),
                          nz * sizeof(apxchol::factor_value_t)), 0)
        << what << ": factor values differ";
}
} // namespace

TEST(FactorizeDeterminism, ParallelSelectionIsReproducibleAtAFixedThreadCount) {
#ifndef _OPENMP
    GTEST_SKIP() << "serial build: there is no parallel selection path";
#else
    if (const char* e = std::getenv("APXCHOL_OMP_THRESHOLD"); e && *e)
        GTEST_SKIP() << "APXCHOL_OMP_THRESHOLD=" << e
                     << " overrides the omp_threshold this test sets";
    // 14400 candidates against a 256-vertex OpenMP gate: most of the ~45
    // elimination rounds take the parallel branch. 16 threads whatever the box
    // has -- oversubscription is welcome here, it widens the window the
    // block_greedy race needed.
    const scoped_threads team(16);
    const auto L = grid_laplacian(120, 120);
    for (const char* sel : {"block_greedy", "luby", "rootset", "baumann_kyng"}) {
        apxchol::factor_options opts;
        opts.seed = 3;
        opts.omp_threshold = 256;
        opts.is_select = sel;
        const auto ref = apxchol::factorize(L, apxchol::graph_storage::vec_pool, opts);
        ASSERT_GT(ref.L.nonZeros(), 70000) << sel;   // the parallel path really ran
        for (int rep = 1; rep <= 3; ++rep) {
            const auto F = apxchol::factorize(L, apxchol::graph_storage::vec_pool, opts);
            expect_same_factor(ref, F, std::string(sel) + " rep " + std::to_string(rep));
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
#endif
}

// The T=1 half of the contract: one thread, byte-identical factor. Cheap and
// unconditional -- it holds on the serial build too.
TEST(FactorizeDeterminism, SingleThreadedFactorizationIsByteIdentical) {
    const scoped_threads team(1);
    const auto L = grid_laplacian(60, 60);
    apxchol::factor_options opts;
    opts.seed = 11;
    const auto a = apxchol::factorize(L, apxchol::graph_storage::vec_pool, opts);
    const auto b = apxchol::factorize(L, apxchol::graph_storage::vec_pool, opts);
    expect_same_factor(a, b, "T=1");
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
    // The GPU-resident PCG is bit-deterministic run to run: its SpMV, vector
    // passes and reductions are our own kernels with fixed grids and fixed-
    // order partial sums (pcg_cuda_kernels.h -- no floating-point atomics),
    // and so is our dataflow SpTRSV backend (the AUTO choice). NOT on
    // cuSPARSE SpSV (APXCHOL_GPU_SPTRSV=cusparse --
    // only where the build opted in
    // with APXCHOL_CUDA_WITH_CUSPARSE): its atomics wobble the residual ~1e-2
    // relative (measured 2e-3 .. 4e-3 on this 8x8 grid) and the iteration
    // count by ±1. Exact where the SpTRSV is ours; a 5e-2 relative gate on
    // cuSPARSE (still catches any real seed/algorithm change, which moves the
    // residual by orders of magnitude; the earlier 1e-3 gate flaked).
    const int  be = apxchol::cuda_sptrsv::backend_from_env();
    const bool maybe_cusparse = apxchol::cuda_sptrsv::cusparse_available() && be < 0;
    if (maybe_cusparse) EXPECT_NEAR(r1.residual, r2.residual, 5e-2 * r1.residual);
    else                EXPECT_DOUBLE_EQ(r1.residual, r2.residual);
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

// ── PCG-loop determinism (fused, OpenMP-parallel vector kernels) ──
// The PCG outer loop and the preconditioner application stream their vectors
// in fused passes with hand-rolled per-thread partial reductions (see the
// detail:: scaffolding in preconditioner.h). Same factor + same b must give
// bit-identical iterations / residual / x run to run at the thread count the
// test process has. n = 10k is above the OpenMP engagement threshold
// (detail::fused_omp_min() == 2000, a constant), so with OMP_NUM_THREADS > 1
// this exercises the parallel path (a reduction() clause or any
// completion-order sum would fail this at T > 1).
static void expect_repeated_solves_bit_identical(const Eigen::SparseMatrix<double>& A,
                                                 const Eigen::VectorXd& b, bool laplacian) {
    apxchol::solve_options opts;
    opts.tol = 1e-8; opts.max_iter = 500; opts.factor_opts.seed = 42;
    apxchol::cpu_solver slv(A, opts);
    auto r1 = slv.solve(b);
    auto r2 = slv.solve(b);
    ASSERT_LT(r1.residual, 1e-8);
    EXPECT_EQ(r1.iterations, r2.iterations);
    EXPECT_EQ(r1.residual, r2.residual);                    // exact, not ULP-tolerant
    EXPECT_TRUE((r1.x.array() == r2.x.array()).all());
    // Warm start (x0 path: r = b - A x0 in one fused pass) is deterministic too.
    auto w1 = slv.solve(b, 1e-10, 500, &r1.x);
    auto w2 = slv.solve(b, 1e-10, 500, &r1.x);
    EXPECT_EQ(w1.iterations, w2.iterations);
    EXPECT_EQ(w1.residual, w2.residual);
    EXPECT_TRUE((w1.x.array() == w2.x.array()).all());
    // The Eigen-facing application (apply -> _solve_impl): same guarantee. On
    // a Laplacian it follows the center-k schedule (APXCHOL_GROUND=center-k,
    // see env_knobs.h): only every K-th application since the last
    // reset_apply_count() runs the two mean passes, so compare applications
    // at the same phase (restart the schedule before each call).
    const auto& pc = slv.preconditioner();
    pc.reset_apply_count();
    Eigen::VectorXd z1 = slv.apply(b);
    pc.reset_apply_count();
    Eigen::VectorXd z2 = slv.apply(b);
    EXPECT_TRUE((z1.array() == z2.array()).all());
    if (laplacian) {
        // The K-th application after a restart centres: mean zero to rounding.
        const int K = apxchol::detail::env_knobs::get().center_k;
        pc.reset_apply_count();
        Eigen::VectorXd zk;
        for (int i = 0; i < K; ++i) zk = slv.apply(b);
        EXPECT_NEAR(zk.mean(), 0.0, 1e-13 * zk.cwiseAbs().maxCoeff());
        // The returned SOLUTION is min-norm regardless of the schedule
        // (cpu_solver::solve centres x once at the end).
        EXPECT_NEAR(r1.x.mean(), 0.0, 1e-14 * r1.x.cwiseAbs().maxCoeff());
        EXPECT_NEAR(w1.x.mean(), 0.0, 1e-14 * w1.x.cwiseAbs().maxCoeff());
    }
}

TEST(PcgFusion, RepeatedSolvesBitIdenticalLaplacian) {
    auto L = grid_laplacian(100, 100);
    auto b = apxchol::generate_test_rhs(L.rows());
    expect_repeated_solves_bit_identical(L, b, /*laplacian=*/true);
}

TEST(PcgFusion, RepeatedSolvesBitIdenticalSDDM) {
    auto M = sddm_grid(100, 100, 0.5);
    Eigen::VectorXd b = Eigen::VectorXd::Random(M.rows());
    expect_repeated_solves_bit_identical(M, b, /*laplacian=*/false);
}

// ── Grounding: min-norm Laplacian solution ──
// cpu_solver::solve returns the MIN-NORM solution of a Laplacian system.
// Under the center-k schedule the preconditioner skips its output
// re-centring on most applications, so the PCG iterate drifts along the null
// space 1 (invisible to the residual); solve_impl subtracts mean(x) once at
// the end -- also for a warm start, whose constant component must not leak
// into the answer.
TEST(Grounding, LaplacianSolutionIsMinNorm) {
    auto L = grid_laplacian(60, 60);        // n = 3600 > detail::fused_omp_min() (2000)
    auto b = apxchol::generate_test_rhs(L.rows());
    apxchol::solve_options opts;
    opts.tol = 1e-8; opts.max_iter = 500; opts.factor_opts.seed = 42;
    apxchol::cpu_solver slv(L, opts);
    auto r = slv.solve(b);
    ASSERT_LT(r.residual, 1e-8);
    EXPECT_NEAR(r.x.mean(), 0.0, 1e-14 * r.x.cwiseAbs().maxCoeff());
    EXPECT_LT((L * r.x - b).norm() / b.norm(), 1e-6);
    // Warm start shifted by a constant, tightened tolerance -> PCG iterates
    // from x0; the answer is still mean-zero and still solves the system.
    Eigen::VectorXd x0 = r.x.array() + 5.0;
    auto w = slv.solve(b, 1e-10, 500, &x0);
    EXPECT_GT(w.iterations, 0);
    EXPECT_NEAR(w.x.mean(), 0.0, 1e-14 * w.x.cwiseAbs().maxCoeff());
    EXPECT_LT((L * w.x - b).norm() / b.norm(), 1e-8);
    // Already-converged warm start (loose tolerance -> 0 iterations): the
    // early exit centres too.
    auto e = slv.solve(b, 1e-6, 500, &x0);
    EXPECT_EQ(e.iterations, 0);
    EXPECT_NEAR(e.x.mean(), 0.0, 1e-14 * e.x.cwiseAbs().maxCoeff());
    EXPECT_LT((L * e.x - b).norm() / b.norm(), 1e-6);
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

// ─── BK residual loop (parallel_residual_threshold) ─────────────────────────
//
// A complete graph is the cheapest way to reach this code from a unit test: its
// independent set is one vertex, so block_greedy trips min_is_fraction on the
// first round and hands the whole graph to the residual path. That is the same
// loop the social graphs enter at 70k-830k active.
namespace {
Eigen::SparseMatrix<double> clique_laplacian(int n) {
    Eigen::SparseMatrix<double> L(n, n);
    std::vector<Eigen::Triplet<double>> t;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            if (i != j) t.emplace_back(i, j, -1.0);
        t.emplace_back(i, i, static_cast<double>(n - 1));
    }
    L.setFromTriplets(t.begin(), t.end());
    return L;
}
}  // namespace

TEST(BkResidualLoop, DrivesTheResidualToTheThresholdAndStaysDeterministic) {
    constexpr int n = 60;
    constexpr size_t thresh = 5;
    const auto L = clique_laplacian(n);
    const auto b = apxchol::generate_test_rhs(L.rows());

    apxchol::factor_options peel_only;              // 60 < the trait's 500, so
    peel_only.seed = 7;                             // the loop is not entered:
    const auto a = apxchol::factorize(              // the peel takes all 60.
        L, apxchol::graph_storage::vec_pool, peel_only);
    EXPECT_TRUE(a.rounds.empty())
        << "block_greedy did not bail on a clique, so this test is not "
           "exercising the residual path any more";

    // Several seeds: the loop must reach the threshold under every one of them.
    // On a clique BK's sample is empty with probability (1-p)^n ~ 3% per round
    // over ~55 rounds, so at least one seed here whiffs — and a whiff must be
    // retried, not treated as "the residual stopped shrinking". Bailing on the
    // first empty round (the behaviour until 2026-08-20) dumps the rest on the
    // serial peel and fails the peel-column bound below.
    for (unsigned seed : {7u, 11u, 42u, 1234u, 20260820u}) {
        apxchol::factor_options with_loop;          // BK rounds down to `thresh`,
        with_loop.seed = seed;                      // then the peel takes the
        with_loop.parallel_residual_threshold = thresh;   // rest.
        const auto c = apxchol::factorize(
            L, apxchol::graph_storage::vec_pool, with_loop);

        size_t in_rounds = 0;
        for (const auto& r : c.rounds) in_rounds += r.is_size;
        EXPECT_GE(in_rounds, static_cast<size_t>(n) - thresh)
            << "seed " << seed << ": the BK residual loop handed "
            << (static_cast<size_t>(n) - in_rounds) << " columns to the serial "
            << "peel, but parallel_residual_threshold = " << thresh
            << " allows at most " << thresh;

        // Same (input, seed, thread count) => same factor, bit for bit.
        const auto c2 = apxchol::factorize(
            L, apxchol::graph_storage::vec_pool, with_loop);
        expect_same_factor(c, c2, "BK residual loop, seed " +
                                  std::to_string(seed));
        if (::testing::Test::HasFatalFailure()) return;

        // And it is still a usable preconditioner.
        const auto res = apxchol::solve(L, b,
            {.tol = 1e-8, .max_iter = 200,
             .storage = apxchol::graph_storage::vec_pool,
             .factor_opts = with_loop});
        EXPECT_LT(res.residual, 1e-8) << "seed " << seed;
    }
}

// The residual loop constructs a fresh baumann_kyng_partitioner partway through
// an elimination, where graph::m() is monotone and 2*m/|active| is inflated by
// every edge the eliminated prefix consumed. BK must honour a caller-supplied
// est_avg_degree on round 0 rather than recomputing one; this pins that seam,
// whose only visible effect is which vertices round 0 samples.
TEST(BaumannKyngSeeding, Round0UsesTheSeedInsteadOfTwoMOverActive) {
    const auto L = grid_laplacian(30, 30);          // 900 vertices, avg degree < 4
    apxchol::partition_context ctx{apxchol::partition_options{}, 42, 2000, nullptr};
    std::vector<apxchol::node_index> active(900);
    std::iota(active.begin(), active.end(), apxchol::node_index{0});

    auto is_size = [&](double seed) {
        auto G = apxchol::make_graph<
            apxchol::graph<apxchol::vec_pool_incidence>>(L);
        apxchol::baumann_kyng_partitioner bk;
        bk.est_avg_degree = seed;                   // 0 = "work it out yourself"
        apxchol::selection sel;
        sel.reset(G.n());
        bk.find_partition(G, std::span<const apxchol::node_index>(active),
                          ctx, sel);
        return sel.finalize().num_regions();
    };

    // Seeding a huge average degree drives p = 1/(c·d) down, so round 0 samples
    // almost nothing; the unseeded call measures the real ~4 and samples freely.
    const auto unseeded = is_size(0.0);
    const auto inflated = is_size(4000.0);
    EXPECT_GT(unseeded, 0u);
    EXPECT_LT(inflated, unseeded)
        << "round 0 ignored the caller's est_avg_degree seed (unseeded="
        << unseeded << ", seeded 4000 => " << inflated << ")";
}
