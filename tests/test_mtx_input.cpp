// Tests for the CLI's Laplacian-vs-adjacency input interpretation
// (src/mtx_input.h): the human-facing layer that decides how to READ a .mtx
// file and prints its decision, with --input-kind to override it.
//
// The LIBRARY-side counterpart -- asserting the operator class rather than
// guessing at it -- lives in test_operator_class.cpp. Note the split: reading
// an adjacency matrix as an operator no longer "breaks down at iteration 0",
// it is REFUSED by the precondition assertion (see
// AdjacencyInputRegression.RawAdjacencyIsRefusedByTheOperatorContract below).
#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "apxchol/solver/factorization.h"
#include "apxchol/solver/solve.h"
#include "mtx_input.h"

using apxchol::adjacency_to_laplacian;
using apxchol::describe_input;
using apxchol::input_kind;
using apxchol::input_scan;
using apxchol::project_laplacian_rhs_components;
using apxchol::resolve_input_kind;
using apxchol::scan_input;

namespace {

using Sparse = Eigen::SparseMatrix<double>;
using Trip   = Eigen::Triplet<double>;

/// Symmetric adjacency matrix of a rows x cols 4-neighbour grid, both
/// triangles stored (what a MatrixMarket `symmetric` file expands to).
Sparse grid_adjacency(int rows, int cols, double w = 1.0) {
    std::vector<Trip> t;
    auto id = [cols](int r, int c) { return r * cols + c; };
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            if (r + 1 < rows) {
                t.emplace_back(id(r, c), id(r + 1, c), w);
                t.emplace_back(id(r + 1, c), id(r, c), w);
            }
            if (c + 1 < cols) {
                t.emplace_back(id(r, c), id(r, c + 1), w);
                t.emplace_back(id(r, c + 1), id(r, c), w);
            }
        }
    Sparse A(rows * cols, rows * cols);
    A.setFromTriplets(t.begin(), t.end());
    return A;
}

/// The matching assembled Laplacian L = D - A.
Sparse grid_laplacian(int rows, int cols, double w = 1.0) {
    Sparse L = grid_adjacency(rows, cols, w);
    adjacency_to_laplacian(L);
    return L;
}

Sparse path_adjacency(int n) { return grid_adjacency(1, n); }

/// nnz of the strictly-off-diagonal part of a factor column set: what the
/// factorization actually produced beyond the trivial diagonal.
Eigen::Index offdiag_factor_nnz(const apxchol::factorization& F) {
    return static_cast<Eigen::Index>(F.L.nonZeros()) -
           static_cast<Eigen::Index>(F.L.cols());
}

} // namespace

// ── scan_input ───────────────────────────────────────

TEST(MtxInputScan, AssembledLaplacian) {
    const Sparse L = grid_laplacian(6, 6);
    const input_scan s = scan_input(L);
    EXPECT_EQ(s.n, 36);
    EXPECT_EQ(s.offdiag_pos, 0);
    EXPECT_EQ(s.offdiag_neg, s.offdiag_nnz);
    EXPECT_EQ(s.diag_pos, 36);
    EXPECT_EQ(s.deficient_rows, 0);
    EXPECT_EQ(s.excess_rows, 0);          // pure Laplacian: every row sums to 0
    EXPECT_EQ(s.upper_nnz, s.lower_nnz);  // symmetric
}

TEST(MtxInputScan, AdjacencyMatrix) {
    const Sparse A = grid_adjacency(6, 6);
    const input_scan s = scan_input(A);
    EXPECT_EQ(s.n, 36);
    EXPECT_EQ(s.offdiag_neg, 0);
    EXPECT_EQ(s.offdiag_pos, s.offdiag_nnz);
    EXPECT_EQ(s.diag_pos, 0);             // no diagonal at all
    EXPECT_EQ(s.deficient_rows, 0);
    EXPECT_EQ(s.excess_rows, 36);         // every row sum is the degree
}

TEST(MtxInputScan, CountsDominanceViolations) {
    // grid_2000.mtx's exact corruption: an extra off-diagonal entry that the
    // diagonal does not account for, leaving two rows with row sum -1.
    Sparse L = grid_laplacian(4, 4);
    std::vector<Trip> t;
    for (int k = 0; k < L.outerSize(); ++k)
        for (Sparse::InnerIterator it(L, k); it; ++it)
            t.emplace_back(static_cast<int>(it.row()), static_cast<int>(it.col()),
                           it.value());
    t.emplace_back(0, 15, -1.0);
    t.emplace_back(15, 0, -1.0);
    Sparse bad(16, 16);
    bad.setFromTriplets(t.begin(), t.end());

    const input_scan s = scan_input(bad);
    EXPECT_EQ(s.deficient_rows, 2);
    EXPECT_NEAR(s.worst_deficit, -1.0, 1e-12);
    EXPECT_GE(s.worst_row, 0);
}

// ── resolve_input_kind: automatic ────────────────────

TEST(MtxInputResolve, AutoPicksLaplacianForAssembledOperator) {
    std::string why;
    EXPECT_EQ(resolve_input_kind(input_kind::automatic,
                                 scan_input(grid_laplacian(6, 6)), false, why),
              input_kind::laplacian);
    EXPECT_NE(why.find("non-positive"), std::string::npos) << why;
}

TEST(MtxInputResolve, AutoPicksAdjacencyForUnweightedGraph) {
    std::string why;
    EXPECT_EQ(resolve_input_kind(input_kind::automatic,
                                 scan_input(grid_adjacency(6, 6)), true, why),
              input_kind::adjacency);
    EXPECT_NE(why.find("pattern"), std::string::npos) << why;
}

TEST(MtxInputResolve, AutoPicksAdjacencyForWeightedGraph) {
    // Positive weights and a few self-loops: still an adjacency matrix.
    Sparse A = grid_adjacency(6, 6, 2.5);
    A.coeffRef(0, 0) = 3.0;
    A.coeffRef(7, 7) = 1.0;
    A.makeCompressed();
    std::string why;
    EXPECT_EQ(resolve_input_kind(input_kind::automatic, scan_input(A), false, why),
              input_kind::adjacency);
    EXPECT_NE(why.find("positive"), std::string::npos) << why;
}

TEST(MtxInputResolve, AutoPicksLaplacianForMixedSignWithFullDiagonal) {
    // A general FEM/structural matrix (parabolic_fem, thermal2, bcsstk*):
    // both off-diagonal signs, but every row carries a positive diagonal.
    Sparse L = grid_laplacian(6, 6);
    L.coeffRef(0, 1) = +0.25;
    L.coeffRef(1, 0) = +0.25;
    L.makeCompressed();
    std::string why;
    EXPECT_EQ(resolve_input_kind(input_kind::automatic, scan_input(L), false, why),
              input_kind::laplacian);
    EXPECT_NE(why.find("mixed-sign"), std::string::npos) << why;
}

TEST(MtxInputResolve, AutoRefusesAmbiguousMixedSign) {
    // Mixed signs AND no diagonal: neither reading is defensible.
    std::vector<Trip> t{{0, 1, +1.0}, {1, 0, +1.0}, {1, 2, -1.0}, {2, 1, -1.0}};
    Sparse M(3, 3);
    M.setFromTriplets(t.begin(), t.end());

    std::string why;
    try {
        resolve_input_kind(input_kind::automatic, scan_input(M), false, why);
        FAIL() << "expected resolve_input_kind to refuse an ambiguous matrix";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("cannot tell"), std::string::npos) << msg;
        EXPECT_NE(msg.find("--input-kind"), std::string::npos) << msg;
    }
}

TEST(MtxInputResolve, RejectsOneSidedMatrix) {
    // Off-diagonals in one triangle only: the file was not read symmetrically,
    // so BOTH readings would build a non-symmetric operator.
    std::vector<Trip> t{{1, 0, -1.0}, {2, 1, -1.0}, {0, 0, 1.0},
                        {1, 1, 2.0},  {2, 2, 1.0}};
    Sparse M(3, 3);
    M.setFromTriplets(t.begin(), t.end());

    std::string why;
    try {
        resolve_input_kind(input_kind::automatic, scan_input(M), false, why);
        FAIL() << "expected resolve_input_kind to reject a one-sided matrix";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("not symmetric"), std::string::npos)
            << e.what();
    }
}

TEST(MtxInputResolve, DiagonalOnlyMatrixIsAnOperator) {
    Sparse M(3, 3);
    M.setIdentity();
    std::string why;
    EXPECT_EQ(resolve_input_kind(input_kind::automatic, scan_input(M), false, why),
              input_kind::laplacian);
}

// ── resolve_input_kind: explicit ─────────────────────

TEST(MtxInputResolve, ExplicitKindWins) {
    const input_scan adj = scan_input(grid_adjacency(6, 6));
    const input_scan lap = scan_input(grid_laplacian(6, 6));
    std::string why;
    EXPECT_EQ(resolve_input_kind(input_kind::laplacian, adj, false, why),
              input_kind::laplacian);
    EXPECT_EQ(resolve_input_kind(input_kind::adjacency, lap, false, why),
              input_kind::adjacency);
}

TEST(MtxInputResolve, ExplicitLaplacianOnAdjacencyWarns) {
    std::string why;
    resolve_input_kind(input_kind::laplacian, scan_input(grid_adjacency(6, 6)),
                       true, why);
    EXPECT_NE(why.find("WARNING"), std::string::npos) << why;
    EXPECT_NE(why.find("--input-kind adjacency"), std::string::npos) << why;
}

// ── describe_input ───────────────────────────────────

TEST(MtxInputDescribe, NamesTheChosenReading) {
    const std::string adj =
        describe_input(input_kind::adjacency, scan_input(grid_adjacency(4, 4)), "r");
    EXPECT_NE(adj.find("ADJACENCY"), std::string::npos) << adj;
    EXPECT_NE(adj.find("L = D - A"), std::string::npos) << adj;

    const std::string lap =
        describe_input(input_kind::laplacian, scan_input(grid_laplacian(4, 4)), "r");
    EXPECT_NE(lap.find("LAPLACIAN/SDDM"), std::string::npos) << lap;
    EXPECT_NE(lap.find("pure Laplacian"), std::string::npos) << lap;
}

// ── adjacency_to_laplacian ───────────────────────────

TEST(AdjacencyToLaplacian, BuildsDegreeMinusAdjacency) {
    Sparse L = grid_adjacency(4, 5, 2.0);
    adjacency_to_laplacian(L);

    ASSERT_EQ(L.rows(), 20);
    for (int k = 0; k < L.outerSize(); ++k) {
        double colsum = 0.0;
        for (Sparse::InnerIterator it(L, k); it; ++it) {
            colsum += it.value();
            if (it.row() == it.col()) EXPECT_GT(it.value(), 0.0);
            else                      EXPECT_DOUBLE_EQ(it.value(), -2.0);
        }
        EXPECT_NEAR(colsum, 0.0, 1e-12);        // pure Laplacian
    }
    // Symmetric.
    const Sparse asym = Sparse(L.transpose()) - L;
    EXPECT_NEAR(asym.norm(), 0.0, 1e-12);
}

TEST(AdjacencyToLaplacian, DropsSelfLoops) {
    Sparse A = path_adjacency(5);
    A.coeffRef(2, 2) = 7.0;
    A.makeCompressed();
    adjacency_to_laplacian(A);

    for (Sparse::InnerIterator it(A, 2); it; ++it)
        if (it.row() == it.col())
            EXPECT_DOUBLE_EQ(it.value(), 2.0);   // degree 2, self-loop ignored
}

TEST(AdjacencyToLaplacian, UsesAbsoluteValues) {
    // The benchmark suite's load_mtx_as_adjacency takes |value|; matching it
    // means --input-kind adjacency reproduces the benchmark's reading.
    Sparse A = path_adjacency(4);
    for (int k = 0; k < A.outerSize(); ++k)
        for (Sparse::InnerIterator it(A, k); it; ++it)
            it.valueRef() = -it.value();
    Sparse B = path_adjacency(4);
    adjacency_to_laplacian(A);
    adjacency_to_laplacian(B);
    EXPECT_NEAR((A - B).norm(), 0.0, 1e-12);
}

TEST(AdjacencyToLaplacian, IsIdempotentOnAPureLaplacian) {
    // Re-reading an assembled pure Laplacian as an adjacency matrix must give
    // the same operator back: |−w| = w, and the rebuilt diagonal is the degree,
    // which for a pure Laplacian is what the diagonal already was.
    const Sparse L = grid_laplacian(5, 6, 1.5);
    Sparse again = L;
    adjacency_to_laplacian(again);
    EXPECT_NEAR((L - again).norm(), 0.0, 1e-12);
}

TEST(ComponentRhsProjection, LeavesConnectedRhsByteIdentical) {
    const Sparse L = grid_laplacian(4, 5);
    Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(L.rows(), -2.0, 3.0);
    const Eigen::VectorXd original = b;

    EXPECT_EQ(project_laplacian_rhs_components(L, b), 1);
    EXPECT_EQ(b, original);
}

TEST(ComponentRhsProjection, CentersEveryDisconnectedComponent) {
    std::vector<Trip> edges{
        {0, 1, 1.0}, {1, 0, 1.0}, {1, 2, 1.0}, {2, 1, 1.0},
        {3, 4, 1.0}, {4, 3, 1.0}, {4, 5, 1.0}, {5, 4, 1.0}};
    Sparse L(6, 6);
    L.setFromTriplets(edges.begin(), edges.end());
    adjacency_to_laplacian(L);
    Eigen::VectorXd b(6);
    b << 1.0, 2.0, 6.0, -3.0, 4.0, 8.0;

    EXPECT_EQ(project_laplacian_rhs_components(L, b), 2);
    EXPECT_NEAR(b.head(3).sum(), 0.0, 1e-15);
    EXPECT_NEAR(b.tail(3).sum(), 0.0, 1e-15);
}

// ── end-to-end regression: the adjacency-as-operator mistake ─────────────────

TEST(AdjacencyInputRegression, RawAdjacencyIsRefusedByTheOperatorContract) {
    // The bug this module exists to prevent, and how it now surfaces. Handing
    // the solver a raw adjacency matrix used to negate every off-diagonal into
    // a negative edge weight, produce a fill-free factor and a PCG that could
    // not take a single step ("iterations 0 / residual 1"). The library now
    // asserts the operator class instead: an adjacency matrix has no positive
    // diagonal anywhere, which is a hard PSD violation, so it is REFUSED --
    // with the explicit conversion named.
    constexpr Eigen::Index kEdges = 2 * 20 * 19;   // 20x20 4-neighbour grid
    const Sparse A = grid_adjacency(20, 20);
    ASSERT_EQ(A.nonZeros(), 2 * kEdges);           // both triangles stored

    for (auto call : {+[](const Sparse& M) {
                          (void)apxchol::factorize(M);
                      },
                      +[](const Sparse& M) {
                          (void)apxchol::solve(
                              M, apxchol::generate_test_rhs(M.rows()),
                              {.tol = 1e-8, .max_iter = 200});
                      }}) {
        try {
            call(A);
            FAIL() << "expected the operator contract to refuse an adjacency matrix";
        } catch (const std::invalid_argument& e) {
            const std::string msg = e.what();
            EXPECT_NE(msg.find("positive diagonal"), std::string::npos) << msg;
            EXPECT_NE(msg.find("ADJACENCY"), std::string::npos) << msg;
            EXPECT_NE(msg.find("apxchol.laplacian(A)"), std::string::npos) << msg;
        }
    }
}

TEST(AdjacencyInputRegression, ConvertedAdjacencyHasPlausibleFillAndConverges) {
    constexpr Eigen::Index kEdges = 2 * 20 * 19;
    Sparse L = grid_adjacency(20, 20);
    adjacency_to_laplacian(L);

    const auto F = apxchol::factorize(L);
    // With the operator right, the randomized elimination emits sampled clique
    // edges, so the factor must hold strictly more than the fill-free count.
    EXPECT_GT(offdiag_factor_nnz(F), kEdges)
        << "factor shows no fill at all -- the clique sampler bailed out again";

    const Eigen::VectorXd b = apxchol::generate_test_rhs(L.rows());
    const auto res = apxchol::solve(L, b, {.tol = 1e-8, .max_iter = 200});
    EXPECT_GT(res.iterations, 0);
    EXPECT_LT(res.residual, 1e-8);
}

TEST(AdjacencyInputRegression, AutoDetectionRoutesAnAdjacencyMatrixToConvergence) {
    // The full CLI decision path, minus the file reading: scan -> resolve ->
    // convert -> solve.
    Sparse M = grid_adjacency(16, 16);
    const input_scan s = scan_input(M);
    std::string why;
    const input_kind kind = resolve_input_kind(input_kind::automatic, s, true, why);
    ASSERT_EQ(kind, input_kind::adjacency);
    adjacency_to_laplacian(M);

    const Eigen::VectorXd b = apxchol::generate_test_rhs(M.rows());
    const auto res = apxchol::solve(M, b, {.tol = 1e-8, .max_iter = 200});
    EXPECT_GT(res.iterations, 0);
    EXPECT_LT(res.residual, 1e-8);
}

TEST(AdjacencyInputRegression, SddmInputIsStillSolvedAsAnOperator) {
    // Requirement: SDDM (positive diagonal excess) must keep working, and must
    // NOT be rebuilt as a Laplacian -- that would discard the excess.
    Sparse L = grid_laplacian(16, 16);
    for (int i = 0; i < L.rows(); ++i) L.coeffRef(i, i) += 0.05;
    L.makeCompressed();

    const input_scan s = scan_input(L);
    EXPECT_EQ(s.excess_rows, L.rows());
    std::string why;
    ASSERT_EQ(resolve_input_kind(input_kind::automatic, s, false, why),
              input_kind::laplacian);

    const Eigen::VectorXd b = apxchol::generate_test_rhs(L.rows());
    const auto res = apxchol::solve(L, b, {.tol = 1e-8, .max_iter = 200});
    EXPECT_GT(res.iterations, 0);
    EXPECT_LT(res.residual, 1e-8);
}
