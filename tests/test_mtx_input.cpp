// Tests for the CLI's Laplacian-vs-adjacency input interpretation
// (src/mtx_input.h): the human-facing layer that reads a .mtx file as the
// operator unless --convert-input is given, and prints how it read it.
//
// The LIBRARY-side counterpart -- asserting the operator class rather than
// guessing at it -- lives in test_operator_class.cpp. Note the split: reading
// an adjacency matrix as an operator no longer "breaks down at iteration 0",
// it is REFUSED by the precondition assertion (see
// AdjacencyInputRegression.RawAdjacencyIsRefusedByTheOperatorContract below).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "apxchol/solver/factorization.h"
#include "apxchol/solver/solve.h"
#include "csc_test_util.h"
#include "mtx_input.h"
#include "test_rhs.h"

using apxchol::describe_input;
using apxchol::input_scan;
using apxchol::test::adjacency_laplacian;
using apxchol::test::is_graph_adjacency;

namespace {

using Sparse = apxchol::test::sparse_matrix;
using Trip   = apxchol::test::triplet;

input_scan scan_input(const Sparse& M) {
    return apxchol::scan_input(apxchol::test::internal_csc(M));
}

std::int64_t project_laplacian_rhs_components(const Sparse& L, std::vector<double>& b) {
    return apxchol::project_laplacian_rhs_components(apxchol::test::internal_csc(L), b);
}

using apxchol::test::set_entry;
/// M(i, j) += value, inserting the entry when it is not stored.
void add_entry(Sparse& M, std::int64_t i, std::int64_t j, double value) {
    set_entry(M, i, j, M.coeff(i, j) + value);
}

/// Symmetric adjacency matrix of a rows x cols 4-neighbour grid, both
/// triangles stored (what a MatrixMarket `symmetric` file expands to).
Sparse grid_adjacency(int rows, int cols, double w = 1.0) {
    std::vector<Trip> t;
    auto id = [cols](int r, int c) { return r * cols + c; };
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            if (r + 1 < rows) {
                t.push_back({id(r, c), id(r + 1, c), w});
                t.push_back({id(r + 1, c), id(r, c), w});
            }
            if (c + 1 < cols) {
                t.push_back({id(r, c), id(r, c + 1), w});
                t.push_back({id(r, c + 1), id(r, c), w});
            }
        }
    return apxchol::test::from_triplets(rows * cols, rows * cols, t);
}

/// The matching assembled Laplacian L = D - A.
Sparse grid_laplacian(int rows, int cols, double w = 1.0) {
    Sparse L = grid_adjacency(rows, cols, w);
    L = adjacency_laplacian(L);
    return L;
}

Sparse path_adjacency(int n) { return grid_adjacency(1, n); }

/// nnz of the strictly-off-diagonal part of a factor column set: what the
/// factorization actually produced beyond the trivial diagonal.
std::int64_t offdiag_factor_nnz(const apxchol::factorization& F) {
    return static_cast<std::int64_t>(F.L.nonZeros()) -
           static_cast<std::int64_t>(F.L.cols());
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
    std::vector<Trip> t = apxchol::test::triplets(L);
    t.push_back({0, 15, -1.0});
    t.push_back({15, 0, -1.0});
    const Sparse bad = apxchol::test::from_triplets(16, 16, t);

    const input_scan s = scan_input(bad);
    EXPECT_EQ(s.deficient_rows, 2);
    EXPECT_NEAR(s.worst_deficit, -1.0, 1e-12);
    EXPECT_GE(s.worst_row, 0);
}

// ── is_graph_adjacency: the rule shared by --convert-input and as_operator ──

TEST(AsOperatorRule, ValidOperatorsPassUnchanged) {
    Sparse sddm = grid_laplacian(6, 6);
    for (int i = 0; i < sddm.rows; ++i) add_entry(sddm, i, i, 0.05);
    Sparse nearly = grid_laplacian(6, 6);   // lumpable positive off-diagonals
    set_entry(nearly, 0, 1, 0.25);
    set_entry(nearly, 1, 0, 0.25);
    const Sparse diagonal = apxchol::test::identity(3);  // a positive diagonal-only matrix
    for (const Sparse& M : {grid_laplacian(6, 6), sddm, nearly, diagonal})
        EXPECT_FALSE(is_graph_adjacency(M));
}

TEST(AsOperatorRule, NonnegativeOffDiagonalsAreAGraph) {
    EXPECT_TRUE(is_graph_adjacency(grid_adjacency(6, 6, 2.5)));
    Sparse loops = grid_adjacency(6, 6);    // self-loops, as a pattern file has
    for (int i = 0; i < loops.rows; ++i) set_entry(loops, i, i, 1.0);
    EXPECT_TRUE(is_graph_adjacency(loops));
}

TEST(AsOperatorRule, AsymmetricOrNonFiniteInputRaises) {
    // edge 0 -> 1 without 1 -> 0
    const std::vector<Trip> t{{0, 1, 1.0}, {1, 2, 1.0}, {2, 1, 1.0}};
    const Sparse asymmetric = apxchol::test::from_triplets(3, 3, t);
    Sparse nan = grid_adjacency(3, 3);
    set_entry(nan, 0, 1, std::numeric_limits<double>::quiet_NaN());
    Sparse inf = grid_adjacency(3, 3);
    set_entry(inf, 1, 0, std::numeric_limits<double>::infinity());
    set_entry(inf, 0, 1, std::numeric_limits<double>::infinity());
    for (const auto& [M, reason] : {std::pair{asymmetric, "differ from their transpose partner"},
                                    std::pair{nan, "NaN or infinite"},
                                    std::pair{inf, "NaN or infinite"}}) {
        try {
            (void)is_graph_adjacency(M);
            ADD_FAILURE() << "expected a refusal: " << reason;
        } catch (const std::invalid_argument& e) {
            const std::string msg = e.what();
            EXPECT_NE(msg.find("needs a square, finite, symmetric matrix"), std::string::npos)
                << msg;
            EXPECT_NE(msg.find(reason), std::string::npos) << msg;
        }
    }
}

TEST(AsOperatorRule, AnyNegativeOffDiagonalLeavesTheMatrixToTheSolver) {
    // Mixed signs, and an operator beyond the default lumping ceiling, are
    // used as given; only the solver decides whether they are valid.
    const std::vector<Trip> t{{0, 1, +1.0}, {1, 0, +1.0}, {1, 2, -1.0}, {2, 1, -1.0}};
    const Sparse mixed = apxchol::test::from_triplets(3, 3, t);
    Sparse heavy = grid_laplacian(4, 4);
    set_entry(heavy, 0, 5, 20.0);   // 45% positive mass
    set_entry(heavy, 5, 0, 20.0);
    for (int i = 0; i < heavy.rows; ++i) add_entry(heavy, i, i, 5.0);
    EXPECT_FALSE(is_graph_adjacency(mixed));
    EXPECT_FALSE(is_graph_adjacency(heavy));
}

// ── describe_input ───────────────────────────────────

TEST(MtxInputDescribe, NamesTheChosenReading) {
    const std::string adj =
        describe_input(true, scan_input(grid_adjacency(4, 4)), "r");
    EXPECT_NE(adj.find("ADJACENCY"), std::string::npos) << adj;
    EXPECT_NE(adj.find("L = D - A"), std::string::npos) << adj;

    const std::string lap =
        describe_input(false, scan_input(grid_laplacian(4, 4)), "r");
    EXPECT_NE(lap.find("LAPLACIAN/SDDM"), std::string::npos) << lap;
    EXPECT_NE(lap.find("pure Laplacian"), std::string::npos) << lap;
}

// ── adjacency_laplacian ───────────────────────────

TEST(AdjacencyToLaplacian, BuildsDegreeMinusAdjacency) {
    Sparse L = grid_adjacency(4, 5, 2.0);
    L = adjacency_laplacian(L);

    ASSERT_EQ(L.rows, 20);
    for (int k = 0; k < L.cols; ++k) {
        double colsum = 0.0;
        for (auto p = L.col_ptr[k]; p < L.col_ptr[k + 1]; ++p) {
            colsum += L.values[p];
            if (L.row_idx[p] == k) EXPECT_GT(L.values[p], 0.0);
            else                   EXPECT_DOUBLE_EQ(L.values[p], -2.0);
        }
        EXPECT_NEAR(colsum, 0.0, 1e-12);        // pure Laplacian
    }
    // Symmetric.
    const Sparse asym = apxchol::test::combine(1.0, apxchol::test::transpose(L), -1.0, L);
    EXPECT_NEAR(apxchol::test::norm(asym.values), 0.0, 1e-12);
}

TEST(AdjacencyToLaplacian, DropsSelfLoops) {
    Sparse A = path_adjacency(5);
    set_entry(A, 2, 2, 7.0);
    A = adjacency_laplacian(A);

    for (auto p = A.col_ptr[2]; p < A.col_ptr[3]; ++p)
        if (A.row_idx[p] == 2) {
            EXPECT_DOUBLE_EQ(A.values[p], 2.0);   // degree 2, self-loop ignored
        }
}

TEST(AdjacencyToLaplacian, IsDegreeMinusAdjacencyWithSelfLoopsCancelling) {
    // L = D - M with D the row sums of M, including the self-loop itself.
    Sparse M = grid_adjacency(3, 4, 1.5);
    set_entry(M, 0, 0, 4.0);
    set_entry(M, 5, 5, 0.5);
    const std::vector<double> rowsum = apxchol::test::multiply(M, std::vector<double>(M.cols, 1.0));
    std::vector<Trip> degree;
    for (int i = 0; i < M.rows; ++i)
        if (rowsum[i] != 0.0) degree.push_back({i, i, rowsum[i]});
    const Sparse expected = apxchol::test::combine(
        1.0, apxchol::test::from_triplets(M.rows, M.cols, degree), -1.0, M);
    const Sparse diff = apxchol::test::combine(1.0, adjacency_laplacian(M), -1.0, expected);
    EXPECT_NEAR(apxchol::test::norm(diff.values), 0.0, 1e-12);
}

TEST(ComponentRhsProjection, LeavesConnectedRhsByteIdentical) {
    const Sparse L = grid_laplacian(4, 5);
    std::vector<double> b(L.rows);
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = -2.0 + 5.0 * double(i) / double(b.size() - 1);
    const std::vector<double> original = b;

    EXPECT_EQ(project_laplacian_rhs_components(L, b), 1);
    EXPECT_EQ(b, original);
}

TEST(ComponentRhsProjection, CentersEveryDisconnectedComponent) {
    std::vector<Trip> edges{
        {0, 1, 1.0}, {1, 0, 1.0}, {1, 2, 1.0}, {2, 1, 1.0},
        {3, 4, 1.0}, {4, 3, 1.0}, {4, 5, 1.0}, {5, 4, 1.0}};
    const Sparse L = adjacency_laplacian(apxchol::test::from_triplets(6, 6, edges));
    std::vector<double> b{1.0, 2.0, 6.0, -3.0, 4.0, 8.0};

    EXPECT_EQ(project_laplacian_rhs_components(L, b), 2);
    EXPECT_NEAR(b[0] + b[1] + b[2], 0.0, 1e-15);
    EXPECT_NEAR(b[3] + b[4] + b[5], 0.0, 1e-15);
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
    constexpr std::int64_t kEdges = 2 * 20 * 19;   // 20x20 4-neighbour grid
    const Sparse A = grid_adjacency(20, 20);
    ASSERT_EQ(A.nnz(), 2 * kEdges);           // both triangles stored

    for (auto call : {+[](const Sparse& M) {
                          (void)apxchol::factorize(apxchol::test::csc(M));
                      },
                      +[](const Sparse& M) {
                          (void)apxchol::solve(
                              apxchol::test::csc(M),
                              apxchol::test::random_rhs(M.rows),
                              1e-8, 200);
                      }}) {
        try {
            call(A);
            FAIL() << "expected the operator contract to refuse an adjacency matrix";
        } catch (const std::invalid_argument& e) {
            const std::string msg = e.what();
            EXPECT_NE(msg.find("positive diagonal"), std::string::npos) << msg;
            EXPECT_NE(msg.find("ADJACENCY"), std::string::npos) << msg;
            EXPECT_NE(msg.find("apxchol.as_operator(M)"), std::string::npos) << msg;
            EXPECT_NE(msg.find("--convert-input"), std::string::npos) << msg;
            EXPECT_NE(msg.find("apxchol_as_operator(M)"), std::string::npos) << msg;
        }
    }
}

TEST(AdjacencyInputRegression, ConvertedAdjacencyHasPlausibleFillAndConverges) {
    constexpr std::int64_t kEdges = 2 * 20 * 19;
    Sparse L = grid_adjacency(20, 20);
    L = adjacency_laplacian(L);

    const auto F = apxchol::factorize(apxchol::test::csc(L));
    // With the operator right, the randomized elimination emits sampled clique
    // edges, so the factor must hold strictly more than the fill-free count.
    EXPECT_GT(offdiag_factor_nnz(F), kEdges)
        << "factor shows no fill at all -- the clique sampler bailed out again";

    const std::vector<double> b = apxchol::test::random_rhs(L.rows);
    const auto res = apxchol::solve(apxchol::test::csc(L), b, 1e-8, 200);
    EXPECT_GT(res.iterations, 0);
    EXPECT_LT(res.residual, 1e-8);
}

TEST(AdjacencyInputRegression, ConversionRoutesAnAdjacencyMatrixToConvergence) {
    // The full CLI decision path, minus the file reading: resolve ->
    // convert -> solve.
    Sparse M = grid_adjacency(16, 16);
    ASSERT_TRUE(is_graph_adjacency(M));
    M = adjacency_laplacian(M);

    const std::vector<double> b = apxchol::test::random_rhs(M.rows);
    const auto res = apxchol::solve(apxchol::test::csc(M), b, 1e-8, 200);
    EXPECT_GT(res.iterations, 0);
    EXPECT_LT(res.residual, 1e-8);
}

TEST(AdjacencyInputRegression, SddmInputIsStillSolvedAsAnOperator) {
    // Requirement: SDDM (positive diagonal excess) must keep working, and must
    // NOT be rebuilt as a Laplacian -- that would discard the excess.
    Sparse L = grid_laplacian(16, 16);
    for (int i = 0; i < L.rows; ++i) add_entry(L, i, i, 0.05);

    EXPECT_EQ(scan_input(L).excess_rows, L.rows);
    ASSERT_FALSE(is_graph_adjacency(L));

    const std::vector<double> b = apxchol::test::random_rhs(L.rows);
    const auto res = apxchol::solve(apxchol::test::csc(L), b, 1e-8, 200);
    EXPECT_GT(res.iterations, 0);
    EXPECT_LT(res.residual, 1e-8);
}
