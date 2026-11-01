// Tests for the operator class apxchol asserts (include/apxchol/operator_class.h)
// and for the M-matrix lumping that repairs a nearly-SDDM SPD operator.
//
// Two things are being pinned down here:
//
//  1. The PRECONDITION. The library states the class it is defined on and
//     names which condition a matrix violates, with counts -- it does not
//     sniff at what the caller probably meant. Reading a matrix as a graph is
//     an explicit request (detail::is_graph_adjacency, shared by the CLI's
//     --convert-input and the bindings' as_operator).
//
//  2. The TRANSFORM. Lumping is applied ONLY to the matrix the preconditioner
//     is built from. `OperatorIsBitIdenticalAfterPreconditionerConstruction`
//     is the load-bearing test: PCG must keep applying the operator the caller
//     passed, or the residual it reports is for a different system.
#include <gtest/gtest.h>
#include "csc_test_util.h"
#include "grid_laplacian.h"
#include "test_rhs.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <limits>
#include <string>
#include <vector>

#include "apxchol/operator_class.h"
#include "apxchol/solver/factorization.h"
#include "apxchol/solver/solve.h"
#include "apxchol/solver/detail/cpu_route.h"

using apxchol::kDefaultLumpMassCeiling;
using apxchol::detail::operator_scan;
using apxchol::detail::operator_view;
using apxchol::detail::require_operator;
using apxchol::detail::scan_operator;
using apxchol::test::require_operator;
using apxchol::test::scan_operator;

namespace {

using Sparse = apxchol::test::sparse_matrix;
using Trip   = apxchol::test::triplet;
using apxchol::test::to_csc;
using apxchol::test::entry;
using apxchol::test::set_entry;

std::int64_t lump_positive_offdiagonals(Sparse& A) { return apxchol::test::lump_in_place(A); }

Sparse assemble(std::int64_t n, const std::vector<Trip>& t) {
    return apxchol::test::from_triplets(n, n, t);
}

/// B x with B the symmetric matrix A's lower triangle defines.
std::vector<double> lower_symmetric_times(const Sparse& A, std::span<const double> x) {
    std::vector<double> y(x.size(), 0.0);
    for (std::int64_t j = 0; j < A.cols; ++j)
        for (auto p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p) {
            const std::int64_t i = A.row_idx[p];
            if (i < j) continue;
            y[i] += A.values[p] * x[j];
            if (i != j) y[j] += A.values[p] * x[i];
        }
    return y;
}

/// Assembled Laplacian of a rows x cols 4-neighbour grid, both triangles
/// stored. Pure Laplacian: every row sums to zero.
using apxchol::test::grid_laplacian;

const apxchol::solve_options on_cpu{.backend = apxchol::solve_backend::cpu};

/// Grid Laplacian + a uniform diagonal excess: a genuine SDDM operator.
Sparse grid_sddm(int rows, int cols, double shift = 0.05) {
    Sparse L = grid_laplacian(rows, cols);
    for (std::int64_t i = 0; i < L.rows; ++i) entry(L, i, i) += shift;
    return L;
}

/// SDDM operator with `flips` of its edges given a POSITIVE off-diagonal of the
/// same magnitude. |a_ij| is unchanged, so the matrix stays strictly diagonally
/// dominant -- SPD, symmetric, and NOT an M-matrix. This is the "nearly SDDM"
/// class lumping exists for.
Sparse nearly_sddm(int rows, int cols, int flips, double shift = 0.05) {
    Sparse A = grid_sddm(rows, cols, shift);
    int done = 0;
    // Deterministic choice: the first `flips` strictly-lower entries in
    // column-major order, mirrored to keep the matrix symmetric.
    std::vector<std::pair<int, int>> picked;
    for (std::int64_t k = 0; k < A.cols && done < flips; ++k)
        for (auto p = A.col_ptr[k]; p < A.col_ptr[k + 1]; ++p)
            if (A.row_idx[p] > k) {
                picked.emplace_back(A.row_idx[p], static_cast<int>(k));
                if (++done == flips) break;
            }
    for (auto [i, j] : picked) {
        const double v = -A.coeff(i, j);
        entry(A, i, j) = v;
        entry(A, j, i) = v;
    }
    return A;
}

/// Tiny strictly diagonally-dominant operator for probing the CPU route's SpMV
/// storage. `upper_01_delta` changes only A(0,1); values within the operator
/// symmetry tolerance are accepted, while the operator must still source both
/// mirrored slots from the lower triangle's A(1,0).
Sparse spmv_probe_sddm(double upper_01_delta = 0.0) {
    std::vector<Trip> t{
        {0, 0, 4.0}, {1, 1, 5.0}, {2, 2, 6.0},
        {1, 0, -1.0}, {0, 1, -1.0 + upper_01_delta},
        {2, 0, -0.5}, {0, 2, -0.5},
        {2, 1, -1.25}, {1, 2, -1.25},
    };
    return assemble(3, t);
}

std::vector<double> spmv_probe_x() { return {1048576.0, 2.0, -4.0}; }

std::vector<double> spmv_probe_b() {
    // Exactly B*x in binary arithmetic for B the symmetric matrix of
    // spmv_probe_sddm()'s lower triangle. Large x[0] makes a mistaken use of
    // the tolerated upper-triangle perturbation plainly observable.
    return {4194304.0, -1048561.0, -524314.5};
}

/// Compressed, sorted CSC with duplicate off-diagonal coordinates whose
/// multiplicity errors cancel in the global upper/lower counts:
///   (1,0) has two lower entries and one upper partner;
///   (2,0) has one lower entry and two upper partners.
/// The dense sums are symmetric, but direct array copying is not equivalent to
/// reading the lower triangle unless the fast path rejects the representation.
Sparse spmv_balanced_duplicate_multiplicity_sddm() {
    Sparse A(3, 3);
    A.col_ptr = {0, 4, 6, 9};
    A.row_idx = {
        0, 1, 1, 2,  // column 0: diagonal + canonical lower entries
        0, 1,        // column 1: one upper partner + diagonal
        0, 0, 2,     // column 2: two upper partners + diagonal
    };
    A.values = {
        4.0, -0.25, -0.75, -2.0,
        -1.0, 4.0,
        -1.0, -1.0, 4.0,
    };
    return A;
}

using apxchol::test::dense_matrix;
dense_matrix dense(const Sparse& A) { return apxchol::test::to_dense(A); }

/// Byte-exact snapshot of a sparse matrix's three storage arrays.
struct raw_snapshot {
    std::vector<std::int32_t> outer, inner;
    std::vector<double> vals;
    std::int64_t rows = 0, cols = 0;

    explicit raw_snapshot(const Sparse& A)
        : outer(A.col_ptr), inner(A.row_idx), vals(A.values), rows(A.rows), cols(A.cols) {}

    bool identical_to(const Sparse& A) const {
        return rows == A.rows && cols == A.cols && outer.size() == A.col_ptr.size() &&
               inner.size() == A.row_idx.size() && vals.size() == A.values.size() &&
               std::memcmp(outer.data(), A.col_ptr.data(), outer.size() * sizeof(std::int32_t)) == 0 &&
               std::memcmp(inner.data(), A.row_idx.data(), inner.size() * sizeof(std::int32_t)) == 0 &&
               std::memcmp(vals.data(), A.values.data(), vals.size() * sizeof(double)) == 0;
    }
};

// Zero-iteration residual check of x0 on a CPU route built from A and F.
apxchol::solve_result unvalidated_route_check(const Sparse& A, apxchol::factorization F,
                                              const std::vector<double>& b,
                                              const std::vector<double>& x0) {
    const auto input = to_csc(A);
    const apxchol::detail::cpu_route route(input, {}, nullptr, &F);
    apxchol::solve_result result;
    std::vector<double> x(b.size());
    const auto n = static_cast<std::int64_t>(b.size());
    const apxchol::detail::const_matrix_ref initial(x0.data(), n);
    route.solve_columns(apxchol::detail::const_matrix_ref(b.data(), n),
                        apxchol::detail::matrix_ref(x.data(), n), &initial,
                        {&result, 1}, nullptr, 1e-15, 0);
    return result;
}

} // namespace

// ── the transform ───────────────────────────────────────────────────────────

TEST(MMatrixLumping, PreservesEveryRowSumExactly) {
    Sparse A = nearly_sddm(6, 7, 9);
    const std::vector<double> ones(static_cast<std::size_t>(A.rows), 1.0);
    const std::vector<double> before = apxchol::test::multiply(A, ones);
    ASSERT_GT(lump_positive_offdiagonals(A), 0);
    const std::vector<double> after = apxchol::test::multiply(A, ones);
    // A 1 = Â 1 because every summand of the transform is
    // a_ij (e_i - e_j)(e_i - e_j)^T and (e_i - e_j)^T 1 = 0.
    EXPECT_LT(apxchol::test::max_abs(apxchol::test::difference(after, before)), 1e-12);
}

TEST(MMatrixLumping, LeavesNoPositiveOffDiagonalAndStaysSymmetric) {
    Sparse A = nearly_sddm(6, 7, 9);
    const std::int64_t removed = lump_positive_offdiagonals(A);
    EXPECT_EQ(removed, 18);                       // 9 pairs, both triangles

    const operator_scan s = scan_operator(A);
    EXPECT_EQ(s.offdiag_pos, 0);
    EXPECT_EQ(s.asymmetric, 0);
    EXPECT_EQ(s.bad_diag, 0);
    EXPECT_LT(apxchol::test::norm(apxchol::test::combine(1, apxchol::test::transpose(A), -1, A).values),
              1e-15);
}

TEST(MMatrixLumping, IsIdempotent) {
    Sparse A = nearly_sddm(5, 5, 6);
    ASSERT_GT(lump_positive_offdiagonals(A), 0);
    const Sparse once = A;
    EXPECT_EQ(lump_positive_offdiagonals(A), 0);  // nothing left to move
    EXPECT_LT(apxchol::test::norm(apxchol::test::combine(1, A, -1, once).values),
              0.0 + 1e-300);                       // and nothing changed
}

TEST(MMatrixLumping, IsANoOpOnAnMMatrix) {
    // The overwhelmingly common case: a Laplacian or SDDM operator must come
    // through bit-identical and cost nothing.
    for (Sparse A : {grid_laplacian(8, 8), grid_sddm(8, 8)}) {
        const raw_snapshot snap(A);
        EXPECT_EQ(lump_positive_offdiagonals(A), 0);
        EXPECT_TRUE(snap.identical_to(A));
    }
}

TEST(MMatrixLumping, MatchesTheRankOneIdentity) {
    // Â = A + Σ_{pairs a_ij > 0} a_ij (e_i - e_j)(e_i - e_j)^T -- the identity
    // the whole soundness argument rests on.
    const Sparse A = nearly_sddm(5, 6, 7);
    Sparse H = A;
    ASSERT_GT(lump_positive_offdiagonals(H), 0);

    dense_matrix expected = dense(A);
    for (std::int64_t k = 0; k < A.cols; ++k)
        for (auto p = A.col_ptr[k]; p < A.col_ptr[k + 1]; ++p) {
            const std::int64_t i = A.row_idx[p];
            const double w = A.values[p];
            if (i > k && w > 0.0) {                // + w (e_i - e_k)(e_i - e_k)^T
                expected(i, i) += w;
                expected(k, k) += w;
                expected(i, k) -= w;
                expected(k, i) -= w;
            }
        }
    EXPECT_LT(apxchol::test::max_abs_difference(dense(H), expected), 1e-12);
}

TEST(MMatrixLumping, NeverWeakensDiagonalDominance) {
    const Sparse A = nearly_sddm(6, 6, 8);
    Sparse H = A;
    ASSERT_GT(lump_positive_offdiagonals(H), 0);

    const dense_matrix Ad = dense(A), Hd = dense(H);
    auto margin = [](const dense_matrix& M, std::int64_t i) {
        double off = 0;
        for (std::int64_t j = 0; j < M.cols; ++j)
            if (j != i) off += std::abs(M(i, j));
        return M(i, i) - off;
    };
    for (std::int64_t i = 0; i < Ad.rows; ++i)
        EXPECT_GE(margin(Hd, i), margin(Ad, i) - 1e-12) << "row " << i;
}

TEST(MMatrixLumping, LumpedMatrixIsPositiveDefiniteWhenTheOriginalIs) {
    // A ⪯ Â because Â - A is a sum of 2 x 2 blocks w [1 -1; -1 1] with w > 0,
    // each positive semidefinite. Checked entrywise, and Â's positive
    // definiteness outright by a dense Cholesky factorization.
    const Sparse A = nearly_sddm(5, 5, 6, /*shift=*/0.1);
    Sparse H = A;
    ASSERT_GT(lump_positive_offdiagonals(H), 0);
    ASSERT_TRUE(apxchol::test::positive_definite(dense(A))) << "test matrix is not SPD";

    // Read the block weights off Â - A: w = -(Â - A)(i, j) for each i > j.
    const dense_matrix D = dense(apxchol::test::combine(1, H, -1, A));
    const std::int64_t n = D.rows;
    dense_matrix blocks(n, n);
    for (std::int64_t j = 0; j < n; ++j)
        for (std::int64_t i = j + 1; i < n; ++i) {
            const double w = -D(i, j);
            if (w == 0.0) continue;
            EXPECT_GT(w, 0.0) << "(" << i << ", " << j << ")";
            blocks(i, i) += w;
            blocks(j, j) += w;
            blocks(i, j) -= w;
            blocks(j, i) -= w;
        }
    EXPECT_LT(apxchol::test::max_abs_difference(D, blocks), 1e-12);
    EXPECT_TRUE(apxchol::test::positive_definite(dense(H)));
}

// ── the precondition ────────────────────────────────────────────────────────

TEST(OperatorContract, AcceptsLaplacianAndSddm) {
    EXPECT_NO_THROW((void)require_operator(grid_laplacian(6, 6)));
    EXPECT_NO_THROW((void)require_operator(grid_sddm(6, 6)));
    Sparse empty(8, 8);                       // n > 0, no entries at all
    EXPECT_NO_THROW((void)require_operator(empty));
    EXPECT_NO_THROW((void)require_operator(apxchol::test::identity(4)));
}

TEST(OperatorContract, RejectsNonSquare) {
    const Sparse A(3, 5);
    try {
        (void)require_operator(A);
        FAIL() << "expected a non-square matrix to be refused";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("not square"), std::string::npos)
            << e.what();
    }
}

TEST(OperatorContract, RejectsAValueAsymmetryNamingTheWorstEntry) {
    // Both (1, 0) and (0, 1) are stored; only one of them is changed, so the
    // VALUE test sees it and can name the entry.
    Sparse A = grid_laplacian(4, 4);
    entry(A, 1, 0) = -0.5;
    const operator_scan s = scan_operator(A);
    EXPECT_EQ(s.asymmetric, 1);
    EXPECT_EQ(s.worst_asym_row, 1);
    EXPECT_EQ(s.worst_asym_col, 0);
    try {
        (void)require_operator(A);
        FAIL() << "expected an asymmetric matrix to be refused";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("Condition FAILED: symmetry"), std::string::npos) << msg;
        EXPECT_NE(msg.find("(1, 0)"), std::string::npos) << msg;
        EXPECT_NE(msg.find("-0.5"), std::string::npos) << msg;
    }
}

TEST(OperatorContract, RejectsAStructuralAsymmetryWithNoStoredPartner) {
    // An entry whose partner is not stored at all. The value test walks the
    // LOWER triangle only, so an upper-only entry is invisible to it -- the
    // upper/lower COUNT imbalance is what catches this one, which is why the
    // message carries a count but no witness entry.
    Sparse A = grid_laplacian(4, 4);
    set_entry(A, 2, 5, -0.5);                 // (5, 2) does not exist
    const operator_scan s = scan_operator(A);
    EXPECT_EQ(s.asymmetric, 1);
    EXPECT_EQ(s.worst_asym_row, -1);          // no witness for this path
    try {
        (void)require_operator(A);
        FAIL() << "expected an asymmetric matrix to be refused";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("Condition FAILED: symmetry"),
                  std::string::npos) << e.what();
    }
}

TEST(OperatorContract, RejectsAOneSidedMatrix) {
    // A `general`-header file holding one triangle only: the value test cannot
    // see the missing partners, the count test can.
    std::vector<Trip> t{{1, 0, -1.0}, {2, 1, -1.0}, {0, 0, 1.0},
                        {1, 1, 2.0},  {2, 2, 1.0}};
    const Sparse A = assemble(3, t);
    try {
        (void)require_operator(A);
        FAIL() << "expected a one-sided matrix to be refused";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("Condition FAILED: symmetry"),
                  std::string::npos) << e.what();
    }
}

TEST(OperatorContract, RejectsANonPositiveDiagonalWithCounts) {
    Sparse A = grid_sddm(4, 4);
    entry(A, 3, 3) = -1.0;                    // e_3^T A e_3 < 0: not PSD
    try {
        (void)require_operator(A);
        FAIL() << "expected a non-positive diagonal to be refused";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("Condition FAILED: positive diagonal"),
                  std::string::npos) << msg;
        EXPECT_NE(msg.find("1 of 16 non-empty rows"), std::string::npos) << msg;
        EXPECT_NE(msg.find("first at row 3"), std::string::npos) << msg;
        // NOT the adjacency case: no conversion suggestion, that would mislead.
        EXPECT_EQ(msg.find("ADJACENCY"), std::string::npos) << msg;
    }
}

TEST(OperatorContract, AdjacencyMatrixGetsTheConversionSuggestion) {
    // No diagonal anywhere + positive off-diagonals: the adjacency signature,
    // reached as a special case of the diagonal condition.
    Sparse L = grid_laplacian(5, 5);
    std::vector<Trip> t;                      // adjacency = -offdiag(L)
    for (const Trip& e : apxchol::test::triplets(L))
        if (e.row != e.col) t.push_back({e.row, e.col, -e.value});
    const Sparse A = assemble(L.rows, t);

    try {
        (void)require_operator(A);
        FAIL() << "expected an adjacency matrix to be refused";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("Condition FAILED: positive diagonal"),
                  std::string::npos) << msg;
        EXPECT_NE(msg.find("ADJACENCY"), std::string::npos) << msg;
        EXPECT_NE(msg.find("apxchol.as_operator(M)"), std::string::npos) << msg;
        EXPECT_NE(msg.find("apxchol_as_operator(M)"), std::string::npos) << msg;
        EXPECT_NE(msg.find("--convert-input"), std::string::npos) << msg;
    }
}

TEST(OperatorContract, ReportsDominanceViolationsWithoutEnforcingThem) {
    // G3_circuit / apache2 in miniature: a Z-matrix with rows the diagonal
    // does not cover. It must be ACCEPTED -- those matrices converge today.
    Sparse A = grid_laplacian(4, 4);
    set_entry(A, 0, 15, -1.0);
    set_entry(A, 15, 0, -1.0);

    const operator_scan s = require_operator(A);
    EXPECT_EQ(s.deficient_rows, 2);
    EXPECT_NEAR(s.worst_row_sum, -1.0, 1e-12);
    EXPECT_GE(s.worst_row, 0);
}

TEST(OperatorContract, PositiveOffDiagonalsAreLumpedNotRejected) {
    const Sparse A = nearly_sddm(6, 6, 5);
    const operator_scan s = require_operator(A);
    EXPECT_EQ(s.offdiag_pos, 10);
    EXPECT_LT(s.positive_mass_fraction(), kDefaultLumpMassCeiling);

    const auto input = to_csc(A);
    const operator_view op(input);
    EXPECT_EQ(op.lumped(), 10);
    EXPECT_EQ(scan_operator(op.matrix()).offdiag_pos, 0);
    // ... and the caller's matrix is untouched.
    EXPECT_EQ(scan_operator(A).offdiag_pos, 10);
}

TEST(OperatorContract, RefusesPastTheLumpMassCeiling) {
    // Every off-diagonal positive: lumping would leave a purely diagonal
    // matrix, i.e. a Jacobi preconditioner bought at the price of a full
    // approximate-Cholesky setup.
    Sparse A = grid_sddm(6, 6, 0.5);
    for (std::int64_t k = 0; k < A.cols; ++k)
        for (auto p = A.col_ptr[k]; p < A.col_ptr[k + 1]; ++p)
            if (A.row_idx[p] != k) A.values[p] = -A.values[p];
    ASSERT_GT(scan_operator(A).positive_mass_fraction(), kDefaultLumpMassCeiling);

    try {
        (void)require_operator(A);
        FAIL() << "expected a matrix past the lumping ceiling to be refused";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("Condition FAILED: non-positive off-diagonals"),
                  std::string::npos) << msg;
        EXPECT_NE(msg.find("of the off-diagonal mass"), std::string::npos) << msg;
        EXPECT_NE(msg.find("lump_max_mass"), std::string::npos) << msg;
    }
}

TEST(OperatorContract, LumpingCeilingIsExplicitPerCall) {
    const auto A = nearly_sddm(6, 6, 1);
    const auto input = to_csc(A);
    EXPECT_THROW(operator_view(input, 0), std::invalid_argument);
    EXPECT_EQ(operator_view(input).lumped(), 2);
    EXPECT_THROW(operator_view(input, 0), std::invalid_argument);

    apxchol::factor_options options;
    options.lump_max_mass = 0;
    EXPECT_THROW(apxchol::factorize(apxchol::test::csc(A), options), std::invalid_argument);
    options.lump_max_mass = 0.25;
    EXPECT_EQ(apxchol::factorize(apxchol::test::csc(A), options).lumped_offdiag, 2);
}

TEST(OperatorContract, RegularizationIsExplicitPerFactorization) {
    const auto A = grid_laplacian(6, 6);
    apxchol::factor_options options;
    EXPECT_FALSE(apxchol::factorize(apxchol::test::csc(A), options).sddm);
    options.regularization = 1e-8;
    EXPECT_TRUE(apxchol::factorize(apxchol::test::csc(A), options).sddm);
    EXPECT_TRUE(apxchol::factorize(apxchol::test::csc(A), apxchol::detail::block_greedy_partitioner{}, options).sddm);
    options.regularization = 0;
    EXPECT_FALSE(apxchol::factorize(apxchol::test::csc(A), options).sddm);
}

TEST(OperatorContract, NumericalOptionsRejectInvalidValues) {
    const Sparse empty(0, 0);
    const auto empty_csc = to_csc(empty);
    for (double value : {-1.0, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
        SCOPED_TRACE(value);
        apxchol::factor_options options;
        options.regularization = value;
        EXPECT_THROW(apxchol::factorize(apxchol::test::csc(empty), options), std::invalid_argument);
        options = {};
        options.lump_max_mass = value;
        EXPECT_THROW(apxchol::factorize(apxchol::test::csc(empty), options), std::invalid_argument);
        EXPECT_THROW(operator_view(empty_csc, value), std::invalid_argument);
    }
    EXPECT_THROW(operator_view(empty_csc, 1.01), std::invalid_argument);
}

TEST(OperatorContract, MassNotCountIsWhatTheCeilingMeasures) {
    // parabolic_fem in miniature: a THIRD of the off-diagonals are positive,
    // but they carry no mass, so the matrix is trivially lumpable. A ceiling
    // on the entry COUNT would have refused it.
    Sparse A = grid_sddm(8, 8);
    std::int64_t flipped = 0;
    for (std::int64_t k = 0; k < A.cols; ++k)
        for (auto p = A.col_ptr[k]; p < A.col_ptr[k + 1]; ++p)
            if (A.row_idx[p] != k && (A.row_idx[p] + k) % 3 == 0) {
                A.values[p] = 1e-9;
                ++flipped;
            }
    const operator_scan s = scan_operator(A);
    ASSERT_EQ(s.offdiag_pos, flipped);
    EXPECT_GT(static_cast<double>(s.offdiag_pos) / s.offdiag_nnz, 0.2);
    EXPECT_LT(s.positive_mass_fraction(), 1e-6);
    EXPECT_NO_THROW((void)require_operator(A));
}

// ── bit-identical triangles: the proof that lets consumers skip partner searches ──

TEST(OperatorScan, BitIdenticalTrianglesAreRecognised) {
    const operator_scan grid = scan_operator(grid_laplacian(7, 5));
    EXPECT_EQ(grid.asymmetric, 0);
    EXPECT_TRUE(grid.triangles_bit_identical);

    const operator_scan sddm = scan_operator(spmv_probe_sddm());
    EXPECT_EQ(sddm.asymmetric, 0);
    EXPECT_TRUE(sddm.triangles_bit_identical);
}

TEST(OperatorScan, ToleratedAsymmetryIsSymmetricButNotBitIdentical) {
    // Below kSymmetryRelTol: still accepted as symmetric, exactly as before,
    // but the two triangles differ in one value bit pattern, so no consumer
    // may substitute an upper entry for its lower partner.
    const Sparse A = spmv_probe_sddm(5e-11);
    ASSERT_NE(A.coeff(0, 1), A.coeff(1, 0));
    const operator_scan s = scan_operator(A);
    EXPECT_EQ(s.asymmetric, 0);
    EXPECT_FALSE(s.triangles_bit_identical);
    EXPECT_NO_THROW((void)require_operator(A));
}

TEST(OperatorScan, ValueAsymmetryKeepsItsCountAndWitness) {
    // The per-entry partner search now runs only after the fingerprints
    // disagree; it must report what the single-pass scan always reported.
    Sparse A = grid_laplacian(6, 6);
    entry(A, 3, 9) *= 1.5;            // upper entry of the (3, 9) pair
    entry(A, 20, 14) *= 3.0;          // lower entry of the (14, 20) pair: the worst one
    const operator_scan s = scan_operator(A);
    EXPECT_EQ(s.asymmetric, 2);
    EXPECT_FALSE(s.triangles_bit_identical);
    EXPECT_EQ(s.worst_asym_row, 20);
    EXPECT_EQ(s.worst_asym_col, 14);
    EXPECT_DOUBLE_EQ(s.worst_asym_a, -3.0);
    EXPECT_DOUBLE_EQ(s.worst_asym_b, -1.0);
    EXPECT_NEAR(s.worst_asym_rel, 2.0 / 3.0, 1e-15);
}

TEST(OperatorScan, SwappedValuesBetweenTwoPairsAreNotBitIdentical) {
    // Same multiset of VALUES in both triangles, same counts -- only the
    // coordinates disagree. The fingerprint is keyed by coordinates, so this
    // must not pass as bit-identical, and the partner search must flag both.
    const Sparse A = assemble(3, {
        {0, 0, 9.0}, {1, 1, 9.0}, {2, 2, 9.0},
        {1, 0, -1.0}, {0, 1, -2.0},
        {2, 0, -2.0}, {0, 2, -1.0},
    });
    const operator_scan s = scan_operator(A);
    EXPECT_FALSE(s.triangles_bit_identical);
    EXPECT_EQ(s.asymmetric, 2);
}

TEST(OperatorScan, ExplicitZerosDuplicatesAndUnsortedColumnsWithholdTheProof) {
    // Each of these is still a valid symmetric operator. None may carry the
    // proof: the graph builder gives stored zeros, duplicate coordinates and
    // unsorted columns their full structural checks.
    {   // explicit zero off-diagonal pair
        Sparse A = spmv_probe_sddm();
        entry(A, 2, 0) = 0.0;
        entry(A, 0, 2) = 0.0;
        ASSERT_EQ(A.nnz(), 9);
        const operator_scan s = scan_operator(A);
        EXPECT_EQ(s.asymmetric, 0);
        EXPECT_FALSE(s.triangles_bit_identical);
    }
    {   // duplicate coordinates, balanced between the triangles
        const Sparse A = spmv_balanced_duplicate_multiplicity_sddm();
        EXPECT_FALSE(scan_operator(A).triangles_bit_identical);
    }
    {   // signed zero against zero is a value-bit difference the scan ignores
        // for symmetry (explicit zeros carry no edge) but must not certify
        Sparse A = spmv_probe_sddm();
        entry(A, 2, 0) = -0.0;
        entry(A, 0, 2) = 0.0;
        EXPECT_FALSE(scan_operator(A).triangles_bit_identical);
    }
}

// ── the operator PCG applies is untouched ───────────────────────────────────

TEST(SpmvLrmBuild, AcceptedNearSymmetryStillUsesCanonicalLowerValues) {
    // require_operator intentionally accepts tiny representation noise. The
    // direct CSC-layout -> CSR path must not start applying A^T; it keeps
    // reading the lower triangle byte-for-byte.
    Sparse A = spmv_probe_sddm(5e-11);
    ASSERT_NE(A.coeff(0, 1), A.coeff(1, 0));
    ASSERT_EQ(scan_operator(A).asymmetric, 0);

    const std::vector<double> x0 = spmv_probe_x();
    const std::vector<double> b = spmv_probe_b();
    apxchol::solver slv(apxchol::test::csc(A), on_cpu);
    const auto res = slv.solve(b, 1e-15, 0, x0);
    EXPECT_EQ(res.iterations, 0);
    EXPECT_EQ(res.residual, 0.0);
}

TEST(SpmvLrmBuild, BitIdenticalFastPathAndWorkSplitApplyTheExactOperator) {
    // Hub-heavy SDDM with bit-identical triangles: the owned copy takes the
    // direct (no partner search) path and the SpMV splits its rows by stored
    // entries. Dyadic weights and integer vectors keep every product and sum
    // exact, so the residual at the true solution is exactly zero in any
    // summation order -- at every team size, if and only if the operator that
    // PCG applies is the caller's operator entry for entry.
    const int n = 3000;   // above the fused kernels' parallel threshold
    std::vector<Trip> t;
    std::vector<double> degree(n, 0.0);
    auto edge = [&](int a, int b, double w) {
        t.emplace_back(a, b, -w); t.emplace_back(b, a, -w);
        degree[a] += w; degree[b] += w;
    };
    for (int v = 1; v < n; ++v) edge(v, 0, 0.125 * (1 + v % 7));          // hub row 0
    for (int v = 2; v < n; v += 2) edge(v, 1, 0.25 * (1 + v % 3));        // hub row 1
    for (int v = 2; v + 1 < n; ++v) edge(v + 1, v, 0.5);                  // a path
    for (int v = 0; v < n; ++v) t.emplace_back(v, v, degree[v] + 1.0);
    const Sparse A = assemble(n, t);
    ASSERT_TRUE(scan_operator(A).triangles_bit_identical);

    std::vector<double> x0(n);
    for (int v = 0; v < n; ++v) x0[v] = static_cast<double>((v * 37) % 17 - 8);
    const std::vector<double> b = lower_symmetric_times(A, x0);

    const int prior = omp_get_max_threads();
    for (int threads : {1, 3, 16}) {
        omp_set_num_threads(threads);
        SCOPED_TRACE(threads);
        apxchol::solver slv(apxchol::test::csc(A), on_cpu);
        const auto res = slv.solve(b, 1e-15, 0, x0);
        EXPECT_EQ(res.iterations, 0);
        EXPECT_EQ(res.residual, 0.0);
    }
    omp_set_num_threads(prior);
}

TEST(SpmvLrmBuild, OwnedFp32CopySurvivesCallerMutationAndDestruction) {
    // Exact binary values select the fp32 operator. Prove the reusable solver
    // does not retain a reference to caller storage.
    auto A = std::make_unique<Sparse>(spmv_probe_sddm());

    const std::vector<double> x0 = spmv_probe_x();
    const std::vector<double> b = spmv_probe_b();
    apxchol::solver slv(apxchol::test::csc(*A), on_cpu);

    std::fill(A->values.begin(), A->values.end(), 0.0);
    const auto after_mutation = slv.solve(b, 1e-15, 0, x0);
    EXPECT_EQ(after_mutation.residual, 0.0);

    A.reset();
    const auto after_destruction = slv.solve(b, 1e-15, 0, x0);
    EXPECT_EQ(after_destruction.residual, 0.0);
}

TEST(SpmvLrmBuild, BalancedDuplicateMultiplicitiesUseGeneralFallback) {
    const Sparse A = spmv_balanced_duplicate_multiplicity_sddm();
    ASSERT_EQ(A.nnz(), 9);

    // A's lower-authoritative operator has unique dense values -1 and -2.
    // Supply a factor for that same operator; max_iter=0 makes this test probe
    // only the SpMV residual at x0, independent of preconditioner quality.
    const std::vector<Trip> canonical_entries{
        {0, 0, 4.0}, {1, 1, 4.0}, {2, 2, 4.0},
        {1, 0, -1.0}, {0, 1, -1.0},
        {2, 0, -2.0}, {0, 2, -2.0},
    };
    const Sparse canonical = assemble(3, canonical_entries);
    auto F = apxchol::factorize(apxchol::test::csc(canonical));
    const std::vector<double> x0{8.0, -4.0, 2.0};
    const std::vector<double> b = lower_symmetric_times(A, x0);

    // solver(A, F) rejects this input, so probe the unvalidated CPU route.
    const auto result = unvalidated_route_check(A, std::move(F), b, x0);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.residual, 0.0);
}

TEST(SpmvLrmBuild, UnsortedCompressedInputUsesGeneralFallback) {
    // Preserve the matrix values while introducing one inversion per column.
    const Sparse canonical = spmv_probe_sddm();
    auto F = apxchol::factorize(apxchol::test::csc(canonical));
    Sparse unsorted = canonical;
    for (std::int64_t col = 0; col < unsorted.cols; ++col) {
        const auto begin = unsorted.col_ptr[col];
        const auto end = unsorted.col_ptr[col + 1];
        ASSERT_GE(end - begin, 2);
        std::swap(unsorted.row_idx[begin], unsorted.row_idx[begin + 1]);
        std::swap(unsorted.values[begin], unsorted.values[begin + 1]);
        ASSERT_GT(unsorted.row_idx[begin], unsorted.row_idx[begin + 1]);
    }

    const std::vector<double> x0 = spmv_probe_x();
    // solver(A, F) rejects this input, so probe the unvalidated CPU route.
    const auto result = unvalidated_route_check(unsorted, std::move(F), spmv_probe_b(), x0);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.residual, 0.0);
}

TEST(SpmvLrmBuild, LowerOnlyRouteInputUsesGeneralFallback) {
    // cpu_route reads its operator through the lower triangle. A lower-only
    // input has no reusable CSC-as-CSR layout, so it must take the general
    // lower-reflection build and still apply the full symmetric operator.
    const Sparse A = spmv_probe_sddm();
    auto F = apxchol::factorize(apxchol::test::csc(A));
    std::vector<Trip> lower_entries = apxchol::test::triplets(A);
    std::erase_if(lower_entries, [](const Trip& e) { return e.row < e.col; });
    const Sparse lower = assemble(A.rows, lower_entries);
    ASSERT_LT(lower.nnz(), A.nnz());

    const std::vector<double> x0 = spmv_probe_x();
    // solver(A, F) rejects this input, so probe the unvalidated CPU route.
    const auto result = unvalidated_route_check(lower, std::move(F), spmv_probe_b(), x0);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.residual, 0.0);
}

TEST(SpmvLrmBuild, OneSidedExplicitZeroKeepsGeneralFallback) {
    // An explicit zero stored in one triangle passes operator validation but
    // leaves no reusable CSC-as-CSR layout, so it must take the general
    // lower-reflection build while the same operator without it is copied directly.
    const std::vector<Trip> entries{
        {0, 0, 4.0}, {1, 1, 4.0}, {2, 2, 4.0}, {3, 3, 4.0},
        {1, 0, -1.0}, {0, 1, -1.0}, {2, 1, -1.5}, {1, 2, -1.5},
        {3, 2, -0.5}, {2, 3, -0.5},
    };
    const Sparse A = assemble(4, entries);
    auto F = apxchol::factorize(apxchol::test::csc(A));
    auto F_for_direct = F;
    Sparse padded = A;
    set_entry(padded, 0, 3, 0.0);
    ASSERT_EQ(padded.nnz(), A.nnz() + 1);

    const std::vector<double> x0{8.0, -4.0, 2.0, 1.0};
    const std::vector<double> b = apxchol::test::multiply(A, x0);
    apxchol::solver direct(apxchol::test::csc(A), std::move(F_for_direct), on_cpu);
    apxchol::solver reflected(apxchol::test::csc(padded), std::move(F), on_cpu);
    const auto res = reflected.solve(b, 1e-15, 0, x0);
    EXPECT_EQ(res.iterations, 0);
    EXPECT_EQ(res.residual, 0.0);

    // More than the exact-x0 probe: with one identical factor installed in
    // both solvers, the direct and reflected operator representations must
    // drive the entire PCG recurrence bit-identically.
    const std::vector<double> rhs{1.0, -2.0, 3.0, 0.5};
    const auto direct_res = direct.solve(rhs, 1e-12, 50);
    const auto reflected_res = reflected.solve(rhs, 1e-12, 50);
    EXPECT_EQ(direct_res.iterations, reflected_res.iterations);
    EXPECT_EQ(direct_res.residual, reflected_res.residual);
    EXPECT_EQ(direct_res.x, reflected_res.x);
}

TEST(MMatrixLumping, OperatorIsBitIdenticalAfterPreconditionerConstruction) {
    // The contract that makes lumping safe: the transform belongs to the
    // PRECONDITIONER, never to the operator. If this ever fails, every
    // residual apxchol reports on a nearly-SDDM matrix is for a system the
    // caller did not ask about.
    Sparse A = nearly_sddm(20, 20, 12);
    const raw_snapshot snap(A);

    apxchol::solver slv(apxchol::test::csc(A), {.backend = apxchol::solve_backend::cpu});
    ASSERT_GT(slv.factor().lumped_offdiag, 0)
        << "the test matrix did not actually trigger lumping";
    EXPECT_TRUE(snap.identical_to(A))
        << "preconditioner construction modified the caller's operator";

    const std::vector<double> b = apxchol::test::random_rhs(A.rows);
    const auto res = slv.solve(b, 1e-8, 300);
    EXPECT_TRUE(snap.identical_to(A)) << "solving modified the caller's operator";

    // And the residual really is measured against the matrix passed in.
    const double rel = apxchol::test::relative_residual(A, res.x, b);
    EXPECT_LT(rel, 1e-8) << "reported residual " << res.residual;
}

// ── end to end ──────────────────────────────────────────────────────────────

TEST(MMatrixLumping, NearlySddmOperatorConvergesAgainstTheTrueOperator) {
    const Sparse A = nearly_sddm(30, 30, 40);
    ASSERT_GT(scan_operator(A).offdiag_pos, 0);

    const std::vector<double> b = apxchol::test::random_rhs(A.rows);
    apxchol::solver slv(apxchol::test::csc(A));
    EXPECT_EQ(slv.factor().lumped_offdiag, 80);   // 40 pairs
    const auto res = slv.solve(b, 1e-8, 300);
    EXPECT_GT(res.iterations, 0);
    EXPECT_LT(res.residual, 1e-8);
    EXPECT_LT(apxchol::test::relative_residual(A, res.x, b), 1e-7);
}

TEST(MMatrixLumping, PureLaplacianAndSddmAreUnchangedByTheFeature) {
    // The regression guard for everything that already worked: no lumping
    // triggered, no behaviour change, and the factor reports zero.
    for (const Sparse& A : {grid_laplacian(30, 30), grid_sddm(30, 30)}) {
        const raw_snapshot snap(A);
        const std::vector<double> b = apxchol::test::random_rhs(A.rows);
        apxchol::solver slv(apxchol::test::csc(A));
        EXPECT_EQ(slv.factor().lumped_offdiag, 0);
        const auto res = slv.solve(b, 1e-8, 300);
        EXPECT_GT(res.iterations, 0);
        EXPECT_LT(res.residual, 1e-8);
        EXPECT_TRUE(snap.identical_to(A));
    }
}

// ── the public CSC views ────────────────────────────────────────────────────

namespace {

// A grid Laplacian's arrays at the given index width.
template<class Index>
struct grid_arrays {
    std::vector<Index> col_ptr, row_idx;
    std::vector<double> values;
    std::int64_t n = 0;

    explicit grid_arrays(const Sparse& A)
        : col_ptr(A.col_ptr.begin(), A.col_ptr.end()), row_idx(A.row_idx.begin(), A.row_idx.end()),
          values(A.values), n(A.rows) {}
    apxchol::basic_csc_view<Index> view() const { return {n, col_ptr, row_idx, values}; }
};

template<class Index>
void expect_invalid_view(const grid_arrays<Index>& arrays, const char* fragment) {
    try {
        apxchol::solver slv(arrays.view(), on_cpu);
        FAIL() << "expected an invalid view to be refused: " << fragment;
    } catch (const std::invalid_argument& e) {
        const std::string message = e.what();
        EXPECT_EQ(message.rfind("apxchol: invalid CSC view: ", 0), 0u) << message;
        EXPECT_NE(message.find(fragment), std::string::npos) << message;
    }
}

} // namespace

TEST(CscView, BothIndexWidthsSolveBitIdentically) {
    const Sparse A = grid_sddm(9, 7);
    const grid_arrays<std::int32_t> narrow(A);
    const grid_arrays<std::int64_t> wide(A);
    const std::vector<double> b = apxchol::test::random_rhs(A.rows);
    apxchol::solver from_narrow(narrow.view(), on_cpu);
    apxchol::solver from_wide(wide.view(), on_cpu);
    const auto x_narrow = from_narrow.solve(b).x;
    const auto x_wide = from_wide.solve(b).x;
    EXPECT_EQ(x_narrow, x_wide);
    EXPECT_EQ(apxchol::solve(wide.view(), b, {}, {}, {}, on_cpu).x, x_wide);
    EXPECT_EQ(apxchol::factorize(narrow.view()).perm, apxchol::factorize(wide.view()).perm);
}

TEST(CscView, InvalidViewsAreRefusedAtTheBoundary) {
    const Sparse A = grid_sddm(4, 4);
    const grid_arrays<std::int32_t> valid(A);
    {
        auto bad = valid;
        bad.col_ptr.pop_back();
        expect_invalid_view(bad, "col_ptr has");
    }
    {
        auto bad = valid;
        bad.n = -1;
        expect_invalid_view(bad, "n is negative");
    }
    {
        auto bad = valid;
        bad.col_ptr[0] = 1;
        expect_invalid_view(bad, "col_ptr[0] must be 0");
    }
    {
        auto bad = valid;
        std::swap(bad.col_ptr[2], bad.col_ptr[3]);
        expect_invalid_view(bad, "col_ptr decreases after column 2");
    }
    {
        auto bad = valid;
        bad.values.pop_back();
        expect_invalid_view(bad, "row_idx and values must have");
    }
    {
        auto bad = valid;
        bad.row_idx[bad.col_ptr[3]] = static_cast<std::int32_t>(bad.n);
        expect_invalid_view(bad, "column 3 row 16 is outside [0, n)");
    }
    {
        auto bad = valid;
        bad.row_idx[bad.col_ptr[5]] = -1;
        expect_invalid_view(bad, "column 5 row -1 is outside [0, n)");
    }
    {
        auto bad = valid;
        std::swap(bad.row_idx[bad.col_ptr[6]], bad.row_idx[bad.col_ptr[6] + 1]);
        expect_invalid_view(bad, "out of order");
    }
    {
        auto bad = valid;
        bad.row_idx[bad.col_ptr[6] + 1] = bad.row_idx[bad.col_ptr[6]];
        expect_invalid_view(bad, "is repeated");
    }
    EXPECT_THROW((void)apxchol::factorize(apxchol::csc_view{2, {}, {}, {}}), std::invalid_argument);
}

TEST(CscView, DenseViewsAreChecked) {
    const Sparse A = grid_sddm(4, 4);
    apxchol::solver slv(apxchol::test::csc(A), on_cpu);
    const std::int64_t n = A.rows;
    std::vector<double> B(static_cast<std::size_t>(2 * n), 1.0), X(B.size());
    const apxchol::const_matrix_view rhs{B.data(), n, 2, n};
    EXPECT_THROW((void)slv.solve(rhs, {X.data(), n, 2, n - 1}), std::invalid_argument);
    EXPECT_THROW((void)slv.solve(rhs, {nullptr, n, 2, n}), std::invalid_argument);
    EXPECT_THROW((void)slv.solve({B.data(), n, -1, n}, {X.data(), n, 2, n}), std::invalid_argument);
    // A null warm start starts every column from zero.
    const auto cold = slv.solve(rhs, {X.data(), n, 2, n});
    std::vector<double> zeros(B.size(), 0.0), Y(B.size());
    const auto warm = slv.solve(rhs, {Y.data(), n, 2, n}, {}, {}, {zeros.data(), n, 2, n});
    EXPECT_EQ(X, Y);
    EXPECT_EQ(cold.columns[0].iterations, warm.columns[0].iterations);
}
