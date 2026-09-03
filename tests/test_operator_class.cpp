// Tests for the operator class apxchol asserts (include/apxchol/operator_class.h)
// and for the M-matrix lumping that repairs a nearly-SDDM SPD operator.
//
// Two things are being pinned down here:
//
//  1. The PRECONDITION. The library states the class it is defined on and
//     names which condition a matrix violates, with counts -- it does not
//     sniff at what the caller probably meant. (The CLI still guesses, out of
//     src/mtx_input.h, because it prints its guess and takes --input-kind.)
//
//  2. The TRANSFORM. Lumping is applied ONLY to the matrix the preconditioner
//     is built from. `OperatorIsBitIdenticalAfterPreconditionerConstruction`
//     is the load-bearing test: PCG must keep applying the operator the caller
//     passed, or the residual it reports is for a different system.
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

#include "apxchol/operator_class.h"
#include "apxchol/solver/factorization.h"
#include "apxchol/solver/solve.h"

using apxchol::kDefaultLumpMassCeiling;
using apxchol::lump_positive_offdiagonals;
using apxchol::operator_scan;
using apxchol::operator_view;
using apxchol::require_operator;
using apxchol::scan_operator;

namespace {

using Sparse = Eigen::SparseMatrix<double>;
using Trip   = Eigen::Triplet<double>;

/// Assembled Laplacian of a rows x cols 4-neighbour grid, both triangles
/// stored. Pure Laplacian: every row sums to zero.
Sparse grid_laplacian(int rows, int cols, double w = 1.0) {
    const int n = rows * cols;
    std::vector<Trip> t;
    std::vector<double> deg(n, 0.0);
    auto id = [cols](int r, int c) { return r * cols + c; };
    auto edge = [&](int a, int b) {
        t.emplace_back(a, b, -w);
        t.emplace_back(b, a, -w);
        deg[a] += w;
        deg[b] += w;
    };
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            if (r + 1 < rows) edge(id(r, c), id(r + 1, c));
            if (c + 1 < cols) edge(id(r, c), id(r, c + 1));
        }
    for (int i = 0; i < n; ++i) t.emplace_back(i, i, deg[i]);
    Sparse L(n, n);
    L.setFromTriplets(t.begin(), t.end());
    L.makeCompressed();
    return L;
}

/// Grid Laplacian + a uniform diagonal excess: a genuine SDDM operator.
Sparse grid_sddm(int rows, int cols, double shift = 0.05) {
    Sparse L = grid_laplacian(rows, cols);
    for (int i = 0; i < L.rows(); ++i) L.coeffRef(i, i) += shift;
    L.makeCompressed();
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
    for (int k = 0; k < A.outerSize() && done < flips; ++k)
        for (Sparse::InnerIterator it(A, k); it; ++it)
            if (it.row() > it.col()) {
                picked.emplace_back(static_cast<int>(it.row()),
                                    static_cast<int>(it.col()));
                if (++done == flips) break;
            }
    for (auto [i, j] : picked) {
        const double v = -A.coeff(i, j);
        A.coeffRef(i, j) = v;
        A.coeffRef(j, i) = v;
    }
    A.makeCompressed();
    return A;
}

/// Tiny strictly diagonally-dominant operator for probing cpu_solver's SpMV
/// storage. `upper_01_delta` changes only A(0,1); values within the operator
/// symmetry tolerance are accepted, while selfadjointView<Lower> semantics
/// must still source both mirrored slots from A(1,0).
Sparse spmv_probe_sddm(double upper_01_delta = 0.0) {
    std::vector<Trip> t{
        {0, 0, 4.0}, {1, 1, 5.0}, {2, 2, 6.0},
        {1, 0, -1.0}, {0, 1, -1.0 + upper_01_delta},
        {2, 0, -0.5}, {0, 2, -0.5},
        {2, 1, -1.25}, {1, 2, -1.25},
    };
    Sparse A(3, 3);
    A.setFromTriplets(t.begin(), t.end());
    A.makeCompressed();
    return A;
}

Eigen::VectorXd spmv_probe_x() {
    Eigen::VectorXd x(3);
    x << 1048576.0, 2.0, -4.0;
    return x;
}

Eigen::VectorXd spmv_probe_b() {
    // Exactly B*x in binary arithmetic for B =
    // spmv_probe_sddm().selfadjointView<Lower>(). Large x[0] makes a mistaken
    // use of the tolerated upper-triangle perturbation plainly observable.
    Eigen::VectorXd b(3);
    b << 4194304.0, -1048561.0, -524314.5;
    return b;
}

/// Compressed, sorted CSC with duplicate off-diagonal coordinates whose
/// multiplicity errors cancel in the global upper/lower counts:
///   (1,0) has two lower entries and one upper partner;
///   (2,0) has one lower entry and two upper partners.
/// The dense sums are symmetric, but direct array copying is not equivalent to
/// selfadjointView<Lower> unless the fast path rejects the representation.
Sparse spmv_balanced_duplicate_multiplicity_sddm() {
    Sparse A(3, 3);
    A.resizeNonZeros(9);
    const int outer[] = {0, 4, 6, 9};
    const int inner[] = {
        0, 1, 1, 2,  // column 0: diagonal + canonical lower entries
        0, 1,        // column 1: one upper partner + diagonal
        0, 0, 2,     // column 2: two upper partners + diagonal
    };
    const double values[] = {
        4.0, -0.25, -0.75, -2.0,
        -1.0, 4.0,
        -1.0, -1.0, 4.0,
    };
    std::copy(std::begin(outer), std::end(outer), A.outerIndexPtr());
    std::copy(std::begin(inner), std::end(inner), A.innerIndexPtr());
    std::copy(std::begin(values), std::end(values), A.valuePtr());
    return A;
}

Eigen::MatrixXd dense(const Sparse& A) { return Eigen::MatrixXd(A); }

/// Byte-exact snapshot of a sparse matrix's three storage arrays.
struct raw_snapshot {
    std::vector<int> outer, inner;
    std::vector<double> vals;
    Eigen::Index rows = 0, cols = 0;

    explicit raw_snapshot(const Sparse& A)
        : outer(A.outerIndexPtr(), A.outerIndexPtr() + A.outerSize() + 1),
          inner(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros()),
          vals(A.valuePtr(), A.valuePtr() + A.nonZeros()),
          rows(A.rows()), cols(A.cols()) {}

    bool identical_to(const Sparse& A) const {
        return rows == A.rows() && cols == A.cols() &&
               static_cast<Eigen::Index>(inner.size()) == A.nonZeros() &&
               std::memcmp(outer.data(), A.outerIndexPtr(),
                           outer.size() * sizeof(int)) == 0 &&
               std::memcmp(inner.data(), A.innerIndexPtr(),
                           inner.size() * sizeof(int)) == 0 &&
               std::memcmp(vals.data(), A.valuePtr(),
                           vals.size() * sizeof(double)) == 0;
    }
};

} // namespace

// ── the transform ───────────────────────────────────────────────────────────

TEST(MMatrixLumping, PreservesEveryRowSumExactly) {
    Sparse A = nearly_sddm(6, 7, 9);
    const Eigen::VectorXd before = A * Eigen::VectorXd::Ones(A.rows());
    ASSERT_GT(lump_positive_offdiagonals(A), 0);
    const Eigen::VectorXd after = A * Eigen::VectorXd::Ones(A.rows());
    // A 1 = Â 1 because every summand of the transform is
    // a_ij (e_i - e_j)(e_i - e_j)^T and (e_i - e_j)^T 1 = 0.
    EXPECT_LT((after - before).cwiseAbs().maxCoeff(), 1e-12);
}

TEST(MMatrixLumping, LeavesNoPositiveOffDiagonalAndStaysSymmetric) {
    Sparse A = nearly_sddm(6, 7, 9);
    const Eigen::Index removed = lump_positive_offdiagonals(A);
    EXPECT_EQ(removed, 18);                       // 9 pairs, both triangles

    const operator_scan s = scan_operator(A);
    EXPECT_EQ(s.offdiag_pos, 0);
    EXPECT_EQ(s.asymmetric, 0);
    EXPECT_EQ(s.bad_diag, 0);
    EXPECT_LT((Sparse(A.transpose()) - A).norm(), 1e-15);
}

TEST(MMatrixLumping, IsIdempotent) {
    Sparse A = nearly_sddm(5, 5, 6);
    ASSERT_GT(lump_positive_offdiagonals(A), 0);
    const Sparse once = A;
    EXPECT_EQ(lump_positive_offdiagonals(A), 0);  // nothing left to move
    EXPECT_LT((A - once).norm(), 0.0 + 1e-300);   // and nothing changed
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

    const int n = static_cast<int>(A.rows());
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(n, n);
    for (int k = 0; k < A.outerSize(); ++k)
        for (Sparse::InnerIterator it(A, k); it; ++it)
            if (it.row() > it.col() && it.value() > 0.0) {
                Eigen::VectorXd e = Eigen::VectorXd::Zero(n);
                e(it.row()) = 1.0;
                e(it.col()) = -1.0;
                S += it.value() * e * e.transpose();
            }
    EXPECT_LT((dense(H) - (dense(A) + S)).cwiseAbs().maxCoeff(), 1e-12);
}

TEST(MMatrixLumping, NeverWeakensDiagonalDominance) {
    const Sparse A = nearly_sddm(6, 6, 8);
    Sparse H = A;
    ASSERT_GT(lump_positive_offdiagonals(H), 0);

    const Eigen::MatrixXd Ad = dense(A), Hd = dense(H);
    for (int i = 0; i < Ad.rows(); ++i) {
        const double ma = Ad(i, i) - (Ad.row(i).cwiseAbs().sum() - std::abs(Ad(i, i)));
        const double mh = Hd(i, i) - (Hd.row(i).cwiseAbs().sum() - std::abs(Hd(i, i)));
        EXPECT_GE(mh, ma - 1e-12) << "row " << i;
    }
}

TEST(MMatrixLumping, LumpedMatrixIsPositiveDefiniteWhenTheOriginalIs) {
    // A ⪯ Â, so λ_k(Â) ≥ λ_k(A) for every k (Weyl). Checked outright on a
    // matrix small enough for a dense symmetric eigendecomposition.
    const Sparse A = nearly_sddm(5, 5, 6, /*shift=*/0.1);
    Sparse H = A;
    ASSERT_GT(lump_positive_offdiagonals(H), 0);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ea(dense(A)), eh(dense(H));
    ASSERT_EQ(ea.info(), Eigen::Success);
    ASSERT_EQ(eh.info(), Eigen::Success);
    EXPECT_GT(ea.eigenvalues().minCoeff(), 0.0) << "test matrix is not SPD";
    EXPECT_GT(eh.eigenvalues().minCoeff(), 0.0);
    const double scale = ea.eigenvalues().cwiseAbs().maxCoeff();
    for (int i = 0; i < ea.eigenvalues().size(); ++i)
        EXPECT_GE(eh.eigenvalues()(i), ea.eigenvalues()(i) - 1e-10 * scale) << i;
}

// ── the precondition ────────────────────────────────────────────────────────

TEST(OperatorContract, AcceptsLaplacianAndSddm) {
    EXPECT_NO_THROW((void)require_operator(grid_laplacian(6, 6)));
    EXPECT_NO_THROW((void)require_operator(grid_sddm(6, 6)));
    Sparse empty(8, 8);                       // n > 0, no entries at all
    EXPECT_NO_THROW((void)require_operator(empty));
    Sparse id(4, 4);
    id.setIdentity();
    EXPECT_NO_THROW((void)require_operator(id));
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
    A.coeffRef(1, 0) = -0.5;
    A.makeCompressed();
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
    A.coeffRef(2, 5) = -0.5;                  // (5, 2) does not exist
    A.makeCompressed();
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
    Sparse A(3, 3);
    A.setFromTriplets(t.begin(), t.end());
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
    A.coeffRef(3, 3) = -1.0;                  // e_3^T A e_3 < 0: not PSD
    A.makeCompressed();
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
    Sparse A(L.rows(), L.cols());             // adjacency = -offdiag(L)
    std::vector<Trip> t;
    for (int k = 0; k < L.outerSize(); ++k)
        for (Sparse::InnerIterator it(L, k); it; ++it)
            if (it.row() != it.col())
                t.emplace_back(static_cast<int>(it.row()),
                               static_cast<int>(it.col()), -it.value());
    A.setFromTriplets(t.begin(), t.end());

    try {
        (void)require_operator(A);
        FAIL() << "expected an adjacency matrix to be refused";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("Condition FAILED: positive diagonal"),
                  std::string::npos) << msg;
        EXPECT_NE(msg.find("ADJACENCY"), std::string::npos) << msg;
        EXPECT_NE(msg.find("apxchol.laplacian(A)"), std::string::npos) << msg;
        EXPECT_NE(msg.find("apxchol_laplacian(A)"), std::string::npos) << msg;
        EXPECT_NE(msg.find("--input-kind"), std::string::npos) << msg;
    }
}

TEST(OperatorContract, ReportsDominanceViolationsWithoutEnforcingThem) {
    // G3_circuit / apache2 in miniature: a Z-matrix with rows the diagonal
    // does not cover. It must be ACCEPTED -- those matrices converge today.
    Sparse A = grid_laplacian(4, 4);
    A.coeffRef(0, 15) = -1.0;
    A.coeffRef(15, 0) = -1.0;
    A.makeCompressed();

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

    const operator_view op(A);
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
    for (int k = 0; k < A.outerSize(); ++k)
        for (Sparse::InnerIterator it(A, k); it; ++it)
            if (it.row() != it.col()) it.valueRef() = -it.value();
    A.makeCompressed();
    ASSERT_GT(scan_operator(A).positive_mass_fraction(), kDefaultLumpMassCeiling);

    try {
        (void)require_operator(A);
        FAIL() << "expected a matrix past the lumping ceiling to be refused";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("Condition FAILED: non-positive off-diagonals"),
                  std::string::npos) << msg;
        EXPECT_NE(msg.find("of the off-diagonal mass"), std::string::npos) << msg;
        EXPECT_NE(msg.find("APXCHOL_LUMP_MAX"), std::string::npos) << msg;
    }
}

TEST(OperatorContract, MassNotCountIsWhatTheCeilingMeasures) {
    // parabolic_fem in miniature: a THIRD of the off-diagonals are positive,
    // but they carry no mass, so the matrix is trivially lumpable. A ceiling
    // on the entry COUNT would have refused it.
    Sparse A = grid_sddm(8, 8);
    Eigen::Index flipped = 0;
    for (int k = 0; k < A.outerSize(); ++k)
        for (Sparse::InnerIterator it(A, k); it; ++it)
            if (it.row() != it.col() && (it.row() + it.col()) % 3 == 0) {
                it.valueRef() = 1e-9;
                ++flipped;
            }
    A.makeCompressed();
    const operator_scan s = scan_operator(A);
    ASSERT_EQ(s.offdiag_pos, flipped);
    EXPECT_GT(static_cast<double>(s.offdiag_pos) / s.offdiag_nnz, 0.2);
    EXPECT_LT(s.positive_mass_fraction(), 1e-6);
    EXPECT_NO_THROW((void)require_operator(A));
}

// ── the operator PCG applies is untouched ───────────────────────────────────

TEST(SpmvLrmBuild, AcceptedNearSymmetryStillUsesCanonicalLowerValues) {
    // require_operator intentionally accepts tiny representation noise. The
    // direct CSC-layout -> CSR path must not start applying A^T; it preserves
    // the old selfadjointView<Lower> contract byte-for-byte.
    Sparse A = spmv_probe_sddm(5e-11);
    ASSERT_NE(A.coeff(0, 1), A.coeff(1, 0));
    ASSERT_EQ(scan_operator(A).asymmetric, 0);

    const Eigen::VectorXd x0 = spmv_probe_x();
    const Eigen::VectorXd b = spmv_probe_b();
    const apxchol::cpu_solver slv(A);
    const auto res = slv.solve(b, 1e-15, 0, &x0);
    EXPECT_EQ(res.iterations, 0);
    EXPECT_EQ(res.residual, 0.0);
}

TEST(SpmvLrmBuild, OwnedFp32CopySurvivesCallerMutationAndDestruction) {
    // Exact binary values select the fp32 operator. Leave the input
    // uncompressed to exercise the local compression path, then prove the
    // reusable solver does not retain a map/reference to caller storage.
    auto A = std::make_unique<Sparse>(spmv_probe_sddm());
    A->uncompress();
    ASSERT_FALSE(A->isCompressed());

    const Eigen::VectorXd x0 = spmv_probe_x();
    const Eigen::VectorXd b = spmv_probe_b();
    const apxchol::cpu_solver slv(*A);

    for (int k = 0; k < A->outerSize(); ++k)
        for (Sparse::InnerIterator it(*A, k); it; ++it)
            it.valueRef() = 0.0;
    const auto after_mutation = slv.solve(b, 1e-15, 0, &x0);
    EXPECT_EQ(after_mutation.residual, 0.0);

    A.reset();
    const auto after_destruction = slv.solve(b, 1e-15, 0, &x0);
    EXPECT_EQ(after_destruction.residual, 0.0);
}

TEST(SpmvLrmBuild, BalancedDuplicateMultiplicitiesUseGeneralFallback) {
    const Sparse A = spmv_balanced_duplicate_multiplicity_sddm();
    ASSERT_TRUE(A.isCompressed());
    ASSERT_EQ(A.nonZeros(), 9);

    // A's lower-authoritative operator has unique dense values -1 and -2.
    // Supply a factor for that same operator; max_iter=0 makes this test probe
    // only the SpMV residual at x0, independent of preconditioner quality.
    const std::vector<Trip> canonical_entries{
        {0, 0, 4.0}, {1, 1, 4.0}, {2, 2, 4.0},
        {1, 0, -1.0}, {0, 1, -1.0},
        {2, 0, -2.0}, {0, 2, -2.0},
    };
    Sparse canonical(3, 3);
    canonical.setFromTriplets(canonical_entries.begin(), canonical_entries.end());
    auto F = apxchol::factorize(canonical);
    Eigen::VectorXd x0(3);
    x0 << 8.0, -4.0, 2.0;
    const Eigen::VectorXd b = A.selfadjointView<Eigen::Lower>() * x0;

    const apxchol::cpu_solver slv(A, std::move(F));
    const auto result = slv.solve(b, 1e-15, 0, &x0);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.residual, 0.0);
}

TEST(SpmvLrmBuild, OneTriangleCustomOperatorKeepsGeneralFallback) {
    // The adopting constructor historically accepts a one-triangle operator.
    // It has no reusable CSC-as-CSR layout, so it must retain the general
    // lower-reflection build while full symmetric inputs take the direct path.
    const Sparse A = spmv_probe_sddm();
    auto F = apxchol::factorize(A);
    auto F_for_direct = F;
    Sparse lower = A.triangularView<Eigen::Lower>();
    lower.makeCompressed();
    ASSERT_LT(lower.nonZeros(), A.nonZeros());

    const Eigen::VectorXd x0 = spmv_probe_x();
    const Eigen::VectorXd b = spmv_probe_b();
    const apxchol::cpu_solver direct(A, std::move(F_for_direct));
    const apxchol::cpu_solver reflected(lower, std::move(F));
    const auto res = reflected.solve(b, 1e-15, 0, &x0);
    EXPECT_EQ(res.iterations, 0);
    EXPECT_EQ(res.residual, 0.0);

    // More than the exact-x0 probe: with one identical factor installed in
    // both solvers, the direct and reflected operator representations must
    // drive the entire PCG recurrence bit-identically.
    Eigen::VectorXd rhs(3);
    rhs << 1.0, -2.0, 3.0;
    const auto direct_res = direct.solve(rhs, 1e-12, 50);
    const auto reflected_res = reflected.solve(rhs, 1e-12, 50);
    EXPECT_EQ(direct_res.iterations, reflected_res.iterations);
    EXPECT_EQ(direct_res.residual, reflected_res.residual);
    EXPECT_TRUE((direct_res.x.array() == reflected_res.x.array()).all());
}

TEST(MMatrixLumping, OperatorIsBitIdenticalAfterPreconditionerConstruction) {
    // The contract that makes lumping safe: the transform belongs to the
    // PRECONDITIONER, never to the operator. If this ever fails, every
    // residual apxchol reports on a nearly-SDDM matrix is for a system the
    // caller did not ask about.
    Sparse A = nearly_sddm(20, 20, 12);
    const raw_snapshot snap(A);

    const apxchol::cpu_solver slv(A, {.tol = 1e-8, .max_iter = 200});
    ASSERT_GT(slv.preconditioner().factor().lumped_offdiag, 0)
        << "the test matrix did not actually trigger lumping";
    EXPECT_TRUE(snap.identical_to(A))
        << "preconditioner construction modified the caller's operator";

    const Eigen::VectorXd b = apxchol::generate_test_rhs(A.rows());
    const auto res = slv.solve(b);
    EXPECT_TRUE(snap.identical_to(A)) << "solving modified the caller's operator";

    // And the residual really is measured against the matrix passed in.
    const double rel = (b - A * res.x).norm() / b.norm();
    EXPECT_LT(rel, 1e-8) << "reported residual " << res.residual;
}

// ── end to end ──────────────────────────────────────────────────────────────

TEST(MMatrixLumping, NearlySddmOperatorConvergesAgainstTheTrueOperator) {
    const Sparse A = nearly_sddm(30, 30, 40);
    ASSERT_GT(scan_operator(A).offdiag_pos, 0);

    const Eigen::VectorXd b = apxchol::generate_test_rhs(A.rows());
    const auto res = apxchol::solve(A, b, {.tol = 1e-8, .max_iter = 300});
    EXPECT_EQ(res.lumped_offdiag, 80);            // 40 pairs
    EXPECT_GT(res.iterations, 0);
    EXPECT_LT(res.residual, 1e-8);
    EXPECT_LT((b - A * res.x).norm() / b.norm(), 1e-7);
}

TEST(MMatrixLumping, PureLaplacianAndSddmAreUnchangedByTheFeature) {
    // The regression guard for everything that already worked: no lumping
    // triggered, no behaviour change, and the factor reports zero.
    for (const Sparse& A : {grid_laplacian(30, 30), grid_sddm(30, 30)}) {
        const raw_snapshot snap(A);
        const Eigen::VectorXd b = apxchol::generate_test_rhs(A.rows());
        const auto res = apxchol::solve(A, b, {.tol = 1e-8, .max_iter = 300});
        EXPECT_EQ(res.lumped_offdiag, 0);
        EXPECT_GT(res.iterations, 0);
        EXPECT_LT(res.residual, 1e-8);
        EXPECT_TRUE(snap.identical_to(A));
    }
}
