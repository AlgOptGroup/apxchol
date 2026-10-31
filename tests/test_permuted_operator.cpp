#include <gtest/gtest.h>
#include "apxchol/solver/detail/permuted_operator.h"
#include "scoped_omp_threads.h"
#include <array>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>

namespace {
using Matrix = Eigen::SparseMatrix<double>;
using apxchol::node_index;
struct Csr {
    std::vector<int> ptr, idx;
    std::vector<double> vals;
    bool exact = true;
};
Csr serial_reference(const Matrix& L, const std::vector<node_index>& perm) {
    std::vector<std::vector<std::pair<int, double>>> rows(L.rows());
    Csr ref;
    for (int col = 0; col < L.cols(); ++col)
        for (Matrix::InnerIterator it(L, col); it; ++it) {
            const int row = it.row();
            if (row < col) continue;
            const double value = it.value();
            ref.exact &= static_cast<double>(static_cast<float>(value)) == value;
            rows[perm[row]].emplace_back(perm[col], value);
            if (row != col) rows[perm[col]].emplace_back(perm[row], value);
        }
    ref.ptr.push_back(0);
    for (auto& row : rows) {
        std::ranges::sort(row);
        for (auto [index, value] : row) { ref.idx.push_back(index); ref.vals.push_back(value); }
        ref.ptr.push_back(static_cast<int>(ref.idx.size()));
    }
    return ref;
}
void check(const Matrix& L, const std::vector<node_index>& perm, bool use_full_builder = false) {
    const auto ref = serial_reference(L, perm);
    apxchol::detail::permuted_operator_csr out;
    if (use_full_builder)
        out = apxchol::detail::build_permuted_full_symmetric_csr(L, perm);
    else
        ASSERT_TRUE(apxchol::detail::try_build_permuted_symmetric_csr(L, perm, out));
    const auto nnz = out.ptr.back();
    ASSERT_EQ(nnz, static_cast<std::int64_t>(ref.idx.size()));
    EXPECT_EQ(out.ptr, ref.ptr);
    EXPECT_EQ(out.fp32_exact, ref.exact);
    if (nnz) {
        EXPECT_EQ(0, std::memcmp(out.idx.get(), ref.idx.data(), nnz * sizeof(int)));
        EXPECT_EQ(0, std::memcmp(out.vals.get(), ref.vals.data(), nnz * sizeof(double)));
    }
}
Matrix from_entries(int n, std::initializer_list<Eigen::Triplet<double>> entries) {
    Matrix L(n, n); L.setFromTriplets(entries.begin(), entries.end()); return L;
}
bool accepted(const Matrix& L) {
    std::vector<node_index> perm(L.rows()); std::iota(perm.begin(), perm.end(), 0);
    apxchol::detail::permuted_operator_csr out;
    return apxchol::detail::try_build_permuted_symmetric_csr(L, perm, out);
}
} // namespace

TEST(PermutedOperator, PermutedArraysMatchIndependentLowerTriangleReference) {
    std::mt19937 rng(418932);
    for (int team : {1, 2, 4}) {
        scoped_omp_threads threads(team);
        for (int trial = 0; trial < 80; ++trial) {
            const int n = trial * 13 % 131;
            std::vector<Eigen::Triplet<double>> entries;
            for (int col = 0; col < n; ++col) {
                if (trial % 3) entries.emplace_back(col, col, 4.0);
                for (int row = col + 1; row < n; ++row) {
                    if ((trial % 2 && row / 8 != col / 8) || rng() % 13) continue;
                    const double value = trial % 5 ? -0.25 * (rng() % 8) : -0.1;
                    entries.emplace_back(row, col, value);
                    // Deliberately different, possibly inexact upper values:
                    // only canonical lower values determine the operator and its precision.
                    entries.emplace_back(col, row, value + 1e-10);
                }
            }
            Matrix L(n, n); L.setFromTriplets(entries.begin(), entries.end());
            std::vector<node_index> perm(n); std::iota(perm.begin(), perm.end(), 0);
            std::shuffle(perm.begin(), perm.end(), rng);
            SCOPED_TRACE("team=" + std::to_string(team) + " trial=" + std::to_string(trial));
            check(L, perm);
        }
    }
}

TEST(PermutedOperator, ExplicitZerosAndMissingDiagonalsPreserveCanonicalPrecision) {
    const auto L = from_entries(5, {{2, 0, -0.0}, {0, 2, 0.1},
                                   {3, 1, -0.5}, {1, 3, -0.500000001}, {4, 4, 2.0}});
    check(L, {3, 0, 4, 2, 1});
    EXPECT_TRUE(serial_reference(L, {3, 0, 4, 2, 1}).exact);
}

TEST(PermutedOperator, RejectsUnpairedStorageEvenWhenTriangleCountsBalance) {
    EXPECT_FALSE(accepted(from_entries(3, {{1, 0, -1.0}, {0, 2, -1.0}})));
    EXPECT_FALSE(accepted(from_entries(3, {{1, 0, -1.0}})));
    EXPECT_FALSE(accepted(from_entries(3, {{0, 1, -1.0}})));
}

TEST(PermutedOperator, MissingPartnerDoesNotPublishPartiallyWrittenArrays) {
    const auto L = from_entries(3, {{1, 0, -1.0}, {0, 2, -1.0}});
    apxchol::detail::permuted_operator_csr out;
    out.idx = std::make_unique<int[]>(1);
    out.vals = std::make_unique<double[]>(1);
    out.idx[0] = 23; out.vals[0] = 4.5;
    const auto* original_idx = out.idx.get();
    const auto* original_vals = out.vals.get();
    out.fp32_exact = false;
    EXPECT_FALSE(apxchol::detail::try_build_permuted_symmetric_csr(
        L, {2, 0, 1}, out));
    EXPECT_EQ(out.idx.get(), original_idx); EXPECT_EQ(out.vals.get(), original_vals);
    EXPECT_EQ(out.idx[0], 23); EXPECT_EQ(out.vals[0], 4.5);
    EXPECT_FALSE(out.fp32_exact);
}

TEST(PermutedOperator, RejectsDuplicatesUnsortedAndUncompressedStorage) {
    Matrix duplicate(2, 2);
    duplicate.resizeNonZeros(4);
    std::copy_n(std::array<int, 3>{0, 2, 4}.data(), 3, duplicate.outerIndexPtr());
    std::copy_n(std::array<int, 4>{1, 1, 0, 0}.data(), 4, duplicate.innerIndexPtr());
    std::fill_n(duplicate.valuePtr(), 4, -1.0);
    EXPECT_FALSE(accepted(duplicate));
    Matrix unsorted = from_entries(3, {{0, 0, 2.}, {1, 0, -1.}, {0, 1, -1.}, {1, 1, 2.}});
    std::swap(unsorted.innerIndexPtr()[0], unsorted.innerIndexPtr()[1]);
    std::swap(unsorted.valuePtr()[0], unsorted.valuePtr()[1]);
    EXPECT_FALSE(accepted(unsorted));
    Matrix uncompressed = from_entries(2, {{0, 0, 2.}, {1, 1, 2.}});
    uncompressed.uncompress();
    EXPECT_FALSE(accepted(uncompressed));
}

TEST(PermutedOperator, GeneralFallbackMatchesIndependentLowerTriangleReference) {
    // Lower-triangle-only storage is unpaired, so the column-ownership path
    // declines it and the atomic count, scatter and per-row sort build the
    // CSR. Fully stored inputs take the column-ownership path. Both must equal
    // the serial reference byte for byte at every team size. No input has
    // duplicate coordinates; those are tested separately below.
    std::mt19937 rng(20261002);
    for (int team : {1, 2, 4}) {
        scoped_omp_threads threads(team);
        for (int trial = 0; trial < 40; ++trial) {
            const int n = 2 + trial * 17 % 97;
            const bool lower_only = trial % 2 == 0;
            std::vector<Eigen::Triplet<double>> entries;
            for (int col = 0; col < n; ++col) {
                entries.emplace_back(col, col, trial % 3 ? 4.0 : 4.1);
                for (int row = col + 1; row < n; ++row) {
                    if (!(row == 1 && col == 0) && rng() % 7) continue;
                    const double value = (trial % 4 ? -0.25 : -0.1) * (1 + rng() % 8);
                    entries.emplace_back(row, col, value);
                    if (!lower_only) entries.emplace_back(col, row, value);
                }
            }
            Matrix L(n, n); L.setFromTriplets(entries.begin(), entries.end());
            std::vector<node_index> perm(n); std::iota(perm.begin(), perm.end(), 0);
            std::shuffle(perm.begin(), perm.end(), rng);
            SCOPED_TRACE("team=" + std::to_string(team) + " trial=" + std::to_string(trial));
            EXPECT_EQ(accepted(L), !lower_only);
            check(L, perm, /*use_full_builder=*/true);
        }
    }
}

TEST(PermutedOperator, GeneralFallbackOrdersDuplicateCoordinatesDeterministically) {
    // Lower-only storage with every off-diagonal coordinate stored two or
    // three times, and a hub row that every column writes to: the scatter's
    // slot order depends on the threads, the sorted result must not.
    const int n = 600;
    std::mt19937 rng(11);
    Matrix L(n, n);
    std::vector<int> outer{0}, inner;
    std::vector<double> values;
    for (int col = 0; col < n; ++col) {
        inner.push_back(col); values.push_back(8.0);
        for (int row = col + 1; row < n; ++row) {
            if (row != n - 1 && rng() % 31) continue;
            for (int copy = 0, copies = 2 + static_cast<int>(rng() % 2); copy < copies; ++copy) {
                inner.push_back(row);
                values.push_back(-0.125 * (1 + static_cast<int>(rng() % 16)));
            }
        }
        outer.push_back(static_cast<int>(inner.size()));
    }
    L.resizeNonZeros(static_cast<Eigen::Index>(inner.size()));
    std::copy(outer.begin(), outer.end(), L.outerIndexPtr());
    std::copy(inner.begin(), inner.end(), L.innerIndexPtr());
    std::copy(values.begin(), values.end(), L.valuePtr());
    ASSERT_FALSE(accepted(L));
    std::vector<node_index> perm(n); std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int team : {1, 2, 4, 8}) {
        scoped_omp_threads threads(team);
        for (int run = 0; run < 3; ++run) {
            SCOPED_TRACE("team=" + std::to_string(team) + " run=" + std::to_string(run));
            check(L, perm, /*use_full_builder=*/true);
        }
    }
}

TEST(PermutedOperator, GeneralFallbackRejectsNonFiniteValuesBeforeSorting) {
    // Lower-only duplicates force the fallback, whose row sort needs finite values.
    for (double bad : {std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity()}) {
        Matrix L(2, 2);
        const std::array<int, 3> outer{0, 3, 4};
        const std::array<int, 4> inner{0, 1, 1, 1};
        const std::array<double, 4> values{2.0, bad, -1.0, 2.0};
        L.resizeNonZeros(4);
        std::copy(outer.begin(), outer.end(), L.outerIndexPtr());
        std::copy(inner.begin(), inner.end(), L.innerIndexPtr());
        std::copy(values.begin(), values.end(), L.valuePtr());
        ASSERT_FALSE(accepted(L));
        const std::vector<node_index> perm{0, 1};
        EXPECT_THROW(apxchol::detail::build_permuted_full_symmetric_csr(L, perm),
                     std::invalid_argument);
    }
}
