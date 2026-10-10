#include <gtest/gtest.h>
#include "apxchol/solver/detail/permuted_operator.h"
#include <array>
#include <bit>
#include <cstring>
#include <numeric>
#include <random>
#include <omp.h>

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
        std::sort(row.begin(), row.end());
        for (auto [index, value] : row) { ref.idx.push_back(index); ref.vals.push_back(value); }
        ref.ptr.push_back(static_cast<int>(ref.idx.size()));
    }
    return ref;
}
void check(const Matrix& L, const std::vector<node_index>& perm) {
    const auto ref = serial_reference(L, perm);
    std::vector<int> ptr;
    std::unique_ptr<int[]> idx;
    std::unique_ptr<double[]> vals;
    std::int64_t nnz = -1;
    bool exact = false;
    ASSERT_TRUE(apxchol::detail::try_build_permuted_symmetric_csr(L, perm, ptr, idx, vals, nnz, exact));
    ASSERT_EQ(nnz, static_cast<std::int64_t>(ref.idx.size()));
    EXPECT_EQ(ptr, ref.ptr);
    EXPECT_EQ(exact, ref.exact);
    if (nnz) {
        EXPECT_EQ(0, std::memcmp(idx.get(), ref.idx.data(), nnz * sizeof(int)));
        EXPECT_EQ(0, std::memcmp(vals.get(), ref.vals.data(), nnz * sizeof(double)));
    }
}
Csr general(const Matrix& L, const std::vector<node_index>& perm) {
    Csr out;
    std::unique_ptr<int[]> idx;
    std::unique_ptr<double[]> vals;
    std::int64_t nnz = -1;
    out.exact = false;
    apxchol::detail::build_permuted_full_symmetric_csr(L, perm, out.ptr, idx, vals, nnz, out.exact);
    out.idx.assign(idx.get(), idx.get() + nnz);
    out.vals.assign(vals.get(), vals.get() + nnz);
    return out;
}
Matrix from_entries(int n, std::initializer_list<Eigen::Triplet<double>> entries) {
    Matrix L(n, n); L.setFromTriplets(entries.begin(), entries.end()); return L;
}
bool accepted(const Matrix& L) {
    std::vector<node_index> perm(L.rows()); std::iota(perm.begin(), perm.end(), 0);
    std::vector<int> ptr; std::unique_ptr<int[]> idx; std::unique_ptr<double[]> vals;
    std::int64_t nnz = -1; bool exact = false;
    return apxchol::detail::try_build_permuted_symmetric_csr(L, perm, ptr, idx, vals, nnz, exact);
}
} // namespace

TEST(PermutedOperator, PermutedArraysMatchIndependentLowerTriangleReference) {
    std::mt19937 rng(418932);
    const int saved = omp_get_max_threads();
    for (int team : {1, 2, 4}) {
        omp_set_num_threads(team);
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
    omp_set_num_threads(saved);
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
    std::vector<int> ptr;
    auto idx = std::make_unique<int[]>(1);
    auto vals = std::make_unique<double[]>(1);
    idx[0] = 23; vals[0] = 4.5;
    const auto* original_idx = idx.get();
    const auto* original_vals = vals.get();
    std::int64_t nnz = 91; bool exact = false;
    EXPECT_FALSE(apxchol::detail::try_build_permuted_symmetric_csr(
        L, {2, 0, 1}, ptr, idx, vals, nnz, exact));
    EXPECT_EQ(idx.get(), original_idx); EXPECT_EQ(vals.get(), original_vals);
    EXPECT_EQ(idx[0], 23); EXPECT_EQ(vals[0], 4.5);
    EXPECT_EQ(nnz, 91); EXPECT_FALSE(exact);
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
    const int saved = omp_get_max_threads();
    for (int team : {1, 2, 4}) {
        omp_set_num_threads(team);
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
            const Csr ref = serial_reference(L, perm);
            const Csr got = general(L, perm);
            EXPECT_EQ(got.ptr, ref.ptr);
            EXPECT_EQ(got.idx, ref.idx);
            ASSERT_EQ(got.vals.size(), ref.vals.size());
            EXPECT_EQ(0, std::memcmp(got.vals.data(), ref.vals.data(), ref.vals.size() * sizeof(double)));
            EXPECT_EQ(got.exact, ref.exact);
        }
    }
    omp_set_num_threads(saved);
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
    // The serial reference with each row's duplicates in value-bit order.
    Csr ref = serial_reference(L, perm);
    for (std::size_t r = 0; r + 1 < ref.ptr.size(); ++r) {
        std::vector<std::pair<int, std::uint64_t>> row;
        for (int p = ref.ptr[r]; p < ref.ptr[r + 1]; ++p)
            row.emplace_back(ref.idx[p], std::bit_cast<std::uint64_t>(ref.vals[p]));
        std::sort(row.begin(), row.end());
        for (int p = ref.ptr[r]; p < ref.ptr[r + 1]; ++p) {
            ref.idx[p] = row[p - ref.ptr[r]].first;
            ref.vals[p] = std::bit_cast<double>(row[p - ref.ptr[r]].second);
        }
    }
    const int saved = omp_get_max_threads();
    for (int team : {1, 2, 4, 8}) {
        omp_set_num_threads(team);
        for (int run = 0; run < 3; ++run) {
            SCOPED_TRACE("team=" + std::to_string(team) + " run=" + std::to_string(run));
            const Csr got = general(L, perm);
            EXPECT_EQ(got.ptr, ref.ptr);
            EXPECT_EQ(got.idx, ref.idx);
            ASSERT_EQ(got.vals.size(), ref.vals.size());
            EXPECT_EQ(0, std::memcmp(got.vals.data(), ref.vals.data(), ref.vals.size() * sizeof(double)));
        }
    }
    omp_set_num_threads(saved);
}
