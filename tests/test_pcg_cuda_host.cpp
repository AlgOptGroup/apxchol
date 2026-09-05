#include <gtest/gtest.h>
#include "apxchol/solver/pcg_cuda_host.h"
#include <array>
#include <cstring>
#include <numeric>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif

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

TEST(GpuPcgHost, PermutedArraysMatchIndependentLowerTriangleReference) {
    std::mt19937 rng(418932);
#ifdef _OPENMP
    const int saved = omp_get_max_threads();
#endif
    for (int team : {1, 2, 4}) {
#ifdef _OPENMP
        omp_set_num_threads(team);
#endif
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
                    // the old GPU builder ignores them, including precision.
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
#ifdef _OPENMP
    omp_set_num_threads(saved);
#endif
}

TEST(GpuPcgHost, ExplicitZerosAndMissingDiagonalsPreserveCanonicalPrecision) {
    const auto L = from_entries(5, {{2, 0, -0.0}, {0, 2, 0.1},
                                   {3, 1, -0.5}, {1, 3, -0.500000001}, {4, 4, 2.0}});
    check(L, {3, 0, 4, 2, 1});
    EXPECT_TRUE(serial_reference(L, {3, 0, 4, 2, 1}).exact);
}

TEST(GpuPcgHost, RejectsUnpairedStorageEvenWhenTriangleCountsBalance) {
    EXPECT_FALSE(accepted(from_entries(3, {{1, 0, -1.0}, {0, 2, -1.0}})));
    EXPECT_FALSE(accepted(from_entries(3, {{1, 0, -1.0}})));
    EXPECT_FALSE(accepted(from_entries(3, {{0, 1, -1.0}})));
}

TEST(GpuPcgHost, MissingPartnerDoesNotPublishPartiallyWrittenArrays) {
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

TEST(GpuPcgHost, RejectsDuplicatesUnsortedAndUncompressedStorage) {
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
