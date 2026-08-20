// Byte-identity tests for the shared CSC→CSR transpose (transpose.h) through
// BOTH its callers: omp_sptrsv::setup (the CPU SpTRSV's CSR of L11) and
// cuda_host::transpose_csr (the GPU backend's host prep: CSR of L from CSR of
// L^T, int32 arrays, plain value copy).
//
// The blocked counting-sort transpose (APXCHOL_PAR_TRANSPOSE, engaged for
// m > kParTransposeMinRows = 50000) must produce a CSR that is byte-identical
// to the classic serial count/prefix/scatter transpose — same row pointers,
// same column indices (ascending column order within every row), same value
// bit patterns — for ANY thread count. These tests build a random sparse
// lower-triangular factor big enough to engage the parallel path and compare
// against an independent serial reference at several thread counts.

#include <gtest/gtest.h>
#include <cstring>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "apxchol/sparse_csc.h"
#include "apxchol/solver/sptrsv/cuda_host.h"
#include "apxchol/solver/sptrsv/omp.h"

using apxchol::edge_index;
using apxchol::node_index;
using apxchol::sparse_csc;
using apxchol::factor_value_t;
using apxchol::sptrsv_value_t;

namespace {

// ── Serial reference transpose ───────────────────────
// Classic count + prefix + column-order scatter. Independent of the
// implementation under test; defines the expected byte layout.
struct ref_csr {
    std::vector<edge_index>     row_ptr;
    std::vector<node_index>     col_idx;
    std::vector<sptrsv_value_t> vals;
};

ref_csr reference_transpose(const sparse_csc& L) {
    const node_index m   = L.rows();
    const edge_index nnz = L.nonZeros();
    const auto* outer = L.outerIndexPtr();
    const auto* inner = L.innerIndexPtr();
    const auto* vals  = L.valuePtr();

    ref_csr R;
    R.row_ptr.assign(static_cast<size_t>(m) + 1, 0);
    for (node_index j = 0; j < m; ++j)
        for (edge_index p = outer[j]; p < outer[j + 1]; ++p)
            R.row_ptr[inner[p] + 1]++;
    for (node_index i = 0; i < m; ++i)
        R.row_ptr[i + 1] += R.row_ptr[i];
    R.col_idx.resize(nnz);
    R.vals.resize(nnz);
    std::vector<edge_index> pos(R.row_ptr.begin(), R.row_ptr.begin() + m);
    for (node_index j = 0; j < m; ++j) {
        // factor_value_t -> sptrsv_value_t through the documented storage
        // contract (RNE narrowing of the value divided by the per-column scale
        // under FP16_SCALED; a plain cast otherwise).
        const float s_j = apxchol::omp_sptrsv::column_scale(vals, outer[j], outer[j + 1]);
        for (edge_index p = outer[j]; p < outer[j + 1]; ++p) {
            const edge_index out = pos[inner[p]]++;
            R.col_idx[out] = j;
            R.vals[out]    = apxchol::omp_sptrsv::narrow_value(vals[p], s_j);
        }
    }
    return R;
}

// ── Random lower-triangular factor ───────────────────
// Every column gets a diagonal entry plus `avg_offdiag` random strictly-lower
// entries on average; a sprinkling of "hub" columns get several hundred to
// exercise phase-3 load imbalance and multi-entry rows.
sparse_csc make_random_lower(node_index m, double avg_offdiag, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uval(0.1, 2.0);
    std::poisson_distribution<int> pcount(avg_offdiag);

    sparse_csc L;
    L.n_ = m;
    L.outer_.assign(static_cast<size_t>(m) + 1, 0);

    std::vector<std::vector<node_index>> col_rows(m);
    for (node_index j = 0; j < m; ++j) {
        int k = pcount(rng);
        if (j % 9973 == 0) k += 500;               // hub columns
        std::vector<node_index>& rows = col_rows[j];
        rows.push_back(j);                          // diagonal, always present
        if (j + 1 < m) {
            std::uniform_int_distribution<node_index> urow(j + 1, m - 1);
            for (int t = 0; t < k; ++t) rows.push_back(urow(rng));
        }
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    }
    for (node_index j = 0; j < m; ++j)
        L.outer_[j + 1] = L.outer_[j] + static_cast<edge_index>(col_rows[j].size());
    L.inner_.resize(static_cast<size_t>(L.outer_[m]));
    L.vals_.resize(static_cast<size_t>(L.outer_[m]));
    for (node_index j = 0; j < m; ++j) {
        edge_index out = L.outer_[j];
        for (node_index r : col_rows[j]) {
            L.inner_[out] = r;
            L.vals_[out]  = static_cast<factor_value_t>(uval(rng));
            ++out;
        }
    }
    return L;
}

void expect_byte_identical(const apxchol::omp_sptrsv& trsv, const ref_csr& R,
                           int threads) {
    SCOPED_TRACE("threads=" + std::to_string(threads));
    ASSERT_EQ(trsv.csr_row_ptr().size(), R.row_ptr.size());
    ASSERT_EQ(trsv.csr_col_idx().size(), R.col_idx.size());
    ASSERT_EQ(trsv.csr_vals().size(), R.vals.size());
    EXPECT_EQ(0, std::memcmp(trsv.csr_row_ptr().data(), R.row_ptr.data(),
                             R.row_ptr.size() * sizeof(edge_index)));
    EXPECT_EQ(0, std::memcmp(trsv.csr_col_idx().data(), R.col_idx.data(),
                             R.col_idx.size() * sizeof(node_index)));
    EXPECT_EQ(0, std::memcmp(trsv.csr_vals().data(), R.vals.data(),
                             R.vals.size() * sizeof(sptrsv_value_t)));
}

} // namespace

// m > 50000 engages the parallel transpose (APXCHOL_PAR_TRANSPOSE defaults
// on). The output must be byte-identical to the serial reference at every
// thread count — including T=1, and thread counts that do not divide m.
TEST(SpTRSVTranspose, ByteIdenticalToSerialReferenceAcrossThreadCounts) {
    const node_index m = 70000;   // > 50000: parallel path
    sparse_csc L = make_random_lower(m, 4.0, /*seed=*/12345);
    const ref_csr R = reference_transpose(L);
    ASSERT_GT(L.nonZeros(), edge_index(4) * m);   // sanity: dense enough

#ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
    // 1..4 under the ctest OMP_NUM_THREADS=4 cap; max_threads additionally
    // covers the full machine when the binary is run standalone.
    for (int threads : {1, 2, 3, 4, max_threads}) {
        omp_set_num_threads(threads);
        apxchol::omp_sptrsv trsv;
        trsv.setup(L, m);
        expect_byte_identical(trsv, R, threads);
    }
    omp_set_num_threads(max_threads);
#else
    apxchol::omp_sptrsv trsv;
    trsv.setup(L, m);
    expect_byte_identical(trsv, R, 1);
#endif
}

// Ascending-column order within each row is what the solve kernels rely on
// (diagonal last in every CSR row). Assert it directly on the parallel
// transpose's output rather than only via the reference comparison.
TEST(SpTRSVTranspose, RowsAreAscendingWithDiagonalLast) {
    const node_index m = 60000;   // > 50000: parallel path
    sparse_csc L = make_random_lower(m, 3.0, /*seed=*/777);

    apxchol::omp_sptrsv trsv;
    trsv.setup(L, m);

    const auto& row_ptr = trsv.csr_row_ptr();
    const auto& col_idx = trsv.csr_col_idx();
    ASSERT_EQ(row_ptr.back(), L.nonZeros());
    for (node_index i = 0; i < m; ++i) {
        ASSERT_LT(row_ptr[i], row_ptr[i + 1]);    // diagonal ⇒ no empty row
        for (edge_index p = row_ptr[i] + 1; p < row_ptr[i + 1]; ++p)
            ASSERT_LT(col_idx[p - 1], col_idx[p]) << "row " << i;
        EXPECT_EQ(col_idx[row_ptr[i + 1] - 1], i) << "diagonal not last in row " << i;
    }
}

// ── The GPU backend's host transpose (cuda_host::transpose_csr) ─────────
// The same shared implementation on int32 arrays with a plain value copy:
// byte-identical to an independent serial reference on the parallel path
// (m > 50000) at every thread count, and on the serial path.
namespace {

template <class V>
struct ref_csr_int {
    std::vector<int> ptr, idx;
    std::vector<V>   vals;
};

template <class V>
ref_csr_int<V> reference_transpose_int(const apxchol::cuda_host::csr_int<V>& A) {
    ref_csr_int<V> R;
    R.ptr.assign(static_cast<size_t>(A.m) + 1, 0);
    for (std::int64_t p = 0; p < A.nnz; ++p) R.ptr[A.idx[p] + 1]++;
    for (int i = 0; i < A.m; ++i) R.ptr[i + 1] += R.ptr[i];
    R.idx.resize(static_cast<size_t>(A.nnz));
    R.vals.resize(static_cast<size_t>(A.nnz));
    std::vector<int> pos(R.ptr.begin(), R.ptr.end());
    for (int i = 0; i < A.m; ++i)
        for (int p = A.ptr[i]; p < A.ptr[i + 1]; ++p) {
            const int d = pos[A.idx[p]]++;
            R.idx[d] = i; R.vals[d] = A.vals[p];
        }
    return R;
}

template <class V>
void expect_gpu_transpose_identical(const apxchol::cuda_host::csr_int<V>& LT, const ref_csr_int<V>& R,
                                    int threads) {
    SCOPED_TRACE("threads=" + std::to_string(threads));
    const auto Lc = apxchol::cuda_host::transpose_csr(LT);
    ASSERT_EQ(Lc.m, LT.m);
    ASSERT_EQ(Lc.nnz, LT.nnz);
    ASSERT_EQ(Lc.ptr.size(), R.ptr.size());
    EXPECT_EQ(0, std::memcmp(Lc.ptr.data(), R.ptr.data(), R.ptr.size() * sizeof(int)));
    EXPECT_EQ(0, std::memcmp(Lc.idx.get(), R.idx.data(), R.idx.size() * sizeof(int)));
    EXPECT_EQ(0, std::memcmp(Lc.vals.get(), R.vals.data(), R.vals.size() * sizeof(V)));
}

} // namespace

TEST(SpTRSVTranspose, GpuHostTransposeIsByteIdenticalToSerialReferenceAcrossThreadCounts) {
    const node_index m = 70000;   // > 50000: parallel path
    sparse_csc L = make_random_lower(m, 4.0, /*seed=*/12345);
    // The GPU backend's int32 CSC of L11 (== CSR of L^T), fp32 values, and its
    // fp16 bit-pattern twin (the fp16 storage transposes uint16_t values).
    const auto LT = apxchol::cuda_host::build_L11_csc_int<factor_value_t>(L, m);
    apxchol::cuda_host::csr_int<std::uint16_t> LT16;
    LT16.m = LT.m; LT16.nnz = LT.nnz; LT16.ptr = LT.ptr;
    LT16.idx  = std::make_unique<int[]>(static_cast<size_t>(LT.nnz));
    LT16.vals = std::make_unique<std::uint16_t[]>(static_cast<size_t>(LT.nnz));
    for (std::int64_t p = 0; p < LT.nnz; ++p) {
        LT16.idx[p]  = LT.idx[p];
        LT16.vals[p] = static_cast<std::uint16_t>((static_cast<std::uint64_t>(p) * 2654435761ull) >> 16);   // arbitrary bit patterns
    }
    const auto R   = reference_transpose_int(LT);
    const auto R16 = reference_transpose_int(LT16);
    ASSERT_GT(LT.nnz, static_cast<std::int64_t>(4) * m);

#ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
    for (int threads : {1, 2, 3, 4, max_threads}) {
        omp_set_num_threads(threads);
        expect_gpu_transpose_identical(LT, R, threads);
        expect_gpu_transpose_identical(LT16, R16, threads);
    }
    omp_set_num_threads(max_threads);
#else
    expect_gpu_transpose_identical(LT, R, 1);
    expect_gpu_transpose_identical(LT16, R16, 1);
#endif
    // And it IS the CPU backend's CSR of L11 (same shared code, same input):
    // row pointers, column indices, values (fp32 storage: a plain cast).
    apxchol::omp_sptrsv trsv;
    trsv.setup(L, m);
    ASSERT_EQ(trsv.csr_row_ptr().size(), R.ptr.size());
    for (node_index i = 0; i <= m; ++i)
        ASSERT_EQ(static_cast<edge_index>(R.ptr[i]), trsv.csr_row_ptr()[i]) << "row_ptr " << i;
    std::uint64_t mism = 0;
    for (std::int64_t k = 0; k < LT.nnz; ++k)
        mism += static_cast<node_index>(R.idx[k]) != trsv.csr_col_idx()[k];
    EXPECT_EQ(mism, 0u);
    for (std::int64_t k = 0; k < LT.nnz; ++k)
        mism += !(static_cast<sptrsv_value_t>(R.vals[k]) == trsv.csr_vals()[k]);
    EXPECT_EQ(mism, 0u);
}

TEST(SpTRSVTranspose, GpuHostTransposeSerialPathMatchesReference) {
    const node_index m = 20000;   // <= 50000: serial path
    sparse_csc L = make_random_lower(m, 5.0, /*seed=*/4242);
    const auto LT = apxchol::cuda_host::build_L11_csc_int<factor_value_t>(L, m);
    const auto R  = reference_transpose_int(LT);
    expect_gpu_transpose_identical(LT, R, 0);
}

// The serial fallback (m <= 50000) must agree with the reference too.
TEST(SpTRSVTranspose, SerialFallbackMatchesReference) {
    const node_index m = 20000;   // <= 50000: serial path
    sparse_csc L = make_random_lower(m, 5.0, /*seed=*/4242);
    const ref_csr R = reference_transpose(L);

    apxchol::omp_sptrsv trsv;
    trsv.setup(L, m);
    expect_byte_identical(trsv, R, 0);
}
