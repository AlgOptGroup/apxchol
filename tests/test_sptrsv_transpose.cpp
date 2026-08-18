// Byte-identity tests for the parallel CSC→CSR transpose in omp_sptrsv::setup.
//
// The blocked counting-sort transpose (APXCHOL_PAR_TRANSPOSE, engaged for
// m > 50000) must produce a CSR that is byte-identical to the classic serial
// count/prefix/scatter transpose — same csr_row_ptr_, same csr_col_idx_
// (ascending column order within every row), same csr_vals_ bit patterns —
// for ANY thread count. These tests build a random sparse lower-triangular
// factor big enough to engage the parallel path and compare against an
// independent serial reference at several thread counts.

#include <gtest/gtest.h>
#include <cstring>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "apxchol/sparse_csc.h"
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
            R.vals[out]    = apxchol::omp_sptrsv::narrow_value(vals[p], s_j, /*fp16_flush_subnormal=*/true);
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

// The serial fallback (m <= 50000) must agree with the reference too.
TEST(SpTRSVTranspose, SerialFallbackMatchesReference) {
    const node_index m = 20000;   // <= 50000: serial path
    sparse_csc L = make_random_lower(m, 5.0, /*seed=*/4242);
    const ref_csr R = reference_transpose(L);

    apxchol::omp_sptrsv trsv;
    trsv.setup(L, m);
    expect_byte_identical(trsv, R, 0);
}
