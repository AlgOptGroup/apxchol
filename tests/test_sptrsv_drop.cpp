// Compaction in omp_sptrsv::setup on default fp32 storage: keep off-diagonals
// with |v| >= drop_tolerance * the pre-drop column maximum, always keep the
// diagonal, remove stored zeros, and report the kept/dropped counts.
// Column-sum compensation is unconditional. Forward/back solves must match
// the reference with dropped entries zeroed and the same compensation applied.
// CSR/CSC arrays must agree byte for byte across thread counts. Zero disables
// threshold dropping; values outside [0, 1] and NaN are invalid.
// GPU-host fp16 preparation is checked below; CPU fp16 storage in test_lowprec.cpp.
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <omp.h>

#include "apxchol/sparse_csc.h"
#include "apxchol/solver/sptrsv/omp.h"

using csc_access = apxchol::detail::sparse_csc_access;

using apxchol::edge_index;
using apxchol::node_index;
using apxchol::sparse_csc;
using apxchol::factor_value_t;
using apxchol::detail::sptrsv_value_t;

namespace {

// Random lower-triangular factor: every column gets a diagonal entry (first,
// positive) plus `avg_offdiag` random strictly-lower entries on average
// (NEGATIVE, like the M-matrix factors the elimination produces; magnitudes
// uniform in [vmin, vmax]); a sprinkling of hub columns get several dozen
// more.
sparse_csc make_random_lower(node_index m, double avg_offdiag, unsigned seed,
                             double vmin = 0.0, double vmax = 0.5) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uval(vmin, vmax);
    std::uniform_real_distribution<double> udiag(1.0, 3.0);
    std::poisson_distribution<int> pcount(avg_offdiag);
    sparse_csc L;
    csc_access::n(L) = m;
    csc_access::outer(L).assign(static_cast<size_t>(m) + 1, 0);
    std::vector<std::vector<node_index>> col_rows(m);
    for (node_index j = 0; j < m; ++j) {
        int k = pcount(rng);
        if (j % 997 == 0) k += 60;
        auto& rows = col_rows[j];
        rows.push_back(j);
        if (j + 1 < m) {
            std::uniform_int_distribution<node_index> urow(j + 1, m - 1);
            for (int t = 0; t < k; ++t) rows.push_back(urow(rng));
        }
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    }
    for (node_index j = 0; j < m; ++j)
        csc_access::outer(L)[j + 1] = csc_access::outer(L)[j] + static_cast<edge_index>(col_rows[j].size());
    csc_access::inner(L).resize(static_cast<size_t>(csc_access::outer(L)[m]));
    csc_access::vals(L).resize(static_cast<size_t>(csc_access::outer(L)[m]));
    for (node_index j = 0; j < m; ++j) {
        edge_index out = csc_access::outer(L)[j];
        for (node_index r : col_rows[j]) {
            csc_access::inner(L)[out] = r;
            csc_access::vals(L)[out]  = static_cast<factor_value_t>(r == j ? udiag(rng) : -uval(rng));
            ++out;
        }
    }
    return L;
}

// Per-column scale of L11 = L.topLeftCorner(m, m) as documented (max
// |off-diagonal|, 1.0 if none; rows >= m excluded -- the Laplacian path's
// grounded last row is not part of what the SpTRSV sees). Computed here
// independently of omp_sptrsv::column_scale.
std::vector<double> reference_scales_L11(const sparse_csc& L, node_index m) {
    std::vector<double> s(m, 1.0);
    for (node_index j = 0; j < m; ++j) {
        double mx = 0.0;
        for (edge_index p = csc_access::outer(L)[j] + 1; p < csc_access::outer(L)[j + 1]; ++p)
            if (csc_access::inner(L)[p] < m) mx = std::max(mx, std::fabs(static_cast<double>(csc_access::vals(L)[p])));
        s[j] = mx > 0.0 ? mx : 1.0;
    }
    return s;
}

// The reference "dense drop": the same factor with the entries the drop
// removes set to ZERO in place (nothing removed). Kept iff |v| >= rel * s_j.
// Each column's dropped mass is folded back into its kept off-diagonals in
// proportion to |v| (v += dropped_sum * |v| / sum|kept|), which preserves the
// column sum -- unconditionally, as in the implementation. Written out here
// without omp_sptrsv::keep_offdiag / setup so predicate and compensation are
// stated twice independently.
struct dense_drop {
    sparse_csc    Lz;         // zeroed (and compensated) copy
    std::uint64_t kept   = 0; // diagonal + kept off-diagonals of L11 (== stored nnz after the drop)
    std::uint64_t zeroed = 0;
};

dense_drop reference_dense_drop(const sparse_csc& L, node_index m, double rel) {
    dense_drop d;
    d.Lz = L;
    const std::vector<double> s = reference_scales_L11(L, m);
    for (node_index j = 0; j < m; ++j) {
        double dropped_sum = 0.0, kept_abs = 0.0;
        for (edge_index p = csc_access::outer(L)[j]; p < csc_access::outer(L)[j + 1]; ++p) {
            if (csc_access::inner(L)[p] >= m) continue;                    // outside L11 (Laplacian last row)
            if (csc_access::inner(L)[p] == j) { ++d.kept; continue; }
            const double v = static_cast<double>(csc_access::vals(L)[p]);
            const bool keep = std::fabs(v) >= rel * s[j];
            if (keep) { ++d.kept; kept_abs += std::fabs(v); }
            else      { ++d.zeroed; dropped_sum += v; csc_access::vals(d.Lz)[p] = 0; }
        }
        // (Same double arithmetic order as setup -- per_abs first -- so the
        // compacted values are bit-identical to these, which the 1e-12 solve
        // comparison below relies on.)
        if (dropped_sum != 0.0 && kept_abs > 0.0) {
            const double per_abs = dropped_sum / kept_abs;
            for (edge_index p = csc_access::outer(L)[j]; p < csc_access::outer(L)[j + 1]; ++p) {
                if (csc_access::inner(L)[p] >= m || csc_access::inner(L)[p] == j || csc_access::vals(d.Lz)[p] == 0) continue;
                const double v = static_cast<double>(csc_access::vals(d.Lz)[p]);
                csc_access::vals(d.Lz)[p] = static_cast<factor_value_t>(v + per_abs * std::fabs(v));
            }
        }
    }
    return d;
}

// Column sums (diagonal + off-diagonals) of L11 = L.topLeftCorner(m, m).
std::vector<double> column_sums_L11(const sparse_csc& L, node_index m) {
    std::vector<double> cs(m, 0.0);
    for (node_index j = 0; j < m; ++j)
        for (edge_index p = csc_access::outer(L)[j]; p < csc_access::outer(L)[j + 1]; ++p)
            if (csc_access::inner(L)[p] < m) cs[j] += static_cast<double>(csc_access::vals(L)[p]);
    return cs;
}

// These fp32 drop checks introduce no storage rounding after factor construction.
constexpr double kStorageEps =
    0.0;
// The fp32 reference uses unscaled factor values.
double pair_scale(double s_pre) {
    (void)s_pre;
    return 1.0;
}
// Widen the stored fp32 value; pair_scale is one for this reference.
double effective(sptrsv_value_t w, double s_pre) { return apxchol::widen(w) * pair_scale(s_pre); }

std::uint64_t L11_nnz_of(const sparse_csc& L, node_index m) {
    std::uint64_t n = 0;
    for (node_index j = 0; j < m; ++j)
        for (edge_index p = csc_access::outer(L)[j]; p < csc_access::outer(L)[j + 1]; ++p) n += csc_access::inner(L)[p] < m;
    return n;
}

// Magnitudes spread over ~9 decades: v = -10^u, u uniform in [-7, 2), so
// about half of every column sits below 1e-4 * s_j; a few explicit zeros.
sparse_csc make_wide_magnitude_lower(node_index n) {
    sparse_csc L = make_random_lower(n, 6.0, 4242 + n, 0.0, 1.0);
    std::mt19937 rng(77 + n);
    std::uniform_real_distribution<double> uexp(-7.0, 2.0);
    for (node_index j = 0; j < n; ++j)
        for (edge_index p = csc_access::outer(L)[j] + 1; p < csc_access::outer(L)[j + 1]; ++p) {
            csc_access::vals(L)[p] = static_cast<factor_value_t>(-std::pow(10.0, uexp(rng)));
            if ((p % 101) == 5) csc_access::vals(L)[p] = 0;   // explicit zeros: always dropped
        }
    return L;
}

} // namespace

// Dropping removes |v| < rel * s_j before building CSR/CSC. Check kept counts,
// diagonal placement and unconditional column-sum compensation, then compare
// both solves with the zeroed reference to about 1e-12. Cover SDDM (m == n),
// Laplacian (m == n-1), and serial/parallel transpose (m > 50000).
TEST(SpTRSVDrop, CompactsToKeptEntriesAndSolvesLikeTheZeroedReference) {

    struct cfg { node_index n; bool laplacian; };
    for (cfg c : {cfg{3000, false}, cfg{3001, true}, cfg{60000, false}, cfg{60001, true}}) {
        const node_index m = c.laplacian ? c.n - 1 : c.n;
        SCOPED_TRACE("n=" + std::to_string(c.n) + (c.laplacian ? " (Laplacian m=n-1)" : " (SDDM m=n)") +
                     ", column sums preserved");
        const sparse_csc L = make_wide_magnitude_lower(c.n);
        const double rel = 1e-4;
        const dense_drop ref = reference_dense_drop(L, m, rel);
        ASSERT_GT(ref.zeroed, ref.kept / 4);   // the test has teeth: a big chunk goes
        const std::uint64_t L11_nnz = L11_nnz_of(L, m);

        // Compacted with the explicit threshold.
        apxchol::detail::omp_sptrsv trsv;
        trsv.setup(L, m, {.drop_tolerance = 1e-4});
        // Reference: dense drop, drop OFF.
        apxchol::detail::omp_sptrsv rf;
        rf.setup(ref.Lz, m, {.drop_tolerance = 0});

        // Counts.
        const auto& st = trsv.drop_stats();
        EXPECT_EQ(st.rel, rel);
        EXPECT_EQ(st.nnz_factor, L11_nnz);
        EXPECT_EQ(st.nnz_stored, ref.kept);
        EXPECT_EQ(trsv.stored_nnz(), ref.kept);
        EXPECT_EQ(st.dropped, ref.zeroed);
        EXPECT_EQ(st.nnz_factor - st.dropped, st.nnz_stored);
        EXPECT_EQ(trsv.csc_vals().size(), ref.kept);
        EXPECT_EQ(trsv.csr_vals().size(), ref.kept);
        EXPECT_EQ(trsv.csc_row_idx().size(), ref.kept);
        EXPECT_EQ(trsv.csr_col_idx().size(), ref.kept);
        EXPECT_EQ(trsv.csc_col_ptr().back(), static_cast<edge_index>(ref.kept));
        EXPECT_EQ(trsv.csr_row_ptr().back(), static_cast<edge_index>(ref.kept));
        // No stored zeros survive the drop; every kept off-diagonal is above
        // the threshold; the diagonal is first in every CSC column and last in
        // every CSR row; with the compensation every column sums to what it
        // summed to before (up to fp32 rounding of the rescaled entries),
        // without it the column sum has lost exactly the dropped mass.
        {
            const std::vector<double> s  = reference_scales_L11(L, m);
            const std::vector<double> cs = column_sums_L11(L, m);
            std::uint64_t zeros = 0, below = 0;
            double worst_cs = 0.0;
            for (node_index j = 0; j < m; ++j) {
                ASSERT_EQ(trsv.csc_row_idx()[trsv.csc_col_ptr()[j]], j) << "diagonal first, col " << j;
                // The diagonal the kernels divide by (fp32 L_jj / s_j under
                // *_SCALED, the fp32 diag_ elsewhere on lowprec, the stored
                // value itself on fp32/fp64), back in the factor's units.
                double sum = apxchol::detail::omp_sptrsv::stored_diag(csc_access::vals(L)[csc_access::outer(L)[j]], static_cast<float>(pair_scale(s[j])))
                           * pair_scale(s[j]);
                double abs = std::fabs(sum);
                for (edge_index p = trsv.csc_col_ptr()[j] + 1; p < trsv.csc_col_ptr()[j + 1]; ++p) {
                    const double v = effective(trsv.csc_vals()[p], s[j]);
                    zeros += v == 0.0;
                    below += std::fabs(v) < rel * s[j] * (1.0 - 2.0 * kStorageEps);
                    sum += v; abs += std::fabs(v);
                }
                worst_cs = std::max(worst_cs, std::fabs(sum - cs[j]) / abs);
            }
            for (node_index i = 0; i < m; ++i)
                ASSERT_EQ(trsv.csr_col_idx()[trsv.csr_row_ptr()[i + 1] - 1], i) << "diagonal last, row " << i;
            EXPECT_EQ(zeros, 0u);
            EXPECT_EQ(below, 0u);
            // fp32 rounding of the rescaled entries (fp64: none), plus each
            // stored entry's own storage rounding on the lowprec builds.
            EXPECT_LT(worst_cs, (sizeof(factor_value_t) == 4 ? 1e-5 : 1e-13) + 2.0 * kStorageEps);
        }
        // Value contract: every stored off-diagonal is narrow_value() of the
        // reference's (compensated) value at the PRE-drop scale s_j -- bit for
        // bit (the same pure function setup uses; a plain cast on fp32/fp64).
        {
            const std::vector<double> s = reference_scales_L11(L, m);
            std::uint64_t mismatches = 0;
            for (node_index j = 0; j < m; ++j) {
                edge_index q = trsv.csc_col_ptr()[j] + 1;
                for (edge_index p = csc_access::outer(L)[j] + 1; p < csc_access::outer(L)[j + 1]; ++p) {
                    if (csc_access::inner(L)[p] >= m) continue;
                    if (std::fabs(static_cast<double>(csc_access::vals(L)[p])) < rel * s[j]) continue;   // dropped
                    ASSERT_LT(q, trsv.csc_col_ptr()[j + 1]);
                    ASSERT_EQ(trsv.csc_row_idx()[q], csc_access::inner(L)[p]);
                    const sptrsv_value_t expect = apxchol::detail::omp_sptrsv::narrow_value(
                        csc_access::vals(ref.Lz)[p], static_cast<float>(pair_scale(s[j])));
                    mismatches += !(trsv.csc_vals()[q] == expect);
                    ++q;
                }
                ASSERT_EQ(q, trsv.csc_col_ptr()[j + 1]);
            }
            EXPECT_EQ(mismatches, 0u);
        }
        // The reference kept everything (drop OFF): its stored nnz is the full L11.
        EXPECT_EQ(rf.drop_stats().rel, 0.0);
        EXPECT_EQ(rf.drop_stats().nnz_stored, L11_nnz);
        EXPECT_EQ(rf.drop_stats().dropped, 0u);

        // Solves agree to ~1e-12 (same arithmetic up to the 4-way accumulator
        // grouping; the zeroed terms contribute exact zeros). Not stated for
        // the *_SCALED variants WITH the compensation: the reference factor's
        // column max is the compensated one, so its per-column scale -- hence
        // its rounding, its forward output D y and its back input -- differ
        // from the compacted factor's pre-drop s_j (the value contract above
        // and the residual below cover that case).
        std::mt19937 rng(9 + c.n);
        std::uniform_real_distribution<double> ux(-1.0, 1.0);
        std::vector<double> x(m), y1(m), y2(m), z1(m), z2(m);
        for (auto& v : x) v = ux(rng);
        trsv.forward_solve(x.data(), y1.data());
        rf.forward_solve(x.data(), y2.data());
        trsv.transpose_solve(x.data(), z1.data());
        rf.transpose_solve(x.data(), z2.data());
        double worst_f = 0.0, worst_b = 0.0, scale_f = 0.0, scale_b = 0.0;
        for (node_index i = 0; i < m; ++i) {
            worst_f = std::max(worst_f, std::fabs(y1[i] - y2[i]));
            worst_b = std::max(worst_b, std::fabs(z1[i] - z2[i]));
            scale_f = std::max(scale_f, std::fabs(y2[i]));
            scale_b = std::max(scale_b, std::fabs(z2[i]));
        }
        const bool same_scales = true;
        if (same_scales) {
            EXPECT_LT(worst_f, 1e-12 * std::max(1.0, scale_f)) << "forward";
            EXPECT_LT(worst_b, 1e-12 * std::max(1.0, scale_b)) << "back";
        }
        // The compacted forward solve must satisfy the stored fp32 factor
        // to componentwise FP64 roundoff.
        {
            const std::vector<double> s = reference_scales_L11(L, m);
            double worst = 0.0;
            for (node_index i = 0; i < m; ++i) {
                double r = x[i], sc = std::fabs(x[i]);
                for (edge_index q = trsv.csr_row_ptr()[i]; q < trsv.csr_row_ptr()[i + 1]; ++q) {
                    const node_index j = trsv.csr_col_idx()[q];
                    const double v = (j == i)
                        ? apxchol::detail::omp_sptrsv::stored_diag(csc_access::vals(L)[csc_access::outer(L)[i]], static_cast<float>(pair_scale(s[i])))
                        : apxchol::widen(trsv.csr_vals()[q]);
                    const double t = v * y1[j];
                    r -= t; sc += std::fabs(t);
                }
                worst = std::max(worst, std::fabs(r) / (sc + 1e-300));
            }
            EXPECT_LT(worst, 1e-11);
        }

        // Default setup uses the same 1e-4 threshold; zero disables dropping.
        {
            apxchol::detail::omp_sptrsv dflt;
            dflt.setup(L, m);
            EXPECT_EQ(dflt.drop_stats().rel, apxchol::factor_options{}.drop_tolerance);
            EXPECT_EQ(dflt.stored_nnz(), ref.kept);
            EXPECT_EQ(dflt.drop_stats().dropped, ref.zeroed);
            EXPECT_EQ(0, std::memcmp(dflt.csc_vals().data(), trsv.csc_vals().data(),
                                     ref.kept * sizeof(sptrsv_value_t)));
            apxchol::detail::omp_sptrsv off;
            off.setup(L, m, {.drop_tolerance = 0});
            EXPECT_EQ(off.drop_stats().dropped, 0u);
            EXPECT_EQ(off.stored_nnz(), L11_nnz);
        }
    }
}

TEST(SpTRSVDrop, ExplicitThresholdValidation) {
    const auto L = make_random_lower(3, 1, 3);
    EXPECT_EQ(apxchol::factor_options{}.drop_tolerance, 1e-4);
    for (double rel : {0.0, 3e-4, 1.0}) {
        apxchol::detail::omp_sptrsv trsv;
        trsv.setup(L, L.rows(), {.drop_tolerance = rel});
        EXPECT_EQ(trsv.drop_stats().rel, rel);
    }
    for (double rel : {-1.0, std::nextafter(1.0, 2.0), 2.0, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
        apxchol::detail::omp_sptrsv trsv;
        EXPECT_THROW(trsv.setup(L, L.rows(), {.drop_tolerance = rel}),
                     std::invalid_argument);
    }
}

// rel = 0 disables threshold dropping; a rel below every entry drops nothing (and
// makes no copy: stored nnz == factor nnz).
TEST(SpTRSVDrop, EdgeCases) {

    const node_index m = 2000;
    sparse_csc L = make_random_lower(m, 4.0, 31, 0.0, 1.0);
    for (edge_index p = 0; p < L.nonZeros(); ++p)
        if ((p % 5) == 2) csc_access::vals(L)[p] = static_cast<factor_value_t>(csc_access::vals(L)[p] * 1e-9);   // far below 1e-4 * s_j
    const std::uint64_t nnz = static_cast<std::uint64_t>(L.nonZeros());
    ASSERT_GT(reference_dense_drop(L, m, 1e-4).zeroed, 100u);   // the default would remove plenty here
    // No value here is zero in fp32, so format_flushes must remove nothing
    // when threshold dropping is disabled.
    const std::vector<double> s_pre = reference_scales_L11(L, m);
    std::uint64_t fmt_zero = 0;
    for (node_index j = 0; j < m; ++j)
        for (edge_index p = csc_access::outer(L)[j] + 1; p < csc_access::outer(L)[j + 1]; ++p)
            fmt_zero += apxchol::detail::omp_sptrsv::format_flushes(csc_access::vals(L)[p], static_cast<float>(s_pre[j]));
    ASSERT_EQ(fmt_zero, 0u);
    {
        apxchol::detail::omp_sptrsv t;
        t.setup(L, m, {.drop_tolerance = 0});
        EXPECT_EQ(t.drop_stats().rel, 0.0);
        EXPECT_EQ(t.drop_stats().dropped, 0u);
        EXPECT_EQ(t.stored_nnz(), nnz);
        EXPECT_EQ(t.csc_vals().size(), nnz);
    }
    {
        // Below everything: only storage-format zeros are dropped.
        apxchol::detail::omp_sptrsv t; t.setup(L, m, {.drop_tolerance = 1e-30});
        EXPECT_EQ(t.drop_stats().rel, 1e-30);
        EXPECT_EQ(t.drop_stats().dropped, fmt_zero);
        EXPECT_EQ(t.drop_stats().dropped_threshold, 0u);
        EXPECT_EQ(t.drop_stats().dropped_flush, fmt_zero);
        EXPECT_EQ(t.stored_nnz(), nnz - fmt_zero);
        EXPECT_EQ(t.csc_col_ptr().back(), static_cast<edge_index>(nnz - fmt_zero));
        if (fmt_zero == 0) {
            // Nothing dropped, so nothing compensated: the CSC IS the factor
            // (through the storage contract: a plain cast on fp32/fp64).
            EXPECT_EQ(0, std::memcmp(t.csc_row_idx().data(), csc_access::inner(L).data(), nnz * sizeof(node_index)));
            std::uint64_t mismatches = 0;
            for (node_index j = 0; j < m; ++j)
                for (edge_index p = csc_access::outer(L)[j]; p < csc_access::outer(L)[j + 1]; ++p)
                    mismatches += !(t.csc_vals()[p] == apxchol::detail::omp_sptrsv::narrow_value(
                        csc_access::vals(L)[p], static_cast<float>(pair_scale(s_pre[j]))));
            EXPECT_EQ(mismatches, 0u);
        }
    }
}

TEST(SpTRSVDrop, DiagonalFactorsWorkAtRangeEndpoints) {
    sparse_csc L;
    csc_access::n(L) = 3;
    csc_access::outer(L) = {0, 1, 2, 3};
    csc_access::inner(L) = {0, 1, 2};
    csc_access::vals(L) = {1, 2, 4};
    for (double rel : {0.0, 1.0}) {
        apxchol::detail::omp_sptrsv trsv;
        trsv.setup(L, 3, {.drop_tolerance = rel});
        EXPECT_EQ(trsv.drop_stats().dropped, 0u);
        EXPECT_EQ(trsv.stored_nnz(), 3u);
        std::vector<double> x(3, 1.0), y(3), z(3);
        trsv.forward_solve(x.data(), y.data());
        trsv.transpose_solve(y.data(), z.data());
        EXPECT_EQ(y, (std::vector<double>{1, 0.5, 0.25}));
        EXPECT_EQ(z, (std::vector<double>{1, 0.25, 0.0625}));
    }
}

// The drop is deterministic: no atomics, no thread-arrival order -- the kept
// set is a per-entry predicate and the compacted order is the input order.
// The CSR (parallel transpose path, m > 50000) and the CSC it produces are
// byte-identical at every thread count, and identical to the compaction of
// the reference dense drop's nonzero pattern.
TEST(SpTRSVDrop, CompactedArraysAreByteIdenticalAcrossThreadCounts) {
    const node_index n = 70001;                       // Laplacian path, m > 50000
    const node_index m = n - 1;
    const sparse_csc L = make_wide_magnitude_lower(n);
    apxchol::detail::omp_sptrsv ref;
    const int max_threads = omp_get_max_threads();
    omp_set_num_threads(1);
    ref.setup(L, m);
    ASSERT_GT(ref.drop_stats().dropped, static_cast<std::uint64_t>(m));   // teeth
    ASSERT_EQ(ref.drop_stats().dropped, reference_dense_drop(L, m, 1e-4).zeroed);
    auto same = [&](const apxchol::detail::omp_sptrsv& t) {
        ASSERT_EQ(t.stored_nnz(), ref.stored_nnz());
        ASSERT_EQ(t.csr_row_ptr().size(), ref.csr_row_ptr().size());
        ASSERT_EQ(t.csr_col_idx().size(), ref.csr_col_idx().size());
        ASSERT_EQ(t.csc_col_ptr().size(), ref.csc_col_ptr().size());
        ASSERT_EQ(t.csc_row_idx().size(), ref.csc_row_idx().size());
        EXPECT_EQ(0, std::memcmp(t.csr_row_ptr().data(), ref.csr_row_ptr().data(),
                                 ref.csr_row_ptr().size() * sizeof(edge_index)));
        EXPECT_EQ(0, std::memcmp(t.csr_col_idx().data(), ref.csr_col_idx().data(),
                                 ref.csr_col_idx().size() * sizeof(node_index)));
        EXPECT_EQ(0, std::memcmp(t.csr_vals().data(), ref.csr_vals().data(),
                                 ref.csr_vals().size() * sizeof(sptrsv_value_t)));
        EXPECT_EQ(0, std::memcmp(t.csc_col_ptr().data(), ref.csc_col_ptr().data(),
                                 ref.csc_col_ptr().size() * sizeof(edge_index)));
        EXPECT_EQ(0, std::memcmp(t.csc_row_idx().data(), ref.csc_row_idx().data(),
                                 ref.csc_row_idx().size() * sizeof(node_index)));
        EXPECT_EQ(0, std::memcmp(t.csc_vals().data(), ref.csc_vals().data(),
                                 ref.csc_vals().size() * sizeof(sptrsv_value_t)));
    };
    for (int threads : {2, 3, 4, max_threads}) {
        SCOPED_TRACE("threads=" + std::to_string(threads));
        omp_set_num_threads(threads);
        apxchol::detail::omp_sptrsv t;
        t.setup(L, m);
        same(t);
    }
    omp_set_num_threads(max_threads);
}

// ── GPU backend host preparation (gpu_factor_prep.h; CUDA-free, so stated here on
// every build) ────────────────────────────────────────────────────────────────
//
// cuda_sptrsv::setup builds L11 as int32 CSC arrays, runs THE compacting drop
// (factor_drop.h -- the same implementation omp_sptrsv::setup runs) on them
// and uploads the result. What it uploads must be what the CPU backend
// stores: same stored nnz, same column pointers / row indices, same values
// (bit-for-bit through the fp32/fp64 storage; through narrow_value() on the
// lowprec builds), for the Laplacian (m = n-1) and SDDM (m = n) paths, drop
// on (explicit and default) and off.
#include "apxchol/solver/sptrsv/gpu_factor_prep.h"

TEST(GpuHostPrep, DropOnTheGpuHostArraysIsTheCpuDrop) {

    struct cfg { node_index n; bool laplacian; };
    for (double rel : {1e-4, 0.0})
    for (cfg c : {cfg{3000, false}, cfg{3001, true}, cfg{60001, true}}) {
        const node_index m = c.laplacian ? c.n - 1 : c.n;
        SCOPED_TRACE("n=" + std::to_string(c.n) + (c.laplacian ? " (Laplacian m=n-1)" : " (SDDM m=n)") +
                     " drop_tolerance=" + std::to_string(rel));
        const sparse_csc L = make_wide_magnitude_lower(c.n);
        // CPU backend.
        apxchol::detail::omp_sptrsv trsv;
        trsv.setup(L, m, {.drop_tolerance = rel});
        // GPU backend's host side, exactly as cuda_sptrsv::setup does it (fp32
        // storage: keep predicate = |v| >= rel * s_j and v != 0).
        auto LT = apxchol::detail::gpu_factor_prep::build_L11_csc_int<factor_value_t>(L, m);
        const std::vector<float> scales = apxchol::detail::gpu_factor_prep::column_scales(LT);
        const std::uint64_t L11_nnz = L11_nnz_of(L, m);
        ASSERT_EQ(static_cast<std::uint64_t>(LT.nnz), L11_nnz);
        const apxchol::detail::factor_drop_stats st = apxchol::detail::gpu_factor_prep::apply_factor_drop(
            LT, scales, rel, /*fp16_storage=*/false);
        // Same statistics ...
        const auto& cs = trsv.drop_stats();
        EXPECT_EQ(st.rel, cs.rel);
        EXPECT_EQ(st.nnz_factor, cs.nnz_factor);
        EXPECT_EQ(st.nnz_stored, cs.nnz_stored);
        EXPECT_EQ(st.dropped, cs.dropped);
        EXPECT_EQ(st.dropped_threshold, cs.dropped_threshold);
        EXPECT_EQ(st.dropped_flush, cs.dropped_flush);
        EXPECT_EQ(static_cast<std::uint64_t>(LT.nnz), trsv.stored_nnz());
        if (rel == 1e-4) { ASSERT_GT(st.dropped, L11_nnz / 4); }   // teeth
        if (rel == 0)    { ASSERT_EQ(st.dropped, 0u); }
        // ... same structure ...
        ASSERT_EQ(trsv.csc_col_ptr().size(), static_cast<size_t>(m) + 1);
        for (node_index j = 0; j <= m; ++j)
            ASSERT_EQ(static_cast<edge_index>(LT.ptr[j]), trsv.csc_col_ptr()[j]) << "col_ptr " << j;
        std::uint64_t idx_mismatch = 0, val_mismatch = 0;
        for (std::int64_t k = 0; k < LT.nnz; ++k) {
            idx_mismatch += static_cast<node_index>(LT.idx[k]) != trsv.csc_row_idx()[k];
        }
        EXPECT_EQ(idx_mismatch, 0u);
        // ... same values: the GPU uploads the compacted fp32 factor value; the
        // CPU stores narrow_value() of it (a plain cast on fp32/fp64, so
        // bit-for-bit; through the storage format on the lowprec builds, at
        // the pre-drop scale s_j -- which is what column_scales() computed).
        for (node_index j = 0; j < m; ++j) {
            const float s = scales[j];
            for (int k = LT.ptr[j]; k < LT.ptr[j + 1]; ++k) {
                const sptrsv_value_t expect = apxchol::detail::omp_sptrsv::narrow_value(
                    LT.vals[k], s);
                val_mismatch += !(trsv.csc_vals()[k] == expect);
            }
        }
        EXPECT_EQ(val_mismatch, 0u);
        // And the GPU-side transpose (CSR of L for the dataflow forward
        // sweep) is the CPU's CSR: same row pointers, column indices, values.
        const auto Lc = apxchol::detail::gpu_factor_prep::transpose_csr(LT);
        ASSERT_EQ(Lc.nnz, LT.nnz);
        for (node_index i = 0; i <= m; ++i)
            ASSERT_EQ(static_cast<edge_index>(Lc.ptr[i]), trsv.csr_row_ptr()[i]) << "row_ptr " << i;
        std::uint64_t t_mismatch = 0;
        for (std::int64_t k = 0; k < Lc.nnz; ++k)
            t_mismatch += static_cast<node_index>(Lc.idx[k]) != trsv.csr_col_idx()[k];
        EXPECT_EQ(t_mismatch, 0u);
        for (node_index i = 0; i < m; ++i)
            for (int k = Lc.ptr[i]; k < Lc.ptr[i + 1]; ++k) {
                const node_index j = static_cast<node_index>(Lc.idx[k]);
                const sptrsv_value_t expect = apxchol::detail::omp_sptrsv::narrow_value(Lc.vals[k], scales[j]);
                t_mismatch += !(trsv.csr_vals()[k] == expect);
            }
        EXPECT_EQ(t_mismatch, 0u);
    }
}

// The fp16 per-column-scaled storage the dataflow backend uploads under
// FP16 factor storage (gpu_factor_prep.h file header): each slot is
// binary16(fp32(v) / s_j) RNE with fp16 subnormals flushed to signed zero
// (restated here with native _Float16 conversions), diag[j] =
// fp32(L_jj) / s_j, inv_scale2[j] = fp64(fp32(1 / s_j))^2. The column's
// rounding residual is always folded into diag[j], compensating the stored
// column sum to fp32 precision. The drop predicate also removes fp16-flushed values.
TEST(GpuHostPrep, Fp16ScaledStorageContract) {
    for (bool laplacian : {false, true}) {
        const node_index n = 4001;
        const node_index m = laplacian ? n - 1 : n;
        SCOPED_TRACE(laplacian ? "Laplacian m=n-1" : "SDDM m=n");
        const sparse_csc L = make_wide_magnitude_lower(n);
        const double rel = 1e-4;
        auto LT = apxchol::detail::gpu_factor_prep::build_L11_csc_int<factor_value_t>(L, m);
        const std::vector<float> scales = apxchol::detail::gpu_factor_prep::column_scales(LT);
        {
            const std::vector<double> ref = reference_scales_L11(L, m);
            for (node_index j = 0; j < m; ++j) ASSERT_EQ(scales[j], static_cast<float>(ref[j])) << j;
        }
        // The fp16 keep predicate: threshold AND not fp16-flushed at |v/s|
        // (with the subnormal flush: < 2^-14 after rounding); at rel = 1e-4
        // > 2^-14 the second clause adds nothing, at rel = 1e-30 it is what
        // remains.
        {
            std::uint64_t only_fmt = 0, disagree = 0;
            for (node_index j = 0; j < m; ++j)
                for (int k = LT.ptr[j] + 1; k < LT.ptr[j + 1]; ++k) {
                    const float v = static_cast<float>(LT.vals[k]);
                    const bool thr = std::fabs(static_cast<double>(v)) >= rel * static_cast<double>(scales[j]);
                    const auto h = _Float16(v / scales[j]);
                    const bool fmt = h != 0 && !apxchol::is_stored_subnormal(h);
                    disagree += apxchol::detail::gpu_factor_prep::keep_offdiag(LT.vals[k], scales[j], rel, true) != (thr && fmt);
                    disagree += apxchol::detail::gpu_factor_prep::keep_offdiag(LT.vals[k], scales[j], 1e-30, true) != (v != 0.0f && fmt);
                    disagree += apxchol::detail::gpu_factor_prep::keep_offdiag(LT.vals[k], scales[j], 1e-30, false) != (v != 0.0f);
                    only_fmt += (v != 0.0f) && !fmt;
                }
            EXPECT_EQ(disagree, 0u);
            EXPECT_GT(only_fmt, 0u);   // teeth: the wide-magnitude factor has fp16-subnormal entries
        }
        const apxchol::detail::factor_drop_stats st = apxchol::detail::gpu_factor_prep::apply_factor_drop(
            LT, scales, rel, /*fp16_storage=*/true);
        ASSERT_GT(st.dropped, 0u);
        {
            const auto h16 = apxchol::detail::gpu_factor_prep::narrow_fp16_scaled(LT, scales);
            ASSERT_EQ(h16.diag.size(), static_cast<size_t>(m));
            ASSERT_EQ(h16.inv_scale2.size(), static_cast<size_t>(m));
            std::uint64_t value_mismatch = 0, sub = 0;
            double worst_cs = 0.0;
            for (node_index j = 0; j < m; ++j) {
                const float s = scales[j];
                const double inv = 1.0f / s;
                EXPECT_EQ(h16.inv_scale2[j], inv * inv);
                ASSERT_EQ(LT.idx[LT.ptr[j]], static_cast<int>(j));
                const float d0 = static_cast<float>(LT.vals[LT.ptr[j]]) / s;   // fp32(L_jj) / s_j
                double resid = 0.0, col_x = 0.0, col_stored = 0.0;
                for (int k = LT.ptr[j]; k < LT.ptr[j + 1]; ++k) {
                    const float v = static_cast<float>(LT.vals[k]);
                    auto h = _Float16(v / s);
                    if (apxchol::is_stored_subnormal(h))
                        h = _Float16(std::copysign(0.0f, float(h)));
                    value_mismatch += h16.vals[k] != h || std::signbit(float(h16.vals[k])) != std::signbit(float(h));
                    if (k == LT.ptr[j]) continue;                                // diagonal slot
                    sub += apxchol::is_stored_subnormal(h16.vals[k]);
                    const double w = static_cast<double>(float(h16.vals[k]));
                    resid      += static_cast<double>(LT.vals[k]) / static_cast<double>(s) - w;
                    col_x      += static_cast<double>(LT.vals[k]) / static_cast<double>(s);
                    col_stored += w;
                }
                const float d_expect = static_cast<float>(static_cast<double>(d0) + resid);
                EXPECT_EQ(h16.diag[j], d_expect) << "diag " << j;
                // Stored column sum vs the fp32 column / s_j: exact (to fp32) --
                // the rounding compensation is unconditional.
                const double stored_sum = static_cast<double>(h16.diag[j]) + col_stored;
                const double x_sum      = static_cast<double>(d0) + col_x;
                const double abs_scale  = std::fabs(static_cast<double>(d0)) + std::fabs(col_x) + 1e-300;
                worst_cs = std::max(worst_cs, std::fabs(stored_sum - x_sum) / abs_scale);
            }
            EXPECT_EQ(value_mismatch, 0u);
            EXPECT_EQ(sub, 0u);                      // subnormals flushed
            EXPECT_EQ(h16.subnormal, 0u);
            EXPECT_LT(worst_cs, 4e-7);               // one fp32 rounding of the diagonal
        }
    }
}

// ── The dataflow backend's host side (gpu_factor_prep.h; CUDA-free) ────────────────
//
TEST(GpuHostPrep, DataflowRowStatsCountLongRowsAndTheEntriesTheyHold) {
    const std::vector<int> len = {1, 2, 255, 256, 257, 512, 1024, 100000};
    std::vector<int> ptr(len.size() + 1);
    std::partial_sum(len.begin(), len.end(), ptr.begin() + 1);
    const auto st = apxchol::detail::gpu_factor_prep::dataflow_row_stats(ptr);
    EXPECT_EQ(apxchol::detail::gpu_dataflow::chunk_size, 256);
    EXPECT_EQ(st.error, 0);
    std::int64_t total = 0; for (int l : len) total += l;
    EXPECT_EQ(st.split_threshold(total), 256);
    EXPECT_EQ(st.max_len, 100000);
    EXPECT_EQ(st.rows_ge[0], 5);                           // 256, 257, 512, 1024, 100000
    EXPECT_EQ(st.nnz_ge[0], 256 + 257 + 512 + 1024 + 100000);
    EXPECT_EQ(st.rows_ge[1], 3);                           // >= 512
    EXPECT_EQ(st.rows_ge[2], 2);                           // >= 1024
    EXPECT_EQ(st.rows_ge[3], 1);                           // >= 2048: only the 100000 row
    EXPECT_EQ(st.rows_ge[8], 1);                           // >= 65536
    EXPECT_EQ(st.rows_ge[9], 0);                           // >= 131072
    EXPECT_EQ(st.nnz_ge[8], 100000);
    // Empty input: no buckets, no division by zero.
    const auto e = apxchol::detail::gpu_factor_prep::dataflow_row_stats({});
    EXPECT_EQ(e.error, 0); EXPECT_EQ(e.max_len, 0); EXPECT_EQ(e.split_threshold(0), 768);
}

TEST(GpuHostPrep, DataflowSplitThresholdFollowsTheLongRowShare) {
    const int C = apxchol::detail::gpu_dataflow::chunk_size;
    auto threshold_for = [&](const std::vector<int>& len) {
        std::vector<int> ptr(len.size() + 1);
        std::partial_sum(len.begin(), len.end(), ptr.begin() + 1);
        return apxchol::detail::gpu_factor_prep::dataflow_row_stats(ptr).split_threshold(ptr.back());
    };
    // Hub-heavy: a handful of very long rows hold nearly all the entries.
    std::vector<int> hub(10000, 4);
    for (int i = 0; i < 20; ++i) hub[static_cast<size_t>(i)] = 20000;
    EXPECT_EQ(threshold_for(hub), C);
    // Shallow: the same few long rows, but the short ones dominate the nnz.
    std::vector<int> flat(10000000, 6);
    for (int i = 0; i < 20; ++i) flat[static_cast<size_t>(i)] = 2000;
    EXPECT_EQ(threshold_for(flat), 3 * C);
    // No long rows at all (grid / IPM factors): the conservative threshold,
    // which no row can reach -- so no row is split.
    EXPECT_EQ(threshold_for(std::vector<int>(1000, 20)), 3 * C);
    // Empty: no division by zero.
    EXPECT_EQ(threshold_for(std::vector<int>{}), 3 * C);
    // The rule is EXACTLY the share test, both sides of it.
    std::vector<int> len(1000, 1);
    len[0] = C;                                   // share = C / (C + 999)
    const double share = static_cast<double>(C) / static_cast<double>(C + 999);
    EXPECT_EQ(threshold_for(len), share >= 0.25 ? C : 3 * C);
}

TEST(GpuSptrsvAdoptionHost, CsrPairIsExactlyTheExistingUploadPair) {
    const node_index n = 4001, m = n - 1;
    const sparse_csc factor = make_wide_magnitude_lower(n);
    auto LT = apxchol::detail::gpu_factor_prep::build_L11_csc_int<factor_value_t>(factor, m);
    const auto scales = apxchol::detail::gpu_factor_prep::column_scales(LT);
    const auto stats = apxchol::detail::gpu_factor_prep::apply_factor_drop(
        LT, scales, 1e-4, /*fp16_storage=*/false);
    ASSERT_EQ(stats.nnz_stored, static_cast<std::uint64_t>(LT.nnz));
    const auto L = apxchol::detail::gpu_factor_prep::transpose_csr(LT);

    EXPECT_EQ(apxchol::detail::gpu_factor_prep::dataflow_factor_structure_check(
                  static_cast<int>(m), LT.nnz,
                  L.ptr.data(), L.idx.get(), LT.ptr.data(), LT.idx.get()),
              "");

    std::vector<int> short_L_ptr = L.ptr;
    --short_L_ptr.back();
    const std::string bad_extent =
        apxchol::detail::gpu_factor_prep::dataflow_factor_structure_check(
            static_cast<int>(m), LT.nnz,
            short_L_ptr.data(), L.idx.get(), LT.ptr.data(), LT.idx.get());
    EXPECT_NE(bad_extent.find("rowptr[m]"), std::string::npos) << bad_extent;

    // Corrupt every pointer/index position in either half, including integer
    // extremes and valid triangular entries that cease to be exact transposes.
    const std::array<std::vector<int>, 4> valid{{
        {0, 1, 3, 5}, {0, 0, 1, 0, 2}, {0, 3, 4, 5}, {0, 1, 2, 1, 2}}};
    const auto check = [](const auto& pair) {
        return apxchol::detail::gpu_factor_prep::dataflow_factor_structure_check(
            3, 5, pair[0].data(), pair[1].data(), pair[2].data(), pair[3].data());
    };
    ASSERT_TRUE(check(valid).empty());
    for (std::size_t array = 0; array < valid.size(); ++array)
        for (std::size_t slot = 0; slot < valid[array].size(); ++slot)
            for (int replacement : {-1, 0, 1, 2, 3, 4, 5, 6,
                                    std::numeric_limits<int>::min(),
                                    std::numeric_limits<int>::max()}) {
                if (replacement == valid[array][slot]) continue;
                auto malformed = valid;
                malformed[array][slot] = replacement;
                EXPECT_FALSE(check(malformed).empty())
                    << "array=" << array << " slot=" << slot << " value=" << replacement;
            }
}

#ifdef APXCHOL_USE_CUDA
// ── The GPU dataflow kernel against an independent CPU reference ────────────
//
// The dataflow kernel (cuda_dataflow.h) is the only GPU SpTRSV backend that is
// ours (the O(n)-schedule level-set kernel it replaced was removed 2026-08-20;
// these tests used to cross-check the two against each other). The independent
// implementation it is now checked against is a SERIAL, DOUBLE-precision host
// sweep over the very arrays the kernel reads -- so the agreement is
// tolerance-level, not bit-level: the kernel accumulates in fp32 (and widens
// fp16 storage to fp32), the reference in fp64.
//
// On a random 60k-row factor (Laplacian path, m = n-1, dense tail rows, the
// default drop) the forward and back sweeps agree with the reference to fp32
// accumulation error, the dataflow result is bit-identical across launches AND
// grid sizes (1 block, 7 blocks, the resident grid), for the fp32 and the fp16
// storage; and through cuda_sptrsv the L L^T pair agrees with the CPU
// omp_sptrsv pair, again bit-identical run to run.
#include "apxchol/solver/sptrsv/cuda.h"
#include "apxchol/solver/sptrsv/cuda_dataflow.h"
#include <cuda_runtime.h>

namespace {
template <class T>
using device_value_t = std::conditional_t<std::is_same_v<T, _Float16>, __half, T>;

template <class T> device_value_t<T>* dev_upload(const T* h, std::size_t count) {
    device_value_t<T>* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, count * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d, h, count * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    return d;
}
template <class T> std::vector<T> dev_download(const T* d, std::size_t count) {
    std::vector<T> h(count);
    EXPECT_EQ(cudaMemcpy(h.data(), d, count * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    return h;
}
// max |a-b| / max |a| and the count of differing entries; every entry finite.
std::pair<double, std::size_t> rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double maxabs = 0, maxdiff = 0; std::size_t nd = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_TRUE(std::isfinite(a[i]) && std::isfinite(b[i])) << "entry " << i;
        maxabs = std::max(maxabs, std::fabs(static_cast<double>(a[i])));
        maxdiff = std::max(maxdiff, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
        nd += a[i] != b[i];
    }
    return {maxdiff / (maxabs > 0 ? maxabs : 1.0), nd};
}

// THE independent reference for the dataflow sweeps: one row at a time, in
// sweep order, accumulating in DOUBLE -- no levels, no batches, no warps, no
// shared code with the kernel. `vals16` != nullptr selects the fp16 storage
// (diagonal from `diag`, rhs scaled by `in_scale` when non-null, exactly the
// gpu_factor_prep.h contract); otherwise the fp32 values are read inline and the
// diagonal is the row's own diagonal slot.
//   out[i] = (rhs[i] * (in_scale ? in_scale[i] : 1) - sum_{j != i} T[i,j] out[j]) / T[i,i]
std::vector<float> host_reference_sweep(int m, bool reverse,
                                        const int* rowptr, const int* colidx,
                                        const float* vals, const _Float16* vals16,
                                        const float* diag, const double* in_scale,
                                        const std::vector<float>& rhs) {
    std::vector<double> out(static_cast<std::size_t>(m), 0.0);
    for (int k = 0; k < m; ++k) {
        const int i = reverse ? m - 1 - k : k;
        double sum = 0.0, d = 0.0;
        for (int p = rowptr[i]; p < rowptr[i + 1]; ++p) {
            const int j = colidx[p];
            const double v = vals16 ? apxchol::widen(vals16[p])
                                    : static_cast<double>(vals[p]);
            if (j == i) { d = vals16 ? static_cast<double>(diag[i]) : v; continue; }
            sum += v * out[j];
        }
        const double r = static_cast<double>(rhs[static_cast<std::size_t>(i)]) * (in_scale ? in_scale[i] : 1.0);
        out[i] = (r - sum) / d;
    }
    std::vector<float> f(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i) f[static_cast<std::size_t>(i)] = static_cast<float>(out[i]);
    return f;
}
} // namespace

// A random factor whose forward AND back solves stay bounded in fp32 (every
// row of L and of L^T strictly diagonally dominant: diagonal in [1, 3],
// off-diagonals in [0, 0.01], hub columns of ~66 entries sum to < 0.7) --
// make_wide_magnitude_lower's 10^2 entries overflow a 60k-row back solve to
// inf, which would make the comparison vacuous. The hub columns are the
// long rows of the back sweep (lane groups of 16); the drop keeps nearly all.
static sparse_csc make_dominant_lower(node_index n) { return make_random_lower(n, 6.0, 515 + n, 0.0, 0.01); }

// ... and one with a REAL HUB TAIL, which the factor above does NOT have (its
// longest row is ~66 entries, i.e. ONE chunk: it exercises nothing of the row
// segmentation). Here a handful of dense COLUMNS give CSR(L^T) -- the back
// sweep -- rows of 5k-20k entries, and a dense ROW TAIL (the last vertices,
// which an elimination puts into everybody's column) gives CSR(L) -- the
// forward sweep -- the same. The hub magnitudes are ~1e-5, so even a 20k-
// entry row sums to ~0.2 against a diagonal in [1, 3]: BOTH triangular solves
// stay bounded in fp32, which the comparisons below need.
static sparse_csc make_hub_tail_lower(node_index n) {
    std::mt19937 rng(9001 + n);
    std::uniform_real_distribution<double> ubase(0.0, 0.01), uhub(0.0, 1e-5), udiag(1.0, 3.0);
    std::poisson_distribution<int> pcount(6);
    const node_index hub_len = 20000 < n / 3 ? 20000 : n / 3;
    std::vector<std::vector<node_index>> col_rows(n);
    std::vector<char> hub_col(n, 0), hub_row(n, 0);
    for (node_index j : {node_index(5), node_index(1000), node_index(2000), node_index(3000)})
        if (j < n) hub_col[j] = 1;
    for (int k = 1; k <= 8; ++k) if (n > static_cast<node_index>(k)) hub_row[n - k] = 1;
    for (node_index j = 0; j < n; ++j) {
        auto& rows = col_rows[j];
        rows.push_back(j);
        if (j + 1 < n) {
            std::uniform_int_distribution<node_index> urow(j + 1, n - 1);
            int k = pcount(rng);
            if (hub_col[j]) k += static_cast<int>(hub_len);
            for (int t = 0; t < k; ++t) rows.push_back(urow(rng));
        }
    }
    for (node_index r = 0; r < n; ++r) {                     // the dense row tail
        if (!hub_row[r]) continue;
        std::uniform_int_distribution<node_index> ucol(0, r > 0 ? r - 1 : 0);
        for (node_index t = 0; t < hub_len && r > 0; ++t) col_rows[ucol(rng)].push_back(r);
    }
    sparse_csc L;
    csc_access::n(L) = n;
    csc_access::outer(L).assign(static_cast<size_t>(n) + 1, 0);
    for (node_index j = 0; j < n; ++j) {
        auto& rows = col_rows[j];
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        csc_access::outer(L)[j + 1] = csc_access::outer(L)[j] + static_cast<edge_index>(rows.size());
    }
    csc_access::inner(L).resize(static_cast<size_t>(csc_access::outer(L)[n]));
    csc_access::vals(L).resize(static_cast<size_t>(csc_access::outer(L)[n]));
    for (node_index j = 0; j < n; ++j) {
        edge_index out = csc_access::outer(L)[j];
        for (node_index r : col_rows[j]) {
            csc_access::inner(L)[out] = r;
            csc_access::vals(L)[out] = static_cast<factor_value_t>(
                r == j ? udiag(rng) : -((hub_col[j] || hub_row[r]) ? uhub(rng) : ubase(rng)));
            ++out;
        }
    }
    return L;
}

// Gates of the tolerance-level comparisons below (see the block header): the
// per-sweep one against the serial fp64 reference on the same arrays, and the
// two pair gates against the CPU backend (fp16 storage rounds every
// off-diagonal to 11 bits, so its pair is a different preconditioner).
// Observed on an RTX 4090 Laptop: sweeps 6.0e-8 (fp32 fwd/bck), 6.5e-8 /
// 1.2e-7 (fp16); pair 1.1e-7 (fp32) and 1.4e-5 (fp16). The gates below sit
// ~10-70x above those, i.e. loose enough for another GPU's rounding and
// tight enough to catch a wrong answer.
static constexpr double kSweepTol    = 1e-6;
static constexpr double kPairTolFp32 = 1e-6;
static constexpr double kPairTolFp16 = 1e-3;

namespace {
namespace df = apxchol::detail::gpu_dataflow;
struct dev_plan {
    unsigned* bs = nullptr;
    df::segment* spec = nullptr;
    int nb = 0, slots = 0;
};
dev_plan upload_plan(const df::packed_plan& pl) {
    return {pl.words.empty() ? nullptr : dev_upload(pl.words.data(), pl.words.size()),
        pl.segments.empty() ? nullptr : dev_upload(pl.segments.data(), pl.segments.size()),
        int(pl.words.size() / df::lanes), int(pl.slots)};
}
void free_plan(dev_plan& d) { cudaFree(d.bs); cudaFree(d.spec); d = {}; }

// Everything one factor needs on the device: the two CSRs, both storages, and
// the host arrays the serial fp64 reference reads. Frees itself.
struct df_factor {
    int m = 0;
    apxchol::detail::gpu_factor_prep::csr_int<float>         LT, Lc;
    apxchol::detail::gpu_factor_prep::csr_int<_Float16> LT16, L16;
    apxchol::detail::gpu_factor_prep::fp16_scaled_arrays     h16;
    std::vector<double> inv_scale2;
    int *d_Lp = nullptr, *d_Li = nullptr, *d_Tp = nullptr, *d_Ti = nullptr;
    float *d_Lv = nullptr, *d_Tv = nullptr, *d_diag = nullptr;
    __half *d_Lv16 = nullptr, *d_Tv16 = nullptr;
    double* d_is2 = nullptr;

    df_factor(const sparse_csc& L, node_index mm) {
        m = static_cast<int>(mm);
        LT = apxchol::detail::gpu_factor_prep::build_L11_csc_int<apxchol::detail::sptrsv_value_t>(L, mm);
        const std::vector<float> scales = apxchol::detail::gpu_factor_prep::column_scales(LT);
        apxchol::detail::gpu_factor_prep::apply_factor_drop(LT, scales, 1e-4, false);
        Lc  = apxchol::detail::gpu_factor_prep::transpose_csr(LT);
        h16 = apxchol::detail::gpu_factor_prep::narrow_fp16_scaled(LT, scales);
        LT16.m = LT.m; LT16.nnz = LT.nnz; LT16.ptr = LT.ptr;
        LT16.idx = std::make_unique_for_overwrite<int[]>(static_cast<size_t>(LT.nnz));
        std::copy_n(LT.idx.get(), LT.nnz, LT16.idx.get());
        LT16.vals = std::make_unique_for_overwrite<_Float16[]>(static_cast<size_t>(LT.nnz));
        std::copy_n(h16.vals.get(), LT.nnz, LT16.vals.get());
        L16 = apxchol::detail::gpu_factor_prep::transpose_csr(LT16);
        inv_scale2 = h16.inv_scale2;
        d_Lp = dev_upload(Lc.ptr.data(), static_cast<size_t>(m) + 1);
        d_Li = dev_upload(Lc.idx.get(), static_cast<size_t>(Lc.nnz));
        d_Lv = dev_upload(Lc.vals.get(), static_cast<size_t>(Lc.nnz));
        d_Tp = dev_upload(LT.ptr.data(), static_cast<size_t>(m) + 1);
        d_Ti = dev_upload(LT.idx.get(), static_cast<size_t>(LT.nnz));
        d_Tv = dev_upload(LT.vals.get(), static_cast<size_t>(LT.nnz));
        d_Lv16 = dev_upload(L16.vals.get(), static_cast<size_t>(L16.nnz));
        d_Tv16 = dev_upload(LT16.vals.get(), static_cast<size_t>(LT16.nnz));
        d_diag = dev_upload(h16.diag.data(), static_cast<size_t>(m));
        d_is2  = dev_upload(inv_scale2.data(), static_cast<size_t>(m));
    }
    ~df_factor() {
        for (void* p : {(void*)d_Lp,(void*)d_Li,(void*)d_Lv,(void*)d_Tp,(void*)d_Ti,(void*)d_Tv,
                        (void*)d_Lv16,(void*)d_Tv16,(void*)d_diag,(void*)d_is2}) cudaFree(p);
    }
    df_factor(const df_factor&) = delete;
    df_factor& operator=(const df_factor&) = delete;

    df::packed_plan plan(bool reverse, unsigned threshold) const {
        const auto& A = reverse ? LT : Lc;
        return df::pack(A.ptr.data(), {unsigned(m), reverse}, std::array{0u, unsigned(m)},
                        threshold);
    }

};

// One sweep through the kernel with a given plan. `tag` must hold
// m + plan.slots words; every invocation initializes them.
void run_sweep(const df_factor& F, const dev_plan& dp, bool reverse, bool fp16,
               unsigned* tag, unsigned* ctrl, int grid,
               const float* d_rhs, float* d_out) {
    const void* values = fp16 ? static_cast<const void*>(reverse ? F.d_Tv16 : F.d_Lv16)
                              : static_cast<const void*>(reverse ? F.d_Tv : F.d_Lv);
    apxchol::detail::dataflow_solve(0, F.m, reverse,
        reverse ? F.d_Tp : F.d_Lp, reverse ? F.d_Ti : F.d_Li, values,
        fp16 ? F.d_diag : nullptr, fp16 && reverse ? F.d_is2 : nullptr,
        dp.bs, dp.spec, dp.nb, tag, F.m + dp.slots, ctrl, grid, d_rhs, d_out);
}
} // namespace

namespace {
struct test_device_allocations {
    std::vector<void*> pointers;

    ~test_device_allocations() {
        for (void* pointer : pointers) (void)cudaFree(pointer);
    }

    template<class T>
    device_value_t<T>* upload(const T* host, std::size_t count) {
        pointers.push_back(nullptr);
        device_value_t<T>* device = nullptr;
        apxchol::detail::check_cuda(
            cudaMalloc(&device, count * sizeof(T)), "test adoption cudaMalloc");
        pointers.back() = device;
        apxchol::detail::check_cuda(
            cudaMemcpy(device, host, count * sizeof(T), cudaMemcpyHostToDevice),
            "test adoption cudaMemcpy");
        return device;
    }

    void relinquish() noexcept { pointers.clear(); }
};

struct prepared_adoption {
    apxchol::detail::cuda_sptrsv_device_factor factor;
    const void* lifetime_probe = nullptr;
    std::int64_t nnz = 0;
    apxchol::detail::factor_drop_stats stats;
};

prepared_adoption prepare_adoption(
        const sparse_csc& source, node_index m, bool fp16,
        bool corrupt_L_extent = false) {
    auto LT = apxchol::detail::gpu_factor_prep::build_L11_csc_int<apxchol::detail::sptrsv_value_t>(
        source, m);
    std::vector<float> scales = apxchol::detail::gpu_factor_prep::column_scales(LT);
    if (fp16)
        for (node_index j = 0; j < m; ++j) {
            const float scale = scales[static_cast<std::size_t>(j)];
            const float diag = static_cast<float>(
                LT.vals[LT.ptr[static_cast<std::size_t>(j)]]);
            if (!std::isfinite(1.0f / scale) ||
                !std::isfinite(diag / scale))
                scales[static_cast<std::size_t>(j)] = 1.0f;
        }
    const auto stats = apxchol::detail::gpu_factor_prep::apply_factor_drop(
        LT, scales, 1e-4, fp16);
    test_device_allocations allocations;
    int current_device = -1;
    apxchol::detail::check_cuda(
        cudaGetDevice(&current_device), "test adoption cudaGetDevice");

    if (!fp16) {
        auto L = apxchol::detail::gpu_factor_prep::transpose_csr(LT);
        auto* d_L_ptr = allocations.upload(
            L.ptr.data(), static_cast<std::size_t>(m) + 1);
        auto* d_L_idx = allocations.upload(
            L.idx.get(), static_cast<std::size_t>(L.nnz));
        auto* d_L_val = allocations.upload(
            L.vals.get(), static_cast<std::size_t>(L.nnz));
        auto* d_LT_ptr = allocations.upload(
            LT.ptr.data(), static_cast<std::size_t>(m) + 1);
        auto* d_LT_idx = allocations.upload(
            LT.idx.get(), static_cast<std::size_t>(LT.nnz));
        auto* d_LT_val = allocations.upload(
            LT.vals.get(), static_cast<std::size_t>(LT.nnz));
        if (corrupt_L_extent) {
            const int bad = L.ptr.back() - 1;
            apxchol::detail::check_cuda(
                cudaMemcpy(d_L_ptr + m, &bad, sizeof(bad), cudaMemcpyHostToDevice),
                "test adoption corrupt rowptr");
        }
        auto owner = apxchol::detail::cuda_sptrsv_device_factor::own_fp32(
            current_device, static_cast<std::int64_t>(m), LT.nnz,
            {{d_L_ptr, (static_cast<std::size_t>(m) + 1) * sizeof(int)},
             {d_L_idx, static_cast<std::size_t>(L.nnz) * sizeof(int)},
             {d_L_val, static_cast<std::size_t>(L.nnz) * sizeof(apxchol::detail::sptrsv_value_t)}},
            {{d_LT_ptr, (static_cast<std::size_t>(m) + 1) * sizeof(int)},
             {d_LT_idx, static_cast<std::size_t>(LT.nnz) * sizeof(int)},
             {d_LT_val, static_cast<std::size_t>(LT.nnz) * sizeof(apxchol::detail::sptrsv_value_t)}},
            stats);
        allocations.relinquish();
        return {std::move(owner), d_L_ptr, LT.nnz, stats};
    }

    auto narrowed = apxchol::detail::gpu_factor_prep::narrow_fp16_scaled(LT, scales);
    const auto& inv_scale2 = narrowed.inv_scale2;
    apxchol::detail::gpu_factor_prep::csr_int<_Float16> LT16;
    LT16.m = LT.m;
    LT16.nnz = LT.nnz;
    LT16.ptr = LT.ptr;
    LT16.idx = std::move(LT.idx);
    LT16.vals = std::move(narrowed.vals);
    auto L16 = apxchol::detail::gpu_factor_prep::transpose_csr(LT16);
    auto* d_L_ptr = allocations.upload(
        L16.ptr.data(), static_cast<std::size_t>(m) + 1);
    auto* d_L_idx = allocations.upload(
        L16.idx.get(), static_cast<std::size_t>(L16.nnz));
    auto* d_L_val = allocations.upload(
        L16.vals.get(), static_cast<std::size_t>(L16.nnz));
    auto* d_LT_ptr = allocations.upload(
        LT16.ptr.data(), static_cast<std::size_t>(m) + 1);
    auto* d_LT_idx = allocations.upload(
        LT16.idx.get(), static_cast<std::size_t>(LT16.nnz));
    auto* d_LT_val = allocations.upload(
        LT16.vals.get(), static_cast<std::size_t>(LT16.nnz));
    auto* d_diag = allocations.upload(
        narrowed.diag.data(), static_cast<std::size_t>(m));
    auto* d_inv_scale2 = allocations.upload(
        inv_scale2.data(), static_cast<std::size_t>(m));
    if (corrupt_L_extent) {
        const int bad = L16.ptr.back() - 1;
        apxchol::detail::check_cuda(
            cudaMemcpy(d_L_ptr + m, &bad, sizeof(bad), cudaMemcpyHostToDevice),
            "test adoption corrupt rowptr");
    }
    auto owner = apxchol::detail::cuda_sptrsv_device_factor::own_fp16_scaled(
        current_device, static_cast<std::int64_t>(m), LT16.nnz,
        {{d_L_ptr, (static_cast<std::size_t>(m) + 1) * sizeof(int)},
         {d_L_idx, static_cast<std::size_t>(L16.nnz) * sizeof(int)},
         {d_L_val, static_cast<std::size_t>(L16.nnz) * sizeof(__half)}},
        {{d_LT_ptr, (static_cast<std::size_t>(m) + 1) * sizeof(int)},
         {d_LT_idx, static_cast<std::size_t>(LT16.nnz) * sizeof(int)},
         {d_LT_val, static_cast<std::size_t>(LT16.nnz) * sizeof(__half)}},
        {d_diag, static_cast<std::size_t>(m) * sizeof(float)},
        {d_inv_scale2, static_cast<std::size_t>(m) * sizeof(double)},
        stats,
        narrowed.flushed, narrowed.subnormal);
    allocations.relinquish();
    return {std::move(owner), d_L_ptr, LT16.nnz, stats};
}

bool live_device_allocation(const void* pointer) {
    cudaPointerAttributes attributes{};
    const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
    if (status != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
#if CUDART_VERSION >= 10000
    return attributes.type == cudaMemoryTypeDevice;
#else
    return attributes.memoryType == cudaMemoryTypeDevice;
#endif
}

void expect_drop_stats_equal(const apxchol::detail::factor_drop_stats& a,
                             const apxchol::detail::factor_drop_stats& b) {
    EXPECT_EQ(a.rel, b.rel);
    EXPECT_EQ(a.nnz_factor, b.nnz_factor);
    EXPECT_EQ(a.nnz_stored, b.nnz_stored);
    EXPECT_EQ(a.dropped, b.dropped);
    EXPECT_EQ(a.dropped_threshold, b.dropped_threshold);
    EXPECT_EQ(a.dropped_flush, b.dropped_flush);
}
} // namespace

// Every sweep, on both factors, against the SERIAL DOUBLE reference over the
// same arrays -- with segmentation off, at the shipped setting, and in the
// TORTURE setting (every row split into 256-entry pieces) -- plus
// bit-identity across launches and grid sizes (1 block = 8 warps against
// thousands of segment items is also the deadlock torture).
TEST(GpuDataflow, MatchesTheSerialDoubleReferenceBothDirectionsAndIsDeterministic) {
    const node_index n = 60001, m = n - 1;
    const int resident = apxchol::detail::dataflow_grid_size();
    ASSERT_GE(resident, 1);
    for (int kind = 0; kind < 2; ++kind) {
        SCOPED_TRACE(kind ? "hub-tail factor" : "no-hub factor");
        const sparse_csc L = kind ? make_hub_tail_lower(n) : make_dominant_lower(n);
        const df_factor F(L, m);
        std::vector<float> x(static_cast<size_t>(m));
        { std::mt19937 rng(9); std::uniform_real_distribution<float> u(-1.f, 1.f); for (auto& v : x) v = u(rng); }
        float* d_x = dev_upload(x.data(), static_cast<size_t>(m));
        float *d_a = nullptr, *d_b = nullptr, *d_y = nullptr;
        ASSERT_EQ(cudaMalloc(&d_a, static_cast<size_t>(m) * 4), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_b, static_cast<size_t>(m) * 4), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_y, static_cast<size_t>(m) * 4), cudaSuccess);
        unsigned* d_ctrl = nullptr;
        ASSERT_EQ(cudaMalloc(&d_ctrl, 2 * sizeof(int)), cudaSuccess);
        ASSERT_EQ(cudaMemset(d_ctrl, 0, 2 * sizeof(int)), cudaSuccess);
        for (unsigned threshold : {0u, 1u, 256u, 2048u}) {
            SCOPED_TRACE(threshold);
            const auto plf = F.plan(false, threshold), plb = F.plan(true, threshold);
            if (!threshold) {
                EXPECT_TRUE(plf.segments.empty() && plb.segments.empty());
            } else if (kind) {
                EXPECT_FALSE(plf.segments.empty());
            }
            dev_plan pf = upload_plan(plf);
            dev_plan pb = upload_plan(plb);
            // ONE tag array for both directions and every sweep below (the
            // fwd / bck / fwd sequence is the slot-aliasing test: the two
            // directions have different slot maps and reuse the same words,
            // reset between sweeps), plus a GUARD BAND: nothing may
            // write past m + n_slots.
            const size_t words = static_cast<size_t>(m) + static_cast<size_t>(std::max(pf.slots, pb.slots));
            const size_t kGuard = 64;
            unsigned* d_tag = nullptr;
            ASSERT_EQ(cudaMalloc(&d_tag, (words + kGuard) * sizeof(unsigned)), cudaSuccess);
            ASSERT_EQ(cudaMemset(d_tag, 0, words * sizeof(unsigned)), cudaSuccess);
            ASSERT_EQ(cudaMemset(d_tag + words, 0xAB, kGuard * sizeof(unsigned)), cudaSuccess);
            for (bool fp16 : {false, true}) {
                SCOPED_TRACE(fp16 ? "fp16 storage" : "fp32 storage");
                // ── forward ──
                const std::vector<float> y_ref = host_reference_sweep(
                    F.m, false, F.Lc.ptr.data(), F.Lc.idx.get(), F.Lc.vals.get(),
                    fp16 ? F.L16.vals.get() : nullptr, F.h16.diag.data(), nullptr, x);
                run_sweep(F, pf, false, fp16, d_tag, d_ctrl, resident, d_x, d_a);
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                const std::vector<float> y_df = dev_download(d_a, static_cast<size_t>(m));
                { const auto [rd, nd] = rel_diff(y_ref, y_df);
                  EXPECT_LT(rd, kSweepTol) << "forward: dataflow vs serial double reference";
                  EXPECT_GT(nd, 0u);
                  std::fprintf(stderr, "[GpuDataflow] %s %s %s forward: rel diff %.3e\n",
                               kind ? "hub" : "plain", std::to_string(threshold).c_str(), fp16 ? "fp16" : "fp32", rd); }
                for (int grid : {resident, 1, 7}) {
                    run_sweep(F, pf, false, fp16, d_tag, d_ctrl, grid, d_x, d_b);
                    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                    const auto [rd, nd] = rel_diff(y_df, dev_download(d_b, static_cast<size_t>(m)));
                    EXPECT_EQ(nd, 0u) << "forward not deterministic at grid " << grid << " (rel diff " << rd << ")";
                }
                // ── back, from the kernel's own forward output ──
                ASSERT_EQ(cudaMemcpy(d_y, y_df.data(), static_cast<size_t>(m) * 4, cudaMemcpyHostToDevice), cudaSuccess);
                const std::vector<float> z_ref = host_reference_sweep(
                    F.m, true, F.LT.ptr.data(), F.LT.idx.get(), F.LT.vals.get(),
                    fp16 ? F.LT16.vals.get() : nullptr, F.h16.diag.data(),
                    fp16 ? F.inv_scale2.data() : nullptr, y_df);
                run_sweep(F, pb, true, fp16, d_tag, d_ctrl, resident, d_y, d_a);
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                const std::vector<float> z_df = dev_download(d_a, static_cast<size_t>(m));
                { const auto [rd, nd] = rel_diff(z_ref, z_df);
                  EXPECT_LT(rd, kSweepTol) << "back: dataflow vs serial double reference";
                  EXPECT_GT(nd, 0u);
                  std::fprintf(stderr, "[GpuDataflow] %s %s %s back: rel diff %.3e\n",
                               kind ? "hub" : "plain", std::to_string(threshold).c_str(), fp16 ? "fp16" : "fp32", rd); }
                for (int grid : {resident, 1, 7}) {
                    run_sweep(F, pb, true, fp16, d_tag, d_ctrl, grid, d_y, d_b);
                    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                    const auto [rd, nd] = rel_diff(z_df, dev_download(d_b, static_cast<size_t>(m)));
                    EXPECT_EQ(nd, 0u) << "back not deterministic at grid " << grid << " (rel diff " << rd << ")";
                }
                const auto ctrl = dev_download(d_ctrl, 2);
                EXPECT_GT(ctrl[0], 0u); EXPECT_EQ(ctrl[1], 0u);
                const auto guard = dev_download(d_tag + words, kGuard);
                for (size_t g = 0; g < kGuard; ++g)
                    ASSERT_EQ(guard[g], 0xABABABABu) << "a sweep wrote past m + n_slots, word " << g;
            }
            cudaFree(d_tag);
            free_plan(pf); free_plan(pb);
        }
        for (void* p : {(void*)d_x, (void*)d_a, (void*)d_b, (void*)d_y, (void*)d_ctrl}) cudaFree(p);
    }
}

TEST(GpuDataflow, ScalarAndPanelsMatchCpuReferenceAndAreDeterministic) {
    const node_index n = 60001, m = n - 1;
    for (int kind = 0; kind < 2; ++kind) {
        SCOPED_TRACE(kind ? "hub-tail factor" : "no-hub factor");
        const sparse_csc L = kind ? make_hub_tail_lower(n) : make_dominant_lower(n);
        std::vector<double> x(m), y_cpu(m), y_df(m), y_df2(m);
        { std::mt19937 rng(11); std::uniform_real_distribution<double> u(-1.0, 1.0); for (auto& v : x) v = u(rng); }
        // The CPU reference pair (fp32 storage, fp64 accumulation) -- one
        // setup, every GPU variant is compared against it. The storage is
        // explicit, and the tolerances are stated against the fp32 pair.
        {
            apxchol::detail::omp_sptrsv c; c.setup(L, m);
            std::vector<double> t(m);
            c.forward_solve(x.data(), t.data());
            c.transpose_solve(t.data(), y_cpu.data());
        }
        for (bool fp16 : {false, true}) {
            SCOPED_TRACE(fp16 ? "fp16" : "fp32");
            const apxchol::factor_options options{.factor_storage = fp16
                ? apxchol::factor_storage_type::fp16 : apxchol::factor_storage_type::fp32};
            apxchol::detail::cuda_sptrsv t; t.setup(L, m, options);
            ASSERT_STREQ(t.backend_name(), "dataflow");
            EXPECT_EQ(t.fp16(), fp16);
            EXPECT_GE(t.dataflow_grid(), 1);
            t.solve_LLt(x.data(), y_df.data());
            t.solve_LLt(x.data(), y_df2.data());
            double maxabs = 0, maxdiff = 0; std::size_t nd = 0, nd2 = 0;
            for (node_index i = 0; i < m; ++i) {
                ASSERT_TRUE(std::isfinite(y_cpu[i]) && std::isfinite(y_df[i])) << i;
                maxabs = std::max(maxabs, std::fabs(y_cpu[i]));
                maxdiff = std::max(maxdiff, std::fabs(y_cpu[i] - y_df[i]));
                nd  += y_cpu[i] != y_df[i];
                nd2 += y_df[i] != y_df2[i];
            }
            EXPECT_LT(maxdiff / maxabs, fp16 ? kPairTolFp16 : kPairTolFp32)
                << "L L^T pair: GPU dataflow vs CPU omp_sptrsv";
            EXPECT_GT(nd, 0u);
            EXPECT_EQ(nd2, 0u) << "dataflow pair not bit-identical run to run";
            for (unsigned columns : {2, 4}) {
                SCOPED_TRACE(columns);
                std::vector<double> panel(m * columns), sentinel(m * columns, 123.0);
                for (node_index row = 0; row < m; ++row)
                    for (unsigned q = 0; q < columns; ++q)
                        panel[row * columns + q] = std::ldexp(x[row], q);
                test_device_allocations memory;
                auto* rhs = memory.upload(panel.data(), panel.size());
                auto* output = memory.upload(sentinel.data(), sentinel.size());
                for (bool partial : {false, true}) {
                    SCOPED_TRACE(partial);
                    const unsigned active[] = {1, 0, unsigned(!partial), 0, 1, 0, unsigned(!partial), 0};
                    auto* mask = memory.upload(active, std::size(active));
                    ASSERT_EQ(cudaMemcpy(output, sentinel.data(), sentinel.size() * sizeof(double),
                                         cudaMemcpyHostToDevice), cudaSuccess);
                    auto solve = [&] {
                        if (columns == 2) t.solve_LLt_dev<2>(rhs, output, mask, 2);
                        else t.solve_LLt_dev<4>(rhs, output, mask, 2);
                    };
                    solve();
                    const auto actual = dev_download(output, panel.size());
                    for (unsigned q = 0; q < columns; ++q) {
                        double error = 0;
                        for (node_index row = 0; row < m; ++row) {
                            const auto k = row * columns + q;
                            ASSERT_TRUE(std::isfinite(actual[k]));
                            if (!active[2 * q]) {
                                EXPECT_EQ(actual[k], sentinel[k]);
                            } else {
                                error = std::max(error, std::abs(actual[k] - std::ldexp(y_cpu[row], q)));
                            }
                        }
                        if (active[2 * q]) {
                            EXPECT_LT(error / std::ldexp(maxabs, q), fp16 ? kPairTolFp16 : kPairTolFp32);
                        }
                    }
                    solve();
                    EXPECT_EQ(actual, dev_download(output, panel.size()));
                }
            }
            std::fprintf(stderr, "[GpuDataflow] %s %s pair: max rel diff vs the CPU pair %.3e\n",
                         kind ? "hub" : "plain", fp16 ? "fp16" : "fp32", maxdiff / maxabs);
        }
    }
}

TEST(GpuSptrsvAdoptionDevice, Fp32AndFp16MatchTheExistingUploadPathBitForBit) {
    const node_index n = 4097, m = n - 1;
    const sparse_csc factor = make_dominant_lower(n);
    std::vector<double> input(static_cast<std::size_t>(m));
    {
        std::mt19937 rng(404);
        std::uniform_real_distribution<double> value(-1.0, 1.0);
        for (double& item : input) item = value(rng);
    }

    for (bool fp16 : {false, true}) {
        SCOPED_TRACE(fp16 ? "fp16" : "fp32");
        const apxchol::factor_options options{.factor_storage = fp16
            ? apxchol::factor_storage_type::fp16 : apxchol::factor_storage_type::fp32};
        apxchol::detail::cuda_sptrsv uploaded;
        uploaded.setup(factor, m, options);
        ASSERT_FALSE(uploaded.adopted_device_factor());
        ASSERT_EQ(uploaded.adoption_host_download_bytes(), 0u);

        prepared_adoption prepared = prepare_adoption(factor, m, fp16);
        const void* lifetime_probe = prepared.lifetime_probe;
        ASSERT_TRUE(live_device_allocation(lifetime_probe));
        std::vector<double> uploaded_result(static_cast<std::size_t>(m));
        std::vector<double> adopted_result(static_cast<std::size_t>(m));
        std::vector<double> adopted_repeat(static_cast<std::size_t>(m));
        {
            apxchol::detail::cuda_sptrsv adopted;
            adopted.setup_adopting_device_factor(
                std::move(prepared.factor), options);
            EXPECT_TRUE(prepared.factor.empty());
            ASSERT_TRUE(adopted.ready());
            EXPECT_TRUE(adopted.adopted_device_factor());
            EXPECT_STREQ(adopted.backend_name(), "dataflow");
            EXPECT_EQ(adopted.fp16(), fp16);
            EXPECT_EQ(adopted.stored_nnz(), uploaded.stored_nnz());
            expect_drop_stats_equal(adopted.drop_stats(), uploaded.drop_stats());
            const std::size_t expected_download =
                2 * (static_cast<std::size_t>(m) + 1) * sizeof(int) +
                2 * static_cast<std::size_t>(prepared.nnz) * sizeof(int) +
                (fp16 ? static_cast<std::size_t>(m) *
                            (sizeof(float) + sizeof(double))
                      : 0u);
            EXPECT_EQ(adopted.adoption_host_download_bytes(), expected_download);

            uploaded.solve_LLt(input.data(), uploaded_result.data());
            adopted.solve_LLt(input.data(), adopted_result.data());
            adopted.solve_LLt(input.data(), adopted_repeat.data());
            EXPECT_EQ(0, std::memcmp(
                uploaded_result.data(), adopted_result.data(),
                adopted_result.size() * sizeof(double)));
            EXPECT_EQ(0, std::memcmp(
                adopted_result.data(), adopted_repeat.data(),
                adopted_result.size() * sizeof(double)));
        }
        EXPECT_FALSE(live_device_allocation(lifetime_probe))
            << "cuda_sptrsv did not release an adopted factor allocation";
    }
}

TEST(GpuSptrsvAdoptionDevice, InvalidStructureRollsBackOwnershipAndDestination) {
    const node_index n = 1025, m = n - 1;
    const sparse_csc factor = make_dominant_lower(n);
    const apxchol::factor_options options{.factor_storage = apxchol::factor_storage_type::fp32};

    apxchol::detail::cuda_sptrsv destination;
    prepared_adoption malformed =
        prepare_adoption(factor, m, /*fp16=*/false,
                         /*corrupt_L_extent=*/true);
    const void* failed_probe = malformed.lifetime_probe;
    ASSERT_TRUE(live_device_allocation(failed_probe));
    try {
        destination.setup_adopting_device_factor(
            std::move(malformed.factor), options);
        FAIL() << "malformed adopted structure was accepted";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find("rowptr[m]"),
                  std::string::npos) << error.what();
    }
    EXPECT_TRUE(malformed.factor.empty());
    EXPECT_FALSE(destination.ready());
    EXPECT_FALSE(destination.adopted_device_factor());
    EXPECT_EQ(destination.adoption_host_download_bytes(), 0u);
    EXPECT_FALSE(live_device_allocation(failed_probe))
        << "failed adoption did not release transferred ownership";

    // The failed transaction leaves no latent partial setup: the same object
    // can immediately accept a valid owner and solve.
    prepared_adoption valid = prepare_adoption(factor, m, /*fp16=*/false);
    const void* success_probe = valid.lifetime_probe;
    destination.setup_adopting_device_factor(
        std::move(valid.factor), options);
    EXPECT_TRUE(destination.ready());
    EXPECT_TRUE(destination.adopted_device_factor());
    EXPECT_TRUE(live_device_allocation(success_probe));
}

namespace {
df::packed_plan download_device_plan(const apxchol::detail::dataflow_device_plan& device,
        test_device_allocations& memory, unsigned lanes = df::lanes) {
    memory.pointers.insert(memory.pointers.end(), {device.words, device.segments});
    df::packed_plan host;
    if (device.n_batches) host.words = dev_download(device.words, std::size_t(device.n_batches) * lanes);
    const auto segments = std::size_t(device.n_slots) + device.n_split;
    if (segments) host.segments = dev_download(device.segments, segments);
    host.slots = device.n_slots;
    return host;
}
}

TEST(GpuDataflowDevicePlan, HostAndDevicePackTheSameTasks) {
    const int choices[] = {1, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257, 769, 2050};
    for (unsigned m : {0, 1, 31, 255, 256, 257, 513, 4097}) {
        for (bool reverse : {false, true}) {
            for (int pattern = 0; pattern < 3; ++pattern) {
                std::vector<int> len(m), ptr(m + 1, 0);
                for (unsigned q = 0; q < m; ++q) {
                    const unsigned requested = pattern == 0 ? 1 : pattern == 1
                        ? choices[(q * 73) % std::size(choices)] : (q % 31 == 0 ? q + 1 : 9);
                    len[reverse ? m - 1 - q : q] = std::min(q + 1, requested);
                }
                for (unsigned row = 0; row < m; ++row) ptr[row + 1] = ptr[row] + len[row];
                const auto stats = apxchol::detail::gpu_factor_prep::dataflow_row_stats(ptr);
                std::vector<unsigned> blocks{0};
                for (unsigned q = 0; q < m; q += 256) blocks.push_back(std::min(m, q + 256));
                for (unsigned lanes : {8, 32}) {
                    for (int threshold : {-1, 0, 1, 256, 768}) {
                        SCOPED_TRACE(::testing::Message() << "m=" << m << " reverse=" << reverse
                            << " pattern=" << pattern << " threshold=" << threshold << " lanes=" << lanes);
                        test_device_allocations memory;
                        auto* d_ptr = memory.upload(ptr.data(), ptr.size());
                        const auto device = apxchol::detail::build_dataflow_device_plan(
                            m, reverse, ptr.back(), d_ptr, threshold, nullptr, nullptr, lanes);
                        const auto plan = download_device_plan(device, memory, lanes);
                        const unsigned split = threshold < 0 ? stats.split_threshold(ptr.back()) : threshold;
                        const auto expected = lanes == 8
                            ? df::pack<8>(ptr.data(), {m, reverse}, blocks, split)
                            : df::pack<32>(ptr.data(), {m, reverse}, blocks, split);
                        EXPECT_EQ(plan.words, expected.words);
                        EXPECT_EQ(plan.slots, expected.slots);
                        ASSERT_EQ(plan.segments.size(), expected.segments.size());
                        if (!plan.segments.empty()) {
                            EXPECT_EQ(std::memcmp(plan.segments.data(), expected.segments.data(),
                                plan.segments.size() * sizeof(df::segment)), 0);
                        }
                        EXPECT_EQ(device.stats.max_len, stats.max_len);
                        EXPECT_LE(device.host_download_bytes, 256u);
                        for (int k = 0; k < apxchol::detail::gpu_dataflow::row_stats::kBuckets; ++k) {
                            EXPECT_EQ(device.stats.rows_ge[k], stats.rows_ge[k]);
                            EXPECT_EQ(device.stats.nnz_ge[k], stats.nnz_ge[k]);
                        }
                    }
                }
            }
        }
    }
}

TEST(GpuDataflowDevicePlan, RejectsInvalidPointersAndScalingMetadata) {
    for (const auto& ptr : std::vector<std::vector<int>>{
            {1, 2, 3, 4, 5}, {0, 1, 1, 3, 4}, {0, 3, 2, 3, 4}, {0, 1, 2, 3, 6}}) {
        test_device_allocations memory;
        auto* device = memory.upload(ptr.data(), ptr.size());
        EXPECT_THROW(apxchol::detail::build_dataflow_device_plan(4, false, 4, device, -1), std::invalid_argument);
    }
    for (int mode = 0; mode < 4; ++mode) {
        const int ptr[] = {0, 1, 2, 3, 4};
        float diag[] = {1, 1, 1, 1};
        double scales[] = {1, 1, 1, 1};
        if (mode == 0) diag[2] = 0;
        if (mode == 1) diag[2] = std::numeric_limits<float>::infinity();
        if (mode == 2) scales[1] = std::numeric_limits<double>::quiet_NaN();
        test_device_allocations memory;
        auto* d_ptr = memory.upload(ptr, std::size(ptr));
        auto* d_diag = memory.upload(diag, std::size(diag));
        auto* d_scale = memory.upload(scales, std::size(scales));
        EXPECT_THROW(apxchol::detail::build_dataflow_device_plan(
            4, false, 4, d_ptr, -1, d_diag, mode == 3 ? nullptr : d_scale), std::invalid_argument);
    }
}

TEST(GpuDataflow, GraphReplayResetsDependenciesForChangedRightHandSides) {
    const df_factor factor(make_dominant_lower(258), 257);
    for (bool fp16 : {false, true}) {
        const auto host = factor.plan(false, 256);
        auto plan = upload_plan(host);
        test_device_allocations memory;
        memory.pointers.insert(memory.pointers.end(), {plan.bs, plan.spec});
        std::vector<unsigned> initial(factor.m + plan.slots, 0xdeadbeefu);
        auto* tags = memory.upload(initial.data(), initial.size());
        const unsigned dirty_ticket = 123;
        auto* ticket = memory.upload(&dirty_ticket, 1);
        std::vector<float> rhs(factor.m, 1);
        auto* input = memory.upload(rhs.data(), rhs.size());
        auto* output = memory.upload(rhs.data(), rhs.size());
        cudaStream_t stream;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
        const int grid = apxchol::detail::dataflow_grid_size();
        ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), cudaSuccess);
        const void* values = fp16 ? static_cast<const void*>(factor.d_Lv16)
                                  : static_cast<const void*>(factor.d_Lv);
        apxchol::detail::dataflow_solve(stream, factor.m, false, factor.d_Lp, factor.d_Li,
            values, fp16 ? factor.d_diag : nullptr, nullptr, plan.bs, plan.spec, plan.nb,
            tags, initial.size(), ticket, grid, input, output);
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
        ASSERT_EQ(cudaGraphInstantiate(&executable, graph, 0), cudaSuccess);
        for (float scale : {1.f, -0.75f, 2.f}) {
            for (int row = 0; row < factor.m; ++row) rhs[row] = scale * std::sin(row + .25f);
            ASSERT_EQ(cudaMemcpyAsync(input, rhs.data(), rhs.size() * sizeof(float),
                cudaMemcpyHostToDevice, stream), cudaSuccess);
            ASSERT_EQ(cudaGraphLaunch(executable, stream), cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            const auto expected = host_reference_sweep(factor.m, false, factor.Lc.ptr.data(),
                factor.Lc.idx.get(), factor.Lc.vals.get(), fp16 ? factor.L16.vals.get() : nullptr,
                factor.h16.diag.data(), nullptr, rhs);
            EXPECT_LT(rel_diff(expected, dev_download(output, factor.m)).first, kSweepTol);
        }
        rhs[0] = std::numeric_limits<float>::quiet_NaN();
        ASSERT_EQ(cudaMemcpyAsync(input, rhs.data(), rhs.size() * sizeof(float),
            cudaMemcpyHostToDevice, stream), cudaSuccess);
        ASSERT_EQ(cudaGraphLaunch(executable, stream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        const auto nonfinite = dev_download(output, factor.m);
        EXPECT_TRUE(std::isnan(nonfinite[0]));
        EXPECT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
        EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
        EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }
}

TEST(GpuDataflowDevicePlan, TilePackingPreservesFp32AndFp16SweepsBitForBit) {
    const node_index n = 8194, m = n - 1;
    for (bool hub : {false, true}) {
        const auto factor = hub ? make_hub_tail_lower(n) : make_dominant_lower(n);
        const df_factor F(factor, m);
        std::vector<float> input(m);
        for (node_index row = 0; row < m; ++row) input[row] = std::sin(row + 0.123);
        for (bool reverse : {false, true}) {
            const auto& ptr = reverse ? F.LT.ptr : F.Lc.ptr;
            for (int threshold : {-1, 0, 1, 256}) {
                const auto stats = apxchol::detail::gpu_factor_prep::dataflow_row_stats(ptr);
                const auto reference = F.plan(reverse,
                    threshold < 0 ? stats.split_threshold(ptr.back()) : threshold);
                test_device_allocations memory;
                const auto device = apxchol::detail::build_dataflow_device_plan(
                    m, reverse, ptr.back(), reverse ? F.d_Tp : F.d_Lp, threshold,
                    F.d_diag, F.d_is2);
                const auto plan = download_device_plan(device, memory);
                auto host_device = upload_plan(reference);
                memory.pointers.insert(memory.pointers.end(), {host_device.bs, host_device.spec});
                const dev_plan tile_device{device.words, device.segments,
                    device.n_batches, device.n_slots};
                const auto words = std::size_t(m) + std::max(plan.slots, reference.slots);
                std::vector<unsigned> zero_tags(words, 0);
                const unsigned zero_ctrl[] = {0, 0};
                auto* tags = memory.upload(zero_tags.data(), words);
                auto* ctrl = memory.upload(zero_ctrl, 2);
                auto* rhs = memory.upload(input.data(), input.size());
                auto* answer = memory.upload(input.data(), input.size());
                    for (bool fp16 : {false, true}) {
                    const int resident = apxchol::detail::dataflow_grid_size();
                    run_sweep(F, host_device, reverse, fp16, tags, ctrl, resident, rhs, answer);
                    const auto expected = dev_download(answer, m);
                    for (int grid : {1, 7, resident}) {
                        SCOPED_TRACE(::testing::Message() << "hub=" << hub << " reverse=" << reverse
                            << " threshold=" << threshold << " fp16=" << fp16 << " grid=" << grid);
                        run_sweep(F, tile_device, reverse, fp16, tags, ctrl, grid, rhs, answer);
                        const auto actual = dev_download(answer, m);
                        ASSERT_EQ(std::memcmp(expected.data(), actual.data(), m * sizeof(float)), 0);
                        for (float value : actual) ASSERT_TRUE(std::isfinite(value));
                    }
                }
            }
        }
    }
}
#endif // APXCHOL_USE_CUDA
