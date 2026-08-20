// The compacting factor drop of omp_sptrsv::setup (APXCHOL_FACTOR_DROP=<rel>,
// default kFactorDropRelDefault = 1e-4; include/apxchol/solver/sptrsv/omp.h):
//   * stored nnz == the number of kept entries (the predicate |v| >= rel * s_j
//     restated here independently of omp_sptrsv::keep_offdiag), the diagonal
//     is always kept (first in every CSC column, last in every CSR row), no
//     stored zeros survive, the statistics say what went;
//   * the column-sum compensation (default): every column of the compacted
//     factor sums to what the original column summed to, and the forward AND
//     back solves through the compacted arrays agree with a drop-OFF
//     omp_sptrsv on the reference factor with the dropped entries ZEROED in
//     place and the same compensation applied (restated independently);
//     (the column-sum compensation is unconditional; the reference is the
//     zeroed factor (a stored zero and an absent entry solve identically);
//   * the CSR/CSC the drop produces are byte-identical across thread counts;
//   * the env contract: unset = default ON at 1e-4, "0" / negative / junk =
//     off, any positive value overrides (rel > 1 keeps only the diagonal).
// Under APXCHOL_SPTRSV_LOWPREC=FP16_SCALED (sptrsv_value_t narrower than the
// factor's factor_value_t; it stores L_ij / s_j with s_j the PRE-drop column
// max) the drop and the compensation run on the fp32 factor
// before the narrowing, so the value contract becomes "stored ==
// narrow_value(compensated reference value, s_j)" bit-for-bit, and the
// column-sum / solve comparisons hold to the storage format's rounding; the
// storage-format-specific drop clause (format_flushes) has its own tests in
// test_lowprec.cpp.
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <utility>
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

// RAII env override: sets one variable for one scope, restores after (unset
// if it was unset), so the tests never leak state into each other.
struct scoped_env {
    std::string name, saved; bool had = false;
    scoped_env(const char* var, const char* value) : name(var) {
        if (const char* e = std::getenv(var)) { had = true; saved = e; }
        if (value) setenv(var, value, 1);
        else       unsetenv(var);
    }
    ~scoped_env() {
        if (had) setenv(name.c_str(), saved.c_str(), 1);
        else     unsetenv(name.c_str());
    }
};
struct scoped_drop_env : scoped_env {
    explicit scoped_drop_env(const char* value) : scoped_env("APXCHOL_FACTOR_DROP", value) {}
};

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
    L.n_ = m;
    L.outer_.assign(static_cast<size_t>(m) + 1, 0);
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
        L.outer_[j + 1] = L.outer_[j] + static_cast<edge_index>(col_rows[j].size());
    L.inner_.resize(static_cast<size_t>(L.outer_[m]));
    L.vals_.resize(static_cast<size_t>(L.outer_[m]));
    for (node_index j = 0; j < m; ++j) {
        edge_index out = L.outer_[j];
        for (node_index r : col_rows[j]) {
            L.inner_[out] = r;
            L.vals_[out]  = static_cast<factor_value_t>(r == j ? udiag(rng) : -uval(rng));
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
        for (edge_index p = L.outer_[j] + 1; p < L.outer_[j + 1]; ++p)
            if (L.inner_[p] < m) mx = std::max(mx, std::fabs(static_cast<double>(L.vals_[p])));
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
        for (edge_index p = L.outer_[j]; p < L.outer_[j + 1]; ++p) {
            if (L.inner_[p] >= m) continue;                    // outside L11 (Laplacian last row)
            if (L.inner_[p] == j) { ++d.kept; continue; }
            const double v = static_cast<double>(L.vals_[p]);
            const bool keep = std::fabs(v) >= rel * s[j];
            if (keep) { ++d.kept; kept_abs += std::fabs(v); }
            else      { ++d.zeroed; dropped_sum += v; d.Lz.vals_[p] = 0; }
        }
        // (Same double arithmetic order as setup -- per_abs first -- so the
        // compacted values are bit-identical to these, which the 1e-12 solve
        // comparison below relies on.)
        if (dropped_sum != 0.0 && kept_abs > 0.0) {
            const double per_abs = dropped_sum / kept_abs;
            for (edge_index p = L.outer_[j]; p < L.outer_[j + 1]; ++p) {
                if (L.inner_[p] >= m || L.inner_[p] == j || d.Lz.vals_[p] == 0) continue;
                const double v = static_cast<double>(d.Lz.vals_[p]);
                d.Lz.vals_[p] = static_cast<factor_value_t>(v + per_abs * std::fabs(v));
            }
        }
    }
    return d;
}

// Column sums (diagonal + off-diagonals) of L11 = L.topLeftCorner(m, m).
std::vector<double> column_sums_L11(const sparse_csc& L, node_index m) {
    std::vector<double> cs(m, 0.0);
    for (node_index j = 0; j < m; ++j)
        for (edge_index p = L.outer_[j]; p < L.outer_[j + 1]; ++p)
            if (L.inner_[p] < m) cs[j] += static_cast<double>(L.vals_[p]);
    return cs;
}

// Relative rounding of ONE stored off-diagonal in this build's storage
// format (what the column-sum / solve comparisons below are held to): 0 on
// the fp32 / fp64 builds (the factor IS the stored value), 2^-11 fp16.
constexpr double kStorageEps =
    0.0;
// The pre-drop per-column scale FP16_SCALED divides by (1.0 off it).
double pair_scale(double s_pre) {
    (void)s_pre;
    return 1.0;
}
// The EFFECTIVE factor value a stored off-diagonal stands for: widened, times
// the column scale under *_SCALED (the kernels run on L~ = L D^-1).
double effective(sptrsv_value_t w, double s_pre) { return apxchol::widen(w) * pair_scale(s_pre); }

std::uint64_t L11_nnz_of(const sparse_csc& L, node_index m) {
    std::uint64_t n = 0;
    for (node_index j = 0; j < m; ++j)
        for (edge_index p = L.outer_[j]; p < L.outer_[j + 1]; ++p) n += L.inner_[p] < m;
    return n;
}

// Magnitudes spread over ~9 decades: v = -10^u, u uniform in [-7, 2), so
// about half of every column sits below 1e-4 * s_j; a few explicit zeros.
sparse_csc make_wide_magnitude_lower(node_index n) {
    sparse_csc L = make_random_lower(n, 6.0, 4242 + n, 0.0, 1.0);
    std::mt19937 rng(77 + n);
    std::uniform_real_distribution<double> uexp(-7.0, 2.0);
    for (node_index j = 0; j < n; ++j)
        for (edge_index p = L.outer_[j] + 1; p < L.outer_[j + 1]; ++p) {
            L.vals_[p] = static_cast<factor_value_t>(-std::pow(10.0, uexp(rng)));
            if ((p % 101) == 5) L.vals_[p] = 0;   // explicit zeros: always dropped
        }
    return L;
}

} // namespace

// APXCHOL_FACTOR_DROP=<rel>: off-diagonals with |v| < rel * s_j are REMOVED
// before the CSR/CSC are built: stored nnz == the number of kept entries, the
// diagonal is always kept (first in every CSC column, last in every CSR row),
// the statistics say what was dropped, the column sums are preserved (default
// compensation) and the forward AND back solves through the compacted arrays
// agree with the reference dense drop (entries zeroed in place, same
// compensation, drop OFF) to ~1e-12. Both compensation modes, both the SDDM
// alias path (m == n) and the Laplacian copy path (m == n-1), serial and
// parallel transpose (m > 50000).
TEST(SpTRSVDrop, CompactsToKeptEntriesAndSolvesLikeTheZeroedReference) {
    // These state the DEFAULT (fp32) storage contract, so pin it: the suite
    // is also run with APXCHOL_SPTRSV_FP16=1 in the environment.
    scoped_env fp32_storage("APXCHOL_SPTRSV_FP16", "0");

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

        // Compacted (explicit env for this setup only).
        apxchol::omp_sptrsv trsv;
        { scoped_drop_env env("1e-4"); trsv.setup(L, m); }
        // Reference: dense drop, drop OFF.
        apxchol::omp_sptrsv rf;
        { scoped_drop_env env("0"); rf.setup(ref.Lz, m); }

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
                double sum = apxchol::omp_sptrsv::stored_diag(L.vals_[L.outer_[j]], static_cast<float>(pair_scale(s[j])))
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
                for (edge_index p = L.outer_[j] + 1; p < L.outer_[j + 1]; ++p) {
                    if (L.inner_[p] >= m) continue;
                    if (std::fabs(static_cast<double>(L.vals_[p])) < rel * s[j]) continue;   // dropped
                    ASSERT_LT(q, trsv.csc_col_ptr()[j + 1]);
                    ASSERT_EQ(trsv.csc_row_idx()[q], L.inner_[p]);
                    const sptrsv_value_t expect = apxchol::omp_sptrsv::narrow_value(
                        ref.Lz.vals_[p], static_cast<float>(pair_scale(s[j])));
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
        // And the compacted forward solve is an exact solve of the STORED L
        // (double accumulation): componentwise residual at roundoff. (On the
        // lowprec build the kernels divide by stored_diag(), not the narrow
        // diagonal slot; under FP16_SCALED y1 is y' = D y on L~ = L D^-1.)
        {
            const std::vector<double> s = reference_scales_L11(L, m);
            double worst = 0.0;
            for (node_index i = 0; i < m; ++i) {
                double r = x[i], sc = std::fabs(x[i]);
                for (edge_index q = trsv.csr_row_ptr()[i]; q < trsv.csr_row_ptr()[i + 1]; ++q) {
                    const node_index j = trsv.csr_col_idx()[q];
                    const double v = (j == i)
                        ? apxchol::omp_sptrsv::stored_diag(L.vals_[L.outer_[i]], static_cast<float>(pair_scale(s[i])))
                        : apxchol::widen(trsv.csr_vals()[q]);
                    const double t = v * y1[j];
                    r -= t; sc += std::fabs(t);
                }
                worst = std::max(worst, std::fabs(r) / (sc + 1e-300));
            }
            EXPECT_LT(worst, 1e-11);
        }

        // The env is read at every setup: unset = the default, which IS 1e-4
        // (same result as the explicit "1e-4" above); "0" = nothing dropped.
        {
            apxchol::omp_sptrsv dflt;
            { scoped_drop_env env(nullptr); dflt.setup(L, m); }
            EXPECT_EQ(dflt.drop_stats().rel, apxchol::kFactorDropRelDefault);
            EXPECT_EQ(dflt.stored_nnz(), ref.kept);
            EXPECT_EQ(dflt.drop_stats().dropped, ref.zeroed);
            EXPECT_EQ(0, std::memcmp(dflt.csc_vals().data(), trsv.csc_vals().data(),
                                     ref.kept * sizeof(sptrsv_value_t)));
            apxchol::omp_sptrsv off;
            { scoped_drop_env env("0"); off.setup(L, m); }
            EXPECT_EQ(off.drop_stats().dropped, 0u);
            EXPECT_EQ(off.stored_nnz(), L11_nnz);
        }
    }
}

// The default is ON at kFactorDropRelDefault (documented as 1e-4 -- the
// measured value; a change of the constant must be a deliberate change here
// too), and rel = 0 / negative / junk / empty resolve as documented.
TEST(SpTRSVDrop, EnvContract) {
    EXPECT_EQ(apxchol::kFactorDropRelDefault, 1e-4);
    { scoped_drop_env env(nullptr); EXPECT_EQ(apxchol::omp_sptrsv::factor_drop_rel_from_env(), 1e-4); }
    { scoped_drop_env env("");      EXPECT_EQ(apxchol::omp_sptrsv::factor_drop_rel_from_env(), 1e-4); }
    { scoped_drop_env env("0");     EXPECT_EQ(apxchol::omp_sptrsv::factor_drop_rel_from_env(), 0.0); }
    { scoped_drop_env env("-1");    EXPECT_EQ(apxchol::omp_sptrsv::factor_drop_rel_from_env(), 0.0); }
    { scoped_drop_env env("abc");   EXPECT_EQ(apxchol::omp_sptrsv::factor_drop_rel_from_env(), 0.0); }
    { scoped_drop_env env("3e-4");  EXPECT_EQ(apxchol::omp_sptrsv::factor_drop_rel_from_env(), 3e-4); }
    { scoped_drop_env env("2");     EXPECT_EQ(apxchol::omp_sptrsv::factor_drop_rel_from_env(), 2.0); }
}

// rel = 0 / negative / junk = off; a rel below every entry drops nothing (and
// makes no copy: stored nnz == factor nnz); the diagonal survives ANY rel
// (even > 1, which drops every off-diagonal); the compacted factor solves as
// a diagonal one then (y = x / diag).
TEST(SpTRSVDrop, EdgeCases) {
    // These state the DEFAULT (fp32) storage contract, so pin it: the suite
    // is also run with APXCHOL_SPTRSV_FP16=1 in the environment.
    scoped_env fp32_storage("APXCHOL_SPTRSV_FP16", "0");

    const node_index m = 2000;
    sparse_csc L = make_random_lower(m, 4.0, 31, 0.0, 1.0);
    for (edge_index p = 0; p < L.nonZeros(); ++p)
        if ((p % 5) == 2) L.vals_[p] = static_cast<factor_value_t>(L.vals_[p] * 1e-9);   // far below 1e-4 * s_j
    std::uint64_t offdiag = 0;
    for (node_index j = 0; j < m; ++j) offdiag += L.outer_[j + 1] - L.outer_[j] - 1;
    const std::uint64_t nnz = static_cast<std::uint64_t>(L.nonZeros());
    ASSERT_GT(reference_dense_drop(L, m, 1e-4).zeroed, 100u);   // the default would remove plenty here
    // What the storage format alone would store as zero (the drop's second
    // clause, omp_sptrsv::format_flushes): nothing on the fp32 / fp64 builds
    // (every entry here is a nonzero fp32); on FP16_SCALED the
    // 1e-9-scaled entries (fp16 flushes them, subnormals included by default).
    const std::vector<double> s_pre = reference_scales_L11(L, m);
    std::uint64_t fmt_zero = 0;
    for (node_index j = 0; j < m; ++j)
        for (edge_index p = L.outer_[j] + 1; p < L.outer_[j + 1]; ++p)
            fmt_zero += apxchol::omp_sptrsv::format_flushes(L.vals_[p], static_cast<float>(s_pre[j]));
    ASSERT_EQ(fmt_zero, 0u);
    for (const char* rel : {"0", "-1", "abc"}) {
        SCOPED_TRACE(std::string("APXCHOL_FACTOR_DROP=") + rel);
        scoped_drop_env env(rel);
        apxchol::omp_sptrsv t; t.setup(L, m);
        EXPECT_EQ(t.drop_stats().rel, 0.0);
        EXPECT_EQ(t.drop_stats().dropped, 0u);
        EXPECT_EQ(t.stored_nnz(), nnz);
        EXPECT_EQ(t.csc_vals().size(), nnz);
    }
    {
        scoped_drop_env env("1e-30");   // below everything: nothing goes (but the format zeros)
        apxchol::omp_sptrsv t; t.setup(L, m);
        EXPECT_EQ(t.drop_stats().rel, 1e-30);
        EXPECT_EQ(t.drop_stats().dropped, fmt_zero);
        EXPECT_EQ(t.drop_stats().dropped_threshold, 0u);
        EXPECT_EQ(t.drop_stats().dropped_flush, fmt_zero);
        EXPECT_EQ(t.stored_nnz(), nnz - fmt_zero);
        EXPECT_EQ(t.csc_col_ptr().back(), static_cast<edge_index>(nnz - fmt_zero));
        if (fmt_zero == 0) {
            // Nothing dropped, so nothing compensated: the CSC IS the factor
            // (through the storage contract: a plain cast on fp32/fp64).
            EXPECT_EQ(0, std::memcmp(t.csc_row_idx().data(), L.inner_.data(), nnz * sizeof(node_index)));
            std::uint64_t mismatches = 0;
            for (node_index j = 0; j < m; ++j)
                for (edge_index p = L.outer_[j]; p < L.outer_[j + 1]; ++p)
                    mismatches += !(t.csc_vals()[p] == apxchol::omp_sptrsv::narrow_value(
                        L.vals_[p], static_cast<float>(pair_scale(s_pre[j]))));
            EXPECT_EQ(mismatches, 0u);
        }
    }
    {
        scoped_drop_env env("2");        // > 1: every off-diagonal goes, diagonal stays
        apxchol::omp_sptrsv t; t.setup(L, m);
        EXPECT_EQ(t.drop_stats().dropped, offdiag);
        EXPECT_EQ(t.stored_nnz(), static_cast<std::uint64_t>(m));
        for (node_index j = 0; j < m; ++j) {
            ASSERT_EQ(t.csc_col_ptr()[j], static_cast<edge_index>(j));
            ASSERT_EQ(t.csc_row_idx()[j], j);
            ASSERT_EQ(t.csr_row_ptr()[j], static_cast<edge_index>(j));
            ASSERT_EQ(t.csr_col_idx()[j], j);
        }
        // Solving with a diagonal factor: the forward sweep divides by the
        // stored diagonal (L_jj; L_jj / s_j under *_SCALED, where it returns
        // y' = D y and the PRE-drop s_j is still the scale) and the pair
        // (forward, then back on the forward's output) applies (L L^T)^-1 =
        // 1 / L_jj^2 -- exactly on fp32/fp64, to fp32(1/s_j)'s 2^-23 under
        // *_SCALED.
        std::vector<double> x(m, 1.0), y(m), z(m);
        t.forward_solve(x.data(), y.data());
        t.transpose_solve(y.data(), z.data());
        for (node_index j = 0; j < m; ++j) {
            const double L_jj = static_cast<double>(L.vals_[L.outer_[j]]);
            const double d    = apxchol::omp_sptrsv::stored_diag(L.vals_[L.outer_[j]], static_cast<float>(pair_scale(s_pre[j])));
            ASSERT_NEAR(y[j], 1.0 / d, 1e-15 / d) << j;
            ASSERT_NEAR(z[j], 1.0 / (L_jj * L_jj), 4e-15 / (L_jj * L_jj)) << j;
        }
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
    scoped_drop_env env("1e-4");
    apxchol::omp_sptrsv ref;
#ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    ref.setup(L, m);
    ASSERT_GT(ref.drop_stats().dropped, static_cast<std::uint64_t>(m));   // teeth
    ASSERT_EQ(ref.drop_stats().dropped, reference_dense_drop(L, m, 1e-4).zeroed);
    auto same = [&](const apxchol::omp_sptrsv& t) {
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
#ifdef _OPENMP
    for (int threads : {2, 3, 4, max_threads}) {
        SCOPED_TRACE("threads=" + std::to_string(threads));
        omp_set_num_threads(threads);
        apxchol::omp_sptrsv t;
        t.setup(L, m);
        same(t);
    }
    omp_set_num_threads(max_threads);
#else
    apxchol::omp_sptrsv t;
    t.setup(L, m);
    same(t);
#endif
}

// ── GPU backend host preparation (cuda_host.h; CUDA-free, so stated here on
// every build) ────────────────────────────────────────────────────────────────
//
// cuda_sptrsv::setup builds L11 as int32 CSC arrays, runs THE compacting drop
// (factor_drop.h -- the same implementation omp_sptrsv::setup runs) on them
// and uploads the result. What it uploads must be what the CPU backend
// stores: same stored nnz, same column pointers / row indices, same values
// (bit-for-bit through the fp32/fp64 storage; through narrow_value() on the
// lowprec builds), for the Laplacian (m = n-1) and SDDM (m = n) paths, drop
// on (explicit and default) and off.
#include "apxchol/solver/sptrsv/cuda_host.h"

TEST(GpuHostPrep, DropOnTheGpuHostArraysIsTheCpuDrop) {
    // These state the DEFAULT (fp32) storage contract, so pin it: the suite
    // is also run with APXCHOL_SPTRSV_FP16=1 in the environment.
    scoped_env fp32_storage("APXCHOL_SPTRSV_FP16", "0");

    struct cfg { node_index n; bool laplacian; };
    for (const char* rel_env : {"1e-4", static_cast<const char*>(nullptr), "0"})
    for (cfg c : {cfg{3000, false}, cfg{3001, true}, cfg{60001, true}}) {
        const node_index m = c.laplacian ? c.n - 1 : c.n;
        SCOPED_TRACE("n=" + std::to_string(c.n) + (c.laplacian ? " (Laplacian m=n-1)" : " (SDDM m=n)") +
                     " APXCHOL_FACTOR_DROP=" + (rel_env ? rel_env : "<unset>"));
        scoped_drop_env env(rel_env);
        const sparse_csc L = make_wide_magnitude_lower(c.n);
        // CPU backend.
        apxchol::omp_sptrsv trsv;
        trsv.setup(L, m);
        // GPU backend's host side, exactly as cuda_sptrsv::setup does it (fp32
        // storage: keep predicate = |v| >= rel * s_j and v != 0).
        const double rel = apxchol::factor_drop_rel_from_env();
        auto LT = apxchol::cuda_host::build_L11_csc_int<factor_value_t>(L, m);
        const std::vector<float> scales = apxchol::cuda_host::column_scales(LT);
        const std::uint64_t L11_nnz = L11_nnz_of(L, m);
        ASSERT_EQ(static_cast<std::uint64_t>(LT.nnz), L11_nnz);
        const apxchol::factor_drop_stats st = apxchol::cuda_host::apply_factor_drop(
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
        if (rel_env && std::string(rel_env) == "1e-4") { ASSERT_GT(st.dropped, L11_nnz / 4); }   // teeth
        if (rel_env && std::string(rel_env) == "0")    { ASSERT_EQ(st.dropped, 0u); }
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
                const sptrsv_value_t expect = apxchol::omp_sptrsv::narrow_value(
                    LT.vals[k], s);
                val_mismatch += !(trsv.csc_vals()[k] == expect);
            }
        }
        EXPECT_EQ(val_mismatch, 0u);
        // And the GPU-side transpose (CSR of L for the dataflow forward
        // sweep) is the CPU's CSR: same row pointers, column indices, values.
        const auto Lc = apxchol::cuda_host::transpose_csr(LT);
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
                const sptrsv_value_t expect = apxchol::omp_sptrsv::narrow_value(Lc.vals[k], scales[j]);
                t_mismatch += !(trsv.csr_vals()[k] == expect);
            }
        EXPECT_EQ(t_mismatch, 0u);
    }
}

// The fp16 per-column-scaled storage the dataflow backend uploads under
// APXCHOL_SPTRSV_FP16=1 (cuda_host.h file header): each slot is
// binary16(fp32(v) / s_j) RNE with fp16 subnormals flushed to signed zero
// (restated here through lowprec.h's fp16_t on the bit level), diag[j] =
// fp32(L_jj) / s_j, inv_scale[j] = fp32(1 / s_j); with diag_comp the column's
// rounding residual is folded into diag[j] so the STORED column (diag +
// widened off-diagonals) sums to the fp32 column / s_j; the drop's fp16 keep
// predicate additionally removes what fp16 flushes. Under the CPU FP16_SCALED
// build the bits equal omp_sptrsv's stored bits entry for entry.
TEST(GpuHostPrep, Fp16ScaledStorageContract) {
    for (bool laplacian : {false, true}) {
        const node_index n = 4001;
        const node_index m = laplacian ? n - 1 : n;
        SCOPED_TRACE(laplacian ? "Laplacian m=n-1" : "SDDM m=n");
        const sparse_csc L = make_wide_magnitude_lower(n);
        const double rel = 1e-4;
        auto LT = apxchol::cuda_host::build_L11_csc_int<factor_value_t>(L, m);
        const std::vector<float> scales = apxchol::cuda_host::column_scales(LT);
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
                    const apxchol::fp16_t h(v / scales[j]);
                    const bool fmt = !(apxchol::fp16_t::is_zero(h.bits) || apxchol::fp16_t::is_subnormal(h.bits));
                    disagree += apxchol::cuda_host::keep_offdiag(LT.vals[k], scales[j], rel, true) != (thr && fmt);
                    disagree += apxchol::cuda_host::keep_offdiag(LT.vals[k], scales[j], 1e-30, true) != (v != 0.0f && fmt);
                    disagree += apxchol::cuda_host::keep_offdiag(LT.vals[k], scales[j], 1e-30, false) != (v != 0.0f);
                    only_fmt += (v != 0.0f) && !fmt;
                }
            EXPECT_EQ(disagree, 0u);
            EXPECT_GT(only_fmt, 0u);   // teeth: the wide-magnitude factor has fp16-subnormal entries
        }
        const apxchol::factor_drop_stats st = apxchol::cuda_host::apply_factor_drop(
            LT, scales, rel, /*fp16_storage=*/true);
        ASSERT_GT(st.dropped, 0u);
        {
            const auto h16 = apxchol::cuda_host::narrow_fp16_scaled(LT, scales);
            ASSERT_EQ(h16.diag.size(), static_cast<size_t>(m));
            ASSERT_EQ(h16.inv_scale.size(), static_cast<size_t>(m));
            std::uint64_t bits_mismatch = 0, sub = 0;
            double worst_cs = 0.0;
            for (node_index j = 0; j < m; ++j) {
                const float s = scales[j];
                EXPECT_EQ(h16.inv_scale[j], 1.0f / s);
                ASSERT_EQ(LT.idx[LT.ptr[j]], static_cast<int>(j));
                const float d0 = static_cast<float>(LT.vals[LT.ptr[j]]) / s;   // fp32(L_jj) / s_j
                double resid = 0.0, col_x = 0.0, col_stored = 0.0;
                for (int k = LT.ptr[j]; k < LT.ptr[j + 1]; ++k) {
                    const float v = static_cast<float>(LT.vals[k]);
                    apxchol::fp16_t h(v / s);                                    // RNE
                    if (apxchol::fp16_t::is_subnormal(h.bits))
                        h = apxchol::fp16_t::from_bits(static_cast<std::uint16_t>(h.bits & 0x8000u));
                    bits_mismatch += h16.vals[k] != h.bits;
                    if (k == LT.ptr[j]) continue;                                // diagonal slot
                    sub += apxchol::fp16_t::is_subnormal(h16.vals[k]);
                    const double w = static_cast<double>(apxchol::fp16_t::from_bits(h16.vals[k]).to_float());
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
            EXPECT_EQ(bits_mismatch, 0u);
            EXPECT_EQ(sub, 0u);                      // subnormals flushed
            EXPECT_EQ(h16.subnormal, 0u);
            EXPECT_LT(worst_cs, 4e-7);               // one fp32 rounding of the diagonal
        }
    }
}

// ── The dataflow backend's host side (cuda_host.h; CUDA-free) ────────────────
//
// dataflow_batches packs the rows of one sweep direction into warps: rows in
// sweep order, each taking a lane group G = dataflow_lane_group(row length,
// pre) -- the smallest power of two <= 32 with G * pre >= length -- aligned to
// a multiple of G, a batch closed exactly when the next row does not fit
// (greedy, so no batch could have taken one more row). The device restates
// the same rule lane by lane, so this table IS the contract between the two.
TEST(GpuHostPrep, DataflowBatchesPackRowsIntoAlignedLaneGroups) {
    for (int pre : {4, 8}) {
        SCOPED_TRACE("pre=" + std::to_string(pre));
        // The rule itself.
        EXPECT_EQ(apxchol::cuda_host::dataflow_lane_group(1, pre), 1);
        EXPECT_EQ(apxchol::cuda_host::dataflow_lane_group(pre, pre), 1);
        EXPECT_EQ(apxchol::cuda_host::dataflow_lane_group(pre + 1, pre), 2);
        EXPECT_EQ(apxchol::cuda_host::dataflow_lane_group(4 * pre, pre), 4);
        EXPECT_EQ(apxchol::cuda_host::dataflow_lane_group(32 * pre, pre), 32);
        EXPECT_EQ(apxchol::cuda_host::dataflow_lane_group(1000 * pre, pre), 32);   // capped: the kernel chunks the rest
        for (int len = 1; len <= 40 * pre; ++len) {
            const int G = apxchol::cuda_host::dataflow_lane_group(len, pre);
            ASSERT_TRUE(G >= 1 && G <= 32 && (G & (G - 1)) == 0) << len;
            if (G < 32) ASSERT_GE(G * pre, len) << len;
            if (G > 1)  ASSERT_LT((G / 2) * pre, len) << len;                     // minimal
        }
        // On a real-shaped factor, both directions.
        const node_index n = 60001, m = n - 1;
        const sparse_csc L = make_wide_magnitude_lower(n);
        scoped_drop_env env("1e-4");
        auto LT = apxchol::cuda_host::build_L11_csc_int<factor_value_t>(L, m);
        { const std::vector<float> sc = apxchol::cuda_host::column_scales(LT);
          apxchol::cuda_host::apply_factor_drop(LT, sc, 1e-4, false); }
        const auto Lc = apxchol::cuda_host::transpose_csr(LT);
        for (bool reverse : {false, true}) {
            SCOPED_TRACE(reverse ? "back (reverse order, CSR of L^T)" : "forward (natural order, CSR of L)");
            const auto& A = reverse ? LT : Lc;
            const std::vector<int> len = apxchol::cuda_host::csr_row_lengths(A.m, A.ptr.data());
            ASSERT_EQ(len.size(), static_cast<size_t>(m));
            for (int i = 0; i < A.m; ++i) ASSERT_EQ(len[i], A.ptr[i + 1] - A.ptr[i]);
            const std::vector<int> bs = apxchol::cuda_host::dataflow_batches(A.m, reverse, len.data(), pre);
            ASSERT_GE(bs.size(), 2u);
            EXPECT_EQ(bs.front(), 0);
            EXPECT_EQ(bs.back(), A.m);
            auto lanes_needed = [&](int q, int pos) {   // aligned start + G of sweep position q placed at lane pos
                const int row = reverse ? A.m - 1 - q : q;
                const int G = apxchol::cuda_host::dataflow_lane_group(len[row], pre);
                const int start = (pos + G - 1) & ~(G - 1);
                return std::pair<int, int>{start, G};
            };
            std::uint64_t rows_seen = 0, full_batches = 0;
            for (size_t b = 0; b + 1 < bs.size(); ++b) {
                ASSERT_LT(bs[b], bs[b + 1]) << "batch " << b;                     // non-empty, increasing
                ASSERT_LE(bs[b + 1] - bs[b], 32) << "batch " << b;
                int pos = 0;
                for (int q = bs[b]; q < bs[b + 1]; ++q) {
                    const auto [start, G] = lanes_needed(q, pos);
                    ASSERT_LE(start + G, 32) << "batch " << b << " row " << q;   // fits
                    ASSERT_EQ(start % G, 0);                                    // aligned
                    pos = start + G;
                    ++rows_seen;
                }
                if (b + 2 < bs.size()) {                                          // greedy: the next row would not fit
                    const auto [start, G] = lanes_needed(bs[b + 1], pos);
                    EXPECT_GT(start + G, 32) << "batch " << b << " closed early";
                }
                full_batches += (bs[b + 1] - bs[b]) >= 8;
            }
            EXPECT_EQ(rows_seen, static_cast<std::uint64_t>(m));
            EXPECT_GT(full_batches, 0u);                                           // teeth: short rows pack many per warp
            EXPECT_LT(bs.size() - 1, static_cast<size_t>(m));                      // ... and long ones do not
        }
    }
}

// ── ROW SEGMENTATION, host side (cuda_host.h) ───────────────────────────────
//
// dataflow_row_stats is the evidence the split threshold is picked from, so
// its buckets are stated here independently: bucket k counts the rows an
// unsplit walk would chew in more than 2^k chunks of C = 32 * pre entries.
TEST(GpuHostPrep, DataflowRowStatsCountLongRowsAndTheEntriesTheyHold) {
    const int pre = 8;                                     // C = 256
    const std::vector<int> len = {1, 2, 255, 256, 257, 512, 1024, 100000};
    const auto st = apxchol::cuda_host::dataflow_row_stats(static_cast<int>(len.size()), len.data(), pre);
    EXPECT_EQ(st.base, 256);
    EXPECT_EQ(st.m, 8);
    std::int64_t total = 0; for (int l : len) total += l;
    EXPECT_EQ(st.nnz, total);
    EXPECT_EQ(st.max_len, 100000);
    EXPECT_DOUBLE_EQ(st.mean_len, static_cast<double>(total) / 8.0);
    EXPECT_EQ(st.rows_ge[0], 5);                           // 256, 257, 512, 1024, 100000
    EXPECT_EQ(st.nnz_ge[0], 256 + 257 + 512 + 1024 + 100000);
    EXPECT_EQ(st.rows_ge[1], 3);                           // >= 512
    EXPECT_EQ(st.rows_ge[2], 2);                           // >= 1024
    EXPECT_EQ(st.rows_ge[3], 1);                           // >= 2048: only the 100000 row
    EXPECT_EQ(st.rows_ge[8], 1);                           // >= 65536
    EXPECT_EQ(st.rows_ge[9], 0);                           // >= 131072
    EXPECT_EQ(st.nnz_ge[8], 100000);
    // Empty input: no buckets, no division by zero.
    const auto e = apxchol::cuda_host::dataflow_row_stats(0, nullptr, pre);
    EXPECT_EQ(e.nnz, 0); EXPECT_EQ(e.max_len, 0); EXPECT_DOUBLE_EQ(e.mean_len, 0.0);
}

namespace {
// A tiny CSR with a diagonal in every row, built from row lengths: row i gets
// `len[i]` entries, the diagonal LAST (the CSR-of-L convention) or FIRST (the
// CSR-of-L^T one), the rest distinct column indices != i.
struct toy_csr {
    std::vector<int> ptr, idx, len;
};
toy_csr make_toy_csr(const std::vector<int>& len, bool diag_first) {
    toy_csr t; t.len = len;
    t.ptr.assign(len.size() + 1, 0);
    for (size_t i = 0; i < len.size(); ++i) t.ptr[i + 1] = t.ptr[i] + len[i];
    t.idx.resize(static_cast<size_t>(t.ptr.back()));
    for (size_t i = 0; i < len.size(); ++i) {
        int col = 0;
        for (int p = t.ptr[i]; p < t.ptr[i + 1]; ++p) {
            const bool is_diag = diag_first ? (p == t.ptr[i]) : (p == t.ptr[i + 1] - 1);
            if (is_diag) { t.idx[static_cast<size_t>(p)] = static_cast<int>(i); continue; }
            if (col == static_cast<int>(i)) ++col;
            t.idx[static_cast<size_t>(p)] = col++;
        }
    }
    return t;
}
} // namespace

// The plan with seg == 0 IS dataflow_batches() -- the rollback switch, and
// what makes "segmentation off" bit-identical to the pre-segmentation kernel.
TEST(GpuHostPrep, DataflowPlanWithoutSegmentationIsTheBatchTable) {
    const node_index n = 60001, m = n - 1;
    const sparse_csc L = make_wide_magnitude_lower(n);
    scoped_drop_env env("1e-4");
    auto LT = apxchol::cuda_host::build_L11_csc_int<factor_value_t>(L, m);
    { const std::vector<float> sc = apxchol::cuda_host::column_scales(LT);
      apxchol::cuda_host::apply_factor_drop(LT, sc, 1e-4, false); }
    const auto Lc = apxchol::cuda_host::transpose_csr(LT);
    for (int pre : {4, 8}) {
        for (bool reverse : {false, true}) {
            SCOPED_TRACE("pre=" + std::to_string(pre) + (reverse ? " back" : " forward"));
            const auto& A = reverse ? LT : Lc;
            const std::vector<int> len = apxchol::cuda_host::csr_row_lengths(A.m, A.ptr.data());
            const std::vector<int> bs = apxchol::cuda_host::dataflow_batches(A.m, reverse, len.data(), pre);
            const apxchol::cuda_host::dataflow_plan pl = apxchol::cuda_host::dataflow_build_plan(
                A.m, reverse, A.ptr.data(), A.idx.get(), len.data(), pre,
                apxchol::cuda_host::dataflow_seg_params{});          // seg = 0
            EXPECT_EQ(pl.batch_start, bs);
            EXPECT_EQ(pl.n_slots, 0);
            EXPECT_EQ(pl.n_split, 0);
            EXPECT_TRUE(pl.spec.empty());
            ASSERT_EQ(pl.batch_spec.size() + 1, pl.batch_start.size());
            for (int v : pl.batch_spec) EXPECT_EQ(v, -1);
            EXPECT_EQ(apxchol::cuda_host::dataflow_plan_check(pl, A.m, reverse, len.data(), pre), "");
        }
    }
}

// With segmentation on: the segments of a split row cover its CSR range
// exactly once, `dpos` IS the diagonal, the items are emitted in sweep order
// with every segment before its finalizer, each special item is alone in its
// batch, batch_start stays monotone and spans [0, m), and the extra state is
// O(n) + O(nnz / seg).
TEST(GpuHostPrep, DataflowPlanSegmentsCoverSplitRowsExactlyOnce) {
    const int pre = 8, C = 32 * pre;
    // Row lengths with a real hub tail; both diagonal conventions.
    std::vector<int> len(4000);
    for (size_t i = 0; i < len.size(); ++i) len[i] = 1 + static_cast<int>(i % 9);
    len[10] = 5000; len[11] = 260; len[2000] = 40000; len[3999] = 1200; len[0] = 1;
    for (bool diag_first : {false, true}) {
        const toy_csr t = make_toy_csr(len, diag_first);
        const int m = static_cast<int>(len.size());
        for (bool reverse : {false, true}) {
            for (const apxchol::cuda_host::dataflow_seg_params p :
                 {apxchol::cuda_host::dataflow_seg_params{1024, 2048, 4096},
                  apxchol::cuda_host::dataflow_seg_params{256, 0, 4096},      // torture: split everything
                  apxchol::cuda_host::dataflow_seg_params{1 << 28, 128, 4096}}) {  // forced S = 1
                SCOPED_TRACE("diag_first=" + std::to_string(diag_first) + " reverse=" + std::to_string(reverse) +
                             " seg=" + std::to_string(p.seg) + " split_min=" + std::to_string(p.split_min));
                const auto pl = apxchol::cuda_host::dataflow_build_plan(
                    m, reverse, t.ptr.data(), t.idx.data(), len.data(), pre, p);
                ASSERT_EQ(apxchol::cuda_host::dataflow_plan_check(pl, m, reverse, len.data(), pre), "");
                // Walk the plan in order and re-derive everything.
                std::vector<char> covered(t.idx.size(), 0);
                int split_rows = 0, slots_seen = 0;
                std::vector<int> pending_slots;                    // slots of the row being segmented
                int pending_row = -1;
                for (const auto& sp : pl.spec) {
                    if (sp.row >= 0) {                             // SEGMENT
                        if (pending_row != sp.row) { EXPECT_TRUE(pending_slots.empty()); pending_row = sp.row; }
                        ASSERT_GT(sp.b, sp.a);
                        ASSERT_GE(sp.a, t.ptr[static_cast<size_t>(sp.row)]);
                        ASSERT_LE(sp.b, t.ptr[static_cast<size_t>(sp.row) + 1]);
                        ASSERT_EQ(sp.c, slots_seen) << "slots are handed out in order";
                        ++slots_seen;
                        pending_slots.push_back(sp.c);
                        // whole chunks, so every segment has the same depth
                        if (sp.b != t.ptr[static_cast<size_t>(sp.row) + 1]) EXPECT_EQ((sp.b - sp.a) % C, 0);
                        for (int q = sp.a; q < sp.b; ++q) {
                            ASSERT_EQ(covered[static_cast<size_t>(q)], 0) << "entry " << q << " covered twice";
                            covered[static_cast<size_t>(q)] = 1;
                        }
                        continue;
                    }
                    const int row = ~sp.row;                       // FINALIZER
                    ASSERT_EQ(row, pending_row) << "finalizer of a row whose segments did not just precede it";
                    ASSERT_EQ(sp.b, static_cast<int>(pending_slots.size()));
                    ASSERT_EQ(sp.a, pending_slots.front());
                    EXPECT_EQ(t.idx[static_cast<size_t>(sp.c)], row) << "dpos must be the diagonal";
                    ASSERT_GE(sp.c, t.ptr[static_cast<size_t>(row)]);
                    ASSERT_LT(sp.c, t.ptr[static_cast<size_t>(row) + 1]);
                    if (p.seg == (1 << 28)) EXPECT_EQ(sp.b, 1) << "seg > any row must give S = 1";
                    // the segments covered the row's whole CSR range
                    for (int q = t.ptr[static_cast<size_t>(row)]; q < t.ptr[static_cast<size_t>(row) + 1]; ++q)
                        ASSERT_EQ(covered[static_cast<size_t>(q)], 1) << "row " << row << " entry " << q;
                    pending_slots.clear();
                    pending_row = -1;
                    ++split_rows;
                }
                EXPECT_TRUE(pending_slots.empty());
                EXPECT_EQ(split_rows, pl.n_split);
                EXPECT_EQ(slots_seen, pl.n_slots);
                EXPECT_EQ(pl.spec.size(), static_cast<size_t>(pl.n_slots + pl.n_split));
                // exactly the rows over the threshold were split
                int expect_split = 0;
                for (int l : len) expect_split += (l - 1 > p.split_min);
                EXPECT_EQ(pl.n_split, expect_split);
                // O(n) + O(nnz / seg) state
                EXPECT_LE(pl.n_slots, static_cast<int>(t.idx.size()) / std::min(p.seg, C) + pl.n_split);
                // the split rows do NOT also appear in a plain batch
                for (const auto& sp : pl.spec) {
                    if (sp.row >= 0) continue;
                    EXPECT_GT(len[static_cast<size_t>(~sp.row)] - 1, p.split_min);
                }
            }
        }
    }
}

// max_seg caps S even for an absurdly long row, so one row can never occupy
// an unbounded slice of the ticket window.
TEST(GpuHostPrep, DataflowPlanCapsTheSegmentCountPerRow) {
    const int pre = 8;
    std::vector<int> len(4, 3);
    len[2] = 1000000;
    const toy_csr t = make_toy_csr(len, false);
    apxchol::cuda_host::dataflow_seg_params p{256, 0, 8};
    const auto pl = apxchol::cuda_host::dataflow_build_plan(
        4, false, t.ptr.data(), t.idx.data(), len.data(), pre, p);
    EXPECT_EQ(apxchol::cuda_host::dataflow_plan_check(pl, 4, false, len.data(), pre), "");
    for (const auto& sp : pl.spec)
        if (sp.row < 0 && ~sp.row == 2) EXPECT_LE(sp.b, p.max_seg);
    EXPECT_LE(pl.n_slots, 8 + 3);
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
template <class T> T* dev_upload(const T* h, std::size_t count) {
    T* d = nullptr;
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
// cuda_host.h contract); otherwise the fp32 values are read inline and the
// diagonal is the row's own diagonal slot.
//   out[i] = (rhs[i] * (in_scale ? in_scale[i] : 1) - sum_{j != i} T[i,j] out[j]) / T[i,i]
std::vector<float> host_reference_sweep(int m, bool reverse,
                                        const int* rowptr, const int* colidx,
                                        const float* vals, const std::uint16_t* vals16,
                                        const float* diag, const double* in_scale,
                                        const std::vector<float>& rhs) {
    std::vector<double> out(static_cast<std::size_t>(m), 0.0);
    for (int k = 0; k < m; ++k) {
        const int i = reverse ? m - 1 - k : k;
        double sum = 0.0, d = 0.0;
        for (int p = rowptr[i]; p < rowptr[i + 1]; ++p) {
            const int j = colidx[p];
            const double v = vals16 ? apxchol::widen(apxchol::fp16_t::from_bits(vals16[p]))
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
    L.n_ = n;
    L.outer_.assign(static_cast<size_t>(n) + 1, 0);
    for (node_index j = 0; j < n; ++j) {
        auto& rows = col_rows[j];
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        L.outer_[j + 1] = L.outer_[j] + static_cast<edge_index>(rows.size());
    }
    L.inner_.resize(static_cast<size_t>(L.outer_[n]));
    L.vals_.resize(static_cast<size_t>(L.outer_[n]));
    for (node_index j = 0; j < n; ++j) {
        edge_index out = L.outer_[j];
        for (node_index r : col_rows[j]) {
            L.inner_[out] = r;
            L.vals_[out] = static_cast<factor_value_t>(
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
// One sweep direction's plan on the device (cuda_host::dataflow_build_plan).
struct dev_plan {
    int*  bs    = nullptr;      // batch_start, nb + 1
    int*  sel   = nullptr;      // batch_spec, nb
    int4* spec  = nullptr;      // the segmentation items (null when there are none)
    int   nb    = 0;
    int   slots = 0;
};
dev_plan upload_plan(const apxchol::cuda_host::dataflow_plan& pl) {
    dev_plan d;
    d.bs  = dev_upload(pl.batch_start.data(), pl.batch_start.size());
    d.sel = dev_upload(pl.batch_spec.data(), pl.batch_spec.size());
    if (!pl.spec.empty()) {
        static_assert(sizeof(apxchol::cuda_host::dataflow_spec) == sizeof(int4), "int4 upload");
        int4* p = nullptr;
        EXPECT_EQ(cudaMalloc(&p, pl.spec.size() * sizeof(int4)), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(p, pl.spec.data(), pl.spec.size() * sizeof(int4), cudaMemcpyHostToDevice), cudaSuccess);
        d.spec = p;
    }
    d.nb = pl.n_batches(); d.slots = pl.n_slots;
    return d;
}
void free_plan(dev_plan& d) { cudaFree(d.bs); cudaFree(d.sel); cudaFree(d.spec); d = dev_plan{}; }

// Everything one factor needs on the device: the two CSRs, both storages, and
// the host arrays the serial fp64 reference reads. Frees itself.
struct df_factor {
    int m = 0;
    apxchol::cuda_host::csr_int<float>         LT, Lc;
    apxchol::cuda_host::csr_int<std::uint16_t> LT16, L16;
    apxchol::cuda_host::fp16_scaled_arrays     h16;
    std::vector<double> inv_scale2;
    std::vector<int>    lenL, lenT;
    int *d_Lp = nullptr, *d_Li = nullptr, *d_Tp = nullptr, *d_Ti = nullptr;
    float *d_Lv = nullptr, *d_Tv = nullptr, *d_diag = nullptr;
    std::uint16_t *d_Lv16 = nullptr, *d_Tv16 = nullptr;
    double* d_is2 = nullptr;

    df_factor(const sparse_csc& L, node_index mm) {
        m = static_cast<int>(mm);
        LT = apxchol::cuda_host::build_L11_csc_int<apxchol::cuda_value_t>(L, mm);
        const std::vector<float> scales = apxchol::cuda_host::column_scales(LT);
        apxchol::cuda_host::apply_factor_drop(LT, scales, 1e-4, false);
        Lc  = apxchol::cuda_host::transpose_csr(LT);
        h16 = apxchol::cuda_host::narrow_fp16_scaled(LT, scales);
        LT16.m = LT.m; LT16.nnz = LT.nnz; LT16.ptr = LT.ptr;
        LT16.idx = std::make_unique_for_overwrite<int[]>(static_cast<size_t>(LT.nnz));
        std::copy_n(LT.idx.get(), LT.nnz, LT16.idx.get());
        LT16.vals = std::make_unique_for_overwrite<std::uint16_t[]>(static_cast<size_t>(LT.nnz));
        std::copy_n(h16.vals.get(), LT.nnz, LT16.vals.get());
        L16 = apxchol::cuda_host::transpose_csr(LT16);
        inv_scale2.resize(static_cast<size_t>(m));
        for (int j = 0; j < m; ++j) inv_scale2[static_cast<size_t>(j)] =
            static_cast<double>(h16.inv_scale[static_cast<size_t>(j)]) * h16.inv_scale[static_cast<size_t>(j)];
        lenL = apxchol::cuda_host::csr_row_lengths(m, Lc.ptr.data());
        lenT = apxchol::cuda_host::csr_row_lengths(m, LT.ptr.data());
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

    apxchol::cuda_host::dataflow_plan plan(bool reverse, const apxchol::cuda_host::dataflow_seg_params& p) const {
        const auto& A = reverse ? LT : Lc;
        const auto& len = reverse ? lenT : lenL;
        auto pl = apxchol::cuda_host::dataflow_build_plan(m, reverse, A.ptr.data(), A.idx.get(),
                                                          len.data(), apxchol::dataflow_prefetch_depth(), p);
        EXPECT_EQ(apxchol::cuda_host::dataflow_plan_check(pl, m, reverse, len.data(),
                                                          apxchol::dataflow_prefetch_depth()), "");
        return pl;
    }
};

// One sweep through the kernel with a given plan. `tag` must hold
// m + plan.slots words and carry no word of `epoch`.
void run_sweep(const df_factor& F, const dev_plan& dp, bool reverse, bool fp16,
               unsigned long long* tag, unsigned epoch, int* ctrl, int grid,
               const float* d_rhs, float* d_out) {
    if (fp16)
        apxchol::dataflow_solve_fp16(0, F.m, reverse,
            reverse ? F.d_Tp : F.d_Lp, reverse ? F.d_Ti : F.d_Li, reverse ? F.d_Tv16 : F.d_Lv16,
            F.d_diag, reverse ? F.d_is2 : nullptr,
            dp.bs, dp.sel, dp.spec, dp.nb, tag, epoch, ctrl, grid, d_rhs, d_out);
    else
        apxchol::dataflow_solve(0, F.m, reverse,
            reverse ? F.d_Tp : F.d_Lp, reverse ? F.d_Ti : F.d_Li, reverse ? F.d_Tv : F.d_Lv,
            dp.bs, dp.sel, dp.spec, dp.nb, tag, epoch, ctrl, grid, d_rhs, d_out);
}
} // namespace

// Every sweep, on both factors, against the SERIAL DOUBLE reference over the
// same arrays -- with segmentation off, at the shipped setting, and in the
// TORTURE setting (every row split into 256-entry pieces) -- plus
// bit-identity across launches and grid sizes (1 block = 8 warps against
// thousands of segment items is also the deadlock torture).
TEST(GpuDataflow, MatchesTheSerialDoubleReferenceBothDirectionsAndIsDeterministic) {
    const node_index n = 60001, m = n - 1;
    scoped_drop_env env("1e-4");
    const int resident = apxchol::dataflow_grid_size(false);
    ASSERT_GE(resident, 1);
    struct seg_cfg { const char* name; apxchol::cuda_host::dataflow_seg_params p; };
    const seg_cfg cfgs[] = {
        {"seg=0 (off)",              {0, 0, 4096}},
        {"seg=1024 split_min=2048",  {1024, 2048, 4096}},
        {"seg=256 split_min=0 (torture)", {256, 0, 4096}},
    };
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
        int* d_ctrl = nullptr;
        ASSERT_EQ(cudaMalloc(&d_ctrl, 2 * sizeof(int)), cudaSuccess);
        ASSERT_EQ(cudaMemset(d_ctrl, 0, 2 * sizeof(int)), cudaSuccess);
        for (const seg_cfg& cfg : cfgs) {
            SCOPED_TRACE(cfg.name);
            const auto plf = F.plan(false, cfg.p), plb = F.plan(true, cfg.p);
            std::fprintf(stderr, "[GpuDataflow] %s %s: split rows fwd %d bck %d, slots %d/%d, batches %d/%d\n",
                         kind ? "hub" : "plain", cfg.name, plf.n_split, plb.n_split,
                         plf.n_slots, plb.n_slots, plf.n_batches(), plb.n_batches());
            // Teeth: the settings must actually reach the segment path (and
            // seg = 0 must not).
            if (cfg.p.seg == 0) { EXPECT_EQ(plf.n_split + plb.n_split, 0); }
            else if (kind)      { EXPECT_GT(plf.n_split, 0) << "hub factor: forward sweep must split"; }
            dev_plan pf = upload_plan(plf);
            dev_plan pb = upload_plan(plb);
            const size_t words = static_cast<size_t>(m) + static_cast<size_t>(std::max(pf.slots, pb.slots));
            unsigned long long* d_tag = nullptr;
            ASSERT_EQ(cudaMalloc(&d_tag, words * sizeof(unsigned long long)), cudaSuccess);
            ASSERT_EQ(cudaMemset(d_tag, 0, words * sizeof(unsigned long long)), cudaSuccess);
            unsigned epoch = 0;
            for (bool fp16 : {false, true}) {
                SCOPED_TRACE(fp16 ? "fp16 storage" : "fp32 storage");
                // ── forward ──
                const std::vector<float> y_ref = host_reference_sweep(
                    F.m, false, F.Lc.ptr.data(), F.Lc.idx.get(), F.Lc.vals.get(),
                    fp16 ? F.L16.vals.get() : nullptr, F.h16.diag.data(), nullptr, x);
                run_sweep(F, pf, false, fp16, d_tag, ++epoch, d_ctrl, resident, d_x, d_a);
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                const std::vector<float> y_df = dev_download(d_a, static_cast<size_t>(m));
                { const auto [rd, nd] = rel_diff(y_ref, y_df);
                  EXPECT_LT(rd, kSweepTol) << "forward: dataflow vs serial double reference";
                  EXPECT_GT(nd, 0u);
                  std::fprintf(stderr, "[GpuDataflow] %s %s %s forward: rel diff %.3e\n",
                               kind ? "hub" : "plain", cfg.name, fp16 ? "fp16" : "fp32", rd); }
                for (int grid : {resident, 1, 7}) {
                    run_sweep(F, pf, false, fp16, d_tag, ++epoch, d_ctrl, grid, d_x, d_b);
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
                run_sweep(F, pb, true, fp16, d_tag, ++epoch, d_ctrl, resident, d_y, d_a);
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                const std::vector<float> z_df = dev_download(d_a, static_cast<size_t>(m));
                { const auto [rd, nd] = rel_diff(z_ref, z_df);
                  EXPECT_LT(rd, kSweepTol) << "back: dataflow vs serial double reference";
                  EXPECT_GT(nd, 0u);
                  std::fprintf(stderr, "[GpuDataflow] %s %s %s back: rel diff %.3e\n",
                               kind ? "hub" : "plain", cfg.name, fp16 ? "fp16" : "fp32", rd); }
                for (int grid : {resident, 1, 7}) {
                    run_sweep(F, pb, true, fp16, d_tag, ++epoch, d_ctrl, grid, d_y, d_b);
                    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                    const auto [rd, nd] = rel_diff(z_df, dev_download(d_b, static_cast<size_t>(m)));
                    EXPECT_EQ(nd, 0u) << "back not deterministic at grid " << grid << " (rel diff " << rd << ")";
                }
                const std::vector<int> ctrl = dev_download(d_ctrl, 2);
                EXPECT_EQ(ctrl[0], 0); EXPECT_EQ(ctrl[1], 0);   // the last warp out resets them
            }
            cudaFree(d_tag);
            free_plan(pf); free_plan(pb);
        }
        for (void* p : {(void*)d_x, (void*)d_a, (void*)d_b, (void*)d_y, (void*)d_ctrl}) cudaFree(p);
    }
}

// PROTOCOL ISOLATION, the strongest single test of the segmentation: force
// S = 1 on every row the unsplit kernel would already walk with all 32 lanes
// (> 16*pre entries), and the split output must be BIT-IDENTICAL to the
// unsplit one -- same lane assignment, same entry order, the diagonal merely
// moved from the group butterfly into the finalizer's divide. Any difference
// is a protocol bug, not a rounding-order change.
TEST(GpuDataflow, ForcedSingleSegmentSplitIsBitIdenticalToTheUnsplitKernel) {
    const node_index n = 60001, m = n - 1;
    scoped_drop_env env("1e-4");
    const int pre = apxchol::dataflow_prefetch_depth();
    const sparse_csc L = make_hub_tail_lower(n);
    const df_factor F(L, m);
    const int resident = apxchol::dataflow_grid_size(false);
    std::vector<float> x(static_cast<size_t>(m));
    { std::mt19937 rng(23); std::uniform_real_distribution<float> u(-1.f, 1.f); for (auto& v : x) v = u(rng); }
    float* d_x = dev_upload(x.data(), static_cast<size_t>(m));
    float *d_a = nullptr, *d_b = nullptr;
    ASSERT_EQ(cudaMalloc(&d_a, static_cast<size_t>(m) * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, static_cast<size_t>(m) * 4), cudaSuccess);
    int* d_ctrl = nullptr;
    ASSERT_EQ(cudaMalloc(&d_ctrl, 2 * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_ctrl, 0, 2 * sizeof(int)), cudaSuccess);
    // seg larger than any row => S = 1; split_min = 16*pre is exactly the
    // G = 32 boundary, so only rows the unsplit kernel gives all 32 lanes are
    // split (splitting a shorter row is correct but re-associates its sum).
    const apxchol::cuda_host::dataflow_seg_params off{0, 0, 4096};
    const apxchol::cuda_host::dataflow_seg_params one{1 << 28, 16 * pre, 4096};
    for (bool reverse : {false, true}) {
        for (bool fp16 : {false, true}) {
            SCOPED_TRACE(std::string(reverse ? "back" : "forward") + (fp16 ? " fp16" : " fp32"));
            const auto pl_one = F.plan(reverse, one);
            int n_one = 0;
            for (const auto& sp : pl_one.spec) if (sp.row < 0) { EXPECT_EQ(sp.b, 1); ++n_one; }
            EXPECT_GT(n_one, 0) << "the hub factor must actually have rows over the threshold";
            dev_plan p0 = upload_plan(F.plan(reverse, off));
            dev_plan p1 = upload_plan(pl_one);
            const size_t words = static_cast<size_t>(m) + static_cast<size_t>(std::max(p0.slots, p1.slots));
            unsigned long long* d_tag = nullptr;
            ASSERT_EQ(cudaMalloc(&d_tag, words * sizeof(unsigned long long)), cudaSuccess);
            ASSERT_EQ(cudaMemset(d_tag, 0, words * sizeof(unsigned long long)), cudaSuccess);
            unsigned epoch = 0;
            run_sweep(F, p0, reverse, fp16, d_tag, ++epoch, d_ctrl, resident, d_x, d_a);
            run_sweep(F, p1, reverse, fp16, d_tag, ++epoch, d_ctrl, resident, d_x, d_b);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            const auto unsplit = dev_download(d_a, static_cast<size_t>(m));
            const auto split   = dev_download(d_b, static_cast<size_t>(m));
            const auto [rd, nd] = rel_diff(unsplit, split);
            EXPECT_EQ(nd, 0u) << "S = 1 split is not bit-identical to the unsplit kernel (rel diff " << rd
                              << ", " << n_one << " rows split)";
            cudaFree(d_tag);
            free_plan(p0); free_plan(p1);
        }
    }
    for (void* p : {(void*)d_x, (void*)d_a, (void*)d_b, (void*)d_ctrl}) cudaFree(p);
}

// The L L^T pair through cuda_sptrsv against the CPU backend's pair on the
// same factor: two independent implementations of the SAME contract (the CPU
// accumulates in double, so this is tolerance-level). Run on the hub-tail
// factor too, and with segmentation both off and on -- the two must agree
// with the CPU to the same tolerance and each be bit-identical run to run.
TEST(GpuDataflow, PairThroughCudaSptrsvMatchesTheCpuPairAndIsDeterministic) {
    const node_index n = 60001, m = n - 1;
    scoped_drop_env env("1e-4");
    for (int kind = 0; kind < 2; ++kind) {
        SCOPED_TRACE(kind ? "hub-tail factor" : "no-hub factor");
        const sparse_csc L = kind ? make_hub_tail_lower(n) : make_dominant_lower(n);
        std::vector<double> x(m), y_cpu(m), y_df(m), y_df2(m);
        { std::mt19937 rng(11); std::uniform_real_distribution<double> u(-1.0, 1.0); for (auto& v : x) v = u(rng); }
        // The CPU reference pair (fp32 storage, fp64 accumulation) -- one
        // setup, every GPU variant is compared against it. The storage is
        // pinned: the suite is also run with APXCHOL_SPTRSV_FP16=1 in the
        // environment, and the tolerances are stated against the fp32 pair.
        {
            scoped_env cpu32("APXCHOL_SPTRSV_FP16", "0");
            apxchol::omp_sptrsv c; c.setup(L, m);
            std::vector<double> t(m);
            c.forward_solve(x.data(), t.data());
            c.transpose_solve(t.data(), y_cpu.data());
        }
        for (const char* seg : {"0", "256"}) {
            for (bool fp16 : {false, true}) {
                SCOPED_TRACE(std::string("APXCHOL_GPU_DF_SPLIT=") + seg + (fp16 ? " fp16" : " fp32"));
                scoped_env f16("APXCHOL_SPTRSV_FP16", fp16 ? "1" : "0");
                scoped_env sg("APXCHOL_GPU_DF_SPLIT", seg);
                {
                    scoped_env be("APXCHOL_GPU_SPTRSV", "dataflow");
                    apxchol::cuda_sptrsv t; t.setup(L, m);
                    ASSERT_TRUE(t.dataflow());
                    EXPECT_STREQ(t.backend_name(), "dataflow");
                    EXPECT_EQ(t.fp16(), fp16);
                    EXPECT_GE(t.dataflow_grid(), 1);
                    t.solve_LLt(x.data(), y_df.data());
                    t.solve_LLt(x.data(), y_df2.data());
                }
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
                std::fprintf(stderr, "[GpuDataflow] %s split=%s %s pair: max rel diff vs the CPU pair %.3e\n",
                             kind ? "hub" : "plain", seg, fp16 ? "fp16" : "fp32", maxdiff / maxabs);
            }
        }
    }
}
#endif // APXCHOL_USE_CUDA
