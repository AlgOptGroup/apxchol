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
//     with APXCHOL_FACTOR_DROP_COMPENSATE=0 the reference is the plain
//     zeroed factor (a stored zero and an absent entry solve identically);
//   * the CSR/CSC the drop produces are byte-identical across thread counts;
//   * the env contract: unset = default ON at 1e-4, "0" / negative / junk =
//     off, any positive value overrides (rel > 1 keeps only the diagonal).
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "apxchol/sparse_csc.h"
#include "apxchol/solver/sptrsv/omp.h"

using apxchol::edge_index;
using apxchol::node_index;
using apxchol::sparse_csc;
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
struct scoped_compensate_env : scoped_env {
    explicit scoped_compensate_env(const char* value) : scoped_env("APXCHOL_FACTOR_DROP_COMPENSATE", value) {}
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
            L.vals_[out]  = static_cast<sptrsv_value_t>(r == j ? udiag(rng) : -uval(rng));
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
// With `compensate`, each column's dropped mass is folded back into its kept
// off-diagonals in proportion to |v| (v += dropped_sum * |v| / sum|kept|),
// which preserves the column sum. Written out here without
// omp_sptrsv::keep_offdiag / setup so predicate and compensation are stated
// twice independently.
struct dense_drop {
    sparse_csc    Lz;         // zeroed (and compensated) copy
    std::uint64_t kept   = 0; // diagonal + kept off-diagonals of L11 (== stored nnz after the drop)
    std::uint64_t zeroed = 0;
};

dense_drop reference_dense_drop(const sparse_csc& L, node_index m, double rel, bool compensate = true) {
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
        if (compensate && dropped_sum != 0.0 && kept_abs > 0.0) {
            const double per_abs = dropped_sum / kept_abs;
            for (edge_index p = L.outer_[j]; p < L.outer_[j + 1]; ++p) {
                if (L.inner_[p] >= m || L.inner_[p] == j || d.Lz.vals_[p] == 0) continue;
                const double v = static_cast<double>(d.Lz.vals_[p]);
                d.Lz.vals_[p] = static_cast<sptrsv_value_t>(v + per_abs * std::fabs(v));
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
            L.vals_[p] = static_cast<sptrsv_value_t>(-std::pow(10.0, uexp(rng)));
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
    struct cfg { node_index n; bool laplacian; };
    for (bool compensate : {true, false})
    for (cfg c : {cfg{3000, false}, cfg{3001, true}, cfg{60000, false}, cfg{60001, true}}) {
        const node_index m = c.laplacian ? c.n - 1 : c.n;
        SCOPED_TRACE("n=" + std::to_string(c.n) + (c.laplacian ? " (Laplacian m=n-1)" : " (SDDM m=n)") +
                     (compensate ? ", column sums preserved" : ", plain removal"));
        scoped_compensate_env comp_env(compensate ? nullptr : "0");
        const sparse_csc L = make_wide_magnitude_lower(c.n);
        const double rel = 1e-4;
        const dense_drop ref = reference_dense_drop(L, m, rel, compensate);
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
        EXPECT_EQ(st.compensate, compensate);
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
                double sum = static_cast<double>(trsv.csc_vals()[trsv.csc_col_ptr()[j]]);
                double abs = std::fabs(sum), dropped_mass = 0.0;
                for (edge_index p = trsv.csc_col_ptr()[j] + 1; p < trsv.csc_col_ptr()[j + 1]; ++p) {
                    const double v = static_cast<double>(trsv.csc_vals()[p]);
                    zeros += v == 0.0;
                    below += std::fabs(v) < rel * s[j];
                    sum += v; abs += std::fabs(v);
                }
                if (!compensate)
                    for (edge_index p = L.outer_[j] + 1; p < L.outer_[j + 1]; ++p)
                        if (L.inner_[p] < m && std::fabs(static_cast<double>(L.vals_[p])) < rel * s[j])
                            dropped_mass += static_cast<double>(L.vals_[p]);
                worst_cs = std::max(worst_cs, std::fabs(sum + dropped_mass - cs[j]) / abs);
            }
            for (node_index i = 0; i < m; ++i)
                ASSERT_EQ(trsv.csr_col_idx()[trsv.csr_row_ptr()[i + 1] - 1], i) << "diagonal last, row " << i;
            EXPECT_EQ(zeros, 0u);
            EXPECT_EQ(below, 0u);
            EXPECT_LT(worst_cs, sizeof(sptrsv_value_t) == 4 ? 1e-5 : 1e-13);
        }
        // The reference kept everything (drop OFF): its stored nnz is the full L11.
        EXPECT_EQ(rf.drop_stats().rel, 0.0);
        EXPECT_EQ(rf.drop_stats().nnz_stored, L11_nnz);
        EXPECT_EQ(rf.drop_stats().dropped, 0u);

        // Solves agree to ~1e-12 (same arithmetic up to the 4-way accumulator
        // grouping; the zeroed terms contribute exact zeros).
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
        EXPECT_LT(worst_f, 1e-12 * std::max(1.0, scale_f)) << "forward";
        EXPECT_LT(worst_b, 1e-12 * std::max(1.0, scale_b)) << "back";
        // And the compacted forward solve is an exact solve of the STORED L
        // (double accumulation): componentwise residual at roundoff.
        double worst = 0.0;
        for (node_index i = 0; i < m; ++i) {
            double r = x[i], sc = std::fabs(x[i]);
            for (edge_index q = trsv.csr_row_ptr()[i]; q < trsv.csr_row_ptr()[i + 1]; ++q) {
                const double t = static_cast<double>(trsv.csr_vals()[q]) * y1[trsv.csr_col_idx()[q]];
                r -= t; sc += std::fabs(t);
            }
            worst = std::max(worst, std::fabs(r) / (sc + 1e-300));
        }
        EXPECT_LT(worst, 1e-11);

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
    // Compensation: on unless APXCHOL_FACTOR_DROP_COMPENSATE=0.
    { scoped_compensate_env env(nullptr); EXPECT_TRUE(apxchol::omp_sptrsv::factor_drop_compensate_from_env()); }
    { scoped_compensate_env env("");      EXPECT_TRUE(apxchol::omp_sptrsv::factor_drop_compensate_from_env()); }
    { scoped_compensate_env env("1");     EXPECT_TRUE(apxchol::omp_sptrsv::factor_drop_compensate_from_env()); }
    { scoped_compensate_env env("0");     EXPECT_FALSE(apxchol::omp_sptrsv::factor_drop_compensate_from_env()); }
}

// rel = 0 / negative / junk = off; a rel below every entry drops nothing (and
// makes no copy: stored nnz == factor nnz); the diagonal survives ANY rel
// (even > 1, which drops every off-diagonal); the compacted factor solves as
// a diagonal one then (y = x / diag).
TEST(SpTRSVDrop, EdgeCases) {
    const node_index m = 2000;
    sparse_csc L = make_random_lower(m, 4.0, 31, 0.0, 1.0);
    for (edge_index p = 0; p < L.nonZeros(); ++p)
        if ((p % 5) == 2) L.vals_[p] = static_cast<sptrsv_value_t>(L.vals_[p] * 1e-9);   // far below 1e-4 * s_j
    std::uint64_t offdiag = 0;
    for (node_index j = 0; j < m; ++j) offdiag += L.outer_[j + 1] - L.outer_[j] - 1;
    const std::uint64_t nnz = static_cast<std::uint64_t>(L.nonZeros());
    ASSERT_GT(reference_dense_drop(L, m, 1e-4).zeroed, 100u);   // the default would remove plenty here
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
        scoped_drop_env env("1e-30");   // below everything: nothing goes
        apxchol::omp_sptrsv t; t.setup(L, m);
        EXPECT_EQ(t.drop_stats().rel, 1e-30);
        EXPECT_EQ(t.drop_stats().dropped, 0u);
        EXPECT_EQ(t.stored_nnz(), nnz);
        EXPECT_EQ(t.csc_col_ptr().back(), static_cast<edge_index>(nnz));
        EXPECT_EQ(0, std::memcmp(t.csc_vals().data(), L.vals_.data(), nnz * sizeof(sptrsv_value_t)));
        EXPECT_EQ(0, std::memcmp(t.csc_row_idx().data(), L.inner_.data(), nnz * sizeof(node_index)));
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
        // Solving with a diagonal factor: y = x / diag, both sweeps.
        std::vector<double> x(m, 1.0), y(m), z(m);
        t.forward_solve(x.data(), y.data());
        t.transpose_solve(x.data(), z.data());
        for (node_index j = 0; j < m; ++j) {
            ASSERT_NEAR(y[j], 1.0 / static_cast<double>(L.vals_[L.outer_[j]]), 1e-15) << j;
            ASSERT_NEAR(z[j], 1.0 / static_cast<double>(L.vals_[L.outer_[j]]), 1e-15) << j;
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
