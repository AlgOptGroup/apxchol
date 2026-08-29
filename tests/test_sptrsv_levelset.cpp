// Bit-identity contract of the ONE level-set kernel (omp_sptrsv::solve_levelset,
// include/apxchol/solver/sptrsv/omp.h), both directions:
//   * a solve is a pure per-row / per-column gather whose arithmetic does not
//     depend on which thread runs the row or when, so forward_solve and
//     transpose_solve return the SAME BYTES at every thread count (in place
//     and out of place) -- the thin `omp single` path, the fat `omp for` path
//     and the fat SIMD path (16-bit storage) included;
//   * one barrier per level (num_barriers() == the level count, both
//     directions);
//   * anchored to a correct solve: the T=1 result satisfies
//     the stored factor's triangular system to double roundoff (so the chain
//     of identities above is a chain of correct solves), stated through the
//     public storage contract (narrow_value / stored_diag / inv_scale) exactly
//     as the storage tests do.
// Two factor shapes: a "mixed" topological-level factor (fat random head,
// blocks of ~100 -> thin levels, a chain -> runs of size-1 levels; both kernel
// paths, long thin runs) and a round-structured all-fat factor through
// set_round_bounds (round-as-level path).
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
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
using apxchol::factor_value_t;
using apxchol::sptrsv_value_t;

namespace {

struct scoped_env {
    std::string name, saved; bool had = false;
    scoped_env(const char* var, const char* value) : name(var) {
        if (const char* e = std::getenv(var)) { had = true; saved = e; }
        if (value) setenv(var, value, 1); else unsetenv(var);
    }
    ~scoped_env() { if (had) setenv(name.c_str(), saved.c_str(), 1); else unsetenv(name.c_str()); }
};

sparse_csc from_cols(node_index m, std::vector<std::vector<node_index>>& col_rows, std::mt19937& rng) {
    std::uniform_real_distribution<double> uval(-0.5, 0.5);
    std::uniform_real_distribution<double> udiag(1.0, 3.0);
    sparse_csc L;
    L.n_ = m;
    L.outer_.assign(static_cast<size_t>(m) + 1, 0);
    for (node_index j = 0; j < m; ++j) {
        auto& r = col_rows[j];
        std::sort(r.begin(), r.end());
        r.erase(std::unique(r.begin(), r.end()), r.end());
        L.outer_[j + 1] = L.outer_[j] + static_cast<edge_index>(r.size());
    }
    L.inner_.resize(static_cast<size_t>(L.outer_[m]));
    L.vals_.resize(static_cast<size_t>(L.outer_[m]));
    for (node_index j = 0; j < m; ++j) {
        edge_index out = L.outer_[j];
        for (node_index r : col_rows[j]) {
            L.inner_[out] = r;
            L.vals_[out]  = static_cast<factor_value_t>(r == j ? udiag(rng) : uval(rng));
            ++out;
        }
    }
    return L;
}

// [0, mr): random lower entries (a fat head of topological levels, a few hub
// columns); then nblk blocks of 100 rows, each row depending on 3 rows of the
// previous block (levels of <= 100 rows); then a chain of `chain` rows, each
// depending on its predecessor (levels of 1 row). Every column starts with its
// diagonal.
sparse_csc make_mixed(node_index mr, node_index nblk, node_index chain, unsigned seed) {
    std::mt19937 rng(seed);
    const node_index m = mr + nblk * 100 + chain;
    std::vector<std::vector<node_index>> col_rows(m);
    for (node_index j = 0; j < m; ++j) col_rows[j].push_back(j);
    std::poisson_distribution<int> pcount(4.0);
    for (node_index j = 0; j < mr; ++j) {
        int k = pcount(rng);
        if (j % 9973 == 0) k += 500;
        if (j + 1 < mr) {
            std::uniform_int_distribution<node_index> urow(j + 1, mr - 1);
            for (int t = 0; t < k; ++t) col_rows[j].push_back(urow(rng));
        }
    }
    for (node_index b = 0; b < nblk; ++b)
        for (node_index q = 0; q < 100; ++q) {
            const node_index i  = mr + b * 100 + q;
            const node_index lo = (b == 0) ? mr - 2000 : mr + (b - 1) * 100;
            const node_index hi = (b == 0) ? mr - 1    : mr + b * 100 - 1;
            std::uniform_int_distribution<node_index> ucol(lo, hi);
            for (int t = 0; t < 3; ++t) col_rows[ucol(rng)].push_back(i);
        }
    for (node_index c = 0; c < chain; ++c) {
        const node_index i = mr + nblk * 100 + c;
        col_rows[i - 1].push_back(i);
        if (c % 7 == 0 && i >= 5) col_rows[i - 5].push_back(i);
    }
    return from_cols(m, col_rows, rng);
}

// R rounds of B mutually independent columns, every off-diagonal pointing to a
// LATER round (round-as-level through set_round_bounds; all levels fat when B >
// kSpTRSVOMPThreshold). Row lengths 0..48 plus a few long ones.
sparse_csc make_round(node_index R, node_index B, unsigned seed, std::vector<node_index>& bounds) {
    std::mt19937 rng(seed);
    const node_index m = R * B;
    std::uniform_int_distribution<int> ulen(0, 48);
    std::vector<std::vector<node_index>> col_rows(m);
    for (node_index j = 0; j < m; ++j) col_rows[j].push_back(j);
    for (node_index i = 0; i < m; ++i) {
        const node_index r = i / B;
        if (r == 0) continue;
        int len = ulen(rng);
        if (i % 1009 == 0) len += 200;
        std::uniform_int_distribution<node_index> ucol(0, r * B - 1);
        for (int t = 0; t < len; ++t) col_rows[ucol(rng)].push_back(i);
    }
    bounds.resize(static_cast<size_t>(R) + 1);
    for (node_index r = 0; r <= R; ++r) bounds[r] = r * B;
    return from_cols(m, col_rows, rng);
}

// Round-as-level factor with a genuinely mixed shape: a fat prefix followed by
// a long contiguous thin tail. This is the structural shape the hybrid path is
// meant to split, and mirrors elimination's large early IS rounds plus BK/peel
// tail without depending on a graph factorization in this unit test.
sparse_csc make_fat_prefix_thin_tail(
        node_index fat_rounds, node_index fat_width,
        node_index thin_rounds, node_index thin_width,
        unsigned seed, std::vector<node_index>& bounds) {
    std::mt19937 rng(seed);
    const node_index rounds = fat_rounds + thin_rounds;
    bounds.assign(static_cast<std::size_t>(rounds) + 1, 0);
    for (node_index r = 0; r < rounds; ++r)
        bounds[r + 1] = bounds[r] + (r < fat_rounds ? fat_width : thin_width);
    const node_index m = bounds.back();
    std::uniform_int_distribution<int> ulen(0, 24);
    std::vector<std::vector<node_index>> col_rows(m);
    for (node_index j = 0; j < m; ++j) col_rows[j].push_back(j);
    for (node_index r = 1; r < rounds; ++r) {
        std::uniform_int_distribution<node_index> ucol(0, bounds[r] - 1);
        for (node_index i = bounds[r]; i < bounds[r + 1]; ++i) {
            int len = ulen(rng);
            if (i % 1009 == 0) len += 100;
            for (int t = 0; t < len; ++t) col_rows[ucol(rng)].push_back(i);
        }
    }
    return from_cols(m, col_rows, rng);
}

std::vector<double> random_x(node_index m, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ux(-1.0, 1.0);
    std::vector<double> x(m);
    for (auto& v : x) v = ux(rng);
    return x;
}

// One solve pair at the current thread count: y = forward(x), z = back(y),
// plus the same in place (w = x; forward(w, w); back(w, w)).
struct pair_out { std::vector<double> y, z, w; };
pair_out solve_pair(const apxchol::omp_sptrsv& t, const std::vector<double>& x) {
    const node_index m = static_cast<node_index>(x.size());
    pair_out o; o.y.resize(m); o.z.resize(m); o.w = x;
    t.forward_solve(x.data(), o.y.data());
    t.transpose_solve(o.y.data(), o.z.data());
    t.forward_solve(o.w.data(), o.w.data());
    t.transpose_solve(o.w.data(), o.w.data());
    return o;
}
#ifdef _OPENMP
// setup() records a four-lane critical plan. Entering solve from an outer team
// while max_active_levels==1 forces its explicitly requested inner team to one
// lane, exercising the real team-mismatch fallback instead of merely changing
// omp_get_max_threads() (the inner region names num_threads(processors)).
pair_out solve_pair_with_forced_team_mismatch(
        const apxchol::omp_sptrsv& t, const std::vector<double>& x) {
    const int saved_levels = omp_get_max_active_levels();
    omp_set_max_active_levels(1);
    pair_out out;
    int outer_threads = 0;
    #pragma omp parallel num_threads(2) shared(out, outer_threads)
    {
        #pragma omp single
        {
            outer_threads = omp_get_num_threads();
            out = solve_pair(t, x);
        }
    }
    omp_set_max_active_levels(saved_levels);
    EXPECT_EQ(outer_threads, 2);
    return out;
}
#endif
void expect_same_bytes(const pair_out& a, const pair_out& b, const char* what) {
    const std::size_t n = a.y.size() * sizeof(double);
    EXPECT_EQ(0, std::memcmp(a.y.data(), b.y.data(), n)) << what << ": forward";
    EXPECT_EQ(0, std::memcmp(a.z.data(), b.z.data(), n)) << what << ": back";
    EXPECT_EQ(0, std::memcmp(a.w.data(), b.w.data(), n)) << what << ": in place";
}

// The stored factor L_s the kernels see (public storage contract): the
// off-diagonal (i, j) is widen(narrow_value(L_ij, s_j, .)), the diagonal
// stored_diag(L_jj, s_j) -- both in the column-scaled frame under FP16_SCALED.
// The pair contract: forward returns y' with L_s y' = x; back, given y',
// returns z with L_s^T z = R y', R = diag(inv_scale(s_j)^2). Residuals in
// double on the same stored values: only summation order differs.
void expect_correct_pair(const sparse_csc& L, const std::vector<double>& x, const pair_out& o) {
    const node_index m = L.rows();
    const auto* outer = L.outerIndexPtr(); const auto* inner = L.innerIndexPtr(); const auto* vals = L.valuePtr();
    std::vector<float> s(m);
    for (node_index j = 0; j < m; ++j) s[j] = apxchol::omp_sptrsv::column_scale(vals, outer[j], outer[j + 1]);
    std::vector<double> ry(m), rz(m);
    for (node_index j = 0; j < m; ++j) {
        ry[j] += apxchol::omp_sptrsv::stored_diag(vals[outer[j]], s[j]) * o.y[j];
        rz[j] += apxchol::omp_sptrsv::stored_diag(vals[outer[j]], s[j]) * o.z[j];
        for (edge_index p = outer[j] + 1; p < outer[j + 1]; ++p) {
            const double v = apxchol::widen(apxchol::omp_sptrsv::narrow_value(vals[p], s[j]));
            ry[inner[p]] += v * o.y[j];      // (L_s y')_i
            rz[j]        += v * o.z[inner[p]];   // (L_s^T z)_j
        }
    }
    double ey = 0.0, ez = 0.0, sy = 0.0, sz = 0.0;
    for (node_index j = 0; j < m; ++j) {
        const double r = apxchol::omp_sptrsv::inv_scale(s[j]);
        ey = std::max(ey, std::fabs(ry[j] - x[j]));
        ez = std::max(ez, std::fabs(rz[j] - o.y[j] * (r * r)));
        sy = std::max(sy, std::fabs(x[j]));
        sz = std::max(sz, std::fabs(o.y[j] * (r * r)));
    }
    EXPECT_LT(ey, 1e-10 * sy) << "forward: L_s y' = x";
    EXPECT_LT(ez, 1e-10 * sz) << "back: L_s^T z = R y'";
}

struct level_shape { std::size_t levels = 0, thin = 0, fat = 0, longest_thin_run = 0; };
level_shape shape(const apxchol::omp_sptrsv& t, bool fwd) {
    std::vector<int> sizes; std::vector<long long> work;
    t.level_stats(fwd, sizes, work);
    level_shape s; s.levels = sizes.size();
    std::size_t run = 0;
    for (int sz : sizes) {
        if (sz <= static_cast<int>(apxchol::kSpTRSVOMPThreshold)) { ++s.thin; ++run; s.longest_thin_run = std::max(s.longest_thin_run, run); }
        else { ++s.fat; run = 0; }
    }
    return s;
}

std::vector<int> thread_counts() {
#ifdef _OPENMP
    const int mx = omp_get_max_threads();
    std::vector<int> t = {1, 2, 3, 4};
    if (std::find(t.begin(), t.end(), mx) == t.end()) t.push_back(mx);
    return t;
#else
    return {1};
#endif
}
void set_threads(int t) {
#ifdef _OPENMP
    omp_set_num_threads(t);
#else
    (void)t;
#endif
}

} // namespace

TEST(SpTRSVRoundRanges, PreserveOldLevelSlotsAndRowOrderAtEmptyBoundaries) {
    using apxchol::detail::directional_round_levels;
    using apxchol::detail::round_level_ranges;

    // Five recorded rounds, including a leading, interior, and trailing empty
    // one, then three residual-peel rows. The old stable bucket fill produced
    // these exact level slots and kept ids ascending inside backward levels.
    const std::vector<node_index> bounds = {0, 0, 3, 3, 7, 7};
    ASSERT_TRUE(round_level_ranges::valid_metadata(bounds, 10));
    const directional_round_levels<true> fwd(bounds, 10);
    const directional_round_levels<false> bck(bounds, 10);
    ASSERT_EQ(fwd.size(), 8u);
    ASSERT_EQ(bck.size(), 8u);

    const std::vector<std::pair<node_index, node_index>> expected_fwd = {
        {0, 0}, {0, 3}, {3, 3}, {3, 7}, {7, 7},
        {7, 8}, {8, 9}, {9, 10},
    };
    std::vector<node_index> fwd_rows, bck_rows;
    for (std::size_t l = 0; l < fwd.size(); ++l) {
        const auto fr = fwd[l];
        EXPECT_EQ(std::pair(fr.first, fr.last), expected_fwd[l]);
        const auto br = bck[l];
        const auto expected_br = expected_fwd[expected_fwd.size() - 1 - l];
        EXPECT_EQ(std::pair(br.first, br.last), expected_br);
        for (std::size_t k = 0; k < fr.size(); ++k) fwd_rows.push_back(fr[k]);
        for (std::size_t k = 0; k < br.size(); ++k) bck_rows.push_back(br[k]);
    }
    EXPECT_EQ(fwd_rows,
              (std::vector<node_index>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
    EXPECT_EQ(bck_rows,
              (std::vector<node_index>{9, 8, 7, 3, 4, 5, 6, 0, 1, 2}));

    // With no peel, the historical max(row_round)+1 sizing omitted trailing
    // phantom rounds. Leading/interior empty ranges remain real barrier slots.
    const std::vector<node_index> trailing = {0, 0, 3, 7, 7, 7};
    const directional_round_levels<true> no_peel(trailing, 7);
    ASSERT_EQ(no_peel.size(), 3u);
    EXPECT_EQ(no_peel[0].size(), 0u);
    EXPECT_EQ(no_peel[1].size(), 3u);
    EXPECT_EQ(no_peel[2].size(), 4u);

    // The previous empty-input histogram initialized max depth to zero.
    const std::vector<node_index> empty = {0, 0, 0};
    const directional_round_levels<true> empty_factor(empty, 0);
    ASSERT_EQ(empty_factor.size(), 1u);
    EXPECT_TRUE(empty_factor[0].empty());
}

TEST(SpTRSVRoundRanges, ImplicitPeelIsByteEquivalentToMaterializedFallback) {
    scoped_env drop_off("APXCHOL_FACTOR_DROP", "0");
    scoped_env mode("APXCHOL_CPU_SPTRSV", "levels");
    scoped_env rounds_on("APXCHOL_ROUND_LEVELS", "1");
    set_threads(4);

    std::vector<node_index> logical_bounds;
    const sparse_csc L = make_round(3, 4, 8801, logical_bounds);
    ASSERT_EQ(logical_bounds,
              (std::vector<node_index>{0, 4, 8, 12}));
    // The first two logical rounds are recorded with phantom empty slots; the
    // final logical round is deliberately represented as a serial peel.
    const std::vector<node_index> bounds = {0, 0, 4, 4, 8, 8};
    const std::vector<int> expected_fwd = {0, 4, 0, 4, 0, 1, 1, 1, 1};
    const std::vector<int> expected_bck = {1, 1, 1, 1, 0, 4, 0, 4, 0};
    const auto x = random_x(L.rows(), 8802);

    for (const char* storage : {"0", "1"}) {
        if (storage[0] == '1' && !apxchol::omp_sptrsv::fp16_supported())
            continue;
        SCOPED_TRACE(std::string("APXCHOL_SPTRSV_FP16=") + storage);
        scoped_env storage_mode("APXCHOL_SPTRSV_FP16", storage);

        apxchol::omp_sptrsv implicit;
        implicit.set_round_bounds(bounds);
        implicit.setup(L, L.rows());

        // No round metadata takes the retained materialized topological path.
        apxchol::omp_sptrsv materialized;
        materialized.setup(L, L.rows());

        std::vector<int> sizes;
        std::vector<long long> work;
        implicit.level_stats(true, sizes, work);
        EXPECT_EQ(sizes, expected_fwd);
        implicit.level_stats(false, sizes, work);
        EXPECT_EQ(sizes, expected_bck);
        EXPECT_EQ(implicit.num_barriers(true), expected_fwd.size());
        EXPECT_EQ(implicit.num_barriers(false), expected_bck.size());
        EXPECT_LT(implicit.memory_bytes(), materialized.memory_bytes());

        const pair_out reference = solve_pair(materialized, x);
        if (storage[0] == '0') expect_correct_pair(L, x, reference);
        expect_same_bytes(reference, solve_pair(implicit, x),
                          "implicit ranges vs materialized fallback");
        set_threads(1);
        expect_same_bytes(reference, solve_pair(implicit, x),
                          "implicit ranges after runtime default-team change");
        set_threads(4);
    }
}

TEST(SpTRSVRoundRanges, PostSetupBoundsMutationDoesNotChangeActiveSchedule) {
    scoped_env fp32_storage("APXCHOL_SPTRSV_FP16", "0");
    scoped_env drop_off("APXCHOL_FACTOR_DROP", "0");
    scoped_env mode("APXCHOL_CPU_SPTRSV", "levels");
    scoped_env rounds_on("APXCHOL_ROUND_LEVELS", "1");
    set_threads(4);

    std::vector<node_index> bounds;
    const sparse_csc L = make_round(4, 32, 8803, bounds);
    const auto x = random_x(L.rows(), 8804);
    apxchol::omp_sptrsv trsv;
    trsv.set_round_bounds(bounds);
    trsv.setup(L, L.rows());
    const pair_out reference = solve_pair(trsv, x);
    std::vector<int> before_sizes, after_sizes;
    std::vector<long long> before_work, after_work;
    trsv.level_stats(true, before_sizes, before_work);

    // This is valid metadata but would collapse dependency-bearing rounds into
    // one unsafe level if the active schedule retained a view of setter state.
    trsv.set_round_bounds({0, L.rows()});
    trsv.level_stats(true, after_sizes, after_work);
    EXPECT_EQ(after_sizes, before_sizes);
    EXPECT_EQ(after_work, before_work);
    expect_same_bytes(reference, solve_pair(trsv, x),
                      "post-setup bounds mutation");
}

TEST(SpTRSVRoundRanges, EmptyFirstTailRangePreservesHybridBoundary) {
    scoped_env fp32_storage("APXCHOL_SPTRSV_FP16", "0");
    scoped_env drop_off("APXCHOL_FACTOR_DROP", "0");
    scoped_env rounds_on("APXCHOL_ROUND_LEVELS", "1");
    set_threads(4);

    std::vector<node_index> bounds;
    const sparse_csc L = make_fat_prefix_thin_tail(
        1, apxchol::kSpTRSVOMPThreshold + 8, 8, 8, 8811, bounds);
    ASSERT_GT(bounds.size(), 2u);
    bounds.insert(bounds.begin() + 2, bounds[1]);
    const auto x = random_x(L.rows(), 8812);

    apxchol::omp_sptrsv level_reference;
    {
        scoped_env mode("APXCHOL_CPU_SPTRSV", "levels");
        level_reference.set_round_bounds(bounds);
        level_reference.setup(L, L.rows());
    }
    const pair_out reference = solve_pair(level_reference, x);

    apxchol::omp_sptrsv hybrid;
    {
        scoped_env mode("APXCHOL_CPU_SPTRSV", "auto");
        hybrid.set_round_bounds(bounds);
        hybrid.setup(L, L.rows());
    }
    ASSERT_TRUE(hybrid.uses_hybrid_schedule());
    expect_same_bytes(reference, solve_pair(hybrid, x),
                      "empty first tail range");

#ifdef _OPENMP
    expect_same_bytes(reference,
                      solve_pair_with_forced_team_mismatch(hybrid, x),
                      "empty first tail range after runtime team mismatch");
#endif
}

// The same bytes at every thread count, both shapes, both directions, in and
// out of place; anchored to a correct solve at T=1.
TEST(SpTRSVLevelset, SameBytesAcrossThreadCounts) {
    // These state the DEFAULT (fp32) storage contract, so pin it: the suite
    // is also run with APXCHOL_SPTRSV_FP16=1 in the environment.
    scoped_env fp32_storage("APXCHOL_SPTRSV_FP16", "0");

    scoped_env drop_off("APXCHOL_FACTOR_DROP", "0");   // state the storage contract on the un-dropped factor
    const std::vector<int> T = thread_counts();
    const int t_max = T.back();

    {
        SCOPED_TRACE("mixed thin/fat topological levels");
        const sparse_csc L = make_mixed(60000, 60, 400, 12345);
        const node_index m = L.rows();
        apxchol::omp_sptrsv trsv;
        trsv.setup(L, m);
        for (bool fwd : {true, false}) {
            const level_shape s = shape(trsv, fwd);
            ASSERT_GT(s.fat, 3u) << "fwd=" << fwd;                 // both kernel paths run
            ASSERT_GT(s.thin, 100u) << "fwd=" << fwd;
            ASSERT_GT(s.longest_thin_run, 50u) << "fwd=" << fwd;   // a long single-thread tail of `omp single` levels
            EXPECT_EQ(trsv.num_barriers(fwd), s.levels);           // one barrier per level
        }
        const auto x = random_x(m, 1);
        set_threads(1);
        const pair_out ref = solve_pair(trsv, x);
        expect_correct_pair(L, x, ref);
        for (int t : T) {
            set_threads(t);
            expect_same_bytes(ref, solve_pair(trsv, x), ("T=" + std::to_string(t)).c_str());
        }
    }
    {
        SCOPED_TRACE("round-structured, all levels fat");
        std::vector<node_index> bounds;
        const sparse_csc L = make_round(8, 2 * apxchol::kSpTRSVOMPThreshold, 4321, bounds);
        const node_index m = L.rows();
        apxchol::omp_sptrsv trsv;
        trsv.set_round_bounds(bounds);
        trsv.setup(L, m);
        for (bool fwd : {true, false}) {
            const level_shape s = shape(trsv, fwd);
            ASSERT_EQ(s.levels, 8u); ASSERT_EQ(s.thin, 0u);
            EXPECT_EQ(trsv.num_barriers(fwd), 8u);
        }
        const auto x = random_x(m, 2);
        set_threads(1);
        const pair_out ref = solve_pair(trsv, x);
        expect_correct_pair(L, x, ref);
        for (int t : T) {
            set_threads(t);
            expect_same_bytes(ref, solve_pair(trsv, x), ("T=" + std::to_string(t)).c_str());
        }
    }
    set_threads(t_max);
}

TEST(CriticalSchedule, HybridRunsFatPrefixAndCriticalThinTailByteIdentically) {
    scoped_env fp32_storage("APXCHOL_SPTRSV_FP16", "0");
    scoped_env drop_off("APXCHOL_FACTOR_DROP", "0");
    std::vector<node_index> bounds;
    const sparse_csc L = make_fat_prefix_thin_tail(
        4, apxchol::kSpTRSVOMPThreshold + 76, 130, 8, 551, bounds);
    const auto x = random_x(L.rows(), 552);
    set_threads(4);

    apxchol::omp_sptrsv levels;
    {
        scoped_env mode("APXCHOL_CPU_SPTRSV", "levels");
        levels.set_round_bounds(bounds);
        levels.setup(L, L.rows());
    }
    apxchol::omp_sptrsv hybrid;
    {
        scoped_env mode("APXCHOL_CPU_SPTRSV", "auto");
        hybrid.set_round_bounds(bounds);
        hybrid.setup(L, L.rows());
    }
    ASSERT_TRUE(hybrid.uses_hybrid_schedule());
    const node_index first_tail_row =
        4 * (apxchol::kSpTRSVOMPThreshold + 76);
    const auto assignment = apxchol::detail::assign_critical_schedule(
        L.rows(), hybrid.csr_row_ptr().data(), hybrid.csr_col_idx().data(),
        4, first_tail_row);
    EXPECT_TRUE(apxchol::detail::critical_assignments_are_valid(
        assignment, L.rows(), hybrid.csr_row_ptr().data(),
        hybrid.csr_col_idx().data(), first_tail_row));
    EXPECT_LT(hybrid.num_barriers(true), levels.num_barriers(true));
    EXPECT_LT(hybrid.num_barriers(false), levels.num_barriers(false));
    const pair_out reference = solve_pair(levels, x);
    expect_same_bytes(reference, solve_pair(hybrid, x), "hybrid vs levels");

#ifdef _OPENMP
    // A true setup/runtime team mismatch must preserve both the bulk-prefix
    // and critical-tail arithmetic, including the fat-prefix template path.
    expect_same_bytes(reference,
                      solve_pair_with_forced_team_mismatch(hybrid, x),
                      "hybrid after forced runtime team mismatch");
#endif
}

TEST(CriticalSchedule, HybridPreservesFp16Arithmetic) {
    if constexpr (!apxchol::omp_sptrsv::fp16_supported()) return;
    scoped_env fp16_storage("APXCHOL_SPTRSV_FP16", "1");
    scoped_env drop_off("APXCHOL_FACTOR_DROP", "0");
    std::vector<node_index> bounds;
    const sparse_csc L = make_fat_prefix_thin_tail(
        2, apxchol::kSpTRSVOMPThreshold + 8, 24, 8, 557, bounds);
    const auto x = random_x(L.rows(), 558);
    set_threads(4);

    apxchol::omp_sptrsv levels;
    {
        scoped_env mode("APXCHOL_CPU_SPTRSV", "levels");
        levels.set_round_bounds(bounds);
        levels.setup(L, L.rows());
    }
    apxchol::omp_sptrsv hybrid;
    {
        scoped_env mode("APXCHOL_CPU_SPTRSV", "auto");
        hybrid.set_round_bounds(bounds);
        hybrid.setup(L, L.rows());
    }
    ASSERT_TRUE(hybrid.uses_hybrid_schedule());
    expect_same_bytes(solve_pair(levels, x), solve_pair(hybrid, x),
                      "fp16 hybrid vs levels");
}

TEST(CriticalSchedule, HybridLimitsArePureLevelsAndPureCritical) {
    scoped_env fp32_storage("APXCHOL_SPTRSV_FP16", "0");
    scoped_env drop_off("APXCHOL_FACTOR_DROP", "0");
    set_threads(4);

    {
        SCOPED_TRACE("all levels fat");
        std::vector<node_index> bounds;
        const sparse_csc L = make_round(
            6, apxchol::kSpTRSVOMPThreshold + 8, 553, bounds);
        const auto x = random_x(L.rows(), 554);
        apxchol::omp_sptrsv levels;
        {
            scoped_env mode("APXCHOL_CPU_SPTRSV", "levels");
            levels.set_round_bounds(bounds);
            levels.setup(L, L.rows());
        }
        apxchol::omp_sptrsv hybrid;
        {
            scoped_env mode("APXCHOL_CPU_SPTRSV", "auto");
            hybrid.set_round_bounds(bounds);
            hybrid.setup(L, L.rows());
        }
        EXPECT_FALSE(hybrid.uses_hybrid_schedule());
        EXPECT_EQ(hybrid.num_barriers(true), levels.num_barriers(true));
        expect_same_bytes(solve_pair(levels, x), solve_pair(hybrid, x),
                          "all-fat hybrid limit");
    }
    {
        SCOPED_TRACE("all levels thin");
        std::vector<node_index> bounds;
        const sparse_csc L = make_round(130, 8, 555, bounds);
        const auto x = random_x(L.rows(), 556);
        apxchol::omp_sptrsv levels;
        {
            scoped_env mode("APXCHOL_CPU_SPTRSV", "levels");
            levels.set_round_bounds(bounds);
            levels.setup(L, L.rows());
        }
        apxchol::omp_sptrsv hybrid;
        {
            scoped_env mode("APXCHOL_CPU_SPTRSV", "auto");
            hybrid.set_round_bounds(bounds);
            hybrid.setup(L, L.rows());
        }
        ASSERT_TRUE(hybrid.uses_hybrid_schedule());
        EXPECT_LT(hybrid.num_barriers(true), levels.num_barriers(true));
        expect_same_bytes(solve_pair(levels, x), solve_pair(hybrid, x),
                          "all-thin hybrid limit");
    }
}

TEST(CriticalSchedule, AutoUsesAnyThinSuffixWithRoundsAndMultipleThreads) {
    scoped_env fp32_storage("APXCHOL_SPTRSV_FP16", "0");
    scoped_env drop_off("APXCHOL_FACTOR_DROP", "0");
    scoped_env mode("APXCHOL_CPU_SPTRSV", "auto");
    std::vector<node_index> bounds;
    const sparse_csc L = make_round(13, 8, 111, bounds);

    set_threads(4);
    apxchol::omp_sptrsv parallel;
    parallel.set_round_bounds(bounds);
    parallel.setup(L, L.rows());
    EXPECT_TRUE(parallel.uses_hybrid_schedule());

    apxchol::omp_sptrsv without_rounds;
    without_rounds.setup(L, L.rows());
    EXPECT_FALSE(without_rounds.uses_hybrid_schedule());

    set_threads(1);
    apxchol::omp_sptrsv serial;
    serial.set_round_bounds(bounds);
    serial.setup(L, L.rows());
    EXPECT_FALSE(serial.uses_hybrid_schedule());
    set_threads(4);
}
