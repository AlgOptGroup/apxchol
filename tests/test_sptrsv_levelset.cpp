// Bit-identity contract of the ONE level-set kernel (omp_sptrsv::solve_levelset,
// include/apxchol/solver/sptrsv/omp.h), both directions:
//   * a solve is a pure per-row / per-column gather whose arithmetic does not
//     depend on which thread runs the row or when, so forward_solve and
//     transpose_solve return the SAME BYTES at every thread count (in place
//     and out of place) -- the thin `omp single` path, the fat `omp for` path
//     and the fat SIMD path (16-bit storage) included;
//   * the two schedule experiments -- APXCHOL_SPTRSV_BALANCE=nnz (fat levels
//     split by nnz, tables cached per team size and rebuilt inside a solve
//     when the team changes) and APXCHOL_SPTRSV_AGGLOMERATE=<K> (runs of thin
//     levels as one barrier-free superstep) -- change only the schedule, so
//     they return the SAME BYTES as the default at every thread count, and the
//     env is read at setup (reported by balance_nnz() / agglomerate());
//   * anchored to a correct solve: the default-schedule T=1 result satisfies
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
void expect_same_bytes(const pair_out& a, const pair_out& b, const char* what) {
    const std::size_t n = a.y.size() * sizeof(double);
    EXPECT_EQ(0, std::memcmp(a.y.data(), b.y.data(), n)) << what << ": forward";
    EXPECT_EQ(0, std::memcmp(a.z.data(), b.z.data(), n)) << what << ": back";
    EXPECT_EQ(0, std::memcmp(a.w.data(), b.w.data(), n)) << what << ": in place";
}

// The stored factor L_s the kernels see (public storage contract): the
// off-diagonal (i, j) is widen(narrow_value(L_ij, k, s_j, ., .)), the diagonal
// stored_diag(L_jj, s_j) -- both in the column-scaled frame under *_SCALED.
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
            const double v = apxchol::widen(apxchol::omp_sptrsv::narrow_value(vals[p], p, s[j], false, true));
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

// Default schedule: the same bytes at every thread count, both shapes, both
// directions, in and out of place; anchored to a correct solve at T=1.
TEST(SpTRSVLevelset, SameBytesAcrossThreadCounts) {
    scoped_env drop_off("APXCHOL_FACTOR_DROP", "0");   // state the storage contract on the un-dropped factor
    scoped_env no_bal("APXCHOL_SPTRSV_BALANCE", nullptr);
    scoped_env no_agg("APXCHOL_SPTRSV_AGGLOMERATE", nullptr);
    const std::vector<int> T = thread_counts();
    const int t_max = T.back();

    {
        SCOPED_TRACE("mixed thin/fat topological levels");
        const sparse_csc L = make_mixed(60000, 60, 400, 12345);
        const node_index m = L.rows();
        apxchol::omp_sptrsv trsv;
        trsv.setup(L, m);
        EXPECT_FALSE(trsv.balance_nnz());
        EXPECT_EQ(trsv.agglomerate(), -1);
        for (bool fwd : {true, false}) {
            const level_shape s = shape(trsv, fwd);
            ASSERT_GT(s.fat, 3u) << "fwd=" << fwd;                 // both kernel paths run
            ASSERT_GT(s.thin, 100u) << "fwd=" << fwd;
            ASSERT_GT(s.longest_thin_run, 50u) << "fwd=" << fwd;   // real supersteps for the agglomerate test
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

// The schedule experiments return the default's bytes at every thread count.
// BALANCE=nnz builds its tables at setup for omp_get_max_threads() and rebuilds
// them inside the solve for any other team (T = 1..4 here); AGGLOMERATE=0
// (whole thin runs) and =3 (capped supersteps) both cover the barrier-free
// multi-level `omp single`; the pair with both on covers their interaction.
TEST(SpTRSVLevelset, ScheduleExperimentsAreBitIdenticalToDefault) {
    scoped_env drop_off("APXCHOL_FACTOR_DROP", "0");
    const std::vector<int> T = thread_counts();
    const int t_max = T.back();
    set_threads(t_max);

    const sparse_csc L = make_mixed(60000, 60, 400, 12345);
    const node_index m = L.rows();
    const auto x = random_x(m, 1);
    apxchol::omp_sptrsv dflt;
    {
        scoped_env no_bal("APXCHOL_SPTRSV_BALANCE", nullptr);
        scoped_env no_agg("APXCHOL_SPTRSV_AGGLOMERATE", nullptr);
        dflt.setup(L, m);
    }
    std::vector<pair_out> ref;
    for (int t : T) { set_threads(t); ref.push_back(solve_pair(dflt, x)); }
    set_threads(t_max);

    struct mode { const char* bal; const char* agg; bool want_bal; int want_agg; };
    for (const mode& md : {mode{"nnz", nullptr, true, -1}, mode{nullptr, "0", false, 0},
                           mode{nullptr, "3", false, 3},   mode{"nnz", "0", true, 0}}) {
        const std::string tag = std::string("BALANCE=") + (md.bal ? md.bal : "<unset>") +
                                " AGGLOMERATE=" + (md.agg ? md.agg : "<unset>");
        SCOPED_TRACE(tag);
        apxchol::omp_sptrsv trsv;
        {
            scoped_env bal("APXCHOL_SPTRSV_BALANCE", md.bal);
            scoped_env agg("APXCHOL_SPTRSV_AGGLOMERATE", md.agg);
            trsv.setup(L, m);
        }
        EXPECT_EQ(trsv.balance_nnz(), md.want_bal);
        EXPECT_EQ(trsv.agglomerate(), md.want_agg);
        // Same level structure (the schedule does not touch the level sets);
        // the barrier count is what AGGLOMERATE changes: off = one per level,
        // 0 = one per fat level + one per run of thin levels, K = runs cut at K.
        for (bool fwd : {true, false}) {
            std::vector<int> a, b; std::vector<long long> wa, wb;
            dflt.level_stats(fwd, a, wa); trsv.level_stats(fwd, b, wb);
            ASSERT_EQ(a, b);
            const level_shape s = shape(dflt, fwd);
            ASSERT_EQ(dflt.num_barriers(fwd), s.levels);
            if (md.want_agg < 0)       EXPECT_EQ(trsv.num_barriers(fwd), s.levels);
            else if (md.want_agg == 0) { EXPECT_LT(trsv.num_barriers(fwd), s.fat + s.thin / 50); EXPECT_GE(trsv.num_barriers(fwd), s.fat + 1); }
            else                       { EXPECT_LT(trsv.num_barriers(fwd), s.levels); EXPECT_GE(trsv.num_barriers(fwd), s.fat + (s.thin + md.want_agg - 1) / md.want_agg); }
        }
        for (std::size_t i = 0; i < T.size(); ++i) {
            set_threads(T[i]);
            expect_same_bytes(ref[i], solve_pair(trsv, x), (tag + " T=" + std::to_string(T[i])).c_str());
        }
        // Back at the setup team size after the rebuilds (cache is per team size).
        set_threads(t_max);
        expect_same_bytes(ref.back(), solve_pair(trsv, x), (tag + " T=max after rebuilds").c_str());
    }
    set_threads(t_max);
}

// A malformed BALANCE value is ignored (default schedule), a negative
// AGGLOMERATE is off; a fresh setup re-reads both.
TEST(SpTRSVLevelset, EnvContract) {
    scoped_env drop_off("APXCHOL_FACTOR_DROP", "0");
    std::vector<node_index> bounds;
    const sparse_csc L = make_round(3, 2 * apxchol::kSpTRSVOMPThreshold, 99, bounds);
    const node_index m = L.rows();
    apxchol::omp_sptrsv trsv;
    trsv.set_round_bounds(bounds);
    {
        scoped_env bal("APXCHOL_SPTRSV_BALANCE", "bogus");
        scoped_env agg("APXCHOL_SPTRSV_AGGLOMERATE", "-5");
        trsv.setup(L, m);
        EXPECT_FALSE(trsv.balance_nnz());
        EXPECT_EQ(trsv.agglomerate(), -1);
    }
    {
        scoped_env bal("APXCHOL_SPTRSV_BALANCE", "nnz");
        scoped_env agg("APXCHOL_SPTRSV_AGGLOMERATE", "8");
        trsv.setup(L, m);
        EXPECT_TRUE(trsv.balance_nnz());
        EXPECT_EQ(trsv.agglomerate(), 8);
    }
    {
        scoped_env bal("APXCHOL_SPTRSV_BALANCE", "rows");
        scoped_env agg("APXCHOL_SPTRSV_AGGLOMERATE", "");
        trsv.setup(L, m);
        EXPECT_FALSE(trsv.balance_nnz());
        EXPECT_EQ(trsv.agglomerate(), -1);
    }
}
