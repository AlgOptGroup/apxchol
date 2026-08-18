// Storage-width-agnostic checks of the OpenMP SpTRSV kernels
// (include/apxchol/solver/sptrsv/omp.h): the compiled storage type matches
// the build flag, and the kernels compute in double from the WIDENED stored
// factor values (whatever sptrsv_value_t this build compiled: fp16-scaled /
// fp32 / fp64 -- see lowprec.h; fp16_t itself and the storage contract of
// setup() are tested in test_lowprec.cpp) -- thin-level (`omp single`) and
// fat-level (`omp for`, SIMD on 16-bit storage) kernels, both gather flavours.

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "apxchol/lowprec.h"
#include "apxchol/sparse_csc.h"
#include "apxchol/solver/sptrsv/omp.h"

using apxchol::edge_index;
using apxchol::node_index;
using apxchol::sparse_csc;
using apxchol::factor_value_t;
using apxchol::sptrsv_value_t;

namespace {

// RAII env override for tests that state a STORAGE contract on the un-dropped
// factor: the compacting drop (APXCHOL_FACTOR_DROP, ON by default at 1e-4 with
// column-sum compensation -- see omp.h) would remove / rescale the tiny
// entries these tests seed. Restores the previous value on scope exit.
struct scoped_env {
    std::string name, saved; bool had = false;
    scoped_env(const char* var, const char* value) : name(var) {
        if (const char* e = std::getenv(var)) { had = true; saved = e; }
        if (value) setenv(var, value, 1); else unsetenv(var);
    }
    ~scoped_env() { if (had) setenv(name.c_str(), saved.c_str(), 1); else unsetenv(name.c_str()); }
};
struct scoped_drop_off : scoped_env {
    scoped_drop_off() : scoped_env("APXCHOL_FACTOR_DROP", "0") {}
};

} // namespace

// The compiled storage width must match the build flag, and value_name /
// value_bytes / lowprec_variant (what the APXCHOL_VERBOSE banner prints) must
// agree with it.
TEST(SpTRSVKernels, SpTRSVValueTypeMatchesBuildFlag) {
#if defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED)
    EXPECT_EQ(sizeof(sptrsv_value_t), 2u);
    EXPECT_STREQ(apxchol::omp_sptrsv::value_name, "fp16 (per-column scaled)");
    EXPECT_STREQ(apxchol::omp_sptrsv::lowprec_variant, "FP16_SCALED");
    EXPECT_TRUE((std::is_same_v<sptrsv_value_t, apxchol::fp16_t>));
    EXPECT_TRUE((std::is_same_v<factor_value_t, float>));
#elif defined(APXCHOL_SPTRSV_FP32)
    EXPECT_EQ(sizeof(sptrsv_value_t), 4u);
    EXPECT_STREQ(apxchol::omp_sptrsv::value_name, "float (fp32)");
    EXPECT_STREQ(apxchol::omp_sptrsv::lowprec_variant, "OFF");
    EXPECT_TRUE((std::is_same_v<sptrsv_value_t, factor_value_t>));
#else
    EXPECT_EQ(sizeof(sptrsv_value_t), 8u);
    EXPECT_STREQ(apxchol::omp_sptrsv::value_name, "double (fp64)");
    EXPECT_STREQ(apxchol::omp_sptrsv::lowprec_variant, "OFF");
    EXPECT_TRUE((std::is_same_v<sptrsv_value_t, factor_value_t>));
#endif
    EXPECT_EQ(apxchol::omp_sptrsv::value_bytes, sizeof(sptrsv_value_t));
    // widen() of every storage type this build can see is the exact value.
    EXPECT_EQ(apxchol::widen(0.75f), 0.75);
    EXPECT_EQ(apxchol::widen(0.75), 0.75);
    EXPECT_EQ(apxchol::widen(apxchol::fp16_t(0.75f)), 0.75);
}

// ── SpTRSV kernels compute in double from the widened stored values ─────
// Build a random unit-ish lower-triangular factor, run forward (L y = x) and
// back (L^T z = y) solves through omp_sptrsv, and check the residuals against
// the values the SpTRSV STORES row by row with a componentwise bound. Because
// the kernels accumulate in double, the residual is at roundoff level even
// when the storage is 16-bit -- if any read did fp16/fp32 arithmetic this
// would blow up to ~2^-11. Under the FP16_SCALED build the stored values
// are: off-diagonals narrowed by omp_sptrsv::narrow_value (RNE of L_ij / s_j,
// s_j the per-column scale) and the DIAGONAL fp32 (omp_sptrsv::stored_diag:
// L_jj / s_j) -- so this test also pins (a) that the kernels divide by that
// diagonal (a narrow diagonal would show as a ~2^-11 residual), (b) that CSR
// (forward) and CSC (back) hold the SAME rounded value for every entry, and
// (c) the FP16_SCALED PAIR CONTRACT (omp.h, "FOLDED INTO THE VECTORS"): the
// kernels run on the stored L~ = L D^-1 with no scale multiplication --
// forward_solve returns y' = D y (y'_j = s_j y_j) and transpose_solve, given
// y', solves L~^T z = R y' with R = diag(r_j^2), r_j = inv_scale(s_j) =
// fp32(1/s_j) -- so the pair applies (L_s L_s^T)^-1 for the stored factor
// L_s = L~ D up to the 2^-23 of r_j. Checked (i) at roundoff against L~ / R
// (what the kernels compute) and (ii) y' == D y, z == L_s^-T y against a
// serial double substitution on L_s (roundoff for y', 2^-23-class for z).
namespace {

// The KERNEL matrix L~ as omp_sptrsv::setup stores it (widened, NOT
// rescaled) for the factor entry with value v in a column with scale s:
// off-diagonals through narrow_value, the diagonal through stored_diag. Under
// the fp32/fp64 builds this is v itself.
double kernel_value(factor_value_t v, bool is_diag, float s) {
    if (is_diag) return apxchol::omp_sptrsv::stored_diag(v, s);
    return apxchol::widen(apxchol::omp_sptrsv::narrow_value(v, s, /*fp16_flush_subnormal=*/true));
}
// The per-column scale the kernels' input / output carry (1.0f off FP16_SCALED).
float pair_scale(const sparse_csc& L, node_index j) {
#if defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED)
    return apxchol::omp_sptrsv::column_scale(L.vals_.data(), L.outer_[j], L.outer_[j + 1]);
#else
    (void)L; (void)j;
    return 1.0f;
#endif
}

sparse_csc make_random_lower(node_index m, double avg_offdiag, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uval(-0.5, 0.5);
    std::uniform_real_distribution<double> udiag(1.0, 3.0);
    std::poisson_distribution<int> pcount(avg_offdiag);
    sparse_csc L;
    L.n_ = m;
    L.outer_.assign(static_cast<size_t>(m) + 1, 0);
    std::vector<std::vector<node_index>> col_rows(m);
    for (node_index j = 0; j < m; ++j) {
        int k = pcount(rng);
        if (j % 997 == 0) k += 60;    // a few hub columns
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
            L.vals_[out]  = static_cast<factor_value_t>(r == j ? udiag(rng) : uval(rng));
            ++out;
        }
    }
    return L;
}

// Componentwise residual of L~ y = b (transpose == false) or L~^T z = b
// (transpose == true) against the KERNEL matrix L~ (kernel_value): what the
// kernels compute, at double roundoff.
void check_kernel_residual(const sparse_csc& L, const std::vector<double>& b,
                           const std::vector<double>& y, bool transpose) {
    const node_index m = L.rows();
    std::vector<double> r(b), scale(m, 0.0);
    for (node_index i = 0; i < m; ++i) scale[i] = std::fabs(b[i]);
    for (node_index j = 0; j < m; ++j) {
        const float s_j = apxchol::omp_sptrsv::column_scale(L.vals_.data(), L.outer_[j], L.outer_[j + 1]);
        for (edge_index p = L.outer_[j]; p < L.outer_[j + 1]; ++p) {
            const node_index i = L.inner_[p];
            const double v = kernel_value(L.vals_[p], /*is_diag=*/i == j, s_j);
            if (!transpose) { r[i] -= v * y[j]; scale[i] += std::fabs(v * y[j]); }
            else            { r[j] -= v * y[i]; scale[j] += std::fabs(v * y[i]); }
        }
    }
    double worst = 0.0;
    for (node_index i = 0; i < m; ++i)
        worst = std::max(worst, std::fabs(r[i]) / (scale[i] + 1e-300));
    // Double accumulation over <= ~70 terms: roundoff ~1e-14; 2^-11 would be
    // the signature of any narrow-precision arithmetic.
    EXPECT_LT(worst, 1e-11) << (transpose ? "back" : "forward") << " kernel residual";
}

// Serial double reference solves on the STORED factor L_s = L~ D (kernel
// off-diagonal * s_j, kernel diagonal * s_j): L_s y = x, then L_s^T z = y.
void reference_pair(const sparse_csc& L, const std::vector<double>& x,
                    std::vector<double>& y, std::vector<double>& z) {
    const node_index m = L.rows();
    // Column-oriented forward substitution.
    y = x;
    for (node_index j = 0; j < m; ++j) {
        const float s_j = apxchol::omp_sptrsv::column_scale(L.vals_.data(), L.outer_[j], L.outer_[j + 1]);
        const double sd = static_cast<double>(pair_scale(L, j));
        const edge_index p0 = L.outer_[j];
        y[j] /= kernel_value(L.vals_[p0], true, s_j) * sd;
        for (edge_index p = p0 + 1; p < L.outer_[j + 1]; ++p)
            y[L.inner_[p]] -= kernel_value(L.vals_[p], false, s_j) * sd * y[j];
    }
    // Column-oriented back substitution (L_s^T z = y: column j of L_s is row j of L_s^T).
    z.assign(m, 0.0);
    for (node_index jj = m; jj-- > 0; ) {
        const node_index j = jj;
        const float s_j = apxchol::omp_sptrsv::column_scale(L.vals_.data(), L.outer_[j], L.outer_[j + 1]);
        const double sd = static_cast<double>(pair_scale(L, j));
        const edge_index p0 = L.outer_[j];
        double acc = y[j];
        for (edge_index p = p0 + 1; p < L.outer_[j + 1]; ++p)
            acc -= kernel_value(L.vals_[p], false, s_j) * sd * z[L.inner_[p]];
        z[j] = acc / (kernel_value(L.vals_[p0], true, s_j) * sd);
    }
}

void run_kernel_precision_check() {
    for (node_index m : {node_index(3000), node_index(60000) /* parallel transpose path */}) {
        SCOPED_TRACE("m=" + std::to_string(m));
        sparse_csc L = make_random_lower(m, 4.0, 99);
        apxchol::omp_sptrsv trsv;
        trsv.setup(L, m);
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> ux(-1.0, 1.0);
        std::vector<double> x(m), yp(m), z(m);
        for (auto& v : x) v = ux(rng);
        // (i) What the kernels compute, at roundoff: L~ y' = x, then
        //     L~^T z = R y'.
        trsv.forward_solve(x.data(), yp.data());
        check_kernel_residual(L, x, yp, /*transpose=*/false);
        trsv.transpose_solve(yp.data(), z.data());
        std::vector<double> ryp(m);
        for (node_index j = 0; j < m; ++j) {
            const double r = apxchol::omp_sptrsv::inv_scale(pair_scale(L, j));
            ryp[j] = yp[j] * (r * r);
        }
        check_kernel_residual(L, ryp, z, /*transpose=*/true);
        // (ii) The pair contract against the serial reference on L_s = L~ D:
        //      y' = D y at roundoff, z = L_s^-T y up to r_j = fp32(1/s_j)
        //      (2^-23 relative on the input, i.e. on z; exact off FP16_SCALED).
        std::vector<double> y_ref, z_ref;
        reference_pair(L, x, y_ref, z_ref);
        double worst_y = 0.0, worst_z = 0.0, sc_y = 0.0, sc_z = 0.0;
        for (node_index j = 0; j < m; ++j) {
            worst_y = std::max(worst_y, std::fabs(yp[j] - static_cast<double>(pair_scale(L, j)) * y_ref[j]));
            worst_z = std::max(worst_z, std::fabs(z[j] - z_ref[j]));
            sc_y = std::max(sc_y, std::fabs(yp[j]));
            sc_z = std::max(sc_z, std::fabs(z_ref[j]));
        }
        EXPECT_LT(worst_y, 1e-10 * sc_y) << "y' = D y";
#if defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED)
        EXPECT_LT(worst_z, 1e-5 * sc_z) << "z (r_j = fp32(1/s_j))";
#else
        EXPECT_LT(worst_z, 1e-10 * sc_z) << "z";
#endif
    }
}

} // namespace

TEST(SpTRSVKernels, SpTRSVKernelsComputeInDoubleFromWidenedStorage) {
    scoped_drop_off drop_off;   // storage contract on the un-dropped factor
    run_kernel_precision_check();
}

// ── Fat-level kernels (the `omp for` paths) ─────────────────────────────
// A round-structured factor (R rounds of B > kSpTRSVOMPThreshold mutually
// independent columns; every off-diagonal points to a LATER round) fed
// through set_round_bounds so every level is fat and the `omp for` kernels
// run -- on 16-bit storage the SIMD ones (simd_dot()): 8-wide vector widen,
// 4-wide step, scalar tail, in both gather flavours (APXCHOL_FP16_GATHER=
// simd | scalar, read at every setup). CSR row lengths are spread over
// 0..48 so every path (8-blocks, the 4-step, tails of 0..3) is exercised.
// Checked exactly like the thin-level kernels: the pair contract at roundoff
// against L~ / R and against the serial reference on L_s, plus the two
// gather flavours agreeing to roundoff.
namespace {

sparse_csc make_round_structured_lower(node_index R, node_index B, unsigned seed,
                                       std::vector<node_index>& round_bounds) {
    const node_index m = R * B;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uval(-0.5, 0.5);
    std::uniform_real_distribution<double> udiag(1.0, 3.0);
    std::uniform_int_distribution<int> ulen(0, 48);
    // Row i (round r = i / B) picks len_i distinct columns from rounds < r.
    std::vector<std::vector<node_index>> col_rows(m);
    for (node_index j = 0; j < m; ++j) col_rows[j].push_back(j);   // diagonal
    for (node_index i = 0; i < m; ++i) {
        const node_index r = i / B;
        if (r == 0) continue;
        int len = ulen(rng);
        if (i % 1009 == 0) len += 200;                        // a few very long rows
        std::uniform_int_distribution<node_index> ucol(0, r * B - 1);
        std::vector<node_index> cols;
        for (int t = 0; t < len; ++t) cols.push_back(ucol(rng));
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
        for (node_index j : cols) col_rows[j].push_back(i);
    }
    sparse_csc L;
    L.n_ = m;
    L.outer_.assign(static_cast<size_t>(m) + 1, 0);
    for (node_index j = 0; j < m; ++j) {
        std::sort(col_rows[j].begin(), col_rows[j].end());
        L.outer_[j + 1] = L.outer_[j] + static_cast<edge_index>(col_rows[j].size());
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
    round_bounds.resize(static_cast<size_t>(R) + 1);
    for (node_index r = 0; r <= R; ++r) round_bounds[r] = r * B;
    return L;
}

} // namespace

TEST(SpTRSVKernels, SpTRSVFatLevelKernelsBothGatherFlavours) {
    scoped_drop_off drop_off;   // storage contract on the un-dropped factor
    const node_index R = 8, B = 2 * apxchol::kSpTRSVOMPThreshold;   // 8 fat levels of 2048
    std::vector<node_index> bounds;
    const sparse_csc L = make_round_structured_lower(R, B, 4321, bounds);
    const node_index m = R * B;
    // Row-length spectrum has teeth: rows of every tail length and 8-blocks.
    {
        std::vector<int> hist(4, 0); int ge8 = 0, ge4 = 0;
        std::vector<int> rowlen(m, 0);
        for (node_index j = 0; j < m; ++j)
            for (edge_index p = L.outer_[j] + 1; p < L.outer_[j + 1]; ++p) rowlen[L.inner_[p]]++;
        for (node_index i = 0; i < m; ++i) { hist[rowlen[i] % 4]++; ge8 += rowlen[i] >= 8; ge4 += rowlen[i] >= 4; }
        for (int t = 0; t < 4; ++t) ASSERT_GT(hist[t], 100) << "tail length " << t;
        ASSERT_GT(ge8, static_cast<int>(m / 4));
        ASSERT_GT(ge4, static_cast<int>(m / 2));
    }
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> ux(-1.0, 1.0);
    std::vector<double> x(m);
    for (auto& v : x) v = ux(rng);
    std::vector<double> y_ref, z_ref;
    reference_pair(L, x, y_ref, z_ref);

    std::vector<std::vector<double>> yps, zs;
    for (const char* mode : {"simd", "scalar"}) {
        SCOPED_TRACE(std::string("APXCHOL_FP16_GATHER=") + mode);
        setenv("APXCHOL_FP16_GATHER", mode, 1);
        apxchol::omp_sptrsv trsv;
        trsv.set_round_bounds(bounds);
        trsv.setup(L, m);
        unsetenv("APXCHOL_FP16_GATHER");
        EXPECT_EQ(trsv.fat_gather_simd(), std::string(mode) == "simd");
#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
        EXPECT_EQ(apxchol::omp_sptrsv::simd_dot(), sizeof(sptrsv_value_t) == 2);
#else
        EXPECT_FALSE(apxchol::omp_sptrsv::simd_dot());
#endif
        // Every level is fat (round-as-level).
        std::vector<int> sizes; std::vector<long long> work;
        trsv.level_stats(/*fwd=*/true, sizes, work);
        ASSERT_EQ(sizes.size(), static_cast<size_t>(R));
        for (int s : sizes) ASSERT_EQ(s, static_cast<int>(B));
        trsv.level_stats(/*fwd=*/false, sizes, work);
        ASSERT_EQ(sizes.size(), static_cast<size_t>(R));
        for (int s : sizes) ASSERT_EQ(s, static_cast<int>(B));

        std::vector<double> yp(m), z(m);
        trsv.forward_solve(x.data(), yp.data());
        check_kernel_residual(L, x, yp, /*transpose=*/false);
        trsv.transpose_solve(yp.data(), z.data());
        std::vector<double> ryp(m);
        for (node_index j = 0; j < m; ++j) {
            const double r = apxchol::omp_sptrsv::inv_scale(pair_scale(L, j));
            ryp[j] = yp[j] * (r * r);
        }
        check_kernel_residual(L, ryp, z,  /*transpose=*/true);
        double worst_y = 0.0, worst_z = 0.0, sc_y = 0.0, sc_z = 0.0;
        for (node_index j = 0; j < m; ++j) {
            worst_y = std::max(worst_y, std::fabs(yp[j] - static_cast<double>(pair_scale(L, j)) * y_ref[j]));
            worst_z = std::max(worst_z, std::fabs(z[j] - z_ref[j]));
            sc_y = std::max(sc_y, std::fabs(yp[j]));
            sc_z = std::max(sc_z, std::fabs(z_ref[j]));
        }
        EXPECT_LT(worst_y, 1e-10 * sc_y) << "y' = D y";
#if defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED)
        EXPECT_LT(worst_z, 1e-5 * sc_z) << "z (r_j = fp32(1/s_j))";
#else
        EXPECT_LT(worst_z, 1e-10 * sc_z) << "z";
#endif
        yps.push_back(yp); zs.push_back(z);
    }
    // The two gather flavours (different summation order) agree to roundoff.
    double dy = 0.0, dz = 0.0, sy = 0.0, sz = 0.0;
    for (node_index j = 0; j < m; ++j) {
        dy  = std::max(dy,  std::fabs(yps[0][j] - yps[1][j]));
        dz  = std::max(dz,  std::fabs(zs[0][j]  - zs[1][j]));
        sy  = std::max(sy,  std::fabs(yps[0][j]));
        sz  = std::max(sz,  std::fabs(zs[0][j]));
    }
    EXPECT_LT(dy,  1e-12 * sy);
    EXPECT_LT(dz,  1e-12 * sz);
}
