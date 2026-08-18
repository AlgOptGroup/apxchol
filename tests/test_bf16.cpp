// bf16_t (include/apxchol/bf16.h): round-trip / rounding / widening tests, plus a
// storage-width-agnostic check that the OpenMP SpTRSV kernels compute in double
// from the WIDENED stored factor values (whatever sptrsv_value_t this build
// compiled: bf16 / bf16-scaled / fp16-scaled / fp24 / fp32 / fp64 -- see
// lowprec.h; fp16_t / fp24_t have their own tests in test_lowprec.cpp).

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

#include "apxchol/bf16.h"
#include "apxchol/sparse_csc.h"
#include "apxchol/solver/sptrsv/omp.h"

using apxchol::bf16_t;
using apxchol::edge_index;
using apxchol::node_index;
using apxchol::sparse_csc;
using apxchol::factor_value_t;
using apxchol::sptrsv_value_t;

namespace {

std::uint32_t f2u(float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; }
float         u2f(std::uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

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

// ── Round-trip bound ───────────────────────────────────
// bf16 keeps 8 significant bits (7 explicit + hidden), so RNE guarantees
// |from_float(to_bf16(x)) - x| <= 2^-8 |x| (half-ulp = 2^-9 relative to the
// power of two below |x|, i.e. at most 2^-8 relative to x itself).
TEST(BF16, RoundTripWithin2PowMinus8Relative) {
    std::mt19937_64 rng(2026);
    // Uniform in the exponent so tiny and huge magnitudes are both covered
    // (normal range only: exponents -120..+120).
    std::uniform_real_distribution<double> uexp(-120.0, 120.0);
    std::uniform_real_distribution<double> umant(1.0, 2.0);
    std::bernoulli_distribution sign(0.5);
    const double bound = std::ldexp(1.0, -8);
    double worst = 0.0;
    for (int t = 0; t < 2'000'000; ++t) {
        const float x = static_cast<float>(
            (sign(rng) ? -1.0 : 1.0) * std::ldexp(umant(rng), static_cast<int>(uexp(rng))));
        const float y = apxchol::to_float(apxchol::from_float(x));
        const double rel = std::fabs(static_cast<double>(y) - static_cast<double>(x)) /
                           std::fabs(static_cast<double>(x));
        worst = std::max(worst, rel);
        ASSERT_LE(rel, bound) << "x=" << x << " y=" << y;
        // widen() must agree with to_float() and never lose the sign.
        ASSERT_EQ(apxchol::widen(apxchol::from_float(x)), static_cast<double>(y));
        ASSERT_EQ(std::signbit(y), std::signbit(x));
    }
    // The bound is tight-ish: RNE must actually get close to it somewhere.
    EXPECT_GT(worst, bound * 0.9);
}

// Values with <= 8 significant bits are exactly representable, so they must
// round-trip bit-for-bit, and so must the special zeros.
TEST(BF16, ExactValuesRoundTripExactly) {
    const float exact[] = {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 1.5f, -3.25f, 255.0f,
                           -127.5f, 2.5f, std::ldexp(1.0f, -100),
                           std::ldexp(1.0f, 100), 1.0078125f /* 1 + 2^-7 */};
    for (float x : exact) {
        const bf16_t h = apxchol::from_float(x);
        EXPECT_EQ(f2u(apxchol::to_float(h)), f2u(x)) << "x=" << x;
        EXPECT_EQ(h.bits, static_cast<std::uint16_t>(f2u(x) >> 16));
    }
}

// Round-to-nearest-even at the exact ties, and the mantissa carry into the
// exponent.
TEST(BF16, RoundToNearestEvenAndCarry) {
    // 1 + 2^-8 is exactly halfway between 1 (mantissa ...0000000, even) and
    // 1 + 2^-7 (mantissa ...0000001, odd) -> ties to even -> 1.0.
    EXPECT_EQ(apxchol::to_float(apxchol::from_float(1.0f + std::ldexp(1.0f, -8))), 1.0f);
    // 1 + 3*2^-8 is halfway between 1 + 2^-7 (odd) and 1 + 2^-6 (even) -> 1 + 2^-6.
    EXPECT_EQ(apxchol::to_float(apxchol::from_float(1.0f + 3.0f * std::ldexp(1.0f, -8))),
              1.0f + std::ldexp(1.0f, -6));
    // Just above / below the tie go to the nearest.
    EXPECT_EQ(apxchol::to_float(apxchol::from_float(u2f(0x3f808001u))), 1.0f + std::ldexp(1.0f, -7));
    EXPECT_EQ(apxchol::to_float(apxchol::from_float(u2f(0x3f807fffu))), 1.0f);
    // Carry: the largest float below 1 (0x3f7fffff = 1 - 2^-24) rounds UP to
    // 1.0f (0x3f80), which is a carry out of the mantissa into the exponent.
    EXPECT_EQ(apxchol::from_float(u2f(0x3f7fffffu)).bits, 0x3f80u);
    // 1.99999988f (0x3fffffff) -> 2.0f (0x4000): carry crosses a binade.
    EXPECT_EQ(apxchol::from_float(u2f(0x3fffffffu)).bits, 0x4000u);
    // Negative mirror of the carry.
    EXPECT_EQ(apxchol::from_float(u2f(0xbf7fffffu)).bits, 0xbf80u);
    // Values within half a bf16 ulp of FLT_MAX overflow to +inf under RNE
    // (correct IEEE behaviour; not expected in a factor, but must not wrap).
    EXPECT_TRUE(std::isinf(apxchol::to_float(apxchol::from_float(std::numeric_limits<float>::max()))));
    // NaN stays NaN (never turns into inf).
    EXPECT_TRUE(std::isnan(apxchol::to_float(apxchol::from_float(std::numeric_limits<float>::quiet_NaN()))));
    // inf stays inf.
    EXPECT_TRUE(std::isinf(apxchol::to_float(apxchol::from_float(std::numeric_limits<float>::infinity()))));
}

// Assignment from double / float / int narrows through the same RNE path
// (this is what omp_sptrsv::setup's CSC copy does with the fp32 factor value).
TEST(BF16, ConvertingConstructorFromDoubleAndInt) {
    const bf16_t from_d = 3.14159265358979;
    const bf16_t from_f = 3.14159265f;
    EXPECT_EQ(from_d.bits, from_f.bits);
    EXPECT_NEAR(static_cast<double>(from_d), 3.14159265358979, std::ldexp(1.0, -8) * 3.2);
    const bf16_t from_i = 7;
    EXPECT_EQ(apxchol::to_float(from_i), 7.0f);
    // static_cast<sptrsv_value_t>(double) -- the transpose's cast form -- and
    // widen() of every storage type must be consistent.
    EXPECT_EQ(apxchol::widen(static_cast<bf16_t>(0.75)), 0.75);
    EXPECT_EQ(apxchol::widen(0.75f), 0.75);
    EXPECT_EQ(apxchol::widen(0.75), 0.75);
}

// The compiled storage width must match the build flag, and value_name /
// value_bytes / lowprec_variant (what the APXCHOL_VERBOSE banner prints) must
// agree with it.
TEST(BF16, SpTRSVValueTypeMatchesBuildFlag) {
#if defined(APXCHOL_SPTRSV_LOWPREC_BF16)
    EXPECT_EQ(sizeof(sptrsv_value_t), 2u);
    EXPECT_STREQ(apxchol::omp_sptrsv::value_name, "bf16");
    EXPECT_STREQ(apxchol::omp_sptrsv::lowprec_variant, "BF16");
    EXPECT_TRUE((std::is_same_v<sptrsv_value_t, bf16_t>));
    EXPECT_TRUE((std::is_same_v<factor_value_t, float>));
#elif defined(APXCHOL_SPTRSV_LOWPREC_BF16_SCALED)
    EXPECT_EQ(sizeof(sptrsv_value_t), 2u);
    EXPECT_STREQ(apxchol::omp_sptrsv::value_name, "bf16 (per-column scaled)");
    EXPECT_STREQ(apxchol::omp_sptrsv::lowprec_variant, "BF16_SCALED");
    EXPECT_TRUE((std::is_same_v<sptrsv_value_t, bf16_t>));
    EXPECT_TRUE((std::is_same_v<factor_value_t, float>));
#elif defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED)
    EXPECT_EQ(sizeof(sptrsv_value_t), 2u);
    EXPECT_STREQ(apxchol::omp_sptrsv::value_name, "fp16 (per-column scaled)");
    EXPECT_STREQ(apxchol::omp_sptrsv::lowprec_variant, "FP16_SCALED");
    EXPECT_TRUE((std::is_same_v<sptrsv_value_t, apxchol::fp16_t>));
    EXPECT_TRUE((std::is_same_v<factor_value_t, float>));
#elif defined(APXCHOL_SPTRSV_LOWPREC_FP24)
    EXPECT_EQ(sizeof(sptrsv_value_t), 3u);
    EXPECT_STREQ(apxchol::omp_sptrsv::value_name, "fp24");
    EXPECT_STREQ(apxchol::omp_sptrsv::lowprec_variant, "FP24");
    EXPECT_TRUE((std::is_same_v<sptrsv_value_t, apxchol::fp24_t>));
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
}

// ── Stochastic rounding (bf16.h: from_float_stochastic) ────────────────
// Per entry: the result is one of the two bf16 neighbours of x (truncation or
// truncation + 1 ulp), so |y - x| < 2^-7 |x|; exact values stay exact; the
// threshold is a pure function of the index (deterministic); and over many
// independent indices the mean rounding error is ~0 (unbiased), which is the
// whole point versus RNE.
TEST(BF16, StochasticRoundingBoundedDeterministicUnbiased) {
    std::mt19937_64 rng(4242);
    std::uniform_real_distribution<double> uexp(-60.0, 60.0);
    std::uniform_real_distribution<double> umant(1.0, 2.0);
    std::bernoulli_distribution sign(0.5);
    const double ulp_rel_bound = std::ldexp(1.0, -7);
    for (int t = 0; t < 200'000; ++t) {
        const float x = static_cast<float>(
            (sign(rng) ? -1.0 : 1.0) * std::ldexp(umant(rng), static_cast<int>(uexp(rng))));
        const std::uint64_t idx = rng();
        const bf16_t h  = apxchol::from_float_stochastic(x, idx);
        const bf16_t h2 = apxchol::from_float_stochastic(x, idx);
        ASSERT_EQ(h.bits, h2.bits) << "not deterministic in idx";
        const float y = apxchol::to_float(h);
        const double rel = std::fabs(static_cast<double>(y) - static_cast<double>(x)) /
                           std::fabs(static_cast<double>(x));
        ASSERT_LT(rel, ulp_rel_bound) << "x=" << x << " y=" << y;
        // One of the two neighbours: truncation, or truncation + 1 ulp (with carry).
        const std::uint16_t trunc = static_cast<std::uint16_t>(f2u(x) >> 16);
        ASSERT_TRUE(h.bits == trunc || h.bits == static_cast<std::uint16_t>(trunc + 1))
            << "x=" << x;
        ASSERT_EQ(std::signbit(y), std::signbit(x));
    }
    // Exact bf16 values never move, whatever the index.
    for (float x : {0.0f, 1.0f, -1.5f, 255.0f, std::ldexp(1.0f, -40)})
        for (std::uint64_t idx = 0; idx < 64; ++idx)
            EXPECT_EQ(f2u(apxchol::to_float(apxchol::from_float_stochastic(x, idx))), f2u(x));
    // Carry into the exponent: 1 - 2^-24 rounds UP to 1.0f for a threshold
    // >= 1 (r = 0xffff), and stays truncated (0x3f7f) only for t == 0.
    {
        const float below1 = u2f(0x3f7fffffu);
        int ups = 0;
        for (std::uint64_t idx = 0; idx < 4096; ++idx)
            ups += apxchol::from_float_stochastic(below1, idx).bits == 0x3f80u;
        EXPECT_GE(ups, 4090);   // P(t == 0) = 2^-16 per index
    }
    // Unbiased: for a fixed x whose discarded fraction is r/2^16, the mean of
    // (y - x) over many indices must be ~0 (RNE would give a fixed nonzero
    // error for every one of them). Sample the fraction at 1/4, 1/2, 3/4.
    for (std::uint32_t frac : {0x4000u, 0x8000u, 0xc000u}) {
        const float x = u2f(0x3f800000u | frac);      // 1 + frac * 2^-23
        const double ulp = std::ldexp(1.0, -7);
        double sum_err = 0.0;
        const int N = 1 << 16;
        for (std::uint64_t idx = 0; idx < static_cast<std::uint64_t>(N); ++idx)
            sum_err += static_cast<double>(apxchol::to_float(apxchol::from_float_stochastic(x, idx)))
                     - static_cast<double>(x);
        // Std error of the mean is ~ ulp * sqrt(p(1-p)/N) <= ulp * 2e-3;
        // allow 5 sigma. RNE's error here would be frac/2^16 * ulp or its
        // complement, i.e. >= 0.25 ulp -- two orders of magnitude off.
        EXPECT_LT(std::fabs(sum_err / N), 0.01 * ulp) << "frac=" << frac;
    }
}

// ── SpTRSV kernels compute in double from the widened stored values ─────
// Build a random unit-ish lower-triangular factor, run forward (L y = x) and
// back (L^T z = y) solves through omp_sptrsv, and check the residuals against
// the values the SpTRSV STORES row by row with a componentwise bound. Because
// the kernels accumulate in double, the residual is at roundoff level even
// when the storage is 16-bit -- if any read did bf16/fp16/fp32 arithmetic this
// would blow up to ~2^-8..2^-11. Under the lowprec builds the stored values
// are: off-diagonals narrowed by omp_sptrsv::narrow_value (RNE, or for the
// bf16 variants stochastic-by-CSC-index under APXCHOL_BF16_STOCHASTIC=1; the
// *_SCALED variants divide by the per-column scale s_j) and the DIAGONAL fp32
// (omp_sptrsv::stored_diag: L_jj, or L_jj / s_j under *_SCALED) -- so this
// test also pins (a) that the kernels divide by that diagonal (a narrow
// diagonal would show as a ~2^-8 residual), (b) that CSR (forward) and CSC
// (back) hold the SAME rounded value for every entry in stochastic mode, and
// (c) the *_SCALED PAIR CONTRACT (omp.h, "FOLDED INTO THE VECTORS"): the
// kernels run on the stored L~ = L D^-1 with no scale multiplication --
// forward_solve returns y' = D y (y'_j = s_j y_j) and transpose_solve, given
// y', solves L~^T z = R y' with R = diag(r_j^2), r_j = inv_scale(s_j) =
// fp32(1/s_j) -- so the pair applies (L_s L_s^T)^-1 for the stored factor
// L_s = L~ D up to the 2^-23 of r_j. Checked (i) at roundoff against L~ / R
// (what the kernels compute) and (ii) y' == D y, z == L_s^-T y against a
// serial double substitution on L_s (roundoff for y', 2^-23-class for z).
namespace {

// The KERNEL matrix L~ as omp_sptrsv::setup stores it (widened, NOT
// rescaled) for the factor entry at CSC position p with value v in a column
// with scale s: off-diagonals through narrow_value, the diagonal through
// stored_diag. Under the fp32/fp64 builds this is v itself.
double kernel_value(factor_value_t v, edge_index p, bool is_diag, bool stochastic, float s) {
    if (is_diag) return apxchol::omp_sptrsv::stored_diag(v, s);
    return apxchol::widen(apxchol::omp_sptrsv::narrow_value(v, p, s, stochastic,
                                                             /*fp16_flush_subnormal=*/true));
}
// The per-column scale the kernels' input / output carry (1.0f off *_SCALED).
float pair_scale(const sparse_csc& L, node_index j) {
#if defined(APXCHOL_SPTRSV_LOWPREC_SCALED)
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
                           const std::vector<double>& y, bool transpose, bool stochastic) {
    const node_index m = L.rows();
    std::vector<double> r(b), scale(m, 0.0);
    for (node_index i = 0; i < m; ++i) scale[i] = std::fabs(b[i]);
    for (node_index j = 0; j < m; ++j) {
        const float s_j = apxchol::omp_sptrsv::column_scale(L.vals_.data(), L.outer_[j], L.outer_[j + 1]);
        for (edge_index p = L.outer_[j]; p < L.outer_[j + 1]; ++p) {
            const node_index i = L.inner_[p];
            const double v = kernel_value(L.vals_[p], p, /*is_diag=*/i == j, stochastic, s_j);
            if (!transpose) { r[i] -= v * y[j]; scale[i] += std::fabs(v * y[j]); }
            else            { r[j] -= v * y[i]; scale[j] += std::fabs(v * y[i]); }
        }
    }
    double worst = 0.0;
    for (node_index i = 0; i < m; ++i)
        worst = std::max(worst, std::fabs(r[i]) / (scale[i] + 1e-300));
    // Double accumulation over <= ~70 terms: roundoff ~1e-14; 2^-8..2^-11
    // would be the signature of any narrow-precision arithmetic.
    EXPECT_LT(worst, 1e-11) << (transpose ? "back" : "forward") << " kernel residual";
}

// Serial double reference solves on the STORED factor L_s = L~ D (kernel
// off-diagonal * s_j, kernel diagonal * s_j): L_s y = x, then L_s^T z = y.
void reference_pair(const sparse_csc& L, const std::vector<double>& x, bool stochastic,
                    std::vector<double>& y, std::vector<double>& z) {
    const node_index m = L.rows();
    // Column-oriented forward substitution.
    y = x;
    for (node_index j = 0; j < m; ++j) {
        const float s_j = apxchol::omp_sptrsv::column_scale(L.vals_.data(), L.outer_[j], L.outer_[j + 1]);
        const double sd = static_cast<double>(pair_scale(L, j));
        const edge_index p0 = L.outer_[j];
        y[j] /= kernel_value(L.vals_[p0], p0, true, stochastic, s_j) * sd;
        for (edge_index p = p0 + 1; p < L.outer_[j + 1]; ++p)
            y[L.inner_[p]] -= kernel_value(L.vals_[p], p, false, stochastic, s_j) * sd * y[j];
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
            acc -= kernel_value(L.vals_[p], p, false, stochastic, s_j) * sd * z[L.inner_[p]];
        z[j] = acc / (kernel_value(L.vals_[p0], p0, true, stochastic, s_j) * sd);
    }
}

void run_kernel_precision_check(bool stochastic) {
    for (node_index m : {node_index(3000), node_index(60000) /* parallel transpose path */}) {
        SCOPED_TRACE("m=" + std::to_string(m) + (stochastic ? " stochastic" : " RNE"));
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
        check_kernel_residual(L, x, yp, /*transpose=*/false, stochastic);
        trsv.transpose_solve(yp.data(), z.data());
        std::vector<double> ryp(m);
        for (node_index j = 0; j < m; ++j) {
            const double r = apxchol::omp_sptrsv::inv_scale(pair_scale(L, j));
            ryp[j] = yp[j] * (r * r);
        }
        check_kernel_residual(L, ryp, z, /*transpose=*/true, stochastic);
        // (ii) The pair contract against the serial reference on L_s = L~ D:
        //      y' = D y at roundoff, z = L_s^-T y up to r_j = fp32(1/s_j)
        //      (2^-23 relative on the input, i.e. on z; exact off *_SCALED).
        std::vector<double> y_ref, z_ref;
        reference_pair(L, x, stochastic, y_ref, z_ref);
        double worst_y = 0.0, worst_z = 0.0, sc_y = 0.0, sc_z = 0.0;
        for (node_index j = 0; j < m; ++j) {
            worst_y = std::max(worst_y, std::fabs(yp[j] - static_cast<double>(pair_scale(L, j)) * y_ref[j]));
            worst_z = std::max(worst_z, std::fabs(z[j] - z_ref[j]));
            sc_y = std::max(sc_y, std::fabs(yp[j]));
            sc_z = std::max(sc_z, std::fabs(z_ref[j]));
        }
        EXPECT_LT(worst_y, 1e-10 * sc_y) << "y' = D y";
#if defined(APXCHOL_SPTRSV_LOWPREC_SCALED)
        EXPECT_LT(worst_z, 1e-5 * sc_z) << "z (r_j = fp32(1/s_j))";
#else
        EXPECT_LT(worst_z, 1e-10 * sc_z) << "z";
#endif
    }
}

} // namespace

TEST(BF16, SpTRSVKernelsComputeInDoubleFromWidenedStorage) {
    scoped_drop_off drop_off;   // storage contract on the un-dropped factor
    run_kernel_precision_check(/*stochastic=*/false);
}

// APXCHOL_BF16_STOCHASTIC=1 is read at every setup(): CSR and CSC must carry
// the same stochastically rounded off-diagonals and the exact diagonal. On the
// non-bf16 builds the env var is ignored and this is the RNE test again.
TEST(BF16, SpTRSVStochasticModeIsConsistentAcrossCSRAndCSC) {
    scoped_drop_off drop_off;   // storage contract on the un-dropped factor
    setenv("APXCHOL_BF16_STOCHASTIC", "1", 1);
    run_kernel_precision_check(/*stochastic=*/true);
    unsetenv("APXCHOL_BF16_STOCHASTIC");
#if defined(APXCHOL_SPTRSV_LOWPREC_BF16) || defined(APXCHOL_SPTRSV_LOWPREC_BF16_SCALED)
    // And it really is a different rounding: the two modes must disagree on
    // some entries of a real factor (else the knob is dead).
    sparse_csc L = make_random_lower(3000, 4.0, 99);
    apxchol::omp_sptrsv rne, sto;
    rne.setup(L, 3000);
    setenv("APXCHOL_BF16_STOCHASTIC", "1", 1);
    sto.setup(L, 3000);
    unsetenv("APXCHOL_BF16_STOCHASTIC");
    ASSERT_EQ(rne.csr_vals().size(), sto.csr_vals().size());
    std::size_t differ = 0;
    for (std::size_t k = 0; k < rne.csr_vals().size(); ++k)
        differ += rne.csr_vals()[k] != sto.csr_vals()[k];
    EXPECT_GT(differ, rne.csr_vals().size() / 10);
#endif
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

TEST(BF16, SpTRSVFatLevelKernelsBothGatherFlavours) {
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
    reference_pair(L, x, /*stochastic=*/false, y_ref, z_ref);

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
        check_kernel_residual(L, x, yp, /*transpose=*/false, /*stochastic=*/false);
        trsv.transpose_solve(yp.data(), z.data());
        std::vector<double> ryp(m);
        for (node_index j = 0; j < m; ++j) {
            const double r = apxchol::omp_sptrsv::inv_scale(pair_scale(L, j));
            ryp[j] = yp[j] * (r * r);
        }
        check_kernel_residual(L, ryp, z,  /*transpose=*/true, /*stochastic=*/false);
        double worst_y = 0.0, worst_z = 0.0, sc_y = 0.0, sc_z = 0.0;
        for (node_index j = 0; j < m; ++j) {
            worst_y = std::max(worst_y, std::fabs(yp[j] - static_cast<double>(pair_scale(L, j)) * y_ref[j]));
            worst_z = std::max(worst_z, std::fabs(z[j] - z_ref[j]));
            sc_y = std::max(sc_y, std::fabs(yp[j]));
            sc_z = std::max(sc_z, std::fabs(z_ref[j]));
        }
        EXPECT_LT(worst_y, 1e-10 * sc_y) << "y' = D y";
#if defined(APXCHOL_SPTRSV_LOWPREC_SCALED)
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
