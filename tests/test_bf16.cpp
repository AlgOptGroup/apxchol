// bf16_t (include/apxchol/bf16.h): round-trip / rounding / widening tests, plus a
// storage-width-agnostic check that the OpenMP SpTRSV kernels compute in double
// from the WIDENED stored factor values (whatever sptrsv_value_t this build
// compiled: bf16 / fp32 / fp64).

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
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
using apxchol::sptrsv_value_t;

namespace {

std::uint32_t f2u(float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; }
float         u2f(std::uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

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

// Assignment from double / int narrows through the same RNE path (this is
// what the factor assembler does with `values[pos] = col.diag`).
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
// value_bytes (what the APXCHOL_VERBOSE banner prints) must agree with it.
TEST(BF16, SpTRSVValueTypeMatchesBuildFlag) {
#if defined(APXCHOL_SPTRSV_BF16)
    EXPECT_EQ(sizeof(sptrsv_value_t), 2u);
    EXPECT_STREQ(apxchol::omp_sptrsv::value_name, "bf16");
    EXPECT_TRUE((std::is_same_v<sptrsv_value_t, bf16_t>));
#elif defined(APXCHOL_SPTRSV_FP32)
    EXPECT_EQ(sizeof(sptrsv_value_t), 4u);
    EXPECT_STREQ(apxchol::omp_sptrsv::value_name, "float (fp32)");
#else
    EXPECT_EQ(sizeof(sptrsv_value_t), 8u);
    EXPECT_STREQ(apxchol::omp_sptrsv::value_name, "double (fp64)");
#endif
    EXPECT_EQ(apxchol::omp_sptrsv::value_bytes, sizeof(sptrsv_value_t));
}

// ── SpTRSV kernels compute in double from the widened stored values ─────
// Build a random unit-ish lower-triangular factor, run forward (L y = x) and
// back (L^T z = y) solves through omp_sptrsv, and check the residual against
// the STORED (widened) values row by row with a componentwise bound. Because
// the kernels accumulate in double, the residual is at roundoff level even
// when the storage is bf16 -- if any read did bf16/fp32 arithmetic, or the
// diagonal read did not widen, this would blow up to ~2^-8.
namespace {

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
            L.vals_[out]  = static_cast<sptrsv_value_t>(r == j ? udiag(rng) : uval(rng));
            ++out;
        }
    }
    return L;
}

// Componentwise residual check of L y = x (transpose == false) or L^T z = x
// (transpose == true) against the widened stored values.
void check_triangular_residual(const sparse_csc& L, const std::vector<double>& x,
                               const std::vector<double>& y, bool transpose) {
    const node_index m = L.rows();
    std::vector<double> r(x), scale(m, 0.0);
    for (node_index i = 0; i < m; ++i) scale[i] = std::fabs(x[i]);
    for (node_index j = 0; j < m; ++j)
        for (edge_index p = L.outer_[j]; p < L.outer_[j + 1]; ++p) {
            const node_index i = L.inner_[p];
            const double v = apxchol::widen(L.vals_[p]);
            if (!transpose) { r[i] -= v * y[j]; scale[i] += std::fabs(v * y[j]); }
            else            { r[j] -= v * y[i]; scale[j] += std::fabs(v * y[i]); }
        }
    double worst = 0.0;
    for (node_index i = 0; i < m; ++i)
        worst = std::max(worst, std::fabs(r[i]) / (scale[i] + 1e-300));
    // Double accumulation over <= ~70 terms: roundoff ~1e-14; 2^-8 = 4e-3
    // would be the signature of any narrow-precision arithmetic.
    EXPECT_LT(worst, 1e-11) << (transpose ? "back" : "forward") << " solve residual";
}

} // namespace

TEST(BF16, SpTRSVKernelsComputeInDoubleFromWidenedStorage) {
    for (node_index m : {node_index(3000), node_index(60000) /* parallel transpose path */}) {
        SCOPED_TRACE("m=" + std::to_string(m));
        sparse_csc L = make_random_lower(m, 4.0, 99);
        apxchol::omp_sptrsv trsv;
        trsv.setup(L, m);
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> ux(-1.0, 1.0);
        std::vector<double> x(m), y(m), z(m);
        for (auto& v : x) v = ux(rng);
        trsv.forward_solve(x.data(), y.data());
        check_triangular_residual(L, x, y, /*transpose=*/false);
        trsv.transpose_solve(y.data(), z.data());
        check_triangular_residual(L, y, z, /*transpose=*/true);
    }
}
