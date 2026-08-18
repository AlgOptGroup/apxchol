// Low-precision SpTRSV storage variants (include/apxchol/lowprec.h):
//   * fp16_t: IEEE binary16 round-trip bound (2^-11 relative, normal range),
//     the documented subnormal / flush-to-zero / overflow behaviour, RNE ties,
//     and an exhaustive cross-check of the bit-level converters against the
//     compiler's _Float16 where available;
//   * fp24_t: 24-bit round-trip bound (2^-16 relative), exact values, RNE
//     ties + carry, NaN;
//   * the storage CONTRACT of omp_sptrsv::setup for whatever variant this
//     build compiled: every stored CSR/CSC value == narrow_value(v, p, s_j),
//     the per-column scales, the off-diagonal flush/subnormal statistics, and
//     the APXCHOL_LOWPREC_DROP diagnostic (fp32/fp64 builds).
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "apxchol/lowprec.h"
#include "apxchol/sparse_csc.h"
#include "apxchol/solver/sptrsv/omp.h"

using apxchol::edge_index;
using apxchol::factor_value_t;
using apxchol::fp16_t;
using apxchol::fp24_t;
using apxchol::node_index;
using apxchol::sparse_csc;
using apxchol::sptrsv_value_t;

namespace {

std::uint32_t f2u(float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; }
float         u2f(std::uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

double rel_err(double y, double x) { return std::fabs(y - x) / std::fabs(x); }

} // namespace

// ── fp16_t ──────────────────────────────────────────────────────────────

// Normal range [2^-14, 65504]: |fp16(x) - x| <= 2^-11 |x| (11 significant
// bits, RNE), the bit-level widen agrees with to_float(), sign is kept.
TEST(FP16, RoundTripWithin2PowMinus11RelativeInNormalRange) {
    std::mt19937_64 rng(1601);
    std::uniform_real_distribution<double> uexp(-14.0, 15.0);
    std::uniform_real_distribution<double> umant(1.0, 2.0);
    std::bernoulli_distribution sign(0.5);
    const double bound = std::ldexp(1.0, -11);
    double worst = 0.0;
    for (int t = 0; t < 2'000'000; ++t) {
        double xd = std::ldexp(umant(rng), static_cast<int>(std::floor(uexp(rng))));
        if (xd > 65504.0) xd = 65504.0;
        const float x = static_cast<float>((sign(rng) ? -1.0 : 1.0) * xd);
        const fp16_t h = x;
        const float  y = h.to_float();
        ASSERT_EQ(f2u(fp16_t::widen_bits(h.bits)), f2u(y)) << "widen_bits vs to_float, x=" << x;
        ASSERT_FALSE(fp16_t::is_subnormal(h.bits)) << "x=" << x;
        ASSERT_FALSE(fp16_t::is_zero(h.bits)) << "x=" << x;
        const double rel = rel_err(y, x);
        worst = std::max(worst, rel);
        ASSERT_LE(rel, bound) << "x=" << x << " y=" << y;
        ASSERT_EQ(apxchol::widen(h), static_cast<double>(y));
        ASSERT_EQ(std::signbit(y), std::signbit(x));
    }
    EXPECT_GT(worst, bound * 0.9);   // the bound is actually approached
}

// Every one of the 65536 fp16 bit patterns that is not a NaN widens exactly
// and narrows back to itself (zeros, subnormals, normals, infinities): the
// narrowing is exact on representable values. Where the compiler has
// _Float16, the bit-level widen must agree with it bit-for-bit.
TEST(FP16, AllBitPatternsRoundTripAndMatchCompilerWiden) {
    for (std::uint32_t hb = 0; hb < 0x10000u; ++hb) {
        const std::uint16_t h = static_cast<std::uint16_t>(hb);
        if (fp16_t::is_inf_or_nan(h) && (h & 0x03ffu) != 0) continue;   // NaN payloads: skip
        const float f = fp16_t::widen_bits(h);
        ASSERT_EQ(fp16_t(f).bits, h) << "pattern " << std::hex << hb;
        ASSERT_EQ(f2u(fp16_t::from_bits(h).to_float()), f2u(f)) << "pattern " << std::hex << hb;
#if defined(__FLT16_MANT_DIG__)
        _Float16 c; std::memcpy(&c, &h, 2);
        ASSERT_EQ(f2u(static_cast<float>(c)), f2u(f)) << "pattern " << std::hex << hb;
#endif
    }
}

#if defined(__FLT16_MANT_DIG__)
// The bit-level fp32 -> fp16 RNE agrees with the compiler's conversion on
// random full-range fp32 patterns (normals, subnormal-range, overflow, both
// signs; NaNs excluded).
TEST(FP16, NarrowingMatchesCompilerConversion) {
    std::mt19937_64 rng(1602);
    std::uniform_int_distribution<std::uint32_t> ubits(0u, 0xffffffffu);
    // Bias the sampling towards the interesting fp16 exponents as well.
    std::uniform_real_distribution<double> uexp(-30.0, 20.0);
    std::uniform_real_distribution<double> umant(1.0, 2.0);
    for (int t = 0; t < 4'000'000; ++t) {
        float x;
        if (t & 1) {
            const std::uint32_t u = ubits(rng);
            if ((u & 0x7fffffffu) > 0x7f800000u) continue;                // NaN
            x = u2f(u);
        } else {
            x = static_cast<float>(std::ldexp(umant(rng), static_cast<int>(std::floor(uexp(rng)))));
            if (t & 2) x = -x;
        }
        const _Float16 c = static_cast<_Float16>(x);
        std::uint16_t cb; std::memcpy(&cb, &c, 2);
        ASSERT_EQ(fp16_t(x).bits, cb) << "x=" << x << " (" << std::hex << f2u(x) << ")";
    }
}
#endif

// The documented flush / subnormal / overflow / tie behaviour.
TEST(FP16, DocumentedSubnormalFlushOverflowAndTies) {
    const float two_m14 = std::ldexp(1.0f, -14);   // smallest normal
    const float two_m24 = std::ldexp(1.0f, -24);   // smallest subnormal
    const float two_m25 = std::ldexp(1.0f, -25);   // half of it: the flush threshold
    // Boundaries.
    EXPECT_EQ(fp16_t(two_m14).bits, 0x0400u);
    EXPECT_EQ(fp16_t(two_m24).bits, 0x0001u);
    EXPECT_EQ(fp16_t(two_m25).bits, 0x0000u);                       // exact tie -> even (zero)
    EXPECT_EQ(fp16_t(-two_m25).bits, 0x8000u);                      // sign kept on the flush
    EXPECT_EQ(fp16_t(std::nextafter(two_m25, 1.0f)).bits, 0x0001u); // just above the tie -> 2^-24
    EXPECT_EQ(fp16_t(std::nextafter(two_m25, 0.0f)).bits, 0x0000u);
    EXPECT_EQ(fp16_t(1e-9f).bits, 0x0000u);                         // deep below: flush
    EXPECT_TRUE(fp16_t::is_zero(fp16_t(1e-9f).bits));
    // Subnormals: absolute error <= 2^-25 (half the subnormal spacing).
    std::mt19937_64 rng(1603);
    std::uniform_real_distribution<float> usub(two_m24, two_m14);
    for (int t = 0; t < 200'000; ++t) {
        const float x = usub(rng);
        const fp16_t h = x;
        ASSERT_TRUE(fp16_t::is_subnormal(h.bits) || h.bits == 0x0400u) << "x=" << x;
        ASSERT_LE(std::fabs(static_cast<double>(h.to_float()) - x), std::ldexp(1.0, -25)) << "x=" << x;
        ASSERT_TRUE(apxchol::is_stored_subnormal(h) || h.bits == 0x0400u);
    }
    // 1e-6 (~2^-20) is subnormal, 1e-3 is normal.
    EXPECT_TRUE(fp16_t::is_subnormal(fp16_t(1e-6f).bits));
    EXPECT_FALSE(fp16_t::is_subnormal(fp16_t(1e-3f).bits));
    // Overflow.
    EXPECT_EQ(fp16_t(65504.0f).bits, 0x7bffu);                      // largest finite, exact
    EXPECT_EQ(fp16_t(65519.0f).bits, 0x7bffu);                      // below the midpoint: rounds down
    EXPECT_EQ(fp16_t(65520.0f).bits, 0x7c00u);                      // midpoint: ties to even == inf
    EXPECT_EQ(fp16_t(1e6f).bits, 0x7c00u);
    EXPECT_EQ(fp16_t(-1e6f).bits, 0xfc00u);
    EXPECT_TRUE(std::isinf(fp16_t(std::numeric_limits<float>::infinity()).to_float()));
    EXPECT_TRUE(std::isnan(fp16_t(std::numeric_limits<float>::quiet_NaN()).to_float()));
    // RNE ties in the normal range: 1 + 2^-11 is halfway between 1 (even) and
    // 1 + 2^-10 (odd) -> 1; 1 + 3*2^-11 is halfway between 1 + 2^-10 (odd)
    // and 1 + 2^-9 (even) -> 1 + 2^-9.
    EXPECT_EQ(fp16_t(1.0f + std::ldexp(1.0f, -11)).to_float(), 1.0f);
    EXPECT_EQ(fp16_t(1.0f + 3.0f * std::ldexp(1.0f, -11)).to_float(), 1.0f + std::ldexp(1.0f, -9));
    // Carry out of the mantissa into the exponent: the largest fp32 below 2
    // rounds up to 2.0.
    EXPECT_EQ(fp16_t(u2f(0x3fffffffu)).bits, 0x4000u);
    // Exact values, and the "max ratio is exactly 1" property the *_SCALED
    // variants rely on (v / v == 1.0f -> representable).
    for (float x : {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 1.5f, 255.0f, 0.75f, -3.25f, 2048.0f})
        EXPECT_EQ(f2u(fp16_t(x).to_float()), f2u(x)) << x;
    // Assignment from double / int narrows through the same path.
    EXPECT_EQ(fp16_t(0.75).bits, fp16_t(0.75f).bits);
    EXPECT_EQ(fp16_t(7).to_float(), 7.0f);
    EXPECT_EQ(sizeof(fp16_t), 2u);
}

// ── fp24_t ──────────────────────────────────────────────────────────────

// Full fp32 normal range: |fp24(x) - x| <= 2^-16 |x| (16 significant bits,
// RNE); the round trip through the 3-byte packing is what is tested.
TEST(FP24, RoundTripWithin2PowMinus16Relative) {
    std::mt19937_64 rng(2401);
    std::uniform_real_distribution<double> uexp(-120.0, 120.0);
    std::uniform_real_distribution<double> umant(1.0, 2.0);
    std::bernoulli_distribution sign(0.5);
    const double bound = std::ldexp(1.0, -16);
    double worst = 0.0;
    for (int t = 0; t < 2'000'000; ++t) {
        const float x = static_cast<float>(
            (sign(rng) ? -1.0 : 1.0) * std::ldexp(umant(rng), static_cast<int>(std::floor(uexp(rng)))));
        const fp24_t q = x;
        const float  y = q.to_float();
        const double rel = rel_err(y, x);
        worst = std::max(worst, rel);
        ASSERT_LE(rel, bound) << "x=" << x << " y=" << y;
        ASSERT_EQ(apxchol::widen(q), static_cast<double>(y));
        ASSERT_EQ(std::signbit(y), std::signbit(x));
        // The stored pattern is the top 24 bits of the widened float, and the
        // widened float has its low 8 bits clear.
        ASSERT_EQ(q.bits(), f2u(y) >> 8);
        ASSERT_EQ(f2u(y) & 0xffu, 0u);
        ASSERT_FALSE(apxchol::is_stored_subnormal(q));
    }
    EXPECT_GT(worst, bound * 0.9);
}

// Values with <= 16 significant bits are exact; RNE ties, carry, NaN, and the
// fp32 subnormals (which fp24 keeps, at 15 bits).
TEST(FP24, ExactValuesTiesCarryNaNSubnormal) {
    for (float x : {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 1.5f, 65535.0f, -32767.5f,
                    1.0f + std::ldexp(1.0f, -15) /* 1 + 2^-15 */, std::ldexp(1.0f, -100), std::ldexp(1.0f, 100)}) {
        const fp24_t q = x;
        EXPECT_EQ(f2u(q.to_float()), f2u(x)) << x;
        EXPECT_EQ(q.bits(), f2u(x) >> 8);
    }
    // 1 + 2^-16 is halfway between 1 (even) and 1 + 2^-15 (odd) -> 1.
    EXPECT_EQ(fp24_t(1.0f + std::ldexp(1.0f, -16)).to_float(), 1.0f);
    // 1 + 3*2^-16 is halfway between 1 + 2^-15 (odd) and 1 + 2^-14 (even) -> 1 + 2^-14.
    EXPECT_EQ(fp24_t(1.0f + 3.0f * std::ldexp(1.0f, -16)).to_float(), 1.0f + std::ldexp(1.0f, -14));
    // Just above / below the tie go to the nearest.
    EXPECT_EQ(fp24_t(u2f(0x3f800081u)).to_float(), 1.0f + std::ldexp(1.0f, -15));
    EXPECT_EQ(fp24_t(u2f(0x3f80007fu)).to_float(), 1.0f);
    // Carry: 1 - 2^-24 (0x3f7fffff) rounds UP to 1.0 (0x3f8000), 0x3fffffff -> 2.0.
    EXPECT_EQ(fp24_t(u2f(0x3f7fffffu)).bits(), 0x3f8000u);
    EXPECT_EQ(fp24_t(u2f(0x3fffffffu)).bits(), 0x400000u);
    EXPECT_EQ(fp24_t(u2f(0xbf7fffffu)).bits(), 0xbf8000u);
    // FLT_MAX overflows to inf (within half an fp24 ulp of the top).
    EXPECT_TRUE(std::isinf(fp24_t(std::numeric_limits<float>::max()).to_float()));
    EXPECT_TRUE(std::isnan(fp24_t(std::numeric_limits<float>::quiet_NaN()).to_float()));
    EXPECT_TRUE(std::isinf(fp24_t(std::numeric_limits<float>::infinity()).to_float()));
    // fp32 subnormals stay (sub)normal-with-15-bits.
    const float sub = std::ldexp(1.0f, -140);
    EXPECT_TRUE(apxchol::is_stored_subnormal(fp24_t(sub)));
    EXPECT_NEAR(fp24_t(sub).to_float(), sub, sub * std::ldexp(1.0, -14));
    // Byte layout: b[0] is the low byte of the 24-bit pattern.
    const fp24_t one = 1.0f;   // 0x3f8000
    EXPECT_EQ(one.b[0], 0x00u); EXPECT_EQ(one.b[1], 0x80u); EXPECT_EQ(one.b[2], 0x3fu);
    EXPECT_EQ(sizeof(fp24_t), 3u);
    EXPECT_EQ(fp24_t(0.75).bits(), fp24_t(0.75f).bits());
    EXPECT_EQ(fp24_t(7).to_float(), 7.0f);
    // A dense array of them really is 3 B/entry.
    fp24_t arr[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_EQ(reinterpret_cast<const char*>(&arr[1]) - reinterpret_cast<const char*>(&arr[0]), 3);
    EXPECT_EQ(arr[3].to_float(), 4.0f);
}

// ── omp_sptrsv storage contract for THIS build's variant ────────────────
namespace {

sparse_csc make_random_lower(node_index m, double avg_offdiag, unsigned seed,
                             double vmin = -0.5, double vmax = 0.5) {
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
            L.vals_[out]  = static_cast<factor_value_t>(r == j ? udiag(rng) : uval(rng));
            ++out;
        }
    }
    return L;
}

// Per-column scale as documented (max |off-diagonal|, 1.0f if none) --
// computed here independently of omp_sptrsv::column_scale.
std::vector<float> reference_scales(const sparse_csc& L) {
    std::vector<float> s(L.rows(), 1.0f);
    for (node_index j = 0; j < L.rows(); ++j) {
        double mx = 0.0;
        for (edge_index p = L.outer_[j] + 1; p < L.outer_[j + 1]; ++p)
            mx = std::max(mx, std::fabs(static_cast<double>(L.vals_[p])));
        s[j] = mx > 0.0 ? static_cast<float>(mx) : 1.0f;
    }
    return s;
}

} // namespace

// Every stored CSC value equals narrow_value(v, p, s_j) bit-for-bit, and so
// does its CSR twin; the scales (if any) match; offdiag count is right.
TEST(LowPrec, SetupStoresNarrowValueOfEveryEntryAndTheColumnScales) {
    for (node_index m : {node_index(3000), node_index(60000) /* parallel transpose */}) {
        SCOPED_TRACE("m=" + std::to_string(m));
        // Wide value range so the *_SCALED variants see ratios spanning many
        // binades (fp16 subnormals / flushes included) and fp24 sees the range.
        sparse_csc L = make_random_lower(m, 4.0, 77, -2.0, 2.0);
        for (edge_index p = 0; p < L.nonZeros(); ++p)
            if ((p % 7) == 3) L.vals_[p] = static_cast<factor_value_t>(L.vals_[p] * 1e-7);
        const std::vector<float> s = reference_scales(L);
        for (node_index j = 0; j < m; ++j)
            ASSERT_EQ(apxchol::omp_sptrsv::column_scale(L.vals_.data(), L.outer_[j], L.outer_[j + 1]), s[j]);

        apxchol::omp_sptrsv trsv;
        trsv.setup(L, m);
        ASSERT_EQ(trsv.csc_vals().size(), static_cast<size_t>(L.nonZeros()));
        std::uint64_t offdiag = 0;
        for (node_index j = 0; j < m; ++j)
            for (edge_index p = L.outer_[j]; p < L.outer_[j + 1]; ++p) {
                const sptrsv_value_t expect = apxchol::omp_sptrsv::narrow_value(L.vals_[p], p, s[j], false);
                ASSERT_TRUE(trsv.csc_vals()[p] == expect) << "csc p=" << p;
                if (L.inner_[p] != j) ++offdiag;
                // CSR twin: locate (row = inner[p], col = j).
                const node_index i = L.inner_[p];
                bool found = false;
                for (edge_index q = trsv.csr_row_ptr()[i]; q < trsv.csr_row_ptr()[i + 1]; ++q)
                    if (trsv.csr_col_idx()[q] == j) {
                        ASSERT_TRUE(trsv.csr_vals()[q] == expect) << "csr q=" << q;
                        found = true;
                        break;
                    }
                ASSERT_TRUE(found);
            }
        EXPECT_EQ(trsv.lowprec_stats().offdiag, offdiag);
        EXPECT_EQ(trsv.lowprec_stats().dropped, 0u);
#if defined(APXCHOL_SPTRSV_LOWPREC_SCALED)
        ASSERT_EQ(trsv.col_scales().size(), static_cast<size_t>(m));
        for (node_index j = 0; j < m; ++j) ASSERT_EQ(trsv.col_scales()[j], s[j]) << "j=" << j;
#endif
#if defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED)
        // The 1e-7-scaled entries sit ~2^-23 below their column max: fp16 must
        // have flushed or subnormalised a good fraction of the off-diagonals.
        EXPECT_GT(trsv.lowprec_stats().flushed + trsv.lowprec_stats().subnormal, offdiag / 20);
#else
        // bf16 / fp24 / fp32 / fp64 have fp32's range: nothing flushes here.
        EXPECT_EQ(trsv.lowprec_stats().flushed, 0u);
        EXPECT_EQ(trsv.lowprec_stats().subnormal, 0u);
#endif
    }
}

// A hand-built column with one flushed and one subnormal ratio: the counts
// are exact under FP16_SCALED and zero everywhere else.
TEST(LowPrec, FlushAndSubnormalCountsAreExact) {
    // 4x4 lower factor: column 0 = diag 2, off-diagonals 1.0 (max), 1e-6 (fp16
    // subnormal after scaling), 1e-9 (flushed). Columns 1..3: diagonal only /
    // one plain off-diagonal.
    sparse_csc L;
    L.n_ = 4;
    L.outer_ = {0, 4, 6, 7, 8};
    L.inner_ = {0, 1, 2, 3,  1, 2,  2,  3};
    L.vals_  = {2.0f, 1.0f, 1e-6f, 1e-9f,  1.5f, -0.25f,  1.0f,  1.0f};
    apxchol::omp_sptrsv trsv;
    trsv.setup(L, 4);
    EXPECT_EQ(trsv.lowprec_stats().offdiag, 4u);
    EXPECT_EQ(trsv.lowprec_stats().dropped, 0u);
#if defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED)
    EXPECT_EQ(trsv.lowprec_stats().flushed, 1u);
    EXPECT_EQ(trsv.lowprec_stats().subnormal, 1u);
    EXPECT_EQ(trsv.col_scales()[0], 1.0f);
    EXPECT_EQ(trsv.col_scales()[1], 0.25f);
    EXPECT_EQ(trsv.col_scales()[2], 1.0f);   // no off-diagonal -> 1
    // What the kernels see: 1e-9 is gone, 1e-6 is a subnormal approximation.
    EXPECT_EQ(apxchol::widen(trsv.csc_vals()[3]), 0.0);
    EXPECT_NEAR(apxchol::widen(trsv.csc_vals()[2]), 1e-6, std::ldexp(1.0, -25));
#else
    EXPECT_EQ(trsv.lowprec_stats().flushed, 0u);
    EXPECT_EQ(trsv.lowprec_stats().subnormal, 0u);
#endif
    // The forward/back solves still run and are exact w.r.t. the STORED
    // values (a tiny system: check L y = x against them).
    std::vector<double> x = {1.0, -2.0, 0.5, 3.0}, y(4);
    trsv.forward_solve(x.data(), y.data());
    for (node_index i = 0; i < 4; ++i) {
        double r = x[i];
        for (node_index j = 0; j <= i; ++j)
            for (edge_index p = L.outer_[j]; p < L.outer_[j + 1]; ++p)
                if (L.inner_[p] == i) {
                    double v = (i == j) ? static_cast<double>(L.vals_[p])
                                        : apxchol::widen(trsv.csc_vals()[p]);
#if defined(APXCHOL_SPTRSV_LOWPREC_SCALED)
                    if (i != j) v *= trsv.col_scales()[j];
#endif
                    r -= v * y[j];
                }
        EXPECT_NEAR(r, 0.0, 1e-12) << "row " << i;
    }
}

// APXCHOL_LOWPREC_DROP=<rel> (fp32/fp64 builds): off-diagonals with |v| <
// rel * s_j are stored as zero, counted in dropped, present as zeros in both
// CSR and CSC, and the kernels use the zeros. Ignored on the lowprec builds.
TEST(LowPrec, DropDiagnosticZeroesSmallOffDiagonalsRelativeToColumnMax) {
    sparse_csc L = make_random_lower(3000, 4.0, 55, -1.0, 1.0);
    const std::vector<float> s = reference_scales(L);
    setenv("APXCHOL_LOWPREC_DROP", "0.3", 1);
    apxchol::omp_sptrsv trsv;
    trsv.setup(L, 3000);
    unsetenv("APXCHOL_LOWPREC_DROP");
    std::uint64_t expect_dropped = 0, csc_zero = 0, csr_zero = 0;
    for (node_index j = 0; j < 3000; ++j)
        for (edge_index p = L.outer_[j]; p < L.outer_[j + 1]; ++p) {
            if (L.inner_[p] == j) continue;
            const bool drop = std::fabs(static_cast<double>(L.vals_[p])) < 0.3 * s[j];
            expect_dropped += drop;
            csc_zero += apxchol::widen(trsv.csc_vals()[p]) == 0.0;
        }
    for (std::size_t q = 0; q < trsv.csr_vals().size(); ++q)
        csr_zero += apxchol::widen(trsv.csr_vals()[q]) == 0.0;
    ASSERT_GT(expect_dropped, 100u);   // the test has teeth
#if defined(APXCHOL_SPTRSV_LOWPREC_ANY)
    // Ignored: nothing dropped, no unexpected zeros beyond the format's own
    // (values here are >= 1e-9 relative -- but fp16 might flush a few; only
    // the dropped counter is asserted).
    EXPECT_EQ(trsv.lowprec_stats().dropped, 0u);
#else
    EXPECT_EQ(trsv.lowprec_stats().dropped, expect_dropped);
    EXPECT_EQ(csc_zero, expect_dropped);
    EXPECT_EQ(csr_zero, expect_dropped);
    // The kernels solve with the DROPPED matrix: L_drop y = x exactly (double
    // accumulation), where L_drop is what the CSC now holds.
    std::mt19937 rng(9);
    std::uniform_real_distribution<double> ux(-1.0, 1.0);
    std::vector<double> x(3000), y(3000);
    for (auto& v : x) v = ux(rng);
    trsv.forward_solve(x.data(), y.data());
    double worst = 0.0;
    for (node_index i = 0; i < 3000; ++i) {
        double r = x[i], sc = std::fabs(x[i]);
        for (edge_index q = trsv.csr_row_ptr()[i]; q < trsv.csr_row_ptr()[i + 1]; ++q) {
            const double t = apxchol::widen(trsv.csr_vals()[q]) * y[trsv.csr_col_idx()[q]];
            r -= t; sc += std::fabs(t);
        }
        worst = std::max(worst, std::fabs(r) / (sc + 1e-300));
    }
    EXPECT_LT(worst, 1e-11);
#endif
    // Without the env var nothing is dropped (read at every setup).
    apxchol::omp_sptrsv plain;
    plain.setup(L, 3000);
    EXPECT_EQ(plain.lowprec_stats().dropped, 0u);
}
