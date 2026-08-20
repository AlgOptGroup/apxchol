// Low-precision SpTRSV storage (include/apxchol/lowprec.h):
//   * fp16_t: IEEE binary16 round-trip bound (2^-11 relative, normal range),
//     the documented subnormal / flush-to-zero / overflow behaviour, RNE ties,
//     and an exhaustive cross-check of the bit-level converters against the
//     compiler's _Float16 where available;
//   * the storage CONTRACT of omp_sptrsv::setup for whatever storage this
//     build compiled: every stored CSR/CSC value == narrow_value(v, s_j),
//     the per-column scales, the off-diagonal flush/subnormal statistics, the
//     fp16 subnormal flush (FP16_SCALED default; APXCHOL_FP16_KEEP_SUBNORMAL=1
//     restores IEEE), and the compacting drop APXCHOL_FACTOR_DROP=<rel>
//     (every build): stored nnz == kept entries, and the compacted SpTRSV
//     solves like the zeroed-but-not-removed reference.
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "apxchol/lowprec.h"
#include "apxchol/sparse_csc.h"
#include "apxchol/solver/sptrsv/omp.h"

using apxchol::edge_index;
using apxchol::factor_value_t;
using apxchol::fp16_t;
using apxchol::node_index;
using apxchol::sparse_csc;
using apxchol::sptrsv_value_t;

namespace {

std::uint32_t f2u(float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; }
float         u2f(std::uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

double rel_err(double y, double x) { return std::fabs(y - x) / std::fabs(x); }

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

// ── omp_sptrsv storage contract for THIS build's storage ────────────────
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

// Every stored CSC value equals narrow_value(v, s_j) bit-for-bit, and so
// does its CSR twin; the scales (if any) match; offdiag count is right.
TEST(LowPrec, SetupStoresNarrowValueOfEveryEntryAndTheColumnScales) {
    // These state the DEFAULT (fp32) storage contract, so pin it: the suite
    // is also run with APXCHOL_SPTRSV_FP16=1 in the environment.
    scoped_env fp32_storage("APXCHOL_SPTRSV_FP16", "0");

    scoped_drop_off drop_off;   // storage contract on the un-dropped factor
    for (node_index m : {node_index(3000), node_index(60000) /* parallel transpose */}) {
        SCOPED_TRACE("m=" + std::to_string(m));
        // Wide value range so FP16_SCALED sees ratios spanning many binades
        // (fp16 subnormals / flushes included).
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
                const sptrsv_value_t expect = apxchol::omp_sptrsv::narrow_value(L.vals_[p], s[j]);
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
        // fp32 / fp64 have fp32's range: nothing flushes here.
        EXPECT_EQ(trsv.lowprec_stats().flushed, 0u);
        EXPECT_EQ(trsv.lowprec_stats().subnormal, 0u);
    }
}

// A hand-built column with one flushed and one subnormal ratio: the counts
// are exact under FP16_SCALED (default: the subnormal is flushed too, so
// flushed=2 / subnormal=0; APXCHOL_FP16_KEEP_SUBNORMAL=1: flushed=1 /
// subnormal=1) and zero everywhere else. The forward solve is exact w.r.t.
// the STORED values either way.
TEST(LowPrec, FlushAndSubnormalCountsAreExact) {
    // These state the DEFAULT (fp32) storage contract, so pin it: the suite
    // is also run with APXCHOL_SPTRSV_FP16=1 in the environment.
    scoped_env fp32_storage("APXCHOL_SPTRSV_FP16", "0");

    scoped_drop_off drop_off;   // the 1e-6 / 1e-9 entries must reach the storage format
    // 4x4 lower factor: column 0 = diag 2, off-diagonals 1.0 (max), 1e-6 (fp16
    // subnormal after scaling), 1e-9 (flushed). Columns 1..3: diagonal only /
    // one plain off-diagonal.
    sparse_csc L;
    L.n_ = 4;
    L.outer_ = {0, 4, 6, 7, 8};
    L.inner_ = {0, 1, 2, 3,  1, 2,  2,  3};
    L.vals_  = {2.0f, 1.0f, 1e-6f, 1e-9f,  1.5f, -0.25f,  1.0f,  1.0f};
    for (int keep_sub = 0; keep_sub < 2; ++keep_sub) {
        SCOPED_TRACE(keep_sub ? "APXCHOL_FP16_KEEP_SUBNORMAL=1" : "default (fp16 subnormals flushed)");
        if (keep_sub) setenv("APXCHOL_FP16_KEEP_SUBNORMAL", "1", 1);
        apxchol::omp_sptrsv trsv;
        trsv.setup(L, 4);
        unsetenv("APXCHOL_FP16_KEEP_SUBNORMAL");
        EXPECT_EQ(trsv.lowprec_stats().offdiag, 4u);
        EXPECT_EQ(trsv.lowprec_stats().dropped, 0u);
        EXPECT_EQ(trsv.lowprec_stats().nnz_factor, 8u);
        EXPECT_EQ(trsv.lowprec_stats().nnz_stored, 8u);
        EXPECT_EQ(trsv.stored_nnz(), 8u);
        EXPECT_EQ(trsv.lowprec_stats().factor_subnormal, 0u);   // every fp32 value here is normal
        EXPECT_EQ(trsv.lowprec_stats().flushed, 0u);
        EXPECT_EQ(trsv.lowprec_stats().subnormal, 0u);
        EXPECT_FALSE(apxchol::omp_sptrsv::format_flushes(1e-9f, 1.0f));   // fp32's range: only 0 stores as 0
        EXPECT_TRUE(apxchol::omp_sptrsv::format_flushes(0.0f, 1.0f));
        // The forward/back solves still run and are exact w.r.t. the STORED
        // values (a tiny system: check L~ y' = x against them -- the kernel
        // matrix L~: widened off-diagonals, stored_diag() diagonal; under
        // FP16_SCALED forward_solve returns y' = D y, see omp.h).
        std::vector<double> x = {1.0, -2.0, 0.5, 3.0}, y(4);
        trsv.forward_solve(x.data(), y.data());
        for (node_index i = 0; i < 4; ++i) {
            double r = x[i];
            for (node_index j = 0; j <= i; ++j)
                for (edge_index p = L.outer_[j]; p < L.outer_[j + 1]; ++p)
                    if (L.inner_[p] == i) {
                        const float sj =
                            1.0f;
                        const double v = (i == j) ? apxchol::omp_sptrsv::stored_diag(L.vals_[p], sj)
                                                  : apxchol::widen(trsv.csc_vals()[p]);
                        r -= v * y[j];
                    }
            EXPECT_NEAR(r, 0.0, 1e-12) << "row " << i;
        }
    }
}

// ── Compacting drop: APXCHOL_FACTOR_DROP=<rel> ──────────────────────────
namespace {

// The reference "dense drop": the same factor with the entries the drop
// removes set to ZERO in place (nothing removed) and each column's dropped
// mass folded back into its kept off-diagonals in proportion to |v| -- the
// unconditional column-sum compensation. Kept iff |v| >= rel * s_j and the
// storage format does not store it as zero anyway (fp32: only an exact zero).
// Written out here without omp_sptrsv::keep_offdiag so the predicate and the
// compensation are stated twice independently.
struct dense_drop {
    sparse_csc    Lz;        // zeroed + compensated copy
    std::uint64_t kept  = 0; // entries with a nonzero-or-diagonal slot (== stored nnz after the drop)
    std::uint64_t zeroed = 0;
};
// Per-column scale of L11 = L.topLeftCorner(m, m) (rows >= m excluded -- the
// Laplacian path's grounded last row is not part of what the SpTRSV sees).
std::vector<float> reference_scales_L11(const sparse_csc& L, node_index m) {
    std::vector<float> s(m, 1.0f);
    for (node_index j = 0; j < m; ++j) {
        double mx = 0.0;
        for (edge_index p = L.outer_[j] + 1; p < L.outer_[j + 1]; ++p)
            if (L.inner_[p] < m) mx = std::max(mx, std::fabs(static_cast<double>(L.vals_[p])));
        s[j] = mx > 0.0 ? static_cast<float>(mx) : 1.0f;
    }
    return s;
}

dense_drop reference_dense_drop(const sparse_csc& L, node_index m, double rel) {
    dense_drop d;
    d.Lz = L;
    const std::vector<float> s = reference_scales_L11(L, m);
    for (node_index j = 0; j < m; ++j) {
        double dropped_sum = 0.0, kept_abs = 0.0;
        for (edge_index p = L.outer_[j]; p < L.outer_[j + 1]; ++p) {
            if (L.inner_[p] >= m) continue;                    // outside L11 (Laplacian last row)
            if (L.inner_[p] == j) { ++d.kept; continue; }
            const factor_value_t v = L.vals_[p];
            bool keep = std::fabs(static_cast<double>(v)) >= rel * static_cast<double>(s[j]);
            if (v == 0) keep = false;
            if (keep) { ++d.kept; kept_abs += std::fabs(static_cast<double>(v)); }
            else      { ++d.zeroed; dropped_sum += static_cast<double>(v); d.Lz.vals_[p] = 0; }
        }
        if (dropped_sum != 0.0 && kept_abs > 0.0) {
            const double per_abs = dropped_sum / kept_abs;
            for (edge_index p = L.outer_[j] + 1; p < L.outer_[j + 1]; ++p) {
                if (L.inner_[p] >= m) continue;
                const double v = static_cast<double>(d.Lz.vals_[p]);
                if (v == 0.0) continue;                        // dropped (or an exact zero)
                d.Lz.vals_[p] = static_cast<factor_value_t>(v + per_abs * std::fabs(v));
            }
        }
    }
    return d;
}

} // namespace

// APXCHOL_FACTOR_DROP=<rel> (every build): off-diagonals with |v| < rel * s_j
// (and those the storage format would store as zero) are REMOVED before the
// CSR/CSC are built: stored nnz == the number of kept entries, the diagonal
// is always kept (first in every CSC column, last in every CSR row), the
// statistics say what was dropped, and the forward AND back solves through
// the compacted arrays agree with the reference dense-drop (entries zeroed
// and the column mass folded back in place, drop off on the reference) to
// ~1e-12. Both the SDDM alias path (m == n) and
// the Laplacian copy path (m == n-1), serial and parallel transpose (m >
// 50000), with a spread of magnitudes so ~half of the off-diagonals fall
// under 1e-4.
TEST(LowPrec, FactorDropCompactsToKeptEntriesAndSolvesLikeTheZeroedReference) {
    // These state the DEFAULT (fp32) storage contract, so pin it: the suite
    // is also run with APXCHOL_SPTRSV_FP16=1 in the environment.
    scoped_env fp32_storage("APXCHOL_SPTRSV_FP16", "0");

    struct cfg { node_index n; bool laplacian; };
    for (cfg c : {cfg{3000, false}, cfg{3001, true}, cfg{60000, false}, cfg{60001, true}}) {
        const node_index m = c.laplacian ? c.n - 1 : c.n;
        SCOPED_TRACE("n=" + std::to_string(c.n) + (c.laplacian ? " (Laplacian m=n-1)" : " (SDDM m=n)"));
        // Magnitudes spread over ~9 decades: |v| = 10^u, u uniform in [-7, 2),
        // so about half of every column sits below 1e-4 * s_j.
        sparse_csc L = make_random_lower(c.n, 6.0, 4242 + c.n, -1.0, 1.0);
        {
            std::mt19937 rng(77 + c.n);
            std::uniform_real_distribution<double> uexp(-7.0, 2.0);
            for (node_index j = 0; j < c.n; ++j)
                for (edge_index p = L.outer_[j] + 1; p < L.outer_[j + 1]; ++p) {
                    const double sign = L.vals_[p] < 0 ? -1.0 : 1.0;
                    L.vals_[p] = static_cast<factor_value_t>(sign * std::pow(10.0, uexp(rng)));
                    if ((p % 101) == 5) L.vals_[p] = 0;   // a few explicit zeros: always dropped
                }
        }
        const double rel = 1e-4;
        const dense_drop ref = reference_dense_drop(L, m, rel);
        ASSERT_GT(ref.zeroed, ref.kept / 4);   // the test has teeth: a big chunk goes

        // Compacted (env set for this setup only).
        apxchol::omp_sptrsv trsv;
        { scoped_env drop("APXCHOL_FACTOR_DROP", "1e-4"); trsv.setup(L, m); }
        // Reference: dense drop, the drop itself off (the zeroed entries stay
        // stored -- a stored zero solves like an absent entry).
        apxchol::omp_sptrsv rf;
        { scoped_drop_off off; rf.setup(ref.Lz, m); }

        // Counts.
        std::uint64_t L11_nnz = 0;
        for (node_index j = 0; j < m; ++j)
            for (edge_index p = L.outer_[j]; p < L.outer_[j + 1]; ++p) L11_nnz += L.inner_[p] < m;
        const auto& st = trsv.lowprec_stats();
        EXPECT_EQ(st.nnz_factor, L11_nnz);
        EXPECT_EQ(st.nnz_stored, ref.kept);
        EXPECT_EQ(trsv.stored_nnz(), ref.kept);
        EXPECT_EQ(st.dropped, ref.zeroed);
        EXPECT_EQ(st.dropped, st.dropped_threshold + st.dropped_flush);
        EXPECT_EQ(st.offdiag + static_cast<std::uint64_t>(m), ref.kept);
        EXPECT_EQ(trsv.csc_vals().size(), ref.kept);
        EXPECT_EQ(trsv.csr_vals().size(), ref.kept);
        EXPECT_EQ(trsv.csc_row_idx().size(), ref.kept);
        EXPECT_EQ(trsv.csr_col_idx().size(), ref.kept);
        EXPECT_EQ(trsv.csc_col_ptr().back(), static_cast<edge_index>(ref.kept));
        EXPECT_EQ(trsv.csr_row_ptr().back(), static_cast<edge_index>(ref.kept));
        // No stored zeros survive the drop; every kept off-diagonal is above the threshold.
        {
            const std::vector<float> s = reference_scales_L11(L, m);
            std::uint64_t zeros = 0, below = 0;
            for (node_index j = 0; j < m; ++j) {
                ASSERT_EQ(trsv.csc_row_idx()[trsv.csc_col_ptr()[j]], j) << "diagonal first, col " << j;
                for (edge_index p = trsv.csc_col_ptr()[j] + 1; p < trsv.csc_col_ptr()[j + 1]; ++p) {
                    zeros += apxchol::widen(trsv.csc_vals()[p]) == 0.0;
                    // Recover |v| >= rel * s_j from the stored value: under
                    // FP16_SCALED the stored ratio is >= rel up to rounding.
                    below += std::fabs(apxchol::widen(trsv.csc_vals()[p])) < rel * static_cast<double>(s[j]) * 0.99;
                }
            }
            for (node_index i = 0; i < m; ++i)
                ASSERT_EQ(trsv.csr_col_idx()[trsv.csr_row_ptr()[i + 1] - 1], i) << "diagonal last, row " << i;
            EXPECT_EQ(zeros, 0u);
            EXPECT_EQ(below, 0u);
        }
        // Reference has the same kept structure count-wise (its stored nnz is the full L11).
        EXPECT_EQ(rf.lowprec_stats().nnz_stored, L11_nnz);
        EXPECT_EQ(rf.lowprec_stats().dropped, 0u);

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
        // And the compacted forward solve is an exact solve of the STORED
        // kernel matrix L~ (widened off-diagonals, stored_diag() diagonal;
        // double accumulation): componentwise residual at roundoff. (Under
        // *_SCALED y1 is y' = D y and L~ = L D^-1, see omp.h.)
        const std::vector<float> s = reference_scales_L11(L, m);
        double worst = 0.0;
        for (node_index i = 0; i < m; ++i) {
            double r = x[i], sc = std::fabs(x[i]);
            for (edge_index q = trsv.csr_row_ptr()[i]; q < trsv.csr_row_ptr()[i + 1]; ++q) {
                double v = apxchol::widen(trsv.csr_vals()[q]);
                const node_index j = trsv.csr_col_idx()[q];
                const double t = v * y1[j];
                r -= t; sc += std::fabs(t);
            }
            worst = std::max(worst, std::fabs(r) / (sc + 1e-300));
        }
        EXPECT_LT(worst, 1e-11);

        // The env is read at every setup: rel = 0 -> nothing is dropped, and
        // unset -> the default kFactorDropRelDefault (== the 1e-4 above, so the
        // same kept set) with the compensation this test switched off.
        apxchol::omp_sptrsv plain;
        { scoped_drop_off off; plain.setup(L, m); }
        EXPECT_EQ(plain.lowprec_stats().rel, 0.0);
        EXPECT_EQ(plain.lowprec_stats().dropped, 0u);
        EXPECT_EQ(plain.lowprec_stats().nnz_stored, L11_nnz);
        EXPECT_EQ(plain.stored_nnz(), L11_nnz);
        apxchol::omp_sptrsv dflt;
        { scoped_env unset("APXCHOL_FACTOR_DROP", nullptr); dflt.setup(L, m); }
        EXPECT_EQ(dflt.lowprec_stats().rel, apxchol::kFactorDropRelDefault);
        EXPECT_EQ(dflt.stored_nnz(), ref.kept);
    }
}

// rel = 0 / unset / negative = off; the diagonal survives ANY rel (even > 1,
// which drops every off-diagonal), the scales are unchanged by the drop, and
// on FP16_SCALED a tiny rel still removes what fp16 flushes (a stored zero is
// a stored entry) -- the drop's second criterion.
TEST(LowPrec, FactorDropEdgeCases) {
    // These state the DEFAULT (fp32) storage contract, so pin it: the suite
    // is also run with APXCHOL_SPTRSV_FP16=1 in the environment.
    scoped_env fp32_storage("APXCHOL_SPTRSV_FP16", "0");

    sparse_csc L = make_random_lower(2000, 4.0, 31, -1.0, 1.0);
    for (edge_index p = 0; p < L.nonZeros(); ++p)
        if ((p % 5) == 2) L.vals_[p] = static_cast<factor_value_t>(L.vals_[p] * 1e-9);   // fp16-flush range
    const std::vector<float> s = reference_scales(L);
    std::uint64_t offdiag = 0, fp16_zero = 0;
    for (node_index j = 0; j < 2000; ++j)
        for (edge_index p = L.outer_[j] + 1; p < L.outer_[j + 1]; ++p) {
            ++offdiag;
            const fp16_t h(static_cast<float>(L.vals_[p]) / s[j]);
            fp16_zero += fp16_t::is_zero(h.bits) || fp16_t::is_subnormal(h.bits);
        }
    ASSERT_GT(fp16_zero, 100u);
    for (const char* rel : {"0", "-1", "abc"}) {
        SCOPED_TRACE(std::string("APXCHOL_FACTOR_DROP=") + rel);
        setenv("APXCHOL_FACTOR_DROP", rel, 1);
        apxchol::omp_sptrsv t; t.setup(L, 2000);
        unsetenv("APXCHOL_FACTOR_DROP");
        EXPECT_EQ(t.lowprec_stats().dropped, 0u);
        EXPECT_EQ(t.stored_nnz(), static_cast<std::uint64_t>(L.nonZeros()));
    }
    {
        setenv("APXCHOL_FACTOR_DROP", "1e-30", 1);   // below everything: only format zeros go
        apxchol::omp_sptrsv t; t.setup(L, 2000);
        unsetenv("APXCHOL_FACTOR_DROP");
        EXPECT_EQ(t.lowprec_stats().dropped, 0u);
        EXPECT_EQ(t.stored_nnz(), static_cast<std::uint64_t>(L.nonZeros()) - t.lowprec_stats().dropped);
    }
    {
        setenv("APXCHOL_FACTOR_DROP", "2", 1);        // > 1: every off-diagonal goes, diagonal stays
        apxchol::omp_sptrsv t; t.setup(L, 2000);
        unsetenv("APXCHOL_FACTOR_DROP");
        EXPECT_EQ(t.lowprec_stats().dropped, offdiag);
        EXPECT_EQ(t.stored_nnz(), 2000u);
        EXPECT_EQ(t.lowprec_stats().offdiag, 0u);
        for (node_index j = 0; j < 2000; ++j) {
            ASSERT_EQ(t.csc_col_ptr()[j], static_cast<edge_index>(j));
            ASSERT_EQ(t.csc_row_idx()[j], j);
        }
        // Solving with a diagonal factor: y' = x / stored diagonal (L_jj, or
        // L_jj / s_j under FP16_SCALED where forward_solve returns y' = D y).
        std::vector<double> x(2000, 1.0), y(2000);
        t.forward_solve(x.data(), y.data());
        for (node_index j = 0; j < 2000; ++j) {
            const double d = apxchol::omp_sptrsv::stored_diag(L.vals_[L.outer_[j]], s[j]);   // s ignored off FP16_SCALED
            ASSERT_NEAR(y[j], 1.0 / d, 1e-15 * std::fabs(1.0 / d)) << j;
        }
    }
}

// ── The fp16 STORAGE at runtime (APXCHOL_SPTRSV_FP16=1) ─────────────────
// The storage is a per-setup choice now, not a build flag, so these state its
// contract against the same binary that runs the fp32 tests above:
//   * every stored slot is narrow_value<fp16_t>(v, s_j) bit-for-bit, in the
//     CSC and in its CSR twin, with fp16 subnormals flushed to signed zero;
//   * col_scales() are the documented per-column maxima;
//   * the fp32 diagonal is stored_diag<fp16_t>() PLUS the column's
//     storage-rounding residual (the compensation is unconditional), so the
//     STORED column sums to what the fp32 column / s_j sums to;
//   * a column whose scale cannot be represented falls back to s_j = 1 and
//     the setup still produces finite, nonzero diagonals (it THROWS
//     otherwise -- see omp.h "DEGENERATE SCALES");
//   * the env is the only switch, and the retired GPU-only name is an alias.
namespace {
bool fp16_available() { return apxchol::omp_sptrsv::fp16_supported(); }
}  // namespace

TEST(LowPrecFp16, StorageContractAndCompensatedDiagonal) {
    if (!fp16_available()) GTEST_SKIP() << "build has no F16C: fp16 storage not compiled";
    scoped_drop_off drop_off;                              // storage contract on the un-dropped factor
    scoped_env fp16("APXCHOL_SPTRSV_FP16", "1");
    for (node_index m : {node_index(3000), node_index(60000) /* parallel transpose */}) {
        SCOPED_TRACE("m=" + std::to_string(m));
        sparse_csc L = make_random_lower(m, 4.0, 77, -2.0, 2.0);
        for (edge_index p = 0; p < L.nonZeros(); ++p)
            if ((p % 7) == 3) L.vals_[p] = static_cast<factor_value_t>(L.vals_[p] * 1e-7);
        const std::vector<float> s = reference_scales(L);

        apxchol::omp_sptrsv trsv;
        trsv.setup(L, m);
        ASSERT_TRUE(trsv.fp16());
        EXPECT_EQ(trsv.value_bytes(), 2u);
        ASSERT_EQ(trsv.csc_vals16().size(), static_cast<size_t>(L.nonZeros()));
        EXPECT_EQ(trsv.csc_vals().size(), 0u);             // the fp32 arrays stay empty
        ASSERT_EQ(trsv.col_scales().size(), static_cast<size_t>(m));
        for (node_index j = 0; j < m; ++j) ASSERT_EQ(trsv.col_scales()[j], s[j]) << "j=" << j;

        std::uint64_t offdiag = 0;
        double worst_colsum = 0.0;
        for (node_index j = 0; j < m; ++j) {
            double x_sum = 0.0, stored_sum = 0.0, resid = 0.0;
            for (edge_index p = L.outer_[j]; p < L.outer_[j + 1]; ++p) {
                const fp16_t expect = apxchol::omp_sptrsv::narrow_value<fp16_t>(L.vals_[p], s[j]);
                ASSERT_TRUE(trsv.csc_vals16()[p] == expect) << "csc p=" << p;
                EXPECT_FALSE(fp16_t::is_subnormal(trsv.csc_vals16()[p].bits));   // always flushed
                const node_index i = L.inner_[p];
                bool found = false;
                for (edge_index q = trsv.csr_row_ptr()[i]; q < trsv.csr_row_ptr()[i + 1]; ++q)
                    if (trsv.csr_col_idx()[q] == j) {
                        ASSERT_TRUE(trsv.csr_vals16()[q] == expect) << "csr q=" << q;
                        found = true;
                        break;
                    }
                ASSERT_TRUE(found);
                if (i == j) continue;
                ++offdiag;
                const double xv = static_cast<double>(L.vals_[p]) / static_cast<double>(s[j]);
                x_sum      += xv;
                stored_sum += apxchol::widen(expect);
                resid      += xv - apxchol::widen(expect);
            }
            // The compensated diagonal, and the column sum it restores.
            const double d0 = apxchol::omp_sptrsv::stored_diag<fp16_t>(L.vals_[L.outer_[j]], s[j]);
            ASSERT_EQ(trsv.fp16_diag()[j], static_cast<float>(d0 + resid)) << "diag " << j;
            const double abs_scale = std::fabs(d0) + std::fabs(x_sum) + 1e-300;
            worst_colsum = std::max(worst_colsum,
                std::fabs((static_cast<double>(trsv.fp16_diag()[j]) + stored_sum) - (d0 + x_sum)) / abs_scale);
        }
        EXPECT_EQ(trsv.lowprec_stats().offdiag, offdiag);
        EXPECT_LT(worst_colsum, 4e-7);                     // one fp32 rounding of the diagonal
        // The 1e-7-scaled entries sit ~2^-23 below their column max: fp16 must
        // have flushed a good fraction of the off-diagonals.
        EXPECT_GT(trsv.lowprec_stats().flushed, offdiag / 20);
        EXPECT_EQ(trsv.lowprec_stats().subnormal, 0u);
        EXPECT_EQ(trsv.lowprec_stats().scale_fallback, 0u);
    }
}

TEST(LowPrecFp16, DegenerateColumnScaleFallsBackToOneAndTheSetupStaysFinite) {
    if (!fp16_available()) GTEST_SKIP() << "build has no F16C: fp16 storage not compiled";
    scoped_drop_off drop_off;
    scoped_env fp16("APXCHOL_SPTRSV_FP16", "1");
    // Column 1's only off-diagonal is ~1e-40: s_j is an fp32 subnormal and
    // fp32(1 / s_j) overflows, so the setup must fall back to s_j = 1.
    sparse_csc L;
    L.n_ = 3;
    L.outer_ = {0, 2, 4, 5};
    L.inner_ = {0, 2,  1, 2,  2};
    L.vals_  = {2.0f, -0.5f,  3.0f, 1e-40f,  4.0f};
    apxchol::omp_sptrsv trsv;
    ASSERT_NO_THROW(trsv.setup(L, 3));
    ASSERT_TRUE(trsv.fp16());
    EXPECT_EQ(trsv.lowprec_stats().scale_fallback, 1u);
    EXPECT_EQ(trsv.col_scales()[1], 1.0f);                 // the fallback
    EXPECT_EQ(trsv.col_scales()[0], 0.5f);                 // untouched
    for (node_index j = 0; j < 3; ++j) {
        EXPECT_TRUE(std::isfinite(trsv.fp16_diag()[j])) << j;
        EXPECT_NE(trsv.fp16_diag()[j], 0.0f) << j;
    }
    // The 1e-40 entry is below anything fp16 can hold next to that diagonal:
    // it stores as (signed) zero.
    EXPECT_EQ(apxchol::widen(trsv.csc_vals16()[3]), 0.0);
    // ... and the solve runs.
    std::vector<double> x = {1.0, -2.0, 0.5}, y(3), z(3);
    trsv.forward_solve(x.data(), y.data());
    trsv.transpose_solve(y.data(), z.data());
    for (node_index j = 0; j < 3; ++j) { EXPECT_TRUE(std::isfinite(y[j])); EXPECT_TRUE(std::isfinite(z[j])); }
}

TEST(LowPrecFp16, TheEnvIsTheOnlySwitchAndTheRetiredGpuNameIsAnAlias) {
    sparse_csc L;
    L.n_ = 2; L.outer_ = {0, 2, 3}; L.inner_ = {0, 1, 1}; L.vals_ = {2.0f, -0.5f, 3.0f};
    auto storage_of = [&](const char* var, const char* value) {
        scoped_env a("APXCHOL_SPTRSV_FP16", nullptr);
        scoped_env b("APXCHOL_GPU_SPTRSV_FP16", nullptr);
        scoped_env c(var, value);
        apxchol::omp_sptrsv t; t.setup(L, 2);
        return t.fp16();
    };
    // Unset = OFF on the CPU (the GPU's default is the opposite: cuda.h).
    EXPECT_FALSE(storage_of("APXCHOL_SPTRSV_FP16", nullptr));
    EXPECT_FALSE(storage_of("APXCHOL_SPTRSV_FP16", "0"));
    EXPECT_EQ(storage_of("APXCHOL_SPTRSV_FP16", "1"), fp16_available());
    // The retired GPU-only name is read as an alias of the unified one.
    EXPECT_EQ(storage_of("APXCHOL_GPU_SPTRSV_FP16", "1"), fp16_available());
    EXPECT_FALSE(storage_of("APXCHOL_GPU_SPTRSV_FP16", "0"));
    // The new name wins over the alias.
    {
        scoped_env a("APXCHOL_SPTRSV_FP16", "0");
        scoped_env b("APXCHOL_GPU_SPTRSV_FP16", "1");
        apxchol::omp_sptrsv t; t.setup(L, 2);
        EXPECT_FALSE(t.fp16());
    }
}
