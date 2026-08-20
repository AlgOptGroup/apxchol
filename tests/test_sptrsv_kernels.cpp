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

// The DEFAULT storage width is fp32, and value_name / value_bytes (what the
// APXCHOL_VERBOSE banner prints) report what the last setup() resolved --
// fp32 unset, fp16 under APXCHOL_SPTRSV_FP16=1 where the build has F16C.
TEST(SpTRSVKernels, StorageWidthFollowsTheRuntimeSwitch) {
    EXPECT_EQ(sizeof(sptrsv_value_t), 4u);
    EXPECT_TRUE((std::is_same_v<sptrsv_value_t, factor_value_t>));
    EXPECT_TRUE((std::is_same_v<factor_value_t, float>));
    // widen() of every storage type this build can see is the exact value.
    EXPECT_EQ(apxchol::widen(0.75f), 0.75);
    EXPECT_EQ(apxchol::widen(0.75), 0.75);
    EXPECT_EQ(apxchol::widen(apxchol::fp16_t(0.75f)), 0.75);

    sparse_csc L; L.n_ = 2; L.outer_ = {0, 2, 3}; L.inner_ = {0, 1, 1}; L.vals_ = {2.0f, -0.5f, 3.0f};
    {
        scoped_env off("APXCHOL_SPTRSV_FP16", "0");
        apxchol::omp_sptrsv t; t.setup(L, 2);
        EXPECT_FALSE(t.fp16());
        EXPECT_EQ(t.value_bytes(), 4u);
        EXPECT_STREQ(t.value_name(), "float (fp32)");
        EXPECT_EQ(t.csc_vals().size(), 3u);
        EXPECT_EQ(t.csc_vals16().size(), 0u);
    }
    {
        scoped_env on("APXCHOL_SPTRSV_FP16", "1");
        apxchol::omp_sptrsv t; t.setup(L, 2);
        // Without F16C the env falls back to fp32 with a note (portable builds).
        EXPECT_EQ(t.fp16(), apxchol::omp_sptrsv::fp16_supported());
        EXPECT_EQ(t.value_bytes(), apxchol::omp_sptrsv::fp16_supported() ? 2u : 4u);
        if (apxchol::omp_sptrsv::fp16_supported()) {
            EXPECT_STREQ(t.value_name(), "fp16 (per-column scaled)");
            EXPECT_EQ(t.csc_vals16().size(), 3u);
            EXPECT_EQ(t.csc_vals().size(), 0u);
        }
    }
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

// THE storage under test, as a runtime flag: what APXCHOL_SPTRSV_FP16
// resolves to for this build (fp16 only where F16C exists).
bool fp16_storage(const char* env_value) {
    return std::string(env_value) == "1" && apxchol::omp_sptrsv::fp16_supported();
}
// The KERNEL matrix L~ as omp_sptrsv::setup stores it (widened, NOT
// rescaled) for the OFF-DIAGONAL entry with value v in a column with scale s:
// through narrow_value. On the fp32 storage this is v itself.
double kernel_offdiag(factor_value_t v, float s, bool fp16) {
    return fp16 ? apxchol::widen(apxchol::omp_sptrsv::narrow_value<apxchol::fp16_t>(v, s))
                : apxchol::widen(apxchol::omp_sptrsv::narrow_value<float>(v, s));
}
// The per-column scale the kernels' input / output carry (1.0f on fp32).
float pair_scale(const sparse_csc& L, node_index j, bool fp16) {
    if (!fp16) return 1.0f;
    return apxchol::omp_sptrsv::column_scale(L.vals_.data(), L.outer_[j], L.outer_[j + 1]);
}
// What the kernels DIVIDE by for column j: the factor diagonal on fp32; on
// fp16 the fp32 stored_diag PLUS the column's storage-rounding residual (the
// compensation omp_sptrsv::setup applies unconditionally -- omp.h).
double kernel_diag(const sparse_csc& L, node_index j, bool fp16) {
    const edge_index p0 = L.outer_[j];
    if (!fp16) return apxchol::omp_sptrsv::stored_diag<float>(L.vals_[p0], 1.0f);
    const float s = pair_scale(L, j, true);
    double d = apxchol::omp_sptrsv::stored_diag<apxchol::fp16_t>(L.vals_[p0], s);
    double resid = 0.0;
    for (edge_index p = p0 + 1; p < L.outer_[j + 1]; ++p)
        resid += static_cast<double>(L.vals_[p]) / static_cast<double>(s)
               - kernel_offdiag(L.vals_[p], s, true);
    return static_cast<float>(d + resid);
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
                           const std::vector<double>& y, bool transpose, bool fp16) {
    const node_index m = L.rows();
    std::vector<double> r(b), scale(m, 0.0);
    for (node_index i = 0; i < m; ++i) scale[i] = std::fabs(b[i]);
    for (node_index j = 0; j < m; ++j) {
        const float s_j = pair_scale(L, j, fp16);
        for (edge_index p = L.outer_[j]; p < L.outer_[j + 1]; ++p) {
            const node_index i = L.inner_[p];
            const double v = (i == j) ? kernel_diag(L, j, fp16)
                                      : kernel_offdiag(L.vals_[p], s_j, fp16);
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
                    std::vector<double>& y, std::vector<double>& z, bool fp16) {
    const node_index m = L.rows();
    // Column-oriented forward substitution.
    y = x;
    for (node_index j = 0; j < m; ++j) {
        const float s_j = pair_scale(L, j, fp16);
        const double sd = static_cast<double>(s_j);
        const edge_index p0 = L.outer_[j];
        y[j] /= kernel_diag(L, j, fp16) * sd;
        for (edge_index p = p0 + 1; p < L.outer_[j + 1]; ++p)
            y[L.inner_[p]] -= kernel_offdiag(L.vals_[p], s_j, fp16) * sd * y[j];
    }
    // Column-oriented back substitution (L_s^T z = y: column j of L_s is row j of L_s^T).
    z.assign(m, 0.0);
    for (node_index jj = m; jj-- > 0; ) {
        const node_index j = jj;
        const float s_j = pair_scale(L, j, fp16);
        const double sd = static_cast<double>(s_j);
        const edge_index p0 = L.outer_[j];
        double acc = y[j];
        for (edge_index p = p0 + 1; p < L.outer_[j + 1]; ++p)
            acc -= kernel_offdiag(L.vals_[p], s_j, fp16) * sd * z[L.inner_[p]];
        z[j] = acc / (kernel_diag(L, j, fp16) * sd);
    }
}

void run_kernel_precision_check() {
    for (const char* env : {"0", "1"})
    for (node_index m : {node_index(3000), node_index(60000) /* parallel transpose path */}) {
        const bool fp16 = fp16_storage(env);
        SCOPED_TRACE("m=" + std::to_string(m) + " APXCHOL_SPTRSV_FP16=" + env);
        scoped_env storage("APXCHOL_SPTRSV_FP16", env);
        sparse_csc L = make_random_lower(m, 4.0, 99);
        apxchol::omp_sptrsv trsv;
        trsv.setup(L, m);
        ASSERT_EQ(trsv.fp16(), fp16);
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> ux(-1.0, 1.0);
        std::vector<double> x(m), yp(m), z(m);
        for (auto& v : x) v = ux(rng);
        // (i) What the kernels compute, at roundoff: L~ y' = x, then
        //     L~^T z = R y'.
        trsv.forward_solve(x.data(), yp.data());
        check_kernel_residual(L, x, yp, /*transpose=*/false, fp16);
        trsv.transpose_solve(yp.data(), z.data());
        std::vector<double> ryp(m);
        for (node_index j = 0; j < m; ++j) {
            const double r = fp16 ? apxchol::omp_sptrsv::inv_scale<apxchol::fp16_t>(pair_scale(L, j, true)) : 1.0;
            ryp[j] = yp[j] * (r * r);
        }
        check_kernel_residual(L, ryp, z, /*transpose=*/true, fp16);
        // (ii) The pair contract against the serial reference on L_s = L~ D:
        //      y' = D y at roundoff, z = L_s^-T y up to r_j = fp32(1/s_j)
        //      (2^-23 relative on the input, i.e. on z; exact on fp32 storage).
        std::vector<double> y_ref, z_ref;
        reference_pair(L, x, y_ref, z_ref, fp16);
        double worst_y = 0.0, worst_z = 0.0, sc_y = 0.0, sc_z = 0.0;
        for (node_index j = 0; j < m; ++j) {
            worst_y = std::max(worst_y, std::fabs(yp[j] - static_cast<double>(pair_scale(L, j, fp16)) * y_ref[j]));
            worst_z = std::max(worst_z, std::fabs(z[j] - z_ref[j]));
            sc_y = std::max(sc_y, std::fabs(yp[j]));
            sc_z = std::max(sc_z, std::fabs(z_ref[j]));
        }
        EXPECT_LT(worst_y, 1e-10 * sc_y) << "y' = D y";
        // z carries R = diag(r_j^2), r_j = fp32(1/s_j): a 2^-23-relative
        // perturbation of the back solve's input under the fp16 storage
        // (D = I on fp32, where the pair is exact to roundoff).
        EXPECT_LT(worst_z, (fp16 ? 1e-6 : 1e-10) * sc_z) << "z";
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
// run -- on 16-bit storage the SIMD ones (simd_fp16_kernel()): 8-wide vector
// widen, 4-wide step, scalar tail. CSR row lengths are spread over 0..48 so
// every path (8-blocks, the 4-step, tails of 0..3) is exercised. Run at BOTH
// storages (APXCHOL_SPTRSV_FP16=0|1) and checked exactly like the thin-level
// kernels: the pair contract at roundoff against L~ / R and against the
// serial reference on L_s.
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
    for (const char* env : {"0", "1"}) {
        const bool fp16 = fp16_storage(env);
        SCOPED_TRACE(std::string("APXCHOL_SPTRSV_FP16=") + env);
        scoped_env storage("APXCHOL_SPTRSV_FP16", env);
        std::vector<double> y_ref, z_ref;
        reference_pair(L, x, y_ref, z_ref, fp16);
        apxchol::omp_sptrsv trsv;
        trsv.set_round_bounds(bounds);
        trsv.setup(L, m);
        ASSERT_EQ(trsv.fp16(), fp16);
#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
        EXPECT_TRUE(apxchol::omp_sptrsv::simd_fp16_kernel());
#else
        EXPECT_FALSE(apxchol::omp_sptrsv::simd_fp16_kernel());
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
        check_kernel_residual(L, x, yp, /*transpose=*/false, fp16);
        trsv.transpose_solve(yp.data(), z.data());
        std::vector<double> ryp(m);
        for (node_index j = 0; j < m; ++j) {
            const double r = fp16 ? apxchol::omp_sptrsv::inv_scale<apxchol::fp16_t>(pair_scale(L, j, true)) : 1.0;
            ryp[j] = yp[j] * (r * r);
        }
        check_kernel_residual(L, ryp, z,  /*transpose=*/true, fp16);
        double worst_y = 0.0, worst_z = 0.0, sc_y = 0.0, sc_z = 0.0;
        for (node_index j = 0; j < m; ++j) {
            worst_y = std::max(worst_y, std::fabs(yp[j] - static_cast<double>(pair_scale(L, j, fp16)) * y_ref[j]));
            worst_z = std::max(worst_z, std::fabs(z[j] - z_ref[j]));
            sc_y = std::max(sc_y, std::fabs(yp[j]));
            sc_z = std::max(sc_z, std::fabs(z_ref[j]));
        }
        EXPECT_LT(worst_y, 1e-10 * sc_y) << "y' = D y";
        // z carries R = diag(r_j^2), r_j = fp32(1/s_j): a 2^-23-relative
        // perturbation of the back solve's input under the fp16 storage
        // (D = I on fp32, where the pair is exact to roundoff).
        EXPECT_LT(worst_z, (fp16 ? 1e-6 : 1e-10) * sc_z) << "z";
    }
}
