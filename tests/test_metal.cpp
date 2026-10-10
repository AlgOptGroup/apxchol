// The Apple GPU PCG (include/apxchol/solver/metal_solver.h). Device
// tests skip when metal_solver::available() is false (no device, kernels not
// compiled, or the double-float self-test failed).
#include <gtest/gtest.h>

#include "apxchol/solver/detail/metal_host.h"
#include "apxchol/solver/factorization.h"
#include "apxchol/solver/metal_solver.h"
#include "apxchol/solver/solve.h"
#include "apxchol/solver/sptrsv/factor_drop.h"
#include "metal_schedule_reference.h"
#include "metal_device.h"

#include <Eigen/Sparse>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using Sparse = Eigen::SparseMatrix<double>;
using apxchol::node_index;
namespace mh = apxchol::detail::metal_host;
namespace ls = apxchol::detail::metal_schedule;

#define REQUIRE_METAL()                                                       \
    do {                                                                      \
        if (!apxchol::metal_solver::available()) GTEST_SKIP() << "no usable Metal device"; \
    } while (0)

// Grid Laplacian (shift = 0) or SDDM (shift > 0) with weights 1 + k/4 (all
// fp32-exact) or, with `inexact`, weights 1 + k/3 (not fp32-exact).
Sparse grid(int rows, int cols, double shift = 0.0, bool inexact = false) {
    const int n = rows * cols;
    std::vector<Eigen::Triplet<double>> t;
    std::vector<double> deg(n, shift);
    auto id = [&](int r, int c) { return r * cols + c; };
    auto edge = [&](int a, int b, double w) {
        t.emplace_back(a, b, -w);
        t.emplace_back(b, a, -w);
        deg[a] += w;
        deg[b] += w;
    };
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const int k = (r * 7 + c * 3) % 5;
            const double w = inexact ? 1.0 + k / 3.0 : 1.0 + 0.25 * k;
            if (r + 1 < rows) edge(id(r, c), id(r + 1, c), w);
            if (c + 1 < cols) edge(id(r, c), id(r, c + 1), w);
        }
    for (int i = 0; i < n; ++i) t.emplace_back(i, i, deg[i]);
    Sparse L(n, n);
    L.setFromTriplets(t.begin(), t.end());
    L.makeCompressed();
    return L;
}

// A grid Laplacian plus `hubs` vertices joined to every (37 + h)-th grid
// vertex: its factor has heavy rows (more than 32 dependencies).
Sparse hub_graph(int rows, int cols, int hubs) {
    const int g = rows * cols, n = g + hubs;
    std::vector<Eigen::Triplet<double>> t;
    std::vector<double> deg(n, 0.0);
    auto edge = [&](int a, int b, double w) {
        t.emplace_back(a, b, -w);
        t.emplace_back(b, a, -w);
        deg[a] += w;
        deg[b] += w;
    };
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            if (r + 1 < rows) edge(r * cols + c, (r + 1) * cols + c, 1.0 + ((r + c) % 3) / 3.0);
            if (c + 1 < cols) edge(r * cols + c, r * cols + c + 1, 1.0);
        }
    for (int h = 0; h < hubs; ++h)
        for (int v = h; v < g; v += 37 + h) edge(g + h, v, 0.5);
    for (int i = 0; i < n; ++i) t.emplace_back(i, i, deg[i]);
    Sparse L(n, n);
    L.setFromTriplets(t.begin(), t.end());
    L.makeCompressed();
    return L;
}

Eigen::VectorXd rhs(Eigen::Index n, unsigned salt, bool centred) {
    Eigen::VectorXd b(n);
    for (Eigen::Index i = 0; i < n; ++i)
        b[i] = std::sin(0.37 * static_cast<double>(i + 1) * salt) + 0.1 * salt;
    if (centred) b.array() -= b.mean();
    return b;
}

bool same_bytes(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), static_cast<std::size_t>(a.size()) * sizeof(double)) == 0;
}

double true_residual(const Sparse& A, const Eigen::VectorXd& b, const Eigen::VectorXd& x) {
    return (b - A * x).norm() / b.norm();
}

apxchol::solve_options keep_opts() {
    apxchol::solve_options o;
    o.keep_factor_values = true;
    return o;
}

// The device's preconditioner application emulated on the host from the same
// factor: permute, power-of-two scale, double-float split, (Laplacian) the
// fixed-tree mean, the fp32 sweeps, (Laplacian) the output mean, unscale.
Eigen::VectorXd emulate_apply(const apxchol::factorization& F, const Eigen::VectorXd& r) {
    const std::size_t n = static_cast<std::size_t>(r.size());
    const bool lap = !F.sddm;
    const std::int64_t m = lap ? static_cast<std::int64_t>(n) - 1 : static_cast<std::int64_t>(n);
    const ls::factor_schedules sch = ls::build_factor_schedules(F.L, m, apxchol::factor_drop_rel_from_env());
    std::vector<double> rp(n);
    mh::scatter(r.data(), F.perm.data(), n, rp.data());
    const double scale = mh::pow2_scale(mh::fold_max_abs(rp.data(), n));
    std::vector<mh::df> w(n);
    for (std::size_t q = 0; q < n; ++q) w[q] = mh::split(scale * rp[q]);
    const mh::df inv_n = mh::inverse_count(n);
    float mu_r = 0.0f;
    if (lap) {
        const mh::df sum = mh::tree_reduce(static_cast<std::uint32_t>(n), [&](std::uint32_t q) { return w[q]; });
        mu_r = mh::df_mul(sum, inv_n).hi;
    }
    std::vector<float> in(n), z(n, 0.0f);
    for (std::size_t q = 0; q < n; ++q) in[q] = w[q].hi - mu_r;
    apxchol::test::emulate_sweep(sch.forward, in.data(), z.data());
    apxchol::test::emulate_sweep(sch.backward, z.data(), z.data());
    float mu_z = 0.0f;
    if (lap) {
        const mh::df sum = mh::tree_reduce(static_cast<std::uint32_t>(n),
                                           [&](std::uint32_t q) { return mh::df{z[q], 0.0f}; });
        mu_z = mh::df_mul(sum, inv_n).hi;
    }
    for (std::size_t q = 0; q < n; ++q) rp[q] = static_cast<double>(z[q] - mu_z) / scale;
    Eigen::VectorXd out(r.size());
    mh::gather(rp.data(), F.perm.data(), n, out.data());
    return out;
}

}  // namespace

TEST(MetalDevice, DoubleFloatProbeExact) {
    REQUIRE_METAL();
    // A second, wider set of cases than available()'s self-test.
    std::vector<float> in;
    std::uint32_t state = 12345u;
    auto next = [&] {
        state = state * 1664525u + 1013904223u;
        return state;
    };
    for (int i = 0; i < 4096; ++i) {
        for (int k = 0; k < 2; ++k) {
            const std::uint32_t bits = next();
            const float hi = std::ldexp(1.0f + static_cast<float>(bits & 0xfffff) / 1048576.0f,
                                        static_cast<int>(bits >> 26) - 32) * ((bits & 0x100000) ? -1.0f : 1.0f);
            const mh::df v = mh::quick_two_sum(hi, hi * std::ldexp(static_cast<float>(next() % 1000) / 1000.0f - 0.5f, -24));
            in.push_back(v.hi);
            in.push_back(v.lo);
        }
    }
    const std::vector<float> out = apxchol::detail::metal::run_probe(in);
    ASSERT_EQ(out.size(), in.size() * 4);
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < in.size() / 4; ++i) {
        const mh::df x{in[4 * i], in[4 * i + 1]}, y{in[4 * i + 2], in[4 * i + 3]};
        const mh::df ts = mh::two_sum(x.hi, y.hi), tp = mh::two_prod(x.hi, y.hi);
        const mh::df da = mh::df_add(x, y), dmul = mh::df_mul(x, y), dmf = mh::df_mul_f(x, y.hi);
        const float expect[] = {ts.hi, ts.lo, tp.hi, tp.lo, da.hi, da.lo, dmul.hi, dmul.lo,
                                dmf.hi, dmf.lo, std::fma(x.hi, y.hi, x.lo),
                                std::fma(-x.hi, y.hi, x.lo) * y.lo};
        for (int k = 0; k < 12; ++k)
            mismatches += std::memcmp(&out[16 * i + k], &expect[k], sizeof(float)) != 0;
        EXPECT_EQ(static_cast<double>(out[16 * i]) + static_cast<double>(out[16 * i + 1]),
                  static_cast<double>(x.hi) + static_cast<double>(y.hi));
    }
    EXPECT_EQ(mismatches, 0u);
    std::printf("[ metal ] device %s, contract pragma %s, %zu probe cases bit-identical to the host\n",
                apxchol::detail::metal::status().name.c_str(),
                apxchol::detail::metal::status().contract_pragma ? "accepted" : "not accepted",
                in.size() / 4);
}

TEST(MetalDevice, ApplyMatchesHostEmulationBitForBit) {
    REQUIRE_METAL();
    for (const double shift : {0.0, 0.5}) {
        SCOPED_TRACE(shift == 0.0 ? "Laplacian" : "SDDM");
        const Sparse A = grid(37, 29, shift, true);
        apxchol::factorization F = apxchol::factorize(A);
        const apxchol::factorization Fcopy = F;
        const apxchol::metal_solver slv(A, std::move(F));
        for (unsigned salt : {1u, 4u}) {
            const Eigen::VectorXd r = rhs(A.rows(), salt, shift == 0.0) * std::ldexp(1.0, 9 * static_cast<int>(salt) - 20);
            const Eigen::VectorXd z = slv.apply(r);
            const Eigen::VectorXd ze = emulate_apply(Fcopy, r);
            EXPECT_TRUE(same_bytes(z, ze)) << "max diff " << (z - ze).cwiseAbs().maxCoeff();
        }
    }
}

// Exercise heavy rows, merged narrow runs and wide levels on the device.
TEST(MetalDevice, HeavyRowsAndNarrowRunsMatchEmulation) {
    REQUIRE_METAL();
    const Sparse A = hub_graph(110, 100, 4);
    apxchol::factorization F = apxchol::factorize(A);
    const apxchol::factorization Fcopy = F;
    ASSERT_FALSE(Fcopy.sddm);
    const ls::factor_schedules sch =
        ls::build_factor_schedules(Fcopy.L, A.rows() - 1, apxchol::factor_drop_rel_from_env());
    int kinds[3] = {0, 0, 0};
    for (const ls::level_solve* sv : {&sch.forward, &sch.backward})
        for (const ls::level_step& st : ls::plan_steps(*sv)) ++kinds[static_cast<int>(st.kind)];
    EXPECT_GT(kinds[0], 0) << "wide light steps";
    EXPECT_GT(kinds[1], 0) << "heavy steps (the hubs' forward rows)";
    EXPECT_GT(kinds[2], 0) << "narrow runs";
    const apxchol::metal_solver slv(A, std::move(F));
    const Eigen::VectorXd r = rhs(A.rows(), 2, true);
    EXPECT_TRUE(same_bytes(slv.apply(r), emulate_apply(Fcopy, r)));

}

// A synthetic factor with heavy rows in both sweeps (hub columns and hub
// rows), on the SDDM (m = n) and Laplacian (m = n - 1) paths.
TEST(MetalDevice, SyntheticHeavyFactorApplyMatchesEmulation) {
    REQUIRE_METAL();
    const node_index n = 5000;
    apxchol::factorization F;
    std::mt19937 rng(19);
    std::uniform_real_distribution<double> mag(-6.0, 0.0), diag(1.0, 3.0);
    std::vector<std::vector<std::pair<node_index, float>>> cols(n);
    for (node_index j = 0; j < n; ++j) {
        cols[j].emplace_back(j, static_cast<float>(diag(rng)));
        if (j + 1 >= n) continue;
        std::uniform_int_distribution<node_index> below(j + 1, n - 1);
        const bool hub = j % 173 == 0;
        for (int t = 0; t < (hub ? 70 : 3); ++t)
            cols[j].emplace_back(below(rng), static_cast<float>(-(hub ? 0.01 : 0.1) * std::pow(10.0, mag(rng))));
    }
    for (node_index i = 89; i < n; i += 241) {
        std::uniform_int_distribution<node_index> above(0, i - 1);
        for (int t = 0; t < 60; ++t)
            cols[above(rng)].emplace_back(i, static_cast<float>(-0.01 * std::pow(10.0, mag(rng))));
    }
    F.L.n_ = n;
    F.L.outer_.assign(n + 1, 0);
    for (node_index j = 0; j < n; ++j) {
        auto& c = cols[j];
        std::sort(c.begin() + 1, c.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        c.erase(std::unique(c.begin() + 1, c.end(), [](const auto& a, const auto& b) { return a.first == b.first; }),
                c.end());
        F.L.outer_[j + 1] = F.L.outer_[j] + static_cast<apxchol::edge_index>(c.size());
        for (const auto& [i, v] : c) {
            F.L.inner_.push_back(i);
            F.L.vals_.push_back(v);
        }
    }
    F.perm.resize(n);
    for (node_index v = 0; v < n; ++v) F.perm[v] = (v * 7 + 3) % n;
    for (const bool sddm : {true, false}) {
        SCOPED_TRACE(sddm ? "SDDM path" : "Laplacian path");
        F.sddm = sddm;
        const ls::factor_schedules sch = ls::build_factor_schedules(
            F.L, sddm ? n : n - 1, apxchol::factor_drop_rel_from_env());
        for (const ls::level_solve* sv : {&sch.forward, &sch.backward}) {
            bool heavy = false;
            for (const ls::level_step& st : ls::plan_steps(*sv)) heavy |= st.kind == ls::step_kind::heavy;
            EXPECT_TRUE(heavy);
        }
        // The operator only has to match in size for an application.
        const Sparse A = grid(50, 100, sddm ? 0.5 : 0.0);
        const apxchol::metal_solver slv(A, apxchol::factorization(F));
        for (unsigned salt : {1u, 6u}) {
            const Eigen::VectorXd r = rhs(A.rows(), salt, !sddm);
            EXPECT_TRUE(same_bytes(slv.apply(r), emulate_apply(F, r)));
        }
    }
}

TEST(MetalDevice, ApplyAgreesWithCpuApply) {
    REQUIRE_METAL();
    for (const double shift : {0.0, 0.25}) {
        SCOPED_TRACE(shift == 0.0 ? "Laplacian" : "SDDM");
        const Sparse A = grid(40, 40, shift);
        apxchol::factorization F = apxchol::factorize(A);
        const apxchol::cpu_solver cpu(A, apxchol::factorization(F));
        const apxchol::metal_solver gpu(A, std::move(F));
        const Eigen::VectorXd r = rhs(A.rows(), 3, shift == 0.0);
        // The CPU's center-k schedule centres only some applications; centre
        // its input and output here, as the device does every time.
        Eigen::VectorXd rc = r;
        if (shift == 0.0) rc.array() -= rc.mean();
        Eigen::VectorXd zc = cpu.apply(rc);
        if (shift == 0.0) zc.array() -= zc.mean();
        const Eigen::VectorXd zg = gpu.apply(r);
        const double rel = (zg - zc).norm() / zc.norm();
        std::printf("[ metal ] %s apply: |z_gpu - z_cpu| / |z_cpu| = %.3e\n",
                    shift == 0.0 ? "Laplacian" : "SDDM", rel);
        EXPECT_LE(rel, 1e-5);
    }
}

TEST(MetalDevice, GridLaplacianOriginalResidual) {
    REQUIRE_METAL();
    const Sparse A = grid(48, 48);
    apxchol::factorization F = apxchol::factorize(A);
    const apxchol::cpu_solver cpu(A, apxchol::factorization(F));
    const apxchol::metal_solver gpu(A, std::move(F));
    const Eigen::VectorXd b = rhs(A.rows(), 2, true);
    for (const double tol : {1e-8, 1e-10}) {
        SCOPED_TRACE(tol);
        const apxchol::solve_result r = gpu.solve(b, tol, 500);
        const double eig = true_residual(A, b, r.x);
        EXPECT_LT(r.residual, tol);
        EXPECT_NEAR(r.residual, eig, 1e-3 * eig + 1e-16);
        EXPECT_LE(std::fabs(r.x.mean()), 1e-12 * r.x.norm());
        EXPECT_GT(r.iterations, 0);
        if (tol == 1e-10) {
            const apxchol::solve_result c = cpu.solve(b, tol, 500);
            const double rel = (r.x - c.x).norm() / c.x.norm();
            std::printf("[ metal ] grid 48x48 tol 1e-10: metal %lld iterations residual %.3e, cpu %lld "
                        "iterations, |x_gpu - x_cpu| / |x_cpu| = %.3e\n",
                        static_cast<long long>(r.iterations), r.residual,
                        static_cast<long long>(c.iterations), rel);
            EXPECT_LE(rel, 1e-7);
        }
    }
}

TEST(MetalDevice, SddmExactAndDoubleFloatOperators) {
    REQUIRE_METAL();
    for (const bool inexact : {false, true}) {
        SCOPED_TRACE(inexact ? "double-float operator" : "fp32-exact operator");
        const Sparse A = grid(30, 50, 0.125, inexact);
        const apxchol::metal_solver slv(A);
        EXPECT_EQ(slv.stats().operator_double_float, inexact);
        EXPECT_TRUE(slv.factor().sddm);
        const Eigen::VectorXd b = rhs(A.rows(), 5, false);
        const apxchol::solve_result r = slv.solve(b, 1e-10, 500);
        EXPECT_LT(r.residual, 1e-10);
        EXPECT_NEAR(r.residual, true_residual(A, b, r.x), 1e-3 * r.residual + 1e-16);
    }
}



TEST(MetalDevice, EarlyExitsTruthful) {
    REQUIRE_METAL();
    const Sparse A = grid(12, 12);
    const apxchol::metal_solver slv(A);
    const Eigen::VectorXd b = rhs(A.rows(), 1, true);
    const apxchol::solve_result none = slv.solve(b, 1e-8, 0);
    EXPECT_EQ(none.iterations, 0);
    EXPECT_EQ(none.residual, 1.0);
    EXPECT_EQ(none.x.norm(), 0.0);

    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(A.rows());
    const apxchol::solve_result z = slv.solve(zero);
    EXPECT_EQ(z.iterations, 0);
    EXPECT_EQ(z.residual, 0.0);
    EXPECT_EQ(z.x.norm(), 0.0);

    const Eigen::VectorXd short_x0 = Eigen::VectorXd::Zero(A.rows() - 1);
    EXPECT_THROW(slv.solve(b, 1e-8, 10, &short_x0), std::invalid_argument);
    EXPECT_THROW(slv.solve(Eigen::VectorXd(Eigen::VectorXd::Ones(3))), std::invalid_argument);
    EXPECT_THROW(slv.apply(Eigen::VectorXd(Eigen::VectorXd::Ones(3))), std::invalid_argument);
}

TEST(MetalDevice, BreakdownIsNotConvergence) {
    REQUIRE_METAL();
    // A constant right-hand side is orthogonal to a Laplacian's range: the
    // centred preconditioner returns 0, p.Ap = 0 on the first iteration.
    const Sparse A = grid(10, 10);
    const apxchol::metal_solver slv(A);
    const Eigen::VectorXd b = Eigen::VectorXd::Ones(A.rows());
    const apxchol::solve_result r = slv.solve(b, 1e-8, 50);
    EXPECT_EQ(r.iterations, 0);
    EXPECT_EQ(r.residual, 1.0);
}

TEST(MetalDevice, UnreachableTolStopsHonestly) {
    REQUIRE_METAL();
    const Sparse A = grid(16, 16, 0.0, true);
    const apxchol::metal_solver slv(A);
    const Eigen::VectorXd b = rhs(A.rows(), 7, true);
    const apxchol::solve_result r = slv.solve(b, 1e-30, 300);
    EXPECT_GE(r.residual, 1e-30);
    EXPECT_LE(r.iterations, 300);
    EXPECT_NEAR(r.residual, true_residual(A, b, r.x), 1e-2 * r.residual);
}

TEST(MetalDevice, PowerOfTwoScalingExact) {
    REQUIRE_METAL();
    const Sparse A = grid(15, 21, 0.3, true);
    const apxchol::metal_solver slv(A);
    const Eigen::VectorXd b = rhs(A.rows(), 2, false);
    const apxchol::solve_result base = slv.solve(b, 1e-10, 400);
    for (const int e : {-40, 7}) {
        SCOPED_TRACE(e);
        const Eigen::VectorXd bs = b * std::ldexp(1.0, e);
        const apxchol::solve_result s = slv.solve(bs, 1e-10, 400);
        EXPECT_EQ(s.iterations, base.iterations);
        EXPECT_TRUE(same_bytes(s.x, Eigen::VectorXd(base.x * std::ldexp(1.0, e))));
    }
}

TEST(MetalDevice, RepeatedAndTwinSolversBitIdentical) {
    REQUIRE_METAL();
    const Sparse A = grid(33, 17);
    apxchol::factorization F = apxchol::factorize(A);
    const apxchol::metal_solver one(A, apxchol::factorization(F));
    const apxchol::metal_solver two(A, std::move(F));
    const Eigen::VectorXd b = rhs(A.rows(), 6, true);
    const apxchol::solve_result a1 = one.solve(b, 1e-10, 400);
    const apxchol::solve_result a2 = one.solve(b, 1e-10, 400);
    const apxchol::solve_result b1 = two.solve(b, 1e-10, 400);
    EXPECT_TRUE(same_bytes(a1.x, a2.x));
    EXPECT_TRUE(same_bytes(a1.x, b1.x));
    EXPECT_EQ(a1.iterations, b1.iterations);
    EXPECT_TRUE(same_bytes(one.apply(b), two.apply(b)));
}



TEST(MetalDevice, LaplacianX0ConstantIrrelevant) {
    REQUIRE_METAL();
    const Sparse A = grid(26, 26);
    const apxchol::metal_solver slv(A);
    const Eigen::VectorXd b = rhs(A.rows(), 8, true);
    const Eigen::VectorXd x0 = rhs(A.rows(), 9, false);
    const Eigen::VectorXd x0c = (x0.array() + 5.0).matrix();
    const apxchol::solve_result r1 = slv.solve(b, 1e-10, 500, &x0);
    const apxchol::solve_result r2 = slv.solve(b, 1e-10, 500, &x0c);
    EXPECT_LT(r1.residual, 1e-10);
    EXPECT_LT(r2.residual, 1e-10);
    EXPECT_LE(std::fabs(r1.x.mean()), 1e-12 * r1.x.norm());
    EXPECT_LE(std::fabs(r2.x.mean()), 1e-12 * r2.x.norm());
    EXPECT_LE((r1.x - r2.x).norm() / r1.x.norm(), 1e-6);
}

TEST(MetalDevice, NonFiniteRhsRejected) {
    REQUIRE_METAL();
    const Sparse A = grid(8, 8, 0.5);
    const apxchol::metal_solver slv(A);
    Eigen::VectorXd b = rhs(A.rows(), 1, false);
    b[3] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(slv.solve(b), std::invalid_argument);
    b[3] = std::numeric_limits<double>::infinity();
    EXPECT_THROW(slv.solve(b), std::invalid_argument);
    EXPECT_THROW(slv.apply(b), std::invalid_argument);
    const Eigen::VectorXd ok = rhs(A.rows(), 1, false);
    Eigen::VectorXd x0 = Eigen::VectorXd::Ones(A.rows());
    x0[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(slv.solve(ok, 1e-8, 10, &x0), std::invalid_argument);
}

TEST(MetalDevice, OutOfRangeOperatorIsADomainError) {
    REQUIRE_METAL();
    Sparse A = grid(6, 6, 0.5);
    A.coeffRef(0, 0) = 1e40;
    apxchol::factorization F = apxchol::factorize(grid(6, 6, 0.5));
    EXPECT_THROW(apxchol::metal_solver(A, std::move(F)), std::domain_error);
}

TEST(MetalDevice, TinySystems) {
    REQUIRE_METAL();
    Sparse one(1, 1);
    one.insert(0, 0) = 4.0;
    one.makeCompressed();
    const apxchol::metal_solver s1(one);
    const apxchol::solve_result r1 = s1.solve(Eigen::VectorXd::Constant(1, 2.0));
    EXPECT_EQ(r1.x[0], 0.5);
    EXPECT_EQ(r1.residual, 0.0);
    for (const int n : {2, 3}) {
        SCOPED_TRACE(n);
        const Sparse A = grid(1, n);  // path Laplacian
        const apxchol::metal_solver s(A);
        const Eigen::VectorXd b = rhs(n, 3, true);
        const apxchol::solve_result r = s.solve(b, 1e-12, 20);
        EXPECT_LT(r.residual, 1e-12);
        EXPECT_LE(std::fabs(r.x.mean()), 1e-14 * r.x.norm());
    }
}

TEST(MetalDevice, StatisticsDescribeTheSetup) {
    REQUIRE_METAL();
    const Sparse A = grid(50, 40);
    const apxchol::metal_solver slv(A, keep_opts());
    const auto st = slv.stats();
    EXPECT_EQ(st.n, A.rows());
    EXPECT_GT(st.levels_forward, 0u);
    EXPECT_GT(st.levels_backward, 0u);
    EXPECT_GT(st.steps_forward, 0u);
    EXPECT_FALSE(st.device.empty());
    EXPECT_FALSE(slv.factor().L.vals_.empty());
    EXPECT_EQ(slv.rows(), A.rows());
    const apxchol::metal_solver released(A);
    EXPECT_TRUE(released.factor().L.vals_.empty());
    EXPECT_EQ(released.factor().perm.size(), static_cast<std::size_t>(A.rows()));
}

TEST(MetalDevice, ReusesFactorAcrossDifferentRightHandSides) {
    REQUIRE_METAL();
    const Sparse A = grid(24, 31, 0.25);
    const apxchol::metal_solver solver(A, keep_opts());
    const auto ptr = solver.factor().L.outer_;
    const auto idx = solver.factor().L.inner_;
    const auto vals = solver.factor().L.vals_;
    for (unsigned salt : {2u, 5u, 2u}) {
        const Eigen::VectorXd b = rhs(A.rows(), salt, false);
        const auto result = solver.solve(b, 1e-9, 400);
        EXPECT_EQ(result.backend, apxchol::solve_backend::gpu);
        EXPECT_LT(true_residual(A, b, result.x), 1e-9);
        EXPECT_EQ(solver.factor().L.outer_, ptr);
        EXPECT_EQ(solver.factor().L.inner_, idx);
        EXPECT_EQ(solver.factor().L.vals_, vals);
    }
}

TEST(MetalDevice, CommonSolverApiAndCpuDefault) {
    REQUIRE_METAL();
    const Sparse A = grid(20, 21, 0.25);
    const Eigen::VectorXd b = rhs(A.rows(), 4, false);
    apxchol::solve_options options;
    const auto cpu = apxchol::solve(A, b, options);
    EXPECT_EQ(cpu.backend, apxchol::solve_backend::cpu);
    options.backend = apxchol::solve_backend::gpu;
    const auto gpu = apxchol::solve(A, b, options);
    EXPECT_EQ(gpu.backend, apxchol::solve_backend::gpu);
    EXPECT_LT(true_residual(A, b, gpu.x), options.tol);
    EXPECT_GT(gpu.timings.total("setup"), 0.0);
    EXPECT_GT(gpu.timings.total("pcg"), 0.0);
}

TEST(MetalDevice, UnsupportedOptionsFailBeforeDeviceSetup) {
    const Sparse A = grid(4, 4, 0.25);
    apxchol::solve_options options;
    options.backend = apxchol::solve_backend::cpu;
    EXPECT_THROW(apxchol::metal_solver(A, options), std::invalid_argument);
    options.backend = apxchol::solve_backend::gpu;
    options.factor_opts.factor_storage = apxchol::factor_storage_type::fp16;
    EXPECT_THROW(apxchol::metal_solver(A, options), std::invalid_argument);
}
