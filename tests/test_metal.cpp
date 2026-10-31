// The Apple GPU PCG (include/apxchol/solver/detail/metal_route.h), through
// solver with backend gpu. Device tests skip when backend_available(gpu) is
// false (no device, kernels not compiled, or the arithmetic capability probe
// failed).
#include <gtest/gtest.h>

#include "apxchol/solver/detail/metal_host.h"
#include "apxchol/solver/factorization.h"
#include "apxchol/detail/checkpoint.h"
#include "apxchol/solver/detail/solver_route.h"
#include "apxchol/solver/solve.h"
#include "apxchol/solver/sptrsv/factor_drop.h"
#include "apxchol/solver/sptrsv/gpu_factor_prep.h"
#include "metal_device.h"
#include "apxchol/solver/detail/solve_backend.h"

#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

using csc_access = apxchol::detail::sparse_csc_access;

namespace {

using Sparse = Eigen::SparseMatrix<double>;
using apxchol::node_index;
namespace mh = apxchol::detail::metal_host;

class MetalDevice : public testing::Test {
    void SetUp() override {
        if (!apxchol::backend_available(apxchol::solve_backend::gpu)) GTEST_SKIP() << "no usable Metal device";
    }
};

apxchol::solve_options gpu_options(apxchol::solve_options options = {}) {
    options.backend = apxchol::solve_backend::gpu;
    return options;
}

const apxchol::detail::metal_route& metal(const apxchol::solver& owner) {
    return std::get<apxchol::detail::metal_route>(apxchol::detail::solver_access::route(owner));
}

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

apxchol::solve_options keep_opts() { return {}; }

// Independent FP64 sweeps of the stored FP32 factor, with host centering at
// the public apply boundary. Device scheduling may change the rounding order.
// An installed factor (solver::factor()) is swept as Metal stores it; a
// construction factor first goes through the drop and storage options.
Eigen::VectorXd emulate_apply(const apxchol::factorization& F, const Eigen::VectorXd& r,
                             const apxchol::factor_options& options = {}) {
    const bool installed = F.storage != apxchol::factor_storage_type::automatic;
    const bool fp16 = (installed ? F.storage : options.factor_storage) ==
                      apxchol::factor_storage_type::fp16;
    const std::size_t n = static_cast<std::size_t>(r.size());
    const bool lap = !F.sddm;
    const std::int64_t m = lap ? static_cast<std::int64_t>(n) - 1 : static_cast<std::int64_t>(n);
    auto backward = apxchol::detail::gpu_factor_prep::build_L11_csc_int<float>(F.L, m);
    if (installed) {
        for (int j = 0; fp16 && j < m; ++j)
            for (int p = backward.ptr[j]; p < backward.ptr[j + 1]; ++p)
                backward.vals[p] = p == backward.ptr[j]
                    ? float(double(backward.vals[p]) * F.scale[j]) : backward.vals[p] * F.scale[j];
    } else {
    auto scales = apxchol::detail::gpu_factor_prep::column_scales(backward);
    if (fp16) apxchol::detail::gpu_factor_prep::normalize_fp16_scales(backward, scales);
    apxchol::detail::gpu_factor_prep::apply_factor_drop(backward, scales, options.drop_tolerance, fp16);
    if (fp16) {
        const auto narrow = apxchol::detail::gpu_factor_prep::narrow_fp16_scaled(backward, scales);
        for (int j = 0; j < m; ++j) {
            for (int p = backward.ptr[j]; p < backward.ptr[j + 1]; ++p)
                backward.vals[p] = float(narrow.vals[p]) * scales[j];
            backward.vals[backward.ptr[j]] = float(double(narrow.diag[j]) * scales[j]);
        }
    }
    }
    const auto forward = apxchol::detail::gpu_factor_prep::transpose_csr(backward);
    std::vector<double> rp(n);
    for (std::size_t v = 0; v < n; ++v) rp[F.perm[v]] = r[v];
    if (lap) {
        const double mean = std::accumulate(rp.begin(), rp.end(), 0.0) / n;
        for (auto& value : rp) value -= mean;
    }
    double max_abs = 0;
    for (double value : rp) max_abs = std::max(max_abs, std::abs(value));
    const double scale = mh::pow2_scale(max_abs);
    std::vector<float> in(n), z(n, 0.0f);
    for (std::size_t q = 0; q < n; ++q) in[q] = static_cast<float>(scale * rp[q]);
    auto sweep = [&](const auto& factor, bool reverse, const float* input) {
        for (std::int64_t q = 0; q < m; ++q) {
            const auto row = reverse ? m - 1 - q : q;
            double sum = input[row], diagonal = 0;
            for (int p = factor.ptr[row]; p < factor.ptr[row + 1]; ++p) {
                if (factor.idx[p] == row) diagonal = factor.vals[p];
                else sum -= double(factor.vals[p]) * z[factor.idx[p]];
            }
            z[row] = float(sum / diagonal);
        }
    };
    sweep(forward, false, in.data());
    sweep(backward, true, z.data());
    for (std::size_t q = 0; q < n; ++q) rp[q] = static_cast<double>(z[q]) / scale;
    if (lap) {
        const double mean = std::accumulate(rp.begin(), rp.end(), 0.0) / n;
        for (auto& value : rp) value -= mean;
    }
    Eigen::VectorXd out(r.size());
    for (std::size_t v = 0; v < n; ++v) out[v] = rp[F.perm[v]];
    return out;
}

}  // namespace

TEST_F(MetalDevice, DoubleFloatProbeExact) {
    // Test full arithmetic separately from available()'s fixed capability sentinels.
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
        EXPECT_EQ(static_cast<double>(out[16 * i + 2]) + static_cast<double>(out[16 * i + 3]),
                  static_cast<double>(x.hi) * static_cast<double>(y.hi));
        const double quotient = mh::join({out[16 * i + 12], out[16 * i + 13]});
        const double reference = mh::join(x) / mh::join(y);
        EXPECT_LE(std::abs(quotient - reference), 1e-12 * std::abs(reference));
        const float rounding[] = {(x.hi - y.hi) * y.lo, mh::quick_two_sum(x.hi, x.lo).lo};
        mismatches += std::memcmp(&out[16 * i + 14], rounding, sizeof rounding) != 0;
    }
    EXPECT_EQ(mismatches, 0u);
    std::printf("[ metal ] device %s, contract pragma %s, %zu probe cases bit-identical to the host\n",
                apxchol::detail::metal::status().name.c_str(),
                apxchol::detail::metal::status().contract_pragma ? "accepted" : "not accepted",
                in.size() / 4);
}

TEST_F(MetalDevice, UniformOperatorScalingPreservesConvergence) {
    for (const auto storage : {apxchol::factor_storage_type::fp32, apxchol::factor_storage_type::fp16}) {
        for (const int exponent : {-90, 0, 90}) {
            SCOPED_TRACE(exponent);
            const Sparse A = grid(4, 5, 0.5, true) * std::scalbn(1.0, exponent);
            apxchol::solve_options options;
            options.factor_opts.seed = 17;
            options.factor_opts.factor_storage = storage;
            apxchol::solver solver(A, gpu_options(options));
            for (unsigned salt : {1u, 4u}) {
                const auto b = rhs(A.rows(), salt, false);
                const auto result = solver.solve(b, 1e-8, 2000);
                EXPECT_LE(true_residual(A, b, result.x), 1e-8);
            }
        }
    }
}

TEST_F(MetalDevice, ScaledMatrixSolvesMatchScalarColumns) {
    enum class start { cold, zero_first, zero_second, warm_first, warm_second, no_iterations, warm_no_iterations };
    for (const auto storage : {apxchol::factor_storage_type::fp32, apxchol::factor_storage_type::fp16}) {
        for (const int exponent : {-90, 0, 90}) {
            const Sparse A = grid(4, 5, 0.5, true) * std::scalbn(1.0, exponent);
            apxchol::solve_options options;
            options.backend = apxchol::solve_backend::gpu;
            options.factor_opts.seed = 17;
            options.factor_opts.factor_storage = storage;
            apxchol::solver owner(A, options);
            Eigen::MatrixXd exact(A.rows(), 7);
            for (int q = 0; q < 7; ++q)
                exact.col(q) = rhs(A.rows(), q + 1, false) * std::scalbn(1.0, -exponent);
            for (const int width : {2, 3, 4, 5, 7}) {
                for (const auto mode : {start::cold, start::zero_first, start::zero_second,
                                       start::warm_first, start::warm_second, start::no_iterations,
                                       start::warm_no_iterations}) {
                    SCOPED_TRACE(testing::Message() << exponent << ' ' << int(storage) << ' ' << width << ' ' << int(mode));
                    Eigen::MatrixXd B = A * exact.leftCols(width);
                    Eigen::MatrixXd initial = Eigen::MatrixXd::Zero(A.rows(), width);
                    if (mode == start::zero_first) B.col(0).setZero();
                    if (mode == start::zero_second) B.col(1).setZero();
                    if (mode == start::warm_first) initial.col(0) = exact.col(0);
                    if (mode == start::warm_second) initial.col(1) = exact.col(1);
                    if (mode == start::warm_no_iterations) initial = exact.leftCols(width);
                    const bool warm = mode == start::warm_first || mode == start::warm_second ||
                                      mode == start::warm_no_iterations;
                    const int budget = mode == start::no_iterations || mode == start::warm_no_iterations ? 0 : 2000;
                    const apxchol::const_matrix_view x0(initial);
                    Eigen::MatrixXd X(A.rows(), width);
                    const auto result = owner.solve(B, X, 1e-8, budget, warm ? &x0 : nullptr);
                    ASSERT_EQ(result.columns.size(), std::size_t(width));
                    for (int q = 0; q < width; ++q) {
                        const Eigen::VectorXd b = B.col(q), guess = initial.col(q);
                        const auto scalar = owner.solve(b, 1e-8, budget, warm ? &guess : nullptr);
                        EXPECT_EQ(result.columns[q].backend, apxchol::solve_backend::gpu);
                        EXPECT_EQ(result.columns[q].iterations, scalar.iterations);
                        EXPECT_EQ(result.columns[q].x.size(), 0);
                        ASSERT_TRUE(X.col(q).allFinite());
                        ASSERT_TRUE(scalar.x.allFinite());
                        EXPECT_LE((X.col(q) - scalar.x).stableNorm(),
                                  1e-12 * std::max(scalar.x.stableNorm(), 1e-300));
                        if (budget || warm) {
                            EXPECT_LE((b - A * X.col(q)).stableNorm() / std::max(b.stableNorm(), 1e-300), 1e-8);
                        }
                        if (!budget || b.isZero(0) || (warm && !guess.isZero(0))) {
                            EXPECT_EQ(result.columns[q].iterations, 0);
                        }
                    }
                }
            }
        }
    }
}

TEST_F(MetalDevice, ScalarAndMatrixSolvesRejectOverflowWhenRestoringScale) {
    Sparse A(4, 4);
    A.setIdentity();
    A *= 1e-30;
    for (const auto storage : {apxchol::factor_storage_type::fp16,
                               apxchol::factor_storage_type::fp32}) {
        apxchol::solve_options options;
        options.backend = apxchol::solve_backend::gpu;
        options.factor_opts.factor_storage = storage;
        apxchol::solver owner(A, options);
        for (const bool warm : {false, true}) {
            SCOPED_TRACE(testing::Message() << int(storage) << ' ' << warm);
            const Eigen::VectorXd b = Eigen::VectorXd::Constant(4, 1e275);
            const Eigen::VectorXd guess = Eigen::VectorXd::Constant(4, 0.25e305);
            const auto scalar = owner.solve(b, 1e-8, 100, warm ? &guess : nullptr);
            ASSERT_TRUE(scalar.x.allFinite());
            EXPECT_LT((b - A * scalar.x).stableNorm() / b.stableNorm(), 1e-8);
            Eigen::MatrixXd B = b.replicate(1, 4), X(4, 4), initial = guess.replicate(1, 4);
            const apxchol::const_matrix_view x0(initial);
            const auto batch = owner.solve(B, X, 1e-8, 100, warm ? &x0 : nullptr);
            ASSERT_TRUE(X.allFinite());
            for (int c = 0; c < X.cols(); ++c) {
                EXPECT_LT((b - A * X.col(c)).stableNorm() / b.stableNorm(), 1e-8);
                EXPECT_LT(batch.columns[c].residual, 1e-8);
            }
            const Eigen::VectorXd huge = Eigen::VectorXd::Constant(4, 1e300);
            EXPECT_THROW(owner.solve(huge, 1e-8, 100, warm ? &guess : nullptr), std::overflow_error);
            B.col(2) = huge;
            EXPECT_THROW(owner.solve(B, X, 1e-8, 100, warm ? &x0 : nullptr), std::overflow_error);
        }
    }
}

TEST_F(MetalDevice, ScaledSolutionAccumulatorsPreserveAccuracy) {
    for (const auto& [diagonal, magnitude] : {std::pair{1e30, 1e-275}, std::pair{1e-30, 1e275},
             std::pair{1.0, 1e308}, std::pair{1.0, 1e-320},
             std::pair{1.0, std::numeric_limits<double>::denorm_min()}}) {
        Sparse A(4, 4);
        A.setIdentity();
        A *= diagonal;
        for (const auto storage : {apxchol::factor_storage_type::fp16,
                                  apxchol::factor_storage_type::fp32}) {
            apxchol::solve_options options;
            options.backend = apxchol::solve_backend::gpu;
            options.factor_opts.factor_storage = storage;
            apxchol::solver solver(A, options);
            for (int width : {1, 2, 4, 7}) {
                Eigen::MatrixXd B(4, width), X(4, width);
                for (int q = 0; q < width; ++q) B.col(q).setConstant(q % 2 ? -magnitude : magnitude);
                for (int warm : {0, 1, 2}) {
                    SCOPED_TRACE(testing::Message() << diagonal << ' ' << magnitude << ' '
                                 << int(storage) << ' ' << width << ' ' << warm);
                    Eigen::MatrixXd initial = B / diagonal;
                    if (warm == 1) initial.row(0).setZero();
                    std::vector<apxchol::solve_result> columns;
                    if (width == 1) {
                        const Eigen::VectorXd rhs = B.col(0), guess = initial.col(0);
                        columns.push_back(solver.solve(rhs, 1e-8, 100, warm ? &guess : nullptr));
                        X.col(0) = columns[0].x;
                    } else {
                        const apxchol::const_matrix_view guess(initial);
                        columns = solver.solve(B, X, 1e-8, 100, warm ? &guess : nullptr).columns;
                    }
                    ASSERT_TRUE(X.allFinite());
                    for (int q = 0; q < width; ++q) {
                        const double scale = mh::pow2_scale(magnitude);
                        const Eigen::VectorXd rhs = B.col(q) * scale;
                        const double residual = ((A * X.col(q)) * scale - rhs).norm() / rhs.norm();
                        EXPECT_LT(residual, 1e-8);
                        EXPECT_NEAR(columns[q].residual, residual, 1e-15);
                        if (warm == 2) {
                            EXPECT_EQ(columns[q].iterations, 0);
                        }
                    }
                }
            }
        }
    }
}

TEST_F(MetalDevice, FourColumnPlansReuseStorageAfterScalarAndWarmSolves) {
    const auto A = grid(40, 41, 0.25, true);
    Eigen::MatrixXd exact(A.rows(), 4);
    for (int q = 0; q < 4; ++q) exact.col(q) = rhs(A.rows(), q + 1, false);
    const Eigen::MatrixXd B = A * exact;
    const Eigen::VectorXd first_rhs = B.col(0);
    Eigen::MatrixXd initial = 0.25 * exact;
    initial.col(0) = exact.col(0);
    const apxchol::const_matrix_view warm(initial);
    auto check = [&](apxchol::solver& owner) {
        for (unsigned pass = 0; pass < 3; ++pass) {
            SCOPED_TRACE(pass);
            const auto scalar = owner.solve(first_rhs, 1e-8, 1000);
            EXPECT_LE(true_residual(A, first_rhs, scalar.x), 1e-8);
            Eigen::MatrixXd X(A.rows(), 4);
            const auto result = owner.solve(B, X, 1e-8, 1000, pass == 2 ? &warm : nullptr);
            ASSERT_EQ(result.columns.size(), 4u);
            if (pass == 2) { EXPECT_EQ(result.columns[0].iterations, 0); }
            for (unsigned q = 0; q < 4; ++q) {
                EXPECT_LE(result.columns[q].residual, 1e-8);
                EXPECT_LE(true_residual(A, B.col(q), X.col(q)), 1e-8);
            }
        }
    };
    for (const auto storage : {apxchol::factor_storage_type::fp16, apxchol::factor_storage_type::fp32}) {
        SCOPED_TRACE(int(storage));
        auto options = keep_opts();
        options.backend = apxchol::solve_backend::gpu;
        options.factor_opts.factor_storage = storage;
        apxchol::solver resident(A, options);
        std::vector<unsigned> row_sizes(A.rows());
        const auto factor = resident.factor();
        for (auto row : csc_access::inner(factor.L)) ++row_sizes[row];
        ASSERT_TRUE(std::ranges::any_of(factor.rounds, [&](const auto& round) {
            const auto first = row_sizes.begin() + A.rows() - round.active;
            return round.is_size > 128 &&
                std::count_if(first, first + round.is_size, [](unsigned size) { return size > 1; }) > 32;
        }));
        check(resident);
        apxchol::solver adopted(A, resident.factor(), options);
        check(adopted);
    }
}

TEST_F(MetalDevice, LongRowMatrixSolvesMatchScalarColumns) {
    const Sparse A = hub_graph(110, 100, 4);
    for (const auto storage : {apxchol::factor_storage_type::fp32, apxchol::factor_storage_type::fp16}) {
        auto options = keep_opts();
        options.backend = apxchol::solve_backend::gpu;
        options.factor_opts.seed = 17;
        options.factor_opts.factor_storage = storage;
        apxchol::solver owner(A, options);
        std::vector<unsigned> row_sizes(A.rows());
        const auto factor = owner.factor();
        for (auto row : csc_access::inner(factor.L))
            if (row + 1 < row_sizes.size()) ++row_sizes[row];
        ASSERT_GT(*std::max_element(row_sizes.begin(), row_sizes.end()), 64u);
        for (const int width : {4, 5}) {
            SCOPED_TRACE(testing::Message() << int(storage) << ' ' << width);
            Eigen::MatrixXd B(A.rows(), width), X(A.rows(), width);
            for (int q = 0; q < width; ++q) B.col(q) = rhs(A.rows(), q + 1, true);
            const auto batch = owner.solve(B, X, 1e-8, 2000);
            ASSERT_EQ(batch.columns.size(), std::size_t(width));
            for (int q = 0; q < width; ++q) {
                const Eigen::VectorXd b = B.col(q);
                const auto scalar = owner.solve(b, 1e-8, 2000);
                ASSERT_TRUE(X.col(q).allFinite());
                ASSERT_TRUE(scalar.x.allFinite());
                EXPECT_LE(true_residual(A, b, X.col(q)), 1e-8);
                EXPECT_LE(true_residual(A, b, scalar.x), 1e-8);
                EXPECT_LE((X.col(q) - scalar.x).norm(), 1e-5 * scalar.x.norm());
            }
        }
    }
}

TEST_F(MetalDevice, MatrixResidualsSurviveOverflowingRhsNorms) {
    Sparse A(16, 16);
    A.setIdentity();
    const double magnitudes[]{1e308, 1.0, 1e200, 1e-200};
    Eigen::MatrixXd B(16, 4), X(16, 4);
    for (int q = 0; q < 4; ++q) B.col(q).setConstant(magnitudes[q]);
    Eigen::MatrixXd initial = B;
    initial.row(0).setZero();
    const apxchol::const_matrix_view guess(initial);
    for (const auto storage : {apxchol::factor_storage_type::fp16,
                              apxchol::factor_storage_type::fp32}) {
        apxchol::solve_options options;
        options.backend = apxchol::solve_backend::gpu;
        options.factor_opts.factor_storage = storage;
        apxchol::solver solver(A, options);
        for (bool warm : {false, true}) {
            for (int budget : {0, 100}) {
                const auto result = solver.solve(B, X, 1e-8, budget, warm ? &guess : nullptr);
                ASSERT_TRUE(X.allFinite());
                for (int q = 0; q < 4; ++q) {
                    const double scale = apxchol::detail::metal_host::pow2_scale(magnitudes[q]);
                    const Eigen::VectorXd rhs = B.col(q) * scale;
                    const double residual = (X.col(q) * scale - rhs).norm() / rhs.norm();
                    EXPECT_NEAR(result.columns[q].residual, residual, 1e-15);
                    if (budget == 0) {
                        EXPECT_DOUBLE_EQ(residual, warm ? 0.25 : 1.0);
                        EXPECT_EQ(result.columns[q].iterations, 0);
                    } else {
                        EXPECT_LT(residual, 1e-8);
                        EXPECT_GT(result.columns[q].iterations, 0);
                    }
                }
            }
        }
    }
}

TEST_F(MetalDevice, ApplyAgreesWithSerialDoubleReference) {
    for (const double shift : {0.0, 0.5}) {
        SCOPED_TRACE(shift == 0.0 ? "Laplacian" : "SDDM");
        const Sparse A = grid(37, 29, shift, true);
        apxchol::factorization F = apxchol::factorize(A);
        const apxchol::factorization Fcopy = F;
        apxchol::solver slv(A, std::move(F), gpu_options());
        for (unsigned salt : {1u, 4u}) {
            const Eigen::VectorXd r = rhs(A.rows(), salt, shift == 0.0) * std::ldexp(1.0, 9 * static_cast<int>(salt) - 20);
            const Eigen::VectorXd z = slv.apply(r);
            const Eigen::VectorXd ze = emulate_apply(Fcopy, r);
            EXPECT_LT((z - ze).norm() / ze.norm(), 2e-5);
        }
    }
}

// Exercise long rows and cross-workgroup dependency stages on the device.
TEST_F(MetalDevice, HubFactorAgreesWithSerialDoubleReference) {
    const Sparse A = hub_graph(110, 100, 4);
    apxchol::factorization F = apxchol::factorize(A);
    const apxchol::factorization Fcopy = F;
    ASSERT_FALSE(Fcopy.sddm);
    apxchol::solver slv(A, std::move(F), gpu_options());
    const Eigen::VectorXd r = rhs(A.rows(), 2, true);
    const auto reference = emulate_apply(Fcopy, r);
    EXPECT_LT((slv.apply(r) - reference).norm() / reference.norm(), 2e-5);

}

// A synthetic factor with heavy rows in both sweeps (hub columns and hub
// rows), on the SDDM (m = n) and Laplacian (m = n - 1) paths.
TEST_F(MetalDevice, SyntheticHeavyFactorAgreesWithSerialDoubleReference) {
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
    csc_access::n(F.L) = n;
    csc_access::outer(F.L).assign(n + 1, 0);
    for (node_index j = 0; j < n; ++j) {
        auto& c = cols[j];
        std::sort(c.begin() + 1, c.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        c.erase(std::unique(c.begin() + 1, c.end(), [](const auto& a, const auto& b) { return a.first == b.first; }),
                c.end());
        csc_access::outer(F.L)[j + 1] = csc_access::outer(F.L)[j] + static_cast<apxchol::edge_index>(c.size());
        for (const auto& [i, v] : c) {
            csc_access::inner(F.L).push_back(i);
            csc_access::vals(F.L).push_back(v);
        }
    }
    F.perm.resize(n);
    for (node_index v = 0; v < n; ++v) F.perm[v] = (v * 7 + 3) % n;
    for (const bool sddm : {true, false}) {
        SCOPED_TRACE(sddm ? "SDDM path" : "Laplacian path");
        F.sddm = sddm;
        // The operator only has to match in size for an application.
        const Sparse A = grid(50, 100, sddm ? 0.5 : 0.0);
        apxchol::solver slv(A, apxchol::factorization(F), gpu_options());
        for (unsigned salt : {1u, 6u}) {
            const Eigen::VectorXd r = rhs(A.rows(), salt, !sddm);
            const auto reference = emulate_apply(F, r);
            EXPECT_LT((slv.apply(r) - reference).norm() / reference.norm(), 2e-5);
        }
    }
}

TEST_F(MetalDevice, ApplyAgreesWithCpuApply) {
    for (const double shift : {0.0, 0.25}) {
        SCOPED_TRACE(shift == 0.0 ? "Laplacian" : "SDDM");
        const Sparse A = grid(40, 40, shift);
        apxchol::factorization F = apxchol::factorize(A);
        apxchol::solver cpu(A, apxchol::factorization(F));
        apxchol::solver gpu(A, std::move(F), gpu_options());
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

TEST_F(MetalDevice, GridLaplacianOriginalResidual) {
    const Sparse A = grid(48, 48);
    apxchol::solver cpu(A, {.backend = apxchol::solve_backend::cpu});
    apxchol::solver gpu(A, gpu_options());
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

TEST_F(MetalDevice, SddmExactAndDoubleFloatOperators) {
    for (const bool inexact : {false, true}) {
        SCOPED_TRACE(inexact ? "double-float operator" : "fp32-exact operator");
        const Sparse A = grid(30, 50, 0.125, inexact);
        apxchol::solver slv(A, gpu_options());
        EXPECT_EQ(metal(slv).stats().operator_double_float, inexact);
        EXPECT_TRUE(slv.factor().sddm);
        const Eigen::VectorXd b = rhs(A.rows(), 5, false);
        const apxchol::solve_result r = slv.solve(b, 1e-10, 500);
        EXPECT_LT(r.residual, 1e-10);
        EXPECT_NEAR(r.residual, true_residual(A, b, r.x), 1e-3 * r.residual + 1e-16);
    }
}

TEST_F(MetalDevice, LongOperatorRowAndPartialTile) {
    // The hub exceeds 256 entries; the final tile contains only one row.
    constexpr int n = 257;
    for (const bool inexact : {false, true}) {
        std::vector<Eigen::Triplet<double>> entries;
        std::vector<double> diagonal(n, 0.125);
        for (int i = 1; i < n; ++i) {
            const double weight = 1.0 + (i % 7) / (inexact ? 3.0 : 4.0);
            entries.emplace_back(0, i, -weight);
            entries.emplace_back(i, 0, -weight);
            diagonal[0] += weight;
            diagonal[i] += weight;
        }
        for (int i = 0; i < n; ++i) entries.emplace_back(i, i, diagonal[i]);
        Sparse A(n, n);
        A.setFromTriplets(entries.begin(), entries.end());
        A.makeCompressed();
        for (const auto storage : {apxchol::factor_storage_type::fp32, apxchol::factor_storage_type::fp16}) {
            SCOPED_TRACE(testing::Message() << "inexact=" << inexact << " storage=" << int(storage));
            apxchol::solve_options options;
            options.factor_opts.factor_storage = storage;
            options.factor_opts.seed = 42;
            apxchol::solver solver(A, gpu_options(options));
            ASSERT_EQ(metal(solver).stats().operator_double_float, inexact);
            Eigen::VectorXd guess;
            for (const unsigned salt : {1u, 3u}) {
                const auto b = rhs(n, salt, false);
                const auto result = solver.solve(b, 1e-8, 500, guess.size() ? &guess : nullptr);
                ASSERT_TRUE(result.x.allFinite());
                EXPECT_LE(true_residual(A, b, result.x), 1e-8);
                guess = 0.5 * result.x;
            }
        }
    }
}


TEST_F(MetalDevice, EarlyExitsTruthful) {
    const Sparse A = grid(12, 12);
    for (const auto storage : {apxchol::factor_storage_type::fp32, apxchol::factor_storage_type::fp16}) {
        SCOPED_TRACE(int(storage));
        apxchol::solve_options options;
        options.factor_opts.factor_storage = storage;
        options.backend = apxchol::solve_backend::gpu;
        apxchol::solver slv(A, options);
        ASSERT_EQ(slv.backend(), apxchol::solve_backend::gpu);
        const Eigen::VectorXd b = rhs(A.rows(), 1, true);
        const apxchol::solve_result none = slv.solve(b, 1e-8, 0);
        EXPECT_EQ(none.iterations, 0);
        EXPECT_EQ(none.residual, 1.0);
        EXPECT_EQ(none.x.norm(), 0.0);

        const Eigen::VectorXd guess = rhs(A.rows(), 2, true);
        const auto seeded = slv.solve(b, 1e-8, 0, &guess);
        EXPECT_EQ(seeded.iterations, 0);
        EXPECT_LT((seeded.x - guess).norm(), 1e-12);
        EXPECT_NEAR(seeded.residual, true_residual(A, b, seeded.x), 1e-12);

        for (int limit : {0, 1, 4, 5}) {
            const auto limited = slv.solve(b, 1e-30, limit);
            EXPECT_EQ(limited.iterations, limit);
            EXPECT_NEAR(limited.residual, true_residual(A, b, limited.x), 1e-12);
            Eigen::MatrixXd B = b.replicate(1, 4), initial = Eigen::MatrixXd::Zero(A.rows(), 4);
            B.col(0).setZero();
            B.col(1) = A * guess;
            initial.col(1) = guess;
            const apxchol::const_matrix_view warm(initial);
            Eigen::MatrixXd storage = Eigen::MatrixXd::Constant(2 * A.rows(), 4, 77.0);
            using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
            Eigen::Map<Eigen::MatrixXd, 0, Stride> X(storage.data(), A.rows(), 4,
                                                   Stride(2 * A.rows(), 2));
            const auto panel = slv.solve(B, X, 1e-8, limit, &warm);
            ASSERT_EQ(panel.columns.size(), 4u);
            for (int q = 0; q < 4; ++q) {
                EXPECT_LE(panel.columns[q].iterations, limit);
                if (q < 2) EXPECT_EQ(panel.columns[q].iterations, 0);
                const double residual = q == 0 ? X.col(q).norm() : true_residual(A, B.col(q), X.col(q));
                EXPECT_NEAR(panel.columns[q].residual, residual, 1e-12);
            }
            for (Eigen::Index i = 1; i < storage.rows(); i += 2)
                EXPECT_TRUE((storage.row(i).array() == 77.0).all());
        }

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
}

TEST_F(MetalDevice, BreakdownIsNotConvergence) {
    // Centering an incompatible constant RHS leaves zero work on the host.
    // That early exit must still report the original-system residual.
    const Sparse A = grid(10, 10);
    for (const auto storage : {apxchol::factor_storage_type::fp32, apxchol::factor_storage_type::fp16}) {
        SCOPED_TRACE(int(storage));
        apxchol::solve_options options;
        options.factor_opts.factor_storage = storage;
        apxchol::solver slv(A, gpu_options(options));
        const Eigen::VectorXd b = Eigen::VectorXd::Ones(A.rows());
        const apxchol::solve_result r = slv.solve(b, 1e-8, 50);
        EXPECT_EQ(r.iterations, 0);
        EXPECT_EQ(r.residual, 1.0);
    }
}

TEST_F(MetalDevice, ProjectedWarmResidualDoesNotConsumeBudget) {
    constexpr int side = 4, n = side * side;
    constexpr double tolerance = 1e-8;
    std::vector<Eigen::Triplet<double>> entries;
    Eigen::VectorXd diagonal = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < n; ++i) {
        for (const int j : {i % side + 1 < side ? i + 1 : -1, i + side < n ? i + side : -1}) {
            if (j < 0) continue;
            entries.emplace_back(i, j, -1.0);
            entries.emplace_back(j, i, -1.0);
            ++diagonal[i];
            ++diagonal[j];
        }
    }
    for (int i = 0; i < n; ++i) entries.emplace_back(i, i, diagonal[i]);
    Sparse A(n, n);
    A.setFromTriplets(entries.begin(), entries.end());
    Eigen::VectorXd guess(n);
    for (int i = 0; i < n; ++i) guess[i] = i % 2 ? -1.0 : 1.0;
    const Eigen::VectorXd compatible = A * guess;
    // The nonzero-mean RHS is incompatible, but its projected warm residual is tiny.
    Eigen::VectorXd incompatible = compatible.array() + 1.0;
    incompatible[0] += 0x1p-30;
    incompatible[1] -= 0x1p-30;
    Eigen::VectorXd projected = incompatible - A * guess;
    projected.array() -= projected.mean();
    ASSERT_GT(true_residual(A, incompatible, guess), tolerance);
    ASSERT_LT(projected.norm() / incompatible.norm(), tolerance);

    for (const auto storage : {apxchol::factor_storage_type::fp32, apxchol::factor_storage_type::fp16}) {
        SCOPED_TRACE(int(storage));
        apxchol::solve_options options;
        options.backend = apxchol::solve_backend::gpu;
        options.factor_opts.factor_storage = storage;
        apxchol::solver slv(A, options);
        ASSERT_EQ(slv.backend(), apxchol::solve_backend::gpu);
        const auto scalar = slv.solve(incompatible, tolerance, 1000, &guess);
        EXPECT_EQ(scalar.iterations, 0);
        EXPECT_FALSE(scalar.converged);
        EXPECT_TRUE(same_bytes(scalar.x, guess));
        EXPECT_NEAR(scalar.residual, true_residual(A, incompatible, scalar.x), 1e-14);

        Eigen::MatrixXd B(n, 4), initial = Eigen::MatrixXd::Zero(n, 4), X(n, 4);
        B.col(0) = incompatible;
        B.col(1) = compatible;
        B.col(2).setZero();
        B.col(3) = A * rhs(n, 3, true);
        initial.col(0) = initial.col(1) = guess;
        const apxchol::const_matrix_view warm(initial);
        const auto batch = slv.solve(B, X, tolerance, 1000, &warm);
        ASSERT_EQ(batch.columns.size(), 4u);
        EXPECT_EQ(batch.columns[0].iterations, 0);
        EXPECT_FALSE(batch.columns[0].converged);
        EXPECT_TRUE(same_bytes(X.col(0), guess));
        EXPECT_EQ(batch.columns[1].iterations, 0);
        EXPECT_EQ(batch.columns[2].iterations, 0);
        EXPECT_GT(batch.columns[3].iterations, 0);
        EXPECT_TRUE(X.allFinite());
        for (int q = 0; q < 4; ++q) {
            const double residual = q == 2 ? X.col(q).norm() : true_residual(A, B.col(q), X.col(q));
            EXPECT_NEAR(batch.columns[q].residual, residual, 1e-14);
            EXPECT_EQ(batch.columns[q].converged, residual < tolerance);
            if (q > 0) EXPECT_TRUE(batch.columns[q].converged);
        }
    }
}

TEST_F(MetalDevice, UnreachableTolStopsHonestly) {
    const Sparse A = grid(16, 16, 0.0, true);
    apxchol::solver slv(A, gpu_options());
    const Eigen::VectorXd b = rhs(A.rows(), 7, true);
    const apxchol::solve_result r = slv.solve(b, 1e-30, 300);
    EXPECT_GE(r.residual, 1e-30);
    EXPECT_LE(r.iterations, 300);
    EXPECT_NEAR(r.residual, true_residual(A, b, r.x), 1e-2 * r.residual);
}

TEST_F(MetalDevice, PowerOfTwoScalingExact) {
    const Sparse A = grid(15, 21, 0.3, true);
    apxchol::solver slv(A, gpu_options());
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

TEST_F(MetalDevice, RepeatedAndTwinSolversBitIdentical) {
    const Sparse A = grid(33, 17);
    apxchol::factorization F = apxchol::factorize(A);
    apxchol::solver one(A, apxchol::factorization(F), gpu_options());
    apxchol::solver two(A, std::move(F), gpu_options());
    const Eigen::VectorXd b = rhs(A.rows(), 6, true);
    const apxchol::solve_result a1 = one.solve(b, 1e-10, 400);
    const Eigen::VectorXd other = rhs(A.rows(), 11, true);
    EXPECT_TRUE(same_bytes(one.apply(other), two.apply(other)));
    const apxchol::solve_result a2 = one.solve(b, 1e-10, 400);
    const apxchol::solve_result b1 = two.solve(b, 1e-10, 400);
    EXPECT_TRUE(same_bytes(a1.x, a2.x));
    EXPECT_TRUE(same_bytes(a1.x, b1.x));
    EXPECT_EQ(a1.iterations, b1.iterations);
    EXPECT_TRUE(same_bytes(one.apply(b), two.apply(b)));
}



TEST_F(MetalDevice, LaplacianX0ConstantIrrelevant) {
    const Sparse A = grid(26, 26);
    apxchol::solver slv(A, gpu_options());
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

TEST_F(MetalDevice, ExactWarmStartUsesTheOriginalOrdering) {
    for (const double shift : {0.0, 0.25}) {
        SCOPED_TRACE(shift);
        const Sparse A = grid(9, 11, shift, true);
        apxchol::solver resident(A, gpu_options(keep_opts()));
        apxchol::solver adopted(A, apxchol::factorization(resident.factor()), gpu_options());
        bool permuted = false;
        for (node_index i = 0; i < A.rows(); ++i)
            permuted |= resident.factor().perm[i] != i;
        ASSERT_TRUE(permuted);
        const Eigen::VectorXd guess = rhs(A.rows(), 9, shift == 0.0);
        const Eigen::VectorXd b = A * guess;
        for (auto* solver : {&resident, &adopted}) {
            const auto result = solver->solve(b, 1e-8, 20, &guess);
            EXPECT_EQ(result.iterations, 0);
            const double residual = true_residual(A, b, result.x);
            EXPECT_LT(residual, 1e-8);
            EXPECT_NEAR(result.residual, residual, 1e-14);
        }
    }
}

TEST_F(MetalDevice, NonFiniteRhsRejected) {
    const Sparse A = grid(8, 8, 0.5);
    apxchol::solver slv(A, gpu_options());
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

TEST_F(MetalDevice, OutOfRangeOperatorIsADomainError) {
    Sparse A(3, 3);
    A.setIdentity();
    const auto F = apxchol::factorize(A);
    apxchol::solve_options options;
    options.backend = apxchol::solve_backend::gpu;
    for (double value : {0x1p-101, 0x1p101}) {
        SCOPED_TRACE(value);
        A.diagonal().setConstant(value);
        EXPECT_THROW(apxchol::solver(A, F, options), std::invalid_argument);
        EXPECT_THROW(apxchol::solver(A, options), std::invalid_argument);
    }
}

TEST_F(MetalDevice, TinyFactorOffDiagonalsRemainUsable) {
    Sparse A(2, 2);
    A.setIdentity();
    const Eigen::Vector2d b(1.0, -2.0);
    for (float value : {-0x1p-110f, -std::numeric_limits<float>::min(),
                        -std::numeric_limits<float>::denorm_min()}) {
        SCOPED_TRACE(value);
        apxchol::factorization F;
        csc_access::n(F.L) = 2;
        csc_access::outer(F.L) = {0, 2, 3};
        csc_access::inner(F.L) = {0, 1, 1};
        csc_access::vals(F.L) = {1.0f, value, 1.0f};
        F.perm = {0, 1};
        F.sddm = true;
        apxchol::solver solver(A, std::move(F), gpu_options());
        EXPECT_EQ(metal(solver).stats().levels_forward, 2u);
        EXPECT_EQ(metal(solver).stats().levels_backward, 2u);
        EXPECT_TRUE(solver.apply(b).allFinite());
        const auto result = solver.solve(b, 1e-8, 10);
        EXPECT_LT(true_residual(A, b, result.x), 1e-8);
    }
}

TEST_F(MetalDevice, ScaledRightHandSides) {
    Sparse A(2, 2);
    A.setIdentity();
    apxchol::solver solver(A, gpu_options());
    for (const double magnitude : {0.0, 1e-200, 1.0, 1e200,
                                   std::numeric_limits<double>::denorm_min()}) {
        SCOPED_TRACE(magnitude);
        Eigen::VectorXd b(2);
        b << magnitude, -2 * magnitude;
        const auto result = solver.solve(b, 1e-10, 10);
        const auto applied = solver.apply(b);
        EXPECT_TRUE(result.x.allFinite());
        EXPECT_TRUE(applied.allFinite());
        if (magnitude == 0) {
            EXPECT_TRUE(result.x.isZero(0.0));
            EXPECT_TRUE(applied.isZero(0.0));
            EXPECT_EQ(result.iterations, 0);
            EXPECT_EQ(result.residual, 0.0);
            continue;
        }
        const double residual = (b - A * result.x).stableNorm() / b.stableNorm();
        EXPECT_GT(result.iterations, 0);
        EXPECT_LT(residual, 1e-10);
        EXPECT_NEAR(result.residual, residual, 1e-15);
        EXPECT_LT((applied - b).stableNorm() / b.stableNorm(), 1e-6);
    }
}

TEST_F(MetalDevice, RelativeResidualSurvivesOverflowingRhsNorm) {
    Sparse A(16, 16);
    A.setIdentity();
    const Eigen::VectorXd rhs = Eigen::VectorXd::Constant(16, 1e308);
    Eigen::VectorXd guess = rhs;
    guess[0] = 0;
    const double scale = 0x1p-1023;
    const Eigen::VectorXd scaled_rhs = rhs * scale;
    for (const auto storage : {apxchol::factor_storage_type::fp16,
                              apxchol::factor_storage_type::fp32}) {
        apxchol::solve_options options;
        options.factor_opts.factor_storage = storage;
        apxchol::solver solver(A, gpu_options(options));
        for (bool warm : {false, true}) {
            const auto stopped = solver.solve(rhs, 1e-8, 0, warm ? &guess : nullptr);
            EXPECT_DOUBLE_EQ(stopped.residual, warm ? 0.25 : 1.0);
            EXPECT_EQ(stopped.iterations, 0);
            const auto solved = solver.solve(rhs, 1e-8, 100, warm ? &guess : nullptr);
            ASSERT_TRUE(solved.x.allFinite());
            const double residual = (solved.x * scale - scaled_rhs).norm() / scaled_rhs.norm();
            EXPECT_LT(residual, 1e-8);
            EXPECT_NEAR(solved.residual, residual, 1e-15);
            EXPECT_GT(solved.iterations, 0);
        }
    }
}

TEST_F(MetalDevice, TinySystems) {
    Sparse one(1, 1);
    one.insert(0, 0) = 4.0;
    one.makeCompressed();
    apxchol::solver s1(one, gpu_options());
    const apxchol::solve_result r1 = s1.solve(Eigen::VectorXd::Constant(1, 2.0));
    EXPECT_EQ(r1.x[0], 0.5);
    EXPECT_EQ(r1.residual, 0.0);
    for (const int n : {2, 3}) {
        SCOPED_TRACE(n);
        const Sparse A = grid(1, n);  // path Laplacian
        apxchol::solver s(A, gpu_options());
        const Eigen::VectorXd b = rhs(n, 3, true);
        const apxchol::solve_result r = s.solve(b, 1e-12, 20);
        EXPECT_LT(r.residual, 1e-12);
        EXPECT_LE(std::fabs(r.x.mean()), 1e-14 * r.x.norm());
    }
}

TEST_F(MetalDevice, StatisticsDescribeTheSetup) {
    const Sparse A = grid(50, 40);
    const apxchol::solver slv(A, gpu_options(keep_opts()));
    const auto st = metal(slv).stats();
    EXPECT_EQ(st.n, A.rows());
    EXPECT_GT(st.levels_forward, 0u);
    EXPECT_GT(st.levels_backward, 0u);
    EXPECT_GT(st.steps_forward, 0u);
    EXPECT_FALSE(st.device.empty());
    EXPECT_FALSE(csc_access::vals(slv.factor().L).empty());
    EXPECT_EQ(slv.rows(), A.rows());
    EXPECT_EQ(slv.factor().perm.size(), static_cast<std::size_t>(A.rows()));
    // The route keeps no host copy of the factor; factor() reads the device.
    EXPECT_TRUE(csc_access::vals(metal(slv).factor().L).empty());
}

TEST_F(MetalDevice, ReusesFactorAcrossDifferentRightHandSides) {
    // Ordinary elimination creates duplicate fill before the 128-vertex tail.
    Eigen::MatrixXd dense = Eigen::MatrixXd::Constant(129, 129, -1.0);
    dense.diagonal().setConstant(129.0);
    const Sparse A = dense.sparseView();
    apxchol::solver solver(A, gpu_options(keep_opts()));
    const auto ptr = csc_access::outer(solver.factor().L);
    const auto idx = csc_access::inner(solver.factor().L);
    const auto vals = csc_access::vals(solver.factor().L);
    for (unsigned salt : {2u, 5u, 2u}) {
        const Eigen::VectorXd b = rhs(A.rows(), salt, false);
        const auto result = solver.solve(b, 1e-9, 400);
        EXPECT_EQ(result.backend, apxchol::solve_backend::gpu);
        EXPECT_LT(true_residual(A, b, result.x), 1e-9);
        EXPECT_EQ(csc_access::outer(solver.factor().L), ptr);
        EXPECT_EQ(csc_access::inner(solver.factor().L), idx);
        EXPECT_EQ(csc_access::vals(solver.factor().L), vals);
    }
}

TEST_F(MetalDevice, CommonSolverApiRespectsBackendSelection) {
    const Sparse A = grid(20, 21, 0.25);
    const Eigen::VectorXd b = rhs(A.rows(), 4, false);
    apxchol::solve_options options;
    for (const auto backend : {apxchol::solve_backend::cpu,
                              apxchol::solve_backend::automatic,
                              apxchol::solve_backend::gpu}) {
        options.backend = backend;
        apxchol::detail::checkpoint timings;
        const auto result = apxchol::detail::solver_access::with_timings(A, options, timings).solve(b);
        EXPECT_EQ(result.backend, backend == apxchol::solve_backend::gpu
            ? apxchol::solve_backend::gpu : apxchol::solve_backend::cpu);
        EXPECT_LT(true_residual(A, b, result.x), apxchol::default_tol);
        EXPECT_GT(timings.total("setup"), 0.0);
        EXPECT_GT(timings.total("pcg"), 0.0);
    }
    Sparse large(32769, 32769);
    large.setIdentity();
    const Eigen::VectorXd large_rhs = Eigen::VectorXd::Ones(large.rows());
    options.backend = apxchol::solve_backend::automatic;
    apxchol::solver automatic(large, options);
    EXPECT_EQ(automatic.backend(), apxchol::solve_backend::gpu);
    EXPECT_LT(true_residual(large, large_rhs, automatic.solve(large_rhs).x), apxchol::default_tol);
    large.coeffRef(0, 1) = 0; // Numerically valid, but the stored zero has no mate.
    large.makeCompressed();
    EXPECT_THROW(apxchol::solver(large, gpu_options(options)), std::invalid_argument);
    apxchol::solver fallback(large, options);
    EXPECT_EQ(fallback.backend(), apxchol::solve_backend::cpu);
    EXPECT_LT(true_residual(large, large_rhs, fallback.solve(large_rhs).x), apxchol::default_tol);
}

TEST_F(MetalDevice, CommonSolverAdoptsHostFactorWithCpuOnlyOptions) {
    const Sparse A = grid(9, 11, 0.25, true);
    auto options = keep_opts();
    options.factor_opts.sampler = apxchol::clique_sampler::trace_cycle;
    const auto F = apxchol::factorize(A, options.factor_opts);
    options.backend = apxchol::solve_backend::gpu;
    EXPECT_THROW(apxchol::solver(A, options), std::invalid_argument);
    apxchol::solver adopted(A, F, options);
    EXPECT_EQ(adopted.backend(), apxchol::solve_backend::gpu);
    Eigen::MatrixXd B(A.rows(), 4), X(A.rows(), 4);
    for (int q = 0; q < B.cols(); ++q) B.col(q) = rhs(A.rows(), q + 1, false);
    const auto batch = adopted.solve(B, X, 1e-8, 1000);
    ASSERT_EQ(batch.columns.size(), 4u);
    for (int q = 0; q < B.cols(); ++q) {
        const Eigen::VectorXd b = B.col(q);
        const auto scalar = adopted.solve(b, 1e-8, 1000);
        EXPECT_EQ(scalar.backend, apxchol::solve_backend::gpu);
        EXPECT_EQ(batch.columns[q].backend, apxchol::solve_backend::gpu);
        EXPECT_LE(true_residual(A, b, scalar.x), 1e-8);
        EXPECT_LE(true_residual(A, b, X.col(q)), 1e-8);
    }
    EXPECT_EQ(adopted.backend(), apxchol::solve_backend::gpu);
    options.backend = apxchol::solve_backend::automatic;
    EXPECT_EQ(apxchol::solver(A, F, options).backend(), apxchol::solve_backend::cpu);
    options.factor_opts.experimental.cpu_schedule = apxchol::cpu_sptrsv_schedule::levels;
    EXPECT_EQ(apxchol::solver(A, F, options).backend(), apxchol::solve_backend::cpu);
    options.backend = apxchol::solve_backend::gpu;
    EXPECT_THROW(apxchol::solver(A, F, options), std::invalid_argument);
    options.backend = static_cast<apxchol::solve_backend>(-1);
    EXPECT_THROW(apxchol::solver(A, F, options), std::invalid_argument);
}

TEST_F(MetalDevice, PerSolverPreparationOptionsAreIndependent) {
    const Sparse A = grid(7, 9);
    auto options = keep_opts();
    EXPECT_FALSE(apxchol::solver(A, gpu_options(options)).factor().sddm);
    options.factor_opts.regularization = 1e-8;
    EXPECT_TRUE(apxchol::solver(A, gpu_options(options)).factor().sddm);
    options.factor_opts.regularization = 0;
    EXPECT_FALSE(apxchol::solver(A, gpu_options(options)).factor().sddm);

    Sparse near = grid(7, 9, 0.25);
    near.coeffRef(0, 1) = near.coeffRef(1, 0) = 1e-6;
    options.factor_opts.lump_max_mass = 0;
    EXPECT_THROW(apxchol::solver(near, gpu_options(options)), std::invalid_argument);
    options.factor_opts.lump_max_mass = 0.25;
    EXPECT_EQ(apxchol::solver(near, gpu_options(options)).factor().lumped_offdiag, 2);
    options.factor_opts.experimental.cpu_schedule = apxchol::cpu_sptrsv_schedule::levels;
    EXPECT_THROW(apxchol::solver(A, gpu_options(options)), std::invalid_argument);
}

TEST_F(MetalDevice, PerSolverStorageAndDropMatchTheReference) {
    const Sparse A = grid(11, 13, 0.25, true);
    const auto factor = apxchol::factorize(A);
    const auto b = rhs(A.rows(), 3, false);
    for (const auto storage : {apxchol::factor_storage_type::fp32, apxchol::factor_storage_type::fp16}) {
        for (double drop : {0.0, 0.1, 1.0, 0.0}) {
            auto options = keep_opts();
            options.factor_opts.factor_storage = storage;
            options.factor_opts.drop_tolerance = drop;
            apxchol::solver solver(A, factor, gpu_options(options));
            const auto expected = emulate_apply(factor, b, options.factor_opts);
            EXPECT_LT((solver.apply(b) - expected).norm() / expected.norm(), 2e-5);
        }
    }
}

TEST(MetalOptions, InvalidDropToleranceFailsBeforeDeviceSetup) {
    const Sparse A = grid(4, 4, 0.25);
    const auto factor = apxchol::factorize(A);
    auto options = keep_opts();
    for (double drop : {-1.0, std::nextafter(1.0, 2.0), 2.0,
                        std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::quiet_NaN()}) {
        options.factor_opts.drop_tolerance = drop;
        EXPECT_THROW(apxchol::solver(A, gpu_options(options)), std::invalid_argument);
        EXPECT_THROW(apxchol::solver(A, factor, gpu_options(options)), std::invalid_argument);
    }
}

TEST_F(MetalDevice, ResidentDropToleranceMatchesHostReference) {
    for (bool laplacian : {false, true}) {
        const Sparse A = grid(7, 9, laplacian ? 0.0 : 0.25, true);
        const auto b = rhs(A.rows(), 3, laplacian);
        for (const auto storage : {apxchol::factor_storage_type::fp32, apxchol::factor_storage_type::fp16}) {
            for (double drop : {0.0, 0.1, 1.0}) {
                SCOPED_TRACE(laplacian);
                SCOPED_TRACE(static_cast<int>(storage));
                SCOPED_TRACE(drop);
                auto options = keep_opts();
                options.factor_opts.factor_storage = storage;
                options.factor_opts.drop_tolerance = drop;
                apxchol::solver solver(A, gpu_options(options));
                const auto expected = emulate_apply(solver.factor(), b, options.factor_opts);
                const auto actual = solver.apply(b);
                EXPECT_TRUE(actual.allFinite());
                EXPECT_LT((actual - expected).norm() / expected.norm(), 2e-5);
                const auto result = solver.solve(b, 1e-8, 1000);
                EXPECT_TRUE(result.converged);
                EXPECT_LE(true_residual(A, b, result.x), 1e-8);

                // factor() is the installed factor: the entries kept after the drop.
                const auto installed = solver.factor();
                const auto& L = installed.L;
                const auto m = static_cast<node_index>(A.rows() - (laplacian ? 1 : 0));
                std::uint64_t stored = 0;
                for (node_index j = 0; j < m; ++j)
                    for (auto p = L.col_ptrs()[j]; p < L.col_ptrs()[j + 1]; ++p)
                        stored += L.row_indices()[p] < m;
                const auto st = metal(solver).stats().drop;
                EXPECT_EQ(st.rel, drop);
                EXPECT_EQ(st.nnz_stored, stored);
                EXPECT_EQ(st.dropped, st.nnz_factor - st.nnz_stored);
                EXPECT_EQ(st.dropped, st.dropped_threshold + st.dropped_flush);
                if (drop == 0.0) EXPECT_EQ(st.dropped, 0u);
                if (drop == 1.0) EXPECT_GT(st.dropped_threshold, 0u);
            }
        }
    }
}

TEST_F(MetalDevice, TinyPositiveDropMatchesHostReference) {
    // Every first pivot has these three ratios, independent of its vertex ID.
    constexpr double weights[]{1.0, 0.4998778998851776, 0x1p-16};
    std::vector<Eigen::Triplet<double>> entries;
    for (int i = 0; i < 4; ++i) {
        entries.emplace_back(i, i, 4.0);
        for (int j = i + 1; j < 4; ++j) {
            const double weight = weights[(i ^ j) - 1];
            entries.emplace_back(i, j, -weight);
            entries.emplace_back(j, i, -weight);
        }
    }
    Sparse A(4, 4);
    A.setFromTriplets(entries.begin(), entries.end());
    A.makeCompressed();
    for (const auto storage : {apxchol::factor_storage_type::fp32, apxchol::factor_storage_type::fp16}) {
        for (double drop : {0.0, 1e-300, 1e-46, 0.1}) {
            SCOPED_TRACE(static_cast<int>(storage));
            SCOPED_TRACE(drop);
            auto options = keep_opts();
            options.factor_opts.factor_storage = storage;
            options.factor_opts.drop_tolerance = drop;
            apxchol::solver solver(A, gpu_options(options));
            for (int i = 0; i < 4; ++i) {
                SCOPED_TRACE(i);
                const Eigen::VectorXd b = Eigen::VectorXd::Unit(4, i);
                const auto expected = emulate_apply(solver.factor(), b, options.factor_opts);
                const auto actual = solver.apply(b);
                EXPECT_TRUE(actual.allFinite());
                EXPECT_LT((actual - expected).norm() / expected.norm(), 2e-6);
            }
        }
    }
}

TEST_F(MetalDevice, VerbosityIsPerSolver) {
    const Sparse A = grid(7, 9, 0.25);
    const auto b = rhs(A.rows(), 4, false);
    for (bool verbose : {false, true, false}) {
        apxchol::solve_options options;
        options.backend = apxchol::solve_backend::gpu;
        options.factor_opts.verbose = verbose;
        testing::internal::CaptureStderr();
        apxchol::solve_result result;
        try { result = apxchol::solve(A, b, std::nullopt, std::nullopt, options); }
        catch (...) { (void)testing::internal::GetCapturedStderr(); throw; }
        const auto output = testing::internal::GetCapturedStderr();
        EXPECT_EQ(output.find("[apxchol] Metal PCG") != std::string::npos, verbose) << output;
        EXPECT_TRUE(result.converged);
        EXPECT_LE(true_residual(A, b, result.x), apxchol::default_tol);
    }
}

TEST_F(MetalDevice, ScalarAndPanelOutputsRejectInputAliasing) {
    const Sparse A = grid(4, 5, 0.25);
    const auto b = rhs(A.rows(), 3, false);
    apxchol::solver direct(A, gpu_options());
    Eigen::VectorXd x = b;
    EXPECT_THROW(direct.solve(x, x), std::invalid_argument);
    EXPECT_EQ(x, b);

    apxchol::solve_options options;
    options.backend = apxchol::solve_backend::gpu;
    apxchol::solver owner(A, options);
    Eigen::MatrixXd B = b.replicate(1, 4), X = B;
    EXPECT_THROW(owner.solve(B, B), std::invalid_argument);
    EXPECT_EQ(B, X);
    const apxchol::const_matrix_view guess(X);
    EXPECT_THROW(owner.solve(B, X, std::nullopt, std::nullopt, &guess), std::invalid_argument);
    EXPECT_EQ(X, B);
    Eigen::MatrixXd wrong(A.rows() - 1, 4);
    EXPECT_THROW(owner.solve(B, wrong), std::invalid_argument);
}

TEST_F(MetalDevice, Float16FactorUsesSharedStorageAndConverges) {
    for (const double shift : {0.0, 0.25}) {
        const auto A = grid(37, 29, shift, true);
        auto options = keep_opts();
        options.factor_opts.factor_storage = apxchol::factor_storage_type::fp16;
        apxchol::solver solver(A, gpu_options(options));
        for (unsigned salt : {1u, 4u}) {
            const auto b = rhs(A.rows(), salt, shift == 0);
            const auto expected = emulate_apply(solver.factor(), b, options.factor_opts);
            EXPECT_LT((solver.apply(b) - expected).norm() / expected.norm(), 2e-5);
            const auto result = solver.solve(b, 1e-9, 1000);
            EXPECT_LT(true_residual(A, b, result.x), 1e-8);
        }
    }
}

TEST_F(MetalDevice, Float16InstalledFactorIsReadoptedExactly) {
    const auto A = grid(32, 33, 0.25);
    auto F = apxchol::factorize(A);
    auto options = keep_opts();
    options.factor_opts.factor_storage = apxchol::factor_storage_type::fp16;
    apxchol::solver half(A, F, gpu_options(options));
    const auto installed = half.factor();
    EXPECT_EQ(installed.storage, apxchol::factor_storage_type::fp16);
    EXPECT_EQ(installed.scale.size(), static_cast<std::size_t>(A.rows()));
    // Conflicting options do not change an installed factor.
    options.factor_opts.factor_storage = apxchol::factor_storage_type::fp32;
    options.factor_opts.drop_tolerance = 0;
    apxchol::solver again(A, installed, gpu_options(options));
    EXPECT_EQ(csc_access::vals(again.factor().L), csc_access::vals(installed.L));
    EXPECT_EQ(again.factor().scale, installed.scale);
    const auto b = rhs(A.rows(), 3, false);
    EXPECT_EQ(again.apply(b), half.apply(b));
}

TEST_F(MetalDevice, Float16TinyColumnScaleKeepsFinitePreconditioning) {
    Sparse A(2, 2);
    A.setIdentity();
    auto options = keep_opts();
    options.factor_opts.factor_storage = apxchol::factor_storage_type::fp16;
    const Eigen::Vector2d b(0.25, 1.0);
    for (float value : {-0x1p-40f, -0x1p-80f, -0x1p-99f, -0x1p-101f, -0x1p-110f, -0x1p-125f, -1e-40f,
                        -std::numeric_limits<float>::denorm_min()}) {
        apxchol::factorization F;
        csc_access::n(F.L) = 2;
        csc_access::outer(F.L) = {0, 2, 3};
        csc_access::inner(F.L) = {0, 1, 1};
        csc_access::vals(F.L) = {1.0f, value, 1.0f};
        F.perm = {0, 1};
        F.sddm = true;
        apxchol::solver solver(A, F, gpu_options(options));
        const auto expected = emulate_apply(F, b, options.factor_opts);
        const auto actual = solver.apply(b);
        EXPECT_TRUE(actual.allFinite());
        EXPECT_LT((actual - expected).norm() / expected.norm(), 2e-5);
        EXPECT_LT(true_residual(A, b, solver.solve(b, 1e-9, 10).x), 1e-8);
    }
    for (float diagonal : {0x1p-101f, 0x1p101f}) {
        auto F = apxchol::factorize(A);
        csc_access::vals(F.L)[0] = diagonal;
        EXPECT_THROW(apxchol::solver(A, F, gpu_options(options)), std::invalid_argument);
    }
}

TEST(MetalOptions, ResidentSetupRejectsUnsupportedOptionsAndLayouts) {
    apxchol::solve_options options;
    options.backend = apxchol::solve_backend::gpu;
    const auto A = grid(4, 4, 0.25);
    options.factor_opts.sampler = apxchol::clique_sampler::trace_cycle;
    EXPECT_THROW(apxchol::solver(A, options), std::invalid_argument);
    options.factor_opts.sampler = apxchol::clique_sampler::gks;
    options.factor_opts.partitioner = apxchol::partitioner_kind::priority_greedy;
    EXPECT_THROW(apxchol::solver(A, options), std::invalid_argument);
    options.factor_opts.partitioner = apxchol::partitioner_kind::block_greedy;
    if (!apxchol::backend_available(apxchol::solve_backend::gpu)) GTEST_SKIP() << "no usable Metal device";
    Sparse lower = A.triangularView<Eigen::Lower>();
    EXPECT_THROW(apxchol::solver(lower, options), std::invalid_argument);
    Sparse missing = A;
    missing.coeffRef(1, 0) = 0;
    missing.prune(0.0);
    EXPECT_THROW(apxchol::solver(missing, options), std::invalid_argument);
    Sparse unsorted = A;
    std::swap(unsorted.innerIndexPtr()[0], unsorted.innerIndexPtr()[1]);
    EXPECT_THROW(apxchol::solver(unsorted, options), std::invalid_argument);
}

TEST_F(MetalDevice, ResidentSetupAcceptsIntermediateDegreeQuantiles) {
    const auto A = grid(17, 19, 0.25, true);
    const auto b = rhs(A.rows(), 3, false);
    for (const auto storage : {apxchol::factor_storage_type::fp16, apxchol::factor_storage_type::fp32}) {
        for (double quantile : {0.1, 0.3, 0.75, 0.9}) {
            SCOPED_TRACE(testing::Message() << int(storage) << ' ' << quantile);
            auto options = keep_opts();
            options.factor_opts.factor_storage = storage;
            options.factor_opts.partition.degree_quantile = quantile;
            apxchol::solver solver(A, gpu_options(options));
            const auto result = solver.solve(b, 1e-8, 300);
            EXPECT_LE(result.residual, 1e-8);
            EXPECT_LE(true_residual(A, b, result.x), 1e-8);
        }
    }
}

TEST_F(MetalDevice, ResidentArenaIgnoresNonGraphEntries) {
    for (unsigned kind : {0u, 1u, 2u}) {
        SCOPED_TRACE(kind);
        Sparse A = grid(3, 3, 0.25);
        if (kind == 0) A.setIdentity();
        else if (kind == 1) A.coeffRef(0, 8) = A.coeffRef(8, 0) = 0.0;
        else A.coeffRef(0, 1) = A.coeffRef(1, 0) = 0.125;
        A.makeCompressed();
        apxchol::solver solver(A, gpu_options());
        if (kind == 2) { EXPECT_EQ(solver.factor().lumped_offdiag, 2); }
        const auto b = rhs(A.rows(), 3, false);
        const auto result = solver.solve(b, 1e-8, 200);
        EXPECT_LT(true_residual(A, b, result.x), 1e-8);
    }
}

TEST_F(MetalDevice, NonemptyFactorGrowthPreservesExportAndSolve) {
    // This ring enters the larger tail reservation after storing ordinary-round columns.
    constexpr int n = 257;
    std::vector<Eigen::Triplet<double>> entries;
    for (int i = 0; i < n; ++i) {
        const int next = (i + 1) % n;
        entries.emplace_back(i, i, 2.25);
        entries.emplace_back(i, next, -1.0);
        entries.emplace_back(next, i, -1.0);
    }
    Sparse A(n, n);
    A.setFromTriplets(entries.begin(), entries.end());
    A.makeCompressed();
    const auto b = rhs(n, 3, false);
    for (const bool fp16 : {false, true}) {
        SCOPED_TRACE(fp16);
        auto options = keep_opts();
        options.factor_opts.seed = 42;
        options.factor_opts.factor_storage = fp16 ? apxchol::factor_storage_type::fp16
                                                 : apxchol::factor_storage_type::fp32;
        apxchol::solver solver(A, gpu_options(options));
        const auto result = solver.solve(b, 1e-8, 200);
        EXPECT_LT(true_residual(A, b, result.x), 1e-8);
        const auto expected = emulate_apply(solver.factor(), b, options.factor_opts);
        EXPECT_LT((solver.apply(b) - expected).norm() / expected.norm(), 2e-5);
    }
}

TEST_F(MetalDevice, ResidentOwnerKeepsNoHostFactorValues) {
    for (const bool laplacian : {false, true}) for (const bool fp16 : {false, true}) {
        SCOPED_TRACE(::testing::Message() << laplacian << ' ' << fp16);
        const auto A = grid(17, 19, laplacian ? 0.0 : 0.25, true);
        auto options = keep_opts();
        options.factor_opts.factor_storage = fp16 ? apxchol::factor_storage_type::fp16
                                                 : apxchol::factor_storage_type::fp32;
        apxchol::solver exported(A, gpu_options(options));
        const auto b = rhs(A.rows(), 3, laplacian);
        const auto expected = emulate_apply(exported.factor(), b, options.factor_opts);
        EXPECT_LT((exported.apply(b) - expected).norm() / expected.norm(), 2e-5);
        apxchol::solver resident(A, gpu_options(options));
        EXPECT_TRUE(csc_access::inner(metal(resident).factor().L).empty());
        EXPECT_TRUE(csc_access::vals(metal(resident).factor().L).empty());
        EXPECT_GT(metal(resident).factor().L.nonZeros(), 0);
        for (unsigned salt : {1u, 2u, 3u}) {
            const auto rhs_value = rhs(A.rows(), salt, laplacian);
            const auto result = resident.solve(rhs_value);
            EXPECT_LT(true_residual(A, rhs_value, result.x), apxchol::default_tol);
        }
    }
}

TEST_F(MetalDevice, ExportedRoundsAreIndependentCpuLevels) {
    for (unsigned kind : {0u, 1u, 2u}) {
        SCOPED_TRACE(kind);
        Sparse A = grid(17, 19, kind == 0 ? 0.0 : 0.25);
        if (kind == 2) {
            Eigen::MatrixXd dense = Eigen::MatrixXd::Constant(129, 129, -1.0);
            dense.diagonal().setConstant(129.0);
            A = dense.sparseView();
        }
        const apxchol::solver exported(A, gpu_options(keep_opts()));
        const auto& F = exported.factor();
        ASSERT_FALSE(F.rounds.empty());
        std::vector<unsigned> levels(A.rows());
        std::size_t first = 0;
        for (unsigned r = 0; r < F.rounds.size(); ++r) {
            const auto& round = F.rounds[r];
            const auto last = first + round.is_size;
            ASSERT_LE(last, levels.size());
            EXPECT_EQ(round.active, levels.size() - first);
            std::fill(levels.begin() + first, levels.begin() + last, r);
            first = last;
        }
        ASSERT_EQ(first, levels.size());
        for (std::size_t col = 0; col < levels.size(); ++col)
            for (auto p = csc_access::outer(F.L)[col] + 1; p < csc_access::outer(F.L)[col + 1]; ++p)
                EXPECT_GT(levels[csc_access::inner(F.L)[p]], levels[col]);

        apxchol::solver grouped(A, F);
        auto ungrouped = F;
        ungrouped.rounds.clear();
        apxchol::solver reference(A, std::move(ungrouped));
        const auto b = rhs(A.rows(), 3, kind == 0);
        const auto expected = reference.apply(b);
        EXPECT_LT((grouped.apply(b) - expected).norm() / expected.norm(), 1e-12);
        EXPECT_LT(true_residual(A, b, grouped.solve(b, 1e-8, 500).x), 1e-8);
    }
}

TEST(MetalOptions, AutomaticSelectionRejectsResidentIncompatibleOptions) {
    Sparse A(32769, 32769);
    A.setIdentity();
    A.makeCompressed();
    apxchol::solve_options options;
    options.factor_opts.sampler = apxchol::clique_sampler::trace_cycle;
    EXPECT_EQ(apxchol::detail::select_solve_backend(options, &A), apxchol::solve_backend::cpu);
    options.factor_opts.sampler = apxchol::clique_sampler::gks;
    A.uncompress();
    EXPECT_EQ(apxchol::detail::select_solve_backend(options, &A), apxchol::solve_backend::cpu);
    options.backend = apxchol::solve_backend::gpu;
    EXPECT_EQ(apxchol::detail::select_solve_backend(options, &A), apxchol::solve_backend::gpu);
}
