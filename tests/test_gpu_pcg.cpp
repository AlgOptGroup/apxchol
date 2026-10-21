#include <gtest/gtest.h>
#include <apxchol/solver/detail/gpu_pcg.h>
#include <apxchol/solver/detail/gpu_scaling.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace pcg = apxchol::detail::gpu_pcg;

namespace {
struct recurrence_math {
    bool positive(double x) const { return x > 0 && std::isfinite(x); }
    bool valid_residual(double x) const { return x >= 0 && std::isfinite(x); }
    bool accept_ratio(double x) const { return std::isfinite(x); }
    double ratio(double x, double y) const { return x / y; }
    bool converged(double x, double threshold) const { return x < threshold; }
    bool stagnated(double previous, double current) const { return previous * 0.25 < current; }
};

struct step_math {
    using wide = double;
    using scalar = double;
    double preconditioned(double z) const { return z; }
    double add_product(double sum, double a, double b) const { return sum + a * b; }
    double update_solution(double x, double alpha, double p) const { return x + alpha * p; }
    double subtract_product(double a, double b, double c) const { return a - b * c; }
    double direction(double z, double p, double beta) const { return z + beta * p; }
    double add_entry(double sum, const double* hi, const double*, unsigned e, double p) const {
        return sum + hi[e] * p;
    }
    double sum_row(double sum) const { return sum; }
};

struct commands {
    unsigned stop;
    bool active = true;
    unsigned batches = 0, initializations = 0, iterations = 0, directions = 0, preconditions = 0;
    template<class Record>
    bool submit(bool first, Record&& record) {
        EXPECT_EQ(first, batches++ == 0);
        record();
        return active;
    }
    void precondition() { preconditions += active; }
    void step(pcg::operation) const {}
    void finalize(unsigned mode, unsigned it) {
        if (!active) return;
        switch (mode) {
            case pcg::kFinalizeRzInit: ++initializations; break;
            case pcg::kFinalizeRz: ++directions; break;
            case pcg::kFinalizeAlpha: EXPECT_EQ(it, iterations + 1); break;
            case pcg::kFinalizeCheck: iterations = it; active = it < stop; break;
        }
    }
};
} // namespace

TEST(GpuPcg, MappedMultiplyPreservesOutputAndDotProduct) {
    auto check = []<unsigned Columns, class Offset>() {
        const std::array<Offset, 3> perm{2, 0, 1};
        constexpr double A[3][3]{{4, -1, 0}, {-1, 5, -2}, {0, -2, 6}};
        for (bool mapped : {false, true}) {
            SCOPED_TRACE(::testing::Message() << Columns << " mapped=" << mapped);
            std::array<Offset, 4> ptr{};
            std::array<Offset, 9> col{};
            std::array<double, 9> values{};
            for (unsigned row = 0; row < 3; ++row) {
                unsigned original = row;
                if (!mapped)
                    while (perm[original] != Offset(row)) original = (original + 1) % 3;
                ptr[row + 1] = ptr[row];
                for (unsigned other = 0; other < 3; ++other) if (A[original][other] != 0) {
                    const auto e = ptr[row + 1]++;
                    col[e] = perm[other];
                    values[e] = A[original][other];
                }
            }
            std::array<double, 3 * Columns> x{}, r{}, ap{}, p{}, z{}, expected{};
            std::array<pcg::solve_state<double, double>, Columns> state{};
            for (unsigned q = 0; q < Columns; ++q) {
                state[q].active = q != 1;
                state[q].alpha = 0.25;
                for (unsigned row = 0; row < 3; ++row)
                    p[perm[row] * Columns + q] = 1 + row + 0.5 * q;
                for (unsigned row = 0; row < 3; ++row)
                    for (unsigned other = 0; other < 3; ++other)
                        expected[perm[row] * Columns + q] += A[row][other] * p[perm[other] * Columns + q];
            }
            ap.fill(-77);
            r.fill(1);
            const pcg::workspace<double, double> work{x.data(), r.data(), ap.data(), p.data(), z.data(), nullptr, state.data()};
            pcg::column_sums<double, Columns> total{};
            for (int offset = 0; offset < 2; ++offset) {
                const pcg::pcg_work<std::int64_t> range{0, 4, 2, offset, 3};
                const auto sum = mapped
                    ? pcg::step_columns<1, Columns>(pcg::operation::multiply, step_math{}, work,
                          ptr.data(), col.data(), values.data(), static_cast<const double*>(nullptr), range, 0, perm.data())
                    : pcg::step_columns<1, Columns>(pcg::operation::multiply, step_math{}, work,
                          ptr.data(), col.data(), values.data(), static_cast<const double*>(nullptr), range, 0);
                for (unsigned q = 0; q < Columns; ++q) total.values[q] += sum.values[q];
            }
            for (unsigned q = 0; q < Columns; ++q) {
                double expected_dot = 0;
                for (unsigned row = 0; row < 3; ++row) {
                    const auto i = row * Columns + q;
                    EXPECT_EQ(ap[i], state[q].active ? expected[i] : -77);
                    expected_dot += expected[i] * p[i];
                }
                EXPECT_EQ(total.values[q], state[q].active ? expected_dot : 0);
            }
            pcg::step_columns<1, Columns>(pcg::operation::residual, step_math{}, work,
                ptr.data(), col.data(), values.data(), static_cast<const double*>(nullptr),
                pcg::pcg_work<std::int64_t>{0, 3, 1, 0, 3}, 0, perm.data());
            for (unsigned row = 0; row < 3; ++row) for (unsigned q = 0; q < Columns; ++q) {
                const auto i = row * Columns + q;
                EXPECT_EQ(x[i], state[q].active ? 0.25 * p[i] : 0);
                EXPECT_EQ(r[i], state[q].active ? 1 - 0.25 * expected[i] : 1);
            }
        }
    };
    check.operator()<1, int>();
    check.operator()<4, int>();
    check.operator()<1, unsigned>();
    check.operator()<4, unsigned>();
}

TEST(GpuPcg, ProductsRequireFinitePositiveValues) {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (auto mode : {pcg::kFinalizeRzInit, pcg::kFinalizeRz, pcg::kFinalizeAlpha}) {
        for (double product : {0.0, -1.0, inf, -inf, nan, 1.0}) {
            SCOPED_TRACE(::testing::Message() << mode << ' ' << product);
            pcg::solve_state<double, double> state{};
            state.rz = 2;
            state.active = 1;
            state.iters = 3;
            state = pcg::advance(recurrence_math{}, state, mode, product, 4);
            EXPECT_EQ(state.active, unsigned(product == 1.0));
            EXPECT_EQ(state.iters, 3u);
            if (product != 1.0) {
                EXPECT_EQ(state.rz, 2);
            } else if (mode == pcg::kFinalizeAlpha) {
                EXPECT_EQ(state.alpha, 2);
            } else {
                EXPECT_EQ(state.rz, 1);
                EXPECT_EQ(state.beta, mode == pcg::kFinalizeRzInit ? 0 : 0.5);
            }
        }
    }
}

TEST(GpuPcg, OverflowingCoefficientsBreakDownBeforeUpdatingTheSolution) {
    const double small = std::numeric_limits<double>::min();
    const double large = std::numeric_limits<double>::max();
    for (auto mode : {pcg::kFinalizeRz, pcg::kFinalizeAlpha}) {
        SCOPED_TRACE(mode);
        pcg::solve_state<double, double> state{};
        state.rz = mode == pcg::kFinalizeRz ? small : large;
        state.active = 1;
        state = pcg::advance(recurrence_math{}, state, mode,
                             mode == pcg::kFinalizeRz ? large : small, 1);
        EXPECT_EQ(state.active, 0u);
        EXPECT_EQ(state.iters, 0u);
    }
}

TEST(GpuPcg, ResidualCheckCountsTheCompletedIterationEvenOnBreakdown) {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (double residual_squared : {-1.0, inf, nan, 0.0}) {
        SCOPED_TRACE(residual_squared);
        pcg::solve_state<double, double> state{};
        state.active = 1;
        state.thr = 1e-16;
        state = pcg::advance(recurrence_math{}, state, pcg::kFinalizeCheck,
                             residual_squared, 4);
        EXPECT_EQ(state.active, 0u);
        EXPECT_EQ(state.iters, 4u);
        if (!std::isnan(residual_squared)) {
            EXPECT_EQ(state.rr, residual_squared);
        }
    }
}

TEST(GpuPcg, DevicePredicationRespectsIterationLimitsAndConvergence) {
    for (unsigned batch : {0u, 1u, 4u}) for (unsigned limit : {0u, 1u, 5u})
    for (unsigned stop : {1u, 3u, 9u}) {
        SCOPED_TRACE(::testing::Message() << batch << ' ' << limit << ' ' << stop);
        commands gpu{stop};
        pcg::solve(gpu, limit, batch);
        const unsigned expected = std::min(limit, stop), width = std::max(batch, 1u);
        EXPECT_EQ(gpu.initializations, unsigned(expected > 0));
        EXPECT_EQ(gpu.iterations, expected);
        EXPECT_EQ(gpu.directions, expected ? expected - 1 : 0);
        EXPECT_EQ(gpu.preconditions, expected);
        EXPECT_EQ(gpu.batches, std::max(1u, (expected + width - 1) / width));
    }
}

TEST(GpuPcg, StagnationChecksOnlyAtWindowBoundaries) {
    for (const unsigned window : {0u, 3u}) {
        SCOPED_TRACE(window);
        pcg::solve_state<double, double> state{};
        state.prev = 16.0;
        state.active = 1;
        state = pcg::advance(recurrence_math{}, state, pcg::kFinalizeCheck, 64.0, 2, window);
        EXPECT_EQ(state.active, 1u);
        EXPECT_EQ(state.prev, 16.0);
        // Halving the norm is sufficient; the recurrence stores squared norms.
        state = pcg::advance(recurrence_math{}, state, pcg::kFinalizeCheck, 4.0, 3, window);
        EXPECT_EQ(state.active, 1u);
        EXPECT_EQ(state.prev, window ? 4.0 : 16.0);
        state = pcg::advance(recurrence_math{}, state, pcg::kFinalizeCheck, 1.1, 6, window);
        EXPECT_EQ(state.active, window ? 0u : 1u);
        EXPECT_EQ(state.iters, 6u);
        EXPECT_EQ(state.rr, 1.1);
    }
}

TEST(GpuPcg, InputScalingKeepsOrdinaryValuesAndNormalizesExtremes) {
    namespace scaling = apxchol::detail::gpu_scaling;
    for (const double value : {0.0, 0x1p-32, 1.0, 0x1p32})
        EXPECT_EQ(scaling::fp32_scale(value), 1.0);
    for (const double value : {1e-100, 1e100, std::numeric_limits<double>::min(),
                               std::numeric_limits<double>::max()}) {
        const double scale = scaling::fp32_scale(value);
        EXPECT_TRUE(std::isfinite(scale));
        EXPECT_GE(value * scale, 1.0);
        EXPECT_LT(value * scale, 2.0);
        EXPECT_EQ((value * scale) / scale, value);
    }
    const double tiny = std::numeric_limits<double>::denorm_min();
    const double scale = scaling::fp32_scale(tiny);
    EXPECT_TRUE(std::isnormal(float(tiny * scale)));
    EXPECT_EQ((tiny * scale) / scale, tiny);
}
