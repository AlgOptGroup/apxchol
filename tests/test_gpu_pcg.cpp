#include <gtest/gtest.h>
#include <apxchol/solver/detail/gpu_pcg.h>
#include <cmath>
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
