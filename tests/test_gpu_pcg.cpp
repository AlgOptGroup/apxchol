#include <gtest/gtest.h>
#include <apxchol/solver/detail/gpu_pcg.h>

namespace pcg = apxchol::detail::gpu_pcg;

namespace {
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
    void direction(unsigned it) {
        if (!active) return;
        if (it == 0) ++initializations;
        else ++directions;
    }
    void multiply(unsigned it) const {
        if (active) { EXPECT_EQ(it, iterations + 1); }
    }
    void update_residual(unsigned it) {
        if (active) { iterations = it; active = it < stop; }
    }
};
} // namespace

TEST(GpuPcg, DevicePredicationRespectsIterationLimitsAndConvergence) {
    for (unsigned batch : {0u, 1u, 4u}) for (unsigned limit : {0u, 1u, 5u})
    for (unsigned stop : {1u, 3u, 9u}) {
        SCOPED_TRACE(::testing::Message() << batch << ' ' << limit << ' ' << stop);
        commands gpu{stop};
        pcg::solve(gpu, limit, batch);
        const unsigned expected = std::min(limit, stop), width = std::max(batch, 1u);
        EXPECT_EQ(gpu.initializations, 1u);
        EXPECT_EQ(gpu.iterations, expected);
        EXPECT_EQ(gpu.directions, expected - unsigned(stop <= limit));
        EXPECT_EQ(gpu.preconditions, gpu.directions + 1);
        EXPECT_EQ(gpu.batches, std::max(1u, (expected + width - 1) / width));
    }
}
