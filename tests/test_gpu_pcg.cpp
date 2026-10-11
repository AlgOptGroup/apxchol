#include <gtest/gtest.h>
#include <apxchol/solver/detail/gpu_pcg.h>

namespace pcg = apxchol::detail::gpu_pcg;

TEST(GpuPcg, ImmediateAndDeferredSubmissionRespectIterationLimitsAndConvergence) {
    for (bool deferred : {false, true}) for (unsigned batch : {0u, 1u, 4u})
    for (unsigned limit : {0u, 1u, 5u}) for (unsigned stop : {1u, 3u, 9u}) {
        SCOPED_TRACE(::testing::Message() << deferred << ' ' << batch << ' ' << limit << ' ' << stop);
        bool active = true;
        unsigned batches = 0, initializations = 0, iterations = 0, directions = 0, preconditions = 0;
        pcg::solve(limit, batch, [&](bool first, auto&& execute) {
            EXPECT_EQ(first, batches++ == 0);
            execute([&](pcg::operation op, unsigned it) {
                EXPECT_LE(it, limit);
                if (!active) return deferred;
                switch (op) {
                case pcg::operation::precondition:
                    ++preconditions;
                    break;
                case pcg::operation::initial_direction:
                    ++initializations;
                    EXPECT_EQ(it, 0u);
                    break;
                case pcg::operation::multiply:
                    EXPECT_EQ(it, iterations + 1);
                    break;
                case pcg::operation::update_residual:
                    iterations = it;
                    active = it < stop;
                    break;
                case pcg::operation::update_direction:
                    ++directions;
                    break;
                }
                return deferred || active;
            });
            return active;
        });
        const unsigned expected = std::min(limit, stop), width = std::max(batch, 1u);
        EXPECT_EQ(initializations, 1u);
        EXPECT_EQ(iterations, expected);
        EXPECT_EQ(directions, expected - unsigned(stop <= limit));
        EXPECT_EQ(preconditions, directions + 1);
        EXPECT_EQ(batches, std::max(1u, (expected + width - 1) / width));
    }
}
