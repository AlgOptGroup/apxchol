#include <gtest/gtest.h>
#include "apxchol/detail/checkpoint.h"
#include <memory>
#include <utility>

TEST(Checkpoint, CopiesAndMovesOwnTheirScopes) {
    for (const bool nested : {false, true}) {
        SCOPED_TRACE(nested);
        auto original = std::make_unique<apxchol::detail::checkpoint>();
        if (nested) {
            original->descend("outer");
            original->descend("inner");
        }
        (*original)("before");
        const auto before = original->report();
        auto copied = *original;
        copied("copy");
        EXPECT_EQ(original->report(), before);

        apxchol::detail::checkpoint assigned;
        assigned.descend("discarded");
        assigned = *original;
        auto moved = std::move(*original);
        original.reset();
        apxchol::detail::checkpoint move_assigned;
        move_assigned.descend("discarded");
        move_assigned = std::move(assigned);

        for (auto* timer : {&copied, &moved, &move_assigned}) {
            (*timer)("after");
            EXPECT_GT(timer->total(nested ? "outer.inner.after" : "after"), 0.0);
            if (nested) {
                timer->ascend();
                (*timer)("parent");
                EXPECT_GT(timer->total("outer.parent"), 0.0);
                timer->ascend();
            }
            (*timer)("root");
            EXPECT_GT(timer->total("root"), 0.0);
            EXPECT_EQ(timer->total("discarded"), 0.0);
        }
    }
}

namespace {
using Checkpoint = apxchol::detail::checkpoint;
Checkpoint::Clock::time_point checkpoint_now;
Checkpoint::Clock::time_point controlled_now() { return checkpoint_now; }
void at_second(int second) { checkpoint_now = Checkpoint::Clock::time_point(std::chrono::seconds(second)); }
}

TEST(Checkpoint, CompleteIntervalsRetainUnlabelledWorkAndReuseTrees) {
    at_second(0);
    Checkpoint cp(controlled_now);
    Checkpoint::interval first(&cp, "setup");
    cp.descend("setup");
    at_second(3); cp.tick(); // Deliberately unlabelled preparation.
    at_second(7); cp("stage");
    cp.ascend();
    at_second(9);
    EXPECT_DOUBLE_EQ(first.finish(), 9);
    EXPECT_DOUBLE_EQ(first.finish(), 9); // Finalization is idempotent.
    EXPECT_DOUBLE_EQ(cp.total("setup.stage"), 4);
    EXPECT_DOUBLE_EQ(cp.total("setup.unattributed"), 5);
    EXPECT_DOUBLE_EQ(cp.total("setup"), 9);

    at_second(10);
    Checkpoint::interval second(&cp, "setup");
    cp.descend("setup");
    at_second(12); cp("stage");
    cp.ascend();
    at_second(15);
    EXPECT_DOUBLE_EQ(second.finish(), 5);
    EXPECT_DOUBLE_EQ(cp.total("setup"), 14);
    EXPECT_DOUBLE_EQ(cp.total("setup.unattributed"), 8);
}

TEST(Checkpoint, NestedIntervalsCountChildrenOnce) {
    at_second(0);
    Checkpoint cp(controlled_now);
    Checkpoint::interval outer(&cp, "setup");
    cp.descend("setup");
    at_second(2); cp("before");
    Checkpoint::interval inner(&cp, "inner");
    cp.descend("inner");
    at_second(5); cp("stage");
    at_second(6); cp.tick();
    at_second(8);
    EXPECT_DOUBLE_EQ(inner.finish(), 6);
    at_second(9); cp("after");
    cp.ascend();
    at_second(10);
    EXPECT_DOUBLE_EQ(outer.finish(), 10);
    EXPECT_DOUBLE_EQ(cp.total("setup.inner"), 6);
    EXPECT_DOUBLE_EQ(cp.total("setup.inner.unattributed"), 3);
    EXPECT_DOUBLE_EQ(cp.total("setup.unattributed"), 1);
    EXPECT_DOUBLE_EQ(cp.total("setup"), 10);
}

TEST(Checkpoint, RejectsOverlappingStageAccounting) {
    at_second(0);
    Checkpoint cp(controlled_now);
    Checkpoint::interval measured(&cp, "setup");
    cp.descend("setup");
    at_second(5); cp("stage");
    at_second(4); // Simulate an inconsistent enclosing interval.
    EXPECT_THROW(measured.finish(), std::logic_error);
}

TEST(Checkpoint, FailedIntervalsRestoreTheCursor) {
    at_second(0);
    Checkpoint cp(controlled_now);
    try {
        Checkpoint::interval measured(&cp, "setup");
        cp.descend("setup");
        cp.descend("unfinished");
        throw std::runtime_error("fixture");
    } catch (const std::runtime_error&) {}
    at_second(2); cp("root");
    EXPECT_DOUBLE_EQ(cp.total("root"), 2);
    EXPECT_DOUBLE_EQ(cp.total("setup.unfinished.root"), 0);
}

TEST(Checkpoint, LongLivedTreesAllowSubsecondIntervals) {
    at_second(0);
    Checkpoint cp(controlled_now);
    {
        Checkpoint::interval prior(&cp, "setup");
        cp.descend("setup");
        at_second(1000000000); cp("stage");
        cp.ascend();
        EXPECT_DOUBLE_EQ(prior.finish(), 1000000000);
    }
    Checkpoint::interval next(&cp, "setup");
    cp.descend("setup");
    checkpoint_now += std::chrono::milliseconds(1); cp("stage");
    cp.ascend();
    EXPECT_DOUBLE_EQ(next.finish(), 0.001);
}

TEST(Checkpoint, CompleteIntervalsRespectAnExistingGroupingScope) {
    at_second(0);
    Checkpoint cp(controlled_now);
    cp.descend("outer");
    Checkpoint::interval measured(&cp, "setup");
    cp.descend("setup");
    at_second(1); cp("stage");
    cp.ascend();
    at_second(2);
    EXPECT_DOUBLE_EQ(measured.finish(), 2);
    EXPECT_DOUBLE_EQ(cp.total("outer.setup"), 2);
    EXPECT_DOUBLE_EQ(cp.total("outer.setup.unattributed"), 1);
    EXPECT_DOUBLE_EQ(cp.total("setup"), 0);
    EXPECT_DOUBLE_EQ(cp.total("outer"), 2);
    cp.ascend();
}
