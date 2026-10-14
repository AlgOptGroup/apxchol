#include <gtest/gtest.h>
#include "apxchol/checkpoint.h"
#include <memory>
#include <utility>

TEST(Checkpoint, CopiesAndMovesOwnTheirScopes) {
    for (const bool nested : {false, true}) {
        SCOPED_TRACE(nested);
        auto original = std::make_unique<apxchol::checkpoint>();
        if (nested) {
            original->descend("outer");
            original->descend("inner");
        }
        (*original)("before");
        const auto before = original->report();
        auto copied = *original;
        copied("copy");
        EXPECT_EQ(original->report(), before);

        apxchol::checkpoint assigned;
        assigned.descend("discarded");
        assigned = *original;
        auto moved = std::move(*original);
        original.reset();
        apxchol::checkpoint move_assigned;
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
