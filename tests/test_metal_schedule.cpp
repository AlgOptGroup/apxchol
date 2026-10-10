#include "apxchol/solver/detail/metal_schedule.h"
#include <array>
#include <gtest/gtest.h>
#include <random>

namespace {
namespace ms = apxchol::detail::metal_schedule;
namespace df = apxchol::detail::gpu_dataflow;

struct triangle {
    std::vector<unsigned> ptr{0}, col;
    std::vector<float> val;
};

triangle make_triangle(unsigned n, bool reverse, unsigned pattern) {
    triangle a;
    std::mt19937 rng(19);
    for (unsigned row = 0; row < n; ++row) {
        std::vector<unsigned> dependencies;
        const unsigned q = reverse ? n - 1 - row : row;
        if (pattern == 1 && q)
            dependencies.push_back(q - 1);
        if (pattern == 2 && q) {
            for (unsigned k = 0; k < (q % 13 ? 3u : 300u); ++k)
                dependencies.push_back(rng() % q);
        }
        if (pattern == 3 && q)
            for (unsigned k = 0; k < q; ++k)
                dependencies.push_back(k);
        for (auto &other : dependencies)
            if (reverse)
                other = n - 1 - other;
        dependencies.push_back(row);
        std::sort(dependencies.begin(), dependencies.end());
        dependencies.erase(std::unique(dependencies.begin(), dependencies.end()), dependencies.end());
        for (unsigned other : dependencies) {
            a.col.push_back(other);
            a.val.push_back(other == row ? 2.f : -0.01f);
        }
        a.ptr.push_back(a.col.size());
    }
    return a;
}

void check_schedule(const triangle &a, bool reverse) {
    const unsigned n = a.ptr.size() - 1;
    const auto s = ms::build(a.ptr, a.col, a.val, reverse);
    const auto &p = s.tasks;
    std::vector<int> stage(n, -1), block(n, -1);
    std::vector<unsigned> position(n);
    std::vector<unsigned> row_count(n), finalizers(n);
    std::vector<std::vector<std::pair<unsigned, unsigned>>> slices(n);
    ASSERT_EQ(p.words.size() % 32, 0u);
    for (unsigned st = 0; st < s.stages.size(); ++st) {
        for (unsigned b = s.stages[st].first; b < s.stages[st].last; ++b) {
            for (unsigned w = p.blocks[b] * 32; w < p.blocks[b + 1] * 32;) {
                const unsigned word = p.words[w];
                if (word == df::empty) {
                    ++w;
                    continue;
                }
                unsigned row, width;
                if (word & df::special_bit) {
                    ASSERT_LT(word & ~df::special_bit, p.segments.size());
                    const auto t = p.segments[word & ~df::special_bit];
                    row = t.row & ~df::special_bit;
                    width = 32;
                    if (t.row & df::special_bit) {
                        ++finalizers[row];
                        EXPECT_EQ(t.output, row);
                        EXPECT_LE(t.end, n + p.slots);
                        EXPECT_GE(t.begin, n);
                    } else {
                        slices[row].emplace_back(t.begin, t.end);
                        EXPECT_GE(t.output, n);
                        EXPECT_LT(t.output, n + p.slots);
                    }
                } else {
                    row = word;
                    width = 1;
                    while (width < 32 && width * 8 < a.ptr[row + 1] - a.ptr[row] - 1)
                        width *= 2;
                    ++row_count[row];
                }
                ASSERT_LT(row, n);
                EXPECT_EQ(w % width, 0u);
                for (unsigned lane = 0; lane < width; ++lane)
                    ASSERT_EQ(p.words[w + lane], word);
                position[row] = w;
                stage[row] = st;
                block[row] = b;
                w += width;
            }
        }
    }
    for (unsigned row = 0; row < n; ++row) {
        const unsigned begin = a.ptr[row] + unsigned(reverse), end = a.ptr[row + 1] - unsigned(!reverse);
        EXPECT_EQ(row_count[row] + finalizers[row], unsigned(begin != end));
        if (!slices[row].empty()) {
            unsigned next = begin;
            for (auto [first, last] : slices[row]) {
                EXPECT_EQ(first, next);
                EXPECT_GT(last, first);
                next = last;
            }
            EXPECT_EQ(next, end);
            EXPECT_EQ(finalizers[row], 1u);
        }
        for (unsigned e = begin; e < end; ++e) {
            const unsigned other = a.col[e];
            EXPECT_TRUE(stage[other] < stage[row] ||
                        (stage[other] == stage[row] && block[other] == block[row] &&
                         position[other] < position[row]));
        }
    }
}
} // namespace

TEST(MetalSchedule, TasksCoverRowsAndConfineEveryDependencyToItsProgressScope) {
    for (unsigned n : {0u, 1u, 31u, 129u, 513u})
        for (bool reverse : {false, true})
            for (unsigned pattern = 0; pattern < 4; ++pattern) {
                SCOPED_TRACE(::testing::Message() << n << " reverse=" << reverse << " pattern=" << pattern);
                check_schedule(make_triangle(n, reverse, pattern), reverse);
            }
}

TEST(MetalSchedule, RejectsInvalidStructureAndValues) {
    auto a = make_triangle(5, false, 1);
    a.ptr[2] = a.ptr[1];
    EXPECT_THROW(ms::build(a.ptr, a.col, a.val, false), std::invalid_argument);
    a = make_triangle(5, false, 1);
    a.col[0] = 1;
    EXPECT_THROW(ms::build(a.ptr, a.col, a.val, false), std::invalid_argument);
    a = make_triangle(5, false, 1);
    a.val[0] = 0;
    EXPECT_THROW(ms::build(a.ptr, a.col, a.val, false), std::invalid_argument);
    for (float value : {0x1p-101f, 0x1p101f, std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
        a = make_triangle(5, false, 1);
        a.val[0] = value;
        EXPECT_THROW(ms::build(a.ptr, a.col, a.val, false), std::domain_error);
    }
}

TEST(GpuDataflowPlan, BoundsPartialStorageForVeryLongRows) {
    const unsigned ptr[] = {0, 1, 20000001};
    const auto plan = df::pack(ptr, {2, false}, {0, 2});
    EXPECT_LE(plan.slots, 4096u);
    EXPECT_EQ(plan.segments.size(), plan.slots + 1);
    EXPECT_EQ(plan.segments.front().begin, 1u);
    EXPECT_EQ(plan.segments[plan.slots - 1].end, 20000000u);
}
