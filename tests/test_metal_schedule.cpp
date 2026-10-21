#include "apxchol/solver/detail/metal_schedule.h"
#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <random>

#define APXCHOL_GPU_CODE(...) __VA_ARGS__
#define APXCHOL_GPU_FUNCTION inline
#include "../src/metal_setup/planner.inc"
#undef APXCHOL_GPU_FUNCTION
#undef APXCHOL_GPU_CODE

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

template<unsigned Lanes = 32>
void check_schedule(const triangle &a, bool reverse) {
    SCOPED_TRACE(::testing::Message() << "lanes=" << Lanes);
    const unsigned n = a.ptr.size() - 1;
    const auto s = ms::build<float, Lanes>(a.ptr, a.col, a.val, reverse);
    const auto &p = s.tasks;
    std::vector<int> stage(n, -1), block(n, -1);
    std::vector<unsigned> position(n);
    std::vector<unsigned> row_count(n), finalizers(n);
    std::vector<std::vector<std::pair<unsigned, unsigned>>> slices(n);
    ASSERT_EQ(p.words.size() % Lanes, 0u);
    for (unsigned st = 0; st < s.stages.size(); ++st) {
        for (unsigned b = s.stages[st].first; b < s.stages[st].last; ++b) {
            for (unsigned w = p.blocks[b] * Lanes; w < p.blocks[b + 1] * Lanes;) {
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
                    width = Lanes;
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
                    while (width < Lanes && width * 8 < a.ptr[row + 1] - a.ptr[row] - 1)
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
                const auto triangle = make_triangle(n, reverse, pattern);
                check_schedule<8>(triangle, reverse);
                check_schedule<32>(triangle, reverse);
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

TEST(MetalSchedule, OffDiagonalsOnlyRequireFiniteValues) {
    for (bool reverse : {false, true}) {
        auto a = make_triangle(2, reverse, 1);
        for (float value : {0x1p-110f, -0x1p-110f, std::numeric_limits<float>::min(),
                            -std::numeric_limits<float>::denorm_min()}) {
            SCOPED_TRACE(::testing::Message() << "reverse=" << reverse << " value=" << value);
            a.val[1] = value;
            EXPECT_NO_THROW(check_schedule(a, reverse));
        }
        for (float value : {std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN()}) {
            SCOPED_TRACE(::testing::Message() << "reverse=" << reverse << " value=" << value);
            a.val[1] = value;
            EXPECT_THROW(ms::build(a.ptr, a.col, a.val, reverse), std::domain_error);
        }
    }
}

TEST(GpuDataflowPlan, BoundsPartialStorageForVeryLongRows) {
    const unsigned ptr[] = {0, 1, 20000001};
    const auto plan = df::pack(ptr, {2, false}, std::array{0u, 2u});
    EXPECT_LE(plan.slots, 4096u);
    EXPECT_EQ(plan.segments.size(), plan.slots + 1);
    EXPECT_EQ(plan.segments.front().begin, 1u);
    EXPECT_EQ(plan.segments[plan.slots - 1].end, 20000000u);
}

TEST(MetalSchedule, Float16UsesSeparateDiagonals) {
    const std::vector<unsigned> ptr{0, 1, 3}, col{0, 0, 1};
    const std::vector<_Float16> values{_Float16(1e10f), _Float16(-1), _Float16(1e10f)};
    for (float d : {1e10f, 0x1p-100f, 0x1p100f}) {
        const std::vector<float> diagonal{d, d};
        const auto plan = apxchol::detail::metal_schedule::build<_Float16>(ptr, col, values, false, diagonal);
        EXPECT_EQ(plan.levels, 2u);
    }
    const std::vector<float> missing{1};
    EXPECT_THROW(apxchol::detail::metal_schedule::build<_Float16>(ptr, col, values, false, missing),
                 std::invalid_argument);
}

TEST(MetalSchedule, CompactedBlocksDoNotRescanEmptyPrefixes) {
    namespace rf = resident_finalize;
    struct counted_index {
        unsigned value;
        unsigned* reads;
        operator unsigned() const { ++*reads; return value; }
    };
    struct memory {
        unsigned n;
        bool reverse;
        unsigned rounds;
        const unsigned* round_ptr;
        unsigned* blocks;
        rf::Step* steps;
        rf::PlanInfo* info;
        void fail(unsigned) const { ADD_FAILURE() << "invalid round boundaries"; }
    };
    const unsigned n = 10 * rf::block_rows + 7;
    const unsigned round_ptr[] = {0, 8 * rf::block_rows, 8 * rf::block_rows + 3, n};
    for (bool reverse : {false, true}) for (bool empty_tail : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "reverse=" << reverse << " empty_tail=" << empty_tail);
        unsigned reads = 0;
        std::vector<counted_index> ptr(n + 1, {0, &reads});
        for (unsigned row = 0; row < n; ++row) {
            const unsigned q = reverse ? n - 1 - row : row;
            const bool work = (q >= round_ptr[1] && q < round_ptr[2]) || (!empty_tail && q >= n - 4);
            ptr[row + 1].value = ptr[row].value + 1 + unsigned(work);
        }
        std::vector<unsigned> blocks(n + 1);
        std::vector<rf::Step> steps(n);
        rf::PlanInfo info{};
        memory plan{n, reverse, 3, round_ptr, blocks.data(), steps.data(), &info};
        rf::plan(plan);
        std::vector<df::counts> counts(info.blocks + 1);
        for (unsigned b = 0; b < info.blocks; ++b)
            counts[b] = df::pack_rows<false>(ptr.data(), {n, reverse}, blocks[b], blocks[b + 1], df::chunk_size);
        rf::compact(plan, counts.data(), df::counts{});
        ASSERT_EQ(info.blocks, empty_tail ? 1u : 2u);
        ASSERT_GT(blocks[1], rf::block_rows);
        if (empty_tail) EXPECT_LT(blocks[info.blocks], n);
        else EXPECT_EQ(blocks[info.blocks], n);
        for (unsigned b = 0; b < info.blocks; ++b) {
            const auto expected = df::pack(ptr.data(), {n, reverse}, std::array{blocks[b], blocks[b + 1]});
            ASSERT_TRUE(expected.segments.empty());
            std::vector<unsigned> words(expected.words.size(), df::empty);
            reads = 0;
            const auto count = df::pack_rows<true>(ptr.data(), {n, reverse},
                rf::block_begin(blocks[b], blocks[b + 1]), blocks[b + 1], df::chunk_size, {}, words.data());
            EXPECT_LE(reads, 2u * rf::block_rows);
            EXPECT_EQ(count.words, counts[b + 1].words - counts[b].words);
            EXPECT_EQ(words, expected.words);
        }
    }
}
