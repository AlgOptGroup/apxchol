#include "apxchol/graph/graph.h"
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

namespace {
struct item { bool backbone; };
struct omp_settings {
#ifdef _OPENMP
    int threads = omp_get_max_threads(), dynamic = omp_get_dynamic();
    ~omp_settings() { omp_set_num_threads(threads); omp_set_dynamic(dynamic); }
#endif
    void set(int threads) {
#ifdef _OPENMP
        omp_set_dynamic(0); omp_set_num_threads(threads);
#else
        (void)threads;
#endif
    }
};
constexpr std::size_t block_items = 16384;
static_assert(apxchol::detail::residual_normalization_block_items == block_items);
std::uint64_t bits(double x) { return std::bit_cast<std::uint64_t>(x); }
auto subject(const std::vector<item>& items, const std::vector<double>& importance,
             double target) {
    return apxchol::detail::normalize_residual_importance(
        std::span<const item>(items), std::span<const double>(importance), target);
}
template<class Scalar>
Scalar scalar_reference(const std::vector<item>& items,
                        const std::vector<double>& importance, double target) {
    Scalar sum = 0;
    for (std::size_t i = 0; i < items.size(); ++i)
        if (!items[i].backbone) sum += static_cast<Scalar>(importance[i]);
    Scalar scale = target > 0.0 && sum > 0 ? static_cast<Scalar>(target) / sum : 0;
    for (unsigned pass = 0; pass < 6 && scale > 0; ++pass) {
        Scalar expected = 0;
        for (std::size_t i = 0; i < items.size(); ++i)
            if (!items[i].backbone)
                expected += std::min(Scalar{1}, scale * static_cast<Scalar>(importance[i]));
        if (expected <= 0) break;
        scale *= static_cast<Scalar>(target) / expected;
    }
    return scale;
}
void fill(std::vector<item>& items, std::vector<double>& importance,
          std::size_t n, unsigned pattern) {
    items.resize(n); importance.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        items[i].backbone = i % 11 == 0;
        if (pattern == 0) importance[i] = 1.0;
        else if (pattern == 1)
            importance[i] = std::sqrt(0.125 + static_cast<double>((i * 13) % 97));
        else importance[i] = std::ldexp(1.0 + static_cast<double>(i % 7) / 8.0,
                                       static_cast<int>(i % 101) - 50);
    }
}
std::size_t off_tree(const std::vector<item>& items) {
    return static_cast<std::size_t>(std::count_if(items.begin(), items.end(),
        [](const item& x) { return !x.backbone; }));
}
}

TEST(ResidualBlockedNormalization, SingleBlockKeepsLegacyArithmetic) {
    omp_settings settings;
    for (std::size_t n : {std::size_t{0}, std::size_t{1}, block_items - 1, block_items}) {
        std::vector<item> items; std::vector<double> importance;
        fill(items, importance, n, 1);
        const double target = 0.25 * off_tree(items);
        const double reference = scalar_reference<double>(items, importance, target);
        for (int threads : {1, 4, 72}) {
            settings.set(threads);
            const auto result = subject(items, importance, target);
            EXPECT_EQ(bits(result.scale), bits(reference));
            EXPECT_EQ(result.blocks, n == 0 ? 0u : 1u);
        }
    }
}

TEST(ResidualBlockedNormalization, FixedBlocksAreBitIdenticalAcrossTeamsAndRepeats) {
    omp_settings settings;
    // The last fixture has more blocks than the largest requested team.
    for (std::size_t n : {block_items + 1, 3 * block_items + 7, 73 * block_items + 5})
        for (unsigned pattern = 0; pattern < 3; ++pattern) {
            SCOPED_TRACE(::testing::Message() << "n=" << n << " pattern=" << pattern);
            std::vector<item> items; std::vector<double> importance;
            fill(items, importance, n, pattern);
            const double target = 0.25 * off_tree(items);
            settings.set(1);
            const auto reference = subject(items, importance, target);
            ASSERT_TRUE(std::isfinite(reference.scale)); ASSERT_GT(reference.scale, 0.0);
            ASSERT_EQ(reference.passes, 6u);
            for (int threads : {1, 2, 4, 72}) {
                settings.set(threads);
                for (int repeat = 0; repeat < 2; ++repeat) {
                    const auto result = subject(items, importance, target);
                    EXPECT_EQ(bits(result.scale), bits(reference.scale));
                    EXPECT_EQ(result.passes, reference.passes);
                    EXPECT_EQ(result.blocks, reference.blocks);
                }
            }
        }
}

TEST(ResidualBlockedNormalization, FiniteProbabilitiesTrackHigherPrecisionReference) {
    if constexpr (std::numeric_limits<long double>::digits <=
                  std::numeric_limits<double>::digits)
        GTEST_SKIP() << "long double does not provide a higher-precision reference";
    omp_settings settings; settings.set(4);
    for (unsigned pattern = 0; pattern < 3; ++pattern) {
        std::vector<item> items; std::vector<double> importance;
        fill(items, importance, 3 * block_items + 7, pattern);
        for (double keep : {1e-6, 0.25, 0.9, 1.0}) {
            SCOPED_TRACE(::testing::Message() << "pattern=" << pattern << " keep=" << keep);
            const double target = keep * off_tree(items);
            const auto actual = subject(items, importance, target);
            const long double reference = scalar_reference<long double>(items, importance, target);
            ASSERT_GT(reference, 0.0L); ASSERT_TRUE(std::isfinite(reference));
            ASSERT_GT(actual.scale, 0.0); ASSERT_TRUE(std::isfinite(actual.scale));
            // Conservative budget for seven positive blocked sums, six
            // division/multiplication updates, and final probability products.
            // These fixtures stay normal and finite, away from overflow.
            const long double u = std::numeric_limits<double>::epsilon() / 2.0L;
            const long double depth = block_items + actual.blocks;
            const long double bound = 64.0L * depth * u / (1.0L - depth * u);
            EXPECT_LE(std::abs(static_cast<long double>(actual.scale) / reference - 1.0L), bound);
            for (std::size_t i = 0; i < items.size(); ++i) {
                const double p = items[i].backbone ? 1.0
                    : std::min(1.0, actual.scale * importance[i]);
                const long double q = items[i].backbone ? 1.0L
                    : std::min(1.0L, reference * importance[i]);
                ASSERT_GT(p, 0.0); ASSERT_LE(p, 1.0); ASSERT_TRUE(std::isfinite(p));
                EXPECT_LE(std::abs(static_cast<long double>(p) / q - 1.0L), bound);
                const double weight = importance[i] * importance[i];
                const double retained_weight = weight / p;
                ASSERT_TRUE(std::isfinite(retained_weight));
                // Conditional Bernoulli/HT mean, not a claim that a finite
                // pseudorandom sample average is exactly its expectation.
                EXPECT_NEAR(p * retained_weight, weight,
                            8 * std::numeric_limits<double>::epsilon() * weight);
            }
        }
    }
}

TEST(ResidualBlockedNormalization, PersistentTeamZeroAndNonfiniteGuardsAreUniform) {
    omp_settings settings;
    std::vector<item> items(3 * block_items + 1, {true});
    std::vector<double> importance(items.size(), 1.0);
    for (int threads : {1, 4, 72}) {
        settings.set(threads);
        for (double target : {0.0, 1.0, -1.0}) {
            const auto result = subject(items, importance, target);
            EXPECT_EQ(bits(result.scale), bits(0.0)); EXPECT_EQ(result.passes, 0u);
        }
        items.front().backbone = false;
        const double inf = std::numeric_limits<double>::infinity();
        const auto infinite = subject(items, importance, inf);
        EXPECT_EQ(infinite.scale, inf); EXPECT_EQ(infinite.passes, 6u);
        const auto nan_target = subject(items, importance,
            std::numeric_limits<double>::quiet_NaN());
        EXPECT_EQ(bits(nan_target.scale), bits(0.0)); EXPECT_EQ(nan_target.passes, 0u);
        items.front().backbone = true;
        // Positive initial scale, but four individual probability products
        // round below denorm_min()/2. The expected<=0 exit must be uniform.
        const std::array<std::size_t, 4> live{0, block_items,
                                            block_items + 1, 2 * block_items};
        for (auto i : live) { items[i].backbone = false; importance[i] = 1e-200; }
        const double target = std::numeric_limits<double>::denorm_min();
        const auto underflow = subject(items, importance, target);
        EXPECT_GT(underflow.scale, 0.0); EXPECT_EQ(underflow.passes, 1u);
        EXPECT_EQ(bits(underflow.scale), bits(target / (4.0 * 1e-200)));
        for (auto i : live) { items[i].backbone = true; importance[i] = 1.0; }
    }
}

namespace {
template<class Incidence>
class ResidualBlockedGraph : public ::testing::Test {};
using block_storages = ::testing::Types<apxchol::vec_pool_incidence,
                                       apxchol::directed_vec_pool_incidence>;
TYPED_TEST_SUITE(ResidualBlockedGraph, block_storages);
}

TYPED_TEST(ResidualBlockedGraph, CrossBlockSamplingIsRepeatableAndConnected) {
    omp_settings settings;
    constexpr apxchol::node_index n = 193; //18,528 distinct edges: two blocks.
    const auto make = [] {
        apxchol::graph<TypeParam> graph(n);
        for (apxchol::node_index u = 0; u < n; ++u)
            for (apxchol::node_index v = u + 1; v < n; ++v)
                graph.add_edge(u, v, u == 0 ? 1024.0 + v
                    : 1.0 + static_cast<double>((u + v) % 13) / 16.0);
        return graph;
    };
    std::vector<apxchol::node_index> active(n);
    std::iota(active.begin(), active.end(), apxchol::node_index{0});
    std::vector<std::tuple<apxchol::node_index,apxchol::node_index,
                           std::uint64_t,apxchol::node_index>> reference;
    for (int threads : {1, 4, 72}) {
        settings.set(threads);
        auto graph = make();
        const auto stats = apxchol::detail::residual_coalescer<TypeParam>::sparsify(
            graph, active, 0.25, 42);
        ASSERT_GT(stats.distinct_before, block_items);
        EXPECT_EQ(stats.backbone_edges, n - 1);
        EXPECT_GE(stats.kept_edges, stats.backbone_edges);
        EXPECT_GT(stats.min_offtree_probability, 0.0);
        std::vector<std::tuple<apxchol::node_index,apxchol::node_index,
                               std::uint64_t,apxchol::node_index>> actual;
        for (apxchol::node_index u = 0; u < n; ++u) {
            bool has_hub = u == 0;
            for (const auto& entry : graph.adj(u)) {
                const auto v = graph.edge_target(entry, u);
                has_hub |= v == 0;
                actual.emplace_back(u, v, bits(graph.edge_weight(entry)),
                                    graph.edge_multiplicity(entry));
            }
            EXPECT_TRUE(has_hub); //The heavy hub forest preserves connectivity.
        }
        if (threads == 1) reference = actual;
        else EXPECT_EQ(actual, reference);
    }
}
