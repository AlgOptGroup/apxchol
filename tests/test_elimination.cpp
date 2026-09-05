#include <gtest/gtest.h>
#include "apxchol/solver/elimination/elimination.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

using namespace apxchol;

namespace {

std::vector<deferred_edge> reference_tree_sample(
        std::vector<weighted_neighbor> neighbors,
        double deg,
        std::uint64_t seed) {
    if (neighbors.empty()) return {};
    std::sort(neighbors.begin(), neighbors.end(),
              [](const auto& a, const auto& b) {
                  return a.weight != b.weight ? a.weight < b.weight
                                              : a.vertex < b.vertex;
              });
    std::vector<double> prefix(neighbors.size());
    prefix[0] = neighbors[0].weight;
    for (size_t i = 1; i < neighbors.size(); ++i)
        prefix[i] = prefix[i - 1] + neighbors[i].weight;

    std::vector<deferred_edge> result;
    edge_emitter out(result);
    random_stream rs{seed};
    for (size_t i = 0; i + 1 < neighbors.size(); ++i) {
        const double suffix_sum = prefix.back() - prefix[i];
        if (suffix_sum <= 0.0) continue;
        const double target = prefix[i] + rs.next_unit() * suffix_sum;
        const auto it = std::upper_bound(
            prefix.begin() + static_cast<std::ptrdiff_t>(i) + 1,
            prefix.end(), target);
        const size_t j = it == prefix.end()
            ? neighbors.size() - 1
            : static_cast<size_t>(it - prefix.begin());
        out(neighbors[i].vertex, neighbors[j].vertex,
            neighbors[i].weight * suffix_sum / deg);
    }
    return result;
}

} // namespace

TEST(EliminatorBounds, TreeReturnsDegMinusOne) {
    EXPECT_EQ(tree_elimination{}.max_clique_edges(0), 0u);
    EXPECT_EQ(tree_elimination{}.max_clique_edges(1), 0u);
    EXPECT_EQ(tree_elimination{}.max_clique_edges(5), 4u);
    // Exact-clique mode: full pair count below the threshold.
    tree_elimination xc{.exact_clique_max_degree = 8};
    EXPECT_EQ(xc.max_clique_edges(5), 10u);
    EXPECT_EQ(xc.max_clique_edges(9), 8u);   // above threshold: sampled, d-1
}

TEST(EliminatorBounds, BoundIsSafeForRandomInput) {
    // Empirical: actual clique edges emitted never exceeds max_clique_edges(deg).
    std::vector<weighted_neighbor> neighbors;
    std::vector<deferred_edge> buf;

    for (node_index deg = 2; deg <= 12; ++deg) {
        neighbors.clear();
        for (node_index i = 0; i < deg; ++i)
            neighbors.emplace_back(i, 1.0 + i);
        const double sum = double(deg) * (deg + 1) / 2.0;

        buf.clear();
        tree_elimination{}.sample_clique(neighbors, sum, /*seed=*/42 + deg,
                                         edge_emitter(buf));
        EXPECT_LE(buf.size(), tree_elimination{}.max_clique_edges(deg))
            << "tree, deg=" << deg;
    }
}

TEST(TreeSampler, MatchesIndependentExactReference) {
    // Retain coverage around the removed degree-512 directory/radix and
    // degree-2048 radix boundaries to guard the byte-identical cleanup.
    for (const node_index degree : {
            node_index{0}, node_index{1}, node_index{2}, node_index{3},
            node_index{15}, node_index{16}, node_index{17},
            node_index{511}, node_index{512},
            node_index{2047}, node_index{2048}, node_index{2049}}) {
        for (int shape = 0; shape < 3; ++shape) {
            std::vector<weighted_neighbor> input;
            input.reserve(degree);
            double sum = 0.0;
            for (node_index i = 0; i < degree; ++i) {
                double weight = 1.0;
                if (shape == 1) {
                    weight = i % 17 == 0
                        ? 1.0 + static_cast<double>(i) * 0.25
                        : 0.001 * static_cast<double>(1 + (i % 11));
                } else if (shape == 2) {
                    weight = i + 4 >= degree
                        ? 1.0 + static_cast<double>(i)
                        : 1e-12 * static_cast<double>(1 + (i % 7));
                }
                input.push_back({degree - 1 - i, weight});
                sum += weight;
            }

            for (const std::uint64_t seed : {
                    std::uint64_t{0}, std::uint64_t{42},
                    std::uint64_t{0x123456789abcdef0ULL}}) {
                const auto expected = reference_tree_sample(input, sum, seed);
                auto production_input = input;
                std::vector<deferred_edge> actual;
                tree_elimination{}.sample_clique(production_input, sum, seed,
                                                  edge_emitter(actual));

                ASSERT_EQ(actual.size(), expected.size());
                for (size_t i = 0; i < expected.size(); ++i) {
                    EXPECT_EQ(actual[i].u, expected[i].u);
                    EXPECT_EQ(actual[i].v, expected[i].v);
                    EXPECT_EQ(std::bit_cast<std::uint64_t>(actual[i].w),
                              std::bit_cast<std::uint64_t>(expected[i].w));
                }
            }
        }
    }
}

TEST(TreeSampler, SuffixUpperBoundPreservesExactBoundarySemantics) {
    const double inf = std::numeric_limits<double>::infinity();
    const double tiny = std::numeric_limits<double>::denorm_min();
    for (const std::vector<double> cdf : {
            std::vector<double>{}, {1.0}, {-0.0, 0.0, 0.0, 1.0},
            {tiny, tiny, 2*tiny, 1.0}, {1.0, 1.0, 1.0, 2.0, 3.0},
            {0x1p53, 0x1p53 + 1.0, 0x1p53 + 2.0, inf}}) {
        std::vector<double> targets{-inf, -0.0, 0.0, 0.5, 1.0, inf,
                                    std::numeric_limits<double>::quiet_NaN()};
        for (double x : cdf) {
            targets.push_back(x);
            targets.push_back(std::nextafter(x, -inf));
            targets.push_back(std::nextafter(x, inf));
        }
        for (std::size_t start=0; start<=cdf.size(); ++start) {
            const auto suffix=std::span<const double>(cdf).subspan(start);
            for (double target : targets) {
                const auto expected=std::upper_bound(suffix.begin(), suffix.end(), target);
                EXPECT_EQ(detail::suffix_upper_bound(suffix,target),
                          static_cast<std::size_t>(expected-suffix.begin()));
            }
        }
    }
}

TEST(TreeSampler, SuffixUpperBoundMatchesAllRandomizedSuffixes) {
    std::mt19937_64 rng(42);
    for (std::size_t size=0; size<=257; ++size) {
        std::vector<double> cdf(size);
        double total=0;
        for (auto& x : cdf) {
            total += static_cast<double>(rng()%5); // includes repeated CDF entries
            x=total;
        }
        for (std::size_t start=0; start<=size; ++start) {
            const auto suffix=std::span<const double>(cdf).subspan(start);
            for (unsigned trial=0; trial<8; ++trial) {
                const double target=static_cast<double>(rng()%(2*size+3))/2;
                const auto expected=std::upper_bound(suffix.begin(), suffix.end(), target);
                ASSERT_EQ(detail::suffix_upper_bound(suffix,target),
                          static_cast<std::size_t>(expected-suffix.begin()));
            }
        }
    }
}
