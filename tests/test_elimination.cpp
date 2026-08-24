#include <gtest/gtest.h>
#include "apxchol/solver/elimination/elimination.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace apxchol;

namespace {

std::vector<deferred_edge> reference_tree_sample(
        std::vector<weighted_neighbor> neighbors,
        double deg,
        std::uint64_t seed) {
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

TEST(TreeSampler, InverseCdfDirectoryMatchesIndependentFullSearch) {
    for (const node_index degree : {node_index{512}, node_index{2049}}) {
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
                    EXPECT_DOUBLE_EQ(actual[i].w, expected[i].w);
                }
            }
        }
    }
}
