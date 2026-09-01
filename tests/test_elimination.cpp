#include <gtest/gtest.h>
#include "apxchol/solver/elimination/elimination.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
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

void expect_edges_exact(const std::vector<deferred_edge>& actual,
                        const std::vector<deferred_edge>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(actual[i].u, expected[i].u) << "edge=" << i;
        EXPECT_EQ(actual[i].v, expected[i].v) << "edge=" << i;
        EXPECT_DOUBLE_EQ(actual[i].w, expected[i].w) << "edge=" << i;
    }
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

TEST(TreeSampler, LocalTwoTreeSparsifierIsAlwaysConnected) {
    const std::vector<weighted_neighbor> input{
        {0, 1.0}, {1, 2.0}, {2, 3.0}, {3, 5.0}, {4, 8.0}, {5, 13.0}};
    tree_elimination local{.local_two_tree_keep = 0.25};
    EXPECT_EQ(local.max_clique_edges(input.size()), 2 * (input.size() - 1));
    for (std::uint64_t seed = 0; seed < 1000; ++seed) {
        auto neighbors = input;
        std::vector<deferred_edge> edges;
        local.sample_clique(neighbors, 32.0, seed, edge_emitter(edges));
        ASSERT_GE(edges.size(), input.size() - 1);
        ASSERT_LE(edges.size(), 2 * (input.size() - 1));

        std::vector<node_index> parent(input.size());
        std::iota(parent.begin(), parent.end(), node_index{0});
        auto find = [&](node_index v) {
            node_index root = v;
            while (parent[root] != root) root = parent[root];
            while (parent[v] != v) {
                const node_index next = parent[v];
                parent[v] = root;
                v = next;
            }
            return root;
        };
        for (const auto& edge : edges) {
            ASSERT_GT(edge.w, 0.0);
            const node_index u = find(edge.u);
            const node_index v = find(edge.v);
            if (u != v) parent[u] = v;
        }
        const node_index component = find(0);
        for (node_index v = 1; v < input.size(); ++v)
            EXPECT_EQ(find(v), component) << "seed=" << seed;
    }
}

TEST(TreeSampler, LocalTwoTreeSparsifierIsUnbiased) {
    const std::vector<weighted_neighbor> input{
        {0, 1.0}, {1, 2.0}, {2, 4.0}, {3, 8.0}};
    constexpr double degree = 15.0;
    constexpr std::uint64_t trials = 200000;
    tree_elimination local{.local_two_tree_keep = 0.5};
    double sums[4][4]{};
    for (std::uint64_t seed = 0; seed < trials; ++seed) {
        auto neighbors = input;
        std::vector<deferred_edge> edges;
        local.sample_clique(neighbors, degree, seed, edge_emitter(edges));
        for (const auto& edge : edges) {
            const node_index u = std::min(edge.u, edge.v);
            const node_index v = std::max(edge.u, edge.v);
            sums[u][v] += edge.w;
        }
    }
    for (node_index u = 0; u < input.size(); ++u)
        for (node_index v = u + 1; v < input.size(); ++v) {
            const double expected =
                input[u].weight * input[v].weight / degree;
            EXPECT_NEAR(sums[u][v] / static_cast<double>(trials), expected,
                        0.04 * expected + 1e-4)
                << "edge=" << u << ',' << v;
        }
}

TEST(TreeSampler, LocalTwoTreeSpreadGateIsExactNoOpOnNarrowStar) {
    const std::vector<weighted_neighbor> input{
        {0, 1.0}, {1, 2.0}, {2, 3.0}, {3, 5.0}, {4, 8.0}};
    const tree_elimination baseline;
    const tree_elimination gated{
        .local_two_tree_keep = 0.2,
        .local_two_tree_trigger_rel = 1e-4};
    for (std::uint64_t seed = 0; seed < 100; ++seed) {
        auto baseline_input = input;
        auto gated_input = input;
        std::vector<deferred_edge> expected;
        std::vector<deferred_edge> actual;
        baseline.sample_clique(baseline_input, 19.0, seed,
                               edge_emitter(expected));
        gated.sample_clique(gated_input, 19.0, seed, edge_emitter(actual));
        expect_edges_exact(actual, expected);
    }
}

TEST(TreeSampler, LocalTwoTreeSpreadGateMatchesUngatedOnSeparatedStar) {
    const std::vector<weighted_neighbor> input{
        {0, 1e-8}, {1, 2.0}, {2, 4.0}, {3, 8.0}};
    const tree_elimination ungated{.local_two_tree_keep = 0.2};
    const tree_elimination gated{
        .local_two_tree_keep = 0.2,
        .local_two_tree_trigger_rel = 1e-4};
    for (std::uint64_t seed = 0; seed < 100; ++seed) {
        auto ungated_input = input;
        auto gated_input = input;
        std::vector<deferred_edge> expected;
        std::vector<deferred_edge> actual;
        ungated.sample_clique(ungated_input, 14.00000001, seed,
                              edge_emitter(expected));
        gated.sample_clique(gated_input, 14.00000001, seed,
                            edge_emitter(actual));
        expect_edges_exact(actual, expected);
    }
}

TEST(TreeSampler, LocalTwoTreeDegreeTwoUsesExactBaselinePath) {
    const std::vector<weighted_neighbor> input{{0, 1e-8}, {1, 8.0}};
    const tree_elimination baseline;
    const tree_elimination local{.local_two_tree_keep = 0.2};
    for (std::uint64_t seed = 0; seed < 100; ++seed) {
        auto baseline_input = input;
        auto local_input = input;
        std::vector<deferred_edge> expected;
        std::vector<deferred_edge> actual;
        baseline.sample_clique(baseline_input, 8.00000001, seed,
                               edge_emitter(expected));
        local.sample_clique(local_input, 8.00000001, seed,
                            edge_emitter(actual));
        expect_edges_exact(actual, expected);
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

TEST(TreeSampler, AcceleratedCanonicalOrderMatchesComparator) {
    const auto canonical_less = [](const auto& a, const auto& b) {
        return a.weight != b.weight ? a.weight < b.weight
                                    : a.vertex < b.vertex;
    };
    for (const node_index degree : {
            node_index{511}, node_index{512},
            node_index{1024}, node_index{2048},
            node_index{4096}}) {
        for (int shape = 0; shape < 4; ++shape) {
            std::vector<weighted_neighbor> input;
            input.reserve(degree);
            for (node_index i = 0; i < degree; ++i) {
                double weight = 1.0;
                if (shape == 1)
                    weight = 0.125 * static_cast<double>(1 + (i * 37) % 29);
                else if (shape == 2 && i == 3)
                    weight = 2.0;  // misses the spread probes intentionally
                else if (shape == 3)
                    weight = static_cast<double>(1 + (i % 7));
                input.push_back({degree - 1 - i, weight});
            }

            auto expected = input;
            std::sort(expected.begin(), expected.end(), canonical_less);
            auto actual = input;
            const bool accelerated = detail::radix_sort_neighbors(actual);
            EXPECT_EQ(accelerated, degree >= 512);
            if (!accelerated)
                std::sort(actual.begin(), actual.end(), canonical_less);

            ASSERT_EQ(actual.size(), expected.size());
            for (size_t i = 0; i < expected.size(); ++i) {
                EXPECT_EQ(actual[i].vertex, expected[i].vertex);
                EXPECT_DOUBLE_EQ(actual[i].weight, expected[i].weight);
            }
        }
    }
}
