#include <gtest/gtest.h>
#include "apxchol/solver/elimination/elimination.h"
#include "apxchol/solver/elimination/volume_tree.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <numeric>
#include <queue>
#include <span>
#include <string>
#include <utility>
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

size_t reference_linear_cdf(std::span<const double> weights, double target) {
    double cumulative = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
        cumulative += weights[i];
        if (target < cumulative) return i;
    }
    return weights.size() - 1;
}

std::vector<size_t> reference_prufer_code(
        std::span<const double> weights,
        std::uint64_t seed) {
    const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    random_stream random{seed};
    std::vector<size_t> code(weights.size() - 2);
    for (size_t& symbol : code)
        symbol = reference_linear_cdf(weights, random.next_unit() * total);
    return code;
}

std::vector<std::pair<size_t, size_t>> reference_heap_decode(
        std::span<const size_t> code,
        size_t n) {
    std::vector<size_t> degree(n, 1);
    for (const size_t symbol : code) ++degree[symbol];

    std::priority_queue<size_t, std::vector<size_t>, std::greater<>> leaves;
    for (size_t i = 0; i < n; ++i)
        if (degree[i] == 1) leaves.push(i);

    std::vector<std::pair<size_t, size_t>> edges;
    edges.reserve(n - 1);
    for (const size_t symbol : code) {
        const size_t leaf = leaves.top();
        leaves.pop();
        edges.emplace_back(leaf, symbol);
        if (--degree[symbol] == 1) leaves.push(symbol);
    }
    const size_t first = leaves.top();
    leaves.pop();
    edges.emplace_back(first, leaves.top());
    return edges;
}

std::vector<deferred_edge> reference_volume_sample(
        std::vector<weighted_neighbor> neighbors,
        double deg,
        std::uint64_t seed) {
    std::sort(neighbors.begin(), neighbors.end(),
              [](const auto& a, const auto& b) {
                  return a.weight != b.weight ? a.weight < b.weight
                                              : a.vertex < b.vertex;
              });
    std::vector<double> weights;
    weights.reserve(neighbors.size());
    for (const auto& neighbor : neighbors) weights.push_back(neighbor.weight);
    const double W = std::accumulate(weights.begin(), weights.end(), 0.0);
    const auto code = reference_prufer_code(weights, seed);
    const auto indices = reference_heap_decode(code, neighbors.size());

    std::vector<deferred_edge> result;
    result.reserve(indices.size());
    for (const auto& [i, j] : indices) {
        const double wi = neighbors[i].weight;
        const double wj = neighbors[j].weight;
        result.push_back({neighbors[i].vertex, neighbors[j].vertex,
                          (W / deg) * wi * wj / (wi + wj)});
    }
    return result;
}

void expect_same_edges(const std::vector<deferred_edge>& actual,
                       const std::vector<deferred_edge>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        SCOPED_TRACE("edge " + std::to_string(i));
        EXPECT_EQ(actual[i].u, expected[i].u);
        EXPECT_EQ(actual[i].v, expected[i].v);
        EXPECT_DOUBLE_EQ(actual[i].w, expected[i].w);
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

TEST(EliminatorBounds, VolumeTreeReturnsDegreeMinusOne) {
    EXPECT_EQ(volume_tree_elimination{}.max_clique_edges(0), 0u);
    EXPECT_EQ(volume_tree_elimination{}.max_clique_edges(1), 0u);
    EXPECT_EQ(volume_tree_elimination{}.max_clique_edges(5), 4u);
    volume_tree_elimination exact{.exact_clique_max_degree = 8};
    EXPECT_EQ(exact.max_clique_edges(5), 10u);
    EXPECT_EQ(exact.max_clique_edges(9), 8u);
}

TEST(Bkz26PruferInternals, InverseCdfMatchesIndependentLinearScan) {
    const std::array<double, 7> weights{
        0.125, 0.5, 1.0, 3.0, 3.0, 8.0, 21.0};
    std::array<double, weights.size()> prefix{};
    std::partial_sum(weights.begin(), weights.end(), prefix.begin());

    std::vector<double> targets{0.0, std::nextafter(prefix.back(), 0.0)};
    for (const double boundary : prefix) {
        if (boundary < prefix.back()) {
            targets.push_back(std::nextafter(boundary, 0.0));
            targets.push_back(boundary);
        }
    }
    random_stream random{0x123456789abcdef0ULL};
    for (int i = 0; i < 100; ++i)
        targets.push_back(random.next_unit() * prefix.back());

    for (const double target : targets) {
        EXPECT_EQ(detail::bkz26_inverse_cdf(prefix, target),
                  reference_linear_cdf(weights, target));
    }
}

TEST(Bkz26PruferInternals, LinearDecodeMatchesIndependentHeapDecode) {
    const std::vector<std::vector<node_index>> codes{
        {}, {3, 0, 1}, {0, 0, 0, 0}, {4, 1, 4, 2, 4, 0}};
    for (const auto& code : codes) {
        const size_t n = code.size() + 2;
        std::vector<node_index> degrees(n, node_index{1});
        std::vector<size_t> reference_code;
        reference_code.reserve(code.size());
        for (const node_index symbol : code) {
            ++degrees[symbol];
            reference_code.push_back(static_cast<size_t>(symbol));
        }
        std::vector<std::pair<size_t, size_t>> actual;
        detail::bkz26_decode_prufer(
            code, degrees,
            [&](size_t u, size_t v) { actual.emplace_back(u, v); });
        EXPECT_EQ(actual, reference_heap_decode(reference_code, n));
    }
}

TEST(Bkz26PruferSampler, Seed42HasExactOrderedOutput) {
    std::vector<weighted_neighbor> input{
        {7, 1.0}, {2, 1.0}, {11, 1.0}, {5, 1.0}, {13, 1.0}};
    auto reversed = input;
    std::reverse(reversed.begin(), reversed.end());
    std::vector<deferred_edge> actual;
    std::vector<deferred_edge> reordered;
    volume_tree_elimination{}.sample_clique(
        input, 5.0, 42, edge_emitter(actual));
    volume_tree_elimination{}.sample_clique(
        reversed, 5.0, 42, edge_emitter(reordered));

    const std::vector<deferred_edge> expected{
        {7, 11, 0.5}, {11, 2, 0.5}, {2, 5, 0.5}, {5, 13, 0.5}};
    expect_same_edges(actual, expected);
    expect_same_edges(reordered, expected);
}

TEST(Bkz26PruferSampler, MatchesIndependentCdfAndHeapDecode) {
    const std::vector<weighted_neighbor> input{
        {0, 1.0}, {1, 7.0}, {2, 2.0}, {3, 11.0}, {4, 3.0}, {5, 5.0}};
    const double degree_with_excess = 32.0;
    for (const std::uint64_t seed : {
            std::uint64_t{0}, std::uint64_t{42},
            std::uint64_t{0x123456789abcdef0ULL}}) {
        const auto expected = reference_volume_sample(
            input, degree_with_excess, seed);
        auto production_input = input;
        std::vector<deferred_edge> actual;
        volume_tree_elimination{}.sample_clique(
            production_input, degree_with_excess, seed, edge_emitter(actual));
        expect_same_edges(actual, expected);
    }
}

TEST(Bkz26PruferSampler, PureLaplacianAndSddmExtensionHaveExactWeights) {
    const std::vector<weighted_neighbor> original{
        {7, 4.0}, {2, 1.0}, {11, 3.0}, {5, 2.0}, {13, 5.0}};
    constexpr double W = 15.0;
    constexpr double sddm_degree = 18.0;

    auto laplacian_input = original;
    auto sddm_input = original;
    std::vector<deferred_edge> laplacian;
    std::vector<deferred_edge> sddm;
    volume_tree_elimination{}.sample_clique(
        laplacian_input, W, 0x12345678ULL, edge_emitter(laplacian));
    volume_tree_elimination{}.sample_clique(
        sddm_input, sddm_degree, 0x12345678ULL, edge_emitter(sddm));
    ASSERT_EQ(laplacian.size(), original.size() - 1);
    ASSERT_EQ(sddm.size(), laplacian.size());

    std::map<node_index, double> weights;
    std::map<node_index, node_index> parent;
    for (const auto& neighbor : original) {
        weights.emplace(neighbor.vertex, neighbor.weight);
        parent.emplace(neighbor.vertex, neighbor.vertex);
    }
    auto root = [&](node_index vertex) {
        while (parent.at(vertex) != vertex) vertex = parent.at(vertex);
        return vertex;
    };

    for (size_t i = 0; i < laplacian.size(); ++i) {
        EXPECT_EQ(laplacian[i].u, sddm[i].u);
        EXPECT_EQ(laplacian[i].v, sddm[i].v);
        const double wu = weights.at(laplacian[i].u);
        const double wv = weights.at(laplacian[i].v);
        const double paper_weight = wu * wv / (wu + wv);
        EXPECT_DOUBLE_EQ(laplacian[i].w, paper_weight);
        EXPECT_DOUBLE_EQ(sddm[i].w, (W / sddm_degree) * paper_weight);

        const node_index ru = root(laplacian[i].u);
        const node_index rv = root(laplacian[i].v);
        EXPECT_NE(ru, rv);
        parent[ru] = rv;
    }
    const node_index component = root(original.front().vertex);
    for (const auto& neighbor : original)
        EXPECT_EQ(root(neighbor.vertex), component);
}

TEST(Bkz26PruferSampler, ExactCliqueOverrideHasExactOutputOrder) {
    std::vector<weighted_neighbor> input{
        {7, 1.0}, {2, 2.0}, {11, 4.0}};
    std::vector<deferred_edge> actual;
    volume_tree_elimination{.exact_clique_max_degree = 3}.sample_clique(
        input, 10.0, 42, edge_emitter(actual));
    const std::vector<deferred_edge> expected{
        {2, 7, 0.2}, {11, 7, 0.4}, {11, 2, 0.8}};
    expect_same_edges(actual, expected);
}

TEST(Bkz26PruferSampler, NonPositiveWeightFallsBackExactlyToGks) {
    const std::vector<weighted_neighbor> input{
        {7, 1.0}, {2, 0.0}, {11, 4.0}, {5, 2.0}};
    auto bkz26_input = input;
    auto gks_input = input;
    std::vector<deferred_edge> bkz26;
    std::vector<deferred_edge> gks;
    volume_tree_elimination{}.sample_clique(
        bkz26_input, 7.0, 42, edge_emitter(bkz26));
    tree_elimination{}.sample_clique(
        gks_input, 7.0, 42, edge_emitter(gks));
    expect_same_edges(bkz26, gks);
}

TEST(Bkz26PruferSampler, ExhaustiveLawIsUnbiasedForSchurClique) {
    constexpr std::array<double, 4> weights{1.0, 2.0, 4.0, 5.0};
    constexpr double W = 12.0;
    constexpr double degree_with_excess = 15.0;
    std::array<std::array<double, 4>, 4> expectation{};

    for (size_t first = 0; first < weights.size(); ++first) {
        for (size_t second = 0; second < weights.size(); ++second) {
            const std::array<size_t, 2> code{first, second};
            const double probability =
                (weights[first] / W) * (weights[second] / W);
            for (auto [u, v] : reference_heap_decode(code, weights.size())) {
                if (u > v) std::swap(u, v);
                expectation[u][v] += probability * (W / degree_with_excess) *
                    weights[u] * weights[v] / (weights[u] + weights[v]);
            }
        }
    }

    for (size_t u = 0; u < weights.size(); ++u)
        for (size_t v = u + 1; v < weights.size(); ++v)
            EXPECT_NEAR(expectation[u][v],
                        weights[u] * weights[v] / degree_with_excess,
                        2e-15);
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
