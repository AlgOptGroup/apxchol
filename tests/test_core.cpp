#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <map>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "apxchol/graph/conversions.h"
#include "apxchol/graph/graph.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// ── Typed test fixture ───────────────────────────────

using AllStorages = ::testing::Types<
    apxchol::vec_incidence,
    apxchol::forward_star_incidence,
    apxchol::bstr_incidence,
    apxchol::vec_pool_incidence,
    apxchol::directed_vec_pool_incidence
>;

template<typename Incidence>
class GraphTest : public ::testing::Test {
protected:
    using Graph = apxchol::graph<Incidence>;

    static Graph make_path(int n) {
        Graph G(n);
        for (int i = 0; i + 1 < n; ++i)
            G.add_edge(i, i + 1, 1.0);
        return G;
    }

    static Graph make_grid(int rows, int cols) {
        Graph G(rows * cols);
        auto id = [cols](int r, int c) { return r * cols + c; };
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                if (r + 1 < rows) G.add_edge(id(r, c), id(r + 1, c), 1.0);
                if (c + 1 < cols) G.add_edge(id(r, c), id(r, c + 1), 1.0);
            }
        return G;
    }
};

TYPED_TEST_SUITE(GraphTest, AllStorages);

// ── Graph tests ──────────────────────────────────────

TEST(ActiveState, ParallelClearsSharingWordsDoNotLoseUpdates) {
    constexpr apxchol::node_index n = 4096;
    apxchol::graph<apxchol::directed_vec_pool_incidence> G(n);

#pragma omp parallel for schedule(static, 1)
    for (apxchol::node_index v = 0; v < n; ++v)
        G.set_inactive_unchecked(v);
    G.bulk_decrement_active(n);

    EXPECT_EQ(G.num_active(), 0u);
    for (apxchol::node_index v = 0; v < n; ++v)
        EXPECT_FALSE(G.is_active(v));
}

TEST(ActiveState, RepeatedDeactivateDoesNotDoubleCount) {
    apxchol::graph<apxchol::directed_vec_pool_incidence> G(65);
    G.deactivate(64);
    G.deactivate(64);
    EXPECT_EQ(G.num_active(), 64u);
    EXPECT_FALSE(G.is_active(64));
    EXPECT_TRUE(G.is_active(63));
}

namespace {

struct assignment_counted_slot {
    int value = 0;
    static inline int assignments = 0;

    assignment_counted_slot() = default;
    assignment_counted_slot(int x) : value(x) {}
    assignment_counted_slot(const assignment_counted_slot&) = default;
    assignment_counted_slot& operator=(const assignment_counted_slot& other) {
        value = other.value;
        ++assignments;
        return *this;
    }
};
} // namespace

TEST(VecPoolIncidence, FilterSkipsWritesBeforeTheFirstRemovedEntry) {
    using Incidence = apxchol::basic_vec_pool_incidence<assignment_counted_slot,
                                                        apxchol::graph_storage::vec_pool>;
    Incidence adj;
    adj.init(1);
    for (int value : {1, 2, 3, 4})
        adj.push(0, value);

    assignment_counted_slot::assignments = 0;
    adj.filter(0, [](const auto&) { return true; });
    EXPECT_EQ(assignment_counted_slot::assignments, 0);

    std::vector<int> kept;
    assignment_counted_slot::assignments = 0;
    adj.filter(
        0, [](const auto& slot) { return slot.value != 2; },
        [&](const auto& slot) { kept.push_back(slot.value); });
    EXPECT_EQ(assignment_counted_slot::assignments, 2);
    EXPECT_EQ(kept, (std::vector<int>{1, 3, 4}));
    ASSERT_EQ(adj[0].size(), 3u);
    EXPECT_EQ(adj[0][0].value, 1);
    EXPECT_EQ(adj[0][1].value, 3);
    EXPECT_EQ(adj[0][2].value, 4);
}

TEST(SegmentedPool, ClaimsNeverCrossSegmentsAndClearReusesTheMap) {
    apxchol::segmented_pool<std::uint32_t, 8, 64> pool;
    pool.prepare();
    const size_t first = pool.claim(6);
    const size_t second = pool.claim(4);

    EXPECT_EQ(first, 0u);
    EXPECT_EQ(second, 8u);
    EXPECT_EQ(pool.size(), 12u);
    pool[second] = 42;
    EXPECT_EQ(pool[second], 42u);

    pool.clear();
    EXPECT_EQ(pool.size(), 0u);
    EXPECT_EQ(pool.claim(3), 0u);
}

TEST(SegmentedPool, ParallelClaimsAreDisjoint) {
    apxchol::segmented_pool<std::uint32_t, 64, 4096> pool;
    pool.prepare();
    constexpr size_t claims = 128;
    constexpr size_t slots_per_claim = 7;
    std::vector<size_t> offsets(claims);

#pragma omp parallel for schedule(static) num_threads(8)
    for (size_t i = 0; i < claims; ++i)
        offsets[i] = pool.claim(slots_per_claim);

    std::sort(offsets.begin(), offsets.end());
    for (size_t i = 0; i < offsets.size(); ++i) {
        EXPECT_LE(offsets[i] % pool.max_claim_slots() + slots_per_claim, pool.max_claim_slots());
        if (i != 0)
            EXPECT_GE(offsets[i], offsets[i - 1] + slots_per_claim);
    }
}

TYPED_TEST(GraphTest, BasicConstruction) {
    using Graph = typename TestFixture::Graph;
    Graph G(5);
    G.add_edge(0, 1, 2.0);
    G.add_edge(1, 2, 3.0);
    EXPECT_EQ(G.n(), 5);
    EXPECT_EQ(G.m(), 2);
}

TYPED_TEST(GraphTest, DeactivateAndPrune) {
    using Graph = typename TestFixture::Graph;
    Graph G(4);
    G.add_edge(0, 1, 1.0);
    G.add_edge(0, 2, 2.0);
    G.add_edge(1, 3, 3.0);
    G.add_edge(2, 3, 4.0);
    EXPECT_EQ(G.num_active(), 4);

    G.deactivate(0);
    EXPECT_FALSE(G.is_active(0));
    EXPECT_EQ(G.num_active(), 3);

    // Vertex 1's neighbor list still contains vertex 0 (deactivated but not pruned).
    // After prune_and_degree, only active neighbors remain.
    G.prune_and_degree(1);
    int count = 0;
    for (auto _ : G.neighbors(1)) ++count;
    EXPECT_EQ(count, 1);  // only vertex 3
}

TYPED_TEST(GraphTest, ForNeighbors) {
    auto G = TestFixture::make_path(4);  // 0-1-2-3
    std::vector<std::pair<int, double>> nbrs;
    for (const auto& e : G.neighbors(1))
        nbrs.push_back({e.to, e.w});
    EXPECT_EQ(nbrs.size(), 2u);
}

TYPED_TEST(GraphTest, WeightedDegree) {
    using Graph = typename TestFixture::Graph;
    Graph G(3);
    G.add_edge(0, 1, 2.0);
    G.add_edge(0, 2, 5.0);
    auto wdeg = [&](apxchol::node_index v) {
        double d = 0;
        for (const auto& e : G.neighbors(v)) d += e.w;
        return d;
    };
    EXPECT_DOUBLE_EQ(wdeg(0), 7.0);
    EXPECT_DOUBLE_EQ(wdeg(1), 2.0);
}

TEST(VecPoolGraph, CoalescePreservesWeightsMultiplicityAndActiveState) {
    using Graph = apxchol::graph<apxchol::vec_pool_incidence>;
    Graph G(6);
    G.add_edge(0, 1, 1.0);
    G.add_edge(0, 1, 2.0);
    G.add_edge(0, 2, 4.0);
    G.add_edge(1, 2, 3.0);
    G.add_edge(1, 2, 5.0);
    G.add_edge(1, 2, 7.0);
    G.add_edge(2, 3, 1.0);
    G.add_edge(3, 4, 9.0);  // dropped: vertex 4 is inactive
    G.add_edge(0, 5, 8.0);  // dropped: vertex 5 is inactive
    G.excess(0) = 0.25;
    G.excess(2) = 0.75;
    G.deactivate(4);
    G.deactivate(5);

    const std::vector<apxchol::node_index> active = {0, 1, 2, 3};
    EXPECT_EQ(G.prune_and_degree(0), 3u);
    EXPECT_EQ(G.prune_and_degree(1), 5u);
    EXPECT_EQ(G.prune_and_degree(2), 5u);
    EXPECT_EQ(G.prune_and_degree(3), 1u);
    EXPECT_DOUBLE_EQ(
        apxchol::detail::residual_coalescer<apxchol::vec_pool_incidence>::
            estimate(G, active, active.size()),
        14.0 / 8.0);

    const auto stats =
        apxchol::detail::residual_coalescer<apxchol::vec_pool_incidence>::
            rebuild(G, active);
    EXPECT_EQ(stats.multi_edges, 7u);
    EXPECT_EQ(stats.distinct_edges, 4u);
    EXPECT_EQ(G.m(), 4u);
    EXPECT_EQ(G.num_active(), 4u);
    EXPECT_FALSE(G.is_active(4));
    EXPECT_FALSE(G.is_active(5));
    EXPECT_DOUBLE_EQ(G.excess(0), 0.25);
    EXPECT_DOUBLE_EQ(G.excess(2), 0.75);

    // Physical adjacency is simple, but the partitioner's degree remains the
    // old multigraph degree through the multiplicity sidecar.
    EXPECT_EQ(G.adj(0).size(), 2u);
    EXPECT_EQ(G.adj(1).size(), 2u);
    EXPECT_EQ(G.prune_and_degree(0), 3u);
    EXPECT_EQ(G.prune_and_degree(1), 5u);
    EXPECT_EQ(G.prune_and_degree(2), 5u);
    EXPECT_EQ(G.prune_and_degree(3), 1u);

    auto find_edge = [&](apxchol::node_index u, apxchol::node_index v) {
        for (const auto idx : G.adj(u)) {
            if (G.edge_target(idx, u) == v)
                return std::pair{G.edge_weight(idx),
                                 G.edge_multiplicity(idx)};
        }
        return std::pair{0.0, apxchol::node_index{0}};
    };
    EXPECT_EQ(find_edge(0, 1), (std::pair{3.0, apxchol::node_index{2}}));
    EXPECT_EQ(find_edge(1, 2), (std::pair{15.0, apxchol::node_index{3}}));
    EXPECT_DOUBLE_EQ(
        apxchol::detail::residual_coalescer<apxchol::vec_pool_incidence>::
            estimate(G, active, active.size()),
        14.0 / 8.0);

    // Future sampled edges append with multiplicity one; the sidecar remains
    // aligned with the edge pool after reserve/write and adjacency insertion.
    G.adj_reserve_for(0, G.adj_count(0) + 1);
    G.adj_reserve_for(3, G.adj_count(3) + 1);
    const auto slot = G.reserve_edge_pool(1);
    G.write_edge_at(slot, 0, 3, 6.0);
    G.adj_atomic_push_reserved(0, slot);
    G.adj_atomic_push_reserved(3, slot);
    EXPECT_EQ(G.edge_multiplicity(slot), 1u);
    EXPECT_EQ(G.prune_and_degree(0), 4u);
    EXPECT_EQ(G.prune_and_degree(3), 2u);
}

namespace {

using SparsifyStorages = ::testing::Types<
    apxchol::vec_pool_incidence,
    apxchol::directed_vec_pool_incidence>;

template<typename Incidence>
class ResidualSparsifyTest : public ::testing::Test {};

TYPED_TEST_SUITE(ResidualSparsifyTest, SparsifyStorages);

template<typename Incidence>
apxchol::graph<Incidence> make_sparsify_graph() {
    apxchol::graph<Incidence> G(8);
    // Two connected components, both with many off-tree alternatives. Distinct
    // weights make the coarse maximum-weight forest deterministic.
    for (const auto [u, v, w] : std::vector<std::tuple<int, int, double>>{
             {0, 1, 20.0}, {1, 2, 18.0}, {2, 3, 16.0},
             {0, 2, 4.0}, {0, 3, 3.0}, {1, 3, 2.0},
             {4, 5, 14.0}, {5, 6, 12.0}, {6, 7, 10.0},
             {4, 6, 4.0}, {4, 7, 3.0}, {5, 7, 2.0}})
        G.add_edge(u, v, w);
    return G;
}

template<typename Incidence>
auto canonical_edges(const apxchol::graph<Incidence>& G) {
    std::vector<std::tuple<apxchol::node_index,
                           apxchol::node_index, double>> result;
    for (apxchol::node_index u = 0; u < G.n(); ++u)
        for (const auto& edge : G.neighbors(u))
            if (edge.to > u)
                result.emplace_back(u, edge.to, edge.w);
    std::ranges::sort(result);
    return result;
}

template<typename Incidence>
bool reachable_within(const apxchol::graph<Incidence>& G,
                      apxchol::node_index first,
                      apxchol::node_index last) {
    std::vector<unsigned char> seen(G.n(), 0);
    std::vector<apxchol::node_index> stack{first};
    seen[first] = 1;
    while (!stack.empty()) {
        const auto u = stack.back();
        stack.pop_back();
        for (const auto& edge : G.neighbors(u))
            if (edge.to >= first && edge.to <= last && !seen[edge.to]) {
                seen[edge.to] = 1;
                stack.push_back(edge.to);
            }
    }
    for (auto v = first; v <= last; ++v)
        if (!seen[v]) return false;
    return true;
}

template<typename Incidence>
auto adjacency_records(const apxchol::graph<Incidence>& G) {
    std::vector<std::tuple<apxchol::node_index,
                           apxchol::node_index, double>> result;
    for (apxchol::node_index u = 0; u < G.n(); ++u)
        for (const auto& edge : G.neighbors(u))
            result.emplace_back(u, edge.to, edge.w);
    return result;
}

}  // namespace

TEST(ResidualSparsifyTest,
     ParallelForestAndDirectedRebuildAreByteEquivalentToSerial) {
#ifdef _OPENMP
    const int previous_threads = omp_get_max_threads();
    omp_set_num_threads(4);
#endif
    using Incidence = apxchol::directed_vec_pool_incidence;
    constexpr apxchol::node_index n = 16;
    std::vector<apxchol::node_index> active(n);
    std::iota(active.begin(), active.end(), apxchol::node_index{0});
    auto make_dense = [=] {
        apxchol::graph<Incidence> G(n);
        for (apxchol::node_index u = 0; u < n; ++u)
            for (apxchol::node_index v = u + 1; v < n; ++v)
                G.add_edge(u, v, 1000.0 - 17.0 * u - v);
        return G;
    };
    auto run = [&](auto& G) {
        return apxchol::detail::residual_coalescer<Incidence>::sparsify(
            G, active, 0.35, 0x123456789abcdef0ULL);
    };

#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    auto serial = make_dense();
    const auto serial_stats = run(serial);
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    auto parallel = make_dense();
    const auto parallel_stats = run(parallel);
#ifdef _OPENMP
    omp_set_num_threads(previous_threads);
#endif
#ifdef _OPENMP
    ASSERT_GT(parallel_stats.forest_shards, 1u);
    ASSERT_LT(parallel_stats.forest_candidates,
              parallel_stats.distinct_before);
    ASSERT_GT(parallel_stats.rebuild_threads, 1u);
#else
    EXPECT_EQ(parallel_stats.forest_shards, 1u);
    EXPECT_EQ(parallel_stats.rebuild_threads, 1u);
#endif
    EXPECT_EQ(serial_stats.physical_before, parallel_stats.physical_before);
    EXPECT_EQ(serial_stats.distinct_before, parallel_stats.distinct_before);
    EXPECT_EQ(serial_stats.kept_edges, parallel_stats.kept_edges);
    EXPECT_EQ(serial_stats.backbone_edges, parallel_stats.backbone_edges);
    EXPECT_EQ(serial.m(), parallel.m());
    EXPECT_EQ(adjacency_records(serial), adjacency_records(parallel));
}

TYPED_TEST(ResidualSparsifyTest,
           BackbonePreservesEveryComponentAndTheRealizationIsDeterministic) {
    const std::vector<apxchol::node_index> active = {0, 1, 2, 3, 4, 5, 6, 7};
    auto A = make_sparsify_graph<TypeParam>();
    auto B = make_sparsify_graph<TypeParam>();
    const auto a = apxchol::detail::residual_coalescer<TypeParam>::sparsify(
        A, active, 1e-6, 0x123456789abcdef0ULL);
    const auto b = apxchol::detail::residual_coalescer<TypeParam>::sparsify(
        B, active, 1e-6, 0x123456789abcdef0ULL);

    EXPECT_EQ(a.backbone_edges, 6u);
    EXPECT_EQ(b.backbone_edges, 6u);
    EXPECT_GE(a.kept_edges, a.backbone_edges);
    EXPECT_TRUE(reachable_within(A, 0, 3));
    EXPECT_TRUE(reachable_within(A, 4, 7));
    EXPECT_EQ(canonical_edges(A), canonical_edges(B));
    const auto edges = canonical_edges(A);
    for (const auto expected : std::vector<std::pair<
             apxchol::node_index, apxchol::node_index>>{
             {0, 1}, {1, 2}, {2, 3}, {4, 5}, {5, 6}, {6, 7}})
        EXPECT_TRUE(std::ranges::any_of(edges, [&](const auto& edge) {
            return std::get<0>(edge) == expected.first &&
                   std::get<1>(edge) == expected.second;
        })) << "coarse maximum-weight forest omitted heavy edge "
            << expected.first << "-" << expected.second;
}

TYPED_TEST(ResidualSparsifyTest, OffTreeWeightsAreUnbiased) {
    using Incidence = TypeParam;
    const std::vector<apxchol::node_index> active = {0, 1, 2};
    constexpr int trials = 4096;
    constexpr double keep_probability = 0.30;
    double weight_sum = 0.0;
    double multiplicity_sum = 0.0;
    int retained = 0;
    for (int seed = 0; seed < trials; ++seed) {
        apxchol::graph<Incidence> G(3);
        G.add_edge(0, 1, 10.0);
        G.add_edge(1, 2, 9.0);
        // Coalesces to one off-tree edge of weight 1 and multiplicity 2.
        G.add_edge(0, 2, 0.25);
        G.add_edge(0, 2, 0.75);
        const auto stats =
            apxchol::detail::residual_coalescer<Incidence>::sparsify(
                G, active, keep_probability, static_cast<std::uint64_t>(seed));
        EXPECT_NEAR(stats.expected_kept_edges,
                    2.0 + keep_probability, 1e-12);
        for (const auto idx : G.adj(0)) {
            if (G.edge_target(idx, 0) != 2) continue;
            ++retained;
            const double weight = G.edge_weight(idx);
            weight_sum += weight;
            EXPECT_NEAR(weight, 1.0 / keep_probability, 1e-6);
            if constexpr (std::same_as<Incidence,
                                       apxchol::vec_pool_incidence>) {
                multiplicity_sum += G.edge_multiplicity(idx);
                EXPECT_TRUE(G.edge_multiplicity(idx) == 6u ||
                            G.edge_multiplicity(idx) == 7u);
            }
        }
        EXPECT_TRUE(reachable_within(G, 0, 2));
    }

    EXPECT_NEAR(static_cast<double>(retained) / trials,
                keep_probability, 0.02);
    EXPECT_NEAR(weight_sum / trials, 1.0, 0.07);
    if constexpr (std::same_as<Incidence, apxchol::vec_pool_incidence>)
        EXPECT_NEAR(multiplicity_sum / trials, 2.0, 0.14);
}

// ── Laplacian tests (storage-independent) ────────────

static apxchol::graph<> make_path_vec(int n) {
    apxchol::graph<> G(n);
    for (int i = 0; i + 1 < n; ++i)
        G.add_edge(i, i + 1, 1.0);
    return G;
}

static apxchol::graph<> make_grid_vec(int rows, int cols) {
    apxchol::graph<> G(rows * cols);
    auto id = [cols](int r, int c) { return r * cols + c; };
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            if (r + 1 < rows) G.add_edge(id(r, c), id(r + 1, c), 1.0);
            if (c + 1 < cols) G.add_edge(id(r, c), id(r, c + 1), 1.0);
        }
    return G;
}

TEST(Laplacian, RowSumsZero) {
    auto G = make_grid_vec(10, 10);
    auto L = apxchol::laplacian(G);
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(L.rows());
    Eigen::VectorXd res = L * ones;
    EXPECT_LT(res.norm(), 1e-12);
}

TEST(Laplacian, Symmetric) {
    auto G = make_grid_vec(8, 8);
    auto L = apxchol::laplacian(G);
    Eigen::SparseMatrix<double> diff = L - Eigen::SparseMatrix<double>(L.transpose());
    EXPECT_LT(diff.norm(), 1e-12);
}

TEST(Laplacian, PositiveDiagonal) {
    auto G = make_grid_vec(5, 5);
    auto L = apxchol::laplacian(G);
    for (int k = 0; k < L.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
            if (it.row() == it.col()) {
                EXPECT_GE(it.value(), 0.0);
            }
        }
}

TEST(Laplacian, PathGraph) {
    auto G = make_path_vec(3);  // 0-1-2, weights 1.0
    auto L = apxchol::laplacian(G);
    // L should be [[1,-1,0],[-1,2,-1],[0,-1,1]]
    EXPECT_EQ(L.rows(), 3);
    EXPECT_NEAR(L.coeff(0, 0), 1.0, 1e-15);
    EXPECT_NEAR(L.coeff(1, 1), 2.0, 1e-15);
    EXPECT_NEAR(L.coeff(0, 1), -1.0, 1e-15);
    EXPECT_NEAR(L.coeff(0, 2), 0.0, 1e-15);
}
