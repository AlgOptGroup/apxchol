#include <gtest/gtest.h>
#include "apxchol/solver/elimination/gpu_round_shadow.h"
#include "apxchol/solver/solve.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

#if defined(APXCHOL_USE_CUDA)
namespace {
using apxchol::node_index;
using apxchol::pool_value_t;
using apxchol::detail::gpu_round_shadow_incidence;
using apxchol::detail::gpu_round_shadow_input;
using edge = gpu_round_shadow_incidence;
constexpr std::size_t block_size = 16384;
std::uint64_t bits(double x) { return std::bit_cast<std::uint64_t>(x); }
double pool(double x) { return static_cast<double>(static_cast<pool_value_t>(x)); }

gpu_round_shadow_input make_input(node_index n, const std::vector<edge>& edges) {
    gpu_round_shadow_input result;
    result.vertex_count = n;
    result.active.assign(n, 1);
    result.excess.resize(n);
    for (node_index v = 0; v < n; ++v) result.excess[v] = v % 3 * 0.125;
    for (const auto& e : edges) {
        result.incidences.push_back({e.owner, e.neighbor, pool(e.weight)});
        result.incidences.push_back({e.neighbor, e.owner, pool(e.weight)});
    }
    std::stable_sort(result.incidences.begin(), result.incidences.end(),
        [](const edge& a, const edge& b) { return a.owner < b.owner; });
    result.owner_offsets.assign(static_cast<std::size_t>(n) + 1, 0);
    for (const auto& e : result.incidences) ++result.owner_offsets[e.owner + 1];
    std::partial_sum(result.owner_offsets.begin(), result.owner_offsets.end(), result.owner_offsets.begin());
    return result;
}

std::uint64_t bucket(double w) {
    if constexpr (sizeof(pool_value_t) == 4)
        return (~std::uint64_t(std::bit_cast<std::uint32_t>(static_cast<float>(w))) & 0xffffffffULL) >> 20;
    else return (~std::bit_cast<std::uint64_t>(w)) >> 52;
}
double draw(node_index u, node_index v, std::uint64_t seed) {
    std::uint64_t x = seed ^ (std::uint64_t(u) * 0x9E3779B97F4A7C15ULL)
                           ^ (std::uint64_t(v) * 0xBF58476D1CE4E5B9ULL);
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return static_cast<double>(x >> 11) * 0x1p-53;
}
struct oracle_output {
    std::vector<edge> canonical, paired;
    std::vector<std::uint8_t> backbone, kept;
    std::vector<double> probabilities;
    double scale = 0, expected = 0, min_probability = 1, total_weight = 0, backbone_weight = 0;
    unsigned passes = 0;
    std::size_t blocks = 0, forest_count = 0;
};

// Independent scalar law, not a call to graph::sparsify_active or the production
// normalization helper. Canonical ordinal is the forest's full tie breaker.
oracle_output reference(const gpu_round_shadow_input& input, std::uint64_t seed,
                        double keep = 0.25, double fixed_scale = -1) {
    oracle_output out;
    std::vector<edge> physical;
    for (const auto& e : input.incidences)
        if (e.owner < e.neighbor) physical.push_back(e);
    std::sort(physical.begin(), physical.end(), [](const edge& a, const edge& b) {
        return std::tie(a.owner, a.neighbor, a.weight) < std::tie(b.owner, b.neighbor, b.weight);
    });
    for (std::size_t i = 0; i < physical.size();) {
        const auto u = physical[i].owner, v = physical[i].neighbor;
        double sum = 0.0; // +0 matters for an all-negative-zero duplicate group.
        do { sum += physical[i++].weight; }
        while (i < physical.size() && physical[i].owner == u && physical[i].neighbor == v);
        out.canonical.push_back({u, v, pool(sum)});
    }
    const auto n = out.canonical.size();
    out.backbone.assign(n, 0); out.kept.assign(n, 0); out.probabilities.resize(n);
    std::vector<std::size_t> order(n); std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return std::pair(bucket(out.canonical[a].weight), a) < std::pair(bucket(out.canonical[b].weight), b);
    });
    std::vector<node_index> component(input.vertex_count);
    std::iota(component.begin(), component.end(), node_index{0});
    // Tiny/dense fixtures permit a simple component relabel instead of the
    // production union-find/Boruvka implementation.
    for (auto i : order) {
        const auto e = out.canonical[i];
        const auto a = component[e.owner], b = component[e.neighbor];
        if (a == b) continue;
        out.backbone[i] = 1; ++out.forest_count; out.backbone_weight += e.weight;
        for (auto& c : component) if (c == b) c = a;
    }
    std::vector<double> importance(n);
    for (std::size_t i = 0; i < n; ++i) importance[i] = std::sqrt(out.canonical[i].weight);
    out.blocks = (n + block_size - 1) / block_size;
    auto folded = [&](bool initial) {
        double total = 0;
        for (std::size_t begin = 0; begin < n; begin += block_size) {
            double partial = 0;
            for (std::size_t i = begin; i < std::min(n, begin + block_size); ++i)
                if (!out.backbone[i]) partial += initial ? importance[i] : std::min(1.0, out.scale * importance[i]);
            total += partial;
        }
        return total;
    };
    const double target = std::clamp(keep, 1e-6, 1.0) * (n - out.forest_count);
    const double mass = folded(true);
    out.scale = target > 0 && mass > 0 ? target / mass : 0;
    for (unsigned pass = 0; pass < 6 && out.scale > 0; ++pass) {
        const double expected = folded(false); ++out.passes;
        if (expected <= 0) break;
        out.scale *= target / expected;
    }
    if (fixed_scale >= 0) out.scale = fixed_scale;
    for (std::size_t i = 0; i < n; ++i) {
        const auto e = out.canonical[i];
        const double p = out.backbone[i] ? 1.0 : std::min(1.0, out.scale * importance[i]);
        out.probabilities[i] = p; out.expected += p; out.total_weight += e.weight;
        if (!out.backbone[i]) out.min_probability = std::min(out.min_probability, p);
        out.kept[i] = draw(e.owner, e.neighbor, seed) < p;
        if (!out.kept[i]) continue;
        // Match the specified two-operation HT rule, not a substituted w/p.
        const double inverse = 1.0 / p;
        const double weight = pool(e.weight * inverse);
        out.paired.push_back({e.owner, e.neighbor, weight});
        out.paired.push_back({e.neighbor, e.owner, weight});
    }
    std::sort(out.paired.begin(), out.paired.end(), [](const edge& a, const edge& b) {
        return std::tie(a.owner, a.neighbor) < std::tie(b.owner, b.neighbor);
    });
    return out;
}
void same_edges(const std::vector<edge>& a, const std::vector<edge>& b) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        SCOPED_TRACE(i);
        EXPECT_EQ(a[i].owner, b[i].owner); EXPECT_EQ(a[i].neighbor, b[i].neighbor);
        EXPECT_EQ(bits(a[i].weight), bits(b[i].weight));
    }
}
template<class Output>
void check(const gpu_round_shadow_input& input, const Output& actual,
           const oracle_output& expected, bool normalization = true) {
    same_edges(actual.canonical, expected.canonical);
    EXPECT_EQ(actual.backbone, expected.backbone); EXPECT_EQ(actual.kept, expected.kept);
    ASSERT_EQ(actual.probabilities.size(), expected.probabilities.size());
    for (std::size_t i = 0; i < actual.probabilities.size(); ++i)
        EXPECT_EQ(bits(actual.probabilities[i]), bits(expected.probabilities[i])) << i;
    same_edges(actual.residual.incidences, expected.paired);
    EXPECT_EQ(actual.residual.vertex_count, input.vertex_count);
    EXPECT_EQ(actual.residual.active, input.active); EXPECT_EQ(actual.residual.excess, input.excess);
    std::vector<std::size_t> offsets(static_cast<std::size_t>(input.vertex_count) + 1);
    for (const auto& e : expected.paired) ++offsets[e.owner + 1];
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
    EXPECT_EQ(actual.residual.owner_offsets, offsets);
    EXPECT_EQ(actual.stats.physical_before, input.incidences.size() / 2);
    EXPECT_EQ(actual.stats.distinct_before, expected.canonical.size());
    EXPECT_EQ(actual.stats.backbone_edges, expected.forest_count);
    EXPECT_EQ(actual.stats.kept_edges, expected.paired.size() / 2);
    EXPECT_EQ(bits(actual.stats.importance_scale), bits(expected.scale));
    // These three diagnostics permit a parallel positive-sum reduction.
    // Policy probabilities above and normalization scale remain bit-exact.
    const double sum_allowance = 64 * std::numeric_limits<double>::epsilon()
        * std::max<std::size_t>(1, expected.canonical.size());
    EXPECT_NEAR(actual.stats.expected_kept_edges, expected.expected,
                sum_allowance * std::max(1.0, std::abs(expected.expected)));
    EXPECT_NEAR(actual.stats.total_weight, expected.total_weight,
                sum_allowance * std::max(1.0, std::abs(expected.total_weight)));
    EXPECT_NEAR(actual.stats.backbone_weight, expected.backbone_weight,
                sum_allowance * std::max(1.0, std::abs(expected.backbone_weight)));
    EXPECT_DOUBLE_EQ(actual.stats.min_offtree_probability, expected.min_probability);
    EXPECT_DOUBLE_EQ(actual.stats.max_inverse_probability, 1.0 / expected.min_probability);
    if (normalization) {
        EXPECT_EQ(actual.stats.normalization_blocks, expected.blocks);
        EXPECT_EQ(actual.stats.normalization_passes, expected.passes);
    }
}
struct scoped_env {
    std::string name, old; bool existed;
    scoped_env(const char* key, const char* value) : name(key), existed(std::getenv(key) != nullptr) {
        if (existed) old = std::getenv(key);
        setenv(key, value, 1);
    }
    ~scoped_env() { if (existed) setenv(name.c_str(), old.c_str(), 1); else unsetenv(name.c_str()); }
};
#define REQUIRE_OWNED_SPARSIFY_DEVICE() do { \
    if (!apxchol::detail::gpu_round_shadow_runtime_available()) \
        GTEST_SKIP() << "CUDA owned sparsifier device unavailable"; \
} while (false)
}

#if defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
TEST(GpuOwnedSparsify, CanonicalBucketsAndCoalescingMatchCpuLaw) {
    REQUIRE_OWNED_SPARSIFY_DEVICE();
    const double one_next = static_cast<double>(std::nextafter(pool_value_t{1}, pool_value_t{2}));
    auto ties = make_input(3, {{1,2,one_next},{0,2,one_next},{0,1,1}});
    auto mixed = make_input(8, {{2,3,4},{0,2,0.25},{0,1,-0.0},{0,1,-0.0},
        {0,2,1},{0,2,0.5},{1,2,1},{0,3,2},{1,3,1},{4,5,4},{5,6,1},{4,6,0.0}});
    mixed.active[7] = 0; // Empty inactive rows are valid owned input.
    unsigned case_id = 0;
    for (const auto& input : {ties, mixed}) {
        SCOPED_TRACE(case_id);
        const auto expected = reference(input, 42);
        const auto result = apxchol::detail::gpu_sparsify_owned_residual_for_test(input, 42);
        check(input, result, expected);
        if (case_id == 0) EXPECT_EQ(expected.backbone, (std::vector<std::uint8_t>{1,1,0}));
        if (case_id == 1) EXPECT_EQ(bits(result.canonical.front().weight), bits(0.0));
        RecordProperty("owned_sparsify_canonical_" + std::to_string(case_id++), "pass");
    }
}

TEST(GpuOwnedSparsify, FixedScaleHashAndHorvitzThompsonMatchCpuLaw) {
    REQUIRE_OWNED_SPARSIFY_DEVICE();
    const auto input = make_input(5, {{0,1,1},{0,2,4},{0,3,16},{0,4,64},
        {1,2,1},{1,3,4},{1,4,16},{2,3,1},{2,4,4},{3,4,0}});
    unsigned case_id = 0;
    for (std::uint64_t seed : {std::uint64_t{0}, std::uint64_t{42}, std::numeric_limits<std::uint64_t>::max()}) {
        const auto expected = reference(input, seed, .25, .125);
        const auto result = apxchol::detail::gpu_sparsify_owned_residual_for_test(input, seed, .25, .125);
        check(input, result, expected, false);
        for (std::size_t i = 0; i < result.canonical.size(); ++i) {
            const double p = result.probabilities[i], w = result.canonical[i].weight;
            if (p > 0) EXPECT_NEAR(p * (w * (1.0 / p)), w, 4 * std::numeric_limits<double>::epsilon() * w);
        }
        RecordProperty("owned_sparsify_fixed_scale_" + std::to_string(case_id++), "pass");
    }
}

TEST(GpuOwnedSparsify, NormalizationUsesFullCanonicalBlockBoundaries) {
    REQUIRE_OWNED_SPARSIFY_DEVICE();
    for (std::size_t count : {block_size - 1, block_size, block_size + 1, 2 * block_size + 3}) {
        const node_index vertices = count > block_size + 1 ? 257 : 183;
        std::vector<edge> edges;
        for (node_index u = 0; u < vertices && edges.size() < count; ++u)
            for (node_index v = u + 1; v < vertices && edges.size() < count; ++v) {
                const auto i = edges.size();
                edges.push_back({u, v, std::ldexp(1.0, 2 * static_cast<int>(i % 4))});
            }
        // One late canonical edge is mandatory in the strict maximum-weight
        // forest, so full-stream chunk boundaries cannot be replaced by an
        // off-tree-only partition.
        edges.back().weight = 4096;
        const auto input = make_input(vertices, edges);
        const auto expected = reference(input, 42);
        ASSERT_EQ(expected.canonical.size(), count);
        ASSERT_EQ(expected.backbone.back(), 1);
        const auto result = apxchol::detail::gpu_sparsify_owned_residual_for_test(input, 42);
        check(input, result, expected);
        EXPECT_EQ(expected.passes, 6u);
        RecordProperty("owned_sparsify_normalization_" + std::to_string(count), "pass");
    }
    // A heavy triangle leaves one heavy off-forest edge. The .9 budget clips
    // that probability during each of the finite normalization updates.
    const auto clipped = make_input(4, {{0,1,64},{0,2,64},{1,2,64},
                                       {0,3,1},{1,3,1},{2,3,1}});
    const auto expected = reference(clipped, 42, .9);
    const auto actual = apxchol::detail::gpu_sparsify_owned_residual_for_test(clipped, 42, .9);
    check(clipped, actual, expected);
    unsigned saturated = 0, fractional = 0;
    for (std::size_t i = 0; i < expected.canonical.size(); ++i) if (!expected.backbone[i]) {
        saturated += expected.probabilities[i] == 1.0;
        fractional += expected.probabilities[i] > 0 && expected.probabilities[i] < 1;
    }
    EXPECT_EQ(saturated, 1u); EXPECT_EQ(fractional, 2u); EXPECT_EQ(expected.passes, 6u);
    RecordProperty("owned_sparsify_normalization_clipped", "pass");
}

TEST(GpuOwnedSparsify, EmptyAllKeptAndInvalidInputsPreserveOwnedContract) {
    REQUIRE_OWNED_SPARSIFY_DEVICE();
    auto empty = make_input(5, {});empty.active[4] = 0;
    const auto first = apxchol::detail::gpu_sparsify_owned_residual_for_test(empty, 42);
    check(empty, first, reference(empty, 42));
    auto input = make_input(4, {{0,1,1},{0,1,2},{0,2,1},{0,3,4},{1,2,2},{2,3,1}});
    const auto original = input;
    const auto expected = reference(input, 42, 1, 1e12);
    auto result = apxchol::detail::gpu_sparsify_owned_residual_for_test(input, 42, 1, 1e12);
    check(input, result, expected, false);
    EXPECT_EQ(result.stats.kept_edges, result.stats.distinct_before);
    same_edges(input.incidences, original.incidences);
    const auto repeated = apxchol::detail::gpu_sparsify_owned_residual_for_test(input, 42, 1, 1e12);
    check(input, repeated, expected, false);
    // Returning host-owned vectors must survive destruction/reuse of the
    // temporary test session and subsequent mutation of the caller's input.
    input.incidences.clear();same_edges(result.residual.incidences, expected.paired);
    check(empty, first, reference(empty, 42));
    auto invalid = original; invalid.active[1] = 0;
    EXPECT_THROW(apxchol::detail::gpu_sparsify_owned_residual_for_test(invalid,42), std::exception);
    auto overflow = make_input(2, {{0,1,double(std::numeric_limits<pool_value_t>::max())},
                                  {0,1,double(std::numeric_limits<pool_value_t>::max())}});
    EXPECT_THROW(apxchol::detail::gpu_sparsify_owned_residual_for_test(overflow,42), std::exception);
    RecordProperty("owned_sparsify_empty_all_kept_invalid", "pass");
}
#endif // APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS

TEST(GpuOwnedSparsify, PreviewReplacementInvalidatesAndResetsHandbackCounts) {
    REQUIRE_OWNED_SPARSIFY_DEVICE();
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
    using graph_type = apxchol::graph<apxchol::directed_vec_pool_incidence>;
    graph_type graph(5);
    std::vector<edge> edges;
    std::vector<apxchol::detail::gpu_topology_edge> topology;
    for (node_index u = 1; u < 5; ++u)
        for (node_index v = u + 1; v < 5; ++v)
            for (unsigned duplicate = 0; duplicate < 2; ++duplicate) {
                graph.add_edge(u, v, .5); edges.push_back({u,v,.5}); topology.push_back({u,v});
            }
    // Session outlives its borrowing selector.
    apxchol::detail::gpu_round_shadow_session session(true);
    apxchol::detail::gpu_block_frontend selector(5, topology);
    std::vector<node_index> active{0,1,2,3,4};
    apxchol::partition_options options; options.degree_quantile = .01;
    selector.prepare(active, options);
    const auto first = selector.select_block_greedy().data;
    ASSERT_EQ(first, (std::vector<node_index>{0}));
    ASSERT_FALSE(session.has_owned_residual());
    const auto initial = session.run_owned_prefix_round(graph, first, selector.device_selection(), 42);
    ASSERT_EQ(initial.raw_fill_edges, 0u);
    active.erase(active.begin());session.advance_selector(selector);
    ASSERT_TRUE(session.has_owned_residual());
    selector.prepare(active, options);
    const auto preview_vertices = selector.select_block_greedy().data;
    ASSERT_FALSE(preview_vertices.empty());
    const auto preview = selector.device_selection();
    const auto binds_before = selector.transfers().owned_residual_binds;
    // All four live vertices are sampled: multiplicity must not double this.
    EXPECT_DOUBLE_EQ(session.probe_owned_distinct_degree(active, selector), 3.0);
    EXPECT_NO_THROW((void)selector.device_selection());
    const auto stats = session.sparsify_owned_residual(selector, 1);
    EXPECT_EQ(selector.transfers().owned_residual_binds, binds_before + 1);
    EXPECT_EQ(stats.physical_before, 12u);EXPECT_EQ(stats.distinct_before, 6u);
    EXPECT_EQ(stats.kept_edges, 5u);
    auto input = make_input(5, edges);input.active[0] = 0;
    std::fill(input.excess.begin(), input.excess.end(), 0.0);
    auto stale_input = input;
    stale_input.pivots = preview_vertices;
    for (auto vertex : preview_vertices)
        stale_input.seeds.push_back(apxchol::detail::gpu_round_shadow_pivot_seed(42, vertex));
    apxchol::detail::round_shadow_reference_detail::validate_input(stale_input);
    auto reject_preview = [&](const char* expected_message) {
        // A fresh disposable consumer isolates each intentional failure from
        // the live owner whose newly published selection must run below.
        apxchol::detail::gpu_round_shadow_device_state rejected;
        try {
            (void)rejected.compute_discover_shape(stale_input, preview, 42);
            FAIL() << "stale sparsification preview was accepted";
        } catch (const std::invalid_argument& error) {
            EXPECT_NE(std::string(error.what()).find(expected_message),
                      std::string::npos) << error.what();
        }
    };
    // Check replacement itself revoked the preview, before prepare could
    // independently invalidate it. Do not ask the live producer to issue an
    // unready capability: that getter may poison the live producer.
    reject_preview("device selection generation is not published");
    const auto law = reference(input, 1);
    ASSERT_EQ(law.paired.size(), 10u);
    graph_type thinned(5);
    for (const auto& e : law.paired) if (e.owner < e.neighbor) thinned.add_edge(e.owner,e.neighbor,e.weight);
    thinned.deactivate(0);
    selector.prepare(active, options);
    const auto selected = selector.select_block_greedy().data;
    ASSERT_FALSE(selected.empty());
    const auto current_selection = selector.device_selection();
    reject_preview("stale device selection generation");
    const auto expected_input = apxchol::detail::make_gpu_round_shadow_input(thinned, selected, 42);
    const auto expected = apxchol::detail::reference_gpu_round_shadow(expected_input);
    ASSERT_GT(expected.raw_fill_edges, 0u); // This fixture exercises the +fill term.
    const auto after = session.run_owned_prefix_round(graph, selected, current_selection, 42);
    EXPECT_EQ(after.raw_fill_edges, expected.raw_fill_edges);
    EXPECT_EQ(after.live_incidences, expected.live_incidences);
    EXPECT_EQ(after.round_state_upload_bytes, 0u);
    std::erase_if(active, [&](node_index v) { return std::find(selected.begin(),selected.end(),v)!=selected.end(); });
    session.advance_selector(selector);
    std::vector<apxchol::detail::factor_col> columns;
    session.materialize_owned_prefix(graph, active, columns);
    EXPECT_EQ(graph.m(), stats.kept_edges + after.raw_fill_edges);
    ASSERT_EQ(columns.size(), 1 + selected.size());
    EXPECT_EQ(columns.front().vertex, 0u);
    const auto actual = apxchol::detail::fingerprint_gpu_round_shadow_input(
        apxchol::detail::make_gpu_round_shadow_input(graph, std::span<const node_index>{}, 42));
    EXPECT_EQ(actual.residual, expected.residual);
    EXPECT_EQ(actual.ordered_residual, expected.ordered_residual);
    EXPECT_EQ(actual.active, expected.active);
    EXPECT_EQ(actual.live_degree, expected.live_degree);
    EXPECT_EQ(actual.excess, expected.canonical_excess);
    EXPECT_EQ(actual.live_incidences, expected.live_incidences);
    RecordProperty("owned_sparsify_preview_handback", "pass");
}
TEST(GpuOwnedSparsify, AutomaticRoundZeroControllerUsesOneAttemptForBothImports) {
    REQUIRE_OWNED_SPARSIFY_DEVICE();
    scoped_env shadow("APXCHOL_GPU_ROUND_SHADOW", "force");
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env frontend("APXCHOL_GPU_BLOCK_FRONTEND", "force");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
    scoped_env trace_enabled("APXCHOL_GPU_BLOCK_TRACE", "1");
    scoped_env sparsify("APXCHOL_RESIDUAL_SPARSIFY", "1");
    scoped_env full_precision_factor("APXCHOL_SPTRSV_FP16", "0");
    scoped_env drop("APXCHOL_FACTOR_DROP", "0");
    constexpr int n = 17;
    std::vector<Eigen::Triplet<double>> entries;
    for (int u = 0; u < n; ++u) for (int v = 0; v < n; ++v)
        entries.emplace_back(u,v,u == v ? double(n) : -1.0);
    Eigen::SparseMatrix<double> original(n,n);original.setFromTriplets(entries.begin(),entries.end());
    original.makeCompressed();
    Eigen::VectorXd exact(n);
    for (int i = 0; i < n; ++i) exact[i] = .125 * (i % 7 - 3);
    const Eigen::VectorXd rhs = original * exact;
    apxchol::solve_options options;options.tol = 1e-8;options.max_iter = 1000;
    options.factor_opts.parallel_residual_threshold = 1;
    options.factor_opts.omp_threshold = 0;
    options.factor_opts.partition.degree_quantile = .8;
    ASSERT_DOUBLE_EQ(options.factor_opts.min_is_fraction, .05);
    auto occurrences = [](const std::string& text, const std::string& needle) {
        std::size_t count = 0, at = 0;
        while ((at = text.find(needle,at)) != std::string::npos) { ++count;at += needle.size(); }
        return count;
    };
    for (bool compressed : {true,false}) {
        SCOPED_TRACE(compressed ? "direct CSC" : "generic host initial");
        auto matrix = original;
        if (!compressed) matrix.uncompress(); // Do this after the copy: Eigen may recompress copies.
        ASSERT_EQ(matrix.isCompressed(), compressed);
        ASSERT_EQ(apxchol::detail::gpu_owned_csc_supported(matrix), compressed);
        // operator_view borrows this M-matrix unchanged; uncompressed storage
        // reaches the existing generic make_graph path, not a test override.
        apxchol::solve_result solved;
        testing::internal::CaptureStderr();
        try { solved = apxchol::solve(matrix,rhs,options); }
        catch (...) { const auto failure = testing::internal::GetCapturedStderr();ADD_FAILURE() << failure;throw; }
        const auto trace = testing::internal::GetCapturedStderr();
        EXPECT_EQ(occurrences(trace,"[gpu-owned-sparsify-gate]"),1u) << trace;
        EXPECT_EQ(occurrences(trace,"[gpu-owned-sparsify]"),1u) << trace;
        EXPECT_NE(trace.find("[gpu-owned-sparsify-gate] round=0 active=17 handoff=1 avg_distinct_degree=16 attempted=1 worthwhile=1"),std::string::npos) << trace;
        const std::string marker = std::string("[gpu-owned-sparsify] implementation=")
            + (compressed ? "device" : "host_initial") + " round=0 ";
        const auto begin = trace.find(marker);ASSERT_NE(begin,std::string::npos) << trace;
        const auto line = trace.substr(begin,trace.find('\n',begin)-begin);
        auto number = [&](const std::string& key) {
            const auto at = line.find(key);
            if (at == std::string::npos) throw std::runtime_error("missing controller receipt " + key);
            return std::stoull(line.substr(at+key.size()));
        };
        EXPECT_EQ(number("physical_before="),136u);EXPECT_EQ(number("distinct_before="),136u);
        EXPECT_LT(number("kept="),136u);EXPECT_GE(number("kept="),16u);
        EXPECT_NE(trace.find("[gpu-owned-factorization] complete"),std::string::npos) << trace;
        if (!compressed) EXPECT_NE(trace.find("fallback=unsupported_stored_format before_device_mutation=1"),std::string::npos) << trace;
        ASSERT_EQ(solved.x.size(),n);
        EXPECT_TRUE(solved.x.allFinite());
        EXPECT_LE((original*solved.x-rhs).norm()/rhs.norm(),1e-8);
        RecordProperty(compressed ? "owned_sparsify_controller_csc" : "owned_sparsify_controller_generic",line);
    }
}
#endif
