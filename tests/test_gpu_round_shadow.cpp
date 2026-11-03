#include <gtest/gtest.h>
#include "scoped_omp_threads.h"
#include "apxchol/solver/solve.h"
#include "csc_test_util.h"
#include "gpu_preconditioner_fixture.h"

#include "apxchol/detail/checkpoint.h"
#include "apxchol/solver/detail/gpu_diagnostics.h"
#include "apxchol/solver/elimination/gpu_round_shadow.h"
#include "apxchol/solver/factorization.h"
#include "apxchol/solver/gpu_block_frontend.h"
#include "apxchol/solver/partition/block_greedy.h"
#include "apxchol/version.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <future>
#include <latch>
#include <limits>
#include <memory>
#include <memory_resource>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(APXCHOL_USE_CUDA)
#include <cuda_runtime_api.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

using csc_access = apxchol::detail::sparse_csc_access;

namespace {

using apxchol::node_index;
using apxchol::detail::gpu_round_shadow_excess_bound;
using apxchol::detail::gpu_round_shadow_input;
using apxchol::detail::gpu_round_shadow_report;


inline constexpr int kResidentProvenanceWorkers = 2;

struct undirected_edge {
    node_index u;
    node_index v;
    double weight;
};

[[maybe_unused]] void establish_r2a_openmp_affinity_if_requested() {
    const char* requested = std::getenv("APXCHOL_R2A_REQUIRE_AFFINITY");
    if (!requested || std::string_view(requested) != "1") return;
#if defined(_OPENMP)
    int observed_team = 0;
#pragma omp parallel
    {
#pragma omp single
        observed_team = omp_get_num_threads();
    }
    const char* expected_text = std::getenv("OMP_NUM_THREADS");
    if (!expected_text || observed_team != std::atoi(expected_text))
        throw std::runtime_error("R2a OpenMP affinity team size differs");
#else
    throw std::runtime_error("R2a affinity provenance requires OpenMP");
#endif
}

gpu_round_shadow_input make_input(
        node_index n, std::span<const undirected_edge> edges,
        std::span<const node_index> pivots,
        std::uint64_t run_seed = 42,
        std::span<const double> excess = {}) {
    gpu_round_shadow_input input;
    input.vertex_count = n;
    input.active.assign(n, 1);
    input.excess.assign(n, 0.0);
    if (!excess.empty()) {
        EXPECT_EQ(excess.size(), static_cast<std::size_t>(n));
        input.excess.assign(excess.begin(), excess.end());
    }
    input.pivots.assign(pivots.begin(), pivots.end());
    for (node_index pivot : pivots)
        input.seeds.push_back(
            apxchol::detail::gpu_round_shadow_pivot_seed(run_seed, pivot));

    std::vector<std::vector<std::pair<node_index, float>>> adjacency(n);
    for (const auto& edge : edges) {
        // A real directed-AoS graph narrows each edge once on insertion and
        // promotes it back to double when process_vertex reads the slab.
        const auto stored = static_cast<float>(edge.weight);
        adjacency[edge.u].push_back({edge.v, stored});
        adjacency[edge.v].push_back({edge.u, stored});
    }
    input.owner_offsets.resize(static_cast<std::size_t>(n) + 1);
    for (node_index owner = 0; owner < n; ++owner) {
        input.owner_offsets[owner] = input.incidences.size();
        for (const auto& [neighbor, weight] : adjacency[owner])
            input.incidences.push_back({owner, neighbor, weight});
    }
    input.owner_offsets[n] = input.incidences.size();
    return input;
}

[[maybe_unused]] gpu_round_shadow_input compact_live_snapshot(gpu_round_shadow_input input) {
    std::vector<apxchol::detail::gpu_round_shadow_incidence> live;
    live.reserve(input.incidences.size());
    std::vector<std::size_t> offsets(
        static_cast<std::size_t>(input.vertex_count) + 1);
    for (node_index owner = 0; owner < input.vertex_count; ++owner) {
        offsets[owner] = live.size();
        if (!input.active[owner]) continue;
        for (std::size_t i = input.owner_offsets[owner];
             i < input.owner_offsets[owner + 1]; ++i) {
            const auto edge = input.incidences[i];
            if (input.active[edge.neighbor]) live.push_back(edge);
        }
    }
    offsets[input.vertex_count] = live.size();
    input.incidences = std::move(live);
    input.owner_offsets = std::move(offsets);
    return input;
}

apxchol::detail::gpu_round_shadow_factor_log materialize_factor_log(
        std::span<const apxchol::detail::factor_col> columns,
        std::size_t entry_base = 0) {
    apxchol::detail::gpu_round_shadow_factor_log result;
    result.columns.reserve(columns.size());
    std::size_t next_entry = entry_base;
    for (const auto& column : columns) {
        if (column.entry_count >
            std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("test factor column exceeds uint32");
        result.columns.push_back({
            column.vertex, column.diag,
            static_cast<std::uint64_t>(next_entry),
            static_cast<std::uint32_t>(column.entry_count)});
        for (std::size_t i = 0; i < column.entry_count; ++i) {
            result.entries.push_back(
                {column.entries[i].neighbor, column.entries[i].value});
        }
        next_entry += column.entry_count;
    }
    return result;
}

#if defined(APXCHOL_USE_CUDA)
apxchol::detail::gpu_round_shadow_factor_log append_factor_logs(
        const apxchol::detail::gpu_round_shadow_factor_log& prefix,
        const apxchol::detail::gpu_round_shadow_factor_log& suffix) {
    auto result = prefix;
    result.columns.reserve(prefix.columns.size() + suffix.columns.size());
    result.entries.reserve(prefix.entries.size() + suffix.entries.size());
    for (auto column : suffix.columns) {
        column.entry_begin += prefix.entries.size();
        result.columns.push_back(column);
    }
    result.entries.insert(result.entries.end(), suffix.entries.begin(),
                          suffix.entries.end());
    return result;
}

void expect_factor_logs_equal(
        const apxchol::detail::gpu_round_shadow_factor_log& expected,
        const apxchol::detail::gpu_round_shadow_factor_log& actual) {
    ASSERT_EQ(actual.columns.size(), expected.columns.size());
    ASSERT_EQ(actual.entries.size(), expected.entries.size());
    for (std::size_t i = 0; i < expected.columns.size(); ++i) {
        SCOPED_TRACE("factor column " + std::to_string(i));
        EXPECT_EQ(actual.columns[i].vertex, expected.columns[i].vertex);
        EXPECT_EQ(std::bit_cast<std::uint32_t>(actual.columns[i].diag),
                  std::bit_cast<std::uint32_t>(expected.columns[i].diag));
        EXPECT_EQ(actual.columns[i].entry_begin,
                  expected.columns[i].entry_begin);
        EXPECT_EQ(actual.columns[i].entry_count,
                  expected.columns[i].entry_count);
    }
    for (std::size_t i = 0; i < expected.entries.size(); ++i) {
        SCOPED_TRACE("factor entry " + std::to_string(i));
        EXPECT_EQ(actual.entries[i].neighbor, expected.entries[i].neighbor);
        EXPECT_EQ(std::bit_cast<std::uint32_t>(actual.entries[i].value),
                  std::bit_cast<std::uint32_t>(expected.entries[i].value));
    }
}
#endif

struct checked_cpu_round {
    apxchol::detail::graph residual;
    apxchol::detail::gpu_round_shadow_cpu_comparison comparison;
    apxchol::detail::gpu_round_shadow_factor_log factor_log;
};

checked_cpu_round run_authoritative_cpu_round(
        std::span<const undirected_edge> edges,
        const gpu_round_shadow_input& input,
        const gpu_round_shadow_report& expected,
        std::span<const gpu_round_shadow_excess_bound> excess_bounds) {
    const std::size_t workers = std::max<std::size_t>(2, input.pivots.size());
    // Match the workspace's explicitly requested team before graph allocation.
    scoped_omp_threads declared_team(static_cast<int>(workers));
    apxchol::detail::graph graph(
        input.vertex_count);
    for (const auto& edge : edges)
        graph.add_edge(edge.u, edge.v, edge.weight);
    for (node_index vertex = 0; vertex < input.vertex_count; ++vertex)
        graph.excess(vertex) = input.excess[vertex];

    apxchol::detail::factorize_workspace workspace;
    workspace.threads.resize(workers);
    for (auto& thread : workspace.threads)
        thread.factor_entries =
            std::make_unique<std::pmr::monotonic_buffer_resource>();

    apxchol::detail::partition_result partition;
    partition.data = input.pivots;
    apxchol::factor_options options;
    options.seed = input.pivots.empty() ? 0 :
        input.seeds.front() ^
        ((std::uint64_t(input.pivots.front()) + 1) *
         0x9E3779B97F4A7C15ULL);
    for (std::size_t i = 0; i < input.pivots.size(); ++i) {
        const std::uint64_t recovered = input.seeds[i] ^
            ((std::uint64_t(input.pivots[i]) + 1) *
             0x9E3779B97F4A7C15ULL);
        if (recovered != options.seed)
            throw std::logic_error("test input seeds do not share one run seed");
    }
    // Multi-pivot cases enter the real directed-AoS OpenMP apply path. A
    // one-pivot round correctly stays on the production serial path.
    options.experimental.omp_threshold = 0;
    std::size_t work_hint = 0;
    for (node_index pivot : input.pivots) {
        for (std::size_t i = input.owner_offsets[pivot];
             i < input.owner_offsets[pivot + 1]; ++i) {
            if (input.active[input.incidences[i].neighbor]) ++work_hint;
        }
    }

    std::vector<apxchol::detail::factor_col> columns;
    apxchol::detail::eliminate_partition(
        apxchol::detail::tree_elimination{}, graph, partition, columns,
        workspace, options, nullptr, work_hint);
    const auto comparison =
        apxchol::detail::compare_gpu_round_shadow_with_cpu(
            expected, excess_bounds, graph, columns);
    auto factor_log = materialize_factor_log(columns);
    return {std::move(graph), comparison, std::move(factor_log)};
}

apxchol::detail::gpu_round_shadow_cpu_comparison run_serial_cpu_round(
        apxchol::detail::graph& graph,
        const gpu_round_shadow_input& input,
        const gpu_round_shadow_report& expected,
        std::span<const gpu_round_shadow_excess_bound> excess_bounds,
        apxchol::detail::gpu_round_shadow_factor_log* factor_log = nullptr) {
    apxchol::detail::factorize_workspace workspace;
    workspace.threads.resize(1);
    workspace.threads.front().factor_entries =
        std::make_unique<std::pmr::monotonic_buffer_resource>();
    apxchol::detail::partition_result partition;
    partition.data = input.pivots;
    apxchol::factor_options options;
    options.seed = input.pivots.empty() ? 0 :
        input.seeds.front() ^
        ((std::uint64_t(input.pivots.front()) + 1) *
         0x9E3779B97F4A7C15ULL);
    std::size_t work_hint = 0;
    for (node_index pivot : input.pivots) {
        for (std::size_t i = input.owner_offsets[pivot];
             i < input.owner_offsets[pivot + 1]; ++i)
            if (input.active[input.incidences[i].neighbor]) ++work_hint;
    }
    std::vector<apxchol::detail::factor_col> columns;
    apxchol::detail::eliminate_partition(
        apxchol::detail::tree_elimination{}, graph, partition, columns,
        workspace, options, nullptr, work_hint);
    if (factor_log) *factor_log = materialize_factor_log(columns);
    return apxchol::detail::compare_gpu_round_shadow_with_cpu(
        expected, excess_bounds, graph, columns);
}

// Independent host replay of the production owned path. Oversized rows use
// the documented fixed warp degree fold, while normal batches fold raw slots
// serially. The audited device path has its separate serial CPU oracle above.
// Preserve exact comparisons: CPU/GPU summation order is part of this oracle.
[[maybe_unused]] apxchol::detail::gpu_round_shadow_factor_log run_owned_host_oracle(
        apxchol::detail::graph& graph,
        const gpu_round_shadow_input& input) {
    std::vector<std::vector<apxchol::detail::factor_entry>> entries(input.pivots.size());
    std::vector<apxchol::detail::factor_col> columns;
    std::vector<apxchol::deferred_edge> fill;
    for (std::size_t k = 0; k < input.pivots.size(); ++k) {
        const auto v = input.pivots[k];
        const auto begin = input.owner_offsets[v], end = input.owner_offsets[v+1];
        std::vector<apxchol::weighted_neighbor> neighbors;
        double degree = 0.0;
        for (std::size_t i = begin; i < end; ++i) {
            const auto& e = input.incidences[i];
            if (!input.active[e.neighbor]) continue;
            degree += e.weight;
            auto found = std::find_if(neighbors.begin(),neighbors.end(),
                [&](const auto& n) { return n.vertex == e.neighbor; });
            if (found == neighbors.end()) neighbors.push_back({e.neighbor,e.weight});
            else found->weight += e.weight;
        }
        if (end-begin > 128) {
            std::array<double,32> lanes{};
            for (std::size_t lane = 0; lane < 32; ++lane)
                for (std::size_t i = begin+lane; i < end; i += 32)
                    if (input.active[input.incidences[i].neighbor])
                        lanes[lane] += input.incidences[i].weight;
            for (unsigned stride = 16; stride; stride /= 2)
                for (unsigned lane = 0; lane+stride < 32; ++lane)
                    lanes[lane] += lanes[lane+stride];
            degree = lanes[0];
        }
        const double excess = input.excess[v];
        degree += excess;
        if (degree <= 0.0) degree = 1.0;
        const double root = std::sqrt(degree);
        for (const auto& n : neighbors) {
            entries[k].push_back({n.vertex,static_cast<apxchol::factor_value_t>(n.weight/root)});
            if (excess > 0.0) graph.excess(n.vertex) += n.weight*excess/degree;
        }
        columns.push_back({v,static_cast<apxchol::factor_value_t>(root),entries[k].data(),
                           static_cast<node_index>(entries[k].size())});
        apxchol::detail::tree_elimination{}.sample_clique(
            neighbors,degree,input.seeds[k],apxchol::edge_emitter(fill));
    }
    for (auto e : fill) graph.add_edge(e.u,e.v,e.w);
    for (auto v : input.pivots) graph.deactivate(v);
    return materialize_factor_log(columns);
}

#if defined(APXCHOL_USE_CUDA) && defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
apxchol::detail::graph graph_from_input(
        const gpu_round_shadow_input& input) {
    apxchol::detail::graph graph(input.vertex_count);
    std::vector<node_index> vertices(input.vertex_count), counts(input.vertex_count);
    std::iota(vertices.begin(), vertices.end(), node_index{0});
    for (auto v : vertices) counts[v] = input.owner_offsets[v + 1] - input.owner_offsets[v];
    graph.adj_bulk_reserve_parallel(vertices.begin(), vertices.end(), counts);
    for (auto v : vertices) {
        for (std::size_t p = input.owner_offsets[v]; p < input.owner_offsets[v + 1]; ++p)
            graph.adj_write_reserved_directed_at(v, p - input.owner_offsets[v],
                input.incidences[p].neighbor, input.incidences[p].weight);
        graph.adj_commit_reserved_directed(v, counts[v]);
        graph.excess(v) = input.excess[v];
        if (!input.active[v]) graph.deactivate(v);
    }
    graph.record_edges_added(input.incidences.size() / 2);
    return graph;
}

void observe_owned_prefix(apxchol::detail::gpu_round_shadow_session& session,
        apxchol::detail::graph& graph,
        std::vector<node_index>& active, std::vector<apxchol::detail::factor_col>& columns) {
    const auto snapshot = session.download_owned_prefix_for_test();
    graph = graph_from_input(snapshot.residual);
    active.clear();
    for (node_index v = 0; v < graph.n(); ++v)
        if (graph.is_active(v)) active.push_back(v);
    columns.clear();
    for (const auto& column : snapshot.columns)
        columns.push_back({column.vertex, column.diag, nullptr, node_index(column.entry_count)});
}

std::uint64_t check_complete_factor_action(
        apxchol::detail::gpu_round_shadow_session& session,
        apxchol::detail::gpu_block_frontend& selector,
        apxchol::detail::graph& oracle,
        apxchol::detail::gpu_round_shadow_factor_log expected, std::uint64_t seed) {
    std::vector<node_index> active;
    for (node_index v = 0; v < oracle.n(); ++v) if (oracle.is_active(v)) active.push_back(v);
    while (!active.empty()) {
        session.advance_selector(selector);
        selector.prepare(active, {});
        const auto selected = selector.select_block_greedy().data;
        if (selected.empty()) throw std::logic_error("owned factor fixture stopped making progress");
        const auto input = apxchol::detail::make_gpu_round_shadow_input(oracle, selected, seed);
        session.run_owned_prefix_round(selector.device_selection(), seed);
        expected = append_factor_logs(expected, run_owned_host_oracle(oracle, input));
        std::erase_if(active, [&](node_index v) { return !oracle.is_active(v); });
    }
    expect_factor_logs_equal(expected, session.download_factor_log_for_test());
    const auto n = oracle.n();
    std::vector<std::vector<apxchol::detail::factor_entry>> entries(n);
    std::vector<apxchol::detail::factor_col> columns;
    for (std::size_t i = 0; i < expected.columns.size(); ++i) {
        const auto& c = expected.columns[i];
        for (std::size_t j = 0; j < c.entry_count; ++j) {
            const auto& e = expected.entries[c.entry_begin + j];
            entries[i].push_back({e.neighbor, e.value});
        }
        columns.push_back({c.vertex, c.diag, entries[i].data(), node_index(entries[i].size())});
    }
    apxchol::factorization reference, metadata;
    apxchol::detail::build_csc(reference, columns, n, nullptr);
    session.complete_owned_factorization(metadata.perm, metadata.L);
    EXPECT_EQ(metadata.perm, reference.perm);
    EXPECT_EQ(csc_access::outer(metadata.L), csc_access::outer(reference.L));
    const apxchol::factor_options factor_options{
        .factor_storage = apxchol::factor_storage_type::fp32, .drop_tolerance = 0};
    apxchol::detail::cuda_sptrsv expected_trsv, actual_trsv;
    expected_trsv.setup(reference.L, n, factor_options);
    auto factor = session.finalize_device_factor(metadata.perm, n, factor_options.factor_storage, 0);
    actual_trsv.setup_adopting_device_factor(std::move(*factor), factor_options);
    std::uint64_t digest = 14695981039346656037ULL;
    for (node_index basis = 0; basis < n; ++basis) {
        std::vector<double> wanted(n), actual(n);
        wanted[basis] = actual[basis] = 1.0;
        expected_trsv.solve_LLt(wanted.data(), wanted.data());
        actual_trsv.solve_LLt(actual.data(), actual.data());
        EXPECT_EQ(actual, wanted) << "basis=" << basis;
        for (double value : actual)
            digest = (digest ^ std::bit_cast<std::uint64_t>(value)) * 1099511628211ULL;
    }
    return digest;
}

gpu_round_shadow_report run_device(
        const gpu_round_shadow_input& input,
        std::vector<gpu_round_shadow_excess_bound>* supplied_bounds = nullptr) {
    std::vector<gpu_round_shadow_excess_bound> bounds;
    (void)apxchol::detail::reference_gpu_round_shadow(input, &bounds);
    if (supplied_bounds) *supplied_bounds = bounds;
    apxchol::detail::gpu_round_shadow_device_state state;
    auto selector = state.initialize_owned_raw_for_test(input);
    if (input.pivots.empty()) {
        const auto actual = state.download_owned_prefix_for_test().residual;
        EXPECT_EQ(apxchol::detail::fingerprint_gpu_round_shadow_input(actual),
                  apxchol::detail::fingerprint_gpu_round_shadow_input(input));
        return {};
    }
    selector->select_for_test(input.pivots);
    const auto seed = input.seeds.front() ^
        ((std::uint64_t(input.pivots.front()) + 1) * 0x9E3779B97F4A7C15ULL);
    const auto report = state.compute_owned_round_for_test(selector->device_selection(), seed);
    const auto factor = state.download_factor_log();
    const auto actual = state.download_owned_prefix_for_test().residual;
    auto oracle = graph_from_input(input);
    const auto expected_factor = run_owned_host_oracle(oracle, input);
    expect_factor_logs_equal(expected_factor, factor);
    const auto expected = compact_live_snapshot(apxchol::detail::make_gpu_round_shadow_input(
        oracle, std::span<const node_index>{}, seed));
    EXPECT_EQ(actual.owner_offsets, expected.owner_offsets);
    EXPECT_EQ(actual.active, expected.active);
    EXPECT_EQ(actual.incidences.size(), expected.incidences.size());
    if (actual.incidences.size() == expected.incidences.size())
        for (std::size_t p = 0; p < actual.incidences.size(); ++p) {
            EXPECT_EQ(actual.incidences[p].owner, expected.incidences[p].owner);
            EXPECT_EQ(actual.incidences[p].neighbor, expected.incidences[p].neighbor);
            EXPECT_EQ(std::bit_cast<std::uint64_t>(double(actual.incidences[p].weight)),
                      std::bit_cast<std::uint64_t>(double(expected.incidences[p].weight)));
        }
    EXPECT_EQ(actual.excess.size(), expected.excess.size());
    if (actual.excess.size() == expected.excess.size())
        for (node_index v = 0; v < input.vertex_count; ++v) {
            if (!actual.active[v]) continue;
            const auto& bound = bounds[v];
            EXPECT_TRUE(std::isfinite(actual.excess[v]));
            EXPECT_GE(actual.excess[v], 0.0);
            if (bound.additions <= 1) {
                EXPECT_EQ(actual.excess[v], expected.excess[v]);
            } else {
                const double terms = double(bound.additions + 1);
                const double tolerance = 32 * std::numeric_limits<double>::epsilon() *
                    terms * bound.absolute_term_sum +
                    32 * std::numeric_limits<double>::denorm_min() * terms;
                EXPECT_LE(std::abs(actual.excess[v] - expected.excess[v]), tolerance);
            }
        }
    return report;
}
#endif

struct parallel_cpu_round_result {
    apxchol::detail::gpu_round_shadow_cpu_comparison comparison;
    std::uint32_t worker_mask = 0;
    int team_width = 0;
};

parallel_cpu_round_result run_parallel_cpu_round(
        apxchol::detail::graph& graph,
        const gpu_round_shadow_input& input,
        const gpu_round_shadow_report& expected,
        std::span<const gpu_round_shadow_excess_bound> excess_bounds) {
    // Keep the explicit team configuration local to this parallel fixture.
    scoped_omp_threads declared_team(kResidentProvenanceWorkers);
    apxchol::detail::factorize_workspace workspace;
    constexpr std::size_t workers = kResidentProvenanceWorkers;
    workspace.threads.resize(workers);
    for (auto& thread : workspace.threads)
        thread.factor_entries =
            std::make_unique<std::pmr::monotonic_buffer_resource>();
    apxchol::detail::partition_result partition;
    partition.data = input.pivots;
    apxchol::factor_options options;
    options.seed = input.pivots.empty() ? 0 :
        input.seeds.front() ^
        ((std::uint64_t(input.pivots.front()) + 1) *
         0x9E3779B97F4A7C15ULL);
    // Stay on the fine-grained production schedule while adjacency work, not
    // an unconditional vertex threshold, selects the two-worker team.
    options.experimental.omp_threshold = input.pivots.size();
    std::size_t work_hint = 0;
    for (node_index pivot : input.pivots) {
        for (std::size_t i = input.owner_offsets[pivot];
             i < input.owner_offsets[pivot + 1]; ++i)
            if (input.active[input.incidences[i].neighbor]) ++work_hint;
    }
    if (apxchol::detail::elimination_round_team_size(
            input.pivots.size(), work_hint, options.experimental.omp_threshold,
            workspace.threads.size()) != workers)
        throw std::logic_error("test did not select the parallel CPU apply path");
    if (input.pivots.size() < workers ||
        apxchol::detail::elimination_compute_chunk(
            input.pivots.size(), options.experimental.omp_threshold) != 1)
        throw std::logic_error("parallel CPU apply test needs one-pivot chunks per worker");

#ifdef _OPENMP
    const int saved_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
#endif
    std::atomic<std::uint32_t> worker_mask{0};
    std::atomic<int> team_width{0};
    std::latch first_samples(workers);
    const apxchol::detail::tree_elimination tree;
    const auto recording_tree = apxchol::as_eliminator(
        [&](std::span<apxchol::weighted_neighbor> neighbors, double degree,
            std::uint64_t seed, apxchol::edge_emitter out) {
#ifdef _OPENMP
            const int worker = omp_get_thread_num();
            const int width = omp_get_num_threads();
#else
            const int worker = 0;
            const int width = 1;
#endif
            team_width.store(width, std::memory_order_relaxed);
            const auto bit = std::uint32_t{1} << worker;
            const auto seen = worker_mask.fetch_or(bit, std::memory_order_relaxed);
            // Meet on each worker's first pivot so dynamic scheduling cannot
            // give all the work to one thread. A reduced team must fail, not hang.
            if (!(seen & bit) && width == kResidentProvenanceWorkers)
                first_samples.arrive_and_wait();
            tree.sample_clique(neighbors, degree, seed, out);
        });
    std::vector<apxchol::detail::factor_col> columns;
    apxchol::detail::eliminate_partition(
        recording_tree, graph, partition, columns,
        workspace, options, nullptr, work_hint);
#ifdef _OPENMP
    omp_set_dynamic(saved_dynamic);
#endif
    return {
        apxchol::detail::compare_gpu_round_shadow_with_cpu(
            expected, excess_bounds, graph, columns),
        worker_mask.load(std::memory_order_relaxed),
        team_width.load(std::memory_order_relaxed)};
}

apxchol::detail::gpu_round_shadow_state_fingerprint fingerprint_graph(
        const apxchol::detail::graph& graph) {
    return apxchol::detail::fingerprint_gpu_round_shadow_input(
        apxchol::detail::make_gpu_round_shadow_input(
            graph, std::span<const node_index>{}, 0));
}

constexpr node_index kResidentProvenanceFirstParallelPivot = 8;
constexpr node_index kResidentProvenanceParallelPivotEnd = 24;
constexpr node_index kResidentProvenanceVertexCount = 536;

apxchol::detail::graph
make_resident_provenance_graph() {
    // run_parallel_cpu_round explicitly requests two workers. The pool sizes
    // its per-worker grow scratch at graph construction, so establish that
    // declared team before allocation even when the process starts with T=1.
    // The guard restores the surrounding test's OpenMP setting on return.
    scoped_omp_threads declared_team(kResidentProvenanceWorkers);
    apxchol::detail::graph graph(
        kResidentProvenanceVertexCount);
    for (const auto& edge : std::vector<undirected_edge>{
        {0, 5, 1.0}, {0, 6, 4.0}, {0, 7, 2.0},
        {1, 3, 3.0}, {1, 4, 5.0}, {1, 5, 7.0},
        {2, 3, 6.0}, {2, 4, 8.0}, {2, 5, 9.0},
        {3, 6, 10.0}, {4, 7, 11.0}, {5, 6, 12.0}})
        graph.add_edge(edge.u, edge.v, edge.weight);
    // Independent degree-512 pivots select the parallel compute/apply path.
    // Their common neighbors permit endpoint-slot claims to differ from the
    // GPU's stable pivot/emission order; an individual outcome may still match.
    for (node_index pivot = kResidentProvenanceFirstParallelPivot;
         pivot < kResidentProvenanceParallelPivotEnd; ++pivot) {
        for (node_index neighbor = kResidentProvenanceParallelPivotEnd;
             neighbor < kResidentProvenanceVertexCount; ++neighbor) {
            const double weight = 1.0 + static_cast<double>(
                (17 * pivot + 13 * neighbor) % 29);
            graph.add_edge(pivot, neighbor, weight);
        }
    }
    graph.excess(1) = 4.0;
    graph.excess(2) = 3.0;
    return graph;
}

#if defined(APXCHOL_USE_CUDA)
std::string gpu_round_trace_field(const std::string& line,
                                  std::string_view name) {
    const std::size_t begin = line.find(name);
    if (begin == std::string::npos)
        throw std::runtime_error("missing GPU round trace field " +
                                 std::string(name));
    const std::size_t value_begin = begin + name.size();
    const std::size_t value_end = line.find_first_of(" )", value_begin);
    return line.substr(value_begin, value_end - value_begin);
}

std::size_t gpu_round_trace_size(const std::string& line,
                                 std::string_view name) {
    return static_cast<std::size_t>(
        std::stoull(gpu_round_trace_field(line, name)));
}

#if defined(APXCHOL_USE_CUDA)
const apxchol::detail::cuda_route& gpu_session(const apxchol::solver& owner) {
    return std::get<apxchol::detail::cuda_route>(apxchol::detail::solver_access::route(owner));
}

void expect_setup_api_receipt(const std::string& trace,
                              const apxchol::detail::cuda_route& solver, bool verbose) {
    const auto begin = trace.find("[gpu-setup-receipt] ");
    if (verbose) {
        ASSERT_NE(begin, std::string::npos) << trace;
        EXPECT_EQ(trace.find("[gpu-setup-receipt] ", begin + 1), std::string::npos);
        const auto line = trace.substr(begin, trace.find('\n', begin) - begin);
        const double wall = std::stod(gpu_round_trace_field(line, "factorize_install_wall_s="));
        EXPECT_TRUE(std::isfinite(wall));
        EXPECT_GT(wall, 0.0);
        const auto& factor = solver.factor();
        EXPECT_EQ(gpu_round_trace_size(line, " n="), factor.perm.size());
        EXPECT_EQ(gpu_round_trace_size(line, "rounds="), factor.rounds.size());
        EXPECT_EQ(gpu_round_trace_size(line, "raw_factor_nnz="), factor.L.nonZeros());
        EXPECT_EQ(gpu_round_trace_size(line, "stored_nnz="), solver.trsv().stored_nnz());
        EXPECT_EQ(gpu_round_trace_size(line, "adopted="), solver.trsv().adopted_device_factor());
        EXPECT_EQ(gpu_round_trace_size(line, "fp16="), solver.trsv().fp16());
        EXPECT_EQ(gpu_round_trace_size(line, "host_factor_array_bytes="),
                  csc_access::inner(factor.L).size() * sizeof(node_index) +
                  csc_access::vals(factor.L).size() * sizeof(apxchol::factor_value_t));
        EXPECT_EQ(gpu_round_trace_size(line, "adoption_download_bytes="),
                  solver.trsv().adoption_host_download_bytes());
        EXPECT_EQ(std::stod(gpu_round_trace_field(line, "factor_drop_rel=")),
                  solver.trsv().drop_stats().rel);
        EXPECT_EQ(gpu_round_trace_size(line, "dropped_threshold="),
                  solver.trsv().drop_stats().dropped_threshold);
        EXPECT_EQ(gpu_round_trace_size(line, "dropped_flush="),
                  solver.trsv().drop_stats().dropped_flush);
    } else {
        (void)solver;
        EXPECT_EQ(begin, std::string::npos);
    }
}
#endif

#if defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
constexpr node_index kResidentSelectionVertexCount = 8;
constexpr std::uint64_t kResidentSelectionSeed = 0x98765432U;

std::vector<undirected_edge> resident_selection_path_edges() {
    std::vector<undirected_edge> edges;
    for (node_index vertex = 0;
         vertex + 1 < kResidentSelectionVertexCount; ++vertex)
        edges.push_back({vertex, vertex + 1,
                         1.0 + static_cast<double>(vertex)});
    return edges;
}

struct resident_selection_boundary {
    apxchol::detail::gpu_round_shadow_device_state state;
    std::unique_ptr<apxchol::detail::gpu_block_frontend> frontend;
    std::vector<node_index> active;
    std::vector<node_index> selected;
    apxchol::detail::gpu_device_selection device_selection;
    std::uint64_t initial_generation = 0;
};

resident_selection_boundary make_resident_selection_boundary() {
    resident_selection_boundary boundary;
    const auto edges = resident_selection_path_edges();
    const auto input = make_input(kResidentSelectionVertexCount, edges, {}, kResidentSelectionSeed);
    boundary.frontend = boundary.state.initialize_owned_raw_for_test(input);
    const std::vector<node_index> first_pivots = {0};
    boundary.frontend->select_for_test(first_pivots);
    const auto first = boundary.state.compute_owned_round_for_test(
        boundary.frontend->device_selection(), kResidentSelectionSeed);
    boundary.initial_generation = first.output_generation;
    boundary.state.advance_selector(*boundary.frontend);
    boundary.active.resize(kResidentSelectionVertexCount - 1);
    std::iota(boundary.active.begin(), boundary.active.end(), node_index{1});
    apxchol::partition_options options;
    options.degree_multiplier = 100.0;
    boundary.frontend->prepare(boundary.active, options);
    boundary.selected = boundary.frontend->select_block_greedy().data;
    if (boundary.selected.size() < 2)
        throw std::runtime_error("resident selection test requires at least two selected vertices");
    boundary.device_selection = boundary.frontend->device_selection();
    return boundary;
}

gpu_round_shadow_input resident_selection_round_input(
        std::span<const node_index> pivots) {
    auto edges = resident_selection_path_edges();
    edges.erase(edges.begin());
    auto input = make_input(
        kResidentSelectionVertexCount, edges, pivots,
        kResidentSelectionSeed);
    input.active[0] = 0;
    return input;
}

#if defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
void inject_device_selection_fault(
        apxchol::detail::gpu_block_frontend& frontend,
        std::span<const node_index> replacement) {
    frontend.inject_device_selection_fault_for_test(replacement);
}
#endif
#endif
#endif

template<class F>
void expect_exception_contains(F&& operation, std::string_view needle) {
    try {
        operation();
        ADD_FAILURE() << "expected an exception containing: " << needle;
    } catch (const std::exception& error) {
        EXPECT_NE(std::string_view(error.what()).find(needle),
                  std::string_view::npos)
            << "actual exception: " << error.what();
    } catch (...) {
        ADD_FAILURE() << "expected a std::exception containing: " << needle;
    }
}



#define REQUIRE_GPU_ROUND_SHADOW_DEVICE()                                      \
    do {                                                                        \
        if (!apxchol::detail::gpu_round_shadow_runtime_available())             \
            GTEST_SKIP() << "CUDA round-shadow runtime/device unavailable";    \
    } while (false)

} // namespace

TEST(GpuRoundShadowReference, SetupDiagnosticClockIsExplicit) {
    using clock = apxchol::detail::gpu_setup_diagnostic_clock;
    EXPECT_EQ(clock::now(false), clock::time_point{});
    EXPECT_NE(clock::now(true), clock::time_point{});
}

TEST(GpuRoundShadowReference, FreshPairedContentMatchesFullValidation) {
    const std::vector<undirected_edge> edges = {
        {0, 1, 1.0}, {0, 1, 2.0}, {1, 2, -0.0}, {3, 4, 3.0}};
    const std::vector<node_index> pivots = {0, 3};
    auto input = make_input(6, edges, pivots);
    input.excess[5] = 0.125;
    const auto expected = apxchol::detail::gpu_round_shadow_selection_content(input);
    for (int threads : {1, 2}) {
        scoped_omp_threads team(threads);
        EXPECT_EQ(apxchol::detail::gpu_round_selection_content_from_paired_input(input), expected);
    }
    // Every graph builder branch still emits canonical lower weights twice.
    for (bool full : {false, true}) {
        std::vector<apxchol::test::triplet> entries = {
            {0, 0, 3}, {1, 1, 3}, {2, 2, 2}, {3, 3, 1},
            {1, 0, -1}, {2, 1, -2}};
        if (full) { entries.push_back({0, 1, -7}); entries.push_back({1, 2, -9}); }
        const auto A = apxchol::test::from_triplets(4, 4, entries);
        const auto graph = apxchol::detail::make_graph(apxchol::test::internal_csc(A));
        const std::vector<node_index> selected = {0};
        const auto fresh = apxchol::detail::make_gpu_round_shadow_input(graph, selected, 42);
        EXPECT_EQ(apxchol::detail::gpu_round_selection_content_from_paired_input(fresh),
                  apxchol::detail::gpu_round_shadow_selection_content(fresh));
    }
}

TEST(GpuRoundShadowReference, FreshPairedInputRetainsLinearValidation) {
    const std::vector<undirected_edge> edges = {{0, 1, 1.0}, {1, 2, 2.0}};
    const std::vector<node_index> pivots = {0};
    const auto good = make_input(4, edges, pivots);
    for (int fault = 0; fault < 11; ++fault) {
        SCOPED_TRACE(fault);
        auto input = good;
        switch (fault) {
        case 0: input.owner_offsets[1] = input.incidences.size() + 1; break;
        case 1: input.active[3] = 0; break;
        case 2: input.excess[3] = std::numeric_limits<double>::quiet_NaN(); break;
        case 3: input.incidences[0].weight = std::numeric_limits<double>::infinity(); break;
        case 4: input.incidences[0].weight = -1; break;
        case 5: input.incidences[0].owner = 2; break;
        case 6: input.incidences[0].neighbor = 4; break;
        case 7: input.incidences[0].neighbor = 0; break;
        case 8: input.pivots.push_back(0); input.seeds.push_back(1); break;
        case 9: input.pivots.push_back(1); input.seeds.push_back(1); break;
        case 10: input.excess.clear(); break;
        }
        EXPECT_THROW(apxchol::detail::gpu_round_selection_content_from_paired_input(input),
                     std::invalid_argument);
    }
}

TEST(GpuRoundShadowReference, PoolStorageRetainsDoubleDegreeAccumulation) {
    const std::vector<node_index> pivots = {0};
    const auto check_degree = [&](std::vector<undirected_edge> edges) {
        const auto input = make_input(4, edges, pivots);
        double expected = 0.0;
        for (const auto& edge : edges)
            expected += double(static_cast<float>(edge.weight));
        const auto report = apxchol::detail::reference_gpu_round_shadow(input);
        ASSERT_EQ(report.pivots.size(), 1u);
        EXPECT_EQ(report.pivots[0].total_degree_bits,
                  std::bit_cast<std::uint64_t>(expected > 0.0 ? expected : 1.0));
    };
    // A float accumulator would lose the two unit contributions.
    check_degree({{0, 1, 0x1p24}, {0, 2, 1.0}, {0, 3, 1.0}});
    for (double weight : {0.1, 0x1.0000000000001p0, 0x1p-150})
        check_degree({{0, 1, weight}});
}

TEST(GpuRoundShadowReference, RawIncidenceDegreePrecedesNeighborDedup) {
    const std::vector<undirected_edge> edges = {
        {0, 1, 1.0e16}, {0, 2, 1.0}, {0, 2, 1.0}, {1, 3, 4.0},
    };
    const std::vector<node_index> pivots = {0};
    const auto input = make_input(4, edges, pivots);
    std::vector<gpu_round_shadow_excess_bound> bounds;
    const auto report =
        apxchol::detail::reference_gpu_round_shadow(input, &bounds);

    double raw_degree = 0.0;
    for (std::size_t i = input.owner_offsets[0];
         i < input.owner_offsets[1]; ++i)
        raw_degree += input.incidences[i].weight;
    const double incorrect_deduplicated_resum =
        input.incidences[input.owner_offsets[0]].weight + 2.0;
    EXPECT_NE(std::bit_cast<std::uint64_t>(raw_degree),
              std::bit_cast<std::uint64_t>(incorrect_deduplicated_resum));
    ASSERT_EQ(report.pivots.size(), 1u);
    EXPECT_EQ(report.gathered_incidences, 3u);
    EXPECT_EQ(report.unique_neighbors, 2u);
    EXPECT_EQ(report.pivots[0].total_degree_bits,
              std::bit_cast<std::uint64_t>(raw_degree));
    EXPECT_NO_THROW((void)run_authoritative_cpu_round(
        edges, input, report, bounds));
}

TEST(GpuRoundShadowReference, CollidingFillRemainsAStoredMultigraph) {
    const std::vector<undirected_edge> edges = {
        {0, 2, 1.0}, {0, 3, 3.0},
        {1, 2, 2.0}, {1, 3, 4.0}, {2, 4, 5.0},
    };
    const std::vector<node_index> pivots = {0, 1};
    const auto input = make_input(5, edges, pivots, 0x12345678ULL);
    std::vector<gpu_round_shadow_excess_bound> bounds;
    const auto report =
        apxchol::detail::reference_gpu_round_shadow(input, &bounds);
    auto cpu = run_authoritative_cpu_round(edges, input, report, bounds);

    EXPECT_EQ(report.raw_fill_edges, 2u);
    EXPECT_EQ(report.surviving_input_incidences, 2u);
    EXPECT_EQ(report.live_incidences, 6u);
    EXPECT_NE(report.residual, apxchol::detail::gpu_round_shadow_digest{});
    std::size_t parallel_fill_incidences = 0;
    for (const auto& [neighbor, weight] : cpu.residual.neighbors(2)) {
        (void)weight;
        if (cpu.residual.is_active(neighbor) && neighbor == 3)
            ++parallel_fill_incidences;
    }
    EXPECT_EQ(parallel_fill_incidences, 2u);
}

TEST(GpuRoundShadowReference,
     ResidualDigestIncludesEndpointsAndStoredWeightBits) {
    const std::vector<node_index> no_pivots;
    const auto endpoints_a = make_input(
        4, std::vector<undirected_edge>{{0, 1, 1.0}, {2, 3, 2.0}},
        no_pivots);
    const auto endpoints_b = make_input(
        4, std::vector<undirected_edge>{{0, 2, 1.0}, {1, 3, 2.0}},
        no_pivots);
    const auto weights_b = make_input(
        4, std::vector<undirected_edge>{{0, 1, 1.25}, {2, 3, 2.0}},
        no_pivots);

    const auto a = apxchol::detail::reference_gpu_round_shadow(endpoints_a);
    const auto b = apxchol::detail::reference_gpu_round_shadow(endpoints_b);
    const auto c = apxchol::detail::reference_gpu_round_shadow(weights_b);

    EXPECT_EQ(a.active, b.active);
    EXPECT_EQ(a.live_degree, b.live_degree);
    EXPECT_EQ(a.canonical_excess, b.canonical_excess);
    EXPECT_NE(a.residual, b.residual);
    EXPECT_NE(a.residual, c.residual);
}

TEST(GpuRoundShadowReference,
     ResidentFingerprintCommitsToLiveSlabOrderAndIgnoresDeadRecords) {
    const std::vector<node_index> no_pivots;
    auto ordered = make_input(
        4, std::vector<undirected_edge>{
            {0, 1, 1.0}, {0, 2, 2.0}, {0, 3, 9.0}},
        no_pivots);
    ordered.active[3] = 0;
    auto reordered = ordered;
    ASSERT_EQ(reordered.owner_offsets[0], 0u);
    ASSERT_GE(reordered.owner_offsets[1] - reordered.owner_offsets[0], 3u);
    std::swap(reordered.incidences[0], reordered.incidences[1]);

    const auto a =
        apxchol::detail::fingerprint_gpu_round_shadow_input(ordered);
    const auto b =
        apxchol::detail::fingerprint_gpu_round_shadow_input(reordered);
    EXPECT_EQ(a.residual, b.residual);
    EXPECT_EQ(a.active, b.active);
    EXPECT_EQ(a.live_degree, b.live_degree);
    EXPECT_EQ(a.excess, b.excess);
    EXPECT_EQ(a.live_incidences, b.live_incidences);
    EXPECT_NE(a.ordered_residual, b.ordered_residual);

    auto dead_reordered = ordered;
    std::rotate(dead_reordered.incidences.begin(),
                dead_reordered.incidences.begin() + 2,
                dead_reordered.incidences.begin() + 3);
    // Moving the dead 0->3 record across the live prefix cannot affect either
    // logical digest. It remains present in the physical snapshot.
    const auto c =
        apxchol::detail::fingerprint_gpu_round_shadow_input(dead_reordered);
    EXPECT_EQ(a.residual, c.residual);
    EXPECT_EQ(a.ordered_residual, c.ordered_residual);
}

TEST(GpuRoundShadowReference,
     SerialCpuRoundMatchesTheReusableOwnerOrder) {
    constexpr std::uint64_t run_seed = 123;
    apxchol::detail::graph graph(6);
    for (const auto& edge : std::vector<undirected_edge>{
        {0, 2, 1.0}, {0, 3, 4.0}, {0, 4, 2.0},
        {1, 2, 3.0}, {1, 3, 5.0}, {1, 5, 7.0},
        {2, 4, 6.0}, {3, 5, 8.0}})
        graph.add_edge(edge.u, edge.v, edge.weight);
    const std::vector<node_index> pivots = {0, 1};
    const auto input = apxchol::detail::make_gpu_round_shadow_input(
        graph, pivots, run_seed);
    std::vector<gpu_round_shadow_excess_bound> bounds;
    const auto expected =
        apxchol::detail::reference_gpu_round_shadow(input, &bounds);
    const auto comparison =
        run_serial_cpu_round(graph, input, expected, bounds);
    ASSERT_EQ(comparison.bounded_excess_vertices, 0u);

    const auto cpu = fingerprint_graph(graph);
    EXPECT_EQ(cpu.residual, expected.residual);
    EXPECT_EQ(cpu.ordered_residual, expected.ordered_residual);
    EXPECT_EQ(cpu.active, expected.active);
    EXPECT_EQ(cpu.live_degree, expected.live_degree);
    EXPECT_EQ(cpu.excess, expected.canonical_excess);
    EXPECT_EQ(cpu.active_count, expected.active_count);
    EXPECT_EQ(cpu.live_incidences, expected.live_incidences);
}

TEST(GpuRoundShadowReference,
     ParallelProductionApplyMatchesTheCanonicalReference) {
#ifndef _OPENMP
    GTEST_SKIP() << "OpenMP build required for the parallel apply path";
#else
    constexpr std::uint64_t run_seed = 0x5eed1234ULL;
    auto graph = make_resident_provenance_graph();

    for (const std::vector<node_index>& pivots :
         {std::vector<node_index>{0}, std::vector<node_index>{1, 2},
          std::vector<node_index>{3}}) {
        const auto input = apxchol::detail::make_gpu_round_shadow_input(
            graph, pivots, run_seed);
        std::vector<gpu_round_shadow_excess_bound> bounds;
        const auto expected =
            apxchol::detail::reference_gpu_round_shadow(input, &bounds);
        (void)run_serial_cpu_round(graph, input, expected, bounds);
    }

    std::vector<node_index> pivots(
        kResidentProvenanceParallelPivotEnd -
        kResidentProvenanceFirstParallelPivot);
    std::iota(pivots.begin(), pivots.end(),
              kResidentProvenanceFirstParallelPivot);
    const auto input = apxchol::detail::make_gpu_round_shadow_input(
        graph, pivots, run_seed);
    std::vector<gpu_round_shadow_excess_bound> bounds;
    const auto expected =
        apxchol::detail::reference_gpu_round_shadow(input, &bounds);
    const auto parallel =
        run_parallel_cpu_round(graph, input, expected, bounds);
    EXPECT_EQ(std::popcount(parallel.worker_mask), 2);
    EXPECT_EQ(parallel.team_width, 2);
    // A parallel schedule may happen to reproduce serial encounter order.
    // Canonical equality is mandatory; order reproducibility is not promised.
    EXPECT_EQ(fingerprint_graph(graph).residual, expected.residual);
#endif
}

TEST(GpuRoundShadowReference, ExcessBoundsNameOnlyCollidingAtomicTargets) {
    const std::vector<undirected_edge> edges = {
        {0, 2, 2.0}, {0, 3, 6.0},
        {1, 2, 5.0}, {1, 3, 7.0}, {2, 4, 1.0},
    };
    const std::vector<node_index> pivots = {0, 1};
    const std::vector<double> excess = {4.0, 3.0, 1.0, 0.0, 0.0};
    const auto input = make_input(5, edges, pivots, 17, excess);
    std::vector<gpu_round_shadow_excess_bound> bounds;
    const auto report =
        apxchol::detail::reference_gpu_round_shadow(input, &bounds);

    EXPECT_EQ(report.excess_updates, 4u);
    EXPECT_EQ(report.excess_targets, 2u);
    EXPECT_EQ(bounds[2].additions, 2u);
    EXPECT_EQ(bounds[3].additions, 2u);
    EXPECT_EQ(bounds[4].additions, 0u);
    const auto cpu = run_authoritative_cpu_round(
        edges, input, report, bounds);
    EXPECT_EQ(cpu.comparison.bounded_excess_vertices, 2u);
}

TEST(GpuRoundShadowReference, OversizedPivotUsesTheAllDegreePath) {
    constexpr node_index degree = 4097;
    std::vector<undirected_edge> edges;
    edges.reserve(degree + 1);
    for (node_index neighbor = 1; neighbor <= degree; ++neighbor) {
        const double weight = neighbor % 11 == 0
            ? 0.0
            : 0.125 * static_cast<double>(1 + (neighbor * 37) % 29);
        edges.push_back({0, neighbor, weight});
    }
    edges.push_back({1, degree + 1, 9.0});
    const std::vector<node_index> pivots = {0};
    const auto input = make_input(degree + 2, edges, pivots, 99);
    const auto report = apxchol::detail::reference_gpu_round_shadow(input);

    ASSERT_EQ(report.pivots.size(), 1u);
    EXPECT_EQ(report.pivots[0].unique_degree, degree);
    EXPECT_EQ(report.pivots[0].emitted_edges, degree - 1);
    EXPECT_EQ(report.raw_fill_edges, degree - 1);
}

TEST(GpuRoundShadowCapacity, CubLimitFailsBeforeAnyDeviceAllocation) {
    gpu_round_shadow_input input;
    input.vertex_count = 1;
    input.owner_offsets = {0, 0};
    input.active = {1};
    input.excess = {0.0};
    gpu_round_shadow_report impossible;
    impossible.gathered_incidences =
        static_cast<std::uint64_t>(std::numeric_limits<int>::max()) + 1;
    EXPECT_THROW(
        apxchol::detail::gpu_round_shadow_validate_capacity(input, impossible),
                 std::overflow_error);
}

TEST(GpuRoundShadowCapacity,
     ResidentOwnerOffsetScanLimitFailsBeforeAnyDeviceAllocation) {
    gpu_round_shadow_input input;
    input.vertex_count = static_cast<node_index>(
        static_cast<std::uint64_t>(std::numeric_limits<int>::max()) + 1);
    gpu_round_shadow_report shape;
    EXPECT_THROW(
        apxchol::detail::gpu_round_shadow_validate_capacity(input, shape),
        std::overflow_error);
}

TEST(GpuRoundShadowValidation, NonIndependentSelectedSetIsRejected) {
    const std::vector<undirected_edge> edges = {{0, 1, 1.0}};
    const std::vector<node_index> pivots = {0, 1};
    const auto input = make_input(2, edges, pivots);
    EXPECT_THROW(apxchol::detail::reference_gpu_round_shadow(input),
                 std::invalid_argument);
}

TEST(GpuRoundShadowValidation, NonOwnerMajorInputIsRejected) {
    const std::vector<undirected_edge> edges = {{0, 1, 1.0}};
    const std::vector<node_index> pivots = {0};
    auto input = make_input(2, edges, pivots);
    input.incidences[0].owner = 1;
    EXPECT_THROW(apxchol::detail::reference_gpu_round_shadow(input),
                 std::invalid_argument);
}

#if defined(APXCHOL_USE_CUDA) && defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
TEST(GpuRoundShadowDevice, ProvenanceAndExecutionMarker) {
    establish_r2a_openmp_affinity_if_requested();
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    const std::vector<undirected_edge> edges = {
        {0, 1, 1.0}, {0, 2, 2.0}, {1, 3, 3.0},
    };
    const std::vector<node_index> pivots = {0};
    const auto input = make_input(4, edges, pivots);
    const auto report = run_device(input);
    EXPECT_TRUE(report.gpu_executed);
    std::cout << "R1_DEVICE_PROVENANCE commit=" << APXCHOL_GIT_SHA
              << " gpu_executed=1\n";
}

TEST(GpuRoundShadowDevice, DiscoversAllStageSizesWithoutCpuShapeInput) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    const std::vector<undirected_edge> edges = {
        {0, 3, 1.0}, {0, 4, 2.0}, {0, 4, 3.0},
        {1, 3, 4.0}, {1, 5, 5.0}, {2, 4, 6.0},
        {2, 5, 7.0}, {3, 6, 8.0}, {4, 6, 9.0},
    };
    const std::vector<node_index> pivots = {0, 1, 2};
    const auto input = make_input(7, edges, pivots, 0x5eedULL,
        std::vector<double>{1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    const auto expected = apxchol::detail::reference_gpu_round_shadow(input);
    const auto discovered = run_device(input);
    EXPECT_TRUE(discovered.gpu_executed);
    EXPECT_EQ(discovered.gathered_incidences, expected.gathered_incidences);
    EXPECT_EQ(discovered.unique_neighbors, expected.unique_neighbors);
    EXPECT_EQ(discovered.raw_fill_edges, expected.raw_fill_edges);
    EXPECT_EQ(discovered.excess_updates, expected.excess_updates);
    EXPECT_EQ(discovered.live_incidences, expected.live_incidences);
    EXPECT_EQ(discovered.factor_log_columns, 3u);
    EXPECT_EQ(discovered.factor_log_entries, 6u);
}

TEST(GpuRoundShadowDevice, ConsumesBlockFrontendSelectionWithoutReupload) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    auto boundary = make_resident_selection_boundary();
    const auto input = resident_selection_round_input(boundary.selected);
    const auto expected = apxchol::detail::reference_gpu_round_shadow(input);
    const auto actual = boundary.state.compute_owned_round_for_test(
        boundary.device_selection, kResidentSelectionSeed);
    EXPECT_EQ(actual.live_incidences, expected.live_incidences);
    EXPECT_EQ(actual.selection_validation_pivots, boundary.selected.size());
    EXPECT_EQ(actual.round_state_upload_bytes, 0u);
    EXPECT_TRUE(actual.resident_selection_consumed);
    const auto snapshot = boundary.state.download_owned_prefix_for_test().residual;
    EXPECT_EQ(apxchol::detail::fingerprint_gpu_round_shadow_input(snapshot).residual,
              expected.residual);
}

TEST(GpuRoundShadowDevice, ResidentSelectionAuditAddsExactlyOneIncidenceScan) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    auto boundary = make_resident_selection_boundary();
    boundary.state.set_selection_audit_for_test(true);
    const auto input = resident_selection_round_input(boundary.selected);
    const auto expected = apxchol::detail::reference_gpu_round_shadow(input);
    const auto actual = boundary.state.compute_owned_round_for_test(
        boundary.device_selection, kResidentSelectionSeed);
    EXPECT_TRUE(actual.resident_selection_consumed);
    EXPECT_EQ(actual.selection_validation_pivots, boundary.selected.size());
    EXPECT_EQ(actual.selection_audit_passes, 1u);
    EXPECT_EQ(actual.selection_audit_incidences, actual.resident_input_incidences);
    EXPECT_EQ(actual.selection_map_initialization_vertices, 0u);
    const auto snapshot = boundary.state.download_owned_prefix_for_test().residual;
    EXPECT_EQ(apxchol::detail::fingerprint_gpu_round_shadow_input(snapshot).residual,
              expected.residual);
}

TEST(GpuRoundShadowDevice, SelectionCapabilityDigestSpansMultipleBlocks) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    constexpr node_index n = 600;
    std::vector<undirected_edge> edges;
    std::vector<node_index> pivots;
    for (node_index vertex = 0; vertex + 1 < n; ++vertex)
        edges.push_back({vertex, vertex + 1, 1.0});
    for (node_index vertex = 0; vertex < n; vertex += 2) pivots.push_back(vertex);
    ASSERT_GT(pivots.size(), 256u);
    const auto report = run_device(make_input(n, edges, pivots, 0x1234abcdULL));
    EXPECT_EQ(report.selection_validation_pivots, pivots.size());
}

TEST(GpuRoundShadowDevice, ChainsADeviceAuthoritativeGenerationWithoutHostState) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    const auto edges = resident_selection_path_edges();
    auto initial = make_input(kResidentSelectionVertexCount, edges, {}, kResidentSelectionSeed);
    auto oracle = graph_from_input(initial);
    apxchol::detail::gpu_round_shadow_device_state state;
    auto selector = state.initialize_owned_raw_for_test(initial);
    std::vector<node_index> active(kResidentSelectionVertexCount);
    std::iota(active.begin(), active.end(), node_index{0});
    apxchol::detail::gpu_round_shadow_factor_log expected_log;
    std::vector<gpu_round_shadow_report> reports;
    for (unsigned round = 0; round < 2; ++round) {
        selector->prepare(active, {});
        const auto selected = selector->select_block_greedy().data;
        ASSERT_FALSE(selected.empty());
        const auto input = apxchol::detail::make_gpu_round_shadow_input(oracle, selected, kResidentSelectionSeed);
        const auto report = state.compute_owned_round_for_test(selector->device_selection(), kResidentSelectionSeed);
        expected_log = append_factor_logs(expected_log, run_owned_host_oracle(oracle, input));
        expect_factor_logs_equal(expected_log, state.download_factor_log());
        const auto actual = state.download_owned_prefix_for_test().residual;
        const auto expected = compact_live_snapshot(apxchol::detail::make_gpu_round_shadow_input(
            oracle, std::span<const node_index>{}, kResidentSelectionSeed));
        EXPECT_EQ(apxchol::detail::fingerprint_gpu_round_shadow_input(actual),
                  apxchol::detail::fingerprint_gpu_round_shadow_input(expected));
        EXPECT_EQ(report.round_state_upload_bytes, 0u);
        EXPECT_TRUE(report.resident_selection_consumed);
        EXPECT_EQ(report.selection_validation_pivots, selected.size());
        EXPECT_EQ(report.factor_log_columns, expected_log.columns.size());
        EXPECT_EQ(report.factor_log_entries, expected_log.entries.size());
        reports.push_back(report);
        std::erase_if(active, [&](node_index v) { return !oracle.is_active(v); });
        state.advance_selector(*selector);
    }
    EXPECT_EQ(reports[1].state_imports, 1u);
    EXPECT_GT(reports[1].state_reuses, reports[0].state_reuses);
    EXPECT_EQ(reports[1].state_buffer_allocations, reports[0].state_buffer_allocations);
    EXPECT_EQ(reports[1].state_buffer_growths, reports[0].state_buffer_growths);
}
#endif

#if defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
#if defined(APXCHOL_USE_CUDA) && defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
TEST(GpuRoundShadowDevice,
     ResidentSelectionValidationRejectsUnsafeIdsAndPoisonsTheGeneration) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    enum class corruption { out_of_range, inactive, duplicate, adjacent };
    const std::pair<corruption, std::string_view> cases[] = {
        {corruption::out_of_range, "out-of-range vertex"},
        {corruption::inactive, "inactive vertex"},
        {corruption::duplicate, "duplicate vertex"},
        {corruption::adjacent, "not independent"},
    };
    for (const auto& [kind, expected_message] : cases) {
        SCOPED_TRACE(expected_message);
        auto boundary = make_resident_selection_boundary();
        auto malformed = boundary.selected;
        switch (kind) {
        case corruption::out_of_range:
            malformed[0] = kResidentSelectionVertexCount;
            break;
        case corruption::inactive:
            malformed[0] = 0;
            break;
        case corruption::duplicate:
            malformed[1] = malformed[0];
            break;
        case corruption::adjacent:
            for (std::size_t i = 0; i < malformed.size(); ++i)
                malformed[i] = static_cast<node_index>(i + 1);
            break;
        }
        inject_device_selection_fault(*boundary.frontend, malformed);
        auto reject_malformed = [&] {
            expect_exception_contains(
                [&] {
                    (void)boundary.state.compute_owned_round_for_test(
                        boundary.device_selection, kResidentSelectionSeed);
                },
                expected_message);
        };
        boundary.state.set_selection_audit_for_test(kind == corruption::adjacent);
        reject_malformed();
        expect_exception_contains(
            [&] {
                (void)boundary.state.compute_owned_round_for_test(
                    boundary.device_selection, kResidentSelectionSeed);
            },
            "poisoned by a failed generation");
    }
}

TEST(GpuRoundShadowDevice,
     ResidentSelectionCapabilityRejectsMutatedIdsWithoutResidualScan) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    auto boundary = make_resident_selection_boundary();
    auto reordered = boundary.selected;
    std::reverse(reordered.begin(), reordered.end());
    ASSERT_NE(reordered, boundary.selected);
    inject_device_selection_fault(*boundary.frontend, reordered);
    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_owned_round_for_test(
                boundary.device_selection, kResidentSelectionSeed);
        },
        "ids do not match the producer capability");
    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_owned_round_for_test(
                boundary.device_selection, kResidentSelectionSeed);
        },
        "poisoned by a failed generation");
}
#endif

TEST(GpuRoundShadowDevice, DeviceSelectionCapabilityCannotBeCallerForged) {
    static_assert(!std::is_aggregate_v<
                  apxchol::detail::gpu_device_selection>);
    static_assert(!std::is_default_constructible_v<
                  apxchol::detail::gpu_device_selection_producer>);
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    auto boundary = make_resident_selection_boundary();
    const apxchol::detail::gpu_device_selection forged;
    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_owned_round_for_test(
                forged, kResidentSelectionSeed);
        },
        "was not issued by a producer");
    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_owned_round_for_test(
                boundary.device_selection, kResidentSelectionSeed);
        },
        "poisoned by a failed generation");
}

TEST(GpuRoundShadowDevice, ProducerRefusesAnEmptyDeviceSelectionCapability) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    apxchol::detail::gpu_round_shadow_device_state state;
    auto owner = state.initialize_owned_raw_for_test(make_input(2, std::vector<undirected_edge>{{0, 1, 1.0}}, {}));
    auto& frontend = *owner;
    const std::vector<node_index> active = {0, 1};
    apxchol::partition_options options;
    options.degree_quantile = 0.0;
    options.degree_multiplier = 0.0;
    EXPECT_EQ(frontend.prepare(active, options).candidate_count, 0u);
    EXPECT_TRUE(frontend.select_block_greedy().data.empty());
    expect_exception_contains([&] { (void)frontend.device_selection(); },
        "no current nonempty device selection");
    expect_exception_contains([&] { (void)frontend.host_candidates(); },
        "selection producer is poisoned");
}

TEST(GpuRoundShadowDevice, ProducerAndConsumerRejectTheWrongCudaDevice) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    int original_device = -1;
    int device_count = 0;
    ASSERT_EQ(cudaGetDevice(&original_device), cudaSuccess);
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count < 2)
        GTEST_SKIP() << "wrong-device regression requires two CUDA devices";
    const int other_device = original_device == 0 ? 1 : 0;

    auto producer = make_resident_selection_boundary();
    ASSERT_EQ(cudaSetDevice(other_device), cudaSuccess);
    expect_exception_contains(
        [&] { (void)producer.frontend->host_candidates(); },
        "wrong CUDA device");
    ASSERT_EQ(cudaSetDevice(original_device), cudaSuccess);
    expect_exception_contains(
        [&] { (void)producer.frontend->device_selection(); },
        "selection producer is poisoned");

    auto consumer = make_resident_selection_boundary();
    ASSERT_EQ(cudaSetDevice(other_device), cudaSuccess);
    expect_exception_contains(
        [&] {
            (void)consumer.state.compute_owned_round_for_test(
                consumer.device_selection, kResidentSelectionSeed);
        },
        "wrong CUDA device");
    ASSERT_EQ(cudaSetDevice(original_device), cudaSuccess);
    expect_exception_contains(
        [&] {
            (void)consumer.state.compute_owned_round_for_test(
                consumer.device_selection, kResidentSelectionSeed);
        },
        "poisoned by a failed generation");
}

TEST(GpuRoundShadowDevice,
     ResidentSelectionCapabilityRejectsStaleAndRetiredViews) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    auto boundary = make_resident_selection_boundary();
    const auto stale = boundary.device_selection;
    apxchol::partition_options options;
    options.degree_multiplier = 100.0;
    boundary.frontend->prepare(boundary.active, options);
    boundary.selected = boundary.frontend->select_block_greedy().data;
    const auto current = boundary.frontend->device_selection();

    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_owned_round_for_test(
                stale, kResidentSelectionSeed);
        },
        "stale device selection generation");
    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_owned_round_for_test(
                current, kResidentSelectionSeed);
        },
        "poisoned by a failed generation");

    auto retired = make_resident_selection_boundary();
    const auto retired_view = retired.device_selection;
    retired.frontend.reset();
    expect_exception_contains(
        [&] {
            (void)retired.state.compute_owned_round_for_test(
                retired_view, kResidentSelectionSeed);
        },
        "producer is no longer alive");

    auto failed_producer = make_resident_selection_boundary();
    const auto failed_view = failed_producer.device_selection;
    expect_exception_contains(
        [&] {
            apxchol::partition_options failed_options;
            (void)failed_producer.frontend->prepare(
                std::span<const node_index>{}, failed_options);
        },
        "active-list size diverged");
    expect_exception_contains(
        [&] {
            (void)failed_producer.state.compute_owned_round_for_test(
                failed_view, kResidentSelectionSeed);
        },
        "selection producer is poisoned");
}

TEST(GpuRoundShadowDevice, ResidentSelectionRejectsConsumedTopologyAndDifferentProducer) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    auto consumed = make_resident_selection_boundary();
    consumed.state.compute_owned_round_for_test(consumed.device_selection, kResidentSelectionSeed);
    EXPECT_THROW(consumed.state.compute_owned_round_for_test(
        consumed.device_selection, kResidentSelectionSeed), std::exception);

    auto cross_producer = make_resident_selection_boundary();
    auto other = make_resident_selection_boundary();
    EXPECT_THROW(cross_producer.state.compute_owned_round_for_test(
        other.device_selection, kResidentSelectionSeed), std::exception);
    expect_exception_contains([&] {
        cross_producer.state.compute_owned_round_for_test(
            cross_producer.device_selection, kResidentSelectionSeed);
    }, "poisoned by a failed generation");
}

TEST(GpuRoundShadowDevice, FirstProducerBindRejectsAccidentalCrossStateReplay) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    auto boundary = make_resident_selection_boundary();
    auto wrong_edges = resident_selection_path_edges();
    wrong_edges.pop_back();
    apxchol::detail::gpu_round_shadow_device_state wrong_state;
    auto selector = wrong_state.initialize_owned_raw_for_test(
        make_input(kResidentSelectionVertexCount, wrong_edges, {}, kResidentSelectionSeed));
    std::vector<node_index> active(kResidentSelectionVertexCount);
    std::iota(active.begin(), active.end(), node_index{0});
    selector->prepare(active, {});
    ASSERT_FALSE(selector->select_block_greedy().data.empty());
    EXPECT_THROW(boundary.state.compute_owned_round_for_test(
        selector->device_selection(), kResidentSelectionSeed), std::exception);
    expect_exception_contains([&] {
        boundary.state.compute_owned_round_for_test(boundary.device_selection, kResidentSelectionSeed);
    }, "poisoned by a failed generation");
}
#endif

#if defined(APXCHOL_USE_CUDA) && \
    defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
#if defined(APXCHOL_USE_CUDA) && defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
TEST(GpuRoundShadowDevice,
     InjectedPostCudaFailurePoisonsPreviouslyAcceptedGeneration) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    auto boundary = make_resident_selection_boundary();
    boundary.state.inject_failure_after_cuda_operation_for_test();
    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_owned_round_for_test(
                boundary.device_selection, kResidentSelectionSeed);
        },
        "injected failure after CUDA operation");
    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_owned_round_for_test(
                boundary.device_selection, kResidentSelectionSeed);
        },
        "poisoned by a failed generation");
}
#endif

TEST(GpuRoundShadowDevice, IntegerDigestsMatchOracleAcrossPartialBlocks) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    // Exercise empty, partial, exact and multiple blocks of pivots and
    // incidences. Parallel edges with equal hashes test XOR cancellation and
    // modular sum separately; distinct weights retain order-sensitive checks.
    for (const node_index count : {1u, 31u, 32u, 33u, 255u, 256u, 257u, 513u}) {
        SCOPED_TRACE(count);
        std::vector<undirected_edge> edges;
        std::vector<node_index> pivots;
        for (node_index i = 0; i < count; ++i) {
            const node_index v = 4 * i;
            pivots.push_back(v);
            edges.push_back({v, v + 1, 1.0});
            edges.push_back({v, v + 1, 1.0});
            edges.push_back({v, v + 2, 0.25});
            edges.push_back({v, v + 2, 2.0});
            edges.push_back({v + 1, v + 3, 0.5});
            edges.push_back({v + 2, v + 3, 4.0});
        }
        auto input = make_input(4 * count + 1, edges, pivots, 0x815eed);
        input.excess.back() = 0.125;  // surviving isolated active vertex
        std::vector<gpu_round_shadow_excess_bound> bounds;
        (void)run_device(input, &bounds);
        EXPECT_NO_THROW((void)run_authoritative_cpu_round(edges, input, apxchol::detail::reference_gpu_round_shadow(input), bounds));
        // The empty selection still checks the uploaded residual snapshot.
        input.pivots.clear(); input.seeds.clear();
        (void)run_device(input);
    }
}

TEST(GpuRoundShadowDevice, RawIncidenceDegreeMatchesAuthoritativeCpu) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    const std::vector<undirected_edge> edges = {
        {0, 1, 1.0e16}, {0, 2, 1.0}, {0, 2, 1.0}, {1, 3, 4.0},
    };
    const std::vector<node_index> pivots = {0};
    const auto input = make_input(4, edges, pivots);
    std::vector<gpu_round_shadow_excess_bound> bounds;
    (void)run_device(input, &bounds);
    EXPECT_NO_THROW((void)run_authoritative_cpu_round(
        edges, input, apxchol::detail::reference_gpu_round_shadow(input), bounds));
}

TEST(GpuRoundShadowDevice, RawCollidingFillsMatchCpuStoredMultigraph) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    const std::vector<undirected_edge> edges = {
        {0, 2, 1.0}, {0, 3, 3.0},
        {1, 2, 2.0}, {1, 3, 4.0}, {2, 4, 5.0},
    };
    const std::vector<node_index> pivots = {0, 1};
    const auto input = make_input(5, edges, pivots, 0x12345678ULL);
    std::vector<gpu_round_shadow_excess_bound> bounds;
    const auto device = run_device(input, &bounds);
    auto cpu = run_authoritative_cpu_round(edges, input, apxchol::detail::reference_gpu_round_shadow(input), bounds);

    std::size_t parallel_fill_incidences = 0;
    for (const auto& [neighbor, weight] : cpu.residual.neighbors(2)) {
        (void)weight;
        if (cpu.residual.is_active(neighbor) && neighbor == 3)
            ++parallel_fill_incidences;
    }
    EXPECT_EQ(parallel_fill_incidences, 2u);
    EXPECT_EQ(device.live_incidences, 6u);
}

TEST(GpuRoundShadowDevice, AtomicExcessCollisionsUseExplicitBounds) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    const std::vector<undirected_edge> edges = {
        {0, 2, 2.0}, {0, 3, 6.0},
        {1, 2, 5.0}, {1, 3, 7.0}, {2, 4, 1.0},
    };
    const std::vector<node_index> pivots = {0, 1};
    const std::vector<double> excess = {4.0, 3.0, 1.0, 0.0, 0.0};
    const auto input = make_input(5, edges, pivots, 17, excess);
    std::vector<gpu_round_shadow_excess_bound> bounds;
    (void)run_device(input, &bounds);
    const auto cpu = run_authoritative_cpu_round(edges, input, apxchol::detail::reference_gpu_round_shadow(input), bounds);
    EXPECT_EQ(cpu.comparison.bounded_excess_vertices, 2u);
}

TEST(GpuRoundShadowDevice, RepeatExecutionIsCanonicallyDeterministic) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    const std::vector<undirected_edge> edges = {
        {0, 2, 1.0}, {0, 3, 5.0}, {0, 4, 2.0},
        {1, 2, 3.0}, {1, 3, 7.0}, {1, 5, 4.0},
        {2, 6, 9.0}, {3, 6, 11.0},
    };
    const std::vector<node_index> pivots = {0, 1};
    const auto input = make_input(7, edges, pivots, 0xabcdefULL);
    std::vector<gpu_round_shadow_excess_bound> bounds;
    const auto baseline = run_device(input, &bounds);
    EXPECT_NO_THROW((void)run_authoritative_cpu_round(
        edges, input, apxchol::detail::reference_gpu_round_shadow(input), bounds));
    for (int repetition = 0; repetition < 5; ++repetition) {
        const auto current = run_device(input);
        apxchol::detail::compare_gpu_round_shadow_reports(
            baseline, current);
    }
}

TEST(GpuRoundShadowDevice, OversizedPivotMatchesAuthoritativeCpu) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    constexpr node_index degree = 4097;
    std::vector<undirected_edge> edges;
    edges.reserve(degree + 1);
    for (node_index neighbor = 1; neighbor <= degree; ++neighbor) {
        const double weight = neighbor % 11 == 0
            ? 0.0
            : 0.125 * static_cast<double>(1 + (neighbor * 37) % 29);
        edges.push_back({0, neighbor, weight});
    }
    edges.push_back({1, degree + 1, 9.0});
    const std::vector<node_index> pivots = {0};
    const auto input = make_input(degree + 2, edges, pivots, 99);
    std::vector<gpu_round_shadow_excess_bound> bounds;
    (void)run_device(input, &bounds);
    EXPECT_NO_THROW((void)run_authoritative_cpu_round(
        edges, input, apxchol::detail::reference_gpu_round_shadow(input), bounds));
}

namespace {
constexpr node_index kOwnedVertices = 98;
constexpr std::uint64_t kOwnedSeed = 123;

apxchol::detail::graph make_owned_order_fixture() {
    using arc = std::pair<node_index, double>;
    std::vector<std::vector<arc>> rows(kOwnedVertices);
    std::size_t edges = 0;
    auto add = [&](node_index u, node_index v, double w) {
        rows[u].push_back({v, w}); rows[v].push_back({u, w}); ++edges;
    };
    for (node_index u = 0; u < 12; ++u)
        for (node_index v = u + 1; v < 12; ++v) add(u, v, 1.0);
    add(0, 1, 0x1p54); add(0, 1, 0x1p-20);
    for (node_index v = 12; v + 1 < kOwnedVertices; ++v)
        if (v != 62 && v != 63 && v != 64) add(v, v + 1, 1.0);
    // Isolated active ids 63/64 cross the active bitmap's word boundary.
    apxchol::detail::graph graph(kOwnedVertices);
    std::vector<node_index> ids(kOwnedVertices), sizes(kOwnedVertices);
    std::iota(ids.begin(), ids.end(), node_index{0});
    for (node_index v : ids) {
        if (v % 2) std::reverse(rows[v].begin(), rows[v].end());
        if (!rows[v].empty()) std::rotate(rows[v].begin(),
            rows[v].begin() + (v + 1) % rows[v].size(), rows[v].end());
        sizes[v] = rows[v].size();
    }
    graph.adj_bulk_reserve_parallel(ids.begin(), ids.end(), sizes);
    for (node_index v : ids) {
        for (node_index k = 0; k < sizes[v]; ++k)
            graph.adj_write_reserved_directed_at(v, k, rows[v][k].first, rows[v][k].second);
        graph.adj_commit_reserved_directed(v, sizes[v]);
        // The dense component stays above the first two candidate degree caps.
        graph.excess(v) = v < 12 ? 0.25 : v == 63 ? 2.0 : v == 64 ? 3.0 : 0.0;
    }
    graph.record_edges_added(edges);
    return graph;
}

struct owned_selector_view_fixture {
    apxchol::detail::graph graph = make_owned_order_fixture();
    std::vector<node_index> active;
    std::unique_ptr<apxchol::detail::gpu_round_shadow_session> session =
        std::make_unique<apxchol::detail::gpu_round_shadow_session>(true);
    std::unique_ptr<apxchol::detail::gpu_block_frontend> selector_owner =
        apxchol::test::initialize_owned_selector(*session, graph);
    apxchol::detail::gpu_block_frontend& selector = *selector_owner;
    apxchol::partition_options options;
    bool degree_tiebreak = true;

    owned_selector_view_fixture() : active(graph.n()) {
        std::iota(active.begin(), active.end(), node_index{0});
        options.degree_quantile = 0.3;
    }

    std::vector<node_index> select() {
        selector.prepare(active, options, 42, 0, degree_tiebreak);
        auto selected = selector.select_block_greedy().data;
        if (selected.empty()) throw std::logic_error("owned selector fixture exhausted too soon");
        return selected;
    }

    gpu_round_shadow_report run_selected(
            std::span<const node_index> selected,
            apxchol::detail::gpu_device_selection capability) {
        auto report = session->run_owned_prefix_round(capability, kOwnedSeed);
        std::erase_if(active, [&](node_index v) {
            return std::binary_search(selected.begin(), selected.end(), v);
        });
        EXPECT_EQ(report.active_count, active.size());
        return report;
    }

    void first_bound_round() {
        const auto selected = select();
        run_selected(selected, selector.device_selection());
        if (active.empty()) throw std::logic_error("owned selector fixture has no second round");
        session->advance_selector(selector);
    }

    void compare_reference(const apxchol::detail::graph& oracle) {
        ASSERT_GT(selector.transfers().owned_residual_binds, 0u);
        const auto prepared = selector.prepare(active, options, 42, 0, degree_tiebreak);
        const auto actual_degrees = selector.host_active_degrees();
        ASSERT_EQ(actual_degrees.size(), active.size());
        std::vector<node_index> expected_degrees;
        std::size_t sum = 0;
        for (auto v : active) {
            node_index degree = 0;
            for (const auto& edge : oracle.neighbors(v)) degree += oracle.is_active(edge.to);
            expected_degrees.push_back(degree); sum += degree;
        }
        EXPECT_EQ(std::vector<node_index>(actual_degrees.begin(), actual_degrees.end()), expected_degrees);
        double threshold = options.degree_multiplier * double(sum) / active.size();
        const double quantile =
            options.degree_quantile.value_or(apxchol::detail::degree_quantile_host_default);
        if (quantile > 0.0 && quantile < 1.0) {
            auto sorted = expected_degrees; std::sort(sorted.begin(), sorted.end());
            threshold = sorted[static_cast<std::size_t>(quantile * (sorted.size()-1))];
        }
        std::vector<node_index> expected_candidates;
        for (std::size_t i = 0; i < active.size(); ++i)
            if (expected_degrees[i] <= threshold) expected_candidates.push_back(active[i]);
        const auto candidates = selector.host_candidates();
        EXPECT_EQ(std::vector<node_index>(candidates.begin(), candidates.end()), expected_candidates);
        EXPECT_EQ(prepared.candidate_count, expected_candidates.size());
        const auto selected = selector.select_block_greedy().data;
        ASSERT_FALSE(selected.empty());
        EXPECT_EQ(selector.select_block_greedy().data, selected);
        std::vector<bool> picked(oracle.n()); std::size_t work = 0;
        for (auto v : selected) {
            ASSERT_TRUE(std::binary_search(expected_candidates.begin(), expected_candidates.end(), v));
            ASSERT_FALSE(picked[v]); picked[v] = true;
            work += expected_degrees[std::lower_bound(active.begin(), active.end(), v)-active.begin()];
        }
        for (auto v : selected) for (const auto& edge : oracle.neighbors(v))
            if (oracle.is_active(edge.to)) {
                EXPECT_FALSE(picked[edge.to]);
            }
        EXPECT_EQ(selector.selected_degree_work(), work);
        // Bounded progress and independence, not the generic CSR maximal law.
    }

};

std::string owned_residual_boundary_receipt(
        const gpu_round_shadow_input& state,
        std::span<const apxchol::detail::factor_col> columns) {
    // Serialize actual fields, including every stored weight bit and offset;
    // an equal digest/count alone would not prove ordered handback identity.
    std::ostringstream out;
    out << state.vertex_count << ";offsets";
    for (auto offset : state.owner_offsets) out << ':' << offset;
    out << ";active";
    for (auto active : state.active) out << ':' << unsigned(active);
    out << ";excess";
    for (double value : state.excess)
        out << ':' << std::bit_cast<std::uint64_t>(value);
    out << ";incidences";
    for (const auto& edge : state.incidences)
        out << ':' << edge.owner << ',' << edge.neighbor << ','
            << std::bit_cast<std::uint64_t>(double(edge.weight));
    out << ";columns";
    for (const auto& column : columns)
        out << ':' << column.vertex << ','
            << std::bit_cast<std::uint32_t>(column.diag) << ',' << column.entry_count;
    return out.str();
}

void expect_owned_residual_boundary(
        std::span<const undirected_edge> edges, node_index n,
        const std::vector<node_index>& wanted_selection,
        const apxchol::partition_options& options,
        std::size_t expected_fill, std::size_t expected_live,
        const std::string& receipt_name) {
    apxchol::detail::graph graph(n), oracle(n);
    for (const auto& edge : edges) {
        graph.add_edge(edge.u, edge.v, edge.weight);
        oracle.add_edge(edge.u, edge.v, edge.weight);
    }
    for (node_index v = 0; v < n; ++v) {
        const double excess = std::find(wanted_selection.begin(),
            wanted_selection.end(), v) == wanted_selection.end() ? 0.25 * (v + 1) : 0.0;
        graph.excess(v) = oracle.excess(v) = excess;
    }
    std::vector<node_index> active(n);
    std::iota(active.begin(), active.end(), node_index{0});
    apxchol::detail::gpu_round_shadow_session session(true);
    auto selector_owner = apxchol::test::initialize_owned_selector(session, graph);
    auto& selector = *selector_owner;
    selector.prepare(active, options);
    const auto selected = selector.select_for_test(wanted_selection).data;
    ASSERT_EQ(selected, wanted_selection);
    const auto report = session.run_owned_prefix_round(selector.device_selection(), kOwnedSeed);
    ASSERT_EQ(report.raw_fill_edges, expected_fill);
    ASSERT_EQ(report.live_incidences, expected_live);
    ASSERT_TRUE(report.pivots.empty());  // This must exercise the owned route.

    std::vector<apxchol::detail::factor_col> columns;
    // This public boundary internally recomputes the post-sort device hash,
    // compares it with the saved topology, and checks the complete handback.
    observe_owned_prefix(session, graph, active, columns);
    const auto actual = apxchol::detail::make_gpu_round_shadow_input(
        graph, std::span<const node_index>{}, kOwnedSeed);

    // The independent CPU round starts only after the GPU-owned window.
    const auto input = apxchol::detail::make_gpu_round_shadow_input(oracle, selected, kOwnedSeed);
    std::vector<gpu_round_shadow_excess_bound> bounds;
    const auto reference = apxchol::detail::reference_gpu_round_shadow(input, &bounds);
    apxchol::detail::gpu_round_shadow_factor_log expected_columns;
    run_serial_cpu_round(oracle, input, reference, bounds, &expected_columns);
    const auto expected = compact_live_snapshot(
        apxchol::detail::make_gpu_round_shadow_input(
            oracle, std::span<const node_index>{}, kOwnedSeed));
    ASSERT_EQ(actual.vertex_count, expected.vertex_count);
    ASSERT_EQ(actual.owner_offsets, expected.owner_offsets);
    ASSERT_EQ(actual.active, expected.active);
    ASSERT_EQ(actual.incidences.size(), expected_live);
    ASSERT_EQ(actual.incidences.size(), expected.incidences.size());
    ASSERT_EQ(actual.excess.size(), expected.excess.size());
    for (std::size_t i = 0; i < actual.excess.size(); ++i)
        EXPECT_EQ(std::bit_cast<std::uint64_t>(actual.excess[i]),
                  std::bit_cast<std::uint64_t>(expected.excess[i]));
    for (std::size_t i = 0; i < actual.incidences.size(); ++i) {
        SCOPED_TRACE(i);
        EXPECT_EQ(actual.incidences[i].owner, expected.incidences[i].owner);
        EXPECT_EQ(actual.incidences[i].neighbor, expected.incidences[i].neighbor);
        EXPECT_EQ(std::bit_cast<std::uint64_t>(double(actual.incidences[i].weight)),
                  std::bit_cast<std::uint64_t>(double(expected.incidences[i].weight)));
    }
    EXPECT_EQ(apxchol::detail::fingerprint_gpu_round_shadow_input(actual),
              apxchol::detail::fingerprint_gpu_round_shadow_input(expected));

    ASSERT_EQ(columns.size(), expected_columns.columns.size());
    for (std::size_t i = 0; i < columns.size(); ++i) {
        EXPECT_EQ(columns[i].vertex, expected_columns.columns[i].vertex);
        EXPECT_EQ(std::bit_cast<std::uint32_t>(columns[i].diag),
                  std::bit_cast<std::uint32_t>(expected_columns.columns[i].diag));
        EXPECT_EQ(columns[i].entry_count, expected_columns.columns[i].entry_count);
    }
    EXPECT_EQ(active.size(), report.active_count);
    ::testing::Test::RecordProperty(receipt_name,
        std::to_string(report.raw_fill_edges) + ";" +
        owned_residual_boundary_receipt(actual, columns));
}

} // namespace

TEST(GpuOwnedPrefix, ResidualBoundaryHandbackMatchesEveryField) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_omp_threads serial(1);
    apxchol::partition_options options;
    options.degree_quantile = 0.0;
    options.degree_multiplier = 0.0;  // Select only isolated vertex 2.
    constexpr double weights[] = {1.0, 1.0, 0.25, -0.0, 0.0, 0x1p-150, 0x1p20, 0.5};
    for (std::size_t count : {1u, 15u, 16u, 17u, 127u, 128u, 129u, 255u, 256u, 257u}) {
        SCOPED_TRACE(count);
        std::vector<undirected_edge> edges;
        for (std::size_t i = 0; i < count; ++i)
            edges.push_back({0, 1, weights[i % std::size(weights)]});
        expect_owned_residual_boundary(edges, 3, {2}, options, 0, 2 * count,
            "owned_residual_boundary_" + std::to_string(2 * count));
    }
}
#endif


#if defined(APXCHOL_USE_CUDA) && defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
TEST(GpuOwnedPrefix, ResidualFillSortPreservesSurvivorOrder) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_omp_threads serial(1);
    apxchol::partition_options options;
    options.degree_quantile = 0.0;
    options.degree_multiplier = 1.0;
    for (std::size_t count : {14u, 15u, 16u, 126u, 127u, 128u, 254u, 255u, 256u}) {
        SCOPED_TRACE(count);
        std::vector<undirected_edge> edges;
        for (std::size_t i = 0; i < count; ++i)
            edges.push_back({0, 1, i % 2 ? 0.25 : 2.0});
        edges.push_back({2, 0, 1.0});
        edges.push_back({2, 1, 1.0});
        // Only degree-2 center 2 is below the average. Its single fill appends
        // owner 0 after the owner-1 survivors, so all-zero keys cannot pass.
        expect_owned_residual_boundary(edges, 3, {2}, options, 1, 2 * (count + 1),
            "owned_residual_fill_sort_" + std::to_string(2 * (count + 1)));
    }
}
#endif


#if defined(APXCHOL_USE_CUDA) && defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
TEST(GpuOwnedPrefix, ResidualFillCollisionsPreserveLaterRoundAssociation) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_omp_threads serial(1);
    apxchol::partition_options options;
    options.degree_quantile = 0.0;
    options.degree_multiplier = 1.0;
    for (std::size_t count : {31u, 32u, 33u}) {
        SCOPED_TRACE(count);
        std::vector<undirected_edge> edges;
        std::vector<double> survivor_weights;
        // Pivot incidences make holes in the middle of both surviving slabs.
        // Two independent pivots then emit colliding new edges onto that pair.
        edges.push_back({2, 0, 1.0});
        for (std::size_t i = 0; i < count; ++i) {
            const double weight = i == 0 ? 0x1p52 : i % 7 == 0 ? -0.0 : 0.25;
            edges.push_back({0, 1, weight});
            survivor_weights.push_back(weight);
            if (i == count / 2) {
                edges.push_back({3, 1, 1.0});
                edges.push_back({2, 1, 1.0});
            }
        }
        edges.push_back({3, 0, 1.0});
        double encounter_sum = 0.0, reordered_sum = 0.0;
        for (double weight : survivor_weights) encounter_sum += weight;
        std::sort(survivor_weights.begin(), survivor_weights.end());
        for (double weight : survivor_weights) reordered_sum += weight;
        // The next pivot's duplicate reduction is association-sensitive; an
        // owner bucket must not reorder its existing, parallel weight stream.
        ASSERT_NE(std::bit_cast<std::uint64_t>(encounter_sum),
                  std::bit_cast<std::uint64_t>(reordered_sum));
        expect_owned_residual_boundary(edges, 4, {2, 3}, options,
            2, 2 * (count + 2), "owned_residual_collision_" + std::to_string(count));

        apxchol::detail::graph graph(4), oracle(4);
        for (const auto& edge : edges) {
            graph.add_edge(edge.u, edge.v, edge.weight);
            oracle.add_edge(edge.u, edge.v, edge.weight);
        }
        graph.excess(0) = oracle.excess(0) = 0.25;
        graph.excess(1) = oracle.excess(1) = 0.5;
        apxchol::detail::gpu_round_shadow_session session(true);
        auto selector_owner = apxchol::test::initialize_owned_selector(session, graph);
        auto& selector = *selector_owner;
        std::vector<node_index> active{0, 1, 2, 3};
        const std::vector<node_index> initial_collision{2, 3};
        std::vector<std::vector<node_index>> sequence;
        while (!active.empty()) {
            const auto round = sequence.size();
            ASSERT_LT(round, 3u);
            selector.prepare(active, options);
            const auto selected = round == 0 ? selector.select_for_test(initial_collision).data
                : selector.select_block_greedy().data;
            ASSERT_FALSE(selected.empty());
            if (round == 0) ASSERT_EQ(selected, initial_collision);
            else ASSERT_EQ(selected.size(), 1u); // The surviving pair is adjacent.
            std::vector<bool> picked(4);
            for (auto v : selected) {
                ASSERT_TRUE(std::binary_search(active.begin(), active.end(), v));
                ASSERT_FALSE(picked[v]); picked[v] = true;
            }
            // New fill only adds parallel0--1 edges already present in graph;
            // its adjacency therefore also checks every selected-set conflict.
            for (auto v : selected) for (const auto& edge : graph.neighbors(v))
                EXPECT_FALSE(picked[edge.to]);
            // Preserve the deliberate first collision, but let the bounded
            // owned selector choose either continuation orientation. The CPU
            // numerical oracle below receives precisely this observed order.
            sequence.push_back(selected);
            const auto report = session.run_owned_prefix_round(selector.device_selection(), kOwnedSeed);
            ASSERT_TRUE(report.pivots.empty());
            ASSERT_EQ(report.raw_fill_edges, round == 0 ? 2u : 0u);
            std::erase_if(active, [&](node_index v) {
                return std::binary_search(selected.begin(), selected.end(), v);
            });
            if (!active.empty()) session.advance_selector(selector);
        }
        std::vector<apxchol::detail::factor_col> columns;
        session.complete_owned_factorization(columns);
        apxchol::detail::gpu_round_shadow_factor_log expected;
        for (const auto& selected : sequence) {
            const auto input = apxchol::detail::make_gpu_round_shadow_input(oracle, selected, kOwnedSeed);
            std::vector<gpu_round_shadow_excess_bound> bounds;
            const auto reference = apxchol::detail::reference_gpu_round_shadow(input, &bounds);
            apxchol::detail::gpu_round_shadow_factor_log round_factor;
            run_serial_cpu_round(oracle, input, reference, bounds, &round_factor);
            expected = append_factor_logs(expected, round_factor);
        }
        ASSERT_EQ(columns.size(), 4u);
        ASSERT_EQ(expected.columns.size(), columns.size());
        std::vector<std::vector<apxchol::detail::factor_entry>> reference_entries(4);
        std::vector<apxchol::detail::factor_col> reference_columns;
        for (std::size_t i = 0; i < columns.size(); ++i) {
            const auto& c = expected.columns[i];
            EXPECT_EQ(columns[i].vertex, c.vertex);
            EXPECT_EQ(std::bit_cast<std::uint32_t>(columns[i].diag),
                      std::bit_cast<std::uint32_t>(c.diag));
            EXPECT_EQ(columns[i].entry_count, c.entry_count);
            for (std::size_t j = 0; j < c.entry_count; ++j) {
                const auto& e = expected.entries[c.entry_begin + j];
                reference_entries[i].push_back({e.neighbor, e.value});
            }
            reference_columns.push_back({c.vertex, c.diag, reference_entries[i].data(),
                node_index(reference_entries[i].size())});
        }
        apxchol::factorization reference_factor;
        // This independent reference is consumed by host-array SpTRSV setup,
        // so it needs values, unlike the device-owned completion metadata.
        apxchol::detail::build_csc(reference_factor, reference_columns, 4, nullptr);
        ASSERT_GT(reference_factor.L.nonZeros(), 0u);
        ASSERT_EQ(csc_access::inner(reference_factor.L).size(), reference_factor.L.nonZeros());
        ASSERT_EQ(csc_access::vals(reference_factor.L).size(), reference_factor.L.nonZeros());
        const apxchol::factor_options factor_options{
            .factor_storage = apxchol::factor_storage_type::fp32, .drop_tolerance = 0};
        apxchol::detail::cuda_sptrsv reference_trsv, actual_trsv;
        reference_trsv.setup(reference_factor.L, 4, factor_options);
        auto device_factor = session.finalize_device_factor(reference_factor.perm, 4, factor_options.factor_storage, 0);
        ASSERT_NE(device_factor, nullptr);
        actual_trsv.setup_adopting_device_factor(std::move(*device_factor), factor_options);
        std::ostringstream receipt;
        for (const auto& selected : sequence) {
            receipt << "selected";
            for (auto v : selected) receipt << ':' << v;
            receipt << ';';
        }
        for (std::size_t basis = 0; basis < 4; ++basis) {
            std::vector<double> wanted(4), actual(4);
            wanted[basis] = actual[basis] = 1.0;
            reference_trsv.solve_LLt(wanted.data(), wanted.data());
            actual_trsv.solve_LLt(actual.data(), actual.data());
            EXPECT_EQ(actual, wanted) << "basis=" << basis;
            for (double value : actual) receipt << std::bit_cast<std::uint64_t>(value) << ':';
        }
        RecordProperty("owned_residual_continuation_" + std::to_string(count), receipt.str());
    }
}
#endif


TEST(GpuOwnedPrefixHost, RejectsMalformedHandbackBeforeTraversalOrPublication) {
    apxchol::detail::gpu_owned_prefix_handback good;
    good.residual.vertex_count = 3;
    good.residual.owner_offsets = {0, 0, 1, 2};
    good.residual.incidences = {{1, 2, 1.0}, {2, 1, 1.0}};
    good.residual.active = {0, 1, 1};
    good.residual.excess = {0.0, 0.0, 0.0};
    good.columns = {{0, 1.0f, 0, 2}};
    good.fingerprint = apxchol::detail::fingerprint_gpu_round_shadow_input(good.residual);
    EXPECT_NO_THROW(apxchol::detail::validate_owned_prefix_handback(good, 2));
    for (int fault = 0; fault < 10; ++fault) {
        SCOPED_TRACE(fault);
        auto bad = good;
        switch (fault) {
            case 0: bad.residual.owner_offsets = {0, 3, 1, 2}; break;
            case 1: bad.residual.owner_offsets = {0, SIZE_MAX, 1, 2}; break;
            case 2: bad.residual.active[1] = 2; break;
            case 3: bad.columns.push_back(bad.columns[0]); break;
            case 4: bad.columns.clear(); break;
            case 5: bad.columns[0].entry_begin = 1; break;
            case 6: bad.columns[0].entry_count = 3; break;
            case 7: bad.columns[0].vertex = 3; break;
            case 8: bad.columns[0].diag = std::numeric_limits<float>::quiet_NaN(); break;
            case 9: bad.residual.owner_offsets.clear(); break;
        }
        EXPECT_THROW(apxchol::detail::validate_owned_prefix_handback(bad, 2), std::exception);
    }
}

#if defined(APXCHOL_USE_CUDA) && defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
TEST(GpuOwnedPrefix, GenericGraphStillRejectsUnpairedStoredArcs) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    auto input = make_input(2, std::vector<undirected_edge>{{0, 1, 1.0}}, {});
    input.incidences.pop_back();
    input.owner_offsets.back() = input.incidences.size();
    apxchol::detail::gpu_round_shadow_device_state state;
    EXPECT_THROW(state.initialize_owned_raw_for_test(input), std::invalid_argument);
}

TEST(GpuOwnedPrefix, TwoRoundsPreserveOrderedHandbackAndCpuPeel) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_omp_threads serial(1);
    auto graph = make_owned_order_fixture();
    auto oracle = make_owned_order_fixture();
    const auto initial = fingerprint_graph(graph);
    apxchol::detail::gpu_round_shadow_session session(true);
    auto selector_owner = apxchol::test::initialize_owned_selector(session, graph);
    auto& selector = *selector_owner;
    std::vector<node_index> active(graph.n());
    std::iota(active.begin(), active.end(), node_index{0});
    std::vector<std::vector<node_index>> selections;
    std::vector<gpu_round_shadow_report> reports;
    apxchol::partition_options options;
    options.degree_quantile = 0.3;
    for (int round = 0; round < 2; ++round) {
        selector.prepare(active, options);
        auto selected = selector.select_block_greedy().data;
        ASSERT_FALSE(selected.empty());
        ASSERT_LT(selected.size(), active.size() - 1);
        reports.push_back(session.run_owned_prefix_round(selector.device_selection(), kOwnedSeed));
        session.advance_selector(selector);
        std::erase_if(active, [&](node_index v) {
            return std::binary_search(selected.begin(), selected.end(), v);
        });
        selections.push_back(std::move(selected));
    }
    // The oracle snapshot/check starts after both production-owned rounds.
    EXPECT_EQ(fingerprint_graph(graph), initial);
    EXPECT_EQ(reports[0].state_imports, 1u);
    EXPECT_EQ(reports[0].round_state_upload_bytes, 0u);
    EXPECT_EQ(reports[1].state_imports, 1u);
    EXPECT_EQ(reports[1].state_reuses, 1u);
    EXPECT_EQ(reports[1].round_state_upload_bytes, 0u);
    EXPECT_EQ(reports[0].resident_input_incidences, initial.live_incidences);
    EXPECT_EQ(reports[1].resident_input_incidences, reports[0].live_incidences);
    for (const auto& report : reports)
        EXPECT_EQ(report.input_incidences, report.resident_input_incidences);
    EXPECT_NE(selections[0].size(), selections[1].size());

    std::vector<apxchol::detail::factor_col> columns;
    observe_owned_prefix(session, graph, active, columns);
    const std::size_t prefix = columns.size();
    ASSERT_EQ(prefix, selections[0].size() + selections[1].size());
    ASSERT_GT(active.size(), 0u);
    EXPECT_EQ(graph.num_active(), active.size());
    EXPECT_FALSE(graph.is_active(63)); EXPECT_FALSE(graph.is_active(64));
    EXPECT_EQ(graph.excess(63), 2.0); EXPECT_EQ(graph.excess(64), 3.0);

    // Independent CPU/reference work begins only AFTER the GPU-owned window.
    apxchol::detail::gpu_round_shadow_factor_log expected_prefix;
    for (std::size_t round = 0; round < 2; ++round) {
        auto input = apxchol::detail::make_gpu_round_shadow_input(oracle, selections[round], kOwnedSeed);
        std::vector<gpu_round_shadow_excess_bound> bounds;
        const auto expected = apxchol::detail::reference_gpu_round_shadow(input, &bounds);
        EXPECT_TRUE(reports[round].pivots.empty());
        EXPECT_EQ(reports[round].active, expected.active);
        EXPECT_EQ(reports[round].raw_fill_edges, expected.raw_fill_edges);
        EXPECT_EQ(reports[round].unique_neighbors, expected.unique_neighbors);
        EXPECT_EQ(reports[round].live_incidences, expected.live_incidences);
        apxchol::detail::gpu_round_shadow_factor_log factor;
        run_serial_cpu_round(oracle, input, expected, bounds, &factor);
        expected_prefix = append_factor_logs(expected_prefix, factor);
    }
    EXPECT_EQ(fingerprint_graph(graph), fingerprint_graph(oracle));

    ASSERT_EQ(columns.size(), expected_prefix.columns.size());
    for (std::size_t i = 0; i < columns.size(); ++i) {
        EXPECT_EQ(columns[i].vertex, expected_prefix.columns[i].vertex);
        EXPECT_EQ(std::bit_cast<std::uint32_t>(columns[i].diag),
                  std::bit_cast<std::uint32_t>(expected_prefix.columns[i].diag));
        EXPECT_EQ(columns[i].entry_count, expected_prefix.columns[i].entry_count);
        EXPECT_EQ(columns[i].entries, nullptr);
    }
    // Unequal duplicate arcs remain individually stored in their owner order.
    EXPECT_EQ(graph.adj_count(0), oracle.adj_count(0));
    EXPECT_GE(std::count_if(graph.neighbors(0).begin(), graph.neighbors(0).end(),
                           [](const auto& e) { return e.to == 1; }), 3);

    apxchol::detail::factorize_workspace actual_ws, expected_ws;
    for (auto* ws : {&actual_ws, &expected_ws}) {
        ws->threads.resize(1);
        ws->threads[0].factor_entries = std::make_unique<std::pmr::monotonic_buffer_resource>();
    }
    apxchol::factor_options tail_options;
    tail_options.seed = kOwnedSeed;
    std::vector<apxchol::detail::factor_col> expected_tail;
    auto oracle_active = active;
    apxchol::detail::eliminate_remaining(apxchol::detail::tree_elimination{}, graph,
        active, columns, actual_ws, tail_options);
    apxchol::detail::eliminate_remaining(apxchol::detail::tree_elimination{}, oracle,
        oracle_active, expected_tail, expected_ws, tail_options);
    ASSERT_EQ(columns.size(), kOwnedVertices);
    ASSERT_EQ(expected_tail.size(), kOwnedVertices - prefix);
    expect_factor_logs_equal(materialize_factor_log(expected_tail),
        materialize_factor_log(std::span<const apxchol::detail::factor_col>(columns).subspan(prefix)));
    std::vector<bool> seen(kOwnedVertices);
    for (const auto& c : columns) { ASSERT_LT(c.vertex, kOwnedVertices); EXPECT_FALSE(seen[c.vertex]); seen[c.vertex] = true; }
    EXPECT_THROW(session.finish(), std::logic_error);
}

TEST(GpuOwnedPrefix, SharedResidualViewPreservesCandidatesAndIndependentProgress) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_omp_threads serial(1);
      for (bool degree_tiebreak : {false, true}) {
       SCOPED_TRACE(degree_tiebreak);
       for (double quantile : {0.0, 0.3}) {
        SCOPED_TRACE(quantile);
        owned_selector_view_fixture f;
        f.options.degree_quantile = quantile;
        f.degree_tiebreak = degree_tiebreak;
        auto oracle = make_owned_order_fixture();
        for (std::size_t round = 0; round < 3; ++round) {
            SCOPED_TRACE(round);
            const auto selected = f.select();
            std::string receipt = std::to_string(f.selector.selected_degree_work());
            for (const auto v : selected) receipt += ":" + std::to_string(v);
            RecordProperty(std::string("owned_selection_") +
                std::to_string(degree_tiebreak) + "_" + std::to_string(quantile) + "_" +
                std::to_string(round), receipt);
            const auto report = f.run_selected(selected, f.selector.device_selection());
            ASSERT_FALSE(f.active.empty());
            f.session->advance_selector(f.selector);

            // The reference consumes its own CPU graph and never supplies the
            // owning numerical state or the borrowed selector with an upload.
            const auto input = apxchol::detail::make_gpu_round_shadow_input(oracle, selected, kOwnedSeed);
            std::vector<gpu_round_shadow_excess_bound> bounds;
            const auto expected = apxchol::detail::reference_gpu_round_shadow(input, &bounds);
            run_serial_cpu_round(oracle, input, expected, bounds);
            EXPECT_EQ(report.active_count, oracle.num_active());
            f.compare_reference(oracle);
            const auto traffic = f.selector.transfers();
            EXPECT_EQ(traffic.owned_residual_binds, round + 2);
            if (round) {
                EXPECT_EQ(report.round_state_upload_bytes, 0u);
            }
        }
       }
      }
}

TEST(GpuOwnedPrefix, SharedResidualViewRejectsPublicReadsAfterNextRound) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_omp_threads serial(1);
    // A rejected operation poisons its selector, so every public entry point
    // gets a fresh owner/selector pair and must reject the expired view first.
    for (int operation = 0; operation < 5; ++operation) {
        SCOPED_TRACE(operation);
        owned_selector_view_fixture f;
        f.first_bound_round();
        const auto selected = f.select();
        const auto capability = f.selector.device_selection();
        (void)f.selector.host_candidates();
        (void)f.selector.host_active_degrees();
        f.run_selected(selected, capability);
        ASSERT_FALSE(f.active.empty());
        expect_exception_contains([&] {
            switch (operation) {
            case 0: (void)f.selector.prepare(f.active, f.options, 42, 0, f.degree_tiebreak); break;
            case 1: (void)f.selector.select_block_greedy(); break;
            case 2: (void)f.selector.host_candidates(); break;
            case 3: (void)f.selector.host_active_degrees(); break;
            case 4: (void)f.selector.device_selection(); break;
            }
        }, "GPU selector residual generation is retired");
        EXPECT_THROW(f.session->advance_selector(f.selector), std::exception);
    }
}

TEST(GpuOwnedPrefix, SharedResidualViewRebindsAfterNextRound) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_omp_threads serial(1);
    owned_selector_view_fixture f;
    f.first_bound_round();
    const auto selected = f.select();
    f.run_selected(selected, f.selector.device_selection());
    ASSERT_FALSE(f.active.empty());
    // The handoff validates the consumed selection without dereferencing the
    // old residual view, which the numerical compute has already revoked.
    ASSERT_NO_THROW(f.session->advance_selector(f.selector));
    ASSERT_NO_THROW((void)f.select());
    EXPECT_NO_THROW((void)f.selector.device_selection());
    EXPECT_EQ(f.selector.transfers().owned_residual_binds, 3u);
}

TEST(GpuOwnedPrefix, SharedResidualViewExpiresWithDestroyedOrReplacedOwner) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_omp_threads serial(1);
    for (bool replace_owner : {false, true}) {
        SCOPED_TRACE(replace_owner);
        owned_selector_view_fixture f;
        f.first_bound_round();
        (void)f.select();
        if (replace_owner)
            *f.session = apxchol::detail::gpu_round_shadow_session(true);
        else
            f.session.reset();
        expect_exception_contains([&] { (void)f.selector.host_active_degrees(); },
                                  "GPU selector residual generation is retired");
        EXPECT_THROW((void)f.selector.device_selection(), std::exception);
    }
}

TEST(GpuOwnedPrefix, FailedNextRoundRevokesSharedResidualView) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_omp_threads serial(1);
    owned_selector_view_fixture f;
    const auto selected = f.select();
    const auto consumed = f.selector.device_selection();
    f.run_selected(selected, consumed);
    f.session->advance_selector(f.selector);
    (void)f.select();
    EXPECT_THROW(f.session->run_owned_prefix_round(consumed, kOwnedSeed), std::exception);
    expect_exception_contains([&] { (void)f.selector.prepare(f.active, f.options); },
                              "GPU selector residual generation is retired");
}

TEST(GpuOwnedPrefix, CompletionRevokesSharedResidualView) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_omp_threads serial(1);
    owned_selector_view_fixture f;
    while (!f.active.empty()) {
        const auto selected = f.select();
        f.run_selected(selected, f.selector.device_selection());
        f.session->advance_selector(f.selector);
    }
    std::vector<apxchol::detail::factor_col> columns;
    ASSERT_NO_THROW(f.session->complete_owned_factorization(columns));
    ASSERT_FALSE(columns.empty());
    expect_exception_contains([&] { (void)f.selector.host_candidates(); },
                              "GPU selector residual generation is retired");
    EXPECT_THROW((void)f.selector.device_selection(), std::exception);
}
#endif

TEST(GpuOwnedMetadataHost, MatchesOriginalConstructionIncludingEmptyAndIrregularColumns) {
    using apxchol::detail::gpu_round_shadow_factor_column;
    const std::vector<std::vector<gpu_round_shadow_factor_column>> fixtures{
        {}, {{0, 1.0f, 0, 0}},
        {{4, 1.25f, 0, 4}, {0, 2.5f, 4, 0}, {5, 0.5f, 4, 2},
         {2, 3.0f, 6, 0}, {1, 4.0f, 6, 1}, {3, 0.25f, 7, 0}},
        {{3, 2.0f, 0, 0}, {1, 1.0f, 0, 0}, {0, 4.0f, 0, 0}, {2, 8.0f, 0, 0}}
    };
    for (const auto& headers : fixtures) {
        SCOPED_TRACE(headers.size());
        std::vector<apxchol::detail::factor_col> old_columns;
        std::size_t entries = 0;
        for (const auto& c : headers) {
            old_columns.push_back({c.vertex, c.diag, nullptr, c.entry_count});
            entries += c.entry_count;
        }
        apxchol::factorization expected, actual;
        const auto n = static_cast<node_index>(headers.size());
        apxchol::detail::build_csc(expected, old_columns, n, nullptr, false);
        actual.perm.resize(n); csc_access::resize(actual.L, n);
        apxchol::detail::validate_owned_factor_columns(headers, entries, actual.perm, csc_access::outer(actual.L));
        EXPECT_EQ(actual.perm, expected.perm);
        EXPECT_EQ(csc_access::outer(actual.L), csc_access::outer(expected.L));
        EXPECT_EQ(actual.L.rows(), expected.L.rows());
        EXPECT_EQ(actual.L.nonZeros(), expected.L.nonZeros());
        EXPECT_TRUE(csc_access::inner(actual.L).empty()); EXPECT_TRUE(csc_access::vals(actual.L).empty());
    }
}

TEST(GpuOwnedMetadataHost, PreservesCoverageAndFiniteChecks) {
    using column = apxchol::detail::gpu_round_shadow_factor_column;
    const std::vector<column> valid{{2, 1.0f, 0, 2}, {0, 2.0f, 2, 0}, {1, 4.0f, 2, 0}};
    std::vector<std::vector<column>> bad(9, valid);
    bad[0][0].vertex = 3;
    bad[1][1].vertex = 2;
    bad[2][0].diag = 0;
    bad[3][0].diag = -1;
    bad[4][0].diag = std::numeric_limits<float>::infinity();
    bad[5][0].diag = std::numeric_limits<float>::quiet_NaN();
    bad[6][0].entry_begin = 1;
    bad[7][1].entry_begin = 1;
    bad[8][0].entry_count = 3;
    for (std::size_t i = 0; i < bad.size(); ++i) {
        SCOPED_TRACE(i);
        std::vector<node_index> permutation(3);
        std::vector<apxchol::edge_index> offsets(4);
        EXPECT_THROW(apxchol::detail::validate_owned_factor_columns(
            bad[i], 2, permutation, offsets), std::logic_error);
        EXPECT_THROW(apxchol::detail::validate_owned_factor_columns(bad[i], 2), std::logic_error);
    }
    std::vector<node_index> permutation(3);
    std::vector<apxchol::edge_index> offsets(4);
    EXPECT_THROW(apxchol::detail::validate_owned_factor_columns(valid, 3, permutation, offsets), std::logic_error);
    EXPECT_THROW(apxchol::detail::validate_owned_factor_columns(valid, 2, {}, offsets), std::invalid_argument);
    EXPECT_THROW(apxchol::detail::validate_owned_factor_columns(valid, 2, permutation, {}), std::invalid_argument);
    EXPECT_NO_THROW(apxchol::detail::validate_owned_factor_columns(valid, 2, permutation, offsets));
}

TEST(GpuOwnedMetadataHost, RawOffsetCapacityMatchesOriginalConstruction) {
    if constexpr (sizeof(apxchol::edge_index) == sizeof(std::uint32_t)) {
        const auto maximum = std::numeric_limits<std::uint32_t>::max();
        using column = apxchol::detail::gpu_round_shadow_factor_column;
        std::vector<column> headers{{1, 1.0f, 0, maximum - 2}, {0, 1.0f, maximum - 2, 0}};
        std::vector<apxchol::detail::factor_col> old_columns{
            {1, 1.0f, nullptr, maximum - 2}, {0, 1.0f, nullptr, 0}};
        apxchol::factorization expected;
        apxchol::detail::build_csc(expected, old_columns, 2, nullptr, false);
        std::vector<node_index> permutation(2);
        std::vector<apxchol::edge_index> offsets(3);
        apxchol::detail::validate_owned_factor_columns(headers, maximum - 2, permutation, offsets);
        EXPECT_EQ(permutation, expected.perm); EXPECT_EQ(offsets, csc_access::outer(expected.L));
        EXPECT_EQ(offsets.back(), maximum);
        // Only two headers are allocated: the large count is metadata, never
        // a request for factor-entry storage.
        headers[0].entry_count = maximum - 1; headers[1].entry_begin = maximum - 1;
        old_columns[0].entry_count = maximum - 1;
        EXPECT_THROW(apxchol::detail::validate_owned_factor_columns(
            headers, maximum - 1, permutation, offsets), std::overflow_error);
        EXPECT_THROW(apxchol::detail::build_csc(
            expected, old_columns, 2, nullptr, false), std::overflow_error);
        EXPECT_THROW(apxchol::detail::build_csc(
            expected, old_columns, 2, nullptr, true), std::overflow_error);
    } else {
        GTEST_SKIP() << "32-bit edge capacity boundary; no giant allocation for 64-bit offsets";
    }
}

#if defined(APXCHOL_USE_CUDA) && defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
TEST(GpuOwnedPrefix, CompleteMetadataMatchesHeaderConstruction) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_omp_threads serial(1);
    apxchol::factorization expected, actual;
    for (bool direct : {false, true}) {
        auto graph = make_owned_order_fixture();
        apxchol::detail::gpu_round_shadow_session session(true);
        auto selector_owner = apxchol::test::initialize_owned_selector(session, graph);
        auto& selector = *selector_owner;
        std::vector<node_index> active(graph.n());
        std::iota(active.begin(), active.end(), node_index{0});
        while (!active.empty()) {
            selector.prepare(active, {});
            const auto selected = selector.select_block_greedy().data;
            ASSERT_FALSE(selected.empty());
            session.run_owned_prefix_round(selector.device_selection(), kOwnedSeed);
            std::erase_if(active, [&](node_index v) {
                return std::binary_search(selected.begin(), selected.end(), v);
            });
            if (!active.empty()) session.advance_selector(selector);
        }
        std::vector<apxchol::detail::factor_col> columns;
        if (direct) {
            session.complete_owned_factorization(actual.perm, actual.L);
            EXPECT_EQ(actual.perm, expected.perm);
            EXPECT_EQ(csc_access::outer(actual.L), csc_access::outer(expected.L));
            EXPECT_TRUE(csc_access::inner(actual.L).empty()); EXPECT_TRUE(csc_access::vals(actual.L).empty());
            // Empty host columns represent a full device log, not a CPU tail.
            EXPECT_NO_THROW(session.finalize_device_factor(actual.perm, graph.n()));
            EXPECT_THROW(session.complete_owned_factorization(actual.perm, actual.L), std::logic_error);
        } else {
            session.complete_owned_factorization(columns);
            apxchol::detail::build_csc(expected, columns, graph.n(), nullptr, false);
        }
        EXPECT_NO_THROW(session.finish());
    }
}

TEST(GpuOwnedPrefix, ContinuousRoundsMatchIndependentReferenceThroughEmptyResidual) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_omp_threads serial(1);
    auto graph = make_owned_order_fixture();
    auto oracle = make_owned_order_fixture();
    const auto initial = fingerprint_graph(graph);
    apxchol::detail::gpu_round_shadow_session session(true, apxchol::clique_sampler::gks, true);
    auto selector_owner = apxchol::test::initialize_owned_selector(session, graph);
    auto& selector = *selector_owner;
    std::vector<node_index> active(graph.n());
    std::iota(active.begin(), active.end(), node_index{0});
    std::vector<std::vector<node_index>> selections;
    std::vector<gpu_round_shadow_report> reports;
    while (!active.empty()) {
        selector.prepare(active, {});
        const auto selected = selector.select_block_greedy().data;
        ASSERT_FALSE(selected.empty());
        reports.push_back(session.run_owned_prefix_round(selector.device_selection(), kOwnedSeed));
        selections.push_back(selected);
        std::erase_if(active, [&](node_index v) {
            return std::binary_search(selected.begin(), selected.end(), v);
        });
        ASSERT_EQ(reports.back().active_count, active.size());
        if (!active.empty()) session.advance_selector(selector);
        ASSERT_LE(reports.size(), kOwnedVertices);
    }
    ASSERT_GT(reports.size(), 2u);
    EXPECT_EQ(fingerprint_graph(graph), initial); // No CPU numerical replay.
    EXPECT_EQ(reports.back().active_count, 0u);
    EXPECT_EQ(reports.back().live_incidences, 0u);
    EXPECT_EQ(reports.back().state_imports, 1u);
    EXPECT_EQ(reports.back().state_reuses, reports.size() - 1);
    std::vector<apxchol::detail::factor_col> columns;
    session.complete_owned_factorization(columns);
    ASSERT_EQ(columns.size(), kOwnedVertices);
    apxchol::detail::gpu_round_shadow_factor_log expected;
    // Independent numerical work starts only after the complete device run.
    for (std::size_t r = 0; r < reports.size(); ++r) {
        auto input = apxchol::detail::make_gpu_round_shadow_input(oracle, selections[r], kOwnedSeed);
        std::vector<gpu_round_shadow_excess_bound> bounds;
        const auto reference = apxchol::detail::reference_gpu_round_shadow(input, &bounds);
        EXPECT_TRUE(reports[r].pivots.empty());
        EXPECT_EQ(reports[r].factor, apxchol::detail::gpu_round_shadow_digest{});
        EXPECT_EQ(reports[r].fill, apxchol::detail::gpu_round_shadow_digest{});
        EXPECT_EQ(reports[r].residual, apxchol::detail::gpu_round_shadow_digest{});
        EXPECT_EQ(reports[r].ordered_residual, apxchol::detail::gpu_round_shadow_digest{});
        EXPECT_EQ(reports[r].live_degree, apxchol::detail::gpu_round_shadow_digest{});
        EXPECT_EQ(reports[r].canonical_excess, apxchol::detail::gpu_round_shadow_digest{});
        EXPECT_EQ(reports[r].active, reference.active);
        EXPECT_EQ(reports[r].raw_fill_edges, reference.raw_fill_edges);
        EXPECT_EQ(reports[r].unique_neighbors, reference.unique_neighbors);
        EXPECT_EQ(reports[r].live_incidences, reference.live_incidences);
        if (r) {
            EXPECT_EQ(reports[r].round_state_upload_bytes, 0u);
            EXPECT_EQ(reports[r].resident_input_incidences, reports[r-1].live_incidences);
        }
        apxchol::detail::gpu_round_shadow_factor_log round_factor;
        run_serial_cpu_round(oracle, input, reference, bounds, &round_factor);
        expected = append_factor_logs(expected, round_factor);
    }
    ASSERT_EQ(expected.columns.size(), columns.size());
    for (std::size_t i = 0; i < columns.size(); ++i) {
        EXPECT_EQ(columns[i].vertex, expected.columns[i].vertex);
        EXPECT_EQ(std::bit_cast<std::uint32_t>(columns[i].diag),
                  std::bit_cast<std::uint32_t>(expected.columns[i].diag));
        EXPECT_EQ(columns[i].entry_count, expected.columns[i].entry_count);
        EXPECT_EQ(columns[i].entries, nullptr);
    }
    // Compare the complete GPU factor's action against independently built
    // CPU factor columns. This checks all entries after audit-only per-round
    // hashes have been removed, including the isolated tail and duplicate arcs.
    std::vector<std::vector<apxchol::detail::factor_entry>> reference_entries(columns.size());
    std::vector<apxchol::detail::factor_col> reference_columns;
    for (std::size_t i = 0; i < columns.size(); ++i) {
        const auto& c = expected.columns[i];
        for (std::size_t j = 0; j < c.entry_count; ++j) {
            const auto& e = expected.entries[c.entry_begin + j];
            reference_entries[i].push_back({e.neighbor, e.value});
        }
        reference_columns.push_back({c.vertex, c.diag, reference_entries[i].data(),
            node_index(reference_entries[i].size())});
    }
    apxchol::factorization reference_factor;
    apxchol::detail::build_csc(reference_factor, reference_columns, kOwnedVertices, nullptr);
    const apxchol::factor_options factor_options{
        .factor_storage = apxchol::factor_storage_type::fp32, .drop_tolerance = 0};
    apxchol::detail::cuda_sptrsv expected_trsv, owned_trsv;
    expected_trsv.setup(reference_factor.L, kOwnedVertices, factor_options);
    auto device_factor = session.finalize_device_factor(reference_factor.perm, kOwnedVertices, factor_options.factor_storage, 0);
    owned_trsv.setup_adopting_device_factor(std::move(*device_factor), factor_options);
    // Basis vectors exercise the full inverse operator, not just one RHS.
    for (std::size_t basis = 0; basis < kOwnedVertices; ++basis) {
        std::vector<double> expected_x(kOwnedVertices), owned_x(kOwnedVertices);
        expected_x[basis] = owned_x[basis] = 1.0;
        expected_trsv.solve_LLt(expected_x.data(), expected_x.data());
        owned_trsv.solve_LLt(owned_x.data(), owned_x.data());
        EXPECT_EQ(owned_x, expected_x) << "basis=" << basis;
    }
    testing::internal::CaptureStderr(); session.finish();
    const auto trace = testing::internal::GetCapturedStderr();
    EXPECT_NE(trace.find("[gpu-owned-factorization] complete"), std::string::npos);
    EXPECT_THROW(session.complete_owned_factorization(columns), std::logic_error);
}

TEST(GpuOwnedPrefix, DegreeScratchLifetimeAcrossMovesCompletionAndNewSizes) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_omp_threads serial(1);
    apxchol::detail::gpu_round_shadow_session session(true);
    const node_index sizes[] = {
        1, 2, 3, 4, 31, 32, 33, 63, 64, 65, 66, 127, 128, 129, 257, 1};
    for (std::size_t fixture = 0; fixture < std::size(sizes); ++fixture) {
        const node_index n = sizes[fixture];
        SCOPED_TRACE(n);
        // Move assignment retires the preceding completed owner's allocations,
        // then binds an independently sized owner, including a final shrink.
        session = apxchol::detail::gpu_round_shadow_session(true);
        auto make_graph = [&] {
            apxchol::detail::graph g(n);
            // Leading isolates leave the highest owner in a triangle, so
            // 2^k+1 domains really exercise the newly required owner bit.
            for (node_index v = n % 3; v + 2 < n; v += 3) {
                g.add_edge(v, v + 1, 1.0);
                g.add_edge(v, v + 2, 1.0);
                g.add_edge(v + 1, v + 2, 1.0);
            }
            for (node_index v = 0; v < n; ++v) g.excess(v) = 0.25;
            return g;
        };
        auto graph = make_graph(), oracle = make_graph();
        auto selector_owner = apxchol::test::initialize_owned_selector(session, graph);
        auto& selector = *selector_owner;
        std::vector<node_index> active(n);
        std::iota(active.begin(), active.end(), node_index{0});
        apxchol::partition_options options;
        options.degree_quantile = 0.0;
        options.degree_multiplier = 100.0;
        std::vector<gpu_round_shadow_report> reports;
        apxchol::detail::gpu_round_shadow_factor_log expected_log;
        std::string receipt;
        while (!active.empty()) {
            selector.prepare(active, options);
            const auto degree_view = selector.host_active_degrees();
            ASSERT_EQ(degree_view.size(), active.size());
            for (std::size_t i = 0; i < active.size(); ++i) {
                std::size_t expected_degree = 0;
                for (const auto& edge : oracle.neighbors(active[i]))
                    expected_degree += oracle.is_active(edge.to);
                EXPECT_EQ(degree_view[i], expected_degree);
                receipt += ":d" + std::to_string(degree_view[i]);
            }
            const auto selected = selector.select_block_greedy().data;
            ASSERT_FALSE(selected.empty());
            const auto report = session.run_owned_prefix_round(selector.device_selection(), kOwnedSeed);
            reports.push_back(report);
            receipt += ":r" + std::to_string(report.live_incidences);
            for (const auto v : selected) receipt += ":v" + std::to_string(v);
            const auto input = apxchol::detail::make_gpu_round_shadow_input(
                oracle, selected, kOwnedSeed);
            std::vector<gpu_round_shadow_excess_bound> bounds;
            const auto expected = apxchol::detail::reference_gpu_round_shadow(input, &bounds);
            apxchol::detail::gpu_round_shadow_factor_log round_log;
            run_serial_cpu_round(oracle, input, expected, bounds, &round_log);
            expected_log = append_factor_logs(expected_log, round_log);
            EXPECT_EQ(report.active, expected.active);
            EXPECT_EQ(report.live_incidences, expected.live_incidences);
            EXPECT_EQ(report.raw_fill_edges, expected.raw_fill_edges);
            std::erase_if(active, [&](node_index v) {
                return std::binary_search(selected.begin(), selected.end(), v);
            });
            if (reports.size() > 1) {
                EXPECT_EQ(report.round_state_upload_bytes, 0u);
                EXPECT_EQ(report.state_buffer_allocations, reports.front().state_buffer_allocations);
                EXPECT_EQ(report.state_buffer_growths, reports.front().state_buffer_growths);
            }
            if (!active.empty()) {
                session.advance_selector(selector);
                // Both move construction and assignment preserve the tracker
                // address and the selector's already published borrowed view.
                auto moved = std::move(session);
                session = std::move(moved);
            }
            ASSERT_LE(reports.size(), 3u);
        }
        if (n >= 3) {
            ASSERT_EQ(reports.size(), 3u);
            EXPECT_GT(reports.front().live_incidences, 0u);
            EXPECT_EQ(reports[1].live_incidences, 0u);
            EXPECT_GT(reports[1].active_count, 0u);
        }
        EXPECT_EQ(reports.back().active_count, 0u);
        std::vector<apxchol::detail::factor_col> columns;
        session.complete_owned_factorization(columns);
        ASSERT_EQ(columns.size(), expected_log.columns.size());
        for (std::size_t i = 0; i < columns.size(); ++i) {
            const auto& actual = columns[i]; const auto& expected = expected_log.columns[i];
            EXPECT_EQ(actual.vertex, expected.vertex);
            EXPECT_EQ(actual.entry_count, expected.entry_count);
            EXPECT_EQ(std::bit_cast<std::uint32_t>(actual.diag), std::bit_cast<std::uint32_t>(expected.diag));
            receipt += ":c" + std::to_string(actual.vertex) + ":" + std::to_string(actual.entry_count)
                + ":" + std::to_string(std::bit_cast<std::uint32_t>(actual.diag));
        }
        RecordProperty("degree_lifetime_case_" + std::to_string(fixture), receipt);
    }
}

TEST(GpuOwnedPrefix, DegreeScratchHandbackPreservesEveryOrderedField) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_omp_threads serial(1);
    for (int rounds : {1, 2, 3}) {
        SCOPED_TRACE(rounds);
        owned_selector_view_fixture f;
        auto oracle = make_owned_order_fixture();
        std::size_t final_live_incidences = 0;
        for (int round = 0; round < rounds; ++round) {
            const auto selected = f.select();
            final_live_incidences = f.run_selected(
                selected, f.selector.device_selection()).live_incidences;
            const auto input = apxchol::detail::make_gpu_round_shadow_input(oracle, selected, kOwnedSeed);
            std::vector<gpu_round_shadow_excess_bound> bounds;
            const auto expected = apxchol::detail::reference_gpu_round_shadow(input, &bounds);
            run_serial_cpu_round(oracle, input, expected, bounds);
            if (round + 1 < rounds) f.session->advance_selector(f.selector);
        }
        std::vector<apxchol::detail::factor_col> columns;
        observe_owned_prefix(*f.session, f.graph, f.active, columns);
        const auto actual = apxchol::detail::make_gpu_round_shadow_input(f.graph, {}, kOwnedSeed);
        const auto stored_cpu = apxchol::detail::make_gpu_round_shadow_input(oracle, {}, kOwnedSeed);
        // CPU slabs can retain arcs to inactive vertices. The owned residual
        // physically contains only active-both incidences; restrict the CPU
        // oracle stably, without changing or filtering the actual handback.
        const auto expected = compact_live_snapshot(stored_cpu);
        if (rounds == 1) {
            EXPECT_GT(stored_cpu.incidences.size(), expected.incidences.size());
        }
        ASSERT_EQ(actual.incidences.size(), final_live_incidences);
        ASSERT_EQ(expected.incidences.size(), final_live_incidences);
        EXPECT_EQ(actual.owner_offsets, expected.owner_offsets);
        EXPECT_EQ(actual.active, expected.active);
        ASSERT_EQ(actual.incidences.size(), expected.incidences.size());
        std::string receipt;
        for (const auto offset : actual.owner_offsets) receipt += ":o" + std::to_string(offset);
        for (std::size_t i = 0; i < actual.active.size(); ++i) {
            EXPECT_EQ(std::bit_cast<std::uint64_t>(actual.excess[i]), std::bit_cast<std::uint64_t>(expected.excess[i]));
            receipt += ":a" + std::to_string(actual.active[i]) + ":x"
                + std::to_string(std::bit_cast<std::uint64_t>(actual.excess[i]));
        }
        for (std::size_t i = 0; i < actual.incidences.size(); ++i) {
            const auto& a = actual.incidences[i]; const auto& e = expected.incidences[i];
            EXPECT_EQ(a.owner, e.owner); EXPECT_EQ(a.neighbor, e.neighbor);
            EXPECT_EQ(std::bit_cast<std::uint64_t>(double(a.weight)), std::bit_cast<std::uint64_t>(double(e.weight)));
            receipt += ":e" + std::to_string(a.owner) + ":" + std::to_string(a.neighbor)
                + ":" + std::to_string(std::bit_cast<std::uint64_t>(double(a.weight)));
        }

        RecordProperty("degree_handback_rounds_" + std::to_string(rounds), receipt);
    }
}

TEST(GpuOwnedPrefix, FactorEntryWritesMatchSerialFactorOnEveryBasis) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_omp_threads serial(1);
    constexpr node_index degrees[] = {0, 1, 2, 31, 32, 33, 63, 64, 65, 255, 256, 257};
    constexpr double weights[] = {0.0, -0.0, 0x1p-20, 0.25, 1.0, 0x1p20, 0x1p52};
    for (node_index count : {1u, 33u, 257u}) {
        SCOPED_TRACE(count);
        // Mixed short/long columns share a warp. Put a long column in the last
        // partial warp/block too; remaining pivots are cheap isolated columns.
        std::vector<undirected_edge> edges;
        node_index n = count;
        for (node_index pivot = 0; pivot < count; ++pivot) {
            const node_index degree = pivot == count - 1 ? 257 :
                pivot < std::size(degrees) ? degrees[pivot] : pivot == 31 ? 65 : 0;
            for (node_index j = 0; j < degree; ++j) {
                const double weight = weights[j % std::size(weights)];
                edges.push_back({pivot, n, weight * 0.75});
                edges.push_back({pivot, n, weight * 0.25});
                ++n;
            }
        }
        auto make_graph = [&] {
            apxchol::detail::graph graph(n);
            for (const auto& edge : edges) graph.add_edge(edge.u, edge.v, edge.weight);
            for (node_index v = 0; v < n; ++v) graph.excess(v) = 0.125;
            return graph;
        };
        auto graph = make_graph(), oracle = make_graph();
        apxchol::detail::gpu_round_shadow_session session(true);
        auto selector_owner = apxchol::test::initialize_owned_selector(session, graph);
        auto& selector = *selector_owner;
        std::vector<node_index> active(n), wanted(count);
        std::iota(active.begin(), active.end(), node_index{0});
        std::iota(wanted.begin(), wanted.end(), node_index{0});
        apxchol::partition_options options;
        options.degree_quantile = 0.0; options.degree_multiplier = 1024.0;
        selector.prepare(active, options, 42, 0, false);
        const auto selected = selector.select_for_test(wanted).data;
        ASSERT_EQ(selected, wanted);
        const auto input = apxchol::detail::make_gpu_round_shadow_input(graph, selected, kOwnedSeed);
        const auto report = session.run_owned_prefix_round(selector.device_selection(), kOwnedSeed);
        ASSERT_TRUE(report.gpu_executed); EXPECT_TRUE(report.pivots.empty());
        std::vector<apxchol::detail::factor_col> actual_columns;
        observe_owned_prefix(session, graph, active, actual_columns);
        const auto expected_log = run_owned_host_oracle(oracle, input);
        ASSERT_EQ(actual_columns.size(), expected_log.columns.size());

        for (std::size_t i = 0; i < count; ++i) {
            const auto& c = expected_log.columns[i];
            EXPECT_EQ(actual_columns[i].vertex, c.vertex);
            EXPECT_EQ(actual_columns[i].entry_count, c.entry_count);
            EXPECT_EQ(std::bit_cast<std::uint32_t>(actual_columns[i].diag),
                      std::bit_cast<std::uint32_t>(c.diag));
        }
        const auto digest = check_complete_factor_action(
            session, selector, oracle, expected_log, kOwnedSeed);
        // Every-basis equality checks complete factor action, not raw factor bytes.
        RecordProperty("owned_factor_entry_" + std::to_string(count),
            std::to_string(n) + ":" + std::to_string(expected_log.entries.size()) + ":" + std::to_string(digest));
    }
}

TEST(GpuOwnedPrefix, NormalBatchesMatchSerialHandbackAndFactorAction) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_omp_threads serial(1);
    const std::vector<std::vector<std::size_t>> profiles = {
        {127, 128, 129},             // Both dispatch paths in the same round.
        {64, 64, 64, 65},            // Packed sums exactly128 and just above128.
        std::vector<std::size_t>(32, 4), // Full32-pivot tile, exactly128 slots.
        [] { std::vector<std::size_t> s(33, 0); s.back() = 129; return s; }(),
        // Six descriptors cross the four-warps-per-block emit boundary.
        // An oversized pivot interrupts packing; a zero row joins the next batch.
        {65, 65, 129, 0, 65, 65, 65, 65}
    }; // A zero-only tile followed by a partial oversized tile must advance.
    constexpr double adversarial[] = {
        0.0, -0.0, 0x1p-40, 1.0, 0x1p52, 0.25, 0x1p-20, 1.0, 0x1p52
    };
    for (std::size_t profile = 0; profile < profiles.size(); ++profile) {
        for (unsigned mode = 0; mode < 2; ++mode) {
            SCOPED_TRACE("profile=" + std::to_string(profile) +
                         " mode=" + std::to_string(mode));
            const auto& spans = profiles[profile];
            const node_index count = static_cast<node_index>(spans.size());
            const node_index n = count + 3;
            const unsigned seed = mode ? std::numeric_limits<unsigned>::max() : 0;
            auto make_graph = [&] {
                apxchol::detail::graph g(n);
                for (node_index pivot = 0; pivot < count; ++pivot) {
                    for (std::size_t j = 0; j < spans[pivot]; ++j) {
                        // Shared receivers create duplicate fill and excess collisions.
                        // Repeated receiver IDs are deliberately interleaved; summing
                        // by a changed association can alter the adversarial bits.
                        const node_index receiver = count + node_index(j % 3);
                        const double w = mode ? (j < 3 ? 1.0 : j % 2 ? -0.0 : 0.0)
                                              : adversarial[j % std::size(adversarial)];
                        g.add_edge(pivot, receiver, w);
                    }
                    // In mode1 each nonempty star has degree+excess=4 exactly;
                    // its three contributions are dyadic. Shared-target updates
                    // therefore compare bitwise to the serial CPU accumulation.
                    g.excess(pivot) = spans[pivot] == 0 ? 0.125 : mode ? 1.0 : 0.0;
                }
                for (node_index v = count; v < n; ++v) g.excess(v) = 0.25;
                return g;
            };
            auto graph = make_graph(), oracle = make_graph();
            // The frontend borrows owned storage; destroy it before its owner.
            apxchol::detail::gpu_round_shadow_session session(true);
            auto selector_owner = apxchol::test::initialize_owned_selector(session, graph);
            auto& selector = *selector_owner;
            std::vector<node_index> active(n), wanted(count);
            std::iota(active.begin(), active.end(), node_index{0});
            std::iota(wanted.begin(), wanted.end(), node_index{0});
            apxchol::partition_options options;
            options.degree_quantile = 0.0;
            options.degree_multiplier = 1024.0;
            selector.prepare(active, options, 42, 0, false);
            const auto selected = selector.select_for_test(wanted).data;
            ASSERT_EQ(selected, wanted);
            const auto input = apxchol::detail::make_gpu_round_shadow_input(graph, selected, seed);

            apxchol::detail::gpu_round_shadow_normal_batch_counts expected_counts;
            for (std::size_t k = 0; k < spans.size(); ++k) {
                const auto begin = input.owner_offsets[k], end = input.owner_offsets[k+1];
                ASSERT_EQ(end-begin, spans[k]);
                std::vector<node_index> live_neighbors;
                for (std::size_t j = begin; j < end; ++j)
                    if (input.active[input.incidences[j].neighbor])
                        live_neighbors.push_back(input.incidences[j].neighbor);
                if (spans[k] <= 128) {
                    ++expected_counts.normal_pivots;
                    expected_counts.normal_spans += spans[k];
                    expected_counts.normal_gathered += live_neighbors.size();
                    std::sort(live_neighbors.begin(), live_neighbors.end());
                    expected_counts.normal_unique += std::unique(
                        live_neighbors.begin(), live_neighbors.end()) - live_neighbors.begin();
                } else {
                    ++expected_counts.oversized_pivots;
                    expected_counts.oversized_spans += spans[k];
                    expected_counts.oversized_gathered += live_neighbors.size();
                }
            }
            for (std::size_t tile = 0; tile < spans.size(); tile += 32) {
                const auto end = std::min(tile + 32, spans.size());
                for (std::size_t k = tile; k < end;) {
                    if (spans[k] > 128) { ++k; continue; }
                    std::size_t sum = 0;
                    do { sum += spans[k++]; }
                    while (k < end && spans[k] <= 128 && sum + spans[k] <= 128);
                    ++expected_counts.batches;
                }
            }
            const auto report = session.run_owned_prefix_round(selector.device_selection(), seed);
            ASSERT_TRUE(report.gpu_executed);
            ASSERT_TRUE(report.pivots.empty());
            const auto& got = report.normal_batch;
            EXPECT_EQ(got.normal_pivots, expected_counts.normal_pivots);
            EXPECT_EQ(got.normal_spans, expected_counts.normal_spans);
            EXPECT_EQ(got.normal_gathered, expected_counts.normal_gathered);
            EXPECT_EQ(got.normal_unique, expected_counts.normal_unique);
            EXPECT_EQ(got.batches, expected_counts.batches);
            EXPECT_EQ(got.oversized_pivots, expected_counts.oversized_pivots);
            EXPECT_EQ(got.oversized_spans, expected_counts.oversized_spans);
            EXPECT_EQ(got.oversized_gathered, expected_counts.oversized_gathered);
            if (profile == 0) {
                EXPECT_EQ(got.normal_pivots, 2u);
                EXPECT_EQ(got.oversized_pivots, 1u);
            }
            if (profile == 1) {
                EXPECT_EQ(got.batches, 3u);
            }
            if (profile == 2) {
                EXPECT_EQ(got.batches, 1u);
            }
            if (profile == 3) {
                EXPECT_EQ(got.normal_pivots, 32u);
                EXPECT_EQ(got.normal_spans, 0u);
                EXPECT_EQ(got.batches, 1u);
                EXPECT_EQ(got.oversized_pivots, 1u);
            }
            if (profile == 4) {
                EXPECT_EQ(got.normal_pivots, 7u);
                EXPECT_EQ(got.normal_spans, 390u);
                EXPECT_EQ(got.normal_gathered, 390u);
                EXPECT_EQ(got.batches, 6u);
                EXPECT_EQ(got.oversized_pivots, 1u);
                EXPECT_EQ(got.oversized_spans, 129u);
            }

            std::vector<apxchol::detail::factor_col> actual_columns;
            observe_owned_prefix(session, graph, active, actual_columns);
            const auto actual = apxchol::detail::make_gpu_round_shadow_input(
                graph, std::span<const node_index>{}, seed);
            // Independent numerical oracle starts after GPU-owned execution.
            std::vector<gpu_round_shadow_excess_bound> bounds;
            const auto reference = apxchol::detail::reference_gpu_round_shadow(input, &bounds);
            const auto expected_log = run_owned_host_oracle(oracle, input);
            const auto expected = compact_live_snapshot(apxchol::detail::make_gpu_round_shadow_input(
                oracle, std::span<const node_index>{}, seed));
            ASSERT_EQ(actual.owner_offsets, expected.owner_offsets);
            ASSERT_EQ(actual.active, expected.active);
            ASSERT_EQ(actual.incidences.size(), expected.incidences.size());
            ASSERT_EQ(actual.excess.size(), expected.excess.size());
            for (std::size_t j = 0; j < actual.incidences.size(); ++j) {
                EXPECT_EQ(actual.incidences[j].owner, expected.incidences[j].owner);
                EXPECT_EQ(actual.incidences[j].neighbor, expected.incidences[j].neighbor);
                EXPECT_EQ(std::bit_cast<std::uint64_t>(double(actual.incidences[j].weight)),
                          std::bit_cast<std::uint64_t>(double(expected.incidences[j].weight)));
            }
            for (std::size_t j = 0; j < actual.excess.size(); ++j)
                EXPECT_EQ(std::bit_cast<std::uint64_t>(actual.excess[j]),
                          std::bit_cast<std::uint64_t>(expected.excess[j]));

            EXPECT_EQ(report.raw_fill_edges, reference.raw_fill_edges);
            EXPECT_EQ(report.unique_neighbors, reference.unique_neighbors);
            EXPECT_EQ(report.excess_updates, reference.excess_updates);
            if (mode) {
                EXPECT_GT(report.excess_updates, 0u);
            }

            ASSERT_EQ(actual_columns.size(), expected_log.columns.size());

            for (std::size_t j = 0; j < count; ++j) {
                const auto& c = expected_log.columns[j];
                EXPECT_EQ(actual_columns[j].vertex, c.vertex);
                EXPECT_EQ(actual_columns[j].entry_count, c.entry_count);
                EXPECT_EQ(std::bit_cast<std::uint32_t>(actual_columns[j].diag),
                          std::bit_cast<std::uint32_t>(c.diag));
            }
            const auto handback_receipt = owned_residual_boundary_receipt(actual, actual_columns);
            const auto digest = check_complete_factor_action(
                session, selector, oracle, expected_log, seed);
            std::ostringstream receipt;
            receipt << got.normal_pivots << ',' << got.normal_spans << ',' << got.normal_gathered
                    << ',' << got.normal_unique << ',' << got.batches << ',' << got.oversized_pivots
                    << ',' << got.oversized_spans << ',' << got.oversized_gathered << ';'
                    << handback_receipt << ";basis_action:" << n << ':' << digest;
            // Every-basis inverse action is checked exactly; this is not a raw
            // factor byte-export claim. Handback fields above are all recorded.
            RecordProperty("normal_batch_"+std::to_string(profile)+"_"+std::to_string(mode),receipt.str());
        }
    }
}

TEST(GpuOwnedPrefix, OversizedImportRejectionAndCompactContinuationMatchOracle) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_omp_threads serial(1);
    constexpr node_index n = 4;
    constexpr unsigned seed = 42;
    auto make_graph = [&] {
        apxchol::detail::graph g(n);
        // The valid fresh owned input has129 live physical slots at pivot0.
        for (unsigned j = 0; j < 129; ++j)
            g.add_edge(0, 1 + j % 3, j < 3 ? 1.0 : j % 2 ? -0.0 : 0.0);
        // Every possible second-round pivot remains oversized after compact
        // publication, independently of the resident selector's tie policy.
        for (node_index u = 1; u < n; ++u)
            for (node_index v = u + 1; v < n; ++v)
                for (unsigned j = 0; j < 65; ++j)
                    g.add_edge(u, v, j == 0 ? 1.0 : j % 2 ? -0.0 : 0.0);
        for (node_index v = 0; v < n; ++v) g.excess(v) = 1.0;
        return g;
    };
    {
        auto inactive = make_graph();
        inactive.deactivate(3);
        apxchol::detail::gpu_round_shadow_session rejected(true);
        EXPECT_THROW(apxchol::test::initialize_owned_selector(rejected, inactive),
                     std::invalid_argument);
    }
    auto graph = make_graph(), oracle = make_graph();
    apxchol::detail::gpu_round_shadow_session session(true);
    auto selector_owner = apxchol::test::initialize_owned_selector(session, graph);
    auto& selector = *selector_owner;
    std::vector<node_index> active{0, 1, 2, 3};
    apxchol::partition_options options;
    options.degree_quantile = 0.0;
    options.degree_multiplier = 1024.0;
    apxchol::detail::gpu_round_shadow_factor_log expected_log;
    std::ostringstream receipt;
    for (unsigned round = 0; round < 2; ++round) {
        SCOPED_TRACE(round);
        selector.prepare(active, options, 42, 0, false);
        const auto selected = round == 0 ? selector.select_for_test(std::vector<node_index>{0}).data
            : selector.select_block_greedy().data;
        ASSERT_EQ(selected.size(), 1u);
        if (round == 0) {
            ASSERT_EQ(selected.front(), 0u);
        }
        auto input = apxchol::detail::make_gpu_round_shadow_input(oracle, selected, seed);
        if (round) input = compact_live_snapshot(input);
        const auto pivot = selected.front();
        const auto span = input.owner_offsets[pivot + 1] - input.owner_offsets[pivot];
        std::size_t live = 0;
        for (std::size_t j = input.owner_offsets[pivot]; j < input.owner_offsets[pivot + 1]; ++j)
            live += input.active[input.incidences[j].neighbor] != 0;
        ASSERT_GT(span, 128u);
        if (round == 0) {
            ASSERT_EQ(span, 129u);
        }
        ASSERT_EQ(span, live);
        const auto report = session.run_owned_prefix_round(selector.device_selection(), seed);
        ASSERT_TRUE(report.gpu_executed);
        EXPECT_EQ(report.normal_batch.oversized_pivots, 1u);
        EXPECT_EQ(report.normal_batch.normal_pivots, 0u);
        EXPECT_EQ(report.normal_batch.oversized_spans, span);
        EXPECT_EQ(report.normal_batch.oversized_gathered, live);
        if (round) {
            EXPECT_EQ(report.round_state_upload_bytes, 0u);
        }
        std::vector<gpu_round_shadow_excess_bound> bounds;
        const auto reference = apxchol::detail::reference_gpu_round_shadow(input, &bounds);
        apxchol::detail::gpu_round_shadow_factor_log round_log;
        run_serial_cpu_round(oracle, input, reference, bounds, &round_log);
        expected_log = append_factor_logs(expected_log, round_log);
        EXPECT_EQ(report.raw_fill_edges, reference.raw_fill_edges);
        EXPECT_EQ(report.excess_updates, reference.excess_updates);
        EXPECT_GT(report.excess_updates, 0u);
        receipt << ";round:" << round << ':' << span << ':' << live
                << ':' << report.excess_updates;
        std::erase_if(active, [&](node_index v) {
            return std::binary_search(selected.begin(), selected.end(), v);
        });
        if (round == 0) session.advance_selector(selector);
    }
    std::vector<apxchol::detail::factor_col> columns;
    observe_owned_prefix(session, graph, active, columns);
    const auto actual = apxchol::detail::make_gpu_round_shadow_input(graph, {}, seed);
    const auto expected = compact_live_snapshot(
        apxchol::detail::make_gpu_round_shadow_input(oracle, {}, seed));
    ASSERT_EQ(actual.owner_offsets, expected.owner_offsets);
    ASSERT_EQ(actual.active, expected.active);
    ASSERT_EQ(actual.incidences.size(), expected.incidences.size());
    ASSERT_EQ(actual.excess.size(), expected.excess.size());
    for (std::size_t j = 0; j < actual.incidences.size(); ++j) {
        EXPECT_EQ(actual.incidences[j].owner, expected.incidences[j].owner);
        EXPECT_EQ(actual.incidences[j].neighbor, expected.incidences[j].neighbor);
        EXPECT_EQ(std::bit_cast<std::uint64_t>(double(actual.incidences[j].weight)),
                  std::bit_cast<std::uint64_t>(double(expected.incidences[j].weight)));
    }
    for (std::size_t j = 0; j < actual.excess.size(); ++j)
        EXPECT_EQ(std::bit_cast<std::uint64_t>(actual.excess[j]),
                  std::bit_cast<std::uint64_t>(expected.excess[j]));

    ASSERT_EQ(columns.size(), expected_log.columns.size());
    for (std::size_t j = 0; j < columns.size(); ++j) {
        EXPECT_EQ(columns[j].vertex, expected_log.columns[j].vertex);
        EXPECT_EQ(columns[j].entry_count, expected_log.columns[j].entry_count);
        EXPECT_EQ(std::bit_cast<std::uint32_t>(columns[j].diag),
                  std::bit_cast<std::uint32_t>(expected_log.columns[j].diag));
    }
    // Exact handback and prefix headers here; the existing NormalBatches
    // fixture independently checks full factor action on every basis.
    receipt << ';' << owned_residual_boundary_receipt(actual, columns);
    RecordProperty("owned_compact_oversized", receipt.str());
}

TEST(GpuOwnedPrefix, ParallelEmissionMatchesSerialOrderedHandback) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_omp_threads serial(1);
    for (node_index degree : {0, 1, 2, 31, 32, 33, 257}) {
        for (unsigned profile = 0; profile < 2; ++profile) {
            SCOPED_TRACE("degree=" + std::to_string(degree) +
                         " profile=" + std::to_string(profile));
            // The CPU factor-options seam accepts an unsigned run seed.
            // Per-pivot SplitMix state and its item offsets remain 64-bit.
            const std::uint64_t seed = profile ? std::numeric_limits<unsigned>::max() : 0;
            auto make_graph = [&] {
                apxchol::detail::graph graph(degree + 1);
                // A prescribed pivot isolates emission for these complete graphs.
                // Duplicate arcs exercise ordered deduplication before emission.
                for (node_index u = 0; u <= degree; ++u)
                    for (node_index v = u + 1; v <= degree; ++v) {
                        double w = 1.0;
                        if (u == 0) {
                            w = profile ? std::ldexp(1.0, (int(v % 7) - 3) * 20)
                                        : (degree == 2 || v % 7 == 0 ? 0.0 : 1.0);
                        }
                        graph.add_edge(u, v, w * 0.75);
                        graph.add_edge(u, v, w * 0.25);
                    }
                return graph;
            };
            auto graph = make_graph();
            auto oracle = make_graph();
            std::vector<node_index> active(graph.n());
            std::iota(active.begin(), active.end(), node_index{0});
            apxchol::detail::gpu_round_shadow_session session(true);
            auto selector_owner = apxchol::test::initialize_owned_selector(session, graph);
            auto& selector = *selector_owner;
            selector.prepare(active, {});
            const auto selected = selector.select_for_test(std::vector<node_index>{0}).data;
            ASSERT_EQ(selected, std::vector<node_index>{0});
            const auto input = apxchol::detail::make_gpu_round_shadow_input(
                oracle, selected, seed);
            std::vector<gpu_round_shadow_excess_bound> bounds;
            const auto reference = apxchol::detail::reference_gpu_round_shadow(input, &bounds);
            const auto expected_factor = run_owned_host_oracle(oracle, input);

            const auto report = session.run_owned_prefix_round(selector.device_selection(), seed);
            EXPECT_TRUE(report.pivots.empty()); // Owned, not audited serial sampler.
            EXPECT_EQ(report.unique_neighbors, static_cast<std::size_t>(degree));
            EXPECT_EQ(report.raw_fill_edges, reference.raw_fill_edges);
            std::vector<apxchol::detail::factor_col> columns;
            observe_owned_prefix(session, graph, active, columns);
            ASSERT_EQ(columns.size(), expected_factor.columns.size());
            for (std::size_t i = 0; i < columns.size(); ++i) {
                EXPECT_EQ(columns[i].vertex, expected_factor.columns[i].vertex);
                EXPECT_EQ(std::bit_cast<std::uint32_t>(columns[i].diag),
                          std::bit_cast<std::uint32_t>(expected_factor.columns[i].diag));
                EXPECT_EQ(columns[i].entry_count, expected_factor.columns[i].entry_count);
                EXPECT_EQ(columns[i].entries, nullptr); // Prefix entries stay on device.
            }
            const auto actual = apxchol::detail::make_gpu_round_shadow_input(graph, {}, seed);
            const auto expected = compact_live_snapshot(
                apxchol::detail::make_gpu_round_shadow_input(oracle, {}, seed));
            EXPECT_EQ(actual.owner_offsets, expected.owner_offsets);
            EXPECT_EQ(actual.active, expected.active);
            EXPECT_EQ(actual.excess, expected.excess);
            ASSERT_EQ(actual.incidences.size(), expected.incidences.size());
            std::uint64_t digest = 14695981039346656037ULL;
            auto fold = [&](std::uint64_t word) { digest = (digest ^ word) * 1099511628211ULL; };
            for (const auto offset : actual.owner_offsets) fold(offset);
            for (std::size_t i = 0; i < actual.incidences.size(); ++i) {
                const auto& a = actual.incidences[i];
                const auto& e = expected.incidences[i];
                EXPECT_EQ(a.owner, e.owner);
                EXPECT_EQ(a.neighbor, e.neighbor);
                EXPECT_EQ(std::bit_cast<std::uint64_t>(double(a.weight)),
                          std::bit_cast<std::uint64_t>(double(e.weight)));
                fold(a.owner); fold(a.neighbor); fold(std::bit_cast<std::uint64_t>(double(a.weight)));
            }
            for (const auto& column : columns) {
                fold(column.vertex); fold(std::bit_cast<std::uint32_t>(column.diag));
                fold(column.entry_count);
            }
            RecordProperty("emission_" + std::to_string(degree) + "_" +
                           std::to_string(profile), std::to_string(digest));
        }
    }
}

TEST(GpuOwnedPrefix, PrematureCompletionPoisonsIncompleteDeviceFactor) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    auto graph = make_owned_order_fixture();
    apxchol::detail::gpu_round_shadow_session session(true);
    auto selector_owner = apxchol::test::initialize_owned_selector(session, graph);
    auto& selector = *selector_owner;
    std::vector<node_index> active(graph.n());
    std::iota(active.begin(), active.end(), node_index{0});
    selector.prepare(active, {});
    (void)selector.select_block_greedy();
    const auto report = session.run_owned_prefix_round(selector.device_selection(), kOwnedSeed);
    ASSERT_GT(report.active_count, 0u);
    std::vector<apxchol::detail::factor_col> columns;
    EXPECT_THROW(session.complete_owned_factorization(columns), std::logic_error);
    EXPECT_TRUE(columns.empty());
}

TEST(GpuOwnedPrefix, StaleSecondSelectionPoisonsBeforeHostPublication) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_omp_threads serial(1);
    auto graph = make_owned_order_fixture();
    const auto initial = fingerprint_graph(graph);
    apxchol::detail::gpu_round_shadow_session session(true);
    auto selector_owner = apxchol::test::initialize_owned_selector(session, graph);
    auto& selector = *selector_owner;
    std::vector<node_index> active(graph.n());
    std::iota(active.begin(), active.end(), node_index{0});
    selector.prepare(active, {});
    (void)selector.select_block_greedy();
    const auto old = selector.device_selection();
    session.run_owned_prefix_round(old, kOwnedSeed);
    session.advance_selector(selector);
    EXPECT_THROW(session.run_owned_prefix_round(old, kOwnedSeed), std::exception);
    std::vector<apxchol::detail::factor_col> columns;
    EXPECT_THROW(observe_owned_prefix(session, graph, active, columns), std::exception);
    EXPECT_TRUE(columns.empty());
    EXPECT_EQ(fingerprint_graph(graph), initial);
}
#endif

#if defined(APXCHOL_USE_CUDA)
namespace {
apxchol::test::sparse_matrix owned_solve_matrix(int n, double shift) {
    std::vector<apxchol::test::triplet> entries;
    for (int i = 0; i < n; ++i) {
        entries.push_back({i, i, 4.0 + shift});
        for (int offset : {1, 8}) {
            int j = (i + offset) % n;
            entries.push_back({i, j, -1.0}); entries.push_back({j, i, -1.0});
        }
    }
    return apxchol::test::from_triplets(n, n, entries);
}
}
#endif

TEST(GpuOwnedPrefix, ConsumingSolveEliminatesEveryColumnOnDevice) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_omp_threads serial(1);
    const int n = 256; // Includes the under-occupied tail, without CPU handoff.
    apxchol::factor_options factor_options;
    factor_options.factor_storage = apxchol::factor_storage_type::fp32;
    factor_options.drop_tolerance = 0;
    factor_options.verbose = true;
    factor_options.partition.degree_quantile = 0.0;
    factor_options.partition.degree_multiplier = 100.0;
    factor_options.experimental.min_is_fraction = 0.0;
    for (double shift : {0.0, 1.0}) {
        SCOPED_TRACE(shift);
        auto A = owned_solve_matrix(n, shift);
        apxchol::solve_options setup;
        setup.backend = apxchol::solve_backend::gpu;
        setup.factor_opts = factor_options;
        apxchol::detail::checkpoint owned_cp;
        testing::internal::CaptureStderr();
        const auto owner = apxchol::test::with_timings(A, setup, owned_cp);
        const std::string trace = testing::internal::GetCapturedStderr();
        const auto& preconditioner = gpu_session(owner);
        ASSERT_NE(trace.find("[gpu-owned-factorization] complete rounds="), std::string::npos) << trace;
        EXPECT_GT(gpu_round_trace_size(trace, "rounds="), 2u);
        EXPECT_NE(trace.find("[gpu-owned-csc] n="), std::string::npos);
        EXPECT_NE(trace.find("factor_entry_download_bytes=0"), std::string::npos);
        expect_setup_api_receipt(trace, preconditioner, factor_options.verbose);
        EXPECT_GT(owned_cp.total("setup.gpu_owned_factorization"), 0.0);
        EXPECT_EQ(owned_cp.total("setup.factor_metadata"), 0.0);
        const auto& factor = preconditioner.factor();
        ASSERT_GT(factor.rounds.size(), 2u);
        std::size_t remaining = A.rows;
        for (const auto& round : factor.rounds) {
            EXPECT_EQ(round.active, remaining);
            ASSERT_GT(round.is_size, 0u); ASSERT_LE(round.is_size, remaining);
            remaining -= round.is_size;
        }
        EXPECT_EQ(remaining, 0u);
        EXPECT_EQ(gpu_round_trace_size(trace, "factor_header_count="), factor.perm.size());
        auto permutation = factor.perm;
        std::sort(permutation.begin(), permutation.end());
        for (std::size_t i = 0; i < permutation.size(); ++i) EXPECT_EQ(permutation[i], i);
        EXPECT_EQ(factor.sddm, shift != 0.0);
        EXPECT_TRUE(csc_access::vals(factor.L).empty()); EXPECT_TRUE(csc_access::inner(factor.L).empty());
        ASSERT_TRUE(preconditioner.trsv().adopted_device_factor());
        std::vector<double> exact(A.rows);
        for (std::size_t i = 0; i < exact.size(); ++i) exact[i] = std::sin(double(i) + 0.25);
        const std::vector<double> b = apxchol::test::multiply(A, exact);
        apxchol::solve_options options;
        options.backend = apxchol::solve_backend::gpu;
        options.factor_opts = factor_options;
        const auto solved = apxchol::solve(apxchol::test::csc(A), b, 1e-8, 1000, {}, options);
        EXPECT_LE(apxchol::test::relative_residual(A, solved.x, b), 1e-8);
    }
#endif
}

TEST(GpuOwnedPrefix, TinyTailAndExportKeepTheirContracts) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    const std::vector<apxchol::test::triplet> entries{{0,0,2},{1,1,2},{0,1,-1},{1,0,-1}};
    const auto tiny = apxchol::test::from_triplets(2, 2, entries);
    const apxchol::factor_options factor_options{
        .factor_storage = apxchol::factor_storage_type::fp32, .drop_tolerance = 0};
    apxchol::solve_options options;
    options.backend = apxchol::solve_backend::gpu;
    options.factor_opts = factor_options;
    const apxchol::solver tiny_owner(apxchol::test::csc(tiny), options);
    const auto& small = gpu_session(tiny_owner);
    EXPECT_EQ(small.factor().perm.size(), 2u);
    EXPECT_TRUE(small.trsv().adopted_device_factor());
    EXPECT_TRUE(csc_access::vals(small.factor().L).empty());
    auto A = owned_solve_matrix(64, 1.0);
    auto exported = apxchol::factorize(apxchol::test::csc(A), factor_options);
    EXPECT_EQ(csc_access::vals(exported.L).size(), exported.L.nonZeros());
    EXPECT_FALSE(apxchol::detail::factorization_access::device_factor(exported));
    apxchol::solver owner(apxchol::test::csc(A), exported, options);
    EXPECT_FALSE(gpu_session(owner).trsv().adopted_device_factor());
    EXPECT_TRUE(csc_access::vals(gpu_session(owner).factor().L).empty());
    EXPECT_EQ(csc_access::vals(exported.L).size(), exported.L.nonZeros());
    auto host_options = options;
    host_options.backend = apxchol::solve_backend::cpu;
    ASSERT_NO_THROW(apxchol::solver(apxchol::test::csc(A), exported, host_options));
    ASSERT_TRUE(exported.sddm);
    std::vector<double> rhs(A.rows), permuted(A.rows), expected(A.rows);
    for (std::int64_t v = 0; v < A.rows; ++v) rhs[v] = 1.0 + double(v) / double(A.rows - 1);
    for (std::int64_t v = 0; v < A.rows; ++v) permuted[exported.perm[v]] = rhs[v];
    // The direct CUDA reference uses the same FP32 arithmetic as the owner.
    apxchol::detail::cuda_sptrsv uploaded;
    uploaded.setup(exported.L, static_cast<node_index>(A.rows), factor_options);
    uploaded.solve_LLt(permuted.data(), expected.data());
    const std::vector<double> actual = owner.apply(rhs);
    for (std::int64_t v = 0; v < A.rows; ++v) EXPECT_EQ(actual[v], expected[exported.perm[v]]) << v;
#endif
}


TEST(GpuCycleSampler, NormalReferenceAndOversizedMomentLaw) {
    // A common normalization can round unequal cutoff numerators to a tie.
    // Numerator comparison deliberately retains the one-ulp improvement.
    const double earlier=0x1.f5bc1aec2f4a9p+6,later=0x1.f5bc1aec2f4a8p+6;
    const double normalization=0x1.b732be6fd2141p+5;
    EXPECT_LT(later,earlier);
    EXPECT_EQ(later/normalization,earlier/normalization);
#if !defined(APXCHOL_USE_CUDA) || !defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
    GTEST_SKIP() << "CUDA test hooks required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_omp_threads serial(1);
    for (auto sampler : {apxchol::clique_sampler::trace_cycle}) {
        for (unsigned profile = 0; profile < 13; ++profile) {
            SCOPED_TRACE(std::to_string(static_cast<int>(sampler))+":"+std::to_string(profile));
            std::vector<unsigned> degree = profile == 0 ?
                std::vector<unsigned>{0,1,2,3,127,128,129} :
                profile == 1 ? std::vector<unsigned>{3,4,5,31,129} :
                               profile == 2 ? std::vector<unsigned>{32,64,65} :
                                              std::vector<unsigned>{3,7,129};
            // Large irregular rows and more than one CTA of rows exercise
            // cooperative searches, partial warp chunks and in-place cycles.
            if(profile==6)degree={3,129,255,256,257,1025,4097};
            if(profile==7) {
                degree.clear();
                for(unsigned row=0;row<37;++row)degree.push_back(129+13*row);
            }
            if(profile==8)degree={129,3};
            if(profile==11)degree={130,3};
            if(profile==12)degree={3,129,160,257,1025,4097};
            if(profile==9)degree={3,4,31,127,128,129,257};
            // Pivot zero's first SplitMix draw is zero, forcing rejection for
            // the 129-way Fisher-Yates draw. Subsequent parent streams must
            // retain that extra draw rather than assuming fixed consumption.
            constexpr std::uint64_t increment=0x9E3779B97F4A7C15ULL;
            const std::uint64_t run_seed=(profile==8 || profile==11) ? (0-increment)^increment : 17;
            const bool numerical_fallback=profile==4 || profile==5 || profile==10;
            const auto count = static_cast<node_index>(degree.size());
            node_index n=count;
            std::vector<undirected_edge> edges;
            std::vector<std::vector<apxchol::weighted_neighbor>> stars(count);
            for(node_index pivot=0;pivot<count;++pivot)for(unsigned j=0;j<degree[pivot];++j) {
                // Profiles 3/4/5 cover representable subnormals, zero inputs,
                // and zero-rounding output, for normal and oversized rows.
                const auto boundary=profile==9 ? (pivot%3==0 ? float{2} :
                    std::nextafter(float{2},pivot%3==1 ? float{0} :
                        std::numeric_limits<float>::infinity())) : float{1};
                const double weight = profile==0 || profile==8 ? 1. : profile==3 ? std::exp2(-70) :
                    profile==4 ? (j==0 ? 0. : 1.) : profile==5 ? std::exp2(sizeof(float)==4?-90:-600) :
                    profile==7 ? 1.+double(j%17)/64. :
                    profile==9 ? (j==0 ? 1. : double(boundary)) :
                    profile==11 ? (j==0 ? std::exp2(-10) : 1.) :
                    profile==12 ? std::ldexp(1.+double((j*37)%127)/128.,-int((j*23)%61)) :
                    profile==10 ? (j==0 ? std::numeric_limits<float>::denorm_min() : std::exp2(sizeof(float)==4?-70:-500)) :
                    std::exp2(int(j%11)-5);
                stars[pivot].push_back({n,weight});
                if(profile==2) {edges.push_back({pivot,n,.25*weight});edges.push_back({pivot,n,.75*weight});}
                else edges.push_back({pivot,n,weight});
                ++n;
            }
            std::vector<std::tuple<node_index,node_index,double>> expected;
            for(node_index pivot=0;pivot<count;++pivot) {
                double D=.125;for(auto edge:stars[pivot])D+=edge.weight;
                std::vector<apxchol::deferred_edge> fill;
                apxchol::detail::tree_elimination{.sampler=numerical_fallback?apxchol::clique_sampler::gks:sampler}.sample_clique(stars[pivot],D,
                    apxchol::detail::gpu_round_shadow_pivot_seed(run_seed,pivot),apxchol::edge_emitter(fill));
                for(auto edge:fill)expected.emplace_back(std::min(edge.u,edge.v),std::max(edge.u,edge.v),
                    static_cast<double>(static_cast<float>(edge.w)));
            }
            std::sort(expected.begin(),expected.end());
            std::vector<std::tuple<node_index,node_index,double>> previous;
            for(unsigned repeat=0;repeat<2;++repeat) {
                apxchol::detail::graph graph(n);
                for(auto edge:edges)graph.add_edge(edge.u,edge.v,edge.weight);
                for(node_index v=0;v<n;++v)graph.excess(v)=.125;
                apxchol::detail::gpu_round_shadow_session session(true,sampler);
                auto selector_owner = apxchol::test::initialize_owned_selector(session, graph);
                auto& selector = *selector_owner;
                std::vector<node_index> active(n),wanted(count);
                std::iota(active.begin(),active.end(),node_index{0});
                std::iota(wanted.begin(),wanted.end(),node_index{0});
                apxchol::partition_options options;
                options.degree_quantile=0;options.degree_multiplier=16384;
                selector.prepare(active,options,42,0,false);
                const auto selected=selector.select_for_test(wanted).data;
                ASSERT_EQ(selected,wanted);
                const auto report=session.run_owned_prefix_round(selector.device_selection(),run_seed);
                ASSERT_TRUE(report.gpu_executed);
                if(profile==7)EXPECT_EQ(report.normal_batch.normal_pivots,0u);
                else EXPECT_GT(report.normal_batch.normal_pivots,0u);
                EXPECT_GT(report.normal_batch.oversized_pivots,0u);
                EXPECT_EQ(report.sampler_numerical_fallbacks,numerical_fallback?count:0u);
                EXPECT_EQ(report.raw_fill_edges,expected.size());
                EXPECT_EQ(report.live_incidences,2*expected.size());
                EXPECT_LE(report.live_incidences,2*edges.size());
                if(profile==1 || profile==3) {
                    EXPECT_EQ(report.live_incidences,2*edges.size()); // f=d uses every slot.
                }
                std::vector<apxchol::detail::factor_col> columns;
                observe_owned_prefix(session,graph,active,columns);
                std::vector<std::tuple<node_index,node_index,double>> got;
                for(node_index v=count;v<n;++v)for(auto [u,w]:graph.neighbors(v)) {
                    EXPECT_NE(u,v);EXPECT_GE(u,count);EXPECT_TRUE(std::isfinite(w));
                    if(numerical_fallback) {EXPECT_GE(w,0.0);} else {EXPECT_GT(w,0.0);}
                    if(v<u)got.emplace_back(v,u,static_cast<double>(w));
                }
                std::sort(got.begin(),got.end());ASSERT_EQ(got.size(),expected.size());
                if (profile==3 && sizeof(float)==4) {
                    for (const auto& edge:got)
                        EXPECT_EQ(std::fpclassify(static_cast<float>(std::get<2>(edge))),FP_SUBNORMAL);
                }
                std::vector<node_index> owner(n),rank(n);
                for(node_index pivot=0;pivot<count;++pivot)
                    for(node_index j=0;j<stars[pivot].size();++j) {
                        owner[stars[pivot][j].vertex]=pivot;rank[stars[pivot][j].vertex]=j;
                    }
                for(std::size_t i=0;i<got.size();++i) {
                    const auto pivot=owner[std::get<0>(got[i])];
                    // Trace rows of degree >= 3 are sampled per item on device: the plan is
                    // serial per pivot, the emission is one thread per neighbour slot, and the
                    // parents invert q directly instead of replaying the host's two-draw
                    // mixture. The realization therefore differs from the scalar host law
                    // while the law itself does not; the block below checks the law for every
                    // such row. Degrees below three keep the established GKS emission and stay
                    // bit-faithful here.
                    if(sampler==apxchol::clique_sampler::trace_cycle &&
                       !numerical_fallback && stars[pivot].size()>=3)continue;
                    EXPECT_EQ(std::get<0>(got[i]),std::get<0>(expected[i]));
                    EXPECT_EQ(std::get<1>(got[i]),std::get<1>(expected[i]));
                    EXPECT_NEAR(std::get<2>(got[i]),std::get<2>(expected[i]),
                        8*std::numeric_limits<float>::epsilon()*std::get<2>(expected[i]));
                }
                // The device emission is item-parallel and its folds need not reproduce the
                // scalar cut, so verify the law rather than the realization for every sampled
                // trace row: the cutoff against its high-precision objective, the core
                // topology and its seeded rejection-aware permutation, the Horvitz-Thompson
                // weights, and each light parent's inverted interval against the same stream.
                if(sampler==apxchol::clique_sampler::trace_cycle && !numerical_fallback)
                for(node_index pivot=0;pivot<count;++pivot) {
                    const auto& star=stars[pivot];const auto d=star.size();
                    if(d<3)continue;
                    std::vector<unsigned> later(d),core_degree(d);
                    std::vector<std::size_t> parent_index(d,d);
                    std::vector<std::tuple<std::size_t,std::size_t,double>> row;
                    for(const auto& [u,v,w]:got)if(owner[u]==pivot) {
                        ASSERT_EQ(owner[v],pivot);
                        auto i=std::size_t(rank[u]),j=std::size_t(rank[v]);
                        if(i>j)std::swap(i,j);
                        ++later[i];parent_index[i]=j;row.emplace_back(i,j,w);
                    }
                    ASSERT_EQ(row.size(),d);
                    std::size_t cut=0;while(cut<d && later[cut]==1)++cut;
                    ASSERT_LE(cut,d-3);const auto h=d-cut;
                    if(profile==11 && d>128) {EXPECT_EQ(cut,1u);EXPECT_EQ(h,129u);}
                    std::vector<long double> suffix(d+1),square(d+1);
                    double D=.125;for(auto edge:star)D+=edge.weight;
                    for(std::size_t i=d;i-->0;) {
                        const long double x=static_cast<long double>(star[i].weight)/star.back().weight;
                        suffix[i]=x+suffix[i+1];square[i]=x*x+square[i+1];
                    }
                    long double parents=0,best=std::numeric_limits<long double>::infinity(),chosen=0;
                    for(std::size_t i=0;i+2<d;++i) {
                        const auto size=d-i;
                        const long double score=parents+(size==3?0.L:(size-3.L)*(size-1.L)*.5L*square[i]);
                        best=std::min(best,score);if(i==cut)chosen=score;
                        const long double x=static_cast<long double>(star[i].weight)/star.back().weight,m=d-i-1;
                        parents+=(m-1)*x*(2*suffix[i+1]+m*x);
                    }
                    EXPECT_LE(chosen-best,128*std::numeric_limits<double>::epsilon()*(d+1)*std::max(1.L,best));
                    std::vector<std::pair<std::size_t,std::size_t>> core;
                    for(const auto& [i,j,w]:row) {
                        const long double ai=star[i].weight,aj=star[j].weight;
                        long double reference;
                        if(i<cut) {
                            EXPECT_EQ(later[i],1u);
                            const long double mass=suffix[i+1]*star.back().weight+(d-i-1)*ai;
                            reference=ai/D*mass/(1+ai/aj);
                        } else {
                            ++core_degree[i];++core_degree[j];core.emplace_back(i,j);
                            reference=ai*aj/D*(h-1)*.5L;
                        }
                        const auto rounded=static_cast<float>(reference);
                        const double tolerance=(8*std::numeric_limits<float>::epsilon()+
                            128*std::numeric_limits<double>::epsilon()*(d+1))*std::abs(double(rounded))+
                            2*double(std::numeric_limits<float>::denorm_min());
                        EXPECT_NEAR(w,double(rounded),tolerance);
                    }
                    ASSERT_EQ(core.size(),h);
                    for(std::size_t i=cut;i<d;++i)EXPECT_EQ(core_degree[i],2u);
                    std::vector<std::size_t> permutation(h);std::iota(permutation.begin(),permutation.end(),cut);
                    apxchol::random_stream rng{apxchol::detail::gpu_round_shadow_pivot_seed(run_seed,pivot)};
                    for(auto k=h;k>1;--k)std::swap(permutation[k-1],permutation[apxchol::detail::uniform_index(rng,k)]);
                    std::vector<std::pair<std::size_t,std::size_t>> expected_core;
                    for(std::size_t k=0;k<h;++k)expected_core.emplace_back(
                        std::min(permutation[k],permutation[(k+1)%h]),std::max(permutation[k],permutation[(k+1)%h]));
                    std::sort(core.begin(),core.end());std::sort(expected_core.begin(),expected_core.end());
                    EXPECT_EQ(core,expected_core);
                    for(std::size_t i=0;i<cut;++i) {
                        const auto j=parent_index[i];ASSERT_GT(j,i);ASSERT_LT(j,d);
                        const long double ai=static_cast<long double>(star[i].weight)/star.back().weight;
                        const long double mass=suffix[i+1]+(d-i-1)*ai;
                        const long double target=rng.next_unit()*mass;
                        const long double upper=suffix[j]+(d-j)*ai;
                        const long double lower=suffix[j+1]+(d-j-1)*ai;
                        const long double tolerance=128*std::numeric_limits<double>::epsilon()*(d+1)*mass;
                        EXPECT_GE(target+tolerance,lower);EXPECT_LE(target-tolerance,upper);
                    }
                }
                if(repeat) {
                    EXPECT_EQ(got,previous);
                }
                previous=got;
            }
        }
    }
#endif
}

TEST(GpuCycleSampler, OwnedSolveReturnsOriginalSystemSolution) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_omp_threads serial(1);
    for(auto sampler:{apxchol::clique_sampler::trace_cycle})
    for(double shift:{0.,1.}) {
        SCOPED_TRACE(std::to_string(static_cast<int>(sampler))+":"+std::to_string(shift));
        auto A=owned_solve_matrix(257,shift);
        std::vector<double> exact(A.rows);for(std::size_t i=0;i<exact.size();++i)exact[i]=std::sin(double(i)+.25);
        const std::vector<double> b=apxchol::test::multiply(A,exact);
        apxchol::solve_options opts;
        opts.backend=apxchol::solve_backend::gpu;
        opts.factor_opts.verbose = true;
        opts.factor_opts.sampler=sampler;opts.factor_opts.partition.degree_quantile=.8;
        testing::internal::CaptureStderr();
        const auto result=apxchol::solve(apxchol::test::csc(A), b, 1e-8, 1000, {}, opts);
        const auto trace=testing::internal::GetCapturedStderr();
        EXPECT_LE(apxchol::test::relative_residual(A,result.x,b),1e-8);
        EXPECT_NE(trace.find("[gpu-owned-factorization] complete rounds="),std::string::npos);
        EXPECT_NE(trace.find("[gpu-clique-sampler] sampler=trace_cycle"),std::string::npos);
        EXPECT_NE(trace.find("device_sampling=1"),std::string::npos);
    }
#endif
}
