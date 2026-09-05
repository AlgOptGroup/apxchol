#include <gtest/gtest.h>

#include "apxchol/checkpoint.h"
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
#include <limits>
#include <memory>
#include <memory_resource>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(APXCHOL_USE_CUDA)
#include <cuda_runtime_api.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using apxchol::node_index;
using apxchol::detail::gpu_round_shadow_excess_bound;
using apxchol::detail::gpu_round_shadow_input;
using apxchol::detail::gpu_round_shadow_report;

class scoped_omp_threads {
public:
    explicit scoped_omp_threads(int threads) {
#ifdef _OPENMP
        before_ = omp_get_max_threads();
        omp_set_num_threads(threads);
#else
        (void)threads;
#endif
    }
    ~scoped_omp_threads() {
#ifdef _OPENMP
        omp_set_num_threads(before_);
#endif
    }

private:
#ifdef _OPENMP
    int before_ = 1;
#endif
};

inline constexpr int kResidentProvenanceWorkers = 2;

struct undirected_edge {
    node_index u;
    node_index v;
    double weight;
};

void establish_r2a_openmp_affinity_if_requested() {
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

    std::vector<std::vector<std::pair<node_index, double>>> adjacency(n);
    for (const auto& edge : edges) {
        // A real directed-AoS graph narrows each edge once on insertion and
        // promotes it back to double when process_vertex reads the slab.
        const double stored = static_cast<double>(
            static_cast<apxchol::pool_value_t>(edge.weight));
        adjacency[edge.u].push_back({edge.v, stored});
        adjacency[edge.v].push_back({edge.u, stored});
    }
    input.owner_offsets.resize(static_cast<std::size_t>(n) + 1);
    for (node_index owner = 0; owner < n; ++owner) {
        input.owner_offsets[owner] = input.incidences.size();
        for (const auto [neighbor, weight] : adjacency[owner])
            input.incidences.push_back({owner, neighbor, weight});
    }
    input.owner_offsets[n] = input.incidences.size();
    return input;
}

gpu_round_shadow_input compact_live_snapshot(gpu_round_shadow_input input) {
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
    apxchol::graph<apxchol::directed_vec_pool_incidence> residual;
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
    apxchol::graph<apxchol::directed_vec_pool_incidence> graph(
        input.vertex_count);
    for (const auto& edge : edges)
        graph.add_edge(edge.u, edge.v, edge.weight);
    for (node_index vertex = 0; vertex < input.vertex_count; ++vertex)
        graph.excess(vertex) = input.excess[vertex];

    apxchol::factorize_workspace workspace;
    workspace.threads.resize(workers);
    for (auto& thread : workspace.threads)
        thread.factor_entries =
            std::make_unique<std::pmr::monotonic_buffer_resource>();

    apxchol::partition_result partition;
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
    options.omp_threshold = 0;
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
        workspace, options, nullptr, false, work_hint);
    const auto comparison =
        apxchol::detail::compare_gpu_round_shadow_with_cpu(
            expected, excess_bounds, graph, columns);
    auto factor_log = materialize_factor_log(columns);
    return {std::move(graph), comparison, std::move(factor_log)};
}

gpu_round_shadow_report run_device(
        const gpu_round_shadow_input& input,
        std::vector<gpu_round_shadow_excess_bound>* bounds = nullptr) {
    return apxchol::detail::run_verified_gpu_round_shadow(input, bounds);
}

apxchol::detail::gpu_round_shadow_cpu_comparison run_serial_cpu_round(
        apxchol::graph<apxchol::directed_vec_pool_incidence>& graph,
        const gpu_round_shadow_input& input,
        const gpu_round_shadow_report& expected,
        std::span<const gpu_round_shadow_excess_bound> excess_bounds,
        apxchol::detail::gpu_round_shadow_factor_log* factor_log = nullptr) {
    apxchol::factorize_workspace workspace;
    workspace.threads.resize(1);
    workspace.threads.front().factor_entries =
        std::make_unique<std::pmr::monotonic_buffer_resource>();
    apxchol::partition_result partition;
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
        workspace, options, nullptr, false, work_hint);
    if (factor_log) *factor_log = materialize_factor_log(columns);
    return apxchol::detail::compare_gpu_round_shadow_with_cpu(
        expected, excess_bounds, graph, columns);
}

struct parallel_cpu_round_result {
    apxchol::detail::gpu_round_shadow_cpu_comparison comparison;
    std::uint32_t worker_mask = 0;
    int team_width = 0;
};

parallel_cpu_round_result run_parallel_cpu_round(
        apxchol::graph<apxchol::directed_vec_pool_incidence>& graph,
        const gpu_round_shadow_input& input,
        const gpu_round_shadow_report& expected,
        std::span<const gpu_round_shadow_excess_bound> excess_bounds) {
    // Keep the explicit team configuration local to this parallel fixture.
    scoped_omp_threads declared_team(kResidentProvenanceWorkers);
    apxchol::factorize_workspace workspace;
    constexpr std::size_t workers = kResidentProvenanceWorkers;
    workspace.threads.resize(workers);
    for (auto& thread : workspace.threads)
        thread.factor_entries =
            std::make_unique<std::pmr::monotonic_buffer_resource>();
    apxchol::partition_result partition;
    partition.data = input.pivots;
    apxchol::factor_options options;
    options.seed = input.pivots.empty() ? 0 :
        input.seeds.front() ^
        ((std::uint64_t(input.pivots.front()) + 1) *
         0x9E3779B97F4A7C15ULL);
    // Stay on the fine-grained production schedule while adjacency work, not
    // an unconditional vertex threshold, selects the two-worker team.
    options.omp_threshold = input.pivots.size();
    std::size_t work_hint = 0;
    for (node_index pivot : input.pivots) {
        for (std::size_t i = input.owner_offsets[pivot];
             i < input.owner_offsets[pivot + 1]; ++i)
            if (input.active[input.incidences[i].neighbor]) ++work_hint;
    }
    if (apxchol::detail::elimination_round_team_size(
            input.pivots.size(), work_hint, options.omp_threshold,
            workspace.threads.size()) != workers)
        throw std::logic_error("test did not select the parallel CPU apply path");
    if (input.pivots.size() < workers)
        throw std::logic_error("parallel CPU apply test needs two pivots");

#ifdef _OPENMP
    const int saved_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
#endif
    std::atomic<std::uint32_t> worker_mask{0};
    std::atomic<int> team_width{0};
    const apxchol::detail::tree_elimination tree;
    const auto recording_tree = apxchol::as_eliminator(
        [&](std::span<apxchol::weighted_neighbor> neighbors, double degree,
            std::uint64_t seed, apxchol::edge_emitter out) {
#ifdef _OPENMP
            const int worker = omp_get_thread_num();
            team_width.store(omp_get_num_threads(), std::memory_order_relaxed);
#else
            const int worker = 0;
            team_width.store(1, std::memory_order_relaxed);
#endif
            worker_mask.fetch_or(std::uint32_t{1} << worker,
                                 std::memory_order_relaxed);
            tree.sample_clique(neighbors, degree, seed, out);
        });
    std::vector<apxchol::detail::factor_col> columns;
    apxchol::detail::eliminate_partition(
        recording_tree, graph, partition, columns,
        workspace, options, nullptr, false, work_hint);
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
        const apxchol::graph<apxchol::directed_vec_pool_incidence>& graph) {
    return apxchol::detail::fingerprint_gpu_round_shadow_input(
        apxchol::detail::make_gpu_round_shadow_input(
            graph, std::span<const node_index>{}, 0));
}

constexpr node_index kResidentProvenanceFirstParallelPivot = 8;
constexpr node_index kResidentProvenanceParallelPivotEnd = 24;
constexpr node_index kResidentProvenanceVertexCount = 536;

apxchol::graph<apxchol::directed_vec_pool_incidence>
make_resident_provenance_graph() {
    // run_parallel_cpu_round explicitly requests two workers. The pool sizes
    // its per-worker grow scratch at graph construction, so establish that
    // declared team before allocation even when the process starts with T=1.
    // The guard restores the surrounding test's OpenMP setting on return.
    scoped_omp_threads declared_team(kResidentProvenanceWorkers);
    apxchol::graph<apxchol::directed_vec_pool_incidence> graph(
        kResidentProvenanceVertexCount);
    for (const auto& edge : std::vector<undirected_edge>{
        {0, 5, 1.0}, {0, 6, 4.0}, {0, 7, 2.0},
        {1, 3, 3.0}, {1, 4, 5.0}, {1, 5, 7.0},
        {2, 3, 6.0}, {2, 4, 8.0}, {2, 5, 9.0},
        {3, 6, 10.0}, {4, 7, 11.0}, {5, 6, 12.0}})
        graph.add_edge(edge.u, edge.v, edge.weight);
    // Sixteen independent degree-512 pivots ensure that the production dynamic
    // CPU compute/apply path uses both workers. Their common neighbor set also
    // permits relaxed endpoint-slot claims to differ from the GPU's stable
    // pivot/emission order; an individual scheduling outcome may still match.
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
std::vector<std::uint64_t> encode_factor_columns(
        std::span<const apxchol::detail::factor_col> columns) {
    std::vector<std::uint64_t> encoded;
    for (const auto& column : columns) {
        encoded.push_back(column.vertex);
        encoded.push_back(std::bit_cast<std::uint32_t>(column.diag));
        encoded.push_back(column.entry_count);
        for (std::size_t i = 0; i < column.entry_count; ++i) {
            encoded.push_back(column.entries[i].neighbor);
            encoded.push_back(
                std::bit_cast<std::uint32_t>(column.entries[i].value));
        }
    }
    return encoded;
}

std::vector<node_index> factor_elimination_order(
        const apxchol::factorization& factor) {
    std::vector<node_index> order(factor.perm.size());
    std::vector<std::uint8_t> seen(factor.perm.size(), 0);
    for (std::size_t vertex = 0; vertex < factor.perm.size(); ++vertex) {
        const std::size_t position = factor.perm[vertex];
        if (position >= order.size() || seen[position])
            throw std::logic_error("factor permutation is not bijective");
        order[position] = static_cast<node_index>(vertex);
        seen[position] = 1;
    }
    return order;
}

void expect_factor_log_matches_assembled(
        const apxchol::factorization& factor, std::size_t first_column,
        const apxchol::detail::gpu_round_shadow_factor_log& expected) {
    const auto order = factor_elimination_order(factor);
    const auto* outer = factor.L.outerIndexPtr();
    const auto* inner = factor.L.innerIndexPtr();
    const auto* values = factor.L.valuePtr();
    ASSERT_LE(first_column + expected.columns.size(), order.size());
    for (std::size_t local = 0; local < expected.columns.size(); ++local) {
        SCOPED_TRACE("assembled factor column " + std::to_string(local));
        const std::size_t column = first_column + local;
        const std::size_t begin = outer[column];
        const std::size_t end = outer[column + 1];
        ASSERT_LT(begin, end);
        ASSERT_EQ(static_cast<std::size_t>(inner[begin]), column);
        const auto& expected_column = expected.columns[local];
        EXPECT_EQ(order[column], expected_column.vertex);
        EXPECT_EQ(std::bit_cast<std::uint32_t>(values[begin]),
                  std::bit_cast<std::uint32_t>(expected_column.diag));

        std::vector<std::pair<node_index, std::uint32_t>> actual_entries;
        for (std::size_t entry = begin + 1; entry < end; ++entry) {
            actual_entries.push_back({
                order[inner[entry]],
                std::bit_cast<std::uint32_t>(
                    static_cast<apxchol::factor_value_t>(-values[entry]))});
        }
        std::vector<std::pair<node_index, std::uint32_t>> expected_entries;
        const std::size_t entry_begin = expected_column.entry_begin;
        ASSERT_LE(entry_begin + expected_column.entry_count,
                  expected.entries.size());
        for (std::size_t entry = 0; entry < expected_column.entry_count;
             ++entry) {
            const auto& value = expected.entries[entry_begin + entry];
            expected_entries.push_back({
                value.neighbor, std::bit_cast<std::uint32_t>(value.value)});
        }
        std::sort(actual_entries.begin(), actual_entries.end());
        std::sort(expected_entries.begin(), expected_entries.end());
        EXPECT_EQ(actual_entries, expected_entries);
    }
}

std::vector<std::string> gpu_round_trace_lines(const std::string& trace) {
    std::vector<std::string> result;
    std::istringstream input(trace);
    for (std::string line; std::getline(input, line);) {
        if (line.starts_with("[gpu-round-shadow] round="))
            result.push_back(std::move(line));
    }
    return result;
}

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

std::pair<std::size_t, std::size_t> gpu_round_trace_size_pair(
        const std::string& line, std::string_view name) {
    const std::string value = gpu_round_trace_field(line, name);
    const std::size_t separator = value.find('/');
    if (separator == std::string::npos)
        throw std::runtime_error("malformed GPU round trace pair " +
                                 std::string(name));
    return {
        static_cast<std::size_t>(std::stoull(value.substr(0, separator))),
        static_cast<std::size_t>(std::stoull(value.substr(separator + 1)))};
}

apxchol::detail::gpu_round_shadow_digest gpu_round_trace_digest(
        const std::string& line, std::string_view name) {
    const std::string value = gpu_round_trace_field(line, name);
    const std::size_t separator = value.find(':');
    if (separator == std::string::npos)
        throw std::runtime_error("malformed GPU round trace digest " +
                                 std::string(name));
    return {
        std::stoull(value.substr(0, separator), nullptr, 16),
        std::stoull(value.substr(separator + 1), nullptr, 16)};
}

void expect_digest_equal(
        const apxchol::detail::gpu_round_shadow_digest& expected,
        const apxchol::detail::gpu_round_shadow_digest& actual) {
    EXPECT_EQ(actual.xor_hash, expected.xor_hash);
    EXPECT_EQ(actual.sum_hash, expected.sum_hash);
}

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

resident_selection_boundary make_resident_selection_boundary(
        bool accept_initial_generation = true) {
    resident_selection_boundary boundary;
    const auto edges = resident_selection_path_edges();
    const std::vector<node_index> first_pivots = {0};
    const auto first_input = make_input(
        kResidentSelectionVertexCount, edges, first_pivots,
        kResidentSelectionSeed);
    const auto first_expected =
        apxchol::detail::reference_gpu_round_shadow(first_input);
    const auto first = boundary.state.compute_discover_shape(first_input);
    apxchol::detail::compare_gpu_round_shadow_reports(first_expected, first);
    boundary.initial_generation = first.output_generation;
    if (accept_initial_generation)
        boundary.state.accept_device_generation(first.output_generation);

    std::vector<apxchol::detail::gpu_topology_edge> topology;
    topology.reserve(edges.size());
    for (const auto& edge : edges) topology.push_back({edge.u, edge.v});
    boundary.frontend =
        std::make_unique<apxchol::detail::gpu_block_frontend>(
            kResidentSelectionVertexCount, topology);
    boundary.frontend->advance(first_pivots, {});
    boundary.active.resize(kResidentSelectionVertexCount - 1);
    std::iota(boundary.active.begin(), boundary.active.end(), node_index{1});
    apxchol::partition_options options;
    options.degree_multiplier = 100.0;
    boundary.frontend->prepare(boundary.active, options);
    boundary.selected = boundary.frontend->select_block_greedy().data;
    if (boundary.selected.size() < 2)
        throw std::runtime_error(
            "resident selection test requires at least two selected vertices");
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

class scoped_env {
public:
    scoped_env(const char* name, const char* value) : name_(name) {
        if (const char* before = std::getenv(name)) {
            had_value_ = true;
            before_ = before;
        }
        setenv(name, value, 1);
    }
    ~scoped_env() {
        if (had_value_)
            setenv(name_.c_str(), before_.c_str(), 1);
        else
            unsetenv(name_.c_str());
    }

private:
    std::string name_;
    std::string before_;
    bool had_value_ = false;
};


#define REQUIRE_GPU_ROUND_SHADOW_DEVICE()                                      \
    do {                                                                        \
        if (!apxchol::detail::gpu_round_shadow_runtime_available())             \
            GTEST_SKIP() << "CUDA round-shadow runtime/device unavailable";    \
    } while (false)

} // namespace

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
    apxchol::graph<apxchol::directed_vec_pool_incidence> graph(6);
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

TEST(GpuRoundShadowEnvironment, OnlyExplicitForceEnablesTheResearchPath) {
    {
        scoped_env env("APXCHOL_GPU_ROUND_SHADOW", "off");
        EXPECT_FALSE(apxchol::detail::gpu_round_shadow_requested());
    }
    {
        scoped_env env("APXCHOL_GPU_ROUND_SHADOW", "force");
        EXPECT_TRUE(apxchol::detail::gpu_round_shadow_requested());
    }
    {
        scoped_env env("APXCHOL_GPU_ROUND_SHADOW", "1");
        EXPECT_THROW(apxchol::detail::gpu_round_shadow_requested(),
                     std::invalid_argument);
    }
}

TEST(GpuRoundShadowEnvironment,
     SelectionCertificateAuditRequiresAnExplicitValidValue) {
    {
        scoped_env env("APXCHOL_GPU_ROUND_SELECTION_AUDIT", "off");
        EXPECT_FALSE(apxchol::detail::gpu_round_selection_audit_requested());
    }
    {
        scoped_env env("APXCHOL_GPU_ROUND_SELECTION_AUDIT", "on");
        EXPECT_TRUE(apxchol::detail::gpu_round_selection_audit_requested());
    }
    {
        scoped_env env("APXCHOL_GPU_ROUND_SELECTION_AUDIT", "force");
        EXPECT_TRUE(apxchol::detail::gpu_round_selection_audit_requested());
    }
    {
        scoped_env env("APXCHOL_GPU_ROUND_SELECTION_AUDIT", "maybe");
        EXPECT_THROW(apxchol::detail::gpu_round_selection_audit_requested(),
                     std::invalid_argument);
    }
}

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
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required for shape-discovering execution";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    const std::vector<undirected_edge> edges = {
        {0, 3, 1.0}, {0, 4, 2.0}, {0, 4, 3.0},
        {1, 3, 4.0}, {1, 5, 5.0}, {2, 4, 6.0},
        {2, 5, 7.0}, {3, 6, 8.0}, {4, 6, 9.0},
    };
    const std::vector<node_index> pivots = {0, 1, 2};
    const std::vector<double> excess = {1.0, 2.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0};
    const auto input = make_input(7, edges, pivots, 0x5eedULL, excess);
    const auto expected =
        apxchol::detail::reference_gpu_round_shadow(input);

    apxchol::detail::gpu_round_shadow_device_state state;
    const auto discovered = state.compute_discover_shape(input);
    EXPECT_TRUE(discovered.gpu_executed);
    apxchol::detail::compare_gpu_round_shadow_reports(expected, discovered);
    EXPECT_EQ(discovered.gathered_incidences,
              expected.gathered_incidences);
    EXPECT_EQ(discovered.unique_neighbors, expected.unique_neighbors);
    EXPECT_EQ(discovered.raw_fill_edges, expected.raw_fill_edges);
    EXPECT_EQ(discovered.excess_updates, expected.excess_updates);
    EXPECT_EQ(discovered.excess_targets, expected.excess_targets);
    EXPECT_EQ(discovered.surviving_input_incidences,
              expected.surviving_input_incidences);
    EXPECT_EQ(discovered.live_incidences, expected.live_incidences);
    EXPECT_EQ(discovered.factor_log_columns, pivots.size());
    EXPECT_EQ(discovered.factor_log_entries, 6u);
    const auto log = state.download_factor_log();
    ASSERT_EQ(log.columns.size(), 3u);
    ASSERT_EQ(log.entries.size(), 6u);
    const double degrees[] = {7.0, 11.0, 13.0};
    const node_index expected_neighbors[][2] = {{3, 4}, {3, 5}, {4, 5}};
    const double expected_weights[][2] = {{1.0, 5.0}, {4.0, 5.0}, {6.0, 7.0}};
    for (std::size_t column = 0; column < 3; ++column) {
        EXPECT_EQ(log.columns[column].vertex, pivots[column]);
        EXPECT_EQ(log.columns[column].diag,
                  static_cast<apxchol::factor_value_t>(
                      std::sqrt(degrees[column])));
        EXPECT_EQ(log.columns[column].entry_begin, 2 * column);
        EXPECT_EQ(log.columns[column].entry_count, 2u);
        for (std::size_t local = 0; local < 2; ++local) {
            const auto& entry = log.entries[2 * column + local];
            EXPECT_EQ(entry.neighbor, expected_neighbors[column][local]);
            EXPECT_EQ(entry.value,
                      static_cast<apxchol::factor_value_t>(
                          expected_weights[column][local] /
                          std::sqrt(degrees[column])));
        }
    }
#endif
}

TEST(GpuRoundShadowDevice, ConsumesBlockFrontendSelectionWithoutReupload) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required for device-selection handoff";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    constexpr node_index n = 8;
    const std::vector<undirected_edge> edges = {
        {0, 1, 1.0}, {1, 2, 2.0}, {2, 3, 3.0}, {3, 4, 4.0},
        {4, 5, 5.0}, {5, 6, 6.0}, {6, 7, 7.0}, {7, 0, 8.0},
        {0, 4, 2.5}, {1, 5, 3.5}, {2, 6, 4.5}, {3, 7, 5.5},
    };
    std::vector<apxchol::detail::gpu_topology_edge> topology;
    topology.reserve(edges.size());
    for (const auto& edge : edges)
        topology.push_back({edge.u, edge.v});

    apxchol::detail::gpu_block_frontend frontend(n, topology);
    std::vector<node_index> active(n);
    std::iota(active.begin(), active.end(), node_index{0});
    apxchol::partition_options options;
    options.degree_multiplier = 100.0;
    frontend.prepare(active, options);
    const auto& selected = frontend.select_block_greedy().data;
    ASSERT_FALSE(selected.empty());
    const auto device_selected = frontend.device_selection();

    constexpr std::uint64_t run_seed = 0x1234abcdULL;
    const auto input = make_input(n, edges, selected, run_seed);
    const auto expected =
        apxchol::detail::reference_gpu_round_shadow(input);
    apxchol::detail::gpu_round_shadow_device_state state;
    const auto actual = state.compute_discover_shape(
        input, device_selected, run_seed);
    apxchol::detail::compare_gpu_round_shadow_reports(expected, actual);
    EXPECT_EQ(actual.selection_validation_pivots, selected.size());
    EXPECT_EQ(actual.selection_audit_passes, 1u);
    EXPECT_EQ(actual.selection_audit_incidences, input.incidences.size());
#endif
}

TEST(GpuRoundShadowDevice, SelectionCapabilityDigestSpansMultipleBlocks) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required for device-selection digest";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    constexpr node_index n = 600;
    std::vector<undirected_edge> edges;
    std::vector<apxchol::detail::gpu_topology_edge> topology;
    for (node_index vertex = 0; vertex + 1 < n; ++vertex) {
        edges.push_back({vertex, vertex + 1, 1.0});
        topology.push_back({vertex, vertex + 1});
    }
    apxchol::detail::gpu_block_frontend frontend(n, topology);
    std::vector<node_index> active(n);
    std::iota(active.begin(), active.end(), node_index{0});
    apxchol::partition_options options;
    options.degree_multiplier = 100.0;
    frontend.prepare(active, options);
    const auto& selected = frontend.select_block_greedy().data;
    ASSERT_GT(selected.size(), 256u);

    constexpr std::uint64_t run_seed = 0x1234abcdULL;
    const auto input = make_input(n, edges, selected, run_seed);
    const auto expected =
        apxchol::detail::reference_gpu_round_shadow(input);
    apxchol::detail::gpu_round_shadow_device_state state;
    const auto actual = state.compute_discover_shape(
        input, frontend.device_selection(), run_seed);
    apxchol::detail::compare_gpu_round_shadow_reports(expected, actual);
#endif
}

TEST(GpuRoundShadowDevice, ChainsADeviceAuthoritativeGenerationWithoutHostState) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required for device-resident round chaining";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    constexpr node_index n = 9;
    constexpr unsigned run_seed = 0x98765432U;
    const std::vector<undirected_edge> edges = {
        {0, 1, 1.0}, {0, 2, 2.0}, {0, 3, 3.0},
        {1, 4, 4.0}, {1, 5, 5.0}, {2, 5, 6.0},
        {2, 6, 7.0}, {3, 6, 8.0}, {3, 7, 9.0},
        {4, 8, 2.5}, {5, 8, 3.5}, {6, 8, 4.5}, {7, 8, 5.5},
    };

    const std::vector<node_index> first_pivots = {0};
    const auto first_input = make_input(n, edges, first_pivots, run_seed);
    ASSERT_EQ(
        first_input.seeds.front() ^
            ((std::uint64_t(first_pivots.front()) + 1) *
             0x9E3779B97F4A7C15ULL),
        run_seed);
    std::vector<gpu_round_shadow_excess_bound> first_bounds;
    const auto first_expected =
        apxchol::detail::reference_gpu_round_shadow(first_input, &first_bounds);
    apxchol::detail::gpu_round_shadow_device_state state;
    const auto first = state.compute_discover_shape(first_input);
    apxchol::detail::compare_gpu_round_shadow_reports(first_expected, first);
    state.accept_device_generation(first.output_generation);
    const auto first_device_log = state.download_factor_log();
    auto cpu = run_authoritative_cpu_round(
        edges, first_input, first, first_bounds);
    expect_factor_logs_equal(cpu.factor_log, first_device_log);

    std::vector<apxchol::detail::gpu_topology_edge> topology;
    for (node_index owner = 0; owner < n; ++owner) {
        if (!cpu.residual.is_active(owner)) continue;
        for (const auto& [neighbor, weight] : cpu.residual.neighbors(owner)) {
            (void)weight;
            if (cpu.residual.is_active(neighbor) && owner < neighbor)
                topology.push_back({owner, neighbor});
        }
    }
    apxchol::detail::gpu_block_frontend frontend(n, topology);
    frontend.advance(first_pivots, {});
    std::vector<node_index> active;
    for (node_index vertex = 0; vertex < n; ++vertex)
        if (cpu.residual.is_active(vertex)) active.push_back(vertex);
    apxchol::partition_options options;
    options.degree_multiplier = 100.0;
    frontend.prepare(active, options);
    const auto& second_pivots = frontend.select_block_greedy().data;
    ASSERT_FALSE(second_pivots.empty());

    auto second_input = compact_live_snapshot(
        apxchol::detail::make_gpu_round_shadow_input(
            cpu.residual, second_pivots, run_seed));
    std::vector<gpu_round_shadow_excess_bound> second_bounds;
    const auto second_expected =
        apxchol::detail::reference_gpu_round_shadow(
            second_input, &second_bounds);
    scoped_env production_validation(
        "APXCHOL_GPU_ROUND_SELECTION_AUDIT", "off");
    const auto second = state.compute_resident(
        frontend.device_selection(), run_seed);
    apxchol::detail::compare_gpu_round_shadow_reports(
        second_expected, second);
    state.accept_device_generation(second.output_generation);
    apxchol::detail::gpu_round_shadow_factor_log second_cpu_log;
    EXPECT_NO_THROW((void)run_serial_cpu_round(
        cpu.residual, second_input, second, second_bounds,
        &second_cpu_log));
    const auto expected_cumulative_log =
        append_factor_logs(cpu.factor_log, second_cpu_log);
    const auto second_device_log = state.download_factor_log();
    expect_factor_logs_equal(expected_cumulative_log, second_device_log);
    EXPECT_TRUE(second.resident_input_reused);
    EXPECT_TRUE(second.resident_selection_consumed);
    EXPECT_EQ(second.selection_validation_pivots, second_pivots.size());
    EXPECT_EQ(second.selection_audit_passes, 0u);
    EXPECT_EQ(second.selection_audit_incidences, 0u);
    EXPECT_EQ(first.selection_map_initialization_vertices,
              static_cast<std::size_t>(n));
    EXPECT_EQ(second.selection_map_initialization_vertices, 0u);
    EXPECT_EQ(second.round_state_upload_bytes, 0u);
    EXPECT_EQ(second.state_imports, 1u);
    EXPECT_EQ(second.state_reuses, 1u);
    // The O(n) selected map was retained from the first round; first selection
    // adds only the two fixed status/digest allocations and grows nothing.
    EXPECT_EQ(second.state_buffer_allocations,
              first.state_buffer_allocations + 2);
    EXPECT_EQ(second.state_buffer_growths, first.state_buffer_growths);
    EXPECT_EQ(second.factor_log_columns,
              first_pivots.size() + second_pivots.size());
    EXPECT_EQ(second.factor_log_columns - first.factor_log_columns,
              second_pivots.size());
    EXPECT_EQ(second.factor_log_entries - first.factor_log_entries,
              second_cpu_log.entries.size());
    EXPECT_EQ(second_device_log.columns.size(), second.factor_log_columns);
    EXPECT_EQ(second_device_log.entries.size(), second.factor_log_entries);
#endif
}

TEST(GpuRoundShadowDevice,
     ResidentSelectionAuditAddsExactlyOneIncidenceScan) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required for resident-selection audit";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env selection_audit("APXCHOL_GPU_ROUND_SELECTION_AUDIT", "1");
    auto boundary = make_resident_selection_boundary();
    const auto expected_input =
        resident_selection_round_input(boundary.selected);
    const auto expected =
        apxchol::detail::reference_gpu_round_shadow(expected_input);
    const auto actual = boundary.state.compute_resident(
        boundary.device_selection, kResidentSelectionSeed);
    apxchol::detail::compare_gpu_round_shadow_reports(expected, actual);
    EXPECT_TRUE(actual.resident_selection_consumed);
    EXPECT_EQ(actual.selection_validation_pivots, boundary.selected.size());
    EXPECT_EQ(actual.selection_audit_passes, 1u);
    EXPECT_EQ(actual.selection_audit_incidences,
              actual.resident_input_incidences);
    EXPECT_EQ(actual.selection_map_initialization_vertices, 0u);
#endif
}

TEST(GpuRoundShadowDevice,
     ProductionSessionConsumesReusableResidentSelection) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required for resident production consumption";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env production_validation(
        "APXCHOL_GPU_ROUND_SELECTION_AUDIT", "off");
    constexpr node_index n = kResidentSelectionVertexCount;
    const auto edges = resident_selection_path_edges();
    apxchol::graph<apxchol::directed_vec_pool_incidence> graph(n);
    std::vector<apxchol::detail::gpu_topology_edge> topology;
    for (const auto& edge : edges) {
        graph.add_edge(edge.u, edge.v, edge.weight);
        topology.push_back({edge.u, edge.v});
    }

    apxchol::detail::gpu_round_shadow_session session(/*active=*/true);
    apxchol::detail::gpu_block_frontend frontend(n, topology);
    apxchol::factorize_workspace workspace;
    workspace.threads.resize(1);
    workspace.threads.front().factor_entries =
        std::make_unique<std::pmr::monotonic_buffer_resource>();
    apxchol::factor_options options;
    options.seed = kResidentSelectionSeed;
    std::vector<apxchol::detail::factor_col> columns;

    auto eliminate_round = [&](std::span<const node_index> selected,
                               const apxchol::detail::gpu_device_selection*
                                   device_selection) {
        workspace.reset_for_round();
        apxchol::partition_result partition;
        partition.data.assign(selected.begin(), selected.end());
        std::size_t work_hint = 0;
        for (node_index pivot : partition.data) {
            for (const auto& [neighbor, weight] : graph.neighbors(pivot)) {
                (void)weight;
                if (graph.is_active(neighbor)) ++work_hint;
            }
        }
        const std::size_t factor_base = columns.size();
        session.begin_round(
            graph, partition.data, options.seed, workspace.round_index,
            /*cpu_order_reproducible=*/true, device_selection);
        apxchol::detail::eliminate_partition(
            apxchol::detail::tree_elimination{}, graph, partition, columns,
            workspace, options, nullptr, false, work_hint);
        session.verify_cpu_round(
            graph, std::span<const apxchol::detail::factor_col>(columns)
                       .subspan(factor_base));
        ++workspace.round_index;
    };

    const std::vector<node_index> first = {0};
    eliminate_round(first, nullptr);
    frontend.advance(first, {});
    std::vector<node_index> active(n - 1);
    std::iota(active.begin(), active.end(), node_index{1});
    apxchol::partition_options partition_options;
    partition_options.degree_multiplier = 100.0;
    frontend.prepare(active, partition_options);
    const auto second = frontend.select_block_greedy().data;
    ASSERT_FALSE(second.empty());
    const auto device_selection = frontend.device_selection();
    testing::internal::CaptureStderr();
    eliminate_round(second, &device_selection);
    session.finish();
    const std::string trace = testing::internal::GetCapturedStderr();
    EXPECT_NE(trace.find("selection=resident"), std::string::npos) << trace;
    EXPECT_NE(trace.find(
                  "selection=resident selection_check=" +
                  std::to_string(second.size()) + "/0/0"),
              std::string::npos)
        << trace;
#endif
}

#if defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
TEST(GpuRoundShadowDevice,
     ResidentSelectionValidationRejectsUnsafeIdsAndPoisonsTheGeneration) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required for resident-selection validation";
#else
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
                    (void)boundary.state.compute_resident(
                        boundary.device_selection, kResidentSelectionSeed);
                },
                expected_message);
        };
        if (kind == corruption::adjacent) {
            scoped_env audit("APXCHOL_GPU_ROUND_SELECTION_AUDIT", "1");
            reject_malformed();
        } else {
            reject_malformed();
        }
        expect_exception_contains(
            [&] {
                (void)boundary.state.compute_resident(
                    boundary.device_selection, kResidentSelectionSeed);
            },
            "poisoned by a failed generation");
    }
#endif
}

TEST(GpuRoundShadowDevice,
     ResidentSelectionCapabilityRejectsMutatedIdsWithoutResidualScan) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required for selection-content validation";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env production_validation(
        "APXCHOL_GPU_ROUND_SELECTION_AUDIT", "off");
    auto boundary = make_resident_selection_boundary();
    auto reordered = boundary.selected;
    std::reverse(reordered.begin(), reordered.end());
    ASSERT_NE(reordered, boundary.selected);
    inject_device_selection_fault(*boundary.frontend, reordered);
    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_resident(
                boundary.device_selection, kResidentSelectionSeed);
        },
        "ids do not match the producer capability");
    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_resident(
                boundary.device_selection, kResidentSelectionSeed);
        },
        "poisoned by a failed generation");
#endif
}
#endif

TEST(GpuRoundShadowDevice, DeviceSelectionCapabilityCannotBeCallerForged) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required for resident-selection capabilities";
#else
    static_assert(!std::is_aggregate_v<
                  apxchol::detail::gpu_device_selection>);
    static_assert(!std::is_default_constructible_v<
                  apxchol::detail::gpu_device_selection_producer>);
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    auto boundary = make_resident_selection_boundary();
    const apxchol::detail::gpu_device_selection forged;
    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_resident(
                forged, kResidentSelectionSeed);
        },
        "was not issued by a producer");
    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_resident(
                boundary.device_selection, kResidentSelectionSeed);
        },
        "poisoned by a failed generation");
#endif
}

TEST(GpuRoundShadowDevice, ProducerRefusesAnEmptyDeviceSelectionCapability) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required for empty-selection publication";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    const std::vector<apxchol::detail::gpu_topology_edge> topology = {{0, 1}};
    apxchol::detail::gpu_block_frontend frontend(2, topology);
    const std::vector<node_index> active = {0, 1};
    apxchol::partition_options options;
    options.degree_quantile = 0.0;
    options.degree_multiplier = 0.0;
    const auto prepared = frontend.prepare(active, options);
    ASSERT_EQ(prepared.candidate_count, 0u);
    EXPECT_TRUE(frontend.select_block_greedy().data.empty());
    expect_exception_contains(
        [&] { (void)frontend.device_selection(); },
        "no current nonempty device selection");
    expect_exception_contains(
        [&] { (void)frontend.resident_region_capacity(); },
        "selection producer is poisoned");
#endif
}

TEST(GpuRoundShadowDevice, ProducerAndConsumerRejectTheWrongCudaDevice) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required for CUDA-device binding";
#else
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
        [&] { (void)producer.frontend->resident_region_capacity(); },
        "wrong CUDA device");
    ASSERT_EQ(cudaSetDevice(original_device), cudaSuccess);
    expect_exception_contains(
        [&] { (void)producer.frontend->device_selection(); },
        "selection producer is poisoned");

    auto consumer = make_resident_selection_boundary();
    ASSERT_EQ(cudaSetDevice(other_device), cudaSuccess);
    expect_exception_contains(
        [&] {
            (void)consumer.state.compute_resident(
                consumer.device_selection, kResidentSelectionSeed);
        },
        "active CUDA device differs from resident state device");
    ASSERT_EQ(cudaSetDevice(original_device), cudaSuccess);
    expect_exception_contains(
        [&] {
            (void)consumer.state.compute_resident(
                consumer.device_selection, kResidentSelectionSeed);
        },
        "poisoned by a failed generation");
#endif
}

TEST(GpuRoundShadowDevice,
     ResidentSelectionCapabilityRejectsStaleAndRetiredViews) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required for resident-selection capabilities";
#else
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
            (void)boundary.state.compute_resident(
                stale, kResidentSelectionSeed);
        },
        "stale device selection generation");
    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_resident(
                current, kResidentSelectionSeed);
        },
        "poisoned by a failed generation");

    auto retired = make_resident_selection_boundary();
    const auto retired_view = retired.device_selection;
    retired.frontend.reset();
    expect_exception_contains(
        [&] {
            (void)retired.state.compute_resident(
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
            (void)failed_producer.state.compute_resident(
                failed_view, kResidentSelectionSeed);
        },
        "selection producer is poisoned");
#endif
}

TEST(GpuRoundShadowDevice,
     ResidentSelectionRejectsConsumedTopologyAndDifferentProducer) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required for resident-selection provenance";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    apxchol::partition_options options;
    options.degree_multiplier = 100.0;

    auto consumed = make_resident_selection_boundary();
    const auto expected_input =
        resident_selection_round_input(consumed.selected);
    const auto expected =
        apxchol::detail::reference_gpu_round_shadow(expected_input);
    const auto actual = consumed.state.compute_resident(
        consumed.device_selection, kResidentSelectionSeed);
    apxchol::detail::compare_gpu_round_shadow_reports(expected, actual);
    consumed.state.accept_device_generation(actual.output_generation);
    expect_exception_contains(
        [&] {
            (void)consumed.state.compute_resident(
                consumed.device_selection, kResidentSelectionSeed);
        },
        "generation was already consumed");

    auto same_topology_boundary = make_resident_selection_boundary();
    const auto first = same_topology_boundary.state.compute_resident(
        same_topology_boundary.device_selection, kResidentSelectionSeed);
    same_topology_boundary.state.accept_device_generation(
        first.output_generation);
    (void)same_topology_boundary.frontend->select_block_greedy();
    const auto same_topology =
        same_topology_boundary.frontend->device_selection();
    expect_exception_contains(
        [&] {
            (void)same_topology_boundary.state.compute_resident(
                same_topology, kResidentSelectionSeed);
        },
        "topology generation was already consumed");

    auto cross_producer = make_resident_selection_boundary();
    const auto bound = cross_producer.state.compute_resident(
        cross_producer.device_selection, kResidentSelectionSeed);
    cross_producer.state.accept_device_generation(bound.output_generation);

    const auto path_edges = resident_selection_path_edges();
    std::vector<apxchol::detail::gpu_topology_edge> topology;
    for (const auto& edge : path_edges)
        topology.push_back({edge.u, edge.v});
    apxchol::detail::gpu_block_frontend other(
        kResidentSelectionVertexCount, topology);
    const std::vector<node_index> first_pivot = {0};
    other.advance(first_pivot, {});
    other.prepare(cross_producer.active, options);
    (void)other.select_block_greedy();
    expect_exception_contains(
        [&] {
            (void)cross_producer.state.compute_resident(
                other.device_selection(), kResidentSelectionSeed);
        },
        "different producer");
#endif
}

TEST(GpuRoundShadowDevice,
     FirstProducerBindRejectsAccidentalCrossStateReplay) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required for content-bound selection replay";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    auto boundary = make_resident_selection_boundary();
    auto wrong_edges = resident_selection_path_edges();
    wrong_edges.pop_back();
    std::vector<apxchol::detail::gpu_topology_edge> wrong_topology;
    for (const auto& edge : wrong_edges)
        wrong_topology.push_back({edge.u, edge.v});
    apxchol::detail::gpu_block_frontend wrong(
        kResidentSelectionVertexCount, wrong_topology);
    const std::vector<node_index> first_pivot = {0};
    wrong.advance(first_pivot, {});
    apxchol::partition_options options;
    options.degree_multiplier = 100.0;
    wrong.prepare(boundary.active, options);
    (void)wrong.select_block_greedy();
    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_resident(
                wrong.device_selection(), kResidentSelectionSeed);
        },
        "topology/active content does not match round state");
    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_resident(
                boundary.device_selection, kResidentSelectionSeed);
        },
        "poisoned by a failed generation");
#endif
}

TEST(GpuRoundShadowDevice,
     ResidentGenerationRequiresExactAuditAcceptance) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required for resident-generation acceptance";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    auto unaccepted = make_resident_selection_boundary(
        /*accept_initial_generation=*/false);
    expect_exception_contains(
        [&] {
            (void)unaccepted.state.compute_resident(
                unaccepted.device_selection, kResidentSelectionSeed);
        },
        "prior device generation was not explicitly accepted");
    expect_exception_contains(
        [&] {
            unaccepted.state.accept_device_generation(
                unaccepted.initial_generation);
        },
        "cannot accept a poisoned device generation");

    auto boundary = make_resident_selection_boundary(
        /*accept_initial_generation=*/false);
    expect_exception_contains(
        [&] {
            boundary.state.accept_device_generation(
                boundary.initial_generation + 1);
        },
        "acceptance is out of order");
    boundary.state.accept_device_generation(boundary.initial_generation);

    const auto expected_input =
        resident_selection_round_input(boundary.selected);
    const auto expected =
        apxchol::detail::reference_gpu_round_shadow(expected_input);
    const auto actual = boundary.state.compute_resident(
        boundary.device_selection, kResidentSelectionSeed);
    apxchol::detail::compare_gpu_round_shadow_reports(expected, actual);
    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_resident(
                boundary.device_selection, kResidentSelectionSeed);
        },
        "prior device generation was not explicitly accepted");
    expect_exception_contains(
        [&] {
            boundary.state.accept_device_generation(actual.output_generation);
        },
        "cannot accept a poisoned device generation");
#endif
}

TEST(GpuRoundShadowDevice,
     ContinuityFailurePoisonsPreviouslyAcceptedGeneration) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required for resident continuity poisoning";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    auto boundary = make_resident_selection_boundary();
    const auto certified_input = resident_selection_round_input({});
    boundary.state.certify_cpu_round(
        /*cpu_order_reproducible=*/true,
        apxchol::detail::fingerprint_gpu_round_shadow_input(certified_input),
        /*cpu_excess_may_differ=*/false);

    auto mismatched = certified_input;
    mismatched.excess[1] = 1.0;
    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_discover_shape(mismatched);
        },
        "excess digest after CPU certification");
    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_resident(
                boundary.device_selection, kResidentSelectionSeed);
        },
        "poisoned by a failed generation");
#endif
}

#if defined(APXCHOL_USE_CUDA) && \
    defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
TEST(GpuRoundShadowDevice,
     InjectedPostCudaFailurePoisonsPreviouslyAcceptedGeneration) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    auto boundary = make_resident_selection_boundary();
    boundary.state.inject_failure_after_cuda_operation_for_test();
    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_resident(
                boundary.device_selection, kResidentSelectionSeed);
        },
        "injected failure after CUDA operation");
    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_resident(
                boundary.device_selection, kResidentSelectionSeed);
        },
        "poisoned by a failed generation");
}
#endif

TEST(GpuRoundShadowDevice, RawIncidenceDegreeMatchesAuthoritativeCpu) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    const std::vector<undirected_edge> edges = {
        {0, 1, 1.0e16}, {0, 2, 1.0}, {0, 2, 1.0}, {1, 3, 4.0},
    };
    const std::vector<node_index> pivots = {0};
    const auto input = make_input(4, edges, pivots);
    std::vector<gpu_round_shadow_excess_bound> bounds;
    const auto device = run_device(input, &bounds);
    EXPECT_NO_THROW((void)run_authoritative_cpu_round(
        edges, input, device, bounds));
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
    auto cpu = run_authoritative_cpu_round(edges, input, device, bounds);

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
    const auto device = run_device(input, &bounds);
    const auto cpu = run_authoritative_cpu_round(edges, input, device, bounds);
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
        edges, input, baseline, bounds));
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
    const auto device = run_device(input, &bounds);
    EXPECT_NO_THROW((void)run_authoritative_cpu_round(
        edges, input, device, bounds));
}

TEST(GpuRoundShadowDevice,
     ResidentReuseRefreshAndParallelReimportHaveExactProvenance) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required for resident-state execution";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
#ifndef _OPENMP
    GTEST_SKIP() << "OpenMP build required for the parallel re-import case";
#else
    constexpr std::uint64_t run_seed = 0x5eed1234ULL;
    constexpr node_index vertex_count = kResidentProvenanceVertexCount;
    auto graph = make_resident_provenance_graph();

    apxchol::detail::gpu_round_shadow_device_state state;

    const std::vector<node_index> first_pivots = {0};
    const auto first_input =
        apxchol::detail::make_gpu_round_shadow_input(
            graph, first_pivots, run_seed);
    std::vector<gpu_round_shadow_excess_bound> first_bounds;
    const auto first_expected =
        apxchol::detail::reference_gpu_round_shadow(
            first_input, &first_bounds);
    const auto first = state.compute(first_input, first_expected);
    apxchol::detail::compare_gpu_round_shadow_reports(first_expected, first);
    EXPECT_FALSE(first.resident_input_reused);
    EXPECT_EQ(first.input_generation, 0u);
    EXPECT_EQ(first.output_generation, 1u);
    EXPECT_EQ(first.state_imports, 1u);
    EXPECT_EQ(first.state_reuses, 0u);
    EXPECT_GT(first.round_state_upload_bytes, 0u);
    const auto first_cpu = run_serial_cpu_round(
        graph, first_input, first, first_bounds);
    state.certify_cpu_round(
        /*cpu_order_reproducible=*/true, fingerprint_graph(graph),
        first_cpu.bounded_excess_vertices != 0);

    const std::vector<node_index> second_pivots = {1, 2};
    const auto second_input =
        apxchol::detail::make_gpu_round_shadow_input(
            graph, second_pivots, run_seed);
    std::vector<gpu_round_shadow_excess_bound> second_bounds;
    const auto second_expected =
        apxchol::detail::reference_gpu_round_shadow(
            second_input, &second_bounds);
    const auto second = state.compute(second_input, second_expected);
    apxchol::detail::compare_gpu_round_shadow_reports(second_expected, second);
    EXPECT_TRUE(second.resident_input_reused);
    EXPECT_EQ(second.input_generation, 1u);
    EXPECT_EQ(second.output_generation, 2u);
    EXPECT_EQ(second.state_imports, 1u);
    EXPECT_EQ(second.state_reuses, 1u);
    EXPECT_EQ(second.order_reimports, 0u);
    EXPECT_EQ(second.round_state_upload_bytes, 0u);
    EXPECT_EQ(second.state_upload_bytes, first.state_upload_bytes);
    EXPECT_EQ(second.state_buffer_allocations,
              first.state_buffer_allocations);
    EXPECT_EQ(second.state_buffer_growths, first.state_buffer_growths);
    EXPECT_EQ(second.resident_input_incidences, first.live_incidences);
    EXPECT_LE(second.resident_input_incidences,
              second.input_incidences);
    const auto second_cpu = run_serial_cpu_round(
        graph, second_input, second, second_bounds);
    ASSERT_GT(second_cpu.bounded_excess_vertices, 0u);
    ASSERT_EQ(second_bounds[3].additions, 2u);
    const double canonical_excess = second_bounds[3].expected;
    const double perturbed_excess =
        std::nextafter(canonical_excess,
                       std::numeric_limits<double>::infinity());
    const double terms = static_cast<double>(second_bounds[3].additions + 1);
    const double tolerance =
        32.0 * std::numeric_limits<double>::epsilon() * terms *
            second_bounds[3].absolute_term_sum +
        32.0 * std::numeric_limits<double>::denorm_min() * terms;
    ASSERT_LE(std::abs(perturbed_excess - canonical_excess), tolerance);
    graph.excess(3) = perturbed_excess;
    state.certify_cpu_round(
        /*cpu_order_reproducible=*/true, fingerprint_graph(graph),
        /*cpu_excess_may_differ=*/true);

    const std::vector<node_index> third_pivots = {3};
    const auto third_input =
        apxchol::detail::make_gpu_round_shadow_input(
            graph, third_pivots, run_seed);
    std::vector<gpu_round_shadow_excess_bound> third_bounds;
    const auto third_expected =
        apxchol::detail::reference_gpu_round_shadow(
            third_input, &third_bounds);
    const auto third = state.compute(third_input, third_expected);
    apxchol::detail::compare_gpu_round_shadow_reports(third_expected, third);
    EXPECT_TRUE(third.resident_input_reused);
    EXPECT_EQ(third.input_generation, 2u);
    EXPECT_EQ(third.output_generation, 3u);
    EXPECT_EQ(third.state_imports, 1u);
    EXPECT_EQ(third.state_reuses, 2u);
    EXPECT_EQ(third.order_reimports, 0u);
    EXPECT_EQ(third.excess_refreshes, 1u);
    EXPECT_EQ(third.round_state_upload_bytes,
              static_cast<std::size_t>(vertex_count) * sizeof(double));
    const auto third_cpu = run_serial_cpu_round(
        graph, third_input, third, third_bounds);
    state.certify_cpu_round(
        /*cpu_order_reproducible=*/true, fingerprint_graph(graph),
        third_cpu.bounded_excess_vertices != 0);

    std::vector<node_index> fourth_pivots(
        kResidentProvenanceParallelPivotEnd -
        kResidentProvenanceFirstParallelPivot);
    std::iota(fourth_pivots.begin(), fourth_pivots.end(),
              kResidentProvenanceFirstParallelPivot);
    const auto fourth_input =
        apxchol::detail::make_gpu_round_shadow_input(
            graph, fourth_pivots, run_seed);
    std::vector<gpu_round_shadow_excess_bound> fourth_bounds;
    const auto fourth_expected =
        apxchol::detail::reference_gpu_round_shadow(
            fourth_input, &fourth_bounds);
    const auto fourth = state.compute(fourth_input, fourth_expected);
    apxchol::detail::compare_gpu_round_shadow_reports(fourth_expected, fourth);
    EXPECT_TRUE(fourth.resident_input_reused);
    EXPECT_EQ(fourth.state_imports, 1u);
    EXPECT_EQ(fourth.state_reuses, 3u);
    EXPECT_EQ(fourth.order_reimports, 0u);
    const auto fourth_cpu = run_parallel_cpu_round(
        graph, fourth_input, fourth, fourth_bounds);
    const auto fourth_cpu_state = fingerprint_graph(graph);
    ASSERT_EQ(fourth_cpu_state.residual, fourth.residual);
    // Reimport is mandatory after parallel apply even if this scheduling
    // outcome happens to have the same ordered fingerprint as the device.
    state.certify_cpu_round(
        /*cpu_order_reproducible=*/false, fourth_cpu_state,
        fourth_cpu.comparison.bounded_excess_vertices != 0);

    const std::vector<node_index> fifth_pivots = {
        kResidentProvenanceParallelPivotEnd};
    const auto fifth_input =
        apxchol::detail::make_gpu_round_shadow_input(
            graph, fifth_pivots, run_seed);
    std::vector<gpu_round_shadow_excess_bound> fifth_bounds;
    const auto fifth_expected =
        apxchol::detail::reference_gpu_round_shadow(
            fifth_input, &fifth_bounds);
    const auto fifth = state.compute(fifth_input, fifth_expected);
    apxchol::detail::compare_gpu_round_shadow_reports(fifth_expected, fifth);
    EXPECT_FALSE(fifth.resident_input_reused);
    EXPECT_EQ(fifth.input_generation, 4u);
    EXPECT_EQ(fifth.output_generation, 5u);
    EXPECT_EQ(fifth.state_imports, 2u);
    EXPECT_EQ(fifth.state_reuses, 3u);
    EXPECT_EQ(fifth.order_reimports, 1u);
    EXPECT_GT(fifth.round_state_upload_bytes, 0u);
    const auto fifth_cpu = run_serial_cpu_round(
        graph, fifth_input, fifth, fifth_bounds);
    state.certify_cpu_round(
        /*cpu_order_reproducible=*/true, fingerprint_graph(graph),
        fifth_cpu.bounded_excess_vertices != 0);

    const std::size_t expected_columns =
        first_pivots.size() + second_pivots.size() + third_pivots.size() +
        fourth_pivots.size() + fifth_pivots.size();
    const std::size_t expected_entries =
        first.factor_entries + second.factor_entries + third.factor_entries +
        fourth.factor_entries + fifth.factor_entries;
    EXPECT_EQ(fifth.factor_log_columns, expected_columns);
    EXPECT_EQ(fifth.factor_log_entries, expected_entries);
    const auto factor_log = state.download_factor_log();
    ASSERT_EQ(factor_log.columns.size(), expected_columns);
    ASSERT_EQ(factor_log.entries.size(), expected_entries);
    EXPECT_EQ(factor_log.columns.front().vertex, first_pivots.front());
    EXPECT_EQ(factor_log.columns.back().vertex, fifth_pivots.back());
    for (const auto& column : factor_log.columns)
        EXPECT_LE(column.entry_begin + column.entry_count,
                  factor_log.entries.size());

    std::cout << "R2A_DEVICE_PROVENANCE commit=" << APXCHOL_GIT_SHA
              << " vertices=" << vertex_count
              << " generations=5 state_imports=" << fifth.state_imports
              << " state_reuses=" << fifth.state_reuses
              << " order_reimports=" << fifth.order_reimports
              << " excess_refreshes=" << fifth.excess_refreshes
              << " second_state_upload_bytes="
              << second.round_state_upload_bytes
              << " excess_refresh_bytes="
              << third.round_state_upload_bytes
              << " state_allocations="
              << fifth.state_buffer_allocations << '\n';
#endif
#endif
}

TEST(GpuRoundShadowDevice,
     ResidentComputeRejectsParallelOrderCertification) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required for resident transition rejection";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    auto boundary = make_resident_selection_boundary();
    const auto cpu_state = apxchol::detail::fingerprint_gpu_round_shadow_input(
        resident_selection_round_input({}));
    boundary.state.certify_cpu_round(
        /*cpu_order_reproducible=*/false, cpu_state,
        /*cpu_excess_may_differ=*/false);

    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_resident(
                boundary.device_selection, kResidentSelectionSeed);
        },
        "pending parallel-order reimport");
    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_resident(
                boundary.device_selection, kResidentSelectionSeed);
        },
        "poisoned by a failed generation");
#endif
}

TEST(GpuRoundShadowDevice,
     ResidentComputeRejectsBoundedExcessRefresh) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required for resident transition rejection";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    auto boundary = make_resident_selection_boundary();
    auto cpu_input = resident_selection_round_input({});
    const auto exact_state =
        apxchol::detail::fingerprint_gpu_round_shadow_input(cpu_input);
    cpu_input.excess[1] = std::nextafter(
        cpu_input.excess[1], std::numeric_limits<double>::infinity());
    const auto bounded_state =
        apxchol::detail::fingerprint_gpu_round_shadow_input(cpu_input);
    ASSERT_NE(bounded_state.excess, exact_state.excess);
    boundary.state.certify_cpu_round(
        /*cpu_order_reproducible=*/true, bounded_state,
        /*cpu_excess_may_differ=*/true);

    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_resident(
                boundary.device_selection, kResidentSelectionSeed);
        },
        "bounded excess refresh requiring a host snapshot");
    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_resident(
                boundary.device_selection, kResidentSelectionSeed);
        },
        "poisoned by a failed generation");
#endif
}

TEST(GpuRoundShadowDevice,
     ResidentComputeRejectsAuthoritativeHostRebuild) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required for resident transition rejection";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    auto boundary = make_resident_selection_boundary();
    const auto cpu_state = apxchol::detail::fingerprint_gpu_round_shadow_input(
        resident_selection_round_input({}));
    boundary.state.certify_cpu_round(
        /*cpu_order_reproducible=*/true, cpu_state,
        /*cpu_excess_may_differ=*/false);
    boundary.state.invalidate_for_authoritative_host_rebuild(cpu_state);

    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_resident(
                boundary.device_selection, kResidentSelectionSeed);
        },
        "pending authoritative host-rebuild reimport");
    expect_exception_contains(
        [&] {
            (void)boundary.state.compute_resident(
                boundary.device_selection, kResidentSelectionSeed);
        },
        "poisoned by a failed generation");
#endif
}

TEST(GpuRoundShadowDevice,
     CertifiedResidentStateRejectsAnUnannouncedHostMutation) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required for resident-state execution";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    constexpr std::uint64_t run_seed = 77;
    apxchol::graph<apxchol::directed_vec_pool_incidence> graph(6);
    for (const auto& edge : std::vector<undirected_edge>{
        {0, 1, 1.0}, {0, 2, 2.0}, {0, 3, 3.0},
        {1, 4, 4.0}, {2, 4, 5.0}, {2, 5, 6.0}, {3, 5, 7.0}})
        graph.add_edge(edge.u, edge.v, edge.weight);

    apxchol::detail::gpu_round_shadow_device_state state;
    const std::vector<node_index> first_pivots = {0};
    const auto first_input =
        apxchol::detail::make_gpu_round_shadow_input(
            graph, first_pivots, run_seed);
    std::vector<gpu_round_shadow_excess_bound> first_bounds;
    const auto first_expected =
        apxchol::detail::reference_gpu_round_shadow(
            first_input, &first_bounds);
    const auto first = state.compute(first_input, first_expected);
    apxchol::detail::compare_gpu_round_shadow_reports(first_expected, first);
    (void)run_serial_cpu_round(graph, first_input, first, first_bounds);
    state.certify_cpu_round(
        /*cpu_order_reproducible=*/true, fingerprint_graph(graph),
        // Exercise the fail-closed gate even when a caller has established a
        // bounded device-vs-CPU excess difference for this generation.
        /*cpu_excess_may_differ=*/true);

    const std::vector<node_index> second_pivots = {2};
    const auto unchanged = apxchol::detail::make_gpu_round_shadow_input(
        graph, second_pivots, run_seed);
    auto reordered = unchanged;
    bool changed = false;
    for (node_index owner = 0; owner < reordered.vertex_count && !changed;
         ++owner) {
        std::vector<std::size_t> live;
        for (std::size_t i = reordered.owner_offsets[owner];
             i < reordered.owner_offsets[owner + 1]; ++i)
            if (reordered.active[reordered.incidences[i].neighbor])
                live.push_back(i);
        if (live.size() >= 2) {
            std::swap(reordered.incidences[live[0]],
                      reordered.incidences[live[1]]);
            changed = true;
        }
    }
    ASSERT_TRUE(changed);
    const auto unchanged_fingerprint =
        apxchol::detail::fingerprint_gpu_round_shadow_input(unchanged);
    const auto reordered_fingerprint =
        apxchol::detail::fingerprint_gpu_round_shadow_input(reordered);
    ASSERT_EQ(unchanged_fingerprint.residual, reordered_fingerprint.residual);
    ASSERT_NE(unchanged_fingerprint.ordered_residual,
              reordered_fingerprint.ordered_residual);
    const auto unchanged_expected =
        apxchol::detail::reference_gpu_round_shadow(unchanged);
    const auto reordered_expected =
        apxchol::detail::reference_gpu_round_shadow(reordered);
    ASSERT_EQ(unchanged_expected.factor, reordered_expected.factor);
    EXPECT_THROW((void)state.compute(reordered, reordered_expected),
                 std::runtime_error);
    expect_exception_contains(
        [&] { (void)state.compute(unchanged, reordered_expected); },
        "poisoned by a failed generation");

    auto import_current_state = [&](
            apxchol::detail::gpu_round_shadow_device_state& target) {
        const std::vector<node_index> no_pivots;
        const auto input = apxchol::detail::make_gpu_round_shadow_input(
            graph, no_pivots, run_seed);
        const auto expected =
            apxchol::detail::reference_gpu_round_shadow(input);
        const auto report = target.compute(input, expected);
        apxchol::detail::compare_gpu_round_shadow_reports(expected, report);
        target.certify_cpu_round(
            /*cpu_order_reproducible=*/true, fingerprint_graph(graph),
            /*cpu_excess_may_differ=*/true);
    };

    apxchol::detail::gpu_round_shadow_device_state excess_state;
    import_current_state(excess_state);
    auto excess_changed = unchanged;
    ASSERT_TRUE(excess_changed.active[1]);
    excess_changed.excess[1] = 1.0;
    const auto excess_changed_expected =
        apxchol::detail::reference_gpu_round_shadow(excess_changed);
    EXPECT_THROW(
        (void)excess_state.compute(excess_changed, excess_changed_expected),
        std::runtime_error);
    expect_exception_contains(
        [&] {
            (void)excess_state.compute(unchanged, excess_changed_expected);
        },
        "poisoned by a failed generation");

    apxchol::detail::gpu_round_shadow_device_state rebuild_state;
    import_current_state(rebuild_state);
    rebuild_state.invalidate_for_authoritative_host_rebuild(
        apxchol::detail::fingerprint_gpu_round_shadow_input(reordered));
    const auto imported =
        rebuild_state.compute(reordered, reordered_expected);
    apxchol::detail::compare_gpu_round_shadow_reports(
        reordered_expected, imported);
    EXPECT_FALSE(imported.resident_input_reused);
    EXPECT_EQ(imported.state_imports, 2u);
    EXPECT_EQ(imported.state_reuses, 0u);
    EXPECT_EQ(imported.order_reimports, 0u);
    EXPECT_EQ(imported.host_rebuild_invalidations, 1u);
    EXPECT_EQ(imported.host_rebuild_reimports, 1u);
    EXPECT_GT(imported.round_state_upload_bytes, 0u);

    // Exercise the parallel-order reimport with a deliberately permuted valid
    // snapshot, independently of which order the host scheduler happened to
    // produce. The canonical factor is unchanged, but reuse must be denied.
    apxchol::detail::gpu_round_shadow_device_state parallel_state;
    auto ordered_snapshot = unchanged;
    ordered_snapshot.pivots.clear();
    ordered_snapshot.seeds.clear();
    const auto ordered_expected =
        apxchol::detail::reference_gpu_round_shadow(ordered_snapshot);
    const auto ordered = parallel_state.compute(ordered_snapshot, ordered_expected);
    apxchol::detail::compare_gpu_round_shadow_reports(ordered_expected, ordered);
    parallel_state.certify_cpu_round(
        /*cpu_order_reproducible=*/false, reordered_fingerprint,
        /*cpu_excess_may_differ=*/false);
    const auto parallel_imported =
        parallel_state.compute(reordered, reordered_expected);
    apxchol::detail::compare_gpu_round_shadow_reports(
        reordered_expected, parallel_imported);
    EXPECT_FALSE(parallel_imported.resident_input_reused);
    EXPECT_EQ(parallel_imported.state_imports, 2u);
    EXPECT_EQ(parallel_imported.order_reimports, 1u);
    EXPECT_EQ(parallel_imported.host_rebuild_reimports, 0u);
    EXPECT_GT(parallel_imported.round_state_upload_bytes, 0u);
    EXPECT_EQ(parallel_imported.factor_log_columns, reordered_expected.pivots.size());
    EXPECT_EQ(parallel_imported.factor_log_entries, reordered_expected.factor_entries);
#endif
}

TEST(GpuRoundShadowIntegration,
     ForcedTinyFactorizationVerifiesCpuRoundAndRemainsNoninterfering) {
    establish_r2a_openmp_affinity_if_requested();
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required for the forced integration path";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    // Preserve this historical test name because the R1/R2a audit parsers
    // name it explicitly. The fixture is now sized from the actual device
    // occupancy boundary: after one path-graph independent set, more than one
    // resident-region capacity remains, forcing factorize() itself through a
    // second GPU-selected round. One host worker makes the first CPU round
    // exactly order-reproducible, which is the production precondition for
    // compute_resident().
    scoped_env gpu_frontend("APXCHOL_GPU_BLOCK_FRONTEND", "force");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
    scoped_env selection_audit("APXCHOL_GPU_ROUND_SELECTION_AUDIT", "off");
    scoped_omp_threads one_cpu_worker(1);

    const std::size_t resident_capacity = [] {
        apxchol::detail::gpu_block_frontend probe(
            1, std::span<const apxchol::detail::gpu_topology_edge>{});
        return probe.resident_region_capacity();
    }();
    ASSERT_GT(resident_capacity, 0u);
    ASSERT_LE(resident_capacity,
              static_cast<std::size_t>(INT_MAX - 1) / 4);
    const std::size_t fixture_size = std::bit_ceil(
        std::max<std::size_t>(64, 4 * resident_capacity));
    ASSERT_LT(fixture_size, static_cast<std::size_t>(INT_MAX));
    const node_index n = static_cast<node_index>(fixture_size);
    const auto make_graph = [n] {
        apxchol::graph<apxchol::directed_vec_pool_incidence> graph(n);
        for (node_index vertex = 0; vertex + 1 < n; ++vertex) {
            graph.add_edge(
                vertex, vertex + 1,
                1.0 + static_cast<double>((vertex * 7) % 13));
        }
        return graph;
    };
    apxchol::factor_options options;
    options.seed = 0x12345678U;
    options.partition.degree_quantile = 0.0;
    options.partition.degree_multiplier = 100.0;
    options.min_is_fraction = 0.0;
    options.omp_threshold = std::numeric_limits<std::size_t>::max();
    options.parallel_residual_threshold = 1;
    const auto baseline = [&] {
        scoped_env env("APXCHOL_GPU_ROUND_SHADOW", "off");
        return apxchol::factorize<apxchol::block_greedy_partitioner>(
            make_graph(), options);
    }();
    apxchol::checkpoint shadow_checkpoint;
    std::string shadow_trace;
    const auto shadowed = [&] {
        scoped_env env("APXCHOL_GPU_ROUND_SHADOW", "force");
        testing::internal::CaptureStderr();
        try {
            auto factor =
                apxchol::factorize<apxchol::block_greedy_partitioner>(
                    make_graph(), options, &shadow_checkpoint);
            shadow_trace = testing::internal::GetCapturedStderr();
            return factor;
        } catch (...) {
            shadow_trace = testing::internal::GetCapturedStderr();
            throw;
        }
    }();

    ASSERT_EQ(baseline.perm, shadowed.perm);
    ASSERT_EQ(baseline.L.cols(), shadowed.L.cols());
    ASSERT_EQ(baseline.L.nonZeros(), shadowed.L.nonZeros());
    const std::size_t columns = static_cast<std::size_t>(baseline.L.cols()) + 1;
    const std::size_t entries =
        static_cast<std::size_t>(baseline.L.nonZeros());
    EXPECT_EQ(std::memcmp(baseline.L.outerIndexPtr(),
                          shadowed.L.outerIndexPtr(),
                          columns * sizeof(apxchol::edge_index)), 0);
    EXPECT_EQ(std::memcmp(baseline.L.innerIndexPtr(),
                          shadowed.L.innerIndexPtr(),
                          entries * sizeof(apxchol::node_index)), 0);
    EXPECT_EQ(std::memcmp(baseline.L.valuePtr(), shadowed.L.valuePtr(),
                          entries * sizeof(apxchol::factor_value_t)), 0);

    ASSERT_GE(shadowed.rounds.size(), 2u);
    ASSERT_EQ(baseline.rounds.size(), shadowed.rounds.size());
    for (std::size_t round = 0; round < shadowed.rounds.size(); ++round) {
        EXPECT_EQ(baseline.rounds[round].active,
                  shadowed.rounds[round].active);
        EXPECT_EQ(baseline.rounds[round].is_size,
                  shadowed.rounds[round].is_size);
    }

    const auto trace_lines = gpu_round_trace_lines(shadow_trace);
    ASSERT_EQ(trace_lines.size(), shadowed.rounds.size()) << shadow_trace;
    ASSERT_EQ(gpu_round_trace_field(trace_lines[0], "state="),
              "host-import");
    ASSERT_EQ(gpu_round_trace_field(trace_lines[0], "selection="),
              "snapshot");
    ASSERT_EQ(gpu_round_trace_field(trace_lines[1], "state="), "resident");
    ASSERT_EQ(gpu_round_trace_field(trace_lines[1], "selection="),
              "resident");
    EXPECT_EQ(gpu_round_trace_field(trace_lines[1], "generation="), "1->2");
    EXPECT_EQ(gpu_round_trace_size(trace_lines[1], "state_upload_bytes="), 0u);
    EXPECT_EQ(gpu_round_trace_size(trace_lines[1], "state_imports="), 1u);
    EXPECT_EQ(gpu_round_trace_size(trace_lines[1], "state_reuses="), 1u);
    EXPECT_EQ(gpu_round_trace_field(trace_lines[1], "selection_check="),
              std::to_string(shadowed.rounds[1].is_size) + "/0/0");

    // Replay the exact first two factorization selections through the
    // independent reference and canonical serial CPU round. This checks the
    // device trace's factor/fill/residual/degree digests, while the assembled
    // factor comparison below checks every selected id and neighbor endpoint.
    auto replay_graph = make_graph();
    const auto elimination_order = factor_elimination_order(shadowed);
    std::size_t factor_column_base = 0;
    std::size_t cumulative_factor_entries = 0;
    std::size_t first_growths = 0;
    for (std::size_t round = 0; round < 2; ++round) {
        const std::size_t round_size = shadowed.rounds[round].is_size;
        ASSERT_GT(round_size, 0u);
        ASSERT_LE(factor_column_base + round_size,
                  elimination_order.size());
        const std::vector<node_index> pivots(
            elimination_order.begin() + factor_column_base,
            elimination_order.begin() + factor_column_base + round_size);
        const auto input = apxchol::detail::make_gpu_round_shadow_input(
            replay_graph, pivots, options.seed);
        std::vector<gpu_round_shadow_excess_bound> bounds;
        const auto expected = apxchol::detail::reference_gpu_round_shadow(
            input, &bounds);
        apxchol::detail::gpu_round_shadow_factor_log cpu_factor_log;
        const auto cpu = run_serial_cpu_round(
            replay_graph, input, expected, bounds, &cpu_factor_log);
        EXPECT_EQ(cpu.bounded_excess_vertices, 0u);
        expect_factor_log_matches_assembled(
            shadowed, factor_column_base, cpu_factor_log);

        const auto& line = trace_lines[round];
        EXPECT_EQ(gpu_round_trace_size(line, "round="), round);
        EXPECT_EQ(gpu_round_trace_size(line, "pivots="), round_size);
        EXPECT_EQ(gpu_round_trace_size(line, "active="),
                  expected.active_count);
        expect_digest_equal(
            expected.factor, gpu_round_trace_digest(line, "factor="));
        expect_digest_equal(
            expected.fill, gpu_round_trace_digest(line, "fill_hash="));
        expect_digest_equal(
            expected.residual, gpu_round_trace_digest(line, "residual="));
        expect_digest_equal(
            expected.live_degree, gpu_round_trace_digest(line, "degree="));

        cumulative_factor_entries += expected.factor_entries;
        EXPECT_EQ(gpu_round_trace_size_pair(line, "factor_log="),
                  (std::pair{factor_column_base + round_size,
                             cumulative_factor_entries}));
        const std::size_t growths =
            gpu_round_trace_size(line, "factor_log_growths=");
        if (round == 0) {
            first_growths = growths;
            EXPECT_EQ(growths, 0u);
        } else {
            // The power-of-two path fixture fills the first exact-size
            // geometric allocation in round one; appending round two must
            // preserve it through one real factor-entry-log growth.
            EXPECT_EQ(growths, first_growths + 1);
            EXPECT_EQ(gpu_round_trace_size(line, "resident_input="),
                      gpu_round_trace_size(trace_lines[0], "live="));
        }
        factor_column_base += round_size;
    }

    // This runtime assertion establishes that FORCE emits the enclosing
    // checkpoint.  Its exact source placement around begin_round() is reviewed
    // separately: a positive duration alone cannot prove which side of that
    // call owns the tick, and no performance conclusion is drawn from it.
    EXPECT_GT(shadow_checkpoint.total("setup.gpu_round_shadow"), 0.0);

    struct rebuild_result {
        apxchol::detail::gpu_round_shadow_state_fingerprint state;
        std::vector<std::uint64_t> factor;
    };
    const auto run_rebuild_sequence = [&](const char* shadow_mode) {
        scoped_env env("APXCHOL_GPU_ROUND_SHADOW", shadow_mode);
        apxchol::graph<apxchol::directed_vec_pool_incidence> graph(8);
        for (node_index u = 0; u < graph.n(); ++u) {
            for (node_index v = u + 1; v < graph.n(); ++v)
                graph.add_edge(
                    u, v, 1.0 + static_cast<double>((13 * u + 7 * v) % 11));
        }

        apxchol::detail::tree_elimination eliminator;
        auto session = apxchol::detail::make_gpu_round_shadow_session<
            apxchol::detail::tree_elimination,
            apxchol::directed_vec_pool_incidence>(eliminator);
        apxchol::factorize_workspace workspace;
        workspace.threads.resize(1);
        workspace.threads.front().factor_entries =
            std::make_unique<std::pmr::monotonic_buffer_resource>();
        apxchol::factor_options sequence_options;
        sequence_options.seed = 0x12345678ULL;
        std::vector<apxchol::detail::factor_col> columns;
        std::vector<node_index> active(graph.n());
        std::iota(active.begin(), active.end(), node_index{0});

        const auto eliminate_one_round = [&](node_index pivot) {
            workspace.reset_for_round();
            apxchol::partition_result partition;
            partition.data = {pivot};
            std::size_t work_hint = 0;
            for (const auto& [neighbor, weight] : graph.neighbors(pivot)) {
                (void)weight;
                if (graph.is_active(neighbor)) ++work_hint;
            }
            const std::size_t factor_base = columns.size();
            session.begin_round(
                graph, partition.data, sequence_options.seed,
                workspace.round_index, /*cpu_order_reproducible=*/true);
            apxchol::detail::eliminate_partition(
                eliminator, graph, partition, columns, workspace,
                sequence_options, nullptr, false, work_hint);
            session.verify_cpu_round(
                graph, std::span<const apxchol::detail::factor_col>(columns)
                           .subspan(factor_base));
            std::erase(active, pivot);
            ++workspace.round_index;
        };

        eliminate_one_round(0);
        const auto before_rebuild = fingerprint_graph(graph);
        const auto stats = apxchol::detail::residual_coalescer<
            apxchol::directed_vec_pool_incidence>::sparsify(
                graph, active, 1.0e-6, sequence_options.seed);
        EXPECT_LT(stats.kept_edges, stats.distinct_before);
        const auto after_rebuild = fingerprint_graph(graph);
        EXPECT_NE(before_rebuild.residual, after_rebuild.residual);
        session.authoritative_host_rebuild(graph);
        eliminate_one_round(1);
        session.finish();
        return rebuild_result{
            fingerprint_graph(graph), encode_factor_columns(columns)};
    };

    const auto rebuilt_baseline = run_rebuild_sequence("off");
    const auto rebuilt_shadow = run_rebuild_sequence("force");
    EXPECT_EQ(rebuilt_baseline.state.residual,
              rebuilt_shadow.state.residual);
    EXPECT_EQ(rebuilt_baseline.state.ordered_residual,
              rebuilt_shadow.state.ordered_residual);
    EXPECT_EQ(rebuilt_baseline.state.active, rebuilt_shadow.state.active);
    EXPECT_EQ(rebuilt_baseline.state.live_degree,
              rebuilt_shadow.state.live_degree);
    EXPECT_EQ(rebuilt_baseline.state.excess, rebuilt_shadow.state.excess);
    EXPECT_EQ(rebuilt_baseline.state.active_count,
              rebuilt_shadow.state.active_count);
    EXPECT_EQ(rebuilt_baseline.state.live_incidences,
              rebuilt_shadow.state.live_incidences);
    EXPECT_EQ(rebuilt_baseline.factor, rebuilt_shadow.factor);
#endif
}

#if defined(APXCHOL_USE_CUDA)
#include "apxchol/solver/sptrsv/cuda.h"
#include "apxchol/solver/sptrsv/omp.h"
#include "apxchol/solver/preconditioner.h"

TEST(GpuFactorFinalize, ResidentPrefixCpuTailPermutationAndGrounding) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env fp32("APXCHOL_SPTRSV_FP16", "0");
    scoped_env drop("APXCHOL_FACTOR_DROP", "0");
    scoped_env dataflow("APXCHOL_GPU_SPTRSV", "dataflow");
    scoped_env segmentation("APXCHOL_GPU_DF_SPLIT", "1");
    const std::vector<undirected_edge> edges = {{1, 0, 2.0}, {1, 3, 3.0}};
    const std::vector<node_index> pivots = {1};
    const auto input = make_input(4, edges, pivots);
    apxchol::detail::gpu_round_shadow_device_state state;
    auto report = state.compute_discover_shape(input);
    apxchol::detail::compare_gpu_round_shadow_reports(
        apxchol::detail::reference_gpu_round_shadow(input), report);
    state.accept_device_generation(report.output_generation);
    const std::vector<node_index> perm = {2, 0, 3, 1};
    apxchol::detail::gpu_round_shadow_factor_log tail;
    tail.columns = {{3, 2.0f, 0, 2}, {0, 1.0f, 2, 1}, {2, 1.0f, 3, 0}};
    tail.entries = {{0, 0.3f}, {2, 0.0f}, {2, 0.2f}};
    // Debug materialization belongs only to this independent reference test.
    auto prefix = state.download_factor_log();
    std::vector<std::vector<apxchol::detail::factor_entry>> entries(4);
    std::vector<apxchol::detail::factor_col> cols;
    for (int i = 0; i < 4; ++i) {
        const auto& c = i == 0 ? prefix.columns[0] : tail.columns[i - 1];
        const auto& es = i == 0 ? prefix.entries : tail.entries;
        for (unsigned j = 0; j < c.entry_count; ++j)
            entries[i].push_back({es[c.entry_begin + j].neighbor, es[c.entry_begin + j].value});
        cols.push_back({c.vertex, c.diag, entries[i].data(), node_index(entries[i].size())});
    }
    apxchol::factorization reference;
    apxchol::detail::build_csc(reference, cols, 4, nullptr);
    ASSERT_EQ(reference.perm, perm);
    for (node_index m : {3u, 4u}) {
        apxchol::cuda_sptrsv ordinary, adopted;
        ordinary.setup(reference.L, m);
        auto factor = state.finalize_fp32(perm, m, tail);
        ASSERT_FALSE(factor->empty());
        adopted.setup_adopting_device_factor_for_research(std::move(*factor));
        ASSERT_TRUE(factor->empty());
        EXPECT_TRUE(adopted.adopted_device_factor());
        EXPECT_EQ(adopted.adoption_host_download_bytes(), 2 * (m + 1) * sizeof(int));
        EXPECT_EQ(adopted.stored_nnz(), ordinary.stored_nnz());
        EXPECT_EQ(adopted.drop_stats().dropped_flush, ordinary.drop_stats().dropped_flush);
        apxchol::omp_sptrsv cpu;
        cpu.setup(reference.L, m);
        for (int k = 0; k < 5; ++k) {
            std::vector<double> a(m), b(m), c(m), scratch(m);
            for (node_index i = 0; i < m; ++i) a[i] = b[i] = c[i] = std::sin(i + 3 * k + 0.1);
            ordinary.solve_LLt(a.data(), a.data()); adopted.solve_LLt(b.data(), b.data());
            cpu.forward_solve(c.data(), scratch.data()); cpu.transpose_solve(scratch.data(), c.data());
            for (node_index i = 0; i < m; ++i) {
                EXPECT_EQ(a[i], b[i]);
                EXPECT_NEAR(b[i], c[i], 2e-6 * std::max(1.0, std::abs(c[i])));
            }
        }
    }
}

TEST(GpuFactorFinalize, InvalidCoveragePermutationAndCoordinatesFailClosed) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env fp32("APXCHOL_SPTRSV_FP16", "0");
    scoped_env drop("APXCHOL_FACTOR_DROP", "0");
    const std::vector<undirected_edge> edges = {{0, 1, 1}, {0, 2, 2}};
    const std::vector<node_index> pivots = {0};
    apxchol::detail::gpu_round_shadow_device_state state;
    auto report = state.compute_discover_shape(make_input(3, edges, pivots));
    const std::vector<node_index> perm = {0, 1, 2};
    apxchol::detail::gpu_round_shadow_factor_log tail;
    tail.columns = {{1, 2.0f, 0, 1}, {2, 1.0f, 1, 0}};
    tail.entries = {{2, 0.2f}};
    EXPECT_THROW(state.finalize_fp32(perm, 3, tail), std::logic_error);
    state.accept_device_generation(report.output_generation);
    EXPECT_THROW(state.finalize_fp32(std::vector<node_index>{0, 1, 1}, 3, tail), std::invalid_argument);
    auto bad = tail; bad.columns.pop_back();
    EXPECT_THROW(state.finalize_fp32(perm, 3, bad), std::invalid_argument);
    bad = tail; bad.entries[0].neighbor = 0;
    EXPECT_THROW(state.finalize_fp32(perm, 3, bad), std::invalid_argument);
    bad = tail; bad.entries.push_back(bad.entries[0]); bad.columns[0].entry_count = 2;
    EXPECT_THROW(state.finalize_fp32(perm, 3, bad), std::invalid_argument);
    bad = tail; bad.columns[0].diag = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(state.finalize_fp32(perm, 3, bad), std::invalid_argument);
    { scoped_env unsupported("APXCHOL_SPTRSV_FP16", "1");
      EXPECT_THROW(state.finalize_fp32(perm, 3, tail), std::invalid_argument); }
    { scoped_env unsupported("APXCHOL_FACTOR_DROP", "0.1");
      EXPECT_THROW(state.finalize_fp32(perm, 3, tail), std::invalid_argument); }
    // Failed read-only finalization must leave the audited prefix reusable.
    EXPECT_NO_THROW(state.finalize_fp32(perm, 3, tail));
    cudaDeviceSynchronize();
    std::size_t before = 0, after = 0, total = 0;
    ASSERT_EQ(cudaMemGetInfo(&before, &total), cudaSuccess);
    for (int i = 0; i < 8; ++i) EXPECT_NO_THROW(state.finalize_fp32(perm, 3, tail));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemGetInfo(&after, &total), cudaSuccess);
    EXPECT_EQ(before, after);
}

TEST(GpuFactorFinalize, NormalPreconditionerInstallsAndReplacesResidentFactors) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env fp32("APXCHOL_SPTRSV_FP16", "0");
    scoped_env drop("APXCHOL_FACTOR_DROP", "0");
    scoped_env shadow("APXCHOL_GPU_ROUND_SHADOW", "force");
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env frontend("APXCHOL_GPU_BLOCK_FRONTEND", "0");
    scoped_omp_threads serial(1);
    apxchol::apx_cholesky preconditioner;
    preconditioner.set_keep_factor(true);
    constexpr int n = 64;
    Eigen::SparseMatrix<double> A(n, n);
    std::vector<Eigen::Triplet<double>> entries;
    for (int i = 0; i < n; ++i) {
        entries.emplace_back(i, i, 5.0); // SDDM, including the last column.
        for (int offset : {1, 8}) {
            const int j = (i + offset) % n;
            entries.emplace_back(i, j, -1.0); entries.emplace_back(j, i, -1.0);
        }
    }
    A.setFromTriplets(entries.begin(), entries.end());
    for (int repetition = 0; repetition < 2; ++repetition) {
        preconditioner.compute(A);
        ASSERT_TRUE(preconditioner.trsv().adopted_device_factor());
        ASSERT_TRUE(preconditioner.factor().sddm);
        const auto& F = preconditioner.factor();
        apxchol::omp_sptrsv cpu;
        cpu.setup(F.L, n);
        Eigen::VectorXd b(n), expected(n), tmp(n), work(n);
        for (int i = 0; i < n; ++i) { b[i] = std::cos(i + 0.5); work[F.perm[i]] = b[i]; }
        cpu.forward_solve(work.data(), tmp.data()); cpu.transpose_solve(tmp.data(), work.data());
        for (int i = 0; i < n; ++i) expected[i] = work[F.perm[i]];
        const Eigen::VectorXd observed = preconditioner.solve(b);
        EXPECT_LT((observed - expected).norm() / expected.norm(), 2e-6);
    }
}

TEST(GpuFactorFinalize, ConsumingSolveOmitsHostArraysAndPreservesExplicitExports) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env fp32("APXCHOL_SPTRSV_FP16", "0");
    scoped_env drop("APXCHOL_FACTOR_DROP", "0");
    scoped_env shadow("APXCHOL_GPU_ROUND_SHADOW", "force");
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env frontend("APXCHOL_GPU_BLOCK_FRONTEND", "0");
    scoped_omp_threads serial(1);
    constexpr int n = 64;
    for (double shift : {0.0, 1.0}) {
        SCOPED_TRACE(shift == 0 ? "grounded Laplacian" : "full SDDM");
        Eigen::SparseMatrix<double> A(n, n);
        std::vector<Eigen::Triplet<double>> entries;
        for (int i = 0; i < n; ++i) {
            entries.emplace_back(i, i, 4.0 + shift);
            for (int offset : {1, 8}) {
                int j = (i + offset) % n;
                entries.emplace_back(i, j, -1.0); entries.emplace_back(j, i, -1.0);
            }
        }
        A.setFromTriplets(entries.begin(), entries.end());
        // The public factorization API remains fully exportable under FORCE.
        auto reference = apxchol::factorize(A, apxchol::graph_storage::vec_pool_aos);
        ASSERT_EQ(reference.L.vals_.size(), reference.L.nonZeros());
        ASSERT_EQ(reference.L.inner_.size(), reference.L.nonZeros());
        apxchol::apx_cholesky exported;
        exported.set_keep_factor(true);
        exported.set_factor(reference);
        ASSERT_EQ(exported.factor().L.vals_.size(), reference.L.nonZeros());
        ASSERT_FALSE(exported.trsv().adopted_device_factor());
        // Copied public factors do not mutate their shared capsule. A uniquely
        // moved factor retains the fast adoption path.
        apxchol::apx_cholesky exported_again;
        exported_again.set_factor(reference);
        EXPECT_FALSE(exported_again.trsv().adopted_device_factor());
        auto moved_factor = apxchol::factorize(A, apxchol::graph_storage::vec_pool_aos);
        apxchol::apx_cholesky moved_export;
        moved_export.set_factor(std::move(moved_factor));
        ASSERT_TRUE(moved_export.trsv().adopted_device_factor());

        apxchol::checkpoint cp;
        apxchol::apx_cholesky consuming;
        consuming.set_checkpoint(&cp);
        consuming.compute(A);
        ASSERT_TRUE(consuming.trsv().adopted_device_factor());
        const auto& F = consuming.factor();
        EXPECT_TRUE(F.L.vals_.empty()); EXPECT_TRUE(F.L.inner_.empty());
        EXPECT_EQ(F.perm, reference.perm);
        EXPECT_EQ(F.L.outer_, reference.L.outer_);
        EXPECT_EQ(F.L.nonZeros(), reference.L.nonZeros());
        EXPECT_EQ(cp.total("setup.assembly"), 0.0);
        EXPECT_GT(cp.total("setup.factor_metadata"), 0.0);
        Eigen::VectorXd b(n);
        for (int i = 0; i < n; ++i) b[i] = std::sin(i + 0.25);
        b.array() -= b.mean();
        const Eigen::VectorXd expected = exported.solve(b);
        const Eigen::VectorXd observed = consuming.solve(b);
        const Eigen::VectorXd copied_export = exported_again.solve(b);
        EXPECT_EQ(std::memcmp(expected.data(), observed.data(), n * sizeof(double)), 0);
        EXPECT_EQ(std::memcmp(expected.data(), copied_export.data(), n * sizeof(double)), 0);
        const Eigen::VectorXd moved_result = moved_export.solve(b);
        EXPECT_EQ(std::memcmp(expected.data(), moved_result.data(), n * sizeof(double)), 0);
        auto install_copy = [reference, b]() mutable {
            apxchol::apx_cholesky solver;
            solver.set_factor(std::move(reference));
            const bool adopted = solver.trsv().adopted_device_factor();
            Eigen::VectorXd result = solver.solve(b);
            return std::pair{adopted, std::move(result)};
        };
        auto first = std::async(std::launch::async, install_copy);
        auto second = std::async(std::launch::async, install_copy);
        for (auto* task : {&first, &second}) {
            auto [adopted, result] = task->get();
            EXPECT_FALSE(adopted);
            EXPECT_EQ(std::memcmp(expected.data(), result.data(), n * sizeof(double)), 0);
        }

        // FORCE-off continues through ordinary assembly and upload, even for
        // a consuming solver; changing the internal call route must not alter it.
        scoped_env ordinary_mode("APXCHOL_GPU_FACTOR_FINALIZE", "off");
        apxchol::checkpoint ordinary_cp;
        apxchol::apx_cholesky ordinary;
        ordinary.set_checkpoint(&ordinary_cp);
        ordinary.compute(A);
        EXPECT_FALSE(ordinary.trsv().adopted_device_factor());
        EXPECT_GT(ordinary_cp.total("setup.assembly"), 0.0);
        EXPECT_EQ(ordinary_cp.total("setup.factor_metadata"), 0.0);
        const Eigen::VectorXd ordinary_result = ordinary.solve(b);
        EXPECT_EQ(std::memcmp(ordinary_result.data(), observed.data(), n * sizeof(double)), 0);
    }
}
#endif
