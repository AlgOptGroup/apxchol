#include <gtest/gtest.h>
#include "apxchol/solver/solve.h"

#include "apxchol/checkpoint.h"
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

#if defined(APXCHOL_USE_CUDA)
void expect_setup_api_receipt(const std::string& trace,
                              const apxchol::apx_cholesky& solver) {
    const auto begin = trace.find("[gpu-setup-receipt] ");
    if (apxchol::detail::gpu_setup_diagnostics()) {
        ASSERT_NE(begin, std::string::npos) << trace;
        EXPECT_EQ(trace.find("[gpu-setup-receipt] ", begin + 1), std::string::npos);
        const auto line = trace.substr(begin, trace.find('\n', begin) - begin);
        EXPECT_EQ(gpu_round_trace_field(line, "diagnostics="),
                  "enabled");
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
                  factor.L.inner_.size() * sizeof(node_index) +
                  factor.L.vals_.size() * sizeof(apxchol::factor_value_t));
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
        if (value) setenv(name, value, 1);
        else unsetenv(name);
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

TEST(GpuRoundShadowReference, SetupDiagnosticsFollowExistingRuntimeSwitches) {
    scoped_env verbose("APXCHOL_VERBOSE", nullptr);
    scoped_env stage("APXCHOL_SPTRSV_SETUP_TRACE", nullptr);
    scoped_env block("APXCHOL_GPU_BLOCK_TRACE", nullptr);
    EXPECT_FALSE(apxchol::detail::gpu_setup_diagnostics());
    { scoped_env enabled("APXCHOL_SPTRSV_SETUP_TRACE", "1");
      EXPECT_TRUE(apxchol::detail::gpu_setup_diagnostics()); }
    { scoped_env enabled("APXCHOL_VERBOSE", "");
      EXPECT_TRUE(apxchol::detail::gpu_setup_diagnostics()); }
    { scoped_env disabled("APXCHOL_GPU_BLOCK_TRACE", "0");
      EXPECT_FALSE(apxchol::detail::gpu_setup_diagnostics()); }
    { scoped_env enabled("APXCHOL_GPU_BLOCK_TRACE", "1");
      EXPECT_TRUE(apxchol::detail::gpu_setup_diagnostics()); }
    EXPECT_FALSE(apxchol::detail::gpu_setup_diagnostics());
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
        Eigen::SparseMatrix<double> A(4, 4);
        std::vector<Eigen::Triplet<double>> entries = {
            {0, 0, 3}, {1, 1, 3}, {2, 2, 2}, {3, 3, 1},
            {1, 0, -1}, {2, 1, -2}};
        if (full) { entries.emplace_back(0, 1, -7); entries.emplace_back(1, 2, -9); }
        A.setFromTriplets(entries.begin(), entries.end());
        const auto graph = apxchol::make_graph<apxchol::graph<apxchol::directed_vec_pool_incidence>>(A);
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

TEST(GpuRoundShadowReference, StreamedFactorAuditNeedsNoPayloadAllocation) {
    scoped_omp_threads team(2);
    const std::vector<undirected_edge> edges = {
        {0, 2, 1.0}, {0, 2, 2.0}, {0, 3, 4.0}, {0, 4, 2.0},
        {1, 2, 3.0}, {1, 3, 5.0}, {1, 5, 7.0},
        {2, 4, 6.0}, {3, 5, 8.0}};
    for (const std::vector<node_index>& pivots :
         {std::vector<node_index>{0}, std::vector<node_index>{0, 1}}) {
        SCOPED_TRACE(pivots.size());
        apxchol::graph<apxchol::directed_vec_pool_incidence> graph(6);
        for (const auto& e : edges) graph.add_edge(e.u, e.v, e.weight);
        graph.excess(0) = 3.0;
        const auto input = apxchol::detail::make_gpu_round_shadow_input(graph, pivots, 42);
        std::vector<gpu_round_shadow_excess_bound> bounds;
        const auto expected = apxchol::detail::reference_gpu_round_shadow(input, &bounds);
        apxchol::factorize_workspace ws;
        ws.threads.resize(2);
        for (auto& t : ws.threads) {
            // Any factor payload allocation would throw, including the first
            // upstream chunk request made by a monotonic resource.
            t.factor_entries = std::make_unique<std::pmr::monotonic_buffer_resource>(
                std::pmr::null_memory_resource());
            t.retain_factor_payload = false;
        }
        apxchol::partition_result part;
        part.data = pivots;
        apxchol::factor_options opts;
        opts.seed = 42;
        opts.omp_threshold = 0;
        std::vector<apxchol::detail::factor_col> columns;
        ASSERT_NO_THROW(apxchol::detail::eliminate_partition(
            apxchol::detail::tree_elimination{}, graph, part, columns,
            ws, opts, nullptr, false, expected.gathered_incidences));
        apxchol::detail::gpu_round_shadow_digest digest;
        for (const auto& t : ws.threads) {
            digest.xor_hash ^= t.streamed_factor_entries[0];
            digest.sum_hash += t.streamed_factor_entries[1];
        }
        std::size_t entries = 0;
        for (const auto& c : columns) {
            EXPECT_EQ(c.entries, nullptr);
            entries += c.entry_count;
        }
        EXPECT_EQ(entries, expected.factor_entries);
        EXPECT_GT(entries, 0u);
        EXPECT_NO_THROW(apxchol::detail::compare_gpu_round_shadow_with_cpu(
            expected, bounds, graph, columns, &digest));
        EXPECT_THROW(apxchol::detail::compare_gpu_round_shadow_with_cpu(
            expected, bounds, graph, columns), std::runtime_error);
        digest.xor_hash ^= 1;
        EXPECT_THROW(apxchol::detail::compare_gpu_round_shadow_with_cpu(
            expected, bounds, graph, columns, &digest), std::runtime_error);
        ws.reset_for_round();
        for (const auto& t : ws.threads)
            EXPECT_EQ(t.streamed_factor_entries, (std::array<std::uint64_t, 2>{}));
    }
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
    scoped_env setup_trace("APXCHOL_SPTRSV_SETUP_TRACE", "1");
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
    if (apxchol::detail::gpu_setup_diagnostics()) {
        EXPECT_NE(trace.find("selection=resident"), std::string::npos) << trace;
        EXPECT_NE(trace.find(
                      "selection=resident selection_check=" +
                      std::to_string(second.size()) + "/0/0"),
                  std::string::npos)
            << trace;
    } else {
        EXPECT_TRUE(trace.empty()) << trace;
    }
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
        const auto actual = run_device(input, &bounds);
        EXPECT_NO_THROW((void)run_authoritative_cpu_round(edges, input, actual, bounds));
        // The zero-pivot report still hashes all residual incidences/state.
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

#if defined(APXCHOL_USE_CUDA)
namespace {
struct resident_selector_fixture {
    static constexpr node_index n = 34;
    static constexpr std::uint64_t seed = 71;
    apxchol::graph<apxchol::directed_vec_pool_incidence> graph{n};
    std::vector<apxchol::detail::gpu_topology_edge> initial;
    std::vector<node_index> active;
    apxchol::detail::gpu_round_shadow_device_state state;
    std::unique_ptr<apxchol::detail::gpu_block_frontend> frontend;
    apxchol::partition_options options;
    std::size_t rounds = 0;

    explicit resident_selector_fixture(bool with_edges = true) {
        scoped_omp_threads one(1);
        auto edge = [&](node_index u, node_index v, double w) {
            graph.add_edge(u, v, w);
            initial.push_back({u, v});
        };
        if (with_edges) {
        for (node_index v = 0; v < 24; ++v) edge(v, (v + 1) % 24, 1.0);
        edge(0, 1, 2.0); // Multigraph multiplicity must survive the projection.
        edge(0, 1, 4.0);
        for (node_index v = 24; v < 31; ++v) edge(v, v + 1, 1.0);
        }
        // Vertices 32 and 33 are isolated; 24..31 form a separate component.
        active.resize(n);
        std::iota(active.begin(), active.end(), node_index{0});
        frontend = std::make_unique<apxchol::detail::gpu_block_frontend>(n, initial);
        options.degree_multiplier = 100.0;
    }

    void round(bool certify = true) {
        scoped_omp_threads one(1);
        frontend->prepare(active, options);
        const auto selected = frontend->select_block_greedy().data;
        if (selected.empty()) throw std::logic_error("fixture exhausted too soon");
        const auto capability = frontend->device_selection();
        const auto input = apxchol::detail::make_gpu_round_shadow_input(graph, selected, seed);
        std::vector<gpu_round_shadow_excess_bound> bounds;
        const auto report = apxchol::detail::run_verified_gpu_round_shadow(
            state, input, &capability, seed, rounds != 0, &bounds);
        EXPECT_EQ(report.resident_selection_consumed, rounds != 0);
        if (rounds) EXPECT_EQ(report.round_state_upload_bytes, 0u);
        const auto cpu = run_serial_cpu_round(graph, input, report, bounds);
        EXPECT_EQ(cpu.bounded_excess_vertices, 0u);
        if (certify) state.certify_cpu_round(true, fingerprint_graph(graph));
        std::erase_if(active, [&](node_index v) { return !graph.is_active(v); });
        ++rounds;
    }

    // Rebuild a separate selector from the CPU graph solely as the oracle.
    // Its uploads are not inputs to frontend or the resident elimination state.
    void compare_reference() {
        const auto snapshot = apxchol::detail::make_gpu_round_shadow_input(
            graph, std::span<const node_index>{}, seed);
        std::vector<apxchol::detail::gpu_topology_edge> edges;
        for (const auto& e : snapshot.incidences)
            if (e.owner < e.neighbor && snapshot.active[e.owner] &&
                snapshot.active[e.neighbor]) edges.push_back({e.owner, e.neighbor});
        apxchol::detail::gpu_block_frontend reference(n, edges);
        std::vector<node_index> inactive;
        for (node_index v = 0; v < n; ++v)
            if (!graph.is_active(v)) inactive.push_back(v);
        reference.advance(inactive, {});
        const auto actual = frontend->prepare(active, options);
        const auto expected = reference.prepare(active, options);
        EXPECT_EQ(actual.candidate_count, expected.candidate_count);
        EXPECT_EQ(actual.average_degree, expected.average_degree);
        EXPECT_EQ(std::vector<node_index>(frontend->host_active_degrees().begin(),
                                         frontend->host_active_degrees().end()),
                  std::vector<node_index>(reference.host_active_degrees().begin(),
                                         reference.host_active_degrees().end()));
        EXPECT_EQ(frontend->select_block_greedy().data,
                  reference.select_block_greedy().data);
    }
};
} // namespace
#endif

#if defined(APXCHOL_USE_CUDA)
namespace {
constexpr node_index kOwnedVertices = 98;
constexpr std::uint64_t kOwnedSeed = 123;

apxchol::graph<apxchol::directed_vec_pool_incidence> make_owned_order_fixture() {
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
    apxchol::graph<apxchol::directed_vec_pool_incidence> graph(kOwnedVertices);
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

std::vector<apxchol::detail::gpu_topology_edge> owned_fixture_topology(
        const apxchol::graph<apxchol::directed_vec_pool_incidence>& graph) {
    std::vector<apxchol::detail::gpu_topology_edge> edges;
    for (node_index v = 0; v < graph.n(); ++v)
        for (const auto& e : graph.neighbors(v))
            if (v < e.to) edges.push_back({v, e.to});
    return edges;
}

struct owned_selector_view_fixture {
    apxchol::graph<apxchol::directed_vec_pool_incidence> graph = make_owned_order_fixture();
    std::vector<node_index> active;
    std::unique_ptr<apxchol::detail::gpu_round_shadow_session> session =
        std::make_unique<apxchol::detail::gpu_round_shadow_session>(true);
    apxchol::detail::gpu_block_frontend selector{graph.n(), owned_fixture_topology(graph)};
    apxchol::partition_options options;

    owned_selector_view_fixture() : active(graph.n()) {
        std::iota(active.begin(), active.end(), node_index{0});
        options.degree_quantile = 0.3;
    }

    std::vector<node_index> select() {
        selector.prepare(active, options);
        auto selected = selector.select_block_greedy().data;
        if (selected.empty()) throw std::logic_error("owned selector fixture exhausted too soon");
        return selected;
    }

    gpu_round_shadow_report run_selected(
            std::span<const node_index> selected,
            apxchol::detail::gpu_device_selection capability) {
        auto report = session->run_owned_prefix_round(graph, selected, capability, kOwnedSeed);
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

    void compare_reference(const apxchol::graph<apxchol::directed_vec_pool_incidence>& oracle) {
        ASSERT_GT(selector.transfers().owned_residual_binds, 0u);
        const auto prepared = selector.prepare(active, options);
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
        if (options.degree_quantile > 0.0 && options.degree_quantile < 1.0) {
            auto sorted = expected_degrees; std::sort(sorted.begin(), sorted.end());
            threshold = sorted[static_cast<std::size_t>(options.degree_quantile * (sorted.size()-1))];
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
            if (oracle.is_active(edge.to)) EXPECT_FALSE(picked[edge.to]);
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
            << std::bit_cast<std::uint64_t>(edge.weight);
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
    apxchol::graph<apxchol::directed_vec_pool_incidence> graph(n), oracle(n);
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
    apxchol::detail::gpu_block_frontend selector(n, owned_fixture_topology(graph));
    selector.prepare(active, options);
    const auto selected = selector.select_block_greedy().data;
    ASSERT_EQ(selected, wanted_selection);
    apxchol::detail::gpu_round_shadow_session session(true);
    const auto report = session.run_owned_prefix_round(
        graph, selected, selector.device_selection(), kOwnedSeed);
    ASSERT_EQ(report.raw_fill_edges, expected_fill);
    ASSERT_EQ(report.live_incidences, expected_live);
    ASSERT_TRUE(report.pivots.empty());  // This must exercise the owned route.

    std::vector<apxchol::detail::factor_col> columns;
    // This public boundary internally recomputes the post-sort device hash,
    // compares it with the saved topology, and checks the complete handback.
    session.materialize_owned_prefix(graph, active, columns);
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
        EXPECT_EQ(std::bit_cast<std::uint64_t>(actual.incidences[i].weight),
                  std::bit_cast<std::uint64_t>(expected.incidences[i].weight));
    }
    EXPECT_EQ(apxchol::detail::fingerprint_gpu_round_shadow_input(actual),
              apxchol::detail::fingerprint_gpu_round_shadow_input(expected));
    EXPECT_EQ(graph.m(), oracle.m());
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

Eigen::SparseMatrix<double> owned_solve_matrix(int n, double shift) {
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
    return A;
}
} // namespace
#endif

TEST(GpuOwnedPrefix, ResidualBoundaryHandbackMatchesEveryField) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
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
#endif
}


TEST(GpuOwnedPrefix, ResidualFillSortPreservesSurvivorOrder) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
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
#endif
}


TEST(GpuOwnedPrefix, ResidualFillCollisionsPreserveLaterRoundAssociation) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
    scoped_env fp32("APXCHOL_SPTRSV_FP16", "0");
    scoped_env drop("APXCHOL_FACTOR_DROP", "0");
    scoped_env tail("APXCHOL_RESIDUAL_SPARSIFY", "0");
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

        apxchol::graph<apxchol::directed_vec_pool_incidence> graph(4), oracle(4);
        for (const auto& edge : edges) {
            graph.add_edge(edge.u, edge.v, edge.weight);
            oracle.add_edge(edge.u, edge.v, edge.weight);
        }
        graph.excess(0) = oracle.excess(0) = 0.25;
        graph.excess(1) = oracle.excess(1) = 0.5;
        apxchol::detail::gpu_round_shadow_session session(true);
        apxchol::detail::gpu_block_frontend selector(4, owned_fixture_topology(graph));
        std::vector<node_index> active{0, 1, 2, 3};
        const std::vector<node_index> initial_collision{2, 3};
        std::vector<std::vector<node_index>> sequence;
        while (!active.empty()) {
            const auto round = sequence.size();
            ASSERT_LT(round, 3u);
            selector.prepare(active, options);
            const auto selected = selector.select_block_greedy().data;
            ASSERT_FALSE(selected.empty());
            EXPECT_EQ(selector.select_block_greedy().data, selected);
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
            const auto report = session.run_owned_prefix_round(
                graph, selected, selector.device_selection(), kOwnedSeed);
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
        ASSERT_EQ(reference_factor.L.inner_.size(), reference_factor.L.nonZeros());
        ASSERT_EQ(reference_factor.L.vals_.size(), reference_factor.L.nonZeros());
        apxchol::cuda_sptrsv reference_trsv, actual_trsv;
        reference_trsv.setup(reference_factor.L, 4);
        auto device_factor = session.finalize_device_factor(columns, reference_factor.perm, 4);
        ASSERT_NE(device_factor, nullptr);
        actual_trsv.setup_adopting_device_factor_for_research(std::move(*device_factor));
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
#endif
}


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

TEST(GpuOwnedPrefix, GenericGraphStillRejectsUnpairedStoredArcs) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_omp_threads serial(1);
    apxchol::graph<apxchol::directed_vec_pool_incidence> graph(2);
    const std::vector<node_index> vertices = {0, 1}, sizes = {1, 0};
    graph.adj_bulk_reserve_parallel(vertices.begin(), vertices.end(), sizes);
    graph.adj_write_reserved_directed_at(0, 0, 1, 1.0);
    graph.adj_commit_reserved_directed(0, 1);
    graph.adj_commit_reserved_directed(1, 0);
    graph.record_edges_added(1);
    const std::vector<apxchol::detail::gpu_topology_edge> topology = {{0, 1}};
    apxchol::detail::gpu_block_frontend selector(2, topology);
    selector.prepare(vertices, {});
    const auto selected = selector.select_block_greedy().data;
    ASSERT_FALSE(selected.empty());
    apxchol::detail::gpu_round_shadow_session session(true);
    // Omit the internal matrix-construction marker: arbitrary graph inputs
    // must continue to validate weighted pairing before any numerical round.
    EXPECT_THROW(session.run_owned_prefix_round(
        graph, selected, selector.device_selection(), 42), std::invalid_argument);
#endif
}

TEST(GpuOwnedPrefix, TwoRoundsPreserveOrderedHandbackAndCpuPeel) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env fp32("APXCHOL_SPTRSV_FP16", "0");
    scoped_env drop("APXCHOL_FACTOR_DROP", "0");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
    scoped_omp_threads serial(1);
    auto graph = make_owned_order_fixture();
    auto oracle = make_owned_order_fixture();
    const auto initial = fingerprint_graph(graph);
    auto topology = owned_fixture_topology(graph);
    apxchol::detail::gpu_block_frontend selector(graph.n(), topology);
    apxchol::detail::gpu_round_shadow_session session(true);
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
        reports.push_back(session.run_owned_prefix_round(
            graph, selected, selector.device_selection(), kOwnedSeed));
        session.advance_selector(selector);
        std::erase_if(active, [&](node_index v) {
            return std::binary_search(selected.begin(), selected.end(), v);
        });
        selections.push_back(std::move(selected));
    }
    // The oracle snapshot/check starts after both production-owned rounds.
    EXPECT_EQ(fingerprint_graph(graph), initial);
    EXPECT_EQ(reports[0].state_imports, 1u);
    EXPECT_GT(reports[0].round_state_upload_bytes, 0u);
    EXPECT_EQ(reports[1].state_imports, 1u);
    EXPECT_EQ(reports[1].state_reuses, 1u);
    EXPECT_EQ(reports[1].round_state_upload_bytes, 0u);
    EXPECT_EQ(reports[0].resident_input_incidences, initial.live_incidences);
    EXPECT_EQ(reports[1].resident_input_incidences, reports[0].live_incidences);
    for (const auto& report : reports)
        EXPECT_EQ(report.input_incidences, report.resident_input_incidences);
    EXPECT_NE(selections[0].size(), selections[1].size());
    EXPECT_EQ(selector.transfers().host_update_bytes, 0u);

    std::vector<apxchol::detail::factor_col> columns;
    session.materialize_owned_prefix(graph, active, columns);
    const std::size_t prefix = columns.size();
    ASSERT_EQ(prefix, selections[0].size() + selections[1].size());
    ASSERT_GT(active.size(), 0u);
    EXPECT_EQ(graph.num_active(), active.size());
    EXPECT_FALSE(graph.is_active(63)); EXPECT_FALSE(graph.is_active(64));
    EXPECT_EQ(graph.excess(63), 2.0); EXPECT_EQ(graph.excess(64), 3.0);
    EXPECT_FALSE(session.active()); // The CPU continuation must not replay prefix rounds.

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
    EXPECT_EQ(graph.m(), oracle.m());
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

    apxchol::factorize_workspace actual_ws, expected_ws;
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
    EXPECT_THROW(session.materialize_owned_prefix(graph, active, columns), std::logic_error);
    session.finish();
#endif
}

TEST(GpuOwnedPrefix, SharedResidualViewPreservesCandidatesAndIndependentProgress) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_omp_threads serial(1);
    for (const char* region_count : {"1", "7", static_cast<const char*>(nullptr)}) {
      scoped_env regions("APXCHOL_GPU_BLOCKS", region_count);
      SCOPED_TRACE(region_count);
      for (bool degree_tiebreak : {false, true}) {
       SCOPED_TRACE(degree_tiebreak);
       for (double quantile : {0.0, 0.3}) {
        SCOPED_TRACE(quantile);
        owned_selector_view_fixture f;
        f.options.degree_quantile = quantile;
        f.options.degree_tiebreak = degree_tiebreak;
        auto oracle = make_owned_order_fixture();
        for (std::size_t round = 0; round < 3; ++round) {
            SCOPED_TRACE(round);
            const auto selected = f.select();
            std::string receipt = std::to_string(f.selector.resident_region_capacity()) + ":" +
                std::to_string(f.selector.selected_degree_work());
            for (const auto v : selected) receipt += ":" + std::to_string(v);
            RecordProperty(std::string("owned_selection_") +
                (region_count ? region_count : "auto") + "_" +
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
            EXPECT_EQ(traffic.owned_residual_binds, round + 1);
            EXPECT_EQ(traffic.host_update_bytes, 0u);
            EXPECT_EQ(traffic.projection_scratch_peak_extra_bytes, 0u);
            if (round) EXPECT_EQ(report.round_state_upload_bytes, 0u);
        }
       }
      }
    }
#endif
}

TEST(GpuOwnedPrefix, SharedResidualViewRejectsPublicReadsAfterNextRound) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
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
            case 0: (void)f.selector.prepare(f.active, f.options); break;
            case 1: (void)f.selector.select_block_greedy(); break;
            case 2: (void)f.selector.host_candidates(); break;
            case 3: (void)f.selector.host_active_degrees(); break;
            case 4: (void)f.selector.device_selection(); break;
            }
        }, "GPU selector residual generation is retired");
        EXPECT_THROW(f.session->advance_selector(f.selector), std::exception);
    }
#endif
}

TEST(GpuOwnedPrefix, SharedResidualViewRebindsAfterNextRound) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
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
    EXPECT_EQ(f.selector.transfers().owned_residual_binds, 2u);
#endif
}

TEST(GpuOwnedPrefix, SharedResidualViewExpiresWithDestroyedOrReplacedOwner) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
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
#endif
}

TEST(GpuOwnedPrefix, FailedNextRoundRevokesSharedResidualView) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
    scoped_omp_threads serial(1);
    owned_selector_view_fixture f;
    const auto selected = f.select();
    const auto consumed = f.selector.device_selection();
    f.run_selected(selected, consumed);
    f.session->advance_selector(f.selector);
    (void)f.select();
    EXPECT_THROW(f.session->run_owned_prefix_round(
        f.graph, selected, consumed, kOwnedSeed), std::exception);
    expect_exception_contains([&] { (void)f.selector.prepare(f.active, f.options); },
                              "GPU selector residual generation is retired");
#endif
}

TEST(GpuOwnedPrefix, OwnedHandbackRevokesSharedResidualView) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
    scoped_omp_threads serial(1);
    owned_selector_view_fixture f;
    f.first_bound_round();
    (void)f.select();
    std::vector<apxchol::detail::factor_col> columns;
    ASSERT_NO_THROW(f.session->materialize_owned_prefix(f.graph, f.active, columns));
    ASSERT_FALSE(columns.empty());
    expect_exception_contains([&] { (void)f.selector.host_candidates(); },
                              "GPU selector residual generation is retired");
    EXPECT_THROW((void)f.selector.device_selection(), std::exception);
#endif
}

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
        actual.perm.resize(n); actual.L.resize(n, n);
        apxchol::detail::validate_owned_factor_columns(headers, entries, actual.perm, actual.L.outer_);
        EXPECT_EQ(actual.perm, expected.perm);
        EXPECT_EQ(actual.L.outer_, expected.L.outer_);
        EXPECT_EQ(actual.L.rows(), expected.L.rows());
        EXPECT_EQ(actual.L.nonZeros(), expected.L.nonZeros());
        EXPECT_TRUE(actual.L.inner_.empty()); EXPECT_TRUE(actual.L.vals_.empty());
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
        EXPECT_EQ(permutation, expected.perm); EXPECT_EQ(offsets, expected.L.outer_);
        EXPECT_EQ(offsets.back(), maximum);
        // Only two headers are allocated: the large count is metadata, never
        // a request for factor-entry storage. Both constructions must abort.
        headers[0].entry_count = maximum - 1; headers[1].entry_begin = maximum - 1;
        old_columns[0].entry_count = maximum - 1;
        EXPECT_DEATH(apxchol::detail::validate_owned_factor_columns(
            headers, maximum - 1, permutation, offsets), "factor_metadata\\(nnz\\)");
        EXPECT_DEATH(apxchol::detail::build_csc(
            expected, old_columns, 2, nullptr, false), "factor_metadata\\(nnz\\)");
    } else {
        GTEST_SKIP() << "32-bit edge capacity boundary; no giant allocation for 64-bit offsets";
    }
}

TEST(GpuOwnedPrefix, CompleteMetadataMatchesHeaderConstruction) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
    scoped_omp_threads serial(1);
    apxchol::factorization expected, actual;
    for (bool direct : {false, true}) {
        auto graph = make_owned_order_fixture();
        apxchol::detail::gpu_block_frontend selector(graph.n(), owned_fixture_topology(graph));
        apxchol::detail::gpu_round_shadow_session session(true);
        std::vector<node_index> active(graph.n());
        std::iota(active.begin(), active.end(), node_index{0});
        while (!active.empty()) {
            selector.prepare(active, {});
            const auto selected = selector.select_block_greedy().data;
            ASSERT_FALSE(selected.empty());
            session.run_owned_prefix_round(graph, selected, selector.device_selection(), kOwnedSeed);
            std::erase_if(active, [&](node_index v) {
                return std::binary_search(selected.begin(), selected.end(), v);
            });
            if (!active.empty()) session.advance_selector(selector);
        }
        std::vector<apxchol::detail::factor_col> columns;
        if (direct) {
            session.complete_owned_factorization(actual.perm, actual.L);
            EXPECT_EQ(actual.perm, expected.perm);
            EXPECT_EQ(actual.L.outer_, expected.L.outer_);
            EXPECT_TRUE(actual.L.inner_.empty()); EXPECT_TRUE(actual.L.vals_.empty());
            // Empty host columns represent a full device log, not a CPU tail.
            EXPECT_NO_THROW(session.finalize_device_factor(columns, actual.perm, graph.n()));
            EXPECT_THROW(session.complete_owned_factorization(actual.perm, actual.L), std::logic_error);
        } else {
            session.complete_owned_factorization(columns);
            apxchol::detail::build_csc(expected, columns, graph.n(), nullptr, false);
        }
        EXPECT_NO_THROW(session.finish());
    }
#endif
}

TEST(GpuOwnedPrefix, ContinuousRoundsMatchIndependentReferenceThroughEmptyResidual) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env setup_trace("APXCHOL_SPTRSV_SETUP_TRACE", "1");
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
    scoped_omp_threads serial(1);
    auto graph = make_owned_order_fixture();
    auto oracle = make_owned_order_fixture();
    const auto initial = fingerprint_graph(graph);
    apxchol::detail::gpu_block_frontend selector(graph.n(), owned_fixture_topology(graph));
    apxchol::detail::gpu_round_shadow_session session(true);
    std::vector<node_index> active(graph.n());
    std::iota(active.begin(), active.end(), node_index{0});
    std::vector<std::vector<node_index>> selections;
    std::vector<gpu_round_shadow_report> reports;
    while (!active.empty()) {
        selector.prepare(active, {});
        const auto selected = selector.select_block_greedy().data;
        ASSERT_FALSE(selected.empty());
        reports.push_back(session.run_owned_prefix_round(
            graph, selected, selector.device_selection(), kOwnedSeed));
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
    EXPECT_EQ(selector.transfers().host_update_bytes, 0u);
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
    scoped_env fp32("APXCHOL_SPTRSV_FP16", "0");
    scoped_env drop("APXCHOL_FACTOR_DROP", "0");
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
    apxchol::cuda_sptrsv expected_trsv, owned_trsv;
    expected_trsv.setup(reference_factor.L, kOwnedVertices);
    auto device_factor = session.finalize_device_factor(columns, reference_factor.perm, kOwnedVertices);
    owned_trsv.setup_adopting_device_factor_for_research(std::move(*device_factor));
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
    if (apxchol::detail::gpu_setup_diagnostics()) {
        EXPECT_NE(trace.find("[gpu-owned-factorization] complete"), std::string::npos);
        EXPECT_NE(trace.find("cpu_tail_columns=0"), std::string::npos);
        EXPECT_NE(trace.find("residual_download_bytes=0"), std::string::npos);
    } else {
        EXPECT_TRUE(trace.empty()) << trace;
    }
    EXPECT_THROW(session.complete_owned_factorization(columns), std::logic_error);
#endif
}

TEST(GpuOwnedPrefix, DegreeScratchLifetimeAcrossMovesCompletionAndNewSizes) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
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
            apxchol::graph<apxchol::directed_vec_pool_incidence> g(n);
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
        apxchol::detail::gpu_block_frontend selector(n, owned_fixture_topology(graph));
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
            const auto report = session.run_owned_prefix_round(
                graph, selected, selector.device_selection(), kOwnedSeed);
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
#endif
}

TEST(GpuOwnedPrefix, DegreeScratchHandbackPreservesEveryOrderedField) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
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
        f.session->materialize_owned_prefix(f.graph, f.active, columns);
        const auto actual = apxchol::detail::make_gpu_round_shadow_input(f.graph, {}, kOwnedSeed);
        const auto stored_cpu = apxchol::detail::make_gpu_round_shadow_input(oracle, {}, kOwnedSeed);
        // CPU slabs can retain arcs to inactive vertices. The owned residual
        // physically contains only active-both incidences; restrict the CPU
        // oracle stably, without changing or filtering the actual handback.
        const auto expected = compact_live_snapshot(stored_cpu);
        if (rounds == 1) EXPECT_GT(stored_cpu.incidences.size(), expected.incidences.size());
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
            EXPECT_EQ(std::bit_cast<std::uint64_t>(a.weight), std::bit_cast<std::uint64_t>(e.weight));
            receipt += ":e" + std::to_string(a.owner) + ":" + std::to_string(a.neighbor)
                + ":" + std::to_string(std::bit_cast<std::uint64_t>(a.weight));
        }
        EXPECT_EQ(f.graph.m(), oracle.m());
        RecordProperty("degree_handback_rounds_" + std::to_string(rounds), receipt);
    }
#endif
}

TEST(GpuOwnedPrefix, FactorEntryWritesMatchSerialFactorOnEveryBasis) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env fp32("APXCHOL_SPTRSV_FP16", "0");
    scoped_env drop("APXCHOL_FACTOR_DROP", "0");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
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
            apxchol::graph<apxchol::directed_vec_pool_incidence> graph(n);
            for (const auto& edge : edges) graph.add_edge(edge.u, edge.v, edge.weight);
            for (node_index v = 0; v < n; ++v) graph.excess(v) = 0.125;
            return graph;
        };
        auto graph = make_graph(), oracle = make_graph();
        apxchol::detail::gpu_round_shadow_session session(true);
        apxchol::detail::gpu_block_frontend selector(n, owned_fixture_topology(graph));
        std::vector<node_index> active(n), wanted(count);
        std::iota(active.begin(), active.end(), node_index{0});
        std::iota(wanted.begin(), wanted.end(), node_index{0});
        apxchol::partition_options options;
        options.degree_quantile = 0.0; options.degree_multiplier = 1024.0;
        options.degree_tiebreak = false;
        selector.prepare(active, options);
        const auto selected = selector.select_block_greedy().data;
        ASSERT_EQ(selected, wanted); // All centers precede their leaves by ID.
        const auto input = apxchol::detail::make_gpu_round_shadow_input(graph, selected, kOwnedSeed);
        const auto report = session.run_owned_prefix_round(graph, selected, selector.device_selection(), kOwnedSeed);
        ASSERT_TRUE(report.gpu_executed); EXPECT_TRUE(report.pivots.empty());
        std::vector<apxchol::detail::factor_col> actual_columns;
        session.materialize_owned_prefix(graph, active, actual_columns);
        std::vector<gpu_round_shadow_excess_bound> bounds;
        const auto reference = apxchol::detail::reference_gpu_round_shadow(input, &bounds);
        apxchol::detail::gpu_round_shadow_factor_log expected_log;
        (void)run_serial_cpu_round(oracle, input, reference, bounds, &expected_log);
        ASSERT_EQ(actual_columns.size(), expected_log.columns.size());
        std::vector<std::vector<apxchol::detail::factor_entry>> prefix_entries(count);
        std::vector<apxchol::detail::factor_col> expected_columns;
        for (std::size_t i = 0; i < count; ++i) {
            const auto& c = expected_log.columns[i];
            EXPECT_EQ(actual_columns[i].vertex, c.vertex);
            EXPECT_EQ(actual_columns[i].entry_count, c.entry_count);
            EXPECT_EQ(std::bit_cast<std::uint32_t>(actual_columns[i].diag),
                      std::bit_cast<std::uint32_t>(c.diag));
            for (std::size_t j = 0; j < c.entry_count; ++j) {
                const auto& e = expected_log.entries[c.entry_begin + j];
                prefix_entries[i].push_back({e.neighbor, e.value});
            }
            expected_columns.push_back({c.vertex, c.diag, prefix_entries[i].data(), c.entry_count});
        }
        // Append the same independently computed CPU tail to both factors. This
        // isolates owned prefix values without exposing a new factor-log API.
        apxchol::factorize_workspace workspace;
        workspace.threads.resize(1);
        workspace.threads[0].factor_entries = std::make_unique<std::pmr::monotonic_buffer_resource>();
        apxchol::factor_options tail_options; tail_options.seed = kOwnedSeed;
        auto remaining = active;
        apxchol::detail::eliminate_remaining(apxchol::detail::tree_elimination{}, oracle,
            remaining, expected_columns, workspace, tail_options);
        actual_columns.insert(actual_columns.end(), expected_columns.begin() + count, expected_columns.end());
        apxchol::factorization reference_factor;
        apxchol::detail::build_csc(reference_factor, expected_columns, n, nullptr);
        apxchol::cuda_sptrsv expected_trsv, actual_trsv;
        expected_trsv.setup(reference_factor.L, n);
        auto factor = session.finalize_device_factor(actual_columns, reference_factor.perm, n);
        ASSERT_NE(factor, nullptr);
        actual_trsv.setup_adopting_device_factor_for_research(std::move(*factor));
        std::uint64_t digest = 14695981039346656037ULL;
        for (node_index basis = 0; basis < n; ++basis) {
            std::vector<double> expected(n), actual(n);
            expected[basis] = actual[basis] = 1.0;
            expected_trsv.solve_LLt(expected.data(), expected.data());
            actual_trsv.solve_LLt(actual.data(), actual.data());
            EXPECT_EQ(actual, expected) << "basis=" << basis;
            for (double value : actual)
                digest = (digest ^ std::bit_cast<std::uint64_t>(value)) * 1099511628211ULL;
        }
        // Every-basis equality checks complete factor action, not raw factor bytes.
        RecordProperty("owned_factor_entry_" + std::to_string(count),
            std::to_string(n) + ":" + std::to_string(expected_log.entries.size()) + ":" + std::to_string(digest));
    }
#endif
}

TEST(GpuOwnedPrefix, NormalBatchesMatchSerialHandbackAndFactorAction) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env fp32("APXCHOL_SPTRSV_FP16", "0");
    scoped_env drop("APXCHOL_FACTOR_DROP", "0");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
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
                apxchol::graph<apxchol::directed_vec_pool_incidence> g(n);
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
            apxchol::detail::gpu_block_frontend selector(n, owned_fixture_topology(graph));
            std::vector<node_index> active(n), wanted(count);
            std::iota(active.begin(), active.end(), node_index{0});
            std::iota(wanted.begin(), wanted.end(), node_index{0});
            apxchol::partition_options options;
            options.degree_quantile = 0.0;
            options.degree_multiplier = 1024.0;
            options.degree_tiebreak = false;
            // Initial generic selection is independent of the experimental
            // resident selector: every lower-ID center precedes its receivers.
            selector.prepare(active, options);
            const auto selected = selector.select_block_greedy().data;
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
            const auto report = session.run_owned_prefix_round(
                graph, selected, selector.device_selection(), seed);
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
            if (profile == 1) EXPECT_EQ(got.batches, 3u);
            if (profile == 2) EXPECT_EQ(got.batches, 1u);
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
            session.materialize_owned_prefix(graph, active, actual_columns);
            const auto actual = apxchol::detail::make_gpu_round_shadow_input(
                graph, std::span<const node_index>{}, seed);
            // Independent numerical oracle starts after GPU-owned execution.
            std::vector<gpu_round_shadow_excess_bound> bounds;
            const auto reference = apxchol::detail::reference_gpu_round_shadow(input, &bounds);
            apxchol::detail::gpu_round_shadow_factor_log expected_log;
            (void)run_serial_cpu_round(oracle, input, reference, bounds, &expected_log);
            const auto expected = compact_live_snapshot(apxchol::detail::make_gpu_round_shadow_input(
                oracle, std::span<const node_index>{}, seed));
            ASSERT_EQ(actual.owner_offsets, expected.owner_offsets);
            ASSERT_EQ(actual.active, expected.active);
            ASSERT_EQ(actual.incidences.size(), expected.incidences.size());
            ASSERT_EQ(actual.excess.size(), expected.excess.size());
            for (std::size_t j = 0; j < actual.incidences.size(); ++j) {
                EXPECT_EQ(actual.incidences[j].owner, expected.incidences[j].owner);
                EXPECT_EQ(actual.incidences[j].neighbor, expected.incidences[j].neighbor);
                EXPECT_EQ(std::bit_cast<std::uint64_t>(actual.incidences[j].weight),
                          std::bit_cast<std::uint64_t>(expected.incidences[j].weight));
            }
            for (std::size_t j = 0; j < actual.excess.size(); ++j)
                EXPECT_EQ(std::bit_cast<std::uint64_t>(actual.excess[j]),
                          std::bit_cast<std::uint64_t>(expected.excess[j]));
            EXPECT_EQ(graph.m(), oracle.m());
            EXPECT_EQ(report.raw_fill_edges, reference.raw_fill_edges);
            EXPECT_EQ(report.unique_neighbors, reference.unique_neighbors);
            EXPECT_EQ(report.excess_updates, reference.excess_updates);
            if (mode) EXPECT_GT(report.excess_updates, 0u);

            ASSERT_EQ(actual_columns.size(), expected_log.columns.size());
            std::vector<std::vector<apxchol::detail::factor_entry>> prefix_entries(count);
            std::vector<apxchol::detail::factor_col> expected_columns;
            for (std::size_t j = 0; j < count; ++j) {
                const auto& c = expected_log.columns[j];
                EXPECT_EQ(actual_columns[j].vertex, c.vertex);
                EXPECT_EQ(actual_columns[j].entry_count, c.entry_count);
                EXPECT_EQ(std::bit_cast<std::uint32_t>(actual_columns[j].diag),
                          std::bit_cast<std::uint32_t>(c.diag));
                for (std::size_t k = 0; k < c.entry_count; ++k) {
                    const auto& entry = expected_log.entries[c.entry_begin + k];
                    prefix_entries[j].push_back({entry.neighbor, entry.value});
                }
                expected_columns.push_back({c.vertex, c.diag, prefix_entries[j].data(), c.entry_count});
            }
            const auto handback_receipt = owned_residual_boundary_receipt(actual, actual_columns);
            apxchol::factorize_workspace workspace;
            workspace.threads.resize(1);
            workspace.threads[0].factor_entries = std::make_unique<std::pmr::monotonic_buffer_resource>();
            apxchol::factor_options tail_options; tail_options.seed = seed;
            auto remaining = active;
            apxchol::detail::eliminate_remaining(apxchol::detail::tree_elimination{}, oracle,
                remaining, expected_columns, workspace, tail_options);
            actual_columns.insert(actual_columns.end(), expected_columns.begin()+count, expected_columns.end());
            apxchol::factorization reference_factor;
            apxchol::detail::build_csc(reference_factor, expected_columns, n, nullptr);
            apxchol::cuda_sptrsv reference_trsv, actual_trsv;
            reference_trsv.setup(reference_factor.L, n);
            auto factor = session.finalize_device_factor(actual_columns, reference_factor.perm, n);
            ASSERT_NE(factor, nullptr);
            actual_trsv.setup_adopting_device_factor_for_research(std::move(*factor));
            std::uint64_t digest = 14695981039346656037ULL;
            for (node_index basis = 0; basis < n; ++basis) {
                std::vector<double> wanted_x(n), actual_x(n);
                wanted_x[basis] = actual_x[basis] = 1.0;
                reference_trsv.solve_LLt(wanted_x.data(), wanted_x.data());
                actual_trsv.solve_LLt(actual_x.data(), actual_x.data());
                EXPECT_EQ(actual_x, wanted_x) << "basis=" << basis;
                for (double value : actual_x)
                    digest = (digest ^ std::bit_cast<std::uint64_t>(value)) * 1099511628211ULL;
            }
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
#endif
}

TEST(GpuOwnedPrefix, OversizedImportRejectionAndCompactContinuationMatchOracle) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
    scoped_omp_threads serial(1);
    constexpr node_index n = 4;
    constexpr unsigned seed = 42;
    auto make_graph = [&] {
        apxchol::graph<apxchol::directed_vec_pool_incidence> g(n);
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
    // Inactive imports are outside the high-level owned-session contract;
    // reject them instead of silently assuming their stored slots are live.
    {
        auto rejected = make_graph();
        rejected.deactivate(3);
        apxchol::detail::gpu_round_shadow_session rejected_session(true);
        apxchol::detail::gpu_block_frontend rejected_selector(
            n, owned_fixture_topology(rejected));
        const std::vector<node_index> inactive{3}, active_probe{0, 1, 2};
        rejected_selector.advance(inactive, {});
        apxchol::partition_options probe_options;
        probe_options.degree_multiplier = 1024.0;
        probe_options.degree_tiebreak = false;
        rejected_selector.prepare(active_probe, probe_options);
        const auto selected = rejected_selector.select_block_greedy().data;
        ASSERT_FALSE(selected.empty());
        EXPECT_THROW(rejected_session.run_owned_prefix_round(rejected, selected,
            rejected_selector.device_selection(), seed), std::invalid_argument);
    }
    auto graph = make_graph(), oracle = make_graph();
    apxchol::detail::gpu_round_shadow_session session(true);
    apxchol::detail::gpu_block_frontend selector(n, owned_fixture_topology(graph));
    std::vector<node_index> active{0, 1, 2, 3};
    apxchol::partition_options options;
    options.degree_quantile = 0.0;
    options.degree_multiplier = 1024.0;
    options.degree_tiebreak = false;
    apxchol::detail::gpu_round_shadow_factor_log expected_log;
    std::ostringstream receipt;
    for (unsigned round = 0; round < 2; ++round) {
        SCOPED_TRACE(round);
        selector.prepare(active, options);
        const auto selected = selector.select_block_greedy().data;
        ASSERT_EQ(selected.size(), 1u);
        if (round == 0) ASSERT_EQ(selected.front(), 0u);
        auto input = apxchol::detail::make_gpu_round_shadow_input(oracle, selected, seed);
        if (round) input = compact_live_snapshot(input);
        const auto pivot = selected.front();
        const auto span = input.owner_offsets[pivot + 1] - input.owner_offsets[pivot];
        std::size_t live = 0;
        for (std::size_t j = input.owner_offsets[pivot]; j < input.owner_offsets[pivot + 1]; ++j)
            live += input.active[input.incidences[j].neighbor] != 0;
        ASSERT_GT(span, 128u);
        if (round == 0) ASSERT_EQ(span, 129u);
        ASSERT_EQ(span, live);
        const auto report = session.run_owned_prefix_round(
            graph, selected, selector.device_selection(), seed);
        ASSERT_TRUE(report.gpu_executed);
        EXPECT_EQ(report.normal_batch.oversized_pivots, 1u);
        EXPECT_EQ(report.normal_batch.normal_pivots, 0u);
        EXPECT_EQ(report.normal_batch.oversized_spans, span);
        EXPECT_EQ(report.normal_batch.oversized_gathered, live);
        if (round) EXPECT_EQ(report.round_state_upload_bytes, 0u);
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
    session.materialize_owned_prefix(graph, active, columns);
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
        EXPECT_EQ(std::bit_cast<std::uint64_t>(actual.incidences[j].weight),
                  std::bit_cast<std::uint64_t>(expected.incidences[j].weight));
    }
    for (std::size_t j = 0; j < actual.excess.size(); ++j)
        EXPECT_EQ(std::bit_cast<std::uint64_t>(actual.excess[j]),
                  std::bit_cast<std::uint64_t>(expected.excess[j]));
    EXPECT_EQ(graph.m(), oracle.m());
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
#endif
}

TEST(GpuOwnedPrefix, ParallelEmissionMatchesSerialOrderedHandback) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
    scoped_omp_threads serial(1);
    for (node_index degree : {0, 1, 2, 31, 32, 33, 257}) {
        for (unsigned profile = 0; profile < 2; ++profile) {
            SCOPED_TRACE("degree=" + std::to_string(degree) +
                         " profile=" + std::to_string(profile));
            // The CPU factor-options seam accepts an unsigned run seed.
            // Per-pivot SplitMix state and its item offsets remain 64-bit.
            const std::uint64_t seed = profile ? std::numeric_limits<unsigned>::max() : 0;
            auto make_graph = [&] {
                apxchol::graph<apxchol::directed_vec_pool_incidence> graph(degree + 1);
                // Equal stored degrees force the selector to choose vertex 0,
                // rather than the leaves of a high-degree star. Duplicate arcs
                // also exercise canonical deduplication before item emission.
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
            apxchol::detail::gpu_block_frontend selector(
                graph.n(), owned_fixture_topology(graph));
            selector.prepare(active, {});
            const auto selected = selector.select_block_greedy().data;
            ASSERT_EQ(selected, std::vector<node_index>{0});
            const auto input = apxchol::detail::make_gpu_round_shadow_input(
                oracle, selected, seed);
            std::vector<gpu_round_shadow_excess_bound> bounds;
            const auto reference = apxchol::detail::reference_gpu_round_shadow(input, &bounds);
            apxchol::detail::gpu_round_shadow_factor_log expected_factor;
            run_serial_cpu_round(oracle, input, reference, bounds, &expected_factor);

            apxchol::detail::gpu_round_shadow_session session(true);
            const auto report = session.run_owned_prefix_round(
                graph, selected, selector.device_selection(), seed);
            EXPECT_TRUE(report.pivots.empty()); // Owned, not audited serial sampler.
            EXPECT_EQ(report.unique_neighbors, static_cast<std::size_t>(degree));
            EXPECT_EQ(report.raw_fill_edges, reference.raw_fill_edges);
            std::vector<apxchol::detail::factor_col> columns;
            session.materialize_owned_prefix(graph, active, columns);
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
                EXPECT_EQ(std::bit_cast<std::uint64_t>(a.weight),
                          std::bit_cast<std::uint64_t>(e.weight));
                fold(a.owner); fold(a.neighbor); fold(std::bit_cast<std::uint64_t>(a.weight));
            }
            for (const auto& column : columns) {
                fold(column.vertex); fold(std::bit_cast<std::uint32_t>(column.diag));
                fold(column.entry_count);
            }
            RecordProperty("emission_" + std::to_string(degree) + "_" +
                           std::to_string(profile), std::to_string(digest));
        }
    }
#endif
}

TEST(GpuOwnedPrefix, PrematureCompletionPoisonsIncompleteDeviceFactor) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
    auto graph = make_owned_order_fixture();
    apxchol::detail::gpu_block_frontend selector(graph.n(), owned_fixture_topology(graph));
    apxchol::detail::gpu_round_shadow_session session(true);
    std::vector<node_index> active(graph.n());
    std::iota(active.begin(), active.end(), node_index{0});
    selector.prepare(active, {});
    const auto selected = selector.select_block_greedy().data;
    const auto report = session.run_owned_prefix_round(
        graph, selected, selector.device_selection(), kOwnedSeed);
    ASSERT_GT(report.active_count, 0u);
    std::vector<apxchol::detail::factor_col> columns;
    EXPECT_THROW(session.complete_owned_factorization(columns), std::logic_error);
    EXPECT_TRUE(columns.empty());
    EXPECT_THROW(session.materialize_owned_prefix(graph, active, columns), std::logic_error);
#endif
}

TEST(GpuOwnedPrefix, StaleSecondSelectionPoisonsBeforeHostPublication) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
    scoped_omp_threads serial(1);
    auto graph = make_owned_order_fixture();
    const auto initial = fingerprint_graph(graph);
    auto topology = owned_fixture_topology(graph);
    apxchol::detail::gpu_block_frontend selector(graph.n(), topology);
    apxchol::detail::gpu_round_shadow_session session(true);
    std::vector<node_index> active(graph.n());
    std::iota(active.begin(), active.end(), node_index{0});
    selector.prepare(active, {});
    const auto selected = selector.select_block_greedy().data;
    const auto old = selector.device_selection();
    session.run_owned_prefix_round(graph, selected, old, kOwnedSeed);
    session.advance_selector(selector);
    EXPECT_THROW(session.run_owned_prefix_round(graph, selected, old, kOwnedSeed), std::exception);
    std::vector<apxchol::detail::factor_col> columns;
    EXPECT_THROW(session.materialize_owned_prefix(graph, active, columns), std::exception);
    EXPECT_TRUE(columns.empty());
    EXPECT_EQ(fingerprint_graph(graph), initial);
#endif
}

TEST(GpuOwnedPrefix, ConsumingSolveEliminatesEveryColumnOnDevice) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env setup_trace("APXCHOL_SPTRSV_SETUP_TRACE", "1");
    scoped_env fp32("APXCHOL_SPTRSV_FP16", "0");
    scoped_env drop("APXCHOL_FACTOR_DROP", "0");
    scoped_env shadow("APXCHOL_GPU_ROUND_SHADOW", "force");
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env frontend("APXCHOL_GPU_BLOCK_FRONTEND", "force");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
    scoped_omp_threads serial(1);
    const int n = 256; // Includes the under-occupied tail, without CPU handoff.
    apxchol::factor_options factor_options;
    factor_options.partition.degree_quantile = 0.0;
    factor_options.partition.degree_multiplier = 100.0;
    factor_options.min_is_fraction = 0.0;
    for (double shift : {0.0, 1.0}) {
        SCOPED_TRACE(shift);
        auto A = owned_solve_matrix(n, shift);
        apxchol::apx_cholesky preconditioner;
        apxchol::checkpoint owned_cp;
        preconditioner.set_checkpoint(&owned_cp);
        preconditioner.set_options(factor_options);
        testing::internal::CaptureStderr();
        preconditioner.compute(A);
        const std::string trace = testing::internal::GetCapturedStderr();
        if (apxchol::detail::gpu_setup_diagnostics()) {
            ASSERT_NE(trace.find("[gpu-owned-factorization] complete rounds="), std::string::npos) << trace;
            EXPECT_GT(gpu_round_trace_size(trace, "rounds="), 2u);
            EXPECT_NE(trace.find("[gpu-owned-csc] n="), std::string::npos);
            EXPECT_NE(trace.find("initial_snapshots=0"), std::string::npos);
            EXPECT_EQ(trace.find("paired_initial=1"), std::string::npos);
            EXPECT_NE(trace.find("cpu_replay_rounds=0"), std::string::npos);
            EXPECT_EQ(trace.find("[gpu-round-shadow] checked"), std::string::npos);
            EXPECT_NE(trace.find("factor_entry_download_bytes=0"), std::string::npos);
            EXPECT_EQ(trace.find("handback_download_bytes="), std::string::npos);
            EXPECT_NE(trace.find("residual_download_bytes=0"), std::string::npos);
            EXPECT_EQ(gpu_round_trace_size(trace, "cpu_tail_columns="), 0u);
            EXPECT_EQ(gpu_round_trace_size(trace, "host_factor_entry_write_bytes="), 0u);
        } else {
            EXPECT_EQ(trace.find("[gpu-setup-receipt]"), std::string::npos) << trace;
            EXPECT_EQ(trace.find("[gpu-owned-"), std::string::npos) << trace;
            EXPECT_EQ(trace.find("[gpu-round-shadow]"), std::string::npos) << trace;
        }
        expect_setup_api_receipt(trace, preconditioner);
        EXPECT_GT(owned_cp.total("setup.gpu_owned_factorization"), 0.0);
        EXPECT_EQ(owned_cp.total("setup.factor_metadata"), 0.0);
        const auto& factor = preconditioner.factor();
        ASSERT_GT(factor.rounds.size(), 2u);
        std::size_t remaining = A.rows();
        for (const auto& round : factor.rounds) {
            EXPECT_EQ(round.active, remaining);
            ASSERT_GT(round.is_size, 0u); ASSERT_LE(round.is_size, remaining);
            remaining -= round.is_size;
        }
        EXPECT_EQ(remaining, 0u);
        if (apxchol::detail::gpu_setup_diagnostics()) {
            EXPECT_EQ(gpu_round_trace_size(trace, "factor_header_count="), factor.perm.size());
        }
        auto permutation = factor.perm;
        std::sort(permutation.begin(), permutation.end());
        for (std::size_t i = 0; i < permutation.size(); ++i) EXPECT_EQ(permutation[i], i);
        EXPECT_EQ(factor.sddm, shift != 0.0);
        EXPECT_TRUE(factor.L.vals_.empty()); EXPECT_TRUE(factor.L.inner_.empty());
        ASSERT_TRUE(preconditioner.trsv().adopted_device_factor());
        Eigen::VectorXd exact(A.rows());
        for (int i = 0; i < exact.size(); ++i) exact[i] = std::sin(i + 0.25);
        Eigen::VectorXd b = A * exact;
        apxchol::solve_options options; options.tol = 1e-8; options.max_iter = 1000;
        options.factor_opts = factor_options;
        const auto solved = apxchol::solve(A, b, options);
        EXPECT_LE((A * solved.x - b).norm() / b.norm(), 1e-8);
    }
#endif
}

TEST(GpuOwnedPrefix, TinyTailAndExportKeepTheirContracts) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env setup_trace("APXCHOL_SPTRSV_SETUP_TRACE", "1");
    scoped_env fp32("APXCHOL_SPTRSV_FP16", "0");
    scoped_env drop("APXCHOL_FACTOR_DROP", "0");
    scoped_env shadow("APXCHOL_GPU_ROUND_SHADOW", "force");
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env frontend("APXCHOL_GPU_BLOCK_FRONTEND", "force");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
    scoped_omp_threads serial(1);
    Eigen::SparseMatrix<double> tiny(2, 2);
    std::vector<Eigen::Triplet<double>> entries{{0,0,2},{1,1,2},{0,1,-1},{1,0,-1}};
    tiny.setFromTriplets(entries.begin(), entries.end());
    apxchol::apx_cholesky small;
    testing::internal::CaptureStderr(); small.compute(tiny);
    const auto zero_trace = testing::internal::GetCapturedStderr();
    if (apxchol::detail::gpu_setup_diagnostics()) {
        EXPECT_NE(zero_trace.find("[gpu-owned-factorization] complete"), std::string::npos);
        EXPECT_EQ(gpu_round_trace_size(zero_trace, "factor_header_count="), 2u);
        EXPECT_EQ(gpu_round_trace_size(zero_trace, "cpu_tail_columns="), 0u);
        EXPECT_EQ(zero_trace.find("[gpu-round-shadow] checked"), std::string::npos);
    } else {
        EXPECT_EQ(zero_trace.find("[gpu-owned-"), std::string::npos);
    }
    expect_setup_api_receipt(zero_trace, small);
    EXPECT_EQ(small.factor().perm.size(), 2u);
    auto A = owned_solve_matrix(64, 1.0);
    apxchol::apx_cholesky kept;
    kept.set_keep_factor(true);
    testing::internal::CaptureStderr(); kept.compute(A);
    const auto kept_trace = testing::internal::GetCapturedStderr();
    EXPECT_EQ(kept_trace.find("[gpu-owned-prefix]"), std::string::npos);
    if (apxchol::detail::gpu_setup_diagnostics()) {
        EXPECT_NE(kept_trace.find("[gpu-round-shadow] checked"), std::string::npos);
    } else {
        EXPECT_EQ(kept_trace.find("[gpu-round-shadow]"), std::string::npos);
    }
    expect_setup_api_receipt(kept_trace, kept);
    EXPECT_EQ(kept.factor().L.vals_.size(), kept.factor().L.nonZeros());
    testing::internal::CaptureStderr(); auto exported = apxchol::factorize(
        A, apxchol::graph_storage::vec_pool_aos);
    const auto exported_trace = testing::internal::GetCapturedStderr();
    EXPECT_EQ(exported_trace.find("[gpu-owned-prefix]"), std::string::npos);
    EXPECT_EQ(exported.L.vals_.size(), exported.L.nonZeros());
    // Ordinary unforced defaults remain on their existing setup route.
    scoped_env shadow_off("APXCHOL_GPU_ROUND_SHADOW", "off");
    scoped_env finalize_off("APXCHOL_GPU_FACTOR_FINALIZE", "off");
    scoped_env frontend_off("APXCHOL_GPU_BLOCK_FRONTEND", "off");
    apxchol::apx_cholesky ordinary;
    testing::internal::CaptureStderr(); ordinary.compute(A);
    const auto ordinary_trace = testing::internal::GetCapturedStderr();
    EXPECT_EQ(ordinary_trace.find("[gpu-owned-prefix]"), std::string::npos);
    EXPECT_FALSE(ordinary.trsv().adopted_device_factor());
    expect_setup_api_receipt(ordinary_trace, ordinary);
#endif
}

TEST(GpuRoundShadowDevice, ResidentSelectorProjectionTwoTransitions) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    resident_selector_fixture f;
    for (int round = 0; round < 3; ++round) {
        f.round();
        f.state.advance_selector(*f.frontend);
        const auto traffic = f.frontend->transfers();
        EXPECT_EQ(traffic.host_update_bytes, 0u);
        EXPECT_EQ(traffic.resident_advances, std::size_t(round + 1));
        f.compare_reference();
    }
    std::fprintf(stderr, "[resident-selector-test] rounds=3 host_update_bytes=%zu "
        "projection_scratch_peak_extra_bound=%zu\n",
        f.frontend->transfers().host_update_bytes,
        f.frontend->transfers().projection_scratch_peak_extra_bytes);
#endif
}

TEST(GpuRoundShadowDevice, ResidentSelectorProjectionHandlesEmptyResidual) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    resident_selector_fixture f(false);
    f.round();
    ASSERT_TRUE(f.active.empty());
    f.state.advance_selector(*f.frontend);
    f.compare_reference();
    EXPECT_EQ(f.frontend->transfers().host_update_bytes, 0u);
#endif
}

TEST(GpuRoundShadowDevice, ResidentSelectorProjectionPreservesOrderAndExcessRefresh) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    for (bool parallel_order : {false, true}) {
        resident_selector_fixture f;
        f.round(false);
        f.state.certify_cpu_round(!parallel_order, fingerprint_graph(f.graph),
                                 !parallel_order);
        f.state.advance_selector(*f.frontend);
        f.frontend->prepare(f.active, f.options);
        f.frontend->select_block_greedy();
        expect_exception_contains([&] {
            f.state.compute_resident(f.frontend->device_selection(), f.seed);
        }, parallel_order ? "parallel-order reimport" : "excess refresh");
        EXPECT_EQ(f.frontend->transfers().host_update_bytes, 0u);
    }
#endif
}

TEST(GpuRoundShadowDevice, ResidentSelectorProjectionRejectsLaterHostSelectedGeneration) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    resident_selector_fixture f;
    f.round();
    ASSERT_FALSE(f.active.empty());
    // A previous producer binding must not authorize a later round that used
    // host-selected pivots, even while its old capability remains published.
    const std::vector<node_index> selected{f.active.front()};
    const auto input = apxchol::detail::make_gpu_round_shadow_input(
        f.graph, selected, f.seed);
    std::vector<gpu_round_shadow_excess_bound> bounds;
    const auto report = apxchol::detail::run_verified_gpu_round_shadow(f.state, input, &bounds);
    const auto cpu = run_serial_cpu_round(f.graph, input, report, bounds);
    f.state.certify_cpu_round(true, fingerprint_graph(f.graph),
                             cpu.bounded_excess_vertices != 0);
    expect_exception_contains([&] { f.state.advance_selector(*f.frontend); }, "CPU-certified");
    EXPECT_EQ(f.frontend->transfers().resident_advances, 0u);
#endif
}

TEST(GpuRoundShadowDevice, ResidentSelectorProjectionRequiresCpuCertification) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    resident_selector_fixture f;
    f.round(false);
    expect_exception_contains([&] { f.state.advance_selector(*f.frontend); },
                              "CPU-certified");
    EXPECT_EQ(f.frontend->transfers().resident_advances, 0u);
    expect_exception_contains([&] { f.state.download_factor_log(); }, "poisoned");
#endif
}

TEST(GpuRoundShadowDevice, ResidentSelectorProjectionRejectsReplayAndInterveningSelection) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    for (bool repeat_projection : {false, true}) {
        resident_selector_fixture f;
        f.round();
        if (repeat_projection) f.state.advance_selector(*f.frontend);
        else f.frontend->select_block_greedy();
        EXPECT_THROW(f.state.advance_selector(*f.frontend), std::exception);
        expect_exception_contains([&] { f.state.download_factor_log(); }, "poisoned");
    }
#endif
}

TEST(GpuRoundShadowDevice, ResidentSelectorProjectionRejectsDifferentProducerAndHostRebuild) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    for (bool host_rebuild : {false, true}) {
        resident_selector_fixture f;
        f.round();
        if (host_rebuild) {
            f.state.invalidate_for_authoritative_host_rebuild(fingerprint_graph(f.graph));
            expect_exception_contains([&] { f.state.advance_selector(*f.frontend); },
                                      "CPU-certified");
        } else {
            apxchol::detail::gpu_block_frontend other(f.n, f.initial);
            std::vector<node_index> all(f.n);
            std::iota(all.begin(), all.end(), node_index{0});
            other.prepare(all, f.options);
            other.select_block_greedy();
            expect_exception_contains([&] { f.state.advance_selector(other); },
                                      "different producer");
            EXPECT_EQ(other.transfers().resident_advances, 0u);
        }
        expect_exception_contains([&] { f.state.download_factor_log(); }, "poisoned");
    }
#endif
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
    // These aggregate counters also include retained radix-sort scratch. The
    // second round selects more pivots and emits more fill, so that scratch may
    // grow despite reusing the resident graph. Generation, import and byte
    // checks above establish actual state reuse without imposing a no-growth
    // requirement on unrelated scratch capacity.
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
    scoped_env setup_trace("APXCHOL_SPTRSV_SETUP_TRACE", "1");
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
    if (apxchol::detail::gpu_setup_diagnostics()) {
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
    } else {
        EXPECT_TRUE(trace_lines.empty()) << shadow_trace;
    }

    // Replay the exact first two factorization selections through the
    // independent reference and canonical serial CPU round. The assembled
    // factor checks remain unconditional; diagnostic mode additionally checks
    // the trace's factor/fill/residual/degree digests.
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

        if (apxchol::detail::gpu_setup_diagnostics()) {
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
        auto factor = state.finalize_device_factor(perm, m, tail);
        ASSERT_FALSE(factor->empty());
        adopted.setup_adopting_device_factor_for_research(std::move(*factor));
        ASSERT_TRUE(factor->empty());
        EXPECT_TRUE(adopted.adopted_device_factor());
        EXPECT_LE(adopted.adoption_host_download_bytes(), 512u); // fixed-size device-plan counts/statistics
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

TEST(GpuFactorFinalize, DefaultStorageMatchesHostCompensationAndHalfBits) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    constexpr node_index n = 10;
    const std::vector<undirected_edge> edges = {{0, 1, 1}, {0, 2, 0.3}};
    const std::vector<node_index> pivots = {0};
    apxchol::detail::gpu_round_shadow_device_state state;
    const auto input = make_input(n, edges, pivots);
    const auto report = state.compute_discover_shape(input);
    state.accept_device_generation(report.output_generation);
    const auto prefix = state.download_factor_log();
    const std::vector<node_index> perm = {0, 2, 1, 3, 4, 5, 6, 7, 8, 9};
    apxchol::detail::gpu_round_shadow_factor_log tail;
    tail.columns = {{2, 2.0f, 0, 6}, {1, 2.0f, 6, 2}, {3, 1.0f, 8, 1},
        {4, 1.0f, 9, 1}, {5, 1.0f, 10, 1}, {6, 1.0f, 11, 1},
        {7, 1.0f, 12, 1}, {8, 1.0f, 13, 1}, {9, 1.0f, 14, 0}};
    // Unsorted tail rows include half ties, threshold drops, format flushes,
    // exact zero, and scale fallback (1/scale or diag/scale overflows).
    tail.entries = {{9, 0.0f}, {8, 0x1p-25f}, {7, 0x1p-14f},
        {6, 0.3333333432674408f}, {5, 1.0f}, {4, 0.500244140625f},
        {3, 0.3333333432674408f}, {4, 0x1p-15f}, {4, 0.25f},
        {5, 0.125f}, {6, 0x1p-130f}, {7, 0x1p-129f},
        {8, 0.00009999f}, {9, 0.25f}};
    auto full = append_factor_logs(prefix, tail);
    std::vector<std::vector<apxchol::detail::factor_entry>> rows(n);
    std::vector<apxchol::detail::factor_col> cols;
    for (node_index i = 0; i < n; ++i) {
        const auto& c = full.columns[i];
        for (unsigned j = 0; j < c.entry_count; ++j) {
            const auto& e = full.entries[c.entry_begin + j];
            rows[i].push_back({e.neighbor, e.value});
        }
        cols.push_back({c.vertex, c.diag, rows[i].data(), node_index(rows[i].size())});
    }
    apxchol::factorization reference;
    apxchol::detail::build_csc(reference, cols, n, nullptr);
    ASSERT_EQ(reference.perm, perm);
    for (const char* fp16 : {"0", "1"}) {
        scoped_env storage("APXCHOL_SPTRSV_FP16", fp16);
        for (const char* rel : {"0", "0.00001", "0.0001", "0.2"}) {
            scoped_env drop("APXCHOL_FACTOR_DROP", rel);
            for (node_index m : {n - 1, n}) {
                SCOPED_TRACE(std::string(fp16) + "/" + rel + "/" + std::to_string(m));
                apxchol::cuda_sptrsv ordinary, adopted;
                ordinary.setup(reference.L, m);
                auto factor = state.finalize_device_factor(perm, m, tail);
#if defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
                auto lt = apxchol::cuda_host::build_L11_csc_int<float>(reference.L, m);
                auto scale = apxchol::cuda_host::column_scales(lt);
                const bool half = std::string_view(fp16) == "1";
                if (half) {
                    for (node_index j = 0; j < m; ++j)
                        if (!std::isfinite(1.0f / scale[j]) ||
                            !std::isfinite(lt.vals[lt.ptr[j]] / scale[j])) scale[j] = 1.0f;
                }
                apxchol::cuda_host::apply_factor_drop(lt, scale, std::atof(rel), half);
                std::vector<std::byte> expected;
                auto append = [&](const auto* data, std::size_t count) {
                    auto bytes = std::as_bytes(std::span(data, count));
                    expected.insert(expected.end(), bytes.begin(), bytes.end());
                };
                if (half) {
                    auto h = apxchol::cuda_host::narrow_fp16_scaled(lt, scale);
                    apxchol::cuda_host::csr_int<std::uint16_t> lt16;
                    lt16.m = m; lt16.nnz = lt.nnz; lt16.ptr = lt.ptr;
                    lt16.idx = std::move(lt.idx); lt16.vals = std::move(h.vals);
                    auto l16 = apxchol::cuda_host::transpose_csr(lt16);
                    append(l16.ptr.data(), m + 1); append(l16.idx.get(), l16.nnz); append(l16.vals.get(), l16.nnz);
                    append(lt16.ptr.data(), m + 1); append(lt16.idx.get(), lt16.nnz); append(lt16.vals.get(), lt16.nnz);
                    append(h.diag.data(), m);
                    std::vector<double> inv2(m);
                    for (node_index j = 0; j < m; ++j) inv2[j] = double(h.inv_scale[j]) * double(h.inv_scale[j]);
                    append(inv2.data(), m);
                } else {
                    auto l = apxchol::cuda_host::transpose_csr(lt);
                    append(l.ptr.data(), m + 1); append(l.idx.get(), l.nnz); append(l.vals.get(), l.nnz);
                    append(lt.ptr.data(), m + 1); append(lt.idx.get(), lt.nnz); append(lt.vals.get(), lt.nnz);
                }
                EXPECT_EQ(state.encode_finalized_factor_for_test(*factor), expected);
#endif
                adopted.setup_adopting_device_factor_for_research(std::move(*factor));
                const auto& want = ordinary.drop_stats(); const auto& got = adopted.drop_stats();
                EXPECT_EQ(got.rel, want.rel); EXPECT_EQ(got.nnz_factor, want.nnz_factor);
                EXPECT_EQ(got.nnz_stored, want.nnz_stored); EXPECT_EQ(got.dropped, want.dropped);
                EXPECT_EQ(got.dropped_threshold, want.dropped_threshold);
                EXPECT_EQ(got.dropped_flush, want.dropped_flush);
                for (node_index basis = 0; basis < m; ++basis) {
                    std::vector<double> a(m), b(m); a[basis] = b[basis] = 1.0;
                    ordinary.solve_LLt(a.data(), a.data()); adopted.solve_LLt(b.data(), b.data());
                    EXPECT_EQ(a, b) << "basis=" << basis;
                }
            }
        }
    }
}

TEST(GpuFactorFinalize, ConsumingSolveAcceptsDefaultFp16AndCompensatedDrop) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env storage("APXCHOL_SPTRSV_FP16", nullptr);
    scoped_env drop("APXCHOL_FACTOR_DROP", nullptr);
    scoped_env shadow("APXCHOL_GPU_ROUND_SHADOW", "force");
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env frontend("APXCHOL_GPU_BLOCK_FRONTEND", "force");
    scoped_env regions("APXCHOL_GPU_BLOCKS", "1");
    scoped_omp_threads serial(1);
    for (double shift : {0.0, 1.0}) {
        // Cross several device-plan tiles and finish with a partial tile.
        auto A = owned_solve_matrix(1025, shift);
        Eigen::VectorXd exact(A.rows());
        for (int i = 0; i < exact.size(); ++i) exact[i] = std::sin(i + 0.25);
        Eigen::VectorXd b = A * exact;
        apxchol::solve_options opts; opts.tol = 1e-8; opts.max_iter = 1000;
        const auto solved = apxchol::solve(A, b, opts);
        EXPECT_LE((A * solved.x - b).norm() / b.norm(), 1e-8);
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
    EXPECT_THROW(state.finalize_device_factor(perm, 3, tail), std::logic_error);
    state.accept_device_generation(report.output_generation);
    EXPECT_THROW(state.finalize_device_factor(std::vector<node_index>{0, 1, 1}, 3, tail), std::invalid_argument);
    auto bad = tail; bad.columns.pop_back();
    EXPECT_THROW(state.finalize_device_factor(perm, 3, bad), std::invalid_argument);
    bad = tail; bad.entries[0].neighbor = 0;
    EXPECT_THROW(state.finalize_device_factor(perm, 3, bad), std::invalid_argument);
    bad = tail; bad.entries.push_back(bad.entries[0]); bad.columns[0].entry_count = 2;
    EXPECT_THROW(state.finalize_device_factor(perm, 3, bad), std::invalid_argument);
    { // Dropping a tiny duplicate must not conceal invalid raw coordinates.
        scoped_env enabled_drop("APXCHOL_FACTOR_DROP", "0.1");
        auto duplicate = tail; duplicate.columns[0].entry_count = 2;
        duplicate.entries.push_back({2, 0.001f});
        EXPECT_THROW(state.finalize_device_factor(perm, 3, duplicate), std::invalid_argument);
    }
    bad = tail; bad.columns[0].diag = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(state.finalize_device_factor(perm, 3, bad), std::invalid_argument);
    // Failed read-only finalization must leave the audited prefix reusable.
    EXPECT_NO_THROW(state.finalize_device_factor(perm, 3, tail));
    // The optional gpu_factor_finalize_leak_check CTest checks these repeated
    // allocations with Compute Sanitizer. cudaMemGetInfo is device-wide and
    // cannot distinguish our leaks from another process's allocations/frees.
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (int i = 0; i < 8; ++i) EXPECT_NO_THROW(state.finalize_device_factor(perm, 3, tail));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST(GpuFactorFinalize, NormalPreconditionerInstallsAndReplacesResidentFactors) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env fp32("APXCHOL_SPTRSV_FP16", "0");
    scoped_env drop("APXCHOL_FACTOR_DROP", "0");
    scoped_env shadow("APXCHOL_GPU_ROUND_SHADOW", "force");
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env frontend("APXCHOL_GPU_BLOCK_FRONTEND", "0");
    // This fixture compares the same CPU-authoritative elimination law, with
    // audited device factor storage; it does not activate the owned selector.
    ASSERT_EQ(apxchol::detail::gpu_block_frontend::configured_block_mode(),
              apxchol::detail::gpu_block_frontend::mode::disabled);
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
    scoped_env setup_trace("APXCHOL_SPTRSV_SETUP_TRACE", "1");
    scoped_env fp32("APXCHOL_SPTRSV_FP16", "0");
    scoped_env drop("APXCHOL_FACTOR_DROP", "0");
    scoped_env shadow("APXCHOL_GPU_ROUND_SHADOW", "force");
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env frontend("APXCHOL_GPU_BLOCK_FRONTEND", "0");
    // This fixture compares the same CPU-authoritative elimination law, with
    // audited device factor storage; it does not activate the owned selector.
    ASSERT_EQ(apxchol::detail::gpu_block_frontend::configured_block_mode(),
              apxchol::detail::gpu_block_frontend::mode::disabled);
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
        testing::internal::CaptureStderr();
        consuming.compute(A);
        const std::string consuming_trace = testing::internal::GetCapturedStderr();
        expect_setup_api_receipt(consuming_trace, consuming);
        EXPECT_EQ(consuming_trace.find("[gpu-owned-"), std::string::npos);
        EXPECT_EQ(cp.total("setup.gpu_owned_factorization"), 0.0);
        ASSERT_TRUE(consuming.trsv().adopted_device_factor());
        const auto& F = consuming.factor();
        EXPECT_TRUE(F.L.vals_.empty()); EXPECT_TRUE(F.L.inner_.empty());
        EXPECT_EQ(F.perm, reference.perm);
        EXPECT_EQ(F.L.outer_, reference.L.outer_);
        EXPECT_EQ(F.L.nonZeros(), reference.L.nonZeros());
        EXPECT_EQ(cp.total("setup.assembly"), 0.0);
        EXPECT_GT(cp.total("setup.factor_metadata"), 0.0);
        if (apxchol::detail::gpu_setup_diagnostics()) {
            EXPECT_EQ(gpu_round_trace_size(consuming_trace, "host_factor_entry_alloc_bytes="), 0u);
            EXPECT_EQ(gpu_round_trace_size(consuming_trace, "host_factor_entry_write_bytes="), 0u);
            EXPECT_EQ(gpu_round_trace_size(consuming_trace, "host_factor_entry_omitted_bytes="),
                      (reference.L.nonZeros() - n) * sizeof(apxchol::detail::factor_entry));
        } else {
            EXPECT_EQ(consuming_trace.find("host_factor_entry_"), std::string::npos);
        }
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

TEST(GpuFactorFinalize, ConsumingPrefixOmissionRetainsTheCpuTailPayload) {
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env setup_trace("APXCHOL_SPTRSV_SETUP_TRACE", "1");
    scoped_env fp32("APXCHOL_SPTRSV_FP16", "0");
    scoped_env drop("APXCHOL_FACTOR_DROP", "0");
    scoped_env shadow("APXCHOL_GPU_ROUND_SHADOW", "force");
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env frontend("APXCHOL_GPU_BLOCK_FRONTEND", "0");
    // This fixture compares the same CPU-authoritative elimination law, with
    // audited device factor storage; it does not activate the owned selector.
    ASSERT_EQ(apxchol::detail::gpu_block_frontend::configured_block_mode(),
              apxchol::detail::gpu_block_frontend::mode::disabled);
    scoped_omp_threads serial(1);
    constexpr node_index n = 20;
    auto factorize = [&](bool retain) {
        apxchol::graph<apxchol::directed_vec_pool_incidence> graph(n);
        for (node_index v = 0; v < 4; ++v) graph.add_edge(v, (v + 1) % 4, 1.0);
        for (node_index v = 4; v < n; ++v) graph.add_edge(v, v % 4, 1.0);
        // This custom partitioner is also statically ineligible for the owned
        // block-greedy path. Both retain modes therefore have the same order.
        // One audited leaf round, then an explicitly empty selection sends
        // the nontrivial four-cycle to the ordinary CPU tail.
        auto partitioner = apxchol::as_partitioner(
            [](auto&, std::span<const node_index> active,
               const apxchol::partition_context&, apxchol::selection& out) {
                for (auto v : active) if (v >= 4) out.add(v);
            });
        apxchol::factor_options opts;
        opts.parallel_residual_threshold = n;
        return apxchol::factorize_impl(apxchol::detail::tree_elimination{},
            partitioner, std::move(graph), opts, nullptr, retain);
    };
    auto exported = factorize(true);
    testing::internal::CaptureStderr();
    auto consuming = factorize(false);
    const std::string trace = testing::internal::GetCapturedStderr();
    EXPECT_EQ(trace.find("[gpu-owned-"), std::string::npos);
    EXPECT_EQ(consuming.perm, exported.perm);
    EXPECT_EQ(consuming.L.outer_, exported.L.outer_);
    if (apxchol::detail::gpu_setup_diagnostics()) {
        const auto allocated = gpu_round_trace_size(trace, "host_factor_entry_alloc_bytes=");
        const auto omitted = gpu_round_trace_size(trace, "host_factor_entry_omitted_bytes=");
        EXPECT_GT(allocated, 0u);
        EXPECT_EQ(omitted, 16 * sizeof(apxchol::detail::factor_entry));
        EXPECT_EQ(allocated + omitted,
                  (exported.L.nonZeros() - n) * sizeof(apxchol::detail::factor_entry));
        EXPECT_EQ(gpu_round_trace_size(trace, "host_factor_entry_write_bytes="), allocated);
    } else {
        EXPECT_EQ(trace.find("host_factor_entry_"), std::string::npos);
        EXPECT_EQ(trace.find("[gpu-setup-receipt]"), std::string::npos);
    }
    apxchol::cuda_sptrsv upload, adopted;
    upload.setup(exported.L, n - 1);
    adopted.setup_adopting_device_factor_for_research(
        std::move(*consuming.research_device_factor));
    std::vector<double> b(n - 1), expected(n - 1), actual(n - 1);
    for (node_index v = 0; v < n - 1; ++v) b[v] = std::cos(v + 0.25);
    upload.solve_LLt(b.data(), expected.data());
    adopted.solve_LLt(b.data(), actual.data());
    EXPECT_EQ(actual, expected);
}
#endif

TEST(GpuCycleSampler, NormalOversizedAndFullDegreeFillMatchHostLaw) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_env one_region("APXCHOL_GPU_BLOCKS", "1");
    scoped_omp_threads serial(1);
    for (auto sampler : {apxchol::clique_sampler::trace_cycle,
                         apxchol::clique_sampler::heavy_core_k2}) {
        for (unsigned profile = 0; profile < 3; ++profile) {
            SCOPED_TRACE(std::to_string(static_cast<int>(sampler))+":"+std::to_string(profile));
            const std::vector<unsigned> degree = profile == 0 ?
                std::vector<unsigned>{0,1,2,3,127,128,129} :
                profile == 1 ? std::vector<unsigned>{3,4,5,31,129} :
                               std::vector<unsigned>{32,64,65};
            const auto count = static_cast<node_index>(degree.size());
            node_index n=count;
            std::vector<undirected_edge> edges;
            std::vector<std::vector<apxchol::weighted_neighbor>> stars(count);
            for(node_index pivot=0;pivot<count;++pivot)for(unsigned j=0;j<degree[pivot];++j) {
                const double weight=profile==0?1.:std::exp2(int(j%11)-5);
                stars[pivot].push_back({n,weight});
                if(profile==2) {edges.push_back({pivot,n,.25*weight});edges.push_back({pivot,n,.75*weight});}
                else edges.push_back({pivot,n,weight});
                ++n;
            }
            std::vector<std::tuple<node_index,node_index,double>> expected;
            for(node_index pivot=0;pivot<count;++pivot) {
                double D=.125;for(auto edge:stars[pivot])D+=edge.weight;
                std::vector<apxchol::deferred_edge> fill;
                apxchol::tree_elimination{.sampler=sampler}.sample_clique(stars[pivot],D,
                    apxchol::detail::gpu_round_shadow_pivot_seed(17,pivot),apxchol::edge_emitter(fill));
                for(auto edge:fill)expected.emplace_back(std::min(edge.u,edge.v),std::max(edge.u,edge.v),
                    static_cast<double>(static_cast<apxchol::pool_value_t>(edge.w)));
            }
            std::sort(expected.begin(),expected.end());
            std::vector<std::tuple<node_index,node_index,double>> previous;
            for(unsigned repeat=0;repeat<2;++repeat) {
                apxchol::graph<apxchol::directed_vec_pool_incidence> graph(n);
                for(auto edge:edges)graph.add_edge(edge.u,edge.v,edge.weight);
                for(node_index v=0;v<n;++v)graph.excess(v)=.125;
                apxchol::detail::gpu_round_shadow_session session(true,sampler);
                apxchol::detail::gpu_block_frontend selector(n,owned_fixture_topology(graph));
                std::vector<node_index> active(n),wanted(count);
                std::iota(active.begin(),active.end(),node_index{0});
                std::iota(wanted.begin(),wanted.end(),node_index{0});
                apxchol::partition_options options;
                options.degree_quantile=0;options.degree_multiplier=2048;options.degree_tiebreak=false;
                selector.prepare(active,options);
                const auto selected=selector.select_block_greedy().data;
                ASSERT_EQ(selected,wanted);
                const auto report=session.run_owned_prefix_round(graph,selected,selector.device_selection(),17);
                ASSERT_TRUE(report.gpu_executed);
                EXPECT_GT(report.normal_batch.normal_pivots,0u);
                EXPECT_GT(report.normal_batch.oversized_pivots,0u);
                EXPECT_EQ(report.sampler_numerical_fallbacks,0u);
                EXPECT_EQ(report.raw_fill_edges,expected.size());
                EXPECT_EQ(report.live_incidences,2*expected.size());
                EXPECT_LE(report.live_incidences,2*edges.size());
                if(profile==1)EXPECT_EQ(report.live_incidences,2*edges.size()); // f=d uses every slot.
                std::vector<apxchol::detail::factor_col> columns;
                session.materialize_owned_prefix(graph,active,columns);
                std::vector<std::tuple<node_index,node_index,double>> got;
                for(node_index v=count;v<n;++v)for(auto [u,w]:graph.neighbors(v)) {
                    EXPECT_NE(u,v);EXPECT_GE(u,count);EXPECT_GT(w,0.0);EXPECT_TRUE(std::isfinite(w));
                    if(v<u)got.emplace_back(v,u,static_cast<double>(w));
                }
                std::sort(got.begin(),got.end());ASSERT_EQ(got.size(),expected.size());
                for(std::size_t i=0;i<got.size();++i) {
                    EXPECT_EQ(std::get<0>(got[i]),std::get<0>(expected[i]));
                    EXPECT_EQ(std::get<1>(got[i]),std::get<1>(expected[i]));
                    EXPECT_NEAR(std::get<2>(got[i]),std::get<2>(expected[i]),
                        8*std::numeric_limits<apxchol::pool_value_t>::epsilon()*std::get<2>(expected[i]));
                }
                if(repeat)EXPECT_EQ(got,previous);
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
    scoped_env frontend("APXCHOL_GPU_BLOCK_FRONTEND","force");
    scoped_env shadow("APXCHOL_GPU_ROUND_SHADOW","force");
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE","force");
    scoped_env verbose("APXCHOL_VERBOSE","1");
    scoped_omp_threads serial(1);
    for(auto sampler:{apxchol::clique_sampler::trace_cycle,apxchol::clique_sampler::heavy_core_k2})
    for(double shift:{0.,1.}) {
        SCOPED_TRACE(std::to_string(static_cast<int>(sampler))+":"+std::to_string(shift));
        auto A=owned_solve_matrix(257,shift);
        Eigen::VectorXd exact(A.rows());for(int i=0;i<exact.size();++i)exact[i]=std::sin(i+.25);
        const Eigen::VectorXd b=A*exact;
        apxchol::solve_options opts;opts.tol=1e-8;opts.max_iter=1000;
        opts.factor_opts.sampler=sampler;opts.factor_opts.partition.degree_quantile=.8;
        testing::internal::CaptureStderr();
        const auto result=apxchol::solve(A,b,opts);
        const auto trace=testing::internal::GetCapturedStderr();
        EXPECT_LE((A*result.x-b).norm()/b.norm(),1e-8);
        EXPECT_NE(trace.find("[gpu-owned-factorization] complete rounds="),std::string::npos);
        EXPECT_NE(trace.find(sampler==apxchol::clique_sampler::trace_cycle?
            "[gpu-clique-sampler] sampler=trace_cycle":"[gpu-clique-sampler] sampler=heavy_core_k2"),std::string::npos);
        EXPECT_NE(trace.find("device_sampling=1"),std::string::npos);
        EXPECT_EQ(trace.find("[gpu-round-shadow] checked"),std::string::npos);
    }
#endif
}

TEST(GpuCycleSampler, AuditedExportRejectsUnsupportedSamplerClearly) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA build required";
#else
    REQUIRE_GPU_ROUND_SHADOW_DEVICE();
    scoped_env frontend("APXCHOL_GPU_BLOCK_FRONTEND","force");
    scoped_env shadow("APXCHOL_GPU_ROUND_SHADOW","force");
    scoped_env finalize("APXCHOL_GPU_FACTOR_FINALIZE","force");
    scoped_omp_threads serial(1);
    auto A=owned_solve_matrix(65,1.);
    for(auto sampler:{apxchol::clique_sampler::trace_cycle,apxchol::clique_sampler::heavy_core_k2}) {
        apxchol::apx_cholesky preconditioner;preconditioner.set_keep_factor(true);
        apxchol::factor_options opts;opts.sampler=sampler;
        preconditioner.set_options(opts);
        try {preconditioner.compute(A);FAIL()<<"audited export unexpectedly accepted";}
        catch(const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("GPU-owned consuming route"),std::string::npos);
        }
    }
#endif
}
