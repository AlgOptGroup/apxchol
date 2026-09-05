#pragma once
/// Research-only full elimination-round CUDA shadow.
///
/// The CPU factorizer remains authoritative. Immediately before a selected
/// independent set mutates the CPU graph, this sidecar snapshots the current
/// directed-AoS residual for audit, receives the exact selected order and
/// per-pivot seeds, and executes the same *round boundary* on CUDA:
///
///   gather -> raw-order degree + deterministic fp64 dedup -> factor digest
///          -> GKS tree sample -> multigraph fill apply -> residual checksums
///
/// R2a retains the compact owner-major CUDA output, active mask, excess vector,
/// and owner offsets as a generation. After the real CPU round verifies and
/// certifies a serial/order-reproducible apply, the next CUDA round consumes
/// that state without a state upload or state-buffer allocation. Parallel CPU
/// apply order is schedule-dependent, so its successor performs an explicit,
/// counted host import. An exact order-sensitive fingerprint rejects any
/// unannounced divergence before reuse.
///
/// CUDA returns no partner list, fill list, or residual adjacency. The host
/// receives only canonical digests, counters, per-pivot scalar counters,
/// stage timings, and peak allocated bytes.  Deterministic fields are compared
/// against both the independent serial implementation in this header and the
/// authoritative CPU round after it runs.  SDDM excess is compared to the CPU
/// with explicit per-vertex roundoff bounds because its OpenMP atomic update
/// order is intentionally unspecified.
///
/// This is deliberately FORCE-only research surface.  Unset/empty/0/off does
/// no shadow work and leaves production decisions/results unchanged.  "force"
/// fails before the corresponding CPU round mutates when the build, runtime,
/// input capacity, or current free device memory is unsupported.

#include "apxchol/graph/graph.h"
#include "apxchol/solver/elimination/elimination.h"
#include "apxchol/solver/gpu_device_selection.h"
#include "apxchol/sparse_csc.h"
#include "apxchol/lowprec.h"
#include "apxchol/solver/sptrsv/factor_drop.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace apxchol { class cuda_sptrsv_device_factor; }

namespace apxchol::detail {

inline bool gpu_factor_finalize_requested() {
    const char* value = std::getenv("APXCHOL_GPU_FACTOR_FINALIZE");
    if (!value || !*value || std::strcmp(value, "0") == 0 || std::strcmp(value, "off") == 0) return false;
    if (std::strcmp(value, "force") == 0) return true;
    throw std::invalid_argument("APXCHOL_GPU_FACTOR_FINALIZE must be unset, 0, off, or force");
}

inline constexpr std::string_view kGpuRoundShadowEnv =
    "APXCHOL_GPU_ROUND_SHADOW";
inline constexpr std::string_view kGpuRoundSelectionAuditEnv =
    "APXCHOL_GPU_ROUND_SELECTION_AUDIT";

inline bool gpu_round_shadow_requested() {
    const char* value = std::getenv(kGpuRoundShadowEnv.data());
    if (!value || !*value || std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "off") == 0)
        return false;
    if (std::strcmp(value, "force") == 0) return true;
    throw std::invalid_argument(
        "APXCHOL_GPU_ROUND_SHADOW must be unset, 0, off, or force");
}

/// The producer capability certifies selection independence against its
/// content-bound topology. Production resident rounds therefore need only the
/// mandatory O(p) range/active/duplicate validator. This knob additionally
/// scans the complete residual O(r) to audit that certificate.
inline bool gpu_round_selection_audit_requested() {
    const char* value = std::getenv(kGpuRoundSelectionAuditEnv.data());
    if (!value || !*value || std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "off") == 0)
        return false;
    if (std::strcmp(value, "1") == 0 ||
        std::strcmp(value, "on") == 0 ||
        std::strcmp(value, "force") == 0)
        return true;
    throw std::invalid_argument(
        "APXCHOL_GPU_ROUND_SELECTION_AUDIT must be unset, 0, off, 1, on, or force");
}

inline constexpr std::uint64_t gpu_round_shadow_pivot_seed(
        std::uint64_t run_seed, node_index pivot) noexcept {
    return run_seed ^
        ((std::uint64_t(pivot) + 1) * 0x9E3779B97F4A7C15ULL);
}

struct gpu_round_shadow_incidence {
    node_index owner = 0;
    node_index neighbor = 0;
    double weight = 0.0;
};

struct gpu_round_shadow_input {
    node_index vertex_count = 0;
    std::vector<gpu_round_shadow_incidence> incidences;
    // Exact owner-major slab boundaries.  This is semantic input: CPU total
    // degree is accumulated over each pivot's RAW live incidences in this
    // encounter order, separately from neighbor deduplication.
    std::vector<std::size_t> owner_offsets;
    std::vector<std::uint8_t> active;
    std::vector<double> excess;
    std::vector<node_index> pivots;
    std::vector<std::uint64_t> seeds;
};

struct gpu_round_shadow_digest {
    std::uint64_t xor_hash = 0;
    std::uint64_t sum_hash = 0;

    friend bool operator==(const gpu_round_shadow_digest&,
                           const gpu_round_shadow_digest&) = default;
};

struct gpu_round_shadow_pivot_counter {
    node_index pivot = 0;
    std::uint64_t unique_degree = 0;
    std::uint64_t emitted_edges = 0;
    std::uint64_t total_degree_bits = 0;

    friend bool operator==(const gpu_round_shadow_pivot_counter&,
                           const gpu_round_shadow_pivot_counter&) = default;
};

/// Device-resident factor append format used by R2b. Columns are stored in
/// elimination order; each column names one contiguous range in entries.
struct gpu_round_shadow_factor_column {
    node_index vertex = 0;
    factor_value_t diag = 0;
    std::uint64_t entry_begin = 0;
    std::uint32_t entry_count = 0;
};

struct gpu_round_shadow_factor_entry {
    node_index neighbor = 0;
    factor_value_t value = 0;
};

struct gpu_round_shadow_factor_log {
    std::vector<gpu_round_shadow_factor_column> columns;
    std::vector<gpu_round_shadow_factor_entry> entries;
};

struct gpu_round_shadow_timings {
    double upload_ms = 0.0;
    double gather_ms = 0.0;
    double dedup_ms = 0.0;
    double factor_ms = 0.0;
    double sample_ms = 0.0;
    double fill_materialize_ms = 0.0;
    double mutate_ms = 0.0;
    double checksums_ms = 0.0;
    double download_ms = 0.0;
    double total_ms = 0.0;
    double reference_ms = 0.0;
};

struct gpu_round_shadow_report {
    gpu_round_shadow_digest factor;
    gpu_round_shadow_digest fill;
    gpu_round_shadow_digest residual;
    // Unlike residual (an order-independent multiset digest), this commits to
    // the live encounter order inside every owner slab.  A retained state may
    // feed another exact CPU-shadow round only when this also matches.
    gpu_round_shadow_digest ordered_residual;
    gpu_round_shadow_digest active;
    gpu_round_shadow_digest live_degree;
    // Canonical device/reference accumulation order only.  The authoritative
    // CPU excess vector is checked separately with per-vertex error bounds.
    gpu_round_shadow_digest canonical_excess;

    std::uint64_t input_incidences = 0;
    // Physical host snapshot entries can include dead slab records; a reused
    // resident generation contains only this many logical live entries.
    std::uint64_t resident_input_incidences = 0;
    std::uint64_t gathered_incidences = 0;
    std::uint64_t unique_neighbors = 0;
    std::uint64_t factor_entries = 0;
    std::uint64_t raw_fill_edges = 0;
    std::uint64_t excess_updates = 0;
    std::uint64_t excess_targets = 0;
    std::uint64_t surviving_input_incidences = 0;
    std::uint64_t active_count = 0;
    std::uint64_t live_incidences = 0;

    std::vector<gpu_round_shadow_pivot_counter> pivots;
    gpu_round_shadow_timings timings;
    std::size_t peak_device_bytes = 0;
    // R2a resident-state provenance.  Counts are cumulative within one device
    // session. round_state_upload_bytes is zero only when no state component
    // was uploaded; a resident reuse that refreshes an already-verified
    // bounded CPU excess difference reports n * sizeof(double).
    std::uint64_t input_generation = 0;
    std::uint64_t output_generation = 0;
    std::size_t round_state_upload_bytes = 0;
    std::size_t state_imports = 0;
    std::size_t state_reuses = 0;
    std::size_t order_reimports = 0;
    std::size_t host_rebuild_invalidations = 0;
    std::size_t host_rebuild_reimports = 0;
    std::size_t excess_refreshes = 0;
    std::size_t state_upload_bytes = 0;
    std::size_t state_buffer_allocations = 0;
    std::size_t state_buffer_growths = 0;
    std::size_t factor_log_columns = 0;
    std::size_t factor_log_entries = 0;
    std::size_t factor_log_allocations = 0;
    std::size_t factor_log_growths = 0;
    // Selection-validation work performed by this round. Every device
    // capability gets one O(p) selected-id pass. The optional independence
    // audit contributes exactly one O(r) incidence pass; zero means it did
    // not run.
    std::size_t selection_validation_pivots = 0;
    std::size_t selection_audit_passes = 0;
    std::size_t selection_audit_incidences = 0;
    std::size_t selection_map_initialization_vertices = 0;
    bool resident_input_reused = false;
    bool resident_selection_consumed = false;
    bool gpu_executed = false;
};

struct gpu_round_shadow_excess_bound {
    double expected = 0.0;
    double absolute_term_sum = 0.0;
    std::uint64_t additions = 0;
};

#if defined(__CUDACC__)
#define APXCHOL_ROUND_SHADOW_HD __host__ __device__
#else
#define APXCHOL_ROUND_SHADOW_HD
#endif

APXCHOL_ROUND_SHADOW_HD inline constexpr std::uint64_t
gpu_round_shadow_mix(std::uint64_t value) noexcept {
    return gpu_device_selection_mix(value);
}

APXCHOL_ROUND_SHADOW_HD inline constexpr std::uint64_t
gpu_round_shadow_item_hash(std::uint64_t tag, std::uint64_t a,
                           std::uint64_t b, std::uint64_t value) noexcept {
    return gpu_device_selection_item_hash(tag, a, b, value);
}

inline void gpu_round_shadow_digest_add(gpu_round_shadow_digest& digest,
                                        std::uint64_t item) noexcept {
    digest.xor_hash ^= item;
    digest.sum_hash += item;
}

namespace gpu_round_shadow_tags {
inline constexpr std::uint64_t factor_diag = 1;
inline constexpr std::uint64_t factor_entry = 2;
inline constexpr std::uint64_t fill = 3;
inline constexpr std::uint64_t active = 4;
inline constexpr std::uint64_t live_degree = 5;
inline constexpr std::uint64_t excess = 6;
inline constexpr std::uint64_t residual = 7;
inline constexpr std::uint64_t ordered_residual = 8;
} // namespace gpu_round_shadow_tags

APXCHOL_ROUND_SHADOW_HD inline constexpr std::uint64_t
gpu_round_shadow_ordered_residual_hash(
        std::uint64_t owner, std::uint64_t ordinal,
        std::uint64_t neighbor, std::uint64_t weight_bits) noexcept {
    const std::uint64_t incidence = gpu_round_shadow_item_hash(
        gpu_round_shadow_tags::residual, owner, neighbor, weight_bits);
    return gpu_round_shadow_item_hash(
        gpu_round_shadow_tags::ordered_residual, owner, ordinal, incidence);
}

inline std::uint64_t gpu_round_shadow_pool_bits(double value) noexcept {
    const pool_value_t stored = static_cast<pool_value_t>(value);
#if defined(APXCHOL_POOL_FP32)
    return std::bit_cast<std::uint32_t>(stored);
#else
    return std::bit_cast<std::uint64_t>(stored);
#endif
}

inline std::uint64_t gpu_round_shadow_factor_bits(
        factor_value_t value) noexcept {
    static_assert(sizeof(factor_value_t) == sizeof(std::uint32_t));
    return std::bit_cast<std::uint32_t>(value);
}

inline std::size_t gpu_round_shadow_checked_add(
        std::size_t a, std::size_t b, const char* what) {
    if (a > std::numeric_limits<std::size_t>::max() - b)
        throw std::overflow_error(std::string("GPU round shadow: ") + what +
                                  " size overflow");
    return a + b;
}

inline std::size_t gpu_round_shadow_checked_mul(
        std::size_t a, std::size_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error(std::string("GPU round shadow: ") + what +
                                  " size overflow");
    return a * b;
}

/// Validate the scalar capacities consumed by CUB and by packed 32-bit device
/// keys.  This function is intentionally separate and unit-testable: no CUDA
/// allocation is attempted until every count and the free-memory fit pass.
inline void gpu_round_shadow_validate_capacity_counts(
        node_index vertex_count, std::size_t incidence_count,
        std::size_t pivot_count, const gpu_round_shadow_report& shape) {
    constexpr std::uint64_t cub_max =
        static_cast<std::uint64_t>(std::numeric_limits<int>::max());
    const std::uint64_t counts[] = {
        vertex_count, incidence_count, pivot_count,
        shape.gathered_incidences, shape.unique_neighbors,
        shape.raw_fill_edges, shape.excess_updates, shape.excess_targets,
        shape.surviving_input_incidences, shape.live_incidences};
    for (std::uint64_t count : counts) {
        if (count > cub_max)
            throw std::overflow_error(
                "GPU round shadow: a CUB item count exceeds INT_MAX");
    }
    if (static_cast<std::uint64_t>(vertex_count) >
        std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error(
            "GPU round shadow: vertex ids do not fit the packed device key");
    const std::size_t residual_bound = gpu_round_shadow_checked_add(
        static_cast<std::size_t>(shape.surviving_input_incidences),
        gpu_round_shadow_checked_mul(
            static_cast<std::size_t>(shape.raw_fill_edges), 2,
            "directed fill"),
        "residual incidence");
    if (residual_bound != shape.live_incidences)
        throw std::logic_error(
            "GPU round shadow: reference residual count is inconsistent");
    if (residual_bound > static_cast<std::size_t>(cub_max))
        throw std::overflow_error(
            "GPU round shadow: directed residual exceeds INT_MAX");
}

inline void gpu_round_shadow_validate_capacity(
        const gpu_round_shadow_input& input,
        const gpu_round_shadow_report& shape) {
    gpu_round_shadow_validate_capacity_counts(
        input.vertex_count, input.incidences.size(), input.pivots.size(), shape);
}

template<incidence_storage Incidence>
gpu_round_shadow_input make_gpu_round_shadow_input(
        const graph<Incidence>& residual,
        std::span<const node_index> pivots,
        std::uint64_t run_seed) {
    if constexpr (!std::is_same_v<Incidence,
                                  directed_vec_pool_incidence>) {
        throw std::invalid_argument(
            "GPU round shadow requires directed vec_pool_aos storage");
    } else {
        gpu_round_shadow_input input;
        input.vertex_count = residual.n();
        input.owner_offsets.resize(
            static_cast<std::size_t>(residual.n()) + 1, 0);
        input.active.resize(static_cast<std::size_t>(residual.n()));
        input.excess.resize(static_cast<std::size_t>(residual.n()));
        input.pivots.assign(pivots.begin(), pivots.end());
        input.seeds.reserve(pivots.size());
        for (node_index pivot : pivots)
            input.seeds.push_back(
                gpu_round_shadow_pivot_seed(run_seed, pivot));

        std::size_t incidence_count = 0;
        for (node_index owner = 0; owner < residual.n(); ++owner) {
            input.owner_offsets[owner] = incidence_count;
            input.active[owner] = residual.is_active(owner) ? 1 : 0;
            input.excess[owner] = residual.excess(owner);
            incidence_count = gpu_round_shadow_checked_add(
                incidence_count,
                static_cast<std::size_t>(residual.adj_count(owner)),
                "host incidence snapshot");
        }
        input.owner_offsets[residual.n()] = incidence_count;
        input.incidences.reserve(incidence_count);
        for (node_index owner = 0; owner < residual.n(); ++owner) {
            for (const auto& [neighbor, weight] : residual.neighbors(owner))
                input.incidences.push_back({owner, neighbor, weight});
        }
        if (input.incidences.size() != incidence_count)
            throw std::logic_error(
                "GPU round shadow: adjacency changed during host snapshot");
        return input;
    }
}

namespace round_shadow_reference_detail {

struct neighbor_record {
    std::uint32_t pivot = 0;
    node_index neighbor = 0;
    double weight = 0.0;
};

struct fill_record {
    node_index u = 0;
    node_index v = 0;
    double weight = 0.0;
};

struct excess_record {
    node_index vertex = 0;
    double delta = 0.0;
};

inline std::uint64_t next_random(std::uint64_t& state) noexcept {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline double next_unit(std::uint64_t& state) noexcept {
    return double(next_random(state) >> 11) * 0x1.0p-53;
}

inline std::uint64_t double_bits(double value) noexcept {
    return std::bit_cast<std::uint64_t>(value);
}

inline void validate_input(const gpu_round_shadow_input& input) {
    const std::size_t n = static_cast<std::size_t>(input.vertex_count);
    if (input.active.size() != n || input.excess.size() != n)
        throw std::invalid_argument(
            "GPU round shadow: active/excess size does not match vertex count");
    if (input.owner_offsets.size() != n + 1 ||
        input.owner_offsets.front() != 0 ||
        input.owner_offsets.back() != input.incidences.size())
        throw std::invalid_argument(
            "GPU round shadow: owner offsets do not describe the incidence stream");
    for (std::size_t owner = 0; owner < n; ++owner) {
        if (input.owner_offsets[owner] > input.owner_offsets[owner + 1])
            throw std::invalid_argument(
                "GPU round shadow: owner offsets are not monotone");
        for (std::size_t i = input.owner_offsets[owner];
             i < input.owner_offsets[owner + 1]; ++i) {
            if (input.incidences[i].owner != static_cast<node_index>(owner))
                throw std::invalid_argument(
                    "GPU round shadow: incidence stream is not owner-major");
        }
        if (!std::isfinite(input.excess[owner]) || input.excess[owner] < 0.0)
            throw std::invalid_argument(
                "GPU round shadow: excess must be finite and nonnegative");
    }
    if (input.pivots.size() != input.seeds.size())
        throw std::invalid_argument(
            "GPU round shadow: pivot/seed count mismatch");

    std::vector<std::uint8_t> selected(n, 0);
    for (node_index pivot : input.pivots) {
        if (static_cast<std::size_t>(pivot) >= n)
            throw std::invalid_argument(
                "GPU round shadow: pivot is outside the graph");
        if (!input.active[pivot])
            throw std::invalid_argument(
                "GPU round shadow: selected pivot is inactive");
        if (std::exchange(selected[pivot], std::uint8_t{1}) != 0)
            throw std::invalid_argument(
                "GPU round shadow: selected pivot appears twice");
    }

    struct live_arc {
        node_index lo;
        node_index hi;
        std::uint64_t weight_bits;
        bool reverse;
    };
    std::vector<live_arc> arcs;
    arcs.reserve(input.incidences.size());
    for (const auto& edge : input.incidences) {
        if (static_cast<std::size_t>(edge.owner) >= n ||
            static_cast<std::size_t>(edge.neighbor) >= n)
            throw std::invalid_argument(
                "GPU round shadow: incidence endpoint is outside the graph");
        if (!std::isfinite(edge.weight) || edge.weight < 0.0)
            throw std::invalid_argument(
                "GPU round shadow: incidence weight must be finite and nonnegative");
        if (edge.owner == edge.neighbor)
            throw std::invalid_argument(
                "GPU round shadow: residual self-incidences are unsupported");
        if (!input.active[edge.owner] || !input.active[edge.neighbor])
            continue;
        if (selected[edge.owner] && selected[edge.neighbor])
            throw std::invalid_argument(
                "GPU round shadow: selected pivots are not independent");
        const bool reverse = edge.owner > edge.neighbor;
        arcs.push_back({std::min(edge.owner, edge.neighbor),
                        std::max(edge.owner, edge.neighbor),
                        double_bits(edge.weight), reverse});
    }
    std::sort(arcs.begin(), arcs.end(), [](const live_arc& a,
                                           const live_arc& b) {
        return std::tie(a.lo, a.hi, a.weight_bits, a.reverse) <
               std::tie(b.lo, b.hi, b.weight_bits, b.reverse);
    });
    for (std::size_t first = 0; first < arcs.size();) {
        std::size_t middle = first;
        while (middle < arcs.size() &&
               arcs[middle].lo == arcs[first].lo &&
               arcs[middle].hi == arcs[first].hi &&
               arcs[middle].weight_bits == arcs[first].weight_bits &&
               !arcs[middle].reverse)
            ++middle;
        std::size_t last = middle;
        while (last < arcs.size() &&
               arcs[last].lo == arcs[first].lo &&
               arcs[last].hi == arcs[first].hi &&
               arcs[last].weight_bits == arcs[first].weight_bits &&
               arcs[last].reverse)
            ++last;
        if (middle - first != last - middle)
            throw std::invalid_argument(
                "GPU round shadow: live directed residual is not symmetric");
        first = last;
    }
}

} // namespace round_shadow_reference_detail

struct gpu_round_shadow_state_fingerprint {
    gpu_round_shadow_digest residual;
    gpu_round_shadow_digest ordered_residual;
    gpu_round_shadow_digest active;
    gpu_round_shadow_digest topology;
    gpu_round_shadow_digest live_degree;
    gpu_round_shadow_digest excess;
    std::uint64_t active_count = 0;
    std::uint64_t live_incidences = 0;
};

/// Fingerprint the logical live state represented by a host snapshot.  Dead
/// incidences deliberately do not participate: the resident CUDA state
/// compacts them at every generation.  ordered_residual is insensitive to the
/// physical placement of owner slabs but commits to each live slab's encounter
/// order, which is the floating-point order used by process_vertex().
inline gpu_round_shadow_state_fingerprint
fingerprint_gpu_round_shadow_input(const gpu_round_shadow_input& input) {
    using round_shadow_reference_detail::double_bits;
    round_shadow_reference_detail::validate_input(input);
    gpu_round_shadow_state_fingerprint result;
    const std::size_t n = static_cast<std::size_t>(input.vertex_count);
    for (std::size_t owner = 0; owner < n; ++owner) {
        if (!input.active[owner]) continue;
        ++result.active_count;
        gpu_round_shadow_digest_add(
            result.active, gpu_device_selection_active_hash(
                static_cast<node_index>(owner)));
        std::uint64_t degree = 0;
        for (std::size_t i = input.owner_offsets[owner];
             i < input.owner_offsets[owner + 1]; ++i) {
            const auto& edge = input.incidences[i];
            if (!input.active[edge.neighbor]) continue;
            const std::uint64_t weight_bits =
                gpu_round_shadow_pool_bits(edge.weight);
            gpu_round_shadow_digest_add(
                result.residual,
                gpu_round_shadow_item_hash(
                    gpu_round_shadow_tags::residual, owner,
                    edge.neighbor, weight_bits));
            gpu_round_shadow_digest_add(
                result.ordered_residual,
                gpu_round_shadow_ordered_residual_hash(
                    owner, degree, edge.neighbor, weight_bits));
            if (owner < static_cast<std::size_t>(edge.neighbor))
                gpu_round_shadow_digest_add(
                    result.topology,
                    gpu_device_selection_topology_hash(
                        static_cast<node_index>(owner), edge.neighbor));
            ++degree;
            ++result.live_incidences;
        }
        gpu_round_shadow_digest_add(
            result.live_degree,
            gpu_round_shadow_item_hash(
                gpu_round_shadow_tags::live_degree, owner, 0, degree));
        gpu_round_shadow_digest_add(
            result.excess,
            gpu_round_shadow_item_hash(
                gpu_round_shadow_tags::excess, owner, 0,
                double_bits(input.excess[owner])));
    }
    return result;
}

inline gpu_device_selection_content gpu_round_shadow_selection_content(
        const gpu_round_shadow_input& input) {
    const auto fingerprint = fingerprint_gpu_round_shadow_input(input);
    return {
        input.vertex_count,
        static_cast<std::size_t>(fingerprint.active_count),
        {fingerprint.topology.xor_hash, fingerprint.topology.sum_hash},
        {fingerprint.active.xor_hash, fingerprint.active.sum_hash},
        {}};
}

/// Independent serial specification of one shadow round.  It intentionally
/// does not call tree_elimination::sample_clique or process_vertex.
inline gpu_round_shadow_report reference_gpu_round_shadow(
        const gpu_round_shadow_input& input,
        std::vector<gpu_round_shadow_excess_bound>* excess_bounds = nullptr) {
    using namespace round_shadow_reference_detail;
    validate_input(input);

    gpu_round_shadow_report report;
    report.input_incidences = input.incidences.size();
    report.pivots.resize(input.pivots.size());

    const std::size_t n = static_cast<std::size_t>(input.vertex_count);
    std::vector<std::int32_t> pivot_by_vertex(n, -1);
    for (std::size_t ordinal = 0; ordinal < input.pivots.size(); ++ordinal) {
        pivot_by_vertex[input.pivots[ordinal]] =
            static_cast<std::int32_t>(ordinal);
        report.pivots[ordinal].pivot = input.pivots[ordinal];
    }

    std::vector<neighbor_record> gathered;
    std::vector<double> raw_edge_degree(input.pivots.size(), 0.0);
    for (const auto& edge : input.incidences) {
        const std::int32_t ordinal = pivot_by_vertex[edge.owner];
        if (ordinal >= 0 && input.active[edge.owner] &&
            input.active[edge.neighbor]) {
            gathered.push_back({static_cast<std::uint32_t>(ordinal),
                                edge.neighbor, edge.weight});
            // Authoritative CPU process_vertex() accumulates degree here, over
            // every raw live slab incidence in encounter order.  It does NOT
            // re-sum the deduplicated neighbor array.
            raw_edge_degree[static_cast<std::size_t>(ordinal)] += edge.weight;
        }
    }
    report.gathered_incidences = gathered.size();
    std::stable_sort(gathered.begin(), gathered.end(),
        [](const neighbor_record& a, const neighbor_record& b) {
            return std::tie(a.pivot, a.neighbor) <
                   std::tie(b.pivot, b.neighbor);
        });

    std::vector<neighbor_record> unique;
    unique.reserve(gathered.size());
    for (std::size_t first = 0; first < gathered.size();) {
        std::size_t last = first + 1;
        double weight = gathered[first].weight;
        while (last < gathered.size() &&
               gathered[last].pivot == gathered[first].pivot &&
               gathered[last].neighbor == gathered[first].neighbor) {
            weight += gathered[last].weight;
            ++last;
        }
        unique.push_back({gathered[first].pivot,
                          gathered[first].neighbor, weight});
        first = last;
    }
    report.unique_neighbors = unique.size();
    report.factor_entries = unique.size();

    std::vector<std::size_t> offsets(input.pivots.size() + 1, 0);
    for (const auto& item : unique) ++offsets[item.pivot + 1];
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
    std::vector<double> total_degree(input.pivots.size(), 1.0);
    for (std::size_t ordinal = 0; ordinal < input.pivots.size(); ++ordinal) {
        const std::size_t begin = offsets[ordinal];
        const std::size_t end = offsets[ordinal + 1];
        double degree = raw_edge_degree[ordinal];
        const double ev = input.excess[input.pivots[ordinal]];
        if (begin == end)
            degree = ev > 0.0 ? ev : 1.0;
        else {
            degree += ev;
            if (degree <= 0.0) degree = 1.0;
        }
        total_degree[ordinal] = degree;
        auto& counter = report.pivots[ordinal];
        counter.unique_degree = end - begin;
        counter.total_degree_bits = double_bits(degree);

        const factor_value_t diag =
            static_cast<factor_value_t>(std::sqrt(degree));
        gpu_round_shadow_digest_add(
            report.factor,
            gpu_round_shadow_item_hash(
                gpu_round_shadow_tags::factor_diag,
                input.pivots[ordinal], input.pivots[ordinal],
                gpu_round_shadow_factor_bits(diag)));
        for (std::size_t i = begin; i < end; ++i) {
            const factor_value_t value = static_cast<factor_value_t>(
                unique[i].weight / std::sqrt(degree));
            gpu_round_shadow_digest_add(
                report.factor,
                gpu_round_shadow_item_hash(
                    gpu_round_shadow_tags::factor_entry,
                    input.pivots[ordinal], unique[i].neighbor,
                    gpu_round_shadow_factor_bits(value)));
        }
    }

    std::vector<neighbor_record> canonical = unique;
    std::sort(canonical.begin(), canonical.end(),
        [](const neighbor_record& a, const neighbor_record& b) {
            if (a.pivot != b.pivot) return a.pivot < b.pivot;
            if (a.weight != b.weight) return a.weight < b.weight;
            return a.neighbor < b.neighbor;
        });

    std::vector<fill_record> fills;
    fills.reserve(unique.size());
    std::vector<excess_record> excess_updates;
    excess_updates.reserve(unique.size());
    std::vector<double> prefix;
    for (std::size_t ordinal = 0; ordinal < input.pivots.size(); ++ordinal) {
        const std::size_t begin = offsets[ordinal];
        const std::size_t end = offsets[ordinal + 1];
        const std::size_t degree = end - begin;
        const double ev = input.excess[input.pivots[ordinal]];
        if (ev > 0.0) {
            for (std::size_t i = begin; i < end; ++i)
                excess_updates.push_back({
                    canonical[i].neighbor,
                    canonical[i].weight * ev / total_degree[ordinal]});
        }
        if (degree < 2) continue;
        prefix.resize(degree);
        prefix[0] = canonical[begin].weight;
        for (std::size_t local = 1; local < degree; ++local)
            prefix[local] = prefix[local - 1] +
                            canonical[begin + local].weight;
        std::uint64_t state = input.seeds[ordinal];
        for (std::size_t local = 0; local + 1 < degree; ++local) {
            const double suffix = prefix.back() - prefix[local];
            if (suffix <= 0.0) continue;
            const double target = prefix[local] + next_unit(state) * suffix;
            const auto found = std::upper_bound(
                prefix.begin() + static_cast<std::ptrdiff_t>(local) + 1,
                prefix.end(), target);
            std::size_t partner = static_cast<std::size_t>(
                found - prefix.begin());
            if (partner >= degree) partner = degree - 1;
            const node_index a = canonical[begin + local].neighbor;
            const node_index b = canonical[begin + partner].neighbor;
            fills.push_back({std::min(a, b), std::max(a, b),
                canonical[begin + local].weight * suffix /
                    total_degree[ordinal]});
            ++report.pivots[ordinal].emitted_edges;
        }
    }
    report.raw_fill_edges = fills.size();
    report.excess_updates = excess_updates.size();

    // graph<directed_vec_pool_incidence> is an append-only multigraph.  Every
    // sampled fill is stored separately and narrowed separately to pool_value_t;
    // colliding endpoint pairs must not be coalesced in fp64 here.
    for (const auto& edge : fills)
        gpu_round_shadow_digest_add(
            report.fill,
            gpu_round_shadow_item_hash(
                gpu_round_shadow_tags::fill, edge.u, edge.v,
                gpu_round_shadow_pool_bits(edge.weight)));

    std::vector<std::uint8_t> active = input.active;
    for (node_index pivot : input.pivots) active[pivot] = 0;
    std::vector<std::uint64_t> degrees(n, 0);
    std::vector<std::uint64_t> encounter_ordinals(n, 0);
    for (const auto& edge : input.incidences) {
        if (active[edge.owner] && active[edge.neighbor]) {
            ++degrees[edge.owner];
            ++report.surviving_input_incidences;
            const std::uint64_t weight_bits =
                gpu_round_shadow_pool_bits(edge.weight);
            gpu_round_shadow_digest_add(
                report.residual,
                gpu_round_shadow_item_hash(
                    gpu_round_shadow_tags::residual,
                    edge.owner, edge.neighbor, weight_bits));
            gpu_round_shadow_digest_add(
                report.ordered_residual,
                gpu_round_shadow_ordered_residual_hash(
                    edge.owner, encounter_ordinals[edge.owner]++,
                    edge.neighbor, weight_bits));
        }
    }
    for (const auto& edge : fills) {
        if (!active[edge.u] || !active[edge.v])
            throw std::logic_error(
                "GPU round shadow: fill touches an eliminated pivot");
        ++degrees[edge.u];
        ++degrees[edge.v];
        const std::uint64_t weight_bits =
            gpu_round_shadow_pool_bits(edge.weight);
        gpu_round_shadow_digest_add(
            report.residual,
            gpu_round_shadow_item_hash(
                gpu_round_shadow_tags::residual,
                edge.u, edge.v, weight_bits));
        gpu_round_shadow_digest_add(
            report.residual,
            gpu_round_shadow_item_hash(
                gpu_round_shadow_tags::residual,
                edge.v, edge.u, weight_bits));
        gpu_round_shadow_digest_add(
            report.ordered_residual,
            gpu_round_shadow_ordered_residual_hash(
                edge.u, encounter_ordinals[edge.u]++, edge.v, weight_bits));
        gpu_round_shadow_digest_add(
            report.ordered_residual,
            gpu_round_shadow_ordered_residual_hash(
                edge.v, encounter_ordinals[edge.v]++, edge.u, weight_bits));
    }
    report.live_incidences = gpu_round_shadow_checked_add(
        static_cast<std::size_t>(report.surviving_input_incidences),
        gpu_round_shadow_checked_mul(fills.size(), 2,
                                     "reference directed fill"),
        "reference residual");

    if (excess_bounds) {
        excess_bounds->resize(n);
        for (std::size_t vertex = 0; vertex < n; ++vertex) {
            (*excess_bounds)[vertex].expected = input.excess[vertex];
            (*excess_bounds)[vertex].absolute_term_sum =
                std::abs(input.excess[vertex]);
        }
        for (const auto& update : excess_updates) {
            auto& bound = (*excess_bounds)[update.vertex];
            bound.absolute_term_sum += std::abs(update.delta);
            ++bound.additions;
        }
    }

    std::stable_sort(excess_updates.begin(), excess_updates.end(),
        [](const excess_record& a, const excess_record& b) {
            return a.vertex < b.vertex;
        });
    std::vector<double> updated_excess = input.excess;
    for (std::size_t first = 0; first < excess_updates.size();) {
        std::size_t last = first + 1;
        double delta = excess_updates[first].delta;
        while (last < excess_updates.size() &&
               excess_updates[last].vertex == excess_updates[first].vertex) {
            delta += excess_updates[last].delta;
            ++last;
        }
        updated_excess[excess_updates[first].vertex] += delta;
        ++report.excess_targets;
        first = last;
    }
    if (excess_bounds) {
        for (std::size_t vertex = 0; vertex < n; ++vertex)
            (*excess_bounds)[vertex].expected = updated_excess[vertex];
    }

    for (node_index vertex = 0; vertex < input.vertex_count; ++vertex) {
        if (!active[vertex]) continue;
        ++report.active_count;
        gpu_round_shadow_digest_add(
            report.active, gpu_device_selection_active_hash(vertex));
        gpu_round_shadow_digest_add(
            report.live_degree,
            gpu_round_shadow_item_hash(
                gpu_round_shadow_tags::live_degree, vertex, 0,
                degrees[vertex]));
        gpu_round_shadow_digest_add(
            report.canonical_excess,
            gpu_round_shadow_item_hash(
                gpu_round_shadow_tags::excess, vertex, 0,
                double_bits(updated_excess[vertex])));
    }

    gpu_round_shadow_validate_capacity(input, report);
    return report;
}

inline void compare_gpu_round_shadow_reports(
        const gpu_round_shadow_report& expected,
        const gpu_round_shadow_report& actual) {
    auto mismatch = [](const char* field) {
        throw std::runtime_error(
            std::string("GPU round shadow mismatch: ") + field);
    };
    if (actual.factor != expected.factor) mismatch("factor digest");
    if (actual.fill != expected.fill) mismatch("fill digest");
    if (actual.residual != expected.residual) mismatch("residual digest");
    if (actual.ordered_residual != expected.ordered_residual)
        mismatch("ordered-residual digest");
    if (actual.active != expected.active) mismatch("active digest");
    if (actual.live_degree != expected.live_degree)
        mismatch("live-degree digest");
    if (actual.canonical_excess != expected.canonical_excess)
        mismatch("canonical excess digest");
#define APXCHOL_ROUND_SHADOW_COMPARE_FIELD(name) \
    if (actual.name != expected.name) mismatch(#name)
    APXCHOL_ROUND_SHADOW_COMPARE_FIELD(input_incidences);
    APXCHOL_ROUND_SHADOW_COMPARE_FIELD(gathered_incidences);
    APXCHOL_ROUND_SHADOW_COMPARE_FIELD(unique_neighbors);
    APXCHOL_ROUND_SHADOW_COMPARE_FIELD(factor_entries);
    APXCHOL_ROUND_SHADOW_COMPARE_FIELD(raw_fill_edges);
    APXCHOL_ROUND_SHADOW_COMPARE_FIELD(excess_updates);
    APXCHOL_ROUND_SHADOW_COMPARE_FIELD(excess_targets);
    APXCHOL_ROUND_SHADOW_COMPARE_FIELD(surviving_input_incidences);
    APXCHOL_ROUND_SHADOW_COMPARE_FIELD(active_count);
    APXCHOL_ROUND_SHADOW_COMPARE_FIELD(live_incidences);
#undef APXCHOL_ROUND_SHADOW_COMPARE_FIELD
    if (actual.pivots.size() != expected.pivots.size())
        mismatch("pivot counter count");
    for (std::size_t i = 0; i < expected.pivots.size(); ++i) {
        if (actual.pivots[i] != expected.pivots[i])
            throw std::runtime_error(
                "GPU round shadow mismatch: pivot counter " +
                std::to_string(i));
    }
}

struct gpu_round_shadow_cpu_comparison {
    std::size_t bounded_excess_vertices = 0;
};

/// Compare the CUDA shadow with the graph and factor columns materialized by
/// the real CPU elimination path.  Factor values, active state, multigraph
/// incidence endpoints/stored weight bits, and raw incidence degrees are exact.
/// Only a vertex receiving two or more excess updates gets a roundoff bound:
/// OpenMP atomics serialize those additions in an unspecified order.
template<incidence_storage Incidence, class FactorColumns>
gpu_round_shadow_cpu_comparison compare_gpu_round_shadow_with_cpu(
        const gpu_round_shadow_report& expected,
        std::span<const gpu_round_shadow_excess_bound> excess_bounds,
        const graph<Incidence>& residual,
        const FactorColumns& columns,
        const gpu_round_shadow_digest* streamed_entries = nullptr) {
    auto mismatch = [](const std::string& field) {
        throw std::runtime_error(
            "GPU round shadow/CPU mismatch: " + field);
    };
    if (columns.size() != expected.pivots.size())
        mismatch("factor-column count");

    gpu_round_shadow_digest factor;
    std::uint64_t factor_entries = 0;
    for (std::size_t ordinal = 0; ordinal < columns.size(); ++ordinal) {
        const auto& column = columns[ordinal];
        if (column.vertex != expected.pivots[ordinal].pivot)
            mismatch("factor-column pivot " + std::to_string(ordinal));
        gpu_round_shadow_digest_add(
            factor,
            gpu_round_shadow_item_hash(
                gpu_round_shadow_tags::factor_diag,
                column.vertex, column.vertex,
                gpu_round_shadow_factor_bits(column.diag)));
        if (streamed_entries) {
            if (column.entries != nullptr)
                mismatch("streamed factor round contains materialized entries");
            factor_entries += column.entry_count;
            continue;
        }
        if (column.entry_count != 0 && column.entries == nullptr)
            mismatch("factor entries missing without a streamed audit");
        for (std::size_t i = 0; i < column.entry_count; ++i) {
            const auto& entry = column.entries[i];
            gpu_round_shadow_digest_add(
                factor,
                gpu_round_shadow_item_hash(
                    gpu_round_shadow_tags::factor_entry,
                    column.vertex, entry.neighbor,
                    gpu_round_shadow_factor_bits(entry.value)));
            ++factor_entries;
        }
    }
    if (streamed_entries) {
        factor.xor_hash ^= streamed_entries->xor_hash;
        factor.sum_hash += streamed_entries->sum_hash;
    }
    if (factor != expected.factor) mismatch("factor digest");
    if (factor_entries != expected.factor_entries)
        mismatch("factor-entry count");

    gpu_round_shadow_digest active;
    gpu_round_shadow_digest live_degree;
    gpu_round_shadow_digest logical_residual;
    std::uint64_t active_count = 0;
    std::uint64_t live_incidences = 0;
    for (node_index owner = 0; owner < residual.n(); ++owner) {
        if (!residual.is_active(owner)) continue;
        ++active_count;
        gpu_round_shadow_digest_add(
            active, gpu_device_selection_active_hash(owner));
        std::uint64_t degree = 0;
        for (const auto& [neighbor, weight] : residual.neighbors(owner)) {
            if (!residual.is_active(neighbor)) continue;
            ++degree;
            ++live_incidences;
            gpu_round_shadow_digest_add(
                logical_residual,
                gpu_round_shadow_item_hash(
                    gpu_round_shadow_tags::residual,
                    owner, neighbor,
                    gpu_round_shadow_pool_bits(weight)));
        }
        gpu_round_shadow_digest_add(
            live_degree,
            gpu_round_shadow_item_hash(
                gpu_round_shadow_tags::live_degree, owner, 0, degree));
    }
    if (active != expected.active) mismatch("active digest");
    if (live_degree != expected.live_degree)
        mismatch("live-degree digest");
    if (logical_residual != expected.residual)
        mismatch("materialized residual endpoint/weight digest");
    if (active_count != expected.active_count) mismatch("active count");
    if (live_incidences != expected.live_incidences)
        mismatch("live-incidence count");

    if (excess_bounds.size() != static_cast<std::size_t>(residual.n()))
        mismatch("excess-bound size");
    gpu_round_shadow_cpu_comparison comparison;
    constexpr double eps = std::numeric_limits<double>::epsilon();
    constexpr double denorm = std::numeric_limits<double>::denorm_min();
    for (node_index vertex = 0; vertex < residual.n(); ++vertex) {
        if (!residual.is_active(vertex)) continue;
        const double actual = residual.excess(vertex);
        const auto& bound = excess_bounds[vertex];
        if (!std::isfinite(actual) || actual < 0.0)
            mismatch("non-finite or negative excess at vertex " +
                     std::to_string(vertex));
        if (bound.additions <= 1) {
            if (std::bit_cast<std::uint64_t>(actual) !=
                std::bit_cast<std::uint64_t>(bound.expected))
                mismatch("exact excess at vertex " +
                         std::to_string(vertex));
            continue;
        }
        ++comparison.bounded_excess_vertices;
        const double terms = static_cast<double>(bound.additions + 1);
        const double tolerance =
            32.0 * eps * terms * bound.absolute_term_sum +
            32.0 * denorm * terms;
        if (std::abs(actual - bound.expected) > tolerance)
            mismatch("bounded excess at vertex " +
                     std::to_string(vertex) + " (difference=" +
                     std::to_string(std::abs(actual - bound.expected)) +
                     ", bound=" + std::to_string(tolerance) + ")");
    }
    return comparison;
}

#if defined(APXCHOL_USE_CUDA)
bool gpu_round_shadow_runtime_available() noexcept;

/// CUDA-owned R2a state machine.  A produced generation is not eligible for
/// reuse until certify_cpu_round() is called after the authoritative CPU
/// comparison with the exact post-CPU fingerprint. cpu_order_reproducible=false
/// keeps the generation resident for provenance, but the next compute imports
/// the host snapshot explicitly. A certified bounded CPU excess collision may
/// refresh only the excess array; every byte is included in the resident-state
/// counters. An authoritative host graph replacement must be announced through
/// invalidate_for_authoritative_host_rebuild(): its fingerprint becomes the
/// new continuity anchor and the next compute performs a counted full import.
class gpu_round_shadow_device_state {
public:
    gpu_round_shadow_device_state();
    ~gpu_round_shadow_device_state();
    gpu_round_shadow_device_state(gpu_round_shadow_device_state&&) noexcept;
    gpu_round_shadow_device_state& operator=(
        gpu_round_shadow_device_state&&) noexcept;
    gpu_round_shadow_device_state(const gpu_round_shadow_device_state&) = delete;
    gpu_round_shadow_device_state& operator=(
        const gpu_round_shadow_device_state&) = delete;

#if defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
    /// Compile-time-only one-shot fault seam. Normal builds contain neither
    /// this method nor its state/branch.
    void inject_failure_after_cuda_operation_for_test();
#endif

    gpu_round_shadow_report compute(
        const gpu_round_shadow_input& input,
        const gpu_round_shadow_report& expected_shape);
    /// Execute one round without using a CPU reference to predict any
    /// intermediate stream size.  The device/CUB counts discovered by each
    /// stage size the following stage and are returned in the report.  This is
    /// the first R2b production boundary; callers may compare the completed
    /// report with an independent reference afterwards, but the reference is
    /// not an input to device execution.
    gpu_round_shadow_report compute_discover_shape(
        const gpu_round_shadow_input& input);
    /// As above, but consume the ordered selected ids from an existing CUDA
    /// allocation. input.pivots/seeds remain the independent host audit and
    /// must describe the same selection; they are not uploaded by this path.
    gpu_round_shadow_report compute_discover_shape(
        const gpu_round_shadow_input& input,
        gpu_device_selection selected,
        std::uint64_t run_seed);
    /// Debug/audit materialization of the persistent factor append log. The
    /// production R3 path will consume its device view instead.
    gpu_round_shadow_factor_log download_factor_log() const;
    // Finalize the resident prefix plus explicit CPU tail into two CUDA CSRs.
    // No prefix factor values or coordinates leave the device. Host permutation
    // and tail are uploaded; the existing adoption stage still downloads indices.
    std::shared_ptr<cuda_sptrsv_device_factor> finalize_fp32(
        std::span<const node_index> permutation, node_index m,
        const gpu_round_shadow_factor_log& tail);
    /// Execute the next round from the preceding CUDA generation without
    /// consuming a host snapshot. The preceding generation must have been
    /// explicitly accepted after its independent audit. Selection ids
    /// must carry a live, newer immutable capability from the same producer,
    /// CUDA device, active set and residual topology. This is an explicit R2b
    /// research boundary; the factorization session uses it only after exact
    /// serial CPU certification and audits the returned report independently.
    gpu_round_shadow_report compute_resident(
        gpu_device_selection selected,
        std::uint64_t run_seed);
    /// Project this accepted, CPU-certified weighted residual into the same
    /// selector that supplied the consumed selection. No CPU ids/endpoints are
    /// uploaded. This does not relax pending order/excess refresh requirements.
    void advance_selector(gpu_block_frontend& frontend);
    /// Mark the current CUDA generation consumable after its independent audit
    /// passed. Passing any other generation fails closed.
    void accept_device_generation(std::uint64_t generation);
    /// Permanently poison the current generation after a failed audit. A new
    /// gpu_round_shadow_device_state is then required.
    void reject_device_generation(std::uint64_t generation) noexcept;
    void certify_cpu_round(
        bool cpu_order_reproducible,
        const gpu_round_shadow_state_fingerprint& cpu_state,
        bool cpu_excess_may_differ = false);
    void invalidate_for_authoritative_host_rebuild(
        const gpu_round_shadow_state_fingerprint& cpu_state);

private:
    void reset() noexcept;
    struct impl;
    std::unique_ptr<impl> impl_;
};

gpu_round_shadow_report compute_gpu_round_shadow(
    const gpu_round_shadow_input& input,
    const gpu_round_shadow_report& expected_shape);
#else
inline bool gpu_round_shadow_runtime_available() noexcept { return false; }
inline gpu_round_shadow_report compute_gpu_round_shadow(
        const gpu_round_shadow_input&,
        const gpu_round_shadow_report&) {
    throw std::runtime_error(
        "GPU round shadow: CUDA support was not compiled");
}
#endif

inline gpu_round_shadow_report run_verified_gpu_round_shadow(
        const gpu_round_shadow_input& input,
        std::vector<gpu_round_shadow_excess_bound>* excess_bounds = nullptr) {
    using clock = std::chrono::steady_clock;
    const auto reference_begin = clock::now();
    const gpu_round_shadow_report expected =
        reference_gpu_round_shadow(input, excess_bounds);
    const double reference_ms = 1e3 * std::chrono::duration<double>(
        clock::now() - reference_begin).count();
    gpu_round_shadow_report actual =
        compute_gpu_round_shadow(input, expected);
    if (!actual.gpu_executed)
        throw std::runtime_error(
            "GPU round shadow mismatch: device execution marker");
    compare_gpu_round_shadow_reports(expected, actual);
    actual.timings.reference_ms = reference_ms;
    return actual;
}

#if defined(APXCHOL_USE_CUDA)
inline gpu_round_shadow_report run_verified_gpu_round_shadow(
        gpu_round_shadow_device_state& state,
        const gpu_round_shadow_input& input,
        const gpu_device_selection* selected,
        std::uint64_t run_seed,
        bool use_resident_selection,
        std::vector<gpu_round_shadow_excess_bound>* excess_bounds = nullptr) {
    using clock = std::chrono::steady_clock;
    // The FORCE-only sidecar still validates its complete host audit input.
    // The resident CUDA execution below does not consume that snapshot: its
    // default selection validator is O(p), while this independent O(r log r)
    // host work remains deliberately outside the performance claim.
    round_shadow_reference_detail::validate_input(input);
    // R2b deliberately executes before the independent host reference.  The
    // reference remains a fail-closed audit, but none of its intermediate
    // stream sizes can influence device allocation or control flow.
    gpu_round_shadow_report actual;
    if (selected && use_resident_selection) {
        actual = state.compute_resident(*selected, run_seed);
        // input_incidences describes the independently audited host snapshot;
        // resident_input_incidences describes the stream actually consumed.
        actual.input_incidences = input.incidences.size();
    } else if (selected) {
        actual = state.compute_discover_shape(input, *selected, run_seed);
    } else {
        actual = state.compute_discover_shape(input);
    }
    try {
        const auto reference_begin = clock::now();
        const gpu_round_shadow_report expected =
            reference_gpu_round_shadow(input, excess_bounds);
        const double reference_ms = 1e3 * std::chrono::duration<double>(
            clock::now() - reference_begin).count();
        if (!actual.gpu_executed)
            throw std::runtime_error(
                "GPU round shadow mismatch: device execution marker");
        compare_gpu_round_shadow_reports(expected, actual);
        state.accept_device_generation(actual.output_generation);
        actual.timings.reference_ms = reference_ms;
        return actual;
    } catch (...) {
        // A CUDA generation that did not pass its independent audit must never
        // become input to compute_resident().
        state.reject_device_generation(actual.output_generation);
        throw;
    }
}

inline gpu_round_shadow_report run_verified_gpu_round_shadow(
        gpu_round_shadow_device_state& state,
        const gpu_round_shadow_input& input,
        std::vector<gpu_round_shadow_excess_bound>* excess_bounds = nullptr) {
    return run_verified_gpu_round_shadow(
        state, input, nullptr, 0, false, excess_bounds);
}
#endif

class gpu_round_shadow_session {
public:
    gpu_round_shadow_session() = default;
    explicit gpu_round_shadow_session(bool active) : active_(active) {
#if defined(APXCHOL_USE_CUDA)
        if (active_)
            device_state_ =
                std::make_unique<gpu_round_shadow_device_state>();
#endif
    }

    gpu_round_shadow_session(gpu_round_shadow_session&&) noexcept = default;
    gpu_round_shadow_session& operator=(
        gpu_round_shadow_session&&) noexcept = default;
    gpu_round_shadow_session(const gpu_round_shadow_session&) = delete;
    gpu_round_shadow_session& operator=(
        const gpu_round_shadow_session&) = delete;

    bool active() const noexcept { return active_; }

    template<incidence_storage Incidence>
    void begin_round(const graph<Incidence>& residual,
                     std::span<const node_index> pivots,
                     std::uint64_t run_seed,
                     std::uint64_t round_index,
                     bool cpu_order_reproducible,
                     const gpu_device_selection* device_selection = nullptr) {
        if (!active_) return;
#if !defined(APXCHOL_USE_CUDA)
        (void)device_selection;
#endif
        if (pending_)
            throw std::logic_error(
                "GPU round shadow: a CPU round was not verified");
        ++attempted_rounds_;
        const gpu_round_shadow_input input = make_gpu_round_shadow_input(
            residual, pivots, run_seed);
#if defined(APXCHOL_USE_CUDA)
        const bool use_resident_selection =
            device_selection && resident_selection_ready_;
        resident_selection_ready_ = false;
        pending_report_ = run_verified_gpu_round_shadow(
            *device_state_, input, device_selection, run_seed,
            use_resident_selection,
            &pending_excess_bounds_);
#else
        pending_report_ = run_verified_gpu_round_shadow(
            input, &pending_excess_bounds_);
#endif
        pending_round_index_ = round_index;
        pending_cpu_order_reproducible_ = cpu_order_reproducible;
        pending_ = true;
    }

    template<incidence_storage Incidence, class FactorColumns>
    void verify_cpu_round(const graph<Incidence>& residual,
                          const FactorColumns& columns,
                          const gpu_round_shadow_digest* streamed_entries = nullptr) {
        if (!active_) return;
        if (!pending_)
            throw std::logic_error(
                "GPU round shadow: CPU verification has no pending round");
        const auto cpu_comparison = compare_gpu_round_shadow_with_cpu(
            pending_report_,
            std::span<const gpu_round_shadow_excess_bound>(
                pending_excess_bounds_),
            residual, columns, streamed_entries);
#if defined(APXCHOL_USE_CUDA)
        const auto cpu_state = fingerprint_gpu_round_shadow_input(
            make_gpu_round_shadow_input(
                residual, std::span<const node_index>{}, 0));
        device_state_->certify_cpu_round(
            pending_cpu_order_reproducible_, cpu_state,
            cpu_comparison.bounded_excess_vertices != 0);
        // A host-order divergence requires a full re-import, and a bounded
        // CPU excess divergence requires the next call to carry the host
        // vector so it can be compared/refreshed. Only the exact serial case
        // can safely take the host-state-free production consumer.
        resident_selection_ready_ = pending_cpu_order_reproducible_ &&
            cpu_comparison.bounded_excess_vertices == 0;
#endif
        const gpu_round_shadow_report& report = pending_report_;
        ++checked_rounds_;
        if (report.gpu_executed) ++device_executions_;
        pivots_ += report.pivots.size();
        input_incidences_ += report.input_incidences;
        raw_fill_edges_ += report.raw_fill_edges;
        peak_device_bytes_ = std::max(peak_device_bytes_,
                                      report.peak_device_bytes);
        state_imports_ = report.state_imports;
        state_reuses_ = report.state_reuses;
        order_reimports_ = report.order_reimports;
        host_rebuild_invalidations_ = report.host_rebuild_invalidations;
        host_rebuild_reimports_ = report.host_rebuild_reimports;
        excess_refreshes_ = report.excess_refreshes;
        state_upload_bytes_ = report.state_upload_bytes;
        state_buffer_allocations_ = report.state_buffer_allocations;
        state_buffer_growths_ = report.state_buffer_growths;
        factor_log_columns_ = report.factor_log_columns;
        factor_log_entries_ = report.factor_log_entries;
        factor_log_allocations_ = report.factor_log_allocations;
        factor_log_growths_ = report.factor_log_growths;
        totals_.upload_ms += report.timings.upload_ms;
        totals_.gather_ms += report.timings.gather_ms;
        totals_.dedup_ms += report.timings.dedup_ms;
        totals_.factor_ms += report.timings.factor_ms;
        totals_.sample_ms += report.timings.sample_ms;
        totals_.fill_materialize_ms += report.timings.fill_materialize_ms;
        totals_.mutate_ms += report.timings.mutate_ms;
        totals_.checksums_ms += report.timings.checksums_ms;
        totals_.download_ms += report.timings.download_ms;
        totals_.total_ms += report.timings.total_ms;
        totals_.reference_ms += report.timings.reference_ms;

        std::fprintf(stderr,
            "[gpu-round-shadow] round=%llu pivots=%zu input=%llu "
            "resident_input=%llu "
            "gathered=%llu unique=%llu fill=%llu active=%llu "
            "live=%llu factor=%016llx:%016llx fill_hash=%016llx:%016llx "
            "residual=%016llx:%016llx degree=%016llx:%016llx "
            "bounded_excess_vertices=%zu device=1 peak_bytes=%zu "
            "state=%s generation=%llu->%llu state_upload_bytes=%zu "
            "state_imports=%zu state_reuses=%zu order_reimports=%zu "
            "host_rebuild_invalidations=%zu host_rebuild_reimports=%zu "
            "excess_refreshes=%zu "
            "state_allocations=%zu state_growths=%zu "
            "factor_log=%zu/%zu factor_log_allocations=%zu "
            "factor_log_growths=%zu selection=%s selection_check=%zu/%zu/%zu "
            "stages_ms=%.6f/%.6f/%.6f/%.6f/%.6f/%.6f/%.6f/%.6f/%.6f "
            "reference_ms=%.6f\n",
            static_cast<unsigned long long>(pending_round_index_),
            report.pivots.size(),
            static_cast<unsigned long long>(report.input_incidences),
            static_cast<unsigned long long>(
                report.resident_input_incidences),
            static_cast<unsigned long long>(report.gathered_incidences),
            static_cast<unsigned long long>(report.unique_neighbors),
            static_cast<unsigned long long>(report.raw_fill_edges),
            static_cast<unsigned long long>(report.active_count),
            static_cast<unsigned long long>(report.live_incidences),
            static_cast<unsigned long long>(report.factor.xor_hash),
            static_cast<unsigned long long>(report.factor.sum_hash),
            static_cast<unsigned long long>(report.fill.xor_hash),
            static_cast<unsigned long long>(report.fill.sum_hash),
            static_cast<unsigned long long>(report.residual.xor_hash),
            static_cast<unsigned long long>(report.residual.sum_hash),
            static_cast<unsigned long long>(report.live_degree.xor_hash),
            static_cast<unsigned long long>(report.live_degree.sum_hash),
            cpu_comparison.bounded_excess_vertices,
            report.peak_device_bytes,
            report.resident_input_reused ? "resident" : "host-import",
            static_cast<unsigned long long>(report.input_generation),
            static_cast<unsigned long long>(report.output_generation),
            report.round_state_upload_bytes,
            report.state_imports, report.state_reuses,
            report.order_reimports, report.host_rebuild_invalidations,
            report.host_rebuild_reimports, report.excess_refreshes,
            report.state_buffer_allocations,
            report.state_buffer_growths,
            report.factor_log_columns, report.factor_log_entries,
            report.factor_log_allocations, report.factor_log_growths,
            report.resident_selection_consumed ? "resident" : "snapshot",
            report.selection_validation_pivots,
            report.selection_audit_passes,
            report.selection_audit_incidences,
            report.timings.upload_ms, report.timings.gather_ms,
            report.timings.dedup_ms, report.timings.factor_ms,
            report.timings.sample_ms, report.timings.fill_materialize_ms,
            report.timings.mutate_ms, report.timings.checksums_ms,
            report.timings.download_ms, report.timings.reference_ms);

        pending_ = false;
        pending_report_ = {};
        pending_excess_bounds_.clear();
    }

    /// Announce an authoritative host graph replacement after the preceding
    /// CPU round was verified. Unannounced mutations remain fail-closed. The
    /// rebuilt host fingerprint is accepted only as the next continuity anchor;
    /// the next device round must import the complete state.
    template<incidence_storage Incidence>
    void authoritative_host_rebuild(const graph<Incidence>& residual) {
        if (!active_) return;
        if (pending_)
            throw std::logic_error(
                "GPU round shadow: host rebuild occurred during a pending round");
#if defined(APXCHOL_USE_CUDA)
        const auto cpu_state = fingerprint_gpu_round_shadow_input(
            make_gpu_round_shadow_input(
                residual, std::span<const node_index>{}, 0));
        device_state_->invalidate_for_authoritative_host_rebuild(cpu_state);
        resident_selection_ready_ = false;
#else
        (void)residual;
#endif
    }

#if defined(APXCHOL_USE_CUDA)
    void advance_selector(gpu_block_frontend& frontend) {
        if (!active_ || pending_ || !device_state_)
            throw std::logic_error("GPU selector handoff requires a completed CPU audit");
        device_state_->advance_selector(frontend);
    }

    template<class Columns>
    std::shared_ptr<cuda_sptrsv_device_factor> finalize_fp32(
            const Columns& columns, std::span<const node_index> permutation,
            node_index m) {
        if (!active_ || pending_ || !device_state_ || factor_log_columns_ == 0)
            throw std::logic_error("GPU finalizer requires a completed, audited device prefix");
        if (factor_log_columns_ > columns.size())
            throw std::logic_error("GPU finalizer prefix exceeds CPU factor");
        gpu_round_shadow_factor_log tail;
        for (std::size_t i = factor_log_columns_; i < columns.size(); ++i) {
            const auto& c = columns[i];
            tail.columns.push_back({c.vertex, c.diag, tail.entries.size(),
                                    static_cast<std::uint32_t>(c.entry_count)});
            for (node_index j = 0; j < c.entry_count; ++j)
                tail.entries.push_back({c.entries[j].neighbor, c.entries[j].value});
        }
        return device_state_->finalize_fp32(permutation, m, tail);
    }
#endif

    void finish() const {
        if (!active_) return;
        if (pending_)
            throw std::logic_error(
                "GPU round shadow: final CPU round was not verified");
        if (state_imports_ + state_reuses_ != checked_rounds_)
            throw std::logic_error(
                "GPU round shadow: resident provenance denominator mismatch");
        std::fprintf(stderr,
            "[gpu-round-shadow] checked %zu/%zu rounds "
            "(deterministic_fields=exact excess=bounded device_executions=%zu "
            "pivots=%zu input_incidences=%llu fill=%llu "
            "peak_bytes=%zu state_imports=%zu state_reuses=%zu "
            "order_reimports=%zu host_rebuild_invalidations=%zu "
            "host_rebuild_reimports=%zu excess_refreshes=%zu "
            "state_upload_bytes=%zu "
            "state_allocations=%zu state_growths=%zu "
            "factor_log=%zu/%zu factor_log_allocations=%zu "
            "factor_log_growths=%zu "
            "total_ms=%.6f reference_ms=%.6f)\n",
            checked_rounds_, attempted_rounds_, device_executions_, pivots_,
            static_cast<unsigned long long>(input_incidences_),
            static_cast<unsigned long long>(raw_fill_edges_),
            peak_device_bytes_, state_imports_, state_reuses_,
            order_reimports_, host_rebuild_invalidations_,
            host_rebuild_reimports_, excess_refreshes_, state_upload_bytes_,
            state_buffer_allocations_, state_buffer_growths_,
            factor_log_columns_, factor_log_entries_, factor_log_allocations_,
            factor_log_growths_,
            totals_.total_ms, totals_.reference_ms);
    }

private:
    bool active_ = false;
    bool pending_ = false;
    std::size_t attempted_rounds_ = 0;
    std::size_t checked_rounds_ = 0;
    std::size_t device_executions_ = 0;
    std::size_t pivots_ = 0;
    std::uint64_t input_incidences_ = 0;
    std::uint64_t raw_fill_edges_ = 0;
    std::size_t peak_device_bytes_ = 0;
    std::size_t state_imports_ = 0;
    std::size_t state_reuses_ = 0;
    std::size_t order_reimports_ = 0;
    std::size_t host_rebuild_invalidations_ = 0;
    std::size_t host_rebuild_reimports_ = 0;
    std::size_t excess_refreshes_ = 0;
    std::size_t state_upload_bytes_ = 0;
    std::size_t state_buffer_allocations_ = 0;
    std::size_t state_buffer_growths_ = 0;
    std::size_t factor_log_columns_ = 0;
    std::size_t factor_log_entries_ = 0;
    std::size_t factor_log_allocations_ = 0;
    std::size_t factor_log_growths_ = 0;
    std::uint64_t pending_round_index_ = 0;
    bool pending_cpu_order_reproducible_ = false;
    gpu_round_shadow_report pending_report_;
    std::vector<gpu_round_shadow_excess_bound> pending_excess_bounds_;
    gpu_round_shadow_timings totals_;
#if defined(APXCHOL_USE_CUDA)
    std::unique_ptr<gpu_round_shadow_device_state> device_state_;
    bool resident_selection_ready_ = false;
#endif
};

template<typename Eliminator, incidence_storage Incidence>
gpu_round_shadow_session make_gpu_round_shadow_session(
        const Eliminator& eliminator) {
    if (gpu_factor_finalize_requested() && !gpu_round_shadow_requested())
        throw std::invalid_argument("GPU finalizer requires APXCHOL_GPU_ROUND_SHADOW=force");
    if (!gpu_round_shadow_requested()) return {};
    if (gpu_factor_finalize_requested() &&
        (sptrsv_fp16_env_tristate() != 0 || factor_drop_rel_from_env() != 0.0))
        throw std::invalid_argument("GPU finalizer prototype requires APXCHOL_SPTRSV_FP16=0 and APXCHOL_FACTOR_DROP=0");
#if !defined(APXCHOL_USE_CUDA)
    (void)eliminator;
    throw std::runtime_error(
        "forced GPU round shadow requires an APXCHOL_USE_CUDA build");
#else
    if constexpr (!std::is_same_v<Incidence,
                                  directed_vec_pool_incidence>) {
        (void)eliminator;
        throw std::invalid_argument(
            "forced GPU round shadow requires vec_pool_aos storage");
    } else if constexpr (!std::is_same_v<std::remove_cvref_t<Eliminator>,
                                         tree_elimination>) {
        (void)eliminator;
        throw std::invalid_argument(
            "forced GPU round shadow requires the built-in tree eliminator");
    } else {
        if (eliminator.exact_clique_max_degree != 0)
            throw std::invalid_argument(
                "forced GPU round shadow requires the default GKS d-1 tree "
                "sampler (exact-clique mode is unsupported)");
        if (!gpu_round_shadow_runtime_available())
            throw std::runtime_error(
                "forced GPU round shadow requires an available CUDA device");
        std::fprintf(stderr,
            "[gpu-round-shadow] enabled FORCE R2a resident research shadow; "
            "CPU residual remains authoritative; deterministic fields exact, "
            "colliding atomic excess bounded; parallel-order reimports "
            "explicitly counted\n");
        return gpu_round_shadow_session(true);
    }
#endif
}

#undef APXCHOL_ROUND_SHADOW_HD

} // namespace apxchol::detail
