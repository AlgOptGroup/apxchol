#pragma once
// CUDA-owned factor construction and its independent host reference routines.
// Production keeps the weighted residual, factor log and numerical rounds on
// the device. Test-only imports preserve raw incidence order and duplicates.

#include "apxchol/solver/detail/gpu_diagnostics.h"
#include "apxchol/graph/graph.h"
#include "apxchol/solver/elimination/elimination.h"
#include "apxchol/solver/gpu_device_selection.h"
#include "apxchol/sparse_csc.h"
#include "apxchol/lowprec.h"
#include "apxchol/solver/sptrsv/factor_drop.h"
#include "apxchol/env_knobs.h"
#if defined(APXCHOL_USE_CUDA)
#include "apxchol/solver/gpu_block_frontend.h"
#endif

#if !defined(__CUDACC__)
#include <Eigen/Sparse>
#include "apxchol/operator_class.h"
#endif
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <climits>
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
#include "apxchol/solver/detail/setup_route.h"
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace apxchol { class cuda_sptrsv_device_factor; }

namespace apxchol::detail {

inline constexpr std::string_view kGpuRoundSelectionAuditEnv =
    "APXCHOL_GPU_ROUND_SELECTION_AUDIT";

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

// Internal host-buffer ABI only. The owning import is synchronous, so these
// borrowed CSC arrays need survive only that call. CUDA never includes Eigen.
struct gpu_owned_csc_buffers {
    int vertex_count = 0;
    std::size_t nonzeros = 0;
    const int* offsets = nullptr;
    const int* rows = nullptr;
    const double* values = nullptr;
};
#if !defined(__CUDACC__)
// Internal eligibility, before any CUDA allocation or mutation.
inline bool gpu_owned_csc_layout_supported(const Eigen::SparseMatrix<double>& A,
                                           std::size_t& offdiagonals) {
    if (!A.isCompressed() || A.rows() != A.cols() || A.rows() <= 0 ||
        A.rows() >= INT_MAX || A.nonZeros() > INT_MAX) return false;
    const int n = static_cast<int>(A.rows());
    const int* ptr = A.outerIndexPtr();
    const int* idx = A.innerIndexPtr();
    if (ptr[0] != 0 || ptr[n] != A.nonZeros()) return false;
    bool sorted = true;
    offdiagonals = 0;
    #pragma omp parallel for schedule(static) reduction(&& : sorted) reduction(+ : offdiagonals)
    for (int v = 0; v < n; ++v) {
        if (ptr[v] < 0 || ptr[v] > ptr[v + 1] || ptr[v + 1] > A.nonZeros()) {
            sorted = false; continue;
        }
        for (int p = ptr[v]; p < ptr[v + 1]; ++p) {
            if (idx[p] < 0 || idx[p] >= n || (p > ptr[v] && idx[p-1] >= idx[p]))
                sorted = false;
            offdiagonals += idx[p] != v;
        }
    }
    return sorted;
}

// Requires the strict, unique compressed layout checked above.
inline bool gpu_owned_csc_paired(const Eigen::SparseMatrix<double>& A) {
    const int n = static_cast<int>(A.rows());
    const int* ptr = A.outerIndexPtr();
    const int* idx = A.innerIndexPtr();
    bool paired = true;
    std::size_t lower = 0, upper = 0;
    #pragma omp parallel for schedule(static) reduction(&& : paired) reduction(+ : lower, upper)
    for (int v = 0; v < n; ++v)
        for (int p = ptr[v]; p < ptr[v+1]; ++p) {
            const int u = idx[p];
            if (u > v) { ++lower; continue; }
            if (u == v) continue;
            ++upper;
            const int* mate = std::lower_bound(idx + ptr[u], idx + ptr[u+1], v);
            if (mate == idx + ptr[u+1] || *mate != v) paired = false;
        }
    return paired && lower == upper;
}
// Raw/test callers have no numerical symmetry certificate and keep the full check.
inline bool gpu_owned_csc_supported(const Eigen::SparseMatrix<double>& A) {
    std::size_t offdiagonals = 0;
    return gpu_owned_csc_layout_supported(A, offdiagonals) && gpu_owned_csc_paired(A);
}

// Internal dispatch passes its freshly constructed view before any input mutation.
// With unique columns and no stored off-diagonal zeros, successful validation
// already proves every lower entry has a nonzero mate and both triangle counts
// agree. A lumped view's scan describes its original input, so it cannot supply
// this proof for the transformed matrix. Stored zeros likewise need pairing.
inline bool gpu_owned_csc_supported(const operator_view& op) {
    const auto& A = op.matrix();
    std::size_t offdiagonals = 0;
    if (!gpu_owned_csc_layout_supported(A, offdiagonals)) return false;
    if (op.lumped() == 0 && offdiagonals == static_cast<std::size_t>(op.scan().offdiag_nnz))
        return true;
    return gpu_owned_csc_paired(A);
}
// Precondition: gpu_owned_csc_supported has succeeded on this same matrix.
inline gpu_owned_csc_buffers gpu_owned_csc_host_buffers(const Eigen::SparseMatrix<double>& A) {
    return {static_cast<int>(A.rows()), static_cast<std::size_t>(A.nonZeros()),
            A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr()};
}
#endif
#if defined(APXCHOL_USE_CUDA) && defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
struct gpu_owned_csc_test_result {
    bool eligible = false;
    gpu_round_shadow_input initial;
    bool sddm = false;
};
gpu_owned_csc_test_result gpu_initial_owned_csc_buffers_for_test(
    const gpu_owned_csc_buffers& input, double reg_eps);
#if !defined(__CUDACC__)
inline gpu_owned_csc_test_result gpu_initial_owned_csc_for_test(
        const Eigen::SparseMatrix<double>& A, double reg_eps = 0.0) {
    if (!gpu_owned_csc_supported(A)) return {};
    return gpu_initial_owned_csc_buffers_for_test(gpu_owned_csc_host_buffers(A), reg_eps);
}
#endif
#endif

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

// Complete owning logs already carry their raw off-diagonal prefix offsets.
// Preserve the existing coverage checks while optionally constructing the
// consuming factor's host metadata directly, without an intermediate factor_col.
inline void validate_owned_factor_columns(
        std::span<const gpu_round_shadow_factor_column> columns,
        std::size_t entry_count, std::span<node_index> permutation = {},
        std::span<edge_index> raw_offsets = {}) {
    const std::size_t n = columns.size();
    const bool metadata = !permutation.empty() || !raw_offsets.empty();
    if (metadata && (permutation.size() != n || raw_offsets.size() != n + 1))
        throw std::invalid_argument("GPU-owned factor metadata dimensions mismatch");
    std::vector<bool> seen(n, false);
    std::size_t next_entry = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const auto& c = columns[i];
        if (c.vertex >= n || seen[c.vertex] ||
            !std::isfinite(c.diag) || c.diag <= 0 || c.entry_begin != next_entry ||
            next_entry > entry_count || c.entry_count > entry_count - next_entry)
            throw std::logic_error("GPU-owned complete factor-header coverage mismatch");
        seen[c.vertex] = true;
        if (metadata) {
            // outer[i] = i + sum_{j<i} offdiagonal_count[j]. Both terms
            // are integers; this is exactly build_csc(false)'s recurrence.
            if (i > sparse_csc::kEdgeMax ||
                next_entry > sparse_csc::kEdgeMax - i)
                edge_index_overflow("factor_metadata(nnz)");
            permutation[c.vertex] = static_cast<node_index>(i);
            raw_offsets[i] = static_cast<edge_index>(next_entry + i);
        }
        next_entry += c.entry_count;
    }
    if (next_entry != entry_count)
        throw std::logic_error("GPU-owned complete factor entry extent mismatch");
    if (metadata) {
        if (n > sparse_csc::kEdgeMax || next_entry > sparse_csc::kEdgeMax - n)
            edge_index_overflow("factor_metadata(nnz)");
        raw_offsets[n] = static_cast<edge_index>(next_entry + n);
    }
}

struct gpu_round_shadow_factor_entry {
    node_index neighbor = 0;
    factor_value_t value = 0;
};

struct gpu_round_shadow_factor_log {
    std::vector<gpu_round_shadow_factor_column> columns;
    std::vector<gpu_round_shadow_factor_entry> entries;
};

struct gpu_round_shadow_timings {
    // False means all optional stage intervals are unavailable, not zero time.
    bool enabled = gpu_setup_diagnostics();
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

// Internal owned-path coverage receipt; zero on the independent audited route.
// Physical spans include inactive slots; gathered counts include live entries.
struct gpu_round_shadow_normal_batch_counts {
    std::uint64_t normal_pivots = 0, normal_spans = 0;
    std::uint64_t normal_gathered = 0, normal_unique = 0, batches = 0;
    std::uint64_t oversized_pivots = 0, oversized_spans = 0;
    std::uint64_t oversized_gathered = 0;
};

struct gpu_round_shadow_report {
    // Owning rounds retain scalar shape/generation and active continuity only.
    // Their factor/fill/weighted/ordered/degree/excess audit digests are zero
    // and pivots is empty. Generic CPU-shadow reports retain the full payload.
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
    gpu_round_shadow_normal_batch_counts normal_batch;
    std::uint64_t gathered_incidences = 0;
    std::uint64_t unique_neighbors = 0;
    std::uint64_t factor_entries = 0;
    std::uint64_t raw_fill_edges = 0;
    std::uint64_t sampler_numerical_fallbacks = 0;
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
    friend bool operator==(const gpu_round_shadow_state_fingerprint&,
                           const gpu_round_shadow_state_fingerprint&) = default;
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

// Only the matrix-consuming caller may use this linear first-import check:
// make_graph has already emitted each lower-canonical weight to both endpoints.
// Generic graph/snapshot inputs retain validate_input's complete paired sort.
// This does not compute unused weighted/order/excess audit fingerprints.
inline gpu_device_selection_content gpu_round_selection_content_from_paired_input(
        const gpu_round_shadow_input& input) {
    const std::size_t n = input.vertex_count, entries = input.incidences.size();
    if (input.active.size() != n || input.excess.size() != n ||
        input.owner_offsets.size() != n + 1 || input.owner_offsets.front() != 0 ||
        input.owner_offsets.back() != entries || entries % 2 ||
        input.pivots.size() != input.seeds.size())
        throw std::invalid_argument("GPU owned initial graph has invalid array extents");
    bool layout_valid = true;
    // Establish every offset bound before any traversal, even for malformed
    // input in a focused test. Each later row then owns a disjoint safe range.
    #pragma omp parallel for schedule(static) reduction(&& : layout_valid)
    for (std::size_t v = 0; v < n; ++v)
        if (input.owner_offsets[v] > input.owner_offsets[v + 1] ||
            input.owner_offsets[v + 1] > entries)
            layout_valid = false;
    if (!layout_valid)
        throw std::invalid_argument("GPU owned initial graph has invalid owner offsets");
    std::vector<std::uint8_t> selected(n, 0);
    for (node_index pivot : input.pivots) {
        if (std::size_t(pivot) >= n || selected[pivot])
            throw std::invalid_argument("GPU owned initial graph has invalid selected vertices");
        selected[pivot] = 1;
    }
    unsigned long long topology_xor = 0, topology_sum = 0;
    unsigned long long active_xor = 0, active_sum = 0, pairs = 0;
    bool valid = true;
    #pragma omp parallel for schedule(static) reduction(&& : valid) \
        reduction(^ : topology_xor, active_xor) reduction(+ : topology_sum, active_sum, pairs)
    for (std::size_t v = 0; v < n; ++v) {
        if (input.active[v] != 1 || !std::isfinite(input.excess[v]) || input.excess[v] < 0)
            valid = false;
        const auto hash = gpu_device_selection_active_hash(static_cast<node_index>(v));
        active_xor ^= hash; active_sum += hash;
        for (std::size_t k = input.owner_offsets[v]; k < input.owner_offsets[v + 1]; ++k) {
            const auto edge = input.incidences[k];
            if (edge.owner != node_index(v) || std::size_t(edge.neighbor) >= n ||
                edge.owner == edge.neighbor || !std::isfinite(edge.weight) || edge.weight < 0) {
                valid = false; continue;
            }
            if (selected[v] && selected[edge.neighbor]) valid = false;
            if (edge.owner < edge.neighbor) {
                const auto item = gpu_device_selection_topology_hash(edge.owner, edge.neighbor);
                topology_xor ^= item; topology_sum += item; ++pairs;
            }
        }
    }
    if (!valid || pairs != entries / 2)
        throw std::invalid_argument("GPU owned initial graph violates its construction contract");
    return {input.vertex_count, n, {topology_xor, topology_sum}, {active_xor, active_sum}, {}};
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

// Independent test observation of a weighted residual and factor headers.
struct gpu_owned_prefix_handback {
    gpu_round_shadow_input residual;
    gpu_round_shadow_state_fingerprint fingerprint;
    std::vector<gpu_round_shadow_factor_column> columns;
    std::size_t download_bytes = 0;
};

// Validate handback layout before a fingerprint traversal or graph publication.
// The general audit validator traverses owners as it checks their offsets;
// here all bounds must be known first because the arrays just crossed devices.
inline void validate_owned_prefix_handback(const gpu_owned_prefix_handback& handback,
                                          std::size_t factor_entries) {
    const auto& input = handback.residual;
    const std::size_t n = input.vertex_count;
    if (input.active.size() != n || input.excess.size() != n ||
        input.owner_offsets.size() != n + 1 || input.owner_offsets.front() != 0 ||
        input.owner_offsets.back() != input.incidences.size())
        throw std::logic_error("GPU-owned prefix handback array extent mismatch");
    for (std::size_t v = 0; v < n; ++v) {
        const auto a = input.owner_offsets[v], b = input.owner_offsets[v + 1];
        if (a > b || b > input.incidences.size() ||
            b - a > std::numeric_limits<node_index>::max() || input.active[v] > 1)
            throw std::logic_error("GPU-owned prefix handback owner bounds mismatch");
    }
    std::vector<bool> seen(n, false);
    std::uint64_t next_entry = 0;
    for (const auto& c : handback.columns) {
        if (c.vertex >= n || seen[c.vertex] || input.active[c.vertex] ||
            !std::isfinite(c.diag) || c.diag < 0 || c.entry_begin != next_entry ||
            next_entry > factor_entries || c.entry_count > factor_entries - next_entry)
            throw std::logic_error("GPU-owned prefix factor-header coverage mismatch");
        seen[c.vertex] = true;
        next_entry += c.entry_count;
    }
    for (std::size_t v = 0; v < n; ++v)
        if (seen[v] == bool(input.active[v]))
            throw std::logic_error("GPU-owned prefix inactive/column coverage mismatch");
    if (next_entry != factor_entries)
        throw std::logic_error("GPU-owned prefix factor entry range mismatch");
    if (fingerprint_gpu_round_shadow_input(input) != handback.fingerprint)
        throw std::logic_error("GPU-owned prefix downloaded residual fingerprint mismatch");
}

#if defined(APXCHOL_USE_CUDA)
bool gpu_round_shadow_runtime_available() noexcept;

struct gpu_owned_sparsify_stats {
    std::size_t physical_before = 0, distinct_before = 0;
    std::size_t backbone_edges = 0, kept_edges = 0, normalization_blocks = 0;
    std::size_t scalar_download_bytes = 0, peak_device_bytes = 0;
    unsigned normalization_passes = 0;
    double importance_scale = 0.0, expected_kept_edges = 0.0;
    double min_offtree_probability = 1.0, max_inverse_probability = 1.0;
    double total_weight = 0.0, backbone_weight = 0.0;
};

// Internal value type: production never requests the edge-sized downloads.
struct gpu_owned_sparsify_test_output {
    gpu_round_shadow_input residual;
    std::vector<gpu_round_shadow_incidence> canonical;
    std::vector<std::uint8_t> backbone, kept;
    std::vector<double> probabilities;
    gpu_owned_sparsify_stats stats;
};
#if defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
gpu_owned_sparsify_test_output gpu_sparsify_owned_residual_for_test(
    const gpu_round_shadow_input&, std::uint64_t seed,
    double keep_probability = 0.25, double scale_override = -1.0);
#endif

// One device owner publishes immutable residual generations to its selector.
// Every failed mutation poisons the owner and revokes outstanding borrows.
class gpu_round_shadow_device_state {
public:
    explicit gpu_round_shadow_device_state(clique_sampler sampler = clique_sampler::gks);
    ~gpu_round_shadow_device_state();
    gpu_round_shadow_device_state(gpu_round_shadow_device_state&&) noexcept;
    gpu_round_shadow_device_state& operator=(gpu_round_shadow_device_state&&) noexcept;
    gpu_round_shadow_device_state(const gpu_round_shadow_device_state&) = delete;
    gpu_round_shadow_device_state& operator=(const gpu_round_shadow_device_state&) = delete;

    void advance_selector(gpu_block_frontend& frontend);
    void reject_device_generation(std::uint64_t generation) noexcept;
    std::shared_ptr<cuda_sptrsv_device_factor> finalize_device_factor(
        std::span<const node_index> permutation, node_index m,
        factor_storage_type storage = factor_storage_type::automatic);
#if defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
    void inject_failure_after_cuda_operation_for_test();
    std::vector<std::byte> encode_finalized_factor_for_test(
        const cuda_sptrsv_device_factor& factor) const;
    std::unique_ptr<gpu_block_frontend> initialize_owned_raw_for_test(
        const gpu_round_shadow_input& input);
    gpu_round_shadow_report compute_owned_round_for_test(
            gpu_device_selection selected, std::uint64_t seed) {
        return compute_owned_prefix(selected, seed);
    }
    gpu_owned_prefix_handback download_owned_prefix_for_test() {
        return download_owned_prefix();
    }
    gpu_round_shadow_factor_log download_factor_log() const;
    std::vector<gpu_round_shadow_factor_column> complete_owned_factorization_for_test() {
        return download_owned_factor_columns();
    }
#endif

private:
    friend class gpu_round_shadow_session;
    double probe_owned_distinct_degree(
        std::span<const node_index> active_ids, gpu_block_frontend& frontend);
    gpu_owned_sparsify_stats sparsify_owned_residual(
        gpu_block_frontend& frontend, std::uint64_t seed);
    std::unique_ptr<gpu_block_frontend> initialize_owned_csc(
        const gpu_owned_csc_buffers& input, double reg_eps, bool& sddm);
    gpu_round_shadow_report compute_owned_prefix(
        gpu_device_selection selected, std::uint64_t run_seed);
    std::vector<gpu_round_shadow_factor_column> download_owned_factor_columns(
        std::span<node_index> permutation = {}, std::span<edge_index> raw_offsets = {});
#if defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
    friend gpu_owned_csc_test_result gpu_initial_owned_csc_buffers_for_test(
        const gpu_owned_csc_buffers&, double);
    gpu_owned_prefix_handback download_owned_prefix();
#endif
    void reset() noexcept;
    struct impl;
    std::unique_ptr<impl> impl_;
};
#else
inline bool gpu_round_shadow_runtime_available() noexcept { return false; }
#endif

class gpu_round_shadow_session {
public:
    gpu_round_shadow_session() = default;
    explicit gpu_round_shadow_session(bool active,
            clique_sampler sampler = clique_sampler::gks)
        : active_(active) {
#if defined(APXCHOL_USE_CUDA)
        if (active_)
            device_state_ =
                std::make_unique<gpu_round_shadow_device_state>(sampler);
#else
        (void)sampler;
#endif
    }

    gpu_round_shadow_session(gpu_round_shadow_session&&) noexcept = default;
    gpu_round_shadow_session& operator=(
        gpu_round_shadow_session&&) noexcept = default;
    gpu_round_shadow_session(const gpu_round_shadow_session&) = delete;
    gpu_round_shadow_session& operator=(
        const gpu_round_shadow_session&) = delete;

    bool active() const noexcept { return active_; }

#if defined(APXCHOL_USE_CUDA)
    // Internal dispatch has checked gpu_owned_csc_supported on the same matrix
    // before its host adapter produces these trusted, synchronously borrowed buffers.
    bool initialize_owned_csc(const gpu_owned_csc_buffers& input,
                              std::unique_ptr<gpu_block_frontend>& frontend) {
        if (!active_ || owned_initialized_ || owned_rounds_ || frontend)
            throw std::logic_error("GPU CSC initialization is outside a fresh owning session");
        const auto& knobs = env_knobs::get();
        const double reg = knobs.ground == grounding_kind::reg ? knobs.reg_eps : 0.0;
        bool sddm = false;
        frontend = device_state_->initialize_owned_csc(
            input, reg, sddm);
        owned_initialized_ = true;
        return sddm;
    }

    bool has_owned_residual() const noexcept {
        return owned_initialized_;
    }

    double probe_owned_distinct_degree(
            std::span<const node_index> active_ids, gpu_block_frontend& frontend) {
        if (!active_ || owned_materialized_ || !has_owned_residual())
            throw std::logic_error("GPU residual probe requires an accepted owned boundary");
        return device_state_->probe_owned_distinct_degree(active_ids, frontend);
    }

    gpu_owned_sparsify_stats sparsify_owned_residual(
            gpu_block_frontend& frontend, std::uint64_t seed) {
        if (!active_ || owned_materialized_ || !has_owned_residual())
            throw std::logic_error("GPU sparsification requires an accepted owned boundary");
        auto stats = device_state_->sparsify_owned_residual(frontend, seed);
        return stats;
    }

    gpu_round_shadow_report run_owned_prefix_round(
            gpu_device_selection selected, std::uint64_t seed) {
        if (!active_ || !owned_initialized_ || owned_materialized_)
            throw std::logic_error("GPU round is outside its initialized owning session");
        auto report = device_state_->compute_owned_prefix(selected, seed);
        ++owned_rounds_;
        factor_log_columns_ = report.factor_log_columns;
        state_imports_ = report.state_imports;
        state_reuses_ = report.state_reuses;
        state_upload_bytes_ = report.state_upload_bytes;
        if (gpu_setup_diagnostics()) std::fprintf(stderr, "[gpu-owned-prefix] round=%zu pivots=%zu generation=%llu "
            "state_upload_bytes=%zu active=%llu live=%llu "
            "peak_round_bytes=%zu normal_batch=%llu/%llu/%llu/%llu/%llu "
            "oversized=%llu/%llu/%llu "
            "stages_ms=%.6f/%.6f/%.6f/%.6f/%.6f/%.6f/%.6f/%.6f/%.6f total_ms=%.6f\n",
            owned_rounds_, report.selection_validation_pivots,
            static_cast<unsigned long long>(report.output_generation),
            report.round_state_upload_bytes,
            static_cast<unsigned long long>(report.active_count),
            static_cast<unsigned long long>(report.live_incidences), report.peak_device_bytes,
            static_cast<unsigned long long>(report.normal_batch.normal_pivots),
            static_cast<unsigned long long>(report.normal_batch.normal_spans),
            static_cast<unsigned long long>(report.normal_batch.normal_gathered),
            static_cast<unsigned long long>(report.normal_batch.normal_unique),
            static_cast<unsigned long long>(report.normal_batch.batches),
            static_cast<unsigned long long>(report.normal_batch.oversized_pivots),
            static_cast<unsigned long long>(report.normal_batch.oversized_spans),
            static_cast<unsigned long long>(report.normal_batch.oversized_gathered),
            report.timings.upload_ms, report.timings.gather_ms, report.timings.dedup_ms,
            report.timings.factor_ms, report.timings.sample_ms, report.timings.fill_materialize_ms,
            report.timings.mutate_ms, report.timings.checksums_ms, report.timings.download_ms,
            report.timings.total_ms);
        return report;
    }

#if defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
    // A complete resident run needs column metadata for the existing host
    // permutation contract, never a residual or factor-entry handback.
    template<class Columns>
    void complete_owned_factorization(Columns& columns) {
        if (!active_ || !owned_rounds_ || owned_materialized_ || !columns.empty())
            throw std::logic_error("GPU-owned factorization completion is out of order");
        try {
            const auto headers = device_state_->download_owned_factor_columns();
            columns.reserve(headers.size());
            for (const auto& c : headers)
                columns.push_back({c.vertex, c.diag, nullptr, static_cast<node_index>(c.entry_count)});
            owned_download_bytes_ = headers.size() * sizeof(gpu_round_shadow_factor_column);
            owned_materialized_ = true;
            owned_complete_ = true;
        } catch (...) {
            device_state_->reject_device_generation(owned_rounds_);
            throw;
        }
    }

#endif

    // Only permutation and raw CSC metadata leave the completed device owner.
    void complete_owned_factorization(std::vector<node_index>& permutation,
                                      sparse_csc& metadata) {
        if (!active_ || !owned_rounds_ || owned_materialized_ ||
            !permutation.empty() || !metadata.outer_.empty())
            throw std::logic_error("GPU-owned factorization completion is out of order");
        try {
            permutation.resize(factor_log_columns_);
            metadata.resize(static_cast<node_index>(factor_log_columns_),
                            static_cast<node_index>(factor_log_columns_));
            const auto headers = device_state_->download_owned_factor_columns(
                permutation, metadata.outer_);
            owned_download_bytes_ = headers.size() * sizeof(gpu_round_shadow_factor_column);
            owned_materialized_ = true;
            owned_complete_ = true;
        } catch (...) {
            device_state_->reject_device_generation(owned_rounds_);
            throw;
        }
    }

    void advance_selector(gpu_block_frontend& frontend) {
        if (!active_ || !device_state_)
            throw std::logic_error("GPU selector handoff requires a completed accepted generation");
        device_state_->advance_selector(frontend);
    }

    std::shared_ptr<cuda_sptrsv_device_factor> finalize_device_factor(
            std::span<const node_index> permutation, node_index m,
            factor_storage_type storage = factor_storage_type::automatic) {
        if (!active_ || !owned_complete_ || !device_state_ ||
            permutation.size() != factor_log_columns_)
            throw std::logic_error("GPU finalizer requires complete owned factor coverage");
        return device_state_->finalize_device_factor(permutation, m, storage);
    }

#if defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
    void initialize_owned_raw_for_test(const gpu_round_shadow_input& input,
            std::unique_ptr<gpu_block_frontend>& frontend) {
        if (!active_ || owned_initialized_ || owned_rounds_ || frontend)
            throw std::logic_error("Raw owned initialization requires a fresh test session");
        frontend = device_state_->initialize_owned_raw_for_test(input);
        owned_initialized_ = true;
    }

    gpu_owned_prefix_handback download_owned_prefix_for_test() {
        return device_state_->download_owned_prefix();
    }
    gpu_round_shadow_factor_log download_factor_log_for_test() const {
        return device_state_->download_factor_log();
    }

#endif
#endif

    void finish() const {
        if (!active_) return;
        if (!owned_rounds_ || !owned_complete_)
            throw std::logic_error("GPU factorization output was not completed");
        if (gpu_setup_diagnostics()) std::fprintf(stderr,
            "[gpu-owned-factorization] complete rounds=%zu state_imports=%zu state_reuses=%zu "
            "state_upload_bytes=%zu factor_header_count=%zu factor_header_download_bytes=%zu "
            "factor_entry_download_bytes=0\n",
            owned_rounds_, state_imports_, state_reuses_, state_upload_bytes_,
            factor_log_columns_, owned_download_bytes_);
    }

private:
    bool active_ = false;
    std::size_t owned_rounds_ = 0;
    bool owned_complete_ = false;
    std::size_t owned_download_bytes_ = 0;
    std::size_t state_imports_ = 0;
    std::size_t state_reuses_ = 0;
    std::size_t state_upload_bytes_ = 0;
    std::size_t factor_log_columns_ = 0;
#if defined(APXCHOL_USE_CUDA)
    bool owned_initialized_ = false;
    bool owned_materialized_ = false;
    std::unique_ptr<gpu_round_shadow_device_state> device_state_;
#endif
};

template<typename Eliminator, incidence_storage Incidence>
gpu_round_shadow_session make_gpu_round_shadow_session(
        const Eliminator& eliminator, setup_route route = setup_route::cpu) {
    if (route == setup_route::cpu) return {};
#if !defined(APXCHOL_USE_CUDA)
    (void)eliminator;
    throw std::runtime_error("GPU factorization requires CUDA support");
#else
    if constexpr (!std::is_same_v<Incidence, directed_vec_pool_incidence> ||
                  !std::is_same_v<std::remove_cvref_t<Eliminator>, tree_elimination>) {
        (void)eliminator;
        throw std::invalid_argument("GPU factorization requires vec_pool_aos and the built-in eliminator");
    } else {
        if (eliminator.exact_clique_max_degree != 0 || eliminator.exact_core_max_h != 0 ||
            eliminator.double_cycle_min_h != 0)
            throw std::invalid_argument("GPU factorization requires a bounded built-in sampler");
        if (!gpu_round_shadow_runtime_available())
            throw std::runtime_error("GPU factorization requires an available CUDA device");
        return gpu_round_shadow_session(true, eliminator.sampler);
    }
#endif
}

#undef APXCHOL_ROUND_SHADOW_HD

} // namespace apxchol::detail
