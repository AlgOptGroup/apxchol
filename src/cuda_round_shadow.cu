#include "apxchol/solver/elimination/gpu_round_shadow.h"
#include "apxchol/solver/sptrsv/cuda.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace apxchol::detail {
namespace {

constexpr int kBlock = 256;
// R3 prototype: finalization reads resident log records in place. Only the
// explicit CPU tail and final permutation enter from the host.
__device__ gpu_round_shadow_factor_column finalizer_column(
        int i, int prefix, const gpu_round_shadow_factor_column* a,
        const gpu_round_shadow_factor_column* b) {
    return i < prefix ? a[i] : b[i - prefix];
}

__global__ void finalizer_counts(
        int n, int m, int prefix,
        const gpu_round_shadow_factor_column* columns,
        const gpu_round_shadow_factor_entry* entries, std::size_t entry_count,
        const gpu_round_shadow_factor_column* tail_columns,
        const gpu_round_shadow_factor_entry* tail_entries, std::size_t tail_count,
        const node_index* perm, int* counts, int* status) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const auto c = finalizer_column(i, prefix, columns, tail_columns);
    const auto* es = i < prefix ? entries : tail_entries;
    const std::size_t cap = i < prefix ? entry_count : tail_count;
    if (c.vertex >= node_index(n) || perm[c.vertex] != node_index(i) ||
        c.entry_begin > cap || c.entry_count > cap - c.entry_begin ||
        !isfinite(c.diag) || c.diag <= 0) {
        atomicExch(status, 1); counts[i + 1] = 0; return;
    }
    int kept = 1, raw = 1;
    for (unsigned j = 0; j < c.entry_count; ++j) {
        const auto e = es[c.entry_begin + j];
        if (e.neighbor >= node_index(n) || !isfinite(e.value)) {
            atomicExch(status, 2); continue;
        }
        const node_index row = perm[e.neighbor];
        if (row <= node_index(i) || row >= node_index(n)) {
            atomicExch(status, 3); continue;
        }
        if (row < node_index(m)) { ++raw; ++kept; }
    }
    counts[i + 1] = kept;
    atomicAdd(status + 1, raw);
}

__global__ void finalizer_emit(
        int m, int prefix, const gpu_round_shadow_factor_column* columns,
        const gpu_round_shadow_factor_entry* entries,
        const gpu_round_shadow_factor_column* tail_columns,
        const gpu_round_shadow_factor_entry* tail_entries,
        const node_index* perm, const int* ptr,
        std::uint64_t* keys, float* vals) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const auto c = finalizer_column(i, prefix, columns, tail_columns);
    const auto* es = i < prefix ? entries : tail_entries;
    int pos = ptr[i];
    keys[pos] = (std::uint64_t(i) << 32) | unsigned(i);
    vals[pos++] = c.diag;
    for (unsigned j = 0; j < c.entry_count; ++j) {
        const auto e = es[c.entry_begin + j];
        const node_index row = perm[e.neighbor];
        if (row >= node_index(m)) continue;
        keys[pos] = (std::uint64_t(i) << 32) | row;
        vals[pos++] = -e.value;
    }
}

__global__ void finalizer_unpack(
        int nnz, const std::uint64_t* keys, int* idx, int* status,
        std::uint64_t* transposed_keys, int* transposed_counts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    const auto key = keys[i];
    const int row = int(key >> 32), col = int(key & 0xffffffffu);
    idx[i] = col;
    if (i > 0 && keys[i - 1] == key) atomicExch(status, 4);
    if (transposed_keys) {
        transposed_keys[i] = (std::uint64_t(col) << 32) | unsigned(row);
        atomicAdd(transposed_counts + col + 1, 1);
    }
}


[[noreturn]] void cuda_failure(cudaError_t error, const char* what) {
    throw std::runtime_error(std::string("GPU round shadow: ") + what +
                             ": " + cudaGetErrorString(error));
}

void cuda_check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) cuda_failure(error, what);
}

int blocks_for(std::size_t count) {
    if (count == 0) return 0;
    const std::size_t blocks = (count + kBlock - 1) / kBlock;
    if (blocks > static_cast<std::size_t>(INT_MAX))
        throw std::overflow_error(
            "GPU round shadow: CUDA grid exceeds INT_MAX blocks");
    return static_cast<int>(blocks);
}

int cub_count(std::size_t count, const char* what) {
    if (count > static_cast<std::size_t>(INT_MAX))
        throw std::overflow_error(std::string("GPU round shadow: ") + what +
                                  " exceeds CUB INT_MAX capacity");
    return static_cast<int>(count);
}

std::size_t geometric_capacity(std::size_t required) {
    if (required <= 1) return required;
    constexpr std::size_t highest_power =
        std::size_t{1} << (std::numeric_limits<std::size_t>::digits - 1);
    return required > highest_power ? required : std::bit_ceil(required);
}

struct allocation_tracker {
    std::size_t current = 0;
    std::size_t peak = 0;

    void add(std::size_t bytes) {
        current = gpu_round_shadow_checked_add(current, bytes,
                                               "device allocation tracker");
        peak = std::max(peak, current);
    }

    void remove(std::size_t bytes) noexcept {
        current -= bytes;
    }

    void begin_round() noexcept { peak = current; }
};

template<class T>
class device_buffer {
public:
    explicit device_buffer(allocation_tracker& tracker) : tracker_(&tracker) {}
    ~device_buffer() {
        if (data_) {
            cudaFree(data_);
            tracker_->remove(bytes_);
        }
    }

    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;

    void allocate(std::size_t count, const char* what) {
        if (data_ || count == 0) return;
        const std::size_t bytes =
            gpu_round_shadow_checked_mul(count, sizeof(T), what);
        // Account first so an overflow cannot happen after cudaMalloc. Roll the
        // accounting back if allocation itself fails; data_ is published only
        // after both operations succeed.
        tracker_->add(bytes);
        T* allocated = nullptr;
        const cudaError_t status = cudaMalloc(
            reinterpret_cast<void**>(&allocated), bytes);
        if (status != cudaSuccess) {
            tracker_->remove(bytes);
            cuda_failure(status, what);
        }
        data_ = allocated;
        bytes_ = bytes;
        count_ = count;
    }

    T* detach() noexcept {
        T* result = data_;
        tracker_->remove(bytes_);
        data_ = nullptr; bytes_ = count_ = 0;
        return result;
    }

    void release(const char* what) {
        if (!data_) return;
        // Retain ownership if cudaFree reports an error, so stack unwinding can
        // make a best-effort retry in the noexcept destructor.
        cuda_check(cudaFree(data_), what);
        tracker_->remove(bytes_);
        data_ = nullptr;
        bytes_ = 0;
        count_ = 0;
    }

    void grow_preserve(std::size_t count, std::size_t used,
                       const char* allocate_what,
                       const char* copy_what,
                       const char* release_what) {
        if (count <= count_) return;
        if (used > count_)
            throw std::logic_error(
                "GPU round shadow: preserved prefix exceeds buffer capacity");
        const std::size_t new_bytes =
            gpu_round_shadow_checked_mul(count, sizeof(T), allocate_what);
        tracker_->add(new_bytes);
        T* allocated = nullptr;
        const cudaError_t allocation = cudaMalloc(
            reinterpret_cast<void**>(&allocated), new_bytes);
        if (allocation != cudaSuccess) {
            tracker_->remove(new_bytes);
            cuda_failure(allocation, allocate_what);
        }
        if (used) {
            const cudaError_t copy = cudaMemcpy(
                allocated, data_, used * sizeof(T), cudaMemcpyDeviceToDevice);
            if (copy != cudaSuccess) {
                cudaFree(allocated);
                tracker_->remove(new_bytes);
                cuda_failure(copy, copy_what);
            }
        }
        if (data_) {
            const cudaError_t release = cudaFree(data_);
            if (release != cudaSuccess) {
                cudaFree(allocated);
                tracker_->remove(new_bytes);
                cuda_failure(release, release_what);
            }
            tracker_->remove(bytes_);
        }
        data_ = allocated;
        bytes_ = new_bytes;
        count_ = count;
    }

    T* get() noexcept { return data_; }
    const T* get() const noexcept { return data_; }
    std::size_t count() const noexcept { return count_; }
    std::size_t bytes() const noexcept { return bytes_; }

private:
    allocation_tracker* tracker_;
    T* data_ = nullptr;
    std::size_t count_ = 0;
    std::size_t bytes_ = 0;
};

class cuda_event {
public:
    explicit cuda_event(const char* what) {
        cuda_check(cudaEventCreate(&event_), what);
    }
    ~cuda_event() {
        if (event_) cudaEventDestroy(event_);
    }

    cuda_event(const cuda_event&) = delete;
    cuda_event& operator=(const cuda_event&) = delete;

    cudaEvent_t get() const noexcept { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

class event_timer {
public:
    event_timer()
        : begin_("create start event"), end_("create end event") {}

    event_timer(const event_timer&) = delete;
    event_timer& operator=(const event_timer&) = delete;

    template<class F>
    double measure_ms(F&& operation) {
        cuda_check(cudaEventRecord(begin_.get()), "record start event");
        operation();
        cuda_check(cudaEventRecord(end_.get()), "record end event");
        cuda_check(cudaEventSynchronize(end_.get()), "synchronize end event");
        float milliseconds = 0.0f;
        cuda_check(cudaEventElapsedTime(
                       &milliseconds, begin_.get(), end_.get()),
                   "read event duration");
        return milliseconds;
    }

private:
    cuda_event begin_;
    cuda_event end_;
};

struct work_record {
    std::uint32_t a;
    std::uint32_t b;
    double value;
};
static_assert(sizeof(work_record) == 16);

struct device_pivot {
    std::uint32_t vertex;
    std::uint32_t incidence_begin;
    std::uint32_t incidence_end;
    std::uint32_t padding;
    std::uint64_t seed;
};

struct device_digest {
    unsigned long long xor_hash;
    unsigned long long sum_hash;
};
static_assert(sizeof(device_digest) == sizeof(gpu_round_shadow_digest));

struct device_semantics {
    device_digest factor;
    device_digest fill;
    device_digest residual;
    device_digest ordered_residual;
    device_digest active;
    device_digest topology;
    device_digest live_degree;
    device_digest canonical_excess;
    unsigned long long active_count;
};

__device__ void digest_add(device_digest* digest,
                           unsigned long long item) {
    atomicXor(&digest->xor_hash, item);
    atomicAdd(&digest->sum_hash, item);
}

__device__ unsigned long long double_bits(double value) {
    return static_cast<unsigned long long>(__double_as_longlong(value));
}

__device__ unsigned long long float_bits(float value) {
    return static_cast<unsigned long long>(__float_as_uint(value));
}

__device__ unsigned long long stored_weight_bits(double value) {
    const pool_value_t stored = static_cast<pool_value_t>(value);
#if defined(APXCHOL_POOL_FP32)
    return static_cast<unsigned long long>(
        __float_as_uint(static_cast<float>(stored)));
#else
    return static_cast<unsigned long long>(
        __double_as_longlong(static_cast<double>(stored)));
#endif
}

__device__ unsigned long long next_random(unsigned long long& state) {
    unsigned long long z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ double next_unit(unsigned long long& state) {
    return double(next_random(state) >> 11) * 0x1.0p-53;
}

__global__ void initialize_selected(
        const device_pivot* pivots, std::size_t pivot_count,
        std::uint64_t selection_epoch,
        std::uint64_t* selected_epoch_by_vertex,
        std::int32_t* selected_ordinal_by_vertex) {
    const std::size_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal < pivot_count) {
        const node_index vertex = pivots[ordinal].vertex;
        selected_ordinal_by_vertex[vertex] =
            static_cast<std::int32_t>(ordinal);
        selected_epoch_by_vertex[vertex] = selection_epoch;
    }
}

__device__ std::int32_t selected_ordinal(
        node_index vertex, std::uint64_t selection_epoch,
        const std::uint64_t* selected_epoch_by_vertex,
        const std::int32_t* selected_ordinal_by_vertex) {
    return selected_epoch_by_vertex[vertex] == selection_epoch
        ? selected_ordinal_by_vertex[vertex] : -1;
}

enum selection_validation_bits : std::uint32_t {
    selection_out_of_range = 1u << 0,
    selection_inactive = 1u << 1,
    selection_duplicate = 1u << 2,
    selection_adjacent = 1u << 3,
    selection_bad_residual_endpoint = 1u << 4,
    selection_content_mismatch = 1u << 5,
};

struct selection_validation_digest_state {
    unsigned long long xor_hash;
    unsigned long long sum_hash;
    unsigned int completed_blocks;
    unsigned int padding;
};
static_assert(sizeof(selection_validation_digest_state) == 24);

/// Validate every selected id before it is used as an array subscript. The
/// atomic claim both detects duplicates and constructs the selected map used
/// by the independent-set pass and, after a clean status readback, the round.
__global__ void validate_selected_vertices(
        const node_index* selected, std::size_t pivot_count,
        std::size_t vertex_count, const std::uint8_t* active,
        std::uint64_t selection_epoch,
        std::uint64_t* selected_epoch_by_vertex,
        std::int32_t* selected_ordinal_by_vertex,
        std::uint32_t* validation_status,
        selection_validation_digest_state* digest_state,
        unsigned long long expected_xor,
        unsigned long long expected_sum) {
    const std::size_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long item_hash = 0;
    if (ordinal < pivot_count) {
        const node_index raw_vertex = selected[ordinal];
        item_hash = gpu_device_selection_selected_hash(ordinal, raw_vertex);
        if (static_cast<std::uint64_t>(raw_vertex) >= vertex_count) {
            atomicOr(validation_status,
                     static_cast<std::uint32_t>(selection_out_of_range));
        } else {
            const std::size_t vertex = static_cast<std::size_t>(raw_vertex);
            auto* epoch = reinterpret_cast<unsigned long long*>(
                selected_epoch_by_vertex + vertex);
            unsigned long long observed = atomicCAS(epoch, 0ULL, 0ULL);
            bool claimed = false;
            while (observed != selection_epoch) {
                const unsigned long long prior = atomicCAS(
                    epoch, observed, selection_epoch);
                if (prior == observed) {
                    claimed = true;
                    break;
                }
                observed = prior;
            }
            if (claimed) {
                selected_ordinal_by_vertex[vertex] =
                    static_cast<std::int32_t>(ordinal);
            } else {
                atomicOr(validation_status,
                         static_cast<std::uint32_t>(selection_duplicate));
            }
            if (!active[vertex])
                atomicOr(validation_status,
                         static_cast<std::uint32_t>(selection_inactive));
        }
    }

    __shared__ unsigned long long block_xor[kBlock];
    __shared__ unsigned long long block_sum[kBlock];
    block_xor[threadIdx.x] = item_hash;
    block_sum[threadIdx.x] = item_hash;
    __syncthreads();
    for (unsigned stride = kBlock / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_xor[threadIdx.x] ^= block_xor[threadIdx.x + stride];
            block_sum[threadIdx.x] += block_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicXor(&digest_state->xor_hash, block_xor[0]);
        atomicAdd(&digest_state->sum_hash, block_sum[0]);
        // Standard threadFenceReduction completion: the final block observes
        // every preceding block's digest contribution before certifying it.
        __threadfence();
        const unsigned int complete =
            atomicAdd(&digest_state->completed_blocks, 1U) + 1U;
        if (complete == gridDim.x &&
            (digest_state->xor_hash != expected_xor ||
             digest_state->sum_hash != expected_sum))
            atomicOr(validation_status,
                     static_cast<std::uint32_t>(selection_content_mismatch));
    }
}

/// The resident incidence stream is scanned only after the selected map is
/// complete. Endpoint guards make this validation pass itself non-OOB even if
/// a previously published residual were corrupt; a clean status then proves
/// every later gather-side endpoint lookup is in range.
__global__ void validate_selection_independence(
        const gpu_round_shadow_incidence* incidences,
        std::size_t incidence_count, std::size_t vertex_count,
        std::uint64_t selection_epoch,
        const std::uint64_t* selected_epoch_by_vertex,
        std::uint32_t* validation_status) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= incidence_count) return;
    const auto edge = incidences[i];
    if (static_cast<std::uint64_t>(edge.owner) >= vertex_count ||
        static_cast<std::uint64_t>(edge.neighbor) >= vertex_count) {
        atomicOr(validation_status,
                 static_cast<std::uint32_t>(
                     selection_bad_residual_endpoint));
        return;
    }
    if (selected_epoch_by_vertex[edge.owner] == selection_epoch &&
        selected_epoch_by_vertex[edge.neighbor] == selection_epoch)
        atomicOr(validation_status,
                 static_cast<std::uint32_t>(selection_adjacent));
}

__global__ void initialize_pivots_from_device_selection(
        const node_index* selected, std::size_t pivot_count,
        std::uint64_t run_seed, device_pivot* pivots) {
    const std::size_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal >= pivot_count) return;
    const node_index vertex = selected[ordinal];
    const std::uint64_t seed = run_seed ^
        ((std::uint64_t(vertex) + 1) * 0x9E3779B97F4A7C15ULL);
    pivots[ordinal] = {
        static_cast<std::uint32_t>(vertex), 0, 0, 0, seed};
}

__global__ void bind_pivot_ranges(
        device_pivot* pivots, std::size_t pivot_count,
        const std::uint32_t* owner_offsets) {
    const std::size_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal >= pivot_count) return;
    const std::uint32_t vertex = pivots[ordinal].vertex;
    pivots[ordinal].incidence_begin = owner_offsets[vertex];
    pivots[ordinal].incidence_end = owner_offsets[vertex + 1];
}

__global__ void gather_selected_neighbors(
        const gpu_round_shadow_incidence* incidences,
        std::size_t incidence_count,
        const std::uint8_t* active,
        std::uint64_t selection_epoch,
        const std::uint64_t* selected_epoch_by_vertex,
        const std::int32_t* selected_ordinal_by_vertex,
        work_record* candidates,
        std::uint8_t* flags) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= incidence_count) return;
    const auto edge = incidences[i];
    const std::int32_t pivot = selected_ordinal(
        edge.owner, selection_epoch, selected_epoch_by_vertex,
        selected_ordinal_by_vertex);
    const bool keep = pivot >= 0 && active[edge.owner] &&
                      active[edge.neighbor];
    flags[i] = keep ? 1 : 0;
    if (keep)
        candidates[i] = {static_cast<std::uint32_t>(pivot),
                         static_cast<std::uint32_t>(edge.neighbor),
                         edge.weight};
}

__global__ void build_pair_keys(const work_record* records,
                                std::size_t count,
                                std::uint64_t* keys) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        keys[i] = (std::uint64_t(records[i].a) << 32) | records[i].b;
}

__global__ void build_first_keys(const work_record* records,
                                 std::size_t count,
                                 std::uint64_t* keys) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) keys[i] = records[i].a;
}

__global__ void build_weight_keys(const work_record* records,
                                  std::size_t count,
                                  std::uint64_t* keys) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const double value = records[i].value;
        keys[i] = value == 0.0 ? 0 : double_bits(value);
    }
}

__global__ void build_owner_keys(
        const gpu_round_shadow_incidence* incidences,
        std::size_t count, std::uint64_t* keys) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) keys[i] = incidences[i].owner;
}

__global__ void reduce_equal_pairs(const work_record* sorted,
                                   std::size_t count,
                                   work_record* candidates,
                                   std::uint8_t* flags) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const bool first = i == 0 || sorted[i - 1].a != sorted[i].a ||
                       sorted[i - 1].b != sorted[i].b;
    flags[i] = first ? 1 : 0;
    if (!first) return;
    std::size_t end = i + 1;
    double sum = sorted[i].value;
    while (end < count && sorted[end].a == sorted[i].a &&
           sorted[end].b == sorted[i].b) {
        sum += sorted[end].value;
        ++end;
    }
    candidates[i] = {sorted[i].a, sorted[i].b, sum};
}

__global__ void count_unique_neighbors(const work_record* unique,
                                       std::size_t count,
                                       std::uint32_t* pivot_counts) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) atomicAdd(pivot_counts + unique[i].a, 1u);
}

__global__ void set_last_offset(std::uint32_t* offsets,
                                std::size_t pivot_count,
                                std::uint32_t unique_count) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        offsets[pivot_count] = unique_count;
}

__global__ void prepare_factor(
        const work_record* unique_by_neighbor,
        const std::uint32_t* offsets,
        const gpu_round_shadow_incidence* incidences,
        const std::uint8_t* active,
        const device_pivot* pivots,
        std::size_t pivot_count,
        const double* excess,
        double* total_degree,
        gpu_round_shadow_pivot_counter* counters,
        device_semantics* semantics,
        gpu_round_shadow_factor_column* factor_columns,
        std::size_t factor_column_base,
        gpu_round_shadow_factor_entry* factor_entries,
        std::size_t factor_entry_base) {
    const std::size_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal >= pivot_count) return;
    const std::uint32_t begin = offsets[ordinal];
    const std::uint32_t end = offsets[ordinal + 1];
    const std::uint32_t vertex = pivots[ordinal].vertex;
    double degree = 0.0;
    // Match process_vertex(): total degree is the raw live slab sum in slab
    // encounter order, not a second sum over deduplicated neighbors.
    for (std::uint32_t i = pivots[ordinal].incidence_begin;
         i < pivots[ordinal].incidence_end; ++i) {
        const auto edge = incidences[i];
        if (active[edge.neighbor]) degree += edge.weight;
    }
    const double ev = excess[vertex];
    if (begin == end)
        degree = ev > 0.0 ? ev : 1.0;
    else {
        degree += ev;
        if (degree <= 0.0) degree = 1.0;
    }
    total_degree[ordinal] = degree;
    counters[ordinal].pivot = static_cast<node_index>(vertex);
    counters[ordinal].unique_degree = end - begin;
    counters[ordinal].emitted_edges = 0;
    counters[ordinal].total_degree_bits = double_bits(degree);

    const double root = sqrt(degree);
    const float diag = static_cast<float>(root);
    factor_columns[factor_column_base + ordinal] = {
        static_cast<node_index>(vertex),
        static_cast<factor_value_t>(diag),
        static_cast<std::uint64_t>(factor_entry_base + begin),
        static_cast<std::uint32_t>(end - begin)};
    digest_add(&semantics->factor,
        gpu_round_shadow_item_hash(
            gpu_round_shadow_tags::factor_diag, vertex, vertex,
            float_bits(diag)));
    for (std::uint32_t i = begin; i < end; ++i) {
        const float value = static_cast<float>(
            unique_by_neighbor[i].value / root);
        factor_entries[factor_entry_base + i] = {
            static_cast<node_index>(unique_by_neighbor[i].b),
            static_cast<factor_value_t>(value)};
        digest_add(&semantics->factor,
            gpu_round_shadow_item_hash(
                gpu_round_shadow_tags::factor_entry, vertex,
                unique_by_neighbor[i].b, float_bits(value)));
    }
}

__global__ void sample_gks_tree(
        const work_record* canonical,
        const std::uint32_t* offsets,
        const device_pivot* pivots,
        std::size_t pivot_count,
        const double* total_degree,
        double* prefix,
        work_record* fill_candidates,
        std::uint8_t* fill_flags,
        gpu_round_shadow_pivot_counter* counters) {
    const std::size_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal >= pivot_count) return;
    const std::uint32_t begin = offsets[ordinal];
    const std::uint32_t end = offsets[ordinal + 1];
    if (begin == end) return;
    prefix[begin] = canonical[begin].value;
    for (std::uint32_t i = begin + 1; i < end; ++i)
        prefix[i] = prefix[i - 1] + canonical[i].value;
    fill_flags[end - 1] = 0;
    unsigned long long state = pivots[ordinal].seed;
    std::uint64_t emitted = 0;
    for (std::uint32_t i = begin; i + 1 < end; ++i) {
        const double suffix = prefix[end - 1] - prefix[i];
        if (suffix <= 0.0) {
            fill_flags[i] = 0;
            continue;
        }
        const double target = prefix[i] + next_unit(state) * suffix;
        std::uint32_t lo = i + 1;
        std::uint32_t hi = end;
        while (lo < hi) {
            const std::uint32_t mid = lo + (hi - lo) / 2;
            if (prefix[mid] <= target)
                lo = mid + 1;
            else
                hi = mid;
        }
        const std::uint32_t partner = lo < end ? lo : end - 1;
        const std::uint32_t a = canonical[i].b;
        const std::uint32_t b = canonical[partner].b;
        fill_candidates[i] = {
            min(a, b), max(a, b),
            canonical[i].value * suffix / total_degree[ordinal]};
        fill_flags[i] = 1;
        ++emitted;
    }
    counters[ordinal].emitted_edges = emitted;
}

__global__ void make_excess_updates(
        const work_record* canonical,
        std::size_t count,
        const device_pivot* pivots,
        const double* total_degree,
        const double* excess,
        work_record* candidates,
        std::uint8_t* flags) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const std::uint32_t ordinal = canonical[i].a;
    const double ev = excess[pivots[ordinal].vertex];
    const bool keep = ev > 0.0;
    flags[i] = keep ? 1 : 0;
    if (keep)
        candidates[i] = {
            canonical[i].b, 0,
            canonical[i].value * ev / total_degree[ordinal]};
}

__global__ void apply_excess_updates(const work_record* updates,
                                     std::size_t count,
                                     double* excess) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) excess[updates[i].a] += updates[i].value;
}

__global__ void hash_raw_fill(const work_record* fill,
                              std::size_t count,
                              device_semantics* semantics) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        digest_add(&semantics->fill,
            gpu_round_shadow_item_hash(
                gpu_round_shadow_tags::fill, fill[i].a, fill[i].b,
                stored_weight_bits(fill[i].value)));
}

__global__ void deactivate_pivots(const device_pivot* pivots,
                                  std::size_t count,
                                  std::uint8_t* active) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) active[pivots[i].vertex] = 0;
}

__global__ void mark_surviving_incidences(
        const gpu_round_shadow_incidence* input,
        std::size_t count,
        const std::uint8_t* active,
        std::uint8_t* flags) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        flags[i] = active[input[i].owner] && active[input[i].neighbor];
}

__global__ void append_raw_fill(
        const work_record* fill,
        std::size_t fill_count,
        gpu_round_shadow_incidence* residual,
        std::size_t base) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= fill_count) return;
    const auto edge = fill[i];
    const pool_value_t stored = static_cast<pool_value_t>(edge.value);
    const double materialized = static_cast<double>(stored);
    residual[base + 2 * i] = {
        static_cast<node_index>(edge.a),
        static_cast<node_index>(edge.b), materialized};
    residual[base + 2 * i + 1] = {
        static_cast<node_index>(edge.b),
        static_cast<node_index>(edge.a), materialized};
}

__global__ void hash_residual_incidences(
        const gpu_round_shadow_incidence* residual,
        std::size_t count,
        device_semantics* semantics) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto edge = residual[i];
    digest_add(&semantics->residual,
        gpu_round_shadow_item_hash(
            gpu_round_shadow_tags::residual,
            edge.owner, edge.neighbor, stored_weight_bits(edge.weight)));
    if (edge.owner < edge.neighbor)
        digest_add(&semantics->topology,
            gpu_device_selection_topology_hash(edge.owner, edge.neighbor));
}

__global__ void count_residual_degrees(
        const gpu_round_shadow_incidence* residual,
        std::size_t count,
        std::uint32_t* degrees) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) atomicAdd(degrees + residual[i].owner, 1U);
}

__global__ void hash_ordered_residual_incidences(
        const gpu_round_shadow_incidence* residual,
        std::size_t count, const std::uint32_t* owner_offsets,
        device_semantics* semantics) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto edge = residual[i];
    const std::uint64_t ordinal =
        i - static_cast<std::size_t>(owner_offsets[edge.owner]);
    digest_add(&semantics->ordered_residual,
        gpu_round_shadow_ordered_residual_hash(
            edge.owner, ordinal, edge.neighbor,
            stored_weight_bits(edge.weight)));
}

__global__ void hash_residual_state(
        std::size_t vertex_count,
        const std::uint8_t* active,
        const std::uint32_t* degrees,
        const double* excess,
        device_semantics* semantics) {
    const std::size_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= vertex_count || !active[vertex]) return;
    atomicAdd(&semantics->active_count, 1ULL);
    digest_add(&semantics->active,
        gpu_device_selection_active_hash(static_cast<node_index>(vertex)));
    digest_add(&semantics->live_degree,
        gpu_round_shadow_item_hash(
            gpu_round_shadow_tags::live_degree, vertex, 0,
            degrees[vertex]));
    digest_add(&semantics->canonical_excess,
        gpu_round_shadow_item_hash(
            gpu_round_shadow_tags::excess, vertex, 0,
            double_bits(excess[vertex])));
}

template<class T>
T* fake_pointer(std::uintptr_t offset) {
    return reinterpret_cast<T*>(std::uintptr_t{0x100000} + offset);
}

std::size_t query_cub_bytes(std::size_t vertex_count,
                            std::size_t incidence_count,
                            std::size_t pivot_count,
                            const gpu_round_shadow_report& shape) {
    std::size_t maximum = 1;
    std::size_t bytes = 0;
    auto query_select_work = [&](std::size_t count) {
        if (!count) return;
        bytes = 0;
        cuda_check(cub::DeviceSelect::Flagged(
            nullptr, bytes,
            fake_pointer<work_record>(0x100),
            fake_pointer<std::uint8_t>(0x200),
            fake_pointer<work_record>(0x300),
            fake_pointer<int>(0x400), cub_count(count, "selection query")),
            "query work-record selection temporary bytes");
        maximum = std::max(maximum, bytes);
    };
    auto query_select_incidence = [&](std::size_t count) {
        if (!count) return;
        bytes = 0;
        cuda_check(cub::DeviceSelect::Flagged(
            nullptr, bytes,
            fake_pointer<gpu_round_shadow_incidence>(0x500),
            fake_pointer<std::uint8_t>(0x600),
            fake_pointer<gpu_round_shadow_incidence>(0x700),
            fake_pointer<int>(0x800), cub_count(count, "incidence selection query")),
            "query incidence selection temporary bytes");
        maximum = std::max(maximum, bytes);
    };
    auto query_sort = [&](std::size_t count) {
        if (!count) return;
        bytes = 0;
        cuda_check(cub::DeviceRadixSort::SortPairs(
            nullptr, bytes,
            fake_pointer<std::uint64_t>(0x900),
            fake_pointer<std::uint64_t>(0xa00),
            fake_pointer<work_record>(0xb00),
            fake_pointer<work_record>(0xc00),
            cub_count(count, "sort query")),
            "query stable radix-sort temporary bytes");
        maximum = std::max(maximum, bytes);
    };
    auto query_incidence_sort = [&](std::size_t count) {
        if (!count) return;
        bytes = 0;
        cuda_check(cub::DeviceRadixSort::SortPairs(
            nullptr, bytes,
            fake_pointer<std::uint64_t>(0xf00),
            fake_pointer<std::uint64_t>(0x1100),
            fake_pointer<gpu_round_shadow_incidence>(0x1300),
            fake_pointer<gpu_round_shadow_incidence>(0x1500),
            cub_count(count, "residual owner-sort query")),
            "query stable residual owner sort temporary bytes");
        maximum = std::max(maximum, bytes);
    };

    query_select_work(incidence_count);
    query_select_work(shape.gathered_incidences);
    query_select_work(shape.unique_neighbors);
    query_select_work(shape.excess_updates);
    query_sort(shape.gathered_incidences);
    query_sort(shape.unique_neighbors);
    query_sort(shape.excess_updates);
    query_incidence_sort(shape.live_incidences);
    query_select_incidence(incidence_count);
    if (pivot_count) {
        bytes = 0;
        cuda_check(cub::DeviceScan::ExclusiveSum(
            nullptr, bytes,
            fake_pointer<std::uint32_t>(0xd00),
            fake_pointer<std::uint32_t>(0xe00),
            cub_count(pivot_count, "scan query")),
            "query pivot-offset scan temporary bytes");
        maximum = std::max(maximum, bytes);
    }
    if (vertex_count) {
        bytes = 0;
        cuda_check(cub::DeviceScan::ExclusiveSum(
            nullptr, bytes,
            fake_pointer<std::uint32_t>(0x1700),
            fake_pointer<std::uint32_t>(0x1900),
            cub_count(vertex_count, "owner-offset scan query")),
            "query owner-offset scan temporary bytes");
        maximum = std::max(maximum, bytes);
    }
    return maximum;
}

std::size_t add_bytes(std::size_t total, std::size_t count,
                      std::size_t element, const char* what) {
    return gpu_round_shadow_checked_add(
        total, gpu_round_shadow_checked_mul(count, element, what), what);
}

std::size_t planned_peak_bytes(std::size_t n, std::size_t r, std::size_t p,
                               const gpu_round_shadow_report& shape,
                               std::size_t cub_bytes) {
    const std::size_t g = static_cast<std::size_t>(shape.gathered_incidences);
    const std::size_t u = static_cast<std::size_t>(shape.unique_neighbors);
    const std::size_t f = static_cast<std::size_t>(shape.raw_fill_edges);
    const std::size_t e = static_cast<std::size_t>(shape.excess_updates);
    const std::size_t q = static_cast<std::size_t>(shape.excess_targets);
    const std::size_t s =
        static_cast<std::size_t>(shape.surviving_input_incidences);
    const std::size_t l = static_cast<std::size_t>(shape.live_incidences);

    std::size_t common = cub_bytes;
    // A zero-result DeviceSelect still receives a valid one-element output
    // sentinel.  Keep a small explicit allowance for those phase-local
    // sentinels so the preflight remains a strict upper bound.
    common = add_bytes(common, 4096, 1, "zero-count sentinels");
    common = add_bytes(common, r, sizeof(gpu_round_shadow_incidence), "input");
    common = add_bytes(common, n, sizeof(std::uint8_t), "active");
    common = add_bytes(common, n, sizeof(double), "excess");
    common = add_bytes(common, n, sizeof(std::uint64_t), "selected epochs");
    common = add_bytes(common, n, sizeof(std::int32_t), "selected ordinals");
    common = add_bytes(common, p, sizeof(device_pivot), "pivots");
    common = add_bytes(common, p, sizeof(gpu_round_shadow_pivot_counter),
                       "pivot counters");
    common = add_bytes(common, p, sizeof(std::uint32_t), "pivot counts");
    common = add_bytes(common, p + 1, sizeof(std::uint32_t), "pivot offsets");
    common = add_bytes(common, p, sizeof(double), "total degree");
    common = add_bytes(common, 1, sizeof(int), "selected count");
    common = add_bytes(common, 1, sizeof(std::uint32_t),
                       "device-selection validation status");
    common = add_bytes(common, 1, sizeof(selection_validation_digest_state),
                       "device-selection validation digest");
    common = add_bytes(common, 1, sizeof(device_semantics), "semantics");

    auto phase = [&](std::initializer_list<std::pair<std::size_t,
                                                      std::size_t>> arrays) {
        std::size_t bytes = common;
        for (const auto [count, size] : arrays)
            bytes = add_bytes(bytes, count, size, "phase allocation");
        return bytes;
    };

    std::size_t peak = common;
    peak = std::max(peak, phase({
        {r, sizeof(work_record)}, {r, sizeof(std::uint8_t)},
        {g, sizeof(work_record)}}));
    peak = std::max(peak, phase({
        {g, sizeof(work_record)}, {g, sizeof(work_record)},
        {g, sizeof(work_record)},
        {2 * g, sizeof(std::uint64_t)}, {g, sizeof(std::uint8_t)},
        {u, sizeof(work_record)}}));
    peak = std::max(peak, phase({
        {2 * u, sizeof(work_record)}, {2 * u, sizeof(std::uint64_t)},
        {u, sizeof(double)}, {u, sizeof(work_record)},
        {u, sizeof(std::uint8_t)}, {f, sizeof(work_record)},
        {e, sizeof(work_record)}, {q, sizeof(work_record)}}));
    peak = std::max(peak, phase({
        {f, sizeof(work_record)}, {r, sizeof(std::uint8_t)},
        {l, sizeof(gpu_round_shadow_incidence)},
        {n, sizeof(unsigned long long)}}));
    (void)s;
    return peak;
}

template<class T>
void copy_to_device(T* destination, const T* source, std::size_t count,
                    const char* what) {
    if (!count) return;
    cuda_check(cudaMemcpy(destination, source, count * sizeof(T),
                          cudaMemcpyHostToDevice), what);
}

template<class T>
void copy_to_host(T* destination, const T* source, std::size_t count,
                  const char* what) {
    if (!count) return;
    cuda_check(cudaMemcpy(destination, source, count * sizeof(T),
                          cudaMemcpyDeviceToHost), what);
}

class cub_workspace {
public:
    cub_workspace(device_buffer<std::byte>& storage,
                  device_buffer<int>& count)
        : storage_(storage), count_(count) {}

    template<class T>
    int select(const T* input, const std::uint8_t* flags, T* output,
               std::size_t count, const char* what) {
        if (!count) return 0;
        std::size_t bytes = 0;
        cuda_check(cub::DeviceSelect::Flagged(
            nullptr, bytes, input, flags, output, count_.get(),
            cub_count(count, what)), "query CUB ordered selection");
        if (bytes > storage_.count())
            throw std::logic_error(
                "GPU round shadow: preflight underestimated CUB selection");
        cuda_check(cub::DeviceSelect::Flagged(
            storage_.get(), bytes, input, flags, output, count_.get(),
            cub_count(count, what)), what);
        int result = 0;
        copy_to_host(&result, count_.get(), 1, "download selection count");
        return result;
    }

    template<class T>
    void sort_pairs(const std::uint64_t* keys_in, std::uint64_t* keys_out,
                    const T* values_in, T* values_out,
                    std::size_t count, const char* what) {
        if (!count) return;
        std::size_t bytes = 0;
        cuda_check(cub::DeviceRadixSort::SortPairs(
            nullptr, bytes, keys_in, keys_out, values_in, values_out,
            cub_count(count, what)), "query CUB stable radix sort");
        if (bytes > storage_.count())
            throw std::logic_error(
                "GPU round shadow: preflight underestimated CUB sort");
        cuda_check(cub::DeviceRadixSort::SortPairs(
            storage_.get(), bytes, keys_in, keys_out, values_in, values_out,
            cub_count(count, what)), what);
    }

    void exclusive_sum(const std::uint32_t* input, std::uint32_t* output,
                       std::size_t count) {
        if (!count) return;
        std::size_t bytes = 0;
        cuda_check(cub::DeviceScan::ExclusiveSum(
            nullptr, bytes, input, output, cub_count(count, "pivot scan")),
            "query CUB pivot scan");
        if (bytes > storage_.count())
            throw std::logic_error(
                "GPU round shadow: preflight underestimated CUB scan");
        cuda_check(cub::DeviceScan::ExclusiveSum(
            storage_.get(), bytes, input, output,
            cub_count(count, "pivot scan")), "CUB pivot-offset scan");
    }

private:
    device_buffer<std::byte>& storage_;
    device_buffer<int>& count_;
};

void require_count(int actual, std::uint64_t expected, const char* stage) {
    if (actual < 0 || static_cast<std::uint64_t>(actual) != expected)
        throw std::runtime_error(
            std::string("GPU round shadow: ") + stage +
            " count disagrees with the independent reference");
}

void require_count_if_expected(int actual,
                               const gpu_round_shadow_report* expected,
                               std::uint64_t expected_count,
                               const char* stage) {
    if (actual < 0)
        throw std::runtime_error(
            std::string("GPU round shadow: ") + stage +
            " produced a negative CUB count");
    if (expected) require_count(actual, expected_count, stage);
}

void require_valid_device_selection(std::uint32_t status) {
    if (status & selection_out_of_range)
        throw std::invalid_argument(
            "GPU round shadow: device selection contains an out-of-range vertex");
    if (status & selection_inactive)
        throw std::invalid_argument(
            "GPU round shadow: device selection contains an inactive vertex");
    if (status & selection_duplicate)
        throw std::invalid_argument(
            "GPU round shadow: device selection contains a duplicate vertex");
    if (status & selection_adjacent)
        throw std::invalid_argument(
            "GPU round shadow: device selection is not independent");
    if (status & selection_bad_residual_endpoint)
        throw std::runtime_error(
            "GPU round shadow: resident residual contains an out-of-range endpoint");
    if (status & selection_content_mismatch)
        throw std::invalid_argument(
            "GPU round shadow: device selection ids do not match the producer capability");
    if (status)
        throw std::runtime_error(
            "GPU round shadow: device selection returned an unknown validation status");
}

} // namespace

bool gpu_round_shadow_runtime_available() noexcept {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

struct gpu_round_shadow_device_state::impl {
    enum class reimport_reason {
        none,
        parallel_order,
        authoritative_host_rebuild,
    };

    allocation_tracker tracker;
    device_buffer<gpu_round_shadow_incidence> residual_0;
    device_buffer<gpu_round_shadow_incidence> residual_1;
    device_buffer<std::uint32_t> owner_offsets_0;
    device_buffer<std::uint32_t> owner_offsets_1;
    device_buffer<std::uint8_t> active;
    device_buffer<double> excess;
    device_buffer<gpu_round_shadow_factor_column> factor_columns;
    device_buffer<gpu_round_shadow_factor_entry> factor_entries;
    // Device-selection validation state is allocated on first use, retained by
    // the session, and reused at the session's fixed vertex bound. The
    // epoch/ordinal map is also the round gather's input. Epoch tagging removes
    // both an O(n) reset and safely bounded cudaMalloc/cudaFree pairs from every
    // round; the epoch array is zeroed only when first allocated.
    device_buffer<std::uint64_t> selected_epoch_by_vertex;
    device_buffer<std::int32_t> selected_ordinal_by_vertex;
    device_buffer<std::uint32_t> selection_validation_status;
    device_buffer<selection_validation_digest_state>
        selection_validation_digest;

    std::size_t current_slot = 0;
    std::size_t resident_count = 0;
    node_index vertex_count = 0;
    bool has_state = false;
    bool certified = false;
    bool reusable = false;
    bool excess_may_differ = false;
    bool generation_accepted = false;
    bool poisoned = false;
    reimport_reason next_reimport = reimport_reason::none;
    std::uint64_t generation = 0;
    int cuda_device = -1;
    bool selection_producer_bound = false;
    std::weak_ptr<const void> selection_producer;
    std::uint64_t last_selection_generation = 0;
    std::uint64_t last_selection_topology_generation = 0;
    gpu_device_selection_content resident_selection_content;
    gpu_round_shadow_state_fingerprint fingerprint;
    gpu_round_shadow_state_fingerprint certified_cpu_fingerprint;
    std::size_t state_imports = 0;
    std::size_t state_reuses = 0;
    std::size_t order_reimports = 0;
    std::size_t host_rebuild_invalidations = 0;
    std::size_t host_rebuild_reimports = 0;
    std::size_t excess_refreshes = 0;
    std::size_t state_upload_bytes = 0;
    std::size_t state_buffer_allocations = 0;
    std::size_t state_buffer_growths = 0;
    std::size_t factor_column_count = 0;
    std::size_t factor_entry_count = 0;
    std::size_t factor_log_allocations = 0;
    std::size_t factor_log_growths = 0;
#if defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
    bool fail_after_cuda_operation_for_test = false;
#endif

    impl()
        : residual_0(tracker), residual_1(tracker),
          owner_offsets_0(tracker), owner_offsets_1(tracker),
          active(tracker), excess(tracker), factor_columns(tracker),
          factor_entries(tracker), selected_epoch_by_vertex(tracker),
          selected_ordinal_by_vertex(tracker),
          selection_validation_status(tracker),
          selection_validation_digest(tracker) {}

    device_buffer<gpu_round_shadow_incidence>& residual(std::size_t slot) {
        return slot == 0 ? residual_0 : residual_1;
    }
    device_buffer<std::uint32_t>& owner_offsets(std::size_t slot) {
        return slot == 0 ? owner_offsets_0 : owner_offsets_1;
    }

    template<class T>
    bool ensure_state_buffer(device_buffer<T>& buffer, std::size_t count,
                             const char* allocate_what,
                             const char* release_what) {
        if (buffer.count() >= count) return false;
        const bool growth = buffer.count() != 0;
        if (growth) buffer.release(release_what);
        buffer.allocate(count, allocate_what);
        ++state_buffer_allocations;
        if (growth) ++state_buffer_growths;
        return true;
    }

    gpu_round_shadow_state_fingerprint validate_continuity(
            const gpu_round_shadow_input& input) const {
        const auto host = fingerprint_gpu_round_shadow_input(input);
        auto mismatch = [](const char* field) {
            throw std::runtime_error(
                std::string("GPU round shadow resident provenance mismatch: ") +
                field);
        };
        if (host.residual != certified_cpu_fingerprint.residual)
            mismatch("logical residual digest");
        if (host.ordered_residual !=
            certified_cpu_fingerprint.ordered_residual)
            mismatch("ordered residual digest after CPU certification");
        if (host.active != certified_cpu_fingerprint.active)
            mismatch("active digest");
        if (host.topology != certified_cpu_fingerprint.topology)
            mismatch("topology digest");
        if (host.live_degree != certified_cpu_fingerprint.live_degree)
            mismatch("live-degree digest");
        if (host.excess != certified_cpu_fingerprint.excess)
            mismatch("excess digest after CPU certification");
        if (host.active_count != certified_cpu_fingerprint.active_count)
            mismatch("active count");
        if (host.live_incidences !=
            certified_cpu_fingerprint.live_incidences)
            mismatch("live-incidence count");
        return host;
    }

    void poison_failed_generation() noexcept {
        poisoned = true;
        generation_accepted = false;
        certified = false;
        reusable = false;
    }

    void require_current_device(const char* operation) {
        int current_device = -1;
        const cudaError_t status = cudaGetDevice(&current_device);
        if (status != cudaSuccess) {
            poison_failed_generation();
            cuda_failure(status, operation);
        }
        if (cuda_device >= 0 && current_device != cuda_device) {
            poison_failed_generation();
            throw std::invalid_argument(
                std::string("GPU round shadow: ") + operation +
                " uses the wrong CUDA device");
        }
    }

    struct failed_generation_guard {
        impl& state;
        bool armed = false;

        ~failed_generation_guard() {
            if (armed) state.poison_failed_generation();
        }
        void arm() noexcept { armed = true; }
        void commit() noexcept { armed = false; }
    };

    gpu_round_shadow_report compute(
        const gpu_round_shadow_input& input,
        const gpu_round_shadow_report* expected_shape,
        const gpu_device_selection* device_selection,
        std::uint64_t run_seed,
        bool resident_only);
    gpu_round_shadow_factor_log download_factor_log();
    void accept_device_generation(std::uint64_t accepted_generation);
    void reject_device_generation(std::uint64_t rejected_generation) noexcept;
    void certify_cpu_round(
        bool cpu_order_reproducible,
        const gpu_round_shadow_state_fingerprint& cpu_state,
        bool cpu_excess_may_differ);
    void invalidate_for_authoritative_host_rebuild(
        const gpu_round_shadow_state_fingerprint& cpu_state);
};

gpu_round_shadow_report gpu_round_shadow_device_state::impl::compute(
        const gpu_round_shadow_input& input,
        const gpu_round_shadow_report* expected_shape,
        const gpu_device_selection* device_selection,
        std::uint64_t run_seed,
        bool resident_only) {
    if (poisoned)
        throw std::logic_error(
            "GPU round shadow: device state is poisoned by a failed generation");

    // Arm before every remaining precondition, provenance check, host
    // continuity check, allocation, or CUDA call. A failed compute attempt
    // must never leave an older accepted generation consumable.
    failed_generation_guard generation_guard{*this};
    generation_guard.arm();

    int current_device = -1;
    const cudaError_t device_status = cudaGetDevice(&current_device);
    if (device_status != cudaSuccess)
        cuda_failure(device_status, "query active CUDA device");
    if (cuda_device >= 0 && current_device != cuda_device)
        throw std::invalid_argument(
            "GPU round shadow: active CUDA device differs from resident state device");
    // Bind ownership before the first allocation/CUDA operation so even a
    // failed first generation is later destroyed on the allocating device.
    if (cuda_device < 0) cuda_device = current_device;

    if (resident_only && !has_state)
        throw std::logic_error(
            "GPU round shadow: device-resident round has no prior generation");
    if (resident_only && !device_selection)
        throw std::logic_error(
            "GPU round shadow: device-resident round requires a device selection");
    if (resident_only && !generation_accepted)
        throw std::logic_error(
            "GPU round shadow: prior device generation was not explicitly accepted");
    if (resident_only && next_reimport != reimport_reason::none) {
        const char* reason = next_reimport == reimport_reason::parallel_order
            ? "parallel-order"
            : next_reimport == reimport_reason::authoritative_host_rebuild
                ? "authoritative host-rebuild"
                : "unknown";
        throw std::logic_error(
            std::string(
                "GPU round shadow: device-resident round cannot bypass pending ") +
            reason + " reimport");
    }
    if (resident_only && excess_may_differ)
        throw std::logic_error(
            "GPU round shadow: device-resident round cannot bypass a bounded "
            "excess refresh requiring a host snapshot");

    gpu_device_selection::snapshot selected;
    if (device_selection) {
        selected = device_selection->inspect();
        if (selection_producer_bound) {
            const auto bound = selection_producer.lock();
            if (!bound || bound.get() != selected.producer_identity.get())
                throw std::invalid_argument(
                    "GPU round shadow: device selection came from a different producer");
            if (selected.generation <= last_selection_generation)
                throw std::invalid_argument(
                    "GPU round shadow: device selection generation was already consumed");
            if (selected.topology_generation <=
                last_selection_topology_generation)
                throw std::invalid_argument(
                    "GPU round shadow: producer topology generation was already consumed");
        }
    }

    if (device_selection && selected.cuda_device != current_device)
        throw std::invalid_argument(
            "GPU round shadow: selection and round state use different CUDA devices");

    const std::size_t n = resident_only
        ? static_cast<std::size_t>(vertex_count)
        : static_cast<std::size_t>(input.vertex_count);
    const std::size_t r = resident_only
        ? resident_count
        : input.incidences.size();
    const std::size_t p = device_selection
        ? selected.size
        : input.pivots.size();
    if (device_selection) {
        if (!resident_only && selected.size != input.pivots.size())
            throw std::invalid_argument(
                "GPU round shadow: device/host selection count mismatch");
        if (!resident_only && input.seeds.size() != input.pivots.size())
            throw std::invalid_argument(
                "GPU round shadow: host audit pivot/seed count mismatch");
        if (p && !selected.data)
            throw std::invalid_argument(
                "GPU round shadow: nonempty device selection has a null pointer");
        if (!resident_only) {
            for (std::size_t i = 0; i < p; ++i) {
                if (input.seeds[i] !=
                    gpu_round_shadow_pivot_seed(run_seed, input.pivots[i]))
                    throw std::invalid_argument(
                        "GPU round shadow: host audit seed does not match the "
                        "device-selection run seed");
            }
        }
        const gpu_device_selection_content expected_content = resident_only
            ? resident_selection_content
            : gpu_round_shadow_selection_content(input);
        if (!gpu_device_selection_state_matches(
                selected.content, expected_content))
            throw std::invalid_argument(
                "GPU round shadow: selection topology/active content does not match round state");
        if (!selected.independence_certified)
            throw std::invalid_argument(
                "GPU round shadow: selection independence is not producer-certified");
        if (!p && selected.content.selection !=
                gpu_device_selection_digest{})
            throw std::invalid_argument(
                "GPU round shadow: empty selection has a nonempty producer digest");
    }
    if (generation == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error(
            "GPU round shadow: resident generation overflow");
    if (factor_column_count > n || p > n - factor_column_count)
        throw std::overflow_error(
            "GPU round shadow: factor log would contain more than n columns");

    bool refresh_excess = false;
    if (has_state && !resident_only) {
        if (!certified)
            throw std::logic_error(
                "GPU round shadow: resident generation was not CPU-certified");
        if (input.vertex_count != vertex_count)
            throw std::runtime_error(
                "GPU round shadow resident provenance mismatch: vertex count");
        // Structural continuity is exact even when a parallel CPU apply made
        // its slab encounter order unreproducible.  The latter case is then a
        // visible re-import, never an implicit fallback.
        const auto host = validate_continuity(input);
        if (reusable &&
            host.ordered_residual != fingerprint.ordered_residual)
            throw std::runtime_error(
                "GPU round shadow resident provenance mismatch: "
                "ordered residual digest");
        if (reusable && host.excess != fingerprint.excess) {
            if (!excess_may_differ)
                throw std::runtime_error(
                    "GPU round shadow resident provenance mismatch: "
                    "exact excess digest");
            refresh_excess = true;
        }
    }
    const bool reuse_input = resident_only || (has_state && reusable);
    const bool import_input = !reuse_input;
    const std::size_t input_count = reuse_input ? resident_count : r;
    const std::size_t next_slot = 1 - current_slot;
    // Host-snapshot FORCE audits retain the complete O(r) adjacency scan.
    // Resident production consumes the authentic producer certificate and
    // performs only the mandatory O(p) bounds/active/duplicate pass unless
    // the explicit audit knob requests the redundant topology scan.
    const bool audit_selection_independence = device_selection &&
        (!resident_only || gpu_round_selection_audit_requested());

    cuda_check(cudaFree(nullptr), "initialize CUDA context");
#if defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
    if (std::exchange(fail_after_cuda_operation_for_test, false))
        throw std::runtime_error(
            "GPU round shadow: injected failure after CUDA operation");
#endif

    // Audit mode receives exact stream sizes from the independent CPU
    // reference.  R2b discovery mode deliberately does not: each stage uses
    // only a proven upper bound until CUB returns its actual scalar count.
    // For one independent pivot set:
    //   gathered <= input, unique/fill/excess <= gathered,
    //   survivors <= input, residual = survivors + 2*fill <= 3*input.
    gpu_round_shadow_report capacity_shape;
    if (expected_shape) {
        capacity_shape = *expected_shape;
    } else {
        const std::size_t upper = std::max(r, input_count);
        const std::size_t residual_upper = gpu_round_shadow_checked_add(
            upper, gpu_round_shadow_checked_mul(upper, 2,
                                                "discovered directed fill"),
            "discovered residual upper bound");
        capacity_shape.gathered_incidences = upper;
        capacity_shape.unique_neighbors = upper;
        capacity_shape.raw_fill_edges = upper;
        capacity_shape.excess_updates = upper;
        capacity_shape.excess_targets = upper;
        capacity_shape.surviving_input_incidences = upper;
        capacity_shape.live_incidences = residual_upper;
    }
    gpu_round_shadow_validate_capacity_counts(
        static_cast<node_index>(n), r, p, capacity_shape);

    const std::size_t cub_bytes = query_cub_bytes(n, r, p, capacity_shape);
    std::size_t planned =
        planned_peak_bytes(n, r, p, capacity_shape, cub_bytes);
    // R1b's plan already covers one input and one phase-local residual.  R2a
    // keeps a second residual slot resident, adds two owner-offset tables, and
    // stably owner-sorts the materialized output before publishing it.
    const std::size_t current_target = std::max(
        residual(current_slot).bytes(),
        gpu_round_shadow_checked_mul(
            std::max<std::size_t>(input_count, 1),
            sizeof(gpu_round_shadow_incidence), "resident input target"));
    const std::size_t next_target = std::max(
        residual(next_slot).bytes(),
        gpu_round_shadow_checked_mul(
            std::max<std::size_t>(
                static_cast<std::size_t>(capacity_shape.live_incidences), 1),
            sizeof(gpu_round_shadow_incidence), "resident output target"));
    const std::size_t planned_input_bytes =
        gpu_round_shadow_checked_mul(r, sizeof(gpu_round_shadow_incidence),
                                     "planned input bytes");
    if (current_target > planned_input_bytes)
        planned = gpu_round_shadow_checked_add(
            planned, current_target - planned_input_bytes,
            "resident input capacity");
    planned = gpu_round_shadow_checked_add(
        planned, next_target, "resident spare output");
    planned = add_bytes(planned, 2 * (n + 1), sizeof(std::uint32_t),
                        "resident owner offsets");
    planned = add_bytes(
        planned,
        2 * static_cast<std::size_t>(capacity_shape.live_incidences),
        sizeof(std::uint64_t),
                        "residual owner-sort keys");
    const std::size_t factor_column_required = factor_column_count + p;
    const std::size_t factor_entry_required_upper =
        gpu_round_shadow_checked_add(
            factor_entry_count,
            static_cast<std::size_t>(capacity_shape.unique_neighbors),
            "factor append upper bound");
    const std::size_t factor_column_target = factor_columns.count() <
            factor_column_required
        ? std::max<std::size_t>(n, factor_column_required)
        : factor_columns.count();
    const std::size_t factor_entry_target = std::max(
        factor_entries.count(), geometric_capacity(factor_entry_required_upper));
    // Growth temporarily holds both old and replacement allocations. Account
    // for that overlap so FORCE fails before mutating the resident state.
    planned = add_bytes(planned, factor_columns.count(),
                        sizeof(gpu_round_shadow_factor_column),
                        "existing factor columns");
    if (factor_column_target > factor_columns.count())
        planned = add_bytes(planned, factor_column_target,
                            sizeof(gpu_round_shadow_factor_column),
                            "grown factor columns");
    planned = add_bytes(planned, factor_entries.count(),
                        sizeof(gpu_round_shadow_factor_entry),
                        "existing factor entries");
    if (factor_entry_target > factor_entries.count())
        planned = add_bytes(planned, factor_entry_target,
                            sizeof(gpu_round_shadow_factor_entry),
                            "grown factor entries");

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes),
               "query free device memory");
    const std::size_t margin = total_bytes / 10;
    const std::size_t available_with_resident =
        gpu_round_shadow_checked_add(tracker.current, free_bytes,
                                     "resident plus free device memory");
    if (available_with_resident <= margin ||
        planned > available_with_resident - margin)
        throw std::runtime_error(
            "GPU round shadow: forced round does not fit in free device "
            "memory with the 10% safety margin (required=" +
            std::to_string(planned) + ", resident=" +
            std::to_string(tracker.current) + ", free=" +
            std::to_string(free_bytes) + ", margin=" +
            std::to_string(margin) + ")");

    tracker.begin_round();
    if (factor_columns.count() < factor_column_required) {
        const bool growth = factor_columns.count() != 0;
        factor_columns.grow_preserve(
            std::max<std::size_t>(n, factor_column_required),
            factor_column_count, "allocate factor-column log",
            "preserve factor-column log", "replace factor-column log");
        ++factor_log_allocations;
        if (growth) ++factor_log_growths;
    }
    ensure_state_buffer(
        residual(current_slot), std::max<std::size_t>(input_count, 1),
        "allocate resident residual input",
        "replace resident residual input");
    ensure_state_buffer(active, n, "allocate resident active mask",
                        "grow resident active mask");
    ensure_state_buffer(excess, n, "allocate resident excess state",
                        "grow resident excess state");
    ensure_state_buffer(owner_offsets(current_slot), n + 1,
                        "allocate resident owner offsets",
                        "grow resident owner offsets");
    ensure_state_buffer(owner_offsets(next_slot), n + 1,
                        "allocate next resident owner offsets",
                        "grow next resident owner offsets");
    const bool selected_epochs_allocated = ensure_state_buffer(
        selected_epoch_by_vertex, n,
        "allocate persistent selected epochs",
        "grow persistent selected epochs");
    ensure_state_buffer(
        selected_ordinal_by_vertex, n,
        "allocate persistent selected ordinals",
        "grow persistent selected ordinals");
    if (selected_epochs_allocated && n)
        cuda_check(cudaMemset(selected_epoch_by_vertex.get(), 0,
                              n * sizeof(std::uint64_t)),
                   "initialize persistent selected epochs");
    if (device_selection) {
        ensure_state_buffer(
            selection_validation_status, 1,
            "allocate persistent device-selection validation status",
            "grow persistent device-selection validation status");
        ensure_state_buffer(
            selection_validation_digest, 1,
            "allocate persistent device-selection validation digest",
            "grow persistent device-selection validation digest");
    }

    auto& device_input = residual(current_slot);
    auto& input_owner_offsets = owner_offsets(current_slot);
    auto& next_owner_offsets = owner_offsets(next_slot);

    device_buffer<std::byte> cub_temp(tracker);
    device_buffer<int> selected_count(tracker);
    device_buffer<device_pivot> pivots(tracker);
    device_buffer<gpu_round_shadow_pivot_counter> pivot_counters(tracker);
    device_buffer<std::uint32_t> pivot_counts(tracker);
    device_buffer<std::uint32_t> pivot_offsets(tracker);
    device_buffer<double> total_degree(tracker);
    device_buffer<device_semantics> semantics(tracker);

    cub_temp.allocate(cub_bytes, "allocate CUB temporary storage");
    selected_count.allocate(1, "allocate CUB selected count");
    pivots.allocate(p, "allocate pivots");
    pivot_counters.allocate(p, "allocate pivot counters");
    pivot_counts.allocate(p, "allocate pivot counts");
    pivot_offsets.allocate(p + 1, "allocate pivot offsets");
    total_degree.allocate(p, "allocate total degrees");
    semantics.allocate(1, "allocate semantic report");

    if (tracker.peak > planned)
        throw std::logic_error(
            "GPU round shadow: common allocations exceeded the preflight plan");

    std::vector<device_pivot> host_pivots;
    if (!device_selection) {
        host_pivots.resize(p);
        for (std::size_t i = 0; i < p; ++i) {
            const node_index pivot = input.pivots[i];
            host_pivots[i] = {
                static_cast<std::uint32_t>(pivot),
                0, 0, 0, input.seeds[i]};
        }
    }
    std::vector<std::uint32_t> host_owner_offsets;
    if (import_input) {
        host_owner_offsets.resize(n + 1);
        for (std::size_t i = 0; i <= n; ++i) {
            if (input.owner_offsets[i] >
                std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error(
                    "GPU round shadow: owner offset exceeds uint32 capacity");
            host_owner_offsets[i] =
                static_cast<std::uint32_t>(input.owner_offsets[i]);
        }
    }

    cub_workspace cub(cub_temp, selected_count);
    event_timer timer;
    cuda_event total_begin("create total start event");
    cuda_event total_end("create total end event");

    gpu_round_shadow_report report;
    report.input_incidences = r;
    report.resident_input_incidences = input_count;
    report.pivots.resize(p);
    report.input_generation = generation;
    report.output_generation = generation + 1;
    report.resident_input_reused = reuse_input;
    report.resident_selection_consumed = resident_only && device_selection;
    report.selection_validation_pivots = device_selection ? p : 0;
    report.selection_audit_passes =
        audit_selection_independence && input_count ? 1 : 0;
    report.selection_audit_incidences = audit_selection_independence
        ? input_count : 0;
    report.selection_map_initialization_vertices =
        selected_epochs_allocated ? n : 0;
    if (reuse_input) {
        ++state_reuses;
        if (refresh_excess) ++excess_refreshes;
    } else {
        ++state_imports;
        if (has_state) {
            if (next_reimport == reimport_reason::authoritative_host_rebuild)
                ++host_rebuild_reimports;
            else if (next_reimport == reimport_reason::parallel_order)
                ++order_reimports;
            else
                throw std::logic_error(
                    "GPU round shadow: resident import has no recorded reason");
        }
    }
    if (import_input) {
        report.round_state_upload_bytes =
            gpu_round_shadow_checked_add(
                gpu_round_shadow_checked_mul(
                    r, sizeof(gpu_round_shadow_incidence),
                    "resident residual upload"),
                gpu_round_shadow_checked_add(
                    gpu_round_shadow_checked_mul(
                        n, sizeof(std::uint8_t), "resident active upload"),
                    gpu_round_shadow_checked_add(
                        gpu_round_shadow_checked_mul(
                            n, sizeof(double), "resident excess upload"),
                        gpu_round_shadow_checked_mul(
                            n + 1, sizeof(std::uint32_t),
                            "resident owner-offset upload"),
                        "resident scalar-state upload"),
                    "resident vertex-state upload"),
                "resident state upload");
    } else if (refresh_excess) {
        report.round_state_upload_bytes = gpu_round_shadow_checked_mul(
            n, sizeof(double), "resident excess refresh");
    }
    if (report.round_state_upload_bytes) {
        state_upload_bytes = gpu_round_shadow_checked_add(
            state_upload_bytes, report.round_state_upload_bytes,
            "cumulative resident upload");
    }
    cuda_check(cudaEventRecord(total_begin.get()), "record total start event");

    report.timings.upload_ms = timer.measure_ms([&] {
        if (import_input) {
            copy_to_device(device_input.get(), input.incidences.data(), r,
                           "upload directed residual state");
            copy_to_device(active.get(), input.active.data(), n,
                           "upload active state");
            copy_to_device(excess.get(), input.excess.data(), n,
                           "upload excess state");
            copy_to_device(input_owner_offsets.get(),
                           host_owner_offsets.data(), n + 1,
                           "upload owner-offset state");
        } else if (refresh_excess) {
            copy_to_device(excess.get(), input.excess.data(), n,
                           "refresh bounded CPU excess state");
        }
        const std::uint64_t selection_epoch = report.output_generation;
        if (device_selection) {
            cuda_check(cudaMemset(selection_validation_status.get(), 0,
                                  sizeof(std::uint32_t)),
                       "clear device-selection validation status");
            cuda_check(cudaMemset(selection_validation_digest.get(), 0,
                                  sizeof(selection_validation_digest_state)),
                       "clear device-selection validation digest");
            if (p) {
                validate_selected_vertices<<<blocks_for(p), kBlock>>>(
                    selected.data, p, n, active.get(), selection_epoch,
                    selected_epoch_by_vertex.get(),
                    selected_ordinal_by_vertex.get(),
                    selection_validation_status.get(),
                    selection_validation_digest.get(),
                    selected.content.selection.xor_hash,
                    selected.content.selection.sum_hash);
                cuda_check(cudaGetLastError(),
                           "validate device-selection vertices");
            }
            if (audit_selection_independence && input_count) {
                validate_selection_independence<<<
                    blocks_for(input_count), kBlock>>>(
                        device_input.get(), input_count, n,
                        selection_epoch,
                        selected_epoch_by_vertex.get(),
                        selection_validation_status.get());
                cuda_check(cudaGetLastError(),
                           "validate device-selection independence");
            }
            // The only selection-validation readback is this fixed 32-bit
            // status word; selected ids and resident graph state stay on CUDA.
            std::uint32_t host_validation_status = 0;
            copy_to_host(&host_validation_status,
                         selection_validation_status.get(), 1,
                         "download device-selection validation status");
            require_valid_device_selection(host_validation_status);
            if (p) {
                initialize_pivots_from_device_selection<<<blocks_for(p), kBlock>>>(
                    selected.data, p, run_seed, pivots.get());
                cuda_check(cudaGetLastError(),
                           "initialize pivots from validated device selection");
            }
        } else {
            copy_to_device(pivots.get(), host_pivots.data(), p,
                           "upload selected pivots and seeds");
        }
        if (p) {
            cuda_check(cudaMemset(pivot_counts.get(), 0,
                                  p * sizeof(std::uint32_t)),
                       "clear pivot counts");
            cuda_check(cudaMemset(pivot_counters.get(), 0,
                                  p * sizeof(gpu_round_shadow_pivot_counter)),
                       "clear pivot counters");
            bind_pivot_ranges<<<blocks_for(p), kBlock>>>(
                pivots.get(), p, input_owner_offsets.get());
            cuda_check(cudaGetLastError(), "bind resident pivot ranges");
            if (!device_selection) {
                initialize_selected<<<blocks_for(p), kBlock>>>(
                    pivots.get(), p, selection_epoch,
                    selected_epoch_by_vertex.get(),
                    selected_ordinal_by_vertex.get());
                cuda_check(cudaGetLastError(),
                           "launch selected-map initializer");
            }
        }
        cuda_check(cudaMemset(semantics.get(), 0, sizeof(device_semantics)),
                   "clear semantic report");
    });

    std::size_t g = 0;
    std::size_t u = 0;
    std::size_t f = 0;
    std::size_t e = 0;
    std::size_t q = 0;
    std::size_t s = 0;
    std::size_t l = 0;
    const std::size_t g_capacity = expected_shape
        ? static_cast<std::size_t>(expected_shape->gathered_incidences)
        : input_count;

    device_buffer<work_record> first(tracker);
    device_buffer<work_record> second(tracker);
    device_buffer<std::uint8_t> flags(tracker);
    first.allocate(input_count, "allocate gather candidates");
    flags.allocate(input_count, "allocate gather flags");
    second.allocate(std::max<std::size_t>(g_capacity, 1),
                    "allocate gathered records");
    report.timings.gather_ms = timer.measure_ms([&] {
        if (input_count) {
            gather_selected_neighbors<<<blocks_for(input_count), kBlock>>>(
                device_input.get(), input_count, active.get(),
                report.output_generation,
                selected_epoch_by_vertex.get(),
                selected_ordinal_by_vertex.get(),
                first.get(), flags.get());
            cuda_check(cudaGetLastError(), "launch selected-neighbor gather");
        }
        const int gathered = cub.select(first.get(), flags.get(), second.get(),
                                        input_count,
                                        "select gathered incidences");
        require_count_if_expected(
            gathered, expected_shape,
            expected_shape ? expected_shape->gathered_incidences : 0,
            "gather");
        g = static_cast<std::size_t>(gathered);
        report.gathered_incidences = g;
    });
    first.release("release gather candidates");
    flags.release("release gather flags");

    device_buffer<std::uint64_t> keys_a(tracker);
    device_buffer<std::uint64_t> keys_b(tracker);
    keys_a.allocate(g, "allocate gather keys A");
    keys_b.allocate(g, "allocate gather keys B");
    first.allocate(g, "allocate sorted gathered records");
    device_buffer<work_record> unique(tracker);
    device_buffer<work_record> scratch(tracker);
    flags.allocate(g, "allocate dedup flags");
    scratch.allocate(g, "allocate dedup candidates");
    const std::size_t u_capacity = expected_shape
        ? static_cast<std::size_t>(expected_shape->unique_neighbors)
        : g;
    unique.allocate(std::max<std::size_t>(u_capacity, 1),
                    "allocate unique neighbors");

    report.timings.dedup_ms = timer.measure_ms([&] {
        if (g) {
            build_pair_keys<<<blocks_for(g), kBlock>>>(second.get(), g,
                                                       keys_a.get());
            cuda_check(cudaGetLastError(), "launch gather-key construction");
            // CUB's LSD radix sort is stable, so equal (pivot,neighbor) keys
            // retain the directed-AoS encounter order used by the fp64 sum.
            cub.sort_pairs(keys_a.get(), keys_b.get(), second.get(), first.get(),
                           g, "stable sort gathered neighbors");
            reduce_equal_pairs<<<blocks_for(g), kBlock>>>(
                first.get(), g, scratch.get(), flags.get());
            cuda_check(cudaGetLastError(), "launch deterministic neighbor dedup");
        }
        const int unique_count = cub.select(
            scratch.get(), flags.get(), unique.get(), g,
            "select unique neighbors");
        require_count_if_expected(
            unique_count, expected_shape,
            expected_shape ? expected_shape->unique_neighbors : 0,
            "dedup");
        u = static_cast<std::size_t>(unique_count);
        report.unique_neighbors = u;
        report.factor_entries = u;
        if (u) {
            count_unique_neighbors<<<blocks_for(u), kBlock>>>(
                unique.get(), u, pivot_counts.get());
            cuda_check(cudaGetLastError(), "launch per-pivot unique counts");
        }
        cub.exclusive_sum(pivot_counts.get(), pivot_offsets.get(), p);
        set_last_offset<<<1, 1>>>(pivot_offsets.get(), p,
                                  static_cast<std::uint32_t>(u));
        cuda_check(cudaGetLastError(), "publish final pivot offset");
    });
    second.release("release gathered records");
    first.release("release sorted gathered records");
    scratch.release("release dedup candidates");
    flags.release("release dedup flags");
    keys_a.release("release gather keys A");
    keys_b.release("release gather keys B");

    const std::size_t factor_column_base = factor_column_count;
    const std::size_t factor_entry_base = factor_entry_count;
    const std::size_t factor_entry_required = gpu_round_shadow_checked_add(
        factor_entry_base, u, "factor append entries");
    if (factor_entries.count() < factor_entry_required) {
        const bool growth = factor_entries.count() != 0;
        factor_entries.grow_preserve(
            geometric_capacity(factor_entry_required), factor_entry_count,
            "allocate factor-entry log", "preserve factor-entry log",
            "replace factor-entry log");
        ++factor_log_allocations;
        if (growth) ++factor_log_growths;
    }
    if (tracker.peak > planned)
        throw std::logic_error(
            "GPU round shadow: factor-log growth exceeded the preflight plan");

    report.timings.factor_ms = timer.measure_ms([&] {
        if (p) {
            prepare_factor<<<blocks_for(p), kBlock>>>(
                unique.get(), pivot_offsets.get(), device_input.get(),
                active.get(), pivots.get(), p,
                excess.get(), total_degree.get(), pivot_counters.get(),
                semantics.get(), factor_columns.get(), factor_column_base,
                factor_entries.get(), factor_entry_base);
            cuda_check(cudaGetLastError(),
                       "launch factor-column append and digest");
        }
    });

    scratch.allocate(u, "allocate canonical-neighbor scratch");
    keys_a.allocate(u, "allocate canonical keys A");
    keys_b.allocate(u, "allocate canonical keys B");
    device_buffer<double> prefix(tracker);
    device_buffer<work_record> fill_candidates(tracker);
    device_buffer<work_record> fills(tracker);
    prefix.allocate(u, "allocate GKS prefix sums");
    fill_candidates.allocate(u, "allocate raw fill candidates");
    flags.allocate(u, "allocate sample flags");
    const std::size_t f_capacity = expected_shape
        ? static_cast<std::size_t>(expected_shape->raw_fill_edges)
        : u;
    fills.allocate(std::max<std::size_t>(f_capacity, 1),
                   "allocate compact raw fill");

    report.timings.sample_ms = timer.measure_ms([&] {
        if (u) {
            // unique[] is (pivot,neighbor).  Stable sort by weight first, then
            // by pivot, yielding (pivot,weight,neighbor) without a 128-bit key.
            build_weight_keys<<<blocks_for(u), kBlock>>>(
                unique.get(), u, keys_a.get());
            cuda_check(cudaGetLastError(), "launch weight-key construction");
            cub.sort_pairs(keys_a.get(), keys_b.get(), unique.get(),
                           scratch.get(), u,
                           "stable sort unique neighbors by weight");
            build_first_keys<<<blocks_for(u), kBlock>>>(
                scratch.get(), u, keys_a.get());
            cuda_check(cudaGetLastError(), "launch pivot-key construction");
            cub.sort_pairs(keys_a.get(), keys_b.get(), scratch.get(),
                           unique.get(), u,
                           "stable regroup canonical neighbors by pivot");
            cuda_check(cudaMemset(flags.get(), 0, u),
                       "clear fill candidate flags");
        }
        if (p) {
            sample_gks_tree<<<blocks_for(p), kBlock>>>(
                unique.get(), pivot_offsets.get(), pivots.get(), p,
                total_degree.get(), prefix.get(), fill_candidates.get(),
                flags.get(), pivot_counters.get());
            cuda_check(cudaGetLastError(), "launch all-degree GKS tree sampler");
        }
        const int raw_fill = cub.select(
            fill_candidates.get(), flags.get(), fills.get(), u,
            "compact raw GKS fill");
        require_count_if_expected(
            raw_fill, expected_shape,
            expected_shape ? expected_shape->raw_fill_edges : 0,
            "sampling");
        f = static_cast<std::size_t>(raw_fill);
        report.raw_fill_edges = f;

        // Excess propagation is another deterministic keyed reduction.  It
        // uses canonical pivot/weight/neighbor order and never uses fp64
        // atomics, so colliding updates are repeatable.
        if (u) {
            make_excess_updates<<<blocks_for(u), kBlock>>>(
                unique.get(), u, pivots.get(), total_degree.get(), excess.get(),
                scratch.get(), flags.get());
            cuda_check(cudaGetLastError(), "launch excess-update construction");
        }
        device_buffer<work_record> compact_excess(tracker);
        const std::size_t e_capacity = expected_shape
            ? static_cast<std::size_t>(expected_shape->excess_updates)
            : u;
        compact_excess.allocate(std::max<std::size_t>(e_capacity, 1),
                                "allocate compact excess updates");
        const int excess_count = cub.select(
            scratch.get(), flags.get(), compact_excess.get(), u,
            "compact excess updates");
        require_count_if_expected(
            excess_count, expected_shape,
            expected_shape ? expected_shape->excess_updates : 0,
            "excess update construction");
        e = static_cast<std::size_t>(excess_count);
        report.excess_updates = e;
        if (e) {
            build_first_keys<<<blocks_for(e), kBlock>>>(
                compact_excess.get(), e, keys_a.get());
            cuda_check(cudaGetLastError(), "launch excess target keys");
            cub.sort_pairs(keys_a.get(), keys_b.get(), compact_excess.get(),
                           unique.get(), e, "stable sort excess updates");
            reduce_equal_pairs<<<blocks_for(e), kBlock>>>(
                unique.get(), e, scratch.get(), flags.get());
            cuda_check(cudaGetLastError(), "launch deterministic excess reduce");
        }
        device_buffer<work_record> reduced_excess(tracker);
        const std::size_t q_capacity = expected_shape
            ? static_cast<std::size_t>(expected_shape->excess_targets)
            : e;
        reduced_excess.allocate(std::max<std::size_t>(q_capacity, 1),
                                "allocate reduced excess updates");
        const int reduced_count = cub.select(
            scratch.get(), flags.get(), reduced_excess.get(), e,
            "compact reduced excess updates");
        require_count_if_expected(
            reduced_count, expected_shape,
            expected_shape ? expected_shape->excess_targets : 0,
            "excess reduction");
        q = static_cast<std::size_t>(reduced_count);
        report.excess_targets = q;
        if (q) {
            apply_excess_updates<<<blocks_for(q), kBlock>>>(
                reduced_excess.get(), q, excess.get());
            cuda_check(cudaGetLastError(), "launch reduced excess apply");
        }
    });
    prefix.release("release GKS prefix sums");
    fill_candidates.release("release raw fill candidates");
    unique.release("release canonical neighbors");
    scratch.release("release canonical scratch");
    flags.release("release canonical flags");
    keys_a.release("release canonical keys A");
    keys_b.release("release canonical keys B");

    report.timings.fill_materialize_ms = timer.measure_ms([&] {
        if (f) {
            hash_raw_fill<<<blocks_for(f), kBlock>>>(
                fills.get(), f, semantics.get());
            cuda_check(cudaGetLastError(), "launch raw multigraph-fill digest");
        }
    });

    device_buffer<gpu_round_shadow_incidence> residual(tracker);
    device_buffer<std::uint32_t> degrees(tracker);
    flags.allocate(input_count, "allocate surviving-incidence flags");
    const std::size_t residual_capacity = expected_shape
        ? static_cast<std::size_t>(expected_shape->live_incidences)
        : gpu_round_shadow_checked_add(
              input_count,
              gpu_round_shadow_checked_mul(f, 2, "discovered fill output"),
              "discovered residual output");
    residual.allocate(std::max<std::size_t>(residual_capacity, 1),
                      "allocate materialized shadow residual");
    degrees.allocate(n, "allocate live degrees");

    report.timings.mutate_ms = timer.measure_ms([&] {
        if (p) {
            deactivate_pivots<<<blocks_for(p), kBlock>>>(
                pivots.get(), p, active.get());
            cuda_check(cudaGetLastError(), "launch pivot removal");
        }
        if (input_count) {
            mark_surviving_incidences<<<blocks_for(input_count), kBlock>>>(
                device_input.get(), input_count, active.get(), flags.get());
            cuda_check(cudaGetLastError(), "launch residual survivor marking");
        }
        const int survivors = cub.select(
            device_input.get(), flags.get(), residual.get(), input_count,
            "compact surviving residual incidences");
        require_count_if_expected(
            survivors, expected_shape,
            expected_shape ? expected_shape->surviving_input_incidences : 0,
            "pivot removal");
        s = static_cast<std::size_t>(survivors);
        report.surviving_input_incidences = s;
        if (f) {
            append_raw_fill<<<blocks_for(f), kBlock>>>(
                fills.get(), f, residual.get(), s);
            cuda_check(cudaGetLastError(), "launch raw multigraph-fill application");
        }
    });
    flags.release("release surviving-incidence flags");
    fills.release("release raw multigraph fill");

    l = gpu_round_shadow_checked_add(
        s, gpu_round_shadow_checked_mul(f, 2, "actual directed fill"),
        "actual residual incidence");
    report.live_incidences = l;
    if (expected_shape && l != expected_shape->live_incidences)
        throw std::runtime_error(
            "GPU round shadow: residual count disagrees with the independent "
            "reference");
    // Re-run the scalar capacity check on the counts discovered by CUDA.  In
    // discovery mode this proves that the conservative preflight bound did
    // not hide a malformed count relation.
    gpu_round_shadow_validate_capacity_counts(
        static_cast<node_index>(n), r, p, report);

    ensure_state_buffer(
        this->residual(next_slot), std::max<std::size_t>(l, 1),
        "allocate resident residual output",
        "grow resident residual output");
    auto& next_residual = this->residual(next_slot);

    keys_a.allocate(l, "allocate residual owner keys A");
    keys_b.allocate(l, "allocate residual owner keys B");

    report.timings.checksums_ms = timer.measure_ms([&] {
        if (n)
            cuda_check(cudaMemset(degrees.get(), 0,
                                  n * sizeof(std::uint32_t)),
                       "clear live degrees");
        if (l) {
            count_residual_degrees<<<blocks_for(l), kBlock>>>(
                residual.get(), l, degrees.get());
            cuda_check(cudaGetLastError(), "launch residual degree count");
            build_owner_keys<<<blocks_for(l), kBlock>>>(
                residual.get(), l, keys_a.get());
            cuda_check(cudaGetLastError(), "launch residual owner keys");
            // The sort is stable: survivors precede this round's fills, and
            // fill records retain pivot/emission order within each owner.  A
            // serial CPU apply creates exactly that slab suffix order.
            cub.sort_pairs(keys_a.get(), keys_b.get(), residual.get(),
                           next_residual.get(), l,
                           "stable sort resident residual by owner");
            cub.exclusive_sum(degrees.get(), next_owner_offsets.get(), n);
            set_last_offset<<<1, 1>>>(
                next_owner_offsets.get(), n, static_cast<std::uint32_t>(l));
            cuda_check(cudaGetLastError(), "publish final owner offset");
            hash_residual_incidences<<<blocks_for(l), kBlock>>>(
                next_residual.get(), l, semantics.get());
            cuda_check(cudaGetLastError(),
                       "launch residual endpoint/weight digest");
            hash_ordered_residual_incidences<<<blocks_for(l), kBlock>>>(
                next_residual.get(), l, next_owner_offsets.get(),
                semantics.get());
            cuda_check(cudaGetLastError(),
                       "launch ordered residual digest");
        } else if (n) {
            cub.exclusive_sum(degrees.get(), next_owner_offsets.get(), n);
            set_last_offset<<<1, 1>>>(next_owner_offsets.get(), n, 0);
            cuda_check(cudaGetLastError(), "publish empty owner offsets");
        }
        if (n) {
            hash_residual_state<<<blocks_for(n), kBlock>>>(
                n, active.get(), degrees.get(), excess.get(), semantics.get());
            cuda_check(cudaGetLastError(), "launch residual checksums");
        }
    });

    device_semantics host_semantics{};
    report.timings.download_ms = timer.measure_ms([&] {
        copy_to_host(&host_semantics, semantics.get(), 1,
                     "download semantic report");
        copy_to_host(report.pivots.data(), pivot_counters.get(), p,
                     "download pivot counters");
    });

    cuda_check(cudaEventRecord(total_end.get()), "record total end event");
    cuda_check(cudaEventSynchronize(total_end.get()),
               "synchronize total end event");
    float total_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(
                   &total_ms, total_begin.get(), total_end.get()),
               "read total event duration");
    report.timings.total_ms = total_ms;

    report.factor = {host_semantics.factor.xor_hash,
                     host_semantics.factor.sum_hash};
    report.fill = {host_semantics.fill.xor_hash,
                   host_semantics.fill.sum_hash};
    report.residual = {host_semantics.residual.xor_hash,
                       host_semantics.residual.sum_hash};
    report.ordered_residual = {
        host_semantics.ordered_residual.xor_hash,
        host_semantics.ordered_residual.sum_hash};
    report.active = {host_semantics.active.xor_hash,
                     host_semantics.active.sum_hash};
    report.live_degree = {host_semantics.live_degree.xor_hash,
                          host_semantics.live_degree.sum_hash};
    report.canonical_excess = {
        host_semantics.canonical_excess.xor_hash,
        host_semantics.canonical_excess.sum_hash};
    report.active_count = host_semantics.active_count;
    report.live_incidences = l;
    report.state_imports = state_imports;
    report.state_reuses = state_reuses;
    report.order_reimports = order_reimports;
    report.host_rebuild_invalidations = host_rebuild_invalidations;
    report.host_rebuild_reimports = host_rebuild_reimports;
    report.excess_refreshes = excess_refreshes;
    report.state_upload_bytes = state_upload_bytes;
    report.state_buffer_allocations = state_buffer_allocations;
    report.state_buffer_growths = state_buffer_growths;
    report.factor_log_columns = factor_column_count + p;
    report.factor_log_entries = factor_entry_count + u;
    report.factor_log_allocations = factor_log_allocations;
    report.factor_log_growths = factor_log_growths;
    report.peak_device_bytes = tracker.peak;
    report.gpu_executed = true;
    if (report.peak_device_bytes > planned)
        throw std::logic_error(
            "GPU round shadow: actual peak exceeded the preflight plan");

    factor_column_count += p;
    factor_entry_count += u;
    current_slot = next_slot;
    resident_count = l;
    vertex_count = static_cast<node_index>(n);
    has_state = true;
    certified = false;
    reusable = false;
    excess_may_differ = false;
    generation_accepted = false;
    next_reimport = reimport_reason::none;
    generation = report.output_generation;
    cuda_device = current_device;
    if (device_selection) {
        selection_producer = selected.producer_identity;
        selection_producer_bound = true;
        last_selection_generation = selected.generation;
        last_selection_topology_generation =
            selected.topology_generation;
    }
    resident_selection_content = {
        static_cast<node_index>(n),
        static_cast<std::size_t>(host_semantics.active_count),
        {host_semantics.topology.xor_hash,
         host_semantics.topology.sum_hash},
        {host_semantics.active.xor_hash,
         host_semantics.active.sum_hash},
        {}};
    fingerprint.residual = report.residual;
    fingerprint.ordered_residual = report.ordered_residual;
    fingerprint.active = report.active;
    fingerprint.topology = {
        host_semantics.topology.xor_hash,
        host_semantics.topology.sum_hash};
    fingerprint.live_degree = report.live_degree;
    fingerprint.excess = report.canonical_excess;
    fingerprint.active_count = report.active_count;
    fingerprint.live_incidences = report.live_incidences;
    generation_guard.commit();
    return report;
}

gpu_round_shadow_factor_log
gpu_round_shadow_device_state::impl::download_factor_log() {
    if (poisoned)
        throw std::logic_error(
            "GPU round shadow: device state is poisoned by a failed generation");
    failed_generation_guard generation_guard{*this};
    generation_guard.arm();
    require_current_device("factor-log download");
    gpu_round_shadow_factor_log result;
    result.columns.resize(factor_column_count);
    result.entries.resize(factor_entry_count);
    copy_to_host(result.columns.data(), factor_columns.get(),
                 factor_column_count, "download factor-column log");
    copy_to_host(result.entries.data(), factor_entries.get(),
                 factor_entry_count, "download factor-entry log");
    generation_guard.commit();
    return result;
}

void gpu_round_shadow_device_state::impl::accept_device_generation(
        std::uint64_t accepted_generation) {
    if (poisoned)
        throw std::logic_error(
            "GPU round shadow: cannot accept a poisoned device generation");
    require_current_device("device-generation acceptance");
    if (!has_state || accepted_generation != generation)
        throw std::logic_error(
            "GPU round shadow: device-generation acceptance is out of order");
    generation_accepted = true;
}

void gpu_round_shadow_device_state::impl::reject_device_generation(
        std::uint64_t rejected_generation) noexcept {
    (void)rejected_generation;
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess ||
        (cuda_device >= 0 && current_device != cuda_device)) {
        (void)cudaGetLastError();
    }
    // Rejection is the caller's explicit statement that the current audit did
    // not pass. Never preserve an older acceptance on an id mismatch.
    poison_failed_generation();
}

void gpu_round_shadow_device_state::impl::certify_cpu_round(
        bool cpu_order_reproducible,
        const gpu_round_shadow_state_fingerprint& cpu_state,
        bool cpu_excess_may_differ) {
    if (poisoned)
        throw std::logic_error(
            "GPU round shadow: cannot certify a poisoned device generation");
    require_current_device("CPU generation certification");
    if (!has_state || certified)
        throw std::logic_error(
            "GPU round shadow: resident generation certification is out of order");
    auto mismatch = [this](const char* field) {
        poison_failed_generation();
        throw std::runtime_error(
            std::string("GPU round shadow CPU certification mismatch: ") +
            field);
    };
    if (cpu_state.residual != fingerprint.residual)
        mismatch("logical residual digest");
    if (cpu_state.active != fingerprint.active)
        mismatch("active digest");
    if (cpu_state.topology != fingerprint.topology)
        mismatch("topology digest");
    if (cpu_state.live_degree != fingerprint.live_degree)
        mismatch("live-degree digest");
    if (cpu_state.active_count != fingerprint.active_count)
        mismatch("active count");
    if (cpu_state.live_incidences != fingerprint.live_incidences)
        mismatch("live-incidence count");
    if (cpu_order_reproducible &&
        cpu_state.ordered_residual != fingerprint.ordered_residual)
        mismatch("serial ordered residual digest");
    if (!cpu_excess_may_differ && cpu_state.excess != fingerprint.excess)
        mismatch("exact excess digest");
    certified_cpu_fingerprint = cpu_state;
    certified = true;
    generation_accepted = true;
    reusable = cpu_order_reproducible;
    excess_may_differ = cpu_excess_may_differ;
    next_reimport = cpu_order_reproducible
        ? reimport_reason::none
        : reimport_reason::parallel_order;
}

void gpu_round_shadow_device_state::impl::
invalidate_for_authoritative_host_rebuild(
        const gpu_round_shadow_state_fingerprint& cpu_state) {
    if (poisoned)
        throw std::logic_error(
            "GPU round shadow: cannot invalidate a poisoned device generation");
    require_current_device("authoritative host-rebuild invalidation");
    if (!has_state) return;
    if (!certified)
        throw std::logic_error(
            "GPU round shadow: host rebuild invalidation requires a certified "
            "resident generation");
    certified_cpu_fingerprint = cpu_state;
    reusable = false;
    excess_may_differ = false;
    next_reimport = reimport_reason::authoritative_host_rebuild;
    ++host_rebuild_invalidations;
}

gpu_round_shadow_device_state::gpu_round_shadow_device_state()
    : impl_(std::make_unique<impl>()) {}

void gpu_round_shadow_device_state::reset() noexcept {
    if (!impl_) return;
    int original_device = -1;
    bool switched_device = false;
    if (impl_->cuda_device >= 0 &&
        cudaGetDevice(&original_device) == cudaSuccess &&
        original_device != impl_->cuda_device &&
        cudaSetDevice(impl_->cuda_device) == cudaSuccess)
        switched_device = true;
    impl_.reset();
    if (switched_device) (void)cudaSetDevice(original_device);
}

gpu_round_shadow_device_state::~gpu_round_shadow_device_state() { reset(); }

gpu_round_shadow_device_state::gpu_round_shadow_device_state(
    gpu_round_shadow_device_state&&) noexcept = default;

gpu_round_shadow_device_state& gpu_round_shadow_device_state::operator=(
        gpu_round_shadow_device_state&& other) noexcept {
    if (this != &other) {
        reset();
        impl_ = std::move(other.impl_);
    }
    return *this;
}

#if defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
void gpu_round_shadow_device_state::
inject_failure_after_cuda_operation_for_test() {
    if (impl_->poisoned)
        throw std::logic_error(
            "GPU round shadow: cannot inject a fault into poisoned state");
    impl_->require_current_device("test fault injection setup");
    impl_->fail_after_cuda_operation_for_test = true;
}
#endif

gpu_round_shadow_report gpu_round_shadow_device_state::compute(
        const gpu_round_shadow_input& input,
        const gpu_round_shadow_report& expected_shape) {
    return impl_->compute(input, &expected_shape, nullptr, 0, false);
}

gpu_round_shadow_report gpu_round_shadow_device_state::compute_discover_shape(
        const gpu_round_shadow_input& input) {
    return impl_->compute(input, nullptr, nullptr, 0, false);
}

gpu_round_shadow_report gpu_round_shadow_device_state::compute_discover_shape(
        const gpu_round_shadow_input& input,
        gpu_device_selection selected,
        std::uint64_t run_seed) {
    return impl_->compute(input, nullptr, &selected, run_seed, false);
}

gpu_round_shadow_report gpu_round_shadow_device_state::compute_resident(
        gpu_device_selection selected,
        std::uint64_t run_seed) {
    const gpu_round_shadow_input no_host_state;
    return impl_->compute(
        no_host_state, nullptr, &selected, run_seed, true);
}

void gpu_round_shadow_device_state::accept_device_generation(
        std::uint64_t generation) {
    impl_->accept_device_generation(generation);
}

void gpu_round_shadow_device_state::reject_device_generation(
        std::uint64_t generation) noexcept {
    impl_->reject_device_generation(generation);
}

std::shared_ptr<cuda_sptrsv_device_factor>
gpu_round_shadow_device_state::finalize_fp32(
        std::span<const node_index> permutation, node_index factor_dim,
        const gpu_round_shadow_factor_log& tail) {
    impl_->require_current_device("factor finalization");
    if (impl_->poisoned || !impl_->has_state || !impl_->generation_accepted)
        throw std::logic_error("GPU finalizer requires an accepted resident generation");
    if (cuda_sptrsv::fp16_resolved() || factor_drop_rel_from_env() != 0.0)
        throw std::invalid_argument("GPU finalizer prototype requires APXCHOL_SPTRSV_FP16=0 and APXCHOL_FACTOR_DROP=0");
    const std::size_t n = permutation.size(), m = factor_dim;
    const std::size_t prefix = impl_->factor_column_count;
    if (m == 0 || m > n || n >= std::size_t(INT_MAX) ||
        prefix == 0 || prefix > n || tail.columns.size() != n - prefix ||
        impl_->factor_entry_count > std::size_t(INT_MAX) - n ||
        tail.entries.size() > std::size_t(INT_MAX) - n - impl_->factor_entry_count)
        throw std::invalid_argument("GPU finalizer dimension, tail coverage or int32 capacity is invalid");
    std::vector<bool> seen(n, false);
    for (node_index p : permutation) {
        if (p >= n || seen[p]) throw std::invalid_argument("GPU finalizer requires a permutation");
        seen[p] = true;
    }
    allocation_tracker tracker;
    device_buffer<node_index> perm(tracker);
    device_buffer<gpu_round_shadow_factor_column> tc(tracker);
    device_buffer<gpu_round_shadow_factor_entry> te(tracker);
    device_buffer<int> status(tracker), lp(tracker), ltp(tracker);
    device_buffer<int> li(tracker), lti(tracker);
    device_buffer<float> lv(tracker), ltv(tracker), input_values(tracker);
    device_buffer<std::uint64_t> keys(tracker), sorted_keys(tracker);
    device_buffer<unsigned char> temporary(tracker);
    perm.allocate(n, "finalizer permutation");
    tc.allocate(tail.columns.size(), "finalizer tail columns");
    te.allocate(tail.entries.size(), "finalizer tail entries");
    copy_to_device(perm.get(), permutation.data(), n, "finalizer permutation upload");
    copy_to_device(tc.get(), tail.columns.data(), tail.columns.size(), "finalizer tail columns upload");
    copy_to_device(te.get(), tail.entries.data(), tail.entries.size(), "finalizer tail entries upload");
    status.allocate(2, "finalizer status");
    lp.allocate(m + 1, "finalizer L pointers");
    ltp.allocate(m + 1, "finalizer LT pointers");
    cuda_check(cudaMemset(status.get(), 0, 2 * sizeof(int)), "finalizer status clear");
    cuda_check(cudaMemset(ltp.get(), 0, sizeof(int)), "finalizer LT pointer origin");
    cuda_check(cudaMemset(lp.get(), 0, (m + 1) * sizeof(int)), "finalizer L pointer counts");
    finalizer_counts<<<blocks_for(m), kBlock>>>(n, m, prefix,
        impl_->factor_columns.get(), impl_->factor_entries.get(), impl_->factor_entry_count,
        tc.get(), te.get(), tail.entries.size(), perm.get(), ltp.get(), status.get());
    cuda_check(cudaGetLastError(), "finalizer count launch");
    int host_status[2] = {};
    copy_to_host(host_status, status.get(), 2, "finalizer status");
    if (host_status[0]) throw std::invalid_argument("GPU finalizer malformed factor log, code " + std::to_string(host_status[0]));
    std::size_t scan_bytes = 0;
    cuda_check(cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, ltp.get(), ltp.get(), int(m + 1)), "finalizer scan size");
    temporary.allocate(scan_bytes, "finalizer scan scratch");
    cuda_check(cub::DeviceScan::InclusiveSum(temporary.get(), scan_bytes, ltp.get(), ltp.get(), int(m + 1)), "finalizer count scan");
    int nnz = 0;
    copy_to_host(&nnz, ltp.get() + m, 1, "finalizer nnz");
    if (nnz < int(m) || nnz > host_status[1]) throw std::logic_error("GPU finalizer invalid output count");
    keys.allocate(nnz, "finalizer COO keys");
    sorted_keys.allocate(nnz, "finalizer sorted keys");
    input_values.allocate(nnz, "finalizer COO values");
    lv.allocate(nnz, "finalizer L values"); ltv.allocate(nnz, "finalizer LT values");
    li.allocate(nnz, "finalizer L indices"); lti.allocate(nnz, "finalizer LT indices");
    finalizer_emit<<<blocks_for(m), kBlock>>>(m, prefix,
        impl_->factor_columns.get(), impl_->factor_entries.get(), tc.get(), te.get(),
        perm.get(), ltp.get(), keys.get(), input_values.get());
    cuda_check(cudaGetLastError(), "finalizer emit launch");
    std::size_t sort_bytes = 0;
    cuda_check(cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, keys.get(), sorted_keys.get(), input_values.get(), ltv.get(), nnz), "finalizer sort size");
    if (sort_bytes > scan_bytes) {
        temporary.release("finalizer replace scratch");
        temporary.allocate(sort_bytes, "finalizer sort scratch");
    }
    cuda_check(cub::DeviceRadixSort::SortPairs(temporary.get(), sort_bytes, keys.get(), sorted_keys.get(), input_values.get(), ltv.get(), nnz), "finalizer sort LT");
    finalizer_unpack<<<blocks_for(nnz), kBlock>>>(nnz, sorted_keys.get(), lti.get(), status.get(), keys.get(), lp.get());
    cuda_check(cudaGetLastError(), "finalizer LT unpack");
    cuda_check(cub::DeviceScan::InclusiveSum(temporary.get(), scan_bytes, lp.get(), lp.get(), int(m + 1)), "finalizer L pointer scan");
    cuda_check(cub::DeviceRadixSort::SortPairs(temporary.get(), sort_bytes, keys.get(), sorted_keys.get(), ltv.get(), lv.get(), nnz), "finalizer sort L");
    finalizer_unpack<<<blocks_for(nnz), kBlock>>>(nnz, sorted_keys.get(), li.get(), status.get(), nullptr, nullptr);
    cuda_check(cudaGetLastError(), "finalizer L unpack");
    copy_to_host(host_status, status.get(), 1, "finalizer duplicate status");
    if (host_status[0]) throw std::invalid_argument("GPU finalizer duplicate coordinates");
    factor_drop_stats stats;
    stats.rel = 0; stats.nnz_factor = host_status[1]; stats.nnz_stored = nnz;
    stats.dropped = stats.dropped_flush = stats.nnz_factor - stats.nnz_stored;
    auto result = std::make_shared<cuda_sptrsv_device_factor>();
    *result = cuda_sptrsv_device_factor::own_fp32(impl_->cuda_device, m, nnz,
        {{lp.get(), (m + 1) * sizeof(int)}, {li.get(), std::size_t(nnz) * sizeof(int)}, {lv.get(), std::size_t(nnz) * sizeof(float)}},
        {{ltp.get(), (m + 1) * sizeof(int)}, {lti.get(), std::size_t(nnz) * sizeof(int)}, {ltv.get(), std::size_t(nnz) * sizeof(float)}}, stats);
    result->finalized_sorted_triangular_ = true;
    lp.detach(); li.detach(); lv.detach(); ltp.detach(); lti.detach(); ltv.detach();
    const std::size_t upload = n * sizeof(node_index) +
        tail.columns.size() * sizeof(gpu_round_shadow_factor_column) +
        tail.entries.size() * sizeof(gpu_round_shadow_factor_entry);
    std::fprintf(stderr, "[gpu-factor-finalize] device_prefix_columns=%zu device_prefix_entries=%zu cpu_tail_columns=%zu cpu_tail_entries=%zu m=%zu nnz=%d host_upload_bytes=%zu scalar_download_bytes=%zu prefix_download_bytes=0 scratch_and_output_peak_bytes=%zu cpu_shadow=retained\n",
        prefix, impl_->factor_entry_count, tail.columns.size(), tail.entries.size(),
        m, nnz, upload, 4 * sizeof(int), tracker.peak);
    return result;
}

gpu_round_shadow_factor_log
gpu_round_shadow_device_state::download_factor_log() const {
    return impl_->download_factor_log();
}

void gpu_round_shadow_device_state::certify_cpu_round(
        bool cpu_order_reproducible,
        const gpu_round_shadow_state_fingerprint& cpu_state,
        bool cpu_excess_may_differ) {
    impl_->certify_cpu_round(
        cpu_order_reproducible, cpu_state, cpu_excess_may_differ);
}

void gpu_round_shadow_device_state::invalidate_for_authoritative_host_rebuild(
        const gpu_round_shadow_state_fingerprint& cpu_state) {
    impl_->invalidate_for_authoritative_host_rebuild(cpu_state);
}

gpu_round_shadow_report compute_gpu_round_shadow(
        const gpu_round_shadow_input& input,
        const gpu_round_shadow_report& expected_shape) {
    gpu_round_shadow_device_state state;
    return state.compute(input, expected_shape);
}

} // namespace apxchol::detail
