#include "apxchol/solver/elimination/gpu_round_shadow.h"
#include "apxchol/solver/sptrsv/cuda.h"
#include "apxchol/solver/gpu_block_frontend.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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

// Residual keys contain only a validated owner < fixed vertex_count. Keep one
// bit for empty/single-vertex domains so no zero-bit copy convention is needed.
constexpr int owner_sort_end_bit(std::size_t vertex_count) {
    return vertex_count > 1
        ? static_cast<int>(std::bit_width(vertex_count - 1)) : 1;
}
static_assert(owner_sort_end_bit(0) == 1 && owner_sort_end_bit(1) == 1);
static_assert(owner_sort_end_bit(2) == 1 && owner_sort_end_bit(3) == 2);
static_assert(owner_sort_end_bit(32) == 5 && owner_sort_end_bit(33) == 6);
static_assert(owner_sort_end_bit(64) == 6 && owner_sort_end_bit(65) == 7);
static_assert(owner_sort_end_bit(INT_MAX) == 31);
static_assert(owner_sort_end_bit(std::numeric_limits<std::size_t>::max()) ==
              std::numeric_limits<std::size_t>::digits);
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


__device__ std::uint16_t finalizer_half(float value) {
    const auto bits = __half_as_ushort(__float2half_rn(value));
    return (bits & 0x7c00u) == 0 ? std::uint16_t(bits & 0x8000u) : bits;
}

__device__ bool finalizer_keep(float value, float scale, double rel, bool fp16) {
    if (!(fabs(double(value)) >= rel * double(scale))) return false;
    return fp16 ? (finalizer_half(__fdiv_rn(value, scale)) & 0x7fffu) != 0
                : value != 0.0f;
}

// Each CUDA thread owns one already sorted factor column. All sums and
// compensation use the same entry order as the host compacting-drop contract.
__global__ void finalizer_column_storage(
        int m, const int* ptr, const std::uint64_t* keys, const float* values,
        double rel, bool fp16, float* scales, int* kept_counts,
        unsigned long long* drop_counts, int* status) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;
    const int begin = ptr[j], end = ptr[j + 1];
    float scale = 0.0f;
    for (int p = begin + 1; p < end; ++p) {
        if (keys[p] == keys[p - 1]) atomicExch(status, 4);
        scale = fmaxf(scale, fabsf(values[p]));
    }
    if (scale == 0.0f) scale = 1.0f;
    if (fp16 && (!isfinite(__fdiv_rn(1.0f, scale)) ||
                 !isfinite(__fdiv_rn(values[begin], scale)))) scale = 1.0f;
    scales[j] = scale;
    int kept = 1;
    unsigned long long threshold = 0, flush = 0;
    for (int p = begin + 1; p < end; ++p) {
        const float v = values[p];
        if (!(rel > 0.0) || finalizer_keep(v, scale, rel, fp16)) ++kept;
        else if (fabs(double(v)) < rel * double(scale)) ++threshold;
        else ++flush;
    }
    kept_counts[j + 1] = kept;
    if (threshold) atomicAdd(drop_counts, threshold);
    if (flush) atomicAdd(drop_counts + 1, flush);
}

__global__ void finalizer_compact_columns(
        int m, const int* old_ptr, const std::uint64_t* old_keys,
        const float* old_values, const float* scales, double rel, bool fp16,
        const int* new_ptr, std::uint64_t* keys, float* values) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;
    const float scale = scales[j];
    int out = new_ptr[j];
    double dropped_sum = 0.0, kept_abs = 0.0;
    for (int p = old_ptr[j]; p < old_ptr[j + 1]; ++p) {
        const float v = old_values[p];
        if (p != old_ptr[j]) {
            if (!finalizer_keep(v, scale, rel, fp16)) {
                dropped_sum += double(v);
                continue;
            }
            kept_abs += fabs(double(v));
        }
        keys[out] = old_keys[p]; values[out] = v; ++out;
    }
    if (dropped_sum != 0.0 && kept_abs > 0.0) {
        const double per_abs = dropped_sum / kept_abs;
        for (int p = new_ptr[j] + 1; p < out; ++p) {
            const double v = double(values[p]);
            values[p] = float(v + per_abs * fabs(v));
        }
    }
}

__global__ void finalizer_narrow_columns(
        int m, const int* ptr, const float* values, const float* scales,
        std::uint16_t* half_values, float* diag, double* inv_scale2,
        unsigned long long* flushed, int* status) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;
    const float scale = scales[j];
    const float inv = __fdiv_rn(1.0f, scale);
    inv_scale2[j] = double(inv) * double(inv);
    float d = __fdiv_rn(values[ptr[j]], scale);
    double residual = 0.0;
    unsigned long long count = 0;
    for (int p = ptr[j]; p < ptr[j + 1]; ++p) {
        const float v = values[p];
        const auto h = finalizer_half(__fdiv_rn(v, scale));
        half_values[p] = h;
        if (p == ptr[j]) continue;
        const float widened = __half2float(__ushort_as_half(h));
        if (v != 0.0f && widened == 0.0f) ++count;
        residual += double(v) / double(scale) - double(widened);
    }
    d = float(double(d) + residual);
    diag[j] = d;
    if (!isfinite(d) || d == 0.0f || !isfinite(inv_scale2[j])) atomicExch(status, 5);
    if (count) atomicAdd(flushed, count);
}


[[noreturn]] void cuda_failure(cudaError_t error, const char* what) {
    throw std::runtime_error(std::string("GPU round shadow: ") + what +
                             ": " + cudaGetErrorString(error));
}

void cuda_check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) cuda_failure(error, what);
}

int blocks_for(std::size_t count, unsigned block_size = kBlock) {
    if (count == 0) return 0;
    const std::size_t blocks = (count + block_size - 1) / block_size;
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
    explicit cuda_event(const char* what, bool enabled) {
        if (enabled)
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
        : enabled_(gpu_setup_diagnostics()),
          begin_("create start event", enabled_), end_("create end event", enabled_) {}

    event_timer(const event_timer&) = delete;
    event_timer& operator=(const event_timer&) = delete;

    template<class F>
    double measure_ms(F&& operation) {
        if (!enabled_) {
            // Every operation stays on the same stream. Host consumers retain
            // their synchronous D2H/status copies; no event publishes data.
            operation();
            return std::numeric_limits<double>::quiet_NaN();
        } else {
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
    }

private:
    const bool enabled_;
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

// XOR and unsigned sum are associative modulo 2^64. Reduce exact audit
// digests inside the block before contending on the one global report word.
// Every thread participates, including inactive lanes in the final block.
__device__ device_digest reduce_block_digest(
        device_digest value, device_digest* warp_values) {
    const unsigned lane = threadIdx.x % 32;
    const unsigned warp = threadIdx.x / 32;
    for (unsigned offset = 16; offset; offset >>= 1) {
        value.xor_hash ^= __shfl_down_sync(0xffffffffU, value.xor_hash, offset);
        value.sum_hash += __shfl_down_sync(0xffffffffU, value.sum_hash, offset);
    }
    if (!lane) warp_values[warp] = value;
    __syncthreads();
    if (!warp) {
        value = lane < kBlock / 32 ? warp_values[lane] : device_digest{};
        for (unsigned offset = 16; offset; offset >>= 1) {
            value.xor_hash ^= __shfl_down_sync(0xffffffffU, value.xor_hash, offset);
            value.sum_hash += __shfl_down_sync(0xffffffffU, value.sum_hash, offset);
        }
    }
    // Callers may immediately reuse the same shared array for another digest.
    __syncthreads();
    return value;  // only thread zero contains the block's complete result
}

__device__ void digest_add_block(device_digest value, device_digest* target,
                                 device_digest* warp_values) {
    value = reduce_block_digest(value, warp_values);
    if (!threadIdx.x) {
        atomicXor(&target->xor_hash, value.xor_hash);
        atomicAdd(&target->sum_hash, value.sum_hash);
    }
}

__device__ void digest_add_local(device_digest& digest, unsigned long long item) {
    digest.xor_hash ^= item;
    digest.sum_hash += item;
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

#include "cuda_cycle_sampler.cuh"

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

template<bool Audit, bool OversizedOnly = false>
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
    device_digest digest{};
    std::uint32_t begin = 0, end = 0;
    double root = 0.0;
    if (ordinal < pivot_count && (!OversizedOnly ||
        pivots[ordinal].incidence_end - pivots[ordinal].incidence_begin > 128u)) {
        begin = offsets[ordinal];
        end = offsets[ordinal + 1];
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
        if constexpr (Audit) {
            counters[ordinal].pivot = static_cast<node_index>(vertex);
            counters[ordinal].unique_degree = end - begin;
            counters[ordinal].emitted_edges = 0;
            counters[ordinal].total_degree_bits = double_bits(degree);
        }

        root = sqrt(degree);
        const float diag = static_cast<float>(root);
        factor_columns[factor_column_base + ordinal] = {
            static_cast<node_index>(vertex),
            static_cast<factor_value_t>(diag),
            static_cast<std::uint64_t>(factor_entry_base + begin),
            static_cast<std::uint32_t>(end - begin)};
        if constexpr (Audit) digest_add_local(digest,
            gpu_round_shadow_item_hash(
                gpu_round_shadow_tags::factor_diag, vertex, vertex,
                float_bits(diag)));
        // Preserve one-thread density on short owned columns and every audited
        // column. Only independent writes for long owned columns cooperate.
        if (Audit || end - begin <= 32u)
        for (std::uint32_t i = begin; i < end; ++i) {
            const float value = static_cast<float>(
                unique_by_neighbor[i].value / root);
            factor_entries[factor_entry_base + i] = {
                static_cast<node_index>(unique_by_neighbor[i].b),
                static_cast<factor_value_t>(value)};
            if constexpr (Audit) digest_add_local(digest,
                gpu_round_shadow_item_hash(
                    gpu_round_shadow_tags::factor_entry, vertex,
                    unique_by_neighbor[i].b, float_bits(value)));
        }
    }
    if constexpr (!Audit) {
        // All32 original pivot lanes participate, including out-of-range lanes.
        // Raw degree, root and column headers above retain their original owner.
        unsigned pending = __ballot_sync(0xffffffffu,
            ordinal < pivot_count && end - begin > 32u);
        const unsigned lane = threadIdx.x & 31u;
        while (pending) {
            const int owner = __ffs(pending) - 1;
            const auto entry_begin = __shfl_sync(0xffffffffu, begin, owner);
            const auto entry_end = __shfl_sync(0xffffffffu, end, owner);
            // Broadcast the original double root, never the rounded diagonal.
            const double entry_root = __shfl_sync(0xffffffffu, root, owner);
            for (std::size_t i = std::size_t(entry_begin) + lane;
                 i < entry_end; i += 32u) {
                const float value = static_cast<float>(
                    unique_by_neighbor[i].value / entry_root);
                factor_entries[factor_entry_base + i] = {
                    static_cast<node_index>(unique_by_neighbor[i].b),
                    static_cast<factor_value_t>(value)};
            }
            pending &= pending - 1;
        }
    }
    if constexpr (Audit) {
        __shared__ device_digest warp_values[kBlock / 32];
        digest_add_block(digest, &semantics->factor, warp_values);
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
    if (counters) counters[ordinal].emitted_edges = emitted;
}

// Keep the serial prefix arithmetic of sample_gks_tree. Only independent
// draws/searches and their disjoint output writes move to the item kernel.
template<bool OversizedOnly = false>
__global__ void prepare_gks_prefix(
        const work_record* canonical,
        const std::uint32_t* offsets,
        std::size_t pivot_count,
        double* prefix, const device_pivot* pivots) {
    const std::size_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal >= pivot_count) return;
    if constexpr (OversizedOnly)
        if (pivots[ordinal].incidence_end - pivots[ordinal].incidence_begin <= 128u) return;
    const std::uint32_t begin = offsets[ordinal];
    const std::uint32_t end = offsets[ordinal + 1];
    if (begin == end) return;
    prefix[begin] = canonical[begin].value;
    for (std::uint32_t i = begin + 1; i < end; ++i)
        prefix[i] = prefix[i - 1] + canonical[i].value;
}

template<bool OversizedOnly = false>
__global__ void sample_gks_items(
        const work_record* canonical,
        std::size_t count,
        const std::uint32_t* offsets,
        const device_pivot* pivots,
        const double* total_degree,
        const double* prefix,
        work_record* fill_candidates,
        std::uint8_t* fill_flags) {
    const std::size_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= count) return;
    const std::uint32_t i = static_cast<std::uint32_t>(item);
    const std::uint32_t ordinal = canonical[i].a;
    if constexpr (OversizedOnly)
        if (pivots[ordinal].incidence_end - pivots[ordinal].incidence_begin <= 128u) return;
    const std::uint32_t begin = offsets[ordinal];
    const std::uint32_t end = offsets[ordinal + 1];
    fill_flags[i] = 0;
    if (i + 1 == end) return;
    const double suffix = prefix[end - 1] - prefix[i];
    if (suffix <= 0.0) return;

    // Nonnegative canonical weights make finite prefixes nondecreasing:
    // skipped zero suffixes are trailing, so this item's draw is i-begin.
    // SplitMix advances additively modulo 2^64; next_unit supplies this draw's
    // final increment and the unchanged mixing/conversion operations.
    unsigned long long state = pivots[ordinal].seed +
        0x9E3779B97F4A7C15ULL * static_cast<unsigned long long>(i - begin);
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
}

// Fixed-size warp batches mirror the normal/oversized split used by the Yves
// implementation. Our CSR encounter order, positive factor convention and GKS
// arithmetic remain authoritative. Count warps own tiles of32 selected pivots;
// consecutive normal rows form batches with at most128 PHYSICAL slots total.
// Count publishes each batch's bounds; emit gives each actual batch its own warp.
constexpr unsigned kNormalSpan = 128;
constexpr unsigned kNormalWarps = 4;
constexpr unsigned kNormalBlock = 32 * kNormalWarps;
struct normal_batch_descriptor {
    std::uint32_t first, end;
};
static_assert(sizeof(normal_batch_descriptor) == 8);
struct normal_record {
    std::uint32_t pivot, neighbor, slot;
    double value;
};
struct normal_neighbor_less {
    __device__ bool operator()(const normal_record& a,
                               const normal_record& b) const {
        if (a.pivot != b.pivot) return a.pivot < b.pivot;
        if (a.neighbor != b.neighbor) return a.neighbor < b.neighbor;
        return a.slot < b.slot; // exact original encounter order for duplicates
    }
};
struct normal_weight_less {
    __device__ bool operator()(const normal_record& a,
                               const normal_record& b) const {
        if (a.pivot != b.pivot) return a.pivot < b.pivot;
        const auto ka = a.value == 0.0 ? 0ULL : double_bits(a.value);
        const auto kb = b.value == 0.0 ? 0ULL : double_bits(b.value);
        if (ka != kb) return ka < kb;
        return a.neighbor < b.neighbor;
    }
};
__device__ normal_record invalid_normal_record() {
    return {UINT32_MAX, UINT32_MAX, UINT32_MAX, 0.0};
}
__device__ void add_batch_count(std::uint64_t* target, std::uint64_t value) {
    atomicAdd(reinterpret_cast<unsigned long long*>(target),
              static_cast<unsigned long long>(value));
}

// Temporary oversized raw counts use the existing pivot counts/offsets before
// the common unique-count scan. Only live entries are counted and gathered.
__global__ void count_oversized_raw(
        const device_pivot* pivots, std::size_t p,
        const gpu_round_shadow_incidence* incidences,
        const std::uint8_t* active, std::uint32_t* counts,
        gpu_round_shadow_normal_batch_counts* stats, bool rows_all_live) {
    const std::size_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = threadIdx.x & 31u;
    device_pivot pivot{};
    if (ordinal < p) { pivot = pivots[ordinal]; counts[ordinal] = 0; }
    const auto span = pivot.incidence_end - pivot.incidence_begin;
    // A completed owned residual contains only active-endpoint incidences.
    // Its current owner offsets are exact even when cached degrees are stale.
    // This launch-uniform branch keeps one pivot per lane and reads no edges.
    if (rows_all_live) {
        if (ordinal < p && span > kNormalSpan) {
            counts[ordinal] = span;
            add_batch_count(&stats->oversized_pivots, 1);
            add_batch_count(&stats->oversized_spans, span);
            add_batch_count(&stats->oversized_gathered, span);
        }
        return;
    }
    // Preserve one-pivot/lane density on small stars; long rows cooperate only
    // for this integer live count, never for the ordered numerical degree sum.
    unsigned pending = __ballot_sync(0xffffffffu, ordinal < p && span > kNormalSpan);
    while (pending) {
        const int owner = __ffs(pending) - 1;
        const auto begin = __shfl_sync(0xffffffffu, pivot.incidence_begin, owner);
        const auto end = __shfl_sync(0xffffffffu, pivot.incidence_end, owner);
        const auto vertex = __shfl_sync(0xffffffffu, pivot.vertex, owner);
        std::uint32_t live = 0;
        for (std::size_t first = begin; first < end; first += 32) {
            const std::size_t i = first + lane;
            const bool keep = i < end && active[vertex] && active[incidences[i].neighbor];
            live += __popc(__ballot_sync(0xffffffffu, keep));
        }
        if (lane == static_cast<unsigned>(owner)) {
            counts[ordinal] = live;
            add_batch_count(&stats->oversized_pivots, 1);
            add_batch_count(&stats->oversized_spans, span);
            add_batch_count(&stats->oversized_gathered, live);
        }
        pending &= pending - 1;
    }
}

// Stable row-local compaction: ballot ranks preserve CSR encounter order and
// never use an atomic output slot. Partial chunks participate with false votes.
__global__ void gather_oversized_raw(
        const device_pivot* pivots, std::size_t p,
        const gpu_round_shadow_incidence* incidences,
        const std::uint8_t* active, const std::uint32_t* raw_offsets,
        work_record* output) {
    const unsigned lane = threadIdx.x & 31u;
    const std::size_t ordinal = (std::size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (ordinal >= p) return;
    const auto pivot = pivots[ordinal];
    if (pivot.incidence_end - pivot.incidence_begin <= kNormalSpan) return;
    std::uint32_t emitted = 0;
    for (std::size_t first = pivot.incidence_begin; first < pivot.incidence_end; first += 32) {
        const std::size_t i = first + lane;
        gpu_round_shadow_incidence edge{};
        if (i < pivot.incidence_end) edge = incidences[i];
        const bool keep = i < pivot.incidence_end && active[pivot.vertex] && active[edge.neighbor];
        const unsigned mask = __ballot_sync(0xffffffffu, keep);
        if (keep) {
            const unsigned rank = __popc(mask & ((1u << lane) - 1u));
            output[raw_offsets[ordinal] + emitted + rank] = {
                static_cast<std::uint32_t>(ordinal),
                static_cast<std::uint32_t>(edge.neighbor), edge.weight};
        }
        emitted += __popc(mask);
    }
}

__global__ void index_oversized_unique(
        const work_record* input, std::size_t count,
        std::uint32_t* starts, std::uint32_t* counts) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || (i && input[i - 1].a == input[i].a)) return;
    std::size_t end = i + 1;
    while (end < count && input[end].a == input[i].a) ++end;
    starts[input[i].a] = static_cast<std::uint32_t>(i);
    counts[input[i].a] = static_cast<std::uint32_t>(end - i);
}
__global__ void scatter_oversized_unique(
        const work_record* input, std::size_t count,
        const std::uint32_t* starts, const std::uint32_t* offsets,
        work_record* output) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const auto record = input[i];
        output[offsets[record.a] + i - starts[record.a]] = record;
    }
}

// Count fixes tile packing and compact append sizes. Emit consumes those exact
// batches independently, with the same neighbor reduction, factor entries before
// local weight sort, and canonical/fill writes at the same common offsets.
// Oversized pivots are skipped individually, never as a whole-round fallback.
template<bool Emit, bool GksFill = true>
__global__ void normal_batch_pass(
        const device_pivot* pivots, std::size_t p,
        normal_batch_descriptor* batches, std::size_t batch_count,
        const gpu_round_shadow_incidence* incidences,
        const std::uint8_t* active, std::uint32_t* counts,
        const std::uint32_t* offsets, gpu_round_shadow_normal_batch_counts* stats,
        const double* excess, double* total_degree,
        gpu_round_shadow_factor_column* columns, std::size_t column_base,
        gpu_round_shadow_factor_entry* entries, std::size_t entry_base,
        work_record* canonical, work_record* fills, std::uint8_t* fill_flags) {
    using sorter = cub::WarpMergeSort<normal_record, 4, 32>;
    static_assert(kNormalWarps * (sizeof(typename sorter::TempStorage) +
                  kNormalSpan * (sizeof(normal_record) + sizeof(double)) +
                  sizeof(unsigned)) <= 48 * 1024,
                  "normal batches must fit baseline CUDA static shared memory");
    __shared__ typename sorter::TempStorage sort_storage[kNormalWarps];
    __shared__ normal_record records[kNormalWarps][kNormalSpan];
    __shared__ double prefixes[kNormalWarps][kNormalSpan];
    __shared__ unsigned unique_counts[kNormalWarps];
    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x & 31u;
    const std::size_t work = std::size_t(blockIdx.x) * kNormalWarps + warp;
    std::size_t tile, tile_end;
    if constexpr (Emit) {
        if (work >= batch_count) return;
        tile = batches[work].first;
        tile_end = batches[work].end;
    } else {
        tile = work * 32;
        if (tile >= p) return;
        tile_end = min(p, tile + 32);
    } // only whole warps exit; no block-wide collective follows
    if constexpr (!Emit) {
        // Every pivot's count is initialized, including empty oversized rows.
        if (tile + lane < tile_end) counts[tile + lane] = 0;
    }
    __syncwarp();
    std::size_t first = tile;
    while (first < tile_end) {
        std::size_t end = tile_end;
        unsigned span = 0;
        if constexpr (Emit) {
            for (std::size_t ordinal = first; ordinal < end; ++ordinal)
                span += pivots[ordinal].incidence_end - pivots[ordinal].incidence_begin;
        } else {
            while (first < tile_end &&
                   pivots[first].incidence_end - pivots[first].incidence_begin > kNormalSpan)
                ++first;
            if (first == tile_end) break;
            end = first;
            while (end < tile_end) {
                const auto next_span = pivots[end].incidence_end - pivots[end].incidence_begin;
                if (next_span > kNormalSpan || next_span > kNormalSpan - span) break;
                span += next_span;
                ++end; // zero-only batches also advance
            }
        }
        for (unsigned i = lane; i < kNormalSpan; i += 32)
            records[warp][i] = invalid_normal_record();
        __syncwarp();
        unsigned base = 0;
        for (std::size_t ordinal = first; ordinal < end; ++ordinal) {
            const auto pivot = pivots[ordinal];
            const unsigned row_span = pivot.incidence_end - pivot.incidence_begin;
            for (unsigned slot = lane; slot < row_span; slot += 32) {
                const auto edge = incidences[pivot.incidence_begin + slot];
                if (active[pivot.vertex] && active[edge.neighbor])
                    records[warp][base + slot] = {
                        static_cast<std::uint32_t>(ordinal),
                        static_cast<std::uint32_t>(edge.neighbor), slot, edge.weight};
            }
            base += row_span;
        }
        __syncwarp();
        if constexpr (Emit) {
            // Raw degree starts at+0 and folds CSR slots before deduplication.
            if (!lane) {
                unsigned base = 0;
                for (std::size_t ordinal = first; ordinal < end; ++ordinal) {
                    const auto pivot = pivots[ordinal];
                    const unsigned row_span = pivot.incidence_end - pivot.incidence_begin;
                    double degree = 0.0;
                    for (unsigned slot = 0; slot < row_span; ++slot) {
                        const auto record = records[warp][base + slot];
                        if (record.pivot != UINT32_MAX) degree += record.value;
                    }
                    total_degree[ordinal] = degree;
                    base += row_span;
                }
            }
        }
        __syncwarp();
        normal_record items[4];
        #pragma unroll
        for (unsigned i = 0; i < 4; ++i) items[i] = records[warp][lane * 4 + i];
        sorter(sort_storage[warp]).Sort(items, normal_neighbor_less{});
        #pragma unroll
        for (unsigned i = 0; i < 4; ++i) records[warp][lane * 4 + i] = items[i];
        __syncwarp();
        if (!lane) {
            unsigned gathered = 0, unique = 0;
            while (gathered < span && records[warp][gathered].pivot != UINT32_MAX) ++gathered;
            for (unsigned i = 0; i < gathered;) {
                auto record = records[warp][i++];
                // Start with the FIRST value, exactly like reduce_equal_pairs.
                while (i < gathered && records[warp][i].pivot == record.pivot &&
                       records[warp][i].neighbor == record.neighbor)
                    record.value += records[warp][i++].value;
                records[warp][unique++] = record;
            }
            unique_counts[warp] = unique;
            if constexpr (!Emit) {
                for (unsigned i = 0; i < unique; ++i) ++counts[records[warp][i].pivot];
                add_batch_count(&stats->normal_pivots, end - first);
                add_batch_count(&stats->normal_spans, span);
                add_batch_count(&stats->normal_gathered, gathered);
                add_batch_count(&stats->normal_unique, unique);
                // Reuse the existing counter atomic to publish one descriptor.
                // Order may vary, but each batch owns disjoint pivot/append slots.
                const auto batch = atomicAdd(
                    reinterpret_cast<unsigned long long*>(&stats->batches), 1ULL);
                batches[batch] = {static_cast<std::uint32_t>(first),
                                  static_cast<std::uint32_t>(end)};
            } else {
                unsigned i = 0;
                for (std::size_t ordinal = first; ordinal < end; ++ordinal) {
                    const auto vertex = pivots[ordinal].vertex;
                    const auto begin = offsets[ordinal], finish = offsets[ordinal + 1];
                    double degree = total_degree[ordinal];
                    const double ev = excess[vertex];
                    if (begin == finish) degree = ev > 0.0 ? ev : 1.0;
                    else { degree += ev; if (degree <= 0.0) degree = 1.0; }
                    total_degree[ordinal] = degree;
                    const double root = sqrt(degree);
                    columns[column_base + ordinal] = {
                        static_cast<node_index>(vertex),
                        static_cast<factor_value_t>(static_cast<float>(root)),
                        static_cast<std::uint64_t>(entry_base + begin), finish - begin};
                    for (std::uint32_t j = begin; j < finish; ++j, ++i)
                        entries[entry_base + j] = {
                            static_cast<node_index>(records[warp][i].neighbor),
                            static_cast<factor_value_t>(static_cast<float>(records[warp][i].value / root))};
                }
            }
        }
        __syncwarp();
        if constexpr (Emit) {
            const unsigned unique = unique_counts[warp];
            #pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                const unsigned position = lane * 4 + i;
                items[i] = position < unique ? records[warp][position] : invalid_normal_record();
            }
            sorter(sort_storage[warp]).Sort(items, normal_weight_less{});
            #pragma unroll
            for (unsigned i = 0; i < 4; ++i) records[warp][lane * 4 + i] = items[i];
            __syncwarp();
            if (!lane) {
                unsigned local_begin = 0;
                for (std::size_t ordinal = first; ordinal < end; ++ordinal) {
                    const auto begin = offsets[ordinal], finish = offsets[ordinal + 1];
                    const unsigned count = finish - begin, local_end = local_begin + count;
                    if (count) {
                        prefixes[warp][local_begin] = records[warp][local_begin].value;
                        for (unsigned i = local_begin + 1; i < local_end; ++i)
                            prefixes[warp][i] = prefixes[warp][i - 1] + records[warp][i].value;
                    }
                    for (unsigned local = local_begin; local < local_end; ++local) {
                        const auto record = records[warp][local];
                        const auto dest = begin + local - local_begin;
                        canonical[dest] = {record.pivot, record.neighbor, record.value};
                        fill_flags[dest] = 0;
                        if constexpr (!GksFill) continue;
                        if (local + 1 == local_end) continue;
                        const double suffix = prefixes[warp][local_end - 1] - prefixes[warp][local];
                        if (suffix <= 0.0) continue;
                        unsigned long long state = pivots[ordinal].seed +
                            0x9E3779B97F4A7C15ULL * static_cast<unsigned long long>(local - local_begin);
                        const double target = prefixes[warp][local] + next_unit(state) * suffix;
                        unsigned lo = local + 1, hi = local_end;
                        while (lo < hi) {
                            const unsigned mid = lo + (hi - lo) / 2;
                            if (prefixes[warp][mid] <= target) lo = mid + 1;
                            else hi = mid;
                        }
                        const unsigned partner = lo < local_end ? lo : local_end - 1;
                        const auto a = record.neighbor, b = records[warp][partner].neighbor;
                        fills[dest] = {min(a, b), max(a, b), record.value * suffix / total_degree[ordinal]};
                        fill_flags[dest] = 1;
                    }
                    local_begin = local_end;
                }
            }
        }
        __syncwarp();
        if constexpr (Emit) break; // one warp consumes exactly one descriptor
        first = end;
    }
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
    const auto item = i < count ? gpu_round_shadow_item_hash(
        gpu_round_shadow_tags::fill, fill[i].a, fill[i].b,
        stored_weight_bits(fill[i].value)) : 0;
    __shared__ device_digest warp_values[kBlock / 32];
    digest_add_block({item, item}, &semantics->fill, warp_values);
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

template<bool Audit>
__global__ void hash_residual_incidences(
        const gpu_round_shadow_incidence* residual,
        std::size_t count,
        device_semantics* semantics) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long item = 0, topology = 0;
    if (i < count) {
        const auto edge = residual[i];
        if constexpr (Audit)
            item = gpu_round_shadow_item_hash(gpu_round_shadow_tags::residual,
                edge.owner, edge.neighbor, stored_weight_bits(edge.weight));
        if (edge.owner < edge.neighbor)
            topology = gpu_device_selection_topology_hash(edge.owner, edge.neighbor);
    }
    __shared__ device_digest warp_values[kBlock / 32];
    if constexpr (Audit)
        digest_add_block({item, item}, &semantics->residual, warp_values);
    digest_add_block({topology, topology}, &semantics->topology, warp_values);
}

// Co-rank a prefix of a stable merge: left (survivor) records win owner ties.
template<class Incidence>
__device__ std::size_t survivor_merge_rank(
        const Incidence* left, std::size_t left_count,
        const Incidence* right, std::size_t right_count,
        std::size_t rank) {
    std::size_t low = rank > right_count ? rank - right_count : 0;
    std::size_t high = rank < left_count ? rank : left_count;
    while (low < high) {
        const std::size_t a = low + (high - low) / 2;
        const std::size_t b = rank - a;
        if (b && a < left_count && right[b - 1].owner >= left[a].owner)
            low = a + 1;
        else
            high = a;
    }
    return low;
}

template<bool HashTopology>
__global__ void merge_survivors_and_fill(
        const gpu_round_shadow_incidence* survivors, std::size_t survivor_count,
        const gpu_round_shadow_incidence* fill, std::size_t fill_count,
        gpu_round_shadow_incidence* output, device_semantics* semantics) {
    const std::size_t total = survivor_count + fill_count;
    const std::size_t begin = std::size_t(blockIdx.x) * kBlock;
    const std::size_t end = begin + kBlock < total ? begin + kBlock : total;
    __shared__ std::size_t left_begin, right_begin, left_size;
    // No default member initializers: shared storage needs no construction.
    struct tile_incidence { node_index owner, neighbor; double weight; };
    __shared__ tile_incidence tile[kBlock];
    if (!threadIdx.x) {
        left_begin = survivor_merge_rank(
            survivors, survivor_count, fill, fill_count, begin);
        right_begin = begin - left_begin;
        left_size = survivor_merge_rank(
            survivors, survivor_count, fill, fill_count, end) - left_begin;
    }
    __syncthreads();
    const std::size_t local = threadIdx.x;
    const std::size_t size = end - begin;
    if (local < size) {
        const auto edge = local < left_size ? survivors[left_begin + local]
            : fill[right_begin + local - left_size];
        tile[local] = {edge.owner, edge.neighbor, edge.weight};
    }
    __syncthreads();
    unsigned long long topology = 0;
    if (local < size) {
        const std::size_t a = survivor_merge_rank(
            tile, left_size, tile + left_size, size - left_size, local);
        const std::size_t b = local - a;
        const auto edge = a < left_size &&
            (b == size - left_size || tile[a].owner <= tile[left_size + b].owner)
            ? tile[a] : tile[left_size + b];
        output[begin + local] = {edge.owner, edge.neighbor, edge.weight};
        if constexpr (HashTopology) {
            if (edge.owner < edge.neighbor)
                topology = gpu_device_selection_topology_hash(edge.owner, edge.neighbor);
        }
    }
    if constexpr (HashTopology) {
        // Every lane joins, including the partial final tile. Keep reduction
        // storage separate: other warps can still be reading the merge tile.
        __shared__ device_digest warp_values[kBlock / 32];
        digest_add_block({topology, topology}, &semantics->topology, warp_values);
    }
}

// Each represented owner has exactly one last incidence: no contested writes.
// Prefix maximum fills gaps for isolated/inactive owners. offset[n] is separate
// so n == INT_MAX does not require an unsupported n+1-item CUB scan.
__global__ void residual_owner_ends(
        const gpu_round_shadow_incidence* residual, std::size_t count,
        std::size_t vertex_count, std::uint32_t* offsets) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count && std::size_t(residual[i].owner) + 1 < vertex_count &&
        (i + 1 == count || residual[i].owner != residual[i + 1].owner))
        offsets[std::size_t(residual[i].owner) + 1] = static_cast<std::uint32_t>(i + 1);
}

__global__ void residual_degrees_from_offsets(
        const std::uint32_t* offsets, std::size_t count, std::uint32_t* degrees) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) degrees[i] = offsets[i + 1] - offsets[i];
}

struct owner_offset_max {
    __host__ __device__ std::uint32_t operator()(
            std::uint32_t a, std::uint32_t b) const { return a < b ? b : a; }
};

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
    unsigned long long item = 0;
    if (i < count) {
        const auto edge = residual[i];
        const std::uint64_t ordinal = i - std::size_t(owner_offsets[edge.owner]);
        item = gpu_round_shadow_ordered_residual_hash(
            edge.owner, ordinal, edge.neighbor, stored_weight_bits(edge.weight));
    }
    __shared__ device_digest warp_values[kBlock / 32];
    digest_add_block({item, item}, &semantics->ordered_residual, warp_values);
}

template<bool Audit>
__global__ void hash_residual_state(
        std::size_t vertex_count,
        const std::uint8_t* active,
        const std::uint32_t* degrees,
        const double* excess,
        device_semantics* semantics) {
    const std::size_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    const bool live = vertex < vertex_count && active[vertex];
    unsigned long long active_hash = 0, degree_hash = 0, excess_hash = 0;
    if (live) {
        active_hash = gpu_device_selection_active_hash(static_cast<node_index>(vertex));
        if constexpr (Audit) {
            degree_hash = gpu_round_shadow_item_hash(
                gpu_round_shadow_tags::live_degree, vertex, 0, degrees[vertex]);
            excess_hash = gpu_round_shadow_item_hash(
                gpu_round_shadow_tags::excess, vertex, 0, double_bits(excess[vertex]));
        }
    }
    __shared__ device_digest warp_values[kBlock / 32];
    digest_add_block({active_hash, active_hash}, &semantics->active, warp_values);
    if constexpr (Audit) {
        digest_add_block({degree_hash, degree_hash}, &semantics->live_degree, warp_values);
        digest_add_block({excess_hash, excess_hash}, &semantics->canonical_excess, warp_values);
    }
    const auto count = reduce_block_digest({0, live ? 1ULL : 0ULL}, warp_values);
    if (!threadIdx.x) atomicAdd(&semantics->active_count, count.sum_hash);
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
            cub_count(count, "residual owner-sort query"),
            0, owner_sort_end_bit(vertex_count)),
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
    query_incidence_sort(gpu_round_shadow_checked_mul(
        static_cast<std::size_t>(shape.raw_fill_edges), 2, "fill-only sort query"));
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
        bytes = 0;
        cuda_check(cub::DeviceScan::InclusiveScan(
            nullptr, bytes, fake_pointer<std::uint32_t>(0x1700),
            fake_pointer<std::uint32_t>(0x1700), owner_offset_max{},
            cub_count(vertex_count, "owner-offset maximum query")),
            "query owner-offset maximum temporary bytes");
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
    common = add_bytes(common, p, sizeof(std::uint32_t), "oversized unique starts");
    common = add_bytes(common, 1, sizeof(gpu_round_shadow_normal_batch_counts), "normal-batch counts");

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

__global__ void csc_initial_counts_excess(
        int n, const int* ptr, const int* idx, const double* values,
        double reg_eps, std::uint32_t* counts, std::uint8_t* active,
        double* excess, std::uint32_t* status) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    double diag = 0.0, degree = 0.0;
    std::uint32_t count = 0;
    for (int k = ptr[v]; k < ptr[v + 1]; ++k) {
        const double value = values[k];
        if (!isfinite(value)) atomicOr(status, 1u);
        if (idx[k] == v) diag = value;
        else {
            // Exact original-column order, before pool rounding or lower-value
            // substitution. Explicit operations exclude an FMA/reassociation.
            degree = __dadd_rn(degree, -value);
            ++count;
        }
    }
    double e = __dadd_rn(diag, -degree);
    e = e > __dmul_rn(diag, 1e-12) ? e : 0.0;
    if (reg_eps > 0.0) {
        const double regularized = __dmul_rn(reg_eps, diag);
        if (e < regularized) e = regularized;
    }
    if (!isfinite(e) || e < 0.0) atomicOr(status, 1u);
    if (e > 0.0) atomicOr(status + 1, 1u);
    counts[v] = count;
    active[v] = 1;
    excess[v] = e;
}

__global__ void csc_initial_incidences(
        int n, const int* ptr, const int* idx, const double* values,
        const std::uint32_t* offsets, gpu_round_shadow_incidence* output,
        std::uint32_t* status) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    std::uint32_t out = offsets[v];
    for (int k = ptr[v]; k < ptr[v + 1]; ++k) {
        const int u = idx[k];
        if (u == v) continue;
        int canonical = k;
        if (u < v) {
            int lo = ptr[u], hi = ptr[u + 1];
            while (lo < hi) {
                const int mid = lo + (hi - lo) / 2;
                if (idx[mid] < v) lo = mid + 1; else hi = mid;
            }
            if (lo == ptr[u + 1] || idx[lo] != v) {
                atomicOr(status, 2u); continue;
            }
            canonical = lo;
        }
        const double original = -values[canonical];
#ifdef APXCHOL_POOL_FP32
        const double weight = static_cast<double>(__double2float_rn(original));
#else
        const double weight = original;
#endif
        if (!isfinite(weight) || weight < 0.0) atomicOr(status, 1u);
        if (out >= offsets[v + 1]) { atomicOr(status, 4u); continue; }
        output[out++] = {static_cast<node_index>(v), static_cast<node_index>(u), weight};
    }
    if (out != offsets[v + 1]) atomicOr(status, 4u);
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
                    std::size_t count, const char* what, int end_bit = 64) {
        if (!count) return;
        std::size_t bytes = 0;
        cuda_check(cub::DeviceRadixSort::SortPairs(
            nullptr, bytes, keys_in, keys_out, values_in, values_out,
            cub_count(count, what), 0, end_bit), "query CUB stable radix sort");
        if (bytes > storage_.count())
            throw std::logic_error(
                "GPU round shadow: preflight underestimated CUB sort");
        cuda_check(cub::DeviceRadixSort::SortPairs(
            storage_.get(), bytes, keys_in, keys_out, values_in, values_out,
            cub_count(count, what), 0, end_bit), what);
    }

    void inclusive_owner_max(std::uint32_t* offsets, std::size_t count) {
        if (!count) return;
        std::size_t bytes = 0;
        cuda_check(cub::DeviceScan::InclusiveScan(
            nullptr, bytes, offsets, offsets, owner_offset_max{},
            cub_count(count, "owner-offset maximum")), "query owner-offset maximum");
        if (bytes > storage_.count())
            throw std::logic_error("GPU round shadow: underestimated owner-offset maximum");
        cuda_check(cub::DeviceScan::InclusiveScan(
            storage_.get(), bytes, offsets, offsets, owner_offset_max{},
            cub_count(count, "owner-offset maximum")), "propagate empty owner offsets");
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

#include "cuda_owned_sparsify.cuh"

struct owned_probe_range { std::uint32_t begin, end, output; };

__global__ void owned_probe_ranges(const node_index* vertices, std::size_t count,
        const std::uint32_t* offsets, const std::uint8_t* active,
        owned_probe_range* ranges, unsigned* invalid) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const auto v = vertices[i];
        if (!active[v]) atomicExch(invalid, 1U);
        ranges[i] = {offsets[v], offsets[v + 1], 0};
    }
}

__global__ void owned_probe_keys(const owned_probe_range* ranges,
        std::size_t count, const gpu_round_shadow_incidence* residual,
        std::uint64_t* keys) {
    const auto row = (std::size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (row >= count) return;
    const auto range = ranges[row];
    for (std::size_t k = range.begin + (threadIdx.x & 31); k < range.end; k += 32)
        keys[range.output + k - range.begin] =
            (std::uint64_t(row) << 32) | std::uint64_t(residual[k].neighbor);
}

__global__ void owned_probe_distinct(const std::uint64_t* keys,
        std::size_t count, std::uint32_t* degrees) {
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count && (!i || keys[i] != keys[i - 1]))
        atomicAdd(degrees + (keys[i] >> 32), 1U);
}

void owned_probe_check_fit(std::size_t extra) {
    std::size_t free = 0, total = 0;
    cuda_check(cudaMemGetInfo(&free, &total), "query owned sparsification fit");
    if (extra > free || total / 10 > free - extra)
        throw std::runtime_error("GPU sparsification scratch does not fit with device margin");
}


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

#if defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
gpu_owned_sparsify_test_output gpu_sparsify_owned_residual_for_test(
        const gpu_round_shadow_input& input, std::uint64_t seed,
        double keep_probability, double scale_override) {
    round_shadow_reference_detail::validate_input(input);
    const std::size_t n = input.vertex_count, live = input.incidences.size();
    cub_count(n, "test sparsification vertices");
    cub_count(live, "test sparsification incidences");
    allocation_tracker tracker;
    device_buffer<gpu_round_shadow_incidence> original(tracker), output(tracker);
    device_buffer<std::uint8_t> active(tracker);
    device_buffer<std::uint32_t> offsets(tracker), degrees(tracker);
    original.allocate(std::max<std::size_t>(live, 1), "allocate test sparse input");
    output.allocate(std::max<std::size_t>(live, 1), "allocate test sparse output");
    active.allocate(n, "allocate test sparse active flags");
    offsets.allocate(n + 1, "allocate test sparse offsets");
    degrees.allocate(n, "allocate test sparse degrees");
    copy_to_device(original.get(), input.incidences.data(), live, "upload test sparse input");
    copy_to_device(active.get(), input.active.data(), n, "upload test sparse active flags");
    gpu_owned_sparsify_test_output result;
    result.stats = sparsify_owned_residual_device(input.vertex_count, live, original.get(),
        active.get(), keep_probability, seed, output.get(), offsets.get(), degrees.get(),
        tracker, &result, scale_override);
    result.residual.vertex_count = input.vertex_count;
    result.residual.active = input.active;
    result.residual.excess = input.excess;
    std::vector<std::uint32_t> host_offsets(n + 1);
    copy_to_host(host_offsets.data(), offsets.get(), n + 1, "download test sparse offsets");
    result.residual.owner_offsets.assign(host_offsets.begin(), host_offsets.end());
    result.residual.incidences.resize(2 * result.stats.kept_edges);
    copy_to_host(result.residual.incidences.data(), output.get(), result.residual.incidences.size(),
                 "download test sparse residual");
    return result;
}
#endif


// These buffers are live throughout compute(). Owning sessions may retain
// their capacity between rounds; every used element is rewritten before read.
// Large phase-local record/incidence arrays keep their existing early releases.
struct round_common_scratch {
    device_buffer<std::byte> cub_temp;
    device_buffer<int> selected_count;
    device_buffer<device_pivot> pivots;
    device_buffer<std::uint32_t> pivot_counts;
    device_buffer<std::uint32_t> pivot_offsets;
    device_buffer<double> total_degree;
    device_buffer<device_semantics> semantics;
    device_buffer<std::uint8_t> flags;
    device_buffer<std::uint32_t> oversized_starts;
    device_buffer<gpu_round_shadow_normal_batch_counts> batch_counts;
    device_buffer<normal_batch_descriptor> batch_descriptors;

    explicit round_common_scratch(allocation_tracker& tracker)
        : cub_temp(tracker), selected_count(tracker), pivots(tracker),
          pivot_counts(tracker), pivot_offsets(tracker), total_degree(tracker),
          semantics(tracker), flags(tracker), oversized_starts(tracker),
          batch_counts(tracker), batch_descriptors(tracker) {}

    // Scratch needs no preserved prefix. Release before growth so old and new
    // capacities never overlap; unchanged/smaller requests allocate nothing.
    template<class T>
    static void prepare(device_buffer<T>& buffer, std::size_t count,
                        const char* what) {
        if (count <= buffer.count()) return;
        buffer.release("replace common round scratch");
        buffer.allocate(count, what);
    }

    // planned_peak_bytes already includes each current requested common
    // buffer in every phase. Add only previously retained capacity in excess
    // of that request, without double-counting the fresh allocation allowance.
    std::size_t retained_excess(std::size_t cub_bytes, std::size_t p) const {
        std::size_t extra = 0;
        auto add = [&](const auto& buffer, std::size_t required) {
            if (buffer.count() > required)
                extra = gpu_round_shadow_checked_add(extra,
                    gpu_round_shadow_checked_mul(buffer.count() - required,
                        sizeof(*buffer.get()), "retained round scratch"),
                    "retained round scratch sum");
        };
        add(cub_temp, cub_bytes); add(selected_count, 1); add(pivots, p);
        add(pivot_counts, p); add(pivot_offsets, p + 1);
        add(total_degree, p); add(semantics, 1);
        add(oversized_starts, p); add(batch_counts, 1);
        add(batch_descriptors, p);
        return extra;
    }

    void release() {
        cub_temp.release("retire common CUB workspace");
        selected_count.release("retire common selected count");
        pivots.release("retire common pivots");
        pivot_counts.release("retire common pivot counts");
        pivot_offsets.release("retire common pivot offsets");
        total_degree.release("retire common total degrees");
        semantics.release("retire common semantic report");
        flags.release("retire owned round flags");
        oversized_starts.release("retire oversized unique starts");
        batch_counts.release("retire normal-batch counters");
        batch_descriptors.release("retire normal-batch descriptors");
    }
};

struct gpu_round_shadow_device_state::impl {
    enum class reimport_reason {
        none,
        parallel_order,
        authoritative_host_rebuild,
    };

    clique_sampler sampler = clique_sampler::gks;
    allocation_tracker tracker;
    round_common_scratch owned_common_scratch;
    device_buffer<double> cycle_workspace;
    device_buffer<std::uint32_t> cycle_status;
    device_buffer<gpu_round_shadow_incidence> residual_0;
    device_buffer<gpu_round_shadow_incidence> residual_1;
    device_buffer<std::uint32_t> owner_offsets_0;
    device_buffer<std::uint32_t> owner_offsets_1;
    device_buffer<std::uint8_t> active;
    device_buffer<double> excess;
    device_buffer<gpu_round_shadow_factor_column> factor_columns;
    device_buffer<gpu_round_shadow_factor_entry> factor_entries;
    // All stable radix sorts use the same two key arrays. Contents are scratch;
    // retain only capacity across stages and rounds instead of synchronously
    // freeing/reallocating both arrays at each sort boundary.
    device_buffer<std::uint64_t> sort_keys_a;
    device_buffer<std::uint64_t> sort_keys_b;
    // Scratch for the fixed-n owning session only. The heap-stable impl owns
    // both this buffer and its tracker; moving the public state does not move
    // either object. Contents are cleared in full before every degree count.
    device_buffer<std::uint32_t> owned_degrees;
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
    bool generation_has_device_selection = false;
    bool device_owned_prefix = false;
    bool owned_prefix_materialized = false;
    bool csc_initial_round_pending = false;
    // Only successful owned-round publication establishes this invariant.
    // Generic imports may retain inactive incidence slots; initial CSC also
    // conservatively uses the existing filtered count.
    bool resident_rows_all_live = false;
    bool poisoned = false;
    reimport_reason next_reimport = reimport_reason::none;
    std::uint64_t generation = 0;
    int cuda_device = -1;
    bool selection_producer_bound = false;
    std::weak_ptr<const void> selection_producer;
    std::uint64_t last_selection_generation = 0;
    std::uint64_t last_selection_topology_generation = 0;
    gpu_device_selection_content resident_selection_content;
    // A private generation lease; only weak borrows leave the numerical owner.
    // Declared after the buffers, so destruction revokes before freeing them.
    std::shared_ptr<const void> owned_residual_epoch;
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

    explicit impl(clique_sampler kind = clique_sampler::gks)
        : sampler(kind), owned_common_scratch(tracker), cycle_workspace(tracker),
          cycle_status(tracker), residual_0(tracker), residual_1(tracker),
          owner_offsets_0(tracker), owner_offsets_1(tracker),
          active(tracker), excess(tracker), factor_columns(tracker),
          factor_entries(tracker), sort_keys_a(tracker), sort_keys_b(tracker),
          owned_degrees(tracker),
          selected_epoch_by_vertex(tracker),
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
        owned_residual_epoch.reset();
        resident_rows_all_live = false;
        poisoned = true;
        generation_accepted = false;
        generation_has_device_selection = false;
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
        bool resident_only,
        const gpu_device_selection_content* paired_initial_content = nullptr);
    gpu_round_shadow_factor_log download_factor_log();
    void accept_device_generation(std::uint64_t accepted_generation);
    void reject_device_generation(std::uint64_t rejected_generation) noexcept;
    void certify_cpu_round(
        bool cpu_order_reproducible,
        const gpu_round_shadow_state_fingerprint& cpu_state,
        bool cpu_excess_may_differ);
    void invalidate_for_authoritative_host_rebuild(
        const gpu_round_shadow_state_fingerprint& cpu_state);

    void validate_owned_preview(gpu_block_frontend& frontend) {
        require_current_device("owned sparsification preview");
        if (poisoned || !has_state || !generation_accepted || !device_owned_prefix ||
            owned_prefix_materialized || next_reimport != reimport_reason::none ||
            excess_may_differ || !owned_residual_epoch)
            throw std::logic_error("GPU sparsification requires an accepted immutable owned residual");
        const auto selected = frontend.device_selection().inspect();
        if (selected.cuda_device != cuda_device || !selected.independence_certified ||
            !gpu_device_selection_state_matches(selected.content, resident_selection_content))
            throw std::invalid_argument("GPU sparsification preview has different device/content");
        if (selection_producer_bound) {
            const auto producer = selection_producer.lock();
            if (!producer || producer != selected.producer_identity ||
                selected.generation <= last_selection_generation ||
                selected.topology_generation <= last_selection_topology_generation)
                throw std::invalid_argument("GPU sparsification preview is stale or from another producer");
        }
    }
};


gpu_round_shadow_report gpu_round_shadow_device_state::impl::compute(
        const gpu_round_shadow_input& input,
        const gpu_round_shadow_report* expected_shape,
        const gpu_device_selection* device_selection,
        std::uint64_t run_seed,
        bool resident_only, const gpu_device_selection_content* paired_initial_content) {
    if (poisoned)
        throw std::logic_error(
            "GPU round shadow: device state is poisoned by a failed generation");

    // Arm before every remaining precondition, provenance check, host
    // continuity check, allocation, or CUDA call. A failed compute attempt
    // must never leave an older accepted generation consumable.
    failed_generation_guard generation_guard{*this};
    generation_guard.arm();
    // Revoke before any validation, allocation, or numerical mutation. A failed
    // next generation cannot leave a selector borrowing the previous arrays.
    owned_residual_epoch.reset();

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

    if (paired_initial_content && (!device_owned_prefix || has_state || resident_only ||
        paired_initial_content->vertex_count != input.vertex_count ||
        paired_initial_content->active_count != input.vertex_count))
        throw std::logic_error("GPU paired initial content escaped its first owned import");

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
            : paired_initial_content ? *paired_initial_content
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
    // An owned generation has no independent CPU audit consumer. Keep only
    // the topology/active certificate that the separate selector consumes;
    // weighted/factor/fill digests and pivot counters belong to shadow audits.
    const bool audit_payload = !device_owned_prefix;
    const bool reuse_input = resident_only || (has_state && reusable);
    const bool import_input = !reuse_input;
    if (import_input) resident_rows_all_live = false;
    const bool count_owned_spans = resident_only && device_owned_prefix &&
                                   resident_rows_all_live;
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
    const std::size_t sampler_components = sampler == clique_sampler::heavy_core_k2 ? 8 :
        sampler == clique_sampler::trace_cycle ? 1 : 0;
    if (sampler_components) {
        const auto required = gpu_round_shadow_checked_mul(
            capacity_shape.unique_neighbors, sampler_components, "cycle sampler moments");
        // Retained capacity overlaps every round phase, including growth.
        planned = add_bytes(planned, cycle_workspace.count(), sizeof(double),
                            "retained cycle sampler moments");
        if (required > cycle_workspace.count())
            planned = add_bytes(planned, required, sizeof(double), "grown cycle sampler moments");
        planned = add_bytes(planned, 2, sizeof(std::uint32_t), "cycle sampler status");
    }
    // Oversized uniques remain live beside the common canonical stream through
    // factor/sample. Retain a conservative full-g allowance in every phase.
    if (device_owned_prefix)
        planned = add_bytes(planned, capacity_shape.gathered_incidences,
                            sizeof(work_record), "oversized unique overlap");
    if (device_owned_prefix)
        planned = add_bytes(planned, p, sizeof(normal_batch_descriptor),
                            "normal-batch descriptors");
    const int normal_blocks = device_owned_prefix && p
        ? static_cast<int>((p + 32 * kNormalWarps - 1) / (32 * kNormalWarps)) : 0;
    const int oversized_blocks = device_owned_prefix
        ? blocks_for(gpu_round_shadow_checked_mul(p, 32, "oversized row launch")) : 0;
    if (device_owned_prefix)
        planned = gpu_round_shadow_checked_add(planned,
            owned_common_scratch.retained_excess(cub_bytes, p),
            "retained common round scratch capacity");
    // Flags previously existed only in some phases, with smaller g/u requests.
    // Retaining input-sized flags overlaps every phase: conservatively add the
    // full capacity, leaving the old phase-local allowances in place. Growth
    // releases scratch first, so old/new capacities do not overlap.
    if (device_owned_prefix)
        planned = add_bytes(planned,
            std::max(owned_common_scratch.flags.count(), input_count),
            sizeof(std::uint8_t), "retained owned round flags");
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
    // The phase-local plan includes dedup/sample keys only in their own
    // phases. Retained keys overlap every phase, so add their entire maximum
    // capacity here (conservatively leaving the old phase allowance in place).
    // Growth releases obsolete scratch first; no old/new overlap is needed.
    const std::size_t sort_key_capacity = std::max({
        sort_keys_a.count(), sort_keys_b.count(),
        static_cast<std::size_t>(capacity_shape.gathered_incidences),
        static_cast<std::size_t>(capacity_shape.unique_neighbors),
        static_cast<std::size_t>(capacity_shape.live_incidences)});
    planned = add_bytes(planned, sort_key_capacity,
                        2 * sizeof(std::uint64_t), "retained radix-sort keys");
    // Retained degrees overlap the early phases and the inter-round state.
    // Keep the old mutation allowance as a conservative overestimate.
    planned = add_bytes(planned,
                        std::max(owned_degrees.count(), device_owned_prefix ? n : 0),
                        sizeof(std::uint32_t), "retained owned degrees");
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

    round_common_scratch local_common_scratch(tracker);
    auto& common = device_owned_prefix ? owned_common_scratch : local_common_scratch;
    auto& cub_temp = common.cub_temp;
    auto& selected_count = common.selected_count;
    auto& pivots = common.pivots;
    auto& pivot_counts = common.pivot_counts;
    auto& pivot_offsets = common.pivot_offsets;
    auto& total_degree = common.total_degree;
    auto& semantics = common.semantics;
    device_buffer<gpu_round_shadow_pivot_counter> pivot_counters(tracker);

    common.prepare(cub_temp, cub_bytes, "allocate CUB temporary storage");
    common.prepare(selected_count, 1, "allocate CUB selected count");
    common.prepare(pivots, p, "allocate pivots");
    if (audit_payload) pivot_counters.allocate(p, "allocate pivot counters");
    common.prepare(pivot_counts, p, "allocate pivot counts");
    common.prepare(pivot_offsets, p + 1, "allocate pivot offsets");
    common.prepare(total_degree, p, "allocate total degrees");
    common.prepare(semantics, 1, "allocate semantic report");

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
    const bool total_diagnostics = gpu_setup_diagnostics();
    cuda_event total_begin("create total start event", total_diagnostics);
    cuda_event total_end("create total end event", total_diagnostics);

    gpu_round_shadow_report report;
    report.input_incidences = r;
    report.resident_input_incidences = input_count;
    if (audit_payload) report.pivots.resize(p);
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
        if (csc_initial_round_pending) csc_initial_round_pending = false;
        else ++state_reuses;
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
    if (total_diagnostics)
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
            if (audit_payload)
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
    device_buffer<work_record> oversized_unique(tracker);
    device_buffer<std::uint8_t> local_flags(tracker);
    auto& flags = device_owned_prefix ? owned_common_scratch.flags : local_flags;
    auto& oversized_starts = common.oversized_starts;
    auto& batch_counts = common.batch_counts;
    auto& batch_descriptors = common.batch_descriptors;
    std::size_t sort_g = 0, oversized_u = 0, normal_batch_count = 0;
    // The common canonical stream remains compact; only the oversized stream
    // visits the three global neighbor/weight sorts in the owned path.
    if (device_owned_prefix) {
        common.prepare(flags, input_count, "allocate owned round flags");
        common.prepare(oversized_starts, p, "allocate oversized unique starts");
        common.prepare(batch_counts, 1, "allocate normal-batch counters");
        common.prepare(batch_descriptors, p, "allocate normal-batch descriptors");
    } else {
        first.allocate(input_count, "allocate gather candidates");
        flags.allocate(input_count, "allocate gather flags");
        second.allocate(std::max<std::size_t>(g_capacity, 1),
                        "allocate gathered records");
    }
    report.timings.gather_ms = timer.measure_ms([&] {
        if (device_owned_prefix) {
            cuda_check(cudaMemset(batch_counts.get(), 0, sizeof(gpu_round_shadow_normal_batch_counts)),
                       "clear normal-batch counters");
            if (p) {
                count_oversized_raw<<<blocks_for(p), kBlock>>>(
                    pivots.get(), p, device_input.get(), active.get(),
                    pivot_counts.get(), batch_counts.get(), count_owned_spans);
                cuda_check(cudaGetLastError(), "count oversized raw rows");
            }
            // Reuse the existing scan/offset allocation for raw oversized rows.
            cub.exclusive_sum(pivot_counts.get(), pivot_offsets.get(), p);
            if (p) {
                normal_batch_pass<false><<<normal_blocks, kNormalBlock>>>(
                    pivots.get(), p, batch_descriptors.get(), 0,
                    device_input.get(), active.get(),
                    pivot_counts.get(), nullptr, batch_counts.get(), nullptr,
                    nullptr, nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr);
                cuda_check(cudaGetLastError(), "count normal batches");
            }
            copy_to_host(&report.normal_batch, batch_counts.get(), 1,
                         "download normal/oversized coverage and counts");
            sort_g = static_cast<std::size_t>(report.normal_batch.oversized_gathered);
            g = gpu_round_shadow_checked_add(sort_g,
                static_cast<std::size_t>(report.normal_batch.normal_gathered),
                "combined gathered count");
            if (g > input_count || report.normal_batch.normal_pivots +
                report.normal_batch.oversized_pivots != p ||
                report.normal_batch.batches > report.normal_batch.normal_pivots)
                throw std::runtime_error("GPU normal batches: invalid partition counts");
            // The existing D2H completes all descriptor writes. Every batch
            // consumes at least one normal pivot, so B<=p (already CUB-bounded).
            normal_batch_count = static_cast<std::size_t>(report.normal_batch.batches);
            second.allocate(sort_g, "allocate oversized gathered records");
            if (sort_g) {
                gather_oversized_raw<<<oversized_blocks, kBlock>>>(
                    pivots.get(), p, device_input.get(), active.get(),
                    pivot_offsets.get(), second.get());
                cuda_check(cudaGetLastError(), "gather oversized CSR rows");
            }
        } else {
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
                                            input_count, "select gathered incidences");
            require_count_if_expected(gathered, expected_shape,
                expected_shape ? expected_shape->gathered_incidences : 0, "gather");
            g = sort_g = static_cast<std::size_t>(gathered);
        }
        report.gathered_incidences = g;
    });
    first.release("release gather candidates");
    if (!device_owned_prefix) flags.release("release gather flags");

    auto& keys_a = sort_keys_a;
    auto& keys_b = sort_keys_b;
    // Keep the original g capacity bound: excess reduction still needs keys for
    // the full canonical stream, even if no oversized row exists this round.
    ensure_state_buffer(keys_a, g, "allocate radix keys A", "grow radix keys A");
    ensure_state_buffer(keys_b, g, "allocate radix keys B", "grow radix keys B");
    first.allocate(sort_g, "allocate sorted gathered records");
    device_buffer<work_record> unique(tracker);
    device_buffer<work_record> scratch(tracker);
    flags.allocate(sort_g, "allocate dedup flags");
    scratch.allocate(sort_g, "allocate dedup candidates");
    const std::size_t u_capacity = expected_shape
        ? static_cast<std::size_t>(expected_shape->unique_neighbors) : g;
    if (device_owned_prefix)
        oversized_unique.allocate(sort_g, "allocate oversized unique neighbors");
    else
        unique.allocate(std::max<std::size_t>(u_capacity, 1), "allocate unique neighbors");

    report.timings.dedup_ms = timer.measure_ms([&] {
        if (sort_g) {
            build_pair_keys<<<blocks_for(sort_g), kBlock>>>(second.get(), sort_g, keys_a.get());
            cuda_check(cudaGetLastError(), "launch gather-key construction");
            cub.sort_pairs(keys_a.get(), keys_b.get(), second.get(), first.get(),
                           sort_g, "stable sort gathered neighbors");
            reduce_equal_pairs<<<blocks_for(sort_g), kBlock>>>(
                first.get(), sort_g, scratch.get(), flags.get());
            cuda_check(cudaGetLastError(), "launch deterministic neighbor dedup");
        }
        auto* dedup_output = device_owned_prefix ? oversized_unique.get() : unique.get();
        const int unique_count = cub.select(scratch.get(), flags.get(), dedup_output,
                                            sort_g, "select unique neighbors");
        require_count_if_expected(unique_count, expected_shape,
            expected_shape ? expected_shape->unique_neighbors : 0, "dedup");
        if (device_owned_prefix) {
            oversized_u = static_cast<std::size_t>(unique_count);
            u = gpu_round_shadow_checked_add(oversized_u,
                static_cast<std::size_t>(report.normal_batch.normal_unique), "combined unique count");
            if (u > g) throw std::runtime_error("GPU normal batches: unique count exceeds gathered");
            if (oversized_u) {
                index_oversized_unique<<<blocks_for(oversized_u), kBlock>>>(
                    oversized_unique.get(), oversized_u, oversized_starts.get(), pivot_counts.get());
                cuda_check(cudaGetLastError(), "index oversized unique rows");
            }
        } else {
            u = static_cast<std::size_t>(unique_count);
            if (u) {
                count_unique_neighbors<<<blocks_for(u), kBlock>>>(unique.get(), u, pivot_counts.get());
                cuda_check(cudaGetLastError(), "launch per-pivot unique counts");
            }
        }
        report.unique_neighbors = report.factor_entries = u;
        cub.exclusive_sum(pivot_counts.get(), pivot_offsets.get(), p);
        set_last_offset<<<1, 1>>>(pivot_offsets.get(), p, static_cast<std::uint32_t>(u));
        cuda_check(cudaGetLastError(), "publish final pivot offset");
    });
    second.release("release gathered records");
    first.release("release sorted gathered records");
    scratch.release("release dedup candidates");
    if (!device_owned_prefix) flags.release("release dedup flags");
    if (device_owned_prefix) {
        unique.allocate(std::max<std::size_t>(u, 1), "allocate combined canonical neighbors");
        if (oversized_u) {
            scatter_oversized_unique<<<blocks_for(oversized_u), kBlock>>>(
                oversized_unique.get(), oversized_u, oversized_starts.get(), pivot_offsets.get(), unique.get());
            cuda_check(cudaGetLastError(), "scatter neighbor-ordered oversized rows");
        }
    }

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
            if (audit_payload) {
                prepare_factor<true><<<blocks_for(p), kBlock>>>(
                    unique.get(), pivot_offsets.get(), device_input.get(),
                    active.get(), pivots.get(), p,
                    excess.get(), total_degree.get(), pivot_counters.get(),
                    semantics.get(), factor_columns.get(), factor_column_base,
                    factor_entries.get(), factor_entry_base);
            } else if (device_owned_prefix) {
                prepare_factor<false, true><<<blocks_for(p), kBlock>>>(
                    unique.get(), pivot_offsets.get(), device_input.get(),
                    active.get(), pivots.get(), p,
                    excess.get(), total_degree.get(), nullptr,
                    semantics.get(), factor_columns.get(), factor_column_base,
                    factor_entries.get(), factor_entry_base);
            } else {
                prepare_factor<false><<<blocks_for(p), kBlock>>>(
                    unique.get(), pivot_offsets.get(), device_input.get(),
                    active.get(), pivots.get(), p,
                    excess.get(), total_degree.get(), nullptr,
                    semantics.get(), factor_columns.get(), factor_column_base,
                    factor_entries.get(), factor_entry_base);
            }
            cuda_check(cudaGetLastError(),
                       "launch factor-column append and digest");
        }
    });

    scratch.allocate(u, "allocate canonical-neighbor scratch");
    // unique <= gathered, so the existing key capacity already covers this
    // stage and its excess-update sorts without another allocation.
    device_buffer<double> prefix(tracker);
    device_buffer<work_record> fill_candidates(tracker);
    device_buffer<work_record> fills(tracker);
    prefix.allocate(u, "allocate GKS prefix sums");
    fill_candidates.allocate(u, "allocate raw fill candidates");
    flags.allocate(u, "allocate sample flags");
    if (sampler_components) {
        ensure_state_buffer(cycle_workspace, gpu_round_shadow_checked_mul(u, sampler_components,
            "cycle sampler workspace"), "allocate cycle sampler moments", "grow cycle sampler moments");
        ensure_state_buffer(cycle_status, 2, "allocate cycle sampler status", "grow cycle sampler status");
        cuda_check(cudaMemset(cycle_status.get(), 0, 2 * sizeof(std::uint32_t)),
                   "clear cycle sampler status");
    }
    const std::size_t f_capacity = expected_shape
        ? static_cast<std::size_t>(expected_shape->raw_fill_edges)
        : u;
    fills.allocate(std::max<std::size_t>(f_capacity, 1),
                   "allocate compact raw fill");

    report.timings.sample_ms = timer.measure_ms([&] {
        if (u) {
            const std::size_t canonical_sort_count = device_owned_prefix ? oversized_u : u;
            auto* sort_input = device_owned_prefix ? oversized_unique.get() : unique.get();
            if (canonical_sort_count) {
                // The oversized stream retains the exact three stable sorts;
                // normal batches use the same keys in warp-local merge sorts.
                build_weight_keys<<<blocks_for(canonical_sort_count), kBlock>>>(
                    sort_input, canonical_sort_count, keys_a.get());
                cuda_check(cudaGetLastError(), "launch weight-key construction");
                cub.sort_pairs(keys_a.get(), keys_b.get(), sort_input,
                               scratch.get(), canonical_sort_count,
                               "stable sort unique neighbors by weight");
                build_first_keys<<<blocks_for(canonical_sort_count), kBlock>>>(
                    scratch.get(), canonical_sort_count, keys_a.get());
                cuda_check(cudaGetLastError(), "launch pivot-key construction");
                cub.sort_pairs(keys_a.get(), keys_b.get(), scratch.get(),
                               sort_input, canonical_sort_count,
                               "stable regroup canonical neighbors by pivot");
                if (device_owned_prefix) {
                    scatter_oversized_unique<<<blocks_for(oversized_u), kBlock>>>(
                        oversized_unique.get(), oversized_u, oversized_starts.get(),
                        pivot_offsets.get(), unique.get());
                    cuda_check(cudaGetLastError(), "scatter canonical oversized rows");
                }
            }
            cuda_check(cudaMemset(flags.get(), 0, u), "clear fill candidate flags");
        }
        if (p) {
            if (sampler != clique_sampler::gks) {
                if (device_owned_prefix && normal_batch_count) {
                    const int emit_blocks = static_cast<int>(
                        (normal_batch_count + kNormalWarps - 1) / kNormalWarps);
                    normal_batch_pass<true, false><<<emit_blocks, kNormalBlock>>>(
                        pivots.get(), p, batch_descriptors.get(), normal_batch_count,
                        device_input.get(), active.get(),
                        pivot_counts.get(), pivot_offsets.get(), nullptr,
                        excess.get(), total_degree.get(), factor_columns.get(),
                        factor_column_base, factor_entries.get(), factor_entry_base,
                        unique.get(), fill_candidates.get(), flags.get());
                    cuda_check(cudaGetLastError(), "emit normal cycle-sampler factor batches");
                }
                sample_cycle_rows<<<blocks_for(p), kBlock>>>(
                    unique.get(), pivot_offsets.get(), pivots.get(), p,
                    total_degree.get(), prefix.get(), cycle_workspace.get(),
                    fill_candidates.get(), flags.get(),
                    audit_payload ? pivot_counters.get() : nullptr, sampler, cycle_status.get());
                cuda_check(cudaGetLastError(), "sample cycle-core fill on device");
                if (sampler == clique_sampler::trace_cycle &&
                    (!device_owned_prefix || report.normal_batch.oversized_pivots)) {
                    // Late rounds can have only a few oversized pivots. Spread
                    // their independent warps across SMs; mixed rounds retain
                    // larger CTAs so skipped normal rows stay inexpensive.
                    const bool all_oversized = device_owned_prefix &&
                        report.normal_batch.normal_pivots == 0;
                    const unsigned trace_block = all_oversized ? 32 : kBlock;
                    const auto trace_blocks = blocks_for(gpu_round_shadow_checked_mul(
                        p, 32, "cooperative trace row launch"), trace_block);
                    sample_large_trace_rows<<<trace_blocks, trace_block>>>(
                        unique.get(), pivot_offsets.get(), pivots.get(), p,
                        total_degree.get(), prefix.get(), cycle_workspace.get(),
                        fill_candidates.get(), flags.get(),
                        audit_payload ? pivot_counters.get() : nullptr, cycle_status.get());
                    cuda_check(cudaGetLastError(), "sample large trace rows cooperatively");
                }
                std::uint32_t status[2]{};
                copy_to_host(status, cycle_status.get(), 2, "read cycle sampler status");
                if (status[0])
                    throw std::domain_error("GPU cycle sampler: invalid plan or edge outside normal pool range");
                report.sampler_numerical_fallbacks = status[1];
                if (gpu_setup_diagnostics()) std::fprintf(stderr,
                    "[gpu-clique-sampler] sampler=%s pivots=%zu normal_pivots=%llu "
                    "oversized_pivots=%llu numerical_gks_fallbacks=%u device_sampling=1\n",
                    sampler == clique_sampler::trace_cycle ? "trace_cycle" : "heavy_core_k2", p,
                    static_cast<unsigned long long>(report.normal_batch.normal_pivots),
                    static_cast<unsigned long long>(report.normal_batch.oversized_pivots), status[1]);
            } else if (audit_payload) {
                sample_gks_tree<<<blocks_for(p), kBlock>>>(
                    unique.get(), pivot_offsets.get(), pivots.get(), p,
                    total_degree.get(), prefix.get(), fill_candidates.get(),
                    flags.get(), pivot_counters.get());
                cuda_check(cudaGetLastError(), "launch all-degree GKS tree sampler");
            } else if (device_owned_prefix) {
                if (normal_batch_count) {
                    const int emit_blocks = static_cast<int>(
                        (normal_batch_count + kNormalWarps - 1) / kNormalWarps);
                    normal_batch_pass<true><<<emit_blocks, kNormalBlock>>>(
                        pivots.get(), p, batch_descriptors.get(), normal_batch_count,
                        device_input.get(), active.get(),
                        pivot_counts.get(), pivot_offsets.get(), nullptr,
                        excess.get(), total_degree.get(), factor_columns.get(),
                        factor_column_base, factor_entries.get(), factor_entry_base,
                        unique.get(), fill_candidates.get(), flags.get());
                    cuda_check(cudaGetLastError(), "emit normal factor/fill batches");
                }
                if (oversized_u) {
                    prepare_gks_prefix<true><<<blocks_for(p), kBlock>>>(
                        unique.get(), pivot_offsets.get(), p, prefix.get(), pivots.get());
                    cuda_check(cudaGetLastError(), "launch oversized ordered prefixes");
                    sample_gks_items<true><<<blocks_for(u), kBlock>>>(
                        unique.get(), u, pivot_offsets.get(), pivots.get(),
                        total_degree.get(), prefix.get(), fill_candidates.get(), flags.get());
                    cuda_check(cudaGetLastError(), "launch oversized GKS emission");
                }
            } else {
                prepare_gks_prefix<false><<<blocks_for(p), kBlock>>>(
                    unique.get(), pivot_offsets.get(), p, prefix.get(), pivots.get());
                cuda_check(cudaGetLastError(), "launch ordered GKS prefixes");
                if (u) {
                    sample_gks_items<false><<<blocks_for(u), kBlock>>>(
                        unique.get(), u, pivot_offsets.get(), pivots.get(),
                        total_degree.get(), prefix.get(), fill_candidates.get(),
                        flags.get());
                    cuda_check(cudaGetLastError(), "launch parallel GKS emission");
                }
            }
        }
        oversized_unique.release("release oversized canonical neighbors");
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
        // Raw fill was already compacted into the separate fills buffer.
        // Reuse its dead u-slot candidate storage for e<=u excess updates;
        // audited/reference callers retain their original local allocations.
        auto* compact_excess_data = fill_candidates.get();
        if (!device_owned_prefix) {
            compact_excess.allocate(std::max<std::size_t>(e_capacity, 1),
                                    "allocate compact excess updates");
            compact_excess_data = compact_excess.get();
        }
        const int excess_count = cub.select(
            scratch.get(), flags.get(), compact_excess_data, u,
            "compact excess updates");
        require_count_if_expected(
            excess_count, expected_shape,
            expected_shape ? expected_shape->excess_updates : 0,
            "excess update construction");
        e = static_cast<std::size_t>(excess_count);
        report.excess_updates = e;
        if (e) {
            build_first_keys<<<blocks_for(e), kBlock>>>(
                compact_excess_data, e, keys_a.get());
            cuda_check(cudaGetLastError(), "launch excess target keys");
            cub.sort_pairs(keys_a.get(), keys_b.get(), compact_excess_data,
                           unique.get(), e, "stable sort excess updates");
            reduce_equal_pairs<<<blocks_for(e), kBlock>>>(
                unique.get(), e, scratch.get(), flags.get());
            cuda_check(cudaGetLastError(), "launch deterministic excess reduce");
        }
        device_buffer<work_record> reduced_excess(tracker);
        const std::size_t q_capacity = expected_shape
            ? static_cast<std::size_t>(expected_shape->excess_targets)
            : e;
        // The sorted input now lives in unique, and its reduction in scratch.
        // The same dead candidate storage fits q<=e<=u without in/out aliasing.
        auto* reduced_excess_data = fill_candidates.get();
        if (!device_owned_prefix) {
            reduced_excess.allocate(std::max<std::size_t>(q_capacity, 1),
                                    "allocate reduced excess updates");
            reduced_excess_data = reduced_excess.get();
        }
        const int reduced_count = cub.select(
            scratch.get(), flags.get(), reduced_excess_data, e,
            "compact reduced excess updates");
        require_count_if_expected(
            reduced_count, expected_shape,
            expected_shape ? expected_shape->excess_targets : 0,
            "excess reduction");
        q = static_cast<std::size_t>(reduced_count);
        report.excess_targets = q;
        if (q) {
            apply_excess_updates<<<blocks_for(q), kBlock>>>(
                reduced_excess_data, q, excess.get());
            cuda_check(cudaGetLastError(), "launch reduced excess apply");
        }
    });
    prefix.release("release GKS prefix sums");
    fill_candidates.release("release raw fill candidates");
    unique.release("release canonical neighbors");
    scratch.release("release canonical scratch");
    if (!device_owned_prefix) flags.release("release canonical flags");

    if (audit_payload && f) {
        report.timings.fill_materialize_ms = timer.measure_ms([&] {
            hash_raw_fill<<<blocks_for(f), kBlock>>>(
                fills.get(), f, semantics.get());
            cuda_check(cudaGetLastError(), "launch raw multigraph-fill digest");
        });
    }

    device_buffer<gpu_round_shadow_incidence> residual(tracker);
    device_buffer<std::uint32_t> local_degrees(tracker);
    auto& degrees = device_owned_prefix ? owned_degrees : local_degrees;
    flags.allocate(input_count, "allocate surviving-incidence flags");
    const std::size_t residual_capacity = expected_shape
        ? static_cast<std::size_t>(expected_shape->live_incidences)
        : gpu_round_shadow_checked_add(
              input_count,
              gpu_round_shadow_checked_mul(f, 2, "discovered fill output"),
              "discovered residual output");
    residual.allocate(std::max<std::size_t>(residual_capacity, 1),
                      "allocate materialized shadow residual");
    if (device_owned_prefix)
        ensure_state_buffer(degrees, n, "allocate owned live degrees",
                            "grow owned live degrees");
    else
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
        const std::size_t directed_fill = gpu_round_shadow_checked_mul(
            f, 2, "directed fill scratch");
        // Paired incidences + independent pivots remove at least twice the
        // gathered degree; supported tree/cycle fill never exceeds that degree.
        if (s > input_count || directed_fill > input_count - s ||
            directed_fill > device_input.count())
            throw std::logic_error("GPU round shadow: fill exceeds retired input scratch");
        if (f) {
            append_raw_fill<<<blocks_for(f), kBlock>>>(
                fills.get(), f, residual.get(), s);
            cuda_check(cudaGetLastError(), "launch raw multigraph-fill application");
        }
    });
    if (!device_owned_prefix) flags.release("release surviving-incidence flags");
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

    const std::size_t directed_fill = gpu_round_shadow_checked_mul(
        f, 2, "directed fill owner sort");
    ensure_state_buffer(keys_a, directed_fill, "allocate radix keys A", "grow radix keys A");
    ensure_state_buffer(keys_b, directed_fill, "allocate radix keys B", "grow radix keys B");

    report.timings.checksums_ms = timer.measure_ms([&] {
        if (n)
            cuda_check(cudaMemset(next_owner_offsets.get(), 0,
                                  n * sizeof(std::uint32_t)),
                       "clear owner boundary ends");
        if (directed_fill) {
            build_owner_keys<<<blocks_for(directed_fill), kBlock>>>(
                residual.get() + s, directed_fill, keys_a.get());
            cuda_check(cudaGetLastError(), "launch new-fill owner keys");
            // Stable survivor selection has completed its last read of input.
            // Reuse that dead buffer for sorted fill, never a new full-L array.
            cub.sort_pairs(keys_a.get(), keys_b.get(), residual.get() + s,
                           device_input.get(), directed_fill,
                           "stable sort new fill by owner", owner_sort_end_bit(n));
        }
        if (l) {
            if (audit_payload)
                merge_survivors_and_fill<false><<<blocks_for(l), kBlock>>>(
                    residual.get(), s, device_input.get(), directed_fill,
                    next_residual.get(), semantics.get());
            else
                merge_survivors_and_fill<true><<<blocks_for(l), kBlock>>>(
                    residual.get(), s, device_input.get(), directed_fill,
                    next_residual.get(), semantics.get());
            cuda_check(cudaGetLastError(), "launch stable survivor-fill merge");
            residual_owner_ends<<<blocks_for(l), kBlock>>>(
                next_residual.get(), l, n, next_owner_offsets.get());
            cuda_check(cudaGetLastError(), "launch unique owner boundary ends");
        }
        if (n) {
            cub.inclusive_owner_max(next_owner_offsets.get(), n);
            set_last_offset<<<1, 1>>>(
                next_owner_offsets.get(), n, static_cast<std::uint32_t>(l));
            cuda_check(cudaGetLastError(), "publish final owner offset");
            residual_degrees_from_offsets<<<blocks_for(n), kBlock>>>(
                next_owner_offsets.get(), n, degrees.get());
            cuda_check(cudaGetLastError(), "derive exact residual degrees");
        }
        if (l) {
            if (audit_payload) {
                hash_residual_incidences<true><<<blocks_for(l), kBlock>>>(
                    next_residual.get(), l, semantics.get());
                cuda_check(cudaGetLastError(), "launch residual endpoint/weight digest");
                hash_ordered_residual_incidences<<<blocks_for(l), kBlock>>>(
                    next_residual.get(), l, next_owner_offsets.get(), semantics.get());
                cuda_check(cudaGetLastError(), "launch ordered residual digest");
            }
        }
        if (n) {
            if (audit_payload)
                hash_residual_state<true><<<blocks_for(n), kBlock>>>(
                    n, active.get(), degrees.get(), excess.get(), semantics.get());
            else
                hash_residual_state<false><<<blocks_for(n), kBlock>>>(
                    n, active.get(), nullptr, nullptr, semantics.get());
            cuda_check(cudaGetLastError(), "launch residual checksums");
        }
    });

    device_semantics host_semantics{};
    report.timings.download_ms = timer.measure_ms([&] {
        copy_to_host(&host_semantics, semantics.get(), 1,
                     "download semantic report");
        if (audit_payload)
            copy_to_host(report.pivots.data(), pivot_counters.get(), p,
                         "download pivot counters");
    });

    if (total_diagnostics) {
        cuda_check(cudaEventRecord(total_end.get()), "record total end event");
        cuda_check(cudaEventSynchronize(total_end.get()),
                   "synchronize total end event");
        float total_ms = 0.0f;
        cuda_check(cudaEventElapsedTime(
                       &total_ms, total_begin.get(), total_end.get()),
                   "read total event duration");
        report.timings.total_ms = total_ms;
    } else {
        // The semantic-report cudaMemcpy above synchronously completes all
        // preceding round work before host validation and buffer retirement.
        report.timings.total_ms = std::numeric_limits<double>::quiet_NaN();
    }

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

    // The synchronous semantic download completed every degree consumer.
    // Retire scratch on the last round before factor installation; no selector
    // borrows degrees. A failed release is still covered by the generation
    // guard and the existing device-aware state teardown.
    if (device_owned_prefix && report.active_count == 0 && l == 0) {
        owned_degrees.release("release completed owned degrees");
        owned_common_scratch.release();
    }

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
    generation_has_device_selection = device_selection != nullptr;
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
    if (device_owned_prefix) owned_residual_epoch = std::make_shared<const int>(0);
    // Survivor filtering, active-neighbor fill and the stable merge above
    // published exact compact rows. No subsequent mutation may inherit this
    // certificate without preserving both the offsets and active endpoints.
    resident_rows_all_live = device_owned_prefix;
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
    resident_rows_all_live = false;
    excess_may_differ = false;
    next_reimport = reimport_reason::authoritative_host_rebuild;
    ++host_rebuild_invalidations;
}

gpu_round_shadow_device_state::gpu_round_shadow_device_state(clique_sampler sampler)
    : impl_(std::make_unique<impl>(sampler)) {
    if (sampler != clique_sampler::gks && sampler != clique_sampler::trace_cycle &&
        sampler != clique_sampler::heavy_core_k2)
        throw std::invalid_argument("unsupported GPU clique sampler");
}

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

gpu_round_shadow_report gpu_round_shadow_device_state::compute_owned_prefix(
        const gpu_round_shadow_input* initial, gpu_device_selection selected,
        std::uint64_t seed, const gpu_device_selection_content* paired_initial_content) {
    impl::failed_generation_guard guard{*impl_};
    guard.arm();
    if (impl_->poisoned || impl_->owned_prefix_materialized ||
        (initial ? impl_->has_state : !impl_->device_owned_prefix))
        throw std::logic_error("GPU-owned prefix generation is out of order");
    impl_->device_owned_prefix = true;
    const gpu_round_shadow_input empty;
    auto report = impl_->compute(initial ? *initial : empty, nullptr, &selected, seed, !initial,
        paired_initial_content);
    // Owned acceptance is intentionally not CPU certification. The current
    // producer/generation and the round's CUDA validation remain mandatory.
    if (!impl_->generation_has_device_selection || !report.gpu_executed ||
        impl_->generation != report.output_generation)
        throw std::logic_error("GPU-owned prefix did not publish its selected generation");
    impl_->generation_accepted = true;
    guard.commit();
    return report;
}

std::unique_ptr<gpu_block_frontend> gpu_round_shadow_device_state::initialize_owned_csc(
        const gpu_owned_csc_buffers& input, double reg_eps, bool& sddm,
        std::size_t& initial_edges) {
    // Private constructor path: dispatch (or the test wrapper) has checked
    // gpu_owned_csc_supported once before entering the owning session.
    // Repeating its full stored-pattern traversal here would undo the import
    // simplification; the device value/status checks below remain mandatory.
    impl::failed_generation_guard guard{*impl_}; guard.arm();
    if (impl_->has_state || impl_->poisoned || impl_->factor_column_count)
        throw std::logic_error("GPU direct CSC initialization requires a fresh numerical owner");
    const auto start = gpu_setup_diagnostic_clock::now();
    const std::size_t n = static_cast<std::size_t>(input.vertex_count);
    const std::size_t nnz = input.nonzeros;
    cuda_check(cudaGetDevice(&impl_->cuda_device), "bind CSC initialization device");
    const auto selector_probe = gpu_block_frontend::probe_runtime(static_cast<node_index>(n), 0);
    if (!selector_probe.cooperative_launch)
        throw std::runtime_error("GPU direct CSC initialization needs cooperative selection");
    std::size_t cub_bytes = 0;
    cuda_check(cub::DeviceScan::ExclusiveSum(nullptr, cub_bytes,
        fake_pointer<std::uint32_t>(0x1700), fake_pointer<std::uint32_t>(0x1900),
        cub_count(n + 1, "CSC owner scan size")), "query CSC owner scan scratch");
    std::size_t planned = selector_probe.estimated_bytes;
    const std::size_t upload_bytes = add_bytes(
        gpu_round_shadow_checked_mul(n + 1, sizeof(int), "CSC pointers"),
        nnz, sizeof(int) + sizeof(double), "CSC values and rows");
    planned = gpu_round_shadow_checked_add(planned, upload_bytes, "CSC upload storage");
    planned = add_bytes(planned, std::max<std::size_t>(nnz, 1),
                         sizeof(gpu_round_shadow_incidence), "initial weighted residual");
    planned = add_bytes(planned, n + 1, 2 * sizeof(std::uint32_t), "CSC counts and offsets");
    planned = add_bytes(planned, n, sizeof(std::uint8_t) + sizeof(double), "initial active and excess");
    planned = add_bytes(planned, 1, sizeof(device_semantics) + 2 * sizeof(std::uint32_t), "CSC status");
    planned = gpu_round_shadow_checked_add(planned, cub_bytes, "CSC scan scratch");
    const std::size_t margin = std::max(selector_probe.total_bytes / 10, std::size_t{256} << 20);
    if (selector_probe.free_bytes <= margin || planned > selector_probe.free_bytes - margin)
        throw std::runtime_error("GPU direct CSC initialization exceeds free memory with margin");
    auto& t = impl_->tracker; t.begin_round();
    device_buffer<int> ptr(t), idx(t);
    device_buffer<double> values(t);
    device_buffer<std::uint32_t> counts(t), status(t);
    device_buffer<std::byte> scratch(t);
    device_buffer<device_semantics> semantics(t);
    ptr.allocate(n + 1, "allocate CSC pointers"); idx.allocate(nnz, "allocate CSC rows");
    values.allocate(nnz, "allocate CSC values"); counts.allocate(n + 1, "allocate CSC row counts");
    status.allocate(2, "allocate CSC status"); scratch.allocate(cub_bytes, "allocate CSC scan scratch");
    semantics.allocate(1, "allocate CSC content digest");
    impl_->ensure_state_buffer(impl_->owner_offsets_0, n + 1,
        "allocate initial CSC offsets", "grow initial CSC offsets");
    impl_->ensure_state_buffer(impl_->active, n, "allocate initial CSC active", "grow initial CSC active");
    impl_->ensure_state_buffer(impl_->excess, n, "allocate initial CSC excess", "grow initial CSC excess");
    event_timer timer;
    const float upload_ms = timer.measure_ms([&] {
        copy_to_device(ptr.get(), input.offsets, n + 1, "upload CSC pointers");
        copy_to_device(idx.get(), input.rows, nnz, "upload CSC rows");
        copy_to_device(values.get(), input.values, nnz, "upload CSC values");
        cuda_check(cudaMemset(status.get(), 0, 2 * sizeof(std::uint32_t)), "clear CSC status");
        cuda_check(cudaMemset(counts.get(), 0, (n + 1) * sizeof(std::uint32_t)), "clear CSC counts");
        cuda_check(cudaMemset(semantics.get(), 0, sizeof(device_semantics)), "clear CSC content digest");
    });
    const float count_ms = timer.measure_ms([&] {
        csc_initial_counts_excess<<<blocks_for(n), kBlock>>>(static_cast<int>(n), ptr.get(), idx.get(),
            values.get(), reg_eps, counts.get(), impl_->active.get(), impl_->excess.get(), status.get());
        cuda_check(cudaGetLastError(), "initialize CSC counts and exact excess");
        std::size_t bytes = cub_bytes;
        cuda_check(cub::DeviceScan::ExclusiveSum(scratch.get(), bytes, counts.get(),
            impl_->owner_offsets_0.get(), cub_count(n + 1, "CSC owner scan")), "scan CSC owner offsets");
    });
    std::uint32_t directed = 0;
    copy_to_host(&directed, impl_->owner_offsets_0.get() + n, 1, "download initial incidence count");
    if (directed > nnz || directed % 2)
        throw std::logic_error("GPU direct CSC incidence count is invalid");
    impl_->ensure_state_buffer(impl_->residual_0, std::max<std::size_t>(directed, 1),
        "allocate initial CSC residual", "grow initial CSC residual");
    const float fill_ms = timer.measure_ms([&] {
        csc_initial_incidences<<<blocks_for(n), kBlock>>>(static_cast<int>(n), ptr.get(), idx.get(),
            values.get(), impl_->owner_offsets_0.get(), impl_->residual_0.get(), status.get());
        cuda_check(cudaGetLastError(), "materialize exact CSC incidences");
        if (directed) {
            hash_residual_incidences<false><<<blocks_for(directed), kBlock>>>(
                impl_->residual_0.get(), directed, semantics.get());
            cuda_check(cudaGetLastError(), "hash initial CSC topology");
        }
        hash_residual_state<false><<<blocks_for(n), kBlock>>>(n, impl_->active.get(),
            nullptr, nullptr, semantics.get());
        cuda_check(cudaGetLastError(), "hash initial CSC active vertices");
    });
    std::uint32_t host_status[2]{}; device_semantics content{};
    copy_to_host(host_status, status.get(), 2, "download CSC validation status");
    copy_to_host(&content, semantics.get(), 1, "download initial CSC content digest");
    if (host_status[0] || content.active_count != n)
        throw std::invalid_argument("GPU direct CSC initialization rejected values or layout");
    if (t.peak > planned)
        throw std::logic_error("GPU CSC owner allocation exceeded initialization plan");
    // CSC is scratch only; release it before the selector and numerical rounds.
    ptr.release("release initial CSC pointers"); idx.release("release initial CSC rows");
    values.release("release initial CSC values"); counts.release("release initial CSC counts");
    scratch.release("release initial CSC scan scratch"); status.release("release initial CSC status");
    semantics.release("release initial CSC content digest");
    impl_->vertex_count = static_cast<node_index>(n);
    impl_->resident_count = directed; impl_->current_slot = 0;
    impl_->has_state = true; impl_->device_owned_prefix = true;
    impl_->generation_accepted = true; impl_->csc_initial_round_pending = true;
    impl_->state_imports = 1; impl_->state_upload_bytes = upload_bytes;
    impl_->resident_selection_content = {static_cast<node_index>(n), n,
        {content.topology.xor_hash, content.topology.sum_hash},
        {content.active.xor_hash, content.active.sum_hash}, {}};
    impl_->fingerprint.topology = {content.topology.xor_hash, content.topology.sum_hash};
    impl_->fingerprint.active = {content.active.xor_hash, content.active.sum_hash};
    impl_->fingerprint.active_count = n; impl_->fingerprint.live_incidences = directed;
    impl_->owned_residual_epoch = std::make_shared<const int>(0);
    const auto selector_start = gpu_setup_diagnostic_clock::now();
    auto frontend = std::unique_ptr<gpu_block_frontend>(new gpu_block_frontend(
        static_cast<node_index>(n), directed / 2, gpu_block_frontend::owned_initialization_tag{}));
    frontend->bind_owned_residual(impl_->residual_0.get(), impl_->owner_offsets_0.get(),
        impl_->active.get(), directed, impl_->resident_selection_content, impl_->owned_residual_epoch);
    const double selector_ms = 1e3 * std::chrono::duration<double>(
        gpu_setup_diagnostic_clock::now() - selector_start).count();
    const double total_ms = 1e3 * std::chrono::duration<double>(
        gpu_setup_diagnostic_clock::now() - start).count();
    sddm = host_status[1] != 0; initial_edges = directed / 2;
    if (gpu_setup_diagnostics()) std::fprintf(stderr, "[gpu-owned-csc] n=%zu directed=%u input_nnz=%zu sddm=%d "
        "host_graph_bytes=0 host_snapshot_bytes=0 selector_edge_upload_bytes=0 "
        "csc_upload_bytes=%zu scalar_download_bytes=%zu owner_tracked_peak_bytes=%zu "
        "planned_total_peak_bytes=%zu upload_ms=%.6f count_excess_ms=%.6f "
        "fill_hash_ms=%.6f selector_bind_ms=%.6f total_ms=%.6f\n",
        n, directed, nnz, sddm ? 1 : 0, upload_bytes,
        sizeof(directed) + sizeof(host_status) + sizeof(content), t.peak, planned,
        double(upload_ms), double(count_ms), double(fill_ms), selector_ms, total_ms);
    guard.commit(); return frontend;
}

#if defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
gpu_owned_csc_test_result gpu_initial_owned_csc_buffers_for_test(
        const gpu_owned_csc_buffers& input, double reg_eps) {
    gpu_owned_csc_test_result result;
    gpu_round_shadow_device_state state;
    std::size_t edges = 0;
    auto frontend = state.initialize_owned_csc(input, reg_eps, result.sddm, edges);
    result.initial = state.download_owned_prefix().residual;
    result.eligible = true;
    return result;
}
#endif

gpu_owned_prefix_handback gpu_round_shadow_device_state::download_owned_prefix() {
    impl::failed_generation_guard guard{*impl_};
    guard.arm();
    impl_->require_current_device("owned prefix handback");
    if (impl_->poisoned || !impl_->device_owned_prefix || impl_->owned_prefix_materialized ||
        !impl_->has_state || !impl_->generation_accepted)
        throw std::logic_error("GPU-owned prefix handback requires its current accepted generation");
    impl_->owned_common_scratch.release(); // Previous round completed synchronously.
    gpu_owned_prefix_handback result;
    auto& input = result.residual;
    input.vertex_count = impl_->vertex_count;
    const std::size_t n = input.vertex_count;
    std::vector<std::uint32_t> offsets(n + 1);
    input.incidences.resize(impl_->resident_count);
    input.active.resize(n);
    input.excess.resize(n);
    result.columns.resize(impl_->factor_column_count);
    copy_to_host(offsets.data(), impl_->owner_offsets(impl_->current_slot).get(), n + 1,
                 "download owned residual offsets");
    input.owner_offsets.assign(offsets.begin(), offsets.end());
    copy_to_host(input.incidences.data(), impl_->residual(impl_->current_slot).get(),
                 input.incidences.size(), "download owned ordered residual");
    copy_to_host(input.active.data(), impl_->active.get(), n, "download owned active mask");
    copy_to_host(input.excess.data(), impl_->excess.get(), n, "download owned excess");
    copy_to_host(result.columns.data(), impl_->factor_columns.get(), result.columns.size(),
                 "download owned factor headers");
    // Full weighted audit is needed only at this explicit host handback, not
    // after every owned round. Compute it independently on the current device
    // arrays, then check the downloaded representation against it below.
    allocation_tracker audit_tracker;
    device_buffer<device_semantics> audit(audit_tracker);
    device_buffer<std::uint32_t> degrees(audit_tracker);
    audit.allocate(1, "allocate handback audit");
    degrees.allocate(n, "allocate handback degrees");
    cuda_check(cudaMemset(audit.get(), 0, sizeof(device_semantics)), "clear handback audit");
    if (n) cuda_check(cudaMemset(degrees.get(), 0, n * sizeof(std::uint32_t)),
                      "clear handback degrees");
    if (impl_->resident_count) {
        const auto* residual = impl_->residual(impl_->current_slot).get();
        count_residual_degrees<<<blocks_for(impl_->resident_count), kBlock>>>(
            residual, impl_->resident_count, degrees.get());
        cuda_check(cudaGetLastError(), "launch handback degree count");
        hash_residual_incidences<true><<<blocks_for(impl_->resident_count), kBlock>>>(
            residual, impl_->resident_count, audit.get());
        cuda_check(cudaGetLastError(), "launch handback weighted digest");
        hash_ordered_residual_incidences<<<blocks_for(impl_->resident_count), kBlock>>>(
            residual, impl_->resident_count,
            impl_->owner_offsets(impl_->current_slot).get(), audit.get());
        cuda_check(cudaGetLastError(), "launch handback ordered digest");
    }
    if (n) {
        hash_residual_state<true><<<blocks_for(n), kBlock>>>(
            n, impl_->active.get(), degrees.get(), impl_->excess.get(), audit.get());
        cuda_check(cudaGetLastError(), "launch handback state digest");
    }
    device_semantics checked{};
    copy_to_host(&checked, audit.get(), 1, "download handback audit");
    result.fingerprint = {
        {checked.residual.xor_hash, checked.residual.sum_hash},
        {checked.ordered_residual.xor_hash, checked.ordered_residual.sum_hash},
        {checked.active.xor_hash, checked.active.sum_hash},
        {checked.topology.xor_hash, checked.topology.sum_hash},
        {checked.live_degree.xor_hash, checked.live_degree.sum_hash},
        {checked.canonical_excess.xor_hash, checked.canonical_excess.sum_hash},
        checked.active_count, impl_->resident_count};
    if (result.fingerprint.active != impl_->fingerprint.active ||
        result.fingerprint.topology != impl_->fingerprint.topology ||
        result.fingerprint.active_count != impl_->fingerprint.active_count)
        throw std::logic_error("GPU-owned handback changed topology/active continuity");
    validate_owned_prefix_handback(result, impl_->factor_entry_count);
    result.download_bytes = offsets.size() * sizeof(offsets[0]) +
        input.incidences.size() * sizeof(input.incidences[0]) +
        input.active.size() + input.excess.size() * sizeof(double) +
        result.columns.size() * sizeof(result.columns[0]) + sizeof(checked);
    impl_->owned_residual_epoch.reset();
    impl_->owned_prefix_materialized = true;
    guard.commit();
    return result;
}

std::vector<gpu_round_shadow_factor_column>
gpu_round_shadow_device_state::download_owned_factor_columns(
        std::span<node_index> permutation, std::span<edge_index> raw_offsets) {
    impl::failed_generation_guard guard{*impl_};
    guard.arm();
    impl_->require_current_device("owned factorization completion");
    if (impl_->poisoned || !impl_->device_owned_prefix || impl_->owned_prefix_materialized ||
        !impl_->has_state || !impl_->generation_accepted ||
        impl_->fingerprint.active_count != 0 || impl_->resident_count != 0 ||
        impl_->factor_column_count != impl_->vertex_count)
        throw std::logic_error("GPU-owned completion requires every vertex eliminated on device");
    std::vector<gpu_round_shadow_factor_column> columns(impl_->factor_column_count);
    copy_to_host(columns.data(), impl_->factor_columns.get(), columns.size(),
                 "download complete factor headers");
    validate_owned_factor_columns(columns, impl_->factor_entry_count,
                                 permutation, raw_offsets);
    // Seal output only after complete coverage; no CPU residual is constructed.
    impl_->owned_residual_epoch.reset();
    impl_->owned_prefix_materialized = true;
    guard.commit();
    return columns;
}

double gpu_round_shadow_device_state::probe_owned_distinct_degree(
        std::span<const node_index> active_ids, gpu_block_frontend& frontend) {
    impl::failed_generation_guard guard{*impl_};
    guard.arm();
    impl_->validate_owned_preview(frontend);
    if (active_ids.size() != impl_->resident_selection_content.active_count ||
        !std::is_sorted(active_ids.begin(), active_ids.end()) ||
        std::adjacent_find(active_ids.begin(), active_ids.end()) != active_ids.end() ||
        (!active_ids.empty() && active_ids.back() >= impl_->vertex_count))
        throw std::invalid_argument("GPU residual probe active IDs are invalid");
    const std::size_t samples = std::min<std::size_t>(256, active_ids.size());
    if (!samples) { guard.commit(); return 0.0; }
    std::vector<node_index> vertices(samples);
    for (std::size_t i = 0; i < samples; ++i)
        vertices[i] = active_ids[i * active_ids.size() / samples];
    std::vector<owned_probe_range> ranges(samples);
    std::vector<std::uint32_t> distinct(samples);
    device_buffer<node_index> ids(impl_->tracker);
    device_buffer<owned_probe_range> device_ranges(impl_->tracker);
    device_buffer<std::uint32_t> degrees(impl_->tracker), invalid(impl_->tracker);
    owned_probe_check_fit(samples * (sizeof(node_index) + sizeof(owned_probe_range) +
                                    sizeof(std::uint32_t)) + sizeof(std::uint32_t));
    ids.allocate(samples, "allocate sampled active IDs");
    device_ranges.allocate(samples, "allocate sampled row ranges");
    degrees.allocate(samples, "allocate sampled distinct degrees");
    invalid.allocate(1, "allocate sampled active validation");
    copy_to_device(ids.get(), vertices.data(), samples, "upload sampled active IDs");
    cuda_check(cudaMemset(invalid.get(), 0, sizeof(std::uint32_t)), "clear sample validation");
    owned_probe_ranges<<<blocks_for(samples), kBlock>>>(ids.get(), samples,
        impl_->owner_offsets(impl_->current_slot).get(), impl_->active.get(),
        device_ranges.get(), invalid.get());
    cuda_check(cudaGetLastError(), "read sampled row ranges");
    copy_to_host(ranges.data(), device_ranges.get(), samples, "download sampled row ranges");
    std::uint32_t bad = 0;
    copy_to_host(&bad, invalid.get(), 1, "download sampled active validation");
    if (bad) throw std::invalid_argument("GPU residual probe sampled an inactive vertex");
    std::size_t count = 0;
    for (auto& range : ranges) {
        if (range.end < range.begin || range.end > impl_->resident_count)
            throw std::logic_error("GPU residual probe has invalid offsets");
        range.output = static_cast<std::uint32_t>(count);
        count = gpu_round_shadow_checked_add(count, range.end - range.begin, "sampled incidences");
    }
    cub_count(count, "sampled incidences");
    copy_to_device(device_ranges.get(), ranges.data(), samples, "upload sampled compact offsets");
    cuda_check(cudaMemset(degrees.get(), 0, samples * sizeof(std::uint32_t)), "clear sampled degrees");
    if (count) {
        std::size_t bytes = 0;
        cuda_check(cub::DeviceRadixSort::SortKeys(nullptr, bytes,
            static_cast<const std::uint64_t*>(nullptr), static_cast<std::uint64_t*>(nullptr),
            static_cast<int>(count)), "query sampled key sort");
        owned_probe_check_fit(add_bytes(bytes, count, 2 * sizeof(std::uint64_t), "sampled sort scratch"));
        device_buffer<std::uint64_t> keys(impl_->tracker), sorted(impl_->tracker);
        device_buffer<std::byte> temporary(impl_->tracker);
        keys.allocate(count, "allocate sampled neighbor keys");
        sorted.allocate(count, "allocate sorted sampled neighbor keys");
        temporary.allocate(bytes, "allocate sampled sort scratch");
        owned_probe_keys<<<blocks_for(samples * 32), kBlock>>>(device_ranges.get(), samples,
            impl_->residual(impl_->current_slot).get(), keys.get());
        cuda_check(cudaGetLastError(), "gather sampled neighbor keys");
        cuda_check(cub::DeviceRadixSort::SortKeys(temporary.get(), bytes, keys.get(), sorted.get(),
            static_cast<int>(count)), "sort sampled neighbor keys");
        owned_probe_distinct<<<blocks_for(count), kBlock>>>(sorted.get(), count, degrees.get());
        cuda_check(cudaGetLastError(), "count sampled distinct neighbors");
        copy_to_host(distinct.data(), degrees.get(), samples, "download sampled distinct degrees");
    }
    std::size_t sum = 0;
    for (auto degree : distinct) sum += degree;
    if (gpu_setup_diagnostics())
        std::fprintf(stderr, "[gpu-owned-sparsify-probe] samples=%zu physical_incidences=%zu "
            "scalar_upload_bytes=%zu scalar_download_bytes=%zu peak_device_bytes=%zu\n",
            samples, count, samples * (sizeof(node_index) + sizeof(owned_probe_range)),
            samples * sizeof(owned_probe_range) + sizeof(bad) +
                (count ? samples * sizeof(std::uint32_t) : 0), impl_->tracker.peak);
    guard.commit();
    return static_cast<double>(sum) / samples;
}

gpu_owned_sparsify_stats gpu_round_shadow_device_state::sparsify_owned_residual(
        gpu_block_frontend& frontend, std::uint64_t seed) {
    impl::failed_generation_guard guard{*impl_};
    guard.arm();
    impl_->validate_owned_preview(frontend);
    const std::size_t n = impl_->vertex_count, live = impl_->resident_count;
    const auto spare = 1 - impl_->current_slot;
    cub_count(n, "sparsification vertices");
    cub_count(live, "sparsification incidences");
    // Only dead numerical scratch is released; selector scratch belongs to
    // its producer. No borrower may observe replacement of the resident rows.
    impl_->owned_residual_epoch.reset();
    impl_->resident_rows_all_live = false;
    impl_->owned_common_scratch.release();
    std::size_t extra = sizeof(device_semantics);
    if (impl_->residual(spare).count() < std::max<std::size_t>(live, 1))
        extra = add_bytes(extra, std::max<std::size_t>(live, 1),
                          sizeof(gpu_round_shadow_incidence), "sparsification spare residual");
    if (impl_->owner_offsets(spare).count() < n + 1)
        extra = add_bytes(extra, n + 1, sizeof(std::uint32_t), "sparsification spare offsets");
    if (impl_->owned_degrees.count() < n)
        extra = add_bytes(extra, n, sizeof(std::uint32_t), "sparsification degrees");
    owned_probe_check_fit(extra);
    impl_->ensure_state_buffer(impl_->residual(spare), std::max<std::size_t>(live, 1),
        "allocate sparsification spare residual", "grow sparsification spare residual");
    impl_->ensure_state_buffer(impl_->owner_offsets(spare), n + 1,
        "allocate sparsification spare offsets", "grow sparsification spare offsets");
    impl_->ensure_state_buffer(impl_->owned_degrees, n,
        "allocate sparsification degrees", "grow sparsification degrees");
    auto report = sparsify_owned_residual_device(static_cast<node_index>(n), live,
        impl_->residual(impl_->current_slot).get(), impl_->active.get(), 0.25, seed,
        impl_->residual(spare).get(), impl_->owner_offsets(spare).get(),
        impl_->owned_degrees.get(), impl_->tracker);
    const auto after = gpu_round_shadow_checked_mul(report.kept_edges, 2, "sparse directed count");
    device_buffer<device_semantics> semantics(impl_->tracker);
    semantics.allocate(1, "allocate sparse semantic receipt");
    cuda_check(cudaMemset(semantics.get(), 0, sizeof(device_semantics)), "clear sparse semantic receipt");
    if (after) {
        hash_residual_incidences<true><<<blocks_for(after), kBlock>>>(
            impl_->residual(spare).get(), after, semantics.get());
        cuda_check(cudaGetLastError(), "hash sparse weighted topology");
        hash_ordered_residual_incidences<<<blocks_for(after), kBlock>>>(
            impl_->residual(spare).get(), after, impl_->owner_offsets(spare).get(), semantics.get());
        cuda_check(cudaGetLastError(), "hash sparse ordered incidences");
    }
    if (n) {
        hash_residual_state<true><<<blocks_for(n), kBlock>>>(n, impl_->active.get(),
            impl_->owned_degrees.get(), impl_->excess.get(), semantics.get());
        cuda_check(cudaGetLastError(), "hash sparse vertex state");
    }
    device_semantics content{};
    copy_to_host(&content, semantics.get(), 1, "download sparse semantic receipt");
    const gpu_round_shadow_digest active_hash{content.active.xor_hash, content.active.sum_hash};
    if (active_hash != impl_->fingerprint.active || content.active_count != impl_->fingerprint.active_count)
        throw std::logic_error("GPU sparsification changed active state");
    // All output checks and consumers completed before publishing this epoch.
    impl_->current_slot = spare;
    impl_->resident_count = after;
    impl_->resident_selection_content.topology = {content.topology.xor_hash, content.topology.sum_hash};
    impl_->fingerprint = {{content.residual.xor_hash, content.residual.sum_hash},
        {content.ordered_residual.xor_hash, content.ordered_residual.sum_hash},
        active_hash, {content.topology.xor_hash, content.topology.sum_hash},
        {content.live_degree.xor_hash, content.live_degree.sum_hash},
        {content.canonical_excess.xor_hash, content.canonical_excess.sum_hash},
        static_cast<std::size_t>(content.active_count), after};
    impl_->owned_residual_epoch = std::make_shared<const int>(0);
    impl_->resident_rows_all_live = true;
    frontend.bind_owned_residual(impl_->residual(spare).get(), impl_->owner_offsets(spare).get(),
        impl_->active.get(), after, impl_->resident_selection_content, impl_->owned_residual_epoch);
    report.scalar_download_bytes += sizeof(content);
    report.peak_device_bytes = impl_->tracker.peak;
    guard.commit();
    return report;
}

void gpu_round_shadow_device_state::advance_selector(gpu_block_frontend& frontend) {
    impl::failed_generation_guard guard{*impl_};
    guard.arm();
    impl_->require_current_device("resident selector handoff");
    if (impl_->poisoned || !impl_->has_state || !impl_->generation_accepted ||
        (!impl_->certified && !(impl_->device_owned_prefix && !impl_->owned_prefix_materialized)) ||
        !impl_->selection_producer_bound ||
        !impl_->generation_has_device_selection ||
        impl_->next_reimport == impl::reimport_reason::authoritative_host_rebuild)
        throw std::logic_error("GPU selector handoff requires an accepted CPU-certified resident generation");
    // Inspect the LIVE published selection before either state is mutated.
    // Equality with the consumed generations also rejects a repeated projection,
    // an intervening prepare/selection/advance, and a different producer.
    const auto selected = frontend.selection_for_residual_handoff().inspect();
    const auto producer = impl_->selection_producer.lock();
    if (!producer || producer != selected.producer_identity ||
        selected.cuda_device != impl_->cuda_device ||
        selected.generation != impl_->last_selection_generation ||
        selected.topology_generation != impl_->last_selection_topology_generation)
        throw std::invalid_argument("GPU selector handoff has a stale or different producer generation");
    if (impl_->device_owned_prefix) {
        // The consumed selection provenance was checked above. This private
        // owning-only operation has a separate receipt, so the completed
        // elimination report keeps its original survivors+2*fill identity.
        frontend.bind_owned_residual(impl_->residual(impl_->current_slot).get(),
            impl_->owner_offsets(impl_->current_slot).get(), impl_->active.get(),
            impl_->resident_count, impl_->resident_selection_content,
            impl_->owned_residual_epoch);
    } else
        frontend.advance_resident(impl_->residual(impl_->current_slot).get(),
            impl_->resident_count, impl_->active.get(), impl_->resident_selection_content);
    guard.commit();
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
gpu_round_shadow_device_state::finalize_device_factor(
        std::span<const node_index> permutation, node_index factor_dim,
        const gpu_round_shadow_factor_log& tail) {
    impl_->require_current_device("factor finalization");
    if (impl_->poisoned || !impl_->has_state || !impl_->generation_accepted ||
        (impl_->device_owned_prefix && !impl_->owned_prefix_materialized))
        throw std::logic_error("GPU finalizer requires an accepted generation and published owned output");
    const bool fp16 = cuda_sptrsv::fp16_resolved();
    const double drop_rel = factor_drop_rel_from_env();
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
    device_buffer<float> col_scale(tracker), dropped_values(tracker), diag(tracker);
    device_buffer<double> inv_scale2(tracker);
    device_buffer<std::uint16_t> lh(tracker), lth(tracker);
    device_buffer<int> dropped_ptr(tracker);
    device_buffer<std::uint64_t> dropped_keys(tracker);
    device_buffer<unsigned long long> storage_counts(tracker);
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
    ltv.allocate(nnz, "finalizer raw LT values");
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
    const int raw_nnz = nnz;
    int* column_ptr = ltp.get();
    std::uint64_t* column_keys = sorted_keys.get();
    float* column_values = ltv.get();
    unsigned long long counts[3] = {};
    std::size_t extra_download = 0;
    if (fp16 || drop_rel > 0.0) {
        col_scale.allocate(m, "finalizer column scales");
        dropped_ptr.allocate(m + 1, "finalizer drop column pointers");
        storage_counts.allocate(3, "finalizer storage counts");
        cuda_check(cudaMemset(dropped_ptr.get(), 0, sizeof(int)), "finalizer clear drop origin");
        cuda_check(cudaMemset(storage_counts.get(), 0, 3 * sizeof(unsigned long long)), "finalizer clear storage counts");
        finalizer_column_storage<<<blocks_for(m), kBlock>>>(
            m, ltp.get(), sorted_keys.get(), ltv.get(), drop_rel, fp16,
            col_scale.get(), dropped_ptr.get(), storage_counts.get(), status.get());
        cuda_check(cudaGetLastError(), "finalizer storage planning");
        cuda_check(cub::DeviceScan::InclusiveSum(temporary.get(), scan_bytes,
            dropped_ptr.get(), dropped_ptr.get(), int(m + 1)), "finalizer drop scan");
        copy_to_host(&nnz, dropped_ptr.get() + m, 1, "finalizer dropped nnz");
        extra_download += sizeof(int);
        if (nnz < int(m) || nnz > raw_nnz)
            throw std::logic_error("GPU finalizer invalid compacted count");
        if (nnz != raw_nnz) {
            dropped_keys.allocate(nnz, "finalizer compacted keys");
            dropped_values.allocate(nnz, "finalizer compacted values");
            finalizer_compact_columns<<<blocks_for(m), kBlock>>>(
                m, ltp.get(), sorted_keys.get(), ltv.get(), col_scale.get(),
                drop_rel, fp16, dropped_ptr.get(), dropped_keys.get(), dropped_values.get());
            cuda_check(cudaGetLastError(), "finalizer compensated drop");
            column_ptr = dropped_ptr.get();
            column_keys = dropped_keys.get();
            column_values = dropped_values.get();
        }
    }
    li.allocate(nnz, "finalizer L indices"); lti.allocate(nnz, "finalizer LT indices");
    finalizer_unpack<<<blocks_for(nnz), kBlock>>>(nnz, column_keys, lti.get(), status.get(), keys.get(), lp.get());
    cuda_check(cudaGetLastError(), "finalizer LT unpack");
    cuda_check(cub::DeviceScan::InclusiveSum(temporary.get(), scan_bytes, lp.get(), lp.get(), int(m + 1)), "finalizer L pointer scan");
    if (fp16) {
        lh.allocate(nnz, "finalizer L half values");
        lth.allocate(nnz, "finalizer LT half values");
        diag.allocate(m, "finalizer scaled diagonal");
        inv_scale2.allocate(m, "finalizer squared inverse scale");
        finalizer_narrow_columns<<<blocks_for(m), kBlock>>>(
            m, column_ptr, column_values, col_scale.get(), lth.get(), diag.get(),
            inv_scale2.get(), storage_counts.get() + 2, status.get());
        cuda_check(cudaGetLastError(), "finalizer compensated fp16 narrowing");
        std::size_t half_sort_bytes = 0;
        cuda_check(cub::DeviceRadixSort::SortPairs(nullptr, half_sort_bytes,
            keys.get(), sorted_keys.get(), lth.get(), lh.get(), nnz), "finalizer half sort size");
        if (half_sort_bytes > std::max(sort_bytes, scan_bytes)) {
            temporary.release("finalizer replace half scratch");
            temporary.allocate(half_sort_bytes, "finalizer half sort scratch");
        }
        cuda_check(cub::DeviceRadixSort::SortPairs(temporary.get(), half_sort_bytes,
            keys.get(), sorted_keys.get(), lth.get(), lh.get(), nnz), "finalizer sort half L");
    } else {
        lv.allocate(nnz, "finalizer L values");
        cuda_check(cub::DeviceRadixSort::SortPairs(temporary.get(), sort_bytes,
            keys.get(), sorted_keys.get(), column_values, lv.get(), nnz), "finalizer sort L");
    }
    finalizer_unpack<<<blocks_for(nnz), kBlock>>>(nnz, sorted_keys.get(), li.get(), status.get(), nullptr, nullptr);
    cuda_check(cudaGetLastError(), "finalizer L unpack");
    copy_to_host(host_status, status.get(), 1, "finalizer storage status");
    if (host_status[0]) throw std::invalid_argument("GPU finalizer invalid coordinates or scaled storage, code " + std::to_string(host_status[0]));
    if (fp16 || drop_rel > 0.0) {
        copy_to_host(counts, storage_counts.get(), 3, "finalizer storage counts");
        extra_download += sizeof(counts);
    }
    factor_drop_stats stats;
    stats.rel = drop_rel; stats.nnz_factor = raw_nnz; stats.nnz_stored = nnz;
    stats.dropped = raw_nnz - nnz;
    stats.dropped_threshold = counts[0]; stats.dropped_flush = counts[1];
    if (stats.dropped != stats.dropped_threshold + stats.dropped_flush)
        throw std::logic_error("GPU finalizer inconsistent drop accounting");
    auto result = std::make_shared<cuda_sptrsv_device_factor>();
    if (fp16) {
        *result = cuda_sptrsv_device_factor::own_fp16_scaled(impl_->cuda_device, m, nnz,
            {{lp.get(), (m + 1) * sizeof(int)}, {li.get(), std::size_t(nnz) * sizeof(int)}, {lh.get(), std::size_t(nnz) * sizeof(std::uint16_t)}},
            {{column_ptr, (m + 1) * sizeof(int)}, {lti.get(), std::size_t(nnz) * sizeof(int)}, {lth.get(), std::size_t(nnz) * sizeof(std::uint16_t)}},
            {diag.get(), m * sizeof(float)}, {inv_scale2.get(), m * sizeof(double)}, stats, counts[2], 0);
        lh.detach(); lth.detach(); diag.detach(); inv_scale2.detach();
    } else {
        *result = cuda_sptrsv_device_factor::own_fp32(impl_->cuda_device, m, nnz,
            {{lp.get(), (m + 1) * sizeof(int)}, {li.get(), std::size_t(nnz) * sizeof(int)}, {lv.get(), std::size_t(nnz) * sizeof(float)}},
            {{column_ptr, (m + 1) * sizeof(int)}, {lti.get(), std::size_t(nnz) * sizeof(int)}, {column_values, std::size_t(nnz) * sizeof(float)}}, stats);
        lv.detach();
        if (column_values == ltv.get()) ltv.detach(); else dropped_values.detach();
    }
    result->finalized_sorted_triangular_ = true;
    lp.detach(); li.detach(); lti.detach();
    if (column_ptr == ltp.get()) ltp.detach(); else dropped_ptr.detach();
    const std::size_t upload = n * sizeof(node_index) +
        tail.columns.size() * sizeof(gpu_round_shadow_factor_column) +
        tail.entries.size() * sizeof(gpu_round_shadow_factor_entry);
    if (gpu_setup_diagnostics()) std::fprintf(stderr, "[gpu-factor-finalize] device_prefix_columns=%zu device_prefix_entries=%zu cpu_tail_columns=%zu cpu_tail_entries=%zu m=%zu nnz=%d host_upload_bytes=%zu scalar_download_bytes=%zu prefix_download_bytes=0 scratch_and_output_peak_bytes=%zu cpu_shadow=%s storage=%s factor_drop_rel=%.17g dropped_threshold=%llu dropped_flush=%llu fp16_flushed=%llu\n",
        prefix, impl_->factor_entry_count, tail.columns.size(), tail.entries.size(),
        m, nnz, upload, 4 * sizeof(int) + extra_download, tracker.peak,
        impl_->device_owned_prefix ? "omitted_owned_prefix" : "retained",
        fp16 ? "fp16" : "fp32", drop_rel,
        static_cast<unsigned long long>(stats.dropped_threshold),
        static_cast<unsigned long long>(stats.dropped_flush), counts[2]);
    return result;
}

#if defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
std::vector<std::byte> gpu_round_shadow_device_state::encode_finalized_factor_for_test(
        const cuda_sptrsv_device_factor& factor) const {
    impl_->require_current_device("test storage snapshot");
    if (factor.cuda_device_ != impl_->cuda_device || factor.empty())
        throw std::invalid_argument("GPU finalizer test snapshot device/storage mismatch");
    std::vector<std::byte> encoded;
    auto append = [&](const void* device, std::size_t bytes) {
        const auto start = encoded.size();
        encoded.resize(start + bytes);
        cuda_check(cudaMemcpy(encoded.data() + start, device, bytes, cudaMemcpyDeviceToHost),
            "test finalized storage download");
    };
    const std::size_t m = factor.m_, nnz = factor.nnz_;
    const bool half = factor.storage_ == cuda_sptrsv_device_factor::storage::fp16_scaled;
    const std::size_t value_bytes = nnz * (half ? sizeof(std::uint16_t) : sizeof(float));
    append(factor.L_row_ptr_, (m + 1) * sizeof(int));
    append(factor.L_col_idx_, nnz * sizeof(int));
    append(factor.L_values_, value_bytes);
    append(factor.LT_row_ptr_, (m + 1) * sizeof(int));
    append(factor.LT_col_idx_, nnz * sizeof(int));
    append(factor.LT_values_, value_bytes);
    if (half) {
        append(factor.diag_, m * sizeof(float));
        append(factor.inv_scale2_, m * sizeof(double));
    }
    return encoded;
}
#endif

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
