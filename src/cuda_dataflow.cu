#include "apxchol/solver/sptrsv/cuda_dataflow.h"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cub/device/device_scan.cuh>
#include <cuda/atomic>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace apxchol {
namespace {
void dataflow_check(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("GPU dataflow: ") + operation + ": " +
                                 cudaGetErrorString(status));
}
__device__ __forceinline__ float dataflow_accumulate(__half *values, float *x, float sum) {
    float a = 0, b = 0;
#pragma unroll
    for (unsigned k = 0; k < 8; k += 2) {
        const float2 v = __half22float2(__halves2half2(values[k], values[k + 1]));
        a = fmaf(v.x, x[k], a);
        b = fmaf(v.y, x[k + 1], b);
    }
    return sum + (a + b);
}
#define APXCHOL_GPU_CODE(...) __VA_ARGS__
#define APXCHOL_GPU_FUNCTION __device__ __forceinline__
#define APXCHOL_UNROLL _Pragma("unroll")
#include "gpu_dataflow.inc"
#undef APXCHOL_GPU_CODE
#undef APXCHOL_GPU_FUNCTION
#undef APXCHOL_UNROLL

template <class Value> struct dataflow_memory {
    using stored_value = Value;
    const int *__restrict__ ptr;
    const int *__restrict__ col;
    const Value *__restrict__ val;
    const unsigned *words;
    const detail::gpu_dataflow::segment *segments;
    unsigned *tags;
    unsigned *ticket;
    const float *__restrict__ rhs;
    float *__restrict__ out;
    unsigned n, batches;
    bool reverse;
    const float *__restrict__ diag = nullptr;
    const double *__restrict__ scale = nullptr;
    const unsigned *active = nullptr;
    const double *rhs64 = nullptr;
    double *out64 = nullptr;

    __device__ unsigned width(unsigned size) const {
        return 1u << min(5u, 32u - __clz((size ? size - 1 : 0) / 8));
    }
    __device__ unsigned claim() const { return atomicAdd(ticket, 1); }
    __device__ unsigned broadcast(unsigned x) const { return __shfl_sync(0xffffffffu, x, 0); }
    __device__ bool any(bool x) const { return __any_sync(0xffffffffu, x); }
    __device__ unsigned ballot(bool x) const { return __ballot_sync(0xffffffffu, x); }
    __device__ float shuffle(float x, unsigned offset) const {
        return __shfl_xor_sync(0xffffffffu, x, offset);
    }
    __device__ unsigned load(unsigned row) const {
        return cuda::atomic_ref<unsigned, cuda::thread_scope_device>(tags[row]).load(
            cuda::memory_order_relaxed);
    }
    __device__ float as_float(unsigned x) const { return __uint_as_float(x); }
    __device__ void reset(unsigned row) const { tags[row] = 0x7f800001u; }
    __device__ void prepare(unsigned row, float &b, float &d) const {
        const float input = rhs64 ? float(rhs64[row]) : rhs[row];
        b = scale ? float(double(input) * scale[row]) : input;
        if constexpr (std::is_same_v<Value, __half>)
            d = diag[row];
        else
            d = val[reverse ? ptr[row] : ptr[row + 1] - 1];
    }
    __device__ float finish(float b, float d, float sum) const { return (b - sum) / d; }
    __device__ void publish(unsigned row, float x) const {
        if (row < n) {
            if (out64) out64[row] = x;
            else out[row] = x;
        }
        // Ready and value share one atomic word. Canonicalize NaNs so a failed
        // numerical solve cannot turn into an endless wait on the pending marker.
        cuda::atomic_ref<unsigned, cuda::thread_scope_device>(tags[row]).store(
            isnan(x) ? 0x7fc00000u : __float_as_uint(x), cuda::memory_order_relaxed);
    }
};

template <class Memory> __global__ void initialize(Memory m, unsigned count) {
    if (m.active && !*m.active) return;
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row == 0)
        *m.ticket = 0;
    if (row >= count)
        return;
    dataflow_initialize(m, row);
}

template <class Memory> __global__ void __launch_bounds__(kDataflowBlock) solve_dataflow(Memory m) {
    if (!m.active || *m.active) dataflow_run(m, threadIdx.x % 32);
}

template <class Value>
void launch(cudaStream_t stream, dataflow_memory<Value> memory, unsigned count, int grid) {
    if (!memory.n)
        return;
    initialize<<<(count + 255) / 256, 256, 0, stream>>>(memory, count);
    if (memory.batches)
        solve_dataflow<<<std::min(unsigned(grid), (memory.batches + 31) / 32), kDataflowBlock, 0, stream>>>(
            memory);
}
} // namespace

int dataflow_prefetch_depth() { return detail::gpu_dataflow::prefetch; }

int dataflow_grid_size(bool fp16) {
    int device, sms, resident;
    dataflow_check(cudaGetDevice(&device), "query device");
    dataflow_check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "query SM count");
    if (fp16)
        dataflow_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &resident, solve_dataflow<dataflow_memory<__half>>, kDataflowBlock, 0),
                       "query FP16 occupancy");
    else
        dataflow_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &resident, solve_dataflow<dataflow_memory<float>>, kDataflowBlock, 0),
                       "query FP32 occupancy");
    const char *requested = std::getenv("APXCHOL_GPU_DATAFLOW_BLOCKS");
    return std::max(1, std::min(sms * resident, requested ? std::atoi(requested) : sms));
}

void dataflow_solve(cudaStream_t stream, int m, bool reverse, const int *ptr, const int *col,
                    const float *val, const unsigned *words, const detail::gpu_dataflow::segment *segments,
                    unsigned batches, unsigned *tags, unsigned tag_count, unsigned *ticket, int grid,
                    const float *rhs, float *out, const unsigned *active,
                    const double *rhs64, double *out64) {
    launch(stream,
           dataflow_memory<float>{ptr, col, val, words, segments, tags, ticket, rhs, out, unsigned(m),
                                  batches, reverse, nullptr, nullptr, active, rhs64, out64},
           tag_count, grid);
}

void dataflow_solve_fp16(cudaStream_t stream, int m, bool reverse, const int *ptr, const int *col,
                         const __half *val, const float *__restrict__ diag, const double *__restrict__ scale,
                         const unsigned *words, const detail::gpu_dataflow::segment *segments,
                         unsigned batches, unsigned *tags, unsigned tag_count, unsigned *ticket, int grid,
                         const float *rhs, float *out, const unsigned *active,
                    const double *rhs64, double *out64) {
    launch(stream,
           dataflow_memory<__half>{ptr, col, val, words, segments, tags, ticket, rhs, out, unsigned(m),
                                   batches, reverse, diag, scale, active, rhs64, out64},
           tag_count, grid);
}
} // namespace apxchol

namespace apxchol::detail {
namespace {
constexpr int kPlanChunk = 256;
constexpr int kPlanBuckets = dataflow_device_plan::kBuckets;
struct plan_row_stats {
    unsigned long long rows[kPlanBuckets], nnz[kPlanBuckets];
    int max_len, error;
};

__global__ void plan_row_statistics(int m, int nnz, const int *ptr, const float *__restrict__ diag,
                                    const double *inv_scale2, plan_row_stats *out) {
    __shared__ plan_row_stats block;
    if (threadIdx.x == 0)
        block = {};
    __syncthreads();
    const std::size_t position = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (blockIdx.x == 0 && threadIdx.x == 0 && (ptr[0] != 0 || ptr[m] != nnz))
        atomicOr(&block.error, 1);
    if (position < static_cast<std::size_t>(m)) {
        const int row = static_cast<int>(position);
        const int lo = ptr[row], hi = ptr[row + 1];
        if (lo < 0 || hi <= lo || hi > nnz)
            atomicOr(&block.error, 1);
        else {
            const int len = hi - lo;
            atomicMax(&block.max_len, len);
            for (int k = 0; k < kPlanBuckets; ++k) {
                if (static_cast<long long>(len) < (static_cast<long long>(32 * gpu_dataflow::prefetch) << k))
                    break;
                atomicAdd(&block.rows[k], 1ULL);
                atomicAdd(&block.nnz[k], static_cast<unsigned long long>(len));
            }
        }
        if (diag && (!isfinite(diag[row]) || diag[row] == 0.0f || !isfinite(inv_scale2[row])))
            atomicOr(&block.error, 2);
    }
    __syncthreads();
    if (threadIdx.x < kPlanBuckets && block.rows[threadIdx.x]) {
        atomicAdd(&out->rows[threadIdx.x], block.rows[threadIdx.x]);
        atomicAdd(&out->nnz[threadIdx.x], block.nnz[threadIdx.x]);
    }
    if (threadIdx.x == 0) {
        atomicMax(&out->max_len, block.max_len);
        atomicOr(&out->error, block.error);
    }
}

using plan_counts = gpu_dataflow::counts;
struct add_plan_counts {
    __host__ __device__ plan_counts operator()(plan_counts a, plan_counts b) const {
        return {a.words + b.words, a.segments + b.segments, a.slots + b.slots};
    }
};

template <bool Emit>
__global__ void pack_chunks(unsigned m, bool reverse, const int *ptr, unsigned split_min, unsigned chunks,
                            plan_counts *counts, const plan_counts *prefix, unsigned *words,
                            gpu_dataflow::segment *segments) {
    const unsigned chunk = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk > chunks || (Emit && chunk == chunks))
        return;
    const unsigned first = min(m, chunk * kPlanChunk), last = min(m, first + kPlanChunk);
    const auto c = gpu_dataflow::pack_rows<Emit>(ptr, {m, reverse}, first, last, split_min,
                                                 Emit ? prefix[chunk] : plan_counts{}, words, segments);
    if constexpr (!Emit)
        counts[chunk] = c;
}

} // namespace

dataflow_device_plan build_dataflow_device_plan(int m, bool reverse, int nnz, const int *rowptr,
                                                int split_min, const float *__restrict__ diag,
                                                const double *inv_scale2) {
    if (m < 0 || m == INT_MAX || nnz < m || !rowptr || split_min < -1 ||
        static_cast<bool>(diag) != static_cast<bool>(inv_scale2))
        throw std::invalid_argument("GPU dataflow plan: invalid dimensions or metadata");
    dataflow_device_plan result;
    plan_row_stats *statistics = nullptr;
    plan_counts *counts_by_chunk = nullptr, *prefix = nullptr;
    void *scratch = nullptr;
    auto cleanup = [&] {
        cudaFree(statistics);
        cudaFree(counts_by_chunk);
        cudaFree(prefix);
        cudaFree(scratch);
    };
    try {
        dataflow_check(cudaMalloc(&statistics, sizeof(plan_row_stats)), "allocate statistics");
        dataflow_check(cudaMemset(statistics, 0, sizeof(plan_row_stats)), "clear statistics");
        plan_row_statistics<<<std::max(1, m / 256 + (m % 256 != 0)), 256>>>(m, nnz, rowptr, diag, inv_scale2,
                                                                            statistics);
        dataflow_check(cudaGetLastError(), "validate/statistics launch");
        plan_row_stats host_stats{};
        dataflow_check(cudaMemcpy(&host_stats, statistics, sizeof(host_stats), cudaMemcpyDeviceToHost),
                       "download statistics");
        result.host_download_bytes = sizeof(host_stats);
        if (host_stats.error)
            throw std::invalid_argument(
                "GPU dataflow plan: invalid CSR row pointers or fp16 metadata (code " +
                std::to_string(host_stats.error) + ")");
        result.max_len = host_stats.max_len;
        for (int k = 0; k < kPlanBuckets; ++k) {
            result.rows_ge[k] = host_stats.rows[k];
            result.nnz_ge[k] = host_stats.nnz[k];
        }
        if (split_min == -1)
            split_min = nnz && host_stats.nnz[0] * 4 >= static_cast<unsigned long long>(nnz)
                            ? 32 * gpu_dataflow::prefetch
                            : 96 * gpu_dataflow::prefetch;
        if (m == 0) {
            cleanup();
            return result;
        }
        const int chunks = m / kPlanChunk + (m % kPlanChunk != 0);
        const std::size_t count_bytes = static_cast<std::size_t>(chunks + 1) * sizeof(plan_counts);
        dataflow_check(cudaMalloc(&counts_by_chunk, count_bytes), "allocate tile counts");
        dataflow_check(cudaMalloc(&prefix, count_bytes), "allocate tile prefixes");
        pack_chunks<false><<<(chunks + 128) / 128, 128>>>(m, reverse, rowptr, split_min, chunks,
                                                          counts_by_chunk, nullptr, nullptr, nullptr);
        dataflow_check(cudaGetLastError(), "tile count launch");
        std::size_t scratch_bytes = 0;
        dataflow_check(cub::DeviceScan::ExclusiveScan(nullptr, scratch_bytes, counts_by_chunk, prefix,
                                                      add_plan_counts{}, plan_counts{}, chunks + 1),
                       "tile scan size");
        dataflow_check(cudaMalloc(&scratch, scratch_bytes), "allocate tile scan scratch");
        dataflow_check(cub::DeviceScan::ExclusiveScan(scratch, scratch_bytes, counts_by_chunk, prefix,
                                                      add_plan_counts{}, plan_counts{}, chunks + 1),
                       "tile prefix scan");
        plan_counts counts{};
        dataflow_check(cudaMemcpy(&counts, prefix + chunks, sizeof(counts), cudaMemcpyDeviceToHost),
                       "download final counts");
        result.host_download_bytes += sizeof(counts);
        if (counts.words > UINT_MAX || counts.segments >= gpu_dataflow::special_bit ||
            counts.slots + static_cast<unsigned long long>(m) >= gpu_dataflow::special_bit)
            throw std::overflow_error("GPU dataflow plan exceeds 32-bit capacity");
        result.n_batches = static_cast<int>(counts.words / gpu_dataflow::lanes);
        result.n_slots = static_cast<int>(counts.slots);
        result.n_split = static_cast<int>(counts.segments - counts.slots);
        const std::size_t word_bytes = counts.words * sizeof(unsigned);
        const std::size_t segment_bytes = counts.segments * sizeof(gpu_dataflow::segment);
        if (word_bytes)
            dataflow_check(cudaMalloc(&result.words, word_bytes), "allocate tasks");
        if (segment_bytes)
            dataflow_check(cudaMalloc(&result.segments, segment_bytes), "allocate segments");
        pack_chunks<true><<<(chunks + 127) / 128, 128>>>(m, reverse, rowptr, split_min, chunks, nullptr,
                                                         prefix, result.words, result.segments);
        dataflow_check(cudaGetLastError(), "emit chunk plans");
        dataflow_check(cudaDeviceSynchronize(), "finish device plan");
        result.bytes = word_bytes + segment_bytes;
        cleanup();
        return result;
    } catch (...) {
        cleanup();
        cudaFree(result.words);
        cudaFree(result.segments);
        throw;
    }
}
} // namespace apxchol::detail
