#include "apxchol/solver/sptrsv/cuda_dataflow.h"
#include "apxchol/solver/detail/cuda_error.h"
#include "cuda_buffer.cuh"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cub/device/device_scan.cuh>
#include <cuda/atomic>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace apxchol::detail {
namespace {
void dataflow_check(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        detail::check_cuda(status, (std::string("GPU dataflow: ") + operation).c_str());
}
__device__ __forceinline__ float dataflow_accumulate(__half *values, float *x, float sum) {
    float a = 0, b = 0;
#pragma unroll
    for (unsigned k = 0; k < 8; k += 2) {
        a = fmaf(__half2float(values[k]), x[k], a);
        b = fmaf(__half2float(values[k + 1]), x[k + 1], b);
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

template <class Value, unsigned Columns = 1, unsigned RhsLanes = 1> struct dataflow_memory {
    using stored_value = Value;
    static constexpr unsigned columns = Columns;
    static constexpr unsigned rhs_lanes = RhsLanes;
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

    unsigned active_stride = 1;
    __device__ bool enabled(unsigned q) const { return !active || active[q * active_stride]; }
    __device__ Value value(unsigned pos, unsigned, unsigned) const { return val[pos]; }
    __device__ unsigned width(unsigned size) const {
        if constexpr (RhsLanes == 1)
            return 1u << min(5u, 32u - __clz((size ? size - 1 : 0) / 8));
        else
            return min(32u / RhsLanes, 1u << min(5u, 32u - __clz((size ? size - 1 : 0) / 8)));
    }
    __device__ unsigned claim() const { return atomicAdd(ticket, 1); }
    __device__ unsigned broadcast(unsigned x) const { return __shfl_sync(0xffffffffu, x, 0); }
    __device__ bool any(bool x) const { return __any_sync(0xffffffffu, x); }
    __device__ unsigned ballot(bool x) const { return __ballot_sync(0xffffffffu, x); }
    __device__ float shuffle(float x, unsigned offset) const {
        return __shfl_xor_sync(0xffffffffu, x, offset);
    }
    __device__ unsigned load(unsigned row, unsigned q) const {
        return cuda::atomic_ref<unsigned, cuda::thread_scope_device>(tags[row * Columns + q]).load(
            cuda::memory_order_relaxed);
    }
    __device__ float as_float(unsigned x) const { return __uint_as_float(x); }
    __device__ void reset(unsigned row, unsigned q) const { tags[row * Columns + q] = 0x7f800001u; }
    __device__ void prepare(unsigned row, unsigned q, float &b, float &d) const {
        const float input = rhs64 ? float(rhs64[row * Columns + q]) : rhs[row * Columns + q];
        b = scale ? float(double(input) * scale[row]) : input;
        if constexpr (std::is_same_v<Value, __half>)
            d = diag[row];
        else
            d = val[reverse ? ptr[row] : ptr[row + 1] - 1];
    }
    __device__ float finish(float b, float d, float sum) const { return (b - sum) / d; }
    __device__ void publish(unsigned row, unsigned q, float x) const {
        if (row < n) {
            if (out64) out64[row * Columns + q] = x;
            else if (out) out[row * Columns + q] = x;
        }
        // Ready and value share one atomic word. Canonicalize NaNs so a failed
        // numerical solve cannot turn into an endless wait on the pending marker.
        cuda::atomic_ref<unsigned, cuda::thread_scope_device>(tags[row * Columns + q]).store(
            isnan(x) ? 0x7fc00000u : __float_as_uint(x), cuda::memory_order_relaxed);
    }
};

template <class Memory> __global__ void initialize(Memory m, unsigned count) {
    // One thread per panel element: the panel is interleaved, so consecutive
    // threads then touch consecutive words at every width.
    const std::uint64_t element = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned row = static_cast<unsigned>(element / Memory::columns);
    const unsigned q = static_cast<unsigned>(element % Memory::columns);
    if (element == 0) {
        // The ticket restarts only when some column solves, as before.
        bool active = false;
        for (unsigned c = 0; c < Memory::columns; ++c) active |= m.enabled(c);
        if (active) *m.ticket = 0;
    }
    if (row >= count || !m.enabled(q))
        return;
    if (row < m.n && m.ptr[row + 1] - m.ptr[row] == 1) {
        float rhs, diagonal;
        m.prepare(row, q, rhs, diagonal);
        m.publish(row, q, m.finish(rhs, diagonal, 0));
    } else
        m.reset(row, q);
}

template <class Memory> __global__ void __launch_bounds__(kDataflowBlock) solve_dataflow(Memory m) {
    dataflow_run<Memory::columns, Memory::rhs_lanes>(m, threadIdx.x % 32);
}

// Dense tail (see dataflow_tail). One warp per (tail row, column); the
// products use all SMs, so no sequential chain remains.
__device__ __forceinline__ float tail_value(float v) { return v; }
__device__ __forceinline__ float tail_value(__half v) { return __half2float(v); }

constexpr unsigned kTailGemvBlock = 256;
// Forward: c = b - (entries before the tail) per tail row, into scratch.
template <class Memory> __global__ void tail_forward_rhs(Memory m, dataflow_tail t) {
    const unsigned lane = threadIdx.x % 32;
    const std::size_t warp = (std::size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (warp >= std::size_t(t.rows) * Memory::columns) return;
    const unsigned r = unsigned(warp / Memory::columns), q = unsigned(warp % Memory::columns);
    if (!m.enabled(q)) return;
    const unsigned row = t.begin + r;
    float sum = 0;
    for (int p = m.ptr[row] + int(lane); p < m.ptr[row + 1]; p += 32) {
        const unsigned col = unsigned(m.col[p]);
        if (col < t.begin) sum = fmaf(tail_value(m.val[p]), m.as_float(m.load(col, q)), sum);
    }
    for (unsigned o = 16; o; o /= 2) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) {
        float b, diagonal;
        m.prepare(row, q, b, diagonal);
        t.scratch[r * Memory::columns + q] = __float_as_uint(b - sum);
    }
}
// Forward: y = inverse * c.
template <class Memory> __global__ void tail_forward_gemv(Memory m, dataflow_tail t) {
    const unsigned lane = threadIdx.x % 32;
    const std::size_t warp = (std::size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (warp >= std::size_t(t.rows) * Memory::columns) return;
    const unsigned r = unsigned(warp / Memory::columns), q = unsigned(warp % Memory::columns);
    if (!m.enabled(q)) return;
    const float *a = t.inverse + std::size_t(r) * t.rows;
    float sum = 0;
    for (unsigned k = lane; k <= r; k += 32)
        sum = fmaf(a[k], __uint_as_float(t.scratch[k * Memory::columns + q]), sum);
    for (unsigned o = 16; o; o /= 2) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) {
        const unsigned at = (t.begin + r) * Memory::columns + q;
        if (m.out) m.out[at] = sum;
        else m.tags[at] = __float_as_uint(sum);
    }
}
// Back: x = inverse^T c, published for the dataflow that follows.
template <class Memory> __global__ void tail_backward_gemv(Memory m, dataflow_tail t) {
    const unsigned lane = threadIdx.x % 32;
    const std::size_t warp = (std::size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (warp >= std::size_t(t.rows) * Memory::columns) return;
    const unsigned j = unsigned(warp / Memory::columns), q = unsigned(warp % Memory::columns);
    if (!m.enabled(q)) return;
    const float *a = t.inverse_t + std::size_t(j) * t.rows;
    float sum = 0;
    for (unsigned i = j + lane; i < t.rows; i += 32) {
        float b, diagonal;
        m.prepare(t.begin + i, q, b, diagonal);
        sum = fmaf(a[i], b, sum);
    }
    for (unsigned o = 16; o; o /= 2) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) m.publish(t.begin + j, q, sum);
}

constexpr unsigned kTailInverseBlock = 512;
__global__ void tail_diagonal(unsigned begin, unsigned T, const int *ptr, const float *values32,
                              const float *diag, double *out) {
    const unsigned r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= T) return;
    const unsigned row = begin + r;
    out[r] = diag ? double(diag[row]) : double(values32[ptr[row + 1] - 1]);
}
// One block per inverse column j: solves L z = e_j over rows [j, T) in fp64,
// 32 rows at a time: warp 0 solves the diagonal tile with shuffles, then the
// block updates the remaining rows. `block` is the strictly lower part.
__global__ void __launch_bounds__(kTailInverseBlock) tail_inverse(unsigned T, const float *block, const double *d,
                                                                  float *inverse, float *inverse_t) {
    extern __shared__ double inverse_shared[];
    double *c = inverse_shared;
    __shared__ double xs[32];
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = blockDim.x / 32;
    for (unsigned j = blockIdx.x; j < T; j += gridDim.x) {
        for (unsigned i = j + threadIdx.x; i < T; i += blockDim.x) c[i] = i == j ? 1.0 : 0.0;
        __syncthreads();
        for (unsigned base = j; base < T; base += 32) {
            const unsigned cnt = min(32u, T - base);
            if (warp == 0) {
                double v = lane < cnt ? c[base + lane] : 0.0;
                for (unsigned k = 0; k < cnt; ++k) {
                    if (lane == k) v = v / d[base + k];
                    const double x = __shfl_sync(0xffffffffu, v, k);
                    if (lane > k && lane < cnt) v = fma(-double(block[std::size_t(base + lane) * T + base + k]), x, v);
                }
                if (lane < cnt) { c[base + lane] = v; xs[lane] = v; }
            }
            __syncthreads();
            for (unsigned r = base + cnt + warp; r < T; r += warps) {
                double s = lane < cnt ? double(block[std::size_t(r) * T + base + lane]) * xs[lane] : 0.0;
                for (unsigned o = 16; o; o /= 2) s += __shfl_xor_sync(0xffffffffu, s, o);
                if (lane == 0) c[r] -= s;
            }
            __syncthreads();
        }
        for (unsigned i = j + threadIdx.x; i < T; i += blockDim.x) {
            const float z = float(c[i]);
            inverse[std::size_t(i) * T + j] = z;
            inverse_t[std::size_t(j) * T + i] = z;
        }
        __syncthreads();
    }
}

template <class Value>
__global__ void tail_block_fill(unsigned begin, unsigned T, const int *ptr, const int *col, const Value *val,
                                float *block) {
    for (unsigned r = blockIdx.x; r < T; r += gridDim.x) {
        const unsigned row = begin + r;
        for (int p = ptr[row] + int(threadIdx.x); p < ptr[row + 1]; p += blockDim.x) {
            const unsigned c = unsigned(col[p]);
            if (c >= begin && c < row) block[std::size_t(r) * T + (c - begin)] = tail_value(val[p]);
        }
    }
}

template <class Value, unsigned Columns, unsigned RhsLanes>
void launch(cudaStream_t stream, dataflow_memory<Value, Columns, RhsLanes> memory, unsigned count, int grid,
            dataflow_tail tail) {
    if (!memory.n)
        return;
    const std::uint64_t elements = static_cast<std::uint64_t>(count) * Columns;
    initialize<<<static_cast<unsigned>((elements + 255) / 256), 256, 0, stream>>>(memory, count);
    const unsigned gemv_blocks =
        unsigned((std::size_t(tail.rows) * Columns * 32 + kTailGemvBlock - 1) / kTailGemvBlock);
    if (tail.rows && memory.reverse)
        tail_backward_gemv<<<gemv_blocks, kTailGemvBlock, 0, stream>>>(memory, tail);
    if (memory.batches)
        solve_dataflow<<<std::min(unsigned(grid), (memory.batches + 31) / 32), kDataflowBlock, 0, stream>>>(
            memory);
    if (tail.rows && !memory.reverse) {
        tail_forward_rhs<<<gemv_blocks, kTailGemvBlock, 0, stream>>>(memory, tail);
        tail_forward_gemv<<<gemv_blocks, kTailGemvBlock, 0, stream>>>(memory, tail);
    }
}
} // namespace

void build_dataflow_tail_inverse(unsigned begin, unsigned rows, const int *L_rowptr, const int *L_colidx,
                                 const void *L_vals, bool fp16, const float *diag, float *inverse,
                                 float *inverse_t) {
    if (rows < 2 || rows > kDataflowTailMaxRows || !inverse || !inverse_t || (fp16 && !diag))
        throw std::invalid_argument("GPU dataflow: dense tail inverse arguments are invalid");
    const std::size_t bytes = std::size_t(rows) * rows * sizeof(float);
    cuda_buffer<float, dataflow_check> block;
    cuda_buffer<double, dataflow_check> d;
    block.allocate(std::size_t(rows) * rows, "allocate tail block");
    d.allocate(rows, "allocate tail diagonal");
    dataflow_check(cudaMemset(block.get(), 0, bytes), "clear tail block");
    dataflow_check(cudaMemset(inverse, 0, bytes), "clear tail inverse");
    dataflow_check(cudaMemset(inverse_t, 0, bytes), "clear tail inverse transpose");
    const unsigned fill_blocks = std::min(rows, 4096u);
    if (fp16)
        tail_block_fill<<<fill_blocks, 128>>>(begin, rows, L_rowptr, L_colidx, static_cast<const __half *>(L_vals),
                                              block.get());
    else
        tail_block_fill<<<fill_blocks, 128>>>(begin, rows, L_rowptr, L_colidx, static_cast<const float *>(L_vals),
                                              block.get());
    tail_diagonal<<<(rows + 255) / 256, 256>>>(begin, rows, L_rowptr,
        fp16 ? nullptr : static_cast<const float *>(L_vals), fp16 ? diag : nullptr, d.get());
    // At most 16 KB of fp64 column: within the default shared-memory limit.
    static_assert(kDataflowTailMaxRows * sizeof(double) <= 48 * 1024);
    tail_inverse<<<rows, kTailInverseBlock, std::size_t(rows) * sizeof(double)>>>(rows, block.get(), d.get(),
                                                                                  inverse, inverse_t);
    dataflow_check(cudaGetLastError(), "tail inverse");
    dataflow_check(cudaDeviceSynchronize(), "finish tail inverse");
}

int dataflow_grid_size() {
    int device, sms;
    dataflow_check(cudaGetDevice(&device), "query device");
    dataflow_check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "query SM count");
    return sms;
}

template<unsigned Columns, unsigned RhsLanes>
void dataflow_solve(cudaStream_t stream, int m, bool reverse, const int *ptr, const int *col,
                    const void *val, const float *diag, const double *scale,
                    const unsigned *words, const detail::gpu_dataflow::segment *segments,
                    unsigned batches, unsigned *tags, unsigned tag_count, unsigned *ticket, int grid,
                    const float *rhs, float *out, const unsigned *active,
                    const double *rhs64, double *out64, unsigned active_stride, dataflow_tail tail) {
    if (tail.rows && (!tail.inverse || !tail.inverse_t || (!reverse && !tail.scratch) ||
                      tail.rows > kDataflowTailMaxRows || m < 0 || tail.begin > unsigned(m) ||
                      tail.rows != unsigned(m) - tail.begin))
        throw std::invalid_argument("GPU dataflow: dense tail is invalid");
    auto run = [&]<class Value>(const Value* values) {
        launch(stream,
               dataflow_memory<Value, Columns, RhsLanes>{ptr, col, values, words, segments, tags, ticket, rhs, out,
                                      unsigned(m), batches, reverse, diag, scale, active, rhs64, out64, active_stride},
               tag_count, grid, tail);
    };
    if (diag) run(static_cast<const __half*>(val));
    else run(static_cast<const float*>(val));
}
template void dataflow_solve<1, 1>(cudaStream_t, int, bool, const int*, const int*, const void*, const float*, const double*, const unsigned*, const detail::gpu_dataflow::segment*, unsigned, unsigned*, unsigned, unsigned*, int, const float*, float*, const unsigned*, const double*, double*, unsigned, dataflow_tail);
template void dataflow_solve<2, 1>(cudaStream_t, int, bool, const int*, const int*, const void*, const float*, const double*, const unsigned*, const detail::gpu_dataflow::segment*, unsigned, unsigned*, unsigned, unsigned*, int, const float*, float*, const unsigned*, const double*, double*, unsigned, dataflow_tail);
template void dataflow_solve<4, 4>(cudaStream_t, int, bool, const int*, const int*, const void*, const float*, const double*, const unsigned*, const detail::gpu_dataflow::segment*, unsigned, unsigned*, unsigned, unsigned*, int, const float*, float*, const unsigned*, const double*, double*, unsigned, dataflow_tail);
template void dataflow_solve<8, 4>(cudaStream_t, int, bool, const int*, const int*, const void*, const float*, const double*, const unsigned*, const detail::gpu_dataflow::segment*, unsigned, unsigned*, unsigned, unsigned*, int, const float*, float*, const unsigned*, const double*, double*, unsigned, dataflow_tail);
template void dataflow_solve<16, 4>(cudaStream_t, int, bool, const int*, const int*, const void*, const float*, const double*, const unsigned*, const detail::gpu_dataflow::segment*, unsigned, unsigned*, unsigned, unsigned*, int, const float*, float*, const unsigned*, const double*, double*, unsigned, dataflow_tail);
template void dataflow_solve<32, 4>(cudaStream_t, int, bool, const int*, const int*, const void*, const float*, const double*, const unsigned*, const detail::gpu_dataflow::segment*, unsigned, unsigned*, unsigned, unsigned*, int, const float*, float*, const unsigned*, const double*, double*, unsigned, dataflow_tail);
} // namespace apxchol::detail

namespace apxchol::detail {
namespace {
constexpr int kPlanChunk = 256;
using plan_row_stats = gpu_dataflow::row_stats;
constexpr int kPlanBuckets = plan_row_stats::kBuckets;

__global__ void plan_row_statistics(int m, int nnz, const int *ptr, const float *__restrict__ diag,
                                    const double *inv_scale2, plan_row_stats *out, int planned_rows) {
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
        else if (row < planned_rows) {
            const int len = hi - lo;
            atomicMax(&block.max_len, len);
            for (int k = 0; k < kPlanBuckets; ++k) {
                if (static_cast<long long>(len) < (static_cast<long long>(gpu_dataflow::chunk_size) << k))
                    break;
                atomicAdd(&block.rows_ge[k], 1ULL);
                atomicAdd(&block.nnz_ge[k], static_cast<unsigned long long>(len));
            }
        }
        if (diag && (!isfinite(diag[row]) || diag[row] == 0.0f || !isfinite(inv_scale2[row])))
            atomicOr(&block.error, 2);
    }
    __syncthreads();
    if (threadIdx.x < kPlanBuckets && block.rows_ge[threadIdx.x]) {
        atomicAdd(&out->rows_ge[threadIdx.x], block.rows_ge[threadIdx.x]);
        atomicAdd(&out->nnz_ge[threadIdx.x], block.nnz_ge[threadIdx.x]);
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

template <bool Emit, unsigned PlanLanes = 32>
__global__ void pack_chunks(unsigned m, bool reverse, const int *ptr, unsigned split_min, unsigned chunks,
                            plan_counts *counts, const plan_counts *prefix, unsigned *words,
                            gpu_dataflow::segment *segments, unsigned q_begin, unsigned q_end) {
    const unsigned chunk = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk > chunks || (Emit && chunk == chunks))
        return;
    // Positions outside [q_begin, q_end) (a dense tail) are not planned.
    const unsigned first = min(q_end, max(q_begin, chunk * kPlanChunk));
    const unsigned last = max(first, min(q_end, chunk * kPlanChunk + kPlanChunk));
    const auto c = gpu_dataflow::pack_rows<Emit, PlanLanes>(ptr, {m, reverse}, first, last, split_min,
                                                 Emit ? prefix[chunk] : plan_counts{}, words, segments);
    if constexpr (!Emit)
        counts[chunk] = c;
}

} // namespace

dataflow_device_plan build_dataflow_device_plan(int m, bool reverse, int nnz, const int *rowptr,
                                                int split_min, const float *__restrict__ diag,
                                                const double *inv_scale2, unsigned plan_lanes,
                                                int planned_rows) {
    if (planned_rows < 0 || planned_rows > m) planned_rows = m;
    // Forward plans rows [0, planned_rows); back plans the same rows, which
    // are positions [m - planned_rows, m) of its reversed order.
    const unsigned q_begin = reverse ? unsigned(m - planned_rows) : 0u;
    const unsigned q_end = reverse ? unsigned(m) : unsigned(planned_rows);
    if (m < 0 || m == INT_MAX || nnz < m || !rowptr || split_min < -1 ||
        static_cast<bool>(diag) != static_cast<bool>(inv_scale2) ||
        (plan_lanes != 8 && plan_lanes != 32))
        throw std::invalid_argument("GPU dataflow plan: invalid dimensions or metadata");
    dataflow_device_plan result;
    // Preserve the temporary cudaFree order while allocating at first use.
    cuda_buffer<char, dataflow_check> scratch;
    cuda_buffer<plan_counts, dataflow_check> counts_by_chunk;
    cuda_buffer<plan_row_stats, dataflow_check> statistics;
    statistics.allocate(1, "allocate statistics");
    dataflow_check(cudaMemset(statistics.get(), 0, sizeof(plan_row_stats)), "clear statistics");
    plan_row_statistics<<<std::max(1, m / 256 + (m % 256 != 0)), 256>>>(m, nnz, rowptr, diag, inv_scale2,
                                                                        statistics.get(), planned_rows);
    dataflow_check(cudaGetLastError(), "validate/statistics launch");
    plan_row_stats host_stats{};
    dataflow_check(cudaMemcpy(&host_stats, statistics.get(), sizeof(host_stats), cudaMemcpyDeviceToHost),
                   "download statistics");
    result.host_download_bytes = sizeof(host_stats);
    if (host_stats.error)
        throw std::invalid_argument(
            "GPU dataflow plan: invalid CSR row pointers or fp16 metadata (code " +
            std::to_string(host_stats.error) + ")");
    result.stats = host_stats;
    if (split_min == -1)
        split_min = host_stats.split_threshold(nnz);
    if (m == 0)
        return result;
    const int chunks = m / kPlanChunk + (m % kPlanChunk != 0);
    counts_by_chunk.allocate(2 * static_cast<std::size_t>(chunks + 1), "allocate tile counts and prefixes");
    plan_counts *prefix = counts_by_chunk.get() + chunks + 1;
    auto count = [&]<unsigned Lanes>() {
        pack_chunks<false, Lanes><<<(chunks + 128) / 128, 128>>>(m, reverse, rowptr, split_min, chunks,
                                                             counts_by_chunk.get(), nullptr, nullptr, nullptr,
                                                             q_begin, q_end);
    };
    if (plan_lanes == 8) count.template operator()<8>();
    else count.template operator()<32>();
    dataflow_check(cudaGetLastError(), "tile count launch");
    std::size_t scratch_bytes = 0;
    dataflow_check(cub::DeviceScan::ExclusiveScan(nullptr, scratch_bytes, counts_by_chunk.get(), prefix,
                                                  add_plan_counts{}, plan_counts{}, chunks + 1),
                   "tile scan size");
    scratch.allocate(scratch_bytes, "allocate tile scan scratch");
    dataflow_check(cub::DeviceScan::ExclusiveScan(scratch.get(), scratch_bytes, counts_by_chunk.get(), prefix,
                                                  add_plan_counts{}, plan_counts{}, chunks + 1),
                   "tile prefix scan");
    plan_counts counts{};
    dataflow_check(cudaMemcpy(&counts, prefix + chunks, sizeof(counts), cudaMemcpyDeviceToHost),
                   "download final counts");
    result.host_download_bytes += sizeof(counts);
    if (counts.words > UINT_MAX || counts.segments >= gpu_dataflow::special_bit ||
        counts.slots + static_cast<unsigned long long>(m) >= gpu_dataflow::special_bit)
        throw std::overflow_error("GPU dataflow plan exceeds 32-bit capacity");
    result.n_batches = static_cast<int>(counts.words / plan_lanes);
    result.n_slots = static_cast<int>(counts.slots);
    result.n_split = static_cast<int>(counts.segments - counts.slots);
    cuda_buffer<unsigned, dataflow_check> words;
    cuda_buffer<gpu_dataflow::segment, dataflow_check> segments;
    words.allocate(counts.words, "allocate tasks");
    segments.allocate(counts.segments, "allocate segments");
    auto emit = [&]<unsigned Lanes>() {
        pack_chunks<true, Lanes><<<(chunks + 127) / 128, 128>>>(m, reverse, rowptr, split_min, chunks, nullptr,
                                                            prefix, words.get(), segments.get(), q_begin, q_end);
    };
    if (plan_lanes == 8) emit.template operator()<8>();
    else emit.template operator()<32>();
    dataflow_check(cudaGetLastError(), "emit chunk plans");
    dataflow_check(cudaDeviceSynchronize(), "finish device plan");
    result.bytes = words.bytes() + segments.bytes();
    result.words = words.release();
    result.segments = segments.release();
    return result;
}
} // namespace apxchol::detail
