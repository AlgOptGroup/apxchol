#include "apxchol/solver/pcg_cuda_kernels.h"

#include <cub/cub.cuh>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

// The GPU-resident PCG's own kernels (see pcg_cuda_kernels.h for the contract):
// CSR SpMV with p.Ap folded in, the fused vector passes, and DETERMINISTIC
// reductions -- fixed grid, fixed per-thread strided assignment, fixed-order
// warp-shuffle + shared-memory tree per block, one partial per block, fixed-
// order single-block final reduce. No floating-point atomics anywhere.
namespace apxchol::pcg_cuda {

namespace {

constexpr int kWarps = kBlockThreads / 32;

// Deterministic block sum: warp shuffle tree (fixed order) -> one value per
// warp in shared memory -> the first warp's shuffle tree over the kWarps
// values (fixed order). Every thread of the block must call it (full masks);
// the result is valid in thread 0.
__device__ __forceinline__ double block_reduce_sum(double v, double* sh) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    double r = 0.0;
    if (warp == 0) {
        r = lane < kWarps ? sh[lane] : 0.0;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) r += __shfl_down_sync(0xffffffffu, r, o);
    }
    return r;
}

// y = A x with LANES threads per row; per-block partial of x.y into
// part[blockIdx.x]. The row-group loop bound (`base < n`) is uniform over the
// block, so every thread reaches every sub-warp shuffle (full mask, width
// LANES) and the block reduce.
template <class VAL, int LANES>
__global__ void __launch_bounds__(kBlockThreads)
spmv_pAp_kernel(int n, const int* __restrict__ rowptr, const int* __restrict__ colidx,
                const VAL* __restrict__ vals, const double* __restrict__ x,
                double* __restrict__ y, double* __restrict__ part) {
    __shared__ double sh[kWarps];
    constexpr int ROWS_PER_BLOCK = kBlockThreads / LANES;
    const int lane      = threadIdx.x & (LANES - 1);
    const int local_row = threadIdx.x / LANES;
    double acc = 0.0;
    for (int base = blockIdx.x * ROWS_PER_BLOCK; base < n; base += gridDim.x * ROWS_PER_BLOCK) {
        const int row = base + local_row;
        double sum = 0.0;
        if (row < n) {
            const int beg = rowptr[row], end = rowptr[row + 1];
            for (int p = beg + lane; p < end; p += LANES)
                sum += static_cast<double>(vals[p]) * x[colidx[p]];   // promote A, fp64 MAC
        }
        #pragma unroll
        for (int o = LANES / 2; o > 0; o >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, o, LANES);
        if (row < n && lane == 0) { y[row] = sum; acc += x[row] * sum; }
    }
    const double bs = block_reduce_sum(acc, sh);
    if (threadIdx.x == 0) part[blockIdx.x] = bs;
}

__global__ void __launch_bounds__(kBlockThreads)
update_xr_kernel(std::int64_t n, double* __restrict__ x, const double* __restrict__ p,
                 double* __restrict__ r, const double* __restrict__ Ap, double alpha,
                 double* __restrict__ part) {
    __shared__ double sh[kWarps];
    double q = 0.0;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        x[i] += alpha * p[i];
        const double ri = r[i] - alpha * Ap[i];
        r[i] = ri;
        q += ri * ri;
    }
    const double bs = block_reduce_sum(q, sh);
    if (threadIdx.x == 0) part[blockIdx.x] = bs;
}

__global__ void __launch_bounds__(kBlockThreads)
dot_kernel(std::int64_t n, const double* __restrict__ a, const double* __restrict__ b,
           double* __restrict__ part) {
    __shared__ double sh[kWarps];
    double q = 0.0;
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x)
        q += a[i] * b[i];
    const double bs = block_reduce_sum(q, sh);
    if (threadIdx.x == 0) part[blockIdx.x] = bs;
}

__global__ void __launch_bounds__(kBlockThreads)
update_p_kernel(std::int64_t n, double* __restrict__ p, const double* __restrict__ z, double beta) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x)
        p[i] = z[i] + beta * p[i];
}

// One block: thread t sums part[t], part[t + kBlockThreads], ... in order,
// then the deterministic block tree.
__global__ void __launch_bounds__(kBlockThreads)
reduce_partials_kernel(const double* __restrict__ part, int blocks, double* __restrict__ out) {
    __shared__ double sh[kWarps];
    double s = 0.0;
    for (int i = threadIdx.x; i < blocks; i += kBlockThreads) s += part[i];
    const double bs = block_reduce_sum(s, sh);
    if (threadIdx.x == 0) out[0] = bs;
}

template <class VAL>
void spmv_pAp_impl(cudaStream_t stream, int n, const int* rowptr, const int* colidx,
                   const VAL* vals, const double* x, double* y, double* part, int lanes) {
    const int grid = pcg_blocks(static_cast<std::int64_t>(n) * lanes);
    switch (lanes) {
        case 1:  spmv_pAp_kernel<VAL, 1> <<<grid, kBlockThreads, 0, stream>>>(n, rowptr, colidx, vals, x, y, part); break;
        case 2:  spmv_pAp_kernel<VAL, 2> <<<grid, kBlockThreads, 0, stream>>>(n, rowptr, colidx, vals, x, y, part); break;
        case 4:  spmv_pAp_kernel<VAL, 4> <<<grid, kBlockThreads, 0, stream>>>(n, rowptr, colidx, vals, x, y, part); break;
        case 8:  spmv_pAp_kernel<VAL, 8> <<<grid, kBlockThreads, 0, stream>>>(n, rowptr, colidx, vals, x, y, part); break;
        case 16: spmv_pAp_kernel<VAL, 16><<<grid, kBlockThreads, 0, stream>>>(n, rowptr, colidx, vals, x, y, part); break;
        default: spmv_pAp_kernel<VAL, 32><<<grid, kBlockThreads, 0, stream>>>(n, rowptr, colidx, vals, x, y, part); break;
    }
}

} // namespace

void spmv_pAp(cudaStream_t stream, int n, const int* rowptr, const int* colidx,
              const double* vals, const double* x, double* y, double* part, int lanes) {
    spmv_pAp_impl(stream, n, rowptr, colidx, vals, x, y, part, lanes);
}

void spmv_pAp(cudaStream_t stream, int n, const int* rowptr, const int* colidx,
              const float* vals, const double* x, double* y, double* part, int lanes) {
    spmv_pAp_impl(stream, n, rowptr, colidx, vals, x, y, part, lanes);
}

void update_xr(cudaStream_t stream, std::int64_t n, double* x, const double* p,
               double* r, const double* Ap, double alpha, double* part) {
    update_xr_kernel<<<pcg_blocks(n), kBlockThreads, 0, stream>>>(n, x, p, r, Ap, alpha, part);
}

void dot(cudaStream_t stream, std::int64_t n, const double* a, const double* b, double* part) {
    dot_kernel<<<pcg_blocks(n), kBlockThreads, 0, stream>>>(n, a, b, part);
}

void update_p(cudaStream_t stream, std::int64_t n, double* p, const double* z, double beta) {
    update_p_kernel<<<pcg_blocks(n), kBlockThreads, 0, stream>>>(n, p, z, beta);
}

void reduce_partials(cudaStream_t stream, const double* part, int blocks, double* out) {
    reduce_partials_kernel<<<1, kBlockThreads, 0, stream>>>(part, blocks, out);
}

namespace {

void operator_check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("GPU operator CSR: ") + cudaGetErrorString(status));
}

// Every temporary owns its allocation until the final, nonthrowing publication.
// cudaFree also retires work using a temporary when an earlier CUDA call throws.
template<class T> struct operator_buffer {
    T* ptr = nullptr;
    explicit operator_buffer(std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::overflow_error("GPU operator CSR allocation overflow");
        if (count) operator_check(cudaMalloc(&ptr, count * sizeof(T)));
    }
    operator_buffer(const operator_buffer&) = delete;
    operator_buffer& operator=(const operator_buffer&) = delete;
    ~operator_buffer() { if (ptr) cudaFree(ptr); }
    void reset() { if (ptr) { operator_check(cudaFree(ptr)); ptr = nullptr; } }
    T* release() { return std::exchange(ptr, nullptr); }
};

struct operator_status {
    // Each count is <= nnz <= INT_MAX: adding packed counts cannot carry from
    // the low (lower triangle) half into the high (upper triangle) half.
    unsigned long long triangles;
    unsigned flags; // bit 0: unsupported layout; bit 1: inexact fp32 lower value
};

__global__ void __launch_bounds__(kBlockThreads)
operator_count_columns(int n, int nnz, const int* outer, const int* inner,
                       const double* values, const std::uint32_t* perm,
                       int* counts, operator_status* status) {
    const int lane = threadIdx.x & 31;
    const std::int64_t warp = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x / 32;
    unsigned long long triangles = 0;
    bool bad = outer[0] != 0 || outer[n] != nnz, inexact = false;
    for (std::int64_t col = warp; col < n; col += stride) {
        const int begin = outer[col], end = outer[col + 1];
        if (begin < 0 || end < begin || end > nnz) { bad = true; continue; }
        if (lane == 0) counts[perm[col]] = end - begin;
        for (std::int64_t p = static_cast<std::int64_t>(begin) + lane; p < end; p += 32) {
            const int row = inner[p];
            if (row < 0 || row >= n || (p > begin && inner[p - 1] >= row)) bad = true;
            if (row < col) triangles += 1ULL << 32;
            else {
                if (row > col) ++triangles;
                if (static_cast<double>(__double2float_rn(values[p])) != values[p]) inexact = true;
            }
        }
    }
    // All 256 lanes participate, including physical tail warps with no column.
    const int any_bad = __syncthreads_or(bad);
    const int any_inexact = __syncthreads_or(inexact);
    using reduce = cub::BlockReduce<unsigned long long, kBlockThreads>;
    __shared__ typename reduce::TempStorage scratch;
    const unsigned long long sum = reduce(scratch).Sum(triangles);
    if (threadIdx.x == 0) {
        atomicAdd(&status->triangles, sum);
        if (any_bad || any_inexact)
            atomicOr(&status->flags, static_cast<unsigned>(any_bad | (any_inexact << 1)));
    }
}

__global__ void __launch_bounds__(kBlockThreads)
operator_fill_columns(int n, const int* outer, const int* inner,
                      const double* input, const std::uint32_t* perm,
                      std::uint64_t* keys, double* values, operator_status* status) {
    const int lane = threadIdx.x & 31;
    const std::int64_t warp = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x / 32;
    bool bad = false;
    for (std::int64_t col = warp; col < n; col += stride) {
        for (std::int64_t p = static_cast<std::int64_t>(outer[col]) + lane;
             p < outer[col + 1]; p += 32) {
            const int row = inner[p];
            double value = input[p];
            if (row < col) {
                int lo = outer[row], hi = outer[row + 1];
                while (lo < hi) {
                    const int mid = lo + (hi - lo) / 2;
                    if (inner[mid] < col) lo = mid + 1;
                    else hi = mid;
                }
                if (lo == outer[row + 1] || inner[lo] != col) bad = true;
                else value = input[lo];
            }
            // Input ordinal p is a unique fixed slot. Sorting this integer key
            // orders CSR rows and their columns without any floating arithmetic.
            keys[p] = (static_cast<std::uint64_t>(perm[col]) << 32) | perm[row];
            values[p] = value;
        }
    }
    if (__syncthreads_or(bad) && threadIdx.x == 0) atomicOr(&status->flags, 1U);
}

__global__ void __launch_bounds__(kBlockThreads)
operator_decode_columns(int nnz, const std::uint64_t* keys, int* columns) {
    for (std::int64_t p = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         p < nnz; p += static_cast<std::int64_t>(gridDim.x) * blockDim.x)
        columns[p] = static_cast<int>(keys[p] & 0xffffffffULL);
}

__global__ void __launch_bounds__(kBlockThreads)
operator_narrow_values(int nnz, const double* input, float* output) {
    for (std::int64_t p = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         p < nnz; p += static_cast<std::int64_t>(gridDim.x) * blockDim.x)
        output[p] = __double2float_rn(input[p]);
}

} // namespace

bool try_build_permuted_operator_csr(
        int n, int nnz, const int* host_outer, const int* host_inner,
        const double* host_values, const std::uint32_t* host_perm,
        int precision, operator_csr& output) {
    if (n < 0 || nnz < 0 || precision < -1 || precision > 1 ||
        !host_outer || (nnz && (!host_inner || !host_values)) || (n && !host_perm))
        return false;
    if (output.row_ptr || output.col_idx || output.values_f64 || output.values_f32)
        throw std::invalid_argument("GPU operator CSR output must not own allocations");
    operator_buffer<int> outer(static_cast<std::size_t>(n) + 1), inner(nnz), counts(n);
    operator_buffer<double> input(nnz);
    operator_buffer<std::uint32_t> perm(n);
    operator_buffer<operator_status> status(1);
    operator_check(cudaMemcpy(outer.ptr, host_outer, (static_cast<std::size_t>(n) + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (nnz) {
        operator_check(cudaMemcpy(inner.ptr, host_inner, static_cast<std::size_t>(nnz) * sizeof(int), cudaMemcpyHostToDevice));
        operator_check(cudaMemcpy(input.ptr, host_values, static_cast<std::size_t>(nnz) * sizeof(double), cudaMemcpyHostToDevice));
    }
    if (n) operator_check(cudaMemcpy(perm.ptr, host_perm, static_cast<std::size_t>(n) * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    operator_check(cudaMemset(status.ptr, 0, sizeof(operator_status)));
    const int column_blocks = pcg_blocks(static_cast<std::int64_t>(n) * 32);
    operator_count_columns<<<column_blocks, kBlockThreads>>>(n, nnz, outer.ptr, inner.ptr, input.ptr, perm.ptr, counts.ptr, status.ptr);
    operator_check(cudaGetLastError());
    operator_status observed{};
    operator_check(cudaMemcpy(&observed, status.ptr, sizeof(observed), cudaMemcpyDeviceToHost));
    if ((observed.flags & 1U) || (observed.triangles & 0xffffffffULL) != (observed.triangles >> 32))
        return false;
    const bool fp32 = precision == 1 || (precision == -1 && !(observed.flags & 2U));

    operator_buffer<std::uint64_t> keys_a(nnz);
    operator_buffer<double> values_a(nnz);
    if (nnz) {
        operator_fill_columns<<<column_blocks, kBlockThreads>>>(n, outer.ptr, inner.ptr, input.ptr, perm.ptr, keys_a.ptr, values_a.ptr, status.ptr);
        operator_check(cudaGetLastError());
        operator_check(cudaMemcpy(&observed, status.ptr, sizeof(observed), cudaMemcpyDeviceToHost));
        if (observed.flags & 1U) return false;
    }
    // The two completed status copies precede retirement of input storage.
    // Do not overlap the full CSC input with both radix-sort key/value buffers.
    outer.reset(); inner.reset(); input.reset(); perm.reset(); status.reset();
    operator_buffer<int> rows(static_cast<std::size_t>(n) + 1);
    if (n) {
        std::size_t bytes = 0;
        operator_check(cub::DeviceScan::ExclusiveSum(nullptr, bytes, counts.ptr, rows.ptr, n));
        operator_buffer<unsigned char> scratch(bytes);
        operator_check(cub::DeviceScan::ExclusiveSum(scratch.ptr, bytes, counts.ptr, rows.ptr, n));
    }
    operator_check(cudaMemcpy(rows.ptr + n, &nnz, sizeof(int), cudaMemcpyHostToDevice));
    counts.reset();
    operator_buffer<std::uint64_t> keys_b(nnz);
    operator_buffer<double> values_b(nnz);
    if (nnz) {
        std::size_t bytes = 0;
        operator_check(cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys_a.ptr, keys_b.ptr, values_a.ptr, values_b.ptr, nnz));
        operator_buffer<unsigned char> scratch(bytes);
        operator_check(cub::DeviceRadixSort::SortPairs(scratch.ptr, bytes, keys_a.ptr, keys_b.ptr, values_a.ptr, values_b.ptr, nnz));
    }
    keys_a.reset(); values_a.reset();
    operator_buffer<int> columns(nnz);
    operator_buffer<float> narrow(fp32 ? nnz : 0);
    if (nnz) {
        operator_decode_columns<<<pcg_blocks(nnz), kBlockThreads>>>(nnz, keys_b.ptr, columns.ptr);
        operator_check(cudaGetLastError());
        if (fp32) {
            operator_narrow_values<<<pcg_blocks(nnz), kBlockThreads>>>(nnz, values_b.ptr, narrow.ptr);
            operator_check(cudaGetLastError());
        }
        operator_check(cudaStreamSynchronize(nullptr));
    }
    // Publication cannot throw; the caller now owns exactly the final arrays.
    output.row_ptr = rows.release();
    output.col_idx = columns.release();
    if (fp32) output.values_f32 = narrow.release();
    else output.values_f64 = values_b.release();
    output.nnz = nnz;
    output.fp32 = fp32;
    return true;
}

} // namespace apxchol::pcg_cuda
