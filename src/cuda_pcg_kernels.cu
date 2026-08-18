#include "apxchol/solver/pcg_cuda_kernels.h"

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

} // namespace apxchol::pcg_cuda
