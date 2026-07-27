#include "apxchol/solver/sptrsv/cuda_levelset.h"

// GPU level-set triangular solve kernels. See cuda_levelset.h for the rationale
// (O(n) schedule vs cuSPARSE's O(nnz) analysis buffer). One kernel launch per
// level; the level partition guarantees every off-diagonal neighbour is in an
// earlier level, so within a level ALL work is independent -- which lets us
// parallelise per-NONZERO (a warp per row) on top of the O(n) schedule, with no
// per-edge dependency tracking. That recovers intra-row parallelism for the fat
// hub rows of social-graph factors without cuSPARSE's O(nnz) memory.
namespace apxchol {

namespace {

using VAL = sptrsv_gpu_value_t;

// One WARP per row of a level. The 32 lanes stride over the row's nonzeros,
// accumulate partial sums, then warp-reduce. Off-diagonal out[j] were finalised in
// earlier levels (prior launches on this stream); out[i] is the warp's exclusive
// write -> race-free. Fat rows (hub degree thousands) now get 32-wide parallelism
// and coalesced colidx/vals reads, instead of one thread grinding them serially.
__global__ void levelset_warp_kernel(const int* __restrict__ rowptr,
                                     const int* __restrict__ colidx,
                                     const VAL* __restrict__ vals,
                                     const int* __restrict__ row_order,
                                     int lvl_begin, int lvl_end,
                                     const VAL* __restrict__ rhs,
                                     VAL* __restrict__ out) {
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = gtid >> 5;       // global warp id -> one row
    const int lane = gtid & 31;
    const int idx = lvl_begin + warp;
    if (idx >= lvl_end) return;
    const int i = row_order[idx];
    const int beg = rowptr[i], end = rowptr[i + 1];
    VAL mysum = VAL(0);
    VAL mydiag = VAL(0);              // only the lane that hits j==i sets it (else 0)
    for (int p = beg + lane; p < end; p += 32) {
        const int j = colidx[p];
        const VAL v = vals[p];
        if (j == i) mydiag = v;
        else        mysum += v * out[j];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        mysum  += __shfl_down_sync(0xffffffffu, mysum,  o);
        mydiag += __shfl_down_sync(0xffffffffu, mydiag, o);   // sum: one nonzero term
    }
    if (lane == 0) out[i] = (rhs[i] - mysum) / mydiag;
}

} // namespace

void levelset_solve(cudaStream_t stream,
                    const int* rowptr, const int* colidx, const VAL* vals,
                    const int* row_order, const int* level_ptr, int num_levels,
                    const VAL* rhs, VAL* out) {
    constexpr int block = 256;        // 8 warps per block
    constexpr int warps_per_block = block / 32;
    for (int l = 0; l < num_levels; ++l) {
        const int begin = level_ptr[l], end = level_ptr[l + 1];
        const int rows = end - begin;
        if (rows <= 0) continue;
        const int grid = (rows + warps_per_block - 1) / warps_per_block;
        levelset_warp_kernel<<<grid, block, 0, stream>>>(
            rowptr, colidx, vals, row_order, begin, end, rhs, out);
    }
}

namespace {

// y = A*x; A values are fp32 (promoted to fp64 per product), x/y fp64, accumulate
// fp64. One warp per row strides the row's nonzeros + warp-reduces. No dependencies
// (SpMV is embarrassingly parallel), so this is a single launch.
__global__ void spmv_f32A_f64_kernel(const int* __restrict__ rowptr,
                                     const int* __restrict__ colidx,
                                     const float* __restrict__ valsf32,
                                     const double* __restrict__ x,
                                     double* __restrict__ y, int n) {
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = gtid >> 5;
    const int lane = gtid & 31;
    if (row >= n) return;
    const int beg = rowptr[row], end = rowptr[row + 1];
    double sum = 0.0;
    for (int p = beg + lane; p < end; p += 32)
        sum += static_cast<double>(valsf32[p]) * x[colidx[p]];   // promote A, fp64 MAC
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, o);
    if (lane == 0) y[row] = sum;
}

} // namespace

void spmv_f32A_f64(cudaStream_t stream, const int* rowptr, const int* colidx,
                   const float* valsf32, const double* x, double* y, int n) {
    constexpr int block = 256;
    const int grid = (n + (block / 32) - 1) / (block / 32);
    spmv_f32A_f64_kernel<<<grid, block, 0, stream>>>(rowptr, colidx, valsf32, x, y, n);
}

} // namespace apxchol
