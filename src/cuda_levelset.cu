#include "apxchol/solver/sptrsv/cuda_levelset.h"
#include <cuda_fp16.h>

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

// fp16 per-column-scaled storage (APXCHOL_GPU_SPTRSV_FP16=1; see
// cuda_levelset.h / cuda_host.h for the contract). Same warp-per-row scheme;
// the differences are per entry (a half -> float widen before the FMA; the
// diagonal slot is skipped instead of captured) and per row (the divisor is
// the fp32 diag[i], the rhs is scaled once by in_scale[i] when given).
__global__ void levelset_warp_kernel_fp16(const int* __restrict__ rowptr,
                                          const int* __restrict__ colidx,
                                          const __half* __restrict__ vals,
                                          const float* __restrict__ diag,
                                          const float* __restrict__ in_scale,
                                          const int* __restrict__ row_order,
                                          int lvl_begin, int lvl_end,
                                          const VAL* __restrict__ rhs,
                                          VAL* __restrict__ out) {
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = gtid >> 5;
    const int lane = gtid & 31;
    const int idx = lvl_begin + warp;
    if (idx >= lvl_end) return;
    const int i = row_order[idx];
    const int beg = rowptr[i], end = rowptr[i + 1];
    VAL mysum = VAL(0);
    for (int p = beg + lane; p < end; p += 32) {
        const int j = colidx[p];
        if (j == i) continue;                          // diagonal slot: unread (diag[] below)
        mysum += static_cast<VAL>(__half2float(vals[p])) * out[j];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        mysum += __shfl_down_sync(0xffffffffu, mysum, o);
    if (lane == 0) {
        const VAL b = in_scale ? rhs[i] * static_cast<VAL>(in_scale[i]) : rhs[i];
        out[i] = (b - mysum) / static_cast<VAL>(diag[i]);
    }
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

void levelset_solve_fp16(cudaStream_t stream,
                         const int* rowptr, const int* colidx, const unsigned short* vals16,
                         const float* diag, const float* in_scale,
                         const int* row_order, const int* level_ptr, int num_levels,
                         const VAL* rhs, VAL* out) {
    constexpr int block = 256;
    constexpr int warps_per_block = block / 32;
    const __half* vals = reinterpret_cast<const __half*>(vals16);
    for (int l = 0; l < num_levels; ++l) {
        const int begin = level_ptr[l], end = level_ptr[l + 1];
        const int rows = end - begin;
        if (rows <= 0) continue;
        const int grid = (rows + warps_per_block - 1) / warps_per_block;
        levelset_warp_kernel_fp16<<<grid, block, 0, stream>>>(
            rowptr, colidx, vals, diag, in_scale, row_order, begin, end, rhs, out);
    }
}

} // namespace apxchol
