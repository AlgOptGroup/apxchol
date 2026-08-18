#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Device kernels of the GPU-resident PCG loop (pcg_cuda.h) -- OUR OWN CSR SpMV
// and fused vector kernels with DETERMINISTIC reductions, replacing cuSPARSE
// SpMV and the cuBLAS dot / nrm2 / axpy / scal the loop used to call, so the
// CUDA library build links cudart only (the cuSPARSE SpSV backend of the
// SpTRSV is a separate opt-in, CMake APXCHOL_CUDA_WITH_CUSPARSE). Defined in
// src/cuda_pcg_kernels.cu (nvcc; the .cpp TUs are host-compiled and cannot
// host __global__), mirroring cuda_cast.cu / cuda_levelset.cu.
//
// The kernels mirror the CPU loop's fusion (src/solve.cpp): spmv+pAp (p.Ap
// folded into the SpMV row loop), update_xr (x += alpha p, r -= alpha Ap, r.r
// in one pass), the preconditioner application, r.z, update_p (p = z + beta
// p) -- and, like the CPU's fixed static_chunk partition + per-thread partials
// summed in thread order, every reduction here is deterministic BY
// CONSTRUCTION: a FIXED grid (a pure function of n, pcg_blocks()), every
// thread accumulating a fixed strided subset in a fixed order, a fixed-order
// warp-shuffle + shared-memory tree per block, one partial per block written
// to part[blockIdx.x], and a fixed-order single-block final reduce of the
// partials (reduce_partials) -- no atomicAdd on floating point anywhere, so
// the loop is bit-identical run to run (SolveTest.DeterministicWithSameSeed
// on the CUDA build states it, together with the dataflow SpTRSV).
//
// SpMV: y = A x with A's CSR values fp64 or fp32 (VAL; the fp32 storage is
// the fp32-exact-operator path -- lossless when every value round-trips
// fp32, detected at setup, same as the CPU's op_fp32_ -- each product is
// promoted to fp64 and accumulated in fp64, so x / y and the recurrence stay
// fp64: an exact-fp32 operator gives the same Ap as fp64 storage). LANES
// consecutive threads cooperate on a row (the CUSP "vector CSR" family: 1 =
// scalar-per-row, 32 = warp-per-row, sub-warp groups between), striding its
// nonzeros and sub-warp-reducing; the host picks LANES from the average
// nnz/row (spmv_lanes_for(): the power of two nearest to avg in log2, capped
// at 32, so grids (~5/row) get 4 lanes and the IPM factors (~15/row) 16). Row groups
// are assigned to threads in a fixed grid-stride pattern, so the per-block
// p.Ap partial is deterministic too.
namespace apxchol::pcg_cuda {

/// Threads per block of every kernel here, and the fixed upper bound on the
/// grid (blocks loop grid-stride over the remaining work); the partials
/// buffer the caller provides must hold kMaxBlocks doubles.
inline constexpr int kBlockThreads = 256;
inline constexpr int kMaxBlocks    = 2048;

/// The fixed grid for `units` work items (n elements of a vector kernel, n *
/// LANES for the SpMV): min(ceil(units / kBlockThreads), kMaxBlocks), at
/// least 1. A pure function of its argument -- this is what makes the
/// per-block partials, and hence every reduction, deterministic.
inline int pcg_blocks(std::int64_t units) {
    const std::int64_t b = (units + kBlockThreads - 1) / kBlockThreads;
    return static_cast<int>(b < 1 ? 1 : b > kMaxBlocks ? kMaxBlocks : b);
}

/// Threads per row of the SpMV for an average of avg_nnz_per_row nonzeros
/// per row: the power of two NEAREST to the average in log2 (i.e. the
/// smallest power of two >= avg / sqrt(2)), in [1, 32] -- grids (~5/row)
/// get 4, the IPM factors (~15/row) 16. Measured on the RTX 4090 Laptop
/// (T=1, gpu_pcg_loop ms/iter, 3 reps): grid_2000 lanes 2 / 4 / 8 = 2.85 /
/// 2.87 / 3.11, iter0040 lanes 8 / 16 / 32 = 1.02 / 0.97 / 1.75 -- the
/// smallest-power-of-two->=avg rule (8 for grids) and warp-per-row (32,
/// the old fp32-operator kernel) both lose. Env APXCHOL_GPU_SPMV_LANES
/// overrides (pcg_cuda.h).
inline int spmv_lanes_for(double avg_nnz_per_row) {
    const double target = avg_nnz_per_row / 1.4142135623730951;
    int L = 1;
    while (L < 32 && static_cast<double>(L) < target) L <<= 1;
    return L;
}

/// y = A x, and per-block partials of x.y (the PCG p.Ap) into part[0..blocks)
/// where blocks = pcg_blocks(n * lanes); lanes in {1, 2, 4, 8, 16, 32}
/// (spmv_lanes_for). A: CSR rowptr[n+1] / colidx[nnz] / vals[nnz] (VAL = double
/// or float, promoted per product); x, y, part fp64 device pointers.
void spmv_pAp(cudaStream_t stream, int n, const int* rowptr, const int* colidx,
              const double* vals, const double* x, double* y, double* part, int lanes);
void spmv_pAp(cudaStream_t stream, int n, const int* rowptr, const int* colidx,
              const float* vals, const double* x, double* y, double* part, int lanes);

/// x += alpha p ; r -= alpha Ap ; per-block partials of r.r (on the UPDATED r)
/// into part[0..pcg_blocks(n)).
void update_xr(cudaStream_t stream, std::int64_t n, double* x, const double* p,
               double* r, const double* Ap, double alpha, double* part);

/// Per-block partials of a.b into part[0..pcg_blocks(n)).
void dot(cudaStream_t stream, std::int64_t n, const double* a, const double* b, double* part);

/// p = z + beta p.
void update_p(cudaStream_t stream, std::int64_t n, double* p, const double* z, double beta);

/// Fixed-order sum of part[0..blocks) into the device scalar out[0] (one
/// block, deterministic tree). The caller copies out[0] back (8 bytes).
void reduce_partials(cudaStream_t stream, const double* part, int blocks, double* out);

} // namespace apxchol::pcg_cuda
