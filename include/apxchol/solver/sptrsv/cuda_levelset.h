#pragma once
#include <cuda_runtime.h>

// Custom GPU level-set sparse triangular solve — a memory-frugal alternative to
// cuSPARSE SpSV. cuSPARSE's analysis buffer is O(nnz) (~23 B/nnz × 2 solves →
// multiple GB on the giant social factors); a level-set needs only the O(n)
// schedule (row order + level boundaries), since the level structure itself is
// intrinsic to the matrix. Each level is one kernel launch over its rows; every
// thread solves one row by GATHERING from already-finished rows (guaranteed by
// the level partition + same-stream kernel ordering), so no atomics are needed.
//
// Defined in src/cuda_levelset.cu (nvcc — the .cpp TUs are host-compiled and
// can't host __global__), mirroring cuda_cast.cu. The value type matches the
// factor width (fp32 under -DAPXCHOL_SPTRSV_FP32), declared here without pulling
// in sparse_csc.h (host C++23, which the C++20-pinned .cu can't compile).
namespace apxchol {

#ifdef APXCHOL_SPTRSV_FP32
using sptrsv_gpu_value_t = float;
#else
using sptrsv_gpu_value_t = double;
#endif

/// Solve a triangular system `T out = rhs` by level sets, where T is given in CSR
/// (row i: entries at colidx[rowptr[i]..rowptr[i+1]), including the diagonal). The
/// off-diagonal neighbours of every row lie in strictly earlier levels, so the
/// gather `out[i] = (rhs[i] - Σ_{j≠i} T[i,j]·out[j]) / T[i,i]` is race-free.
///   row_order : rows sorted by level (device)
///   level_ptr : level boundaries into row_order, size num_levels+1 (HOST array —
///               read to size each per-level launch)
/// Works for forward (CSR of L, lower-tri, levels ascending) and back (CSR of
/// L^T, upper-tri, levels descending) by passing the matching CSR + schedule.
/// rhs and out must be distinct buffers. One WARP per row (32 lanes stride the
/// row's nonzeros + warp-reduce) so fat hub rows get intra-row parallelism.
void levelset_solve(cudaStream_t stream,
                    const int* rowptr, const int* colidx, const sptrsv_gpu_value_t* vals,
                    const int* row_order, const int* level_ptr, int num_levels,
                    const sptrsv_gpu_value_t* rhs, sptrsv_gpu_value_t* out);

/// y = A*x with A's CSR values stored in fp32 (memory saving; LOSSLESS when A is
/// exactly fp32-representable, e.g. unweighted/pattern graphs) but each product
/// promoted to fp64 and accumulated in fp64 -- so x stays fp64 and the result is
/// fp64-accurate. Unlike a pure-fp32 cuSPARSE SpMV (fp32 vectors + fp32 accumulate),
/// this does NOT floor the PCG recurrence: an exact-fp32 operator gives the same Ap
/// as fp64 storage. One warp per row.
void spmv_f32A_f64(cudaStream_t stream, const int* rowptr, const int* colidx,
                   const float* valsf32, const double* x, double* y, int n);

} // namespace apxchol
