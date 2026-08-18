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

/// The same level-set solve on the FP16 PER-COLUMN-SCALED storage of the
/// opt-in APXCHOL_GPU_SPTRSV_FP16=1 mode (cuda_host.h states the contract):
/// `vals16` are IEEE binary16 bit patterns (uint16_t on the host side, read as
/// __half here) of the column-scaled factor L~ = L D^-1 -- the row's diagonal
/// SLOT is still present in the CSR but is skipped; the diagonal the row
/// divides by is the fp32 `diag[i]` (fp32(L_ii / s_i), plus the column's
/// rounding residual under the default diag_comp) -- and the row's rhs is
/// scaled once by `in_scale[i]` if `in_scale` != nullptr (the back solve
/// passes inv_scale^2 = fp32(1/s_i)^2, folding D^-2 into its input; the
/// forward solve passes nullptr):
///   out[i] = (rhs[i] * in_scale[i] - sum_{j != i} widen(vals16[p]) * out[j]) / diag[i].
/// Every product is formed and accumulated in sptrsv_gpu_value_t (float on
/// the default fp32 build) after the half -> float widen; the fixed warp-
/// reduce order keeps it deterministic per factor. Same schedule / same
/// per-level launch structure as levelset_solve.
void levelset_solve_fp16(cudaStream_t stream,
                         const int* rowptr, const int* colidx, const unsigned short* vals16,
                         const float* diag, const float* in_scale,
                         const int* row_order, const int* level_ptr, int num_levels,
                         const sptrsv_gpu_value_t* rhs, sptrsv_gpu_value_t* out);

} // namespace apxchol
