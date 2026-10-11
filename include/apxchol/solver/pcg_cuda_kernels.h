#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "apxchol/solver/detail/gpu_pcg.h"

// CUDA launch boundary for the common PCG kernels. Both operator storage
// precisions accumulate in FP64. Reductions use a fixed grid and tree, without
// floating-point atomics; launch geometry depends only on the input size.
namespace apxchol::pcg_cuda {

using solve_state = detail::gpu_pcg::solve_state<double, double>;

// Private experiment ABI. Input arrays are valid borrowed host arrays of
// lengths n+1,nnz,nnz,n; permutation is the factor's validated bijection.
// Accept only unique sorted fully paired CSC. Canonical LOWER values and
// sorted permuted rows match the existing host constructor, including zeros.
// precision: -1=AUTO exact-fp32,0=fp64,1=forced-fp32. Unsupported input returns
// false without publishing output. CUDA/allocation errors throw after freeing
// temporary allocations. Success transfers cudaFree ownership to the caller.
struct operator_csr {
    int* row_ptr = nullptr;
    int* col_idx = nullptr;
    double* values_f64 = nullptr;
    float* values_f32 = nullptr;
    int nnz = 0;
    bool fp32 = false;
};
bool try_build_permuted_operator_csr(
    int n, int nnz, const int* host_outer, const int* host_inner,
    const double* host_values, const std::uint32_t* host_perm,
    int precision, operator_csr& output);

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

// Nearest power of two to average nnz/row, capped at one warp. The measured
// mapping and APXCHOL_GPU_SPMV_LANES experiment override are retained.
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
              const double* vals, const double* x, double* y, double* part, int lanes, const solve_state* state);
void spmv_pAp(cudaStream_t stream, int n, const int* rowptr, const int* colidx,
              const float* vals, const double* x, double* y, double* part, int lanes, const solve_state* state);

/// x += alpha p ; r -= alpha Ap ; per-block partials of r.r (on the UPDATED r)
/// into part[0..pcg_blocks(n)).
void update_xr(cudaStream_t stream, std::int64_t n, double* x, const double* p,
               double* r, const double* Ap, const solve_state* state, double* part);

/// Per-block partials of a.b into part[0..pcg_blocks(n)).
void dot(cudaStream_t stream, std::int64_t n, const double* a, const double* b, double* part, const solve_state* state);

/// p = z + beta p.
void update_p(cudaStream_t stream, std::int64_t n, double* p, const double* z, const solve_state* state);

// Fold partials in the fixed tree and advance the device recurrence.
void finalize(cudaStream_t stream, const double* part, int blocks, solve_state* state,
              unsigned mode, unsigned iteration, double bnorm);

} // namespace apxchol::pcg_cuda
