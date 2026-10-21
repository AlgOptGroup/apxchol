#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <array>
#include "apxchol/solver/detail/gpu_pcg.h"

// CUDA launch boundary for the common PCG kernels. Both operator storage
// precisions accumulate in FP64. Reductions use a fixed grid and tree, without
// floating-point atomics; launch geometry depends only on the input size.
namespace apxchol::pcg_cuda {

using solve_state = detail::gpu_pcg::solve_state<double, double>;
struct workspace : detail::gpu_pcg::workspace<double, double> {
    double* rhs = nullptr;
};
enum exit_flag : unsigned {
    kVerifySolution = 1, kSolutionOverflow = 2, kSolutionUnderflow = 4
};

// Private experiment ABI. Input arrays are valid borrowed host arrays of
// lengths n+1,nnz,nnz,n; permutation is the factor's validated bijection.
// Accept only unique sorted fully paired CSC. Canonical LOWER values and
// sorted permuted rows match the existing host constructor, including zeros.
// precision: -1=AUTO exact-fp32,0=fp64,1=require-exact-fp32. Layout failures return
// false without publishing output. Requiring FP32 with inexact values throws
// std::invalid_argument. CUDA/allocation errors also throw; temporary allocations
// are freed on failure. Success transfers cudaFree ownership to the caller.
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
/// buffer the caller provides must hold Columns * kMaxBlocks doubles.
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

// Keep the uploaded RHS intact; initialize the scaled recurrence and zero x.
template<unsigned Columns = 1>
void initialize(cudaStream_t stream, std::int64_t n, workspace vectors,
                std::array<double, Columns> scales);

// Launch one shared vector operation; only multiply reads the operator.
// Partials use pcg_blocks(n * lanes) for multiply, pcg_blocks(n) otherwise.
template<unsigned Columns = 1>
void step(cudaStream_t stream, detail::gpu_pcg::operation op, std::int64_t n,
          workspace vectors, const operator_csr& matrix, int lanes);

// Fold partials in the fixed tree and advance the device recurrence.
template<unsigned Columns = 1>
void finalize(cudaStream_t stream, const double* part, int blocks, solve_state* state,
              unsigned mode, unsigned iteration, std::array<double, Columns> bnorm, unsigned window = 0);

// Restore the returned solution on device and check the original system,
// independently of the recurrence's exit reason. Reuses the dead PCG scratch.
// rr receives the relative residual; active stores exit_flag bits.
template<unsigned Columns = 1>
void finish(cudaStream_t stream, std::int64_t n, workspace vectors,
            const operator_csr& matrix, int lanes, bool laplacian,
            std::array<double, Columns> scales, std::array<double, Columns> bnorm);

} // namespace apxchol::pcg_cuda
