#pragma once
#include "apxchol/solver/detail/gpu_dataflow_plan.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace apxchol {

using sptrsv_gpu_value_t = float;
inline constexpr int kDataflowBlock = 256;
int dataflow_prefetch_depth();
int dataflow_grid_size(bool fp16);

// Each sweep initializes atomic FP32 words and solves dependency-free rows,
// then resident warps consume topologically ordered batches. A word contains
// either a value or the pending NaN marker; publishing needs no separate flag
// or memory fence. The shared CUDA/Metal worker is in src/gpu_dataflow.inc.
//
// Dynamic ticket claims ensure a dependency's earlier batch is already running
// or complete. Within a batch, ready rows publish while other rows keep polling.
// Multi-chunk rows and segment finalizers occupy a full warp. No dependency can
// be hidden behind a waiting consumer, including within one warp.
//
// The owner supplies tag_count >= m + number of segment partials, and one
// ticket word. Initialization resets both on every invocation,
// including graph replay. rhs and out must be distinct; sweeps share workspace
// only when ordered on the same stream. Factor CSR includes its diagonal.
void dataflow_solve(cudaStream_t stream, int m, bool reverse,
    const int* rowptr, const int* colidx, const float* vals,
    const unsigned* words, const detail::gpu_dataflow::segment* segments, unsigned batches,
    unsigned* tags, unsigned tag_count, unsigned* ticket, int grid, const float* rhs, float* out, const unsigned* active = nullptr,
    const double* rhs64 = nullptr, double* out64 = nullptr);

// PCG may supply rhs64/out64 to fuse its vector conversions into the sweep.

// FP16 keeps the existing compensated column scaling: the diagonal is FP32,
// and optional backward RHS scaling is multiplied in FP64 before narrowing.
void dataflow_solve_fp16(cudaStream_t stream, int m, bool reverse,
    const int* rowptr, const int* colidx, const __half* vals, const float* diag, const double* scale,
    const unsigned* words, const detail::gpu_dataflow::segment* segments, unsigned batches,
    unsigned* tags, unsigned tag_count, unsigned* ticket, int grid, const float* rhs, float* out, const unsigned* active = nullptr,
    const double* rhs64 = nullptr, double* out64 = nullptr);

namespace detail {
// Returned allocations transfer to the caller. Only fixed-size statistics and
// counts cross to the host; trusted device-owned factors remain on the device.
struct dataflow_device_plan {
    static constexpr int kBuckets = 12;
    unsigned* words = nullptr;
    gpu_dataflow::segment* segments = nullptr;
    int n_batches = 0, n_slots = 0, n_split = 0, max_len = 0;
    std::uint64_t rows_ge[kBuckets]{};
    std::uint64_t nnz_ge[kBuckets]{};
    std::size_t bytes = 0, host_download_bytes = 0;
};

// split_min: -1 derives the threshold from row lengths, 0 disables splitting.
// The private factor finalizer guarantees sorted triangular CSR with diagonal
// at the proper end. Optional FP16 metadata is validated in the same row pass.
dataflow_device_plan build_dataflow_device_plan(
    int m, bool reverse, int nnz, const int* rowptr, int split_min,
    const float* diag = nullptr, const double* inv_scale2 = nullptr);
} // namespace detail
} // namespace apxchol
