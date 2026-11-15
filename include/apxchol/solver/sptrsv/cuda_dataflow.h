#pragma once
#include "apxchol/solver/detail/gpu_dataflow_plan.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace apxchol::detail {

inline constexpr int kDataflowBlock = 256;
int dataflow_grid_size();

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
// PCG may supply rhs64/out64 to fuse its vector conversions into the sweep.
// A non-null diag selects FP16 values with compensated column scaling;
// otherwise values are FP32. Backward RHS scaling uses FP64 before narrowing.
//
// Dense tail: the last rows [begin, begin + rows == m) of the factor leave the
// plan, because they form one chain of `rows` dependent rows. They are applied as products with the block's exact inverse
// (diagonal included; row-major, inverse_t its transpose), one warp per row
// and column over all SMs: after the dataflow sweep forward, before it back.
// The forward sweep keeps the tail's right-hand side in `scratch` (rows x
// Columns words, row-interleaved), which must not alias `out`, or `tags` when
// `out` is null. Forward writes the tail rows to `out`, or to `tags` when
// `out` is null; it never writes out64.
struct dataflow_tail {
    unsigned begin = 0, rows = 0;
    const float* inverse = nullptr;
    const float* inverse_t = nullptr;
    unsigned* scratch = nullptr;
};

template<unsigned Columns, unsigned RhsLanes>
void dataflow_solve(cudaStream_t stream, int m, bool reverse,
    const int* rowptr, const int* colidx, const void* vals, const float* diag, const double* scale,
    const unsigned* words, const detail::gpu_dataflow::segment* segments, unsigned batches,
    unsigned* tags, unsigned tag_count, unsigned* ticket, int grid, const float* rhs, float* out, const unsigned* active,
    const double* rhs64, double* out64, unsigned active_stride, dataflow_tail tail = {});

// Largest dense tail the inverse kernels accept: 2 x 16 MB of inverse.
inline constexpr unsigned kDataflowTailMaxRows = 2048;

// Exact inverse of rows [begin, begin + rows) of CSR(L) restricted to the
// columns [begin, begin + rows), diagonal included (fp16: diag[row]; else the
// last entry of the row). Computed in fp64 from a temporary fp32 copy of the
// block that is freed before returning; writes the row-major inverse and its
// transpose as fp32 (both rows x rows). 2 <= rows <= kDataflowTailMaxRows.
void build_dataflow_tail_inverse(unsigned begin, unsigned rows, const int* L_rowptr,
    const int* L_colidx, const void* L_vals, bool fp16, const float* diag,
    float* inverse, float* inverse_t);

inline void dataflow_solve(cudaStream_t stream, int m, bool reverse,
    const int* rowptr, const int* colidx, const void* vals, const float* diag, const double* scale,
    const unsigned* words, const detail::gpu_dataflow::segment* segments, unsigned batches,
    unsigned* tags, unsigned tag_count, unsigned* ticket, int grid, const float* rhs, float* out,
    const unsigned* active = nullptr, const double* rhs64 = nullptr, double* out64 = nullptr) {
    dataflow_solve<1, 1>(stream, m, reverse, rowptr, colidx, vals, diag, scale, words, segments,
                     batches, tags, tag_count, ticket, grid, rhs, out, active, rhs64, out64, 1);
}

// Returned allocations transfer to the caller. Only fixed-size statistics and
// counts cross to the host; trusted device-owned factors remain on the device.
struct dataflow_device_plan {
    unsigned* words = nullptr;
    gpu_dataflow::segment* segments = nullptr;
    int n_batches = 0, n_slots = 0, n_split = 0;
    gpu_dataflow::row_stats stats{};
    std::size_t bytes = 0, host_download_bytes = 0;
};

// split_min: -1 derives the threshold from row lengths, 0 disables splitting.
// The private factor finalizer guarantees sorted triangular CSR with diagonal
// at the proper end. Optional FP16 metadata is validated in the same row pass.
// planned_rows < m (a dense tail): only rows [0, planned_rows) enter the plan
// and its row-length statistics.
dataflow_device_plan build_dataflow_device_plan(
    int m, bool reverse, int nnz, const int* rowptr, int split_min,
    const float* diag = nullptr, const double* inv_scale2 = nullptr, unsigned plan_lanes = 32,
    int planned_rows = -1);
} // namespace apxchol::detail
