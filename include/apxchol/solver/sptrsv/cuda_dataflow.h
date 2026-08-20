#pragma once
#include "apxchol/solver/sptrsv/cuda_levelset.h"   // sptrsv_gpu_value_t
#include <cuda_runtime.h>

// Sync-free ("dataflow") GPU sparse triangular solve with O(n) STATE -- the
// third SpTRSV backend of cuda.h (env APXCHOL_GPU_SPTRSV=dataflow).
//
// WHY. cuSPARSE SpSV is ~1.5-2x faster per sweep than our level-set kernel on
// this GPU (both far below bandwidth; the level-set pays one launch + one
// stream sync per level, 70-100 levels per sweep) but keeps an O(nnz)
// analysis buffer (~23 B/nnz per direction: a level-permuted copy of the
// matrix -- its find_colors / create_perm / number_of_deps kernels -- plus
// per-row dependency counts), which is what OOMs the giant social factors.
// This kernel is ONE persistent launch per sweep, works on the very CSR the
// level-set backend already holds, and its per-solve state is 8 bytes per
// ROW.
//
// SCHEME (T out = rhs, T triangular in CSR with the diagonal inside every
// row). Every row i owns a TAGGED WORD tag[i] = {value, epoch} (one 64-bit
// word). A sweep gets a fresh epoch e (the caller counts them); a row is
// "published" for this sweep when tag[i] carries e.
//   * Warps claim BATCHES of consecutive rows in SWEEP ORDER with one
//     atomicAdd on a ticket counter -- the natural row order for the forward
//     solve (T lower triangular: row i's neighbours j < i are earlier rows),
//     its reverse for the back solve (T upper triangular). Consecutive rows
//     mean contiguous CSR data (coalesced structure / value loads) and, for
//     an elimination factor, roughly level-ordered work (a round's rows are
//     contiguous in the elimination order and mutually independent).
//   * The host packs the rows into 32-lane batches (cuda_host::dataflow_
//     batches): a row of at most kPre entries takes one lane, longer rows a
//     group of G = 2, 4, ..., 32 lanes (G aligned). The G lanes of a row
//     stride its CSR entries (lane sub takes entries sub, sub+G, ...), so
//     every row's structure -- column indices, values, rhs, diagonal -- is
//     PREFETCHED into registers before the wait; rows beyond G*kPre entries
//     continue in G*kPre chunks. The dense tail rows of an elimination
//     factor (dozens of off-diagonals) are why: with lane-per-row plus a
//     serial warp fallback they cost several round trips each, serialised
//     inside the warp, and the tail IS the critical path.
//   * Each lane POLLS the tagged words of its still-missing neighbours (all
//     loads of the chunk in flight together; a lane keeps a pending mask, so
//     a re-poll only re-reads the laggards), between polls it spins on the
//     SM clock (~100-800 ns; __nanosleep's granularity is ~1 us on this GPU).
//     When every lane of the row's group has all its neighbours -- a
//     shuffle butterfly inside the group -- the row is ready:
//       out[i] = (rhs[i] - sum_{j != i} T[i,j] * value(tag[j])) / T[i,i]
//     accumulated in entry order per lane, then a fixed butterfly over the
//     group; the leader lane writes out[i] (the plain result vector) and
//     tag[i] = {out[i], e} in one 64-bit store.
//   * A row's word carries value AND flag, so the data path needs NO fence,
//     NO counter and NO atomic anywhere: single-copy atomicity of the aligned
//     64-bit gpu-scope relaxed load/store is all (the reader gets the value
//     it saw the epoch of). The only atomics are the ticket claims and the
//     last-warp reset of the two control ints (which do carry fences).
//
// DEADLOCK FREEDOM. A lane only ever waits on rows with SMALLER tickets
// (a forward neighbour j < i is earlier in the natural order, a back
// neighbour k > i earlier in the reversed one), and a ticket is only ever
// held by a warp that is running and stays running until its batch is done
// -- tickets are claimed dynamically at run time, never assigned to a
// not-yet-resident block, so unlike a static "block b solves rows [bB,
// (b+1)B)" scheme there is no waiting on work that has not been scheduled.
// Inside one warp a row whose producer is another row of the same batch is
// served by the loop: ready rows publish, the others re-poll. The grid is
// still sized to exactly the RESIDENT block count (cudaOccupancyMaxActive-
// BlocksPerMultiprocessor x SMs, dataflow_grid_size()) -- every block loops
// over tickets until they run out, so queued non-resident blocks would only
// add a launch tail.
//
// STATE (device, O(n)): tag[m] (8 B/row, mutable, shared by both directions
// -- the forward sweep uses epoch e, the back sweep e+1), the two batch
// tables (<= m+1 ints each, read-only), ctrl[2] per direction = {ticket,
// finished warps} (zero at entry and at exit: the last warp out resets
// them). No per-nonzero scratch, no in-degree array, no transposed
// dependency structure -- each sweep touches only its own CSR (CSR of L
// forward, CSR of L^T back), which the level-set backend holds anyway.
// The epoch is a 32-bit sweep count; the owner clears tag[] and restarts it
// long before it could wrap onto a stale word.
//
// DETERMINISM. Which warp computes a row and when does not enter the value:
// a lane's partial sum runs in entry order, the group's butterfly is fixed,
// and the lane group of a row is a function of its length alone -- so out is
// bit-identical run to run AND across grid sizes (cuSPARSE SpSV is not; the
// level-set backend is). Verified: tests/test_sptrsv_drop.cpp (CUDA build).
//
// Defined in src/cuda_dataflow.cu (nvcc TU). fp32 values (the tagged word
// packs a 4-byte value with a 4-byte epoch -- which is why the removed fp64
// SpTRSV storage never had this backend); the fp16 storage of cuda_host.h is the same kernel
// with the half -> float widen, the fp32 diag[] and the optional per-row
// input scale (dataflow_solve_fp16), mirroring levelset_solve_fp16.
//
// fp16 PATH DESIGN (2026-08-19 rework; measurements RTX 4090 Laptop,
// interleaved warm per-sweep medians): the values are loaded as raw 2-byte
// bits (the diagonal slot zeroed -- it can be fp16 Inf, see cuda_host.h
// diag_bad), widened only in the accumulate as ONE packed half2 -> float2
// per entry pair, and accumulated branchlessly in two ILP chains (dead
// slots contribute +-0 because their y is 0). Rationale: a hub-heavy BACK
// sweep is a sequential publish chain of multi-chunk rows, so anything on
// the load->poll->accumulate->publish path is paid once per chain link --
// the original widen-at-load put a cvt behind every 2-byte load and, past
// the register cliff (see the kernel comment in the .cu), serialized the
// 8-deep load pipeline: kron_g500 back sweep 24 -> 13 ms after the rework,
// com-LiveJournal solve 9.6 -> 5.4 s, at unchanged iteration counts.
// Result vs fp32 storage: faster or equal on every forward sweep and on
// IPM / grid factors (iter0040 pair 0.92x, grid_2000 0.97x), a small
// residual on hub-graph back sweeps only (kron 1.06x, LJ 1.03-1.08x; the
// per-link widen+select tax) -- sub-noise at total-solve level. Measured
// dead ends (fp16 compute, mixed-width fp32-for-hub-rows, pair loads +
// register cap) are recorded at the kernel in the .cu.
namespace apxchol {

/// Threads per block of the persistent kernel.
inline constexpr int kDataflowBlock = 256;

/// The kernel's per-lane prefetch depth (entries of the row held in
/// registers per lane; a compile-time constant of the .cu). The host's batch
/// packing (cuda_host::dataflow_batches) needs it.
int dataflow_prefetch_depth();

/// The persistent grid: cudaOccupancyMaxActiveBlocksPerMultiprocessor of the
/// (fp32 or fp16) kernel x multiprocessor count on the current device -- the
/// exact resident block count. Env APXCHOL_GPU_DATAFLOW_BLOCKS=<n> overrides
/// (tuning / A/B only; read here). >= 1.
int dataflow_grid_size(bool fp16);

/// One sweep `T out = rhs`, one kernel launch. See the file header for the
/// contract. `tag` and `ctrl` are device state owned by the caller; `rhs`
/// and `out` must be distinct.
///   m, reverse : rows; the sweep order is the natural row order (T lower
///                triangular, forward) or its reverse (T upper triangular, back)
///   rowptr/colidx/vals : CSR of T (diagonal inside each row)
///   batch_start, n_batches : the warp batches (cuda_host::dataflow_batches
///                for this direction; n_batches + 1 ints on the device)
///   tag : m tagged words {value, epoch}; a row's word is (re)written by this
///         sweep with `epoch`, and consumers poll it -- pass an epoch no
///         word in the array carries (the caller counts sweeps; 0 = never)
///   ctrl : 2 ints {ticket, finished warps}, zero at entry and at exit
///   grid : blocks (dataflow_grid_size()); block = kDataflowBlock
void dataflow_solve(cudaStream_t stream, int m, bool reverse,
                    const int* rowptr, const int* colidx, const sptrsv_gpu_value_t* vals,
                    const int* batch_start, int n_batches,
                    unsigned long long* tag, unsigned epoch, int* ctrl, int grid,
                    const sptrsv_gpu_value_t* rhs, sptrsv_gpu_value_t* out);

/// The same sweep on the fp16 per-column-scaled storage (cuda_host.h
/// contract; identical semantics to levelset_solve_fp16: `vals16` binary16
/// bit patterns, the diagonal slot skipped, divide by the fp32 `diag[i]`,
/// rhs[i] scaled once by the DOUBLE `in_scale[i]` when non-null -- r_i^2
/// exact, product in double, one narrowing cast; see cuda_levelset.h).
void dataflow_solve_fp16(cudaStream_t stream, int m, bool reverse,
                         const int* rowptr, const int* colidx, const unsigned short* vals16,
                         const float* diag, const double* in_scale,
                         const int* batch_start, int n_batches,
                         unsigned long long* tag, unsigned epoch, int* ctrl, int grid,
                         const sptrsv_gpu_value_t* rhs, sptrsv_gpu_value_t* out);

} // namespace apxchol
