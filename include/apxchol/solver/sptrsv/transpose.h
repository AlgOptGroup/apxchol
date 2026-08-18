#pragma once
// THE CSC -> CSR TRANSPOSE of the factor -- the ONE host implementation shared
// by both SpTRSV backends: omp_sptrsv::setup builds its CSR of L11 (the
// forward solve's operand) from the factor's CSC through it, and the GPU
// backend's host prep (cuda_host::transpose_csr) builds CSR of L from CSR of
// L^T -- the very same arrays, int32 -- for our two kernel backends (dataflow /
// level-set) through it. Header-only, CUDA-free, OpenMP-parallel, templated on
// the offset / index / value types (edge_index / node_index / sptrsv_value_t
// on the CPU, int / int / cuda_value_t or uint16_t on the GPU) and on a
// `store(v, j)` functor that maps the input value of an entry in column j to
// the stored value (the CPU's narrow_value() through the storage format; a
// plain copy on the GPU) -- the same code, so what the GPU uploads is what the
// CPU stores (tests/test_sptrsv_transpose.cpp states the byte identity of
// both callers against one serial reference at several thread counts).
//
// Blocked counting-sort parallel transpose (APXCHOL_PAR_TRANSPOSE, default on
// for m > kParTransposeMinRows): O(nnz) TOTAL work. It replaced the row-range
// rescan scheme on the CPU, where every thread scanned ALL nnz twice filtering
// to its own row range -- O(threads*nnz) total work, so the stage's wall time
// was flat in thread count (71-76% of sptrsv_setup at T=16) -- and the plain
// serial scatter on the GPU host path (+155 ms on grid_2000's 22M-nnz factor,
// 330 vs 175 ms sptrsv_setup against cuSPARSE, which needs no transpose).
//
//   Phase 1 (parallel): threads own contiguous, nnz-balanced COLUMN ranges.
//     Rows are split into NB <= 4*nt blocks of power-of-two width
//     (block-of-row is a shift). Thread t histograms its column range's
//     entries per row block into cnt[t][b].
//   Phase 2 (parallel): a serial exclusive prefix over cnt in (b major, t
//     minor) order gives every (thread, block) pair an exact segment in an
//     nnz-sized bucket. Thread t re-scans its column range in ascending-column
//     order, appending each entry (row, col, value) to its segment of the
//     entry's row block.
//   Phase 3 (parallel over blocks): block b's bucket region
//     [blk_off[b], blk_off[b+1]) holds exactly its rows' entries in
//     ascending-column order (segments are concatenated in thread order ==
//     ascending column ranges, each internally ascending). A per-block
//     counting sort -- count rows, exclusive prefix based at blk_off[b] (which
//     IS the final CSR offset of the block's first row, since blocks partition
//     rows contiguously), then a stable scatter -- lands every entry in its
//     final CSR slot and fills the block's out_ptr range.
//
// Determinism / byte-identity: within each row, the stable per-block scatter
// preserves the bucket's ascending-column order -- exactly the order the
// serial scatter (columns walked 0..m-1) produces -- and out_ptr is uniquely
// determined by the row counts. All three CSR arrays are therefore
// byte-identical to the serial result for ANY thread count (verified by the
// SpTRSVTranspose.* unit tests).
//
// Memory: one transient bucket of nnz * (2 * sizeof(Idx) + sizeof(OutVal))
// bytes (12 B/nnz on the default 32-bit-index fp32 build, 10 B/nnz for the
// GPU's fp16 storage), allocated uninitialized (every slot is written exactly
// once in phase 2) and freed on return, plus the nt x NB count matrix
// (<= 32*129*8 B ~ 33 KB).
//
// Rejected alternatives:
//   (a) per-thread full row histograms -- nt x m x 4 B per call wins on
//       small-n high-fanout factors but the allocation explodes on
//       multi-million-row grids (256 MB at m=4M, nt=16);
//   (b) atomic-claim on shared O(m) counters -- avoids the n x nt memory but
//       cache-line ping-pong on the m-sized counters under high thread counts
//       makes it slower than serial on both shapes;
//   (c) row-range rescan (the previous CPU scheme) -- no extra memory, but
//       O(threads*nnz) total work: flat wall time in thread count.
//
// Below the size threshold the SERIAL scatter wins (thread dispatch overhead
// exceeds the work) -- both callers go through use_parallel_transpose(m).
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace apxchol {

/// Rows above which the parallel transpose is used (below it the serial
/// scatter wins: thread dispatch overhead exceeds the work).
inline constexpr std::int64_t kParTransposeMinRows = 50000;

/// APXCHOL_PAR_TRANSPOSE (env, read once per process): 0 disables the
/// parallel path everywhere (A/B only); unset / anything else = on.
inline bool par_transpose_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("APXCHOL_PAR_TRANSPOSE");
        return e ? std::atoi(e) != 0 : true;
    }();
    return v;
}

/// The rule both backends apply: the parallel path iff enabled, m >
/// kParTransposeMinRows and more than one OpenMP thread is available (at one
/// thread the plain serial scatter is faster -- two passes and no bucket vs
/// the blocked sort's three; both are byte-identical anyway).
inline bool use_parallel_transpose(std::int64_t m) {
#ifdef _OPENMP
    return par_transpose_enabled() && m > kParTransposeMinRows && omp_get_max_threads() > 1;
#else
    (void)m;
    return false;
#endif
}

/// Transpose an m x m CSC (in_ptr[m+1], in_idx[nnz], in_vals[nnz]; == a CSR of
/// the transpose) into the CSR out_ptr[m+1], out_idx[nnz], out_vals[nnz] --
/// all three output arrays allocated by the caller, every slot written exactly
/// once (out_ptr in full, out_ptr[m] == nnz). Entry (i, j) with input value v
/// lands in row i with column j and value store(v, j) (Store: (InVal, Idx) ->
/// OutVal, pure); within each output row the columns are ascending -- the
/// order the serial column walk produces -- on both paths. `parallel` selects
/// the blocked counting-sort path (use_parallel_transpose(m) is the rule;
/// without OpenMP it is the serial path regardless). Off / Idx are the
/// offset / index types (unsigned or signed, 32 or 64 bit).
template <class Off, class Idx, class InVal, class OutVal, class Store>
inline void transpose_csc_to_csr(Idx m, const Off* in_ptr, const Idx* in_idx, const InVal* in_vals,
                                 Off* out_ptr, Idx* out_idx, OutVal* out_vals,
                                 Store&& store, bool parallel) {
    const Off nnz = in_ptr[m];
#ifdef _OPENMP
    if (parallel && m > 0) {
        // Transient bucket, deliberately UNINITIALIZED (a std::vector would
        // serially memset up to nnz*12 B): phase 2 writes every slot exactly
        // once before phase 3 reads it.
        std::unique_ptr<Idx[]>    bkt_row(new Idx[nnz]);
        std::unique_ptr<Idx[]>    bkt_col(new Idx[nnz]);
        std::unique_ptr<OutVal[]> bkt_val(new OutVal[nnz]);
        std::vector<Off> cnt;      // cnt[t*NB + b]: per-(thread, block) counts
        std::vector<Off> seg_off;  // segment starts, same layout as cnt
        std::vector<Off> blk_off;  // NB+1 block starts (== final CSR offsets)
        int shift = 0;             // log2(rows per block)
        Idx NB    = 1;             // number of row blocks
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const int nt  = omp_get_num_threads();
            #pragma omp single
            {
                // Row-block width: smallest power of two >= ceil(m/(4*nt)), so
                // NB <= 4*nt (~4 blocks/thread for phase-3 balance) and
                // block-of-row is a shift.
                const Idx target = static_cast<Idx>(
                    (static_cast<std::uint64_t>(m) + 4 * static_cast<std::uint64_t>(nt) - 1) /
                    (4 * static_cast<std::uint64_t>(nt)));
                Idx rpb = 1;
                shift = 0;
                while (rpb < target) { rpb <<= 1; ++shift; }
                NB = static_cast<Idx>(((static_cast<std::uint64_t>(m) - 1) >> shift) + 1);   // == ceil(m/rpb)
                cnt.assign(static_cast<std::size_t>(nt) * static_cast<std::size_t>(NB), 0);
            }   // implicit barrier: cnt/NB/shift visible to all threads
            // Contiguous nnz-balanced column range for this thread (a single
            // column is never split, so one pathologically dense column bounds
            // the imbalance). Boundaries are monotone in tid, so ranges are
            // disjoint and cover every entry.
            auto col_at = [&](int t) {
                const Off tgt = static_cast<Off>(
                    static_cast<std::uint64_t>(nnz) * static_cast<std::uint64_t>(t) / static_cast<std::uint64_t>(nt));
                return static_cast<Idx>(std::lower_bound(in_ptr, in_ptr + m + 1, tgt) - in_ptr);
            };
            const Idx c_lo = col_at(tid);
            const Idx c_hi = col_at(tid + 1);
            // Phase 1: per-(thread, block) histogram.
            Off* my_cnt = cnt.data() + static_cast<std::size_t>(tid) * static_cast<std::size_t>(NB);
            for (Idx j = c_lo; j < c_hi; ++j)
                for (Off p = in_ptr[j]; p < in_ptr[j + 1]; ++p)
                    my_cnt[in_idx[p] >> shift]++;
            #pragma omp barrier
            #pragma omp single
            {
                // Exclusive prefix in (b major, t minor) order: segment
                // concatenation inside each block follows ascending column
                // ranges. O(nt*NB) -- trivially serial.
                seg_off.resize(cnt.size());
                blk_off.resize(static_cast<std::size_t>(NB) + 1);
                Off run = 0;
                for (Idx b = 0; b < NB; ++b) {
                    blk_off[b] = run;
                    for (int t = 0; t < nt; ++t) {
                        seg_off[static_cast<std::size_t>(t) * NB + b] = run;
                        run += cnt[static_cast<std::size_t>(t) * NB + b];
                    }
                }
                blk_off[NB] = run;   // == nnz
            }   // implicit barrier
            // Phase 2: scatter into the bucket (private cursors; each (thread,
            // block) segment is written by one thread only).
            std::vector<Off> cur(seg_off.begin() + static_cast<std::size_t>(tid) * NB,
                                 seg_off.begin() + static_cast<std::size_t>(tid + 1) * NB);
            for (Idx j = c_lo; j < c_hi; ++j)
                for (Off p = in_ptr[j]; p < in_ptr[j + 1]; ++p) {
                    const Idx row = in_idx[p];
                    const Off out = cur[row >> shift]++;
                    bkt_row[out] = row;
                    bkt_col[out] = j;
                    bkt_val[out] = store(in_vals[p], j);
                }
            #pragma omp barrier
            // Phase 3: per-block stable counting sort into the final CSR (each
            // block owns disjoint row + output ranges -- race-free and
            // schedule-independent).
            #pragma omp for schedule(dynamic, 1)
            for (Idx b = 0; b < NB; ++b) {
                const Idx r_lo = static_cast<Idx>(static_cast<std::uint64_t>(b) << shift);
                const Idx r_hi = static_cast<Idx>(std::min<std::uint64_t>(
                    static_cast<std::uint64_t>(m), static_cast<std::uint64_t>(b + 1) << shift));
                std::vector<Off> pos(static_cast<std::size_t>(r_hi - r_lo), 0);
                for (Off k = blk_off[b]; k < blk_off[b + 1]; ++k)
                    pos[bkt_row[k] - r_lo]++;
                Off run = blk_off[b];   // final CSR offset of row r_lo
                for (Idx r = r_lo; r < r_hi; ++r) {
                    const Off c = pos[r - r_lo];
                    out_ptr[r]     = run;
                    pos[r - r_lo]  = run;
                    run += c;
                }
                for (Off k = blk_off[b]; k < blk_off[b + 1]; ++k) {
                    const Off out = pos[bkt_row[k] - r_lo]++;
                    out_idx[out]  = bkt_col[k];
                    out_vals[out] = bkt_val[k];
                }
            }   // implicit barrier
        }
        out_ptr[m] = nnz;
        return;
    }
#else
    (void)parallel;
#endif
    // Serial count / prefix / column-order scatter.
    std::fill(out_ptr, out_ptr + m + 1, Off(0));
    for (Idx j = 0; j < m; ++j)
        for (Off p = in_ptr[j]; p < in_ptr[j + 1]; ++p)
            out_ptr[in_idx[p] + 1]++;
    for (Idx i = 0; i < m; ++i)
        out_ptr[i + 1] += out_ptr[i];
    std::vector<Off> pos(out_ptr, out_ptr + m);
    for (Idx j = 0; j < m; ++j)
        for (Off p = in_ptr[j]; p < in_ptr[j + 1]; ++p) {
            const Off out = pos[in_idx[p]]++;
            out_idx[out]  = j;
            out_vals[out] = store(in_vals[p], j);
        }
}

} // namespace apxchol
