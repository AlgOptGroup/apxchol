#pragma once
// HOST-side preparation of the GPU SpTRSV backend (cuda.h): everything
// cuda_sptrsv::setup does to the factor BEFORE the upload -- the L11 extraction
// into cuSPARSE's int32 arrays, the compacting factor drop (the shared
// factor_drop.h implementation, the same one omp_sptrsv::setup runs), the
// fp16 per-column-scaled narrowing of our kernel backends' opt-in fp16
// storage (APXCHOL_SPTRSV_FP16=1), the CSR transpose (the shared
// transpose.h implementation, the one omp_sptrsv::setup runs), the dataflow
// schedules and the dataflow batch tables. Deliberately CUDA-FREE (no cuda_runtime.h, no __half: fp16
// values are IEEE binary16 BIT PATTERNS, std::uint16_t, produced by
// lowprec.h's fp16_t -- the same RNE the CPU FP16_SCALED build uses -- and
// reinterpreted as __half on the device) so the CPU unit tests can state,
// without a GPU, that what the GPU backend uploads is what the CPU backend
// stores (tests/test_sptrsv_drop.cpp, "GpuHostPrep*").
//
// fp16 STORAGE CONTRACT (mirrors omp.h's FP16_SCALED, "FOLDED INTO THE
// VECTORS"): what is stored is the column-scaled factor L~ = L D^-1, D =
// diag(s_j), s_j = factor_column_scale (max |off-diagonal| of column j, 1.0f
// if none; the PRE-drop max, which the drop never removes): off-diagonal
// slots hold fp16(fp32(L_ij) / s_j) (RNE; fp16 subnormals flushed to signed
// zero, always), the
// diagonal is NOT read from its (still present, fp16) slot but from a
// separate fp32 diag[j] = fp32(L_jj) / s_j, and inv_scale[j] = fp32(1 / s_j).
// The kernels never multiply a scale back: the forward solve on L~ returns
// y' = D y, the back solve reads its input scaled by inv_scale[j]^2 and
// solves L~^T z = D^-2 y', so the pair applies (L_s L_s^T)^-1 for the stored
// factor L_s = L~ D. The rounding residual of every column,
// sum_i (x_ij - widen(h_ij)) with x_ij = fp64(L_ij) / s_j, is folded into
// diag[j], always (the CPU SpTRSV does the same; the env knob that used
// to gate it, APXCHOL_LOWPREC_DIAG_COMP, was removed 2026-08-20 -- it was OFF
// there as a diagnostic, ON here because the fp16 GPU mode is new and was
// measured with it: without it the Laplacian path pays iter0040 45 -> 64
// PCG iterations, with it fp16 matches fp32's counts) so the stored column
// sums to what the fp32 column sums to.
#include "apxchol/sparse_csc.h"
#include "apxchol/solver/sptrsv/factor_drop.h"
#include "apxchol/solver/sptrsv/transpose.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace apxchol::cuda_host {

/// An owning m x m CSR (or CSC -- the same three arrays) with int32 offsets
/// and indices (cuSPARSE CUSPARSE_INDEX_32I) and Val values. idx / vals are
/// allocated uninitialized by the builders below (every slot written once).
template <class Val>
struct csr_int {
    int                     m   = 0;
    std::int64_t            nnz = 0;
    std::vector<int>        ptr;    // m+1
    std::unique_ptr<int[]>  idx;    // nnz
    std::unique_ptr<Val[]>  vals;   // nnz
};

/// L11 = top-left m x m block of the factor L as int CSC arrays (== CSR of
/// L11^T). For the Laplacian case (m < n) entries in the grounded last row
/// (n-1) are filtered out. Offsets are read as edge_index and emitted as int:
/// a factor exceeding int32 cannot use cuSPARSE's 32-bit index API anyway.
template <class Val>
inline csr_int<Val> build_L11_csc_int(const sparse_csc& L, std::int64_t m) {
    const edge_index* Lo = L.outerIndexPtr();
    const node_index* Li = L.innerIndexPtr();
    const auto*       Lv = L.valuePtr();   // factor_value_t*
    const std::int64_t n = static_cast<std::int64_t>(L.rows());
    csr_int<Val> out;
    if (m > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
        Lo[m] > static_cast<edge_index>(std::numeric_limits<int>::max()))
        throw std::runtime_error("apxchol cuda_sptrsv: the factor (m=" + std::to_string(m) + ", nnz=" +
                                 std::to_string(static_cast<unsigned long long>(Lo[m])) +
                                 ") exceeds the GPU backend's 32-bit index range (cuSPARSE CUSPARSE_INDEX_32I)");
    out.m = static_cast<int>(m);
    out.ptr.assign(static_cast<std::size_t>(m) + 1, 0);
    if (m == n) {
        for (std::int64_t j = 0; j < m; ++j)
            out.ptr[j + 1] = out.ptr[j] + static_cast<int>(Lo[j + 1] - Lo[j]);
        const edge_index nnz = Lo[m];
        out.nnz  = static_cast<std::int64_t>(nnz);
        out.idx  = std::make_unique_for_overwrite<int[]>(static_cast<std::size_t>(nnz));
        out.vals = std::make_unique_for_overwrite<Val[]>(static_cast<std::size_t>(nnz));
        #pragma omp parallel for schedule(static)
        for (edge_index p = 0; p < nnz; ++p) {
            out.idx[p]  = static_cast<int>(Li[p]);
            out.vals[p] = static_cast<Val>(Lv[p]);
        }
    } else {
        const node_index drop = static_cast<node_index>(n - 1);
        std::vector<int> kept(static_cast<std::size_t>(m));
        #pragma omp parallel for schedule(static)
        for (std::int64_t j = 0; j < m; ++j) {
            int k = 0;
            for (edge_index p = Lo[j]; p < Lo[j + 1]; ++p)
                if (Li[p] != drop) ++k;
            kept[j] = k;
        }
        for (std::int64_t j = 0; j < m; ++j) out.ptr[j + 1] = out.ptr[j] + kept[j];
        out.nnz  = out.ptr[static_cast<std::size_t>(m)];
        out.idx  = std::make_unique_for_overwrite<int[]>(static_cast<std::size_t>(out.nnz));
        out.vals = std::make_unique_for_overwrite<Val[]>(static_cast<std::size_t>(out.nnz));
        #pragma omp parallel for schedule(static)
        for (std::int64_t j = 0; j < m; ++j) {
            int o = out.ptr[j];
            for (edge_index p = Lo[j]; p < Lo[j + 1]; ++p) {
                if (Li[p] == drop) continue;
                out.idx[o]  = static_cast<int>(Li[p]);
                out.vals[o] = static_cast<Val>(Lv[p]);
                ++o;
            }
        }
    }
    return out;
}

/// The fp16 storage's flush clause: true iff fp16(x) is zero or a subnormal
/// -- the entries the format stores as (signed) zero (subnormals are always
/// flushed; see lowprec.h). Pure.
inline bool fp16_flushes(float x) {
    const fp16_t h(x);
    return fp16_t::is_zero(h.bits) || fp16_t::is_subnormal(h.bits);
}

/// THE GPU backend's compacting-drop predicate (omp_sptrsv::keep_offdiag
/// restated for its two storage formats): the off-diagonal v of a column with
/// scale s survives APXCHOL_FACTOR_DROP=rel iff |v| >= rel * s and the storage
/// format does not map it to zero -- an exact zero under fp32 / fp64 storage;
/// under the fp16 storage (`fp16_storage`) also everything fp16 flushes at
/// |v / s| (fp16_flushes). The diagonal is never passed through this.
template <class Val>
inline bool keep_offdiag(Val v, float s, double rel, bool fp16_storage) {
    if (!(std::fabs(static_cast<double>(v)) >= rel * static_cast<double>(s))) return false;
    if (fp16_storage) return !fp16_flushes(static_cast<float>(v) / s);
    return v != Val(0);
}

/// Per-column scales s_j of the columns of `A` (factor_column_scale contract:
/// max |off-diagonal|, 1.0f if none; every column starts with its diagonal).
template <class Val>
inline std::vector<float> column_scales(const csr_int<Val>& A) {
    std::vector<float> s(static_cast<std::size_t>(A.m));
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < A.m; ++j)
        s[j] = factor_column_scale(A.vals.get(), A.ptr[j], A.ptr[j + 1]);
    return s;
}

/// The compacting factor drop (factor_drop.h -- the implementation the CPU
/// backend runs) applied IN PLACE to the columns of `L11`, with the GPU
/// backend's keep predicate (above): if anything is dropped the compacted
/// arrays replace L11's (nnz shrinks), else L11 is untouched. `col_scale` =
/// column_scales(L11) computed BEFORE the drop (the fp16 storage keeps using
/// it). Returns what happened (drop_stats()).
template <class Val>
inline factor_drop_stats apply_factor_drop(csr_int<Val>& L11, const std::vector<float>& col_scale,
                                           double rel,
                                           bool fp16_storage) {
    factor_drop_stats st;
    std::vector<int>       out_ptr;
    std::unique_ptr<int[]> out_idx;
    std::unique_ptr<Val[]> out_vals;
    const bool compacted = compact_factor_columns<int, int, Val>(
        L11.m, L11.ptr.data(), L11.idx.get(), L11.vals.get(), col_scale.data(), rel,
        [=](Val v, float s) { return keep_offdiag(v, s, rel, fp16_storage); },
        out_ptr, out_idx, out_vals, st);
    if (compacted) {
        L11.ptr.swap(out_ptr);
        L11.idx  = std::move(out_idx);
        L11.vals = std::move(out_vals);
        L11.nnz  = static_cast<std::int64_t>(st.nnz_stored);
    }
    assert(static_cast<std::uint64_t>(L11.nnz) == st.nnz_stored);
    return st;
}

/// The fp16 per-column-scaled storage of an int CSC (see the file header for
/// the contract): vals = binary16 bit patterns of fp32(L_ij) / s_j (diagonal
/// slot included, unread by the kernels), diag = fp32(L_jj) / s_j (+ the
/// column's rounding residual, always), inv_scale = fp32(1 / s_j),
/// plus the storage statistics over the off-diagonals.
struct fp16_scaled_arrays {
    std::unique_ptr<std::uint16_t[]> vals;       // nnz
    std::vector<float>               diag;       // m
    std::vector<float>               inv_scale;  // m
    std::uint64_t flushed   = 0;   // stored off-diagonals with v != 0 stored as zero
    std::uint64_t subnormal = 0;   // stored off-diagonals that are fp16 subnormals (0: they are always flushed)
};

/// The narrowing itself: what THIS entry stores. Pure (both stored copies --
/// CSR of L and CSR of L^T -- carry the same bits for the same entry).
inline std::uint16_t narrow_fp16_scaled_value(float v, float s) {
    const fp16_t h(v / s);                                    // RNE
    if (fp16_t::is_subnormal(h.bits))
        return static_cast<std::uint16_t>(h.bits & 0x8000u);  // signed zero
    return h.bits;
}
inline float widen_fp16(std::uint16_t bits) { return fp16_t::from_bits(bits).to_float(); }

template <class Val>
inline fp16_scaled_arrays narrow_fp16_scaled(const csr_int<Val>& L11, const std::vector<float>& col_scale) {
    fp16_scaled_arrays out;
    out.vals = std::make_unique_for_overwrite<std::uint16_t[]>(static_cast<std::size_t>(L11.nnz));
    out.diag.resize(static_cast<std::size_t>(L11.m));
    out.inv_scale.resize(static_cast<std::size_t>(L11.m));
    std::uint64_t n_flush = 0, n_sub = 0;
    #pragma omp parallel for schedule(static) reduction(+ : n_flush, n_sub)
    for (int j = 0; j < L11.m; ++j) {
        const float s = col_scale[j];
        out.inv_scale[j] = 1.0f / s;
        assert(L11.idx[L11.ptr[j]] == j && "factor column must start with its diagonal");
        float  d     = static_cast<float>(L11.vals[L11.ptr[j]]) / s;   // stored_diag(): fp32(L_jj) / s_j
        double resid = 0.0;                                            // sum over the off-diagonals of (x - widen(stored))
        for (int p = L11.ptr[j]; p < L11.ptr[j + 1]; ++p) {
            const float v = static_cast<float>(L11.vals[p]);
            const std::uint16_t h = narrow_fp16_scaled_value(v, s);
            out.vals[p] = h;
            if (L11.idx[p] == j) continue;   // the diagonal SLOT: written, never read (see the file header)
            const float w = widen_fp16(h);
            if (v != 0.0f && w == 0.0f) ++n_flush;
            else if (fp16_t::is_subnormal(h)) ++n_sub;
            resid += static_cast<double>(L11.vals[p]) / static_cast<double>(s) - static_cast<double>(w);
        }
        d = static_cast<float>(static_cast<double>(d) + resid);
        out.diag[j] = d;
    }
    out.flushed = n_flush; out.subnormal = n_sub;
    return out;
}

/// Transpose a square m x m CSR (values of any trivially copyable V). The
/// The dataflow forward solve needs CSR of L (row access) while setup
/// builds CSR of L^T (== the factor's CSC); this produces the one from the
/// other. THE shared transpose (transpose.h -- the very code omp_sptrsv::setup
/// runs for its CSC -> CSR): the blocked counting-sort parallel path, O(nnz)
/// total work, for m > kParTransposeMinRows (APXCHOL_PAR_TRANSPOSE=0 disables
/// it), the serial column-order scatter below; within each output row the
/// input-row order is preserved on both paths, and the three arrays are
/// byte-identical to the serial result at ANY thread count
/// (SpTRSVTranspose.GpuHostTransposeIsByteIdenticalToSerialReference).
template <class V>
inline csr_int<V> transpose_csr(const csr_int<V>& in) {
    csr_int<V> out;
    out.m = in.m; out.nnz = in.nnz;
    out.ptr.resize(static_cast<std::size_t>(in.m) + 1);
    out.idx  = std::make_unique_for_overwrite<int[]>(static_cast<std::size_t>(in.nnz));
    out.vals = std::make_unique_for_overwrite<V[]>(static_cast<std::size_t>(in.nnz));
    transpose_csc_to_csr<int, int, V, V>(
        in.m, in.ptr.data(), in.idx.get(), in.vals.get(),
        out.ptr.data(), out.idx.get(), out.vals.get(),
        [](V v, int) { return v; }, use_parallel_transpose(in.m));
    return out;
}

/// The dataflow backend's lane-group size of a row of `entries` CSR entries
/// (diagonal slot included): the smallest power of two G <= 32 with G * pre
/// >= entries (`pre` = the kernel's per-lane prefetch depth,
/// dataflow_prefetch_depth()). Restated on the device; both sides must agree.
inline int dataflow_lane_group(int entries, int pre) {
    int G = 1;
    while (G < 32 && G * pre < entries) G <<= 1;
    return G;
}

/// THE PLAIN BATCH TABLE, kept as the planner's REFERENCE IMPLEMENTATION:
/// dataflow_build_plan() below produces exactly this at seg == 0, and a test
/// says so on a real-shaped factor in both directions. Setup itself goes
/// through the planner; this is the independently written greedy packer the
/// planner's common path is checked against.
///
/// The dataflow backend's BATCH table for one sweep direction: consecutive
/// rows in sweep order (q = 0..m-1; the row is q, or m-1-q when the sweep is
/// reversed) packed greedily into warps of 32 lanes -- row q takes G(q)
/// lanes (dataflow_lane_group of its CSR row length `len[row]`), aligned to
/// a multiple of G(q); when a row does not fit the batch is closed and the
/// row starts the next one. Returns batch_start (n_batches + 1 sweep
/// positions; batch b = positions [batch_start[b], batch_start[b+1])). O(m)
/// memory, one pass.
inline std::vector<int> dataflow_batches(int m, bool reverse, const int* len, int pre) {
    std::vector<int> bs;
    bs.reserve(static_cast<std::size_t>(m) / 32 + 2);
    bs.push_back(0);
    int pos = 0;
    for (int q = 0; q < m; ++q) {
        const int row = reverse ? m - 1 - q : q;
        const int G = dataflow_lane_group(len[row], pre);
        const int aligned = (pos + G - 1) & ~(G - 1);
        if (aligned + G > 32) { bs.push_back(q); pos = 0; }
        pos = ((pos + G - 1) & ~(G - 1)) + G;
    }
    bs.push_back(m);
    return bs;
}

/// CSR row lengths (diagonal included), rowptr[i+1] - rowptr[i].
///
/// SERIAL ON PURPOSE. The loop is embarrassingly parallel and this is the only
/// part of `cuda_sptrsv::setup_kernel_backend` that is not `dataflow_batches`,
/// so `#pragma omp parallel for schedule(static)` looks free -- it is a
/// MEASURED REGRESSION (2026-08-20, RTX 4090 Laptop, T=16, `bg+tree[vec_pool]`,
/// 28 samples/arm from `APXCHOL_SPTRSV_SETUP_TRACE`, both arms with a warm CUDA
/// context): stage `kernel_backend_tables` grid_2000 **12.30 -> 20.25 ms**
/// median (min 11.43 -> 13.31), iter0040 2.71 -> 2.89. The reason is the
/// CONSUMER: `dataflow_batches` reads `len` back SERIALLY on one thread, and a
/// parallel fill scatters those 4m bytes across every core's private caches
/// first, so the greedy pack then pulls the whole array back cross-core. Do not
/// re-add the pragma on its own -- it only pays off together with a parallel
/// (chunk-and-fixup) `dataflow_batches`, which is where the stage's ~85% sits
/// (host micro-benchmark at m = 4M, avg 5.5 nnz/row: 8.97 ms per direction vs
/// 1.5 ms for this function).
inline std::vector<int> csr_row_lengths(int m, const int* rowptr) {
    std::vector<int> len(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i) len[i] = rowptr[i + 1] - rowptr[i];
    return len;
}

/// ── row segmentation (see cuda_dataflow.h "ROW SEGMENTATION") ──────────────
///
/// A row of `len` CSR entries is walked by its lane group in chunks of
/// G * pre entries, and G caps at 32, so an unsplit row costs
/// ceil(len / C) chunk round trips ON THE CRITICAL PATH with C = 32 * pre.
/// On a hub factor the elimination tail is a CHAIN of such rows, so the sweep
/// pays (chain length) x (len / C) round trips. Segmentation cuts a long row
/// into S pieces that run as independent warps plus a finalizer that sums
/// their partials.

/// Row-length statistics of one sweep direction -- the evidence behind the
/// segmentation threshold (APXCHOL_GPU_SPTRSV_STATS=1 prints them at setup).
/// Bucket k counts the rows with len >= (32 * pre) << k, i.e. the rows an
/// unsplit walk chews in more than 2^k chunks, and the entries they hold.
struct dataflow_len_stats {
    static constexpr int kBuckets = 12;
    int          m        = 0;
    std::int64_t nnz      = 0;
    int          max_len  = 0;
    double       mean_len = 0.0;
    int          base     = 0;            // C = 32 * pre, one unsplit chunk
    std::int64_t rows_ge[kBuckets]{};     // rows with len >= base << k
    std::int64_t nnz_ge[kBuckets]{};      // and the entries they hold
};

inline dataflow_len_stats dataflow_row_stats(int m, const int* len, int pre) {
    dataflow_len_stats s;
    s.m = m;
    s.base = 32 * pre;
    for (int i = 0; i < m; ++i) {
        const int l = len[i];
        s.nnz += l;
        if (l > s.max_len) s.max_len = l;
        for (int k = 0; k < dataflow_len_stats::kBuckets; ++k) {
            if (l < (static_cast<std::int64_t>(s.base) << k)) break;
            ++s.rows_ge[k];
            s.nnz_ge[k] += l;
        }
    }
    s.mean_len = m > 0 ? static_cast<double>(s.nnz) / static_cast<double>(m) : 0.0;
    return s;
}

/// The nnz share (of rows at least one chunk long) above which a direction
/// counts as HUB-HEAVY and gets the aggressive split threshold. Measured
/// separation: com-Amazon fwd 11% / back 0.2%, iter0040 0.01%, grid_500 0%
/// on one side; com-LiveJournal back 42% / fwd 63%, com-Orkut 69% / 84%,
/// kron_g500-logn16 54% / 90% on the other.
inline constexpr double kDataflowSplitShare = 0.25;

/// The DEFAULT split threshold of ONE sweep direction, from that direction's
/// own row-length statistics. A pure function of row lengths -- never of
/// occupancy, grid size or device properties -- so the plan, and the output,
/// stay bit-identical across grid sizes.
///
/// Splitting a row of `len` entries trades a walk of ceil(len / C) chunk
/// round trips for one segment chunk + one finalizer chunk + the publish
/// (depth ~3), at the price of S + 1 extra ticket claims and batches. Where
/// the long rows hold most of the work that price is nothing and the
/// aggressive threshold C wins; where they are a rounding error the extra
/// batches are pure overhead and only rows past the depth break-even (3C) are
/// worth cutting. Measured (RTX 4090 Laptop, gpu_pcg_loop ms/iteration,
/// medians of 6, interleaved):
///   kron_g500-logn16 (fwd share 90%): unsplit 17.00 | C 8.87 | 2C 9.07 | 3C 10.48
///   com-Amazon       (fwd share 11%): unsplit  1.99 | C 2.62 | 2C 2.09 | 3C  1.95
/// -- i.e. the same threshold that is 2x on the hub factor is +31% on the
/// shallow one, which is why this is derived and not a constant.
inline int dataflow_split_threshold(const dataflow_len_stats& st) {
    const double share = st.nnz ? static_cast<double>(st.nnz_ge[0]) / static_cast<double>(st.nnz) : 0.0;
    return share >= kDataflowSplitShare ? st.base : 3 * st.base;
}

/// Parameters of the segmentation. A row with more than `split_min`
/// off-diagonals is cut into segments of about `seg` entries -- rounded UP to
/// a whole chunk C = 32 * pre so every segment has the same chunk depth and
/// none is a straggler -- at most `max_seg` of them. `seg == 0` disables
/// segmentation: the plan is then exactly dataflow_batches() (the rollback
/// switch, APXCHOL_GPU_DF_SPLIT=0).
///
/// The rule is a pure function of (row lengths, params) and must NEVER read
/// occupancy, grid size or device properties: that is what keeps the result
/// bit-identical across grid sizes.
struct dataflow_seg_params {
    int seg       = 0;
    int split_min = 0;
    int max_seg   = 4096;
};

/// One special item of the plan; 16 bytes, uploaded as int4.
///   SEGMENT   : row =  i, a = lo, b = hi, c = slot  -- CSR range [lo, hi)
///   FINALIZER : row = ~i, a = s0, b = S,  c = dpos  -- slots [s0, s0 + S)
struct dataflow_spec { int row = 0, a = 0, b = 0, c = 0; };

/// The batch plan of one sweep direction. `batch_start` is the same monotone
/// table dataflow_batches() returns; `batch_spec[b]` selects the path: -1 =
/// plain (the kernel's untouched common path), >= 0 = index into `spec`.
/// A special item is ALONE in its batch and uses all 32 lanes; a segment's
/// batch is ZERO-WIDTH in sweep positions and the finalizer's batch owns the
/// split row's position, so `batch_start` stays monotone and the plain
/// batches stay contiguous position runs.
struct dataflow_plan {
    std::vector<int>           batch_start;   // nb + 1
    std::vector<int>           batch_spec;    // nb, -1 = plain
    std::vector<dataflow_spec> spec;          // n_slots + n_split entries
    int n_slots = 0;                          // partial slots this direction needs
    int n_split = 0;                          // rows that were split
    int n_batches() const { return static_cast<int>(batch_spec.size()); }
};

/// Build the plan of one sweep direction. `rowptr` / `colidx` are that
/// direction's CSR (CSR of L forward, CSR of L^T back), `len` its row
/// lengths, `pre` = dataflow_prefetch_depth().
///
/// Segments cover the row's WHOLE CSR range, diagonal slot included: the
/// kernel is passed the true row id and drops the `jj == i` slot exactly as
/// the unsplit walk does (fp16 zeroes its bits at load, fp32 parks it in the
/// unused `d`), while the finalizer owns the divide through `dpos`. Covering
/// the whole range -- rather than trimming the diagonal off one end -- is
/// what makes a forced S = 1 split BIT-IDENTICAL to the unsplit kernel in
/// BOTH directions: lane `sub` gets exactly the entries it would have got
/// there. `dpos` is found by scanning the row; the diagonal-first (CSR of
/// L^T) / diagonal-last (CSR of L) convention is not relied on.
inline dataflow_plan dataflow_build_plan(int m, bool reverse, const int* rowptr,
                                         const int* colidx, const int* len, int pre,
                                         const dataflow_seg_params& p) {
    const int C = 32 * pre;
    dataflow_plan pl;
    pl.batch_start.reserve(static_cast<std::size_t>(m) / 32 + 2);
    pl.batch_start.push_back(0);
    int open = 0;   // sweep position the currently open batch starts at
    int pos  = 0;   // lane cursor inside the open PLAIN batch
    for (int q = 0; q < m; ++q) {
        const int row = reverse ? m - 1 - q : q;
        const int l   = len[row];
        if (p.seg > 0 && l - 1 > p.split_min) {
            const int beg = rowptr[row], end = rowptr[row + 1];
            int dpos = -1;
            for (int t = beg; t < end; ++t)
                if (colidx[t] == row) { dpos = t; break; }
            if (dpos < 0)
                throw std::runtime_error("apxchol cuda_sptrsv: row " + std::to_string(row) +
                                         " has no diagonal entry; the dataflow plan needs it");
            const int lo  = beg, hi = end, span = hi - lo;
            int S = (span + p.seg - 1) / p.seg;
            if (S > p.max_seg) S = p.max_seg;
            if (S < 1) S = 1;
            int seg_len = (span + S - 1) / S;
            seg_len = ((seg_len + C - 1) / C) * C;                  // whole chunks
            const int S_act = (span + seg_len - 1) / seg_len;       // <= S
            if (q > open) { pl.batch_start.push_back(q); pl.batch_spec.push_back(-1); }
            const int s0 = pl.n_slots;
            for (int s = 0; s < S_act; ++s) {
                pl.spec.push_back(dataflow_spec{row, lo + s * seg_len,
                                                std::min(hi, lo + (s + 1) * seg_len), s0 + s});
                pl.batch_start.push_back(q);                        // zero-width
                pl.batch_spec.push_back(static_cast<int>(pl.spec.size()) - 1);
            }
            pl.n_slots += S_act;
            pl.spec.push_back(dataflow_spec{~row, s0, S_act, dpos});
            pl.batch_start.push_back(q + 1);                        // owns the row's position
            pl.batch_spec.push_back(static_cast<int>(pl.spec.size()) - 1);
            ++pl.n_split;
            open = q + 1;
            pos  = 0;
            continue;
        }
        const int G = dataflow_lane_group(l, pre);
        int aligned = (pos + G - 1) & ~(G - 1);
        if (aligned + G > 32) {
            pl.batch_start.push_back(q); pl.batch_spec.push_back(-1);
            open = q; pos = 0; aligned = 0;
        }
        pos = aligned + G;
    }
    if (open < m) { pl.batch_start.push_back(m); pl.batch_spec.push_back(-1); }
    return pl;
}

/// The plan's SAFETY PROPERTY, checked on the host because the kernel relies
/// on it implicitly: the chunk walk parks the lanes of rows that are not
/// ready in an inner spin, so an item that needs MORE THAN ONE chunk must
/// never share a warp with a possible producer. Without segmentation that
/// holds by accident (multi-chunk implies G = 32 implies one row per batch);
/// segments and finalizers keep it by being alone in their batch. Also
/// re-states the table invariants (monotone starts, <= 32 lanes per plain
/// batch, every sweep position covered exactly once, segments before their
/// finalizer). Returns "" when the plan is sound, else what is wrong. O(m).
inline std::string dataflow_plan_check(const dataflow_plan& pl, int m, bool reverse,
                                       const int* len, int pre) {
    const int nb = pl.n_batches();
    if (static_cast<int>(pl.batch_start.size()) != nb + 1) return "batch_start size != n_batches + 1";
    if (pl.batch_start.front() != 0 || pl.batch_start.back() != m) return "batch_start does not span [0, m)";
    std::vector<char> seen(static_cast<std::size_t>(m), 0);
    for (int b = 0; b < nb; ++b) {
        const int q0 = pl.batch_start[b], q1 = pl.batch_start[b + 1];
        if (q1 < q0) return "batch_start is not monotone at batch " + std::to_string(b);
        const int sb = pl.batch_spec[b];
        if (sb >= static_cast<int>(pl.spec.size())) return "batch_spec out of range at batch " + std::to_string(b);
        if (sb >= 0) {
            const dataflow_spec& sp = pl.spec[static_cast<std::size_t>(sb)];
            if (sp.row >= 0) {                        // SEGMENT: zero-width
                if (q1 != q0) return "segment batch " + std::to_string(b) + " is not zero-width";
            } else {                                  // FINALIZER: owns its row's position
                if (q1 != q0 + 1) return "finalizer batch " + std::to_string(b) + " does not own one position";
                const int row = reverse ? m - 1 - q0 : q0;
                if (~sp.row != row) return "finalizer batch " + std::to_string(b) + " is at the wrong position";
                seen[static_cast<std::size_t>(q0)] = 1;
            }
            continue;
        }
        if (q1 == q0) return "empty plain batch " + std::to_string(b);
        int pos = 0;
        for (int q = q0; q < q1; ++q) {
            const int row = reverse ? m - 1 - q : q;
            const int G = dataflow_lane_group(len[row], pre);
            pos = (pos + G - 1) & ~(G - 1);
            if (pos + G > 32) return "plain batch " + std::to_string(b) + " overflows 32 lanes";
            if (len[row] > G * pre && q1 - q0 != 1)   // the implicit warp-sharing property
                return "multi-chunk row " + std::to_string(row) + " shares batch " + std::to_string(b);
            pos += G;
            seen[static_cast<std::size_t>(q)] = 1;
        }
    }
    for (int q = 0; q < m; ++q)
        if (!seen[static_cast<std::size_t>(q)]) return "sweep position " + std::to_string(q) + " is in no batch";
    for (std::size_t k = 0; k < pl.spec.size(); ++k) {
        const dataflow_spec& sp = pl.spec[k];
        if (sp.row >= 0) continue;
        const int s0 = sp.a, S = sp.b;
        if (S < 1 || s0 < 0 || s0 + S > pl.n_slots) return "finalizer slot range out of bounds";
        if (static_cast<int>(k) < S) return "finalizer precedes its segments";
        for (int s = 0; s < S; ++s) {
            const dataflow_spec& sg = pl.spec[k - static_cast<std::size_t>(S) + static_cast<std::size_t>(s)];
            if (sg.row != ~sp.row || sg.c != s0 + s) return "segment/finalizer mismatch";
            if (sg.b <= sg.a) return "empty segment";
        }
    }
    return std::string();
}

} // namespace apxchol::cuda_host
