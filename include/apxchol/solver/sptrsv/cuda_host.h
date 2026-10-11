#pragma once
// HOST-side preparation of the GPU SpTRSV backend (cuda.h): everything
// cuda_sptrsv::setup does to the factor BEFORE the upload -- the L11 extraction
// into the dataflow kernels' int32 arrays, the compacting factor drop (the shared
// factor_drop.h implementation, the same one omp_sptrsv::setup runs), the
// fp16 per-column-scaled narrowing of the runtime fp16
// storage (APXCHOL_FACTOR_STORAGE=float16), the CSR transpose (the shared
// transpose.h implementation, the one omp_sptrsv::setup runs), the dataflow
// schedules and the dataflow batch tables. Deliberately CUDA-FREE: host FP16
// values use _Float16 and share lowprec.h's narrowing helper with CPU storage.
// Upload copies their binary16 representation into CUDA __half arrays.
// CPU unit tests can check, without a GPU, that the GPU upload matches
// CPU storage (tests/test_sptrsv_drop.cpp, "GpuHostPrep*").
//
// fp16 STORAGE CONTRACT (mirrors omp.h's FP16_SCALED, "FOLDED INTO THE
// VECTORS"): what is stored is the column-scaled factor L~ = L D^-1, D =
// diag(s_j), s_j = factor_column_scale (max |off-diagonal| of column j, 1.0f
// if none; the PRE-drop max, which the drop never removes): off-diagonal
// slots hold fp16(fp32(L_ij) / s_j) (native conversion; fp16 subnormals flushed
// to signed zero), the
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
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "apxchol/csc_work.h"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace apxchol::cuda_host {

/// An owning m x m CSR (or CSC -- the same three arrays) with int32 offsets
/// and indices, as required by the dataflow kernels, and Val values. idx / vals are
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
/// a factor exceeding int32 cannot use the dataflow kernels' index type.
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
                                 ") exceeds the dataflow GPU backend's 32-bit index range");
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
        #pragma omp parallel
        {
#ifdef _OPENMP
    const int bal_tid = omp_get_thread_num(), bal_nt = omp_get_num_threads();
#else
    const int bal_tid = 0, bal_nt = 1;
#endif
        const auto [bal_lo, bal_hi] = detail::work_balanced_range(Lo, m, bal_tid, bal_nt);
        for (std::int64_t j = bal_lo; j < bal_hi; ++j) {
            int k = 0;
            for (edge_index p = Lo[j]; p < Lo[j + 1]; ++p)
                if (Li[p] != drop) ++k;
            kept[j] = k;
        }
        }
        for (std::int64_t j = 0; j < m; ++j) out.ptr[j + 1] = out.ptr[j] + kept[j];
        out.nnz  = out.ptr[static_cast<std::size_t>(m)];
        out.idx  = std::make_unique_for_overwrite<int[]>(static_cast<std::size_t>(out.nnz));
        out.vals = std::make_unique_for_overwrite<Val[]>(static_cast<std::size_t>(out.nnz));
        #pragma omp parallel
        {
#ifdef _OPENMP
    const int bal_tid = omp_get_thread_num(), bal_nt = omp_get_num_threads();
#else
    const int bal_tid = 0, bal_nt = 1;
#endif
        const auto [bal_lo, bal_hi] = detail::work_balanced_range(Lo, m, bal_tid, bal_nt);
        for (std::int64_t j = bal_lo; j < bal_hi; ++j) {
            int o = out.ptr[j];
            for (edge_index p = Lo[j]; p < Lo[j + 1]; ++p) {
                if (Li[p] == drop) continue;
                out.idx[o]  = static_cast<int>(Li[p]);
                out.vals[o] = static_cast<Val>(Lv[p]);
                ++o;
            }
        }
        }
    }
    return out;
}

/// THE GPU backend's compacting-drop predicate (omp_sptrsv::keep_offdiag
/// restated for its two storage formats): the off-diagonal v of a column with
/// scale s survives APXCHOL_FACTOR_DROP=rel iff |v| >= rel * s and the storage
/// format does not map it to zero -- an exact zero under fp32 / fp64 storage;
/// under the fp16 storage (`fp16_storage`) also everything fp16 flushes at
/// |v / s| (fp16_flushes). The diagonal is never passed through this.
template <class Val>
inline bool keep_offdiag(Val v, float s, double rel, bool fp16_storage) {
    if (!(std::fabs(v) >= rel * s)) return false;
    if (fp16_storage) return !detail::fp16_flushes(static_cast<float>(v) / s);
    return v != Val(0);
}

/// Per-column scales s_j of the columns of `A` (factor_column_scale contract:
/// max |off-diagonal|, 1.0f if none; every column starts with its diagonal).
template <class Val>
inline std::vector<float> column_scales(const csr_int<Val>& A) {
    std::vector<float> s(static_cast<std::size_t>(A.m));
    #pragma omp parallel
    {
#ifdef _OPENMP
    const int bal_tid = omp_get_thread_num(), bal_nt = omp_get_num_threads();
#else
    const int bal_tid = 0, bal_nt = 1;
#endif
    const auto [bal_lo, bal_hi] = detail::work_balanced_range(A.ptr.data(), A.m, bal_tid, bal_nt);
    for (int j = bal_lo; j < bal_hi; ++j)
        s[j] = factor_column_scale(A.vals.get(), A.ptr[j], A.ptr[j + 1]);
    }
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
    std::unique_ptr<_Float16[]>       vals;       // nnz
    std::vector<float>               diag;       // m
    std::vector<float>               inv_scale;  // m
    std::uint64_t flushed   = 0;   // stored off-diagonals with v != 0 stored as zero
    std::uint64_t subnormal = 0;   // stored off-diagonals that are fp16 subnormals (0: they are always flushed)
};

template <class Val>
inline fp16_scaled_arrays narrow_fp16_scaled(const csr_int<Val>& L11, const std::vector<float>& col_scale) {
    fp16_scaled_arrays out;
    out.vals = std::make_unique_for_overwrite<_Float16[]>(static_cast<std::size_t>(L11.nnz));
    out.diag.resize(static_cast<std::size_t>(L11.m));
    out.inv_scale.resize(static_cast<std::size_t>(L11.m));
    std::uint64_t n_flush = 0;
    #pragma omp parallel reduction(+ : n_flush)
    {
#ifdef _OPENMP
    const int bal_tid = omp_get_thread_num(), bal_nt = omp_get_num_threads();
#else
    const int bal_tid = 0, bal_nt = 1;
#endif
    const auto [bal_lo, bal_hi] = detail::work_balanced_range(L11.ptr.data(), L11.m, bal_tid, bal_nt);
    for (int j = bal_lo; j < bal_hi; ++j) {
        const float s = col_scale[j];
        out.inv_scale[j] = 1.0f / s;
        assert(L11.idx[L11.ptr[j]] == j && "factor column must start with its diagonal");
        float  d     = static_cast<float>(L11.vals[L11.ptr[j]]) / s;   // stored_diag(): fp32(L_jj) / s_j
        double resid = 0.0;                                            // sum over the off-diagonals of (x - widen(stored))
        for (int p = L11.ptr[j]; p < L11.ptr[j + 1]; ++p) {
            const float v = static_cast<float>(L11.vals[p]);
            const _Float16 h = detail::narrow_scaled_fp16(v, s);
            out.vals[p] = h;
            if (L11.idx[p] == j) continue;   // the diagonal SLOT: written, never read (see the file header)
            const float w = float(h);
            if (v != 0.0f && w == 0.0f) ++n_flush;
            resid += static_cast<double>(L11.vals[p]) / s - w;
        }
        d = static_cast<float>(static_cast<double>(d) + resid);
        out.diag[j] = d;
    }
    }
    out.flushed = n_flush;
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
    transpose_csc_to_csr<int, int, V>(
        in.m, in.ptr.data(), in.idx.get(), in.vals.get(),
        out.ptr.data(), out.idx.get(), out.vals.get(),
        use_parallel_transpose(in.m));
    return out;
}

/// Validate the two int32 CSR structures accepted by the research-only
/// device-factor adoption boundary in cuda.h. `L` is the lower-triangular
/// forward operand and `LT` is its exact structural transpose (the
/// upper-triangular back operand). Values deliberately do not cross this host
/// boundary; the caller separately promises that corresponding value arrays
/// are transposes of one another.
///
/// The external-capsule adoption path downloads both structures to build its
/// host dataflow plans. Private finalized capsules instead plan on device. Do the complete, collision-free structural
/// check over those bytes before any pointer ownership is committed instead
/// of relying on a digest. Returns an empty string on success and a precise
/// rejection reason otherwise. CUDA-free so malformed-boundary tests run in
/// every build.
inline std::string dataflow_factor_structure_check(
        int m, std::int64_t nnz,
        const int* L_ptr, const int* L_idx,
        const int* LT_ptr, const int* LT_idx) {
    if (m <= 0) return "dimension must be positive";
    if (nnz < m || nnz > std::numeric_limits<int>::max())
        return "nnz must contain one diagonal per row and fit int32";
    if (!L_ptr || !L_idx || !LT_ptr || !LT_idx)
        return "CSR pointers must be non-null";

    if (L_ptr[0] != 0) return "CSR(L) rowptr[0] is not zero";
    if (L_ptr[m] != nnz) return "CSR(L) rowptr[m] does not equal nnz";
    if (LT_ptr[0] != 0) return "CSR(L^T) rowptr[0] is not zero";
    if (LT_ptr[m] != nnz) return "CSR(L^T) rowptr[m] does not equal nnz";

    // Validate LT and count its transpose. An exact match to this transpose
    // proves L is also sorted, triangular and has exactly one diagonal per row.
    std::vector<int> expected_ptr(static_cast<std::size_t>(m) + 1, 0);
    for (int row = 0; row < m; ++row) {
        const int beg = LT_ptr[row], end = LT_ptr[row + 1];
        if (beg < 0 || end < beg || end > nnz)
            return "CSR(L^T) rowptr is not a bounded monotone prefix at row " +
                   std::to_string(row);
        int previous = -1;
        for (int p = beg; p < end; ++p) {
            const int col = LT_idx[p];
            if (col < 0 || col >= m)
                return "CSR(L^T) has an out-of-range column at entry " + std::to_string(p);
            if (col <= previous)
                return "CSR(L^T) row " + std::to_string(row) + " is not strictly column-sorted";
            if (col < row)
                return "CSR(L^T) has an entry outside its triangular half at (" +
                       std::to_string(row) + "," + std::to_string(col) + ")";
            ++expected_ptr[static_cast<std::size_t>(col) + 1];
            previous = col;
        }
        if (beg == end || LT_idx[beg] != row)
            return "CSR(L^T) row " + std::to_string(row) + " does not contain exactly one diagonal";
    }
    for (int row = 0; row < m; ++row) {
        expected_ptr[static_cast<std::size_t>(row) + 1] +=
            expected_ptr[static_cast<std::size_t>(row)];
        if (expected_ptr[static_cast<std::size_t>(row) + 1] != L_ptr[row + 1])
            return "CSR(L) row pointers are not the transpose of CSR(L^T)";
    }

    // Consume the validated row starts as cursors in stable source-row order.
    // Their counts bound every L_idx read without trusting the supplied L_ptr.
    for (int row = 0; row < m; ++row)
        for (int p = LT_ptr[row]; p < LT_ptr[row + 1]; ++p) {
            const int out = expected_ptr[static_cast<std::size_t>(LT_idx[p])]++;
            if (L_idx[out] != row)
                return "CSR(L) column indices are not the transpose of CSR(L^T) at entry " +
                       std::to_string(out);
        }
    return {};
}

// Row-length histograms choose the split threshold and support setup diagnostics.
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

inline dataflow_len_stats dataflow_row_stats(int m, const int* rowptr, int pre) {
    dataflow_len_stats s;
    s.m = m;
    s.base = 32 * pre;
    // Integer sums and a max, so the reduction order cannot change a result.
    std::int64_t nnz = 0; int max_len = s.max_len;
    std::int64_t rows_ge[dataflow_len_stats::kBuckets]{};
    std::int64_t nnz_ge[dataflow_len_stats::kBuckets]{};
    #pragma omp parallel for schedule(static) if(m > 16384) \
        reduction(+ : nnz) reduction(max : max_len) \
        reduction(+ : rows_ge[:dataflow_len_stats::kBuckets]) \
        reduction(+ : nnz_ge[:dataflow_len_stats::kBuckets])
    for (int i = 0; i < m; ++i) {
        const int l = rowptr[i + 1] - rowptr[i];
        nnz += l;
        if (l > max_len) max_len = l;
        for (int k = 0; k < dataflow_len_stats::kBuckets; ++k) {
            if (l < (static_cast<std::int64_t>(s.base) << k)) break;
            ++rows_ge[k];
            nnz_ge[k] += l;
        }
    }
    s.nnz = nnz; s.max_len = max_len;
    for (int k = 0; k < dataflow_len_stats::kBuckets; ++k) {
        s.rows_ge[k] += rows_ge[k];
        s.nnz_ge[k] += nnz_ge[k];
    }
    s.mean_len = m > 0 ? static_cast<double>(s.nnz) / static_cast<double>(m) : 0.0;
    return s;
}

// Split after one chunk when long rows hold at least a quarter of the entries;
// otherwise wait until three chunks. The historical trade-off is documented in
// docs/implementation-history.md. Neither grid size nor scheduling affects it.
inline constexpr double kDataflowSplitShare = 0.25;
inline int dataflow_split_threshold(const dataflow_len_stats& st) {
    const double share = st.nnz ? static_cast<double>(st.nnz_ge[0]) / static_cast<double>(st.nnz) : 0.0;
    return share >= kDataflowSplitShare ? st.base : 3 * st.base;
}

} // namespace apxchol::cuda_host
