#pragma once
// The COMPACTING FACTOR DROP -- the ONE host implementation shared by every
// SpTRSV backend (omp_sptrsv::setup on the CPU, cuda_sptrsv::setup before the
// upload on the GPU, through cuda_host.h). What it does, why, and the
// measurements behind the default are documented at length in the omp.h file
// header ("COMPACTING DROP" / "COLUMN-SUM COMPENSATION"); in one paragraph:
//
//   Off-diagonal (i, j) of the factor is KEPT iff |L_ij| >= rel * s_j, s_j =
//   column j's max |off-diagonal| (column_scale(), computed BEFORE the drop --
//   the drop never removes the max), AND the backend's storage format does not
//   store it as zero anyway (the `keep` predicate the backend passes in --
//   omp_sptrsv::keep_offdiag on the CPU, cuda_host::keep_offdiag on the GPU).
//   The diagonal is always kept (every column keeps its first entry). The
//   dropped mass of each column is folded back into its kept off-diagonals in
//   proportion to |v| (a uniform rescale for the all-negative M-matrix columns
//   the elimination produces), so every column sums to what it summed to
//   (`compensate`, default ON): a Laplacian factor's zero column sums are what
//   keep L L^T a Laplacian and the grounded L11 L11^T a reduced Laplacian with
//   the right grounding mass -- plain removal costs iter0040 45 -> 67 PCG
//   iterations on the Laplacian path, the compensated drop 45 -> 45 while
//   removing 52% of the entries. O(nnz) parallel work, no atomics; the kept
//   set is a per-entry predicate, the output order is the input order and the
//   compensation sums each column in a fixed order: deterministic.
//
// Header-only, CUDA-free, OpenMP-parallel; templated on the offset / index /
// value types so the CPU backend runs it on the factor's (edge_index,
// node_index, factor_value_t) arrays and the GPU backend on its int32 cuSPARSE
// arrays -- the same code, so the two backends' compacted factors are
// identical (tests/test_sptrsv_drop.cpp states that).
#include "apxchol/types.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

namespace apxchol {

// DEFAULT relative threshold of the COMPACTING FACTOR DROP: at setup, factor
// off-diagonal (i, j) is stored in the SpTRSV's CSR/CSC iff
//   |L_ij| >= kFactorDropRelDefault * s_j,   s_j = column j's max |off-diagonal|,
// the dropped mass of each column is folded back into its kept off-diagonals
// (column sums preserved), and the diagonal is always kept. Env
// APXCHOL_FACTOR_DROP=<rel> overrides (read at every setup): 0 (or any value
// <= 0) disables the drop, any other value replaces the threshold.
//
// ON by default at 1e-4 by measurement. Branch perf/factor-drop @ fbbfd72
// (pre-center-k base, where the IPM matrix classified as SDDM; fp32 build,
// T=1, tol 1e-8, bg+tree[vec_pool], Iters/RelRes drop-off vs drop-on):
// grid_500 40/40, grid_2000 50/50, iter0040 44/44 -- 0 PCG iterations change;
// grids drop 0 entries (exact no-op), iter0040 drops 51.8% of its
// off-diagonals (stored nnz 9063231 -> 4639013), i.e. the SpTRSV's CSR/CSC
// bytes and per-sweep work roughly halve there. Re-measured on main
// (center-k grounding, iter0040 is a rank-(n-1) Laplacian factor there),
// same protocol, this port with the column-sum compensation: grid_500 40/40
// (RelRes 8.345e-09 both), grid_2000 47/47 (8.161e-09 both), iter0040 45/45
// (7.196e-09 -> 7.224e-09), com-Amazon 38/38 (7.977e-09 both, 0 dropped);
// iter0040 drops 51.6% (9032674 -> 4639053). WITHOUT the compensation
// (APXCHOL_FACTOR_DROP_COMPENSATE=0, the branch's plain-removal semantics)
// the same 1e-4 costs iter0040 45 -> 67 iterations on main's Laplacian path
// (48 -> 48 with APXCHOL_GROUND=reg, i.e. on the SDDM path the branch
// measured on, where it costs nothing either way); see the omp.h header
// comment for why. With the compensation iter0040 stays at 45 up to rel=3e-3
// (67% dropped), so 1e-4 is a conservative default, kept as the value the
// branch measured; raising it needs the IPM ladder, not one matrix.
inline constexpr double kFactorDropRelDefault = 1e-4;

// APXCHOL_FACTOR_DROP=<rel> resolution, read at every setup(): unset (or
// empty) -> kFactorDropRelDefault; set -> its value, where anything <= 0
// (including "0" and unparsable text, which atof reads as 0) turns the
// drop OFF.
inline double factor_drop_rel_from_env() {
    const char* e = std::getenv("APXCHOL_FACTOR_DROP");
    if (!e || !*e) return kFactorDropRelDefault;
    const double r = std::atof(e);
    return r > 0.0 ? r : 0.0;
}
// APXCHOL_FACTOR_DROP_COMPENSATE: unset / anything but "0" -> the
// column-sum compensation is applied (default); "0" -> plain removal (the
// A/B switch; see the omp.h file header). Read at every setup().
inline bool factor_drop_compensate_from_env() {
    const char* e = std::getenv("APXCHOL_FACTOR_DROP_COMPENSATE");
    return !(e && *e && std::atoi(e) == 0);
}

// Per-column scale contract: s_j of the factor column [first, last) whose
// FIRST entry is the diagonal: max |off-diagonal|, or 1.0f if the column has
// no nonzero off-diagonal (so v / s_j is always defined). The reference of the
// compacting drop's threshold (computed BEFORE the drop; the drop never
// removes the column max, so it is the same after) and the *_SCALED storage
// variants' scale (CPU: omp_sptrsv::column_scale; GPU fp16: cuda_host).
// Computed in double, stored fp32 (the factor is fp32 on every build that
// uses it, so this is exact).
template <class Val, class Off>
inline float factor_column_scale(const Val* vals, Off first, Off last) {
    double mx = 0.0;
    for (Off p = first + 1; p < last; ++p)
        mx = std::max(mx, std::fabs(static_cast<double>(vals[p])));
    return mx > 0.0 ? static_cast<float>(mx) : 1.0f;
}

// What the compacting drop did to the m columns handed to it (the SpTRSV
// backends expose it as drop_stats()): the threshold in effect (rel; 0 = off)
// and whether the compensation was applied, nnz before / after (nnz_stored is
// what the CSR and the CSC each hold), and how many off-diagonals went and
// why (dropped = dropped_threshold + dropped_flush -- below the threshold, or
// stored as zero by the storage format anyway; a dropped entry is attributed
// to the threshold first).
struct factor_drop_stats {
    double        rel               = 0.0;
    bool          compensate        = true;
    std::uint64_t nnz_factor        = 0;   // nnz of the columns as factorized (before the drop)
    std::uint64_t nnz_stored        = 0;   // nnz after the drop (== nnz_factor when nothing dropped)
    std::uint64_t dropped           = 0;   // = dropped_threshold + dropped_flush
    std::uint64_t dropped_threshold = 0;   //   ... because |L_ij| < rel * s_j
    std::uint64_t dropped_flush     = 0;   //   ... because the storage format stores them as zero anyway
};

/// THE compacting drop over m CSC columns (outer[m+1], inner / vals at
/// [outer[j], outer[j+1]); every column starts with its diagonal, inner == j).
/// `keep(v, s)` is the backend's off-diagonal predicate (|v| >= rel * s AND
/// the storage format keeps it -- rel is passed separately only to attribute a
/// dropped entry to the threshold vs the format); `col_scale[j]` = s_j
/// (factor_column_scale, computed by the caller BEFORE the drop -- the
/// *_SCALED storage variants keep it as their scale). If anything is dropped
/// the compacted arrays are returned in `out_outer` (m+1), `out_inner` /
/// `out_vals` (nnz_stored, allocated UNINITIALIZED -- every slot is written
/// exactly once), with the column-sum compensation applied to the kept
/// off-diagonals of every column that lost mass (when `compensate`), and the
/// function returns true; otherwise the outputs are left untouched (no copy
/// for the exact-no-op case, e.g. grids) and it returns false. `st` gets
/// nnz_factor / nnz_stored / dropped* either way (rel / compensate too).
/// O(nnz) parallel, no atomics: per-column kept count -> serial prefix over
/// m+1 -> parallel compacted copy + per-column compensation, all at the
/// factor's precision (Val), before any narrowing to a storage format.
template <class Off, class Idx, class Val, class Keep>
inline bool compact_factor_columns(Idx m, const Off* outer, const Idx* inner, const Val* vals,
                                   const float* col_scale, double rel, bool compensate, Keep keep,
                                   std::vector<Off>& out_outer, std::unique_ptr<Idx[]>& out_inner,
                                   std::unique_ptr<Val[]>& out_vals, factor_drop_stats& st) {
    const Off nnz = outer[m];
    st.rel        = rel;
    st.compensate = compensate;
    st.nnz_factor = static_cast<std::uint64_t>(nnz);
    st.nnz_stored = static_cast<std::uint64_t>(nnz);
    st.dropped = st.dropped_threshold = st.dropped_flush = 0;
    if (!(rel > 0.0)) return false;

    std::vector<Off> drop_outer(static_cast<std::size_t>(m) + 1);
    std::uint64_t n_thr = 0, n_fmt = 0;
    #pragma omp parallel for schedule(static) reduction(+ : n_thr, n_fmt)
    for (Idx j = 0; j < m; ++j) {
        const float s = col_scale[j];
        Off kept = 0;
        for (Off p = outer[j]; p < outer[j + 1]; ++p) {
            const Val v = vals[p];
            if (inner[p] == j) { ++kept; continue; }                 // diagonal: always
            if (keep(v, s)) { ++kept; continue; }
            // Dropped: attribute to the threshold first (the format-only
            // reason is what the drop removes ON TOP of the threshold).
            if (std::fabs(static_cast<double>(v)) < rel * static_cast<double>(s)) ++n_thr;
            else ++n_fmt;
        }
        drop_outer[static_cast<std::size_t>(j) + 1] = kept;
    }
    drop_outer[0] = 0;
    for (Idx j = 0; j < m; ++j)
        drop_outer[static_cast<std::size_t>(j) + 1] += drop_outer[j];
    const Off nnz_kept = drop_outer[m];
    st.dropped_threshold = n_thr;
    st.dropped_flush     = n_fmt;
    st.dropped           = n_thr + n_fmt;
    st.nnz_stored        = static_cast<std::uint64_t>(nnz_kept);
    assert(st.dropped == static_cast<std::uint64_t>(nnz - nnz_kept));
    if (nnz_kept == nnz) return false;                               // nothing dropped: no copy

    std::unique_ptr<Idx[]> drop_inner(new Idx[static_cast<std::size_t>(nnz_kept)]);
    std::unique_ptr<Val[]> drop_vals(new Val[static_cast<std::size_t>(nnz_kept)]);
    #pragma omp parallel for schedule(static)
    for (Idx j = 0; j < m; ++j) {
        const float s = col_scale[j];
        const Off first = drop_outer[j];
        Off out = first;
        double dropped_sum = 0.0, kept_abs = 0.0;   // over the off-diagonals
        for (Off p = outer[j]; p < outer[j + 1]; ++p) {
            const Val v = vals[p];
            if (inner[p] != j) {
                if (!keep(v, s)) {
                    dropped_sum += static_cast<double>(v);
                    continue;
                }
                kept_abs += std::fabs(static_cast<double>(v));
            }
            drop_inner[out] = inner[p];
            drop_vals[out]  = v;
            ++out;
        }
        assert(out == drop_outer[static_cast<std::size_t>(j) + 1]);
        // Column-sum compensation (see the omp.h file header): spread the
        // dropped mass over the kept off-diagonals in proportion to |v|, so
        // the column sum -- hence L~ L~^T's row sums, the Laplacian structure
        // and the grounding mass -- is what it was. Same fixed order in every
        // thread: deterministic.
        if (compensate && dropped_sum != 0.0 && kept_abs > 0.0) {
            const double per_abs = dropped_sum / kept_abs;
            for (Off q = first; q < out; ++q) {
                if (drop_inner[q] == j) continue;
                const double v = static_cast<double>(drop_vals[q]);
                drop_vals[q] = static_cast<Val>(v + per_abs * std::fabs(v));
            }
        }
    }
    out_outer.swap(drop_outer);
    out_inner = std::move(drop_inner);
    out_vals  = std::move(drop_vals);
    return true;
}

} // namespace apxchol
