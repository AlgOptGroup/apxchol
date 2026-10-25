#pragma once
// Compact host factor columns before storage conversion. Keep the diagonal and
// off-diagonals accepted by the threshold/format predicate. Redistribute dropped
// mass among retained off-diagonals when any remain; otherwise keep the diagonal
// alone. Preserve entry order and ordered per-column compensation.
// See docs/precision.md for the storage contract.
#include "apxchol/csc_work.h"
#include "apxchol/types.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace apxchol {

// Per-column scale contract: s_j of the factor column [first, last) whose
// FIRST entry is the diagonal: max |off-diagonal|, or 1.0f if the column has
// no nonzero off-diagonal (so v / s_j is always defined). The reference of the
// compacting drop's threshold is computed before dropping; the same scale is
// used for normalized FP16 storage.
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
// backends expose it as drop_stats()): the threshold in effect (rel; 0 = off),
// nnz before / after (nnz_stored is
// what the CSR and the CSC each hold), and how many off-diagonals went and
// why (dropped = dropped_threshold + dropped_flush -- below the threshold, or
// stored as zero by the storage format anyway; a dropped entry is attributed
// to the threshold first).
struct factor_drop_stats {
    double        rel               = 0.0;
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
/// off-diagonals of every column that lost mass, and the
/// function returns true; otherwise the outputs are left untouched (no copy
/// for the exact-no-op case, e.g. grids) and it returns false. `st` gets
/// rel / nnz_factor / nnz_stored / dropped* either way.
/// O(nnz) parallel, no atomics: per-column kept count -> serial prefix over
/// m+1 -> parallel compacted copy + per-column compensation, all at the
/// factor's precision (Val), before any narrowing to a storage format.
template <class Off, class Idx, class Val, class Keep>
inline bool compact_factor_columns(Idx m, const Off* outer, const Idx* inner, const Val* vals,
                                   const float* col_scale, double rel, Keep keep,
                                   std::vector<Off>& out_outer, std::unique_ptr<Idx[]>& out_inner,
                                   std::unique_ptr<Val[]>& out_vals, factor_drop_stats& st) {
    const Off nnz = outer[m];
    st.rel        = rel;
    st.nnz_factor = static_cast<std::uint64_t>(nnz);
    st.nnz_stored = static_cast<std::uint64_t>(nnz);
    st.dropped = st.dropped_threshold = st.dropped_flush = 0;
    if (!(rel > 0.0)) return false;

    std::vector<Off> drop_outer(static_cast<std::size_t>(m) + 1);
    std::uint64_t n_thr = 0, n_fmt = 0;
    // Split by stored entries, not by column count: the body is proportional to
    // the column's length, and the equal-count split leaves one thread with up
    // to 4x the mean on a hub factor (detail::work_balanced_range). Every
    // column writes its own slot and the reductions are integer counts, so the
    // split cannot change a result.
    #pragma omp parallel reduction(+ : n_thr, n_fmt)
    {
    const auto [bal_lo, bal_hi] = detail::work_balanced_range(outer, m);
    for (Idx j = bal_lo; j < bal_hi; ++j) {
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
    #pragma omp parallel
    {
    const auto [bal_lo, bal_hi] = detail::work_balanced_range(outer, m);
    for (Idx j = bal_lo; j < bal_hi; ++j) {
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
        // Column-sum compensation (see docs/precision.md): spread the
        // dropped mass over the kept off-diagonals in proportion to |v|, so
        // the column sum -- hence L~ L~^T's row sums, the Laplacian structure
        // and the grounding mass -- is what it was. Same fixed order in every
        // thread: deterministic.
        if (dropped_sum != 0.0 && kept_abs > 0.0) {
            const double per_abs = dropped_sum / kept_abs;
            for (Off q = first; q < out; ++q) {
                if (drop_inner[q] == j) continue;
                const double v = static_cast<double>(drop_vals[q]);
                drop_vals[q] = static_cast<Val>(v + per_abs * std::fabs(v));
            }
        }
    }
    }
    out_outer.swap(drop_outer);
    out_inner = std::move(drop_inner);
    out_vals  = std::move(drop_vals);
    return true;
}

} // namespace apxchol
