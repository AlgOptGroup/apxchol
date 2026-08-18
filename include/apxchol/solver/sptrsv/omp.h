#pragma once
#include "apxchol/types.h"
#include "apxchol/sparse_csc.h"
#include "apxchol/big_alloc.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string_view>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace apxchol {

// Minimum level-set size before engaging OpenMP for SpTRSV.
// Below this threshold, the thread-dispatch overhead exceeds the
// computational benefit of parallelizing a single level's rows/cols.
inline constexpr node_index kSpTRSVOMPThreshold = 1024;

// DEFAULT relative threshold of the COMPACTING FACTOR DROP (see the "COMPACTING
// DROP" section of the omp_sptrsv header comment below): at setup, factor
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
// measured on, where it costs nothing either way); see the header comment
// for why. With the compensation iter0040 stays at 45 up to rel=3e-3
// (67% dropped), so 1e-4 is a conservative default, kept as the value the
// branch measured; raising it needs the IPM ladder, not one matrix.
inline constexpr double kFactorDropRelDefault = 1e-4;

/// OpenMP parallel sparse triangular solver with two interchangeable schedulers:
///
///   1. **Level-set scheduler** (Anderson & Saad 1989; Naumov 2011):
///      Pre-computes topological levels of the dependency DAG.  Each
///      level is a parallel-for, with an implicit barrier between
///      levels.  Best when the factor has few, fat levels, or when a
///      few fat levels carry nearly all the work.
///
///   2. **Synchronization-free scheduler** (Liu, Smelyanskiy, Chow 2016;
///      Park et al. 2014; Su, Yang, Zhao 2020 "SyncFree"):
///      No barriers.  Each row carries an atomic counter of unresolved
///      dependencies; threads pick rows dynamically and busy-wait on
///      the counter, then notify dependents via atomic decrement.
///      Could win only on genuinely deep factors of uniformly-thin
///      levels with no fat work head.
///
/// Both schedulers reuse the same CSR (forward) / CSC (back) storage.
/// The default `forward_solve` and `transpose_solve` entry points both use
/// the level-set scheduler: paired A/B across the benchmark suite showed
/// sync-free never beats level-set on any factor the current elimination
/// produces (power-law factors have a tiny AVERAGE level size, but a few
/// very fat early levels carry nearly all the work — see the
/// APXCHOL_LEVEL_DUMP work-concentration stats).  The sync-free back solve
/// remains as an opt-in escape hatch via APXCHOL_BCK_SCHED=syncfree for
/// orderings that do produce deep-thin factors.

// sptrsv_value_t (fp32 under -DAPXCHOL_SPTRSV_FP32, else fp64) is defined in
// sparse_csc.h and shared with the CUDA backend. It halves csr_vals_/csc_vals_
// -- the two largest factor copies -- cutting memory and bandwidth on the
// bandwidth-bound (~94% of peak) triangular solve; diag + outer PCG stay fp64.
//
// COMPACTING DROP (APXCHOL_FACTOR_DROP=<rel>, default kFactorDropRelDefault =
// 1e-4, read at every setup; <= 0 = off): setup() REMOVES -- not zeroes --
// factor off-diagonals BEFORE the CSR/CSC are built: entry (i, j), i != j, is
// KEPT iff
//   |L_ij| >= rel * s_j   (s_j = column j's max |off-diagonal|, column_scale(),
//                          computed from the factor BEFORE the drop).
// keep_offdiag() is the pure predicate. The diagonal is always kept, so every
// column keeps its first entry (the diagonal-first CSC / diagonal-last CSR
// invariants the solves rely on hold) and s_j is unchanged by the drop (the
// column max itself is never below rel * s_j for rel <= 1). Exact zeros
// always go (|0| < rel * s_j for any rel > 0). The drop is O(nnz) parallel
// work with no atomics -- per-column kept count -> serial prefix over m+1 ->
// parallel compacted copy into uninitialized buffers -- and the compacted
// arrays REPLACE L11_{outer,inner,vals} / nnz for everything downstream: the
// CSR transpose, the CSC copy, the level sets (the topological scan reads the
// compacted CSR/CSC; round-as-level bounds are per column and unaffected) and
// the sync-free counters are all drop-agnostic. Result: nnz(L stored) -- the
// CSR/CSC bytes and the per-sweep work -- shrink. Deterministic: the kept set
// is a per-entry predicate on the factor, the output order is the input
// order, and the compensation below sums each column in a fixed order (the
// factor itself is deterministic at T=1; at T>1 only fp merge-order ulps
// differ, as before). When nothing is below the threshold (grids) the arrays
// are left untouched -- no copy is made.
//
// COLUMN-SUM COMPENSATION (default ON; APXCHOL_FACTOR_DROP_COMPENSATE=0, read
// at every setup, gives plain removal for A/B): every column of the factor of
// a Laplacian sums to zero -- L(j,j) = sqrt(deg_j), L(i,j) = -w_ij/sqrt(deg_j)
// -- which is exactly why L L^T again has zero row sums (a Laplacian in the
// generalized sense: A + a zero-row-sum sampling error) and why grounding the
// last vertex costs nothing: (L L^T)_11 = L11 L11^T is the reduced form of
// that approximation, with the same total grounding mass as the true A_11. Plain removal of an entry L(i,j) breaks the column sum: to
// first order it deletes the tiny edge (i,j) from the approximate Laplacian
// AND leaves its weight behind as a self-loop at both endpoints -- in the
// grounded (Laplacian, m = n-1) view, an extra edge to ground. Modes that are
// only weakly grounded in A_11 (regions far from the hub in conductance) then
// see a preconditioner that grounds them much more strongly, and PCG pays:
// measured on main's iter0040 (Laplacian path) plain removal at rel=1e-4
// costs 45 -> 67 iterations, while on the SDDM path (APXCHOL_GROUND=reg,
// where residuals stay orthogonal to the one near-null direction) it costs
// nothing (48 -> 48; the branch that introduced the drop measured 44 -> 44
// on that path). The compensation folds each column's dropped mass back into
// its kept off-diagonals in proportion to |v| (for the M-matrix factors we
// produce -- all off-diagonals <= 0 -- that is a uniform rescale of the kept
// off-diagonals by 1 + |dropped| / |kept|; the |v| weighting only keeps it
// bounded on a mixed-sign column). Column sums -- hence the zero row sums of
// L~ L~^T and the total grounding mass of L~11 L~11^T (= sum of squared column
// sums of L~11 = the hub row's) -- are preserved exactly (up to fp32
// rounding); the perturbation is a bounded relative change of the kept edges
// instead of an unbounded relative change of the grounding. Measured on iter0040 (main): 45 -> 45 iterations at rel = 1e-4,
// 1e-3 and 3e-3 (52 / 65 / 67% of the off-diagonals dropped). This is the
// same idea as MILU's row-sum preservation. Measurement behind the default:
// see kFactorDropRelDefault.
//
// STATISTICS: drop_stats() / stored_nnz() report the factor's nnz before and
// after the drop and how many off-diagonals went; APXCHOL_VERBOSE prints one
// "[apxchol] sptrsv storage" line per setup.

class omp_sptrsv {
public:
    omp_sptrsv() = default;

    // Compiled width of the off-diagonal factor values, read straight off the
    // type the compiler actually built. 4 == this TU was compiled with
    // -DAPXCHOL_SPTRSV_FP32, 8 == fp64. Printed at startup so the build flag is
    // observable at runtime (no inferring from residuals).
    static constexpr std::size_t value_bytes = sizeof(sptrsv_value_t);
    static constexpr const char* value_name =
        sizeof(sptrsv_value_t) == 4 ? "float (fp32)" : "double (fp64)";

    // THE compacting-drop predicate (public so the tests can state it): the
    // off-diagonal v of a column whose max |off-diagonal| is s survives
    // APXCHOL_FACTOR_DROP=rel iff |v| >= rel * s. The diagonal is never
    // passed through this (always kept). Pure.
    static bool keep_offdiag(sptrsv_value_t v, double s, double rel) {
        return std::fabs(static_cast<double>(v)) >= rel * s;
    }

    // Per-column scale contract (public for the tests): s_j of the factor
    // column [first, last) whose FIRST entry is the diagonal (the assembler's
    // CSC invariant, which the back solve relies on too): max |off-diagonal|,
    // or 1.0 if the column has no nonzero off-diagonal. The drop's threshold
    // reference; computed BEFORE the drop (the drop never removes the column
    // max, so it is the same after).
    static double column_scale(const sptrsv_value_t* vals, edge_index first, edge_index last) {
        double mx = 0.0;
        for (edge_index p = first + 1; p < last; ++p)
            mx = std::max(mx, std::fabs(static_cast<double>(vals[p])));
        return mx > 0.0 ? mx : 1.0;
    }

    // APXCHOL_FACTOR_DROP=<rel> resolution, read at every setup(): unset (or
    // empty) -> kFactorDropRelDefault; set -> its value, where anything <= 0
    // (including "0" and unparsable text, which atof reads as 0) turns the
    // drop OFF.
    static double factor_drop_rel_from_env() {
        const char* e = std::getenv("APXCHOL_FACTOR_DROP");
        if (!e || !*e) return kFactorDropRelDefault;
        const double r = std::atof(e);
        return r > 0.0 ? r : 0.0;
    }
    // APXCHOL_FACTOR_DROP_COMPENSATE: unset / anything but "0" -> the
    // column-sum compensation is applied (default); "0" -> plain removal (the
    // A/B switch; see the file header). Read at every setup().
    static bool factor_drop_compensate_from_env() {
        const char* e = std::getenv("APXCHOL_FACTOR_DROP_COMPENSATE");
        return !(e && *e && std::atoi(e) == 0);
    }

    // Statistics of the last setup() (see the file header): what the
    // compacting drop did to L11 = L.topLeftCorner(m, m).
    struct drop_statistics {
        double        rel        = 0.0;  // threshold in effect (0 = drop off)
        bool          compensate = true; // column-sum compensation applied to the dropped columns
        std::uint64_t nnz_factor = 0;    // nnz of L11 as factorized (before the drop)
        std::uint64_t nnz_stored = 0;    // nnz the CSR (and the CSC) each hold (after the drop)
        std::uint64_t dropped    = 0;    // off-diagonals removed (== nnz_factor - nnz_stored)
    };
    const drop_statistics& drop_stats() const { return stats_; }
    // nnz held by the SpTRSV's CSR / CSC (each) after the last setup().
    std::uint64_t stored_nnz() const { return stats_.nnz_stored; }

    /// Analyze L11 = L.topLeftCorner(m, m): build CSR, CSC, and level sets.
    /// L is read only; the caller keeps it.
    void setup(const sparse_csc& L, node_index m) { setup_impl(L, m, nullptr); }

    /// Same analysis, but the SpTRSV CONSUMES L: its row/value arrays are
    /// released (sparse_csc::release_values) at the first point setup no longer
    /// reads them, instead of surviving until the caller frees them after setup.
    /// On the Laplacian path (m == n-1) that is right after the L11 copy, so
    /// the setup transient -- L11 + the transpose bucket + CSR (+ CSC) -- no
    /// longer sits on top of a dead full copy of the factor (nnz*(4+sizeof
    /// value) B: ~70 MB on iter0040, ~180 MB on grid_2000). On the SDDM path
    /// L11 aliases L, so the release happens after the compacting drop copied it
    /// (if it did) or else after the CSC copy. Column pointers stay (nonZeros()
    /// still works). For callers that keep the factor, use setup().
    void setup_consuming(sparse_csc& L, node_index m) { setup_impl(L, m, &L); }

private:
    // `consumed` != nullptr: `L` may be released as soon as it is dead (see
    // setup_consuming); it then always aliases &L.
    void setup_impl(const sparse_csc& L, node_index m, sparse_csc* consumed) {
        m_ = m;
        const bool trace = std::getenv("APXCHOL_SPTRSV_SETUP_TRACE") != nullptr;
        auto now = []() {
            return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        };
        const double t_start = trace ? now() : 0;
        double t_last = t_start;
        auto mark = [&](const char* label) {
            if (!trace) return;
            const double t = now();
            std::fprintf(stderr, "[sptrsv-setup] %-20s %.0f ms\n", label, t - t_last);
            t_last = t;
        };
        // For SDDM (m == n) we alias L's CSC storage directly — no copy.
        // For Laplacian (m == n-1) we build L11 = L.topLeftCorner(m, m) in
        // parallel: keep entries from columns 0..m-1 with row < m, drop
        // any row==n-1 entries that fall in the kept columns. Eigen's
        // built-in topLeftCorner + makeCompressed is serial and costs
        // ~60 ms on IPM iter40 — manual parallel version drops it under 10 ms.
        std::vector<edge_index>     L11_outer_local;
        std::vector<node_index>     L11_inner_local;
        std::vector<sptrsv_value_t> L11_vals_local;
        const edge_index*     L11_outer;
        const node_index*     L11_inner;
        const sptrsv_value_t* L11_vals;     // fp32 under the flag (diag read separately)
        edge_index nnz;
        if (m == L.rows()) {
            // SDDM full matrix: alias L (assume already compressed by
            // assemble_csc). Pointer copies only — nominally ~0 ms.
            L11_outer = L.outerIndexPtr();
            L11_inner = L.innerIndexPtr();
            L11_vals  = L.valuePtr();
            nnz = L.nonZeros();
        } else {
            // Parallel build of L11 = L.topLeftCorner(m, m).
            const edge_index* L_outer = L.outerIndexPtr();
            const node_index* L_inner = L.innerIndexPtr();
            const sptrsv_value_t* L_vals = L.valuePtr();
            const node_index drop_row = L.rows() - 1;
            // PASS 1 (parallel): per-column count of entries with row < m.
            std::vector<node_index> col_kept(m_, 0);
            #pragma omp parallel for schedule(static)
            for (node_index j = 0; j < m_; ++j) {
                node_index kept = 0;
                for (edge_index p = L_outer[j]; p < L_outer[j + 1]; ++p)
                    if (L_inner[p] != drop_row) ++kept;
                col_kept[j] = kept;
            }
            // Prefix sum (serial, m_+1 entries — sub-ms).
            L11_outer_local.resize(static_cast<size_t>(m_) + 1);
            L11_outer_local[0] = 0;
            for (node_index j = 0; j < m_; ++j)
                L11_outer_local[j + 1] = L11_outer_local[j] + col_kept[j];
            nnz = L11_outer_local[m_];
            L11_inner_local.resize(static_cast<size_t>(nnz));
            L11_vals_local.resize(static_cast<size_t>(nnz));
            // PASS 2 (parallel): scatter entries to compacted positions.
            #pragma omp parallel for schedule(static)
            for (node_index j = 0; j < m_; ++j) {
                edge_index out = L11_outer_local[j];
                for (edge_index p = L_outer[j]; p < L_outer[j + 1]; ++p) {
                    if (L_inner[p] == drop_row) continue;
                    L11_inner_local[out] = L_inner[p];
                    L11_vals_local[out]  = L_vals[p];
                    ++out;
                }
            }
            L11_outer = L11_outer_local.data();
            L11_inner = L11_inner_local.data();
            L11_vals  = L11_vals_local.data();
            // L11 is a full copy: the input factor is dead from here on.
            if (consumed) consumed->release_values();
        }
        mark("L11_alias_or_copy");

        // ── Compacting drop (APXCHOL_FACTOR_DROP; see the file header) ──
        // O(nnz) parallel work, no atomics: per-column kept count -> serial
        // prefix over m_+1 -> parallel compacted copy. The compacted arrays
        // REPLACE L11_{outer,inner,vals} / nnz for everything below (transpose,
        // CSC copy, level sets), so the rest of setup is drop-agnostic. Order
        // within a column is preserved (diagonal stays first). The buffers are
        // allocated uninitialized (every slot is written exactly once). If no
        // entry is below the threshold the original arrays stay in place (no
        // second copy of the factor for the exact-no-op case, e.g. grids).
        const double factor_drop_rel = factor_drop_rel_from_env();
        const bool   drop_compensate = factor_drop_compensate_from_env();
        stats_ = drop_statistics{};
        stats_.rel        = factor_drop_rel;
        stats_.compensate = drop_compensate;
        stats_.nnz_factor = static_cast<std::uint64_t>(nnz);
        std::vector<edge_index>           drop_outer;
        std::unique_ptr<node_index[]>     drop_inner;
        std::unique_ptr<sptrsv_value_t[]> drop_vals;
        if (factor_drop_rel > 0.0) {
            // Per-column threshold reference s_j, from the factor BEFORE the
            // drop (column_scale() contract).
            std::vector<double> col_scale(m_);
            #pragma omp parallel for schedule(static)
            for (node_index j = 0; j < m_; ++j)
                col_scale[j] = column_scale(L11_vals, L11_outer[j], L11_outer[j + 1]);
            drop_outer.resize(static_cast<size_t>(m_) + 1);
            #pragma omp parallel for schedule(static)
            for (node_index j = 0; j < m_; ++j) {
                const double s = col_scale[j];
                edge_index kept = 0;
                for (edge_index p = L11_outer[j]; p < L11_outer[j + 1]; ++p)
                    if (L11_inner[p] == j || keep_offdiag(L11_vals[p], s, factor_drop_rel)) ++kept;
                drop_outer[j + 1] = kept;
            }
            drop_outer[0] = 0;
            for (node_index j = 0; j < m_; ++j)
                drop_outer[j + 1] += drop_outer[j];
            const edge_index nnz_kept = drop_outer[m_];
            if (nnz_kept != nnz) {
                drop_inner.reset(new node_index[nnz_kept]);
                drop_vals.reset(new sptrsv_value_t[nnz_kept]);
                #pragma omp parallel for schedule(static)
                for (node_index j = 0; j < m_; ++j) {
                    const double s = col_scale[j];
                    const edge_index first = drop_outer[j];
                    edge_index out = first;
                    double dropped_sum = 0.0, kept_abs = 0.0;   // over the off-diagonals
                    for (edge_index p = L11_outer[j]; p < L11_outer[j + 1]; ++p) {
                        const sptrsv_value_t v = L11_vals[p];
                        if (L11_inner[p] != j) {
                            if (!keep_offdiag(v, s, factor_drop_rel)) {
                                dropped_sum += static_cast<double>(v);
                                continue;
                            }
                            kept_abs += std::fabs(static_cast<double>(v));
                        }
                        drop_inner[out] = L11_inner[p];
                        drop_vals[out]  = v;
                        ++out;
                    }
                    assert(out == drop_outer[j + 1]);
                    // Column-sum compensation (see the file header): spread the
                    // dropped mass over the kept off-diagonals in proportion to
                    // |v|, so the column sum -- hence L~ L~^T's row sums, the
                    // Laplacian structure and the grounding mass -- is what it
                    // was. Same fixed order in every thread: deterministic.
                    if (drop_compensate && dropped_sum != 0.0 && kept_abs > 0.0) {
                        const double per_abs = dropped_sum / kept_abs;
                        for (edge_index q = first; q < out; ++q) {
                            if (drop_inner[q] == j) continue;
                            const double v = static_cast<double>(drop_vals[q]);
                            drop_vals[q] = static_cast<sptrsv_value_t>(v + per_abs * std::fabs(v));
                        }
                    }
                }
                stats_.dropped = static_cast<std::uint64_t>(nnz - nnz_kept);
                L11_outer = drop_outer.data();
                L11_inner = drop_inner.get();
                L11_vals  = drop_vals.get();
                nnz = nnz_kept;
                // The Laplacian path's L11 copy is dead now: free it before the
                // transpose allocates its bucket (peak memory, not speed).
                // (`v = {}` would only clear -- it keeps the capacity; swapping
                // with an empty vector actually returns the memory.)
                std::vector<node_index>().swap(L11_inner_local);
                std::vector<sptrsv_value_t>().swap(L11_vals_local);
                std::vector<edge_index>().swap(L11_outer_local);
                // SDDM path: L11 aliased the input factor, which the compacted
                // copy has just replaced -- the input is dead now.
                if (consumed && m == L.rows())
                    consumed->release_values();
            } else {
                std::vector<edge_index>().swap(drop_outer);   // nothing dropped: keep aliasing the input
            }
            mark("factor_drop");
        }
        stats_.nnz_stored = static_cast<std::uint64_t>(nnz);
        if (std::getenv("APXCHOL_VERBOSE")) {
            const std::uint64_t off0 = stats_.nnz_factor - static_cast<std::uint64_t>(m_);
            std::fprintf(stderr,
                "[apxchol] sptrsv storage %s: stored_nnz=%llu (L11_nnz=%llu);"
                " factor drop (APXCHOL_FACTOR_DROP=%g%s%s): dropped=%llu (%.4f%% of %llu off-diagonals)\n",
                value_name,
                static_cast<unsigned long long>(stats_.nnz_stored),
                static_cast<unsigned long long>(stats_.nnz_factor),
                factor_drop_rel, factor_drop_rel > 0.0 ? "" : ", off",
                factor_drop_rel > 0.0 ? (drop_compensate ? ", column sums preserved"
                                                         : ", plain removal (COMPENSATE=0)") : "",
                static_cast<unsigned long long>(stats_.dropped),
                100.0 * static_cast<double>(stats_.dropped) / (off0 ? static_cast<double>(off0) : 1.0),
                static_cast<unsigned long long>(off0));
        }

        // ── CSC → CSR of L11 (for forward solve) ─────
        // Blocked counting-sort parallel transpose (APXCHOL_PAR_TRANSPOSE,
        // default on for large m): O(nnz) TOTAL work. Replaces the row-range
        // rescan scheme, where every thread scanned ALL nnz twice filtering to
        // its own row range — O(threads·nnz) total work, so the stage's wall
        // time was flat in thread count (71-76% of sptrsv_setup at T=16).
        //
        //   Phase 1 (parallel): threads own contiguous, nnz-balanced COLUMN
        //     ranges. Rows are split into NB ≤ 4·nt blocks of power-of-two
        //     width (block-of-row is a shift). Thread t histograms its column
        //     range's entries per row block into cnt[t][b].
        //   Phase 2 (parallel): a serial exclusive prefix over cnt in
        //     (b major, t minor) order gives every (thread, block) pair an
        //     exact segment in an nnz-sized bucket. Thread t re-scans its
        //     column range in ascending-column order, appending each entry
        //     (row, col, value) to its segment of the entry's row block.
        //   Phase 3 (parallel over blocks): block b's bucket region
        //     [blk_off[b], blk_off[b+1]) holds exactly its rows' entries in
        //     ascending-column order (segments are concatenated in thread
        //     order == ascending column ranges, each internally ascending).
        //     A per-block counting sort — count rows, exclusive prefix based
        //     at blk_off[b] (which IS the final CSR offset of the block's
        //     first row, since blocks partition rows contiguously), then a
        //     stable scatter — lands every entry in its final CSR slot and
        //     fills the block's csr_row_ptr_ range.
        //
        // Determinism / byte-identity: within each row, the stable per-block
        // scatter preserves the bucket's ascending-column order — exactly the
        // order the serial scatter (columns walked 0..m-1) produces — and
        // csr_row_ptr_ is uniquely determined by the row counts. All three
        // CSR arrays are therefore byte-identical to the serial result for
        // ANY thread count (verified by SpTRSVTranspose.* unit tests).
        //
        // Memory: one transient bucket of nnz·(2·sizeof(node_index) +
        // sizeof(sptrsv_value_t)) bytes (12 B/nnz on the default
        // 32-bit-index fp32 build), allocated uninitialized (every slot is
        // written exactly once in phase 2) and freed before the level-set
        // build, plus the nt×NB count matrix (≤ 32·129·8 B ≈ 33 KB).
        //
        // Rejected alternatives:
        //   (a) per-thread full row histograms — nt × m × 4 B per call wins
        //       on small-n high-fanout factors but the allocation explodes on
        //       multi-million-row grids (256 MB at m=4M, nt=16);
        //   (b) atomic-claim on shared O(m) counters — avoids the n × nt
        //       memory but cache-line ping-pong on the m-sized counters under
        //       high thread counts makes it slower than serial on both shapes;
        //   (c) row-range rescan (the previous scheme) — no extra memory, but
        //       O(threads·nnz) total work: flat wall time in thread count.
        //
        // Below the size threshold the SERIAL scatter wins (thread dispatch
        // overhead exceeds the work).
        static const bool kParTranspose = [] {
            const char* e = std::getenv("APXCHOL_PAR_TRANSPOSE");
            return e ? std::atoi(e) != 0 : true;
        }();
        const bool par_tr = kParTranspose && m_ > 50000;
        csr_row_ptr_.assign(static_cast<size_t>(m_) + 1, 0);
        csr_col_idx_.resize(nnz);
        csr_vals_.resize(nnz);
        if (par_tr) {
            // Transient bucket, deliberately UNINITIALIZED (a std::vector
            // would serially memset up to nnz·12 B): phase 2 writes every
            // slot exactly once before phase 3 reads it.
            std::unique_ptr<node_index[]>     bkt_row(new node_index[nnz]);
            std::unique_ptr<node_index[]>     bkt_col(new node_index[nnz]);
            std::unique_ptr<sptrsv_value_t[]> bkt_val(new sptrsv_value_t[nnz]);
            std::vector<edge_index> cnt;      // cnt[t*NB + b]: per-(thread, block) counts
            std::vector<edge_index> seg_off;  // segment starts, same layout as cnt
            std::vector<edge_index> blk_off;  // NB+1 block starts (== final CSR offsets)
            int        shift = 0;             // log2(rows per block)
            node_index NB    = 1;             // number of row blocks
            #pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                const int nt  = omp_get_num_threads();
                #pragma omp single
                {
                    // Row-block width: smallest power of two >= ceil(m/(4·nt)),
                    // so NB <= 4·nt (~4 blocks/thread for phase-3 balance) and
                    // block-of-row is a shift.
                    const node_index target =
                        (m_ + 4 * static_cast<node_index>(nt) - 1) / (4 * static_cast<node_index>(nt));
                    node_index rpb = 1;
                    shift = 0;
                    while (rpb < target) { rpb <<= 1; ++shift; }
                    NB = ((m_ - 1) >> shift) + 1;   // == ceil(m/rpb); m_ > 50000 here
                    cnt.assign(static_cast<size_t>(nt) * NB, 0);
                }   // implicit barrier: cnt/NB/shift visible to all threads
                // Contiguous nnz-balanced column range for this thread (a
                // single column is never split, so one pathologically dense
                // column bounds the imbalance). Boundaries are monotone in
                // tid, so ranges are disjoint and cover every entry.
                auto col_at = [&](int t) {
                    const edge_index tgt = static_cast<edge_index>(
                        static_cast<std::uint64_t>(nnz) * t / nt);
                    return static_cast<node_index>(
                        std::lower_bound(L11_outer, L11_outer + m_ + 1, tgt) - L11_outer);
                };
                const node_index c_lo = col_at(tid);
                const node_index c_hi = col_at(tid + 1);
                // Phase 1: per-(thread, block) histogram.
                edge_index* my_cnt = cnt.data() + static_cast<size_t>(tid) * NB;
                for (node_index j = c_lo; j < c_hi; ++j)
                    for (edge_index p = L11_outer[j]; p < L11_outer[j + 1]; ++p)
                        my_cnt[L11_inner[p] >> shift]++;
                #pragma omp barrier
                #pragma omp single
                {
                    // Exclusive prefix in (b major, t minor) order: segment
                    // concatenation inside each block follows ascending
                    // column ranges. O(nt·NB) — trivially serial.
                    seg_off.resize(cnt.size());
                    blk_off.resize(static_cast<size_t>(NB) + 1);
                    edge_index run = 0;
                    for (node_index b = 0; b < NB; ++b) {
                        blk_off[b] = run;
                        for (int t = 0; t < nt; ++t) {
                            seg_off[static_cast<size_t>(t) * NB + b] = run;
                            run += cnt[static_cast<size_t>(t) * NB + b];
                        }
                    }
                    blk_off[NB] = run;   // == nnz
                }   // implicit barrier
                // Phase 2: scatter into the bucket (private cursors; each
                // (thread, block) segment is written by one thread only).
                std::vector<edge_index> cur(
                    seg_off.begin() + static_cast<size_t>(tid) * NB,
                    seg_off.begin() + static_cast<size_t>(tid + 1) * NB);
                for (node_index j = c_lo; j < c_hi; ++j)
                    for (edge_index p = L11_outer[j]; p < L11_outer[j + 1]; ++p) {
                        const node_index row = L11_inner[p];
                        const edge_index out = cur[row >> shift]++;
                        bkt_row[out] = row;
                        bkt_col[out] = j;
                        bkt_val[out] = static_cast<sptrsv_value_t>(L11_vals[p]);
                    }
                #pragma omp barrier
                // Phase 3: per-block stable counting sort into the final CSR
                // (each block owns disjoint row + output ranges — race-free
                // and schedule-independent).
                #pragma omp for schedule(dynamic, 1)
                for (node_index b = 0; b < NB; ++b) {
                    const node_index r_lo = static_cast<node_index>(
                        static_cast<std::uint64_t>(b) << shift);
                    const node_index r_hi = static_cast<node_index>(std::min<std::uint64_t>(
                        m_, static_cast<std::uint64_t>(b + 1) << shift));
                    std::vector<edge_index> pos(r_hi - r_lo, 0);
                    for (edge_index k = blk_off[b]; k < blk_off[b + 1]; ++k)
                        pos[bkt_row[k] - r_lo]++;
                    edge_index run = blk_off[b];   // final CSR offset of row r_lo
                    for (node_index r = r_lo; r < r_hi; ++r) {
                        const edge_index c = pos[r - r_lo];
                        csr_row_ptr_[r] = run;
                        pos[r - r_lo]   = run;
                        run += c;
                    }
                    for (edge_index k = blk_off[b]; k < blk_off[b + 1]; ++k) {
                        const edge_index out = pos[bkt_row[k] - r_lo]++;
                        csr_col_idx_[out] = bkt_col[k];
                        csr_vals_[out]    = bkt_val[k];
                    }
                }   // implicit barrier
            }
            csr_row_ptr_[m_] = nnz;
        } else {
            for (node_index j = 0; j < m_; ++j)
                for (edge_index p = L11_outer[j]; p < L11_outer[j + 1]; ++p)
                    csr_row_ptr_[L11_inner[p] + 1]++;
            for (node_index i = 0; i < m_; ++i)
                csr_row_ptr_[i + 1] += csr_row_ptr_[i];
            std::vector<edge_index> pos(csr_row_ptr_.begin(),
                                        csr_row_ptr_.begin() + m_);
            for (node_index j = 0; j < m_; ++j) {
                for (edge_index p = L11_outer[j]; p < L11_outer[j + 1]; ++p) {
                    const node_index row = L11_inner[p];
                    const edge_index out = pos[row]++;
                    csr_col_idx_[out] = j;
                    csr_vals_[out]    = static_cast<sptrsv_value_t>(L11_vals[p]);
                }
            }
        }

        // No separate diagonal array: the solve reads L(i,i) inline from the factor
        // -- the LAST entry of CSR row i (forward: sum loop stops one short) and the
        // FIRST entry of CSC column j (back: sum loop starts one in) -- at the same
        // precision as the off-diagonals (fp32 under the flag). This is bit-identical
        // to the old fp64 diag_ (L11_vals is already fp32 under the flag, so diag_ only
        // ever held the fp32 value widened to double) and matches the GPU backend,
        // which has always read the diagonal inline.
        mark("csc_to_csr");

        // ── CSC of L11 (for back solve) ─────────────────────────
        // Parallel memcpy of the three arrays.
        csc_col_ptr_.resize(static_cast<size_t>(m_) + 1);
        csc_row_idx_.resize(nnz);
        csc_vals_.resize(nnz);
        #pragma omp parallel for schedule(static)
        for (node_index i = 0; i <= m_; ++i)
            csc_col_ptr_[i] = L11_outer[i];
        #pragma omp parallel for schedule(static)
        for (edge_index k = 0; k < nnz; ++k) {
            csc_row_idx_[k] = L11_inner[k];
            csc_vals_[k]    = static_cast<sptrsv_value_t>(L11_vals[k]);
        }
        mark("csc_copy");
        // Last read of L11 -- whichever of the input factor, the Laplacian-path
        // copy or the compacted (drop) copy the L11_* pointers aliased. The
        // level sets and counters below use only the SpTRSV's own arrays, so
        // release all three sources HERE rather than at return (nnz-sized;
        // swap-with-empty / reset, since `v = {}` / clear() keep the capacity).
        if (consumed) consumed->release_values();
        std::vector<edge_index>().swap(L11_outer_local);
        std::vector<node_index>().swap(L11_inner_local);
        std::vector<sptrsv_value_t>().swap(L11_vals_local);
        std::vector<edge_index>().swap(drop_outer);
        drop_inner.reset();
        drop_vals.reset();
        L11_outer = nullptr; L11_inner = nullptr; L11_vals = nullptr;

        // ── Level sets ──────────────────────────────────────────
        // Round-as-level (DEFAULT when round boundaries are available;
        // APXCHOL_ROUND_LEVELS=0 forces the topological scan): assign each column
        // its elimination-round index as its level. This is correct: same-round IS
        // columns are mutually independent (no edge, no fill between them -> L=0),
        // so they share a level; every L-dependency points to an earlier round
        // (forward) / later round (back, reversed), never the same level. Trailing
        // residual-peel columns get their own sequential levels.
        //
        // Correctness hinges on round_bounds_ exactly matching the factor's
        // column->round mapping. It does, now that the elimination loop no longer
        // records the bailed (min_is_fraction) round -- that round eliminates no
        // columns, and counting it used to desync the boundaries and mislabel the
        // sequential peel tail as one independent level (intermittent solve
        // failures on dense IPM factors). See factorization_impl.h.
        //
        // SOLVE optimization (-6..-18% across grids/SS/IPM): the topological scan
        // minimizes level COUNT but leaves levels imbalanced; the round boundaries
        // distribute columns far more evenly -> better per-level load balance.
        static const bool kRoundLevelsDisabled = [] {
            const char* e = std::getenv("APXCHOL_ROUND_LEVELS");
            return e && std::atoi(e) == 0;
        }();
        const bool use_rounds = !kRoundLevelsDisabled
            && !round_bounds_.empty()
            && round_bounds_.back() <= static_cast<node_index>(m_);
        if (use_rounds) {
            std::vector<int> rof(m_);
            const node_index R = static_cast<node_index>(round_bounds_.size()) - 1;
            const node_index last = round_bounds_.back();
            node_index r = 0;
            for (node_index c = 0; c < m_; ++c) {
                if (c >= last) { rof[c] = static_cast<int>(R + (c - last)); continue; }
                while (r + 1 < round_bounds_.size() && c >= round_bounds_[r + 1]) ++r;
                rof[c] = static_cast<int>(r);
            }
            int maxd = 0;
            for (node_index c = 0; c < m_; ++c) maxd = std::max(maxd, rof[c]);
            // Pre-size every level exactly (one O(m) histogram pass) so the
            // fill below never reallocates; contents and order are unchanged.
            // The fill itself stays serial: it is a stable multi-bucket
            // append whose parallelization would need per-(thread, level)
            // exact offsets — another full two-pass — for an O(m) loop that
            // is small next to the transpose above.
            std::vector<node_index> lvl_cnt(static_cast<size_t>(maxd) + 1, 0);
            for (node_index c = 0; c < m_; ++c) lvl_cnt[rof[c]]++;
            fwd_levels_.assign(maxd + 1, {});
            bck_levels_.assign(maxd + 1, {});
            for (int d = 0; d <= maxd; ++d) {
                fwd_levels_[d].reserve(lvl_cnt[d]);
                bck_levels_[maxd - d].reserve(lvl_cnt[d]);
            }
            for (node_index c = 0; c < m_; ++c) {
                fwd_levels_[rof[c]].push_back(c);          // L solve: round order
                bck_levels_[maxd - rof[c]].push_back(c);   // L^T solve: reversed
            }
            mark("fwd_levels"); mark("bck_levels");
        } else {
        // ── Forward level sets (for L solve) ────────────────────
        // Row i depends on columns j < i where L(i,j) ≠ 0.
        // The depth recurrence is inherently sequential (depth[i] reads the
        // depths of earlier rows along the DAG's longest path), so it stays
        // serial; the fill is pre-sized from a depth histogram so its
        // push_backs never reallocate (contents and order unchanged).
        {
            std::vector<int> depth(m_, 0);
            int max_depth = 0;
            for (node_index i = 0; i < m_; ++i) {
                int d = 0;
                for (edge_index p = csr_row_ptr_[i]; p < csr_row_ptr_[i + 1] - 1; ++p)
                    d = std::max(d, depth[csr_col_idx_[p]] + 1);
                depth[i] = d;
                max_depth = std::max(max_depth, d);
            }
            std::vector<node_index> lvl_cnt(static_cast<size_t>(max_depth) + 1, 0);
            for (node_index i = 0; i < m_; ++i) lvl_cnt[depth[i]]++;
            fwd_levels_.resize(max_depth + 1);
            for (int d = 0; d <= max_depth; ++d)
                fwd_levels_[d].reserve(lvl_cnt[d]);
            for (node_index i = 0; i < m_; ++i)
                fwd_levels_[depth[i]].push_back(i);
        }
        mark("fwd_levels");

        // ── Backward level sets (for L^T solve) ─────────────────
        // Column j: z[j] = (y[j] - Σ L(k,j)*z[k]) / L(j,j)  for k > j.
        // So column j depends on z[k] for k > j where L(k,j) ≠ 0.
        // Same structure as the forward scan: sequential depth recurrence
        // (reverse direction), histogram-pre-sized fill.
        {
            std::vector<int> depth(m_, 0);
            int max_depth = 0;
            for (node_index j = m_; j-- > 0; ) {   // reverse: m_-1 .. 0 (unsigned-safe)
                int d = 0;
                for (edge_index p = csc_col_ptr_[j]; p < csc_col_ptr_[j + 1]; ++p) {
                    node_index k = csc_row_idx_[p];
                    if (k > j)
                        d = std::max(d, depth[k] + 1);
                }
                depth[j] = d;
                max_depth = std::max(max_depth, d);
            }
            std::vector<node_index> lvl_cnt(static_cast<size_t>(max_depth) + 1, 0);
            for (node_index j = 0; j < m_; ++j) lvl_cnt[depth[j]]++;
            bck_levels_.resize(max_depth + 1);
            for (int d = 0; d <= max_depth; ++d)
                bck_levels_[d].reserve(lvl_cnt[d]);
            for (node_index j = 0; j < m_; ++j)
                bck_levels_[depth[j]].push_back(j);
        }
        mark("bck_levels");
        }
        if (const char* e = std::getenv("APXCHOL_LEVEL_DUMP"); e && *e) {
            long long fwd_max = 0, bck_max = 0;
            for (auto& l : fwd_levels_) fwd_max = std::max<long long>(fwd_max, (long long)l.size());
            for (auto& l : bck_levels_) bck_max = std::max<long long>(bck_max, (long long)l.size());
            // Back-solve WORK concentration. avg/count of levels is a MISLEADING
            // sync-free signal: power-law factors have a tiny average level size
            // yet a few very fat early levels carry nearly all the work (so
            // level-set wins). The predictive signals are per-level off-diagonal
            // WORK (= SpTRSV flops): bck_work_top1_frac (share in the single
            // fattest level — high => fat head => level-set) and bck_work_in_tiny
            // (share in levels below the tiny-level threshold). A genuine
            // sync-free candidate needs HIGH tiny-work AND LOW top1 (work truly
            // spread across many small levels, no dominant head).
            constexpr long long kTinyLevelSize = 512;   // diagnostic bucket only
            long long total_w = 0, max_lvl_w = 0, tiny_w = 0;
            for (auto& lvl : bck_levels_) {
                long long w = 0;
                for (node_index j : lvl)
                    w += (long long)(csc_col_ptr_[j + 1] - csc_col_ptr_[j] - 1);
                total_w += w;
                max_lvl_w = std::max(max_lvl_w, w);
                if ((long long)lvl.size() < kTinyLevelSize) tiny_w += w;
            }
            const double top1 = total_w ? (double)max_lvl_w / total_w : 0.0;
            const double tiny = total_w ? (double)tiny_w / total_w : 0.0;
            std::fprintf(stderr,
                "[trsv-levels] m=%lld fwd_lvls=%zu (max=%lld) bck_lvls=%zu (max=%lld) "
                "bck_work_top1_frac=%.4f bck_work_in_tiny_frac=%.4f\n",
                (long long)m_, fwd_levels_.size(), fwd_max, bck_levels_.size(), bck_max,
                top1, tiny);
        }

        // ── Sync-free initial dependency counts (backward only) ──
        // Column j depends on (csc_col_ptr[j+1] - csc_col_ptr[j] - 1)
        // off-diagonal rows k > j.  Pre-computed once and copied into the
        // live counter buffer at the start of each sync-free back solve.
        // (Forward always uses the level-set scheduler — see forward_solve —
        // so no forward sync-free counters are built.)
        bck_unsolved_init_.resize(m_);
        #pragma omp parallel for schedule(static)
        for (node_index j = 0; j < m_; ++j)
            bck_unsolved_init_[j] =
                static_cast<int>(csc_col_ptr_[j + 1] - csc_col_ptr_[j] - 1);
        bck_unsolved_.resize(m_);

        ready_ = true;
    }

public:

    /// Forward solve: L * y = x.  Reads x[0..m-1], writes y[0..m-1].
    /// Always the level-set scheduler: empirically beats sync-free across
    /// all level counts we tested (BG/luby/BK with thousands of levels and
    /// rootset with ~65 levels), including the thin-level regime where the
    /// back solve prefers sync-free — so the forward sync-free variant was
    /// dropped.  (The back solve still chooses between the two: see
    /// transpose_solve.)
    void forward_solve(const double* x_in, double* y_out) const {
        forward_solve_levelset(x_in, y_out);
    }

    /// Back solve: L^T * z = y.  Default: level-set scheduler.  Paired A/B
    /// across the benchmark suite shows the sync-free back-solve never beats
    /// level-set on any factor the current elimination produces: power-law
    /// factors have a tiny *average* level size that hides a few very fat
    /// early levels carrying nearly all the work (which parallelize best
    /// under level-set), while sync-free pays atomic-spin + load-imbalance
    /// over the whole column set.  Sync-free stays available as an opt-in
    /// escape hatch (APXCHOL_BCK_SCHED=syncfree) for orderings that produce
    /// genuinely deep, uniformly-thin factors.
    void transpose_solve(const double* x_in, double* y_out) const {
        static const bool syncfree = []{
            const char* e = std::getenv("APXCHOL_BCK_SCHED");
            return e && std::string_view(e) == "syncfree";
        }();
        if (syncfree) transpose_solve_syncfree(x_in, y_out);
        else          transpose_solve_levelset(x_in, y_out);
    }

    // ── Level-set scheduler ──────────────────────────────────

    void forward_solve_levelset(const double* x_in, double* y_out) const {
        // No initializing copy of x_in into y_out: the input is folded
        // straight into the recurrence (row i reads x_in[i] in its update
        // below), removing a serial full-vector memory pass per solve.
        // Bit-identical to the old copy-then-update form: y_out[i] was only
        // ever read at row i's own update, where it still held x_in[i]
        // (levels partition 0..m-1, each index written exactly once); the
        // gathered terms read y_out entries written by earlier levels.
        // In-place calls (x_in == y_out) stay valid for the same reason.

        // ONE persistent OpenMP team across all levels.  Issuing a fresh
        // `#pragma omp parallel for` per level costs ~10µs per fork-join
        // at T=16, which dominates solve time when a factor has many
        // levels (e.g. ~240 levels × 50 PCG iters × 2 solves = 24k
        // fork-joins per solve on IPM iter40).  By spawning threads once
        // and iterating levels inside the parallel region with `omp for`
        // (implicit barrier between levels carries the level-dependency
        // guarantee that level k+1 reads y_out written by level k), we
        // collapse the fork-joins to one.  Tiny levels run on thread 0
        // via `omp single` (also has an implicit barrier).
        #pragma omp parallel
        {
            for (const auto& level : fwd_levels_) {
                const node_index level_sz = static_cast<node_index>(level.size());
                if (level_sz <= kSpTRSVOMPThreshold) {
                    #pragma omp single
                    {
                        for (node_index k = 0; k < level_sz; ++k) {
                            node_index i = level[k];
                            // Two-stage prefetch: pull metadata 8 ahead
                            // (cheap csr_row_ptr_[] load), pull actual nnz
                            // payload 4 ahead where most cache-miss stalls
                            // occur.
                            if (k + 8 < level_sz)
                                __builtin_prefetch(&csr_row_ptr_[level[k + 8]]);
                            if (k + 4 < level_sz) {
                                node_index ip = level[k + 4];
                                __builtin_prefetch(&csr_col_idx_[csr_row_ptr_[ip]]);
                                __builtin_prefetch(&csr_vals_[csr_row_ptr_[ip]]);
                            }
                            // 4-way accumulator split (see solve.cpp:31 for rationale).
                            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                            const edge_index row_start = csr_row_ptr_[i];
                            const edge_index row_end   = csr_row_ptr_[i + 1] - 1;
                            edge_index p = row_start;
                            for (; p + 4 <= row_end; p += 4) {
                                s0 += csr_vals_[p + 0] * y_out[csr_col_idx_[p + 0]];
                                s1 += csr_vals_[p + 1] * y_out[csr_col_idx_[p + 1]];
                                s2 += csr_vals_[p + 2] * y_out[csr_col_idx_[p + 2]];
                                s3 += csr_vals_[p + 3] * y_out[csr_col_idx_[p + 3]];
                            }
                            double sum = (s0 + s1) + (s2 + s3);
                            for (; p < row_end; ++p)
                                sum += csr_vals_[p] * y_out[csr_col_idx_[p]];
                            y_out[i] = (x_in[i] - sum) / csr_vals_[csr_row_ptr_[i + 1] - 1];
                        }
                    } // implicit barrier
                } else {
                    #pragma omp for schedule(static)
                    for (node_index k = 0; k < level_sz; ++k) {
                        node_index i = level[k];
                        if (k + 8 < level_sz)
                            __builtin_prefetch(&csr_row_ptr_[level[k + 8]]);
                        if (k + 4 < level_sz) {
                            node_index ip = level[k + 4];
                            __builtin_prefetch(&csr_col_idx_[csr_row_ptr_[ip]]);
                            __builtin_prefetch(&csr_vals_[csr_row_ptr_[ip]]);
                        }
                        double sum = 0.0;
                        for (edge_index p = csr_row_ptr_[i]; p < csr_row_ptr_[i + 1] - 1; ++p)
                            sum += csr_vals_[p] * y_out[csr_col_idx_[p]];
                        y_out[i] = (x_in[i] - sum) / csr_vals_[csr_row_ptr_[i + 1] - 1];
                    } // implicit barrier on omp for
                }
            }
        }
    }

    void transpose_solve_levelset(const double* x_in, double* y_out) const {
        // No initializing copy — x_in is folded into the recurrence exactly
        // as in forward_solve_levelset (this solve is also a pure gather:
        // column j reads x_in[j] once at its own update, and the sum gathers
        // y_out entries of later columns written by earlier backward levels).
        // Bit-identical to the old copy-then-update form; in-place calls
        // (x_in == y_out) stay valid.

        // ONE persistent OpenMP team across all backward levels.  See
        // forward_solve_levelset for the rationale: collapses per-level
        // fork-joins to one outer parallel region with implicit barriers
        // between `omp for` constructs.  Tiny-level path runs on thread 0
        // via `omp single` (also has an implicit barrier).
        #pragma omp parallel
        {
            // Process backward levels from depth 0 (no dependencies) upward.
            for (const auto& level : bck_levels_) {
                const node_index level_sz = static_cast<node_index>(level.size());
                if (level_sz <= kSpTRSVOMPThreshold) {
                    #pragma omp single
                    {
                        for (node_index k = 0; k < level_sz; ++k) {
                            node_index j = level[k];
                            // Two-stage prefetch: csc_col_ptr_ 8 ahead,
                            // csc_row_idx_/vals 4 ahead (where cache-miss
                            // stalls hit).
                            if (k + 8 < level_sz)
                                __builtin_prefetch(&csc_col_ptr_[level[k + 8]]);
                            if (k + 4 < level_sz) {
                                node_index jp = level[k + 4];
                                __builtin_prefetch(&csc_row_idx_[csc_col_ptr_[jp]]);
                                __builtin_prefetch(&csc_vals_[csc_col_ptr_[jp]]);
                            }
                            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                            const edge_index col_start = csc_col_ptr_[j] + 1;
                            const edge_index col_end   = csc_col_ptr_[j + 1];
                            edge_index p = col_start;
                            for (; p + 4 <= col_end; p += 4) {
                                s0 += csc_vals_[p + 0] * y_out[csc_row_idx_[p + 0]];
                                s1 += csc_vals_[p + 1] * y_out[csc_row_idx_[p + 1]];
                                s2 += csc_vals_[p + 2] * y_out[csc_row_idx_[p + 2]];
                                s3 += csc_vals_[p + 3] * y_out[csc_row_idx_[p + 3]];
                            }
                            double sum = (s0 + s1) + (s2 + s3);
                            for (; p < col_end; ++p)
                                sum += csc_vals_[p] * y_out[csc_row_idx_[p]];
                            y_out[j] = (x_in[j] - sum) / csc_vals_[csc_col_ptr_[j]];
                        }
                    } // implicit barrier
                } else {
                    #pragma omp for schedule(static)
                    for (node_index k = 0; k < level_sz; ++k) {
                        node_index j = level[k];
                        if (k + 8 < level_sz)
                            __builtin_prefetch(&csc_col_ptr_[level[k + 8]]);
                        if (k + 4 < level_sz) {
                            node_index jp = level[k + 4];
                            __builtin_prefetch(&csc_row_idx_[csc_col_ptr_[jp]]);
                            __builtin_prefetch(&csc_vals_[csc_col_ptr_[jp]]);
                        }
                        double sum = 0.0;
                        for (edge_index p = csc_col_ptr_[j] + 1; p < csc_col_ptr_[j + 1]; ++p)
                            sum += csc_vals_[p] * y_out[csc_row_idx_[p]];
                        y_out[j] = (x_in[j] - sum) / csc_vals_[csc_col_ptr_[j]];
                    } // implicit barrier on omp for
                }
            }
        }
    }

    // ── Synchronization-free scheduler (back solve only) ─────
    //
    // Per-column dependency counters (bck_unsolved_) start at the
    // off-diagonal nonzero count and decrement as producers finish.
    // Threads pick columns dynamically; each column busy-waits on its
    // counter to reach zero, computes, then publishes the result by
    // atomically decrementing each dependent's counter.  Memory
    // ordering: release on the publish, acquire on the spin-wait read,
    // which establishes happens-before between the producer's write to
    // y_out[j] and the consumer's load of y_out[j].
    // (The forward solve always uses the level-set scheduler, which wins
    // universally there, so it has no sync-free variant.)

    void transpose_solve_syncfree(const double* x_in, double* y_out) const {
        std::copy(x_in, x_in + m_, y_out);

        #pragma omp parallel for schedule(static)
        for (node_index j = 0; j < m_; ++j)
            bck_unsolved_[j] = bck_unsolved_init_[j];

        const node_index m = m_;
        int* unsolved = bck_unsolved_.data();

        // Process columns in reverse: column j depends on z[k] for k > j,
        // so larger indices become ready first.  Reverse iteration in
        // the dynamic-for keeps cache locality similar to the forward case
        // and lets the index-(m-1) row (no dependencies) start immediately.
        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic, 64) nowait
            for (node_index j_r = 0; j_r < m; ++j_r) {
                node_index j = m - 1 - j_r;
                while (__atomic_load_n(&unsolved[j], __ATOMIC_ACQUIRE) != 0) {
                    #if defined(__x86_64__) || defined(__i386__)
                    __builtin_ia32_pause();
                    #endif
                }

                double sum = 0.0;
                for (edge_index p = csc_col_ptr_[j] + 1; p < csc_col_ptr_[j + 1]; ++p)
                    sum += csc_vals_[p] * y_out[csc_row_idx_[p]];
                y_out[j] = (y_out[j] - sum) / csc_vals_[csc_col_ptr_[j]];

                // Notify dependents: columns i < j with L(j, i) ≠ 0.
                // CSR row j lists exactly those, with diagonal last.
                for (edge_index p = csr_row_ptr_[j]; p < csr_row_ptr_[j + 1] - 1; ++p) {
                    node_index i = csr_col_idx_[p];
                    __atomic_sub_fetch(&unsolved[i], 1, __ATOMIC_RELEASE);
                }
            }
        }
    }

    int num_fwd_levels() const { return static_cast<int>(fwd_levels_.size()); }
    int num_bck_levels() const { return static_cast<int>(bck_levels_.size()); }

    // Diagnostics: per-level row/col counts and per-level off-diagonal nnz.
    void level_stats(bool fwd, std::vector<int>& sizes, std::vector<long long>& work) const {
        const auto& levels = fwd ? fwd_levels_ : bck_levels_;
        sizes.clear(); work.clear();
        sizes.reserve(levels.size()); work.reserve(levels.size());
        for (const auto& lvl : levels) {
            long long w = 0;
            for (node_index v : lvl) {
                if (fwd)
                    w += (csr_row_ptr_[v + 1] - csr_row_ptr_[v] - 1);
                else
                    w += (csc_col_ptr_[v + 1] - csc_col_ptr_[v] - 1);
            }
            sizes.push_back(static_cast<int>(lvl.size()));
            work.push_back(w);
        }
    }
    bool ready() const { return ready_; }

    // Read-only views of the CSR built by setup's CSC→CSR transpose and of the
    // CSC copy. Used by the SpTRSVTranspose unit tests to byte-compare the
    // parallel transpose against a serial reference and by the SpTRSVDrop
    // tests to check what the compacting drop stored (and available for
    // diagnostics).
    const auto& csr_row_ptr() const { return csr_row_ptr_; }
    const auto& csr_col_idx() const { return csr_col_idx_; }
    const auto& csr_vals()    const { return csr_vals_; }
    const auto& csc_col_ptr() const { return csc_col_ptr_; }
    const auto& csc_row_idx() const { return csc_row_idx_; }
    const auto& csc_vals()    const { return csc_vals_; }

    /// Bytes held by this object's arrays (capacities, heap only): CSR + CSC +
    /// level sets + round bounds + sync-free counters. After setup() this is
    /// everything the SpTRSV keeps -- setup's transients (L11 copy, compacted
    /// copy, transpose bucket, scratch) are all released before it returns
    /// (guarded by tests/test_sptrsv_memory.cpp).
    std::size_t memory_bytes() const {
        std::size_t b = csr_row_ptr_.capacity() * sizeof(edge_index)
                      + csr_col_idx_.capacity() * sizeof(node_index)
                      + csr_vals_.capacity()    * sizeof(sptrsv_value_t)
                      + csc_col_ptr_.capacity() * sizeof(edge_index)
                      + csc_row_idx_.capacity() * sizeof(node_index)
                      + csc_vals_.capacity()    * sizeof(sptrsv_value_t)
                      + round_bounds_.capacity() * sizeof(node_index)
                      + bck_unsolved_init_.capacity() * sizeof(int)
                      + bck_unsolved_.capacity() * sizeof(int);
        for (const auto* lv : {&fwd_levels_, &bck_levels_}) {
            b += lv->capacity() * sizeof(std::vector<node_index>);
            for (const auto& l : *lv) b += l.capacity() * sizeof(node_index);
        }
        return b;
    }

private:
    node_index m_ = 0;
    bool ready_ = false;
    drop_statistics stats_;   // what the last setup()'s compacting drop did

    // SpTRSV's CSR/CSC arrays are read by the inner forward/back loops at
    // ~50 GB/s effective on T=16. With default 4 KB pages and ~32 MB per
    // array, the L2 dTLB (3072 entries × 4 KB = 12 MB coverage) gets blown
    // out on every level-set sweep — measured dTLB miss rate is 19% on
    // IPM iter40. big_alloc (mmap + MADV_HUGEPAGE + MADV_POPULATE_WRITE)
    // backs each array with 2 MB hugepages, dropping dTLB pressure from
    // ~16384 pages/array to ~16. Expected -5-10% PCG solve win on
    // bandwidth-bound IPM workloads.
    template<typename T>
    using big_vec = std::vector<T, util::big_alloc<T>>;

    // CSR of L11 (forward solve). row_ptr is an offset array (edge_index);
    // col_idx holds column ids (node_index).
    big_vec<edge_index>     csr_row_ptr_;
    big_vec<node_index>     csr_col_idx_;
    big_vec<sptrsv_value_t> csr_vals_;   // fp32 under APXCHOL_SPTRSV_FP32

    // CSC of L11 (back solve). col_ptr is an offset array (edge_index);
    // row_idx holds row ids (node_index).
    big_vec<edge_index>     csc_col_ptr_;
    big_vec<node_index>     csc_row_idx_;
    big_vec<sptrsv_value_t> csc_vals_;   // fp32 under APXCHOL_SPTRSV_FP32

    // Level sets.
    std::vector<std::vector<node_index>> fwd_levels_;
    std::vector<std::vector<node_index>> bck_levels_;
    // Optional: cumulative IS-size per elimination round (column->round boundaries).
    // When set + APXCHOL_ROUND_LEVELS, levels are read off the rounds instead of
    // an O(nnz) topological depth scan (same structure -- same-round IS columns are
    // independent; peel columns get their own sequential level).
    std::vector<node_index> round_bounds_;
public:
    void set_round_bounds(std::vector<node_index> b) { round_bounds_ = std::move(b); }
private:

    // Sync-free back-solve dependency counters.
    // bck_unsolved_init_ holds the immutable initial counts (computed once in
    // setup); bck_unsolved_ is the live counter buffer reset at each solve.
    std::vector<int> bck_unsolved_init_;
    mutable std::vector<int> bck_unsolved_;
};

} // namespace apxchol
