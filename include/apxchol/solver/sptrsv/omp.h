#pragma once
#include "apxchol/types.h"
#include "apxchol/sparse_csc.h"
#include "apxchol/big_alloc.h"
#include "apxchol/solver/sptrsv/factor_drop.h"   // the compacting drop (shared with the CUDA backend)
#include "apxchol/solver/sptrsv/transpose.h"     // the CSC -> CSR transpose (shared with the CUDA backend)
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
#include <immintrin.h>   // fat-level SIMD kernels (16-bit storage): _mm256_cvtph_ps / _mm256_fmadd_pd
#endif

namespace apxchol {

// Minimum level-set size before engaging OpenMP for SpTRSV.
// Below this threshold, the thread-dispatch overhead exceeds the
// computational benefit of parallelizing a single level's rows/cols.
inline constexpr node_index kSpTRSVOMPThreshold = 1024;

// kFactorDropRelDefault (the compacting drop's default threshold, 1e-4, and
// the measurements behind it), factor_drop_rel_from_env(),
// factor_column_scale() and the drop itself
// (compact_factor_columns) live in factor_drop.h -- ONE host implementation
// shared with the CUDA backend; the "COMPACTING DROP" section below is the
// authoritative description of what it does and why.

/// OpenMP parallel sparse triangular solver: ONE level-set kernel for both
/// sweeps (Anderson & Saad 1989; Naumov 2011).
///
/// setup() pre-computes the topological levels of the dependency DAG (or reads
/// them off the elimination rounds, see round_bounds_); a solve is one
/// persistent OpenMP team walking the levels in order with a barrier between
/// levels. The forward solve (L y = x) walks the CSR of L11 -- row i, diagonal
/// slot LAST, levels fwd_levels_ -- and the back solve (L^T z = y) walks the
/// CSC -- column j, diagonal slot FIRST, levels bck_levels_ (already stored
/// deepest-dependency-first, so both walks are index-ascending). Both are pure
/// gathers of the same shape (per row / column: one dot product over the
/// off-diagonal slots against y_out, one subtract, one divide by the diagonal),
/// so they are ONE templated kernel, solve_levelset<Dir, V>, whose direction
/// policy (fwd_dir / bck_dir, below the kernel) supplies the arrays, the level
/// list, the off-diagonal slot range and diagonal slot of a row / column, and
/// the input transform (identity, or the fp16 storage's folded x * r_j^2 on
/// the back solve), and whose STORAGE type V is float or fp16_t.
/// forward_solve / transpose_solve are thin wrappers that branch on the
/// storage the last setup() chose -- ONCE per solve, never per row.
///
/// Levels of size <= kSpTRSVOMPThreshold ("thin") run on one thread inside
/// `omp single` (4-way scalar kernel dot_thin); larger ("fat") levels are an
/// `omp for schedule(static)` (plain scalar loop on fp32, SIMD dot_fat_simd
/// on fp16 storage); one barrier per level either way (num_barriers()). A
/// solve is a pure gather -- each row's arithmetic is independent of which
/// thread runs it and when -- so both sweeps return the SAME BYTES at every
/// thread count (tests/test_sptrsv_levelset.cpp).
///
/// Two schedule experiments were tried against this and REMOVED 2026-08-18
/// as negative results (neither beat the schedule above; both were
/// bit-identical to it): APXCHOL_SPTRSV_BALANCE=nnz (fat levels split into
/// per-thread contiguous row ranges of equal stored-entry counts, cached per
/// team size, instead of `omp for schedule(static)` by row count -- the
/// motivation was `schedule(dynamic, 64)`'s -20% SpTRSV on IPM at +23% on
/// grids) and APXCHOL_SPTRSV_AGGLOMERATE=<K> (runs of consecutive thin levels
/// as one barrier-free `omp single` superstep -- iter0040 has 60-94 thin
/// levels per direction in ONE contiguous run, grid_2000 54-61, so this
/// removed most of a solve's barriers: iter0040 ~37-42 instead of ~100-130).
/// Same verdict as the sync-free (Liu/Smelyanskiy/Chow 2016) back solve that
/// used to sit behind APXCHOL_BCK_SCHED=syncfree, removed before them: paired
/// A/B across the suite never showed it winning on any factor the
/// elimination produces (power-law factors have a tiny AVERAGE level size,
/// but a few very fat early levels carry nearly all the work), and the
/// point-to-point experiment showed the barrier COUNT is not what the wall
/// time is made of. APXCHOL_LEVEL_DUMP=1 still prints the thin-level counts
/// and thin-run length histogram per direction (the structure those
/// experiments targeted) next to the work-concentration signals.

// STORAGE (RUNTIME, per setup): the SpTRSV's CSR/CSC value arrays -- the two
// largest copies of the factor -- hold either fp32 (`float`, the default) or
// the fp16 per-column-scaled form (`fp16_t`, lowprec.h). Which one is a
// RUNTIME choice, read from the unified env APXCHOL_SPTRSV_FP16=0|1 -- the
// same variable the GPU backend reads, sptrsv_fp16_env_tristate() in
// lowprec.h being the one reader -- at every setup; unset means OFF on the
// CPU and ON on the GPU. It is NOT a build option (the CMake variable
// APXCHOL_SPTRSV_LOWPREC that used to select it was removed 2026-08-20): the
// CPU kernel is INSTANTIATED for both storage types and setup picks one, so a
// single binary can do either and the choice needs no rebuild.
//
// Narrowing to fp16 halves the value stream (the 8 B/nnz idx+val stream drops
// to 6 B/nnz) on a bandwidth-bound solve; compute is unchanged, since every
// read of a stored value in the kernels below goes through widen(): the
// arithmetic is ALWAYS done in double, whatever the storage width, and the
// outer PCG stays fp64. It is a preconditioner-QUALITY knob (PCG iteration
// count), never a residual-floor one.
//
// DISTRIBUTION GUARD: the fp16 storage is only compiled, and only offered,
// where the target has F16C (__F16C__ -- any x86 since Ivy Bridge / Zen; the
// -march=native default has it). Without F16C the fp16 -> fp32 widen is a
// libgcc __extendhfdf2 CALL in the SpTRSV inner loop (measured 3x slower
// solve), so a PORTABLE build (APXCHOL_NATIVE_ARCH=OFF, i.e. the PyPI wheels'
// baseline x86-64) compiles the fp32 storage ONLY; the env then falls back to
// fp32 with a one-shot stderr note. fp16_supported() is the compile-time
// predicate, and every fp16 instantiation sits behind `if constexpr` on it.
//
// DIAGONAL under fp16: the fp32 storage reads L(i,i) inline from the factor
// (last entry of CSR row i / first entry of CSC column j) at the same
// precision as the off-diagonals. The fp16 storage does NOT: a narrow
// diagonal was measured to be the dominant iteration-count damage (iter0040
// T=1, the removed all-bf16 variant: diag-only bf16 314 PCG iters,
// off-diag-only 185, both 348, fp32 65), so it keeps a separate fp32
// `diag_[m]`, filled at setup from the factor (factor_value_t == float, i.e.
// BEFORE any narrowing) -- the SCALED diagonal L_jj / s_j (one fp32 division,
// stored_diag()), see below -- and both solves divide by diag_[i]. The narrow
// diagonal slot stays in csr_vals16_/csc_vals16_ (written like every other
// entry through narrow_value, i.e. fp16(L_jj / s_j), which may even overflow
// fp16 to inf) but is never read: keeping it leaves the CSR/CSC layout, the
// transpose, and every "diagonal is entry X" invariant untouched.
// diag<Dir, V>() below is the single switch. (Reading the fp16 slot instead
// was a real, measured option -- env APXCHOL_FP16_DIAG -- and lost: iter0040
// 44 -> 50 PCG iterations, +14%; see "Retired knobs" in AGENTS.md.)
//
// PER-COLUMN SCALE under fp16 -- FOLDED INTO THE VECTORS:
// scale_[j] = s_j = max |L_ij| over the off-diagonals of column j (1.0f if
// there is none), fp32, computed at setup from the factor (column_scale()).
// What is stored is the COLUMN-SCALED factor L~ = L D^-1, D = diag(s_j):
// off-diagonals narrow(L_ij / s_j) and diag_[j] = fp32(L_jj / s_j). The
// kernels never multiply a scale back -- they run on L~ as stored, so every
// kernel path (forward / back x thin / fat) is ONE source for both storage
// types, the only per-type difference being the widen() overload:
//   forward:  L y = x  <=>  L~ (D y) = x.  forward_solve runs the plain forward
//             kernel on L~ and returns y' = D y (y'_j = s_j y_j), NOT y.
//   back:     L^T z = y  <=>  D L~^T z = y  <=>  L~^T z = D^-1 y = D^-2 y'.
//             transpose_solve takes y' (the forward's output, THE pair
//             contract) and runs the plain back kernel on L~^T with the input
//             read x_in[j] scaled once per column: z_j = (x_in[j] * r_j^2 -
//             sum_k h_kj z_k) / diag_[j], r_j = fp32(1 / s_j) (inv_scale_[j],
//             set at setup; r_j^2 is exact in double). One per-COLUMN load
//             and multiply, nothing per entry, no pass between the sweeps.
// The pair therefore applies z = L~^-T R L~^-1 x with R = diag(r_j^2) =
// D^-2 (1 + O(2^-23)): symmetric positive definite, and with exact r_j
// exactly (L_s L_s^T)^-1 for the STORED factor L_s = L~ D (effective diagonal
// fp32(L_jj / s_j) * s_j, effective off-diagonals widen(h_ij) * s_j) -- the
// unit tests state this pair contract (y' = D y, then z) against a serial
// double reference on L_s. On the fp32 storage D = I: forward_solve returns y
// and transpose_solve solves L^T z = x_in, the scaled-only pieces
// (inv_scale_, the x_in scaling) compile out of THAT instantiation, and its
// inner loops are instruction-identical to the pre-fold ones (objdump of the
// outlined `omp` bodies: same FP instruction stream -- thin: vcvtss2sd +
// vfmadd231sd x4, fat: vcvtss2sd / vmulsd / vaddsd, epilogue vsubsd /
// vcvtss2sd / vdivsd).
//
// DEGENERATE SCALES (fp16 storage; the same contract the GPU backend states
// in cuda.h): a column whose scale cannot be represented -- fp32 1/s_j
// overflows (s_j < ~3e-39) or the scaled diagonal fp32(L_jj)/s_j overflows
// (off-diagonals >= ~1e38x below the diagonal) -- FALLS BACK to s_j = 1 (the
// "no off-diagonals" convention). Its off-diagonals are below anything even
// an fp32 sweep could see next to that diagonal, and the drop/flush removes
// them. The fallback is applied BEFORE the drop so the threshold and the
// storage agree on the scale, and it is counted
// (lowprec_stats().scale_fallback). setup() then VERIFIES that every diag_[j]
// is finite and nonzero and every r_j^2 finite, and THROWS otherwise: a
// factor the fp16 storage cannot represent must fail loudly, never dissolve
// into NaN.
//
// SIMD CONVERT (fat levels, the `omp for` paths; fp16 storage only; needs
// AVX2 + F16C + FMA, the -march=native default on any x86 since Haswell /
// Zen): 8 stored values per vector convert (widen8(): F16C _mm256_cvtph_ps --
// overloads for float / double exist so flipping the fp32 kernel over is one
// constant, simd_dot_v, but it is deliberately NOT flipped: the fp32 kernel
// stays as measured) -> two 4-double lanes, through an 8-double stack buffer
// (which the compiler turns into register lane extracts) into a 4-way scalar
// FMA chain over scalar y gathers. A 4-wide step and a scalar tail finish the
// row / column. Thin levels (the `omp single` paths) keep the 4-way scalar
// kernel; the fp32 fat-level loop is the plain scalar loop it always was.
// Different summation orders (2 lanes vs 4-way vs 1): same accuracy, not
// bit-identical. (The vector-gather flavour -- _mm256_i32gather_pd feeding
// _mm256_fmadd_pd instead of the stack buffer -- was an env A/B,
// APXCHOL_FP16_GATHER=simd, and LOST: vgatherdpd's latency sits on the
// critical path of the short grid rows, ~10-15% slower solve on grid_2000
// T=1, equal on iter0040. See "Retired knobs" in AGENTS.md.)
//
// ROUNDING: RNE (the bf16 variants' opt-in stochastic rounding went with
// them; a Laplacian factor's systematically signed RNE errors are what the
// column-sum compensation below absorbs instead).
//
// FP16 SUBNORMALS: a stored fp16 subnormal (|L_ij / s_j| in [2^-25, 2^-14))
// carries between 1 and 10 significant bits and, per the drop measurement
// below, that magnitude range is dead weight for the preconditioner.
// narrow_value therefore ALWAYS flushes fp16 subnormals to (signed) zero at
// storage time. (Keeping them was an env A/B, APXCHOL_FP16_KEEP_SUBNORMAL=1,
// and bought nothing: iter0040 64 iterations either way -- see "Retired
// knobs".) The flushed count below includes them.
//
// COMPACTING DROP (both storages; APXCHOL_FACTOR_DROP=<rel>, default
// kFactorDropRelDefault = 1e-4, read at every setup; <= 0 = off): setup()
// REMOVES -- not zeroes -- factor off-diagonals BEFORE the CSR/CSC are built:
// entry (i, j), i != j, is KEPT iff
//   |L_ij| >= rel * s_j   (s_j = column j's max |off-diagonal|, column_scale(),
//                          computed from the factor BEFORE the drop)
//   AND the storage format does not map it to zero anyway (format_flushes():
//   an exact zero on the fp32 storage; on fp16 also everything fp16 flushes --
//   |L_ij / s_j| < 2^-14 with the subnormal flush above -- since a stored zero
//   is still a stored entry; at the default rel = 1e-4 > 2^-14 this second
//   clause adds nothing).
// keep_offdiag() is the pure predicate. The diagonal is always kept, so every
// column keeps its first entry (the diagonal-first CSC / diagonal-last CSR
// invariants the solves rely on hold) and s_j is unchanged by the drop (the
// column max itself is never below rel * s_j for rel <= 1). Exact zeros
// always go (|0| < rel * s_j for any rel > 0). The drop is O(nnz) parallel
// work with no atomics -- per-column kept count -> serial prefix over m+1 ->
// parallel compacted copy into uninitialized buffers -- and the compacted
// arrays REPLACE L11_{outer,inner,vals} / nnz for everything downstream: the
// CSR transpose, the CSC copy and the level sets (the topological scan reads
// the compacted CSR/CSC; round-as-level bounds are per column and unaffected)
// are all drop-agnostic. Result: nnz(L stored) -- the
// CSR/CSC bytes and the per-sweep work -- shrink. The drop happens at the
// FACTOR's precision (factor_value_t), before narrow_value(): dropped entries
// never reach the storage format. Deterministic: the kept set is a per-entry
// predicate on the factor, the output order is the input order, and the
// compensation below sums each column in a fixed order (the factor itself is
// deterministic at T=1; at T>1 only fp merge-order ulps differ, as before).
// When nothing is below the threshold (grids) the arrays are left untouched
// -- no copy is made.
//
// COLUMN-SUM COMPENSATION (UNCONDITIONAL; the APXCHOL_FACTOR_DROP_COMPENSATE=0
// A/B switch was removed 2026-08-20 -- the numbers below are why): every
// column of the factor of
// a Laplacian sums to zero -- L(j,j) = sqrt(deg_j), L(i,j) = -w_ij/sqrt(deg_j)
// -- which is exactly why L L^T again has zero row sums (a Laplacian in the
// generalized sense: A + a zero-row-sum sampling error) and why grounding the
// last vertex costs nothing: (L L^T)_11 = L11 L11^T is the reduced form of
// that approximation, with the same total grounding mass as the true A_11.
// Plain removal of an entry L(i,j) breaks the column sum: to first order it
// deletes the tiny edge (i,j) from the approximate Laplacian AND leaves its
// weight behind as a self-loop at both endpoints -- in the grounded
// (Laplacian, m = n-1) view, an extra edge to ground. Modes that are only
// weakly grounded in A_11 (regions far from the hub in conductance) then see
// a preconditioner that grounds them much more strongly, and PCG pays:
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
// instead of an unbounded relative change of the grounding. Measured on
// iter0040 (main): 45 -> 45 iterations at rel = 1e-4, 1e-3 and 3e-3 (52 / 65
// / 67% of the off-diagonals dropped). This is the same idea as MILU's
// row-sum preservation. Measurement behind the default: see
// kFactorDropRelDefault. Under fp16 the compensation runs on the un-scaled
// fp32 factor column and only THEN is the column narrowed (L_ij / s_j); s_j
// is the pre-drop column max, which the drop never removes.
//
// COLUMN-SUM COMPENSATION OF THE ROUNDING (fp16 storage; UNCONDITIONAL): the
// per-entry storage rounding delta_ij of a narrow off-diagonal is exactly the
// kind of perturbation the paragraph above is about -- it breaks the zero
// column sums of a Laplacian factor, i.e. it grounds every vertex a little
// (~2^-11 relative under fp16) at both endpoints of every edge -- and on the
// Laplacian (m = n-1, center-k) path PCG pays for it exactly as it pays for
// plain removal: MEASURED iter0040 (T=1, tol 1e-8, bg+tree[vec_pool], drop
// on) fp32 45 -> fp16 64 WITHOUT it (67 with the drop off), while on the SDDM
// path (APXCHOL_GROUND=reg) both take 48. Since the diagonal lives in a
// separate fp32 diag_[], the residual is absorbed there: diag_[j] =
// fp32(x_jj + sum_i (x_ij - widen(h_ij))) with x = the value store() narrows
// (L_ij / s_j), computed in the CSC pass -- the stored column then sums to
// what the fp32 column sums to (up to fp32), and the diagonal moves by at
// most 2^-11 relative (|sum_i L_ij| = L_jj for our factors). The GPU backend
// applies the same compensation, always (cuda_host::narrow_fp16_scaled).
//
// STATISTICS (lowprec_stats() == drop_stats(), printed under APXCHOL_VERBOSE:
// one "sptrsv storage" line per setup, plus a "factor drop" line when the drop
// is on): the threshold in effect (rel; 0 = off) and whether the compensation
// the factor's nnz before / after the drop
// (nnz_factor, nnz_stored -- the latter is what the CSR and the CSC each
// hold), how many off-diagonals the drop removed and why (dropped =
// dropped_threshold + dropped_flush), and over the STORED off-diagonals: how
// many stored values flushed to zero (v != 0, stored == 0), how many are
// subnormal in the storage format, plus the SUBNORMAL CENSUS
// factor_subnormal = number of factor entries (diagonal included,
// factor_value_t = fp32) that are fp32 subnormals -- on the fp32 storage
// these ARE the stored values, so this is exactly "how many stored fp32
// factor values are subnormal". Measured 0 of 4.6M (iter0040) and 0 of 21.9M
// (grid_2000) -- which is why the MXCSR FTZ+DAZ knob that used to ask the
// same question at PCG entry is gone (see "Retired knobs" in AGENTS.md).

class omp_sptrsv {
public:
    omp_sptrsv() = default;

    // ── Storage selection (runtime, per setup) ───────────────────────────
    /// Whether THIS BUILD can do the fp16 factor storage at all: the target
    /// must have F16C (the one-instruction fp16 -> fp32 widen). A portable
    /// baseline-x86-64 build (the distributed wheels) compiles the fp32
    /// storage only -- see the file header, "DISTRIBUTION GUARD".
    static constexpr bool fp16_supported() {
#if defined(__F16C__)
        return true;
#else
        return false;
#endif
    }
    /// THE storage choice of the next setup(): the unified env
    /// APXCHOL_SPTRSV_FP16=0|1 (lowprec.h; the GPU backend reads the same
    /// variable), unset = OFF on the CPU. Asking for fp16 on a build without
    /// F16C falls back to fp32 with a one-shot stderr note.
    static bool fp16_from_env() {
        const bool want = sptrsv_fp16_env_tristate() == 1;
        if (want && !fp16_supported()) {
            static const bool warned = [] {
                std::fprintf(stderr,
                    "[apxchol] APXCHOL_SPTRSV_FP16=1 ignored by the CPU SpTRSV: this build has no F16C"
                    " (compiled for baseline x86-64 / without -march=native), where the fp16 -> fp32 widen"
                    " becomes a libgcc call in the inner loop (measured 3x slower solve); using fp32 storage\n");
                return true;
            }();
            (void)warned;
            return false;
        }
        return want;
    }
    /// Which storage the LAST setup() chose.
    bool fp16() const { return fp16_; }
    /// Width / name of the stored off-diagonal values after the last setup().
    std::size_t value_bytes() const { return fp16_ ? sizeof(fp16_t) : sizeof(float); }
    const char* value_name() const { return fp16_ ? "fp16 (per-column scaled)" : "float (fp32)"; }
    /// Whether the fat-level kernels of the fp16 storage are the SIMD ones on
    /// this target (AVX2 + F16C + FMA).
    static constexpr bool simd_fp16_kernel() { return simd_dot_v<fp16_t>; }

    // THE storage contract (public so the unit tests can state it): what
    // setup() stores for the factor entry with value v, in a column whose
    // per-column scale is s (fp16: s_j = max |off-diagonal| of column j, 1.0f
    // if none; the fp32 storage ignores s -- pass 1.0f). fp16 subnormals are
    // flushed to signed zero (see the file header). A PURE function of its
    // arguments: the CSR transpose and the CSC copy both call it, so the two
    // stored copies of every entry agree bit-for-bit (a no-op cast on the fp32
    // storage, exactly the static_cast it always did). The compacting drop
    // (APXCHOL_FACTOR_DROP) happens BEFORE this: dropped entries never reach it.
    template <class V = sptrsv_value_t>
    static V narrow_value(factor_value_t v, float s) {
        if constexpr (std::is_same_v<V, fp16_t>) {
            const float x = static_cast<float>(v) / s;   // |x| <= 1 for the off-diagonals as factorized (s is
                                                         // their max; the drop's compensation may lift kept
                                                         // entries a little above -- fp16 has range to spare)
            const fp16_t h(x);                           // RNE, subnormals / flush per IEEE
            if (fp16_t::is_subnormal(h.bits))
                return fp16_t::from_bits(static_cast<std::uint16_t>(h.bits & 0x8000u));   // signed zero
            return h;
        } else {
            (void)s;
            return static_cast<V>(v);
        }
    }

    // True iff the storage format V maps the off-diagonal v (in a column with
    // scale s) to zero: an exact zero on the fp32 storage (a nonzero factor
    // entry never rounds to zero there); on fp16 everything fp16 flushes
    // (|v / s| < 2^-25 under RNE) plus the fp16 subnormal range (< 2^-14 after
    // rounding), which narrow_value flushes. Pure.
    template <class V = sptrsv_value_t>
    static bool format_flushes(factor_value_t v, float s) {
        if constexpr (std::is_same_v<V, fp16_t>) {
            const fp16_t h(static_cast<float>(v) / s);
            return fp16_t::is_zero(h.bits) || fp16_t::is_subnormal(h.bits);
        } else {
            (void)s;
            return v == 0;
        }
    }

    // THE compacting-drop predicate (public so the tests can state it): the
    // off-diagonal v of a column with scale s survives APXCHOL_FACTOR_DROP=rel
    // iff |v| >= rel * s and the storage format does not map it to zero. The
    // diagonal is never passed through this (always kept).
    template <class V = sptrsv_value_t>
    static bool keep_offdiag(factor_value_t v, float s, double rel) {
        return std::fabs(static_cast<double>(v)) >= rel * static_cast<double>(s) &&
               !format_flushes<V>(v, s);
    }

    // Per-column scale contract (public for the tests): the fp16 storage's
    // s_j, computed by setup() from the factor column [first, last) whose
    // FIRST entry is the diagonal: max |off-diagonal|, or 1.0f if the column
    // has no nonzero off-diagonal (so v / s_j is always defined). Also the
    // reference of the compacting drop's threshold (computed BEFORE the drop;
    // the drop never removes the column max, so it is the same after).
    // Computed in double, stored fp32 (the factor is fp32, so this is exact).
    static float column_scale(const factor_value_t* vals, edge_index first, edge_index last) {
        return factor_column_scale(vals, first, last);   // factor_drop.h (shared with the GPU backend)
    }

    // Diagonal contract (public for the tests): what the kernels DIVIDE by for
    // a column whose factor diagonal is L_jj and whose scale is s (1.0f on the
    // fp32 storage). The fp16 storage keeps the scaled diagonal
    // fp32(L_jj / s) -- one fp32 division -- in diag_[j] (see the file header,
    // "FOLDED INTO THE VECTORS"), plus the column's rounding residual; the
    // fp32 storage reads the factor value inline (== L_jj).
    template <class V = sptrsv_value_t>
    static double stored_diag(factor_value_t L_jj, float s) {
        if constexpr (std::is_same_v<V, fp16_t>) {
            return static_cast<double>(static_cast<float>(L_jj) / s);
        } else {
            (void)s;
            return widen(static_cast<V>(L_jj));   // inline read: factor_value_t == V here
        }
    }
    // Back-input scale contract (public for the tests): transpose_solve reads
    // x_in[j] * r_j^2 with r_j = inv_scale(s_j) (fp32 reciprocal; r_j^2 exact
    // in double); 1.0 on the fp32 storage.
    template <class V = sptrsv_value_t>
    static double inv_scale(float s) {
        if constexpr (std::is_same_v<V, fp16_t>) {
            return static_cast<double>(1.0f / s);
        } else {
            (void)s;
            return 1.0;
        }
    }

    // APXCHOL_FACTOR_DROP=<rel> resolution, read at every setup(): unset (or
    // empty) -> kFactorDropRelDefault; set -> its value, where anything <= 0
    // (including "0" and unparsable text, which atof reads as 0) turns the
    // drop OFF.
    static double factor_drop_rel_from_env() { return apxchol::factor_drop_rel_from_env(); }

    // Statistics of the last setup() (see the file header): what the
    // compacting drop did to L11 = L.topLeftCorner(m, m) (rel / nnz_factor /
    // nnz_stored / dropped*) and what the storage format did to
    // what remained. offdiag / flushed / subnormal / factor_subnormal are over
    // the STORED factor (after the drop); the dropped_* counts are what the
    // drop removed; nnz_factor is L11's nnz before the drop and nnz_stored
    // after (== nnz_factor when the drop is off).
    struct lowprec_statistics {
        double        rel               = 0.0;  // drop threshold in effect (0 = drop off)
        std::uint64_t offdiag           = 0;   // number of stored off-diagonal entries of L11
        std::uint64_t flushed           = 0;   // stored as zero although the factor value was nonzero
        std::uint64_t subnormal         = 0;   // stored as a subnormal of the storage format
        std::uint64_t factor_subnormal  = 0;   // factor entries (fp32, diagonal incl.) that are fp32 subnormals
        std::uint64_t dropped           = 0;   // off-diagonals removed by APXCHOL_FACTOR_DROP (= the two below)
        std::uint64_t dropped_threshold = 0;   //   ... because |L_ij| < rel * s_j
        std::uint64_t dropped_flush     = 0;   //   ... because the storage format stores them as zero anyway
        std::uint64_t nnz_factor        = 0;   // nnz of L11 as factorized (before the drop)
        std::uint64_t nnz_stored        = 0;   // nnz the CSR (and the CSC) hold (after the drop)
        // fp16 storage only (0 on fp32): columns whose scale could not be
        // represented and fell back to s_j = 1 (see the file header,
        // "DEGENERATE SCALES"), and columns WITH an off-diagonal whose
        // L_jj < s_j (the diagonal-dominance sanity count; off-diagonal-free
        // columns have s_j = 1.0f).
        std::uint64_t scale_fallback    = 0;
        std::uint64_t diag_below_scale  = 0;
    };
    const lowprec_statistics& lowprec_stats() const { return stats_; }
    // The same record under the name the drop-only callers use:
    // rel / nnz_factor / nnz_stored / dropped are its drop half.
    const lowprec_statistics& drop_stats() const { return stats_; }
    // nnz held by the SpTRSV's CSR / CSC (each) after the last setup().
    std::uint64_t stored_nnz() const { return stats_.nnz_stored; }

    /// Analyze L11 = L.topLeftCorner(m, m): build CSR, CSC, and level sets.
    /// L is read only; the caller keeps it. The storage width is resolved from
    /// the env HERE, once (fp16_from_env()).
    void setup(const sparse_csc& L, node_index m) { setup_dispatch(L, m, nullptr); }

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
    void setup_consuming(sparse_csc& L, node_index m) { setup_dispatch(L, m, &L); }

private:
    void setup_dispatch(const sparse_csc& L, node_index m, sparse_csc* consumed) {
        if (fp16_from_env()) {
            // `if constexpr` so a build without F16C never instantiates the
            // fp16 setup / kernels at all (see the file header).
            if constexpr (fp16_supported()) { fp16_ = true; setup_impl<fp16_t>(L, m, consumed); return; }
        }
        fp16_ = false;
        setup_impl<float>(L, m, consumed);
    }

    // `consumed` != nullptr: `L` may be released as soon as it is dead (see
    // setup_consuming); it then always aliases &L. V is the storage type.
    template <class V>
    void setup_impl(const sparse_csc& L, node_index m, sparse_csc* consumed) {
        constexpr bool kScaled = std::is_same_v<V, fp16_t>;   // narrow_value / stored_diag read s_j
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
        std::vector<factor_value_t> L11_vals_local;
        const edge_index*     L11_outer;
        const node_index*     L11_inner;
        const factor_value_t* L11_vals;     // the FACTOR's width: fp32
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
            const factor_value_t* L_vals = L.valuePtr();
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

        // APXCHOL_FACTOR_DROP=<rel> (both storages; default kFactorDropRelDefault,
        // <= 0 = off). The column-sum compensation is unconditional.
        const double factor_drop_rel = factor_drop_rel_from_env();
        stats_ = lowprec_statistics{};
        stats_.rel        = factor_drop_rel;
        stats_.nnz_factor = static_cast<std::uint64_t>(nnz);
        // Per-column scale s_j (the fp16 storage's scale_ -- a member, the
        // storage divides by it -- and the compacting drop's threshold
        // reference; not needed otherwise). See column_scale() for the
        // contract. Computed from the factor BEFORE the drop. On the fp32
        // storage it is a drop-only transient (m floats), freed right after
        // the drop block: the storage there ignores the scale.
        std::vector<float> col_scale_local;
        float* col_scale = nullptr;
        bool need_scale = false;
        if constexpr (kScaled) {
            scale_.resize(m_);
            col_scale = scale_.data();
            need_scale = true;
        } else {
            if (factor_drop_rel > 0.0) col_scale_local.resize(m_);
            col_scale = col_scale_local.data();
            need_scale = factor_drop_rel > 0.0;
        }
        if (need_scale) {
            #pragma omp parallel for schedule(static)
            for (node_index j = 0; j < m_; ++j)
                col_scale[j] = column_scale(L11_vals, L11_outer[j], L11_outer[j + 1]);
            mark("col_scale");
        }
        if constexpr (kScaled) {
            // DEGENERATE SCALES (see the file header; the GPU backend states
            // the same contract in cuda.h): a column whose fp32 1/s_j or
            // fp32(L_jj)/s_j would overflow falls back to s_j = 1. BEFORE the
            // drop, so the threshold and the storage agree on the scale.
            std::uint64_t fallback = 0;
            #pragma omp parallel for schedule(static) reduction(+ : fallback)
            for (node_index j = 0; j < m_; ++j) {
                const float s = col_scale[j];
                const float d = static_cast<float>(L11_vals[L11_outer[j]]);
                if (!std::isfinite(1.0f / s) || !std::isfinite(d / s)) { col_scale[j] = 1.0f; ++fallback; }
            }
            stats_.scale_fallback = fallback;
            // r_j = fp32(1 / s_j): the back solve's per-column input scale (see
            // the file header, "FOLDED INTO THE VECTORS"; inv_scale() is the
            // contract).
            inv_scale_.resize(m_);
            #pragma omp parallel for schedule(static)
            for (node_index j = 0; j < m_; ++j)
                inv_scale_[j] = 1.0f / col_scale[j];
        }

        // ── Compacting drop (APXCHOL_FACTOR_DROP; see the file header) ──
        // O(nnz) parallel work, no atomics: per-column kept count -> serial
        // prefix over m_+1 -> parallel compacted copy (+ the column-sum
        // compensation, per column, in the same pass). The compacted arrays
        // REPLACE L11_{outer,inner,vals} / nnz for everything below (transpose,
        // CSC copy, level sets), so the rest of setup is drop-agnostic. Order
        // within a column is preserved (diagonal stays first). The buffers are
        // allocated uninitialized (every slot is written exactly once). If no
        // entry is below the threshold the original arrays stay in place (no
        // second copy of the factor for the exact-no-op case, e.g. grids). All
        // of it at the FACTOR's precision (factor_value_t): narrowing to the
        // storage type happens later, in store().
        std::vector<edge_index>           drop_outer;
        std::unique_ptr<node_index[]>     drop_inner;
        std::unique_ptr<factor_value_t[]> drop_vals;
        {
            factor_drop_stats ds;
            const bool compacted = compact_factor_columns<edge_index, node_index, factor_value_t>(
                m_, L11_outer, L11_inner, L11_vals, col_scale, factor_drop_rel,
                [=](factor_value_t v, float s) { return keep_offdiag<V>(v, s, factor_drop_rel); },
                drop_outer, drop_inner, drop_vals, ds);
            stats_.dropped_threshold = ds.dropped_threshold;
            stats_.dropped_flush     = ds.dropped_flush;
            stats_.dropped           = ds.dropped;
            stats_.nnz_stored        = ds.nnz_stored;
            if (compacted) {
                L11_outer = drop_outer.data();
                L11_inner = drop_inner.get();
                L11_vals  = drop_vals.get();
                nnz = static_cast<edge_index>(ds.nnz_stored);
                // The Laplacian path's L11 copy is dead now: free it before the
                // transpose allocates its bucket (peak memory, not speed).
                // (`v = {}` would only clear -- it keeps the capacity; swapping
                // with an empty vector actually returns the memory.)
                std::vector<node_index>().swap(L11_inner_local);
                std::vector<factor_value_t>().swap(L11_vals_local);
                std::vector<edge_index>().swap(L11_outer_local);
                // SDDM path: L11 aliased the input factor, which the compacted
                // copy has just replaced -- the input is dead now.
                if (consumed && m == L.rows())
                    consumed->release_values();
            }
            if (factor_drop_rel > 0.0) mark("factor_drop");
        }
        assert(stats_.nnz_stored == static_cast<std::uint64_t>(nnz));

        // store(v, j): the factor entry with value v in column j of the
        // (possibly compacted) L11 -> the SpTRSV's storage width, via
        // narrow_value() (see its contract above). Both stored copies of an
        // entry (CSR transpose below, CSC copy) go through this same pure
        // function.
        const auto store = [=](factor_value_t v, node_index j) -> V {
            const float s = kScaled ? col_scale[j] : 1.0f;
            return narrow_value<V>(v, s);
        };
        if constexpr (kScaled) {
            // fp32 diagonal, straight from the factor (factor_value_t == float;
            // NOT via the narrowing path): the scaled L_jj / s_j (stored_diag()
            // is the contract). L(j,j) is the FIRST entry of CSC column j --
            // the invariant the back solve has always relied on.
            diag_.resize(m_);
            #pragma omp parallel for schedule(static)
            for (node_index j = 0; j < m_; ++j) {
                assert(L11_inner[L11_outer[j]] == j && "factor column must start with its diagonal");
                diag_[j] = static_cast<float>(stored_diag<V>(L11_vals[L11_outer[j]], col_scale[j]));
            }
            mark("diag_fp32");
        }

        // ── CSC → CSR of L11 (for forward solve) ─────
        // THE shared transpose (transpose.h; the GPU backend's host prep runs
        // the same code): the blocked counting-sort parallel transpose --
        // O(nnz) total work, byte-identical to the serial column-order scatter
        // at ANY thread count (SpTRSVTranspose.* unit tests) -- for m above
        // kParTransposeMinRows (APXCHOL_PAR_TRANSPOSE=0 disables it), the
        // serial scatter below it. Every stored value goes through store().
        // The design, the memory transient (one nnz-sized bucket, freed on
        // return) and the rejected alternatives are documented in transpose.h.
        auto& csr_vals = vals_csr(std::type_identity<V>{});
        auto& csc_vals = vals_csc(std::type_identity<V>{});
        csr_row_ptr_.resize(static_cast<size_t>(m_) + 1);
        csr_col_idx_.resize(nnz);
        csr_vals.resize(nnz);
        transpose_csc_to_csr<edge_index, node_index, factor_value_t, V>(
            m_, L11_outer, L11_inner, L11_vals,
            csr_row_ptr_.data(), csr_col_idx_.data(), csr_vals.data(),
            store, use_parallel_transpose(static_cast<std::int64_t>(m_)));

        // fp32 storage: no separate diagonal array -- the solve reads L(i,i)
        // inline from the factor, the LAST entry of CSR row i (forward: sum loop
        // stops one short) and the FIRST entry of CSC column j (back: sum loop
        // starts one in), at the same precision as the off-diagonals (the read
        // widens like every other one). This matches the GPU backend, which has
        // always read the diagonal inline. The fp16 storage keeps the exact-fp32
        // diag_ filled above instead (see diag<Dir, V>); its narrow diagonal
        // slots in the CSR/CSC are written like every other entry but never read.
        mark("csc_to_csr");

        // ── CSC of L11 (for back solve) ─────────────────────────
        // Parallel copy of the three arrays (values through store()), column
        // by column so store() knows the entry's column; this pass also
        // gathers the off-diagonal storage statistics (each entry once) and,
        // under fp16, folds each column's storage-rounding residual into
        // diag_[j] (the file header, "COLUMN-SUM COMPENSATION OF THE
        // ROUNDING"; unconditional).
        csc_col_ptr_.resize(static_cast<size_t>(m_) + 1);
        csc_row_idx_.resize(nnz);
        csc_vals.resize(nnz);
        #pragma omp parallel for schedule(static)
        for (node_index i = 0; i <= m_; ++i)
            csc_col_ptr_[i] = L11_outer[i];
        {
            std::uint64_t n_off = 0, n_flush = 0, n_sub = 0, n_fsub = 0, n_dlt = 0;
            #pragma omp parallel for schedule(static) reduction(+ : n_off, n_flush, n_sub, n_fsub, n_dlt)
            for (node_index j = 0; j < m_; ++j) {
                double resid = 0.0;                                // sum over the off-diagonals of (x - widen(stored))
                for (edge_index k = L11_outer[j]; k < L11_outer[j + 1]; ++k) {
                    const node_index    i  = L11_inner[k];
                    const factor_value_t v = L11_vals[k];
                    const V              w = store(v, j);
                    csc_row_idx_[k] = i;
                    csc_vals[k]     = w;
                    if (is_stored_subnormal(v)) ++n_fsub;          // census: the FACTOR value (fp32), diagonal incl.
                    if constexpr (kScaled) {
                        if (i != j) {
                            const double x = static_cast<double>(v) / static_cast<double>(col_scale[j]);
                            resid += x - widen(w);
                        }
                    }
                    if (i == j) {                                  // diagonal slot: never read under fp16 ...
                        if constexpr (kScaled) {
                            // ... but count L_jj < s_j among columns that HAVE
                            // an off-diagonal (s_j is the placeholder 1.0f
                            // otherwise): the diagonal-dominance sanity signal.
                            if (L11_outer[j + 1] - L11_outer[j] > 1 &&
                                static_cast<double>(v) < static_cast<double>(col_scale[j])) ++n_dlt;
                        }
                        continue;
                    }
                    ++n_off;
                    if (v != 0 && widen(w) == 0.0) {
                        ++n_flush;                                 // zeroed by the storage format
                    } else if (is_stored_subnormal(w)) {
                        ++n_sub;
                    }
                }
                if constexpr (kScaled)
                    diag_[j] = static_cast<float>(static_cast<double>(diag_[j]) + resid);
                else
                    (void)resid;
            }
            stats_.offdiag = n_off; stats_.flushed = n_flush;
            stats_.subnormal = n_sub; stats_.factor_subnormal = n_fsub;
            stats_.diag_below_scale = n_dlt;
            if constexpr (kScaled) {
                // FAIL LOUDLY on a factor the fp16 storage cannot represent
                // (unreachable after the scale fallback above -- a guard, not a
                // policy): a non-finite / zero diagonal or a non-finite r_j^2
                // would dissolve the solve into NaN with no error anywhere.
                // Same contract as cuda_sptrsv::setup.
                for (node_index j = 0; j < m_; ++j) {
                    const double r = static_cast<double>(inv_scale_[j]);
                    if (!(std::isfinite(diag_[j]) && diag_[j] != 0.0f && std::isfinite(r * r)))
                        throw std::runtime_error(
                            "apxchol omp_sptrsv: fp16 factor storage cannot represent column " + std::to_string(j) +
                            " (diag=" + std::to_string(diag_[j]) + ", inv_scale^2=" + std::to_string(r * r) +
                            "); set APXCHOL_SPTRSV_FP16=0");
                }
                if (std::getenv("APXCHOL_VERBOSE"))
                    std::fprintf(stderr,
                        "[apxchol] fp16 storage: diag = fp32(L_jj / s_j) + the column's rounding residual;"
                        " scale fallbacks (s_j := 1)=%llu, columns with L_jj < s_j=%llu; fat-level kernel=%s\n",
                        static_cast<unsigned long long>(stats_.scale_fallback),
                        static_cast<unsigned long long>(n_dlt),
                        simd_fp16_kernel() ? "simd" : "scalar (no AVX2/F16C/FMA at compile time)");
            }
            if (std::getenv("APXCHOL_VERBOSE")) {
                const double den = n_off ? static_cast<double>(n_off) : 1.0;
                std::fprintf(stderr,
                    "[apxchol] sptrsv storage %s: stored_nnz=%llu offdiag=%llu"
                    " flushed_to_zero=%llu (%.6f%%) subnormal=%llu (%.6f%%)"
                    " factor_subnormal(fp32 census, diag incl.)=%llu\n",
                    value_name(),
                    static_cast<unsigned long long>(stats_.nnz_stored),
                    static_cast<unsigned long long>(n_off),
                    static_cast<unsigned long long>(n_flush), 100.0 * n_flush / den,
                    static_cast<unsigned long long>(n_sub),   100.0 * n_sub / den,
                    static_cast<unsigned long long>(n_fsub));
                if (factor_drop_rel > 0.0) {
                    // Fractions are of the factor's ORIGINAL off-diagonals.
                    const std::uint64_t off0 = n_off + stats_.dropped;
                    const double den0 = off0 ? static_cast<double>(off0) : 1.0;
                    std::fprintf(stderr,
                        "[apxchol] factor drop (APXCHOL_FACTOR_DROP=%g, column sums preserved): dropped=%llu"
                        " (%.4f%% of %llu off-diagonals; threshold=%llu, format_zero=%llu)"
                        " stored_nnz %llu -> %llu (%.4f%% of factor)\n",
                        factor_drop_rel,
                        static_cast<unsigned long long>(stats_.dropped), 100.0 * stats_.dropped / den0,
                        static_cast<unsigned long long>(off0),
                        static_cast<unsigned long long>(stats_.dropped_threshold),
                        static_cast<unsigned long long>(stats_.dropped_flush),
                        static_cast<unsigned long long>(stats_.nnz_factor),
                        static_cast<unsigned long long>(stats_.nnz_stored),
                        100.0 * stats_.nnz_stored / (stats_.nnz_factor ? stats_.nnz_factor : 1));
                }
            }
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
        std::vector<factor_value_t>().swap(L11_vals_local);
        std::vector<float>().swap(col_scale_local);
        std::vector<edge_index>().swap(drop_outer);
        drop_inner.reset();
        drop_vals.reset();
        L11_outer = nullptr; L11_inner = nullptr; L11_vals = nullptr; col_scale = nullptr;

        build_levels(mark);
        ready_ = true;
    }

    // ── Level sets (storage-INDEPENDENT: they read only the CSR/CSC
    // structure, never a value) ─────────────────────────────────────────
    template <class Mark>
    void build_levels(Mark&& mark) {
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
            // signal: power-law factors have a tiny average level size yet a
            // few very fat early levels carry nearly all the work (which is why
            // level-set beat the removed sync-free schedule everywhere). The
            // predictive signals are per-level off-diagonal WORK (= SpTRSV
            // flops): bck_work_top1_frac (share in the single fattest level --
            // high => fat head) and bck_work_in_tiny (share in levels below the
            // tiny-level threshold).
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
            // Thin-level structure: per direction, the number of levels of size
            // <= kSpTRSVOMPThreshold (each an `omp single` + barrier), and the
            // length distribution of the RUNS of consecutive thin levels (the
            // single-thread tail of a solve), as "len:count" pairs.
            for (const auto* lv : {&fwd_levels_, &bck_levels_}) {
                std::size_t thin_n = 0, fat_n = 0, runs = 0;
                std::vector<std::pair<std::size_t, std::size_t>> hist;   // (run length, count), by length
                std::size_t run = 0;
                auto flush = [&] {
                    if (!run) return;
                    ++runs;
                    auto it = std::find_if(hist.begin(), hist.end(), [&](const auto& h) { return h.first == run; });
                    if (it == hist.end()) hist.emplace_back(run, 1); else ++it->second;
                    run = 0;
                };
                for (const auto& l : *lv) {
                    if (l.size() <= static_cast<std::size_t>(kSpTRSVOMPThreshold)) { ++thin_n; ++run; }
                    else { ++fat_n; flush(); }
                }
                flush();
                std::sort(hist.begin(), hist.end());
                std::fprintf(stderr, "[trsv-levels] %s: levels=%zu thin(<=%u)=%zu fat=%zu thin_runs=%zu barriers/solve=%zu"
                             " run_lengths(len:count)=",
                             lv == &fwd_levels_ ? "fwd" : "bck", lv->size(),
                             static_cast<unsigned>(kSpTRSVOMPThreshold), thin_n, fat_n, runs,
                             num_barriers(lv == &fwd_levels_));
                for (const auto& h : hist) std::fprintf(stderr, " %zu:%zu", h.first, h.second);
                std::fprintf(stderr, "\n");
            }
        }
    }

public:

    /// Forward solve: L * y = x.  Reads x[0..m-1], writes y[0..m-1].
    /// Under the fp16 storage the kernel runs on the stored L~ = L D^-1
    /// and writes y' = D y (y'_j = s_j * y_j) -- the value transpose_solve
    /// expects as its input; the pair (forward_solve, transpose_solve) applies
    /// (L L^T)^-1 either way. See the file header, "FOLDED INTO THE VECTORS".
    /// Valid in place (x_in == y_out).
    void forward_solve(const double* x_in, double* y_out) const {
        solve_dispatch<fwd_dir>(x_in, y_out);
    }

    /// Back solve: L^T * z = y.  Under the fp16 storage the input is the
    /// forward's y' = D y and the kernel solves L~^T z = D^-2 y' (input read
    /// scaled once per column by inv_scale_[j]^2), i.e. z = L^-T y as before;
    /// D = I on the fp32 storage. Valid in place (x_in == y_out).
    void transpose_solve(const double* x_in, double* y_out) const {
        solve_dispatch<bck_dir>(x_in, y_out);
    }

private:
    // ONE branch per solve on the storage the last setup() chose -- never per
    // row. On a build without F16C the fp16 instantiation does not exist.
    template <class Dir>
    void solve_dispatch(const double* x_in, double* y_out) const {
        if (fp16_) {
            if constexpr (fp16_supported()) { solve_levelset<Dir, fp16_t>(x_in, y_out); return; }
        }
        solve_levelset<Dir, float>(x_in, y_out);
    }

    // ── THE level-set kernel (both directions, both storages) ──────────
    //
    // No initializing copy of x_in into y_out: the input is folded straight
    // into the recurrence (row v reads x_in[v] in its own update), which
    // removes a serial full-vector memory pass per solve. Bit-identical to the
    // copy-then-update form: y_out[v] was only ever read at row v's own
    // update, where it still held x_in[v] (levels partition 0..m-1, each index
    // written exactly once); the gathered terms read y_out entries written by
    // earlier levels. In-place calls (x_in == y_out) stay valid for the same
    // reason (the row's x_in[v] is read before its y_out[v] is written).
    //
    // ONE persistent OpenMP team across all levels. Issuing a fresh `#pragma
    // omp parallel for` per level costs ~10us per fork-join at T=16, which
    // dominates when a factor has many levels (~240 levels x 50 PCG iters x 2
    // solves = 24k fork-joins per solve on IPM iter40). Spawning the team once
    // and iterating levels inside it -- an `omp for` (fat) or `omp single`
    // (thin) per level, whose implicit barrier carries the level-dependency
    // guarantee that level k+1 reads y_out written by level k -- collapses
    // that to one fork-join per solve.
    //
    // Schedule: thin level -> `omp single` over its rows (one thread, level
    // order); fat level -> `omp for schedule(static)` over its rows; the
    // implicit barrier of each carries the level dependency.
    template <class Dir, class V>
    void solve_levelset(const double* x_in, double* y_out) const {
        const auto& levels  = Dir::levels(*this);
        const std::size_t L = levels.size();
        #pragma omp parallel
        {
            for (std::size_t l = 0; l < L; ++l) {
                const auto& level = levels[l];
                const node_index level_sz = static_cast<node_index>(level.size());
                const node_index* lv = level.data();
                if (level_sz <= kSpTRSVOMPThreshold) {
                    #pragma omp single
                    for (node_index k = 0; k < level_sz; ++k) {
                        prefetch_ahead<Dir, V>(lv, k, level_sz);
                        solve_row<Dir, V, /*Fat=*/false>(lv[k], x_in, y_out);
                    } // implicit barrier on omp single
                } else {
                    #pragma omp for schedule(static)
                    for (node_index k = 0; k < level_sz; ++k) {
                        prefetch_ahead<Dir, V>(lv, k, level_sz);
                        solve_row<Dir, V, /*Fat=*/true>(lv[k], x_in, y_out);
                    } // implicit barrier on omp for
                }
            }
        }
    }

    // Two-stage prefetch: pull the row-pointer of the row 8 ahead (cheap ptr[]
    // load), the nnz payload (idx / vals) of the row 4 ahead, where most of
    // the cache-miss stalls occur.
    template <class Dir, class V>
    void prefetch_ahead(const node_index* level, node_index k, node_index level_sz) const {
        const auto* ptr = Dir::ptr(*this).data();
        if (k + 8 < level_sz)
            __builtin_prefetch(&ptr[level[k + 8]]);
        if (k + 4 < level_sz) {
            const node_index vp = level[k + 4];
            __builtin_prefetch(&Dir::idx(*this)[ptr[vp]]);
            __builtin_prefetch(&Dir::template vals<V>(*this)[ptr[vp]]);
        }
    }

public:
    int num_fwd_levels() const { return static_cast<int>(fwd_levels_.size()); }
    int num_bck_levels() const { return static_cast<int>(bck_levels_.size()); }
    // Barriers one solve in that direction executes: one per level (thin or
    // fat) under the level-set schedule -- the critical-path length of a solve.
    std::size_t num_barriers(bool fwd) const {
        return (fwd ? fwd_levels_ : bck_levels_).size();
    }

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

    // Read-only views of the CSR built by setup's CSC→CSR transpose. Used by
    // the SpTRSVTranspose unit tests to byte-compare the parallel transpose
    // against a serial reference (and available for diagnostics). The value
    // arrays are per storage: exactly ONE of the two pairs is non-empty after
    // a setup -- csr_vals() / csc_vals() are the fp32 ones (the default
    // storage), csr_vals16() / csc_vals16() the fp16 ones.
    const auto& csr_row_ptr() const { return csr_row_ptr_; }
    const auto& csr_col_idx() const { return csr_col_idx_; }
    const auto& csr_vals()    const { return csr_vals32_; }
    const auto& csc_col_ptr() const { return csc_col_ptr_; }
    const auto& csc_row_idx() const { return csc_row_idx_; }
    const auto& csc_vals()    const { return csc_vals32_; }
    const auto& csr_vals16()  const { return csr_vals16_; }
    const auto& csc_vals16()  const { return csc_vals16_; }
    // Per-column scales s_j of the last setup() (see column_scale()); empty
    // unless the fp16 storage was chosen.
    const auto& col_scales()  const { return scale_; }
    // The fp16 storage's fp32 scaled diagonal (empty otherwise).
    const auto& fp16_diag()   const { return diag_; }

    /// Bytes held by this object's arrays (capacities, heap only): CSR + CSC +
    /// level sets + round bounds (+ the fp16 storage's fp32 diag_ / scale_ /
    /// inv_scale_). After setup() this is everything the
    /// SpTRSV keeps -- setup's transients (L11 copy, compacted copy, transpose
    /// bucket, scratch) are all released before it returns (guarded by
    /// tests/test_sptrsv_memory.cpp).
    std::size_t memory_bytes() const {
        std::size_t b = csr_row_ptr_.capacity() * sizeof(edge_index)
                      + csr_col_idx_.capacity() * sizeof(node_index)
                      + csr_vals32_.capacity()  * sizeof(float)
                      + csr_vals16_.capacity()  * sizeof(fp16_t)
                      + csc_col_ptr_.capacity() * sizeof(edge_index)
                      + csc_row_idx_.capacity() * sizeof(node_index)
                      + csc_vals32_.capacity()  * sizeof(float)
                      + csc_vals16_.capacity()  * sizeof(fp16_t)
                      + round_bounds_.capacity() * sizeof(node_index)
                      + diag_.capacity()      * sizeof(float)
                      + scale_.capacity()     * sizeof(float)
                      + inv_scale_.capacity() * sizeof(float);
        for (const auto* lv : {&fwd_levels_, &bck_levels_}) {
            b += lv->capacity() * sizeof(std::vector<node_index>);
            for (const auto& l : *lv) b += l.capacity() * sizeof(node_index);
        }
        return b;
    }

private:
    node_index m_ = 0;
    bool ready_ = false;
    bool fp16_  = false;   // the storage the last setup() resolved

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
    // col_idx holds column ids (node_index). Exactly ONE of the two value
    // arrays is populated by a setup: the storage it resolved.
    big_vec<edge_index>     csr_row_ptr_;
    big_vec<node_index>     csr_col_idx_;
    big_vec<float>          csr_vals32_;
    big_vec<fp16_t>         csr_vals16_;

    // CSC of L11 (back solve). col_ptr is an offset array (edge_index);
    // row_idx holds row ids (node_index).
    big_vec<edge_index>     csc_col_ptr_;
    big_vec<node_index>     csc_row_idx_;
    big_vec<float>          csc_vals32_;
    big_vec<fp16_t>         csc_vals16_;

    // Storage-array selection by type, for the templated setup / kernels.
    big_vec<float>&  vals_csr(std::type_identity<float>)  { return csr_vals32_; }
    big_vec<fp16_t>& vals_csr(std::type_identity<fp16_t>) { return csr_vals16_; }
    big_vec<float>&  vals_csc(std::type_identity<float>)  { return csc_vals32_; }
    big_vec<fp16_t>& vals_csc(std::type_identity<fp16_t>) { return csc_vals16_; }
    const big_vec<float>&  vals_csr(std::type_identity<float>)  const { return csr_vals32_; }
    const big_vec<fp16_t>& vals_csr(std::type_identity<fp16_t>) const { return csr_vals16_; }
    const big_vec<float>&  vals_csc(std::type_identity<float>)  const { return csc_vals32_; }
    const big_vec<fp16_t>& vals_csc(std::type_identity<fp16_t>) const { return csc_vals16_; }

    // fp32 diagonal, i < m_ (the fp16 storage only; see the file header): the
    // scaled L(i,i) / s_i (one fp32 division, stored_diag()) plus the column's
    // storage-rounding residual.
    big_vec<float> diag_;
    // Per-column scale s_j = max |off-diagonal| of L11 column j (1.0f if
    // none), fp32 (column_scale()); the stored factor is L~ = L D^-1, D =
    // diag(s_j). Used at setup (narrowing, the drop threshold, diag_) and by
    // col_scales(); the kernels never read it. inv_scale_[j] = fp32(1 / s_j)
    // is what the back solve reads (once per column) to fold D^-2 into its
    // input -- see the file header, "FOLDED INTO THE VECTORS".
    big_vec<float> scale_;
    big_vec<float> inv_scale_;
    lowprec_statistics stats_;

    // ── The kernels: ONE source for both storage types AND both directions ──
    // Per stored entry the work is (value load, widen, index load, y gather,
    // fma); the storage type only changes the widen() overload. Per row /
    // column: the diagonal division (diag<Dir, V>) and, on the back solve of
    // the fp16 storage, the input scale (bck_dir::rhs). The DIRECTION is a
    // policy: what differs between the sweeps is only which arrays (CSR vs
    // CSC), which level list, where the diagonal slot sits (last vs first),
    // and the input transform -- everything else is solve_levelset / solve_row.
    struct fwd_dir {
        // Forward: L~ y = x on the CSR (row i; diagonal slot LAST).
        static const big_vec<edge_index>& ptr (const omp_sptrsv& s) { return s.csr_row_ptr_; }
        static const big_vec<node_index>& idx (const omp_sptrsv& s) { return s.csr_col_idx_; }
        template <class V>
        static const big_vec<V>& vals(const omp_sptrsv& s) { return s.vals_csr(std::type_identity<V>{}); }
        static const std::vector<std::vector<node_index>>& levels(const omp_sptrsv& s) { return s.fwd_levels_; }
        // Off-diagonal slots of row i: [ptr[i], ptr[i+1] - 1); diagonal at ptr[i+1] - 1.
        static edge_index first(const edge_index* ptr, node_index i) { return ptr[i]; }
        static edge_index last (const edge_index* ptr, node_index i) { return ptr[i + 1] - 1; }
        static edge_index diag_slot(const edge_index* ptr, node_index i) { return ptr[i + 1] - 1; }
        // Input transform: identity.
        template <class V>
        static double rhs(const omp_sptrsv&, node_index i, const double* x_in) { return x_in[i]; }
    };
    struct bck_dir {
        // Back: L~^T z = D^-2 y' on the CSC (column j; diagonal slot FIRST).
        static const big_vec<edge_index>& ptr (const omp_sptrsv& s) { return s.csc_col_ptr_; }
        static const big_vec<node_index>& idx (const omp_sptrsv& s) { return s.csc_row_idx_; }
        template <class V>
        static const big_vec<V>& vals(const omp_sptrsv& s) { return s.vals_csc(std::type_identity<V>{}); }
        static const std::vector<std::vector<node_index>>& levels(const omp_sptrsv& s) { return s.bck_levels_; }
        // Off-diagonal slots of column j: [ptr[j] + 1, ptr[j+1]); diagonal at ptr[j].
        static edge_index first(const edge_index* ptr, node_index j) { return ptr[j] + 1; }
        static edge_index last (const edge_index* ptr, node_index j) { return ptr[j + 1]; }
        static edge_index diag_slot(const edge_index* ptr, node_index j) { return ptr[j]; }
        // Input transform: x_in[j] * r_j^2 (r_j = fp32(1/s_j), r_j^2 exact in
        // double) under the fp16 storage -- D^-2 folded into the input read,
        // see the file header -- x_in[j] itself on fp32.
        template <class V>
        static double rhs(const omp_sptrsv& s, node_index j, const double* x_in) {
            if constexpr (std::is_same_v<V, fp16_t>) {
                const double r = static_cast<double>(s.inv_scale_[j]);
                return x_in[j] * (r * r);
            } else {
                (void)s;
                return x_in[j];
            }
        }
    };

    // L~(v,v) as the sweep divides by it -- THE single place the diagonal's
    // storage is chosen: the fp32 storage keeps the inline read (the
    // direction's diagonal slot: last of the CSR row / first of the CSC
    // column), the fp16 storage reads diag_[] (scaled by 1/s_j, with the
    // column's rounding residual folded in).
    template <class Dir, class V>
    double diag(node_index v) const {
        if constexpr (std::is_same_v<V, fp16_t>)
            return static_cast<double>(diag_[v]);
        else
            return widen(Dir::template vals<V>(*this)[Dir::diag_slot(Dir::ptr(*this).data(), v)]);
    }

    // Fat-level SIMD kernel availability: AVX2 + F16C + FMA target AND 16-bit
    // storage (fp16_t). widen8()/widen4() overloads exist for float
    // and double too, so enabling the SIMD path for the fp32 storage is
    // this one constant -- deliberately not done: that kernel stays the
    // scalar loop it was measured as.
    static constexpr bool kSimdIsa =
#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
        true;
#else
        false;
#endif
    template <class V>
    static constexpr bool simd_dot_v = kSimdIsa && sizeof(V) == 2;

    // sum over q in [p, end) of widen(vals[q]) * y[idx[q]] -- the thin-level
    // kernel: scalar, 4-way accumulators (see solve.cpp:31 for the rationale).
    template <class V>
    static double dot_thin(const V* __restrict vals, const node_index* __restrict idx,
                           edge_index p, edge_index end, const double* __restrict y) {
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        for (; p + 4 <= end; p += 4) {
            s0 += widen(vals[p + 0]) * y[idx[p + 0]];
            s1 += widen(vals[p + 1]) * y[idx[p + 1]];
            s2 += widen(vals[p + 2]) * y[idx[p + 2]];
            s3 += widen(vals[p + 3]) * y[idx[p + 3]];
        }
        double sum = (s0 + s1) + (s2 + s3);
        for (; p < end; ++p)
            sum += widen(vals[p]) * y[idx[p]];
        return sum;
    }

#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
    // Vector widen: 8 (or 4) consecutive stored values -> two (one) lanes of 4
    // doubles. One overload per storage type. (The `double` pair went with the
    // fp64 storage on 2026-08-20; the `float` pair is unused by design -- see
    // simd_dot_v below.)
    static inline void widen8(const fp16_t* v, __m256d& lo, __m256d& hi) {
        const __m256 f = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(v)));   // F16C
        lo = _mm256_cvtps_pd(_mm256_castps256_ps128(f));
        hi = _mm256_cvtps_pd(_mm256_extractf128_ps(f, 1));
    }
    static inline __m256d widen4(const fp16_t* v) {
        return _mm256_cvtps_pd(_mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(v))));
    }
    static inline void widen8(const float* v, __m256d& lo, __m256d& hi) {
        lo = _mm256_cvtps_pd(_mm_loadu_ps(v));
        hi = _mm256_cvtps_pd(_mm_loadu_ps(v + 4));
    }
    static inline __m256d widen4(const float* v) { return _mm256_cvtps_pd(_mm_loadu_ps(v)); }
    // The same sum as dot_thin, fat-level SIMD flavour (16-bit storage): 8
    // stored values per widen8, through an 8-double stack buffer (which the
    // compiler turns into register lane extracts) feeding a 4-way scalar FMA
    // chain over scalar y gathers. Then a 4-wide step and a scalar tail.
    // Different summation order from dot_thin: same accuracy, not
    // bit-identical. A template on the value type so it is only instantiated
    // where simd_dot_v selects it.
    template <class V>
    static double dot_fat_simd(const V* __restrict vals, const node_index* __restrict idx,
                               edge_index p, edge_index end, const double* __restrict y) {
        alignas(32) double hb[8];
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        for (; p + 8 <= end; p += 8) {
            __m256d h0, h1;
            widen8(vals + p, h0, h1);
            _mm256_store_pd(hb,     h0);
            _mm256_store_pd(hb + 4, h1);
            s0 += hb[0] * y[idx[p + 0]];
            s1 += hb[1] * y[idx[p + 1]];
            s2 += hb[2] * y[idx[p + 2]];
            s3 += hb[3] * y[idx[p + 3]];
            s0 += hb[4] * y[idx[p + 4]];
            s1 += hb[5] * y[idx[p + 5]];
            s2 += hb[6] * y[idx[p + 6]];
            s3 += hb[7] * y[idx[p + 7]];
        }
        if (p + 4 <= end) {
            _mm256_store_pd(hb, widen4(vals + p));
            s0 += hb[0] * y[idx[p + 0]];
            s1 += hb[1] * y[idx[p + 1]];
            s2 += hb[2] * y[idx[p + 2]];
            s3 += hb[3] * y[idx[p + 3]];
            p += 4;
        }
        double sum = (s0 + s1) + (s2 + s3);
        for (; p < end; ++p)
            sum += widen(vals[p]) * y[idx[p]];
        return sum;
    }
#endif

    // One row (forward: CSR row i) / column (back: CSC column j) of the sweep:
    //   y_out[v] = (rhs(v) - sum over the off-diagonal slots of widen(vals) * y_out[idx]) / L~_vv
    // The gather source is y_out itself (earlier levels' results); valid in
    // place (x_in == y_out: rhs reads x_in[v] before y_out[v] is written).
    // Fat: the `omp for` levels -- the SIMD kernel on 16-bit storage, else the
    // plain single-accumulator loop (instruction-identical to the pre-fold fp32
    // kernel); thin: dot_thin (4-way).
    template <class Dir, class V, bool Fat>
    void solve_row(node_index v, const double* x_in, double* y_out) const {
        const edge_index* ptr = Dir::ptr(*this).data();
        const edge_index p0 = Dir::first(ptr, v);
        const edge_index p1 = Dir::last(ptr, v);
        double sum;
        if constexpr (Fat && simd_dot_v<V>) {
#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
            sum = dot_fat_simd<V>(Dir::template vals<V>(*this).data(), Dir::idx(*this).data(), p0, p1, y_out);
#endif
        } else if constexpr (Fat) {
            const V* vals = Dir::template vals<V>(*this).data();
            const node_index* idx = Dir::idx(*this).data();
            sum = 0.0;
            for (edge_index p = p0; p < p1; ++p)
                sum += widen(vals[p]) * y_out[idx[p]];
        } else {
            sum = dot_thin<V>(Dir::template vals<V>(*this).data(), Dir::idx(*this).data(), p0, p1, y_out);
        }
        y_out[v] = (Dir::template rhs<V>(*this, v, x_in) - sum) / diag<Dir, V>(v);
    }

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
};

} // namespace apxchol
