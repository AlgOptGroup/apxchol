#pragma once
#include "apxchol/types.h"
#include "apxchol/sparse_csc.h"
#include "apxchol/big_alloc.h"
#include "apxchol/solver/sptrsv/factor_drop.h"   // the compacting drop (shared with the CUDA backend)
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
#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
#include <immintrin.h>   // fat-level SIMD kernels (16-bit storage): _mm256_cvtph_ps / _mm256_i32gather_pd / _mm256_fmadd_pd
#endif

namespace apxchol {

// Minimum level-set size before engaging OpenMP for SpTRSV.
// Below this threshold, the thread-dispatch overhead exceeds the
// computational benefit of parallelizing a single level's rows/cols.
inline constexpr node_index kSpTRSVOMPThreshold = 1024;

// kFactorDropRelDefault (the compacting drop's default threshold, 1e-4, and
// the measurements behind it), factor_drop_rel_from_env() /
// factor_drop_compensate_from_env(), factor_column_scale() and the drop itself
// (compact_factor_columns) live in factor_drop.h -- ONE host implementation
// shared with the CUDA backend; the "COMPACTING DROP" section below is the
// authoritative description of what it does and why.

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

// sptrsv_value_t (bf16_t / fp16_t / fp24_t under the APXCHOL_SPTRSV_LOWPREC
// variants, fp32 under -DAPXCHOL_SPTRSV_FP32, else fp64) is defined in
// sparse_csc.h (variants: lowprec.h) and shared with the CUDA backend
// (fp32/fp64 only). It narrows csr_vals_/csc_vals_ -- the two largest factor
// copies -- cutting memory and bandwidth on the bandwidth-bound (~94% of peak)
// triangular solve. Every read of a stored value in the solve kernels below
// goes through widen(): the arithmetic is ALWAYS done in double, whatever the
// storage width; the outer PCG stays fp64.
//
// DIAGONAL under the LOWPREC variants: the fp32/fp64 builds read L(i,i) inline
// from the factor (last entry of CSR row i / first entry of CSC column j) at
// the same precision as the off-diagonals. The lowprec builds do NOT: an 8-bit
// diagonal was measured to be the dominant iteration-count damage (iter0040
// T=1: diag-only bf16 314 PCG iters, off-diag-only 185, both 348, fp32 65), so
// every lowprec build keeps a separate fp32 `diag_[m]`, filled at setup from
// the factor (factor_value_t == float, i.e. BEFORE any narrowing) -- exactly
// L_jj under BF16 / FP24, the SCALED diagonal L_jj / s_j (one fp32 division,
// stored_diag()) under the *_SCALED variants, see below -- and both solves
// divide by diag_[i]. The narrow diagonal slot stays in csr_vals_/csc_vals_
// (written like every other entry through narrow_value -- under the *_SCALED
// variants that is narrow(L_jj / s_j), which may even overflow fp16 to inf --
// and is read by the kernels ONLY under APXCHOL_FP16_DIAG, below; keeping it
// leaves the CSR/CSC layout, the transpose, and every "diagonal is entry X"
// invariant untouched). fwd_diag()/bck_diag() below are the single switch.
//
// PER-COLUMN SCALE under the *_SCALED variants -- FOLDED INTO THE VECTORS:
// scale_[j] = s_j = max |L_ij| over the off-diagonals of column j (1.0f if
// there is none), fp32, computed at setup from the factor (column_scale()).
// What is stored is the COLUMN-SCALED factor L~ = L D^-1, D = diag(s_j):
// off-diagonals narrow(L_ij / s_j) and diag_[j] = fp32(L_jj / s_j). The
// kernels never multiply a scale back -- they run on L~ as stored, so every
// kernel path (forward / back x thin / fat, sync-free) is ONE source for every
// storage type, the only per-type difference being the widen() overload:
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
// double reference on L_s. What changed vs the pre-fold kernels: the per-ENTRY
// scale gather scale_[csr_col_idx_[p]] of the forward sweep and the per-entry
// scale multiply of both sweeps are gone (the fma reads the stored value
// directly), the diagonal is stored pre-scaled (2^-24 relative rounding of
// L_jj / s_j, vs the exact fp32 L_jj before), and the forward output / back
// input carry D. On the non-scaled builds
// (fp32 / fp64 / BF16 / FP24) D = I: forward_solve returns y and
// transpose_solve solves L^T z = x_in exactly as before -- the SCALED-only
// pieces (inv_scale_, the x_in scaling) compile out and the fp32 kernels'
// inner loops are instruction-identical to the pre-fold ones (objdump of the
// outlined `omp` bodies: same FP instruction stream -- thin: vcvtss2sd +
// vfmadd231sd x4, fat: vcvtss2sd / vmulsd / vaddsd, epilogue vsubsd /
// vcvtss2sd / vdivsd -- only register allocation in the prologues moved).
//
// SIMD CONVERT (fat levels, the `omp for` paths; 16-bit storage only, i.e.
// the BF16 / BF16_SCALED / FP16_SCALED builds; needs AVX2 + F16C + FMA, the
// -march=native default on any x86 since Haswell / Zen): 8 stored values per
// vector convert (widen8(): F16C _mm256_cvtph_ps for fp16, a 16-bit shift for
// bf16 -- overloads for float / double exist so flipping the fp32 build over
// is one constant, kSimdDot, but it is deliberately NOT flipped: the fp32
// kernel stays as measured) -> two 4-double lanes -> _mm256_fmadd_pd into two
// accumulators; the y gather is either _mm256_i32gather_pd (or i64gather
// under 64-bit node indices) -- gather mode "simd" -- or the converted values
// go through an 8-double stack buffer (which the compiler turns into register
// lane extracts) and the gather is scalar loads feeding a 4-way scalar FMA
// chain -- gather mode "scalar". Env APXCHOL_FP16_GATHER=simd|scalar (read
// at every setup, every build) selects; the default is the measured winner,
// "scalar" (kFatGatherSimdDefault). A 4-wide step and a scalar tail
// finish the row / column. Thin levels (the `omp single` paths) keep the 4-way
// scalar kernel, the sync-free back solve its single accumulator; the fp32 /
// fp64 / FP24 fat-level loop is the plain scalar loop it always was. Different
// summation orders (2 lanes vs 4-way vs 1): same accuracy, not bit-identical.
//
// APXCHOL_FP16_DIAG=1 (env, read at every setup; FP16_SCALED only): the
// kernels read the diagonal from its fp16 slot in csr_vals_/csc_vals_ --
// fp16(L_jj / s_j), what the slot has always held -- instead of the fp32
// diag_[j] = fp32(L_jj / s_j): the same scaled quantity at 11 instead of 24
// significant bits, nothing else differs. This is the SOUND fp16 diagonal:
// for the factors apxchol produces L_jj >= s_j (each eliminated column is a
// Laplacian / SDDM Schur-complement column: L_jj = sqrt(d_j), L_ij = -w_ij /
// sqrt(d_j), so |L_ij| / L_jj = w_ij / d_j <= 1; setup counts violations,
// lowprec_stats().diag_below_scale), so L_jj / s_j >= 1 is a NORMAL fp16
// (2^-11 relative) unless it is >= 65520 and rounds to +inf (a hub whose
// weighted degree exceeds 65520x its heaviest edge); storing L_jj unscaled
// would have fp16's range problem on any weighted matrix. setup counts the
// slots that are not normal finite fp16 (lowprec_stats().diag_fp16_bad); if
// any exist the mode is REFUSED (stderr warning, diag_[] used for every
// column) so a run under the env is either all-fp32 or all-fp16 diagonal,
// never a mix. Purpose: a T=1 iteration-count test of an 11-bit diagonal
// (bf16's 8 bits were the dominant damage) -- iteration-neutral would have
// licensed dropping diag_ (4 B/row) in favour of the slot already in the
// CSR/CSC streams. MEASURED (fp16s build, T=1, tol 1e-8, bg+tree[vec_pool],
// with and without APXCHOL_FACTOR_DROP=1e-4): grid_500 40 -> 40, grid_2000
// 47 -> 47, iter0040 44 -> 50 PCG iterations (+14%; 0 slots refused, 0
// columns with L_jj < s_j on all three) -- NOT neutral on IPM, so diag_ stays
// fp32 and the mode stays a diagnostic.
//
// ROUNDING mode: RNE for every variant. For the two bf16 variants,
// APXCHOL_BF16_STOCHASTIC=1 (env, read at every setup) switches the
// off-diagonal narrowing to unbiased stochastic rounding with a deterministic
// per-entry threshold (a hash of the entry's CSC index -- see bf16.h), so it
// is run-to-run and thread-count deterministic given the factor, and CSR/CSC
// agree entry-for-entry (a disagreement would make the preconditioner L1 L2^T
// non-symmetric). Motivation: RNE's per-entry error is a deterministic
// function of the value; on a Laplacian-structured factor those errors can be
// systematically signed and accumulate through the triangular solve, whereas
// unbiased rounding's errors average out. Ignored (no effect) on the other
// builds.
//
// FP16 SUBNORMALS (FP16_SCALED only): a stored fp16 subnormal (|L_ij / s_j| in
// [2^-25, 2^-14)) carries between 1 and 10 significant bits and, per the drop
// measurement below, that magnitude range is dead weight for the
// preconditioner. narrow_value therefore flushes fp16 subnormals to (signed)
// zero at storage time BY DEFAULT; env APXCHOL_FP16_KEEP_SUBNORMAL=1 (read at
// every setup) restores IEEE behaviour (subnormals stored as such). Ignored on
// every other build. The flushed count below includes them.
//
// COMPACTING DROP (every build; APXCHOL_FACTOR_DROP=<rel>, default
// kFactorDropRelDefault = 1e-4, read at every setup; <= 0 = off): setup()
// REMOVES -- not zeroes -- factor off-diagonals BEFORE the CSR/CSC are built:
// entry (i, j), i != j, is KEPT iff
//   |L_ij| >= rel * s_j   (s_j = column j's max |off-diagonal|, column_scale(),
//                          computed from the factor BEFORE the drop)
//   AND the storage format does not map it to zero anyway (format_flushes():
//   an exact zero on the fp32/fp64/bf16/fp24 builds; on FP16_SCALED also
//   everything fp16 flushes -- |L_ij / s_j| < 2^-25, or < 2^-14 with the
//   subnormal flush above -- since a stored zero is still a stored entry; at
//   the default rel = 1e-4 > 2^-14 this second clause adds nothing).
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
// CSR/CSC bytes and the per-sweep work -- shrink. The drop happens at the
// FACTOR's precision (factor_value_t), before narrow_value(): dropped entries
// never reach the storage format. Deterministic: the kept set is a per-entry
// predicate on the factor, the output order is the input order, and the
// compensation below sums each column in a fixed order (the factor itself is
// deterministic at T=1; at T>1 only fp merge-order ulps differ, as before).
// When nothing is below the threshold (grids) the arrays are left untouched
// -- no copy is made. This supersedes the earlier APXCHOL_LOWPREC_DROP
// diagnostic (same threshold, same numerics -- a stored zero and an absent
// entry solve identically -- but that one saved flops, not bytes); the
// diagnostic was removed.
//
// COLUMN-SUM COMPENSATION (default ON; APXCHOL_FACTOR_DROP_COMPENSATE=0, read
// at every setup, gives plain removal for A/B): every column of the factor of
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
// kFactorDropRelDefault. Under the *_SCALED variants the compensation runs on
// the un-scaled fp32 factor column and only THEN is the column narrowed
// (L_ij / s_j); s_j is the pre-drop column max, which the drop never removes.
//
// COLUMN-SUM COMPENSATION OF THE ROUNDING (lowprec builds; env
// APXCHOL_LOWPREC_DIAG_COMP=1, read at every setup, DEFAULT OFF -- a
// diagnostic, measured on the integration branch): the per-entry storage
// rounding delta_ij of a narrow off-diagonal is exactly the kind of
// perturbation the paragraph above is about -- it breaks the zero column
// sums of a Laplacian factor, i.e. it grounds every vertex a little (~2^-11
// relative under fp16) at both endpoints of every edge -- and on the
// Laplacian (m = n-1, center-k) path PCG pays for it exactly as it pays for
// plain removal: MEASURED iter0040 (T=1, tol 1e-8, bg+tree[vec_pool], drop
// on) fp32 45 -> FP16_SCALED 64 (67 with the drop off; 64 with fp16
// subnormals kept), while on the SDDM path (APXCHOL_GROUND=reg) both take 48.
// The lowprec builds keep the diagonal in a separate fp32 diag_[], so the
// residual can be absorbed there: with the knob, diag_[j] = fp32(x_jj +
// sum_i (x_ij - widen(h_ij))) with x = the value store() narrows (L_ij / s_j
// under *_SCALED), computed in the CSC pass -- the stored column then sums
// to what the fp32 column sums to (up to fp32), the diagonal moves by at
// most 2^-11 relative (|sum_i L_ij| = L_jj for our factors), and the
// stored_diag() contract / the unit tests describe the knob-OFF diagonal.
// Ignored under APXCHOL_FP16_DIAG (the slot is fp16). Whether this
// recovers the Laplacian-path iterations is what the knob is for; adopting
// it as the default means restating stored_diag() and its tests.
//
// STATISTICS (lowprec_stats() == drop_stats(), printed under APXCHOL_VERBOSE:
// one "sptrsv storage" line per setup, plus a "factor drop" line when the drop
// is on): the threshold in effect (rel; 0 = off) and whether the compensation
// was applied (compensate), the factor's nnz before / after the drop
// (nnz_factor, nnz_stored -- the latter is what the CSR and CSC each hold), how
// many off-diagonals the drop removed and why (dropped = dropped_threshold +
// dropped_flush), and over the STORED off-diagonals: how many stored values
// flushed to zero (v != 0, stored == 0), how many are subnormal in the storage
// format, plus the SUBNORMAL CENSUS factor_subnormal = number of factor
// entries (diagonal included, factor_value_t = fp32 on the default builds)
// that are fp32 subnormals -- on the fp32 build these ARE the stored values,
// so this is exactly "how many stored fp32 factor values are subnormal" (the
// DAZ/FTZ question; see APXCHOL_FTZ in solve.cpp).

class omp_sptrsv {
public:
    omp_sptrsv() = default;

    // Compiled width of the off-diagonal factor values, read straight off the
    // type the compiler actually built. 2 == this TU was compiled with
    // APXCHOL_SPTRSV_LOWPREC = BF16 / BF16_SCALED / FP16_SCALED, 3 == FP24,
    // 4 == -DAPXCHOL_SPTRSV_FP32, 8 == fp64. Printed at startup so the build
    // flag is observable at runtime (no inferring from residuals).
    static constexpr std::size_t value_bytes = sizeof(sptrsv_value_t);
    static constexpr const char* value_name =
#if defined(APXCHOL_SPTRSV_LOWPREC_BF16)
        "bf16";
#elif defined(APXCHOL_SPTRSV_LOWPREC_BF16_SCALED)
        "bf16 (per-column scaled)";
#elif defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED)
        "fp16 (per-column scaled)";
#elif defined(APXCHOL_SPTRSV_LOWPREC_FP24)
        "fp24";
#else
        sizeof(sptrsv_value_t) == 4 ? "float (fp32)" : "double (fp64)";
#endif
    // The APXCHOL_SPTRSV_LOWPREC variant this TU compiled ("OFF" for fp32/fp64).
    static constexpr const char* lowprec_variant = sptrsv_lowprec_variant;

    // THE storage contract (public so the unit tests can state it): what
    // setup() stores for the factor entry at CSC position k with value v, in a
    // column whose per-column scale is s (the *_SCALED variants: s_j = max
    // |off-diagonal| of column j, 1.0f if none; every other variant ignores s
    // -- pass 1.0f). `stochastic` selects bf16 stochastic rounding (bf16
    // variants only; ignored elsewhere); `fp16_flush_subnormal` flushes fp16
    // subnormals to signed zero (FP16_SCALED only, the default there -- see
    // the file header; ignored elsewhere). A PURE function of its arguments:
    // the CSR transpose and the CSC copy both call it, so the two stored
    // copies of every entry agree bit-for-bit whatever the rounding mode
    // (needed for the stochastic mode; a no-op cast on the fp32/fp64 builds,
    // exactly the static_cast they always did). The compacting drop
    // (APXCHOL_FACTOR_DROP) happens BEFORE this: dropped entries never reach
    // it.
    static sptrsv_value_t narrow_value(factor_value_t v, edge_index k, float s, bool stochastic,
                                       bool fp16_flush_subnormal) {
#if defined(APXCHOL_SPTRSV_LOWPREC_SCALED)
        const float x = static_cast<float>(v) / s;      // |x| <= 1 for the off-diagonals as factorized (s is
                                                        // their max; the drop's compensation may lift kept
                                                        // entries a little above -- fp16 has range to spare)
#else
        (void)s;
        const factor_value_t x = v;
#endif
#if defined(APXCHOL_SPTRSV_LOWPREC_BF16) || defined(APXCHOL_SPTRSV_LOWPREC_BF16_SCALED)
        (void)fp16_flush_subnormal;
        return stochastic ? from_float_stochastic(x, static_cast<std::uint64_t>(k))
                          : bf16_t(x);                   // RNE
#elif defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED)
        (void)k; (void)stochastic;
        const fp16_t h(x);                               // RNE, subnormals / flush per IEEE
        if (fp16_flush_subnormal && fp16_t::is_subnormal(h.bits))
            return fp16_t::from_bits(static_cast<std::uint16_t>(h.bits & 0x8000u));   // signed zero
        return h;
#elif defined(APXCHOL_SPTRSV_LOWPREC_FP24)
        (void)k; (void)stochastic; (void)fp16_flush_subnormal;
        return fp24_t(x);                                // RNE on the dropped 8 bits
#else
        (void)k; (void)stochastic; (void)fp16_flush_subnormal;
        return static_cast<sptrsv_value_t>(x);
#endif
    }

    // True iff THIS build's storage format maps the off-diagonal v (in a
    // column with scale s) to zero regardless of rounding mode: an exact zero
    // on the fp32 / fp64 / bf16 / fp24 builds (they keep fp32's exponent range,
    // so a nonzero fp32 factor entry never rounds to zero -- bf16 stochastic
    // rounding included, whose two candidates bracket a nonzero x); on
    // FP16_SCALED everything fp16 flushes (|v / s| < 2^-25 under RNE) plus,
    // with `fp16_flush_subnormal`, the fp16 subnormal range (< 2^-14 after
    // rounding). Pure; independent of the entry's position k.
    static bool format_flushes(factor_value_t v, float s, bool fp16_flush_subnormal) {
#if defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED)
        const fp16_t h(static_cast<float>(v) / s);
        return fp16_t::is_zero(h.bits) || (fp16_flush_subnormal && fp16_t::is_subnormal(h.bits));
#else
        (void)s; (void)fp16_flush_subnormal;
        return v == 0;
#endif
    }

    // THE compacting-drop predicate (public so the tests can state it): the
    // off-diagonal v of a column with scale s survives APXCHOL_FACTOR_DROP=rel
    // iff |v| >= rel * s and the storage format does not map it to zero. The
    // diagonal is never passed through this (always kept).
    static bool keep_offdiag(factor_value_t v, float s, double rel, bool fp16_flush_subnormal) {
        return std::fabs(static_cast<double>(v)) >= rel * static_cast<double>(s) &&
               !format_flushes(v, s, fp16_flush_subnormal);
    }

    // Per-column scale contract (public for the tests): the *_SCALED variants'
    // s_j, computed by setup() from the factor column [first, last) whose FIRST
    // entry is the diagonal: max |off-diagonal|, or 1.0f if the column has no
    // nonzero off-diagonal (so v / s_j is always defined). Also the reference
    // of the compacting drop's threshold (computed BEFORE the drop; the drop
    // never removes the column max, so it is the same after). Computed in
    // double, stored fp32 (the factor is fp32 on every build that uses it, so
    // this is exact).
    static float column_scale(const factor_value_t* vals, edge_index first, edge_index last) {
        return factor_column_scale(vals, first, last);   // factor_drop.h (shared with the GPU backend)
    }

    // Diagonal contract (public for the tests): what the kernels DIVIDE by for
    // a column whose factor diagonal is L_jj and whose scale is s (1.0f on the
    // non-scaled builds). The *_SCALED variants store the scaled diagonal
    // fp32(L_jj / s) -- one fp32 division -- in diag_[j] (see the file header,
    // "FOLDED INTO THE VECTORS"); the other lowprec builds store L_jj itself
    // in diag_[j]; the fp32/fp64 builds read the factor value inline (== L_jj at
    // the factor's width). Under APXCHOL_FP16_DIAG (FP16_SCALED) the kernels
    // divide by widen(fp16(L_jj / s)) instead, i.e. narrow_value(L_jj, ., s, .).
    static double stored_diag(factor_value_t L_jj, float s) {
#if defined(APXCHOL_SPTRSV_LOWPREC_SCALED)
        return static_cast<double>(static_cast<float>(L_jj) / s);
#elif defined(APXCHOL_SPTRSV_LOWPREC_ANY)
        (void)s;
        return static_cast<double>(static_cast<float>(L_jj));       // fp32 diag_
#else
        (void)s;
        return widen(L_jj);   // inline read: factor_value_t == sptrsv_value_t here
#endif
    }
    // Back-input scale contract (public for the tests): transpose_solve reads
    // x_in[j] * r_j^2 with r_j = inv_scale(s_j) (fp32 reciprocal; r_j^2 exact
    // in double); 1.0 on the non-scaled builds.
    static double inv_scale(float s) {
#if defined(APXCHOL_SPTRSV_LOWPREC_SCALED)
        return static_cast<double>(1.0f / s);
#else
        (void)s;
        return 1.0;
#endif
    }

    // APXCHOL_FACTOR_DROP=<rel> resolution, read at every setup(): unset (or
    // empty) -> kFactorDropRelDefault; set -> its value, where anything <= 0
    // (including "0" and unparsable text, which atof reads as 0) turns the
    // drop OFF.
    static double factor_drop_rel_from_env() { return apxchol::factor_drop_rel_from_env(); }
    // APXCHOL_FACTOR_DROP_COMPENSATE: unset / anything but "0" -> the
    // column-sum compensation is applied (default); "0" -> plain removal (the
    // A/B switch; see the file header). Read at every setup().
    static bool factor_drop_compensate_from_env() { return apxchol::factor_drop_compensate_from_env(); }

    // Statistics of the last setup() (see the file header): what the
    // compacting drop did to L11 = L.topLeftCorner(m, m) (rel / compensate /
    // nnz_factor / nnz_stored / dropped*, every build) and what the storage
    // format did to what remained. offdiag / flushed / subnormal /
    // factor_subnormal are over the STORED factor (after the drop); the
    // dropped_* counts are what the drop removed; nnz_factor is L11's nnz
    // before the drop and nnz_stored after (== nnz_factor when the drop is off).
    struct lowprec_statistics {
        double        rel               = 0.0;  // drop threshold in effect (0 = drop off)
        bool          compensate        = true; // column-sum compensation applied to the dropped columns
        std::uint64_t offdiag           = 0;   // number of stored off-diagonal entries of L11
        std::uint64_t flushed           = 0;   // stored as zero although the factor value was nonzero
        std::uint64_t subnormal         = 0;   // stored as a subnormal of the storage format
        std::uint64_t factor_subnormal  = 0;   // factor entries (fp32, diagonal incl.) that are fp32 subnormals
        std::uint64_t dropped           = 0;   // off-diagonals removed by APXCHOL_FACTOR_DROP (= the two below)
        std::uint64_t dropped_threshold = 0;   //   ... because |L_ij| < rel * s_j
        std::uint64_t dropped_flush     = 0;   //   ... because the storage format stores them as zero anyway
        std::uint64_t nnz_factor        = 0;   // nnz of L11 as factorized (before the drop)
        std::uint64_t nnz_stored        = 0;   // nnz the CSR (and the CSC) hold (after the drop)
        // FP16_SCALED only (0 elsewhere): diagonal slots fp16(L_jj / s_j) that
        // are not a normal finite fp16 (inf / nan / zero / subnormal -- what
        // refuses APXCHOL_FP16_DIAG), and columns WITH an off-diagonal whose
        // L_jj < s_j (the diagonal-dominance sanity count the fp16 diagonal's
        // soundness rests on; off-diagonal-free columns have s_j = 1.0f).
        std::uint64_t diag_fp16_bad     = 0;
        std::uint64_t diag_below_scale  = 0;
    };
    const lowprec_statistics& lowprec_stats() const { return stats_; }
    // The same record under the name the drop-only (fp32/fp64) callers use:
    // rel / compensate / nnz_factor / nnz_stored / dropped are its drop half.
    const lowprec_statistics& drop_stats() const { return stats_; }
    // nnz held by the SpTRSV's CSR / CSC (each) after the last setup().
    std::uint64_t stored_nnz() const { return stats_.nnz_stored; }
    // FP16_SCALED: true iff the last setup() honoured APXCHOL_FP16_DIAG=1 (the
    // kernels divide by the fp16 diagonal slot instead of the fp32 diag_[]);
    // always false on every other build.
    bool fp16_diag() const { return fp16_diag_; }
    // Lowprec builds: true iff the last setup() applied APXCHOL_LOWPREC_DIAG_COMP=1
    // (diag_[j] carries the column's storage-rounding residual, see the file
    // header); always false on the fp32/fp64 builds.
    bool lowprec_diag_comp() const { return diag_comp_; }
    // Whether the fat-level 16-bit-storage kernels use the SIMD gather
    // (APXCHOL_FP16_GATHER=simd) rather than the stack-buffer + scalar-gather
    // flavour (=scalar) -- the value the last setup() resolved (default:
    // kFatGatherSimdDefault). Read on every build; only acted on where
    // simd_dot() is true.
    bool fat_gather_simd() const { return gather_simd_; }
    // Whether THIS build's fat-level kernels are the SIMD ones (16-bit storage
    // on an AVX2 + F16C + FMA target).
    static constexpr bool simd_dot() { return kSimdDot; }

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
        std::vector<factor_value_t> L11_vals_local;
        const edge_index*     L11_outer;
        const node_index*     L11_inner;
        const factor_value_t* L11_vals;     // the FACTOR's width: fp32 under FP32/BF16, else fp64
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

        // Storage-mode envs, read at every setup (any build; each only has an
        // effect on its own variant): bf16 rounding mode, fp16 subnormal flush
        // (default ON, see file header).
        const bool bf16_stochastic = [] {
            const char* e = std::getenv("APXCHOL_BF16_STOCHASTIC");
            return e && std::atoi(e) != 0;
        }();
        const bool fp16_flush_subnormal = [] {
            const char* e = std::getenv("APXCHOL_FP16_KEEP_SUBNORMAL");
            return !(e && std::atoi(e) != 0);
        }();
        // APXCHOL_FACTOR_DROP=<rel> (every build; default kFactorDropRelDefault,
        // <= 0 = off) and APXCHOL_FACTOR_DROP_COMPENSATE (see the file header).
        const double factor_drop_rel = factor_drop_rel_from_env();
        const bool   drop_compensate = factor_drop_compensate_from_env();
        stats_ = lowprec_statistics{};
        stats_.rel        = factor_drop_rel;
        stats_.compensate = drop_compensate;
        stats_.nnz_factor = static_cast<std::uint64_t>(nnz);
        // Per-column scale s_j (the *_SCALED variants' scale_ -- a member, the
        // storage divides by it -- and the compacting drop's threshold
        // reference; not needed otherwise). See column_scale() for the
        // contract. Computed from the factor BEFORE the drop. On the non-scaled
        // builds it is a drop-only transient (m floats), freed right after the
        // drop block: the storage there ignores the scale.
#if defined(APXCHOL_SPTRSV_LOWPREC_SCALED)
        constexpr bool kStoreScaled = true;    // narrow_value / stored_diag read s_j
        scale_.resize(m_);
        float* const col_scale = scale_.data();
        const bool need_scale = true;
#else
        constexpr bool kStoreScaled = false;   // ... and ignore it here
        std::vector<float> col_scale_local;
        if (factor_drop_rel > 0.0) col_scale_local.resize(m_);
        float* const col_scale = col_scale_local.data();
        const bool need_scale = factor_drop_rel > 0.0;
#endif
        if (need_scale) {
            #pragma omp parallel for schedule(static)
            for (node_index j = 0; j < m_; ++j)
                col_scale[j] = column_scale(L11_vals, L11_outer[j], L11_outer[j + 1]);
            mark("col_scale");
        }
#if defined(APXCHOL_SPTRSV_LOWPREC_SCALED)
        // r_j = fp32(1 / s_j): the back solve's per-column input scale (see the
        // file header, "FOLDED INTO THE VECTORS"; inv_scale() is the contract).
        inv_scale_.resize(m_);
        #pragma omp parallel for schedule(static)
        for (node_index j = 0; j < m_; ++j)
            inv_scale_[j] = 1.0f / col_scale[j];
#endif

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
                m_, L11_outer, L11_inner, L11_vals, col_scale, factor_drop_rel, drop_compensate,
                [=](factor_value_t v, float s) { return keep_offdiag(v, s, factor_drop_rel, fp16_flush_subnormal); },
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
#if !defined(APXCHOL_SPTRSV_LOWPREC_SCALED)
        // Last read of the drop-only scales on the non-scaled builds (store()
        // and the diagonal below do not use them): free, not at return.
        std::vector<float>().swap(col_scale_local);
#endif

        // store(v, k, i, j): the factor entry L(i,j) at CSC position k of the
        // (possibly compacted) L11 -> the SpTRSV's storage width, via
        // narrow_value() (see its contract above). Both stored copies of an
        // entry (CSR transpose below, CSC copy) go through this same pure
        // function.
        const auto store = [=](factor_value_t v, edge_index k, node_index /*i*/, node_index j) -> sptrsv_value_t {
            const float s = kStoreScaled ? col_scale[j] : 1.0f;
            return narrow_value(v, k, s, bf16_stochastic, fp16_flush_subnormal);
        };
#if defined(APXCHOL_SPTRSV_LOWPREC_ANY)
        // fp32 diagonal, straight from the factor (factor_value_t == float
        // here; NOT via the narrowing path): L_jj, or under the *_SCALED
        // variants the scaled L_jj / s_j (stored_diag() is the contract). L(j,j)
        // is the FIRST entry of CSC column j -- the invariant the back solve has
        // always relied on.
        diag_.resize(m_);
        #pragma omp parallel for schedule(static)
        for (node_index j = 0; j < m_; ++j) {
            assert(L11_inner[L11_outer[j]] == j && "factor column must start with its diagonal");
            diag_[j] = static_cast<float>(stored_diag(L11_vals[L11_outer[j]], kStoreScaled ? col_scale[j] : 1.0f));
        }
        mark("diag_fp32");
#endif

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
        // 32-bit-index fp32 build, 10-11 B/nnz under the lowprec variants),
        // allocated uninitialized (every slot is
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
                        bkt_val[out] = store(L11_vals[p], p, row, j);
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
                    csr_vals_[out]    = store(L11_vals[p], p, row, j);
                }
            }
        }

        // fp32/fp64 builds: no separate diagonal array -- the solve reads L(i,i)
        // inline from the factor, the LAST entry of CSR row i (forward: sum loop
        // stops one short) and the FIRST entry of CSC column j (back: sum loop
        // starts one in), at the same precision as the off-diagonals (the read
        // widens like every other one). This is bit-identical to the old fp64
        // diag_ (L11_vals is already fp32 under the flag, so diag_ only ever held
        // the fp32 value widened to double) and matches the GPU backend, which has
        // always read the diagonal inline. The lowprec builds keep the exact-fp32
        // diag_ filled above instead (see fwd_diag / bck_diag); their narrow
        // diagonal slots in csr_vals_/csc_vals_ are written like every other
        // entry but never read.
        mark("csc_to_csr");

        // ── CSC of L11 (for back solve) ─────────────────────────
        // Parallel copy of the three arrays (values through store()), column
        // by column so store() knows the entry's column; this pass also
        // gathers the off-diagonal storage statistics (each entry once).
        csc_col_ptr_.resize(static_cast<size_t>(m_) + 1);
        csc_row_idx_.resize(nnz);
        csc_vals_.resize(nnz);
        #pragma omp parallel for schedule(static)
        for (node_index i = 0; i <= m_; ++i)
            csc_col_ptr_[i] = L11_outer[i];
        {
#if defined(APXCHOL_SPTRSV_LOWPREC_ANY)
            // APXCHOL_LOWPREC_DIAG_COMP=1 (env, read at every setup; lowprec
            // builds only; default OFF -- a DIAGNOSTIC, see the file header,
            // "COLUMN-SUM COMPENSATION OF THE ROUNDING"): fold each column's
            // storage-rounding residual sum_i (x_ij - widen(h_ij)) (x = the
            // value store() narrows, i.e. L_ij / s_j under *_SCALED) into the
            // fp32 diag_[j], so the STORED column sums to what the fp32
            // (compensated-drop) column sums to.
            const bool diag_comp = [] {
                const char* e = std::getenv("APXCHOL_LOWPREC_DIAG_COMP");
                return e && std::atoi(e) != 0;
            }();
            diag_comp_ = diag_comp;
#endif
            std::uint64_t n_off = 0, n_flush = 0, n_sub = 0, n_fsub = 0, n_dbad = 0, n_dlt = 0;
            #pragma omp parallel for schedule(static) reduction(+ : n_off, n_flush, n_sub, n_fsub, n_dbad, n_dlt)
            for (node_index j = 0; j < m_; ++j) {
                double resid = 0.0;                                // sum over the off-diagonals of (x - widen(stored))
                for (edge_index k = L11_outer[j]; k < L11_outer[j + 1]; ++k) {
                    const node_index    i  = L11_inner[k];
                    const factor_value_t v = L11_vals[k];
                    const sptrsv_value_t w = store(v, k, i, j);
                    csc_row_idx_[k] = i;
                    csc_vals_[k]    = w;
                    if (is_stored_subnormal(v)) ++n_fsub;          // census: the FACTOR value (fp32), diagonal incl.
#if defined(APXCHOL_SPTRSV_LOWPREC_ANY)
                    if (diag_comp && i != j) {
                        const double x = static_cast<double>(v) / (kStoreScaled ? static_cast<double>(col_scale[j]) : 1.0);
                        resid += x - widen(w);
                    }
#endif
                    if (i == j) {                                  // diagonal slot: unread under lowprec ...
#if defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED)
                        // ... except under APXCHOL_FP16_DIAG (see file header):
                        // fp16(L_jj / s_j) must be a normal finite fp16.
                        if (fp16_t::is_inf_or_nan(w.bits) || fp16_t::is_zero(w.bits) || fp16_t::is_subnormal(w.bits)) ++n_dbad;
                        // L_jj < s_j among columns that HAVE an off-diagonal (s_j
                        // is the placeholder 1.0f otherwise).
                        if (L11_outer[j + 1] - L11_outer[j] > 1 &&
                            static_cast<double>(v) < static_cast<double>(col_scale[j])) ++n_dlt;
#endif
                        continue;
                    }
                    ++n_off;
                    if (v != 0 && widen(w) == 0.0) {
                        ++n_flush;                                 // zeroed by the storage format
                    } else if (is_stored_subnormal(w)) {
                        ++n_sub;
                    }
                }
#if defined(APXCHOL_SPTRSV_LOWPREC_ANY)
                if (diag_comp)
                    diag_[j] = static_cast<float>(static_cast<double>(diag_[j]) + resid);
#else
                (void)resid;
#endif
            }
            stats_.offdiag = n_off; stats_.flushed = n_flush;
            stats_.subnormal = n_sub; stats_.factor_subnormal = n_fsub;
            stats_.diag_fp16_bad = n_dbad; stats_.diag_below_scale = n_dlt;
            // Fat-level gather flavour of the 16-bit-storage SIMD kernels (see
            // the file header, "SIMD CONVERT"): env read at every setup, on
            // every build (acted on only where simd_dot()).
            gather_simd_ = [] {
                const char* e = std::getenv("APXCHOL_FP16_GATHER");
                if (!e || !*e) return kFatGatherSimdDefault;
                const std::string_view s(e);
                if (s == "simd")   return true;
                if (s == "scalar") return false;
                std::fprintf(stderr, "[apxchol] APXCHOL_FP16_GATHER=%s ignored (expected simd|scalar); using the default\n", e);
                return kFatGatherSimdDefault;
            }();
#if defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED)
            // APXCHOL_FP16_DIAG (see the file header): honoured only if every
            // diagonal slot is a normal finite fp16.
            const bool want_fp16_diag = [] {
                const char* e = std::getenv("APXCHOL_FP16_DIAG");
                return e && std::atoi(e) != 0;
            }();
            fp16_diag_ = want_fp16_diag && n_dbad == 0;
            if (want_fp16_diag && !fp16_diag_)
                std::fprintf(stderr,
                    "[apxchol] APXCHOL_FP16_DIAG=1 REFUSED: %llu of %llu diagonal slots fp16(L_jj / s_j) are not a normal"
                    " finite fp16 (L_jj / s_j >= 65520 rounds to inf); the fp32 diag_[] is used for every column\n",
                    static_cast<unsigned long long>(n_dbad), static_cast<unsigned long long>(m_));
            if (std::getenv("APXCHOL_VERBOSE"))
                std::fprintf(stderr,
                    "[apxchol] fp16 diag: %s (APXCHOL_FP16_DIAG=%d; slots not normal fp16=%llu, columns with L_jj < s_j=%llu);"
                    " fat-level kernel=%s\n",
                    fp16_diag_ ? "fp16 slot fp16(L_jj / s_j), 11-bit" : "fp32 diag_[] = fp32(L_jj / s_j)", want_fp16_diag ? 1 : 0,
                    static_cast<unsigned long long>(n_dbad), static_cast<unsigned long long>(n_dlt),
                    !kSimdDot ? "scalar (no AVX2/F16C/FMA at compile time)" : gather_simd_ ? "simd, gather=simd" : "simd, gather=scalar");
#endif
            if (std::getenv("APXCHOL_VERBOSE")) {
                const double den = n_off ? static_cast<double>(n_off) : 1.0;
                std::fprintf(stderr,
                    "[apxchol] sptrsv storage %s (lowprec=%s%s): stored_nnz=%llu offdiag=%llu"
                    " flushed_to_zero=%llu (%.6f%%) subnormal=%llu (%.6f%%)"
                    " factor_subnormal(fp32 census, diag incl.)=%llu\n",
                    value_name, lowprec_variant, diag_comp_ ? ", diag_comp" : "",
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
                        "[apxchol] factor drop (APXCHOL_FACTOR_DROP=%g%s%s): dropped=%llu (%.4f%% of %llu off-diagonals;"
                        " threshold=%llu, format_zero=%llu) stored_nnz %llu -> %llu (%.4f%% of factor)\n",
                        factor_drop_rel,
                        drop_compensate ? ", column sums preserved" : ", plain removal (COMPENSATE=0)",
#if defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED)
                        fp16_flush_subnormal ? ", fp16 subnormals flushed" : ", fp16 subnormals kept",
#else
                        "",
#endif
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
    /// Under the *_SCALED variants the kernel runs on the stored L~ = L D^-1
    /// and writes y' = D y (y'_j = s_j * y_j) -- the value transpose_solve
    /// expects as its input; the pair (forward_solve, transpose_solve) applies
    /// (L L^T)^-1 on every build. See the file header, "FOLDED INTO THE
    /// VECTORS". Always the level-set scheduler: empirically beats sync-free across
    /// all level counts we tested (BG/luby/BK with thousands of levels and
    /// rootset with ~65 levels), including the thin-level regime where the
    /// back solve prefers sync-free — so the forward sync-free variant was
    /// dropped.  (The back solve still chooses between the two: see
    /// transpose_solve.)
    void forward_solve(const double* x_in, double* y_out) const {
        forward_solve_levelset(x_in, y_out);
    }

    /// Back solve: L^T * z = y.  Under the *_SCALED variants the input is the
    /// forward's y' = D y and the kernel solves L~^T z = D^-2 y' (input read
    /// scaled once per column by inv_scale_[j]^2), i.e. z = L^-T y as before;
    /// D = I elsewhere. Default: level-set scheduler.  Paired A/B
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
                            fwd_row</*Fat=*/false>(i, x_in, y_out);
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
                        fwd_row</*Fat=*/true>(i, x_in, y_out);
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
                            bck_col</*Fat=*/false>(j, x_in, y_out);
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
                        bck_col</*Fat=*/true>(j, x_in, y_out);
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
                    sum += widen(csc_vals_[p]) * y_out[csc_row_idx_[p]];
                y_out[j] = (bck_rhs(j, y_out) - sum) / bck_diag(j);   // x_in was copied into y_out above

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

    // Read-only views of the CSR built by setup's CSC→CSR transpose. Used by
    // the SpTRSVTranspose unit tests to byte-compare the parallel transpose
    // against a serial reference (and available for diagnostics).
    const auto& csr_row_ptr() const { return csr_row_ptr_; }
    const auto& csr_col_idx() const { return csr_col_idx_; }
    const auto& csr_vals()    const { return csr_vals_; }
    const auto& csc_col_ptr() const { return csc_col_ptr_; }
    const auto& csc_row_idx() const { return csc_row_idx_; }
    const auto& csc_vals()    const { return csc_vals_; }
#if defined(APXCHOL_SPTRSV_LOWPREC_SCALED)
    // Per-column scales s_j of the last setup() (see column_scale()).
    const auto& col_scales()  const { return scale_; }
#endif

    /// Bytes held by this object's arrays (capacities, heap only): CSR + CSC +
    /// level sets + round bounds + sync-free counters (+ the lowprec builds'
    /// fp32 diag_ and the *_SCALED variants' scale_ / inv_scale_). After
    /// setup() this is everything the SpTRSV keeps -- setup's transients (L11
    /// copy, compacted copy, transpose bucket, scratch) are all released
    /// before it returns (guarded by tests/test_sptrsv_memory.cpp).
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
#if defined(APXCHOL_SPTRSV_LOWPREC_ANY)
        b += diag_.capacity() * sizeof(float);
#endif
#if defined(APXCHOL_SPTRSV_LOWPREC_SCALED)
        b += scale_.capacity() * sizeof(float) + inv_scale_.capacity() * sizeof(float);
#endif
        for (const auto* lv : {&fwd_levels_, &bck_levels_}) {
            b += lv->capacity() * sizeof(std::vector<node_index>);
            for (const auto& l : *lv) b += l.capacity() * sizeof(node_index);
        }
        return b;
    }

private:
    node_index m_ = 0;
    bool ready_ = false;

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
    big_vec<sptrsv_value_t> csr_vals_;   // fp32 under APXCHOL_SPTRSV_FP32; bf16/fp16/fp24 under the LOWPREC variants

    // CSC of L11 (back solve). col_ptr is an offset array (edge_index);
    // row_idx holds row ids (node_index).
    big_vec<edge_index>     csc_col_ptr_;
    big_vec<node_index>     csc_row_idx_;
    big_vec<sptrsv_value_t> csc_vals_;   // fp32 under APXCHOL_SPTRSV_FP32; bf16/fp16/fp24 under the LOWPREC variants

#if defined(APXCHOL_SPTRSV_LOWPREC_ANY)
    // fp32 diagonal, i < m_ (lowprec builds only; see the file header): L(i,i)
    // under BF16 / FP24 -- exactly what the fp32 build divides by, so those
    // differ from fp32 ONLY in the off-diagonals -- and the scaled L(i,i) / s_i
    // (one fp32 division, stored_diag()) under the *_SCALED variants.
    big_vec<float> diag_;
#endif
#if defined(APXCHOL_SPTRSV_LOWPREC_SCALED)
    // Per-column scale s_j = max |off-diagonal| of L11 column j (1.0f if
    // none), fp32 (column_scale()); the stored factor is L~ = L D^-1, D =
    // diag(s_j). Used at setup (narrowing, the drop threshold, diag_) and by
    // col_scales(); the kernels never read it. inv_scale_[j] = fp32(1 / s_j)
    // is what the back solve reads (once per column) to fold D^-2 into its
    // input -- see the file header, "FOLDED INTO THE VECTORS".
    big_vec<float> scale_;
    big_vec<float> inv_scale_;
#endif
    lowprec_statistics stats_;

    // ── The kernels: ONE source for every storage type ─────────────────
    // Per stored entry the work is (value load, widen, index load, y gather,
    // fma); the storage type only changes the widen() overload. Per row /
    // column: the diagonal division (fwd_diag / bck_diag) and, on the back
    // solve of the *_SCALED variants, the input scale (bck_rhs).

    // L(i,i) as the forward solve (CSR row i) / back solve (CSC column j)
    // divides by it -- THE single place the diagonal's storage is chosen: the
    // fp32/fp64 builds keep the inline read (last of CSR row / first of CSC
    // column, byte-identical to before), the lowprec builds read diag_[]
    // (scaled by 1/s_j under *_SCALED), and FP16_SCALED under
    // APXCHOL_FP16_DIAG (fp16_diag_) the fp16 diagonal slot itself.
    double fwd_diag(node_index i) const {
#if defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED)
        return fp16_diag_ ? widen(csr_vals_[csr_row_ptr_[i + 1] - 1]) : static_cast<double>(diag_[i]);
#elif defined(APXCHOL_SPTRSV_LOWPREC_ANY)
        return static_cast<double>(diag_[i]);
#else
        return widen(csr_vals_[csr_row_ptr_[i + 1] - 1]);
#endif
    }
    double bck_diag(node_index j) const {
#if defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED)
        return fp16_diag_ ? widen(csc_vals_[csc_col_ptr_[j]]) : static_cast<double>(diag_[j]);
#elif defined(APXCHOL_SPTRSV_LOWPREC_ANY)
        return static_cast<double>(diag_[j]);
#else
        return widen(csc_vals_[csc_col_ptr_[j]]);
#endif
    }
    // The back solve's input for column j: x_in[j] * r_j^2 (r_j = fp32(1/s_j),
    // r_j^2 exact in double) under the *_SCALED variants -- D^-2 folded into
    // the input read, see the file header -- x_in[j] itself elsewhere.
    double bck_rhs(node_index j, const double* x_in) const {
#if defined(APXCHOL_SPTRSV_LOWPREC_SCALED)
        const double r = static_cast<double>(inv_scale_[j]);
        return x_in[j] * (r * r);
#else
        return x_in[j];
#endif
    }

    // Fat-level SIMD kernel availability: AVX2 + F16C + FMA target AND 16-bit
    // storage (bf16_t / fp16_t). widen8()/widen4() overloads exist for float
    // and double too, so enabling the SIMD path for the fp32/fp64 builds is
    // this one constant -- deliberately not done: those kernels stay the
    // scalar loops they were measured as.
    static constexpr bool kSimdIsa =
#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
        true;
#else
        false;
#endif
    static constexpr bool kSimdDot = kSimdIsa && sizeof(sptrsv_value_t) == 2;
    // Default fat-level gather flavour (APXCHOL_FP16_GATHER unset): "scalar"
    // -- the winner of the unlocked A/B (fp16s, T=1: grid_2000 ~10-15% faster
    // solve than the vector gather, iter0040 equal; T=8 within noise; see the
    // commit message). vgatherdpd's latency sits on the critical path of the
    // short grid rows; the compiler turns the "stack buffer" into register
    // lane extracts feeding vfmadd231sd with a memory operand, so the scalar
    // flavour has no store/reload either.
    static constexpr bool kFatGatherSimdDefault = false;
    // Kernel state set by setup() (see there).
    bool diag_comp_   = false;   // lowprec builds: APXCHOL_LOWPREC_DIAG_COMP=1 applied at the last setup()
    bool fp16_diag_   = false;   // FP16_SCALED: APXCHOL_FP16_DIAG=1 honoured at the last setup()
    bool gather_simd_ = false;   // fat-level SIMD kernels: vector gather (true) or stack buffer + scalar gather

    // sum over q in [p, end) of widen(vals[q]) * y[idx[q]] -- the thin-level
    // kernel: scalar, 4-way accumulators (see solve.cpp:31 for the rationale).
    static double dot_thin(const sptrsv_value_t* __restrict vals, const node_index* __restrict idx,
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
    // doubles. One overload per storage type; fp24_t has none (3-byte packed
    // -> kSimdDot is false there anyway).
    static inline void widen8(const fp16_t* v, __m256d& lo, __m256d& hi) {
        const __m256 f = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(v)));   // F16C
        lo = _mm256_cvtps_pd(_mm256_castps256_ps128(f));
        hi = _mm256_cvtps_pd(_mm256_extractf128_ps(f, 1));
    }
    static inline __m256d widen4(const fp16_t* v) {
        return _mm256_cvtps_pd(_mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(v))));
    }
    static inline void widen8(const bf16_t* v, __m256d& lo, __m256d& hi) {
        // bf16 -> fp32 is the bit pattern << 16 (exact, like to_float()).
        const __m256i u = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(v)));
        const __m256  f = _mm256_castsi256_ps(_mm256_slli_epi32(u, 16));
        lo = _mm256_cvtps_pd(_mm256_castps256_ps128(f));
        hi = _mm256_cvtps_pd(_mm256_extractf128_ps(f, 1));
    }
    static inline __m256d widen4(const bf16_t* v) {
        const __m128i u = _mm_cvtepu16_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(v)));
        return _mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(u, 16)));
    }
    static inline void widen8(const float* v, __m256d& lo, __m256d& hi) {
        lo = _mm256_cvtps_pd(_mm_loadu_ps(v));
        hi = _mm256_cvtps_pd(_mm_loadu_ps(v + 4));
    }
    static inline __m256d widen4(const float* v) { return _mm256_cvtps_pd(_mm_loadu_ps(v)); }
    static inline void widen8(const double* v, __m256d& lo, __m256d& hi) {
        lo = _mm256_loadu_pd(v);
        hi = _mm256_loadu_pd(v + 4);
    }
    static inline __m256d widen4(const double* v) { return _mm256_loadu_pd(v); }
    // Four doubles y[idx[0..3]] (32- or 64-bit node indices).
    static inline __m256d gather4(const double* y, const node_index* idx) {
        if constexpr (sizeof(node_index) == 4)
            return _mm256_i32gather_pd(y, _mm_loadu_si128(reinterpret_cast<const __m128i*>(idx)), 8);
        else
            return _mm256_i64gather_pd(y, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(idx)), 8);
    }
    static inline double hsum4(__m256d v) {
        const __m128d lo = _mm_add_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd(v, 1));
        return _mm_cvtsd_f64(_mm_add_sd(lo, _mm_unpackhi_pd(lo, lo)));
    }
    // The same sum as dot_thin, fat-level SIMD flavour (16-bit storage): 8
    // stored values per widen8; the y gather is a vector gather + vector FMA
    // (2 accumulators) when gather_simd_, else the 8 widened doubles go through
    // a stack buffer and scalar gathers feed a 4-way scalar FMA chain. Then a
    // 4-wide step and a scalar tail. Different summation order from dot_thin
    // (and between the two flavours): same accuracy, not bit-identical. A
    // template on the value type so it is only instantiated where kSimdDot
    // selects it (fp24_t has no widen8).
    template <class V>
    double dot_fat_simd(const V* __restrict vals, const node_index* __restrict idx,
                        edge_index p, edge_index end, const double* __restrict y) const {
        if (gather_simd_) {
            __m256d acc0 = _mm256_setzero_pd(), acc1 = _mm256_setzero_pd();
            for (; p + 8 <= end; p += 8) {
                __m256d h0, h1;
                widen8(vals + p, h0, h1);
                acc0 = _mm256_fmadd_pd(h0, gather4(y, idx + p),     acc0);
                acc1 = _mm256_fmadd_pd(h1, gather4(y, idx + p + 4), acc1);
            }
            if (p + 4 <= end) {
                acc0 = _mm256_fmadd_pd(widen4(vals + p), gather4(y, idx + p), acc0);
                p += 4;
            }
            double sum = hsum4(_mm256_add_pd(acc0, acc1));
            for (; p < end; ++p)
                sum += widen(vals[p]) * y[idx[p]];
            return sum;
        }
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

    // Forward row i (CSR row i, diagonal slot LAST): y_i = (x_i - sum_j L~_ij
    // y_j) / L~_ii. Fat: the `omp for` levels -- the SIMD kernel on 16-bit
    // storage, else the plain single-accumulator loop (instruction-identical
    // to the pre-fold fp32 kernel); thin: dot_thin.
    template <bool Fat>
    void fwd_row(node_index i, const double* x_in, double* y_out) const {
        const edge_index row_start = csr_row_ptr_[i];
        const edge_index row_end   = csr_row_ptr_[i + 1] - 1;
        double sum;
        if constexpr (Fat && kSimdDot) {
#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
            sum = dot_fat_simd(csr_vals_.data(), csr_col_idx_.data(), row_start, row_end, y_out);
#endif
        } else if constexpr (Fat) {
            sum = 0.0;
            for (edge_index p = row_start; p < row_end; ++p)
                sum += widen(csr_vals_[p]) * y_out[csr_col_idx_[p]];
        } else {
            sum = dot_thin(csr_vals_.data(), csr_col_idx_.data(), row_start, row_end, y_out);
        }
        y_out[i] = (x_in[i] - sum) / fwd_diag(i);
    }
    // Back column j (CSC column j, diagonal slot FIRST): z_j = (x_j [* r_j^2]
    // - sum_k L~_kj z_k) / L~_jj; the gather source is y_out (z) itself. Valid
    // in place (x_in == y_out: x_in[j] is read before y_out[j] is written).
    template <bool Fat>
    void bck_col(node_index j, const double* x_in, double* y_out) const {
        const edge_index col_start = csc_col_ptr_[j] + 1;
        const edge_index col_end   = csc_col_ptr_[j + 1];
        double sum;
        if constexpr (Fat && kSimdDot) {
#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
            sum = dot_fat_simd(csc_vals_.data(), csc_row_idx_.data(), col_start, col_end, y_out);
#endif
        } else if constexpr (Fat) {
            sum = 0.0;
            for (edge_index p = col_start; p < col_end; ++p)
                sum += widen(csc_vals_[p]) * y_out[csc_row_idx_[p]];
        } else {
            sum = dot_thin(csc_vals_.data(), csc_row_idx_.data(), col_start, col_end, y_out);
        }
        y_out[j] = (bck_rhs(j, x_in) - sum) / bck_diag(j);
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
private:

    // Sync-free back-solve dependency counters.
    // bck_unsolved_init_ holds the immutable initial counts (computed once in
    // setup); bck_unsolved_ is the live counter buffer reset at each solve.
    std::vector<int> bck_unsolved_init_;
    mutable std::vector<int> bck_unsolved_;
};

} // namespace apxchol
