#pragma once
#include "apxchol/types.h"
#include <Eigen/Sparse>
#include <algorithm>
#include <atomic>
#include <numeric>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace apxchol {

// Minimum level-set size before engaging OpenMP for SpTRSV.
// Below this threshold, the thread-dispatch overhead exceeds the
// computational benefit of parallelizing a single level's rows/cols.
inline constexpr index_t kSpTRSVOMPThreshold = 1024;

// Maximum number of (forward or backward) levels at which the
// classical level-set scheduler is competitive.  Beyond this many
// levels each contains very few rows and the per-level barrier
// dominates; the synchronization-free scheduler then wins.
inline constexpr int kSpTRSVLevelSetMaxLevels = 256;

/// OpenMP parallel sparse triangular solver with two interchangeable schedulers:
///
///   1. **Level-set scheduler** (Anderson & Saad 1989; Naumov 2011):
///      Pre-computes topological levels of the dependency DAG.  Each
///      level is a parallel-for, with an implicit barrier between
///      levels.  Best when the factor has few, fat levels (e.g. the
///      Blelloch rootset elimination yields ≈ 65 levels on 524k rows).
///
///   2. **Synchronization-free scheduler** (Liu, Smelyanskiy, Chow 2016;
///      Park et al. 2014; Su, Yang, Zhao 2020 "SyncFree"):
///      No barriers.  Each row carries an atomic counter of unresolved
///      dependencies; threads pick rows dynamically and busy-wait on
///      the counter, then notify dependents via atomic decrement.
///      Best when the factor has many short levels (≫ 256), where the
///      level-set barrier overhead dominates: e.g. block-greedy gives
///      ~5770 levels on the same matrix and the level-set scheduler
///      becomes barrier-bound.
///
/// Both schedulers reuse the same CSR (forward) / CSC (back) storage
/// and share the diagonal vector.  The default `forward_solve` and
/// `transpose_solve` entry points pick the scheduler at solve time
/// based on the level count detected during `setup`.
class omp_sptrsv {
public:
    omp_sptrsv() = default;

    /// Analyze L11 = L.topLeftCorner(m, m): build CSR, CSC, and level sets.
    void setup(const Eigen::SparseMatrix<double>& L, Eigen::Index m) {
        m_ = static_cast<index_t>(m);

        Eigen::SparseMatrix<double> L11 = L.topLeftCorner(m, m);
        L11.makeCompressed();

        // ── CSR of L11 (for forward solve) ──────────────────────
        csr_row_ptr_.assign(m_ + 1, 0);
        for (index_t j = 0; j < m_; ++j)
            for (Eigen::SparseMatrix<double>::InnerIterator it(L11, j); it; ++it)
                csr_row_ptr_[it.row() + 1]++;
        for (index_t i = 0; i < m_; ++i)
            csr_row_ptr_[i + 1] += csr_row_ptr_[i];

        index_t nnz = csr_row_ptr_[m_];
        csr_col_idx_.resize(nnz);
        csr_vals_.resize(nnz);
        {
            std::vector<index_t> pos(csr_row_ptr_.begin(), csr_row_ptr_.begin() + m_);
            for (index_t j = 0; j < m_; ++j)
                for (Eigen::SparseMatrix<double>::InnerIterator it(L11, j); it; ++it) {
                    index_t p = pos[it.row()]++;
                    csr_col_idx_[p] = static_cast<index_t>(j);
                    csr_vals_[p] = it.value();
                }
        }

        // Diagonal extraction from CSR (diagonal = last entry in each row).
        diag_.resize(m_);
        for (index_t i = 0; i < m_; ++i)
            diag_[i] = csr_vals_[csr_row_ptr_[i + 1] - 1];

        // ── CSC of L11 (for back solve) ─────────────────────────
        // Eigen already stores CSC; just copy the arrays.
        csc_col_ptr_.assign(L11.outerIndexPtr(), L11.outerIndexPtr() + m_ + 1);
        csc_row_idx_.assign(L11.innerIndexPtr(), L11.innerIndexPtr() + nnz);
        csc_vals_.assign(L11.valuePtr(), L11.valuePtr() + nnz);

        // ── Forward level sets (for L solve) ────────────────────
        // Row i depends on columns j < i where L(i,j) ≠ 0.
        {
            std::vector<int> depth(m_, 0);
            int max_depth = 0;
            for (index_t i = 0; i < m_; ++i) {
                int d = 0;
                for (index_t p = csr_row_ptr_[i]; p < csr_row_ptr_[i + 1] - 1; ++p)
                    d = std::max(d, depth[csr_col_idx_[p]] + 1);
                depth[i] = d;
                max_depth = std::max(max_depth, d);
            }
            fwd_levels_.resize(max_depth + 1);
            for (index_t i = 0; i < m_; ++i)
                fwd_levels_[depth[i]].push_back(i);
        }

        // ── Backward level sets (for L^T solve) ─────────────────
        // Column j: z[j] = (y[j] - Σ L(k,j)*z[k]) / L(j,j)  for k > j.
        // So column j depends on z[k] for k > j where L(k,j) ≠ 0.
        {
            std::vector<int> depth(m_, 0);
            int max_depth = 0;
            for (index_t j = m_ - 1; j >= 0; --j) {
                int d = 0;
                for (index_t p = csc_col_ptr_[j]; p < csc_col_ptr_[j + 1]; ++p) {
                    index_t k = csc_row_idx_[p];
                    if (k > j)
                        d = std::max(d, depth[k] + 1);
                }
                depth[j] = d;
                max_depth = std::max(max_depth, d);
            }
            bck_levels_.resize(max_depth + 1);
            for (index_t j = 0; j < m_; ++j)
                bck_levels_[depth[j]].push_back(j);
        }

        // ── Sync-free initial dependency counts ─────────────────
        // Forward: row i depends on (csr_row_ptr[i+1] - csr_row_ptr[i] - 1)
        //          off-diagonal columns j < i.
        // Back   : column j depends on (csc_col_ptr[j+1] - csc_col_ptr[j] - 1)
        //          off-diagonal rows k > j.
        // Pre-computed once and copied into the live counter buffers
        // at the start of each solve.
        fwd_unsolved_init_.resize(m_);
        bck_unsolved_init_.resize(m_);
        for (index_t i = 0; i < m_; ++i)
            fwd_unsolved_init_[i] =
                static_cast<int>(csr_row_ptr_[i + 1] - csr_row_ptr_[i] - 1);
        for (index_t j = 0; j < m_; ++j)
            bck_unsolved_init_[j] =
                static_cast<int>(csc_col_ptr_[j + 1] - csc_col_ptr_[j] - 1);
        fwd_unsolved_.resize(m_);
        bck_unsolved_.resize(m_);

        ready_ = true;
    }

    /// Forward solve: L * y = x.  Reads x[0..m-1], writes y[0..m-1].
    /// Default scheduler is the level-set variant: empirically beats sync-free
    /// across all level counts we tested (BG/luby/BK with thousands of levels
    /// and rootset with ~65 levels).  The sync-free variant remains available
    /// as `forward_solve_syncfree` for benchmarking.
    void forward_solve(const double* x_in, double* y_out) const {
        forward_solve_levelset(x_in, y_out);
    }

    /// Back solve: L^T * z = y.  See `forward_solve` note on scheduler choice.
    void transpose_solve(const double* x_in, double* y_out) const {
        transpose_solve_levelset(x_in, y_out);
    }

    // ── Level-set scheduler ──────────────────────────────────

    void forward_solve_levelset(const double* x_in, double* y_out) const {
        std::copy(x_in, x_in + m_, y_out);

        for (const auto& level : fwd_levels_) {
            const index_t level_sz = static_cast<index_t>(level.size());
            if (level_sz <= kSpTRSVOMPThreshold) {
                for (index_t k = 0; k < level_sz; ++k) {
                    index_t i = level[k];
                    if (k + 4 < level_sz) {
                        index_t ip = level[k + 1];
                        __builtin_prefetch(&csr_col_idx_[csr_row_ptr_[ip]]);
                        __builtin_prefetch(&csr_vals_[csr_row_ptr_[ip]]);
                    }
                    double sum = 0.0;
                    for (index_t p = csr_row_ptr_[i]; p < csr_row_ptr_[i + 1] - 1; ++p)
                        sum += csr_vals_[p] * y_out[csr_col_idx_[p]];
                    y_out[i] = (y_out[i] - sum) / diag_[i];
                }
            } else {
                #pragma omp parallel for schedule(static)
                for (index_t k = 0; k < level_sz; ++k) {
                    index_t i = level[k];
                    if (k + 4 < level_sz) {
                        index_t ip = level[k + 1];
                        __builtin_prefetch(&csr_col_idx_[csr_row_ptr_[ip]]);
                        __builtin_prefetch(&csr_vals_[csr_row_ptr_[ip]]);
                    }
                    double sum = 0.0;
                    for (index_t p = csr_row_ptr_[i]; p < csr_row_ptr_[i + 1] - 1; ++p)
                        sum += csr_vals_[p] * y_out[csr_col_idx_[p]];
                    y_out[i] = (y_out[i] - sum) / diag_[i];
                }
            }
        }
    }

    void transpose_solve_levelset(const double* x_in, double* y_out) const {
        std::copy(x_in, x_in + m_, y_out);

        // Process backward levels from depth 0 (no dependencies) upward.
        for (const auto& level : bck_levels_) {
            const index_t level_sz = static_cast<index_t>(level.size());
            // No `if` clause: the OMP parallel-region construct itself
            // costs several µs per level even when the `if` is false; with
            // thousands of small backward levels (typical for BG/luby/BK)
            // that overhead dominated solve time on 1 thread.  We use the
            // num_threads-clamp to fall back to serial for tiny levels
            // without instantiating a new parallel region.
            if (level_sz <= kSpTRSVOMPThreshold) {
                for (index_t k = 0; k < level_sz; ++k) {
                    index_t j = level[k];
                    // Prefetch the next column's metadata + values to hide
                    // memory latency on the consecutive walk.
                    if (k + 4 < level_sz) {
                        index_t jp = level[k + 4];
                        __builtin_prefetch(&csc_col_ptr_[jp]);
                        __builtin_prefetch(&csc_row_idx_[csc_col_ptr_[level[k + 1]]]);
                        __builtin_prefetch(&csc_vals_[csc_col_ptr_[level[k + 1]]]);
                    }
                    double sum = 0.0;
                    for (index_t p = csc_col_ptr_[j] + 1; p < csc_col_ptr_[j + 1]; ++p)
                        sum += csc_vals_[p] * y_out[csc_row_idx_[p]];
                    y_out[j] = (y_out[j] - sum) / diag_[j];
                }
            } else {
                #pragma omp parallel for schedule(static)
                for (index_t k = 0; k < level_sz; ++k) {
                    index_t j = level[k];
                    if (k + 4 < level_sz) {
                        index_t jp = level[k + 1];
                        __builtin_prefetch(&csc_row_idx_[csc_col_ptr_[jp]]);
                        __builtin_prefetch(&csc_vals_[csc_col_ptr_[jp]]);
                    }
                    double sum = 0.0;
                    for (index_t p = csc_col_ptr_[j] + 1; p < csc_col_ptr_[j + 1]; ++p)
                        sum += csc_vals_[p] * y_out[csc_row_idx_[p]];
                    y_out[j] = (y_out[j] - sum) / diag_[j];
                }
            }
        }
    }

    // ── Synchronization-free scheduler ───────────────────────
    //
    // Per-row dependency counters (fwd_unsolved_, bck_unsolved_) start
    // at the off-diagonal nonzero count and decrement as producers
    // finish.  Threads pick rows dynamically; each row busy-waits on
    // its counter to reach zero, computes, then publishes the result
    // by atomically decrementing each dependent's counter.  Memory
    // ordering: release on the publish, acquire on the spin-wait read,
    // which establishes happens-before between the producer's write to
    // y_out[i] and the consumer's load of y_out[i].

    void forward_solve_syncfree(const double* x_in, double* y_out) const {
        std::copy(x_in, x_in + m_, y_out);

        // Reset live counters in parallel (just an O(m) memcpy).
        #pragma omp parallel for schedule(static)
        for (index_t i = 0; i < m_; ++i)
            fwd_unsolved_[i] = fwd_unsolved_init_[i];

        const index_t m = m_;
        int* unsolved = fwd_unsolved_.data();

        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic, 64) nowait
            for (index_t i = 0; i < m; ++i) {
                // Spin until all dependencies of row i are satisfied.
                while (__atomic_load_n(&unsolved[i], __ATOMIC_ACQUIRE) != 0) {
                    #if defined(__x86_64__) || defined(__i386__)
                    __builtin_ia32_pause();
                    #endif
                }

                double sum = 0.0;
                for (index_t p = csr_row_ptr_[i]; p < csr_row_ptr_[i + 1] - 1; ++p)
                    sum += csr_vals_[p] * y_out[csr_col_idx_[p]];
                y_out[i] = (y_out[i] - sum) / diag_[i];

                // Notify dependents: rows k > i with L(k, i) ≠ 0.
                // CSC column i lists those, with diagonal at csc_col_ptr_[i].
                for (index_t p = csc_col_ptr_[i] + 1; p < csc_col_ptr_[i + 1]; ++p) {
                    index_t k = csc_row_idx_[p];
                    __atomic_sub_fetch(&unsolved[k], 1, __ATOMIC_RELEASE);
                }
            }
        }
    }

    void transpose_solve_syncfree(const double* x_in, double* y_out) const {
        std::copy(x_in, x_in + m_, y_out);

        #pragma omp parallel for schedule(static)
        for (index_t j = 0; j < m_; ++j)
            bck_unsolved_[j] = bck_unsolved_init_[j];

        const index_t m = m_;
        int* unsolved = bck_unsolved_.data();

        // Process columns in reverse: column j depends on z[k] for k > j,
        // so larger indices become ready first.  Reverse iteration in
        // the dynamic-for keeps cache locality similar to the forward case
        // and lets the index-(m-1) row (no dependencies) start immediately.
        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic, 64) nowait
            for (index_t j_r = 0; j_r < m; ++j_r) {
                index_t j = m - 1 - j_r;
                while (__atomic_load_n(&unsolved[j], __ATOMIC_ACQUIRE) != 0) {
                    #if defined(__x86_64__) || defined(__i386__)
                    __builtin_ia32_pause();
                    #endif
                }

                double sum = 0.0;
                for (index_t p = csc_col_ptr_[j] + 1; p < csc_col_ptr_[j + 1]; ++p)
                    sum += csc_vals_[p] * y_out[csc_row_idx_[p]];
                y_out[j] = (y_out[j] - sum) / diag_[j];

                // Notify dependents: columns i < j with L(j, i) ≠ 0.
                // CSR row j lists exactly those, with diagonal last.
                for (index_t p = csr_row_ptr_[j]; p < csr_row_ptr_[j + 1] - 1; ++p) {
                    index_t i = csr_col_idx_[p];
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
            for (index_t v : lvl) {
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

private:
    index_t m_ = 0;
    bool ready_ = false;

    // CSR of L11 (forward solve).
    std::vector<index_t>    csr_row_ptr_;
    std::vector<index_t>    csr_col_idx_;
    std::vector<double> csr_vals_;
    std::vector<double> diag_;

    // CSC of L11 (back solve).
    std::vector<index_t>    csc_col_ptr_;
    std::vector<index_t>    csc_row_idx_;
    std::vector<double> csc_vals_;

    // Level sets.
    std::vector<std::vector<index_t>> fwd_levels_;
    std::vector<std::vector<index_t>> bck_levels_;

    // Sync-free dependency counters.
    // *_init_ holds the immutable initial counts (computed once in setup);
    // *_unsolved_ is the live counter buffer reset at the start of each solve.
    std::vector<int> fwd_unsolved_init_;
    std::vector<int> bck_unsolved_init_;
    mutable std::vector<int> fwd_unsolved_;
    mutable std::vector<int> bck_unsolved_;
};

} // namespace apxchol
