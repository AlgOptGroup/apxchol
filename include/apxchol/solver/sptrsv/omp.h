#pragma once
#include "apxchol/types.h"
#include <Eigen/Sparse>
#include <algorithm>
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

/// OpenMP level-set parallel sparse triangular solver.
///
/// Builds two representations of L11:
///   - CSR of L with forward level sets → parallel forward solve L*y=x (gather)
///   - CSC of L with backward level sets → parallel back solve L^T*z=y (gather)
///
/// Both solves are gather-based (no atomics).  For our approximate Cholesky
/// factors, level 0 typically contains ~50% of all rows.
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

        ready_ = true;
    }

    /// Forward solve: L * y = x.  Reads x[0..m-1], writes y[0..m-1].
    /// Uses CSR of L; gather-based (no atomics).
    void forward_solve(const double* x_in, double* y_out) const {
        std::copy(x_in, x_in + m_, y_out);

        for (const auto& level : fwd_levels_) {
            const index_t level_sz = static_cast<index_t>(level.size());
            #pragma omp parallel for schedule(static) if(level_sz > kSpTRSVOMPThreshold)
            for (index_t k = 0; k < level_sz; ++k) {
                index_t i = level[k];
                double sum = 0.0;
                for (index_t p = csr_row_ptr_[i]; p < csr_row_ptr_[i + 1] - 1; ++p)
                    sum += csr_vals_[p] * y_out[csr_col_idx_[p]];
                y_out[i] = (y_out[i] - sum) / diag_[i];
            }
        }
    }

    /// Back solve: L^T * z = y.  Reads x[0..m-1], writes y[0..m-1].
    /// Uses CSC of L; gather-based (no atomics).
    void transpose_solve(const double* x_in, double* y_out) const {
        std::copy(x_in, x_in + m_, y_out);

        // Process backward levels from depth 0 (no dependencies) upward.
        for (const auto& level : bck_levels_) {
            const index_t level_sz = static_cast<index_t>(level.size());
            #pragma omp parallel for schedule(static) if(level_sz > kSpTRSVOMPThreshold)
            for (index_t k = 0; k < level_sz; ++k) {
                index_t j = level[k];
                double sum = 0.0;
                // In L's CSC, column j has diagonal at index csc_col_ptr_[j]
                // (row j), followed by off-diagonal rows k > j.
                // Skip the diagonal entry and gather from k > j only.
                for (index_t p = csc_col_ptr_[j] + 1; p < csc_col_ptr_[j + 1]; ++p)
                    sum += csc_vals_[p] * y_out[csc_row_idx_[p]];
                y_out[j] = (y_out[j] - sum) / diag_[j];
            }
        }
    }

    int num_fwd_levels() const { return static_cast<int>(fwd_levels_.size()); }
    int num_bck_levels() const { return static_cast<int>(bck_levels_.size()); }
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
};

} // namespace apxchol
