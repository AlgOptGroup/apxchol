#pragma once
#include "apxchol/checkpoint.h"
#include "apxchol/solver/factorization.h"
#include <Eigen/Core>
#include <Eigen/Sparse>

#if defined(APXCHOL_USE_CUDA)
#include "apxchol/solver/sptrsv/cuda.h"
#else
#include "apxchol/solver/sptrsv/omp.h"
#endif

namespace apxchol {

/// Eigen-compatible preconditioner using approximate Cholesky factorization.
///
/// Inherits SparseSolverBase for the standard solve() / _solve_impl() pattern.
/// Use with Eigen's PCG:
///   Eigen::ConjugateGradient<SpMat, Eigen::Lower, apxchol::apx_cholesky> cg;
///   cg.compute(L);   // triggers apx_cholesky::compute(L) internally
///   x = cg.solve(b);
///
/// Triangular solve backend selected at compile time:
///   - Default: OpenMP level-set parallel SpTRSV
///   - APXCHOL_USE_CUDA: cuSPARSE GPU-accelerated SpSV
class apx_cholesky : public Eigen::SparseSolverBase<apx_cholesky> {
    using Base = Eigen::SparseSolverBase<apx_cholesky>;

protected:
    using Base::m_isInitialized;

public:
    using Scalar      = double;
    using RealScalar  = double;
    using StorageIndex = index_t;
    enum {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic
    };

    apx_cholesky() = default;

    const factorization& factor() const { return F_; }
    const factor_options& options() const { return opts_; }
    void set_options(const factor_options& opts) { opts_ = opts; }
    void set_storage(graph_storage s) { storage_ = s; }
    void set_checkpoint(checkpoint* cp) { cp_ = cp; }

    /// No-op: approximate Cholesky has no separate symbolic analysis phase.
    apx_cholesky& analyzePattern(const auto&) {
        m_analysisIsOk = true;
        m_isInitialized = true;
        m_info = Eigen::Success;
        return *this;
    }

    /// Compute the approximate Cholesky factorization of A.
    apx_cholesky& factorize(const auto& A) {
        eigen_assert(m_analysisIsOk && "analyzePattern() should be called first");
        n_ = A.rows();
        F_ = apxchol::factorize(A, storage_, opts_, cp_);
        scratch_.resize(n_);

        // For Laplacians (rank n-1) the last row/col is unused.
        // For SDDM (full rank) we use the complete n×n factor.
        Eigen::Index factor_dim = F_.sddm ? n_ : n_ - 1;
        trsv_.setup(F_.L, factor_dim);

        m_factorizationIsOk = true;
        m_info = Eigen::Success;
        return *this;
    }

    /// Shortcut for analyzePattern() + factorize().
    apx_cholesky& compute(const auto& A) {
        analyzePattern(A);
        factorize(A);
        return *this;
    }

    Eigen::Index rows() const { return n_; }
    Eigen::Index cols() const { return n_; }
    Eigen::ComputationInfo info() const { return m_info; }

    /// Apply M^{-1} to rhs via approximate Cholesky.
    ///
    /// Laplacian path (rank n-1): center → P → L^{-1} → L^{-T} → P^{-1} → center.
    /// SDDM path     (rank n):   P → L^{-1} → L^{-T} → P^{-1}.
    void _solve_impl(const auto& b, auto& x) const {
        eigen_assert(m_factorizationIsOk && "factorize() should be called first");

        if (cp_) { cp_->descend("solve"); cp_->tick(); }

        if (F_.sddm) {
            // ── SDDM path: full-rank factor, no centering ──
            x = F_.perm * b;
            if (cp_) (*cp_)("permute");

            trsv_.forward_solve(x.data(), scratch_.data());
            if (cp_) (*cp_)("forward");
            trsv_.transpose_solve(scratch_.data(), x.data());
            if (cp_) (*cp_)("back");

            scratch_ = x;
            x = F_.perm.inverse() * scratch_;
            if (cp_) (*cp_)("unpermute");
        } else {
            // ── Laplacian path: rank-(n-1) factor with centering ──
            const Eigen::Index m = n_ - 1;

            scratch_.array() = b.array() - b.mean();
            x = F_.perm * scratch_;
            if (cp_) (*cp_)("permute");

#if defined(APXCHOL_USE_CUDA)
            trsv_.solve_LLt(x.data(), scratch_.data());
            x.head(m) = Eigen::Map<const Eigen::VectorXd>(scratch_.data(), m);
#else
            trsv_.forward_solve(x.data(), scratch_.data());
            if (cp_) (*cp_)("forward");
            trsv_.transpose_solve(scratch_.data(), x.data());
#endif
            if (cp_) (*cp_)("back");

            x(m) = 0.0;

            scratch_ = x;
            x = F_.perm.inverse() * scratch_;
            x.array() -= x.mean();
            if (cp_) (*cp_)("unpermute");
        }

        if (cp_) cp_->ascend();
    }

private:
    struct factorization F_;
    factor_options opts_;
    graph_storage storage_ = graph_storage::forward_star;
    checkpoint* cp_ = nullptr;
    mutable Eigen::VectorXd scratch_;
    Eigen::Index n_ = 0;
    Eigen::ComputationInfo m_info = Eigen::Success;
    bool m_analysisIsOk = false;
    bool m_factorizationIsOk = false;

#if defined(APXCHOL_USE_CUDA)
    mutable cuda_sptrsv trsv_;
#else
    omp_sptrsv trsv_;
#endif
};

} // namespace apxchol
