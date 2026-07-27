#pragma once
#include "apxchol/checkpoint.h"
#include "apxchol/solver/factorization.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <stdexcept>
#include <utility>

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
    // Eigen-facing index type for the SolverBase interface. Kept signed `int`
    // (Eigen requires a signed StorageIndex); independent of apxchol's internal
    // unsigned node_index/edge_index, which the factor now owns directly.
    using StorageIndex = int;
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
    /// Keep the factor's row/value arrays alive after SpTRSV setup (they are
    /// otherwise freed to save one factor-sized copy).  Needed by callers that
    /// want to read factor().L afterwards, e.g. the Python binding's L/D/P
    /// export, or to pass the factor on to set_factor() elsewhere.  Must be
    /// set before factorize()/compute()/set_factor().
    void set_keep_factor(bool keep) { keep_factor_ = keep; }

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
        install_factor();
        return *this;
    }

    /// Install an externally computed factorization (e.g. from a factorize()
    /// call with a custom eliminator or partitioner) and run the SpTRSV setup.
    /// After this the preconditioner is ready for _solve_impl / Eigen CG.
    /// The factorization must still own its row/value arrays — a factor read
    /// back from a default-configured solver has had them released (see
    /// set_keep_factor); pass a freshly computed factorization instead.
    apx_cholesky& set_factor(factorization F) {
        if (F.L.nonZeros() > 0 && F.L.vals_.empty())
            throw std::invalid_argument(
                "apx_cholesky::set_factor: factorization values were released; "
                "pass a freshly computed factorization");
        n_ = static_cast<Eigen::Index>(F.L.rows());
        F_ = std::move(F);
        m_analysisIsOk = true;
        m_isInitialized = true;
        install_factor();
        return *this;
    }

private:
    /// Shared tail of factorize()/set_factor(): SpTRSV setup from F_.
    void install_factor() {
        scratch_.resize(n_);

        // For Laplacians (rank n-1) the last row/col is unused.
        // For SDDM (full rank) we use the complete n×n factor.
        const node_index factor_dim = static_cast<node_index>(F_.sddm ? n_ : n_ - 1);
        // sptrsv setup (level sets / cuSPARSE analysis) is a real ~900ms
        // cost on IPM-scale matrices. Track it as part of "setup" so the
        // bench's setup_time / solve_time split is honest.
        if (cp_) { cp_->descend("setup"); cp_->tick(); }
#if !defined(APXCHOL_USE_CUDA)
        // Round-as-level: hand the SpTRSV the per-round IS-column boundaries so it
        // builds level sets straight off the elimination rounds (same-round IS
        // columns are mutually independent -> one level) instead of two O(nnz)
        // topological depth scans. Each region is one factor column placed in round
        // order; trailing residual-peel columns get sequential levels inside the
        // SpTRSV. The backend uses these by default (round-as-level is a -6..-18%
        // solve win); APXCHOL_ROUND_LEVELS=0 forces the topological scan instead.
        if (!F_.rounds.empty()) {
            std::vector<node_index> bounds;
            bounds.reserve(F_.rounds.size() + 1);
            bounds.push_back(0);
            node_index acc = 0;
            for (const auto& r : F_.rounds) {
                acc += static_cast<node_index>(r.is_size);
                if (acc > factor_dim) acc = static_cast<node_index>(factor_dim);
                bounds.push_back(acc);
            }
            trsv_.set_round_bounds(std::move(bounds));
        }
#endif
        trsv_.setup(F_.L, factor_dim);
        if (cp_) { (*cp_)("sptrsv_setup"); cp_->ascend(); }

        // The SpTRSV (and the CUDA path) have copied the factor into their own
        // CSR/CSC; the PCG loop only uses trsv_, never F_.L. Free the factor's
        // big row/value arrays -> one fewer full copy of the factor held during
        // the solve (~nnz*12 B; e.g. ~1.4 GB on com-LiveJournal). nonZeros() (the
        // fill stat) still works via the retained column pointers.
        // set_keep_factor(true) skips the free for callers that export L.
        if (!keep_factor_)
            F_.L.release_values();

        m_factorizationIsOk = true;
        m_info = Eigen::Success;
    }

public:

    /// Shortcut for analyzePattern() + factorize().
    apx_cholesky& compute(const auto& A) {
        analyzePattern(A);
        factorize(A);
        return *this;
    }

    Eigen::Index rows() const { return n_; }
    Eigen::Index cols() const { return n_; }
    Eigen::ComputationInfo info() const { return m_info; }

#if defined(APXCHOL_USE_CUDA)
    /// Device-resident SpTRSV state (cuda_pcg uses solve_LLt_dev directly).
    cuda_sptrsv& trsv() { return trsv_; }
    const cuda_sptrsv& trsv() const { return trsv_; }
#else
    omp_sptrsv& trsv() { return trsv_; }
    const omp_sptrsv& trsv() const { return trsv_; }
#endif

    /// Apply M^{-1} to rhs via approximate Cholesky.
    ///
    /// Laplacian path (rank n-1): center → P → L^{-1} → L^{-T} → P^{-1} → center.
    /// SDDM path     (rank n):   P → L^{-1} → L^{-T} → P^{-1}.
    void _solve_impl(const auto& b, auto& x) const {
        eigen_assert(m_factorizationIsOk && "factorize() should be called first");

        if (cp_) { cp_->descend("solve"); cp_->tick(); }

        // Permutation as explicit gather/scatter (replaces Eigen::PermutationMatrix
        // products). perm[v] = new position of original vertex v.
        //   to permuted   : dst[perm[v]] = src[v]   (scatter)
        //   from permuted : dst[v] = src[perm[v]]   (gather)
        // Both are bijective writes -> trivially parallel (Eigen's perm product
        // was serial; this is a free parallelism win on million-row systems).
        const node_index* P = F_.perm.data();
        const Eigen::Index n = n_;
        auto permute_to = [&](const auto& src, auto& dst) {
            #pragma omp parallel for schedule(static)
            for (Eigen::Index v = 0; v < n; ++v) dst(P[v]) = src(v);
        };
        auto permute_from = [&](const auto& src, auto& dst) {
            #pragma omp parallel for schedule(static)
            for (Eigen::Index v = 0; v < n; ++v) dst(v) = src(P[v]);
        };

        if (F_.sddm) {
            // ── SDDM path: full-rank factor, no centering ──
            permute_to(b, x);
            if (cp_) (*cp_)("permute");

#if defined(APXCHOL_USE_CUDA)
            trsv_.solve_LLt(x.data(), scratch_.data());
            x = scratch_;
#else
            trsv_.forward_solve(x.data(), scratch_.data());
            if (cp_) (*cp_)("forward");
            trsv_.transpose_solve(scratch_.data(), x.data());
#endif
            if (cp_) (*cp_)("back");

            scratch_ = x;
            permute_from(scratch_, x);
            if (cp_) (*cp_)("unpermute");
        } else {
            // ── Laplacian path: rank-(n-1) factor with centering ──
            const Eigen::Index m = n_ - 1;

            scratch_.array() = b.array() - b.mean();
            permute_to(scratch_, x);
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
            permute_from(scratch_, x);
            x.array() -= x.mean();
            if (cp_) (*cp_)("unpermute");
        }

        if (cp_) cp_->ascend();
    }

private:
    struct factorization F_;
    factor_options opts_;
    graph_storage storage_ = graph_storage::vec_pool;
    checkpoint* cp_ = nullptr;
    bool keep_factor_ = false;
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
