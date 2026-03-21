#pragma once
#include "factorization.h"
#include <Eigen/Core>
#include <Eigen/Sparse>

namespace apxchol {

/// Eigen-compatible preconditioner wrapping an approximate Cholesky factorization.
///
/// Usage with Eigen's PCG:
///   auto F = apxchol::factorize(L);
///   apxchol::preconditioner M(F);
///   Eigen::ConjugateGradient<SpMat, Eigen::Lower|Eigen::Upper, apxchol::preconditioner> cg;
///   cg.preconditioner() = M;
///   cg.compute(L);
///   x = cg.solve(b);
class preconditioner {
public:
    using Scalar      = double;
    using RealScalar  = double;
    using StorageIndex = int;

    preconditioner() = default;
    explicit preconditioner(const factorization& F);

    template <typename MatrixType>
    preconditioner& analyzePattern(const MatrixType&) { return *this; }

    template <typename MatrixType>
    preconditioner& factorize(const MatrixType&) { return *this; }

    template <typename MatrixType>
    preconditioner& compute(const MatrixType& A) {
        n_ = A.rows();
        return *this;
    }

    Eigen::Index rows() const { return n_; }
    Eigen::Index cols() const { return n_; }
    Eigen::ComputationInfo info() const { return info_; }

    /// Apply M^{-1} to rhs:  P^T * L^{-T} * L^{-1} * P * rhs, centered.
    template <typename Rhs>
    Eigen::VectorXd solve(const Rhs& rhs) const {
        const int m = n_ - 1;
        Eigen::VectorXd b = rhs;
        b.array() -= b.mean();

        // Permute
        Eigen::VectorXd y = F_->perm * b;

        // Forward solve: L * z = y  (only first m rows)
        auto L11 = F_->L.topLeftCorner(m, m);
        Eigen::VectorXd z = L11.template triangularView<Eigen::Lower>().solve(y.head(m));

        // Back solve: L^T * w = z
        Eigen::VectorXd w(n_);
        w.head(m) = L11.transpose().template triangularView<Eigen::Upper>().solve(z);
        w(m) = 0.0;

        // Unpermute
        Eigen::VectorXd x = F_->perm.transpose() * w;
        x.array() -= x.mean();
        return x;
    }

private:
    const struct factorization* F_ = nullptr;
    Eigen::Index n_ = 0;
    Eigen::ComputationInfo info_ = Eigen::Success;
};

} // namespace apxchol

// Eigen traits (required for PCG integration)
namespace Eigen { namespace internal {
    template<>
    struct traits<apxchol::preconditioner>
        : traits<Eigen::SparseMatrix<double>> {};
}}
