#pragma once

/// @file laplacian_preconditioner.h
/// CRTP base for Laplacian preconditioners compatible with Eigen's PCG.
///
/// Usage:
///   class MyPreconditioner : public laplacian_preconditioner<MyPreconditioner> {
///   public:
///     using laplacian_preconditioner::laplacian_preconditioner;
///     // Required: apply preconditioner M^{-1} to rhs
///     Eigen::VectorXd apply(const Eigen::VectorXd& rhs) const { ... }
///   };
///
///   // Then use with Eigen's PCG:
///   Eigen::ConjugateGradient<SpMat, Eigen::Lower|Eigen::Upper, MyPreconditioner> cg;
///   cg.compute(L);
///   x = cg.solve(b);

#include <Eigen/Core>
#include <Eigen/Sparse>

template <typename Derived>
class laplacian_preconditioner {
public:
    using Scalar = double;
    using RealScalar = double;
    using StorageIndex = int;

    laplacian_preconditioner() = default;

    template <typename MatrixType>
    laplacian_preconditioner& analyzePattern(const MatrixType&) { return *this; }

    template <typename MatrixType>
    laplacian_preconditioner& factorize(const MatrixType&) { return *this; }

    template <typename MatrixType>
    laplacian_preconditioner& compute(const MatrixType& A) {
        n_ = A.rows();
        info_ = Eigen::Success;
        return *this;
    }

    Eigen::Index rows() const { return n_; }
    Eigen::Index cols() const { return n_; }
    Eigen::ComputationInfo info() const { return info_; }

    /// Solve M^{-1} * rhs, centering both input and output.
    template <typename Rhs>
    Eigen::VectorXd solve(const Rhs& rhs) const {
        Eigen::VectorXd b = rhs;
        b.array() -= b.mean();
        Eigen::VectorXd x = static_cast<const Derived*>(this)->apply(b);
        x.array() -= x.mean();
        return x;
    }

protected:
    Eigen::Index n_ = 0;
    Eigen::ComputationInfo info_ = Eigen::Success;
};

// Eigen traits specialization (required for PCG integration)
namespace Eigen { namespace internal {
    template <typename Derived>
    struct traits<laplacian_preconditioner<Derived>>
        : traits<Eigen::SparseMatrix<double>> {};
}}
