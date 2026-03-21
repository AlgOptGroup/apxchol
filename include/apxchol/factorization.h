#pragma once
#include <Eigen/Sparse>

namespace apxchol {

/// Result of the approximate Cholesky factorization.
///
/// The factorization produces a sparse lower-triangular matrix L and a
/// permutation P such that P^T L L^T P ≈ original Laplacian.
/// Used as a preconditioner for PCG.
struct factorization {
    Eigen::SparseMatrix<double> L;                          // lower-triangular factor
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> perm;
    int n       = 0;
    int nnz     = 0;     // nonzeros in L
    double fillin = 0.0; // nnz(L) / nnz(input)
};

struct factor_options {
    unsigned seed = 42;
    // Future: parallelism settings
};

/// Compute approximate Cholesky factorization of a graph Laplacian.
///
/// Implements the Kyng-Sachdeva (2016) algorithm:
///   1. Find a large independent set of low-degree vertices
///   2. Eliminate them via random clique sampling
///   3. Repeat on the Schur complement
///   4. Return the sparse triangular factor
///
/// Input: Laplacian matrix L (n×n, symmetric, zero row-sums).
/// Output: factorization struct with L_factor and permutation.
factorization factorize(const Eigen::SparseMatrix<double>& L,
                        const factor_options& opts = {});

} // namespace apxchol
