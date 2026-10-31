#pragma once
// Eigen fixtures in the library's own CSC storage, for tests of internals.
#include "apxchol/detail/eigen_boundary.h"
#include "apxchol/operator_class.h"

#include <Eigen/Sparse>
#include <cstdint>

namespace apxchol::test {

/// Owned copy of a square Eigen matrix (compressed; explicit entries kept).
inline detail::csc_matrix<double> to_csc(const Eigen::SparseMatrix<double>& A) {
    const detail::eigen_csc input(A);
    return detail::csc_cast<double>(input.ref());
}

/// M-matrix lumping applied to an Eigen matrix in place; returns the entries
/// removed. Leaves A unchanged when nothing is positive off the diagonal.
inline Eigen::Index lump_in_place(Eigen::SparseMatrix<double>& A) {
    if (A.rows() != A.cols() || A.rows() == 0) return 0;
    A.makeCompressed();
    std::int64_t removed = 0;
    const detail::eigen_csc input(A);
    const auto lumped = detail::lump_positive_offdiagonals(input.ref(), removed);
    if (removed != 0) A = detail::to_eigen(lumped.ref());
    return removed;
}

} // namespace apxchol::test
