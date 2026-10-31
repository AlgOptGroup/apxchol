#pragma once
#include "apxchol/solver/solve.h"
#include "apxchol/detail/csc_matrix.h"
#include "apxchol/detail/eigen_boundary.h"

namespace apxchol::detail {
// Resolve configuration and optional input eligibility before constructing an owner.
solve_backend select_solve_backend(const solve_options& opts,
                                  const csc_ref<double>* A = nullptr);

// The same for Eigen input (tests and the benchmark); a matrix that is not
// square is never eligible for a device route.
inline solve_backend select_solve_backend(const solve_options& opts,
                                          const Eigen::SparseMatrix<double>* A) {
    if (!A) return select_solve_backend(opts);
    const eigen_csc input(*A);
    const csc_ref<double> empty{};
    return select_solve_backend(opts, input.square() ? &input.ref() : &empty);
}
} // namespace apxchol::detail
