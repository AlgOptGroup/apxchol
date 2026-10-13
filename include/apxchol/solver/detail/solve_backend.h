#pragma once
#include "apxchol/solver/solve.h"

namespace apxchol::detail {
// Resolve configuration and optional input eligibility before constructing an owner.
solve_backend select_solve_backend(const solve_options& opts,
                                  const Eigen::SparseMatrix<double>* A = nullptr);
} // namespace apxchol::detail
