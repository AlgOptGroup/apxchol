#pragma once
#include "apxchol/solver/solve.h"
#include "apxchol/detail/csc_matrix.h"

namespace apxchol::detail {
// Resolve configuration and optional input eligibility before constructing an owner.
solve_backend select_solve_backend(const solve_options& opts,
                                  const csc_ref<double>* A = nullptr);
} // namespace apxchol::detail
