#pragma once
#include "apxchol/solver/solve.h"

namespace apxchol::detail {
// Resolve configuration before constructing either concrete owner.
solve_backend select_solve_backend(const solve_options& opts);
} // namespace apxchol::detail
