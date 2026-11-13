#pragma once
#include "apxchol/solver/solve.h"
#include "apxchol/detail/csc_matrix.h"

namespace apxchol::detail {
// Resolve configuration and optional input eligibility before constructing an owner.
solve_backend select_solve_backend(const solve_options& opts,
                                  const csc_ref<double>* A = nullptr);

// For entry points that factor and solve in one call (Python solve, Octave
// apxchol_solve, CLI --rhs): choose the clique sampler from the number of
// right-hand-side columns the factor will serve. Trace-cycle from 16 columns
// on CPU, 64 on CUDA and 8 on Metal, where its setup plus batched solve beat
// GKS on most measured inputs. The backend is resolved
// without the input, so very small automatic inputs may still run on CPU.
// Callers here cannot set the sampler themselves, so nothing is overridden.
void choose_sampler(solve_options& opts, std::int64_t columns);
} // namespace apxchol::detail
