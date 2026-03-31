#pragma once
/// Configurable options for the approximate Cholesky factorization.

#include <cstddef>

namespace apxchol {

/// Enumerate available IS selection strategies for runtime dispatch.
enum class is_strategy { block_greedy, luby, baumann_kyng };

struct factor_options {
    unsigned seed = 42;
    double degree_multiplier = 2.0;  // IS degree threshold = multiplier × avg_degree
    double min_is_fraction = 0.05;   // fall back to sequential when IS < 5% of active
    size_t omp_threshold = 2000;     // min active/IS vertices before engaging OpenMP
    is_strategy is_select = is_strategy::block_greedy;  // IS selection strategy (runtime dispatch)
};

} // namespace apxchol
