#pragma once
/// Configurable options for the approximate Cholesky factorization.

#include <cstddef>

namespace apxchol {

/// Enumerate available IS selection strategies for runtime dispatch.
enum class is_strategy { block_greedy, luby, baumann_kyng, rootset, hybrid };

/// Enumerate available elimination (clique sampling) strategies for runtime dispatch.
enum class elimination_strategy { tree, star, clique };

struct factor_options {
    unsigned seed = 42;
    double degree_multiplier = 2.0;  // IS degree threshold = multiplier × avg_degree
    double min_is_fraction = 0.05;   // fall back to sequential when IS < 5% of active
    size_t omp_threshold = 2000;     // min active/IS vertices before engaging OpenMP
    is_strategy is_select = is_strategy::block_greedy;  // IS selection strategy (runtime dispatch)
    elimination_strategy elim = elimination_strategy::tree;  // Elimination strategy (runtime dispatch)
    double bk_sampling_constant = 0.3;  // BK: sample prob = 1/(c·d_max); lower c → larger IS
};

} // namespace apxchol
