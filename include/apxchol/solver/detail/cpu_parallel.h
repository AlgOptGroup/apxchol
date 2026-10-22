#pragma once
#include <algorithm>
#include <cstddef>
#include <omp.h>

namespace apxchol::detail {

#ifdef __APPLE__
inline constexpr std::size_t cpu_solve_grain = 32 * 1024;
#else
inline constexpr std::size_t cpu_solve_grain = 4 * 1024;
#endif

// Grow the solve team with the available work instead of waking every worker
// as soon as a small problem crosses the serial threshold.
inline int cpu_solve_threads(std::size_t work,
                             std::size_t grain = cpu_solve_grain) noexcept {
    if (work <= grain) return 1;
    const auto workers = work / grain + (work % grain != 0);
    return static_cast<int>(std::min<std::size_t>(
        workers, omp_get_max_threads()));
}

inline int cpu_spmv_threads(std::size_t rows, std::size_t nnz) noexcept {
#ifndef __APPLE__
    const int workers = cpu_solve_threads(rows);
    if (workers > 1) return workers;
#else
    (void)rows;
#endif
    return cpu_solve_threads(nnz, 32 * 1024);
}

} // namespace apxchol::detail
