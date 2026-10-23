#pragma once
#include <algorithm>
#include <cstddef>
#include <limits>
#include <cstdint>
#define APXCHOL_GPU_CODE(...) __VA_ARGS__
#define APXCHOL_GPU_FUNCTION inline
#define APXCHOL_GPU_UINT64 std::uint64_t
#include "residual_sparsify.inc"
#undef APXCHOL_GPU_UINT64
#undef APXCHOL_GPU_FUNCTION
#undef APXCHOL_GPU_CODE
namespace apxchol::detail {
// The IS-yield bailout measures the selector against the vertices it was
// actually allowed to choose, not against all active vertices.  With the
// default degree_quantile=0.2, using |active| as the denominator silently made
// min_is_fraction=0.05 mean "select at least 25% of the candidate pool".
//
// Near the partitioner's handoff threshold there is no profitable fallback:
// BK stops at the threshold and a bail just above it sends almost the whole
// threshold-sized residual to the singleton peel. Protect twice the handoff
// size so integer/seed noise cannot turn a productive main selector into that
// dependency-level cliff. selected==0 remains the unconditional escape from a
// stuck custom selector.
inline bool is_yield_too_small(size_t selected, size_t candidates,
                               size_t active, double min_fraction,
                               size_t residual_handoff_threshold) {
    if (selected == 0) return true;
    if (residual_handoff_threshold != std::numeric_limits<size_t>::max()) {
        const size_t protected_tail = residual_handoff_threshold >
                std::numeric_limits<size_t>::max() / 2
            ? std::numeric_limits<size_t>::max()
            : 2 * residual_handoff_threshold;
        if (active <= protected_tail) return false;
    }
    return selected < candidates * min_fraction;
}

inline bool selection_should_handoff(
        size_t selected, size_t candidates, size_t active, double min_fraction,
        size_t residual_handoff_threshold, size_t omp_threshold) {
    if (selected == 0) return true;
    if (omp_threshold != 0 && selected >= omp_threshold) return false;
    return is_yield_too_small(selected, candidates, active,
                              min_fraction, residual_handoff_threshold);
}

// The relative yield at which the main selector stops should reflect both the
// size and the density of the residual. BG/priority-greedy rescan it; BK samples
// bounded edge work after an O(active) hash pass. Keep min_is_fraction as the
// sparse/small-residual base and preserve zero as "disable yield bailout".
//
// Adapt only while at least four handoff-sized chunks remain, so BK has enough
// runway to amortize taking over and small residuals do not fall off a level
// cliff. A residual whose average degree is itself at least the final-tail size
// triples the base. Sparse residuals keep the user's base: switching them early
// creates thousands of thin BK levels and can make the solve dominate the setup
// saving. The 0.15 ceiling bounds the quality trade even if a caller chose a
// larger handoff threshold.
inline double adaptive_is_yield_fraction(
        double base, size_t active, double average_degree,
        size_t residual_handoff_threshold) {
    if (!(base > 0.0) ||
        residual_handoff_threshold == std::numeric_limits<size_t>::max() ||
        residual_handoff_threshold > std::numeric_limits<size_t>::max() / 4 ||
        active <= 4 * residual_handoff_threshold ||
        average_degree < static_cast<double>(residual_handoff_threshold))
        return base;
    return std::max(base, std::min(0.15, 3.0 * base));
}

inline constexpr double kResidualSparsifyKeepProbability = 0.25;
struct residual_policy_math {
    using wide = long double;
    static wide from_float(float x) { return x; }
    static wide subtract(wide a, wide b) { return a - b; }
    static wide multiply(wide a, wide b) { return a * b; }
    static bool less(wide a, wide b) { return a < b; }
};

// Conservative traffic gate for the one-shot BK-residual sparsifier. The
// 256-vertex probe estimates distinct live edges E. A spanning forest needs at
// most active-1 of them; fixed-p sampling is therefore expected to remove D
// edges. The discarded main-selector round supplies a residual-observed yield:
// (active-threshold)/selected-vertices estimates how many similarly sized BK
// rounds remain. Require their avoided edge traffic to repay eight full
// E-equivalent
// passes (coalesce/sort, collect, bucket order, forest, and the two-pass
// rebuild, with margin). Solve-side factor savings are deliberately omitted.
inline bool residual_sparsify_worthwhile(
        std::size_t active, std::size_t residual_threshold,
        std::size_t handoff_vertices, double avg_distinct_degree,
        double keep_probability = kResidualSparsifyKeepProbability) {
    if (active <= residual_threshold || handoff_vertices == 0 ||
        !(avg_distinct_degree > 0.0) ||
        !(keep_probability > 0.0 && keep_probability < 1.0))
        return false;
    const long double estimated_edges =
        0.5L * static_cast<long double>(active) * avg_distinct_degree;
    const long double forest_upper =
        static_cast<long double>(active - 1);
    const long double estimated_rounds =
        static_cast<long double>(active - residual_threshold) /
        static_cast<long double>(handoff_vertices);
    return residual_sparsify::worthwhile(residual_policy_math{}, estimated_edges,
        forest_upper, estimated_rounds, static_cast<long double>(keep_probability));
}

}
