#pragma once
/// Configurable options for the approximate Cholesky factorization.

#include <cstddef>
#include <limits>
#include <string>
#include <string_view>
#include <stdexcept>

#include "apxchol/types.h"
#include "apxchol/detail/operator_options.h"
#include "apxchol/lowprec.h"
#include "apxchol/solver/elimination/sampler.h"

namespace apxchol {

enum class cpu_sptrsv_schedule { automatic, levels };

namespace detail {

inline cpu_sptrsv_schedule parse_cpu_schedule(std::string_view value) {
    if (value == "auto") return cpu_sptrsv_schedule::automatic;
    if (value == "levels") return cpu_sptrsv_schedule::levels;
    throw std::invalid_argument("cpu_schedule must be auto or levels");
}

} // namespace detail

/// Pivot order for the serial tail after parallel elimination and BK handoff.
enum class residual_peel_strategy {
    natural,     // peel in original active-list order (default)
    min_degree,  // lazy degree-heap heuristic; not exact current minimum degree
    bk_serial    // sample √|active| vertices and peel the min-degree one
};

/// Resolve `exact_core_max_h` to 4 on CPU or 0 for GPU-owned setup.
inline constexpr std::size_t exact_core_by_route =
    std::numeric_limits<std::size_t>::max();

/// Default CPU exact-core cap; see `exact_core_max_h`.
inline constexpr std::size_t exact_core_host_default = 4;

/// Resolve `degree_quantile` from the factorization route.
inline constexpr double degree_quantile_by_route = -1.0;

/// CPU default, also used by partitioners invoked without a factorization route.
inline constexpr double degree_quantile_host_default = 0.2;

/// CUDA default; Metal uses the CPU default.
inline constexpr double degree_quantile_device_default = 0.5;

/// Treat the unresolved core cap as off when checking GPU compatibility.
inline constexpr std::size_t exact_core_or_off(std::size_t h) noexcept {
    return h == exact_core_by_route ? 0u : h;
}

/// Resolve an unset quantile to the CPU default, preserving explicit values.
inline constexpr double degree_quantile_or_host_default(double q) noexcept {
    return q < 0.0 ? degree_quantile_host_default : q;
}

/// Selection options for built-in partitioners, passed as opts.partition.
struct partition_options {
    /// Admit the lowest-degree quantile of active vertices as IS candidates,
    /// including ties at the cutoff. Values in (0,1) replace degree_multiplier;
    /// 0 uses that multiplier instead. The -1 sentinel selects 0.2 on CPU/Metal
    /// or 0.5 on CUDA, whose wider rounds amortize device launch overhead.
    /// Metal requires a quantile in (0,1) and rounds the cutoff up to its
    /// degree bucket's upper bound. CPU/CUDA preserve explicit values.
    double degree_quantile = degree_quantile_by_route;

    /// Fallback IS degree cap when degree_quantile == 0:
    /// threshold = degree_multiplier × avg_degree.
    double degree_multiplier = 2.0;

    /// Prefer lower-degree candidates in conflicts. block_greedy breaks degree
    /// ties by index; priority_greedy uses its hashed priority to break ties.
    bool degree_tiebreak = true;
};

struct factor_options {
    unsigned seed = 42;
    /// Matrix-input diagonal regularization as a fraction of each diagonal; zero disables it.
    double regularization = 0;
    /// Matrix-input lumping ceiling; zero rejects positive off-diagonals.
    double lump_max_mass = kDefaultLumpMassCeiling;
    /// Triangular-solve storage; construction arithmetic and exports are unchanged.
    /// automatic uses the backend default (CPU/Metal fp32, CUDA fp16).
    factor_storage_type factor_storage = factor_storage_type::automatic;
    cpu_sptrsv_schedule cpu_schedule = cpu_sptrsv_schedule::automatic;
    /// Relative off-diagonal drop threshold in [0, 1]; zero disables dropping.
    double drop_tolerance = 1e-4;
    bool verbose = false;
    partition_options partition{};   // selection options (see above)
    /// Minimum yield for scan-based selectors (selected regions / degree-eligible
    /// vertices) before BK handoff; 0 disables yield-based handoff.
    /// Zero progress always hands off.
    /// Nonempty selections are retained within twice the handoff threshold, or
    /// when their size reaches omp_threshold. For residuals larger than four
    /// handoff thresholds with average degree at least that threshold, use
    /// max(base, min(0.15, 3*base)); sparse/small residuals use the base value.
    double min_is_fraction = 0.05;
    /// OpenMP work threshold for selection and elimination. Smaller rounds size
    /// their team from selected pivots and live-degree work. Changing this may
    /// change selection order and the factor across thread counts.
    size_t omp_threshold = 2000;
    std::string is_select = "block_greedy";  // Partitioner name (runtime dispatch via dispatch_partitioner)

    /// After the main selector hands off, run BK rounds until the residual
    /// reaches this size, then use residual_peel. SIZE_MAX selects the
    /// partitioner's default: 500 for block_greedy and priority_greedy.
    /// Keeping the default natural-order serial tail small limits fill.
    size_t parallel_residual_threshold = std::numeric_limits<size_t>::max();

    /// Serial pivot order: natural visits the supplied order; min_degree uses
    /// a lazy heap (not exact current minimum degree); bk_serial samples
    /// sqrt(active) vertices with replacement and picks the lowest live degree.
    residual_peel_strategy residual_peel = residual_peel_strategy::natural;
    /// Emit the exact Schur clique for degrees at most this value; 0 disables
    /// it. This removes sampling variance at the cost of d(d-1)/2 fill edges,
    /// so it is intended for small degrees. GPU-owned setup requires 0.
    size_t exact_clique_max_degree = 0;

    /// Trace-cycle core rules for h near-equal heavy neighbors. For
    /// h <= exact_core_max_h, emit the exact core clique; otherwise, when
    /// h >= double_cycle_min_h > 0, emit two independent cycles at half weight.
    /// Both preserve expected pair weights; 0 disables the corresponding rule.
    /// The exact-core sentinel resolves to 4 on CPU, balancing fill against
    /// solve cost, or 0 for GPU-owned setup. Other host samplers ignore these
    /// rules, but GPU-owned setup requires them off regardless of sampler.
    size_t exact_core_max_h = exact_core_by_route;
    size_t double_cycle_min_h = 0;

    clique_sampler sampler = clique_sampler::gks;


};

namespace detail {
inline void validate_drop_tolerance(double value) {
    if (!(value >= 0 && value <= 1))
        throw std::invalid_argument("drop_tolerance must be in [0, 1]");
}

inline void validate_factor_options(const factor_options& options) {
    validate_regularization(options.regularization);
    validate_lump_max_mass(options.lump_max_mass);
    validate_drop_tolerance(options.drop_tolerance);
    const double quantile = options.partition.degree_quantile;
    if (!std::isfinite(quantile) ||
        (quantile != degree_quantile_by_route && !(quantile >= 0 && quantile < 1)))
        throw std::invalid_argument("degree_quantile must be -1 (automatic) or in [0, 1)");
    if (!std::isfinite(options.partition.degree_multiplier) ||
        options.partition.degree_multiplier <= 0)
        throw std::invalid_argument("degree_multiplier must be finite and positive");
    if (!std::isfinite(options.min_is_fraction) ||
        options.min_is_fraction < 0 || options.min_is_fraction > 1)
        throw std::invalid_argument("min_is_fraction must be in [0, 1]");
    if (options.cpu_schedule != cpu_sptrsv_schedule::automatic &&
        options.cpu_schedule != cpu_sptrsv_schedule::levels)
        throw std::invalid_argument("invalid CPU triangular-solve schedule");
    if (options.factor_storage != factor_storage_type::automatic &&
        options.factor_storage != factor_storage_type::fp16 &&
        options.factor_storage != factor_storage_type::fp32)
        throw std::invalid_argument("invalid factor storage");
    if (options.sampler != clique_sampler::gks && options.sampler != clique_sampler::trace_cycle)
        throw std::invalid_argument("invalid clique sampler");
    if (options.residual_peel != residual_peel_strategy::natural &&
        options.residual_peel != residual_peel_strategy::min_degree &&
        options.residual_peel != residual_peel_strategy::bk_serial)
        throw std::invalid_argument("invalid residual peel strategy");
}
inline void validate_graph_options(const factor_options& options) {
    validate_factor_options(options);
    if (options.regularization != 0 || options.lump_max_mass != kDefaultLumpMassCeiling)
        throw std::invalid_argument("regularization and lump_max_mass require matrix input; "
                                    "a supplied graph must already have valid positive weights");
}
} // namespace detail
} // namespace apxchol
