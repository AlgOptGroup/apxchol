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

/// CPU triangular-solve scheduling: choose automatically or use levels only.
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
    /// Lowest-degree fraction admitted as selection candidates, including ties.
    /// Use -1 for the backend default (CPU/Metal 0.2, CUDA 0.5), or a value in
    /// [0, 1). Zero uses degree_multiplier instead.
    /// Metal requires (0, 1) and rounds the cutoff to its degree-bucket boundary.
    double degree_quantile = degree_quantile_by_route;

    /// Positive degree cap multiplier when degree_quantile is zero:
    /// admit degrees at most degree_multiplier times the average degree.
    double degree_multiplier = 2.0;

    /// Prefer lower-degree candidates in conflicts. block_greedy breaks degree
    /// ties by index; priority_greedy uses its hashed priority to break ties.
    bool degree_tiebreak = true;
};

struct factor_options {
    /// Seed for factor construction; see the precision guide for repeatability limits.
    unsigned seed = 42;
    /// Minimum diagonal excess of the preconditioner, as a fraction of each
    /// diagonal after lump_max_mass lumping. Excess below 1e-12 of the diagonal
    /// counts as zero first, as row-sum rounding noise. The operator is not
    /// modified. Zero disables it.
    double regularization = 0;
    /// Allowed positive off-diagonal mass fraction in [0, 1]. Zero rejects it.
    /// Accepted positive entries are moved to the preconditioner's diagonal.
    double lump_max_mass = kDefaultLumpMassCeiling;
    /// Triangular-solve storage; construction arithmetic and exports are unchanged.
    /// automatic uses the backend default (CPU/Metal fp32, CUDA fp16).
    factor_storage_type factor_storage = factor_storage_type::automatic;
    /// CPU triangular-solve schedule; automatic chooses the installed schedule.
    cpu_sptrsv_schedule cpu_schedule = cpu_sptrsv_schedule::automatic;
    /// Relative off-diagonal drop threshold in [0, 1]; zero disables dropping.
    double drop_tolerance = 1e-4;
    /// Print setup and solve diagnostics.
    bool verbose = false;
    /// Candidate selection controls for the chosen partitioner.
    partition_options partition{};
    /// Base selection-yield threshold before switching to Baumann-Kyng rounds.
    /// Measured as selected regions divided by degree-eligible vertices; dense
    /// residuals may use a higher threshold. Zero disables yield-based handoff,
    /// but a round with no progress always hands off.
    double min_is_fraction = 0.05;
    /// Work threshold for parallel selection and elimination. Smaller rounds
    /// may use fewer threads; changing this can change the constructed factor.
    size_t omp_threshold = 2000;
    /// Built-in partitioner name used by runtime dispatch.
    std::string is_select = "block_greedy";

    /// Residual size at which Baumann-Kyng rounds switch to residual_peel.
    /// SIZE_MAX uses the partitioner's default: 500 for block_greedy and
    /// priority_greedy. A small serial tail limits fill.
    size_t parallel_residual_threshold = std::numeric_limits<size_t>::max();

    /// Serial pivot order: natural visits the supplied order; min_degree uses
    /// a lazy heap (not exact current minimum degree); bk_serial samples
    /// sqrt(active) vertices with replacement and picks the lowest live degree.
    residual_peel_strategy residual_peel = residual_peel_strategy::natural;
    /// Emit an exact Schur clique for degrees at most this value; zero disables
    /// it. Intended for small degrees because it adds d(d-1)/2 edges.
    /// GPU factor construction requires zero.
    size_t exact_clique_max_degree = 0;

    /// Trace-cycle exact-clique limit for near-equal heavy neighbors; zero
    /// disables it. exact_core_by_route selects 4 on CPU or 0 on GPU.
    /// Other CPU samplers ignore it; GPU construction requires zero.
    size_t exact_core_max_h = exact_core_by_route;
    /// Trace-cycle rule: with at least this many near-equal heavy neighbors, emit
    /// two half-weight cycles unless the exact-core rule applies. Zero disables
    /// it; GPU construction requires zero. Expected pair weights are preserved.
    size_t double_cycle_min_h = 0;

    /// Clique-sampling algorithm for factor construction.
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
