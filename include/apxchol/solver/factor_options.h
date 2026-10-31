#pragma once
/// Configurable options for the approximate Cholesky factorization.

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string_view>
#include <stdexcept>

#include "apxchol/types.h"
#include "apxchol/detail/operator_options.h"
#include "apxchol/detail/option_names.h"
#include "apxchol/lowprec.h"
#include "apxchol/solver/elimination/sampler.h"

namespace apxchol {

/// CPU triangular-solve scheduling (experimental): choose automatically or use levels only.
enum class cpu_sptrsv_schedule { automatic, levels };

namespace detail {

inline cpu_sptrsv_schedule parse_cpu_schedule(std::string_view value) {
    if (value == "auto") return cpu_sptrsv_schedule::automatic;
    if (value == "levels") return cpu_sptrsv_schedule::levels;
    throw std::invalid_argument("cpu_schedule must be auto or levels");
}

} // namespace detail

/// Built-in partitioner used by runtime dispatch.
enum class partitioner_kind { block_greedy, priority_greedy, baumann_kyng };

namespace detail {

/// Frontend names, in partitioner_kind and partitioner_list order.
inline constexpr std::array<option_name<partitioner_kind>, 3> partitioner_names{{
    {"block_greedy", partitioner_kind::block_greedy},
    {"priority_greedy", partitioner_kind::priority_greedy},
    {"baumann_kyng", partitioner_kind::baumann_kyng},
}};

// Route defaults for unset options. CPU and Metal use the host values;
// partitioners invoked without a factorization route use the host quantile.
inline constexpr double degree_quantile_host_default = 0.2;
inline constexpr double degree_quantile_device_default = 0.5;
inline constexpr std::size_t exact_core_host_default = 4;

} // namespace detail

/// Selection options for built-in partitioners, passed as opts.partition.
struct partition_options {
    /// Lowest-degree fraction admitted as selection candidates, including ties.
    /// Unset uses the backend default (CPU/Metal 0.2, CUDA 0.5); a set value
    /// must be in [0, 1). Zero uses degree_multiplier instead.
    /// Metal requires (0, 1) and rounds the cutoff to its degree-bucket boundary.
    std::optional<double> degree_quantile;

    /// Positive degree cap multiplier when degree_quantile is zero:
    /// admit degrees at most degree_multiplier times the average degree.
    double degree_multiplier = 2.0;
};

/// Research controls outside the 1.x compatibility promise. Members may change
/// or disappear in any release; keep the defaults for ordinary use.
/// GPU routes require the defaults of cpu_schedule, exact_clique_max_degree,
/// exact_core_max_h and double_cycle_min_h; Metal also requires degree_tiebreak.
struct experimental_options {
    /// CPU triangular-solve schedule; automatic chooses the installed schedule.
    cpu_sptrsv_schedule cpu_schedule = cpu_sptrsv_schedule::automatic;
    /// Prefer lower-degree candidates in conflicts. block_greedy breaks degree
    /// ties by index; priority_greedy uses its hashed priority to break ties.
    bool degree_tiebreak = true;
    /// Base selection-yield threshold before switching to Baumann-Kyng rounds.
    /// Measured as selected regions divided by degree-eligible vertices; dense
    /// residuals may use a higher threshold. Zero disables yield-based handoff,
    /// but a round with no progress always hands off.
    double min_is_fraction = 0.05;
    /// Work threshold for parallel selection and elimination. Smaller rounds
    /// may use fewer threads; changing this can change the constructed factor.
    size_t omp_threshold = 2000;
    /// Residual size at which Baumann-Kyng rounds switch to the serial peel.
    /// Unset uses the partitioner's default: 500 for block_greedy and
    /// priority_greedy. A small serial tail limits fill.
    std::optional<size_t> parallel_residual_threshold;
    /// Emit an exact Schur clique for degrees at most this value; zero disables
    /// it. Intended for small degrees because it adds d(d-1)/2 edges.
    size_t exact_clique_max_degree = 0;
    /// Trace-cycle exact-clique limit for near-equal heavy neighbors; zero
    /// disables it. Unset selects 4 on CPU or 0 on GPU.
    /// Other CPU samplers ignore it.
    std::optional<size_t> exact_core_max_h;
    /// Trace-cycle rule: with at least this many near-equal heavy neighbors, emit
    /// two half-weight cycles unless the exact-core rule applies. Zero disables
    /// it. Expected pair weights are preserved.
    size_t double_cycle_min_h = 0;
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
    /// Relative off-diagonal drop threshold in [0, 1]; zero disables dropping.
    double drop_tolerance = 1e-4;
    /// Print setup and solve diagnostics.
    bool verbose = false;
    /// Candidate selection controls for the chosen partitioner.
    partition_options partition{};
    /// Built-in partitioner; GPU construction requires block_greedy.
    partitioner_kind partitioner = partitioner_kind::block_greedy;
    /// Clique-sampling algorithm for factor construction.
    clique_sampler sampler = clique_sampler::gks;
    /// Research controls outside the 1.x compatibility promise.
    experimental_options experimental{};
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
    if (const auto quantile = options.partition.degree_quantile;
        quantile && !(*quantile >= 0 && *quantile < 1))
        throw std::invalid_argument("degree_quantile must be in [0, 1)");
    if (!std::isfinite(options.partition.degree_multiplier) ||
        options.partition.degree_multiplier <= 0)
        throw std::invalid_argument("degree_multiplier must be finite and positive");
    if (options.partitioner != partitioner_kind::block_greedy &&
        options.partitioner != partitioner_kind::priority_greedy &&
        options.partitioner != partitioner_kind::baumann_kyng)
        throw std::invalid_argument("invalid partitioner");
    if (options.factor_storage != factor_storage_type::automatic &&
        options.factor_storage != factor_storage_type::fp16 &&
        options.factor_storage != factor_storage_type::fp32)
        throw std::invalid_argument("invalid factor storage");
    if (options.sampler != clique_sampler::gks && options.sampler != clique_sampler::trace_cycle)
        throw std::invalid_argument("invalid clique sampler");
    const auto& experimental = options.experimental;
    if (!std::isfinite(experimental.min_is_fraction) ||
        experimental.min_is_fraction < 0 || experimental.min_is_fraction > 1)
        throw std::invalid_argument("min_is_fraction must be in [0, 1]");
    if (experimental.cpu_schedule != cpu_sptrsv_schedule::automatic &&
        experimental.cpu_schedule != cpu_sptrsv_schedule::levels)
        throw std::invalid_argument("invalid CPU triangular-solve schedule");
}
inline void validate_graph_options(const factor_options& options) {
    validate_factor_options(options);
    if (options.regularization != 0 || options.lump_max_mass != kDefaultLumpMassCeiling)
        throw std::invalid_argument("regularization and lump_max_mass require matrix input; "
                                    "a supplied graph must already have valid positive weights");
}
} // namespace detail
} // namespace apxchol
