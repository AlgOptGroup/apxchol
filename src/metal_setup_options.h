#pragma once
#include <limits>
#include <optional>
#include "apxchol/solver/factor_options.h"
#include "apxchol/solver/elimination/gpu_round_shadow.h"
#include "apxchol/solver/detail/metal_schedule.h"

namespace apxchol::detail {

inline double metal_degree_quantile(std::optional<double> quantile) {
    return quantile.value_or(degree_quantile_host_default);
}

inline bool metal_setup_options_supported(const factor_options &options) {
    const double quantile = metal_degree_quantile(options.experimental.partition.degree_quantile);
    return sizeof(node_index) == sizeof(std::uint32_t) &&
           options.experimental.cpu_schedule == cpu_sptrsv_schedule::automatic &&
           options.experimental.partitioner == partitioner_kind::block_greedy &&
           options.experimental.sampler == clique_sampler::gks &&
           options.experimental.exact_core_max_h.value_or(0) == 0 &&
           options.experimental.double_cycle_min_h == 0 &&
           quantile > 0 && quantile < 1;
}

inline bool metal_setup_input_supported(csc_ref<double> A) {
    std::size_t offdiagonals = 0;
    bool zeros = false;
    // Numerical validation proves pairing except for stored zeros.
    if (!gpu_owned_csc_layout_supported(A, offdiagonals, &zeros) ||
        (zeros && !gpu_owned_csc_paired(A)))
        return false;
    bool valid = true;
#pragma omp parallel for schedule(static) reduction(&& : valid)
    for (std::int64_t p = 0; p < static_cast<std::int64_t>(A.nnz()); ++p)
        valid = valid && metal_schedule::in_range(A.val[p]);
    return valid;
}

} // namespace apxchol::detail
