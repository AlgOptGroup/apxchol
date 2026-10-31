#pragma once
#include <limits>
#include <optional>
#include "apxchol/solver/factor_options.h"
#include "apxchol/solver/elimination/gpu_round_shadow.h"
#include "apxchol/solver/detail/metal_schedule.h"

namespace apxchol::detail {

inline constexpr auto metal_regularization_error =
    "Metal setup requires regularization in [2^-100, 2^100] and positive normal FP32 diagonal products";

inline bool metal_setup_regularization_supported(double regularization,
                                                 const Eigen::SparseMatrix<double>* A = nullptr) {
    if (regularization == 0) return true;
    if (!(regularization > 0) || !metal_schedule::in_range(regularization)) return false;
    if (A) {
        if (A->rows() != A->cols()) return false;
        const double epsilon = float(regularization);
        for (Eigen::Index row = 0; row < A->rows(); ++row) {
            const double ground = A->coeff(row, row) * epsilon;
            if (!(ground >= std::numeric_limits<float>::min() &&
                  ground <= std::numeric_limits<float>::max())) return false;
        }
    }
    return true;
}

inline double metal_degree_quantile(std::optional<double> quantile) {
    return quantile.value_or(degree_quantile_host_default);
}

inline bool metal_setup_options_supported(const factor_options &options) {
    const double quantile = metal_degree_quantile(options.partition.degree_quantile);
    return sizeof(node_index) == sizeof(std::uint32_t) &&
           options.experimental.cpu_schedule == cpu_sptrsv_schedule::automatic &&
           options.partitioner == partitioner_kind::block_greedy &&
           options.sampler == clique_sampler::gks &&
           options.experimental.exact_clique_max_degree == 0 &&
           options.experimental.exact_core_max_h.value_or(0) == 0 &&
           options.experimental.double_cycle_min_h == 0 &&
           options.experimental.degree_tiebreak &&
           quantile > 0 && quantile < 1;
}

inline bool metal_setup_input_supported(const Eigen::SparseMatrix<double> &A) {
    std::size_t offdiagonals = 0;
    bool zeros = false;
    // Numerical validation proves pairing except for stored zeros.
    if (!gpu_owned_csc_layout_supported(A, offdiagonals, &zeros) ||
        (zeros && !gpu_owned_csc_paired(A)))
        return false;
    bool valid = true;
#pragma omp parallel for schedule(static) reduction(&& : valid)
    for (Eigen::Index p = 0; p < A.nonZeros(); ++p)
        valid = valid && metal_schedule::in_range(A.valuePtr()[p]);
    return valid;
}

} // namespace apxchol::detail
