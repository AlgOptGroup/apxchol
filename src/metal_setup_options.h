#pragma once
#include "apxchol/env_knobs.h"
#include <limits>
#include "apxchol/solver/factor_options.h"
#include "apxchol/solver/elimination/gpu_round_shadow.h"
#include "apxchol/solver/detail/metal_schedule.h"

namespace apxchol::detail {

inline constexpr auto metal_regularization_error =
    "Metal setup requires APXCHOL_REG_EPS in [2^-100, 2^100] and positive normal FP32 diagonal products";

inline bool metal_setup_regularization_supported(const env_knobs& environment = env_knobs::get(),
                                                 const Eigen::SparseMatrix<double>* A = nullptr) {
    if (environment.ground != grounding_kind::reg) return true;
    if (!(environment.reg_eps > 0) || !metal_schedule::in_range(environment.reg_eps)) return false;
    if (A) {
        if (A->rows() != A->cols()) return false;
        const double epsilon = float(environment.reg_eps);
        for (Eigen::Index row = 0; row < A->rows(); ++row) {
            const double ground = A->coeff(row, row) * epsilon;
            if (!(ground >= std::numeric_limits<float>::min() &&
                  ground <= std::numeric_limits<float>::max())) return false;
        }
    }
    return true;
}

inline unsigned metal_degree_divisor(double quantile) {
    if (quantile == degree_quantile_by_route || quantile == 0.2) return 5;
    return quantile == 0.5 ? 2 : 0;
}

inline bool metal_setup_options_supported(const factor_options &options) {
#ifndef APXCHOL_POOL_FP32
    (void)options;
    return false;
#else
    return sizeof(node_index) == sizeof(std::uint32_t) && options.is_select == "block_greedy" &&
           options.sampler == clique_sampler::gks && options.exact_clique_max_degree == 0 &&
           exact_core_or_off(options.exact_core_max_h) == 0 && options.double_cycle_min_h == 0 &&
           options.partition.degree_tiebreak &&
           options.residual_peel == residual_peel_strategy::natural &&
           metal_degree_divisor(options.partition.degree_quantile) != 0;
#endif
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
