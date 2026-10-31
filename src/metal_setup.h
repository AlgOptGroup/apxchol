#pragma once
#include "apxchol/solver/detail/metal_schedule.h"
#include <optional>
#include "apxchol/detail/checkpoint.h"
#include "apxchol/solver/factorization.h"
#include "apxchol/solver/solve.h"
#import <Metal/Metal.h>

namespace apxchol::detail::metal_setup {

struct triangle {
    id<MTLBuffer> ptr = nil, idx = nil, val = nil;
    id<MTLBuffer> words = nil, blocks = nil, segments = nil;
    std::vector<metal_schedule::step> stages{};
    unsigned levels = 0, tag_count = 0;
    bool resident_round_plan = false;
};

struct resident_factor {
    triangle forward, backward;
    id<MTLBuffer> metadata = nil;
    id<MTLBuffer> operator_ptr = nil, operator_idx = nil;
    id<MTLBuffer> operator_row_map = nil;
    id<MTLBuffer> operator_hi = nil, operator_lo = nil;
    factorization factor;
    bool half = false, operator_exact = true, operator_triangles_identical = false;
    unsigned max_operator_row = 0;
    float preconditioner_scale = 1;
};

void prepare_lane_plans(id<MTLDevice> device, const factorization& factor,
                        triangle& forward, triangle& backward);
/// The single-lane round plans resident setup builds, for an installed factor
/// whose rounds are elimination rounds (round_plans_apply), so a reinstalled
/// factor is solved in the same order as when it was built.
void prepare_round_plans(id<MTLDevice> device, const factorization& factor,
                         triangle& forward, triangle& backward);
/// The device operator resident setup builds (input row order, permuted
/// columns, double-float values), for an installed factor with permutation
/// perm, or nullopt when A lacks the paired 32-bit CSC layout it needs. Only
/// the operator fields of the result are set.
std::optional<resident_factor> prepare_input_operator(id<MTLDevice> device,
    const Eigen::SparseMatrix<double>& A, const std::vector<node_index>& perm,
    double lump_max_mass);

resident_factor factorize(id<MTLDevice> device, const Eigen::SparseMatrix<double> &A,
                          const solve_options &options, checkpoint *cp,
                          bool input_prechecked = false);

} // namespace apxchol::detail::metal_setup
