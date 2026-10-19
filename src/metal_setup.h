#pragma once
#include "apxchol/solver/detail/metal_schedule.h"
#include "apxchol/solver/factorization.h"
#include "apxchol/solver/solve.h"
#import <Metal/Metal.h>

namespace apxchol::detail::metal_setup {

struct triangle {
    id<MTLBuffer> ptr = nil, idx = nil, val = nil;
    id<MTLBuffer> words = nil, blocks = nil, segments = nil;
    std::vector<metal_schedule::step> stages;
    unsigned levels = 0, tag_count = 0;
};

struct resident_factor {
    triangle forward, backward;
    id<MTLBuffer> metadata = nil;
    id<MTLBuffer> operator_ptr = nil, operator_idx = nil;
    id<MTLBuffer> operator_hi = nil, operator_lo = nil;
    factorization factor;
    bool half = false, operator_exact = true, operator_triangles_identical = false;
    unsigned max_operator_row = 0;
    float preconditioner_scale = 1;
};

resident_factor factorize(id<MTLDevice> device, const Eigen::SparseMatrix<double> &A,
                          const solve_options &options, checkpoint *cp,
                          bool input_prechecked = false);

} // namespace apxchol::detail::metal_setup
