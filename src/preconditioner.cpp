#include "apxchol/preconditioner.h"

namespace apxchol {

preconditioner::preconditioner(const factorization& F)
    : F_(&F), n_(F.n), info_(Eigen::Success) {}

// solve() is implemented in the header as a template,
// but the core triangular-solve logic lives here.

} // namespace apxchol
