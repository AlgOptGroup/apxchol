#include "apxchol/factorization.h"
#include <stdexcept>

namespace apxchol {

factorization factorize(const Eigen::SparseMatrix<double>& L,
                        const factor_options& /*opts*/) {
    const int n = static_cast<int>(L.rows());
    if (n != L.cols())
        throw std::invalid_argument("factorize: matrix must be square");

    // TODO: implement Kyng-Sachdeva approximate Gaussian elimination
    //
    // Algorithm outline (arXiv:1605.02353):
    //   1. Build adjacency from L
    //   2. While vertices remain:
    //      a. Find independent set S of low-degree vertices
    //      b. For each v in S, eliminate v:
    //         - Sample a random neighbor proportional to edge weight
    //         - Form a clique on remaining neighbors with sampled weights
    //         - Record column of L factor
    //      c. Update adjacency (Schur complement)
    //   3. Permute and assemble sparse lower-triangular factor

    throw std::runtime_error("factorize: not yet implemented");
}

} // namespace apxchol
