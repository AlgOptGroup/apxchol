#pragma once
#include "apxchol/solver/solve.h"
#include <filesystem>
#include <span>

namespace apxchol {
namespace detail {

// The caller validates retained factor values before reconstructing the operator.
void save_solver_data(const std::filesystem::path& path,
                      const Eigen::SparseMatrix<double>& A, const factorization& F, Eigen::Index input_entries);

// Convert an owned symmetric CSR operator back to original-coordinate FP64 CSC.
// Only the adopted Metal route stores both coordinates in factor order.
template<class Scalar>
Eigen::SparseMatrix<double> archive_operator_csc(int n, const int* outer,
        const int* inner, const Scalar* values, std::span<const node_index> perm = {}) {
    using Csr = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;
    const Eigen::Map<const Csr> stored(n, n, outer[n], outer, inner, values);
    Eigen::SparseMatrix<double> result = stored.template cast<double>();
    if (!perm.empty()) {
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> P(n);
        for (int i = 0; i < n; ++i) P.indices()[i] = static_cast<int>(perm[i]);
        result = (P.transpose() * result * P).eval();
    }
    return result;
}
} // namespace detail
} // namespace apxchol
