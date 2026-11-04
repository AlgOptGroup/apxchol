#pragma once
#include "apxchol/solver/solve.h"
#include "apxchol/detail/csc_matrix.h"
#include <cstdint>
#include <filesystem>
#include <span>

namespace apxchol {
namespace detail {

// The caller validates retained factor values before reconstructing the operator.
void save_solver_data(const std::filesystem::path& path,
                      csc_ref<double> A, const factorization& F, std::int64_t input_entries);

// Convert an owned symmetric CSR operator back to original-coordinate FP64
// CSC with sorted columns. Only the adopted Metal route stores both
// coordinates in factor order (perm[original] = factor position).
template<class Scalar>
csc_matrix<double> archive_operator_csc(csc_ref<Scalar> stored,
                                        std::span<const node_index> perm = {}) {
    // The CSR arrays are the CSC arrays of the transpose.
    auto result = csc_transpose<double>(stored);
    if (!perm.empty()) result = csc_unpermute_symmetric(result.ref(), perm);
    return result;
}
} // namespace detail
} // namespace apxchol
