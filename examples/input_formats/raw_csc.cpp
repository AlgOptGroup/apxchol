/// Your own CSR or CSC arrays, e.g. exported from PETSc or Trilinos.
#include "apxchol.h"
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    // Laplacian of the path 0-1-2-3 with edge weights 1, 2, 1, stored as CSR.
    // The matrix is symmetric, so its CSR arrays are also its CSC arrays.
    const std::vector<std::int32_t> row_ptr{0, 2, 5, 8, 10};
    const std::vector<std::int32_t> col_idx{0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    const std::vector<double> values{1, -1, -1, 3, -2, -2, 3, -1, -1, 1};
    const std::vector<double> b{1, 0, 0, -1};   // a Laplacian RHS sums to zero

    const apxchol::csc_view A{4, row_ptr, col_idx, values};
    const auto result = apxchol::solve(A, b, 1e-10);

    // 64-bit index arrays use csc_view64.
    const std::vector<std::int64_t> row_ptr64(row_ptr.begin(), row_ptr.end());
    const std::vector<std::int64_t> col_idx64(col_idx.begin(), col_idx.end());
    const apxchol::csc_view64 A64{4, row_ptr64, col_idx64, values};
    const auto result64 = apxchol::solve(A64, b, 1e-10);

    std::printf("raw CSC: residual %.1e (32-bit), %.1e (64-bit)\n",
                result.residual, result64.residual);
    return result.converged && result64.converged ? 0 : 1;
}
