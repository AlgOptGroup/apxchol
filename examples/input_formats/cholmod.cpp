/// SuiteSparse CHOLMOD: a cholmod_sparse with 64-bit indices (cholmod_l_*).
#include "apxchol.h"
#include <cholmod.h>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    cholmod_common c;
    cholmod_l_start(&c);

    // Laplacian of the path 0-1-2-3 with edge weights 1, 2, 1. CHOLMOD usually
    // stores a symmetric matrix as one triangle: here the lower one (stype < 0).
    cholmod_triplet* T = cholmod_l_allocate_triplet(4, 4, 7, -1, CHOLMOD_REAL, &c);
    if (!T) {
        cholmod_l_finish(&c);
        return 2;
    }
    auto* row = static_cast<std::int64_t*>(T->i);
    auto* col = static_cast<std::int64_t*>(T->j);
    auto* val = static_cast<double*>(T->x);
    const double weight[] = {1, 2, 1};
    for (std::int64_t u = 0; u < 4; ++u) {
        row[T->nnz] = u; col[T->nnz] = u;
        val[T->nnz++] = (u > 0 ? weight[u - 1] : 0) + (u < 3 ? weight[u] : 0);
    }
    for (std::int64_t u = 0; u < 3; ++u) {
        row[T->nnz] = u + 1; col[T->nnz] = u; val[T->nnz++] = -weight[u];
    }
    cholmod_sparse* lower = cholmod_l_triplet_to_sparse(T, 0, &c);

    // apxchol needs both triangles (stype 0) in packed, sorted columns.
    cholmod_sparse* A = lower ? cholmod_l_copy(lower, 0, 1, &c) : nullptr;
    if (A && !A->sorted) cholmod_l_sort(A, &c);
    int status = 2;  // a CHOLMOD failure
    if (A && c.status == CHOLMOD_OK) {
        const auto n = static_cast<std::size_t>(A->ncol);
        const auto nnz = static_cast<std::size_t>(cholmod_l_nnz(A, &c));
        const apxchol::csc_view64 view{static_cast<std::int64_t>(n),
                                       {static_cast<const std::int64_t*>(A->p), n + 1},
                                       {static_cast<const std::int64_t*>(A->i), nnz},
                                       {static_cast<const double*>(A->x), nnz}};

        const std::vector<double> b{1, 0, 0, -1};
        const auto result = apxchol::solve(view, b, 1e-10);
        std::printf("CHOLMOD: residual %.1e\n", result.residual);
        status = result.converged ? 0 : 1;
    }

    // The free functions accept null pointers.
    cholmod_l_free_sparse(&A, &c);
    cholmod_l_free_sparse(&lower, &c);
    cholmod_l_free_triplet(&T, &c);
    cholmod_l_finish(&c);
    return status;
}
