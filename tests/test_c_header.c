/* A pure C11 consumer of apxchol/c_api.h: the header must compile as C, the
 * struct layout is part of the ABI, and one Laplacian solve runs through it. */
#include "apxchol/c_api.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>

_Static_assert(sizeof(apxchol_options) == 64, "apxchol_options layout");
_Static_assert(offsetof(apxchol_options, backend) == 4, "apxchol_options layout");
_Static_assert(offsetof(apxchol_options, tol) == 8, "apxchol_options layout");
_Static_assert(offsetof(apxchol_options, max_iter) == 16, "apxchol_options layout");
_Static_assert(offsetof(apxchol_options, stagnation_window) == 20, "apxchol_options layout");
_Static_assert(offsetof(apxchol_options, seed) == 24, "apxchol_options layout");
_Static_assert(offsetof(apxchol_options, threads) == 28, "apxchol_options layout");
_Static_assert(offsetof(apxchol_options, sampler) == 32, "apxchol_options layout");
_Static_assert(offsetof(apxchol_options, partitioner) == 36, "apxchol_options layout");
_Static_assert(offsetof(apxchol_options, storage) == 40, "apxchol_options layout");
_Static_assert(offsetof(apxchol_options, keep_factor_values) == 44, "apxchol_options layout");
_Static_assert(offsetof(apxchol_options, degree_quantile) == 48, "apxchol_options layout");
_Static_assert(offsetof(apxchol_options, exact_clique_max_degree) == 56, "apxchol_options layout");

_Static_assert(sizeof(apxchol_solve_info) == 32, "apxchol_solve_info layout");
_Static_assert(offsetof(apxchol_solve_info, converged) == 4, "apxchol_solve_info layout");
_Static_assert(offsetof(apxchol_solve_info, iterations) == 8, "apxchol_solve_info layout");
_Static_assert(offsetof(apxchol_solve_info, relative_residual) == 16, "apxchol_solve_info layout");
_Static_assert(offsetof(apxchol_solve_info, solve_seconds) == 24, "apxchol_solve_info layout");

_Static_assert(sizeof(apxchol_stats) == 64, "apxchol_stats layout");
_Static_assert(offsetof(apxchol_stats, backend) == 4, "apxchol_stats layout");
_Static_assert(offsetof(apxchol_stats, n) == 8, "apxchol_stats layout");
_Static_assert(offsetof(apxchol_stats, factor_nnz) == 16, "apxchol_stats layout");
_Static_assert(offsetof(apxchol_stats, lumped_offdiag) == 24, "apxchol_stats layout");
_Static_assert(offsetof(apxchol_stats, rounds) == 32, "apxchol_stats layout");
_Static_assert(offsetof(apxchol_stats, peak_graph_bytes) == 40, "apxchol_stats layout");
_Static_assert(offsetof(apxchol_stats, setup_seconds) == 48, "apxchol_stats layout");
_Static_assert(offsetof(apxchol_stats, sddm) == 56, "apxchol_stats layout");
_Static_assert(offsetof(apxchol_stats, setup_max_threads) == 60, "apxchol_stats layout");

int main(void) {
    /* Path Laplacian on 3 vertices, 1-based CSC. */
    const int64_t colptr[] = {1, 3, 6, 8};
    const int64_t rowval[] = {1, 2, 1, 2, 3, 2, 3};
    const double nzval[] = {1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0};
    const double b[] = {1.0, 0.0, -1.0};
    double x[3] = {0.0, 0.0, 0.0};
    char error[256];

    apxchol_options options;
    if (apxchol_options_default(&options, sizeof options) != APXCHOL_STATUS_SUCCESS) return 1;
    options.tol = 1e-10;
    options.threads = 1;

    apxchol_solver* solver = NULL;
    if (apxchol_solver_create(3, colptr, rowval, nzval, 1, &options, &solver, error,
                              sizeof error) != APXCHOL_STATUS_SUCCESS) {
        fprintf(stderr, "create: %s\n", error);
        return 1;
    }
    apxchol_solve_info info;
    info.struct_size = sizeof info;
    const apxchol_status status =
        apxchol_solver_solve(solver, b, NULL, x, -1.0, -1, &info, error, sizeof error);
    apxchol_solver_destroy(solver);
    if (status != APXCHOL_STATUS_SUCCESS || info.converged != 1) {
        fprintf(stderr, "solve: %d %s\n", (int)status, error);
        return 1;
    }
    /* L x = b with mean(x) = 0: x = (1, 0, -1). */
    if (fabs(x[0] - 1.0) > 1e-8 || fabs(x[1]) > 1e-8 || fabs(x[2] + 1.0) > 1e-8) {
        fprintf(stderr, "solution %g %g %g\n", x[0], x[1], x[2]);
        return 1;
    }
    printf("apxchol %s, C ABI %d: ok\n", apxchol_version(), (int)apxchol_c_abi_version());
    return 0;
}
