/* A pure C11 consumer of apxchol/c_api.h: the header must compile as C, the
 * struct layout is part of the ABI, and batched Laplacian operations run through it,
 * once factorizing and once adopting a factor handle. */
#include "apxchol/c_api.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>

_Static_assert(APXCHOL_C_ABI_VERSION == 3, "C ABI version");
_Static_assert(sizeof(apxchol_options) == 24, "apxchol_options layout");
_Static_assert(offsetof(apxchol_options, backend) == 4, "apxchol_options layout");
_Static_assert(offsetof(apxchol_options, seed) == 8, "apxchol_options layout");
_Static_assert(offsetof(apxchol_options, threads) == 12, "apxchol_options layout");
_Static_assert(offsetof(apxchol_options, keep_factor_values) == 16, "apxchol_options layout");
_Static_assert(offsetof(apxchol_options, factor_storage) == 20, "apxchol_options layout");
_Static_assert(APXCHOL_BACKEND_CPU == 0 && APXCHOL_BACKEND_GPU == 1 && APXCHOL_BACKEND_AUTO == 2,
               "backend values");
_Static_assert(APXCHOL_FACTOR_STORAGE_AUTO == 0, "automatic factor storage value");
_Static_assert(APXCHOL_FACTOR_STORAGE_FLOAT16 == 16, "float16 factor storage value");
_Static_assert(APXCHOL_FACTOR_STORAGE_FLOAT32 == 32, "float32 factor storage value");

_Static_assert(sizeof(apxchol_solve_info) == 24, "apxchol_solve_info layout");
_Static_assert(offsetof(apxchol_solve_info, iterations) == 0, "apxchol_solve_info layout");
_Static_assert(offsetof(apxchol_solve_info, relative_residual) == 8, "apxchol_solve_info layout");
_Static_assert(offsetof(apxchol_solve_info, converged) == 16, "apxchol_solve_info layout");

_Static_assert(sizeof(apxchol_stats) == 40, "apxchol_stats layout");
_Static_assert(offsetof(apxchol_stats, backend) == 4, "apxchol_stats layout");
_Static_assert(offsetof(apxchol_stats, n) == 8, "apxchol_stats layout");
_Static_assert(offsetof(apxchol_stats, factor_nnz) == 16, "apxchol_stats layout");
_Static_assert(offsetof(apxchol_stats, lumped_offdiag) == 24, "apxchol_stats layout");
_Static_assert(offsetof(apxchol_stats, sddm) == 32, "apxchol_stats layout");
_Static_assert(offsetof(apxchol_stats, factor_storage) == 36, "apxchol_stats layout");

_Static_assert(APXCHOL_INDEX_INT32 == 32 && APXCHOL_INDEX_INT64 == 64, "index types");
_Static_assert(sizeof(apxchol_csc_view) == 48, "apxchol_csc_view layout");
_Static_assert(offsetof(apxchol_csc_view, n) == 0, "apxchol_csc_view layout");
_Static_assert(offsetof(apxchol_csc_view, colptr) == 8, "apxchol_csc_view layout");
_Static_assert(offsetof(apxchol_csc_view, rowval) == 16, "apxchol_csc_view layout");
_Static_assert(offsetof(apxchol_csc_view, nzval) == 24, "apxchol_csc_view layout");
_Static_assert(offsetof(apxchol_csc_view, colptr_type) == 32, "apxchol_csc_view layout");
_Static_assert(offsetof(apxchol_csc_view, rowval_type) == 36, "apxchol_csc_view layout");
_Static_assert(offsetof(apxchol_csc_view, index_base) == 40, "apxchol_csc_view layout");
_Static_assert(sizeof(apxchol_const_dense_view) == 24, "const dense view layout");
_Static_assert(sizeof(apxchol_dense_view) == 24, "dense view layout");
_Static_assert(offsetof(apxchol_dense_view, row_stride) == 8, "dense view layout");
_Static_assert(offsetof(apxchol_dense_view, column_stride) == 16, "dense view layout");

/* Two RHSs with padded/reversed views and distinct warm-start/output buffers. */
static int solve_path(apxchol_solver* solver, const char* what) {
    const double b[] = {1.0, 2.0, NAN, NAN, 0.0, -3.0, NAN, NAN, -1.0, 1.0, NAN, NAN};
    const double x0[] = {0.25, 0.25, 0.25, 0.25, -0.25, 0.5};
    const double expected[] = {1.0, 0.0, -1.0, 1.0, -1.0, 0.0};
    const double r[] = {2.0, 1.0, 0.0, 4.0, 0.0, 2.0};
    double x[12], z[12];
    for (size_t i = 0; i < 12; ++i) x[i] = z[i] = NAN;
    const apxchol_const_dense_view rhs = {b, 4, 1}, initial = {x0 + 2, -1, 3}, residual = {r, 1, 3};
    const apxchol_dense_view solution = {x + 8, -4, 1}, applied = {z + 1, 1, 6};
    char error[256];
    apxchol_solve_info info[2] = {{-1, NAN, -1}, {-1, NAN, -1}};
    const apxchol_status status =
        apxchol_solver_solve(solver, 2, &rhs, &initial, &solution, 1e-10, 200, info, sizeof info[0],
                             error, sizeof error);
    if (status != APXCHOL_STATUS_SUCCESS) {
        fprintf(stderr, "%s solve: %d %s\n", what, (int)status, error);
        return 1;
    }
    if (apxchol_solver_apply(solver, 2, &residual, &applied, error, sizeof error) !=
            APXCHOL_STATUS_SUCCESS) {
        fprintf(stderr, "%s apply: %s\n", what, error);
        return 1;
    }
    for (size_t col = 0; col < 2; ++col) {
        double x_sum = 0.0, z_sum = 0.0;
        if (info[col].converged != 1 || info[col].iterations < 0 ||
            !isfinite(info[col].relative_residual) || info[col].relative_residual >= 1e-10) {
            fprintf(stderr, "%s solve info column %zu\n", what, col);
            return 1;
        }
        for (size_t row = 0; row < 3; ++row) {
            const size_t i = 3 * col + row;
            const size_t xi = 8 - 4 * row + col, zi = 1 + row + 6 * col;
            if (!isfinite(x[xi]) || fabs(x[xi] - expected[i]) > 1e-8 || !isfinite(z[zi])) {
                fprintf(stderr, "%s column %zu row %zu: x=%g z=%g\n", what, col, row, x[xi], z[zi]);
                return 1;
            }
            x_sum += x[xi];
            z_sum += z[zi];
        }
        if (fabs(x_sum) > 1e-8 || fabs(z_sum) > 1e-8) {
            fprintf(stderr, "%s column %zu is not centered\n", what, col);
            return 1;
        }
    }
    for (size_t i = 0; i < 12; ++i) {
        if (i % 4 >= 2 && !isnan(x[i])) return 1;
        if ((i % 6 == 0 || i % 6 >= 4) && !isnan(z[i])) return 1;
    }
    return 0;
}

int main(void) {
    /* Path Laplacian on 3 vertices, 1-based CSC. */
    const int32_t colptr[] = {1, 3, 6, 8};
    const int64_t wide_colptr[] = {1, 3, 6, 8};
    const int32_t narrow_rowval[] = {1, 2, 1, 2, 3, 2, 3};
    const int64_t rowval[] = {1, 2, 1, 2, 3, 2, 3};
    const double nzval[] = {1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0};
    char error[256];
    apxchol_csc_view matrix = {3, colptr, rowval, nzval, APXCHOL_INDEX_INT32, APXCHOL_INDEX_INT64, 1};

    apxchol_options options;
    if (apxchol_c_abi_version() != APXCHOL_C_ABI_VERSION) return 1;
    if (!apxchol_backend_available(APXCHOL_BACKEND_CPU)) return 1;
    if (apxchol_options_default(&options, sizeof options) != APXCHOL_STATUS_SUCCESS) return 1;
    if (options.factor_storage != APXCHOL_FACTOR_STORAGE_AUTO ||
        options.backend != APXCHOL_BACKEND_AUTO) return 1;
    options.threads = 1;
    options.factor_storage = APXCHOL_FACTOR_STORAGE_FLOAT32;

    apxchol_solver* solver = NULL;
    if (apxchol_solver_create(&matrix, NULL, &options, &solver, error,
                              sizeof error) != APXCHOL_STATUS_SUCCESS) {
        fprintf(stderr, "create: %s\n", error);
        return 1;
    }
    const int solved = solve_path(solver, "create");
    apxchol_solver_destroy(solver);
    if (solved != 0) return 1;

    /* A factor handle: its stats size the export, and a solver adopts it. */
    apxchol_factor* factor = NULL;
    matrix.colptr = wide_colptr;
    matrix.colptr_type = APXCHOL_INDEX_INT64;
    matrix.rowval = narrow_rowval;
    matrix.rowval_type = APXCHOL_INDEX_INT32;
    if (apxchol_factor_create(&matrix, &options, &factor, error,
                              sizeof error) != APXCHOL_STATUS_SUCCESS) {
        fprintf(stderr, "factor_create: %s\n", error);
        return 1;
    }
    apxchol_stats stats;
    stats.struct_size = sizeof stats;
    int64_t lcolptr[4], lrowval[16], perm[3];
    float lnzval[16];
    if (apxchol_factor_stats(factor, &stats) != APXCHOL_STATUS_SUCCESS || stats.n != 3 ||
        stats.sddm != 0 || stats.factor_nnz < 3 || stats.factor_nnz > 16 ||
        stats.factor_storage != APXCHOL_FACTOR_STORAGE_FLOAT32 ||
        apxchol_factor_export(factor, 0, lcolptr, lrowval, lnzval, perm, error,
                              sizeof error) != APXCHOL_STATUS_SUCCESS ||
        lcolptr[3] != stats.factor_nnz) {
        fprintf(stderr, "factor stats/export: %s\n", error);
        apxchol_factor_destroy(factor);
        return 1;
    }
    solver = NULL;
    options.keep_factor_values = 1;
    const apxchol_status adopted = apxchol_solver_create(
        &matrix, factor, &options, &solver, error, sizeof error);
    apxchol_factor_destroy(factor); /* the solver holds its own copy */
    if (adopted != APXCHOL_STATUS_SUCCESS) {
        fprintf(stderr, "create with factor: %s\n", error);
        return 1;
    }
    int failed = solve_path(solver, "create with factor");
    if (apxchol_solver_stats(solver, &stats) != APXCHOL_STATUS_SUCCESS ||
        stats.factor_storage != APXCHOL_FACTOR_STORAGE_FLOAT32) failed = 1;
    if (apxchol_solver_export_factor(solver, 0, lcolptr, lrowval, lnzval, perm,
                                     error, sizeof error) != APXCHOL_STATUS_SUCCESS ||
        lcolptr[3] != stats.factor_nnz) failed = 1;
    /* The solver's retained factor, copied back out into a new handle. */
    factor = NULL;
    if (apxchol_solver_copy_factor(solver, &factor, error, sizeof error) !=
            APXCHOL_STATUS_SUCCESS ||
        apxchol_factor_stats(factor, &stats) != APXCHOL_STATUS_SUCCESS || stats.n != 3) {
        fprintf(stderr, "copy_factor: %s\n", error);
        failed = 1;
    }
    apxchol_factor_destroy(factor);
    apxchol_solver_destroy(solver);
    if (failed != 0) return 1;
    printf("apxchol %s, C ABI %d: ok\n", apxchol_version(), (int)apxchol_c_abi_version());
    return 0;
}
