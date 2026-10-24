#ifndef APXCHOL_C_API_H
#define APXCHOL_C_API_H
/* Exception-safe C ABI for apxchol (pure C99; the `apxchol_c` CMake target).
 *
 * Every fallible call returns an apxchol_status. SUCCESS (0) and
 * NOT_CONVERGED (1) both write every requested output: NOT_CONVERGED means the
 * solution and info were written but the relative residual is not below tol.
 * It is never a relaxed acceptance. Every other status is an error: a call
 * that returns a new handle sets it to NULL. A mismatched info_size leaves
 * the whole info array untouched. Validated info and stats outputs are zeroed
 * on later errors (stats keeps struct_size). Other outputs are unspecified.
 * When error_message is non-NULL and
 * error_capacity > 0, a NUL-terminated (possibly truncated) message is
 * written ("" on SUCCESS).
 *
 * No C++ exception crosses this boundary. Conditions that terminate the
 * process instead of returning a status: a factor or pool offset beyond
 * edge_index (2^32-1 by default; rebuild with APXCHOL_64BIT_EDGE_INDICES=ON)
 * aborts, and an exception escaping an OpenMP region calls std::terminate.
 *
 * Input: a square CSC matrix with int64 colptr[n+1] and rowval/nzval[nnz],
 * nnz = colptr[n] - index_base, index_base 0 or 1 for both index arrays. Rows
 * may be unsorted; duplicates are summed. The matrix must satisfy the
 * operator contract of apxchol/operator_class.h (Laplacian, SDDM or a
 * lumpable M-matrix); it is validated, never reinterpreted. Limits in every
 * build: n <= 2^31-1 and nnz <= 2^31-1.
 *
 * Laplacian systems return the min-norm solution (unless an SDDM factor is
 * adopted, see apxchol_solver_create); the right-hand side must be
 * compatible with every connected component (it is not projected).
 *
 * A solver handle is not safe for overlapping calls; distinct handles may be
 * used concurrently from different threads. A factor handle is immutable: the
 * calls that read it may overlap, but destroy must not overlap any of them.
 * For solve, x must not overlap b or x0; for apply, z must not overlap r.
 * Exact and partial overlap both return INVALID_ARGUMENT. b and x0 may overlap.
 * info must not overlap any dense input or output buffer.
 *
 * Versioning: options and stats carry struct_size; solve takes info_size.
 * The caller supplies sizeof(struct) from its header. This library accepts
 * the sizes of its own header; a larger (newer) size returns UNSUPPORTED and a
 * smaller one INVALID_ARGUMENT. */

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define APXCHOL_C_API __attribute__((visibility("default")))
#else
#define APXCHOL_C_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define APXCHOL_C_ABI_VERSION 2

typedef int32_t apxchol_status;
enum {
    APXCHOL_STATUS_SUCCESS = 0,
    APXCHOL_STATUS_NOT_CONVERGED = 1,
    APXCHOL_STATUS_INVALID_ARGUMENT = 2, /* pointers, sizes, CSC, option values */
    APXCHOL_STATUS_INVALID_OPERATOR = 3, /* the operator contract rejected A */
    APXCHOL_STATUS_NO_FACTOR_VALUES = 4, /* factor export without keep_factor_values */
    APXCHOL_STATUS_UNSUPPORTED = 5,      /* backend not built/available; newer struct; an
                                            operator the backend cannot represent; FP16
                                            storage on a CPU route without FP16 support */
    APXCHOL_STATUS_OUT_OF_MEMORY = 6,    /* host or device memory */
    APXCHOL_STATUS_INTERNAL_ERROR = 7
};

typedef int32_t apxchol_backend;
/* GPU means the device backend compiled into the library (CUDA or Metal).
 * AUTO follows apxchol::solver's input, option and device compatibility policy;
 * adopting a factor with AUTO selects CPU. Once selected, a route never retries
 * on another backend. CUDA cannot adopt a host factor and returns UNSUPPORTED.
 * stats.backend reports the route chosen, never AUTO. */
enum { APXCHOL_BACKEND_CPU = 0, APXCHOL_BACKEND_GPU = 1, APXCHOL_BACKEND_AUTO = 2 };

typedef int32_t apxchol_sampler;
enum { APXCHOL_SAMPLER_GKS = 0, APXCHOL_SAMPLER_TRACE_CYCLE = 1 };

typedef int32_t apxchol_partitioner;
enum {
    APXCHOL_PARTITIONER_BLOCK_GREEDY = 0,
    APXCHOL_PARTITIONER_PRIORITY_GREEDY = 1,
    APXCHOL_PARTITIONER_BAUMANN_KYNG = 2
};

typedef int32_t apxchol_factor_storage;
enum {
    APXCHOL_FACTOR_STORAGE_AUTO = 0,
    APXCHOL_FACTOR_STORAGE_FLOAT16 = 16,
    APXCHOL_FACTOR_STORAGE_FLOAT32 = 32
};

/* 56 bytes. Initialize with apxchol_options_default, then override fields. */
typedef struct apxchol_options {
    uint32_t struct_size;
    apxchol_backend backend;
    double tol;                       /* relative residual target, finite > 0 */
    double degree_quantile;           /* -1: chosen by route; otherwise [0, 1) */
    int32_t max_iter;                 /* >= 0 */
    int32_t stagnation_window;        /* >= 0; 0 disables the stagnation stop */
    uint32_t seed;
    int32_t threads;                  /* > 0: call-scoped worker ceiling; 0: inherit */
    apxchol_sampler sampler;
    apxchol_partitioner partitioner;
    int32_t keep_factor_values;       /* 0/1; factor value export requires 1 */
    apxchol_factor_storage factor_storage; /* triangular-solve storage;
                                              AUTO uses the backend default */
} apxchol_options;

/* 24 bytes; pass sizeof(apxchol_solve_info) as info_size to solve. */
typedef struct apxchol_solve_info {
    int64_t iterations;
    double relative_residual;         /* ||b - A x|| / ||b|| of the returned x,
                                         recomputed in fp64 from the original
                                         operator on every backend */
    int32_t converged;                /* 1 iff relative_residual < tol */
} apxchol_solve_info;

/* 40 bytes; the caller sets struct_size. */
typedef struct apxchol_stats {
    uint32_t struct_size;
    apxchol_backend backend;          /* the route chosen: CPU or GPU */
    int64_t n;
    int64_t factor_nnz;               /* stored entries of L, diagonal included */
    int64_t lumped_offdiag;           /* positive off-diagonals lumped (2 per pair) */
    int32_t sddm;                     /* the factor's class: 1 SDDM, 0 Laplacian
                                         (min-norm solutions) */
    apxchol_factor_storage factor_storage; /* resolved solver storage; FLOAT32 for a host factor */
} apxchol_stats;

typedef struct apxchol_solver apxchol_solver;
typedef struct apxchol_factor apxchol_factor;

APXCHOL_C_API const char* apxchol_version(void); /* static "<version>+<git sha>" */
APXCHOL_C_API int32_t apxchol_c_abi_version(void);
/* 1 if this build has the backend and (for GPU) a usable device; AUTO: 1. */
APXCHOL_C_API int32_t apxchol_backend_available(apxchol_backend backend);

/* Writes the library defaults (those of apxchol::solve_options) and sets
 * options->struct_size. struct_size is the caller's sizeof(apxchol_options). */
APXCHOL_C_API apxchol_status apxchol_options_default(apxchol_options* options,
                                                     size_t struct_size);

/* Builds a reusable solver. factor NULL factorizes A; otherwise the solver
 * adopts its own copy of factor, which may come from a different operator.
 * The dimension must match. A Laplacian factor cannot precondition an SDDM
 * operator (INVALID_ARGUMENT); an SDDM factor may precondition a Laplacian,
 * whose solution is then not centred. Adopted factorization options are
 * validated but unused. options NULL selects defaults. */
APXCHOL_C_API apxchol_status apxchol_solver_create(
    int64_t n, const int64_t* colptr, const int64_t* rowval, const double* nzval,
    int32_t index_base, const apxchol_factor* factor, const apxchol_options* options,
    apxchol_solver** out_solver, char* error_message, size_t error_capacity);
APXCHOL_C_API void apxchol_solver_destroy(apxchol_solver* solver); /* NULL is a no-op */

/* nrhs right-hand sides, column-major n*nrhs. nrhs = 0 is a no-op requiring
 * a valid solver; dense-array and info pointers are optional and info_size is
 * ignored. Negative or overflowing dimensions are INVALID_ARGUMENT.
 * Finite tol < 0 / max_iter < 0
 * select the handle's options. x0 NULL starts from zero. info is NULL or an
 * array of nrhs structs, one per column. info_size is ignored when info is NULL.
 * Native bounded-panel solves reuse
 * the factor and workspace; one column uses the native scalar kernels.
 * All b and x0 values are checked before x is written. Returns
 * NOT_CONVERGED if any column did not converge. The backend rescales extreme
 * inputs internally; CPU inputs that cannot share a usable scale are
 * INVALID_ARGUMENT. A solution outside the representable range is UNSUPPORTED. */
APXCHOL_C_API apxchol_status apxchol_solver_solve(
    apxchol_solver* solver, int64_t nrhs, const double* b, const double* x0, double* x,
    double tol, int32_t max_iter, apxchol_solve_info* info, size_t info_size,
    char* error_message, size_t error_capacity);

/* Apply z = M^{-1} r to nrhs columns, with the same layout and dimension rules
 * as solve. All r values are checked before any output is written. It is a
 * fixed linear map; input and output are centred on every call when the
 * factor is Laplacian (stats.sddm == 0). */
APXCHOL_C_API apxchol_status apxchol_solver_apply(
    apxchol_solver* solver, int64_t nrhs, const double* r, double* z,
    char* error_message, size_t error_capacity);

APXCHOL_C_API apxchol_status apxchol_solver_stats(const apxchol_solver* solver,
                                                  apxchol_stats* stats);

/* The lower factor L (CSC in the permuted space, diagonal first in each
 * column, native fp32 values) and perm[v] = elimination position of vertex v,
 * with index_base applied to every index: A[p,p] ~= L*L^T for p = inv(perm).
 * Array lengths: colptr[n+1], rowval/nzval[factor_nnz], perm[n]. Passing
 * colptr = rowval = nzval = NULL exports perm only, which is always
 * available; L itself requires keep_factor_values = 1. */
APXCHOL_C_API apxchol_status apxchol_solver_export_factor(
    const apxchol_solver* solver, int32_t index_base,
    int64_t* colptr, int64_t* rowval, float* nzval, int64_t* perm,
    char* error_message, size_t error_capacity);

/* Factor handles: one host factorization (L, perm, statistics), always with
 * its values, that any number of solvers can adopt. A solver adopts a copy,
 * so the factor and its solvers may be destroyed in any order.
 * apxchol_factor_create builds on the CPU; apxchol_solver_create builds on
 * the selected backend. */

/* Factorizes A without building a solver: the same validation, lumping and
 * call-scoped threads as apxchol_solver_create, with the factorization
 * options (seed, sampler, partitioner, degree_quantile). The other options are validated
 * and unused; keep_factor_values is ignored. options NULL = defaults. */
APXCHOL_C_API apxchol_status apxchol_factor_create(
    int64_t n, const int64_t* colptr, const int64_t* rowval, const double* nzval,
    int32_t index_base, const apxchol_options* options,
    apxchol_factor** out_factor, char* error_message, size_t error_capacity);
APXCHOL_C_API void apxchol_factor_destroy(apxchol_factor* factor); /* NULL is a no-op */

/* The factor's statistics (lumped_offdiag: of the matrix it was built from)
 * with backend APXCHOL_BACKEND_CPU. */
APXCHOL_C_API apxchol_status apxchol_factor_stats(const apxchol_factor* factor,
                                                  apxchol_stats* stats);

/* As apxchol_solver_export_factor; L is always available. */
APXCHOL_C_API apxchol_status apxchol_factor_export(
    const apxchol_factor* factor, int32_t index_base,
    int64_t* colptr, int64_t* rowval, float* nzval, int64_t* perm,
    char* error_message, size_t error_capacity);

/* Copies the solver's factor into a new factor handle. Requires
 * keep_factor_values = 1 (NO_FACTOR_VALUES otherwise). */
APXCHOL_C_API apxchol_status apxchol_solver_copy_factor(
    const apxchol_solver* solver, apxchol_factor** out_factor,
    char* error_message, size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif /* APXCHOL_C_API_H */
