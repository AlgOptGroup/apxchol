#ifndef APXCHOL_C_API_H
#define APXCHOL_C_API_H
/* Exception-safe C ABI for apxchol (pure C99; the `apxchol_c` CMake target).
 *
 * Every fallible call returns an apxchol_status. SUCCESS (0) and
 * NOT_CONVERGED (1) both write every requested output: NOT_CONVERGED means the
 * solution and info were written but the relative residual is not below tol.
 * It is never a relaxed acceptance. Every other status is an error: a call
 * that returns a new handle sets it to NULL, an info or stats output whose
 * struct_size matches is zeroed except struct_size (one whose struct_size
 * does not match is left untouched and decides the status), and other
 * outputs are unspecified. When error_message is non-NULL and
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
 * adopted, see apxchol_solver_create_from_factor); the right-hand side must be
 * compatible with every connected component (it is not projected).
 *
 * A solver handle is not safe for overlapping calls; distinct handles may be
 * used concurrently from different threads. A factor handle is immutable: the
 * calls that read it may overlap, but destroy must not overlap any of them.
 * b, x0 and x (and r, z) may be the same pointer; partial overlap is
 * undefined.
 *
 * Versioning: every struct carries struct_size, which the caller sets to
 * sizeof(struct) from the header it compiled against. This library accepts
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

#define APXCHOL_C_ABI_VERSION 1

typedef int32_t apxchol_status;
enum {
    APXCHOL_STATUS_SUCCESS = 0,
    APXCHOL_STATUS_NOT_CONVERGED = 1,
    APXCHOL_STATUS_INVALID_ARGUMENT = 2, /* pointers, sizes, CSC, option values */
    APXCHOL_STATUS_INVALID_OPERATOR = 3, /* the operator contract rejected A */
    APXCHOL_STATUS_NO_FACTOR_VALUES = 4, /* factor export without keep_factor_values */
    APXCHOL_STATUS_UNSUPPORTED = 5,      /* backend not built/available; newer struct; an
                                            operator the backend cannot represent */
    APXCHOL_STATUS_OUT_OF_MEMORY = 6,    /* host or device memory */
    APXCHOL_STATUS_INTERNAL_ERROR = 7
};

typedef int32_t apxchol_backend;
enum { APXCHOL_BACKEND_CPU = 0, APXCHOL_BACKEND_METAL = 1 };

typedef int32_t apxchol_sampler;
enum { APXCHOL_SAMPLER_GKS = 0, APXCHOL_SAMPLER_TRACE_CYCLE = 1 };

typedef int32_t apxchol_partitioner;
enum {
    APXCHOL_PARTITIONER_BLOCK_GREEDY = 0,
    APXCHOL_PARTITIONER_PRIORITY_GREEDY = 1,
    APXCHOL_PARTITIONER_BAUMANN_KYNG = 2
};

/* Values of apxchol::graph_storage; the retired values 1 and 3 are rejected. */
typedef int32_t apxchol_storage;
enum {
    APXCHOL_STORAGE_VEC = 0,
    APXCHOL_STORAGE_BSTR = 2,
    APXCHOL_STORAGE_VEC_POOL_AOS = 4
};

/* 72 bytes. Initialize with apxchol_options_default, then override fields. */
typedef struct apxchol_options {
    uint32_t struct_size;
    apxchol_backend backend;
    double tol;                       /* relative residual target, finite > 0 */
    int32_t max_iter;                 /* >= 0 */
    int32_t stagnation_window;        /* >= 0; 0 disables the stagnation stop */
    uint32_t seed;
    int32_t threads;                  /* > 0: call-scoped OpenMP team; 0: inherit */
    apxchol_sampler sampler;
    apxchol_partitioner partitioner;
    apxchol_storage storage;
    int32_t keep_factor_values;       /* 0/1; factor value export requires 1 */
    double degree_quantile;           /* < 0: chosen by route (factor_options.h) */
    uint64_t exact_clique_max_degree;
    uint64_t omp_threshold;           /* min active vertices before the factorization
                                         engages OpenMP, and the selection size kept
                                         whatever its yield (factor_options.h); 0:
                                         OpenMP at every size, that safeguard off.
                                         A set APXCHOL_OMP_THRESHOLD overrides it,
                                         as for every apxchol entry point */
} apxchol_options;

/* 32 bytes; the caller sets struct_size. */
typedef struct apxchol_solve_info {
    uint32_t struct_size;
    int32_t converged;                /* 1 iff relative_residual < tol */
    int64_t iterations;
    double relative_residual;         /* at exit: CPU the PCG recursion's; METAL
                                         ||b - A x|| / ||b|| recomputed in fp64 */
    double solve_seconds;
} apxchol_solve_info;

/* 64 bytes; the caller sets struct_size. */
typedef struct apxchol_stats {
    uint32_t struct_size;
    apxchol_backend backend;
    int64_t n;
    int64_t factor_nnz;               /* stored entries of L, diagonal included */
    int64_t lumped_offdiag;           /* positive off-diagonals lumped (2 per pair) */
    int64_t rounds;
    int64_t peak_graph_bytes;
    double setup_seconds;             /* wall time of the call that created the handle */
    int32_t sddm;                     /* the factor's class: 1 SDDM, 0 Laplacian
                                         (min-norm solutions) */
    int32_t setup_max_threads;        /* OpenMP team limit during setup; 1 if serial */
} apxchol_stats;

typedef struct apxchol_solver apxchol_solver;
typedef struct apxchol_factor apxchol_factor;

APXCHOL_C_API const char* apxchol_version(void); /* static "<version>+<git sha>" */
APXCHOL_C_API int32_t apxchol_c_abi_version(void);
APXCHOL_C_API int32_t apxchol_openmp_enabled(void);
/* omp_get_max_threads() of the calling thread; 1 in a serial build. */
APXCHOL_C_API int32_t apxchol_get_max_threads(void);
/* 1 if this build has the backend and (for METAL) a usable device. */
APXCHOL_C_API int32_t apxchol_backend_available(apxchol_backend backend);

/* Writes the library defaults (those of apxchol::solve_options) and sets
 * options->struct_size. struct_size is the caller's sizeof(apxchol_options). */
APXCHOL_C_API apxchol_status apxchol_options_default(apxchol_options* options,
                                                     size_t struct_size);

/* Factorizes A and builds the reusable solver. options NULL = defaults. */
APXCHOL_C_API apxchol_status apxchol_solver_create(
    int64_t n, const int64_t* colptr, const int64_t* rowval, const double* nzval,
    int32_t index_base, const apxchol_options* options,
    apxchol_solver** out_solver, char* error_message, size_t error_capacity);
APXCHOL_C_API void apxchol_solver_destroy(apxchol_solver* solver); /* NULL is a no-op */

/* One right-hand side. tol < 0 / max_iter < 0 select the handle's options;
 * x0 NULL starts from zero. info is nullable. */
APXCHOL_C_API apxchol_status apxchol_solver_solve(
    apxchol_solver* solver, const double* b, const double* x0, double* x,
    double tol, int32_t max_iter, apxchol_solve_info* info,
    char* error_message, size_t error_capacity);

/* k right-hand sides, column-major n*k. x0 NULL starts every column from zero.
 * The per-column outputs are nullable arrays of length k. Every column uses
 * the single-solve stopping rule; the CPU backend solves columns in order,
 * METAL advances up to 64 columns in lockstep; on both, every column is
 * bit-identical to its single solve. Returns NOT_CONVERGED if any column did
 * not converge. */
APXCHOL_C_API apxchol_status apxchol_solver_solve_block(
    apxchol_solver* solver, int64_t k, const double* b, const double* x0, double* x,
    double tol, int32_t max_iter, int64_t* iterations, double* relative_residuals,
    int32_t* converged, char* error_message, size_t error_capacity);

/* One preconditioner application z = M^{-1} r. With an SDDM factor
 * (stats.sddm = 1) it is a fixed linear map. With a Laplacian factor z is
 * defined up to a constant vector: METAL centres input and output on every
 * application, so the same r gives the same z; CPU centres them only on
 * every K-th application of the solver's count (APXCHOL_GROUND=center-k,
 * K = APXCHOL_CENTER_K, default 10; every solve restarts the count), so for
 * a mean-zero r repeated applications agree up to a constant vector. */
APXCHOL_C_API apxchol_status apxchol_solver_apply(
    apxchol_solver* solver, const double* r, double* z,
    char* error_message, size_t error_capacity);

APXCHOL_C_API apxchol_status apxchol_solver_stats(const apxchol_solver* solver,
                                                  apxchol_stats* stats);

/* The lower factor L (CSC in the permuted space, diagonal first in each
 * column, fp32 values widened) and perm[v] = elimination position of vertex v,
 * with index_base applied to every index: A[p,p] ~= L*L^T for p = inv(perm).
 * Array lengths: colptr[n+1], rowval/nzval[factor_nnz], perm[n]. Passing
 * colptr = rowval = nzval = NULL exports perm only, which is always
 * available; L itself requires keep_factor_values = 1. */
APXCHOL_C_API apxchol_status apxchol_solver_export_factor(
    const apxchol_solver* solver, int32_t index_base,
    int64_t* colptr, int64_t* rowval, double* nzval, int64_t* perm,
    char* error_message, size_t error_capacity);

/* Factor handles: one host factorization (L, perm, statistics), always with
 * its values, that any number of solvers can adopt. A solver adopts a copy,
 * so the factor and its solvers may be destroyed in any order.
 * apxchol_solver_create is apxchol_factor_create followed by
 * apxchol_solver_create_from_factor on the same matrix and options. */

/* Factorizes A without building a solver: the same validation, lumping and
 * call-scoped threads as apxchol_solver_create, with the factorization
 * options (seed, sampler, partitioner, storage, degree_quantile,
 * exact_clique_max_degree, omp_threshold). The other options are validated
 * and unused; keep_factor_values is ignored. options NULL = defaults. */
APXCHOL_C_API apxchol_status apxchol_factor_create(
    int64_t n, const int64_t* colptr, const int64_t* rowval, const double* nzval,
    int32_t index_base, const apxchol_options* options,
    apxchol_factor** out_factor, char* error_message, size_t error_capacity);
APXCHOL_C_API void apxchol_factor_destroy(apxchol_factor* factor); /* NULL is a no-op */

/* The factor's statistics (lumped_offdiag: of the matrix it was built from)
 * with backend APXCHOL_BACKEND_CPU. setup_seconds and setup_max_threads are
 * those of apxchol_factor_create, or of the solver it was copied from. */
APXCHOL_C_API apxchol_status apxchol_factor_stats(const apxchol_factor* factor,
                                                  apxchol_stats* stats);

/* As apxchol_solver_export_factor; L is always available. */
APXCHOL_C_API apxchol_status apxchol_factor_export(
    const apxchol_factor* factor, int32_t index_base,
    int64_t* colptr, int64_t* rowval, double* nzval, int64_t* perm,
    char* error_message, size_t error_capacity);

/* Builds a solver for A preconditioned by a copy of factor instead of
 * factorizing. A is validated as in apxchol_solver_create and may differ
 * from the matrix the factor was built from (a stale factor for a nearby
 * operator), but must have its dimension (INVALID_ARGUMENT otherwise). A
 * Laplacian factor (rank n-1, centred) cannot precondition an SDDM operator
 * (INVALID_ARGUMENT); an SDDM factor may precondition a Laplacian, whose
 * solution is then not centred. backend, tol, max_iter, stagnation_window,
 * threads and keep_factor_values apply as in create; the factorization
 * options are validated and unused. Stats are the factor's, with this call's
 * setup_seconds and setup_max_threads. On another operator convergence
 * depends on its distance from the factor's; NOT_CONVERGED reports it. */
APXCHOL_C_API apxchol_status apxchol_solver_create_from_factor(
    int64_t n, const int64_t* colptr, const int64_t* rowval, const double* nzval,
    int32_t index_base, const apxchol_factor* factor, const apxchol_options* options,
    apxchol_solver** out_solver, char* error_message, size_t error_capacity);

/* Copies the solver's factor into a new factor handle. Requires
 * keep_factor_values = 1 (NO_FACTOR_VALUES otherwise). */
APXCHOL_C_API apxchol_status apxchol_solver_copy_factor(
    const apxchol_solver* solver, apxchol_factor** out_factor,
    char* error_message, size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif /* APXCHOL_C_API_H */
