// C ABI over apxchol::solver, with reusable host-factor handles.
// See include/apxchol/c_api.h for the contract: statuses, struct versioning,
// aliasing and the abort paths.
#include "apxchol/c_api.h"

#include "apxchol/operator_class.h"
#include "apxchol/detail/csc_import.h"
#include "apxchol/lowprec.h"
#include "apxchol/solver/detail/thread_budget.h"
#include "apxchol/solver/detail/solve_backend.h"
#include "apxchol/solver/factorization.h"
#include "apxchol/solver/solve.h"
#include "apxchol/solver/sptrsv/omp.h"
#include "apxchol/version.h"
#if defined(APXCHOL_USE_CUDA)
#include "apxchol/solver/elimination/gpu_round_shadow.h"
#elif defined(APXCHOL_USE_METAL)
#include "apxchol/solver/metal_solver.h"
#endif

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace {

/// A validation failure with its own status (never re-classified).
struct c_api_error : std::runtime_error {
    apxchol_status status;
    c_api_error(apxchol_status s, const std::string& message)
        : std::runtime_error(message), status(s) {}
};

[[noreturn]] void fail(apxchol_status status, const std::string& message) {
    throw c_api_error(status, message);
}

void require(bool condition, const char* message) {
    if (!condition) fail(APXCHOL_STATUS_INVALID_ARGUMENT, message);
}

void write_message(char* buffer, std::size_t capacity, const char* message) noexcept {
    if (buffer == nullptr || capacity == 0) return;
    std::size_t length = std::strlen(message);
    if (length > capacity - 1) length = capacity - 1;
    std::memcpy(buffer, message, length);
    buffer[length] = '\0';
}

/// Runs `body` and maps every exception to a status. During create, a
/// std::invalid_argument comes from the operator contract (the options were
/// validated before); elsewhere it is an argument error.
template <class Body>
apxchol_status guarded(char* message, std::size_t capacity, bool creating,
                       Body&& body) noexcept {
    write_message(message, capacity, "");
    try {
        return body();
    } catch (const c_api_error& e) {
        write_message(message, capacity, e.what());
        return e.status;
    } catch (const std::bad_alloc& e) {
        // A plain std::bad_alloc only says "std::bad_alloc"; the Metal
        // device's names the buffer. Neither path allocates.
        write_message(message, capacity,
                      typeid(e) == typeid(std::bad_alloc) ? "out of memory" : e.what());
        return APXCHOL_STATUS_OUT_OF_MEMORY;
    } catch (const std::invalid_argument& e) {
        write_message(message, capacity, e.what());
        return creating ? APXCHOL_STATUS_INVALID_OPERATOR
                        : APXCHOL_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        write_message(message, capacity, e.what());
        return APXCHOL_STATUS_INTERNAL_ERROR;
    } catch (...) {
        write_message(message, capacity, "unknown C++ exception");
        return APXCHOL_STATUS_INTERNAL_ERROR;
    }
}

/// struct_size check shared by every versioned struct.
apxchol_status check_struct_size(std::size_t given, std::size_t expected) noexcept {
    if (given < expected) return APXCHOL_STATUS_INVALID_ARGUMENT;
    if (given > expected) return APXCHOL_STATUS_UNSUPPORTED;
    return APXCHOL_STATUS_SUCCESS;
}

void require_struct_size(std::size_t given, std::size_t expected, const char* what) {
    const apxchol_status s = check_struct_size(given, expected);
    if (s == APXCHOL_STATUS_INVALID_ARGUMENT)
        fail(s, std::string(what) + ".struct_size is smaller than this library's struct");
    if (s == APXCHOL_STATUS_UNSUPPORTED)
        fail(s, std::string(what) + ".struct_size is newer than this library supports");
}

apxchol_options default_options() {
    const apxchol::solve_options so;
    apxchol_options o{};
    o.struct_size = sizeof(apxchol_options);
    o.backend = so.backend == apxchol::solve_backend::cpu ? APXCHOL_BACKEND_CPU
              : so.backend == apxchol::solve_backend::gpu ? APXCHOL_BACKEND_GPU
                                                            : APXCHOL_BACKEND_AUTO;
    o.seed = so.factor_opts.seed;
    o.threads = so.num_threads;
    o.keep_factor_values = so.keep_factor_values ? 1 : 0;
    o.factor_storage = APXCHOL_FACTOR_STORAGE_AUTO;
    return o;
}

/// The caller's options (struct_size checked before anything else is read, so
/// an older, smaller struct is never read past its end), or the defaults.
apxchol_options read_options(const apxchol_options* options) {
    if (options == nullptr) return default_options();
    require_struct_size(options->struct_size, sizeof(apxchol_options), "options");
    return *options;
}

/// Validates every option before any work, so a later std::invalid_argument
/// during create can only come from the operator contract.
apxchol::solve_options to_solve_options(const apxchol_options& o) {
    require(o.backend == APXCHOL_BACKEND_CPU || o.backend == APXCHOL_BACKEND_GPU ||
                o.backend == APXCHOL_BACKEND_AUTO,
            "options.backend must be APXCHOL_BACKEND_CPU, GPU or AUTO");
    require(o.keep_factor_values == 0 || o.keep_factor_values == 1,
            "options.keep_factor_values must be 0 or 1");
    require(o.factor_storage == APXCHOL_FACTOR_STORAGE_AUTO ||
                o.factor_storage == APXCHOL_FACTOR_STORAGE_FLOAT16 ||
                o.factor_storage == APXCHOL_FACTOR_STORAGE_FLOAT32,
            "options.factor_storage must be AUTO, FLOAT16 or FLOAT32");
    apxchol::solve_options so;
    so.backend = o.backend == APXCHOL_BACKEND_CPU ? apxchol::solve_backend::cpu
               : o.backend == APXCHOL_BACKEND_GPU ? apxchol::solve_backend::gpu
                                                    : apxchol::solve_backend::automatic;
    so.factor_opts.factor_storage = o.factor_storage == APXCHOL_FACTOR_STORAGE_AUTO
        ? apxchol::factor_storage_type::automatic
        : o.factor_storage == APXCHOL_FACTOR_STORAGE_FLOAT16
            ? apxchol::factor_storage_type::fp16 : apxchol::factor_storage_type::fp32;
    so.num_threads = o.threads;
    so.collect_timings = false;
    so.keep_factor_values = o.keep_factor_values != 0;
    so.factor_opts.seed = o.seed;
    try {
        apxchol::detail::validate_solve_options(so);
    } catch (const std::invalid_argument& error) {
        fail(APXCHOL_STATUS_INVALID_ARGUMENT, error.what());
    }
    return so;
}

bool gpu_available() {
#if defined(APXCHOL_USE_CUDA)
    return apxchol::detail::gpu_round_shadow_runtime_available();
#elif defined(APXCHOL_USE_METAL)
    return apxchol::metal_solver::available();
#else
    return false;
#endif
}

/// Follow native routing, translating unsupported explicit requests to C statuses.
apxchol::solve_backend resolve_route(const apxchol::solve_options& so,
                                     const Eigen::SparseMatrix<double>& A, bool adopt) {
    if (so.backend == apxchol::solve_backend::cpu ||
        (adopt && so.backend == apxchol::solve_backend::automatic))
        return apxchol::solve_backend::cpu;
    if (so.backend == apxchol::solve_backend::gpu) {
        if (!gpu_available())
            fail(APXCHOL_STATUS_UNSUPPORTED, "no usable GPU backend in this build");
        if (adopt) return apxchol::solve_backend::gpu;
    }
    try {
        return apxchol::detail::select_solve_backend(so, &A);
    } catch (const std::invalid_argument& error) {
        if (so.backend == apxchol::solve_backend::gpu) {
            fail(APXCHOL_STATUS_UNSUPPORTED, error.what());
        }
        throw;
    }
}

/// Reject unsupported CPU FP16 storage before factorization, with its C status.
void require_storage(const apxchol::solve_options& so, apxchol::solve_backend route) {
    const bool fp16 = apxchol::detail::resolve_fp16_storage(so.factor_opts.factor_storage, false);
    if (route == apxchol::solve_backend::cpu && fp16 && !apxchol::omp_sptrsv::fp16_supported())
        fail(APXCHOL_STATUS_UNSUPPORTED,
             "FP16 factor storage on the CPU route requires AVX/F16C or Apple ARM64");
}

template<class Function>
auto with_indices(const void* data, apxchol_index_type type, Function&& function) {
    switch (type) {
        case APXCHOL_INDEX_INT32: return function(static_cast<const std::int32_t*>(data));
        case APXCHOL_INDEX_INT64: return function(static_cast<const std::int64_t*>(data));
        default: fail(APXCHOL_STATUS_INVALID_ARGUMENT, "index type must be INT32 or INT64");
    }
}

Eigen::SparseMatrix<double> import_csc(const apxchol_csc_view* view) {
    require(view != nullptr, "A is NULL");
    require(view->n >= 1, "n must be in [1, 2^31-1]");
    try {
        return with_indices(view->colptr, view->colptr_type, [&](const auto* colptr) {
            return with_indices(view->rowval, view->rowval_type, [&](const auto* rowval) {
                return apxchol::detail::import_csc(view->n, colptr, rowval, view->nzval,
                                                  view->index_base);
            });
        });
    } catch (const std::invalid_argument& error) {
        fail(APXCHOL_STATUS_INVALID_ARGUMENT, error.what());
    }
}

void require_finite(apxchol::const_matrix_view source, const char* what) {
    if (!source.allFinite())
        fail(APXCHOL_STATUS_INVALID_ARGUMENT, std::string(what) + " contains a non-finite value");
}

void require_block_size(Eigen::Index n, std::int64_t nrhs) {
    require(nrhs >= 0, "nrhs must be non-negative");
    require(nrhs <= std::numeric_limits<std::ptrdiff_t>::max() /
                        static_cast<std::int64_t>(sizeof(double)) / n,
            "n * nrhs * sizeof(double) overflows");
}

template<class View>
auto dense_map(const View* view, Eigen::Index rows, Eigen::Index columns) {
    require(view != nullptr && view->data != nullptr, "dense view or data is NULL");
    // Eigen treats a zero Ref stride as a default; inactive axes need no stride.
    const auto row_stride = rows > 1 ? view->row_stride : 1;
    const auto column_stride = columns > 1 ? view->column_stride : 1;
    require(row_stride != 0 && column_stride != 0, "active dense strides must be nonzero");
    const auto magnitude = [](std::int64_t value) -> std::uint64_t {
        return value < 0 ? std::uint64_t(-(value + 1)) + 1 : std::uint64_t(value);
    };
    constexpr auto limit = std::uint64_t(std::numeric_limits<std::ptrdiff_t>::max()) /
                           sizeof(double);
    const auto extent = [&](Eigen::Index size, std::int64_t stride) {
        const auto step = magnitude(stride);
        require(std::uint64_t(size - 1) <= limit / step, "dense stride extent overflows");
        return std::uint64_t(size - 1) * step;
    };
    const auto row_extent = extent(rows, row_stride);
    const auto column_extent = extent(columns, column_stride);
    require(row_extent < limit && column_extent < limit - row_extent,
            "dense address extent overflows");
    const auto before = (row_stride < 0 ? row_extent : 0) +
                        (column_stride < 0 ? column_extent : 0);
    const auto after = row_extent + column_extent - before + 1;
    const auto address = reinterpret_cast<std::uintptr_t>(view->data);
    require(address >= before * sizeof(double) &&
                address <= std::numeric_limits<std::uintptr_t>::max() - after * sizeof(double),
            "dense address extent wraps");
    using Scalar = std::remove_pointer_t<decltype(view->data)>;
    if constexpr (!std::is_const_v<Scalar>) {
        if (rows > 1 && columns > 1) {
            const auto r = magnitude(row_stride), c = magnitude(column_stride);
            const auto divisor = std::gcd(r, c);
            require(std::uint64_t(rows) <= c / divisor ||
                        std::uint64_t(columns) <= r / divisor,
                    "dense output elements overlap");
        }
    }
    using Matrix = std::conditional_t<std::is_const_v<Scalar>, const Eigen::MatrixXd,
                                      Eigen::MatrixXd>;
    using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    return Eigen::Map<Matrix, 0, Stride>(view->data, rows, columns,
                                        Stride(column_stride, row_stride));
}

/// The factorization's own SDDM test (make_graph in graph/conversions.h): a
/// column whose diagonal exceeds its off-diagonal weight by more than 1e-12
/// of the diagonal, summed in the same order.
bool has_positive_excess(const Eigen::SparseMatrix<double>& A) {
    for (Eigen::Index k = 0; k < A.outerSize(); ++k) {
        double diag = 0.0, wdeg = 0.0;
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            if (it.row() != it.col()) wdeg -= it.value();
            else diag = it.value();
        }
        const double excess = diag - wdeg;
        if (excess > diag * 1e-12) return true;
    }
    return false;
}

/// Asserts the operator contract on an adopting solver's operator and that
/// the factor can precondition it. A Laplacian factor is rank n-1 and every
/// solve centres x, so it cannot serve an SDDM operator; the test runs on the
/// matrix the factorization would see (lumped if needed), so a factor never
/// refuses its own operator.
void check_adoptable(const apxchol::factorization& F, const Eigen::SparseMatrix<double>& A) {
    if (F.sddm) {
        apxchol::require_operator(A);
        return;
    }
    const apxchol::operator_view op(A);
    if (has_positive_excess(op.matrix()))
        fail(APXCHOL_STATUS_INVALID_ARGUMENT,
             "a Laplacian factor (rank n-1, centred solutions) cannot precondition an SDDM "
             "operator");
}

bool has_values(const apxchol::factorization& F) {
    const auto nnz = static_cast<std::size_t>(F.L.nonZeros());
    return F.L.inner_.size() == nnz && F.L.vals_.size() == nnz;
}

constexpr const char* kReleasedValues =
    "the factor values were released after setup; create the solver with "
    "keep_factor_values = 1";

/// L (CSC, permuted space) and perm with index_base applied; see the header.
void export_arrays(const apxchol::factorization& F, std::size_t n, std::int32_t index_base,
                   std::int64_t* colptr, std::int64_t* rowval, float* nzval,
                   std::int64_t* perm) {
    require(index_base == 0 || index_base == 1, "index_base must be 0 or 1");
    const bool any_l = colptr != nullptr || rowval != nullptr || nzval != nullptr;
    const bool all_l = colptr != nullptr && rowval != nullptr && nzval != nullptr;
    require(!any_l || all_l, "colptr, rowval and nzval must all be NULL or all non-NULL");
    require(any_l || perm != nullptr, "nothing to export: every output is NULL");
    if (F.perm.size() != n || static_cast<std::size_t>(F.L.rows()) != n)
        fail(APXCHOL_STATUS_INTERNAL_ERROR, "factor dimension differs from the operator");
    if (all_l && !has_values(F)) fail(APXCHOL_STATUS_NO_FACTOR_VALUES, kReleasedValues);
    if (all_l) {
        const auto nnz = static_cast<std::size_t>(F.L.nonZeros());
        for (std::size_t j = 0; j <= n; ++j)
            colptr[j] = static_cast<std::int64_t>(F.L.outer_[j]) + index_base;
        for (std::size_t p = 0; p < nnz; ++p) {
            rowval[p] = static_cast<std::int64_t>(F.L.inner_[p]) + index_base;
            nzval[p] = F.L.vals_[p];
        }
    }
    if (perm != nullptr)
        for (std::size_t v = 0; v < n; ++v)
            perm[v] = static_cast<std::int64_t>(F.perm[v]) + index_base;
}

/// The stats calls' prologue: a mismatched struct_size leaves *stats untouched
/// and is the status; a matching one is zeroed (struct_size kept) before
/// anything else can fail.
apxchol_status begin_stats(apxchol_stats* stats) noexcept {
    if (stats == nullptr) return APXCHOL_STATUS_INVALID_ARGUMENT;
    const apxchol_status s = check_struct_size(stats->struct_size, sizeof(apxchol_stats));
    if (s != APXCHOL_STATUS_SUCCESS) return s;
    std::memset(stats, 0, sizeof(*stats));
    stats->struct_size = sizeof(apxchol_stats);
    return APXCHOL_STATUS_SUCCESS;
}

/// Shared by the solver and factor stats (after begin_stats).
void write_stats(const apxchol::factorization& F, apxchol_backend backend, Eigen::Index n,
                 apxchol_stats* stats) {
    stats->backend = backend;
    stats->n = static_cast<int64_t>(n);
    stats->factor_nnz = static_cast<int64_t>(F.L.nonZeros());
    stats->lumped_offdiag = static_cast<int64_t>(F.lumped_offdiag);
    stats->sddm = F.sddm ? 1 : 0;
    stats->factor_storage = APXCHOL_FACTOR_STORAGE_FLOAT32;
}

}  // namespace

/// Immutable after creation: every call on it only reads.
struct apxchol_factor {
    apxchol::factorization F;  // values always retained
};

struct apxchol_solver {
    apxchol::solver owner;

    explicit apxchol_solver(apxchol::solver value) : owner(std::move(value)) {}

    const apxchol::factorization& factor() const { return owner.factor(); }
    Eigen::Index n() const { return owner.rows(); }

};

extern "C" {

apxchol_status apxchol_solver_create(const apxchol_csc_view* matrix,
                             const apxchol_factor* factor,
                             const apxchol_options* options, apxchol_solver** out_solver,
                             char* error_message, size_t error_capacity) {
    if (out_solver != nullptr) *out_solver = nullptr;
    return guarded(error_message, error_capacity, true, [&]() -> apxchol_status {
        require(out_solver != nullptr, "out_solver is NULL");
        const apxchol_options opt = read_options(options);
        apxchol::solve_options so = to_solve_options(opt);
        const apxchol::detail::thread_budget scope(opt.threads);
        const Eigen::SparseMatrix<double> A = import_csc(matrix);
        if (factor != nullptr)
            require(static_cast<std::size_t>(A.rows()) == factor->F.perm.size(),
                    "the factor's dimension differs from the operator's");

        const auto route = resolve_route(so, A, factor != nullptr);
        const bool gpu = route == apxchol::solve_backend::gpu;
        require_storage(so, route);
        if (factor != nullptr) check_adoptable(factor->F, A);
        std::unique_ptr<apxchol_solver> solver;
        try {
            solver = std::make_unique<apxchol_solver>(factor ? apxchol::solver(A, factor->F, so)
                                                            : apxchol::solver(A, so));
        } catch (const std::invalid_argument& e) {
            if (gpu) {
                // Distinguish an invalid operator from a valid but unsupported
                // GPU input, without repeating validation on successful setup.
                apxchol::require_operator(A, so.factor_opts.lump_max_mass);
                fail(APXCHOL_STATUS_UNSUPPORTED, e.what());
            }
            throw;
        } catch (const std::domain_error& e) {
            if (gpu) fail(APXCHOL_STATUS_UNSUPPORTED, e.what());
            throw;
        } catch (const std::length_error& e) {
            if (gpu) fail(APXCHOL_STATUS_UNSUPPORTED, e.what());
            throw;
        } catch (const std::overflow_error& e) {
            if (gpu) fail(APXCHOL_STATUS_UNSUPPORTED, e.what());
            throw;
        }
        *out_solver = solver.release();
        return APXCHOL_STATUS_SUCCESS;
    });
}

const char* apxchol_version(void) { return APXCHOL_VERSION "+" APXCHOL_GIT_SHA; }

int32_t apxchol_c_abi_version(void) { return APXCHOL_C_ABI_VERSION; }

int32_t apxchol_backend_available(apxchol_backend backend) {
    if (backend == APXCHOL_BACKEND_CPU || backend == APXCHOL_BACKEND_AUTO) return 1;
    try {
        if (backend == APXCHOL_BACKEND_GPU) return gpu_available() ? 1 : 0;
    } catch (...) {
        return 0;
    }
    return 0;
}

apxchol_status apxchol_options_default(apxchol_options* options, size_t struct_size) {
    if (options == nullptr) return APXCHOL_STATUS_INVALID_ARGUMENT;
    const apxchol_status s = check_struct_size(struct_size, sizeof(apxchol_options));
    if (s != APXCHOL_STATUS_SUCCESS) return s;
    try {
        *options = default_options();
    } catch (...) {
        return APXCHOL_STATUS_INTERNAL_ERROR;
    }
    return APXCHOL_STATUS_SUCCESS;
}

void apxchol_solver_destroy(apxchol_solver* solver) { delete solver; }

apxchol_status apxchol_solver_solve(apxchol_solver* solver, int64_t nrhs,
                                    const apxchol_const_dense_view* b,
                                    const apxchol_const_dense_view* x0,
                                    const apxchol_dense_view* x, double tol, int32_t max_iter,
                                    apxchol_solve_info* info, size_t info_size,
                                    char* error_message, size_t error_capacity) {
    return guarded(error_message, error_capacity, false, [&]() -> apxchol_status {
        require(solver != nullptr, "solver is NULL");
        apxchol::detail::validate_solve_parameters(tol, max_iter);
        require_block_size(solver->n(), nrhs);
        if (nrhs == 0) return APXCHOL_STATUS_SUCCESS;
        if (info != nullptr) {
            const apxchol_status s = check_struct_size(info_size, sizeof(apxchol_solve_info));
            if (s != APXCHOL_STATUS_SUCCESS)
                fail(s, "info_size does not match this library");
            require(nrhs <= std::numeric_limits<std::ptrdiff_t>::max() /
                                static_cast<int64_t>(sizeof(apxchol_solve_info)),
                    "nrhs * sizeof(apxchol_solve_info) overflows");
            std::memset(info, 0, static_cast<std::size_t>(nrhs) * sizeof(*info));
        }
        const auto rhs = dense_map(b, solver->n(), nrhs);
        const auto initial = x0 ? dense_map(x0, solver->n(), nrhs) : rhs;
        auto out = dense_map(x, solver->n(), nrhs);
        // Validate every column before native panel execution can write outputs.
        require_finite(rhs, "b");
        if (x0 != nullptr) require_finite(initial, "x0");
        const apxchol::const_matrix_view initial_view(initial);
        apxchol::batch_solve_result result;
        try {
            result = solver->owner.solve(rhs, out, tol, max_iter,
                                         x0 ? &initial_view : nullptr);
        } catch (const std::overflow_error& e) {
            fail(APXCHOL_STATUS_UNSUPPORTED, e.what());
        } catch (const std::underflow_error& e) {
            fail(APXCHOL_STATUS_UNSUPPORTED, e.what());
        }
        bool all_converged = true;
        for (int64_t c = 0; c < nrhs; ++c) {
            const auto& column = result.columns[c];
            all_converged = all_converged && column.converged;
            if (info != nullptr) {
                info[c].iterations = static_cast<int64_t>(column.iterations);
                info[c].relative_residual = column.residual;
                info[c].converged = column.converged ? 1 : 0;
            }
        }
        if (all_converged) return APXCHOL_STATUS_SUCCESS;
        write_message(error_message, error_capacity,
                      "did not converge: some column's relative residual is not below tol");
        return APXCHOL_STATUS_NOT_CONVERGED;
    });
}

apxchol_status apxchol_solver_apply(apxchol_solver* solver, int64_t nrhs,
                                    const apxchol_const_dense_view* r,
                                    const apxchol_dense_view* z,
                                    char* error_message, size_t error_capacity) {
    return guarded(error_message, error_capacity, false, [&]() -> apxchol_status {
        require(solver != nullptr, "solver is NULL");
        require_block_size(solver->n(), nrhs);
        if (nrhs == 0) return APXCHOL_STATUS_SUCCESS;
        const auto rhs = dense_map(r, solver->n(), nrhs);
        auto out = dense_map(z, solver->n(), nrhs);
        require_finite(rhs, "r");
        solver->owner.apply(rhs, out);
        return APXCHOL_STATUS_SUCCESS;
    });
}

apxchol_status apxchol_solver_stats(const apxchol_solver* solver, apxchol_stats* stats) {
    const apxchol_status s = begin_stats(stats);
    if (s != APXCHOL_STATUS_SUCCESS) return s;
    if (solver == nullptr) return APXCHOL_STATUS_INVALID_ARGUMENT;
    write_stats(solver->factor(),
                solver->owner.backend() == apxchol::solve_backend::cpu ? APXCHOL_BACKEND_CPU
                                                                         : APXCHOL_BACKEND_GPU,
                solver->n(), stats);
    stats->factor_storage = solver->owner.factor_storage() == apxchol::factor_storage_type::fp16
        ? APXCHOL_FACTOR_STORAGE_FLOAT16 : APXCHOL_FACTOR_STORAGE_FLOAT32;
    return APXCHOL_STATUS_SUCCESS;
}

apxchol_status apxchol_solver_export_factor(const apxchol_solver* solver, int32_t index_base,
                                            int64_t* colptr, int64_t* rowval, float* nzval,
                                            int64_t* perm, char* error_message,
                                            size_t error_capacity) {
    return guarded(error_message, error_capacity, false, [&]() -> apxchol_status {
        require(solver != nullptr, "solver is NULL");
        export_arrays(solver->factor(), static_cast<std::size_t>(solver->n()), index_base, colptr,
                      rowval, nzval, perm);
        return APXCHOL_STATUS_SUCCESS;
    });
}

apxchol_status apxchol_solver_copy_factor(const apxchol_solver* solver,
                                          apxchol_factor** out_factor, char* error_message,
                                          size_t error_capacity) {
    if (out_factor != nullptr) *out_factor = nullptr;
    return guarded(error_message, error_capacity, false, [&]() -> apxchol_status {
        require(solver != nullptr && out_factor != nullptr, "solver or out_factor is NULL");
        const apxchol::factorization& F = solver->factor();
        if (!has_values(F)) fail(APXCHOL_STATUS_NO_FACTOR_VALUES, kReleasedValues);
        auto factor = std::make_unique<apxchol_factor>();
        factor->F = F;
        *out_factor = factor.release();
        return APXCHOL_STATUS_SUCCESS;
    });
}

apxchol_status apxchol_factor_create(const apxchol_csc_view* matrix,
                                     const apxchol_options* options,
                                     apxchol_factor** out_factor, char* error_message,
                                     size_t error_capacity) {
    if (out_factor != nullptr) *out_factor = nullptr;
    return guarded(error_message, error_capacity, true, [&]() -> apxchol_status {
        require(out_factor != nullptr, "out_factor is NULL");
        const apxchol_options opt = read_options(options);
        const apxchol::solve_options so = to_solve_options(opt);
        const apxchol::detail::thread_budget scope(opt.threads);
        const Eigen::SparseMatrix<double> A = import_csc(matrix);
        auto factor = std::make_unique<apxchol_factor>();
        // Public factorization retains native FP32 values on every build.
        factor->F = apxchol::factorize(A, so.factor_opts);
        *out_factor = factor.release();
        return APXCHOL_STATUS_SUCCESS;
    });
}

void apxchol_factor_destroy(apxchol_factor* factor) { delete factor; }

apxchol_status apxchol_factor_stats(const apxchol_factor* factor, apxchol_stats* stats) {
    const apxchol_status s = begin_stats(stats);
    if (s != APXCHOL_STATUS_SUCCESS) return s;
    if (factor == nullptr) return APXCHOL_STATUS_INVALID_ARGUMENT;
    write_stats(factor->F, APXCHOL_BACKEND_CPU, static_cast<Eigen::Index>(factor->F.perm.size()),
                stats);
    return APXCHOL_STATUS_SUCCESS;
}

apxchol_status apxchol_factor_export(const apxchol_factor* factor, int32_t index_base,
                                     int64_t* colptr, int64_t* rowval, float* nzval,
                                     int64_t* perm, char* error_message,
                                     size_t error_capacity) {
    return guarded(error_message, error_capacity, false, [&]() -> apxchol_status {
        require(factor != nullptr, "factor is NULL");
        export_arrays(factor->F, factor->F.perm.size(), index_base, colptr, rowval, nzval, perm);
        return APXCHOL_STATUS_SUCCESS;
    });
}

}  // extern "C"
