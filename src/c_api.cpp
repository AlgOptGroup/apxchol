// C ABI over apxchol::cpu_solver and, in APXCHOL_USE_METAL builds,
// apxchol::metal_solver. See include/apxchol/c_api.h for the contract:
// statuses, struct versioning, aliasing and the abort paths.
#include "apxchol/c_api.h"

#include "apxchol/solver/solve.h"
#include "apxchol/version.h"
#if defined(APXCHOL_USE_METAL)
#include "apxchol/solver/metal_solver.h"
#endif

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using clock_type = std::chrono::steady_clock;

double seconds_since(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}

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
    } catch (const std::bad_alloc&) {
        write_message(message, capacity, "out of memory");
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

/// Applies a positive `threads` to the calling thread's OpenMP team limit
/// for the duration of one call and restores the previous limit.
class thread_scope {
public:
    explicit thread_scope(std::int32_t threads) noexcept {
#ifdef _OPENMP
        if (threads > 0) {
            previous_ = omp_get_max_threads();
            omp_set_num_threads(threads);
            restore_ = true;
        }
#else
        (void)threads;
#endif
    }
    ~thread_scope() {
#ifdef _OPENMP
        if (restore_) omp_set_num_threads(previous_);
#endif
    }
    thread_scope(const thread_scope&) = delete;
    thread_scope& operator=(const thread_scope&) = delete;

private:
    int previous_ = 1;
    bool restore_ = false;
};

int max_threads() noexcept {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
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

constexpr const char* kPartitionerNames[] = {"block_greedy", "priority_greedy",
                                             "baumann_kyng"};

apxchol_options default_options() {
    const apxchol::solve_options so;
    apxchol_options o{};
    o.struct_size = sizeof(apxchol_options);
    o.backend = APXCHOL_BACKEND_CPU;
    o.tol = so.tol;
    o.max_iter = so.max_iter;
    o.stagnation_window = so.stagnation_window;
    o.seed = so.factor_opts.seed;
    o.threads = 0;
    o.sampler = so.factor_opts.sampler == apxchol::clique_sampler::trace_cycle
        ? APXCHOL_SAMPLER_TRACE_CYCLE : APXCHOL_SAMPLER_GKS;
    o.partitioner = APXCHOL_PARTITIONER_BLOCK_GREEDY;
    for (int p = 0; p < 3; ++p)
        if (so.factor_opts.is_select == kPartitionerNames[p]) o.partitioner = p;
    o.storage = static_cast<apxchol_storage>(so.storage);
    o.keep_factor_values = so.keep_factor_values ? 1 : 0;
    o.degree_quantile = so.factor_opts.partition.degree_quantile;
    o.exact_clique_max_degree = so.factor_opts.exact_clique_max_degree;
    return o;
}

/// Validates every option before any work, so a later std::invalid_argument
/// during create can only come from the operator contract.
apxchol::solve_options to_solve_options(const apxchol_options& o) {
    require_struct_size(o.struct_size, sizeof(apxchol_options), "options");
    require(o.backend == APXCHOL_BACKEND_CPU || o.backend == APXCHOL_BACKEND_METAL,
            "options.backend must be APXCHOL_BACKEND_CPU or APXCHOL_BACKEND_METAL");
    require(std::isfinite(o.tol) && o.tol > 0.0, "options.tol must be finite and positive");
    require(o.max_iter >= 0, "options.max_iter must be non-negative");
    require(o.stagnation_window >= 0, "options.stagnation_window must be non-negative");
    require(o.threads >= 0, "options.threads must be non-negative");
    require(o.sampler == APXCHOL_SAMPLER_GKS || o.sampler == APXCHOL_SAMPLER_TRACE_CYCLE,
            "options.sampler must be APXCHOL_SAMPLER_GKS or APXCHOL_SAMPLER_TRACE_CYCLE");
    require(o.partitioner >= 0 && o.partitioner < 3,
            "options.partitioner is not an apxchol_partitioner value");
    require(o.storage == APXCHOL_STORAGE_VEC || o.storage == APXCHOL_STORAGE_BSTR ||
                o.storage == APXCHOL_STORAGE_VEC_POOL_AOS,
            "options.storage must be VEC, BSTR or VEC_POOL_AOS");
    require(o.keep_factor_values == 0 || o.keep_factor_values == 1,
            "options.keep_factor_values must be 0 or 1");
    require(std::isfinite(o.degree_quantile) && o.degree_quantile <= 1.0,
            "options.degree_quantile must be finite and at most 1");
    if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t))
        require(o.exact_clique_max_degree <= std::numeric_limits<std::size_t>::max(),
                "options.exact_clique_max_degree exceeds size_t");

    apxchol::solve_options so;
    so.tol = o.tol;
    so.max_iter = o.max_iter;
    so.stagnation_window = o.stagnation_window;
    so.storage = static_cast<apxchol::graph_storage>(o.storage);
    so.keep_factor_values = o.keep_factor_values != 0;
    so.factor_opts.seed = o.seed;
    so.factor_opts.is_select = kPartitionerNames[o.partitioner];
    so.factor_opts.sampler = o.sampler == APXCHOL_SAMPLER_TRACE_CYCLE
        ? apxchol::clique_sampler::trace_cycle : apxchol::clique_sampler::gks;
    so.factor_opts.partition.degree_quantile = o.degree_quantile;
    so.factor_opts.exact_clique_max_degree =
        static_cast<std::size_t>(o.exact_clique_max_degree);
    return so;
}

/// CSC import with overflow-safe validation. Columns with strictly increasing
/// rows are copied directly into the compressed layout setFromTriplets would
/// produce; anything else goes through setFromTriplets (duplicates summed).
Eigen::SparseMatrix<double> import_csc(std::int64_t n, const std::int64_t* colptr,
                                       const std::int64_t* rowval, const double* nzval,
                                       std::int32_t base) {
    constexpr std::int64_t kMax = std::numeric_limits<int>::max();
    require(n >= 1 && n <= kMax, "n must be in [1, 2^31-1]");
    require(colptr != nullptr, "colptr is NULL");
    require(base == 0 || base == 1, "index_base must be 0 or 1");
    require(colptr[0] == base, "colptr[0] must equal index_base");
    for (std::int64_t j = 0; j < n; ++j)
        require(colptr[j + 1] >= colptr[j], "colptr must be nondecreasing");
    const std::int64_t nnz = colptr[n] - base;
    require(nnz <= kMax, "nnz exceeds 2^31-1");
    require(nnz == 0 || (rowval != nullptr && nzval != nullptr), "rowval or nzval is NULL");

    bool sorted = true;
    for (std::int64_t j = 0; j < n; ++j) {
        const std::int64_t begin = colptr[j] - base, end = colptr[j + 1] - base;
        for (std::int64_t p = begin; p < end; ++p) {
            const std::int64_t r = rowval[p];
            require(r >= base && r - base < n, "CSC row index out of range");
            if (p > begin && r <= rowval[p - 1]) sorted = false;
        }
    }

    const auto ni = static_cast<int>(n);
    Eigen::SparseMatrix<double> A(ni, ni);
    if (sorted) {
        A.resizeNonZeros(static_cast<Eigen::Index>(nnz));
        int* outer = A.outerIndexPtr();
        int* inner = A.innerIndexPtr();
        double* values = A.valuePtr();
        for (std::int64_t j = 0; j <= n; ++j) outer[j] = static_cast<int>(colptr[j] - base);
        for (std::int64_t p = 0; p < nnz; ++p) {
            inner[p] = static_cast<int>(rowval[p] - base);
            values[p] = nzval[p];
        }
        return A;
    }
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<std::size_t>(nnz));
    for (std::int64_t j = 0; j < n; ++j)
        for (std::int64_t p = colptr[j] - base; p < colptr[j + 1] - base; ++p)
            triplets.emplace_back(static_cast<int>(rowval[p] - base), static_cast<int>(j),
                                  nzval[p]);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();
    return A;
}

/// Copies a vector argument into reusable storage (which makes aliasing with
/// the output safe) and rejects non-finite entries.
void load_vector(const double* source, Eigen::Index n, Eigen::VectorXd& target,
                 const char* what) {
    target.resize(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        if (!std::isfinite(source[i]))
            fail(APXCHOL_STATUS_INVALID_ARGUMENT, std::string(what) + " contains a non-finite value");
        target[i] = source[i];
    }
}

}  // namespace

struct apxchol_solver {
    apxchol_options options{};
    std::unique_ptr<apxchol::cpu_solver> cpu;
#if defined(APXCHOL_USE_METAL)
    std::unique_ptr<apxchol::metal_solver> metal;  // exactly one of cpu / metal is set
#endif
    Eigen::VectorXd b_buffer, x0_buffer;
    Eigen::Index n = 0;
    double setup_seconds = 0.0;
    std::int32_t setup_max_threads = 1;

    const apxchol::factorization& factor() const {
#if defined(APXCHOL_USE_METAL)
        if (metal) return metal->factor();
#endif
        return cpu->preconditioner().factor();
    }
    double tol_for(double tol) const { return tol < 0.0 ? options.tol : tol; }
    int max_iter_for(std::int32_t max_iter) const {
        return max_iter < 0 ? options.max_iter : max_iter;
    }

    /// One right-hand side through the handle's backend into caller memory.
    apxchol::solve_result solve_one(const double* b, const double* x0, double* x,
                                    double tol, int max_iter) {
        load_vector(b, n, b_buffer, "b");
        if (x0 != nullptr) load_vector(x0, n, x0_buffer, "x0");
        Eigen::Map<Eigen::VectorXd> out(x, n);
#if defined(APXCHOL_USE_METAL)
        if (metal) {
            apxchol::solve_result r =
                metal->solve(b_buffer, tol, max_iter, x0 != nullptr ? &x0_buffer : nullptr);
            out = r.x;
            r.x.resize(0);
            return r;
        }
#endif
        return cpu->solve(b_buffer, out, tol, max_iter, x0 != nullptr ? &x0_buffer : nullptr);
    }

    Eigen::VectorXd apply(const Eigen::VectorXd& r) const {
#if defined(APXCHOL_USE_METAL)
        if (metal) return metal->apply(r);
#endif
        return cpu->apply(r);
    }
};

namespace {

void check_call_tolerances(double tol, std::int32_t max_iter) {
    (void)max_iter;  // negative selects the handle's value; every other value is valid
    require(!std::isnan(tol) && tol != 0.0 && !(tol > 0.0 && !std::isfinite(tol)),
            "tol must be finite and positive (or negative for the handle's value)");
}

}  // namespace

extern "C" {

const char* apxchol_version(void) { return APXCHOL_VERSION "+" APXCHOL_GIT_SHA; }

int32_t apxchol_c_abi_version(void) { return APXCHOL_C_ABI_VERSION; }

int32_t apxchol_openmp_enabled(void) {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

int32_t apxchol_get_max_threads(void) { return max_threads(); }

int32_t apxchol_backend_available(apxchol_backend backend) {
    if (backend == APXCHOL_BACKEND_CPU) return 1;
#if defined(APXCHOL_USE_METAL)
    if (backend == APXCHOL_BACKEND_METAL) return apxchol::metal_solver::available() ? 1 : 0;
#endif
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

apxchol_status apxchol_solver_create(int64_t n, const int64_t* colptr, const int64_t* rowval,
                                     const double* nzval, int32_t index_base,
                                     const apxchol_options* options,
                                     apxchol_solver** out_solver, char* error_message,
                                     size_t error_capacity) {
    if (out_solver != nullptr) *out_solver = nullptr;
    return guarded(error_message, error_capacity, true, [&]() -> apxchol_status {
        require(out_solver != nullptr, "out_solver is NULL");
        const apxchol_options opt = options != nullptr ? *options : default_options();
        const apxchol::solve_options so = to_solve_options(opt);
#if defined(APXCHOL_USE_METAL)
        if (opt.backend == APXCHOL_BACKEND_METAL && !apxchol::metal_solver::available())
            fail(APXCHOL_STATUS_UNSUPPORTED, "no usable Metal device for APXCHOL_BACKEND_METAL");
#else
        if (opt.backend != APXCHOL_BACKEND_CPU)
            fail(APXCHOL_STATUS_UNSUPPORTED,
                 "this apxchol build has no Metal backend (configure with -DAPXCHOL_USE_METAL=ON)");
#endif
        const Eigen::SparseMatrix<double> A = import_csc(n, colptr, rowval, nzval, index_base);

        auto solver = std::make_unique<apxchol_solver>();
        solver->options = opt;
        solver->n = A.rows();
        const thread_scope scope(opt.threads);
        solver->setup_max_threads = max_threads();
        const auto start = clock_type::now();
#if defined(APXCHOL_USE_METAL)
        if (opt.backend == APXCHOL_BACKEND_METAL)
            solver->metal = std::make_unique<apxchol::metal_solver>(A, so);
        else
#endif
            solver->cpu = std::make_unique<apxchol::cpu_solver>(A, so);
        solver->setup_seconds = seconds_since(start);
        *out_solver = solver.release();
        return APXCHOL_STATUS_SUCCESS;
    });
}

void apxchol_solver_destroy(apxchol_solver* solver) { delete solver; }

apxchol_status apxchol_solver_solve(apxchol_solver* solver, const double* b, const double* x0,
                                    double* x, double tol, int32_t max_iter,
                                    apxchol_solve_info* info, char* error_message,
                                    size_t error_capacity) {
    if (info != nullptr) {
        const apxchol_status s = check_struct_size(info->struct_size, sizeof(apxchol_solve_info));
        if (s != APXCHOL_STATUS_SUCCESS) {
            write_message(error_message, error_capacity, "info.struct_size does not match this library");
            return s;
        }
        *info = apxchol_solve_info{};
        info->struct_size = sizeof(apxchol_solve_info);
    }
    return guarded(error_message, error_capacity, false, [&]() -> apxchol_status {
        require(solver != nullptr && b != nullptr && x != nullptr, "solver, b or x is NULL");
        check_call_tolerances(tol, max_iter);
        const double t = solver->tol_for(tol);
        const thread_scope scope(solver->options.threads);
        const auto start = clock_type::now();
        const apxchol::solve_result r =
            solver->solve_one(b, x0, x, t, solver->max_iter_for(max_iter));
        const bool converged = r.residual < t;
        if (info != nullptr) {
            info->converged = converged ? 1 : 0;
            info->iterations = static_cast<int64_t>(r.iterations);
            info->relative_residual = r.residual;
            info->solve_seconds = seconds_since(start);
        }
        if (converged) return APXCHOL_STATUS_SUCCESS;
        write_message(error_message, error_capacity,
                      "did not converge: the relative residual is not below tol");
        return APXCHOL_STATUS_NOT_CONVERGED;
    });
}

apxchol_status apxchol_solver_solve_block(apxchol_solver* solver, int64_t k, const double* b,
                                          const double* x0, double* x, double tol,
                                          int32_t max_iter, int64_t* iterations,
                                          double* relative_residuals, int32_t* converged,
                                          char* error_message, size_t error_capacity) {
    return guarded(error_message, error_capacity, false, [&]() -> apxchol_status {
        require(solver != nullptr, "solver is NULL");
        require(k >= 0, "k must be non-negative");
        if (k == 0) return APXCHOL_STATUS_SUCCESS;
        require(b != nullptr && x != nullptr, "b or x is NULL");
        require(k <= std::numeric_limits<int64_t>::max() / solver->n, "n*k overflows");
        check_call_tolerances(tol, max_iter);
        const double t = solver->tol_for(tol);
        const int mi = solver->max_iter_for(max_iter);
        const thread_scope scope(solver->options.threads);
#if defined(APXCHOL_USE_METAL)
        if (solver->metal) {
            // Lockstep device batches; every column is bit-identical to its
            // single-RHS solve. B and X0 are copied (and checked) first, so
            // aliasing with X is safe.
            const Eigen::Index n = solver->n;
            const Eigen::Map<const Eigen::MatrixXd> Bm(b, n, k);
            const Eigen::MatrixXd Bc = Bm;
            if (!Bc.allFinite()) fail(APXCHOL_STATUS_INVALID_ARGUMENT, "b contains a non-finite value");
            Eigen::MatrixXd X0c;
            if (x0 != nullptr) {
                X0c = Eigen::Map<const Eigen::MatrixXd>(x0, n, k);
                if (!X0c.allFinite()) fail(APXCHOL_STATUS_INVALID_ARGUMENT, "x0 contains a non-finite value");
            }
            const apxchol::metal_solver::block_cref X0r(X0c);
            Eigen::Map<Eigen::MatrixXd> Xm(x, n, k);
            const apxchol::metal_block_result r = solver->metal->solve(
                apxchol::metal_solver::block_cref(Bc), Xm, t, mi, x0 != nullptr ? &X0r : nullptr);
            bool all = true;
            for (int64_t c = 0; c < k; ++c) {
                const apxchol::metal_column_result& col = r.columns[static_cast<std::size_t>(c)];
                const bool ok = col.residual < t;
                all = all && ok;
                if (iterations != nullptr) iterations[c] = static_cast<int64_t>(col.iterations);
                if (relative_residuals != nullptr) relative_residuals[c] = col.residual;
                if (converged != nullptr) converged[c] = ok ? 1 : 0;
            }
            if (all) return APXCHOL_STATUS_SUCCESS;
            write_message(error_message, error_capacity,
                          "did not converge: some column's relative residual is not below tol");
            return APXCHOL_STATUS_NOT_CONVERGED;
        }
#endif
        bool all = true;
        for (int64_t c = 0; c < k; ++c) {
            const int64_t off = c * solver->n;
            const apxchol::solve_result r = solver->solve_one(
                b + off, x0 != nullptr ? x0 + off : nullptr, x + off, t, mi);
            const bool ok = r.residual < t;
            all = all && ok;
            if (iterations != nullptr) iterations[c] = static_cast<int64_t>(r.iterations);
            if (relative_residuals != nullptr) relative_residuals[c] = r.residual;
            if (converged != nullptr) converged[c] = ok ? 1 : 0;
        }
        if (all) return APXCHOL_STATUS_SUCCESS;
        write_message(error_message, error_capacity,
                      "did not converge: some column's relative residual is not below tol");
        return APXCHOL_STATUS_NOT_CONVERGED;
    });
}

apxchol_status apxchol_solver_apply(apxchol_solver* solver, const double* r, double* z,
                                    char* error_message, size_t error_capacity) {
    return guarded(error_message, error_capacity, false, [&]() -> apxchol_status {
        require(solver != nullptr && r != nullptr && z != nullptr, "solver, r or z is NULL");
        const thread_scope scope(solver->options.threads);
        load_vector(r, solver->n, solver->b_buffer, "r");
        Eigen::Map<Eigen::VectorXd>(z, solver->n) = solver->apply(solver->b_buffer);
        return APXCHOL_STATUS_SUCCESS;
    });
}

apxchol_status apxchol_solver_stats(const apxchol_solver* solver, apxchol_stats* stats) {
    if (solver == nullptr || stats == nullptr) return APXCHOL_STATUS_INVALID_ARGUMENT;
    const apxchol_status s = check_struct_size(stats->struct_size, sizeof(apxchol_stats));
    if (s != APXCHOL_STATUS_SUCCESS) return s;
    const apxchol::factorization& F = solver->factor();
    *stats = apxchol_stats{};
    stats->struct_size = sizeof(apxchol_stats);
    stats->backend = solver->options.backend;
    stats->n = static_cast<int64_t>(solver->n);
    stats->factor_nnz = static_cast<int64_t>(F.L.nonZeros());
    stats->lumped_offdiag = static_cast<int64_t>(F.lumped_offdiag);
    stats->rounds = static_cast<int64_t>(F.rounds.size());
    stats->peak_graph_bytes = static_cast<int64_t>(F.peak_graph_bytes);
    stats->setup_seconds = solver->setup_seconds;
    stats->sddm = F.sddm ? 1 : 0;
    stats->setup_max_threads = solver->setup_max_threads;
    return APXCHOL_STATUS_SUCCESS;
}

apxchol_status apxchol_solver_export_factor(const apxchol_solver* solver, int32_t index_base,
                                            int64_t* colptr, int64_t* rowval, double* nzval,
                                            int64_t* perm, char* error_message,
                                            size_t error_capacity) {
    return guarded(error_message, error_capacity, false, [&]() -> apxchol_status {
        require(solver != nullptr, "solver is NULL");
        require(index_base == 0 || index_base == 1, "index_base must be 0 or 1");
        const bool any_l = colptr != nullptr || rowval != nullptr || nzval != nullptr;
        const bool all_l = colptr != nullptr && rowval != nullptr && nzval != nullptr;
        require(!any_l || all_l, "colptr, rowval and nzval must all be NULL or all non-NULL");
        require(any_l || perm != nullptr, "nothing to export: every output is NULL");

        const apxchol::factorization& F = solver->factor();
        const auto n = static_cast<std::size_t>(solver->n);
        if (F.perm.size() != n || static_cast<std::size_t>(F.L.rows()) != n)
            fail(APXCHOL_STATUS_INTERNAL_ERROR, "factor dimension differs from the operator");
        const auto nnz = static_cast<std::size_t>(F.L.nonZeros());
        if (all_l && (F.L.inner_.size() != nnz || F.L.vals_.size() != nnz))
            fail(APXCHOL_STATUS_NO_FACTOR_VALUES,
                 "the factor values were released after setup; create the solver with "
                 "keep_factor_values = 1");
        if (all_l) {
            for (std::size_t j = 0; j <= n; ++j)
                colptr[j] = static_cast<int64_t>(F.L.outer_[j]) + index_base;
            for (std::size_t p = 0; p < nnz; ++p) {
                rowval[p] = static_cast<int64_t>(F.L.inner_[p]) + index_base;
                nzval[p] = static_cast<double>(F.L.vals_[p]);
            }
        }
        if (perm != nullptr)
            for (std::size_t v = 0; v < n; ++v)
                perm[v] = static_cast<int64_t>(F.perm[v]) + index_base;
        return APXCHOL_STATUS_SUCCESS;
    });
}

}  // extern "C"
