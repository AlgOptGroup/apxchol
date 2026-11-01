#pragma once
// The route a solver selected at setup. solver is the only caller of the
// route interface; tests and the benchmark read route state through
// solver_access::route.
#include "apxchol/solver/solve.h"
#include "apxchol/solver/detail/cpu_route.h"
#if defined(APXCHOL_USE_CUDA)
#include "apxchol/solver/detail/cuda_route.h"
#elif defined(APXCHOL_USE_METAL)
#include "apxchol/solver/detail/metal_route.h"
#endif
#include "apxchol/detail/eigen_boundary.h"
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <variant>

namespace apxchol::detail {

// Every route is constructed from (A, opts, cp, F), A a square csc_ref, and
// provides solve_widths,
// apply_widths, solve_columns, apply_columns, factor (metadata),
// installed_factor, factor_nnz, factor_storage and original_operator.
using solver_route = std::variant<cpu_route
#if defined(APXCHOL_USE_CUDA)
    , cuda_route
#elif defined(APXCHOL_USE_METAL)
    , metal_route
#endif
    >;

struct solver_access {
    static const solver_route& route(const solver& owner);
    // Attach a timing tree for the CLI's -v report, tests and the benchmark:
    // setup records under "setup", and every solve on the returned solver
    // under "pcg". The tree must outlive the solver; copies start without one.
    static solver with_timings(const Eigen::SparseMatrix<double>& A,
                               const solve_options& opts, checkpoint& timings);
    // A solver adopting F for an operator in the library's own storage.
    static solver make(csc_ref<double> A, factorization F, const solve_options& opts);
    // A converted public input as the route's operator; throws the route's
    // errors for a matrix that is not square.
    static const csc_ref<double>& solver_input(const eigen_csc& input,
                                               const solve_options& opts, bool adopted);
    static void save(const std::filesystem::path& path, const solver& owner);
    // started is when load() began, so setup_seconds includes reading the file.
    static void finish_load(solver& owner, std::int64_t entries,
                            std::chrono::steady_clock::time_point started);
};

} // namespace apxchol::detail
