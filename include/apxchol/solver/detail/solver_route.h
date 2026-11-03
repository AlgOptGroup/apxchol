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
    // A solver for an operator in the library's own storage (the CLI, tests
    // and the benchmark). A timing tree, when given, records setup under
    // "setup" and every solve on the returned solver under "pcg"; it must
    // outlive the solver, and copies start without one.
    static solver make(csc_ref<double> A, const solve_options& opts,
                       checkpoint* timings = nullptr);
    // A solver adopting F for an operator in the library's own storage.
    static solver make(csc_ref<double> A, factorization F, const solve_options& opts);
    static void save(const std::filesystem::path& path, const solver& owner);
    // started is when load() began, so setup_seconds includes reading the file.
    static void finish_load(solver& owner, std::int64_t entries,
                            std::chrono::steady_clock::time_point started);
    // Measure setup_seconds from started, e.g. before checking a public view.
    static void set_setup_start(solver& owner, std::chrono::steady_clock::time_point started);
};

} // namespace apxchol::detail
