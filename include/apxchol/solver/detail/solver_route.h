#pragma once
// The route a solver selected at setup. solver is the only caller of the
// route interface; tests and the benchmark read route state through
// solver_access::route.
#include "apxchol/solver/solve.h"
#include "apxchol/solver/detail/cpu_route.h"
#if defined(APXCHOL_USE_CUDA)
#include "apxchol/solver/detail/gpu_solve_session.h"
#elif defined(APXCHOL_USE_METAL)
#include "apxchol/solver/detail/metal_route.h"
#endif
#include <filesystem>
#include <variant>

namespace apxchol::detail {

// Every route is constructed from (A, opts, cp, F) and provides solve_widths,
// apply_widths, solve_columns, apply_columns, factor, factor_storage and
// original_operator.
using solver_route = std::variant<cpu_route
#if defined(APXCHOL_USE_CUDA)
    , gpu_solve_session
#elif defined(APXCHOL_USE_METAL)
    , metal_route
#endif
    >;

struct solver_access {
    static const solver_route& route(const solver& owner);
    static void save(const std::filesystem::path& path, const solver& owner);
    static void restore_input_entries(solver& owner, Eigen::Index entries);
};

} // namespace apxchol::detail
