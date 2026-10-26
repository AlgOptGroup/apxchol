#pragma once

#include "apxchol/solver/solve.h"
#include <filesystem>

namespace apxchol {

/// Save the solver's original operator and retained FP32 factor in portable NPZ.
/// Creates a new file; an existing destination is an error. Factor retention
/// must have been enabled at construction. No extra matrix is kept between saves.
void save_solver(const std::filesystem::path& path, const solver& owner);

/// Restore a solver using the saved operator and factor. Backend, storage and
/// solve options apply as with solver(A, F, options); automatic adoption uses CPU.
solver load_solver(const std::filesystem::path& path, const solve_options& options = {});

} // namespace apxchol
