#pragma once

#include "apxchol/solver/solve.h"
#include <filesystem>

namespace apxchol {

/// Save the original operator and retained FP32 factor in a portable NPZ file.
/// Creates a new file; an existing destination is an error. A must be the
/// operator intended for later solves, not the factor's approximate product.
void save_solver(const std::filesystem::path& path,
                 const Eigen::SparseMatrix<double>& A, const factorization& F);

/// Restore a solver using the saved operator and factor. Backend, storage and
/// solve options apply as with solver(A, F, options); automatic adoption uses CPU.
solver load_solver(const std::filesystem::path& path, const solve_options& options = {});

} // namespace apxchol
