#pragma once
/// Input interpretation for the CLI.
///
/// A .mtx file can hold either an assembled operator or graph input:
///
///   * an ALREADY-ASSEMBLED Laplacian / SDDM operator (positive diagonal,
///     non-positive off-diagonals) — solve it as given; or
///   * a graph ADJACENCY / pattern matrix (no diagonal, non-negative
///     off-diagonals) — the operator is L = D - A, which must be assembled
///     first.
///
/// The CLI reads the file as the operator. `--convert-input` applies the rule
/// it shares with the bindings' `as_operator` (`detail::is_graph_adjacency` in
/// include/apxchol/operator_class.h): a finite symmetric matrix whose
/// off-diagonal entries are all positive becomes its graph Laplacian, any
/// other finite symmetric matrix is used as given, and the rest is an error.

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <string>

namespace apxchol {

/// Structural facts about an input matrix, gathered in a single pass.
/// Column sums stand in for row sums: the operator contract requires a
/// symmetric matrix, and `one_sided` below catches the case where it isn't.
struct input_scan {
    Eigen::Index n            = 0;  ///< rows (== cols; the caller checks)
    Eigen::Index offdiag_nnz  = 0;  ///< stored off-diagonal entries
    Eigen::Index offdiag_pos  = 0;  ///< ... with a strictly positive value
    Eigen::Index offdiag_neg  = 0;  ///< ... with a strictly negative value
    Eigen::Index upper_nnz    = 0;  ///< ... in the strict upper triangle
    Eigen::Index lower_nnz    = 0;  ///< ... in the strict lower triangle
    Eigen::Index diag_pos     = 0;  ///< columns with a strictly positive diagonal
    /// Columns whose sum is below -1e-10 * max(|diagonal|, 1) — i.e. rows that
    /// break diagonal dominance. Diagnostic only: plenty of legitimate SPD
    /// test matrices (G3_circuit, apache2) are not diagonally dominant and
    /// converge fine, so this never gates a run.
    Eigen::Index deficient_rows = 0;
    double       worst_deficit  = 0.0;  ///< most negative column sum seen
    Eigen::Index worst_row      = -1;   ///< where it was seen (-1 = none)
    /// Columns whose sum is above +1e-10 * max(|diagonal|, 1) — the SDDM
    /// diagonal excess.
    Eigen::Index excess_rows = 0;
};

/// One pass over `M`, gathering what the CLI reports about it.
input_scan scan_input(const Eigen::SparseMatrix<double>& M);

/// Project `b` onto the range of a graph Laplacian by subtracting its mean
/// independently on every connected component of `L`. Returns the component
/// count. A connected matrix leaves `b` byte-for-byte unchanged, preserving
/// the historical --random-rhs stream on the common case.
///
/// The caller must already know that `L` is a pure Laplacian. Applying this to
/// an SDDM/SPD operator would change a perfectly valid right-hand side.
Eigen::Index project_laplacian_rhs_components(
    const Eigen::SparseMatrix<double>& L, Eigen::VectorXd& b);

/// The single line the CLI logs to say how it read the file.
/// `graph` says whether the file was read as an adjacency matrix.
std::string describe_input(bool graph, const input_scan& s, const std::string& reason);

/// Extra diagnosis appended to the CLI's failure message. Empty when the scan
/// found nothing that would explain a breakdown.
std::string diagnose_failure(bool graph, const input_scan& s);

} // namespace apxchol
