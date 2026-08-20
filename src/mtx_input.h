#pragma once
/// Input interpretation for the CLI.
///
/// A .mtx file can hold either of two very different things, and the two are
/// numerically indistinguishable to the solver core:
///
///   * an ALREADY-ASSEMBLED Laplacian / SDDM operator (positive diagonal,
///     non-positive off-diagonals) — solve it as given; or
///   * a graph ADJACENCY / pattern matrix (no diagonal, non-negative
///     off-diagonals) — the operator is L = D - A, which must be assembled
///     first.
///
/// Handing an adjacency matrix straight to the solver negates every off-
/// diagonal into a NEGATIVE edge weight; `tree_elimination::sample_clique`
/// early-returns on a non-positive weighted degree, so the factorization
/// silently produces zero fill and PCG breaks down on iteration 1 with
/// `p·Ap <= 0` (reported as "iterations 0 / residual 1"). This header exists
/// so that never happens without the user being told.
///
/// Everything here depends on Eigen only, so the unit tests can link it.

#include <Eigen/Sparse>
#include <string>

namespace apxchol {

/// How the CLI should interpret the matrix it just read.
enum class input_kind {
    automatic,  ///< decide from the matrix itself (the default)
    laplacian,  ///< already an assembled Laplacian / SDDM operator
    adjacency,  ///< a graph adjacency / pattern matrix; assemble L = D - A
};

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

/// One pass over `M`, gathering everything `resolve_input_kind` needs.
input_scan scan_input(const Eigen::SparseMatrix<double>& M);

/// Resolve `requested` (possibly `automatic`) against the scan.
///
/// `pattern_field` says whether the MatrixMarket header declared the
/// `pattern` field; it only sharpens the explanation, the decision itself
/// stands without it.
///
/// On success returns `laplacian` or `adjacency` and writes a human-readable
/// justification into `reason`. Throws `std::runtime_error` — naming what was
/// detected — when the matrix is structurally unusable, or when `automatic`
/// genuinely cannot tell the two kinds apart.
input_kind resolve_input_kind(input_kind requested, const input_scan& s,
                              bool pattern_field, std::string& reason);

/// Reinterpret `M` as a weighted adjacency matrix and replace it with the
/// graph Laplacian L = D - A, where A_ij = |M_ij| for i != j and D is the
/// weighted-degree diagonal. Diagonal entries of `M` (graph self-loops, which
/// contribute nothing to a Laplacian) are dropped.
///
/// Taking |value| matches the benchmark suite's `load_mtx_as_adjacency`, so
/// `--input-kind adjacency` reproduces the benchmark's reading of any file.
void adjacency_to_laplacian(Eigen::SparseMatrix<double>& M);

/// The single line the CLI logs to say how it read the file.
std::string describe_input(input_kind kind, const input_scan& s,
                           const std::string& reason);

/// Extra diagnosis appended to the CLI's failure message. Empty when the scan
/// found nothing that would explain a breakdown.
std::string diagnose_failure(input_kind kind, const input_scan& s);

} // namespace apxchol
