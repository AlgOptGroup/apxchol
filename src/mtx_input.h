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
/// The same confusion reaches the in-memory bindings (python/, octave/) by a
/// different route -- a caller hands `scipy.io.mmread('com-Amazon.mtx')`
/// straight to the solver -- so `detect_adjacency_signature` below is shared
/// with them; they compile this TU too.
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

/// Counts behind the ADJACENCY SIGNATURE check, gathered in one early-exiting
/// pass. See `detect_adjacency_signature`.
struct adjacency_signature {
    Eigen::Index n                    = 0;      ///< rows in the matrix
    /// Some row carries a strictly positive diagonal entry. The scan stops at
    /// the first one, so this is a flag, not a count -- when it is false, ALL
    /// `n` rows lack a positive diagonal, which is what the message reports.
    bool         any_positive_diagonal = false;
    Eigen::Index positive_offdiag     = 0;      ///< strictly positive off-diagonals

    /// The UNAMBIGUOUS adjacency signature: not one row carries a positive
    /// diagonal entry -- which a Laplacian/SDDM always does -- yet positive
    /// off-diagonal entries exist, which an M-matrix never has.
    ///
    /// Deliberately NARROWER than `resolve_input_kind`'s automatic decision:
    /// it is the rule the in-memory bindings REJECT on, so it must never fire
    /// on input that works today. Mixed-sign FEM/structural operators
    /// (parabolic_fem, thermal2, bcsstk*, G3_circuit, apache2) all carry
    /// positive diagonals and are therefore untouched. Equivalent, on a full
    /// scan, to `scan_input(M).diag_pos == 0 && scan_input(M).offdiag_pos > 0`.
    bool detected() const {
        return n > 0 && !any_positive_diagonal && positive_offdiag > 0;
    }
};

/// Test `M` for the adjacency signature, counting what an error message needs.
///
/// Cost is O(nnz) and, in practice, free for every assembled operator: the
/// walk stops at the FIRST positive diagonal entry, which a Laplacian/SDDM
/// carries in its very first column. Only a matrix that really has no positive
/// diagonal anywhere is walked whole -- and that one is about to be rejected.
adjacency_signature detect_adjacency_signature(const Eigen::SparseMatrix<double>& M);

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
