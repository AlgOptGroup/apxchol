#pragma once
/// The operator class apxchol is DEFINED on — asserted, not guessed — plus the
/// M-matrix lumping that repairs a *nearly* SDDM operator into one.
///
/// ── The contract ────────────────────────────────────────────────────────────
/// apxchol factorizes a symmetric M-matrix (a Stieltjes matrix): the assembled
/// Laplacian / SDDM operator
///
///     A = A^T,   a_ii > 0,   a_ij <= 0 for i != j.
///
/// Diagonal dominance (a_ii >= sum_{j != i} |a_ij|) is what makes it SDDM, and
/// it is REPORTED but NOT enforced: published operators such as G3_circuit and
/// apache2 carry rows that break it and converge fine today, so requiring it
/// would reject matrices that work.
///
/// `require_operator` reports violated conditions with counts. When the matrix
/// has no negative off-diagonal entry, its diagnostic suggests reading it as a
/// graph adjacency matrix (`detail::is_graph_adjacency`).
///
/// ── M-matrix lumping ────────────────────────────────────────────────────────
/// A symmetric operator that is SPD but carries positive off-diagonals is not
/// an M-matrix, and handing it to the factorization unrepaired produces
/// NEGATIVE edge weights, which break `tree_elimination::sample_clique`: its
/// `prefix` array is a running sum of the sorted neighbour weights, and
/// `std::upper_bound` over it needs that array to be non-decreasing. One
/// negative weight and the sampler draws from a broken distribution, and the
/// bad fill spreads by additive merging.
///
/// How bad depends on how much of the off-diagonal MASS is positive, and the
/// range is wide (all measured 2026-08-20, CLI, T=8, tol 1e-8, vs the same
/// build without the transform):
///   * thermal2 (840 of 7.35M entries, 0.0017% of the mass) SURVIVES it --
///     46 iterations unrepaired, 44 with lumping. A handful of broken draws in
///     a 1.2M-vertex elimination is simply not enough to matter.
///   * parabolic_fem (33% of entries, 0.00016% of the mass): 44 -> 42.
///   * bcsstk13 (44% of entries, 39.7% of the mass) is DESTROYED: unrepaired
///     it runs 500 iterations to a residual of NaN. Lumped (past the default
///     ceiling, so this needs a larger lump_max_mass) it stagnates at 29.4 instead
///     -- finite, still useless. That is the ceiling's whole point.
///
/// Lumping restores the class before the preconditioner is built. For every
/// positive off-diagonal pair a_ij = a_ji > 0 (i != j):
///
///     a_ii += a_ij,   a_jj += a_ij,   a_ij = a_ji = 0.
///
/// Equivalently, in one line:
///
///     Â = A + sum_{pairs with a_ij > 0}  a_ij (e_i - e_j)(e_i - e_j)^T.       (*)
///
/// (*) is the whole soundness argument:
///   * each summand is a POSITIVE multiple of a rank-1 PSD matrix, so Â >= A
///     in the Loewner order. A PSD => Â PSD; A PD => Â PD, since
///     x^T Â x = x^T A x + sum a_ij (x_i - x_j)^2. Â is therefore always a
///     valid PCG preconditioner source when A is.
///   * (e_i - e_j)^T 1 = 0, so Â 1 = A 1: every ROW SUM is preserved exactly.
///     Diagonal dominance is preserved and in fact strengthened — row i's
///     dominance margin grows by 2 * (its positive off-diagonal mass).
///   * off-diagonals are non-positive by construction, so the transform is
///     idempotent and the edge weights it hands the sampler are all >= 0.
///   * A <= Â means every eigenvalue of Â^{-1} A lies in (0, 1], and
///     Â - A has rank at most the number of positive off-diagonal PAIRS, so in
///     exact arithmetic CG on the true operator with an exact Â preconditioner
///     terminates in at most (that many + 1) iterations.
///
/// The transform is applied ONLY to the matrix the preconditioner is built
/// from. PCG keeps applying the TRUE operator A, so the residual it reports and
/// the answer it returns are for the system the caller asked about — the same
/// contract rchol uses for its own non-SDD compensation (arXiv:2011.07769
/// §5.2.1, where apache2 and G3_circuit are benchmarked as published).
///
/// Everything here works on the library's own CSC storage (csc_matrix.h).

#include "apxchol/detail/csc_matrix.h"
#include "apxchol/detail/operator_options.h"
#include <cstdint>
#include <string>

namespace apxchol::detail {

/// One-pass structural census of a candidate operator matrix.
///
/// Column-major facts stand in for row-major ones wherever the two coincide for
/// a symmetric matrix — legitimate because `require_operator` rejects an
/// asymmetric matrix before anything else reads these fields.
struct operator_scan {
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::int64_t stored_nnz   = 0;  ///< stored entries, explicit zeros included
    std::int64_t nonfinite    = 0;  ///< ... whose value is NaN or +-Inf
    std::int64_t offdiag_nnz  = 0;  ///< stored off-diagonal entries with value != 0
    std::int64_t offdiag_pos  = 0;  ///< ... with a strictly positive value
    /// Columns with at least one nonzero entry.
    std::int64_t nonempty     = 0;
    /// ... of which carry a strictly positive diagonal entry.
    std::int64_t diag_pos     = 0;
    /// Non-empty columns whose diagonal is <= 0 (missing counts as 0). A PSD
    /// matrix cannot have one: e_i^T A e_i = a_ii would be <= 0 with a nonzero
    /// row. This is the hard diagonal condition.
    std::int64_t bad_diag     = 0;
    std::int64_t first_bad_diag = -1;
    double       first_bad_diag_value = 0.0;

    /// Off-diagonal entries whose transpose partner differs by more than
    /// `kSymmetryRelTol` relatively (a missing partner counts as 0).
    std::int64_t asymmetric   = 0;
    /// The strictly lower and strictly upper triangles hold the same entries
    /// bit for bit: every column is strictly index-sorted (so no duplicate
    /// coordinates), no off-diagonal entry is an explicit zero or non-finite,
    /// and the two triangles' entry fingerprints agree. Stronger than
    /// `asymmetric == 0`, which tolerates `kSymmetryRelTol`. A consumer that
    /// needs the canonical LOWER value of an upper entry may then read the
    /// entry itself instead of searching for its transpose partner.
    bool         triangles_bit_identical = false;
    double       worst_asym_rel = 0.0;
    std::int64_t worst_asym_row = -1, worst_asym_col = -1;
    double       worst_asym_a = 0.0, worst_asym_b = 0.0;

    /// Sum over stored off-diagonals of |a_ij|, and of a_ij where a_ij > 0.
    /// Their ratio is what the lumping ceiling is expressed in.
    double offdiag_abs_mass = 0.0;
    double offdiag_pos_mass = 0.0;

    /// Rows breaking diagonal dominance, i.e. with a negative row sum.
    /// DIAGNOSTIC ONLY — never gates a run (G3_circuit / apache2 have them and
    /// converge). Because lumping preserves row sums exactly, this count is
    /// also the number of rows the LUMPED matrix fails to dominate.
    std::int64_t deficient_rows = 0;
    double       worst_row_sum  = 0.0;
    std::int64_t worst_row      = -1;
    /// Rows carrying a positive diagonal excess (the SDDM surplus).
    std::int64_t excess_rows = 0;

    bool square() const { return rows == cols; }
    /// Fraction of the off-diagonal |mass| that lumping would delete. This —
    /// not the entry COUNT — is what predicts the damage: a handful of huge
    /// positive entries hurts more than many tiny ones (see
    /// `lump_mass_ceiling`).
    double positive_mass_fraction() const {
        return offdiag_abs_mass > 0.0 ? offdiag_pos_mass / offdiag_abs_mass : 0.0;
    }
};

/// Relative tolerance on |a_ij - a_ji| before an entry counts as asymmetric.
inline constexpr double kSymmetryRelTol = 1e-10;

/// Scan operator properties without materializing a transpose. Symmetry
/// fingerprints provide a fast path; otherwise a second pass binary-searches
/// transpose entries. Uses per-thread witness storage.
operator_scan scan_operator(csc_ref<double> A);

/// Assert that `A` belongs to the class apxchol is defined on, and return the
/// census. Throws `std::invalid_argument` naming WHICH condition failed and
/// with what counts; when the matrix has no negative off-diagonal entry, the
/// message also names the conversion to a graph Laplacian on every surface.
///
/// Diagonal dominance is reported in the census but never enforced.
operator_scan require_operator(csc_ref<double> A,
                               double lump_max_mass = kDefaultLumpMassCeiling);

/// The error require_operator reports for a rows x cols matrix that is not square.
[[noreturn]] void throw_operator_not_square(std::int64_t rows, std::int64_t cols);

/// The M-matrix lumping of `A` as a new matrix; `removed` receives the number
/// of stored positive off-diagonal ENTRIES removed (two per pair, both
/// triangles). Every entry that ends up exactly zero is dropped, as by
/// Eigen's prune(0, 0); the remaining entries keep their order.
///
/// Precondition: `A` is symmetric and every non-empty row carries a stored
/// positive diagonal — i.e. `require_operator` has accepted it. That is what
/// makes the transform structure-free: the mass only ever moves onto a
/// diagonal entry that already exists, so nothing is inserted.
csc_matrix<double> lump_positive_offdiagonals(csc_ref<double> A, std::int64_t& removed);

/// Validated, factorization-ready view of a caller's operator.
///
/// Construction asserts the operator contract and, when the matrix carries
/// positive off-diagonals below the requested mass ceiling,
/// holds the LUMPED copy. `matrix()` is what the preconditioner must be built
/// from; the caller's matrix is never touched, which is what keeps the true
/// operator PCG applies bit-identical.
///
/// When nothing needs lumping — every Laplacian, every SDDM matrix — no copy
/// is made and `matrix()` is a reference to the caller's own matrix.
class operator_view {
public:
    explicit operator_view(csc_ref<double> A,
                           double lump_max_mass = kDefaultLumpMassCeiling);

    /// The lumped copy, or the caller's own matrix when nothing was lumped.
    csc_ref<double> matrix() const {
        return lumped_entries_ > 0 ? lumped_.ref() : src_;
    }
    /// Stored positive off-diagonal entries lumped away (0 = untouched input).
    std::int64_t lumped() const { return lumped_entries_; }
    const operator_scan& scan() const { return scan_; }

private:
    csc_ref<double> src_;
    csc_matrix<double> lumped_;
    operator_scan scan_;
    std::int64_t lumped_entries_ = 0;
};

/// A one-line summary of a scanned operator:
/// how it was classified, what was lumped, and any dominance violations.
std::string describe_operator(const operator_scan& s, std::int64_t lumped);

/// The rule behind the CLI's `--convert-input`, Python's `apxchol.as_operator`
/// and MATLAB/Octave's `apxchol_as_operator`.
/// Returns true when M has off-diagonal entries and none is negative (a graph
/// adjacency matrix: use its Laplacian), false otherwise (use M unchanged; the
/// solver validates it). Throws `std::invalid_argument` when M is not square,
/// finite and symmetric.
bool is_graph_adjacency(csc_ref<double> M);

/// The error is_graph_adjacency reports for a matrix that is not square.
[[noreturn]] void throw_adjacency_not_square(std::int64_t rows, std::int64_t cols);

/// The graph Laplacian L = D - M, with D the diagonal of row sums of M. A
/// diagonal entry of M (a self-loop) adds to D and is subtracted again, so it
/// cancels: L_ii is the sum of row i's off-diagonal weights.
csc_matrix<double> adjacency_laplacian(csc_ref<double> M);

} // namespace apxchol::detail

// Eigen overloads of these checks for the CLI, the bindings and the benchmark,
// which still hold Eigen matrices.
#include "apxchol/detail/eigen_boundary.h"
