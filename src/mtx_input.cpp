#include "mtx_input.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace apxchol {
namespace {

/// Relative slack on a column sum before it counts as a dominance violation
/// (or as SDDM excess). Absorbs fp accumulation-order ulps.
constexpr double kRowSumTol = 1e-10;

std::string plural(Eigen::Index k, const char* singular, const char* pluralf) {
    return std::to_string(k) + " " + (k == 1 ? singular : pluralf);
}

} // namespace

input_scan scan_input(const Eigen::SparseMatrix<double>& M) {
    input_scan s;
    s.n = M.rows();

    // Column-major walk: one sequential pass, no per-row scatter.
    for (Eigen::Index k = 0; k < M.outerSize(); ++k) {
        double colsum = 0.0;
        double diag   = 0.0;
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
            const double v = it.value();
            colsum += v;
            if (it.row() == it.col()) {
                diag = v;
                if (v > 0.0) ++s.diag_pos;
            } else {
                ++s.offdiag_nnz;
                if (v > 0.0)      ++s.offdiag_pos;
                else if (v < 0.0) ++s.offdiag_neg;
                if (it.row() < it.col()) ++s.upper_nnz;
                else                     ++s.lower_nnz;
            }
        }
        const double slack = kRowSumTol * std::max(std::abs(diag), 1.0);
        if (colsum < -slack) {
            ++s.deficient_rows;
            if (colsum < s.worst_deficit) {
                s.worst_deficit = colsum;
                s.worst_row     = k;
            }
        } else if (colsum > slack) {
            ++s.excess_rows;
        }
    }
    return s;
}

adjacency_signature detect_adjacency_signature(const Eigen::SparseMatrix<double>& M) {
    adjacency_signature a;
    a.n = M.rows();

    for (Eigen::Index k = 0; k < M.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
            if (it.row() == it.col()) {
                // One positive diagonal entry is enough: an adjacency matrix
                // has none, so nothing further can make the signature hold.
                // Every assembled operator hits this in its first column,
                // which is what keeps the check free on valid input.
                if (it.value() > 0.0) {
                    a.any_positive_diagonal = true;
                    return a;
                }
            } else if (it.value() > 0.0) {
                ++a.positive_offdiag;
            }
        }
    return a;
}

input_kind resolve_input_kind(input_kind requested, const input_scan& s,
                              bool pattern_field, std::string& reason) {
    // ── Structural gate, whatever the requested kind ──
    // Off-diagonal entries confined to one triangle means the file was not
    // read as a symmetric matrix (a one-sided `general` file, say). Both
    // interpretations would then build a non-symmetric operator and PCG would
    // be solving something other than what the user thinks.
    if (s.offdiag_nnz > 0 && (s.upper_nnz == 0 || s.lower_nnz == 0))
        throw std::runtime_error(
            "matrix is not symmetric: all " + std::to_string(s.offdiag_nnz) +
            " off-diagonal entries are in the " +
            (s.lower_nnz == 0 ? std::string("upper") : std::string("lower")) +
            " triangle. apxchol needs a symmetric Laplacian/SDDM or adjacency "
            "matrix; if the file stores one triangle only, declare it "
            "`symmetric` in the MatrixMarket header so both triangles are read");

    const std::string offdiag_counts =
        plural(s.offdiag_pos, "positive off-diagonal entry", "positive off-diagonal entries") +
        " and " +
        plural(s.offdiag_neg, "negative off-diagonal entry", "negative off-diagonal entries");

    // ── Explicit request: honour it, but say when it looks wrong ──
    if (requested == input_kind::laplacian) {
        reason = "--input-kind laplacian";
        if (s.diag_pos == 0 && s.n > 0)
            reason += "; WARNING none of the " + std::to_string(s.n) +
                      " rows carries a positive diagonal entry, which a "
                      "Laplacian/SDDM always does — did you mean "
                      "--input-kind adjacency?";
        else if (s.offdiag_pos > 0)
            reason += "; WARNING " + std::to_string(s.offdiag_pos) +
                      " off-diagonal entries are positive, so they become "
                      "NEGATIVE edge weights";
        return input_kind::laplacian;
    }
    if (requested == input_kind::adjacency) {
        reason = "--input-kind adjacency";
        return input_kind::adjacency;
    }

    // ── automatic ──
    if (s.offdiag_nnz == 0) {
        reason = "no off-diagonal entries (a diagonal operator)";
        return input_kind::laplacian;
    }
    if (s.offdiag_pos == 0) {
        // Every off-diagonal is <= 0: cannot be an adjacency matrix.
        reason = "all " + std::to_string(s.offdiag_nnz) +
                 " off-diagonal entries are non-positive";
        return input_kind::laplacian;
    }
    if (s.offdiag_neg == 0) {
        // Every off-diagonal is >= 0: cannot be a Laplacian/SDDM operator,
        // whose off-diagonals are non-positive by definition.
        reason = pattern_field
            ? "MatrixMarket 'pattern' field, " + std::to_string(s.offdiag_nnz) +
              " entries, unit weights"
            : "all " + std::to_string(s.offdiag_nnz) +
              " off-diagonal entries are positive, which a Laplacian/SDDM "
              "cannot have";
        return input_kind::adjacency;
    }

    // Mixed signs. An assembled operator always carries an explicit diagonal;
    // an adjacency matrix carries (essentially) none. Use that, and refuse
    // when even that does not settle it.
    if (s.diag_pos == s.n) {
        reason = "mixed-sign off-diagonals (" + offdiag_counts +
                 "), but every one of the " + std::to_string(s.n) +
                 " rows carries a positive diagonal — an assembled operator";
        return input_kind::laplacian;
    }
    throw std::runtime_error(
        "cannot tell whether this matrix is an assembled Laplacian/SDDM "
        "operator or a graph adjacency matrix: it has " + offdiag_counts +
        ", and only " + std::to_string(s.diag_pos) + " of " +
        std::to_string(s.n) + " rows carry a positive diagonal entry. "
        "Re-run with --input-kind laplacian (solve the matrix as given) or "
        "--input-kind adjacency (solve L = D - A)");
}

void adjacency_to_laplacian(Eigen::SparseMatrix<double>& M) {
    const Eigen::Index n = M.rows();
    M.makeCompressed();

    // A = |off-diagonal of M|, negated in place; the diagonal (self-loops,
    // which cancel between D and A) is zeroed and pruned away.
    std::vector<double> deg(static_cast<std::size_t>(n), 0.0);
    for (Eigen::Index k = 0; k < M.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
            if (it.row() == it.col()) { it.valueRef() = 0.0; continue; }
            const double w = std::abs(it.value());
            it.valueRef() = -w;
            deg[static_cast<std::size_t>(it.row())] += w;
        }
    M.prune(0.0, 0.0);   // drop the zeroed diagonal (and any explicit zero)

    Eigen::SparseMatrix<double> D(n, n);
    D.reserve(Eigen::VectorXi::Constant(static_cast<int>(n), 1));
    for (Eigen::Index i = 0; i < n; ++i)
        D.insert(i, i) = deg[static_cast<std::size_t>(i)];
    D.makeCompressed();

    Eigen::SparseMatrix<double> L = M + D;
    L.makeCompressed();
    M.swap(L);
}

std::string describe_input(input_kind kind, const input_scan& s,
                           const std::string& reason) {
    if (kind == input_kind::adjacency)
        return "input: ADJACENCY graph — solving the Laplacian L = D - A (" +
               reason + ")";

    // The diagonal-excess wording only means anything once the matrix really
    // looks like an assembled operator; on a diagonal-free adjacency matrix
    // read as `laplacian` the "excess" is just the vertex degree.
    std::string tail;
    if (s.diag_pos < s.n)
        tail = "; only " + std::to_string(s.diag_pos) + " of " +
               std::to_string(s.n) + " rows carry a positive diagonal entry";
    else if (s.deficient_rows > 0)
        tail = "; " + std::to_string(s.deficient_rows) +
               " of " + std::to_string(s.n) + " rows are not diagonally dominant";
    else if (s.excess_rows > 0)
        tail = "; SDDM, " + std::to_string(s.excess_rows) +
               " rows carry a positive diagonal excess";
    else
        tail = "; pure Laplacian, every row sums to zero";
    return "input: LAPLACIAN/SDDM operator — solving the matrix as given (" +
           reason + tail + ")";
}

std::string diagnose_failure(input_kind kind, const input_scan& s) {
    std::vector<std::string> found;
    if (kind == input_kind::laplacian && s.offdiag_pos > 0)
        found.push_back(std::to_string(s.offdiag_pos) +
                        " off-diagonal entries are positive, so they become "
                        "negative edge weights and generate no fill");
    if (kind == input_kind::laplacian && s.diag_pos < s.n)
        found.push_back("only " + std::to_string(s.diag_pos) + " of " +
                        std::to_string(s.n) + " rows carry a positive diagonal "
                        "entry — if this file is a graph adjacency/pattern "
                        "matrix, re-run with --input-kind adjacency");
    if (s.deficient_rows > 0)
        found.push_back(std::to_string(s.deficient_rows) + " of " +
                        std::to_string(s.n) + " rows break diagonal dominance "
                        "(worst row sum " + std::to_string(s.worst_deficit) +
                        " at row " + std::to_string(s.worst_row) + "), so the "
                        "operator need not be positive semidefinite");
    if (found.empty()) return {};

    std::string out = " Detected: ";
    for (std::size_t i = 0; i < found.size(); ++i) {
        if (i) out += "; ";
        out += found[i];
    }
    return out + ".";
}

} // namespace apxchol
