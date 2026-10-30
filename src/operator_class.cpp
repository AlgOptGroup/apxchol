#include "apxchol/operator_class.h"

#include "apxchol/csc_work.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

namespace apxchol {
namespace {

/// Relative slack on a row sum before it counts as a dominance violation (or as
/// SDDM excess). Absorbs fp accumulation-order ulps.
constexpr double kRowSumRelTol = 1e-10;

std::string count_of(Eigen::Index k, Eigen::Index n, const char* what) {
    return std::to_string(k) + " of " + std::to_string(n) + " " + what;
}

std::string num(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    return buf;
}

/// Half-open stored range of column `k`, valid for a compressed OR uncompressed
/// SparseMatrix. Eigen keeps each column's inner indices sorted either way,
/// which is what the binary search below relies on.
struct col_range {
    const int* inner;
    const double* value;
    Eigen::Index begin, end;
};

/// Merge sums in thread order for reproducibility at a fixed team size.
/// Witnesses use deterministic tie-breaking. Align per-thread storage to
/// cache-line boundaries to avoid false sharing during the scan.
struct alignas(64) witness {
    double abs_mass = 0.0;
    double pos_mass = 0.0;
    Eigen::Index bad_diag = -1;
    double bad_diag_value = 0.0;
    double asym_rel = 0.0;
    Eigen::Index asym_row = -1, asym_col = -1;
    double asym_a = 0.0, asym_b = 0.0;
    double row_sum = 0.0;
    Eigen::Index row = -1;
};

col_range column(const Eigen::SparseMatrix<double>& A, Eigen::Index k) {
    const int* outer = A.outerIndexPtr();
    const Eigen::Index b = outer[k];
    const Eigen::Index e = A.isCompressed() ? outer[k + 1]
                                            : b + A.innerNonZeroPtr()[k];
    return {A.innerIndexPtr(), A.valuePtr(), b, e};
}

/// A(row, col) by binary search inside `A` itself. 0 when not stored.
/// No transpose is materialized, so the symmetry test costs no memory.
double lookup(const Eigen::SparseMatrix<double>& A, Eigen::Index row,
              Eigen::Index col) {
    const col_range c = column(A, col);
    const int* first = c.inner + c.begin;
    const int* last  = c.inner + c.end;
    const int* it = std::lower_bound(first, last, static_cast<int>(row));
    if (it == last || *it != static_cast<int>(row)) return 0.0;
    return c.value[c.begin + (it - first)];
}

} // namespace

operator_scan scan_operator(const Eigen::SparseMatrix<double>& A) {
    operator_scan s;
    s.rows = A.rows();
    s.cols = A.cols();
    if (!s.square() || s.cols == 0) return s;

    // Column-major walk, one pass. Counts go through OpenMP reductions; the
    // WITNESSES (which row, which entry) are kept per-thread and merged
    // serially afterwards with a tie-break that does not depend on the
    // schedule -- smallest index wins every tie -- so the message a user sees
    // is the same on every run and at every thread count. No critical section
    // anywhere: G3_circuit has 665k rows that break dominance, and one
    // critical entry each would serialize the whole scan.
    const int nt = omp_get_max_threads();
    std::vector<witness> wit(static_cast<std::size_t>(nt));

    Eigen::Index stored = 0, nonfinite = 0, offdiag = 0, offpos = 0;
    Eigen::Index nonempty = 0, diagpos = 0, baddiag = 0;
    Eigen::Index asym = 0, deficient = 0, excess = 0;
    Eigen::Index upper = 0, lower = 0;
    // Symmetry is decided in two stages. This pass streams an order-independent
    // fingerprint of every nonzero off-diagonal entry into one sum per triangle
    // (detail::symmetric_entry_hash) and records whether each column is strictly
    // index-sorted. Equal sums and counts on a sorted, duplicate-free matrix mean
    // the two triangles are bit-identical, which passes the relative-tolerance
    // test trivially: `asymmetric` is 0 and there is no witness to report. Only
    // otherwise does the second pass below search the transpose partner of every
    // lower entry, exactly as this scan always did. That search costs about
    // log2(column length) cache misses per entry and lands in the hub columns of
    // power-law and IPM operators (9.2 ns per stored entry on as-Skitter against
    // 0.7 on a grid), so it is worth paying only when something is actually
    // asymmetric.
    std::uint64_t hash_lower = 0, hash_upper = 0;
    Eigen::Index offdiag_zero = 0;
    bool ordered = true;

    // schedule(static), not dynamic: it fixes which columns a thread owns, so
    // the per-thread mass sums merged below are reproducible.
    #pragma omp parallel for schedule(static) \
        reduction(+ : stored, nonfinite, offdiag, offpos, nonempty, diagpos, \
                      baddiag, deficient, excess, upper, lower, offdiag_zero, \
                      hash_lower, hash_upper) \
        reduction(&& : ordered)
    for (Eigen::Index k = 0; k < s.cols; ++k) {
        const int tid = omp_get_thread_num();
        witness& w = wit[static_cast<std::size_t>(tid)];
        const col_range c = column(A, k);
        double colsum = 0.0, diag = 0.0;
        bool any_nonzero = false;
        for (Eigen::Index p = c.begin; p < c.end; ++p) {
            const Eigen::Index i = c.inner[p];
            const double v = c.value[p];
            if (p > c.begin && c.inner[p - 1] >= c.inner[p]) ordered = false;
            ++stored;
            if (!std::isfinite(v)) { ++nonfinite; continue; }
            colsum += v;
            if (v != 0.0) any_nonzero = true;
            if (i == k) { diag = v; continue; }
            if (v == 0.0) { ++offdiag_zero; continue; }   // explicit zero: no edge, no sign
            ++offdiag;
            w.abs_mass += std::abs(v);
            if (v > 0.0) { ++offpos; w.pos_mass += v; }
            if (i > k) {
                ++lower;
                hash_lower += detail::symmetric_entry_hash(
                    static_cast<std::uint64_t>(k), static_cast<std::uint64_t>(i), 0, v);
            } else {
                ++upper;
                hash_upper += detail::symmetric_entry_hash(
                    static_cast<std::uint64_t>(i), static_cast<std::uint64_t>(k), 0, v);
            }
        }
        if (any_nonzero) {
            ++nonempty;
            if (diag > 0.0) {
                ++diagpos;
            } else {
                ++baddiag;
                if (w.bad_diag < 0 || k < w.bad_diag) {
                    w.bad_diag = k;
                    w.bad_diag_value = diag;
                }
            }
        }
        const double slack = kRowSumRelTol * std::max(std::abs(diag), 1.0);
        if (colsum < -slack) {
            ++deficient;
            if (colsum < w.row_sum || (colsum == w.row_sum && k < w.row)) {
                w.row_sum = colsum;
                w.row = k;
            }
        } else if (colsum > slack) {
            ++excess;
        }
    }

    const bool triangles_match =
        ordered && lower == upper && hash_lower == hash_upper;
    s.triangles_bit_identical =
        triangles_match && offdiag_zero == 0 && nonfinite == 0;
    if (!triangles_match) {
        // Symmetry: check the LOWER triangle against its transpose partner.
        // Distinct lower entries map to distinct upper positions, so once the
        // two counts are known equal, a match on every lower entry means every
        // upper entry is matched too. Same columns per thread as the pass
        // above, and the witness tie-break does not depend on the schedule.
        #pragma omp parallel for schedule(static) reduction(+ : asym)
        for (Eigen::Index k = 0; k < s.cols; ++k) {
            const int tid = omp_get_thread_num();
            witness& w = wit[static_cast<std::size_t>(tid)];
            const col_range c = column(A, k);
            for (Eigen::Index p = c.begin; p < c.end; ++p) {
                const Eigen::Index i = c.inner[p];
                const double v = c.value[p];
                if (i <= k || v == 0.0 || !std::isfinite(v)) continue;
                const double t = lookup(A, k, i);
                const double scale = std::max(std::abs(v), std::abs(t));
                const double d = std::abs(v - t);
                if (d > kSymmetryRelTol * scale) {
                    ++asym;
                    const double rel = scale > 0.0 ? d / scale : d;
                    if (rel > w.asym_rel ||
                        (rel == w.asym_rel && i < w.asym_row)) {
                        w.asym_rel = rel;
                        w.asym_row = i; w.asym_col = k; w.asym_a = v; w.asym_b = t;
                    }
                }
            }
        }
    }

    s.stored_nnz = stored;
    s.nonfinite  = nonfinite;
    s.offdiag_nnz = offdiag;
    s.offdiag_pos = offpos;
    s.nonempty = nonempty;
    s.diag_pos = diagpos;
    s.bad_diag = baddiag;
    s.asymmetric = asym;
    s.deficient_rows = deficient;
    s.excess_rows = excess;

    // Structural counterpart of the value test: an upper entry with no lower
    // partner is invisible to it, but it does unbalance the two counts.
    if (upper != lower) s.asymmetric += std::max(upper, lower) - std::min(upper, lower);

    for (const witness& w : wit) {
        s.offdiag_abs_mass += w.abs_mass;
        s.offdiag_pos_mass += w.pos_mass;
        if (w.bad_diag >= 0 && (s.first_bad_diag < 0 || w.bad_diag < s.first_bad_diag)) {
            s.first_bad_diag = w.bad_diag;
            s.first_bad_diag_value = w.bad_diag_value;
        }
        if (w.asym_row >= 0 && (w.asym_rel > s.worst_asym_rel ||
                                (w.asym_rel == s.worst_asym_rel &&
                                 w.asym_row < s.worst_asym_row))) {
            s.worst_asym_rel = w.asym_rel;
            s.worst_asym_row = w.asym_row;
            s.worst_asym_col = w.asym_col;
            s.worst_asym_a = w.asym_a;
            s.worst_asym_b = w.asym_b;
        }
        if (w.row >= 0 && (w.row_sum < s.worst_row_sum ||
                           (w.row_sum == s.worst_row_sum && w.row < s.worst_row))) {
            s.worst_row_sum = w.row_sum;
            s.worst_row = w.row;
        }
    }
    return s;
}

operator_scan require_operator(const Eigen::SparseMatrix<double>& A, double lump_max_mass) {
    detail::validate_lump_max_mass(lump_max_mass);
    const std::string kContract =
        "apxchol solves symmetric SDDM / Laplacian operators (A = A^T, "
        "a_ii > 0, a_ij <= 0 for i != j). ";

    if (A.rows() != A.cols())
        throw std::invalid_argument(
            kContract + "This matrix is not square: " +
            std::to_string(A.rows()) + " rows, " + std::to_string(A.cols()) +
            " columns.");

    const operator_scan s = scan_operator(A);

    if (s.nonfinite > 0)
        throw std::invalid_argument(
            kContract + "Condition FAILED: finite values. " +
            count_of(s.nonfinite, s.stored_nnz, "stored entries") +
            " are NaN or infinite.");

    if (s.asymmetric > 0) {
        std::string msg = kContract + "Condition FAILED: symmetry. " +
            std::to_string(s.asymmetric) + " of " +
            std::to_string(s.offdiag_nnz) +
            " off-diagonal entries differ from their transpose partner by more "
            "than " + num(kSymmetryRelTol) + " relative";
        if (s.worst_asym_row >= 0)
            msg += " (worst at (" + std::to_string(s.worst_asym_row) + ", " +
                   std::to_string(s.worst_asym_col) + "): " +
                   num(s.worst_asym_a) + " vs " + num(s.worst_asym_b) + ")";
        msg += ". If the file stores one triangle only, declare it `symmetric` "
               "in the MatrixMarket header so both triangles are read.";
        throw std::invalid_argument(msg);
    }

    if (s.bad_diag > 0) {
        std::string msg = kContract + "Condition FAILED: positive diagonal. " +
            count_of(s.bad_diag, s.nonempty, "non-empty rows") +
            " carry a diagonal entry that is missing or <= 0";
        if (s.first_bad_diag >= 0)
            msg += " (first at row " + std::to_string(s.first_bad_diag) +
                   ", a_ii = " + num(s.first_bad_diag_value) + ")";
        msg += "; a positive semidefinite operator cannot have one, since "
               "e_i^T A e_i = a_ii.";
        // An adjacency matrix fails exactly here. Name the reading that fixes it.
        if (detail::is_adjacency_matrix(A))
            msg += " The matrix looks like a graph ADJACENCY matrix (symmetric, "
                   "nonnegative off-diagonals, zero diagonal), not an assembled "
                   "operator. Read it as a graph: apxchol.operator(M, \"adjacency\") "
                   "or apxchol.operator(M, \"auto\") in Python, "
                   "apxchol_operator(M, 'adjacency') or apxchol_operator(M, 'auto') "
                   "in MATLAB/Octave, `--input-kind adjacency` or `--input-kind auto` "
                   "on the CLI.";
        throw std::invalid_argument(msg);
    }

    // Off-diagonal sign: NOT a failure on its own — `operator_view` repairs it
    // by M-matrix lumping. It only fails past the ceiling, where lumping would
    // delete more of the operator than it preserves.
    if (s.offdiag_pos > 0) {
        const double rho = s.positive_mass_fraction();
        if (lump_max_mass == 0 || rho > lump_max_mass)
            throw std::invalid_argument(
                kContract + "Condition FAILED: non-positive off-diagonals. " +
                count_of(s.offdiag_pos, s.offdiag_nnz, "off-diagonal entries") +
                " are positive, carrying " + num(100.0 * rho) +
                "% of the off-diagonal mass; lump_max_mass permits at most " +
                num(100.0 * lump_max_mass) + "%.");
    }

    return s;
}

Eigen::Index lump_positive_offdiagonals(Eigen::SparseMatrix<double>& A) {
    if (A.rows() != A.cols() || A.rows() == 0) return 0;
    A.makeCompressed();

    const Eigen::Index n = A.cols();
    const int* outer = A.outerIndexPtr();
    const int* inner = A.innerIndexPtr();
    double* value = A.valuePtr();

    // PASS 1: zero every positive off-diagonal and accumulate, per column, the
    // mass it carried. Column k holds a_ik for all i, so its positive mass is
    // (by symmetry) row k's — exactly what a_kk must absorb.
    std::vector<double> add(static_cast<std::size_t>(n), 0.0);
    Eigen::Index removed = 0;
    #pragma omp parallel for schedule(dynamic, 1024) reduction(+ : removed)
    for (Eigen::Index k = 0; k < n; ++k) {
        double p = 0.0;
        for (Eigen::Index q = outer[k]; q < outer[k + 1]; ++q) {
            if (inner[q] == k) continue;
            const double v = value[q];
            if (v > 0.0) { p += v; value[q] = 0.0; ++removed; }
        }
        add[static_cast<std::size_t>(k)] = p;
    }
    if (removed == 0) return 0;

    // PASS 2: move the mass onto the diagonal. `require_operator` has already
    // established that every row carrying entries carries a positive diagonal,
    // so the slot always exists and nothing is inserted.
    #pragma omp parallel for schedule(dynamic, 1024)
    for (Eigen::Index k = 0; k < n; ++k) {
        const double p = add[static_cast<std::size_t>(k)];
        if (p == 0.0) continue;
        for (Eigen::Index q = outer[k]; q < outer[k + 1]; ++q)
            if (inner[q] == k) { value[q] += p; break; }
    }

    // Drop the zeroed entries: a stored 0 would otherwise reach make_graph as a
    // zero-WEIGHT edge and inflate the vertex degrees it feeds the sampler.
    A.prune(0.0, 0.0);
    return removed;
}

operator_view::operator_view(const Eigen::SparseMatrix<double>& A, double lump_max_mass)
    : src_(&A), scan_(require_operator(A, lump_max_mass)) {
    if (scan_.offdiag_pos > 0) {
        // Only here does a copy happen: a Laplacian or SDDM operator (nothing
        // positive off the diagonal) is handed straight through.
        lumped_ = A;
        lumped_entries_ = lump_positive_offdiagonals(lumped_);
    }
}

std::string describe_operator(const operator_scan& s, Eigen::Index lumped) {
    std::string out = "operator: n = " + std::to_string(s.rows) + ", nnz = " +
                      std::to_string(s.stored_nnz);
    if (s.offdiag_pos == 0 && s.deficient_rows == 0 && s.excess_rows == 0)
        out += ", pure Laplacian (every row sums to zero)";
    else if (s.excess_rows > 0)
        out += ", SDDM (" + std::to_string(s.excess_rows) +
               " rows carry a positive diagonal excess)";
    if (lumped > 0)
        out += "; M-matrix lumping moved " + std::to_string(lumped) +
               " positive off-diagonal entries (" + std::to_string(lumped / 2) +
               " pairs, " + num(100.0 * s.positive_mass_fraction()) +
               "% of the off-diagonal mass) onto the diagonal — preconditioner "
               "only, PCG applies the operator as given";
    else if (s.offdiag_pos > 0)
        out += "; " + std::to_string(s.offdiag_pos) +
               " positive off-diagonal entries";
    if (s.deficient_rows > 0)
        out += "; " + count_of(s.deficient_rows, s.rows, "rows") +
               " break diagonal dominance (worst row sum " +
               num(s.worst_row_sum) + " at row " +
               std::to_string(s.worst_row) +
               "), so the operator need not be SDDM — reported, not enforced";
    return out;
}

namespace detail {

input_kind parse_input_kind(const std::string& name) {
    if (name == "operator") return input_kind::as_operator;
    if (name == "adjacency") return input_kind::adjacency;
    if (name == "auto") return input_kind::automatic;
    throw std::invalid_argument("input kind must be operator, adjacency or auto, got '" +
                                name + "'");
}

bool is_adjacency_matrix(const Eigen::SparseMatrix<double>& M) {
    if (M.rows() != M.cols()) return false;
    for (Eigen::Index k = 0; k < M.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it)
            if (!std::isfinite(it.value()) || it.value() < 0.0 ||
                (it.row() == it.col() && it.value() != 0.0))
                return false;
    return scan_operator(M).asymmetric == 0;
}

input_kind resolve_input_kind(const Eigen::SparseMatrix<double>& M, input_kind kind,
                              bool pattern) {
    if (kind != input_kind::automatic) return kind;
    try {
        require_operator(M);
        return input_kind::as_operator;
    } catch (const std::invalid_argument&) {
        return pattern || is_adjacency_matrix(M) ? input_kind::adjacency
                                                 : input_kind::as_operator;
    }
}

Eigen::SparseMatrix<double> adjacency_laplacian(const Eigen::SparseMatrix<double>& M) {
    if (M.rows() != M.cols())
        throw std::invalid_argument("adjacency matrix must be square");
    const Eigen::Index n = M.rows();
    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(static_cast<std::size_t>(M.nonZeros() + n));
    Eigen::VectorXd degree = Eigen::VectorXd::Zero(n);
    for (Eigen::Index k = 0; k < M.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
            const double w = std::abs(it.value());
            if (it.row() == it.col() || w == 0.0) continue;
            entries.emplace_back(it.row(), it.col(), -w);
            degree[it.row()] += w;
        }
    for (Eigen::Index i = 0; i < n; ++i)
        if (degree[i] != 0.0) entries.emplace_back(i, i, degree[i]);
    Eigen::SparseMatrix<double> L(n, n);
    L.setFromTriplets(entries.begin(), entries.end());
    return L;
}

} // namespace detail

} // namespace apxchol
