#include "apxchol/operator_class.h"

#include "apxchol/env_knobs.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

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

/// Per-thread accumulator for the scan below. Merged serially in thread order,
/// so both the fp sums and the reported witnesses are schedule-independent:
/// the witnesses tie-break on the smallest index, and the two mass sums are
/// added up in thread order over the fixed `schedule(static)` partition (no
/// `reduction(+:)` on a double anywhere — libgomp combines those in thread
/// COMPLETION order, which varies run to run; same rule as the fused PCG
/// passes in preconditioner.h).
struct witness {
    double abs_mass = 0.0;
    double pos_mass = 0.0;
    Eigen::Index bad_diag = -1;
    double bad_diag_value = 0.0;
    double asym_rel = 0.0;
    Eigen::Index asym_row = -1, asym_col = -1;
    double asym_a = 0.0, asym_b = 0.0;
    double row_sum = 0.0;
    Eigen::Index row = -1;
    // One cache line per thread: the per-column writes below would otherwise
    // false-share across the whole scan.
    char pad[64];
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
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
#endif
    std::vector<witness> wit(static_cast<std::size_t>(nt));

    Eigen::Index stored = 0, nonfinite = 0, offdiag = 0, offpos = 0;
    Eigen::Index nonempty = 0, diagpos = 0, baddiag = 0;
    Eigen::Index asym = 0, deficient = 0, excess = 0;
    Eigen::Index upper = 0, lower = 0;

    // schedule(static), not dynamic: it fixes which columns a thread owns, so
    // the per-thread mass sums merged below are reproducible.
    #pragma omp parallel for schedule(static) \
        reduction(+ : stored, nonfinite, offdiag, offpos, nonempty, diagpos, \
                      baddiag, asym, deficient, excess, upper, lower)
    for (Eigen::Index k = 0; k < s.cols; ++k) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        witness& w = wit[static_cast<std::size_t>(tid)];
        const col_range c = column(A, k);
        double colsum = 0.0, diag = 0.0;
        bool any_nonzero = false;
        for (Eigen::Index p = c.begin; p < c.end; ++p) {
            const Eigen::Index i = c.inner[p];
            const double v = c.value[p];
            ++stored;
            if (!std::isfinite(v)) { ++nonfinite; continue; }
            colsum += v;
            if (v != 0.0) any_nonzero = true;
            if (i == k) { diag = v; continue; }
            if (v == 0.0) continue;         // explicit zero: no edge, no sign
            ++offdiag;
            w.abs_mass += std::abs(v);
            if (v > 0.0) { ++offpos; w.pos_mass += v; }
            if (i > k) ++lower; else ++upper;
            // Symmetry: check the LOWER triangle against its transpose partner.
            // Distinct lower entries map to distinct upper positions, so once
            // the two counts are known equal, a match on every lower entry
            // means every upper entry is matched too.
            if (i > k) {
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

operator_scan require_operator(const Eigen::SparseMatrix<double>& A) {
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
        // The adjacency signature is exactly this condition failing EVERYWHERE
        // while positive off-diagonals exist. Name the explicit conversion.
        if (s.diag_pos == 0 && s.offdiag_pos > 0)
            msg += " Not one of the " + std::to_string(s.nonempty) +
                   " non-empty rows carries a positive diagonal, while " +
                   std::to_string(s.offdiag_pos) +
                   " off-diagonal entries are positive: this is a graph "
                   "ADJACENCY matrix, not an assembled operator. Assemble the "
                   "Laplacian first — apxchol.laplacian(A) in Python, "
                   "apxchol_laplacian(A) in Octave/MATLAB, `--input-kind "
                   "adjacency` on the CLI.";
        throw std::invalid_argument(msg);
    }

    // Off-diagonal sign: NOT a failure on its own — `operator_view` repairs it
    // by M-matrix lumping. It only fails past the ceiling, where lumping would
    // delete more of the operator than it preserves.
    if (s.offdiag_pos > 0) {
        const auto& k = detail::env_knobs::get();
        const double rho = s.positive_mass_fraction();
        const double ceiling = k.lump_max_mass > 0.0 ? k.lump_max_mass
                                                     : kDefaultLumpMassCeiling;
        if (k.lump && rho > ceiling)
            throw std::invalid_argument(
                kContract + "Condition FAILED: non-positive off-diagonals. " +
                count_of(s.offdiag_pos, s.offdiag_nnz, "off-diagonal entries") +
                " are positive, carrying " + num(100.0 * rho) +
                "% of the off-diagonal mass. apxchol repairs a nearly-SDDM "
                "operator by lumping positive off-diagonals onto the diagonal "
                "(row sums preserved, the preconditioner only — PCG keeps "
                "applying the matrix you passed), but past " +
                num(100.0 * ceiling) +
                "% of the mass that deletes more of the operator than it "
                "keeps and the factorization stops paying for itself. Raise "
                "the ceiling with APXCHOL_LUMP_MAX=<fraction> if you want it "
                "anyway.");
        // APXCHOL_LUMP=0 exists so the two arms of an A/B can both RUN, so it
        // lets the violation through rather than refusing -- but loudly, and
        // not only under APXCHOL_VERBOSE: the factorization is about to be
        // handed negative edge weights, which is a preconditioner-quality
        // failure with no other symptom than a bad residual at the end.
        if (!k.lump)
            std::fprintf(stderr,
                "[apxchol] WARNING: M-matrix lumping is disabled "
                "(APXCHOL_LUMP=0) and %lld of %lld off-diagonal entries are "
                "positive (%s%% of the off-diagonal mass). They become "
                "NEGATIVE edge weights, which breaks the clique sampler's "
                "prefix distribution -- expect a poor preconditioner.\n",
                static_cast<long long>(s.offdiag_pos),
                static_cast<long long>(s.offdiag_nnz),
                num(100.0 * rho).c_str());
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

operator_view::operator_view(const Eigen::SparseMatrix<double>& A)
    : src_(&A), scan_(require_operator(A)) {
    const auto& knobs = detail::env_knobs::get();
    if (scan_.offdiag_pos > 0 && knobs.lump) {
        // Only here does a copy happen: a Laplacian or SDDM operator (nothing
        // positive off the diagonal) is handed straight through.
        lumped_ = A;
        lumped_entries_ = lump_positive_offdiagonals(lumped_);
    }
    if (std::getenv("APXCHOL_VERBOSE"))
        std::fprintf(stderr, "[apxchol] %s\n",
                     describe_operator(scan_, lumped_entries_).c_str());
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
               " positive off-diagonal entries left in place (APXCHOL_LUMP=0)";
    if (s.deficient_rows > 0)
        out += "; " + count_of(s.deficient_rows, s.rows, "rows") +
               " break diagonal dominance (worst row sum " +
               num(s.worst_row_sum) + " at row " +
               std::to_string(s.worst_row) +
               "), so the operator need not be SDDM — reported, not enforced";
    return out;
}

} // namespace apxchol
