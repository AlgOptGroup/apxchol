#include "apxchol/operator_class.h"

#include "apxchol/csc_work.h"
#include "apxchol/detail/csc_input.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

namespace apxchol::detail {
namespace {

/// Relative slack on a row sum before it counts as a dominance violation (or as
/// SDDM excess). Absorbs fp accumulation-order ulps.
constexpr double kRowSumRelTol = 1e-10;

std::string count_of(std::int64_t k, std::int64_t n, const char* what) {
    return std::to_string(k) + " of " + std::to_string(n) + " " + what;
}

std::string num(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    return buf;
}

/// Half-open stored range of column `k`. Binary searches below need sorted
/// columns; unsorted input only loses their exactness, which the scan's own
/// `ordered` flag reports.
struct col_range {
    const node_index* inner;
    const double* value;
    std::int64_t begin, end;
};

/// Merge sums in thread order for reproducibility at a fixed team size.
/// Witnesses use deterministic tie-breaking. Align per-thread storage to
/// cache-line boundaries to avoid false sharing during the scan.
struct alignas(64) witness {
    double abs_mass = 0.0;
    double pos_mass = 0.0;
    std::int64_t bad_diag = -1;
    double bad_diag_value = 0.0;
    double asym_rel = 0.0;
    std::int64_t asym_row = -1, asym_col = -1;
    double asym_a = 0.0, asym_b = 0.0;
    double row_sum = 0.0;
    std::int64_t row = -1;
};

col_range column(csc_ref<double> A, std::int64_t k) {
    return {A.idx, A.val, static_cast<std::int64_t>(A.ptr[k]),
            static_cast<std::int64_t>(A.ptr[k + 1])};
}

/// A(row, col) by binary search inside `A` itself. 0 when not stored.
/// No transpose is materialized, so the symmetry test costs no memory.
double lookup(csc_ref<double> A, std::int64_t row, std::int64_t col) {
    const col_range c = column(A, col);
    const node_index* first = c.inner + c.begin;
    const node_index* last  = c.inner + c.end;
    const node_index* it = std::lower_bound(first, last, static_cast<node_index>(row));
    if (it == last || *it != static_cast<node_index>(row)) return 0.0;
    return c.value[c.begin + (it - first)];
}

} // namespace

operator_scan scan_operator(csc_ref<double> A) {
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

    std::int64_t stored = 0, nonfinite = 0, offdiag = 0, offpos = 0;
    std::int64_t nonempty = 0, diagpos = 0, baddiag = 0;
    std::int64_t asym = 0, deficient = 0, excess = 0;
    std::int64_t upper = 0, lower = 0;
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
    std::int64_t offdiag_zero = 0;
    bool ordered = true;

    // schedule(static), not dynamic: it fixes which columns a thread owns, so
    // the per-thread mass sums merged below are reproducible.
    #pragma omp parallel for schedule(static) \
        reduction(+ : stored, nonfinite, offdiag, offpos, nonempty, diagpos, \
                      baddiag, deficient, excess, upper, lower, offdiag_zero, \
                      hash_lower, hash_upper) \
        reduction(&& : ordered)
    for (std::int64_t k = 0; k < s.cols; ++k) {
        const int tid = omp_get_thread_num();
        witness& w = wit[static_cast<std::size_t>(tid)];
        const col_range c = column(A, k);
        double colsum = 0.0, diag = 0.0;
        bool any_nonzero = false;
        for (std::int64_t p = c.begin; p < c.end; ++p) {
            const std::int64_t i = c.inner[p];
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
        for (std::int64_t k = 0; k < s.cols; ++k) {
            const int tid = omp_get_thread_num();
            witness& w = wit[static_cast<std::size_t>(tid)];
            const col_range c = column(A, k);
            for (std::int64_t p = c.begin; p < c.end; ++p) {
                const std::int64_t i = c.inner[p];
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

static const char* const kConversion =
    "Input conversion needs a square, finite, symmetric matrix. ";

// Shared by the operator check and the input conversion.
static std::string nonfinite_entries(const operator_scan& s) {
    return count_of(s.nonfinite, s.stored_nnz, "stored entries") + " are NaN or infinite.";
}

static std::string asymmetry(const operator_scan& s) {
    std::string msg = std::to_string(s.asymmetric) + " of " +
        std::to_string(s.offdiag_nnz) +
        " off-diagonal entries differ from their transpose partner by more "
        "than " + num(kSymmetryRelTol) + " relative";
    if (s.worst_asym_row >= 0)
        msg += " (worst at (" + std::to_string(s.worst_asym_row) + ", " +
               std::to_string(s.worst_asym_col) + "): " +
               num(s.worst_asym_a) + " vs " + num(s.worst_asym_b) + ")";
    return msg + ". If the file stores one triangle only, declare it `symmetric` "
                 "in the MatrixMarket header so both triangles are read.";
}

// A symmetric matrix with no negative off-diagonal is read as a graph by the
// conversion helpers; name them when the operator reading fails.
static std::string adjacency_hint(const operator_scan& s) {
    if (s.offdiag_pos != s.offdiag_nnz) return {};
    return " The matrix has no negative off-diagonal entry, so it may be a graph "
           "ADJACENCY matrix rather than an assembled operator. To use its "
           "Laplacian L = D - A, pass --convert-input on the CLI, or call "
           "apxchol.as_operator(M) in Python or apxchol_as_operator(M) in "
           "MATLAB/Octave.";
}

static const char* const kContract =
    "apxchol solves symmetric SDDM / Laplacian operators (A = A^T, "
    "a_ii > 0, a_ij <= 0 for i != j). ";

void throw_operator_not_square(std::int64_t rows, std::int64_t cols) {
    throw std::invalid_argument(
        std::string(kContract) + "This matrix is not square: " +
        std::to_string(rows) + " rows, " + std::to_string(cols) + " columns.");
}

operator_scan require_operator(csc_ref<double> A, double lump_max_mass) {
    detail::validate_lump_max_mass(lump_max_mass);
    const operator_scan s = scan_operator(A);

    if (s.nonfinite > 0)
        throw std::invalid_argument(
            std::string(kContract) + "Condition FAILED: finite values. " + nonfinite_entries(s));

    if (s.asymmetric > 0) {
        std::string msg = std::string(kContract) + "Condition FAILED: symmetry. " + asymmetry(s);
        throw std::invalid_argument(msg);
    }

    if (s.bad_diag > 0) {
        std::string msg = std::string(kContract) + "Condition FAILED: positive diagonal. " +
            count_of(s.bad_diag, s.nonempty, "non-empty rows") +
            " carry a diagonal entry that is missing or <= 0";
        if (s.first_bad_diag >= 0)
            msg += " (first at row " + std::to_string(s.first_bad_diag) +
                   ", a_ii = " + num(s.first_bad_diag_value) + ")";
        msg += "; a positive semidefinite operator cannot have one, since "
               "e_i^T A e_i = a_ii.";
        throw std::invalid_argument(msg + adjacency_hint(s));
    }

    // Off-diagonal sign: NOT a failure on its own — `operator_view` repairs it
    // by M-matrix lumping. It only fails past the ceiling, where lumping would
    // delete more of the operator than it preserves.
    if (s.offdiag_pos > 0) {
        const double rho = s.positive_mass_fraction();
        if (lump_max_mass == 0 || rho > lump_max_mass)
            throw std::invalid_argument(
                std::string(kContract) + "Condition FAILED: non-positive off-diagonals. " +
                count_of(s.offdiag_pos, s.offdiag_nnz, "off-diagonal entries") +
                " are positive, carrying " + num(100.0 * rho) +
                "% of the off-diagonal mass; lump_max_mass permits at most " +
                num(100.0 * lump_max_mass) + "%." + adjacency_hint(s));
    }

    return s;
}

// Value of a stored entry after lumping: positive off-diagonals become zero
// and the first stored diagonal entry gains the column's positive mass p.
static double lumped_value(node_index row, std::int64_t col, double v, double p, bool& absorbed) {
    if (row != static_cast<node_index>(col)) return v > 0.0 ? 0.0 : v;
    if (absorbed) return v;
    absorbed = true;
    return v + p;
}

csc_matrix<double> lump_positive_offdiagonals(csc_ref<double> A, std::int64_t& removed) {
    removed = 0;
    const std::int64_t n = A.n;
    const edge_index* outer = A.ptr;
    const node_index* inner = A.idx;
    const double* value = A.val;

    // PASS 1, per column k: the positive off-diagonal mass it carries (by
    // symmetry row k's, which a_kk absorbs) in stored order, and the entries
    // that stay. An entry stays unless its final value is exactly zero: a
    // positive off-diagonal is zeroed, a diagonal gains the mass, the rest
    // keep their value. Each column's work is serial, so the split cannot
    // change a value.
    std::vector<double> add(static_cast<std::size_t>(n), 0.0);
    std::vector<edge_index> kept(static_cast<std::size_t>(n) + 1, 0);
    std::int64_t positives = 0;
    #pragma omp parallel reduction(+ : positives)
    {
        const auto [lo, hi] = work_balanced_range(outer, n);
        for (std::int64_t k = lo; k < hi; ++k) {
            double p = 0.0;
            for (edge_index q = outer[k]; q < outer[k + 1]; ++q) {
                if (inner[q] == static_cast<node_index>(k)) continue;
                const double v = value[q];
                if (v > 0.0) { p += v; ++positives; }
            }
            add[static_cast<std::size_t>(k)] = p;
            edge_index count = 0;
            bool absorbed = p == 0.0;
            for (edge_index q = outer[k]; q < outer[k + 1]; ++q) {
                const double result = lumped_value(inner[q], k, value[q], p, absorbed);
                count += result != 0.0;
            }
            kept[static_cast<std::size_t>(k) + 1] = count;
        }
    }
    removed = positives;
    for (std::int64_t k = 0; k < n; ++k)
        kept[static_cast<std::size_t>(k) + 1] += kept[static_cast<std::size_t>(k)];

    // PASS 2: write the kept entries. `require_operator` has already
    // established that every row carrying entries carries a positive
    // diagonal, so the mass always lands on a stored slot and nothing is
    // inserted. Dropping the zeroed entries keeps a stored 0 from reaching
    // make_graph as a zero-WEIGHT edge that would inflate the vertex degrees
    // it feeds the sampler.
    csc_matrix<double> out(A.n, kept[static_cast<std::size_t>(n)]);
    #pragma omp parallel
    {
        const auto [lo, hi] = work_balanced_range(outer, n);
        for (std::int64_t k = lo; k < hi; ++k) {
            const double p = add[static_cast<std::size_t>(k)];
            edge_index at = kept[static_cast<std::size_t>(k)];
            out.ptr()[k] = at;
            bool absorbed = p == 0.0;
            for (edge_index q = outer[k]; q < outer[k + 1]; ++q) {
                const double result = lumped_value(inner[q], k, value[q], p, absorbed);
                if (result == 0.0) continue;
                out.idx()[at] = inner[q];
                out.val()[at] = result;
                ++at;
            }
        }
    }
    out.ptr()[n] = kept[static_cast<std::size_t>(n)];
    return out;
}

operator_view::operator_view(csc_ref<double> A, double lump_max_mass)
    : src_(A), scan_(require_operator(A, lump_max_mass)) {
    if (scan_.offdiag_pos > 0) {
        // Only here does a copy happen: a Laplacian or SDDM operator (nothing
        // positive off the diagonal) is handed straight through.
        lumped_ = lump_positive_offdiagonals(A, lumped_entries_);
    }
}

std::string describe_operator(const operator_scan& s, std::int64_t lumped) {
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

bool is_graph_adjacency(csc_ref<double> M) {
    const operator_scan s = scan_operator(M);
    if (s.nonfinite > 0)
        throw std::invalid_argument(std::string(kConversion) + nonfinite_entries(s));
    if (s.asymmetric > 0)
        throw std::invalid_argument(std::string(kConversion) + asymmetry(s));
    // Any negative off-diagonal marks an operator, which the solver validates.
    // A matrix without off-diagonal entries is a diagonal operator, not a graph.
    return s.offdiag_nnz > 0 && s.offdiag_pos == s.offdiag_nnz;
}

void throw_adjacency_not_square(std::int64_t rows, std::int64_t cols) {
    throw std::invalid_argument(std::string(kConversion) + "This matrix has " +
        std::to_string(rows) + " rows and " + std::to_string(cols) + " columns.");
}

csc_matrix<double> adjacency_laplacian(csc_ref<double> M) {
    const node_index n = M.n;
    std::vector<triplet> entries;
    entries.reserve(static_cast<std::size_t>(M.nnz()) + n);
    std::vector<double> degree(n, 0.0);
    // L = D - M: a self-loop m_ii adds to D_ii and is subtracted again.
    for (node_index k = 0; k < n; ++k)
        for (edge_index p = M.ptr[k]; p < M.ptr[k + 1]; ++p) {
            const node_index row = M.idx[p];
            const double w = M.val[p];
            if (row == k || w == 0.0) continue;
            entries.push_back({row, k, -w});
            degree[row] += w;
        }
    for (node_index i = 0; i < n; ++i)
        if (degree[i] != 0.0) entries.push_back({i, i, degree[i]});
    return csc_from_triplets(n, entries);
}

namespace {

[[noreturn]] void bad_view(const std::string& what) {
    throw std::invalid_argument("apxchol: invalid CSC view: " + what);
}

// Index p of a column that holds a row outside [0, n), out of order or
// repeated, or end when the column is valid.
template<class Index>
std::int64_t first_bad_row(const Index* rows, std::int64_t begin, std::int64_t end,
                           std::int64_t n) {
    for (std::int64_t p = begin; p < end; ++p) {
        const auto row = static_cast<std::int64_t>(rows[p]);
        if (row < 0 || row >= n || (p > begin && row <= static_cast<std::int64_t>(rows[p - 1])))
            return p;
    }
    return end;
}

} // namespace

template<class Index>
void csc_input::init(basic_csc_view<Index> A) {
    const std::int64_t n = A.n;
    if (n < 0) bad_view("n is negative");
    if (static_cast<std::uint64_t>(n) >= std::numeric_limits<node_index>::max())
        bad_view("n = " + std::to_string(n) + " exceeds this build's " +
                 std::to_string(sizeof(node_index) * 8) + "-bit vertex indices");
    if (n == 0 && A.col_ptr.empty() && A.row_idx.empty() && A.values.empty()) return;
    if (A.col_ptr.size() != static_cast<std::size_t>(n) + 1)
        bad_view("col_ptr has " + std::to_string(A.col_ptr.size()) + " entries, expected n+1 = " +
                 std::to_string(n + 1));
    const Index* ptr = A.col_ptr.data();
    if (ptr[0] != 0) bad_view("col_ptr[0] must be 0");
    std::int64_t decreasing = n;
    #pragma omp parallel for schedule(static) reduction(min : decreasing)
    for (std::int64_t j = 0; j < n; ++j)
        if (ptr[j + 1] < ptr[j]) decreasing = std::min(decreasing, j);
    if (decreasing < n)
        bad_view("col_ptr decreases after column " + std::to_string(decreasing));
    const auto nnz = static_cast<std::int64_t>(ptr[n]);
    if (A.row_idx.size() != static_cast<std::size_t>(nnz) ||
        A.values.size() != static_cast<std::size_t>(nnz))
        bad_view("row_idx and values must have col_ptr[n] = " + std::to_string(nnz) +
                 " entries, got " + std::to_string(A.row_idx.size()) + " and " +
                 std::to_string(A.values.size()));
    if (static_cast<std::uint64_t>(nnz) > std::numeric_limits<edge_index>::max())
        bad_view(std::to_string(nnz) + " stored entries exceed this build's " +
                 std::to_string(sizeof(edge_index) * 8) + "-bit offsets");
    const Index* rows = A.row_idx.data();
    std::int64_t bad_column = n;
    #pragma omp parallel reduction(min : bad_column)
    {
        const auto [lo, hi] = work_balanced_range(ptr, n);
        for (std::int64_t j = lo; j < hi; ++j)
            if (first_bad_row(rows, ptr[j], ptr[j + 1], n) < static_cast<std::int64_t>(ptr[j + 1])) {
                bad_column = j;
                break;
            }
    }
    if (bad_column < n) {
        const std::int64_t begin = ptr[bad_column];
        const std::int64_t p = first_bad_row(rows, begin, ptr[bad_column + 1], n);
        const auto row = static_cast<std::int64_t>(rows[p]);
        const std::string where = "column " + std::to_string(bad_column) + " row " +
                                  std::to_string(row);
        if (row < 0 || row >= n) bad_view(where + " is outside [0, n)");
        bad_view(where + (row == static_cast<std::int64_t>(rows[p - 1]) ? " is repeated"
                                                                          : " is out of order") +
                 "; rows must be sorted with no duplicates within a column");
    }

    const edge_index* offsets = nullptr;
    if constexpr (sizeof(Index) == sizeof(edge_index)) {
        // Same-width signed and unsigned integers may alias; the values are nonnegative.
        offsets = reinterpret_cast<const edge_index*>(ptr);
    } else {
        offsets_.resize(static_cast<std::size_t>(n) + 1);
        edge_index* out = offsets_.data();
        #pragma omp parallel for schedule(static)
        for (std::int64_t j = 0; j <= n; ++j) out[j] = static_cast<edge_index>(ptr[j]);
        offsets = out;
    }
    const node_index* indices = nullptr;
    if constexpr (sizeof(Index) == sizeof(node_index)) {
        indices = reinterpret_cast<const node_index*>(rows);
    } else {
        rows_.resize(static_cast<std::size_t>(nnz));
        node_index* out = rows_.data();
        #pragma omp parallel for schedule(static)
        for (std::int64_t p = 0; p < nnz; ++p) out[p] = static_cast<node_index>(rows[p]);
        indices = out;
    }
    ref_ = {static_cast<node_index>(n), offsets, indices, A.values.data()};
}

csc_input::csc_input(csc_view A) { init(A); }
csc_input::csc_input(csc_view64 A) { init(A); }

namespace {

template<class Ref, class View>
Ref dense_view(View M) {
    if (M.rows < 0 || M.cols < 0)
        throw std::invalid_argument("apxchol: matrix view dimensions must be nonnegative");
    if (M.col_stride < M.rows)
        throw std::invalid_argument("apxchol: matrix view col_stride must be at least rows");
    if (M.data == nullptr && M.rows != 0 && M.cols != 0)
        throw std::invalid_argument("apxchol: matrix view data is null");
    return Ref(M.data, M.rows, M.cols, 1, M.col_stride);
}

} // namespace

const_matrix_ref dense_input(const_matrix_view M) { return dense_view<const_matrix_ref>(M); }
matrix_ref dense_input(matrix_view M) { return dense_view<matrix_ref>(M); }

} // namespace apxchol::detail
