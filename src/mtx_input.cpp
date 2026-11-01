#include "mtx_input.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace apxchol {
namespace {

/// Relative slack on a column sum before it counts as a dominance violation
/// (or as SDDM excess). Absorbs fp accumulation-order ulps.
constexpr double kRowSumTol = 1e-10;

} // namespace

input_scan scan_input(detail::csc_ref<double> M) {
    input_scan s;
    s.n = M.n;

    // Column-major walk: one sequential pass, no per-row scatter.
    for (std::int64_t k = 0; k < s.n; ++k) {
        double colsum = 0.0;
        double diag   = 0.0;
        for (edge_index p = M.ptr[k]; p < M.ptr[k + 1]; ++p) {
            const std::int64_t row = M.idx[p];
            const double v = M.val[p];
            colsum += v;
            if (row == k) {
                diag = v;
                if (v > 0.0) ++s.diag_pos;
            } else {
                ++s.offdiag_nnz;
                if (v > 0.0)      ++s.offdiag_pos;
                else if (v < 0.0) ++s.offdiag_neg;
                if (row < k) ++s.upper_nnz;
                else         ++s.lower_nnz;
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

std::int64_t project_laplacian_rhs_components(
        detail::csc_ref<double> L, std::span<double> b) {
    const std::int64_t n = L.n;
    if (std::cmp_not_equal(b.size(), n))
        throw std::invalid_argument(
            "component RHS projection requires a square matrix and matching RHS");

    std::vector<std::int64_t> component(static_cast<std::size_t>(n), -1);
    std::vector<std::int64_t> stack;
    std::vector<double> sums;
    std::vector<std::int64_t> counts;
    stack.reserve(static_cast<std::size_t>(n));

    for (std::int64_t root = 0; root < n; ++root) {
        if (component[static_cast<std::size_t>(root)] >= 0) continue;
        const std::int64_t id = static_cast<std::int64_t>(sums.size());
        sums.push_back(0.0);
        counts.push_back(0);
        component[static_cast<std::size_t>(root)] = id;
        stack.push_back(root);
        while (!stack.empty()) {
            const std::int64_t v = stack.back();
            stack.pop_back();
            sums[static_cast<std::size_t>(id)] += b[static_cast<std::size_t>(v)];
            ++counts[static_cast<std::size_t>(id)];
            for (edge_index p = L.ptr[v]; p < L.ptr[v + 1]; ++p) {
                const std::int64_t u = L.idx[p];
                if (u == v || L.val[p] == 0.0 ||
                    component[static_cast<std::size_t>(u)] >= 0)
                    continue;
                component[static_cast<std::size_t>(u)] = id;
                stack.push_back(u);
            }
        }
    }

    if (sums.size() == 1) return 1;
    for (std::int64_t v = 0; v < n; ++v) {
        const std::size_t id = static_cast<std::size_t>(
            component[static_cast<std::size_t>(v)]);
        b[static_cast<std::size_t>(v)] -= sums[id] / static_cast<double>(counts[id]);
    }
    return static_cast<std::int64_t>(sums.size());
}

std::string describe_input(bool graph, const input_scan& s,
                           const std::string& reason) {
    if (graph)
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

std::string diagnose_failure(bool graph, const input_scan& s) {
    std::vector<std::string> found;
    if (!graph && s.offdiag_pos > 0)
        found.push_back(std::to_string(s.offdiag_pos) +
                        " off-diagonal entries are positive, so they become "
                        "negative edge weights and generate no fill");
    if (!graph && s.diag_pos < s.n)
        found.push_back("only " + std::to_string(s.diag_pos) + " of " +
                        std::to_string(s.n) + " rows carry a positive diagonal "
                        "entry — if this file is a graph adjacency/pattern "
                        "matrix, re-run with --convert-input");
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
