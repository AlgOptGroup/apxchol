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

Eigen::Index project_laplacian_rhs_components(
        const Eigen::SparseMatrix<double>& L, Eigen::VectorXd& b) {
    const Eigen::Index n = L.rows();
    if (L.cols() != n || b.size() != n)
        throw std::invalid_argument(
            "component RHS projection requires a square matrix and matching RHS");

    std::vector<Eigen::Index> component(static_cast<std::size_t>(n), -1);
    std::vector<Eigen::Index> stack;
    std::vector<double> sums;
    std::vector<Eigen::Index> counts;
    stack.reserve(static_cast<std::size_t>(n));

    for (Eigen::Index root = 0; root < n; ++root) {
        if (component[static_cast<std::size_t>(root)] >= 0) continue;
        const Eigen::Index id = static_cast<Eigen::Index>(sums.size());
        sums.push_back(0.0);
        counts.push_back(0);
        component[static_cast<std::size_t>(root)] = id;
        stack.push_back(root);
        while (!stack.empty()) {
            const Eigen::Index v = stack.back();
            stack.pop_back();
            sums[static_cast<std::size_t>(id)] += b[v];
            ++counts[static_cast<std::size_t>(id)];
            for (Eigen::SparseMatrix<double>::InnerIterator it(L, v); it; ++it) {
                const Eigen::Index u = it.row();
                if (u == v || it.value() == 0.0 ||
                    component[static_cast<std::size_t>(u)] >= 0)
                    continue;
                component[static_cast<std::size_t>(u)] = id;
                stack.push_back(u);
            }
        }
    }

    if (sums.size() == 1) return 1;
    for (Eigen::Index v = 0; v < n; ++v) {
        const std::size_t id = static_cast<std::size_t>(
            component[static_cast<std::size_t>(v)]);
        b[v] -= sums[id] / static_cast<double>(counts[id]);
    }
    return static_cast<Eigen::Index>(sums.size());
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
    if (kind == input_kind::as_operator && s.offdiag_pos > 0)
        found.push_back(std::to_string(s.offdiag_pos) +
                        " off-diagonal entries are positive, so they become "
                        "negative edge weights and generate no fill");
    if (kind == input_kind::as_operator && s.diag_pos < s.n)
        found.push_back("only " + std::to_string(s.diag_pos) + " of " +
                        std::to_string(s.n) + " rows carry a positive diagonal "
                        "entry — if this file is a graph adjacency/pattern "
                        "matrix, re-run with --input-kind adjacency or auto");
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
