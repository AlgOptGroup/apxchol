#pragma once
// CUDA-free preparation for the GPU PCG operator. Internal only: the caller
// supplies the same valid original->permuted bijection used by the factor.
#include "apxchol/types.h"
#include <Eigen/Sparse>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace apxchol::detail {

// A fully stored symmetric CSC column k owns row perm[k] of the permuted
// symmetric CSR. This removes shared row counters and scatter cursors. The
// inner indices still need sorting after permutation. As in the general
// builder, values come from the canonical LOWER triangle and fp32_exact
// examines only that triangle; an accepted near-symmetric upper value must
// neither replace its lower partner nor change the precision decision.
//
// Return false for uncompressed, unsorted, duplicate or unpaired storage so
// the established general builder retains those cases. On false, row_ptr is
// scratch and the other outputs are unchanged. No CUDA runtime is required.
inline bool try_build_permuted_symmetric_csr(
        const Eigen::SparseMatrix<double>& L,
        const std::vector<node_index>& perm,
        std::vector<int>& row_ptr,
        std::unique_ptr<int[]>& col_idx,
        std::unique_ptr<double[]>& vals,
        std::int64_t& nnz,
        bool& fp32_exact) {
    if (!L.isCompressed() || L.rows() != L.cols() ||
        L.rows() > std::numeric_limits<int>::max() ||
        L.nonZeros() > std::numeric_limits<int>::max())
        return false;
    const int n = static_cast<int>(L.rows());
    const int* outer = L.outerIndexPtr();
    const int* inner = L.innerIndexPtr();
    const double* input = L.valuePtr();
    row_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    bool eligible = true, exact = true;
    std::int64_t lower = 0, upper = 0;
    #pragma omp parallel for schedule(static) reduction(&& : eligible, exact) reduction(+ : lower, upper)
    for (int col = 0; col < n; ++col) {
        row_ptr[perm[col] + 1] = outer[col + 1] - outer[col];
        for (int p = outer[col]; p < outer[col + 1]; ++p) {
            const int row = inner[p];
            if (p > outer[col] && inner[p - 1] >= row) eligible = false;
            if (row < col) {
                ++upper;
            } else {
                if (row > col) ++lower;
                if (static_cast<double>(static_cast<float>(input[p])) != input[p])
                    exact = false;
            }
        }
    }
    if (!eligible || lower != upper) return false;
    for (int row = 0; row < n; ++row) row_ptr[row + 1] += row_ptr[row];
    const int count = row_ptr[n];
    auto out_idx = std::make_unique_for_overwrite<int[]>(static_cast<std::size_t>(count));
    auto out_vals = std::make_unique_for_overwrite<double[]>(static_cast<std::size_t>(count));
    // Only now is every partner range known to be sorted. Keep the arrays
    // local until every upper entry has a canonical lower partner; balanced
    // triangle counts alone do not establish per-coordinate symmetry.
    bool paired = true;
    #pragma omp parallel reduction(&& : paired)
    {
        std::vector<std::pair<int, double>> entries;
        #pragma omp for schedule(static)
        for (int col = 0; col < n; ++col) {
            entries.clear();
            entries.reserve(static_cast<std::size_t>(outer[col + 1] - outer[col]));
            for (int p = outer[col]; p < outer[col + 1]; ++p) {
                const int row = inner[p];
                double value = input[p];
                if (row < col) {
                    const int* end = inner + outer[row + 1];
                    const int* partner = std::lower_bound(
                        inner + outer[row], end, col);
                    if (partner == end || *partner != col) {
                        paired = false;
                        continue;
                    }
                    value = input[partner - inner];
                }
                entries.emplace_back(static_cast<int>(perm[row]), value);
            }
            std::sort(entries.begin(), entries.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            int target = row_ptr[perm[col]];
            for (const auto& [index, value] : entries) {
                out_idx[target] = index;
                out_vals[target] = value;
                ++target;
            }
        }
    }
    if (!paired) return false;
    col_idx = std::move(out_idx);
    vals = std::move(out_vals);
    nnz = count;
    fp32_exact = exact;
    return true;
}

} // namespace apxchol::detail
