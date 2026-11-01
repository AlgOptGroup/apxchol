#pragma once
// Internal host preparation of P A P^T for device solvers. The caller supplies
// the validated operator and the original-to-permuted bijection of its factor.
#include "apxchol/csc_work.h"
#include "apxchol/solver/detail/lower_triangle_csr.h"
#include "apxchol/types.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <omp.h>

namespace apxchol::detail {

struct permuted_operator_csr {
    std::vector<int> ptr;
    std::unique_ptr<int[]> idx;
    std::unique_ptr<double[]> vals;
    bool fp32_exact = true;
};

// A fully stored symmetric CSC column k owns row perm[k] of the permuted
// symmetric CSR, avoiding atomic scatter. Values and fp32_exact use only the
// canonical lower triangle, even when upper values differ within tolerance.
//
// Return false for unsorted, duplicate or unpaired storage so the caller can
// choose a fallback. On false, out.ptr is scratch and the other members are
// unchanged.
inline bool try_build_permuted_symmetric_csr(
        csc_ref<double> L,
        const std::vector<node_index>& perm,
        permuted_operator_csr& out) {
    if (static_cast<std::uint64_t>(L.n) > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        static_cast<std::uint64_t>(L.nnz()) > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        return false;
    const int n = static_cast<int>(L.n);
    const edge_index* outer = L.ptr;
    const node_index* inner = L.idx;
    const double* input = L.val;
    auto& row_ptr = out.ptr;
    row_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    bool eligible = true, exact = true;
    std::int64_t lower = 0, upper = 0;
    // Balance by stored entries; each column owns a disjoint output row.
    #pragma omp parallel reduction(&& : eligible, exact) reduction(+ : lower, upper)
    {
        const auto [c_lo, c_hi] = work_balanced_range(
            outer, n, omp_get_thread_num(), omp_get_num_threads());
        for (int col = c_lo; col < c_hi; ++col) {
            row_ptr[perm[col] + 1] = static_cast<int>(outer[col + 1] - outer[col]);
            for (edge_index p = outer[col]; p < outer[col + 1]; ++p) {
                const int row = static_cast<int>(inner[p]);
                if (p > outer[col] && static_cast<int>(inner[p - 1]) >= row) eligible = false;
                if (row < col) {
                    ++upper;
                } else {
                    if (row > col) ++lower;
                    if (static_cast<double>(static_cast<float>(input[p])) != input[p])
                        exact = false;
                }
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
        const auto [c_lo, c_hi] = work_balanced_range(
            outer, n, omp_get_thread_num(), omp_get_num_threads());
        for (int col = c_lo; col < c_hi; ++col) {
            entries.clear();
            entries.reserve(static_cast<std::size_t>(outer[col + 1] - outer[col]));
            for (edge_index p = outer[col]; p < outer[col + 1]; ++p) {
                const int row = static_cast<int>(inner[p]);
                const double value = canonical_lower_value(outer, inner, input, col, row, p, paired);
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
    out.idx = std::move(out_idx);
    out.vals = std::move(out_vals);
    out.fp32_exact = exact;
    return true;
}

// Build sorted, full-symmetric CSR of P L P^T from L's canonical lower triangle.
// Requires a valid permutation; the caller must ensure that the expanded
// operator fits int offsets. Duplicate entries remain separate, ordered by
// value. fp32_exact reports lossless FP32 storage.
inline permuted_operator_csr build_permuted_full_symmetric_csr(
    csc_ref<double> L,
    const std::vector<node_index>& perm)
{
    permuted_operator_csr out;
    if (try_build_permuted_symmetric_csr(L, perm, out))
        return out;
    out.ptr.resize(static_cast<std::size_t>(L.n) + 1);
    out.fp32_exact = reflect_lower_to_csr<true>(L, std::span<int>(out.ptr),
        [p = perm.data()](std::int64_t i) { return static_cast<int>(p[i]); },
        [&](int count) {
            out.idx = std::make_unique_for_overwrite<int[]>(count);
            out.vals = std::make_unique_for_overwrite<double[]>(count);
            return std::pair{out.idx.get(), out.vals.get()};
        });
    return out;
}

} // namespace apxchol::detail
