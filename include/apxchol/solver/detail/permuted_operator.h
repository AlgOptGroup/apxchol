#pragma once
// Internal host preparation of P A P^T for device solvers. The caller supplies
// the validated operator and the original-to-permuted bijection of its factor.
#include "apxchol/csc_work.h"
#include "apxchol/types.h"
#include <Eigen/Sparse>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <omp.h>

namespace apxchol::detail {

// A fully stored symmetric CSC column k owns row perm[k] of the permuted
// symmetric CSR, avoiding atomic scatter. Values and fp32_exact use only the
// canonical lower triangle, even when upper values differ within tolerance.
//
// Return false for uncompressed, unsorted, duplicate or unpaired storage so
// the caller can choose a fallback. On false, row_ptr is scratch and the other
// outputs are unchanged.
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
    // Balance by stored entries; each column owns a disjoint output row.
    #pragma omp parallel reduction(&& : eligible, exact) reduction(+ : lower, upper)
    {
        const auto [c_lo, c_hi] = work_balanced_range(
            outer, n, omp_get_thread_num(), omp_get_num_threads());
        for (int col = c_lo; col < c_hi; ++col) {
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

// Build sorted, full-symmetric CSR of P L P^T from L's canonical lower triangle.
// Requires compressed square storage and a valid permutation; the caller must
// ensure that the expanded operator fits int offsets. Duplicate entries remain
// separate, ordered by value bits. fp32_exact reports lossless FP32 storage.
inline void build_permuted_full_symmetric_csr(
    const Eigen::SparseMatrix<double>& L,
    const std::vector<node_index>& perm,
    std::vector<int>& row_ptr,
    std::unique_ptr<int[]>& col_idx,
    std::unique_ptr<double[]>& vals,
    std::int64_t& nnz,
    bool& fp32_exact)
{
    if (try_build_permuted_symmetric_csr(
            L, perm, row_ptr, col_idx, vals, nnz, fp32_exact))
        return;
    const int n = static_cast<int>(L.rows());
    const int* L_outer = L.outerIndexPtr();
    const int* L_inner = L.innerIndexPtr();
    const double* L_vals = L.valuePtr();
    const node_index* p_idx = perm.data();

    // Count both mirrored entries of each off-diagonal lower value.
    row_ptr.assign(n + 1, 0);
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; ++k) {
        const int pk = p_idx[k];
        for (int p = L_outer[k]; p < L_outer[k + 1]; ++p) {
            const int row = L_inner[p];
            if (row < k) continue;
            const int pr = p_idx[row];
            __atomic_fetch_add(&row_ptr[pr + 1], 1, __ATOMIC_RELAXED);
            if (row != k)
                __atomic_fetch_add(&row_ptr[pk + 1], 1, __ATOMIC_RELAXED);
        }
    }
    for (int i = 0; i < n; ++i)
        row_ptr[i + 1] += row_ptr[i];
    const int total = row_ptr[n];
    nnz = total;
    // The scatter writes every slot, allowing parallel first touch.
    col_idx = std::make_unique_for_overwrite<int[]>(static_cast<std::size_t>(total));
    vals    = std::make_unique_for_overwrite<double[]>(static_cast<std::size_t>(total));

    // Scatter and check FP32 exactness in the same pass over the lower triangle.
    std::vector<int> pos(row_ptr.begin(), row_ptr.begin() + n);
    bool exact = true;
    #pragma omp parallel for schedule(static) reduction(&&:exact)
    for (int k = 0; k < n; ++k) {
        const int pk = p_idx[k];
        for (int p = L_outer[k]; p < L_outer[k + 1]; ++p) {
            const int row = L_inner[p];
            if (row < k) continue;
            const double v = L_vals[p];
            if (static_cast<double>(static_cast<float>(v)) != v) exact = false;  // lossless-fp32 check
            const int pr = p_idx[row];
            // A_perm[pr, pk] = v
            const int slot_pr = __atomic_fetch_add(&pos[pr], 1, __ATOMIC_RELAXED);
            col_idx[slot_pr] = pk;
            vals[slot_pr]    = v;
            if (row != k) {
                // A_perm[pk, pr] = v
                const int slot_pk = __atomic_fetch_add(&pos[pk], 1, __ATOMIC_RELAXED);
                col_idx[slot_pk] = pr;
                vals[slot_pk]    = v;
            }
        }
    }
    fp32_exact = exact;

    // Sort by column and then value bits so duplicate order is independent
    // of the threads' scatter order. Reuse one scratch buffer per worker.
    #pragma omp parallel
    {
        std::vector<std::pair<int, double>> kv;
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            const int rs = row_ptr[i], re = row_ptr[i + 1];
            if (re - rs < 2) continue;
            kv.clear();
            kv.reserve(re - rs);
            for (int p = rs; p < re; ++p)
                kv.emplace_back(col_idx[p], vals[p]);
            std::sort(kv.begin(), kv.end(), [](const auto& a, const auto& b) {
                if (a.first != b.first) return a.first < b.first;
                return std::bit_cast<std::uint64_t>(a.second) < std::bit_cast<std::uint64_t>(b.second);
            });
            for (int p = rs; p < re; ++p) {
                col_idx[p] = kv[p - rs].first;
                vals[p]    = kv[p - rs].second;
            }
        }
    }
}

} // namespace apxchol::detail
