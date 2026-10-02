#pragma once
// CUDA-free preparation for the GPU PCG operator. Internal only: the caller
// supplies the same valid original->permuted bijection used by the factor.
#include "apxchol/csc_work.h"
#include "apxchol/types.h"
#include <Eigen/Sparse>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

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
    // Columns are split by stored entries, not by count: the work of a column
    // is its length, and with the natural labelling the heaviest equal-count
    // chunk carries 2.0x the mean on the IPM normal equations and 4.0x on
    // as-Skitter (see detail::work_balanced_range). Both loops here write
    // per-column disjoint output and reduce only integers and booleans, so the
    // split cannot change a single stored byte.
    #pragma omp parallel reduction(&& : eligible, exact) reduction(+ : lower, upper)
    {
#ifdef _OPENMP
    const int tid = omp_get_thread_num(), nt = omp_get_num_threads();
#else
    const int tid = 0, nt = 1;
#endif
    const auto [c_lo, c_hi] = work_balanced_range(outer, n, tid, nt);
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
#ifdef _OPENMP
        const int tid = omp_get_thread_num(), nt = omp_get_num_threads();
#else
        const int tid = 0, nt = 1;
#endif
        const auto [c_lo, c_hi] = work_balanced_range(outer, n, tid, nt);
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

// Build full-symmetric CSR of A_perm = P L P^T from a (lower-half-stored)
// symmetric matrix L and its permutation P. The factor F_.L was built on
// A_perm, so running PCG in permuted space matches what trsv_.solve_LLt_dev
// expects per iter. Used by cuda_pcg::setup and CUDA-free, so any backend that
// runs its PCG in the permuted space can share it; requires compressed storage.
//
// perm.indices()[orig_v] = new_idx ⇒  A_perm[i,j] = L[iperm(i), iperm(j)]
// where iperm = P^{-1}. The permutation acts on BOTH row and col of L.
// Output: row_ptr/col_idx/vals = CSR of A_perm (full symmetric, sorted),
// nnz = row_ptr[n] (col_idx/vals hold exactly that many entries; they are
// plain arrays, not vectors — see the allocation note below).
//
// Fully paired, unique sorted CSC uses column ownership: source column k
// owns output row perm[k], retaining the sort by permuted column indices.
// The general fallback below counts and scatters through atomic row
// counters, then sorts each row. Both preserve canonical lower values.
// fp32_exact (out) := every operator value round-trips fp32 (v == double(float(v))),
// so storing A in fp32 is LOSSLESS. Computed FOR FREE as an OMP reduction in PASS 2's
// existing value loop -- no separate scan. (A is symmetric; PASS 2 visits the upper
// triangle incl. diagonal = every distinct value.) This is the "detect at input"
// gate that lets exact matrices use the half-size fp32 operator while Krylov compute
// stays fp64 (so the 1e-8 residual floor is preserved).
inline void build_permuted_full_symmetric_csr(
    const Eigen::SparseMatrix<double>& L,
    const std::vector<node_index>& perm,
    std::vector<int>& row_ptr,
    std::unique_ptr<int[]>& col_idx,
    std::unique_ptr<double[]>& vals,
    std::int64_t& nnz,
    bool& fp32_exact)
{
    if (detail::try_build_permuted_symmetric_csr(
            L, perm, row_ptr, col_idx, vals, nnz, fp32_exact))
        return;
    const int n = static_cast<int>(L.rows());
    const int* L_outer = L.outerIndexPtr();
    const int* L_inner = L.innerIndexPtr();
    const double* L_vals = L.valuePtr();
    // perm_[v] = new_idx for original vertex v.
    const node_index* p_idx = perm.data();

    // PASS 1 (parallel): atomic count per-row of A_perm.
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
    // Prefix sum (serial, m+1 entries — sub-ms even for n=4M).
    for (int i = 0; i < n; ++i)
        row_ptr[i + 1] += row_ptr[i];
    const int total = row_ptr[n];
    nnz = total;
    // UNINITIALIZED, deliberately: PASS 2 below writes every one of the
    // `total` slots exactly once (its scatter is the same walk PASS 1 just
    // counted), so a zero fill is pure waste -- and a SERIAL one, 80 MB of
    // int + 160 MB of double on grid_2000, memset on one thread and then
    // immediately overwritten. Same idiom (and same reason) as the fp32
    // operator cast in setup(). The prefix sum above stays serial: it is
    // n+1 entries, sub-ms even at n = 4M.
    // Worth less than it looks: the page faults just move from the memset
    // into PASS 2's (parallel) first touch, so the measured `gpu_pcg_setup`
    // win is only grid_2000 79.9 -> 77.7 ms, iter0040 64.3 -> 63.3 (medians
    // of 48, RTX 4090 Laptop, T=16, warm context). Kept because it is
    // strictly less work and strictly less peak-transient traffic.
    col_idx = std::make_unique_for_overwrite<int[]>(static_cast<std::size_t>(total));
    vals    = std::make_unique_for_overwrite<double[]>(static_cast<std::size_t>(total));

    // PASS 2 (parallel): atomic-claim slot, scatter. Non-deterministic
    // per-row order across threads; restored by per-row sort below. The fp32
    // exactness reduction rides along for free (every value v is read here anyway).
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

    // Sort each row's (col, val) ascending: sorted CSR gives the SpMV its
    // best locality on the x gathers. Per-thread kv buffer reused across
    // rows (avoids n tiny mallocs).
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
            std::sort(kv.begin(), kv.end(),
                      [](const auto& a, const auto& b){ return a.first < b.first; });
            for (int p = rs; p < re; ++p) {
                col_idx[p] = kv[p - rs].first;
                vals[p]    = kv[p - rs].second;
            }
        }
    }
}

} // namespace apxchol::detail
