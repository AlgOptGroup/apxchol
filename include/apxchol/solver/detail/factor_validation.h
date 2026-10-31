#pragma once

#include "apxchol/solver/factorization.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace apxchol::detail {

// Operator checks shared by adoption and portable archive creation.
inline void validate_adopted_operator(const Eigen::SparseMatrix<double>& A,
                                      const factorization& F, double lump_max_mass) {
    if (A.rows() == 0 || A.rows() != A.cols() || static_cast<Eigen::Index>(F.L.rows()) != A.rows())
        throw std::invalid_argument("solver: factorization dimension mismatch");
    require_operator(A, lump_max_mass);
    if (!F.sddm) {
        for (Eigen::Index col = 0; col < A.cols(); ++col) {
            double diagonal = 0, positive = 0, degree = 0;
            for (Eigen::SparseMatrix<double>::InnerIterator entry(A, col); entry; ++entry)
                if (entry.row() == entry.col()) diagonal = entry.value();
                else if (entry.value() > 0) positive += entry.value();
                else degree -= entry.value();
            diagonal += positive;
            if (diagonal - degree > diagonal * 1e-12)
                throw std::invalid_argument("a Laplacian factor cannot precondition an SDDM operator");
        }
    }
}

// External factors have caller-editable permutations and rounds, and may have
// released values. Validate them before setup or permutation indexing;
// internally generated factors do not need this scan.
inline void validate_adopted_factor(const factorization& factor) {
    const auto& L = factor.L;
    const auto outer = L.col_ptrs();
    const auto inner = L.row_indices();
    const auto vals = L.values();
    const std::size_t n = L.rows();
    if (n == 0 || n == std::numeric_limits<std::size_t>::max() ||
        outer.size() != n + 1 || factor.perm.size() != n)
        throw std::invalid_argument("adopted factor has invalid dimensions");
    if (outer.front() != 0 || outer.back() != inner.size() ||
        inner.size() != vals.size())
        throw std::invalid_argument("adopted factor has invalid or released CSC arrays");

    std::vector<bool> seen(n, false);
    for (const auto row : factor.perm) {
        if (row >= n || seen[row])
            throw std::invalid_argument("adopted factor permutation is not a bijection");
        seen[row] = true;
    }
    std::size_t covered = 0;
    for (const auto& round : factor.rounds) {
        if (round.is_size > n - covered)
            throw std::invalid_argument("adopted factor round sizes exceed its dimension");
        covered += round.is_size;
    }

    const bool installed = factor.storage != factor_storage_type::automatic;
    const bool fp16 = factor.storage == factor_storage_type::fp16;
    if (fp16 ? factor.scale.size() != n : !factor.scale.empty())
        throw std::invalid_argument("adopted factor needs one column scale per column for fp16 "
                                    "storage and none otherwise");
    for (const float s : factor.scale)
        if (!(s > 0) || !std::isfinite(s) || !std::isfinite(1.0f / s))
            throw std::invalid_argument("adopted factor scales must be positive and finite");

    const std::size_t m = n - !factor.sddm;
    std::size_t round = 0, round_end = 0;
    for (std::size_t col = 0; col < n; ++col) {
        const auto first = outer[col], last = outer[col + 1];
        if (first > last || last > vals.size())
            throw std::invalid_argument("adopted factor column pointers are invalid");
        if (col < m && (first == last || inner[first] != col || !(vals[first] > 0)))
            throw std::invalid_argument("adopted factor requires a positive first diagonal");
        while (col >= round_end && round < factor.rounds.size())
            round_end += factor.rounds[round++].is_size;
        const auto dependency_begin = std::min(m, std::max(col + 1, round_end));
        for (auto p = first; p < last; ++p) {
            const auto row = inner[p];
            if (row >= n || row < col || (p > first && row == col) ||
                !std::isfinite(vals[p]))
                throw std::invalid_argument("adopted factor must be finite and lower triangular");
            if (row > col && row < dependency_begin)
                throw std::invalid_argument("adopted factor has a dependency within one round");
            if (installed && row >= m)
                throw std::invalid_argument("an installed Laplacian factor has an empty last row and column");
            if (fp16 && row != col &&
                static_cast<float>(static_cast<_Float16>(detail::sparse_csc_access::vals(L)[p])) != detail::sparse_csc_access::vals(L)[p])
                throw std::invalid_argument("fp16 factor off-diagonals must be binary16 values");
        }
    }
}

} // namespace apxchol::detail
