#pragma once

#include "apxchol/solver/factorization.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace apxchol::detail {

// External factors expose mutable arrays. Validate them before setup or
// permutation indexing; internally generated factors do not need this scan.
inline void validate_adopted_factor(const factorization& factor) {
    const auto& L = factor.L;
    const std::size_t n = L.rows();
    if (n == 0 || n == std::numeric_limits<std::size_t>::max() ||
        L.outer_.size() != n + 1 || factor.perm.size() != n)
        throw std::invalid_argument("adopted factor has invalid dimensions");
    if (L.outer_.front() != 0 || L.outer_.back() != L.inner_.size() ||
        L.inner_.size() != L.vals_.size())
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

    const std::size_t m = n - !factor.sddm;
    std::size_t round = 0, round_end = 0;
    for (std::size_t col = 0; col < n; ++col) {
        const auto first = L.outer_[col], last = L.outer_[col + 1];
        if (first > last || last > L.vals_.size())
            throw std::invalid_argument("adopted factor column pointers are invalid");
        if (col < m && (first == last || L.inner_[first] != col || !(L.vals_[first] > 0)))
            throw std::invalid_argument("adopted factor requires a positive first diagonal");
        while (col >= round_end && round < factor.rounds.size())
            round_end += factor.rounds[round++].is_size;
        const auto dependency_begin = std::min(m, std::max(col + 1, round_end));
        for (auto p = first; p < last; ++p) {
            const auto row = L.inner_[p];
            if (row >= n || row < col || (p > first && row == col) ||
                !std::isfinite(L.vals_[p]))
                throw std::invalid_argument("adopted factor must be finite and lower triangular");
            if (row > col && row < dependency_begin)
                throw std::invalid_argument("adopted factor has a dependency within one round");
        }
    }
}

} // namespace apxchol::detail
