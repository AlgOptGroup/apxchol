#pragma once

#include <Eigen/Sparse>
#include <concepts>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace apxchol::detail {

// Copy foreign CSC arrays into Eigen's 32-bit layout. Operator validation is
// separate; unsorted columns and duplicates retain setFromTriplets semantics.
template<std::integral Offset, std::integral Index>
Eigen::SparseMatrix<double> import_csc(std::int64_t n, const Offset* outer,
                                     const Index* inner, const double* values,
                                     int base = 0) {
    constexpr auto limit = std::numeric_limits<int>::max();
    if (n < 0 || n > limit)
        throw std::invalid_argument("n must be in [0, 2^31-1]");
    if (outer == nullptr)
        throw std::invalid_argument("colptr is NULL");
    if (base != 0 && base != 1)
        throw std::invalid_argument("index_base must be 0 or 1");
    if (!std::cmp_equal(outer[0], base))
        throw std::invalid_argument("colptr[0] must equal index_base");
    for (std::int64_t j = 0; j < n; ++j)
        if (outer[j + 1] < outer[j])
            throw std::invalid_argument("colptr must be nondecreasing");
    if (std::cmp_greater(outer[n], std::int64_t(limit) + base))
        throw std::invalid_argument("nnz exceeds 2^31-1");
    const auto nnz = static_cast<int>(outer[n] - base);
    if (nnz != 0 && (inner == nullptr || values == nullptr))
        throw std::invalid_argument("rowval or nzval is NULL");

    bool sorted = true;
    for (std::int64_t j = 0; j < n; ++j) {
        const auto begin = outer[j] - base, end = outer[j + 1] - base;
        for (auto p = begin; p < end; ++p) {
            const auto row = inner[p];
            if (std::cmp_less(row, base) || std::cmp_greater_equal(row, n + base))
                throw std::invalid_argument("CSC row index out of range");
            if (p > begin && row <= inner[p - 1]) sorted = false;
        }
    }

    const auto size = static_cast<int>(n);
    Eigen::SparseMatrix<double> A(size, size);
    if (sorted) {
        A.resizeNonZeros(nnz);
        for (std::int64_t j = 0; j <= n; ++j)
            A.outerIndexPtr()[j] = static_cast<int>(outer[j] - base);
        for (int p = 0; p < nnz; ++p) {
            A.innerIndexPtr()[p] = static_cast<int>(inner[p] - base);
            A.valuePtr()[p] = values[p];
        }
        return A;
    }

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(static_cast<std::size_t>(nnz));
    for (int j = 0; j < size; ++j)
        for (auto p = outer[j] - base; p < outer[j + 1] - base; ++p)
            triplets.emplace_back(static_cast<int>(inner[p] - base), j, values[p]);
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

} // namespace apxchol::detail
