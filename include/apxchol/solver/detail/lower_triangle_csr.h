#pragma once
#include <Eigen/Sparse>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace apxchol::detail {

// Reflect compressed L's lower triangle into sorted CSR, retaining duplicates
// ordered by value. map maps input vertices to output rows; allocate returns
// writable index/value arrays after the counts establish their size. Each
// scatter slot is written once. Non-finite values are rejected before sorting.
template<bool CheckFp32 = false, class Map, class Allocate>
bool reflect_lower_to_csr(const Eigen::SparseMatrix<double>& L, std::span<int> row_ptr,
                          Map map, Allocate allocate) {
    const Eigen::Index n = L.rows();
    const int* outer = L.outerIndexPtr();
    const int* inner = L.innerIndexPtr();
    const double* input = L.valuePtr();
    std::fill_n(row_ptr.begin(), n + 1, 0);
    #pragma omp parallel for schedule(static)
    for (Eigen::Index col = 0; col < n; ++col) {
        const int target = map(col);
        for (int p = outer[col]; p < outer[col + 1]; ++p) {
            const int row = inner[p];
            if (row < col) continue;
            std::atomic_ref(row_ptr[map(row) + 1]).fetch_add(1, std::memory_order_relaxed);
            if (row != col)
                std::atomic_ref(row_ptr[target + 1]).fetch_add(1, std::memory_order_relaxed);
        }
    }
    for (Eigen::Index row = 0; row < n; ++row)
        row_ptr[row + 1] += row_ptr[row];
    const auto output = allocate(row_ptr[n]);
    int* indices = output.first;
    double* values = output.second;
    std::vector<int> pos(row_ptr.begin(), row_ptr.begin() + n);
    bool exact = true, finite = true;
    #pragma omp parallel for schedule(static) reduction(&&:exact, finite)
    for (Eigen::Index col = 0; col < n; ++col) {
        const int target = map(col);
        for (int p = outer[col]; p < outer[col + 1]; ++p) {
            const int row = inner[p];
            if (row < col) continue;
            const int mapped_row = map(row);
            const double value = input[p];
            if (!std::isfinite(value)) finite = false;
            if constexpr (CheckFp32)
                if (double(float(value)) != value) exact = false;
            const int slot = std::atomic_ref(pos[mapped_row]).fetch_add(1, std::memory_order_relaxed);
            indices[slot] = target;
            values[slot] = value;
            if (row != col) {
                const int mirror = std::atomic_ref(pos[target]).fetch_add(1, std::memory_order_relaxed);
                indices[mirror] = mapped_row;
                values[mirror] = value;
            }
        }
    }
    if (!finite)
        throw std::invalid_argument("operator contains a non-finite value");
    #pragma omp parallel
    {
        std::vector<std::pair<int, double>> entries;
        #pragma omp for schedule(static)
        for (Eigen::Index row = 0; row < n; ++row) {
            const int begin = row_ptr[row], end = row_ptr[row + 1];
            if (end - begin < 2) continue;
            entries.clear();
            entries.reserve(end - begin);
            for (int p = begin; p < end; ++p)
                entries.emplace_back(indices[p], values[p]);
            std::sort(entries.begin(), entries.end());
            for (int p = begin; p < end; ++p) {
                indices[p] = entries[p - begin].first;
                values[p] = entries[p - begin].second;
            }
        }
    }
    return exact;
}

} // namespace apxchol::detail
