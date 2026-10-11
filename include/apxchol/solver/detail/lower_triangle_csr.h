#pragma once
#include <Eigen/Sparse>
#include <algorithm>
#include <utility>
#include <vector>

namespace apxchol::detail {

// Reflect compressed L's lower triangle into sorted CSR, retaining duplicates.
// map maps input vertices to output rows; allocate returns writable index/value
// arrays after the counts establish their size. Each scatter slot is written once.
template<bool CheckFp32 = false, class Map, class Allocate, class Less>
bool reflect_lower_to_csr(const Eigen::SparseMatrix<double>& L, int* row_ptr,
                          Map map, Allocate allocate, Less less) {
    const Eigen::Index n = L.rows();
    const int* outer = L.outerIndexPtr();
    const int* inner = L.innerIndexPtr();
    const double* input = L.valuePtr();
    std::fill_n(row_ptr, n + 1, 0);
    #pragma omp parallel for schedule(static)
    for (Eigen::Index col = 0; col < n; ++col) {
        const int target = map(col);
        for (int p = outer[col]; p < outer[col + 1]; ++p) {
            const int row = inner[p];
            if (row < col) continue;
            __atomic_fetch_add(&row_ptr[map(row) + 1], 1, __ATOMIC_RELAXED);
            if (row != col)
                __atomic_fetch_add(&row_ptr[target + 1], 1, __ATOMIC_RELAXED);
        }
    }
    for (Eigen::Index row = 0; row < n; ++row)
        row_ptr[row + 1] += row_ptr[row];
    const auto output = allocate(row_ptr[n]);
    int* indices = output.first;
    double* values = output.second;
    std::vector<int> pos(row_ptr, row_ptr + n);
    bool exact = true;
    #pragma omp parallel for schedule(static) reduction(&&:exact)
    for (Eigen::Index col = 0; col < n; ++col) {
        const int target = map(col);
        for (int p = outer[col]; p < outer[col + 1]; ++p) {
            const int row = inner[p];
            if (row < col) continue;
            const int mapped_row = map(row);
            const double value = input[p];
            if constexpr (CheckFp32)
                if (double(float(value)) != value) exact = false;
            const int slot = __atomic_fetch_add(&pos[mapped_row], 1, __ATOMIC_RELAXED);
            indices[slot] = target;
            values[slot] = value;
            if (row != col) {
                const int mirror = __atomic_fetch_add(&pos[target], 1, __ATOMIC_RELAXED);
                indices[mirror] = mapped_row;
                values[mirror] = value;
            }
        }
    }
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
            std::sort(entries.begin(), entries.end(), less);
            for (int p = begin; p < end; ++p) {
                indices[p] = entries[p - begin].first;
                values[p] = entries[p - begin].second;
            }
        }
    }
    return exact;
}

} // namespace apxchol::detail
