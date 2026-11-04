#pragma once
#include "apxchol/detail/csc_matrix.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace apxchol::detail {

// Reflect L's lower triangle into sorted CSR, retaining duplicates ordered by
// value. map maps input vertices to output rows; allocate returns writable
// index/value arrays after the counts establish their size. Each scatter slot
// is written once. Non-finite values are rejected before sorting.
template<bool CheckFp32 = false, class Offset, class Map, class Allocate>
bool reflect_lower_to_csr(csc_ref<double> L, std::span<Offset> row_ptr,
                          Map map, Allocate allocate) {
    const std::int64_t n = L.n;
    const edge_index* outer = L.ptr;
    const node_index* inner = L.idx;
    const double* input = L.val;
    std::fill_n(row_ptr.begin(), n + 1, Offset(0));
    #pragma omp parallel for schedule(static)
    for (std::int64_t col = 0; col < n; ++col) {
        const auto target = map(col);
        for (edge_index p = outer[col]; p < outer[col + 1]; ++p) {
            const node_index row = inner[p];
            if (row < static_cast<node_index>(col)) continue;
            std::atomic_ref(row_ptr[map(row) + 1]).fetch_add(1, std::memory_order_relaxed);
            if (row != static_cast<node_index>(col))
                std::atomic_ref(row_ptr[target + 1]).fetch_add(1, std::memory_order_relaxed);
        }
    }
    for (std::int64_t row = 0; row < n; ++row)
        row_ptr[row + 1] += row_ptr[row];
    const auto output = allocate(row_ptr[n]);
    auto* indices = output.first;
    double* values = output.second;
    using Index = std::remove_pointer_t<decltype(indices)>;
    std::vector<Offset> pos(row_ptr.begin(), row_ptr.begin() + n);
    bool exact = true, finite = true;
    #pragma omp parallel for schedule(static) reduction(&&:exact, finite)
    for (std::int64_t col = 0; col < n; ++col) {
        const auto target = map(col);
        for (edge_index p = outer[col]; p < outer[col + 1]; ++p) {
            const node_index row = inner[p];
            if (row < static_cast<node_index>(col)) continue;
            const auto mapped_row = map(row);
            const double value = input[p];
            if (!std::isfinite(value)) finite = false;
            if constexpr (CheckFp32)
                if (double(float(value)) != value) exact = false;
            const Offset slot = std::atomic_ref(pos[mapped_row]).fetch_add(1, std::memory_order_relaxed);
            indices[slot] = static_cast<Index>(target);
            values[slot] = value;
            if (row != static_cast<node_index>(col)) {
                const Offset mirror = std::atomic_ref(pos[target]).fetch_add(1, std::memory_order_relaxed);
                indices[mirror] = static_cast<Index>(mapped_row);
                values[mirror] = value;
            }
        }
    }
    if (!finite)
        throw std::invalid_argument("operator contains a non-finite value");
    #pragma omp parallel
    {
        std::vector<std::pair<Index, double>> entries;
        #pragma omp for schedule(static)
        for (std::int64_t row = 0; row < n; ++row) {
            const Offset begin = row_ptr[row], end = row_ptr[row + 1];
            if (end - begin < 2) continue;
            entries.clear();
            entries.reserve(end - begin);
            for (Offset p = begin; p < end; ++p)
                entries.emplace_back(indices[p], values[p]);
            std::sort(entries.begin(), entries.end());
            for (Offset p = begin; p < end; ++p) {
                indices[p] = entries[p - begin].first;
                values[p] = entries[p - begin].second;
            }
        }
    }
    return exact;
}

} // namespace apxchol::detail
