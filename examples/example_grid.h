#pragma once
/// Shared helper for the examples: a k×k 5-point grid Laplacian
/// (weighted graph Laplacian; singular, rank n−1) in CSC arrays.
#include "apxchol/views.h"
#include <cstdint>
#include <vector>

/// CSC arrays owned by the caller; view() borrows them for apxchol calls.
struct csc_arrays {
    std::vector<std::int32_t> col_ptr, row_idx;
    std::vector<double> values;

    apxchol::csc_view view() const {
        return {static_cast<std::int64_t>(col_ptr.size()) - 1, col_ptr, row_idx, values};
    }
};

inline csc_arrays grid_laplacian(int k) {
    csc_arrays A;
    A.col_ptr.reserve(static_cast<size_t>(k) * k + 1);
    A.col_ptr.push_back(0);
    auto id = [k](int i, int j) { return i * k + j; };
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) {
            // Rows in ascending order: up, left, diagonal, right, down.
            const bool up = i > 0, left = j > 0, right = j + 1 < k, down = i + 1 < k;
            const double degree = up + left + right + down;
            auto entry = [&](int row, double value) {
                A.row_idx.push_back(row);
                A.values.push_back(value);
            };
            if (up) entry(id(i - 1, j), -1.0);
            if (left) entry(id(i, j - 1), -1.0);
            entry(id(i, j), degree);
            if (right) entry(id(i, j + 1), -1.0);
            if (down) entry(id(i + 1, j), -1.0);
            A.col_ptr.push_back(static_cast<std::int32_t>(A.row_idx.size()));
        }
    return A;
}
