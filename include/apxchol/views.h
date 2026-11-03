#pragma once
/// Borrowed views of caller storage: the input and output types of the public
/// API. Nothing is copied when a view is made; the caller keeps the arrays
/// alive for the duration of the call that receives the view.

#include <cstdint>
#include <span>

namespace apxchol {

/// Square CSC matrix borrowed from the caller: n columns, col_ptr has n+1
/// entries, row_idx/values have col_ptr[n]. Rows sorted, no duplicates within
/// a column. Symmetric input: a CSR matrix is passed as the same arrays.
template<class Index>
struct basic_csc_view {
    std::int64_t n = 0;
    std::span<const Index> col_ptr;
    std::span<const Index> row_idx;
    std::span<const double> values;
};
using csc_view   = basic_csc_view<std::int32_t>;
using csc_view64 = basic_csc_view<std::int64_t>;

/// Column-major dense matrix borrowed from the caller; column j starts at
/// data + j * col_stride, col_stride >= rows.
struct matrix_view       { double* data = nullptr;       std::int64_t rows = 0, cols = 0, col_stride = 0; };
struct const_matrix_view { const double* data = nullptr; std::int64_t rows = 0, cols = 0, col_stride = 0; };

} // namespace apxchol
