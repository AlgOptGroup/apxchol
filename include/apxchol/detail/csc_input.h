#pragma once
/// The public views (apxchol/views.h) as the library's own storage types. The
/// public API converts here, once; everything behind it works on csc_ref and
/// dense_ref. Not part of the public API.

#include "apxchol/views.h"
#include "apxchol/detail/buffer.h"
#include "apxchol/detail/csc_matrix.h"
#include "apxchol/detail/dense.h"

namespace apxchol::detail {

/// A caller's CSC view, checked and presented as a csc_ref. Index arrays as
/// wide as node_index/edge_index are borrowed; the others are converted once
/// into owned copies. Values are always borrowed. Throws
/// std::invalid_argument ("apxchol: ...") when col_ptr does not have n+1
/// nondecreasing entries starting at 0, row_idx/values do not have col_ptr[n]
/// entries, or a column holds a row outside [0, n), out of order or twice.
class csc_input {
public:
    explicit csc_input(csc_view A);
    explicit csc_input(csc_view64 A);
    csc_input(const csc_input&) = delete;
    csc_input& operator=(const csc_input&) = delete;

    const csc_ref<double>& ref() const noexcept { return ref_; }
    operator csc_ref<double>() const noexcept { return ref_; }

private:
    template<class Index> void init(basic_csc_view<Index> A);

    buffer<edge_index> offsets_;
    buffer<node_index> rows_;
    csc_ref<double> ref_;
};

/// A caller's dense view as a strided ref. Throws std::invalid_argument for
/// negative dimensions, col_stride < rows, or a null pointer with elements.
const_matrix_ref dense_input(const_matrix_view M);
matrix_ref dense_input(matrix_view M);

} // namespace apxchol::detail
