#pragma once
/// CSC arrays from the Python and MATLAB/Octave bindings as a public CSC view.
/// Not part of the public API.

#include "apxchol/detail/csc_matrix.h"
#include "apxchol/views.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace apxchol::detail {

/// Foreign n x n CSC arrays (outer has n+1 entries) as a view the solver
/// accepts. Sorted, duplicate-free columns are borrowed; otherwise the entries
/// are assembled into owned storage with csc_from_triplets semantics
/// (duplicates summed in order). Operator validation is separate.
template<class Index>
class imported_csc {
public:
    imported_csc(std::int64_t n, const Index* outer, const Index* inner, const double* values) {
        if (n < 0) throw std::invalid_argument("n must be nonnegative");
        if (outer == nullptr) throw std::invalid_argument("colptr is NULL");
        if (outer[0] != 0) throw std::invalid_argument("colptr[0] must be 0");
        for (std::int64_t j = 0; j < n; ++j)
            if (outer[j + 1] < outer[j])
                throw std::invalid_argument("colptr must be nondecreasing");
        const auto nnz = static_cast<std::int64_t>(outer[n]);
        if (nnz != 0 && (inner == nullptr || values == nullptr))
            throw std::invalid_argument("rowval or nzval is NULL");

        bool sorted = true;
        for (std::int64_t j = 0; j < n; ++j)
            for (std::int64_t p = outer[j]; p < outer[j + 1]; ++p) {
                if (inner[p] < 0 || inner[p] >= n)
                    throw std::invalid_argument("CSC row index out of range");
                if (p > outer[j] && inner[p] <= inner[p - 1]) sorted = false;
            }
        const auto size = static_cast<std::size_t>(n);
        if (sorted) {
            view_ = {n, {outer, size + 1}, {inner, static_cast<std::size_t>(nnz)},
                     {values, static_cast<std::size_t>(nnz)}};
            return;
        }

        if (static_cast<std::uint64_t>(n) >= std::numeric_limits<node_index>::max())
            throw std::invalid_argument("n exceeds this build's vertex index width");
        std::vector<triplet> triplets;
        triplets.reserve(static_cast<std::size_t>(nnz));
        for (std::int64_t j = 0; j < n; ++j)
            for (std::int64_t p = outer[j]; p < outer[j + 1]; ++p)
                triplets.push_back({static_cast<node_index>(inner[p]),
                                    static_cast<node_index>(j), values[p]});
        const auto assembled = csc_from_triplets(static_cast<node_index>(n), triplets);
        std::vector<triplet>().swap(triplets);
        const auto stored = static_cast<std::size_t>(assembled.ptr()[n]);
        outer_.assign(assembled.ptr(), assembled.ptr() + size + 1);
        inner_.assign(assembled.idx(), assembled.idx() + stored);
        values_.assign(assembled.val(), assembled.val() + stored);
        view_ = {n, outer_, inner_, values_};
    }
    imported_csc(const imported_csc&) = delete;
    imported_csc& operator=(const imported_csc&) = delete;

    const basic_csc_view<Index>& view() const noexcept { return view_; }

private:
    std::vector<Index> outer_, inner_;
    std::vector<double> values_;
    basic_csc_view<Index> view_;
};

} // namespace apxchol::detail
