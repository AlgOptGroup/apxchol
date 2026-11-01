#pragma once
/// The one place where Eigen types meet the library's own storage. The public
/// API still takes and returns Eigen types; it converts here, once, and
/// everything behind it works on csc_ref/csc_matrix and dense_ref. The CLI and
/// the Python and Octave bindings also convert here. Not part of the public API.

#include "apxchol/detail/csc_matrix.h"
#include "apxchol/detail/dense.h"
#include "apxchol/operator_class.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace apxchol::detail {

/// Square Eigen sparse matrix as a csc_ref. A compressed matrix is borrowed
/// in place; only index arrays narrower than node_index/edge_index are
/// widened into owned copies (none in the default 32-bit build). An
/// uncompressed matrix is compressed into owned storage. Reports non-square
/// input through rows()/cols() and an empty ref.
class eigen_csc {
public:
    explicit eigen_csc(const Eigen::SparseMatrix<double>& A)
        : rows_(A.rows()), cols_(A.cols()) {
        if (rows_ != cols_) return;
        using StorageIndex = Eigen::SparseMatrix<double>::StorageIndex;
        if (A.isCompressed()) {
            // StorageIndex is a signed int and node_index/edge_index, when
            // equally wide, its unsigned counterpart, which may alias it.
            // Valid indices are nonnegative and keep their values.
            const auto n = static_cast<std::size_t>(rows_);
            const auto nnz = static_cast<std::int64_t>(A.nonZeros());
            const edge_index* ptr = nullptr;
            const node_index* idx = nullptr;
            if constexpr (sizeof(StorageIndex) == sizeof(edge_index)) {
                ptr = reinterpret_cast<const edge_index*>(A.outerIndexPtr());
            } else {
                offsets_.resize(n + 1);
                std::copy_n(A.outerIndexPtr(), n + 1, offsets_.data());
                ptr = offsets_.data();
            }
            if constexpr (sizeof(StorageIndex) == sizeof(node_index)) {
                idx = reinterpret_cast<const node_index*>(A.innerIndexPtr());
            } else {
                rows_copy_.resize(static_cast<std::size_t>(nnz));
                const StorageIndex* inner = A.innerIndexPtr();
                node_index* out = rows_copy_.data();
                #pragma omp parallel for schedule(static)
                for (std::int64_t p = 0; p < nnz; ++p) out[p] = static_cast<node_index>(inner[p]);
                idx = rows_copy_.data();
            }
            ref_ = {static_cast<node_index>(rows_), ptr, idx, A.valuePtr()};
            return;
        }
        const auto n = static_cast<node_index>(rows_);
        const StorageIndex* outer = A.outerIndexPtr();
        const StorageIndex* inner = A.innerIndexPtr();
        const StorageIndex* counts = A.innerNonZeroPtr();  // uncompressed: never null
        const auto range = [&](Eigen::Index col) {
            const Eigen::Index first = outer[col];
            const Eigen::Index last = first + counts[col];
            if (first < 0 || last < first || last > A.data().size())
                throw std::invalid_argument("apxchol: invalid Eigen sparse storage");
            return std::pair{first, last};
        };
        Eigen::Index stored = 0;
        for (Eigen::Index col = 0; col < rows_; ++col) {
            const auto [first, last] = range(col);
            stored += last - first;
        }
        owned_.resize(n, static_cast<edge_index>(stored));
        edge_index at = 0;
        owned_.ptr()[0] = 0;
        for (Eigen::Index col = 0; col < rows_; ++col) {
            const auto [first, last] = range(col);
            for (Eigen::Index p = first; p < last; ++p, ++at) {
                owned_.idx()[at] = static_cast<node_index>(inner[p]);
                owned_.val()[at] = A.valuePtr()[p];
            }
            owned_.ptr()[col + 1] = at;
        }
        ref_ = owned_.ref();
    }
    eigen_csc(const eigen_csc&) = delete;
    eigen_csc& operator=(const eigen_csc&) = delete;

    Eigen::Index rows() const noexcept { return rows_; }
    Eigen::Index cols() const noexcept { return cols_; }
    bool square() const noexcept { return rows_ == cols_; }
    const csc_ref<double>& ref() const noexcept { return ref_; }
    operator csc_ref<double>() const noexcept { return ref_; }

private:
    Eigen::Index rows_ = 0, cols_ = 0;
    buffer<edge_index> offsets_;
    buffer<node_index> rows_copy_;
    csc_matrix<double> owned_;
    csc_ref<double> ref_;
};

/// Owned copy of the library's CSC as an Eigen matrix (int indices).
inline Eigen::SparseMatrix<double> to_eigen(csc_ref<double> A) {
    constexpr auto limit = static_cast<std::uint64_t>(std::numeric_limits<int>::max());
    if (static_cast<std::uint64_t>(A.n) > limit || static_cast<std::uint64_t>(A.nnz()) > limit)
        throw std::length_error("apxchol: matrix exceeds Eigen's 32-bit sparse indices");
    Eigen::SparseMatrix<double> out(static_cast<int>(A.n), static_cast<int>(A.n));
    out.resizeNonZeros(static_cast<int>(A.nnz()));
    for (std::size_t j = 0; j <= A.n; ++j) out.outerIndexPtr()[j] = static_cast<int>(A.ptr[j]);
    for (edge_index p = 0; p < A.nnz(); ++p) {
        out.innerIndexPtr()[p] = static_cast<int>(A.idx[p]);
        out.valuePtr()[p] = A.val[p];
    }
    return out;
}

/// An Eigen dense object with direct access as a strided view; writable when
/// the object is.
template<class Matrix>
auto dense_of(Matrix&& M) {
    auto& m = M.derived();
    using Derived = std::remove_cvref_t<decltype(m)>;
    static_assert((Derived::Flags & Eigen::DirectAccessBit) != 0);
    using T = std::remove_pointer_t<decltype(m.data())>;
    const std::int64_t row_step = Derived::IsRowMajor ? m.outerStride() : m.innerStride();
    const std::int64_t col_step = Derived::IsRowMajor ? m.innerStride() : m.outerStride();
    return dense_ref<T>(m.data(), m.rows(), m.cols(), row_step, col_step);
}

// Operator checks on Eigen input, for the CLI, the bindings and the benchmark.
inline operator_scan scan_operator(const Eigen::SparseMatrix<double>& A) {
    const eigen_csc input(A);
    if (!input.square()) {
        operator_scan s;
        s.rows = A.rows();
        s.cols = A.cols();
        return s;
    }
    return scan_operator(input.ref());
}

inline operator_scan require_operator(const Eigen::SparseMatrix<double>& A,
                                      double lump_max_mass = kDefaultLumpMassCeiling) {
    validate_lump_max_mass(lump_max_mass);
    const eigen_csc input(A);
    if (!input.square()) throw_operator_not_square(A.rows(), A.cols());
    return require_operator(input.ref(), lump_max_mass);
}

inline bool is_graph_adjacency(const Eigen::SparseMatrix<double>& M) {
    const eigen_csc input(M);
    if (!input.square()) throw_adjacency_not_square(M.rows(), M.cols());
    return is_graph_adjacency(input.ref());
}

inline Eigen::SparseMatrix<double> adjacency_laplacian(const Eigen::SparseMatrix<double>& M) {
    if (M.rows() != M.cols()) throw std::invalid_argument("adjacency matrix must be square");
    const eigen_csc input(M);
    return to_eigen(adjacency_laplacian(input.ref()).ref());
}

} // namespace apxchol::detail
