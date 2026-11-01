#pragma once
// Eigen fixtures as the library's views and storage. The tests still hold
// Eigen containers; these adapters convert them at each call.
#include "apxchol/detail/csc_input.h"
#include "apxchol/operator_class.h"
#include "apxchol/solver/detail/solve_backend.h"
#include "apxchol/solver/detail/solver_route.h"
#include "apxchol/views.h"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace apxchol::test {

using Sparse = Eigen::SparseMatrix<double>;

/// A square, compressed Eigen matrix as a public view of its own arrays.
inline csc_view csc(const Sparse& A) {
    if (A.rows() != A.cols()) throw std::logic_error("test matrix must be square");
    if (!A.isCompressed()) throw std::logic_error("test matrix must be compressed");
    if (A.rows() == 0) return {};
    const auto n = static_cast<std::size_t>(A.rows());
    const auto nnz = static_cast<std::size_t>(A.nonZeros());
    return {A.rows(), {A.outerIndexPtr(), n + 1}, {A.innerIndexPtr(), nnz}, {A.valuePtr(), nnz}};
}

/// A contiguous Eigen vector as a span.
template<class Vector>
std::span<const double> cspan(const Vector& v) {
    static_assert((std::remove_cvref_t<Vector>::Flags & Eigen::DirectAccessBit) != 0);
    if (v.size() > 1 && v.innerStride() != 1) throw std::logic_error("test vector must be contiguous");
    return {v.data(), static_cast<std::size_t>(v.size())};
}
template<class Vector>
std::span<double> mspan(Vector&& v) {
    if (v.size() > 1 && v.innerStride() != 1) throw std::logic_error("test vector must be contiguous");
    return {v.data(), static_cast<std::size_t>(v.size())};
}

/// A column-major Eigen matrix (or Map/Ref/block with unit row stride) as a view.
template<class Matrix>
const_matrix_view cmat(const Matrix& M) {
    using M0 = std::remove_cvref_t<Matrix>;
    static_assert((M0::Flags & Eigen::DirectAccessBit) != 0);
    if (M0::IsRowMajor && M.cols() > 1) throw std::logic_error("test matrix must be column-major");
    if (M.rows() > 1 && M.innerStride() != 1) throw std::logic_error("test matrix rows must be contiguous");
    return {M.data(), M.rows(), M.cols(), M.cols() > 1 ? std::int64_t(M.outerStride()) : std::int64_t(M.rows())};
}
template<class Matrix>
matrix_view mmat(Matrix&& M) {
    using M0 = std::remove_cvref_t<Matrix>;
    static_assert((M0::Flags & Eigen::DirectAccessBit) != 0);
    if (M0::IsRowMajor && M.cols() > 1) throw std::logic_error("test matrix must be column-major");
    if (M.rows() > 1 && M.innerStride() != 1) throw std::logic_error("test matrix rows must be contiguous");
    return {M.data(), M.rows(), M.cols(), M.cols() > 1 ? std::int64_t(M.outerStride()) : std::int64_t(M.rows())};
}

/// A solution vector as an Eigen expression operand.
inline Eigen::Map<const Eigen::VectorXd> vec(const std::vector<double>& x) {
    return {x.data(), static_cast<Eigen::Index>(x.size())};
}

/// Square Eigen sparse matrix as a csc_ref. A compressed matrix is borrowed
/// in place; only index arrays narrower than node_index/edge_index are
/// widened into owned copies. An uncompressed matrix is compressed into
/// owned storage. Reports non-square input through rows()/cols() and an
/// empty ref.
class eigen_csc {
public:
    explicit eigen_csc(const Sparse& A) : rows_(A.rows()), cols_(A.cols()) {
        if (rows_ != cols_) return;
        using StorageIndex = Sparse::StorageIndex;
        const auto n = static_cast<node_index>(rows_);
        if (A.isCompressed()) {
            const auto nnz = static_cast<std::size_t>(A.nonZeros());
            const edge_index* ptr = nullptr;
            const node_index* idx = nullptr;
            if constexpr (sizeof(StorageIndex) == sizeof(edge_index)) {
                ptr = reinterpret_cast<const edge_index*>(A.outerIndexPtr());
            } else {
                offsets_.assign(A.outerIndexPtr(), A.outerIndexPtr() + rows_ + 1);
                ptr = offsets_.data();
            }
            if constexpr (sizeof(StorageIndex) == sizeof(node_index)) {
                idx = reinterpret_cast<const node_index*>(A.innerIndexPtr());
            } else {
                rows_copy_.assign(A.innerIndexPtr(), A.innerIndexPtr() + nnz);
                idx = rows_copy_.data();
            }
            ref_ = {n, ptr, idx, A.valuePtr()};
            return;
        }
        Sparse compressed = A;
        compressed.makeCompressed();
        const auto nnz = static_cast<edge_index>(compressed.nonZeros());
        owned_.resize(n, nnz);
        for (Eigen::Index j = 0; j <= rows_; ++j)
            owned_.ptr()[j] = static_cast<edge_index>(compressed.outerIndexPtr()[j]);
        for (edge_index p = 0; p < nnz; ++p) {
            owned_.idx()[p] = static_cast<node_index>(compressed.innerIndexPtr()[p]);
            owned_.val()[p] = compressed.valuePtr()[p];
        }
        ref_ = owned_.ref();
    }
    eigen_csc(const eigen_csc&) = delete;
    eigen_csc& operator=(const eigen_csc&) = delete;

    Eigen::Index rows() const noexcept { return rows_; }
    Eigen::Index cols() const noexcept { return cols_; }
    bool square() const noexcept { return rows_ == cols_; }
    const detail::csc_ref<double>& ref() const noexcept { return ref_; }
    operator detail::csc_ref<double>() const noexcept { return ref_; }

private:
    Eigen::Index rows_ = 0, cols_ = 0;
    std::vector<edge_index> offsets_;
    std::vector<node_index> rows_copy_;
    detail::csc_matrix<double> owned_;
    detail::csc_ref<double> ref_;
};

/// Owned copy of a square Eigen matrix (compressed; explicit entries kept).
inline detail::csc_matrix<double> to_csc(const Sparse& A) {
    const eigen_csc input(A);
    return detail::csc_cast<double>(input.ref());
}

/// Owned copy of the library's CSC as an Eigen matrix (int indices).
inline Sparse to_eigen(detail::csc_ref<double> A) {
    Sparse out(static_cast<int>(A.n), static_cast<int>(A.n));
    out.resizeNonZeros(static_cast<int>(A.nnz()));
    for (std::size_t j = 0; j <= A.n; ++j) out.outerIndexPtr()[j] = static_cast<int>(A.ptr[j]);
    for (edge_index p = 0; p < A.nnz(); ++p) {
        out.innerIndexPtr()[p] = static_cast<int>(A.idx[p]);
        out.valuePtr()[p] = A.val[p];
    }
    return out;
}

/// An Eigen dense object with direct access as a strided internal view;
/// writable when the object is.
template<class Matrix>
auto dense_of(Matrix&& M) {
    auto& m = M.derived();
    using Derived = std::remove_cvref_t<decltype(m)>;
    static_assert((Derived::Flags & Eigen::DirectAccessBit) != 0);
    using T = std::remove_pointer_t<decltype(m.data())>;
    const std::int64_t row_step = Derived::IsRowMajor ? m.outerStride() : m.innerStride();
    const std::int64_t col_step = Derived::IsRowMajor ? m.innerStride() : m.outerStride();
    return detail::dense_ref<T>(m.data(), m.rows(), m.cols(), row_step, col_step);
}

/// M-matrix lumping applied to an Eigen matrix in place; returns the entries
/// removed. Leaves A unchanged when nothing is positive off the diagonal.
inline Eigen::Index lump_in_place(Sparse& A) {
    if (A.rows() != A.cols() || A.rows() == 0) return 0;
    A.makeCompressed();
    std::int64_t removed = 0;
    const eigen_csc input(A);
    const auto lumped = detail::lump_positive_offdiagonals(input.ref(), removed);
    if (removed != 0) A = to_eigen(lumped.ref());
    return removed;
}

/// Operator checks on Eigen matrices; a matrix that is not square reports
/// the library's not-square error.
inline detail::operator_scan scan_operator(const Sparse& A) {
    const eigen_csc input(A);
    if (!input.square()) {
        detail::operator_scan s;
        s.rows = A.rows();
        s.cols = A.cols();
        return s;
    }
    return detail::scan_operator(input.ref());
}
inline detail::operator_scan require_operator(const Sparse& A,
                                              double lump_max_mass = kDefaultLumpMassCeiling) {
    detail::validate_lump_max_mass(lump_max_mass);
    const eigen_csc input(A);
    if (!input.square()) detail::throw_operator_not_square(A.rows(), A.cols());
    return detail::require_operator(input.ref(), lump_max_mass);
}
inline bool is_graph_adjacency(const Sparse& M) {
    const eigen_csc input(M);
    if (!input.square()) detail::throw_adjacency_not_square(M.rows(), M.cols());
    return detail::is_graph_adjacency(input.ref());
}
inline Sparse adjacency_laplacian(const Sparse& M) {
    const eigen_csc input(M);
    return to_eigen(detail::adjacency_laplacian(input.ref()).ref());
}

/// Route selection for an Eigen matrix.
inline solve_backend select_solve_backend(const solve_options& opts, const Sparse& A) {
    const eigen_csc input(A);
    return detail::select_solve_backend(opts, &input.ref());
}

/// A solver with a timing tree attached (detail::solver_access::make).
inline solver with_timings(const Sparse& A, const solve_options& opts, detail::checkpoint& timings) {
    const eigen_csc input(A);
    return detail::solver_access::make(input, opts, &timings);
}

} // namespace apxchol::test
