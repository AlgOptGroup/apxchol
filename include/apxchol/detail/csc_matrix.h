#pragma once
/// Square compressed sparse storage on node_index/edge_index, borrowed
/// (csc_ref) or owned (csc_matrix), and the sparse routines the library needs:
/// transpose, symmetric permutation, triplet assembly. A symmetric operator
/// has the same arrays in CSC and CSR, so the same types hold the row-major
/// PCG operator. Not part of the public API.

#include "apxchol/detail/buffer.h"
#include "apxchol/types.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace apxchol::detail {

/// Borrowed n x n compressed matrix: column k holds idx/val[ptr[k], ptr[k+1]),
/// ptr[0] == 0. Producers state whether columns are sorted or unique.
template<class T>
struct csc_ref {
    node_index n = 0;
    const edge_index* ptr = nullptr;
    const node_index* idx = nullptr;
    const T* val = nullptr;

    node_index rows() const noexcept { return n; }
    node_index cols() const noexcept { return n; }
    /// Stored entries.
    edge_index nnz() const noexcept { return ptr ? ptr[n] : edge_index(0); }
};

/// Owned n x n compressed matrix. resize() leaves the arrays uninitialized;
/// the producer writes every element.
template<class T>
class csc_matrix {
public:
    csc_matrix() = default;
    csc_matrix(node_index n, edge_index nnz) { resize(n, nnz); }

    void resize(node_index n, edge_index nnz) {
        n_ = n;
        ptr_.resize(static_cast<std::size_t>(n) + 1);
        idx_.resize(static_cast<std::size_t>(nnz));
        val_.resize(static_cast<std::size_t>(nnz));
    }
    /// Reallocate the index and value arrays, keeping the offsets.
    void resize_nonzeros(edge_index nnz) {
        idx_.resize(static_cast<std::size_t>(nnz));
        val_.resize(static_cast<std::size_t>(nnz));
    }
    void clear() noexcept {
        n_ = 0;
        ptr_.clear();
        idx_.clear();
        val_.clear();
    }

    node_index rows() const noexcept { return n_; }
    node_index cols() const noexcept { return n_; }
    /// Allocated entries; equals ptr()[rows()] once the producer has filled it.
    edge_index nnz() const noexcept { return static_cast<edge_index>(idx_.size()); }
    bool empty() const noexcept { return ptr_.empty(); }

    edge_index* ptr() noexcept { return ptr_.data(); }
    node_index* idx() noexcept { return idx_.data(); }
    T* val() noexcept { return val_.data(); }
    const edge_index* ptr() const noexcept { return ptr_.data(); }
    const node_index* idx() const noexcept { return idx_.data(); }
    const T* val() const noexcept { return val_.data(); }

    csc_ref<T> ref() const noexcept { return {n_, ptr_.data(), idx_.data(), val_.data()}; }
    operator csc_ref<T>() const noexcept { return ref(); }

private:
    node_index n_ = 0;
    buffer<edge_index> ptr_;
    buffer<node_index> idx_;
    buffer<T> val_;
};

/// Element-wise converting copy (e.g. fp32 storage widened to fp64).
template<class To, class From>
csc_matrix<To> csc_cast(csc_ref<From> A) {
    csc_matrix<To> out(A.n, A.nnz());
    std::copy_n(A.ptr, static_cast<std::size_t>(A.n) + 1, out.ptr());
    std::copy_n(A.idx, A.nnz(), out.idx());
    std::transform(A.val, A.val + A.nnz(), out.val(), [](From v) { return static_cast<To>(v); });
    return out;
}

/// Transpose by a stable counting scatter: column j of the result lists the
/// entries of row j in the order of their source columns, so the result is
/// sorted whenever every source column holds each row at most once, and
/// duplicates keep their source order. Values are converted to To.
template<class To = void, class From>
auto csc_transpose(csc_ref<From> A) {
    using Out = std::conditional_t<std::is_void_v<To>, From, To>;
    const std::size_t n = A.n;
    const edge_index nnz = A.nnz();
    csc_matrix<Out> out(A.n, nnz);
    edge_index* ptr = out.ptr();
    std::fill_n(ptr, n + 1, edge_index(0));
    for (edge_index p = 0; p < nnz; ++p) ++ptr[A.idx[p] + 1];
    for (std::size_t j = 0; j < n; ++j) ptr[j + 1] += ptr[j];
    std::vector<edge_index> next(ptr, ptr + n);
    for (std::size_t col = 0; col < n; ++col)
        for (edge_index p = A.ptr[col]; p < A.ptr[col + 1]; ++p) {
            const edge_index slot = next[A.idx[p]]++;
            out.idx()[slot] = static_cast<node_index>(col);
            out.val()[slot] = static_cast<Out>(A.val[p]);
        }
    return out;
}

/// B = P^T A P for the permutation perm (perm[i] = new position of i), i.e.
/// B(i, j) = A(perm[i], perm[j]): the inverse of the symmetric permutation
/// that placed entry (i, j) at (perm[i], perm[j]). Columns of the result are
/// sorted; duplicate coordinates keep their source order.
template<class T>
csc_matrix<T> csc_unpermute_symmetric(csc_ref<T> A, std::span<const node_index> perm) {
    const std::size_t n = A.n;
    if (perm.size() != n) throw std::invalid_argument("permutation size mismatch");
    std::vector<node_index> inverse(n);
    for (std::size_t i = 0; i < n; ++i) inverse[perm[i]] = static_cast<node_index>(i);
    // Relabel rows and columns, then two stable counting transposes sort each
    // column by row while keeping duplicate runs in source order.
    csc_matrix<T> relabeled(A.n, A.nnz());
    relabeled.ptr()[0] = 0;
    edge_index at = 0;
    for (std::size_t j = 0; j < n; ++j) {
        const std::size_t source = perm[j];
        for (edge_index p = A.ptr[source]; p < A.ptr[source + 1]; ++p, ++at) {
            relabeled.idx()[at] = inverse[A.idx[p]];
            relabeled.val()[at] = A.val[p];
        }
        relabeled.ptr()[j + 1] = at;
    }
    const auto rows_major = csc_transpose(relabeled.ref());
    return csc_transpose(rows_major.ref());
}

/// One (row, col, value) entry for csc_from_triplets.
struct triplet {
    node_index row;
    node_index col;
    double value;
};

/// Assemble an n x n CSC matrix: sorted, unique columns; entries with equal coordinates are summed in their
/// order in `entries`, ((v1 + v2) + v3); sums that cancel to zero stay stored.
/// Throws std::out_of_range for coordinates outside [0, n).
inline csc_matrix<double> csc_from_triplets(node_index n, std::span<const triplet> entries) {
    const std::size_t count = entries.size(), size = n;
    if (count > static_cast<std::size_t>(std::numeric_limits<edge_index>::max()))
        throw std::length_error("apxchol: too many matrix entries for edge_index");
    for (const auto& t : entries)
        if (t.row >= n || t.col >= n) throw std::out_of_range("triplet coordinate out of range");
    // Stable counting sort by row, then by column: within a column, rows
    // ascend and equal coordinates keep their input order.
    std::vector<edge_index> by_row_ptr(size + 1, 0);
    for (const auto& t : entries) ++by_row_ptr[t.row + 1];
    for (std::size_t i = 0; i < size; ++i) by_row_ptr[i + 1] += by_row_ptr[i];
    std::vector<edge_index> order(count);
    {
        std::vector<edge_index> next(by_row_ptr.begin(), by_row_ptr.end() - 1);
        for (std::size_t k = 0; k < count; ++k) order[next[entries[k].row]++] = static_cast<edge_index>(k);
    }
    std::vector<edge_index> col_ptr(size + 1, 0);
    for (const auto& t : entries) ++col_ptr[t.col + 1];
    for (std::size_t j = 0; j < size; ++j) col_ptr[j + 1] += col_ptr[j];
    std::vector<edge_index> sorted(count);
    {
        std::vector<edge_index> next(col_ptr.begin(), col_ptr.end() - 1);
        for (const edge_index k : order) sorted[next[entries[k].col]++] = k;
    }
    std::vector<edge_index>().swap(order);
    // Sum duplicates within each column, in input order.
    std::vector<edge_index> unique_ptr(size + 1, 0);
    for (std::size_t j = 0; j < size; ++j) {
        edge_index kept = 0;
        for (edge_index p = col_ptr[j]; p < col_ptr[j + 1]; ++p)
            if (p == col_ptr[j] || entries[sorted[p]].row != entries[sorted[p - 1]].row) ++kept;
        unique_ptr[j + 1] = unique_ptr[j] + kept;
    }
    csc_matrix<double> out(n, unique_ptr[size]);
    std::copy(unique_ptr.begin(), unique_ptr.end(), out.ptr());
    for (std::size_t j = 0; j < size; ++j) {
        edge_index q = unique_ptr[j];
        for (edge_index p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
            const triplet& t = entries[sorted[p]];
            if (p != col_ptr[j] && t.row == entries[sorted[p - 1]].row) {
                out.val()[q - 1] += t.value;
            } else {
                out.idx()[q] = t.row;
                out.val()[q] = t.value;
                ++q;
            }
        }
    }
    return out;
}

} // namespace apxchol::detail
