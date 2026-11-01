#pragma once
// Small matrices and vectors for the tests: an owned CSC matrix with int32
// indices (the public csc_view's type), a column-major dense matrix, a few
// vector reductions and a dense Cholesky for reference checks. They favour
// plain loops over speed; the library's own kernels are what the tests check.
#include "apxchol/detail/csc_input.h"
#include "apxchol/operator_class.h"
#include "apxchol/solver/detail/solve_backend.h"
#include "apxchol/solver/detail/solver_route.h"
#include "apxchol/views.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace apxchol::test {

struct triplet {
    std::int64_t row = 0, col = 0;
    double value = 0;
};

/// Owned rows x cols CSC matrix. Column j holds row_idx/values in
/// [col_ptr[j], col_ptr[j + 1]). from_triplets() produces sorted, unique
/// columns; tests that need other layouts fill the arrays directly.
struct sparse_matrix {
    std::int64_t rows = 0, cols = 0;
    std::vector<std::int32_t> col_ptr{0};
    std::vector<std::int32_t> row_idx;
    std::vector<double> values;

    sparse_matrix() = default;
    /// An empty rows x cols matrix.
    sparse_matrix(std::int64_t r, std::int64_t c)
        : rows(r), cols(c), col_ptr(static_cast<std::size_t>(c) + 1, 0) {}

    std::int64_t nnz() const noexcept { return static_cast<std::int64_t>(row_idx.size()); }
    /// Sum of the stored entries at (i, j); 0 when there are none.
    double coeff(std::int64_t i, std::int64_t j) const {
        double sum = 0;
        for (auto p = col_ptr[j]; p < col_ptr[j + 1]; ++p)
            if (row_idx[p] == i) sum += values[p];
        return sum;
    }
};

/// Assembles rows x cols from triplets: columns sorted by row, duplicates summed
/// in input order, explicit zeros kept.
inline sparse_matrix from_triplets(std::int64_t rows, std::int64_t cols,
                                   std::span<const triplet> entries) {
    std::vector<std::size_t> order(entries.size());
    std::iota(order.begin(), order.end(), std::size_t(0));
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        const auto& x = entries[a];
        const auto& y = entries[b];
        return x.col != y.col ? x.col < y.col : x.row < y.row;
    });
    sparse_matrix A(rows, cols);
    for (std::size_t k = 0; k < order.size(); ++k) {
        const auto& e = entries[order[k]];
        if (e.row < 0 || e.row >= rows || e.col < 0 || e.col >= cols)
            throw std::out_of_range("test triplet outside the matrix");
        const bool repeat = k > 0 && entries[order[k - 1]].row == e.row &&
                            entries[order[k - 1]].col == e.col;
        if (repeat) {
            A.values.back() += e.value;
        } else {
            A.row_idx.push_back(static_cast<std::int32_t>(e.row));
            A.values.push_back(e.value);
            ++A.col_ptr[e.col + 1];
        }
    }
    for (std::int64_t j = 0; j < cols; ++j) A.col_ptr[j + 1] += A.col_ptr[j];
    return A;
}

/// The stored entry (i, j), which must exist.
inline double& entry(sparse_matrix& A, std::int64_t i, std::int64_t j) {
    for (auto p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p)
        if (A.row_idx[p] == i) return A.values[p];
    throw std::logic_error("test matrix has no stored entry there");
}

/// Sets entry (i, j) of a matrix with sorted, unique columns, inserting it
/// (even as an explicit zero) when it is not stored.
inline void set_entry(sparse_matrix& A, std::int64_t i, std::int64_t j, double value) {
    auto p = A.col_ptr[j];
    while (p < A.col_ptr[j + 1] && A.row_idx[p] < i) ++p;
    if (p < A.col_ptr[j + 1] && A.row_idx[p] == i) {
        A.values[p] = value;
        return;
    }
    A.row_idx.insert(A.row_idx.begin() + p, static_cast<std::int32_t>(i));
    A.values.insert(A.values.begin() + p, value);
    for (std::int64_t k = j + 1; k <= A.cols; ++k) ++A.col_ptr[k];
}
/// A copy of A with entry (i, j) set as by set_entry.
inline sparse_matrix with_entry(sparse_matrix A, std::int64_t i, std::int64_t j, double value) {
    set_entry(A, i, j, value);
    return A;
}

/// The stored entries in column order.
inline std::vector<triplet> triplets(const sparse_matrix& A) {
    std::vector<triplet> out;
    out.reserve(A.row_idx.size());
    for (std::int64_t j = 0; j < A.cols; ++j)
        for (auto p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p)
            out.push_back({A.row_idx[p], j, A.values[p]});
    return out;
}

inline sparse_matrix identity(std::int64_t n) {
    std::vector<triplet> d;
    for (std::int64_t i = 0; i < n; ++i) d.push_back({i, i, 1.0});
    return from_triplets(n, n, d);
}

inline sparse_matrix transpose(const sparse_matrix& A) {
    auto t = triplets(A);
    for (auto& e : t) std::swap(e.row, e.col);
    return from_triplets(A.cols, A.rows, t);
}

/// alpha * A + beta * B; entries present in either operand are kept.
inline sparse_matrix combine(double alpha, const sparse_matrix& A, double beta, const sparse_matrix& B) {
    if (A.rows != B.rows || A.cols != B.cols) throw std::logic_error("test matrix sizes differ");
    auto t = triplets(A);
    for (auto& e : t) e.value *= alpha;
    for (auto e : triplets(B)) {
        e.value *= beta;
        t.push_back(e);
    }
    return from_triplets(A.rows, A.cols, t);
}
inline sparse_matrix scaled(const sparse_matrix& A, double s) {
    sparse_matrix out = A;
    for (double& v : out.values) v *= s;
    return out;
}

/// A * x.
inline std::vector<double> multiply(const sparse_matrix& A, std::span<const double> x) {
    if (static_cast<std::int64_t>(x.size()) != A.cols) throw std::logic_error("test vector size");
    std::vector<double> y(static_cast<std::size_t>(A.rows), 0.0);
    for (std::int64_t j = 0; j < A.cols; ++j)
        for (auto p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p) y[A.row_idx[p]] += A.values[p] * x[j];
    return y;
}

// ── vectors ─────────────────────────────────────────────────────────────────

inline double dot(std::span<const double> a, std::span<const double> b) {
    if (a.size() != b.size()) throw std::logic_error("test vector sizes differ");
    double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}
inline double norm(std::span<const double> a) { return std::sqrt(dot(a, a)); }
inline double max_abs(std::span<const double> a) {
    double m = 0;
    for (double v : a) m = std::max(m, std::abs(v));
    return m;
}
inline double mean(std::span<const double> a) {
    return a.empty() ? 0.0 : std::accumulate(a.begin(), a.end(), 0.0) / static_cast<double>(a.size());
}
inline bool all_finite(std::span<const double> a) {
    return std::all_of(a.begin(), a.end(), [](double v) { return std::isfinite(v); });
}
/// a - b.
inline std::vector<double> difference(std::span<const double> a, std::span<const double> b) {
    if (a.size() != b.size()) throw std::logic_error("test vector sizes differ");
    std::vector<double> d(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) d[i] = a[i] - b[i];
    return d;
}
/// ||A x - b|| / ||b||.
inline double relative_residual(const sparse_matrix& A, std::span<const double> x,
                                std::span<const double> b) {
    return norm(difference(multiply(A, x), b)) / norm(b);
}
/// The vector with its mean removed.
inline std::vector<double> centered(std::span<const double> a) {
    const double m = mean(a);
    std::vector<double> c(a.begin(), a.end());
    for (double& v : c) v -= m;
    return c;
}

/// Euclidean norm without overflow or underflow in the squares.
inline double stable_norm(std::span<const double> a) {
    const double scale = max_abs(a);
    if (scale == 0.0 || !std::isfinite(scale)) return scale;
    double sum = 0;
    for (double v : a) sum += (v / scale) * (v / scale);
    return scale * std::sqrt(sum);
}

/// n values evenly spaced from first to last (last alone when n == 1).
inline std::vector<double> linspaced(std::int64_t n, double first, double last) {
    std::vector<double> v(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i)
        v[i] = n == 1 ? last : first + (last - first) * static_cast<double>(i) / static_cast<double>(n - 1);
    return v;
}

/// n values uniform in [-1, 1) from a fixed seed.
inline std::vector<double> random_vector(std::int64_t n, std::uint64_t seed = 1) {
    std::mt19937_64 rng(seed);
    std::vector<double> v(static_cast<std::size_t>(n));
    for (double& x : v) x = static_cast<double>(rng() >> 11) * 0x1p-52 - 1.0;
    return v;
}

/// Zero-mean, unit-norm right-hand side from a fixed seed.
inline std::vector<double> random_rhs(std::int64_t n, std::uint64_t seed = 1) {
    auto b = centered(random_vector(n, seed));
    const double s = norm(b);
    for (double& v : b) v /= s;
    return b;
}

// ── dense ───────────────────────────────────────────────────────────────────

/// Owned column-major rows x cols matrix, zero-initialized.
struct dense_matrix {
    std::int64_t rows = 0, cols = 0;
    std::vector<double> data;

    dense_matrix() = default;
    dense_matrix(std::int64_t r, std::int64_t c, double fill = 0.0)
        : rows(r), cols(c), data(static_cast<std::size_t>(r * c), fill) {}

    double& operator()(std::int64_t i, std::int64_t j) { return data[i + j * rows]; }
    double operator()(std::int64_t i, std::int64_t j) const { return data[i + j * rows]; }
    std::span<double> col(std::int64_t j) {
        return {data.data() + j * rows, static_cast<std::size_t>(rows)};
    }
    std::span<const double> col(std::int64_t j) const {
        return {data.data() + j * rows, static_cast<std::size_t>(rows)};
    }
    matrix_view view() { return {data.data(), rows, cols, rows}; }
    const_matrix_view view() const { return {data.data(), rows, cols, rows}; }
};

inline dense_matrix to_dense(const sparse_matrix& A) {
    dense_matrix D(A.rows, A.cols);
    for (std::int64_t j = 0; j < A.cols; ++j)
        for (auto p = A.col_ptr[j]; p < A.col_ptr[j + 1]; ++p) D(A.row_idx[p], j) += A.values[p];
    return D;
}
/// The nonzero entries of D.
inline sparse_matrix to_sparse(const dense_matrix& D) {
    std::vector<triplet> t;
    for (std::int64_t j = 0; j < D.cols; ++j)
        for (std::int64_t i = 0; i < D.rows; ++i)
            if (D(i, j) != 0.0) t.push_back({i, j, D(i, j)});
    return from_triplets(D.rows, D.cols, t);
}
inline dense_matrix multiply(const dense_matrix& A, const dense_matrix& B) {
    if (A.cols != B.rows) throw std::logic_error("test matrix sizes differ");
    dense_matrix C(A.rows, B.cols);
    for (std::int64_t j = 0; j < B.cols; ++j)
        for (std::int64_t k = 0; k < A.cols; ++k)
            for (std::int64_t i = 0; i < A.rows; ++i) C(i, j) += A(i, k) * B(k, j);
    return C;
}
inline dense_matrix transpose(const dense_matrix& A) {
    dense_matrix T(A.cols, A.rows);
    for (std::int64_t j = 0; j < A.cols; ++j)
        for (std::int64_t i = 0; i < A.rows; ++i) T(j, i) = A(i, j);
    return T;
}
/// Largest |A(i, j) - B(i, j)|.
inline double max_abs_difference(const dense_matrix& A, const dense_matrix& B) {
    if (A.rows != B.rows || A.cols != B.cols) throw std::logic_error("test matrix sizes differ");
    return max_abs(difference(A.data, B.data));
}

/// Lower Cholesky factor of a symmetric matrix; throws std::domain_error
/// when a pivot is not positive (the matrix is not positive definite).
inline dense_matrix cholesky(const dense_matrix& A) {
    if (A.rows != A.cols) throw std::logic_error("test matrix must be square");
    const std::int64_t n = A.rows;
    dense_matrix L(n, n);
    for (std::int64_t j = 0; j < n; ++j) {
        double d = A(j, j);
        for (std::int64_t k = 0; k < j; ++k) d -= L(j, k) * L(j, k);
        if (!(d > 0.0)) throw std::domain_error("test matrix is not positive definite");
        L(j, j) = std::sqrt(d);
        for (std::int64_t i = j + 1; i < n; ++i) {
            double s = A(i, j);
            for (std::int64_t k = 0; k < j; ++k) s -= L(i, k) * L(j, k);
            L(i, j) = s / L(j, j);
        }
    }
    return L;
}
inline bool positive_definite(const dense_matrix& A) {
    try {
        (void)cholesky(A);
        return true;
    } catch (const std::domain_error&) {
        return false;
    }
}
/// Solves A x = b for a symmetric positive definite A.
inline std::vector<double> cholesky_solve(const dense_matrix& A, std::span<const double> b) {
    const dense_matrix L = cholesky(A);
    const std::int64_t n = A.rows;
    std::vector<double> x(b.begin(), b.end());
    for (std::int64_t i = 0; i < n; ++i) {
        for (std::int64_t k = 0; k < i; ++k) x[i] -= L(i, k) * x[k];
        x[i] /= L(i, i);
    }
    for (std::int64_t i = n - 1; i >= 0; --i) {
        for (std::int64_t k = i + 1; k < n; ++k) x[i] -= L(k, i) * x[k];
        x[i] /= L(i, i);
    }
    return x;
}

// ── as the library's types ──────────────────────────────────────────────────

/// A square matrix as a public view of its own arrays.
inline csc_view csc(const sparse_matrix& A) {
    if (A.rows != A.cols) throw std::logic_error("test matrix must be square");
    return {A.rows, A.col_ptr, A.row_idx, A.values};
}

/// A matrix as the library's internal csc_ref, without validation. Index
/// arrays narrower than node_index/edge_index are widened into owned copies;
/// values are borrowed. Reports a matrix that is not square through
/// rows()/cols() and an empty ref.
class internal_csc {
public:
    explicit internal_csc(const sparse_matrix& A) : rows_(A.rows), cols_(A.cols) {
        if (rows_ != cols_) return;
        const edge_index* ptr = nullptr;
        const node_index* idx = nullptr;
        if constexpr (sizeof(edge_index) == sizeof(std::int32_t)) {
            ptr = reinterpret_cast<const edge_index*>(A.col_ptr.data());
        } else {
            offsets_.assign(A.col_ptr.begin(), A.col_ptr.end());
            ptr = offsets_.data();
        }
        if constexpr (sizeof(node_index) == sizeof(std::int32_t)) {
            idx = reinterpret_cast<const node_index*>(A.row_idx.data());
        } else {
            rows_copy_.assign(A.row_idx.begin(), A.row_idx.end());
            idx = rows_copy_.data();
        }
        ref_ = {static_cast<node_index>(rows_), ptr, idx, A.values.data()};
    }
    internal_csc(const internal_csc&) = delete;
    internal_csc& operator=(const internal_csc&) = delete;

    std::int64_t rows() const noexcept { return rows_; }
    std::int64_t cols() const noexcept { return cols_; }
    bool square() const noexcept { return rows_ == cols_; }
    const detail::csc_ref<double>& ref() const noexcept { return ref_; }
    operator detail::csc_ref<double>() const noexcept { return ref_; }

private:
    std::int64_t rows_ = 0, cols_ = 0;
    std::vector<edge_index> offsets_;
    std::vector<node_index> rows_copy_;
    detail::csc_ref<double> ref_;
};

/// Owned copy of a square matrix in the library's storage.
inline detail::csc_matrix<double> to_csc(const sparse_matrix& A) {
    const internal_csc input(A);
    return detail::csc_cast<double>(input.ref());
}

/// Owned copy of the library's CSC (int32 indices).
template<class T>
sparse_matrix from_csc(detail::csc_ref<T> A) {
    sparse_matrix out(A.n, A.n);
    for (std::size_t j = 0; j <= static_cast<std::size_t>(A.n); ++j)
        out.col_ptr[j] = static_cast<std::int32_t>(A.ptr[j]);
    out.row_idx.assign(A.idx, A.idx + A.nnz());
    out.values.assign(A.val, A.val + A.nnz());
    return out;
}
template<class T>
sparse_matrix from_csc(const detail::csc_matrix<T>& A) { return from_csc(A.ref()); }

/// A dense matrix as the library's strided ref.
inline detail::matrix_ref dense_of(dense_matrix& M) {
    return {M.data.data(), M.rows, M.cols, 1, M.rows};
}
inline detail::const_matrix_ref dense_of(const dense_matrix& M) {
    return {M.data.data(), M.rows, M.cols, 1, M.rows};
}

/// M-matrix lumping applied in place; returns the entries removed. Leaves A
/// unchanged when nothing is positive off the diagonal.
inline std::int64_t lump_in_place(sparse_matrix& A) {
    if (A.rows != A.cols || A.rows == 0) return 0;
    std::int64_t removed = 0;
    const internal_csc input(A);
    const auto lumped = detail::lump_positive_offdiagonals(input.ref(), removed);
    if (removed != 0) A = from_csc(lumped.ref());
    return removed;
}

/// Operator checks; a matrix that is not square reports the library's
/// not-square error.
inline detail::operator_scan scan_operator(const sparse_matrix& A) {
    const internal_csc input(A);
    if (!input.square()) {
        detail::operator_scan s;
        s.rows = A.rows;
        s.cols = A.cols;
        return s;
    }
    return detail::scan_operator(input.ref());
}
inline detail::operator_scan require_operator(const sparse_matrix& A,
                                              double lump_max_mass = kDefaultLumpMassCeiling) {
    detail::validate_lump_max_mass(lump_max_mass);
    const internal_csc input(A);
    if (!input.square()) detail::throw_operator_not_square(A.rows, A.cols);
    return detail::require_operator(input.ref(), lump_max_mass);
}
inline bool is_graph_adjacency(const sparse_matrix& M) {
    const internal_csc input(M);
    if (!input.square()) detail::throw_adjacency_not_square(M.rows, M.cols);
    return detail::is_graph_adjacency(input.ref());
}
inline sparse_matrix adjacency_laplacian(const sparse_matrix& M) {
    const internal_csc input(M);
    return from_csc(detail::adjacency_laplacian(input.ref()).ref());
}

/// Route selection for a matrix.
inline solve_backend select_solve_backend(const solve_options& opts, const sparse_matrix& A) {
    const internal_csc input(A);
    return detail::select_solve_backend(opts, &input.ref());
}

/// A solver with a timing tree attached (detail::solver_access::make).
inline solver with_timings(const sparse_matrix& A, const solve_options& opts, detail::checkpoint& timings) {
    const internal_csc input(A);
    return detail::solver_access::make(input, opts, &timings);
}

} // namespace apxchol::test

