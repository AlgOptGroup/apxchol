#pragma once
/// Strided dense views and the vector kernels the solver routes use outside
/// their fused PCG passes. Not part of the public API.
///
/// Reductions use one fixed association, independent of the thread count and
/// of the view's stride: the index range is cut into kReduceBlock-element
/// blocks; inside a block, element i accumulates into lane i mod kReduceLanes;
/// the lanes combine pairwise; block sums add in block order. Blocks may run
/// in parallel without changing a bit. Contiguous views take a loop the
/// compiler vectorizes; strided views run the same arithmetic element by
/// element.

#include "apxchol/solver/detail/cpu_parallel.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace apxchol::detail {

/// Column-major strided view: element (i, j) is data[i * row_stride + j * col_stride].
template<class T>
class dense_ref {
public:
    dense_ref() = default;
    dense_ref(T* data, std::int64_t rows, std::int64_t cols,
              std::int64_t row_stride, std::int64_t col_stride) noexcept
        : data_(data), rows_(rows), cols_(cols), row_stride_(row_stride), col_stride_(col_stride) {}
    /// A contiguous vector of n elements.
    dense_ref(T* data, std::int64_t n) noexcept : dense_ref(data, n, 1, 1, n) {}
    template<class U>
        requires (std::is_same_v<const U, T> && !std::is_same_v<U, T>)
    dense_ref(const dense_ref<U>& other) noexcept
        : dense_ref(other.data(), other.rows(), other.cols(), other.row_stride(), other.col_stride()) {}

    T* data() const noexcept { return data_; }
    std::int64_t rows() const noexcept { return rows_; }
    std::int64_t cols() const noexcept { return cols_; }
    std::int64_t size() const noexcept { return rows_ * cols_; }
    std::int64_t row_stride() const noexcept { return row_stride_; }
    std::int64_t col_stride() const noexcept { return col_stride_; }

    T& operator()(std::int64_t i, std::int64_t j) const noexcept {
        return data_[i * row_stride_ + j * col_stride_];
    }
    /// Element i of a single-column view.
    T& operator[](std::int64_t i) const noexcept { return data_[i * row_stride_]; }
    dense_ref col(std::int64_t j) const noexcept {
        return {data_ + j * col_stride_, rows_, 1, row_stride_, col_stride_};
    }
    dense_ref middle_cols(std::int64_t first, std::int64_t count) const noexcept {
        return {data_ + first * col_stride_, rows_, count, row_stride_, col_stride_};
    }

private:
    T* data_ = nullptr;
    std::int64_t rows_ = 0, cols_ = 0, row_stride_ = 1, col_stride_ = 0;
};

using matrix_ref = dense_ref<double>;
using const_matrix_ref = dense_ref<const double>;

namespace vec {

inline constexpr std::int64_t kReduceBlock = 8192;
inline constexpr int kReduceLanes = 8;

/// Sum of term(i) over [first, last), last - first <= kReduceBlock, in the
/// fixed lane association.
template<class Term>
inline double block_sum(std::int64_t first, std::int64_t last, Term term) noexcept {
    double lane[kReduceLanes] = {};
    std::int64_t i = first;
    for (; i + kReduceLanes <= last; i += kReduceLanes)
        for (int k = 0; k < kReduceLanes; ++k) lane[k] += term(i + k);
    for (int k = 0; i < last; ++i, ++k) lane[k] += term(i);
    return ((lane[0] + lane[1]) + (lane[2] + lane[3])) +
           ((lane[4] + lane[5]) + (lane[6] + lane[7]));
}

/// Σ term(i) for i in [0, n) in the fixed association described above.
template<class Term>
double ordered_sum(std::int64_t n, Term term) {
    if (n <= 0) return 0.0;
    const std::int64_t blocks = (n + kReduceBlock - 1) / kReduceBlock;
    const int workers = blocks > 1 ? cpu_solve_threads(static_cast<std::size_t>(n)) : 1;
    if (workers <= 1) {
        double total = 0.0;
        for (std::int64_t b = 0; b < blocks; ++b)
            total += block_sum(b * kReduceBlock, std::min(n, (b + 1) * kReduceBlock), term);
        return total;
    }
    std::vector<double> partial(static_cast<std::size_t>(blocks));
    #pragma omp parallel for schedule(static) num_threads(workers)
    for (std::int64_t b = 0; b < blocks; ++b)
        partial[static_cast<std::size_t>(b)] =
            block_sum(b * kReduceBlock, std::min(n, (b + 1) * kReduceBlock), term);
    double total = 0.0;
    for (const double value : partial) total += value;
    return total;
}

/// Bitwise-OR of word(i) over [0, n); order-free, so it may run in any split.
template<class Word>
std::uint64_t any_bits(std::int64_t n, Word word) {
    std::uint64_t bits = 0;
    const int workers = cpu_solve_threads(static_cast<std::size_t>(std::max<std::int64_t>(n, 0)));
    #pragma omp parallel for schedule(static) reduction(| : bits) if(workers > 1) num_threads(workers)
    for (std::int64_t i = 0; i < n; ++i) bits |= word(i);
    return bits;
}

/// Maximum of key(i) over [0, n); order-free.
template<class Key>
std::uint64_t max_bits(std::int64_t n, Key key) {
    std::uint64_t best = 0;
    const int workers = cpu_solve_threads(static_cast<std::size_t>(std::max<std::int64_t>(n, 0)));
    #pragma omp parallel for schedule(static) reduction(max : best) if(workers > 1) num_threads(workers)
    for (std::int64_t i = 0; i < n; ++i) best = std::max(best, key(i));
    return best;
}

inline constexpr std::uint64_t kExponentBits = 0x7ff0000000000000ull;
inline constexpr std::uint64_t kMagnitudeBits = 0x7fffffffffffffffull;

/// Applies f to a contiguous pointer when the view has unit stride, so the
/// compiler sees a plain indexed loop; otherwise to the strided view.
template<class F>
decltype(auto) with_access(const_matrix_ref v, F f) {
    if (v.row_stride() == 1) {
        const double* p = v.data();
        return f([p](std::int64_t i) { return p[i]; });
    }
    return f([v](std::int64_t i) { return v[i]; });
}

/// Σ v[i]^2 of a single-column view.
inline double squared_norm(const_matrix_ref v) {
    return with_access(v, [&](auto at) {
        return ordered_sum(v.rows(), [at](std::int64_t i) { const double x = at(i); return x * x; });
    });
}

/// Σ (scale * v[i])^2, rounding each product as if the scaled vector were stored.
inline double scaled_squared_norm(const_matrix_ref v, double scale) {
    return with_access(v, [&](auto at) {
        return ordered_sum(v.rows(), [at, scale](std::int64_t i) {
            const double x = scale * at(i);
            return x * x;
        });
    });
}

/// Σ v[i] of a single-column view.
inline double sum(const_matrix_ref v) {
    return with_access(v, [&](auto at) { return ordered_sum(v.rows(), at); });
}

/// Σ v[i] / n, or 0 for an empty view.
inline double mean(const_matrix_ref v) {
    return v.rows() == 0 ? 0.0 : sum(v) / static_cast<double>(v.rows());
}

/// max |v[i]|, compared through the magnitude bits: for non-negative doubles
/// the integer order is the numeric order, so the result is exact and needs
/// no floating-point max reduction. A NaN compares above +infinity. 0 for an
/// empty view.
inline double max_abs(const_matrix_ref v) {
    const std::uint64_t bits = with_access(v, [&](auto at) {
        return max_bits(v.rows(), [at](std::int64_t i) {
            return std::bit_cast<std::uint64_t>(at(i)) & kMagnitudeBits;
        });
    });
    return std::bit_cast<double>(bits);
}

/// Every element is finite (its exponent field is not all ones).
inline bool all_finite(const_matrix_ref v) {
    return with_access(v, [&](auto at) {
        return any_bits(v.rows(), [at](std::int64_t i) {
            return static_cast<std::uint64_t>(
                (std::bit_cast<std::uint64_t>(at(i)) & kExponentBits) == kExponentBits);
        }) == 0;
    });
}

/// Every element compares equal to zero (either sign).
inline bool is_zero(const_matrix_ref v) {
    return with_access(v, [&](auto at) {
        return any_bits(v.rows(), [at](std::int64_t i) {
            return std::bit_cast<std::uint64_t>(at(i)) & kMagnitudeBits;
        }) == 0;
    });
}

/// Every element of every column is finite.
inline bool all_finite_matrix(const_matrix_ref m) {
    for (std::int64_t j = 0; j < m.cols(); ++j)
        if (!all_finite(m.col(j))) return false;
    return true;
}

} // namespace vec

} // namespace apxchol::detail
