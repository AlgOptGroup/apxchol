#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace cmg_native {
// Exact binary64 sum in units of 2^-1074. Every finite double needs at most
// 2098 bits; the driver's signed-32-bit entry cap adds at most31 carry bits.
// Two34-limb unsigned sums therefore cannot overflow. Per-column scratch
// avoids a multiprecision object for every graph vertex.
struct exact_binary64_sum {
    std::array<std::uint64_t, 34> positive{}, negative{};
    static void add_word(std::array<std::uint64_t, 34>& sum, unsigned word, std::uint64_t value) {
        while (value) {
            if (word >= sum.size()) throw std::runtime_error("exact sum capacity exceeded");
            const std::uint64_t old = sum[word];
            sum[word] += value;
            value = sum[word] < old;
            ++word;
        }
    }
    void add(double value) {
        static_assert(std::numeric_limits<double>::is_iec559 &&
                      std::numeric_limits<double>::digits == 53 && sizeof(double) == 8,
                      "CMG gauge requires IEEE binary64");
        if (!std::isfinite(value)) throw std::runtime_error("non-finite exact sum input");
        std::uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        const unsigned exponent = static_cast<unsigned>((bits >> 52) & 2047);
        const std::uint64_t mantissa = (bits & ((std::uint64_t{1} << 52) - 1)) |
                                      (exponent ? std::uint64_t{1} << 52 : 0);
        const unsigned shift = exponent ? exponent - 1 : 0;
        const unsigned word = shift / 64, offset = shift % 64;
        auto& sum = bits >> 63 ? negative : positive;
        add_word(sum, word, mantissa << offset);
        if (offset) add_word(sum, word + 1, mantissa >> (64 - offset));
    }
    bool zero() const { return positive == negative; }
};
// Canonical CMG grounds one coordinate at its coarse Laplacian solve. A
// disconnected Laplacian needs one gauge per component. Select a principal
// system only in that case; all coefficients and retained RHS values survive
// unchanged. The driver restores zero at removed coordinates and grades A*x-b
// against the original input. No diagonal shift or RHS projection is applied.
struct component_gauge {
    std::vector<int> retained;
    int components = 0;
    int grounded = 0;
};

template<class Matrix, class Vector>
component_gauge select_component_gauge(const Matrix& A, const Vector& b) {
    const int n = static_cast<int>(A.rows());
    if (A.cols() != n || b.size() != n)
        throw std::runtime_error("component gauge shape mismatch");
    std::vector<int> parent(n), size(n, 1);
    std::iota(parent.begin(), parent.end(), 0);
    auto root = [&](int v) {
        while (parent[v] != v) {
            parent[v] = parent[parent[v]];
            v = parent[v];
        }
        return v;
    };
    bool nonpositive_offdiagonal = true;
    for (int col = 0; col < n; ++col)
        for (typename Matrix::InnerIterator e(A, col); e; ++e) {
            if (!std::isfinite(e.value()))
                throw std::runtime_error("non-finite component operator");
            if (e.row() != col && e.value() > 0)
                nonpositive_offdiagonal = false;
            if (e.row() < col && e.value() != 0) {
                int a = root(static_cast<int>(e.row())), c = root(col);
                if (a != c) {
                    if (size[a] < size[c]) std::swap(a, c);
                    parent[c] = a;
                    size[a] += size[c];
                }
            }
        }
    component_gauge result;
    for (int i = 0; i < n; ++i) {
        parent[i] = root(i);
        result.components += parent[i] == i;
    }
    // Preserve canonical behavior for connected inputs and unsupported signs.
    if (result.components == 1 || !nonpositive_offdiagonal) return result;
    std::vector<long double> rhs_sum(n, 0), rhs_abs(n, 0);
    std::vector<bool> laplacian(n, true);
    std::vector<int> first(n, n);
    for (int col = 0; col < n; ++col) {
        exact_binary64_sum column_sum;
        for (typename Matrix::InnerIterator e(A, col); e; ++e) column_sum.add(e.value());
        // Columns equal rows for a symmetric operator; the driver checks
        // original symmetry before performing any selected transformation.
        laplacian[parent[col]] = laplacian[parent[col]] && column_sum.zero();
    }
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(b[i])) throw std::runtime_error("non-finite component RHS");
        const int c = parent[i];
        rhs_sum[c] += static_cast<long double>(b[i]);
        rhs_abs[c] += std::abs(static_cast<long double>(b[i]));
        first[c] = std::min(first[c], i);
    }
    for (int i = 0; i < n; ++i) if (parent[i] == i && laplacian[i]) {
        const long double bound = 128 * std::numeric_limits<double>::epsilon() * rhs_abs[i];
        if (std::abs(rhs_sum[i]) > bound)
            throw std::runtime_error("RHS incompatible with a Laplacian component");
        ++result.grounded;
    }
    if (!result.grounded) return result;
    result.retained.reserve(n - result.grounded);
    for (int i = 0; i < n; ++i)
        if (!laplacian[parent[i]] || i != first[parent[i]]) result.retained.push_back(i);
    return result;
}
} // namespace cmg_native
