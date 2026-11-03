#include <apxchol.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

static_assert(__cplusplus > 202002L, "apxchol must propagate C++23");

int main() {
    for (int n : {17, 33, 65}) {
        // Tridiagonal SDDM matrix 3 on the diagonal, -1 beside it, in CSC.
        std::vector<std::int32_t> col_ptr{0}, row_idx;
        std::vector<double> values;
        for (int j = 0; j < n; ++j) {
            for (int i = j - 1; i <= j + 1; ++i) {
                if (i < 0 || i >= n) continue;
                row_idx.push_back(i);
                values.push_back(i == j ? 3.0 : -1.0);
            }
            col_ptr.push_back(static_cast<std::int32_t>(row_idx.size()));
        }
        auto multiply = [&](const std::vector<double>& x) {
            std::vector<double> y(n, 0.0);
            for (int j = 0; j < n; ++j)
                for (auto p = col_ptr[j]; p < col_ptr[j + 1]; ++p) y[row_idx[p]] += values[p] * x[j];
            return y;
        };
        std::vector<double> expected(n);
        for (int i = 0; i < n; ++i) expected[i] = std::sin(0.13 * i);
        const std::vector<double> rhs = multiply(expected);
        const auto result = apxchol::solve(apxchol::csc_view{n, col_ptr, row_idx, values}, rhs);
        const std::vector<double> ax = multiply(result.x);
        double r2 = 0, b2 = 0;
        for (int i = 0; i < n; ++i) {
            r2 += (ax[i] - rhs[i]) * (ax[i] - rhs[i]);
            b2 += rhs[i] * rhs[i];
        }
        const double residual = std::sqrt(r2 / b2);
        if (!std::isfinite(residual) || residual > 1e-8) {
            std::cerr << "original-system residual failed at size " << n << '\n';
            return 1;
        }
    }
    std::cout << "C++23, three solves passed\n";
}
