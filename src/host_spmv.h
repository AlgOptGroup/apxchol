#pragma once

#include <array>

namespace apxchol::detail {

// Four independent sums preserve the CPU SpMV's accumulation order.
// x(row, column) supports both packed panels and strided vector views.
template<int C, class S, class Index, class Offset, class Entry>
std::array<double, C> sparse_row_product(const Index* col, const S* val,
                                         Offset k, Offset end, Entry x) {
    std::array<double, C> a{}, b{}, c{}, d{};
    for (; end - k >= 4; k += 4)
        for (int q = 0; q < C; q++) {
            a[q] += val[k + 0] * x(col[k + 0], q);
            b[q] += val[k + 1] * x(col[k + 1], q);
            c[q] += val[k + 2] * x(col[k + 2], q);
            d[q] += val[k + 3] * x(col[k + 3], q);
        }
    std::array<double, C> sum{};
    for (int q = 0; q < C; q++)
        sum[q] = (a[q] + b[q]) + (c[q] + d[q]);
    for (; k < end; k++)
        for (int q = 0; q < C; q++)
            sum[q] += val[k] * x(col[k], q);
    return sum;
}

} // namespace apxchol::detail
