/// A Matrix Market file read with fast_matrix_market, then assembled as CSC.
#include "apxchol.h"
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <sstream>
#include <tuple>
#include <vector>

namespace fmm = fast_matrix_market;

// Laplacian of the path 1-2-3-4 with edge weights 1, 2, 1 (lower triangle).
const char* const example_file = R"(%%MatrixMarket matrix coordinate real symmetric
4 4 7
1 1 1
2 1 -1
2 2 3
3 2 -2
3 3 3
4 3 -1
4 4 1
)";

int main(int argc, char** argv) {
    std::ifstream file;
    std::istringstream text(example_file);
    if (argc > 1) file.open(argv[1]);
    std::istream& in = argc > 1 ? static_cast<std::istream&>(file) : text;

    // Symmetric files store one triangle; the reader mirrors it into both.
    std::int64_t rows = 0, cols = 0;
    std::vector<std::int32_t> row, col;
    std::vector<double> val;
    fmm::read_matrix_market_triplet(in, rows, cols, row, col, val);

    // Order the triplets by column, then row, and sum duplicates.
    std::vector<std::size_t> order(val.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return std::tie(col[a], row[a]) < std::tie(col[b], row[b]);
    });
    std::vector<std::int32_t> col_ptr(static_cast<std::size_t>(cols) + 1, 0), row_idx;
    std::vector<double> values;
    for (std::size_t k = 0, last = 0; k < order.size(); last = order[k++]) {
        const std::size_t e = order[k];
        if (k > 0 && col[e] == col[last] && row[e] == row[last]) {
            values.back() += val[e];
        } else {
            row_idx.push_back(row[e]);
            values.push_back(val[e]);
            ++col_ptr[static_cast<std::size_t>(col[e]) + 1];
        }
    }
    std::partial_sum(col_ptr.begin(), col_ptr.end(), col_ptr.begin());

    const apxchol::csc_view A{cols, col_ptr, row_idx, values};
    std::vector<double> b(static_cast<std::size_t>(cols), 0.0);
    b.front() = 1.0;   // a Laplacian RHS sums to zero
    b.back() = -1.0;
    const auto result = apxchol::solve(A, b, 1e-10);
    std::printf("Matrix Market: %lld x %lld, residual %.1e\n",
                static_cast<long long>(rows), static_cast<long long>(cols), result.residual);
    return result.converged ? 0 : 1;
}
