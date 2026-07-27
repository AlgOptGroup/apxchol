#pragma once
/// Shared helper for the examples: a k×k 5-point grid Laplacian
/// (weighted graph Laplacian; singular, rank n−1).
#include <Eigen/Sparse>
#include <vector>

inline Eigen::SparseMatrix<double> grid_laplacian(int k) {
    const int n = k * k;
    std::vector<Eigen::Triplet<double>> t;
    t.reserve(static_cast<size_t>(n) * 5);
    auto id = [k](int i, int j) { return i * k + j; };
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) {
            int deg = 0;
            auto edge = [&](int i2, int j2) {
                if (i2 < 0 || j2 < 0 || i2 >= k || j2 >= k) return;
                t.emplace_back(id(i, j), id(i2, j2), -1.0);
                ++deg;
            };
            edge(i - 1, j); edge(i + 1, j); edge(i, j - 1); edge(i, j + 1);
            t.emplace_back(id(i, j), id(i, j), double(deg));
        }
    Eigen::SparseMatrix<double> L(n, n);
    L.setFromTriplets(t.begin(), t.end());
    return L;
}
