#include "../include/graphs.h"
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>

static inline double harmonic_mean(double a, double b) {
    return (2.0 * a * b) / (a + b);
}

std::vector<std::vector<Edge>> grid_graph(int rows, int cols,
                                          double a_left,
                                          double a_right,
                                          int jump_col) {
    int n = rows * cols;
    std::vector<std::vector<Edge>> adj(n);

    auto id = [cols](int r, int c) { return r * cols + c; };
    const int interface_col = (jump_col < 0) ? (cols / 2) : jump_col;
    auto coeff = [&](int r, int c) {
        return (c < interface_col) ? a_left : a_right;
    };

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int u = id(r, c);
            double au = coeff(r, c);
            if (r > 0) {
                int v = id(r - 1, c);
                double w = harmonic_mean(au, coeff(r - 1, c));
                adj[u].push_back({v, w});
            }
            if (r + 1 < rows) {
                int v = id(r + 1, c);
                double w = harmonic_mean(au, coeff(r + 1, c));
                adj[u].push_back({v, w});
            }
            if (c > 0) {
                int v = id(r, c - 1);
                double w = harmonic_mean(au, coeff(r, c - 1));
                adj[u].push_back({v, w});
            }
            if (c + 1 < cols) {
                int v = id(r, c + 1);
                double w = harmonic_mean(au, coeff(r, c + 1));
                adj[u].push_back({v, w});
            }
        }
    }
    return adj;
}

std::vector<std::vector<Edge>> grid_graph_checkerboard(int rows, int cols,
                                                       double a_high,
                                                       double a_low,
                                                       int tile) {
    int n = rows * cols;
    std::vector<std::vector<Edge>> adj(n);

    auto id = [cols](int r, int c) { return r * cols + c; };
    const int t = (tile <= 0) ? 1 : tile;

    auto coeff = [&](int r, int c) {
        int rr = r / t;
        int cc = c / t;
        return ((rr + cc) & 1) ? a_low : a_high;
    };

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int u = id(r, c);
            double au = coeff(r, c);
            if (r > 0) {
                int v = id(r - 1, c);
                double w = harmonic_mean(au, coeff(r - 1, c));
                adj[u].push_back({v, w});
            }
            if (r + 1 < rows) {
                int v = id(r + 1, c);
                double w = harmonic_mean(au, coeff(r + 1, c));
                adj[u].push_back({v, w});
            }
            if (c > 0) {
                int v = id(r, c - 1);
                double w = harmonic_mean(au, coeff(r, c - 1));
                adj[u].push_back({v, w});
            }
            if (c + 1 < cols) {
                int v = id(r, c + 1);
                double w = harmonic_mean(au, coeff(r, c + 1));
                adj[u].push_back({v, w});
            }
        }
    }
    return adj;
}

std::vector<std::vector<Edge>> erdos_renyi_graph(int n, double p, unsigned seed, double w) {
    std::vector<std::vector<Edge>> adj(n);
    std::mt19937 rng(seed);
    std::bernoulli_distribution bern(p);

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (bern(rng)) {
                adj[i].push_back({j, w});
                adj[j].push_back({i, w});
            }
        }
    }
    return adj;
}
