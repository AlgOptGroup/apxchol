#include "graphs.h"
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

std::vector<std::vector<Edge>> grid_graph_3d(int n, double w) {
    int total = n * n * n;
    std::vector<std::vector<Edge>> adj(total);

    auto id = [n](int x, int y, int z) { return (x * n + y) * n + z; };

    for (int x = 0; x < n; ++x) {
        for (int y = 0; y < n; ++y) {
            for (int z = 0; z < n; ++z) {
                int u = id(x, y, z);
                if (x > 0)     adj[u].push_back({id(x - 1, y, z), w});
                if (x + 1 < n) adj[u].push_back({id(x + 1, y, z), w});
                if (y > 0)     adj[u].push_back({id(x, y - 1, z), w});
                if (y + 1 < n) adj[u].push_back({id(x, y + 1, z), w});
                if (z > 0)     adj[u].push_back({id(x, y, z - 1), w});
                if (z + 1 < n) adj[u].push_back({id(x, y, z + 1), w});
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
    if (n <= 1 || p <= 0.0) return adj;
    if (p >= 1.0) p = 1.0 - 1e-15;

    std::mt19937_64 rng(seed);
    // Fast O(m) generation using geometric distribution to skip non-edges.
    // Each potential edge (i<j) is present independently with probability p.
    // We enumerate edges in lexicographic order of (i,j) and jump by
    // Geometric(p) gaps.
    double log1mp = std::log(1.0 - p);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    long long total_pairs = (long long)n * (n - 1) / 2;
    long long pos = -1; // linear index into upper triangle

    while (true) {
        double u = unif(rng);
        if (u == 0.0) u = 1e-300; // avoid log(0)
        long long skip = (long long)(std::log(u) / log1mp);
        pos += 1 + skip;
        if (pos >= total_pairs) break;
        // Decode linear index to (i, j):
        // pos = i*n - i*(i+1)/2 + (j - i - 1)
        // Solve for i: i ≈ n - 0.5 - sqrt((n-0.5)^2 - 2*pos)
        double nd = (double)n;
        int i = (int)(nd - 0.5 - std::sqrt((nd - 0.5) * (nd - 0.5) - 2.0 * pos));
        if (i < 0) i = 0;
        long long row_start = (long long)i * n - (long long)i * (i + 1) / 2;
        while (row_start + (n - i - 1) <= pos && i < n - 1) {
            i++;
            row_start = (long long)i * n - (long long)i * (i + 1) / 2;
        }
        int j = (int)(pos - row_start) + i + 1;
        if (i >= 0 && i < n && j > i && j < n) {
            adj[i].push_back({j, w});
            adj[j].push_back({i, w});
        }
    }
    return adj;
}
