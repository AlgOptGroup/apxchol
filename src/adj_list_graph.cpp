#include "apxchol/adj_list_graph.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace apxchol {

adj_list_graph::adj_list_graph(int n) : n_(n), adj_(n) {}

edge_index adj_list_graph::add_edge(node_index u, node_index v, double w) {
    auto idx = static_cast<edge_index>(edges_.size());
    edges_.push_back({u, v, w});
    adj_[u].push_back(idx);
    adj_[v].push_back(idx);
    return idx;
}

void adj_list_graph::for_neighbors(node_index v,
                                   const std::function<void(node_index, double)>& cb) const {
    for (edge_index ei : adj_[v]) {
        const auto& e = edges_[ei];
        cb(e.traverse(v), e.w);
    }
}

// ── Matrix Market loader ────────────────────────────

adj_list_graph load_mtx(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open())
        throw std::runtime_error("Cannot open " + path);

    std::string line;

    // Header
    if (!std::getline(in, line))
        throw std::runtime_error("Empty file: " + path);
    if (line.find("%%MatrixMarket") == std::string::npos)
        throw std::runtime_error("Not a MatrixMarket file: " + path);

    std::string lower = line;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    bool symmetric = lower.find("symmetric") != std::string::npos;
    bool pattern   = lower.find("pattern")   != std::string::npos;

    if (lower.find("array") != std::string::npos)
        throw std::runtime_error("Dense (array) format not supported: " + path);

    // Skip comments
    while (std::getline(in, line))
        if (!line.empty() && line[0] != '%') break;

    // Dimensions
    int rows, cols, nnz;
    {
        std::istringstream iss(line);
        if (!(iss >> rows >> cols >> nnz))
            throw std::runtime_error("Bad dimension line in " + path);
    }
    if (rows != cols)
        throw std::runtime_error("Matrix is not square in " + path);

    adj_list_graph G(rows);

    for (int k = 0; k < nnz; ++k) {
        if (!std::getline(in, line))
            throw std::runtime_error("Premature end of file: " + path);
        std::istringstream iss(line);
        int i, j;
        double w = 1.0;
        if (!(iss >> i >> j))
            throw std::runtime_error("Bad entry in " + path);
        if (!pattern) iss >> w;
        --i; --j;  // 1-indexed → 0-indexed

        if (i == j) continue;  // skip diagonal

        w = std::abs(w);  // Laplacian off-diag entries are negative; use abs

        if (symmetric) {
            // Symmetric MTX stores only lower triangle; add_edge handles both directions.
            G.add_edge(i, j, w);
        } else {
            // General MTX stores both (i,j) and (j,i). Only add once (lower triangle).
            if (i > j)
                G.add_edge(i, j, w);
        }
    }

    return G;
}

} // namespace apxchol
