#include "mmio.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

MtxResult load_mtx_as_adjacency(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open())
        throw std::runtime_error("Cannot open " + path);

    std::string line;
    bool is_symmetric = false;
    bool is_pattern = false;

    // Parse header line
    if (!std::getline(in, line))
        throw std::runtime_error("Empty file: " + path);
    if (line.find("%%MatrixMarket") == std::string::npos)
        throw std::runtime_error("Not a MatrixMarket file: " + path);

    // Check for symmetric and pattern
    std::string lower_line = line;
    std::transform(lower_line.begin(), lower_line.end(), lower_line.begin(), ::tolower);
    is_symmetric = lower_line.find("symmetric") != std::string::npos;
    is_pattern = lower_line.find("pattern") != std::string::npos;

    if (lower_line.find("array") != std::string::npos)
        throw std::runtime_error("Array (dense) format not supported, need coordinate format: " + path);

    // Skip comments
    while (std::getline(in, line))
        if (!line.empty() && line[0] != '%') break;

    // Parse dimensions
    int rows, cols, nnz;
    {
        std::istringstream iss(line);
        if (!(iss >> rows >> cols >> nnz))
            throw std::runtime_error("Bad dimension line in " + path);
    }
    if (rows != cols)
        throw std::runtime_error("Matrix is not square in " + path);

    int n = rows;
    std::vector<std::vector<Edge>> adj(n);

    // Parse entries
    for (int k = 0; k < nnz; ++k) {
        if (!std::getline(in, line))
            throw std::runtime_error("Premature end of file: " + path);
        std::istringstream iss(line);
        int i, j;
        double w = 1.0;
        if (!(iss >> i >> j))
            throw std::runtime_error("Bad entry line in " + path);
        if (!is_pattern)
            iss >> w;
        --i; --j; // 1-indexed to 0-indexed

        if (i == j) continue; // skip diagonal (we build Laplacian from adjacency)

        w = std::abs(w); // off-diagonal Laplacian entries are negative; use abs as edge weight

        adj[i].push_back({j, w});
        if (is_symmetric && i != j)
            adj[j].push_back({i, w});
    }

    return {std::move(adj), n, nnz};
}
