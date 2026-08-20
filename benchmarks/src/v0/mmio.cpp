#include "mmio.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace {

// What the banner + dimension lines of a coordinate MatrixMarket file say.
// Shared by both readers so the two kinds can never drift apart on the parts
// they legitimately agree about (geometry and storage), only on the part they
// are supposed to differ on: what the VALUES mean.
struct MtxHeader {
    bool symmetric = false;
    bool pattern   = false;
    int  rows = 0, cols = 0;
    long nnz  = 0;
};

MtxHeader read_header(std::ifstream& in, const std::string& path) {
    MtxHeader h;
    std::string line;

    if (!std::getline(in, line))
        throw std::runtime_error("Empty file: " + path);
    if (line.find("%%MatrixMarket") == std::string::npos)
        throw std::runtime_error("Not a MatrixMarket file: " + path);

    std::string lower_line = line;
    std::transform(lower_line.begin(), lower_line.end(), lower_line.begin(), ::tolower);
    h.symmetric = lower_line.find("symmetric") != std::string::npos;
    h.pattern   = lower_line.find("pattern")   != std::string::npos;

    if (lower_line.find("array") != std::string::npos)
        throw std::runtime_error("Array (dense) format not supported, need coordinate format: " + path);

    // Skip comments
    while (std::getline(in, line))
        if (!line.empty() && line[0] != '%') break;

    {
        std::istringstream iss(line);
        if (!(iss >> h.rows >> h.cols >> h.nnz))
            throw std::runtime_error("Bad dimension line in " + path);
    }
    if (h.rows != h.cols)
        throw std::runtime_error("Matrix is not square in " + path);

    return h;
}

} // namespace

MtxResult load_mtx_as_adjacency(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open())
        throw std::runtime_error("Cannot open " + path);

    const MtxHeader h = read_header(in, path);
    const int n = h.rows;
    std::vector<std::vector<Edge>> adj(n);

    std::string line;
    // Parse entries
    for (long k = 0; k < h.nnz; ++k) {
        if (!std::getline(in, line))
            throw std::runtime_error("Premature end of file: " + path);
        std::istringstream iss(line);
        int i, j;
        double w = 1.0;
        if (!(iss >> i >> j))
            throw std::runtime_error("Bad entry line in " + path);
        if (!h.pattern)
            iss >> w;
        --i; --j; // 1-indexed to 0-indexed

        if (i == j) continue; // skip diagonal (we build Laplacian from adjacency)

        w = std::abs(w); // off-diagonal Laplacian entries are negative; use abs as edge weight

        adj[i].push_back({j, w});
        if (h.symmetric && i != j)
            adj[j].push_back({i, w});
    }

    return {std::move(adj), n, static_cast<int>(h.nnz)};
}

MtxOperator load_mtx_as_operator(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open())
        throw std::runtime_error("Cannot open " + path);

    const MtxHeader h = read_header(in, path);
    if (h.pattern)
        throw std::runtime_error(
            "kind=operator was declared for " + path +
            ", but its MatrixMarket header says `pattern`: the file carries no "
            "values, so there is no published operator to solve. A pattern file "
            "can only be kind=graph (solved as L = D - A with unit weights).");

    const int n = h.rows;
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<size_t>(h.symmetric ? 2 * h.nnz : h.nnz));

    std::string line;
    for (long k = 0; k < h.nnz; ++k) {
        if (!std::getline(in, line))
            throw std::runtime_error("Premature end of file: " + path);
        std::istringstream iss(line);
        int i, j;
        double v;
        if (!(iss >> i >> j >> v))
            throw std::runtime_error("Bad entry line in " + path);
        --i; --j; // 1-indexed to 0-indexed

        // Published value, sign and all — including the diagonal, which is the
        // whole point of reading the file as an operator.
        trips.emplace_back(i, j, v);
        if (h.symmetric && i != j)
            trips.emplace_back(j, i, v);
    }

    Eigen::SparseMatrix<double> A(n, n);
    A.setFromTriplets(trips.begin(), trips.end());   // sums duplicates, as MatrixMarket specifies
    A.makeCompressed();

    return {std::move(A), n, h.nnz, h.symmetric};
}
