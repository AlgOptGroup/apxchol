#pragma once
#include <string>
#include <vector>
#include "graphs.h"

// Load a symmetric sparse matrix from Matrix Market (.mtx) format.
// Returns the adjacency list (off-diagonal entries become edge weights).
// Throws std::runtime_error on failure.
struct MtxResult {
    std::vector<std::vector<Edge>> adj;
    int n;
    int nnz_file; // nonzeros as reported in file header
};

MtxResult load_mtx_as_adjacency(const std::string& path);
