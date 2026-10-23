#pragma once
#include <Eigen/Sparse>
#include <string>
#include <vector>
#include "graphs.h"

// A .mtx file holds one of two different things, and the benchmark is told
// WHICH by the registry (--kind), never by sniffing the contents:
//
//   graph     the file is an adjacency / pattern matrix. The system it defines
//             is the graph Laplacian L = D - A, assembled here from |value|
//             (unit weights for a `pattern` file) -> load_mtx_as_adjacency.
//   operator  the file is an already-assembled Laplacian / SDDM operator. The
//             system it defines is the matrix itself, diagonal included, signs
//             as published -> load_mtx_as_operator.
//
// Both throw std::runtime_error on failure.

// Adjacency reading: off-diagonal entries become edge weights, the file's own
// diagonal is DROPPED (a graph self-loop contributes nothing to a Laplacian)
// and every weight is |value|. This is the graph-kind reading, unchanged.
struct MtxResult {
    std::vector<std::vector<Edge>> adj;
    int n;
    int nnz_file; // nonzeros as reported in file header
};

MtxResult load_mtx_as_adjacency(const std::string& path);

// Operator reading: every stored entry is kept at its published value, the
// diagonal included, and a `symmetric` header is expanded into both triangles.
// Duplicate entries are summed (MatrixMarket's own convention).
//
// A `pattern` file is REJECTED: it carries no values at all, so it cannot be an
// assembled operator — which makes this the loud failure for a registry entry
// declared `operator` by mistake.
struct MtxOperator {
    Eigen::SparseMatrix<double> A;
    int  n;
    long nnz_file;   // nonzeros as reported in the file header
    bool symmetric;  // header declared `symmetric` (one triangle stored)
};

MtxOperator load_mtx_as_operator(const std::string& path);
