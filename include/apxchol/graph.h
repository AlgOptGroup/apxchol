#pragma once
#include <Eigen/Sparse>
#include <functional>
#include <cstdint>

namespace apxchol {

using node_index = int;
using edge_index = int;

/// Weighted undirected graph interface.
///
/// Vertices are numbered [0, n). Edges are undirected with positive weights.
/// Concrete representations (adjacency list, CSC, etc.) derive from this.
class graph {
public:
    virtual ~graph() = default;

    virtual int n() const = 0;
    virtual int m() const = 0;  // number of undirected edges

    /// Call `callback(neighbor, weight)` for each neighbor of `v`.
    virtual void for_neighbors(node_index v,
                               const std::function<void(node_index, double)>& callback) const = 0;

    /// Weighted degree of vertex v.
    virtual double weighted_degree(node_index v) const;

    /// Build the graph Laplacian L = D - A as a sparse matrix.
    Eigen::SparseMatrix<double> laplacian() const;
};

} // namespace apxchol
