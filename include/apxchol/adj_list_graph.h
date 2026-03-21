#pragma once
#include "graph.h"
#include <vector>

namespace apxchol {

/// Adjacency-list graph representation.
///
/// Edges stored in a flat array; each vertex holds indices into it.
/// Follows the cp-algo pattern: one edge object per undirected edge,
/// referenced from both endpoints.
class adj_list_graph final : public graph {
public:
    struct edge {
        node_index u, v;
        double w;

        node_index traverse(node_index from) const { return u ^ v ^ from; }
    };

    adj_list_graph() = default;
    explicit adj_list_graph(int n);

    /// Add an undirected edge {u, v} with weight w. Returns edge index.
    edge_index add_edge(node_index u, node_index v, double w);

    int n() const override { return n_; }
    int m() const override { return static_cast<int>(edges_.size()); }

    void for_neighbors(node_index v,
                       const std::function<void(node_index, double)>& cb) const override;

    const edge& get_edge(edge_index e) const { return edges_[e]; }
    const std::vector<edge_index>& adj(node_index v) const { return adj_[v]; }

private:
    int n_ = 0;
    std::vector<edge> edges_;
    std::vector<std::vector<edge_index>> adj_;
};

/// Load a symmetric sparse matrix from Matrix Market (.mtx) format.
/// Off-diagonal entries become undirected edges; diagonal is ignored.
adj_list_graph load_mtx(const std::string& path);

} // namespace apxchol
