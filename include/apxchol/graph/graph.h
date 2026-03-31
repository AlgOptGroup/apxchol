#pragma once
#include "apxchol/graph/incidence_list.h"
#include <ranges>
#include <vector>

namespace apxchol {

/// Mutable incidence-list graph with a flat edge pool.
///
/// Each undirected edge {u,v,w} is stored once in the pool with XOR-encoded
/// endpoints (u^v).  Both u and v reference the same pool entry;
/// the target is recovered by XORing the stored value with the source vertex.
/// The Incidence template parameter controls how per-vertex lists of
/// edge indices are stored (vec_incidence, forward_star_incidence,
/// small_vec_incidence, or any user-provided type with the same interface).
template<incidence_storage Incidence = forward_star_incidence>
class graph {
public:
    struct edge {
        node_index to; double w;
    private:
        friend graph;
        node_index traverse(node_index from) const { return to ^ from; }
    };

    graph() = default;

    explicit graph(index_t n)
        : n_(n), m_(0), num_active_(n), active_(n, true), excess_(n, 0.0) {
        adj_.init(n);
    }

    index_t n() const { return n_; }
    index_t m() const { return m_; }

    auto neighbors(node_index v) const {
        return adj_[v] | std::views::transform(
            [this, v](edge_index i) -> edge {
                return {edges_[i].traverse(v), edges_[i].w};
            });
    }

    // ── mutation ──

    void add_edge(node_index u, node_index v, double w) {
        auto idx = static_cast<edge_index>(edges_.size());
        edges_.push_back({u ^ v, w});
        adj_.push(u, idx);
        adj_.push(v, idx);
        ++m_;
    }

    void deactivate(node_index v) {
        if (!active_[v]) return;
        active_[v] = false;
        --num_active_;
        adj_.clear(v);
    }

    /// Merge parallel edges to the same neighbor into one, accumulating weights.
    /// Uses a vertex-indexed bucket (thread_local, lazily grown to n) for O(d)
    /// deduplication instead of O(d log d) sorting.
    void merge_parallel_edges(node_index v) {
        // first[u] = pool index of the surviving representative edge to neighbor u,
        // or npos if no edge to u has been seen yet this call.
        static constexpr edge_index npos = edge_index(-1);
        thread_local static std::vector<edge_index> first;
        thread_local static std::vector<node_index> touched;
        if (first.size() < size_t(n_)) first.assign(n_, npos);
        touched.clear();

        // Single pass: prune dead neighbours + deduplicate + accumulate.
        adj_.filter(v, [&](edge_index idx) {
            auto u = edges_[idx].traverse(v);         // decoded neighbor
            if (!active_[u]) return false;             // dead neighbour
            if (first[u] == npos) {
                first[u] = idx;                       // first edge to this neighbour
                touched.push_back(u);
                return true;                          // keep in adj list
            }
            edges_[first[u]].w += edges_[idx].w;      // accumulate into representative
            return false;                             // discard duplicate
        });

        // Reset bucket via touched list — O(degree), not O(n).
        for (auto t : touched) first[t] = npos;
    }

    /// Prune dead edges and return surviving (active) degree in one pass.
    index_t prune_and_degree(node_index v) {
        index_t count = 0;
        adj_.filter(v, [&](edge_index idx) {
            return active_[edges_[idx].traverse(v)];
        }, [&](edge_index) { ++count; });
        return count;
    }

    // ── active-aware queries ──

    bool is_active(node_index v) const { return active_[v]; }
    index_t num_active() const { return num_active_; }

    /// Raw access to adjacency list and edge pool (for tight inner loops).
    auto adj(node_index v) const { return adj_[v]; }
    node_index edge_target(edge_index idx, node_index from) const {
        return edges_[idx].traverse(from);
    }

    /// Per-vertex excess diagonal (for SDDM matrices).
    /// For a pure Laplacian this is zero everywhere.
    /// For SDDM,  excess[v] = M[v,v] - sum of incident edge weights.
    double  excess(node_index v) const { return excess_[v]; }
    double& excess(node_index v)       { return excess_[v]; }

    /// Approximate heap memory usage in bytes.
    std::size_t memory_bytes() const {
        return edges_.capacity() * sizeof(edge)
             + adj_.memory_bytes()
             + active_.capacity() * sizeof(char)
             + excess_.capacity() * sizeof(double);
    }


private:
    index_t n_ = 0;
    index_t m_ = 0;
    index_t num_active_ = 0;
    std::vector<edge> edges_;
    Incidence adj_;
    std::vector<char> active_;
    std::vector<double> excess_;
};

} // namespace apxchol
