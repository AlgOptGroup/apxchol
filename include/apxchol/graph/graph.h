#pragma once
#include "apxchol/graph/incidence_list.h"
#include <algorithm>
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
        : n_(n), m_(0), num_active_(n), active_(n, true) {
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

    void merge_parallel_edges(node_index v) {
        // Collect active edges into a local buffer with pre-read weights.
        struct edge_info { node_index xor_to; edge_index idx; double w; };
        std::vector<edge_info> buf;
        for (auto idx : adj_[v])
            if (active_[edges_[idx].traverse(v)])
                buf.push_back({edges_[idx].to, idx, edges_[idx].w});

        if (buf.size() < 2) {
            // No merging possible — just prune dead edges.
            adj_.filter(v, [&](edge_index idx) {
                return active_[edges_[idx].traverse(v)];
            });
            return;
        }

        std::ranges::sort(buf, {}, &edge_info::xor_to);

        // Accumulate weight into first of each group; zero the rest.
        for (size_t i = 0; i < buf.size();) {
            auto xor_to = buf[i].xor_to;
            double sum = buf[i].w;
            size_t first = i++;
            while (i < buf.size() && buf[i].xor_to == xor_to) {
                sum += buf[i].w;
                edges_[buf[i].idx].w = 0;
                ++i;
            }
            edges_[buf[first].idx].w = sum;
        }

        // Filter out dead edges and zeroed-weight duplicates.
        // No pool allocation — safe for concurrent use on disjoint vertex sets.
        adj_.filter(v, [&](edge_index idx) {
            return edges_[idx].w > 0.0 && active_[edges_[idx].traverse(v)];
        });
    }

    /// Remove edges to inactive vertices from v's incidence list.
    /// O(degree) cleanup that speeds up future traversals.
    /// For forward_star, relinks the chain in-place (no pool growth).
    void prune_inactive_edges(node_index v) {
        adj_.filter(v, [&](edge_index idx) {
            return active_[edges_[idx].traverse(v)];
        });
    }

    /// Prune dead edges and return surviving (active) degree in one pass.
    int prune_and_degree(node_index v) {
        int count = 0;
        adj_.filter(v, [&](edge_index idx) {
            return active_[edges_[idx].traverse(v)];
        }, [&](edge_index) { ++count; });
        return count;
    }

    // ── active-aware queries ──

    bool is_active(node_index v) const { return active_[v]; }
    index_t num_active() const { return num_active_; }

    int active_degree(node_index v) const {
        int deg = 0;
        for (auto e : neighbors(v))
            if (active_[e.to]) ++deg;
        return deg;
    }

    template<typename F>
    void for_active_neighbors(node_index v, F&& cb) const {
        for (auto e : neighbors(v)) {
            if (active_[e.to])
                cb(e.to, e.w);
        }
    }

private:
    index_t n_ = 0;
    index_t m_ = 0;
    index_t num_active_ = 0;
    std::vector<edge> edges_;
    Incidence adj_;
    std::vector<char> active_;
};

} // namespace apxchol
