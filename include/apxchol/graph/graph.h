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

    // ── Bulk parallel apply (forward_star fast path) ─────────────
    //
    // Used by eliminate_set to atomically commit per-thread deferred
    // edge buffers, excess updates, and IS-vertex deactivations
    // without serializing on graph mutation.  Only enabled for the
    // forward_star backend, where the underlying linked list supports
    // lock-free CAS prepending.
    //
    // Pre: IS vertices are pairwise non-adjacent and disjoint from
    //      neighbors written via excess/edges → no head_[] conflict
    //      between deactivations and atomic pushes.

    /// Reserve `N` slots in the edge pool; returns starting `edge_index`.
    /// Caller fills slots in parallel.  Used together with `reserve_adj_pool`
    /// and `link_edge_atomic` for lock-free bulk insertion.
    edge_index reserve_edge_pool(index_t N) {
        auto start = static_cast<edge_index>(edges_.size());
        edges_.resize(start + N);
        m_ += N;  // bookkeeping; queried via m()
        return start;
    }

    /// Reserve `extra` adjacency-list slots (forward_star only).
    template<typename I = Incidence>
        requires std::same_as<I, forward_star_incidence>
    index_t reserve_adj_pool(index_t extra) {
        return adj_.reserve_pool(extra);
    }

    /// Write edge data at a pre-allocated pool slot.  Lock-free; safe
    /// to call concurrently from different threads on different slots.
    void write_edge_at(edge_index slot, node_index u, node_index v, double w) {
        edges_[slot] = {u ^ v, w};
    }

    /// Atomically prepend `e_slot` to vertex v's adjacency chain at
    /// pre-allocated adjacency slot `a_slot` (forward_star only).
    template<typename I = Incidence>
        requires std::same_as<I, forward_star_incidence>
    void adj_push_atomic(node_index v, index_t a_slot, edge_index e_slot) {
        adj_.push_atomic(v, a_slot, e_slot);
    }

    /// Compact adjacency pool (forward_star only).  Walks each chain,
    /// re-emits survivors into a fresh contiguous buffer.  Restores
    /// cache-friendly traversal after long sequences of filter() calls
    /// have fragmented the chains.
    template<typename I = Incidence>
        requires std::same_as<I, forward_star_incidence>
    void compact_adj() { adj_.compact(); }

    /// Pool occupancy ratio (forward_star only).  Used as auto-compact trigger.
    template<typename I = Incidence>
        requires std::same_as<I, forward_star_incidence>
    double adj_live_fraction() const { return adj_.live_fraction(); }

    /// Switch forward_star adjacency to append-on-filter mode.  When on,
    /// every filter() call writes survivors contiguously at the end of the
    /// pool (instead of in-place re-link).  Pool grows; reclaim with
    /// compact_adj() periodically.  Forward_star only.
    template<typename I = Incidence>
        requires std::same_as<I, forward_star_incidence>
    void set_adj_filter_append(bool on) { adj_.set_filter_append(on); }

    /// Atomically add `delta` to excess[v]; safe under data races on v.
    void atomic_add_excess(node_index v, double delta) {
        #pragma omp atomic
        excess_[v] += delta;
    }

    /// Mark vertex inactive without decrementing num_active_; the
    /// caller must invoke `bulk_decrement_active(k)` once after the
    /// parallel batch.  Adjacency chain is cleared (head reset to npos).
    void set_inactive_unchecked(node_index v) {
        active_[v] = false;
        adj_.clear(v);
    }

    /// Apply a bulk decrement to `num_active_` after a parallel batch
    /// of `set_inactive_unchecked` calls.
    void bulk_decrement_active(index_t k) { num_active_ -= k; }

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

    /// Prune dead edges and call visitor(target_vertex) for each survivor.
    /// Returns degree (number of surviving edges).
    template<typename Visitor>
    index_t prune_and_visit(node_index v, Visitor&& visit) {
        index_t count = 0;
        adj_.filter(v, [&](edge_index idx) {
            auto u = edges_[idx].traverse(v);
            if (!active_[u]) return false;
            visit(u);
            ++count;
            return true;
        });
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
