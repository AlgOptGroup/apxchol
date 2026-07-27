#pragma once
#include "apxchol/graph/incidence_list.h"
#include <ranges>
#include <vector>

namespace apxchol {

// Residual-pool edge weight storage type. fp32 under -DAPXCHOL_POOL_FP32 (ON by
// default; pass =OFF for fp64). The residual graph is a MULTIGRAPH: add_edge/write_edge_at
// APPEND new edges and weights are never re-summed in place (the per-vertex
// dedup/weighted-degree aggregation is done transiently in fp64 at read), so each
// stored weight is a TERMINAL write -- fp32 is safe (verified: PCG iters/residual
// unchanged across IPM incl. high-DR, social, grid). fp32 halves the 16 B edge
// {node_index + double, alignment-padded} to 8 B {node_index + float}, ~halving
// edges_ (the largest setup-transient array) with no convergence cost. This is
// faithful to Kyng-Sachdeva (keep multi-edges, don't merge): merging would also
// save memory but raises max leverage -> worse conditioning -> more PCG iters,
// whereas fp32-multigraph keeps the conditioning and just narrows the bytes.
#ifdef APXCHOL_POOL_FP32
using pool_value_t = float;
#else
using pool_value_t = double;
#endif

/// Mutable incidence-list graph with a flat edge pool.
///
/// Each undirected edge {u,v,w} is stored once in the pool with XOR-encoded
/// endpoints (u^v).  Both u and v reference the same pool entry;
/// the target is recovered by XORing the stored value with the source vertex.
/// The Incidence template parameter controls how per-vertex lists of
/// edge indices are stored (vec_incidence, forward_star_incidence,
/// bstr_incidence, or any user-provided type with the same interface).
template<incidence_storage Incidence = vec_pool_incidence>
class graph {
public:
    /// Expose the incidence backend for callers that want to specialize
    /// (e.g. parallel make_graph picks the vec_pool fast path via this).
    using incidence_type = Incidence;

    struct edge {
        node_index to; pool_value_t w;   // w fp32 under -DAPXCHOL_POOL_FP32
    private:
        friend graph;
        node_index traverse(node_index from) const { return to ^ from; }
    };

    graph() = default;

    explicit graph(node_index n)
        : n_(n), m_(0), num_active_(n), active_(n, true), excess_(n, 0.0) {
        adj_.init(n);
    }

    node_index n() const { return n_; }   // vertex count
    edge_index m() const { return m_; }   // edge count (can exceed 2^31)

    // Yields {to, w} with w PROMOTED to double, so all consumers (elimination,
    // weighted-degree, conversions) compute in fp64 regardless of pool storage.
    auto neighbors(node_index v) const {
        return adj_[v] | std::views::transform(
            [this, v](edge_index i) {
                struct nbr { node_index to; double w; };
                return nbr{edges_[i].traverse(v), static_cast<double>(edges_[i].w)};
            });
    }

    // ── mutation ──

    void add_edge(node_index u, node_index v, double w) {
        auto idx = static_cast<edge_index>(edges_.size());
        edges_.push_back({u ^ v, static_cast<pool_value_t>(w)});
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
    edge_index reserve_edge_pool(edge_index N) {
        auto start = static_cast<edge_index>(edges_.size());
        edges_.resize(start + N);
        m_ += N;  // bookkeeping; queried via m()
        return start;
    }

    /// Pre-reserve N additional edge slots for parallel atomic claim.
    /// Grows edges_ size by N (default-initialised — cheap, trivial type)
    /// and resets the per-section claim counter. Must precede any
    /// claim_edge_slot() calls. finalize_edge_pool() trims and
    /// updates m_ after the parallel section completes.
    void reserve_edge_pool_atomic(edge_index N) {
        edges_pool_base_  = static_cast<edge_index>(edges_.size());
        edges_pool_count_ = 0;
        edges_.resize(edges_pool_base_ + N);
    }

    /// Atomic-claim a single slot from the pre-reserved range.
    /// Returns an absolute edge_index in [base, base + N).
    /// Thread-safe via __atomic_fetch_add.
    edge_index claim_edge_slot() {
        const edge_index off = __atomic_fetch_add(
            &edges_pool_count_, 1, __ATOMIC_RELAXED);
        return edges_pool_base_ + off;
    }

    /// Trim edges_ to the actual claim count and update m_. Called
    /// once per parallel section, after all claim_edge_slot() callers
    /// have finished.
    void finalize_edge_pool() {
        const edge_index claimed = edges_pool_count_;
        edges_.resize(edges_pool_base_ + claimed);
        m_ += claimed;
    }

    /// Reserve `extra` adjacency-list slots (forward_star only).
    template<typename I = Incidence>
        requires std::same_as<I, forward_star_incidence>
    edge_index reserve_adj_pool(edge_index extra) {
        return adj_.reserve_pool(extra);
    }

    /// forward_star only: pre-reserve N adj-pool slots and arm an
    /// atomic counter that adj_push_inline draws from.
    template<typename I = Incidence>
        requires std::same_as<I, forward_star_incidence>
    void reserve_adj_pool_atomic(edge_index N) {
        adj_.reserve_pool_atomic(N);
    }

    /// Push edge slot e_slot onto v's adjacency. Backend-dispatched:
    ///   forward_star: atomic adj-pool slot claim + push_atomic on head_[v].
    ///                 Tolerates concurrent inter-thread pushes to same v.
    ///                 Caller MUST have called reserve_adj_pool_atomic.
    ///   vec/bstr: adj_[v].push_back(e_slot). Caller's contract:
    ///                  no other thread writes to adj_[v] (UB if violated).
    void adj_push_inline(node_index v, edge_index e_slot) {
        if constexpr (std::same_as<Incidence, forward_star_incidence>) {
            const edge_index a_slot = adj_.claim_pool_slot();
            adj_.push_atomic(v, a_slot, e_slot);
        } else {
            adj_.push(v, e_slot);
        }
    }

    /// True if an inline adj push to v will NOT reallocate shared storage.
    /// Only vec_pool uses a shared pool arena whose grow() reallocs it under
    /// sibling threads (heap-use-after-free in the parallel multi-vertex
    /// section); for it, report spare slab capacity so a full slab can defer
    /// instead. Other backends never realloc shared state on an inline push:
    /// forward_star draws from a pre-reserved atomic pool, and vec/bstr
    /// use independent per-vertex containers (no cross-thread shared arena).
    bool adj_has_inline_capacity(node_index v) const {
        if constexpr (std::same_as<Incidence, vec_pool_incidence>)
            return adj_.has_inline_capacity(v);
        else
            return true;
    }

    /// Write edge data at a pre-allocated pool slot.  Lock-free; safe
    /// to call concurrently from different threads on different slots.
    void write_edge_at(edge_index slot, node_index u, node_index v, double w) {
        edges_[slot] = {u ^ v, static_cast<pool_value_t>(w)};
    }

    /// Atomically prepend `e_slot` to vertex v's adjacency chain at
    /// pre-allocated adjacency slot `a_slot` (forward_star only).
    template<typename I = Incidence>
        requires std::same_as<I, forward_star_incidence>
    void adj_push_atomic(node_index v, edge_index a_slot, edge_index e_slot) {
        adj_.push_atomic(v, a_slot, e_slot);
    }

    /// Compact adjacency pool (forward_star only).  Walks each chain,
    /// re-emits survivors into a fresh contiguous buffer.  Restores
    /// cache-friendly traversal after long sequences of filter() calls
    /// have fragmented the chains.
    template<typename I = Incidence>
        requires std::same_as<I, forward_star_incidence>
    void compact_adj() { adj_.compact(); }

    /// vec_pool only: pre-reserve cap_[v] >= need before a parallel
    /// adj_atomic_push_reserved phase.  NOT thread-safe.
    template<typename I = Incidence>
        requires std::same_as<I, vec_pool_incidence>
    void adj_reserve_for(node_index v, node_index need) {
        adj_.reserve_for(v, need);
    }

    /// vec_pool only: bulk parallel reserve_for. Must be called from
    /// inside an OpenMP parallel region; implementation has its own
    /// single + for split.
    template<typename I = Incidence, typename TouchedIter>
        requires std::same_as<I, vec_pool_incidence>
    void adj_bulk_reserve_parallel(TouchedIter begin, TouchedIter end,
                                   const std::vector<node_index>& incoming) {
        adj_.bulk_reserve_parallel(begin, end, incoming);
    }

    /// vec_pool only: current adjacency count for vertex v (used for
    /// reserve_for pre-pass).
    template<typename I = Incidence>
        requires std::same_as<I, vec_pool_incidence>
    node_index adj_count(node_index v) const {
        return static_cast<node_index>(adj_[v].size());
    }

    /// vec_pool only: lock-free push into v's pre-reserved slab.  Caller
    /// must have called adj_reserve_for to ensure cap > current_count + N
    /// for the parallel section.
    template<typename I = Incidence>
        requires std::same_as<I, vec_pool_incidence>
    void adj_atomic_push_reserved(node_index v, edge_index e_slot) {
        adj_.atomic_push_reserved(v, e_slot);
    }

    /// Sort vertex v's adj slab in increasing edge_index order — used to
    /// restore deterministic ordering after a parallel atomic-push phase.
    template<typename I = Incidence>
        requires std::same_as<I, vec_pool_incidence>
    void adj_sort_slab(node_index v) {
        adj_.sort_slab(v);
    }

    /// vec_pool only: compact pool to remove abandoned slabs from doubling.
    /// Reduces memory; rebuilds with tight cap=count for all vertices.
    template<typename I = Incidence>
        requires std::same_as<I, vec_pool_incidence>
    void compact_adj() { adj_.compact(); }

    /// vec_pool only: live fraction (1 - abandoned/pool_size).
    template<typename I = Incidence>
        requires std::same_as<I, vec_pool_incidence>
    double adj_live_fraction() const { return adj_.live_fraction(); }

    /// vec_pool only: pool diagnostics (grow/compact counts, worst fragmentation).
    template<typename I = Incidence>
        requires std::same_as<I, vec_pool_incidence>
    size_t adj_grow_count() const { return adj_.grow_count(); }
    template<typename I = Incidence>
        requires std::same_as<I, vec_pool_incidence>
    size_t adj_compact_count() const { return adj_.compact_count(); }
    template<typename I = Incidence>
        requires std::same_as<I, vec_pool_incidence>
    double adj_min_live_fraction() const { return adj_.min_live_fraction(); }
    template<typename I = Incidence>
        requires std::same_as<I, vec_pool_incidence>
    void adj_note_live_fraction() { adj_.note_live_fraction(); }

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

    template<typename I = Incidence>
        requires std::same_as<I, forward_star_incidence>
    bool adj_filter_append_enabled() const { return adj_.filter_append(); }

    /// Pre-size the adjacency pool for a parallel-append phase.  Every
    /// filter() call inside the parallel region will CAS-reserve its
    /// survivor slot via __atomic_fetch_add, avoiding push_back races.
    /// Must be paired with end_parallel_append_adj() after the region.
    /// Forward_star + filter_append mode only.
    template<typename I = Incidence>
        requires std::same_as<I, forward_star_incidence>
    void begin_parallel_append_adj() {
        adj_.begin_parallel_append(adj_.live_count());
    }

    template<typename I = Incidence>
        requires std::same_as<I, forward_star_incidence>
    void end_parallel_append_adj() { adj_.end_parallel_append(); }

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
    void bulk_decrement_active(node_index k) { num_active_ -= k; }

    /// Prune dead edges and (optionally) visit each survivor.  Returns
    /// the surviving (active) degree count.  The default no-op visitor
    /// makes this serve as a pure degree count too.
    template<typename Visitor>
    node_index prune_and_visit(node_index v, Visitor&& visit) {
        node_index count = 0;
        adj_.filter(v, [&](edge_index idx) {
            auto u = edges_[idx].traverse(v);
            if (!active_[u]) return false;
            visit(u);
            ++count;
            return true;
        });
        return count;
    }

    node_index prune_and_visit(node_index v) {
        return prune_and_visit(v, [](node_index){});
    }

    /// Convenience wrapper — prune + count, no visitor.
    node_index prune_and_degree(node_index v) {
        return prune_and_visit(v);
    }

    // ── active-aware queries ──

    bool is_active(node_index v) const { return active_[v]; }
    node_index num_active() const { return num_active_; }

    /// Raw access to adjacency list and edge pool (for tight inner loops).
    auto adj(node_index v) const { return adj_[v]; }
    node_index edge_target(edge_index idx, node_index from) const {
        return edges_[idx].traverse(from);
    }
    double edge_weight(edge_index idx) const { return edges_[idx].w; }
    node_index other_endpoint(node_index v, edge_index idx) const {
        return edges_[idx].traverse(v);
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
    node_index n_ = 0;          // vertex count
    edge_index m_ = 0;          // edge count (can exceed 2^31)
    node_index num_active_ = 0; // active vertex count
    std::vector<edge> edges_;
    Incidence adj_;
    std::vector<char> active_;
    std::vector<double> excess_;
    edge_index edges_pool_base_  = 0;
    edge_index edges_pool_count_ = 0;
};

} // namespace apxchol
