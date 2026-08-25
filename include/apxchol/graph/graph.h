#pragma once
#include "apxchol/graph/incidence_list.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <ranges>
#include <span>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace apxchol {

namespace detail {
template<incidence_storage Incidence>
struct residual_coalescer;
}

// Residual-pool edge weight storage type. fp32 under -DAPXCHOL_POOL_FP32 (ON by
// default; pass =OFF for fp64). The residual graph is a MULTIGRAPH: add_edge/write_edge_at
// APPEND new edges and weights are never re-summed in place (the per-vertex
// dedup/weighted-degree aggregation is done transiently in fp64 at read), so each
// stored weight is a TERMINAL write -- fp32 is safe (verified: PCG iters/residual
// unchanged across IPM incl. high-DR, social, grid). fp32 halves the 16 B edge
// {node_index + double, alignment-padded} to 8 B {node_index + float}, ~halving
// edges_ (the largest setup-transient array) with no convergence cost. The
// main elimination remains append-only: maintaining one physical edge per
// active endpoint pair would require a concurrent dynamic pair map, while
// physically merging only the pivot's adjacency was measured slower than the
// current transient fp64 aggregation in process_vertex(). A governed one-shot
// rebuild may coalesce the late residual; it stores multiplicity separately so
// the partitioner's degree heuristic remains the same.
/// Mutable incidence-list graph.
///
/// The traditional backends store each undirected edge {u,v,w} once in a flat
/// pool with XOR-encoded endpoints (u^v); both endpoints store its edge index.
/// directed_vec_pool_incidence instead duplicates {neighbor, weight} inline at
/// both endpoints, avoiding the indexed pool read during adjacency scans.
/// The Incidence template parameter controls how per-vertex lists are stored.
/// The traditional implementations contain edge indices (vec_incidence,
/// forward_star_incidence and bstr_incidence); directed_vec_pool_incidence
/// contains the inline records themselves.
template<incidence_storage Incidence = vec_pool_incidence>
class graph {
public:
    /// Expose the incidence backend for callers that want to specialize
    /// (e.g. parallel make_graph picks the vec_pool fast path via this).
    using incidence_type = Incidence;
    static constexpr bool stores_directed_incidence = [] {
        if constexpr (requires { typename Incidence::value_type; })
            return std::same_as<typename Incidence::value_type,
                                directed_pool_edge>;
        else
            return false;
    }();

    struct edge {
        node_index to; pool_value_t w;   // w fp32 under -DAPXCHOL_POOL_FP32
    private:
        friend graph;
        node_index traverse(node_index from) const { return to ^ from; }
    };

    graph() = default;

    explicit graph(node_index n)
        : n_(n), m_(0), num_active_(n),
          active_(active_word_count(n), ~active_word{0}),
          excess_(n, 0.0) {
        adj_.init(n);
    }

    node_index n() const { return n_; }   // vertex count
    /// Edges EVER added (can exceed 2^31). MONOTONE: elimination prunes edges
    /// but never decrements this, so mid-elimination `m()` is not the live edge
    /// count and `2*m()/active` is not the live average degree — it is inflated
    /// by every edge the eliminated prefix consumed. Anything needing a live
    /// degree must measure one (prune_and_degrees) or be handed one.
    edge_index m() const { return m_; }

    // Yields {to, w} with w PROMOTED to double, so all consumers (elimination,
    // weighted-degree, conversions) compute in fp64 regardless of pool storage.
    auto neighbors(node_index v) const {
        if constexpr (stores_directed_incidence) {
            return adj_[v] | std::views::transform(
                [](const directed_pool_edge& i) {
                    struct nbr { node_index to; double w; };
                    return nbr{i.to, static_cast<double>(i.w)};
                });
        } else {
            return adj_[v] | std::views::transform(
                [this, v](edge_index i) {
                    struct nbr { node_index to; double w; };
                    return nbr{edges_[i].traverse(v),
                               static_cast<double>(edges_[i].w)};
                });
        }
    }

    // ── mutation ──

    void add_edge(node_index u, node_index v, double w) {
        if constexpr (stores_directed_incidence) {
            adj_.push(u, {v, static_cast<pool_value_t>(w)});
            adj_.push(v, {u, static_cast<pool_value_t>(w)});
        } else {
            auto idx = static_cast<edge_index>(edges_.size());
            edges_.push_back({u ^ v, static_cast<pool_value_t>(w)});
            if (!multiplicity_.empty()) multiplicity_.push_back(1);
            adj_.push(u, idx);
            adj_.push(v, idx);
        }
        ++m_;
    }

    void deactivate(node_index v) {
        if (!clear_active_bit(v)) return;
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
        static_assert(!stores_directed_incidence,
                      "directed incidences must be inserted directly");
        const edge_index start = static_cast<edge_index>(edges_.size());
        edges_.resize(start + N);
        if (!multiplicity_.empty()) multiplicity_.resize(start + N, 1);
        m_ += N;  // bookkeeping; queried via m()
        return start;
    }

    /// Directed-inline vec_pool: account for a bulk set of undirected edges
    /// whose two incidences are written directly, without an edge-pool stage.
    void record_edges_added(edge_index count) {
        static_assert(stores_directed_incidence);
        m_ += count;
    }

    /// Directed-inline vec_pool: materialize one endpoint incidence directly
    /// into a slab already sized by adj_bulk_reserve_parallel().
    void adj_atomic_push_directed(node_index from, node_index to, double w) {
        static_assert(stores_directed_incidence);
        adj_.atomic_push_reserved(
            from, directed_pool_edge{to, static_cast<pool_value_t>(w)});
    }

    /// Directed-inline vec_pool: write an incidence at a caller-assigned
    /// offset in the pre-reserved suffix without an atomic slot claim.
    void adj_write_reserved_directed_at(node_index from, node_index offset,
                                        node_index to, double w) {
        static_assert(stores_directed_incidence);
        adj_.write_reserved_at(
            from, offset,
            directed_pool_edge{to, static_cast<pool_value_t>(w)});
    }

    /// Publish all preassigned suffix slots for one vec_pool vertex.
    template<typename I = Incidence>
        requires is_vec_pool_incidence_v<I>
    void adj_commit_reserved_directed(node_index v, node_index count) {
        static_assert(stores_directed_incidence);
        adj_.commit_reserved(v, count);
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
        if (!multiplicity_.empty())
            multiplicity_.resize(edges_pool_base_ + N, 1);
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
        if (!multiplicity_.empty())
            multiplicity_.resize(edges_pool_base_ + claimed);
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

    /// Write edge data at a pre-allocated pool slot.  Lock-free; safe
    /// to call concurrently from different threads on different slots.
    void write_edge_at(edge_index slot, node_index u, node_index v, double w) {
        edges_[slot] = {u ^ v, static_cast<pool_value_t>(w)};
        if (!multiplicity_.empty()) multiplicity_[slot] = 1;
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
        requires is_vec_pool_incidence_v<I>
    void adj_reserve_for(node_index v, node_index need) {
        adj_.reserve_for(v, need);
    }

    /// vec_pool only: bulk parallel reserve_for. Must be called from
    /// inside an OpenMP parallel region; implementation has its own
    /// single + for split.
    template<typename I = Incidence, typename TouchedIter>
        requires is_vec_pool_incidence_v<I>
    void adj_bulk_reserve_parallel(TouchedIter begin, TouchedIter end,
                                   const std::vector<node_index>& incoming) {
        adj_.bulk_reserve_parallel(begin, end, incoming);
    }

    /// vec_pool only: current adjacency count for vertex v (used for
    /// reserve_for pre-pass).
    template<typename I = Incidence>
        requires is_vec_pool_incidence_v<I>
    node_index adj_count(node_index v) const {
        return static_cast<node_index>(adj_[v].size());
    }

    /// vec_pool only: lock-free push into v's pre-reserved slab.  Caller
    /// must have called adj_reserve_for to ensure cap > current_count + N
    /// for the parallel section.
    template<typename I = Incidence>
        requires is_vec_pool_incidence_v<I>
    void adj_atomic_push_reserved(node_index v, edge_index e_slot) {
        static_assert(!stores_directed_incidence,
                      "directed incidences must be inserted directly");
        adj_.atomic_push_reserved(v, e_slot);
    }

    /// Sort vertex v's adjacency slab — by edge index for the indexed backend,
    /// or by (target, weight) for the directed-inline backend — to restore
    /// deterministic ordering after a parallel atomic-push phase.
    template<typename I = Incidence>
        requires is_vec_pool_incidence_v<I>
    void adj_sort_slab(node_index v) {
        adj_.sort_slab(v);
    }

    /// vec_pool only: compact pool to remove abandoned slabs from doubling.
    /// Reduces memory; rebuilds with tight cap=count for all vertices.
    template<typename I = Incidence>
        requires is_vec_pool_incidence_v<I>
    void compact_adj() { adj_.compact(); }

    /// vec_pool only: live fraction (1 - abandoned/pool_size).
    template<typename I = Incidence>
        requires is_vec_pool_incidence_v<I>
    double adj_live_fraction() const { return adj_.live_fraction(); }

    /// vec_pool only: pool diagnostics (grow/compact counts, worst fragmentation).
    template<typename I = Incidence>
        requires is_vec_pool_incidence_v<I>
    size_t adj_grow_count() const { return adj_.grow_count(); }
    template<typename I = Incidence>
        requires is_vec_pool_incidence_v<I>
    size_t adj_compact_count() const { return adj_.compact_count(); }
    template<typename I = Incidence>
        requires is_vec_pool_incidence_v<I>
    double adj_min_live_fraction() const { return adj_.min_live_fraction(); }
    template<typename I = Incidence>
        requires is_vec_pool_incidence_v<I>
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
        const bool was_active = clear_active_bit(v);
        assert(was_active);
        (void)was_active;
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
        adj_.filter(v, [&](const auto& idx) {
            auto u = edge_target(idx, v);
            if (!is_active(u)) return false;
            visit(u);
            count += edge_multiplicity(idx);
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

    bool is_active(node_index v) const {
        const active_word word = __atomic_load_n(
            &active_[static_cast<size_t>(v) / kActiveWordBits],
            __ATOMIC_RELAXED);
        return (word & active_mask(v)) != 0;
    }
    node_index num_active() const { return num_active_; }

    /// Raw access to adjacency list and edge pool (for tight inner loops).
    auto adj(node_index v) const { return adj_[v]; }
    template<typename Entry>
    node_index edge_target(const Entry& idx, node_index from) const {
        if constexpr (std::same_as<std::remove_cvref_t<Entry>,
                                   directed_pool_edge>)
            return idx.to;
        else
            return edges_[static_cast<edge_index>(idx)].traverse(from);
    }
    template<typename Entry>
    double edge_weight(const Entry& idx) const {
        if constexpr (std::same_as<std::remove_cvref_t<Entry>,
                                   directed_pool_edge>)
            return idx.w;
        else
            return edges_[static_cast<edge_index>(idx)].w;
    }
    template<typename Entry>
    node_index edge_multiplicity(const Entry& idx) const {
        if constexpr (std::same_as<std::remove_cvref_t<Entry>,
                                   directed_pool_edge>)
            return 1;
        else {
            const edge_index edge_id = static_cast<edge_index>(idx);
            return multiplicity_.empty() ? node_index{1}
                                         : multiplicity_[edge_id];
        }
    }

    /// Per-vertex excess diagonal (for SDDM matrices).
    /// For a pure Laplacian this is zero everywhere.
    /// For SDDM,  excess[v] = M[v,v] - sum of incident edge weights.
    double  excess(node_index v) const { return excess_[v]; }
    double& excess(node_index v)       { return excess_[v]; }

    /// Approximate heap memory usage in bytes.
    std::size_t memory_bytes() const {
        return edges_.capacity() * sizeof(edge)
             + multiplicity_.capacity() * sizeof(node_index)
             + adj_.memory_bytes()
             + active_.capacity() * sizeof(active_word)
             + excess_.capacity() * sizeof(double);
    }

private:
    template<incidence_storage>
    friend struct detail::residual_coalescer;

    struct coalesce_stats {
        std::size_t multi_edges = 0;
        std::size_t distinct_edges = 0;
        std::size_t bytes_before = 0;
        std::size_t bytes_after = 0;
    };

    /// Estimate active multiedge / endpoint-pair ratio from evenly spaced
    /// active vertices. This cheap guard avoids rebuilding low-duplication
    /// residuals merely to discover that compaction cannot repay its pass.
    template<typename I = Incidence>
        requires std::same_as<I, vec_pool_incidence>
    double estimate_active_duplicate_ratio(
        std::span<const node_index> active,
        std::size_t sample_vertices = 256) const {
        const std::size_t samples =
            std::min(sample_vertices, active.size());
        if (samples == 0) return 1.0;
        unsigned long long multi_total = 0;
        unsigned long long distinct_total = 0;
        #pragma omp parallel reduction(+:multi_total, distinct_total)
        {
            std::vector<node_index> targets;
            #pragma omp for schedule(static)
            for (std::size_t s = 0; s < samples; ++s) {
                const node_index u = active[s * active.size() / samples];
                targets.clear();
                unsigned long long local_multi = 0;
                for (const edge_index idx : adj_[u]) {
                    const node_index v = edges_[idx].traverse(u);
                    if (!is_active(v)) continue;
                    targets.push_back(v);
                    local_multi += edge_multiplicity(idx);
                }
                std::ranges::sort(targets);
                const auto unique_end = std::ranges::unique(targets).begin();
                multi_total += local_multi;
                distinct_total +=
                    static_cast<unsigned long long>(unique_end - targets.begin());
            }
        }
        return distinct_total
            ? static_cast<double>(multi_total) / distinct_total
            : 1.0;
    }

    /// Rebuild the active vec_pool residual with one physical edge per
    /// endpoint pair. The multiplicity sidecar keeps prune_and_degree()
    /// exactly faithful to the old multigraph while process_vertex() sees the
    /// same summed numerical weight. This is a one-shot late-residual rebuild,
    /// not a dynamic pair map.
    template<typename I = Incidence>
        requires std::same_as<I, vec_pool_incidence>
    coalesce_stats coalesce_active(std::span<const node_index> active) {
        struct pending_edge {
            node_index u, v;
            double weight;
            node_index multiplicity;
        };
        struct local_neighbor {
            node_index v;
            double weight;
            node_index multiplicity;
        };

        int num_threads = 1;
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#endif
        std::vector<std::vector<pending_edge>> pending(
            static_cast<std::size_t>(num_threads));

        #pragma omp parallel num_threads(num_threads)
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            auto& out = pending[static_cast<std::size_t>(tid)];
            std::vector<local_neighbor> neighbors;
            #pragma omp for schedule(static)
            for (std::size_t k = 0; k < active.size(); ++k) {
                const node_index u = active[k];
                neighbors.clear();
                for (const edge_index idx : adj_[u]) {
                    const node_index v = edges_[idx].traverse(u);
                    if (!is_active(v) || v <= u) continue;
                    neighbors.push_back({v, static_cast<double>(edges_[idx].w),
                                         edge_multiplicity(idx)});
                }
                std::ranges::sort(neighbors, {}, &local_neighbor::v);
                for (std::size_t i = 0; i < neighbors.size();) {
                    const node_index v = neighbors[i].v;
                    double weight = 0.0;
                    node_index multiplicity = 0;
                    do {
                        weight += neighbors[i].weight;
                        multiplicity += neighbors[i].multiplicity;
                        ++i;
                    } while (i < neighbors.size() && neighbors[i].v == v);
                    out.push_back({u, v, weight, multiplicity});
                }
            }
        }

        coalesce_stats stats;
        stats.bytes_before = memory_bytes();
        for (const auto& list : pending) {
            stats.distinct_edges += list.size();
            for (const auto& edge : list)
                stats.multi_edges += edge.multiplicity;
        }

        graph rebuilt(n_);
        rebuilt.active_ = active_;
        rebuilt.num_active_ = num_active_;
        rebuilt.excess_ = excess_;
        rebuilt.edges_.reserve(stats.distinct_edges);
        rebuilt.multiplicity_.reserve(stats.distinct_edges);

        std::vector<node_index> physical_degree(static_cast<std::size_t>(n_), 0);
        for (const auto& list : pending) {
            for (const auto& edge : list) {
                ++physical_degree[edge.u];
                ++physical_degree[edge.v];
            }
        }
        for (const node_index v : active)
            rebuilt.adj_.reserve_for(v, physical_degree[v]);

        for (const auto& list : pending) {
            for (const auto& edge : list) {
                const edge_index idx = static_cast<edge_index>(rebuilt.edges_.size());
                rebuilt.edges_.push_back(
                    {edge.u ^ edge.v, static_cast<pool_value_t>(edge.weight)});
                rebuilt.multiplicity_.push_back(edge.multiplicity);
                rebuilt.adj_.push(edge.u, idx);
                rebuilt.adj_.push(edge.v, idx);
                ++rebuilt.m_;
            }
        }
        stats.bytes_after = rebuilt.memory_bytes();
        *this = std::move(rebuilt);
        return stats;
    }


private:
    using active_word = std::uint64_t;
    static constexpr size_t kActiveWordBits = 8 * sizeof(active_word);

    static constexpr size_t active_word_count(node_index n) {
        return (static_cast<size_t>(n) + kActiveWordBits - 1) /
               kActiveWordBits;
    }

    static constexpr active_word active_mask(node_index v) {
        return active_word{1} <<
               (static_cast<size_t>(v) % kActiveWordBits);
    }

    bool clear_active_bit(node_index v) {
        const active_word mask = active_mask(v);
        const active_word old = __atomic_fetch_and(
            &active_[static_cast<size_t>(v) / kActiveWordBits], ~mask,
            __ATOMIC_RELAXED);
        return (old & mask) != 0;
    }

    node_index n_ = 0;          // vertex count
    edge_index m_ = 0;          // edge count (can exceed 2^31)
    node_index num_active_ = 0; // active vertex count
    std::vector<edge> edges_;
    // Empty until an exact coalesced-residual rebuild. Thereafter one count per
    // physical edge; newly sampled edges use count 1.
    std::vector<node_index> multiplicity_;
    Incidence adj_;
    // Active status is probed randomly in nearly every residual-edge scan, so
    // keep one bit per vertex rather than one byte.  Parallel elimination can
    // clear distinct vertices that share a word; all reads and clears therefore
    // use relaxed atomic operations.  Round barriers provide the phase ordering
    // needed by later graph traversals.
    std::vector<active_word> active_;
    std::vector<double> excess_;
    edge_index edges_pool_base_  = 0;
    edge_index edges_pool_count_ = 0;
};

namespace detail {

/// Internal access seam for the one-shot late-residual rebuild. Keeping these
/// operations here avoids growing graph's public API for one factorizer policy.
template<incidence_storage Incidence>
struct residual_coalescer {
    static double estimate(const graph<Incidence>& g,
                           std::span<const node_index> active,
                           std::size_t sample_vertices = 256) {
        return g.estimate_active_duplicate_ratio(active, sample_vertices);
    }

    static auto rebuild(graph<Incidence>& g,
                        std::span<const node_index> active) {
        return g.coalesce_active(active);
    }
};

} // namespace detail

} // namespace apxchol
