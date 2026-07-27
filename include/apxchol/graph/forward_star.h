#pragma once
/// Chained forward star — flat-array linked adjacency list.
///
/// Each vertex has a singly-linked chain of nodes threaded through
/// a flat pool.  Data and link are co-located in each node for
/// cache-friendly traversal.
/// push(v, data) prepends a node to v's chain in O(1) — no copies.
/// Iteration over v's chain is O(degree(v)).
/// No per-element erasure — use clear(v) to disconnect all elements.
///
/// Index roles (see types.h): a *vertex id* indexes head_ and is `node_index`
/// (node_index). A *pool offset* indexes nodes_ -- the chain links (`next`,
/// head_ values, npos, claimed slots) -- and is `edge_index`, since the pool
/// holds one node per live/fill edge and can exceed 2^31 on dense graphs.

#include "apxchol/types.h"
#include "apxchol/big_alloc.h"
#include <cstddef>
#include <iterator>
#include <limits>
#include <ranges>
#include <span>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace apxchol {

template<typename T>
struct forward_star {
    // npos is a POOL offset sentinel (slot 0 is reserved), hence edge_index.
    static constexpr edge_index npos = 0;
    static constexpr graph_storage tag = graph_storage::forward_star;
    // Pool offsets must stay <= this; exceeding it would wrap a chain link.
    static constexpr edge_index kPoolMax = std::numeric_limits<edge_index>::max();

    struct node { T data; edge_index next; };

    forward_star() = default;

    void init(node_index n) {
        head_.assign(n, npos);
        nodes_.resize(1);
    }

    void push(node_index v, const T& val) {
        nodes_.push_back({val, head_[v]});
        if (nodes_.size() > static_cast<std::size_t>(kPoolMax))
            edge_index_overflow("forward_star::push");
        head_[v] = static_cast<edge_index>(nodes_.size()) - 1;
    }

    void clear(node_index v) { head_[v] = npos; }

    // ── Lock-free batched insertion ───────────────────────────
    //
    // Two-step parallel push for use after a serial pre-allocation:
    //   1. Caller invokes `reserve_pool(N)` to grow `nodes_` by N
    //      slots and obtain the starting index.
    //   2. Multiple threads call `push_atomic(v, slot, data)` with
    //      distinct `slot` values from that range, prepending each
    //      pre-allocated node to head_[v] via a CAS loop.
    // After all atomic pushes complete, head_[v] points at the most
    // recent insertion (LIFO order; non-deterministic across runs).

    /// Reserve `extra` slots in the node pool; returns starting index.
    /// The pool grows once, so subsequent `push_atomic` calls do not
    /// race on `nodes_` resizing.
    edge_index reserve_pool(edge_index extra) {
        const std::size_t start = nodes_.size();
        if (start + extra > static_cast<std::size_t>(kPoolMax))
            edge_index_overflow("forward_star::reserve_pool");
        nodes_.resize(start + extra);
        return static_cast<edge_index>(start);
    }

    /// Reserve `extra` slots and arm an atomic claim counter for
    /// concurrent inline pushes. Pairs with claim_pool_slot().
    void reserve_pool_atomic(edge_index extra) {
        atomic_pool_base_  = reserve_pool(extra);
        atomic_pool_count_ = 0;
    }

    /// Atomic-claim a slot from the pre-reserved range. Thread-safe.
    edge_index claim_pool_slot() {
        const edge_index off = __atomic_fetch_add(
            &atomic_pool_count_, 1, __ATOMIC_RELAXED);
        return atomic_pool_base_ + off;
    }

    /// Lock-free prepend of the pre-allocated node at `slot` to
    /// vertex v's chain.  Multiple threads may call this concurrently
    /// for the same v with distinct slots; the CAS retries on conflict.
    /// Pre: nodes_[slot] is in the range returned by `reserve_pool`.
    void push_atomic(node_index v, edge_index slot, const T& data) {
        nodes_[slot].data = data;
        edge_index old = __atomic_load_n(&head_[v], __ATOMIC_RELAXED);
        do {
            nodes_[slot].next = old;
        } while (!__atomic_compare_exchange_n(
            &head_[v], &old, slot,
            /*weak=*/false, __ATOMIC_RELEASE, __ATOMIC_RELAXED));
    }

    /// Remove chain elements where pred(data) is false.
    /// Calls on_keep(data) for each surviving element.
    ///
    /// Two implementations selected by `set_filter_append`:
    ///  * in-place (default): re-link around dead nodes.  O(L), no allocation.
    ///    Survivors stay where they are -> chain may fragment after many calls.
    ///  * append: walk the chain, push each survivor at the end of nodes_ as a
    ///    fresh contiguous run, then point head_[v] at it.  Old slots become
    ///    dead (reclaimed by next compact()).  Trades pool growth for
    ///    immediate chain locality.
    void filter(node_index v, auto&& pred, auto&& on_keep) {
        // Append mode inside a parallel region requires a caller-provided
        // begin_parallel_append() to pre-size the pool and expose an atomic
        // free-slot counter.  Without that we can't use push_back safely,
        // so fall back to in-place re-link.
#ifdef _OPENMP
        bool can_append = filter_append_ &&
                          (!omp_in_parallel() || parallel_append_active_);
#else
        bool can_append = filter_append_;
#endif
        if (can_append) {
            filter_append_impl(v, pred, on_keep);
            return;
        }
        // In-place re-link.  After many filter() calls chains fragment
        // across the node pool and each step is a random access.  We
        // prefetch one hop ahead to overlap the pred()/on_keep() work
        // with the next cache-line fetch; this cuts prune time by 3-5x
        // on late-round IS selection where fragmentation peaks.
        edge_index* prev = &head_[v];
        edge_index cur = *prev;
        while (cur != npos) {
            edge_index nxt = nodes_[cur].next;
            if (nxt != npos) __builtin_prefetch(&nodes_[nxt]);
            if (pred(nodes_[cur].data)) {
                on_keep(nodes_[cur].data);
                prev = &nodes_[cur].next;
            } else {
                *prev = nxt;
            }
            cur = nxt;
        }
    }

    void filter(node_index v, auto&& pred) {
        filter(v, pred, [](const T&) {});
    }

    /// Switch filter() to append-on-survive mode (see filter() docs).
    void set_filter_append(bool on) { filter_append_ = on; }
    bool filter_append() const { return filter_append_; }

    // ── Parallel-safe append ─────────────────────────────────
    //
    // begin_parallel_append(ub) resizes the pool by `ub` (upper bound on
    // survivors across all chains) and installs an atomic free-slot
    // pointer.  While active, filter() in append mode allocates slots via
    // __atomic_fetch_add instead of push_back -- parallel-safe, no
    // reallocation races.  Caller MUST invoke end_parallel_append() after
    // the parallel region to shrink the pool back to the actual tail.
    //
    // Total live count across all chains is the exact upper bound; use
    // live_count() (or count_live() internally) to compute it cheaply.
    void begin_parallel_append(edge_index upper_bound) {
        const std::size_t tail = nodes_.size();
        if (tail + upper_bound > static_cast<std::size_t>(kPoolMax))
            edge_index_overflow("forward_star::begin_parallel_append");
        parallel_tail_ = static_cast<edge_index>(tail);
        nodes_.resize(tail + upper_bound);
        parallel_append_active_ = true;
    }

    void end_parallel_append() {
        if (!parallel_append_active_) return;
        nodes_.resize(parallel_tail_);
        parallel_append_active_ = false;
    }

    /// Count of nodes reachable from any head[v] -- exact upper bound for
    /// begin_parallel_append.  O(live edges) serial, but trivially
    /// parallelisable by the caller if needed.
    edge_index live_count() const {
        std::size_t kept = 0;
        for (node_index v = 0; v < static_cast<node_index>(head_.size()); ++v)
            for (edge_index cur = head_[v]; cur != npos; cur = nodes_[cur].next)
                ++kept;
        return static_cast<edge_index>(kept);
    }

  private:
    void filter_append_impl(node_index v, auto&& pred, auto&& on_keep) {
        // Walk old chain, snapshot head, sever it, then re-emit survivors
        // contiguously at the end of nodes_.
        edge_index cur = head_[v];
        head_[v] = npos;
        edge_index new_head = npos;
        edge_index new_tail = npos;
        while (cur != npos) {
            edge_index nxt = nodes_[cur].next;
            // Snapshot the data before push_back may invalidate references
            // into nodes_ via reallocation.
            T data = nodes_[cur].data;
            if (pred(data)) {
                on_keep(data);
                edge_index slot;
                if (parallel_append_active_) {
                    slot = __atomic_fetch_add(&parallel_tail_, 1,
                                              __ATOMIC_RELAXED);
                    nodes_[slot] = {data, npos};
                } else {
                    nodes_.push_back({data, npos});
                    if (nodes_.size() > static_cast<std::size_t>(kPoolMax))
                        edge_index_overflow("forward_star::filter_append");
                    slot = static_cast<edge_index>(nodes_.size()) - 1;
                }
                if (new_head == npos) new_head = slot;
                else nodes_[new_tail].next = slot;
                new_tail = slot;
            }
            cur = nxt;
        }
        head_[v] = new_head;
    }

  public:

    /// Rebuild the entire pool so each vertex's chain is laid out contiguously
    /// in nodes_ in adjacency order.  After many filter() calls, chains
    /// fragment: nodes_[head_[v]].next can jump arbitrarily through the pool,
    /// killing prefetch.  compact() fixes this by walking each chain once
    /// and re-emitting survivors into a fresh contiguous buffer.
    ///
    /// Cost: O(total_surviving_edges).  Reclaims memory from filtered-out
    /// nodes — pool size after compact() == surviving edges + 1 sentinel.
    void compact() {
        const node_index n = static_cast<node_index>(head_.size());

        // Per-vertex chain length, then prefix-sum to get the start
        // offset of each vertex's run in the new pool.  Each vertex's
        // output range is disjoint, so the rebuild loop has no
        // contention and runs in parallel without atomics.
        std::vector<std::size_t> start(n + 1, 0);
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (node_index v = 0; v < n; ++v) {
            std::size_t len = 0;
            for (edge_index cur = head_[v]; cur != npos; cur = nodes_[cur].next)
                ++len;
            start[v + 1] = len;
        }
        // Reserve slot 0 as the sentinel.  Convert per-vertex lengths
        // (currently in start[v+1]) into start offsets via running sum:
        //   start[0]   = 1
        //   start[v+1] = start[v] + chain_len_v
        start[0] = 1;
        for (node_index v = 0; v < n; ++v)
            start[v + 1] += start[v];
        const std::size_t total = start[n];  // includes sentinel
        if (total > static_cast<std::size_t>(kPoolMax))
            edge_index_overflow("forward_star::compact");

        std::vector<node, util::big_alloc<node>> new_nodes(total);  // sentinel at index 0
        std::vector<edge_index> new_head(n, npos);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (node_index v = 0; v < n; ++v) {
            std::size_t off = start[v];
            std::size_t end = start[v + 1];
            if (off == end) continue;
            new_head[v] = static_cast<edge_index>(off);
            edge_index cur = head_[v];
            for (std::size_t k = off; k < end; ++k) {
                new_nodes[k].data = nodes_[cur].data;
                new_nodes[k].next = (k + 1 < end)
                    ? static_cast<edge_index>(k + 1) : npos;
                cur = nodes_[cur].next;
            }
        }
        head_ = std::move(new_head);
        nodes_ = std::move(new_nodes);
    }

    /// Pool occupancy ratio — fraction of pool slots actually reachable
    /// from a head[].  After many filter() calls this drops.  Use as a
    /// trigger heuristic for compact().
    double live_fraction() const {
        if (nodes_.size() <= 1) return 1.0;
        return double(live_count()) / double(nodes_.size() - 1);
    }

    /// Approximate memory usage in bytes (heap only).
    std::size_t memory_bytes() const {
        return head_.capacity() * sizeof(edge_index)
             + nodes_.capacity() * sizeof(node);
    }

    template<typename Fs>
    struct iterator_ {
        using fs_type       = std::remove_reference_t<Fs>;
        using value_type    = std::conditional_t<std::is_const_v<fs_type>, const T, T>;
        using difference_type = std::ptrdiff_t;

        fs_type*   fs = nullptr;
        edge_index idx = npos;

        value_type& operator*()  const { return fs->nodes_[idx].data; }
        value_type* operator->() const { return &fs->nodes_[idx].data; }
        iterator_& operator++()    { idx = fs->nodes_[idx].next; return *this; }
        iterator_  operator++(int) { return {fs, std::exchange(idx, fs->nodes_[idx].next)}; }
        friend bool operator==(const iterator_& it, std::default_sentinel_t) {
            return it.idx == npos;
        }
    };

    template<typename Self>
    auto operator[](this Self&& self, node_index v) {
        return std::ranges::subrange(
            iterator_<Self>{&self, self.head_[v]}, std::default_sentinel);
    }

private:
    // head_[v] is a POOL offset (edge_index); indexed by vertex id (node_index).
    std::vector<edge_index> head_;
    // big_alloc: nodes_ can grow to hundreds of MB on grid_3000+.  See
    // big_alloc.h for rationale (THP + MAP_POPULATE).
    std::vector<node, util::big_alloc<node>> nodes_;
    bool filter_append_ = false;
    bool parallel_append_active_ = false;
    edge_index parallel_tail_ = 0;  // atomic when parallel_append_active_
    edge_index atomic_pool_base_  = 0;
    edge_index atomic_pool_count_ = 0;
};

} // namespace apxchol
