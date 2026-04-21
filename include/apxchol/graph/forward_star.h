#pragma once
/// Chained forward star (链式前向星) — flat-array linked adjacency list.
///
/// Each vertex has a singly-linked chain of nodes threaded through
/// a flat pool.  Data and link are co-located in each node for
/// cache-friendly traversal.
/// push(v, data) prepends a node to v's chain in O(1) — no copies.
/// Iteration over v's chain is O(degree(v)).
/// No per-element erasure — use clear(v) to disconnect all elements.

#include "apxchol/types.h"
#include <cstddef>
#include <iterator>
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
    static constexpr index_t npos = 0;
    static constexpr graph_storage tag = graph_storage::forward_star;

    struct node { T data; index_t next; };

    forward_star() = default;

    void init(index_t n) {
        head_.assign(n, npos);
        nodes_.resize(1);
    }

    void push(index_t v, const T& val) {
        nodes_.push_back({val, head_[v]});
        head_[v] = static_cast<index_t>(nodes_.size()) - 1;
    }

    void reserve(index_t m) { nodes_.reserve(m); }
    void clear(index_t v) { head_[v] = npos; }

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
    index_t reserve_pool(index_t extra) {
        index_t start = static_cast<index_t>(nodes_.size());
        nodes_.resize(start + extra);
        return start;
    }

    /// Lock-free prepend of the pre-allocated node at `slot` to
    /// vertex v's chain.  Multiple threads may call this concurrently
    /// for the same v with distinct slots; the CAS retries on conflict.
    /// Pre: nodes_[slot] is in the range returned by `reserve_pool`.
    void push_atomic(index_t v, index_t slot, const T& data) {
        nodes_[slot].data = data;
        index_t old = __atomic_load_n(&head_[v], __ATOMIC_RELAXED);
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
    void filter(index_t v, auto&& pred, auto&& on_keep) {
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
        index_t* prev = &head_[v];
        while (*prev != npos) {
            if (pred(nodes_[*prev].data)) {
                on_keep(nodes_[*prev].data);
                prev = &nodes_[*prev].next;
            } else
                *prev = nodes_[*prev].next;
        }
    }

    void filter(index_t v, auto&& pred) {
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
    void begin_parallel_append(index_t upper_bound) {
        parallel_tail_ = static_cast<index_t>(nodes_.size());
        nodes_.resize(parallel_tail_ + upper_bound);
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
    index_t live_count() const {
        std::size_t kept = 0;
        for (index_t v = 0; v < static_cast<index_t>(head_.size()); ++v)
            for (index_t cur = head_[v]; cur != npos; cur = nodes_[cur].next)
                ++kept;
        return static_cast<index_t>(kept);
    }

  private:
    void filter_append_impl(index_t v, auto&& pred, auto&& on_keep) {
        // Walk old chain, snapshot head, sever it, then re-emit survivors
        // contiguously at the end of nodes_.
        index_t cur = head_[v];
        head_[v] = npos;
        index_t new_head = npos;
        index_t new_tail = npos;
        while (cur != npos) {
            index_t nxt = nodes_[cur].next;
            // Snapshot the data before push_back may invalidate references
            // into nodes_ via reallocation.
            T data = nodes_[cur].data;
            if (pred(data)) {
                on_keep(data);
                index_t slot;
                if (parallel_append_active_) {
                    slot = __atomic_fetch_add(&parallel_tail_, 1,
                                              __ATOMIC_RELAXED);
                    nodes_[slot] = {data, npos};
                } else {
                    nodes_.push_back({data, npos});
                    slot = static_cast<index_t>(nodes_.size()) - 1;
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
        // Count survivors first to pre-size the new pool exactly.
        std::size_t kept = 1;  // sentinel
        for (index_t v = 0; v < static_cast<index_t>(head_.size()); ++v)
            for (index_t cur = head_[v]; cur != npos; cur = nodes_[cur].next)
                ++kept;

        std::vector<node> new_nodes;
        new_nodes.reserve(kept);
        new_nodes.resize(1);  // sentinel at index 0 (== npos)
        std::vector<index_t> new_head(head_.size(), npos);

        for (index_t v = 0; v < static_cast<index_t>(head_.size()); ++v) {
            index_t prev = npos;
            for (index_t cur = head_[v]; cur != npos; cur = nodes_[cur].next) {
                new_nodes.push_back({nodes_[cur].data, npos});
                index_t new_idx = static_cast<index_t>(new_nodes.size()) - 1;
                if (prev == npos)
                    new_head[v] = new_idx;
                else
                    new_nodes[prev].next = new_idx;
                prev = new_idx;
            }
        }
        head_ = std::move(new_head);
        nodes_ = std::move(new_nodes);
    }

    /// Pool occupancy ratio — fraction of pool slots actually reachable
    /// from a head[].  After many filter() calls this drops.  Use as a
    /// trigger heuristic for compact().
    double live_fraction() const {
        std::size_t kept = 0;
        for (index_t v = 0; v < static_cast<index_t>(head_.size()); ++v)
            for (index_t cur = head_[v]; cur != npos; cur = nodes_[cur].next)
                ++kept;
        return nodes_.size() > 1 ? double(kept) / double(nodes_.size() - 1) : 1.0;
    }

    std::size_t size() const { return head_.size(); }

    /// Approximate memory usage in bytes (heap only).
    std::size_t memory_bytes() const {
        return head_.capacity() * sizeof(index_t)
             + nodes_.capacity() * sizeof(node);
    }

    template<typename Fs>
    struct iterator_ {
        using fs_type       = std::remove_reference_t<Fs>;
        using value_type    = std::conditional_t<std::is_const_v<fs_type>, const T, T>;
        using difference_type = std::ptrdiff_t;

        fs_type* fs = nullptr;
        index_t  idx = npos;

        value_type& operator*()  const { return fs->nodes_[idx].data; }
        value_type* operator->() const { return &fs->nodes_[idx].data; }
        iterator_& operator++()    { idx = fs->nodes_[idx].next; return *this; }
        iterator_  operator++(int) { return {fs, std::exchange(idx, fs->nodes_[idx].next)}; }
        friend bool operator==(const iterator_& it, std::default_sentinel_t) {
            return it.idx == npos;
        }
    };

    using iterator       = iterator_<forward_star>;
    using const_iterator = iterator_<const forward_star>;

    template<typename Self>
    auto operator[](this Self&& self, index_t v) {
        return std::ranges::subrange(
            iterator_<Self>{&self, self.head_[v]}, std::default_sentinel);
    }

private:
    std::vector<index_t> head_;
    std::vector<node> nodes_;
    bool filter_append_ = false;
    bool parallel_append_active_ = false;
    index_t parallel_tail_ = 0;  // atomic when parallel_append_active_
};

} // namespace apxchol
