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

    /// Remove chain elements where pred(data) is false, by relinking.
    /// Calls on_keep(data) for each surviving element.
    /// O(chain length), no allocation, no pool growth.
    void filter(index_t v, auto&& pred, auto&& on_keep) {
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
};

} // namespace apxchol
