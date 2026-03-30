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

    template<std::ranges::input_range R>
        requires std::convertible_to<std::ranges::range_value_t<R>, T>
    void replace(index_t v, R&& values) {
        head_[v] = npos;
        for (const auto& val : values)
            push(v, val);
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

    std::size_t size() const { return head_.size(); }

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
