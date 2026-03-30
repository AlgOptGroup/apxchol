#pragma once
/// Incidence list storage backends for graph.
///
/// Each backend manages per-vertex lists of edge_index values.
/// The graph itself owns the flat edge pool; these types only
/// store indices into that pool.
///
/// Required interface:
///   init(n)                   — set up n empty vertex lists
///   operator[](v) const       — range of edge_index for vertex v
///   push(v, idx)              — append edge_index to vertex v
///   clear(v)                  — clear vertex v's list
///   replace(v, range)          — replace vertex v's list
///   filter(v, pred)           — erase entries where pred is false
///   filter(v, pred, on_keep)  — same, calling on_keep for survivors
///   memory_bytes() const      — approximate heap usage in bytes
///   tag                       — static constexpr graph_storage value

#include "apxchol/types.h"
#include "apxchol/trivial_char_traits.h"
#include "apxchol/graph/forward_star.h"
#include <algorithm>
#include <boost/container/small_vector.hpp>
#include <concepts>
#include <cstddef>
#include <functional>
#include <ranges>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace apxchol {

namespace detail {
    struct edge_pred  { bool operator()(edge_index) const; };
    struct edge_visit { void operator()(edge_index) const; };
}

/// Concept for incidence list storage backends.
template<typename T>
concept incidence_storage = requires(T t, const T ct, index_t n, index_t v,
                                      edge_index idx, std::span<const edge_index> s,
                                      detail::edge_pred pred, detail::edge_visit visit) {
    t.init(n);
    { ct[v] } -> std::ranges::input_range;
    requires std::same_as<std::ranges::range_value_t<decltype(ct[v])>, edge_index>;
    t.push(v, idx);
    t.clear(v);
    t.replace(v, s);
    t.filter(v, pred);
    t.filter(v, pred, visit);
    { ct.memory_bytes() } -> std::convertible_to<std::size_t>;
    { T::tag } -> std::convertible_to<const graph_storage&>;
};

/// Default small-buffer capacity for small_vec_incidence.
inline constexpr std::size_t default_sso_capacity = 4;

// ── Contiguous storage (vector / small_vector / …) ──

template<typename Container, graph_storage Tag>
struct contiguous_incidence {
    static constexpr graph_storage tag = Tag;

    void init(index_t n) { adj_.assign(n, {}); }

    std::span<const edge_index> operator[](index_t v) const { return adj_[v]; }

    void push(index_t v, edge_index idx) { adj_[v].push_back(idx); }
    void clear(index_t v) { adj_[v].clear(); }

    template<std::ranges::input_range R>
        requires std::convertible_to<std::ranges::range_value_t<R>, edge_index>
    void replace(index_t v, R&& indices) {
        adj_[v].assign(std::ranges::begin(indices), std::ranges::end(indices));
    }

    /// Remove elements from v's list where pred(idx) is false.
    /// Calls on_keep(idx) for each surviving element.
    void filter(index_t v, auto&& pred, auto&& on_keep) {
        auto& list = adj_[v];
        list.erase(std::ranges::remove_if(list, std::not_fn(pred)).begin(),
                   list.end());
        for (auto idx : list) on_keep(idx);
    }

    void filter(index_t v, auto&& pred) {
        filter(v, pred, [](edge_index) {});
    }

    /// Approximate memory usage in bytes (heap only).
    std::size_t memory_bytes() const {
        std::size_t total = adj_.capacity() * sizeof(Container);
        for (const auto& c : adj_)
            total += c.capacity() * sizeof(edge_index);
        return total;
    }

private:
    std::vector<Container> adj_;
};

using vec_incidence = contiguous_incidence<std::vector<edge_index>, graph_storage::vec>;

template<std::size_t N = default_sso_capacity>
using small_vec_incidence_n = contiguous_incidence<
    boost::container::small_vector<edge_index, N>, graph_storage::small_vec>;

using small_vec_incidence = small_vec_incidence_n<>;

// ── basic_string SSO storage ──

using bstr_incidence = contiguous_incidence<
    std::basic_string<edge_index>, graph_storage::bstr>;

// ── forward_star<edge_index> satisfies incidence_storage directly ──

using forward_star_incidence = forward_star<edge_index>;

} // namespace apxchol
