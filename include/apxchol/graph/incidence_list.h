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

#include "apxchol/types.h"
#include "apxchol/graph/forward_star.h"
#include <algorithm>
#include <boost/container/small_vector.hpp>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

// ── char_traits<edge_index> ──
// libstdc++ provides a default char_traits primary template for any type,
// but libc++ (LLVM ≥ 15) intentionally leaves it undefined — only
// char/wchar_t/char8_t/char16_t/char32_t have specializations.
// We provide an explicit specialization so basic_string<edge_index> works
// on both standard libraries.  Note: specializing std templates for
// non-program-defined types (like int32_t) is technically non-conforming
// per [namespace.std] p2, but is harmless in practice.
#if defined(_LIBCPP_VERSION)
template<>
struct std::char_traits<apxchol::edge_index> {
    using char_type  = apxchol::edge_index;
    using int_type   = std::int64_t;
    using off_type   = std::streamoff;
    using pos_type   = std::streampos;
    using state_type = std::mbstate_t;

    static constexpr void assign(char_type& r, const char_type& a) noexcept { r = a; }
    static constexpr char_type* assign(char_type* p, std::size_t n, char_type a) {
        for (std::size_t i = 0; i < n; ++i) p[i] = a;
        return p;
    }
    static constexpr bool eq(char_type a, char_type b) noexcept { return a == b; }
    static constexpr bool lt(char_type a, char_type b) noexcept { return a < b; }
    static constexpr char_type* move(char_type* d, const char_type* s, std::size_t n) {
        if (d < s) for (std::size_t i = 0; i < n; ++i) d[i] = s[i];
        else       for (std::size_t i = n; i-- > 0;)   d[i] = s[i];
        return d;
    }
    static constexpr char_type* copy(char_type* d, const char_type* s, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) d[i] = s[i];
        return d;
    }
    static constexpr int compare(const char_type* a, const char_type* b, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            if (lt(a[i], b[i])) return -1;
            if (lt(b[i], a[i])) return  1;
        }
        return 0;
    }
    static constexpr std::size_t length(const char_type* s) {
        std::size_t n = 0; while (!eq(s[n], char_type())) ++n; return n;
    }
    static constexpr const char_type* find(const char_type* s, std::size_t n, const char_type& a) {
        for (std::size_t i = 0; i < n; ++i) if (eq(s[i], a)) return s + i;
        return nullptr;
    }
    static constexpr char_type  to_char_type(int_type c) noexcept { return static_cast<char_type>(c); }
    static constexpr int_type   to_int_type(char_type c) noexcept { return static_cast<int_type>(c); }
    static constexpr bool       eq_int_type(int_type a, int_type b) noexcept { return a == b; }
    static constexpr int_type   eof()       noexcept { return -1; }
    static constexpr int_type   not_eof(int_type e) noexcept { return eq_int_type(e, eof()) ? 0 : e; }
};
#endif

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
};

/// Default small-buffer capacity for small_vec_incidence.
inline constexpr std::size_t default_sso_capacity = 12;

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
        auto out = list.begin();
        for (auto it = list.begin(); it != list.end(); ++it) {
            if (pred(*it)) {
                on_keep(*it);
                if (out != it) *out = std::move(*it);
                ++out;
            }
        }
        list.erase(out, list.end());
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
