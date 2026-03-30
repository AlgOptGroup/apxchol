#pragma once
/// char_traits<edge_index> — satisfies the CharTraits named requirement.
///
/// Enables std::basic_string<edge_index> as an SSO-capable dynamic array.
/// The standard only mandates char_traits for char/wchar_t/char8_t/char16_t/char32_t;
/// libstdc++ provides a default primary template but libc++ does not.
/// We always provide our own for portability.
///
/// Specializing char_traits for non-program-defined types (e.g. int32_t)
/// is technically non-conforming per [namespace.std] p2, but is
/// well-established practice and works on all major implementations.

#include "apxchol/types.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

template<>
struct std::char_traits<apxchol::edge_index> {
    using char_type  = apxchol::edge_index;
    using int_type   = std::int64_t;   // wider than char_type; fits eof()
    using off_type   = std::streamoff;
    using pos_type   = std::streampos;
    using state_type = std::mbstate_t;

    static constexpr void assign(char_type& r, const char_type& a) noexcept { r = a; }

    static constexpr char_type* assign(char_type* p, std::size_t n, char_type a) noexcept {
        std::fill_n(p, n, a);
        return p;
    }

    static constexpr bool eq(char_type a, char_type b) noexcept { return a == b; }
    static constexpr bool lt(char_type a, char_type b) noexcept { return a < b; }

    static constexpr int compare(const char_type* a, const char_type* b, std::size_t n) {
        auto r = std::lexicographical_compare_three_way(a, a + n, b, b + n);
        return r < 0 ? -1 : r > 0 ? 1 : 0;
    }

    static constexpr std::size_t length(const char_type* s) {
        const char_type* p = s;
        while (*p != char_type{}) ++p;
        return static_cast<std::size_t>(p - s);
    }

    static constexpr const char_type* find(const char_type* s, std::size_t n, const char_type& a) {
        auto it = std::find(s, s + n, a);
        return it == s + n ? nullptr : it;
    }

    static constexpr char_type* move(char_type* d, const char_type* s, std::size_t n) {
        if (d < s)       std::copy_n(s, n, d);
        else if (d > s)  std::copy_backward(s, s + n, d + n);
        return d;
    }

    static constexpr char_type* copy(char_type* d, const char_type* s, std::size_t n) {
        std::copy_n(s, n, d);
        return d;
    }

    static constexpr char_type  to_char_type(int_type c) noexcept { return static_cast<char_type>(c); }
    static constexpr int_type   to_int_type(char_type c) noexcept { return static_cast<int_type>(c); }
    static constexpr bool       eq_int_type(int_type a, int_type b) noexcept { return a == b; }
    static constexpr int_type   eof()     noexcept { return -1; }
    static constexpr int_type   not_eof(int_type e) noexcept { return e != eof() ? e : 0; }
};
