#pragma once
/// char_traits specialization for trivially copyable types.
///
/// Enables std::basic_string<T> as an SSO-capable dynamic array.
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
#include <cstring>
#include <string>

template<>
struct std::char_traits<apxchol::edge_index> {
    using char_type  = apxchol::edge_index;
    using int_type   = std::int64_t;   // wider than char_type; fits eof()
    using off_type   = std::streamoff;
    using pos_type   = std::streampos;
    using state_type = std::mbstate_t;

    static constexpr void assign(char_type& r, const char_type& a) noexcept { r = a; }

    static constexpr char_type* assign(char_type* p, std::size_t n, char_type a) {
        std::fill_n(p, n, a);
        return p;
    }

    static constexpr bool eq(char_type a, char_type b) noexcept { return a == b; }
    static constexpr bool lt(char_type a, char_type b) noexcept { return a < b; }

    static constexpr char_type* move(char_type* d, const char_type* s, std::size_t n) {
        if (std::is_constant_evaluated()) {
            // constexpr path: manual overlap-safe copy
            if (d < s) std::copy_n(s, n, d);
            else       std::copy_backward(s, s + n, d + n);
        } else {
            std::memmove(d, s, n * sizeof(char_type));
        }
        return d;
    }

    static constexpr char_type* copy(char_type* d, const char_type* s, std::size_t n) {
        if (std::is_constant_evaluated())
            std::copy_n(s, n, d);
        else
            std::memcpy(d, s, n * sizeof(char_type));
        return d;
    }

    static constexpr int compare(const char_type* a, const char_type* b, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            if (a[i] < b[i]) return -1;
            if (b[i] < a[i]) return  1;
        }
        return 0;
    }

    static constexpr std::size_t length(const char_type* s) {
        std::size_t n = 0;
        while (s[n] != char_type{}) ++n;
        return n;
    }

    static constexpr const char_type* find(const char_type* s, std::size_t n, const char_type& a) {
        for (std::size_t i = 0; i < n; ++i)
            if (s[i] == a) return s + i;
        return nullptr;
    }

    static constexpr char_type  to_char_type(int_type c) noexcept { return static_cast<char_type>(c); }
    static constexpr int_type   to_int_type(char_type c) noexcept { return static_cast<int_type>(c); }
    static constexpr bool       eq_int_type(int_type a, int_type b) noexcept { return a == b; }
    static constexpr int_type   eof()     noexcept { return -1; }
    static constexpr int_type   not_eof(int_type e) noexcept { return eq_int_type(e, eof()) ? 0 : e; }
};
