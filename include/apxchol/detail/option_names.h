#pragma once
/// Option name tables shared by the CLI and the Python and MATLAB/Octave
/// bindings, so every frontend accepts the same spellings.

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

namespace apxchol::detail {

template<class E>
struct option_name {
    std::string_view name;
    E value;
};

/// Map a frontend name to its enum value. The error lists every accepted name.
template<class E, std::size_t N>
E parse_option_name(const std::array<option_name<E>, N>& table,
                    std::string_view value, std::string_view option) {
    for (const auto& entry : table)
        if (entry.name == value) return entry.value;
    std::string message = std::string(option) + " must be ";
    for (std::size_t i = 0; i < N; ++i) {
        if (i) message += i + 1 == N ? " or " : ", ";
        message += table[i].name;
    }
    throw std::invalid_argument(message);
}

} // namespace apxchol::detail
