#pragma once
/// The list of all built-in partitioners.  Adding a new partitioner =
/// add one line to `partitioner_list` (and a header include above), plus a
/// matching `partitioner_kind` value and `detail::partitioner_names` entry.

#include "apxchol/solver/partitioner.h"
#include "apxchol/solver/partition/block_greedy.h"
#include "apxchol/solver/partition/priority_greedy.h"
#include "apxchol/solver/partition/baumann_kyng.h"
#include <optional>
#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace apxchol {

using partitioner_list = std::tuple<
    block_greedy_partitioner,
    priority_greedy_partitioner,
    baumann_kyng_partitioner
>;

/// Concept check: every entry in partitioner_list satisfies `partitioner`,
/// and the list matches partitioner_kind and its frontend names.
namespace detail {
template<typename... Ts>
constexpr bool all_partitioners(std::tuple<Ts...>*) {
    return (... && partitioner<Ts>);
}
template<typename... Ts>
constexpr bool partitioners_match_kinds(std::tuple<Ts...>*) {
    const std::string_view names[] = {Ts::name...};
    if (sizeof...(Ts) != partitioner_names.size()) return false;
    for (std::size_t i = 0; i < sizeof...(Ts); ++i)
        if (names[i] != partitioner_names[i].name ||
            static_cast<std::size_t>(std::to_underlying(partitioner_names[i].value)) != i)
            return false;
    return true;
}
}
static_assert(detail::all_partitioners(static_cast<partitioner_list*>(nullptr)),
              "every entry in partitioner_list must satisfy `partitioner`");
static_assert(detail::partitioners_match_kinds(static_cast<partitioner_list*>(nullptr)),
              "partitioner_list must match partitioner_kind and detail::partitioner_names");

/// Dispatch by kind.  `fn` is a generic lambda
/// `[]<typename P>() -> Result { ... }`.  Throws std::invalid_argument
/// if `kind` is not a partitioner_kind value.
template<typename Result, typename Fn>
Result dispatch_partitioner(partitioner_kind kind, Fn&& fn) {
    std::optional<Result> result;
    [&]<typename... Ts>(std::tuple<Ts...>*) {
        std::size_t i = 0;
        (void)(... || (i++ == static_cast<std::size_t>(std::to_underlying(kind)) &&
                       (result.emplace(fn.template operator()<Ts>()), true)));
    }(static_cast<partitioner_list*>(nullptr));
    if (!result) throw std::invalid_argument("invalid partitioner");
    return std::move(*result);
}

} // namespace apxchol
