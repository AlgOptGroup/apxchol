#pragma once
/// The list of all built-in partitioners.  Adding a new partitioner =
/// add one line to `partitioner_list` (and a header include above).

#include "apxchol/solver/partitioner.h"
#include "apxchol/solver/partition/block_greedy.h"
#include "apxchol/solver/partition/luby.h"
#include "apxchol/solver/partition/baumann_kyng.h"
#include "apxchol/solver/partition/rootset.h"
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace apxchol {

using partitioner_list = std::tuple<
    block_greedy_partitioner,
    luby_partitioner,
    baumann_kyng_partitioner,
    rootset_partitioner
>;

/// Concept check: every entry in partitioner_list satisfies `partitioner`.
namespace detail {
template<typename... Ts>
constexpr bool all_partitioners(std::tuple<Ts...>*) {
    return (... && partitioner<Ts>);
}
}
static_assert(detail::all_partitioners(static_cast<partitioner_list*>(nullptr)),
              "every entry in partitioner_list must satisfy `partitioner`");

/// Dispatch by name.  `fn` is a generic lambda
/// `[]<typename P>() -> Result { ... }`.  Throws std::invalid_argument
/// if `name` matches no entry, with a helpful "did you mean..." message.
template<typename Result, typename Fn>
Result dispatch_partitioner(std::string_view name, Fn&& fn) {
    std::optional<Result> result;
    auto try_one = [&]<typename P>() -> bool {
        if (P::name == name) {
            result = fn.template operator()<P>();
            return true;
        }
        return false;
    };
    [&]<typename... Ts>(std::tuple<Ts...>*) {
        bool ok = (... || try_one.template operator()<Ts>());
        if (!ok) {
            std::string valid;
            ((valid += (valid.empty() ? "" : ", "), valid += Ts::name), ...);
            throw std::invalid_argument(
                "unknown partitioner: '" + std::string(name) +
                "' - valid: " + valid);
        }
    }(static_cast<partitioner_list*>(nullptr));
    return std::move(*result);
}

} // namespace apxchol
