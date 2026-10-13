#pragma once
#include "apxchol/sparse_csc.h"
#include <type_traits>

namespace apxchol::detail {

// Raw append format shared by host assembly and device factor exports.
struct factor_entry {
    node_index neighbor = 0;
    factor_value_t value = 0;
};
static_assert(std::is_trivially_copyable_v<factor_entry>);

} // namespace apxchol::detail
