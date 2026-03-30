#pragma once
/// Core index types for the apxchol library.
///
/// By default, all graph indices are 32-bit signed integers.
/// Define APXCHOL_64BIT_INDICES at compile time (or via CMake option)
/// to switch to 64-bit indices for very large graphs.

#include <cstdint>

namespace apxchol {

#ifdef APXCHOL_64BIT_INDICES
using index_t = int64_t;
#else
using index_t = int32_t;
#endif

using node_index = index_t;
using edge_index = index_t;

/// graph_storage enumerates the available incidence list backends
/// for runtime dispatch (CLI, factor_options).
enum class graph_storage { vec, forward_star, small_vec };

} // namespace apxchol
