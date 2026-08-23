#pragma once

#include <algorithm>
#include <cstdlib>

namespace apxchol::detail {

/// Number of parallel fixed-priority passes before an exact fallback. One
/// process-stable value is shared by the CPU and optional GPU implementation.
inline int priority_greedy_parallel_passes() {
    static const int value = [] {
        const char* e = std::getenv("APXCHOL_PRIORITY_GREEDY_PASSES");
        return e ? std::max(0, std::atoi(e)) : 16;
    }();
    return value;
}

} // namespace apxchol::detail
