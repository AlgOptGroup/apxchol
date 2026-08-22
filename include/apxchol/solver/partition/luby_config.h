#pragma once

#include <cstdlib>

namespace apxchol::detail {

/// One process-stable value shared by the CPU and optional GPU implementation.
inline int luby_iteration_limit() {
    static const int value = [] {
        const char* e = std::getenv("APXCHOL_LUBY_ITERS");
        return e ? std::atoi(e) : 16;
    }();
    return value;
}

} // namespace apxchol::detail
