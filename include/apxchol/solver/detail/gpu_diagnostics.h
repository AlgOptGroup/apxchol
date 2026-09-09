#pragma once

#include <chrono>
#include <cstdlib>
#include <cstring>

namespace apxchol::detail {

// Keep setup quiet unless an existing verbose/stage/front-end trace requests
// diagnostics. Specific memory, nnz and SpTRSV statistics retain their own gates.
inline bool gpu_setup_diagnostics() noexcept {
    if (std::getenv("APXCHOL_VERBOSE") || std::getenv("APXCHOL_SPTRSV_SETUP_TRACE"))
        return true;
    const char* block_trace = std::getenv("APXCHOL_GPU_BLOCK_TRACE");
    return block_trace && *block_trace && std::strcmp(block_trace, "0") != 0;
}

struct gpu_setup_diagnostic_clock {
    using time_point = std::chrono::steady_clock::time_point;
    static time_point now() noexcept {
        return gpu_setup_diagnostics() ? std::chrono::steady_clock::now() : time_point{};
    }
};

} // namespace apxchol::detail
