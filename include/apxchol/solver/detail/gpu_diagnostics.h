#pragma once

#include <chrono>

namespace apxchol::detail {

struct gpu_setup_diagnostic_clock {
    using time_point = std::chrono::steady_clock::time_point;
    static time_point now(bool enabled) noexcept {
        return enabled ? std::chrono::steady_clock::now() : time_point{};
    }
};

} // namespace apxchol::detail
