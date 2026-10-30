#pragma once
#include <mutex>

namespace apxchol::detail {

// Serializes const calls that share one object's mutable workspace. A copied
// or moved-to object gets its own unlocked mutex.
class call_mutex {
public:
    call_mutex() = default;
    call_mutex(const call_mutex&) noexcept {}
    call_mutex& operator=(const call_mutex&) noexcept { return *this; }

    [[nodiscard]] std::unique_lock<std::mutex> lock() const { return std::unique_lock(mutex_); }

private:
    mutable std::mutex mutex_;
};

}  // namespace apxchol::detail
