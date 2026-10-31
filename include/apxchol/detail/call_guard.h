#pragma once
#include <atomic>
#include <stdexcept>
#include <thread>

namespace apxchol::detail {

// Rejects a call that would overlap another thread's call on one object's
// mutable workspace. The owning thread may re-enter. A copied or assigned
// object starts unowned.
class call_guard {
public:
    call_guard() = default;
    call_guard(const call_guard&) noexcept {}
    call_guard& operator=(const call_guard&) noexcept { return *this; }

    // Releases ownership taken by enter(); empty for a re-entrant call.
    class scope {
    public:
        scope() = default;
        scope(const scope&) = delete;
        scope& operator=(const scope&) = delete;
        ~scope() {
            if (guard_) guard_->owner_.store(std::thread::id{}, std::memory_order_release);
        }

    private:
        friend call_guard;
        explicit scope(const call_guard* guard) noexcept : guard_(guard) {}
        const call_guard* guard_ = nullptr;
    };

    [[nodiscard]] scope enter() const {
        const auto self = std::this_thread::get_id();
        std::thread::id owner{};
        if (owner_.compare_exchange_strong(owner, self, std::memory_order_acquire,
                                           std::memory_order_relaxed))
            return scope(this);
        if (owner == self) return scope();
        throw std::runtime_error(
            "this solver is already in use by another thread; use a separate solver "
            "per thread, or pass all right-hand sides to solve() as one matrix");
    }

private:
    mutable std::atomic<std::thread::id> owner_{};
};

}  // namespace apxchol::detail
