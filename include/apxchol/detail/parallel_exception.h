#pragma once
#include <atomic>
#include <exception>

namespace apxchol::detail {

// Capture inside the team; rethrow only after all workers have joined.
class parallel_exception {
public:
    template<class Work>
    void run(Work&& work) noexcept {
        try {
            work();
        } catch (...) {
            if (!failed_.exchange(true)) error_ = std::current_exception();
        }
    }

    bool failed() const noexcept { return failed_.load(); }
    void rethrow() const {
        if (error_) std::rethrow_exception(error_);
    }

private:
    std::atomic<bool> failed_{false};
    std::exception_ptr error_;
};

} // namespace apxchol::detail
