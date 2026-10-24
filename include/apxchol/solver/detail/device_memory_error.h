#pragma once
#include <new>
#include <string>
#include <utility>

namespace apxchol::detail {
// Preserve the allocation diagnostic while allowing callers to catch bad_alloc.
class device_memory_error : public std::bad_alloc {
public:
    explicit device_memory_error(std::string message) : message_(std::move(message)) {}
    const char* what() const noexcept override { return message_.c_str(); }

private:
    std::string message_;
};
} // namespace apxchol::detail
