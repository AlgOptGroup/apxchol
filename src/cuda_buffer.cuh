#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace apxchol::detail {

struct cuda_untracked_allocation {
    static std::size_t size_bytes(std::size_t count, std::size_t width, const char*) {
        return count * width;
    }
    void add(std::size_t) {}
    void remove(std::size_t) noexcept {}
};

// Each caller keeps its error/size checks and optional allocation accounting.
// A replacement remains owned until the old allocation is successfully retired.
template<class T, auto Check, class Tracker = cuda_untracked_allocation>
class cuda_buffer {
public:
    explicit cuda_buffer(std::size_t count = 0) { allocate(count, "cudaMalloc"); }
    explicit cuda_buffer(Tracker& tracker) : tracker_(&tracker) {}
    ~cuda_buffer() {
        if (data_) { cudaFree(data_); release(); }
    }
    cuda_buffer(const cuda_buffer&) = delete;
    cuda_buffer& operator=(const cuda_buffer&) = delete;

    void allocate(std::size_t count, const char* what) {
        if (data_ || !count) return;
        const std::size_t size = Tracker::size_bytes(count, sizeof(T), what);
        // Account before allocating: an accounting overflow cannot leak memory.
        if (tracker_) tracker_->add(size);
        T* next = nullptr;
        const auto status = cudaMalloc(reinterpret_cast<void**>(&next), size);
        if (status != cudaSuccess) {
            if (tracker_) tracker_->remove(size);
            Check(status, what);
        }
        data_ = next;
        count_ = count;
    }

    T* release() noexcept {
        if (tracker_) tracker_->remove(bytes());
        count_ = 0;
        return std::exchange(data_, nullptr);
    }

    void reset(const char* what = "cudaFree") {
        if (!data_) return;
        // Keep ownership on failure so destruction can retry without throwing.
        Check(cudaFree(data_), what);
        release();
    }

    void grow_preserve(std::size_t count, std::size_t used,
                       const char* allocate_what, const char* copy_what,
                       const char* release_what) {
        if (count <= count_) return;
        if (used > count_)
            throw std::logic_error(
                "GPU round shadow: preserved prefix exceeds buffer capacity");
        cuda_buffer next;
        next.tracker_ = tracker_;
        next.allocate(count, allocate_what);
        if (used)
            Check(cudaMemcpy(next.data_, data_, used * sizeof(T),
                             cudaMemcpyDeviceToDevice), copy_what);
        reset(release_what);
        std::swap(data_, next.data_);
        std::swap(count_, next.count_);
    }

    void reserve(std::size_t count) {
        grow_preserve(count, 0, "cudaMalloc", nullptr, "cudaFree during growth");
    }
    T* get() noexcept { return data_; }
    const T* get() const noexcept { return data_; }
    std::size_t count() const noexcept { return count_; }
    std::size_t bytes() const noexcept { return count_ * sizeof(T); }

private:
    Tracker* tracker_ = nullptr;
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

} // namespace apxchol::detail
