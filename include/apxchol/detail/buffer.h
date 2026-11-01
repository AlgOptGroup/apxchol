#pragma once
/// Owned, 64-byte aligned array of trivial values for operator storage and
/// solver workspaces. resize() leaves new elements uninitialized and keeps no
/// contents when the size changes, so callers write every element before
/// reading it and the writing threads perform first touch. Not part of the
/// public API.

#include <algorithm>
#include <cstddef>
#include <new>
#include <span>
#include <type_traits>
#include <utility>

namespace apxchol::detail {

template<class T>
class buffer {
    static_assert(std::is_trivially_copyable_v<T> &&
                  std::is_trivially_default_constructible_v<T> &&
                  std::is_trivially_destructible_v<T>);

public:
    static constexpr std::size_t alignment = 64;

    buffer() = default;
    explicit buffer(std::size_t n) { resize(n); }
    buffer(const buffer& other) : buffer(other.size_) {
        std::copy_n(other.data_, size_, data_);
    }
    buffer(buffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}
    buffer& operator=(const buffer& other) {
        if (this != &other) {
            buffer copy(other);
            swap(copy);
        }
        return *this;
    }
    buffer& operator=(buffer&& other) noexcept {
        buffer moved(std::move(other));
        swap(moved);
        return *this;
    }
    ~buffer() { release(); }

    /// Reallocate to n uninitialized elements; a no-op when n == size().
    void resize(std::size_t n) {
        if (n == size_) return;
        release();
        if (n != 0) {
            if (n > static_cast<std::size_t>(-1) / sizeof(T)) throw std::bad_array_new_length();
            data_ = static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(alignment)));
        }
        size_ = n;
    }
    /// Free the storage.
    void clear() noexcept { release(); }
    void swap(buffer& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    T& operator[](std::size_t i) noexcept { return data_[i]; }
    const T& operator[](std::size_t i) const noexcept { return data_[i]; }
    T* begin() noexcept { return data_; }
    T* end() noexcept { return data_ + size_; }
    const T* begin() const noexcept { return data_; }
    const T* end() const noexcept { return data_ + size_; }
    operator std::span<T>() noexcept { return {data_, size_}; }
    operator std::span<const T>() const noexcept { return {data_, size_}; }

private:
    void release() noexcept {
        if (data_) ::operator delete(data_, std::align_val_t(alignment));
        data_ = nullptr;
        size_ = 0;
    }

    T* data_ = nullptr;
    std::size_t size_ = 0;
};

} // namespace apxchol::detail
