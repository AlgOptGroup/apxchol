#pragma once
/// Memory allocator that uses transparent huge pages and pre-population for
/// large allocations. Adapted from cp-algo (Oleksandr Kulkov's CP library):
/// https://lib.cp-algorithms.com/cp-algo/util/big_alloc.hpp
///
/// For allocations >= 1 MB:
///   - mmap with MAP_PRIVATE | MAP_ANONYMOUS
///   - madvise MADV_HUGEPAGE: ask kernel to back with 2 MB pages (THP)
///   - by default, madvise MADV_POPULATE_WRITE pre-faults all pages so there
///     are no per-page minor faults during use. The Populate=false allocator
///     variant leaves pages lazy for over-allocated containers whose unused
///     capacity should not become resident.
///
/// Effect: removes the per-page minor faults that std::vector value-init
/// triggers during first touch. For 128 MB allocations with 4 KB pages this
/// was ~30k page faults @ ~3 us each = ~100 ms. With 2 MB pages: ~64 faults.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <sys/mman.h>

// Linux 5.14 uapi constant; absent from older glibc headers (e.g. the
// manylinux_2_28 wheel-build image, glibc 2.28). Define the raw value and let
// pre-5.14 kernels return EINVAL — the madvise calls below are best-effort
// advice and their return values are deliberately ignored.
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif

namespace apxchol::util {

inline constexpr std::size_t MEGABYTE = 1u << 20;

inline std::size_t round_up(std::size_t n) {
    return (n + 4095) & ~std::size_t(4095);
}

template <typename T, std::size_t Align = 32, bool Populate = true>
class big_alloc {
public:
    using value_type = T;
    template <class U> struct rebind {
        using other = big_alloc<U, Align, Populate>;
    };

    big_alloc() = default;
    template <class U>
    big_alloc(const big_alloc<U, Align, Populate>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        std::size_t padded = round_up(n * sizeof(T));
        std::size_t align  = std::max<std::size_t>(alignof(T), Align);
        if (padded >= MEGABYTE) {
            void* raw = mmap(nullptr, padded,
                             PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (raw == MAP_FAILED) throw std::bad_alloc{};
            madvise(raw, padded, MADV_HUGEPAGE);
            if constexpr (Populate)
                madvise(raw, padded, MADV_POPULATE_WRITE);
            return static_cast<T*>(raw);
        }
        return static_cast<T*>(::operator new(padded, std::align_val_t(align)));
    }

    void deallocate(T* p, std::size_t n) noexcept {
        if (!p) return;
        std::size_t padded = round_up(n * sizeof(T));
        if (padded >= MEGABYTE) { munmap(p, padded); return; }
        std::size_t align = std::max<std::size_t>(alignof(T), Align);
        // The sized aligned-delete overload is optional in Clang unless
        // -fsized-deallocation is enabled.  The matching unsized aligned
        // overload is always sufficient for memory obtained from aligned new.
        ::operator delete(p, std::align_val_t(align));
    }

    bool operator==(const big_alloc&) const noexcept { return true; }
    bool operator!=(const big_alloc&) const noexcept { return false; }
};

}  // namespace apxchol::util
