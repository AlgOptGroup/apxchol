#pragma once

// One-launch stable compaction and exclusive sum for short inputs. A CUB
// device-wide select or scan launches two kernels (tile-state init and sweep)
// and does a host-side size query; for the few thousand items of a late
// elimination round, one block does the same work in one launch. Outputs are
// the same as CUB's: the kept items in input order with their count, and the
// wrapping uint32 exclusive prefix sum.

#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace apxchol::detail {
namespace {

constexpr unsigned kSmallScanBlock = 1024;
constexpr unsigned kSmallScanItems = 4;
// Inputs up to this many items take the one-block path.
constexpr std::size_t kSmallScanLimit = std::size_t{kSmallScanBlock} * kSmallScanItems;

// keep(i) decides item i; kept input[i] go to output in order, the count to
// *device_count.
template<class Input, class T, class Keep>
__global__ void __launch_bounds__(kSmallScanBlock)
small_stable_select(Input input, T* output, int count, int* device_count, Keep keep) {
    using scan = cub::BlockScan<int, kSmallScanBlock>;
    __shared__ typename scan::TempStorage storage;
    const int begin = static_cast<int>(threadIdx.x * kSmallScanItems);
    bool kept[kSmallScanItems];
    int local = 0;
    #pragma unroll
    for (unsigned k = 0; k < kSmallScanItems; ++k) {
        const int i = begin + static_cast<int>(k);
        kept[k] = i < count && keep(i);
        local += kept[k] ? 1 : 0;
    }
    int offset = 0, total = 0;
    scan(storage).ExclusiveSum(local, offset, total);
    #pragma unroll
    for (unsigned k = 0; k < kSmallScanItems; ++k)
        if (kept[k]) output[offset++] = input[begin + static_cast<int>(k)];
    if (threadIdx.x == 0) *device_count = total;
}

// output[i] = input[0] + ... + input[i-1] (mod 2^32); with write_last,
// output[count] = last as well.
__global__ void __launch_bounds__(kSmallScanBlock)
small_exclusive_sum(const std::uint32_t* input, std::uint32_t* output, int count,
                    bool write_last, std::uint32_t last) {
    using scan = cub::BlockScan<std::uint32_t, kSmallScanBlock>;
    __shared__ typename scan::TempStorage storage;
    const int begin = static_cast<int>(threadIdx.x * kSmallScanItems);
    std::uint32_t items[kSmallScanItems];
    std::uint32_t local = 0;
    #pragma unroll
    for (unsigned k = 0; k < kSmallScanItems; ++k) {
        const int i = begin + static_cast<int>(k);
        items[k] = i < count ? input[i] : 0u;
        local += items[k];
    }
    std::uint32_t offset = 0;
    scan(storage).ExclusiveSum(local, offset);
    #pragma unroll
    for (unsigned k = 0; k < kSmallScanItems; ++k) {
        const int i = begin + static_cast<int>(k);
        if (i < count) output[i] = offset;
        offset += items[k];
    }
    if (write_last && threadIdx.x == 0) output[count] = last;
}

struct flag_is_set {
    const std::uint8_t* flags;
    __device__ bool operator()(int i) const { return flags[i] != 0; }
};

template<class Input, class Predicate>
struct predicate_holds {
    Input input;
    Predicate predicate;
    __device__ bool operator()(int i) const { return predicate(input[i]); }
};

} // namespace
} // namespace apxchol::detail
