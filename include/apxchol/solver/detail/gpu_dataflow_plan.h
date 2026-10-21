#pragma once
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#define APXCHOL_GPU_CODE(...) __VA_ARGS__
#ifdef __CUDACC__
#define APXCHOL_GPU_FUNCTION __host__ __device__
#else
#define APXCHOL_GPU_FUNCTION
#endif
#define APXCHOL_GPU_BUFFER(T) T*
#define APXCHOL_GPU_UINT64 std::uint64_t
#define APXCHOL_GPU_CONSTANT inline constexpr
#include "gpu_dataflow_pack.inc"
#undef APXCHOL_GPU_CONSTANT
#undef APXCHOL_GPU_UINT64
#undef APXCHOL_GPU_BUFFER
#undef APXCHOL_GPU_FUNCTION
#undef APXCHOL_GPU_CODE

namespace apxchol::detail::gpu_dataflow {

// Shared host/device histogram; CUDA adds validation errors in the same row pass.
struct row_stats {
    static constexpr int kBuckets = 12;
    unsigned long long rows_ge[kBuckets], nnz_ge[kBuckets];
    int max_len, error;

    // Split after one chunk when long rows hold at least a quarter of entries.
    unsigned split_threshold(std::uint64_t nnz) const {
        return nnz && nnz_ge[0] * 4 >= nnz ? chunk_size : 3 * chunk_size;
    }
};

struct packed_plan {
    std::vector<unsigned> words, blocks{0};
    std::vector<segment> segments;
    unsigned slots = 0;
};

template <unsigned PlanLanes = lanes, class Pointer>
inline packed_plan pack(const Pointer *ptr, row_order order, std::span<const unsigned> blocks,
                        unsigned split_min = chunk_size) {
    std::vector<counts> prefix(blocks.size());
    for (std::size_t b = 1; b < blocks.size(); ++b) {
        auto c = pack_rows<false, PlanLanes>(ptr, order, blocks[b - 1], blocks[b], split_min);
        auto p = prefix[b - 1];
        prefix[b] = {p.words + c.words, p.segments + c.segments, p.slots + c.slots};
    }
    const auto total = prefix.back();
    if (order.n >= special_bit || total.words > std::numeric_limits<unsigned>::max() ||
        total.segments >= special_bit || total.slots + order.n >= special_bit)
        throw std::length_error("GPU dataflow plan exceeds 32-bit capacity");
    packed_plan out;
    out.words.resize(total.words);
    out.segments.resize(total.segments);
    out.slots = unsigned(total.slots);
    for (std::size_t b = 1; b < blocks.size(); ++b) {
        pack_rows<true, PlanLanes>(ptr, order, blocks[b - 1], blocks[b], split_min, prefix[b - 1], out.words.data(),
                        out.segments.data());
        out.blocks.push_back(unsigned(prefix[b].words / PlanLanes));
    }
    return out;
}
} // namespace apxchol::detail::gpu_dataflow
