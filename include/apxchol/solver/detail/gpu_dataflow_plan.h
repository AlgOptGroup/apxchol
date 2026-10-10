#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef __CUDACC__
#define APXCHOL_PLAN_FUNCTION __host__ __device__
#else
#define APXCHOL_PLAN_FUNCTION
#endif

namespace apxchol::detail::gpu_dataflow {

inline constexpr unsigned lanes = 32, prefetch = 8, chunk_size = lanes * prefetch;
inline constexpr unsigned special_bit = 1u << 31, empty = ~0u;

// A special task computes a row segment, or folds partials when row's high bit
// is set. Normal tasks carry the row directly. Both retain 31-bit row indices.
struct alignas(16) segment {
    unsigned row, begin, end, output;
};
static_assert(sizeof(segment) == 16);
struct counts {
    std::uint64_t words = 0, segments = 0, slots = 0;
};

struct row_order {
    unsigned n;
    bool reverse;
    const unsigned *rows = nullptr;
    APXCHOL_PLAN_FUNCTION unsigned operator()(unsigned q) const {
        return rows ? rows[q] : reverse ? n - 1 - q : q;
    }
};

// Count and emit use the same walk on CPU and CUDA. Each invocation closes its
// last batch, so independent tiles/workgroups can be packed with prefix sums.
template <bool Emit, class Pointer>
APXCHOL_PLAN_FUNCTION counts pack_rows(const Pointer *ptr, row_order order, unsigned first, unsigned last,
                                       unsigned split_min, counts base = {}, unsigned *words = nullptr,
                                       segment *segments = nullptr) {
    counts c;
    auto pad = [&](unsigned alignment) {
        while (c.words % alignment) {
            if constexpr (Emit)
                words[base.words + c.words] = empty;
            ++c.words;
        }
    };
    auto special = [&](segment s) {
        pad(lanes);
        if constexpr (Emit) {
            segments[base.segments + c.segments] = s;
            for (unsigned i = 0; i < lanes; ++i)
                words[base.words + c.words + i] = special_bit | unsigned(base.segments + c.segments);
        }
        ++c.segments;
        c.words += lanes;
    };
    for (unsigned q = first; q < last; ++q) {
        const unsigned row = order(q);
        const unsigned begin = ptr[row] + unsigned(order.reverse);
        const unsigned end = ptr[row + 1] - unsigned(!order.reverse);
        const unsigned size = end - begin;
        if (!size)
            continue; // Initialization solves dependency-free rows.
        if (split_min && size > split_min) {
            // Cap partials for a single row, retaining the existing memory bound.
            const unsigned pieces = (size + chunk_size - 1) / chunk_size;
            const unsigned bounded = pieces < 4096 ? pieces : 4096;
            const unsigned width =
                ((size + bounded - 1) / bounded + chunk_size - 1) / chunk_size * chunk_size;
            const unsigned first_slot = order.n + unsigned(base.slots + c.slots);
            for (unsigned p = begin; p < end;) {
                const unsigned next = end - p < width ? end : p + width;
                special({row, p, next, order.n + unsigned(base.slots + c.slots)});
                ++c.slots;
                p = next;
            }
            special({row | special_bit, first_slot, order.n + unsigned(base.slots + c.slots), row});
        } else {
            unsigned group = 1;
            while (group < lanes && group * prefetch < size)
                group *= 2;
            pad(group);
            if constexpr (Emit)
                for (unsigned i = 0; i < group; ++i)
                    words[base.words + c.words + i] = row;
            c.words += group;
        }
    }
    pad(lanes);
    return c;
}

struct packed_plan {
    std::vector<unsigned> words, blocks{0};
    std::vector<segment> segments;
    unsigned slots = 0;
};

template <class Pointer>
inline packed_plan pack(const Pointer *ptr, row_order order, const std::vector<unsigned> &blocks,
                        unsigned split_min = chunk_size) {
    std::vector<counts> prefix(blocks.size());
    for (std::size_t b = 1; b < blocks.size(); ++b) {
        auto c = pack_rows<false>(ptr, order, blocks[b - 1], blocks[b], split_min);
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
        pack_rows<true>(ptr, order, blocks[b - 1], blocks[b], split_min, prefix[b - 1], out.words.data(),
                        out.segments.data());
        out.blocks.push_back(unsigned(prefix[b].words / lanes));
    }
    return out;
}
} // namespace apxchol::detail::gpu_dataflow

#undef APXCHOL_PLAN_FUNCTION
