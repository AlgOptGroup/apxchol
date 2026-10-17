APXCHOL_GPU_CODE(
namespace apxchol { namespace detail { namespace gpu_dataflow {

APXCHOL_GPU_CONSTANT unsigned lanes = 32, prefetch = 8, chunk_size = lanes * prefetch;
APXCHOL_GPU_CONSTANT unsigned special_bit = 1u << 31, empty = ~0u;

// A special task computes a row segment, or folds partials when row's high bit
// is set. Normal tasks carry the row directly. Both retain 31-bit row indices.
struct alignas(16) segment {
    unsigned row, begin, end, output;
};
static_assert(sizeof(segment) == 16, "segment layout");
struct counts {
    APXCHOL_GPU_UINT64 words = 0, segments = 0, slots = 0;
};

struct row_order {
    unsigned n;
    bool reverse;
    APXCHOL_GPU_BUFFER(const unsigned) rows = nullptr;
    APXCHOL_GPU_FUNCTION unsigned operator()(unsigned q) const {
        return rows ? rows[q] : reverse ? n - 1 - q : q;
    }
};

// Count and emit use the same walk on CPU and GPU. Each invocation closes its
// last batch, so independent tiles/workgroups can be packed with prefix sums.
template <bool Emit, class Pointer>
APXCHOL_GPU_FUNCTION counts pack_rows(APXCHOL_GPU_BUFFER(const Pointer) ptr, row_order order, unsigned first, unsigned last,
                                       unsigned split_min, counts base = {}, APXCHOL_GPU_BUFFER(unsigned) words = nullptr,
                                       APXCHOL_GPU_BUFFER(segment) segments = nullptr) {
    counts c;
    auto pad = [&](unsigned alignment) {
        const auto padding = (-c.words) & (alignment - 1);
        if (Emit)
            for (unsigned i = 0; i < padding; ++i)
                words[base.words + c.words + i] = empty;
        c.words += padding;
    };
    auto special = [&](segment s) {
        pad(lanes);
        if (Emit) {
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
            if (Emit)
                for (unsigned i = 0; i < group; ++i)
                    words[base.words + c.words + i] = row;
            c.words += group;
        }
    }
    pad(lanes);
    return c;
}

} } }
)
