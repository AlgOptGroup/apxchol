APXCHOL_GPU_CODE(
namespace resident_finalize {
enum : unsigned { block_rows = 128 };
struct Step { unsigned first, last; };
// levels counts the independent elimination rounds before workgroup coalescing.
struct PlanInfo { unsigned rows, blocks, stages, levels; };

// Compaction can prepend empty blocks; only the final original block has work.
APXCHOL_GPU_FUNCTION unsigned block_begin(unsigned first, unsigned last) {
    return last - first > block_rows ? last - block_rows : first;
}

template<class Memory>
APXCHOL_GPU_FUNCTION void plan(Memory s) {
    *s.info = {};
    s.blocks[0] = 0;
    if (s.round_ptr[0] != 0 || s.round_ptr[s.rounds] < s.n) { s.fail(32); return; }
    for (unsigned r = 0; r < s.rounds; ++r)
        if (s.round_ptr[r] > s.round_ptr[r + 1]) { s.fail(32); return; }
    unsigned written = 0, blocks = 0, stages = 0, pending = 0, levels = 0;
    auto flush = [&] {
        if (!pending) return;
        s.steps[stages++] = {blocks, blocks + 1};
        written += pending;
        s.blocks[++blocks] = written;
        pending = 0;
    };
    for (unsigned q = 0; q < s.rounds; ++q) {
        const unsigned r = s.reverse ? s.rounds - 1 - q : q;
        const unsigned lo = s.round_ptr[r] < s.n ? s.round_ptr[r] : s.n;
        const unsigned hi = s.round_ptr[r + 1] < s.n ? s.round_ptr[r + 1] : s.n;
        const unsigned count = hi - lo;
        if (!count) continue;
        ++levels;
        if (count <= block_rows) {
            if (pending + count > block_rows) flush();
            pending += count;
        } else {
            flush();
            Step stage{blocks, 0};
            const unsigned end = written + count;
            while (written < end) {
                written += end - written < block_rows ? end - written : block_rows;
                s.blocks[++blocks] = written;
            }
            stage.last = blocks;
            s.steps[stages++] = stage;
        }
    }
    flush();
    *s.info = {written, blocks, stages, levels};
}

// Root rows are initialized before the solve. Remove empty workgroups/stages,
// and reuse their ranges for adjacent groups: those extra rows emit no tasks.
template<class Memory, class Counts, class Count>
APXCHOL_GPU_FUNCTION void compact(Memory s, Counts counts, Count sum) {
    unsigned blocks = 0, stages = 0;
    for (unsigned st = 0; st < s.info->stages; ++st) {
        const Step original = s.steps[st];
        Step kept{blocks, 0};
        for (unsigned b = original.first; b < original.last; ++b) {
            const auto value = counts[b];
            if (!value.words) continue;
            counts[blocks] = sum;
            sum.words += value.words;
            sum.segments += value.segments;
            sum.slots += value.slots;
            s.blocks[++blocks] = s.blocks[b + 1];
        }
        kept.last = blocks;
        if (kept.first != kept.last) s.steps[stages++] = kept;
    }
    counts[blocks] = sum;
    s.info->blocks = blocks;
    s.info->stages = stages;
}
}
)
