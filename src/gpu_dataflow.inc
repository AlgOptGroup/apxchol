APXCHOL_GPU_CODE(
struct alignas(16) dataflow_segment { unsigned row, begin, end, output; };
static_assert(sizeof(dataflow_segment) == 16, "Matches gpu_dataflow::segment in the plan");
struct dataflow_task { unsigned row, begin, end, output, kind; };
template<class Memory>
APXCHOL_GPU_FUNCTION dataflow_task dataflow_describe(Memory m, unsigned word) {
    if (word >> 31) {
        auto s = m.segments[word & 0x7fffffffu];
        return {s.row & 0x7fffffffu, s.begin, s.end, s.output, (s.row >> 31) ? 2u : 1u};
    }
    return {word, unsigned(m.ptr[word]) + unsigned(m.reverse),
            unsigned(m.ptr[word + 1]) - unsigned(!m.reverse), word, 0};
}
template<class Pointer>
APXCHOL_GPU_FUNCTION float dataflow_accumulate(Pointer v, Pointer y, float sum) {
    APXCHOL_UNROLL
    for (unsigned k = 0; k < 8; ++k) sum = fma(v[k], y[k], sum);
    return sum;
}
template<class Memory>
APXCHOL_GPU_FUNCTION void dataflow_initialize(Memory m, unsigned row) {
    if (row < m.n && m.ptr[row + 1] - m.ptr[row] == 1) {
        float rhs, diagonal;
        m.prepare(row, rhs, diagonal);
        m.publish(row, m.finish(rhs, diagonal, 0));
    } else m.reset(row);
}
template<class Memory>
APXCHOL_GPU_FUNCTION void dataflow_run(Memory m, unsigned lane) {
    using Value = typename Memory::stored_value;
    for (;;) {
        unsigned task = lane == 0 ? m.claim() : 0;
        task = m.broadcast(task);
        if (task >= m.batches) break;
        const unsigned word = m.words[task * 32 + lane];
        bool have = word != ~0u;
        const dataflow_task t = have ? dataflow_describe(m, word) : dataflow_task{};
        const unsigned g = t.kind ? 32 : m.width(t.end - t.begin), sub = lane & (g - 1);
        unsigned p = t.begin + sub, end = t.end;
        float rhs = 0, diagonal = 1;
        // Hide operand loads behind the dependency wait, including FP16 scaling.
        if (have && t.kind != 1) m.prepare(t.row, rhs, diagonal);
        float sum = 0;
        bool reload = have;
        unsigned pending = 0;
        unsigned cols[8];
        Value vals[8];
        float values[8];
        while (m.any(have)) {
            if (reload) {
                pending = 0;
                APXCHOL_UNROLL
                for (unsigned k = 0; k < 8; ++k) {
                    const unsigned pos = p + k * g;
                    const bool ok = have && pos < end;
                    cols[k] = ok ? (t.kind == 2 ? pos : m.col[pos]) : 0;
                    vals[k] = ok ? (t.kind == 2 ? Value(1.0f) : m.val[pos]) : Value(0.0f);
                    values[k] = 0;
                    if (ok) pending |= 1u << k;
                }
                reload = false;
            }
            unsigned bits[8];
            APXCHOL_UNROLL
            for (unsigned k = 0; k < 8; ++k)
                bits[k] = (pending >> k) & 1 ? m.load(cols[k]) : 0;
            APXCHOL_UNROLL
            for (unsigned k = 0; k < 8; ++k)
                if (((pending >> k) & 1) && bits[k] != 0x7f800001u) {
                    values[k] = m.as_float(bits[k]);
                    pending &= ~(1u << k);
                }
            const unsigned group_mask = (0xffffffffu >> (32 - g)) << (lane & ~(g - 1));
            const unsigned waiting = m.ballot(pending != 0);
            const bool ready = have && !(waiting & group_mask);
            if (ready) sum = dataflow_accumulate(vals, values, sum);
            float total = sum;
            for (unsigned o = 16; o; o /= 2) {
                const float other = m.shuffle(total, o);
                if (o < g) total += other;
            }
            if (ready) {
                p += g * 8;
                if (p - sub < end) reload = true;
                else {
                    if (sub == 0) m.publish(t.output, t.kind == 1 ? total : m.finish(rhs, diagonal, total));
                    have = false;
                }
            }
        }
    }
}
)
