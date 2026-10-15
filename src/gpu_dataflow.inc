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
template<unsigned Columns = 1, class Memory>
APXCHOL_GPU_FUNCTION void dataflow_initialize(Memory m, unsigned row) {
    for (unsigned q = 0; q < Columns; ++q) {
        if (!m.enabled(q)) continue;
        if (row < m.n && m.ptr[row + 1] - m.ptr[row] == 1) {
            float rhs, diagonal;
            m.prepare(row, q, rhs, diagonal);
            m.publish(row, q, m.finish(rhs, diagonal, 0));
        } else m.reset(row, q);
    }
}
template<unsigned Columns = 1, class Memory>
APXCHOL_GPU_FUNCTION void dataflow_run(Memory m, unsigned lane) {
    using Value = typename Memory::stored_value;
    bool enabled[Columns], active = false;
    for (unsigned q = 0; q < Columns; ++q) active |= enabled[q] = m.enabled(q);
    if (!active) return;
    for (;;) {
        unsigned task = lane == 0 ? m.claim() : 0;
        task = m.broadcast(task);
        if (task >= m.batches) break;
        const unsigned word = m.words[task * 32 + lane];
        bool have = word != ~0u;
        const dataflow_task t = have ? dataflow_describe(m, word) : dataflow_task{};
        const unsigned g = t.kind ? 32 : m.width(t.end - t.begin), sub = lane & (g - 1);
        unsigned p = t.begin + sub, end = t.end;
        float rhs[Columns]{}, diagonal[Columns]{}, sum[Columns]{};
        if (have && t.kind != 1)
            for (unsigned q = 0; q < Columns; ++q)
                if (enabled[q]) m.prepare(t.row, q, rhs[q], diagonal[q]);
        bool reload = have;
        unsigned pending[Columns]{};
        unsigned cols[8];
        Value vals[8];
        float values[Columns][8];
        while (m.any(have)) {
            if (reload) {
                unsigned mask = 0;
                APXCHOL_UNROLL
                for (unsigned k = 0; k < 8; ++k) {
                    const unsigned pos = p + k * g;
                    const bool ok = have && pos < end;
                    cols[k] = ok ? (t.kind == 2 ? pos : m.col[pos]) : 0;
                    vals[k] = ok ? (t.kind == 2 ? Value(1.0f) : m.value(pos, t.row, cols[k])) : Value(0.0f);
                    for (unsigned q = 0; q < Columns; ++q) values[q][k] = 0;
                    if (ok) mask |= 1u << k;
                }
                for (unsigned q = 0; q < Columns; ++q) pending[q] = enabled[q] ? mask : 0;
                reload = false;
            }
            unsigned pending_any = 0;
            for (unsigned q = 0; q < Columns; ++q) {
                unsigned bits[8];
                APXCHOL_UNROLL
                for (unsigned k = 0; k < 8; ++k)
                    bits[k] = (pending[q] >> k) & 1 ? m.load(cols[k], q) : 0;
                APXCHOL_UNROLL
                for (unsigned k = 0; k < 8; ++k)
                    if (((pending[q] >> k) & 1) && bits[k] != 0x7f800001u) {
                        values[q][k] = m.as_float(bits[k]);
                        pending[q] &= ~(1u << k);
                    }
                pending_any |= pending[q];
            }
            const unsigned group_mask = (0xffffffffu >> (32 - g)) << (lane & ~(g - 1));
            const unsigned waiting = m.ballot(pending_any != 0);
            const bool ready = have && !(waiting & group_mask);
            float total[Columns];
            for (unsigned q = 0; q < Columns; ++q) {
                if (ready && enabled[q]) sum[q] = dataflow_accumulate(vals, values[q], sum[q]);
                total[q] = sum[q];
                for (unsigned o = 16; o; o /= 2) {
                    const float other = m.shuffle(total[q], o);
                    if (o < g) total[q] += other;
                }
            }
            if (ready) {
                p += g * 8;
                if (p - sub < end) reload = true;
                else {
                    if (sub == 0)
                        for (unsigned q = 0; q < Columns; ++q)
                            if (enabled[q]) m.publish(t.output, q,
                                t.kind == 1 ? total[q] : m.finish(rhs[q], diagonal[q], total[q]));
                    have = false;
                }
            }
        }
    }
}
)
