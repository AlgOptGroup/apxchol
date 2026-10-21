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
template<unsigned Columns = 1, unsigned RhsLanes = 1, class Memory>
APXCHOL_GPU_FUNCTION void dataflow_run(Memory m, unsigned lane, bool completed_real_rows = false) {
    static_assert(RhsLanes && RhsLanes <= 32 && !(RhsLanes & (RhsLanes - 1)));
    static_assert(Columns % RhsLanes == 0);
    constexpr unsigned SparseLanes = 32 / RhsLanes, LocalColumns = Columns / RhsLanes;
    const unsigned sparse_lane = lane / RhsLanes, first_column = lane % RhsLanes;
    using Value = typename Memory::stored_value;
    bool enabled[LocalColumns], active = false;
    for (unsigned q = 0; q < LocalColumns; ++q)
        active |= enabled[q] = m.enabled(first_column + q * RhsLanes);
    if (RhsLanes == 1) {
        if (!active) return;
    } else if (!m.any(active)) return;
    for (;;) {
        unsigned task = lane == 0 ? m.claim() : 0;
        task = m.broadcast(task);
        if (task >= m.batches) break;
        const unsigned word = m.words[task * SparseLanes + sparse_lane];
        bool have = word != ~0u;
        const dataflow_task t = have ? dataflow_describe(m, word) : dataflow_task{};
        // Finalizers still wait for partials produced by other SIMD groups.
        const bool completed = completed_real_rows && t.kind != 2;
        unsigned g;
        if (RhsLanes == 1)
            g = t.kind ? 32 : m.width(t.end - t.begin);
        else {
            const unsigned width = m.width(t.end - t.begin);
            g = t.kind || width > SparseLanes ? SparseLanes : width;
        }
        const unsigned sub = sparse_lane & (g - 1);
        unsigned p = t.begin + sub, end = t.end;
        float rhs[LocalColumns]{}, diagonal[LocalColumns]{}, sum[LocalColumns]{};
        if (have && t.kind != 1)
            for (unsigned q = 0; q < LocalColumns; ++q)
                if (enabled[q]) m.prepare(t.row, first_column + q * RhsLanes, rhs[q], diagonal[q]);
        bool reload = have;
        unsigned pending[LocalColumns]{};
        unsigned cols[8];
        Value vals[8];
        float values[LocalColumns][8];
        while (m.any(have)) {
            if (reload) {
                unsigned mask = 0;
                APXCHOL_UNROLL
                for (unsigned k = 0; k < 8; ++k) {
                    const unsigned pos = p + k * g;
                    const bool ok = have && pos < end;
                    cols[k] = ok ? (t.kind == 2 ? pos : m.col[pos]) : 0;
                    vals[k] = ok ? (t.kind == 2 ? Value(1.0f) : m.value(pos, t.row, cols[k])) : Value(0.0f);
                    for (unsigned q = 0; q < LocalColumns; ++q)
                        values[q][k] = completed && ok && enabled[q]
                            ? m.as_float(m.load(cols[k], first_column + q * RhsLanes)) : 0;
                    if (!completed && ok) mask |= 1u << k;
                }
                for (unsigned q = 0; q < LocalColumns; ++q) pending[q] = enabled[q] ? mask : 0;
                reload = false;
            }
            bool ready = have;
            if (!completed) {
                unsigned pending_any = 0;
                for (unsigned q = 0; q < LocalColumns; ++q) {
                    unsigned bits[8];
                    APXCHOL_UNROLL
                    for (unsigned k = 0; k < 8; ++k)
                        bits[k] = (pending[q] >> k) & 1 ? m.load(cols[k], first_column + q * RhsLanes) : 0;
                    APXCHOL_UNROLL
                    for (unsigned k = 0; k < 8; ++k)
                        if (((pending[q] >> k) & 1) && bits[k] != 0x7f800001u) {
                            values[q][k] = m.as_float(bits[k]);
                            pending[q] &= ~(1u << k);
                        }
                    pending_any |= pending[q];
                }
                const unsigned physical_width = g * RhsLanes;
                const unsigned group_mask =
                    (0xffffffffu >> (32 - physical_width)) << (lane & ~(physical_width - 1));
                const unsigned waiting = m.ballot(pending_any != 0);
                ready = have && !(waiting & group_mask);
            }
            const bool reduce = m.any(ready && g > 1 && p + g * 8 - sub >= end);
            float total[LocalColumns];
            for (unsigned q = 0; q < LocalColumns; ++q) {
                if (ready && enabled[q]) sum[q] = dataflow_accumulate(vals, values[q], sum[q]);
                total[q] = sum[q];
                for (unsigned o = reduce ? SparseLanes / 2 : 0; o; o /= 2) {
                    const float other = m.shuffle(total[q], o * RhsLanes);
                    if (o < g) total[q] += other;
                }
            }
            if (ready) {
                p += g * 8;
                if (p - sub < end) reload = true;
                else {
                    if (sub == 0)
                        for (unsigned q = 0; q < LocalColumns; ++q)
                            if (enabled[q]) m.publish(t.output, first_column + q * RhsLanes,
                                t.kind == 1 ? total[q] : m.finish(rhs[q], diagonal[q], total[q]));
                    have = false;
                }
            }
        }
    }
}
)
