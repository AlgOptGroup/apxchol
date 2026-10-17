APXCHOL_GPU_CODE(
namespace resident_finalize {
enum Stage : unsigned {
    stage_initialize = 0,
    stage_prepare = 1,
    stage_emit = 2,
    stage_count_transpose = 3,
    stage_initialize_cursor = 4,
    stage_emit_transpose = 5,
};

struct Metadata { float scale, diagonal; };
struct Row { unsigned count; float scale, diagonal; };
struct Shape { unsigned n, m; float drop_hi, drop_lo; };

template<class Memory>
APXCHOL_GPU_FUNCTION void permutation(Memory s, unsigned row) {
    if (row < s.shape.n) s.perm[s.columns[row].node] = row;
}

// Raw append positions also reserve one diagonal per preceding column.
// Grounding is applied before measuring scales and compensated dropping.
template<class Memory, class Math>
APXCHOL_GPU_FUNCTION void prepare(Memory s, Math math, unsigned row) {
    if (row >= s.shape.m) return;
    const auto col = s.columns[row];
    auto entries = s.scratch + col.begin + row;
    unsigned count = 0;
    float scale = 0;
    for (unsigned p = 0; p < col.count; ++p) {
        const auto edge = s.factors[col.begin + p];
        const unsigned other = s.perm[edge.node];
        if (other >= s.shape.m) continue;
        if (other <= row) { s.fail(1); return; }
        entries[count++] = {other, -edge.value};
        scale = math.maximum(scale, edge.value);
    }
    if (scale == 0) scale = 1;
    if (s.fp16 && (!math.finite_float(1.0f / scale) || !math.finite_float(col.diagonal / scale)))
        scale = 1;
    auto dropped = math.from_float(0), kept = dropped;
    unsigned written = 0;
    for (unsigned p = 0; p < count; ++p) {
        const float v = entries[p].value;
        const bool keep = !(s.shape.drop_hi > 0) ||
            (!math.less(math.from_float(math.absolute(v)),
                        math.multiply(math.drop(s.shape), math.from_float(scale))) &&
             (s.fp16 ? !math.flushes(v / scale) : v != 0));
        if (keep) {
            kept = math.add(kept, math.from_float(-v));
            entries[written++] = entries[p];
        } else dropped = math.add(dropped, math.from_float(v));
    }
    if (math.to_float(dropped) != 0 && math.to_float(kept) > 0) {
        const auto per_abs = math.divide(dropped, kept);
        for (unsigned p = 0; p < written; ++p) {
            const float v = entries[p].value;
            entries[p].value = math.to_float(math.add(math.from_float(v),
                math.multiply(per_abs, math.from_float(-v))));
        }
    }
    s.rows[row] = {written + 1, scale, col.diagonal};
    s.ptr[row] = written + 1;
}

template<class Memory, class Math>
APXCHOL_GPU_FUNCTION void emit(Memory s, Math math, unsigned row) {
    if (row >= s.shape.m) return;
    const auto col = s.columns[row];
    const auto info = s.rows[row];
    const auto entries = s.scratch + col.begin + row;
    const unsigned begin = s.ptr[row];
    s.idx[begin] = row;
    s.val[begin] = math.store(info.diagonal, info.scale, s.fp16);
    auto residual = math.from_float(0);
    for (unsigned p = 0; p + 1 < info.count; ++p) {
        const auto edge = entries[p];
        const auto stored = math.store(edge.value, info.scale, s.fp16);
        s.idx[begin + p + 1] = edge.node;
        s.val[begin + p + 1] = stored;
        if (s.fp16) residual = math.add(residual, math.subtract(
            math.divide(math.from_float(edge.value), math.from_float(info.scale)),
            math.from_float(float(stored))));
    }
    float diagonal = info.diagonal;
    if (s.fp16) {
        const float scaled = math.to_float(math.add(math.from_float(diagonal / info.scale), residual));
        diagonal = scaled * info.scale;
    }
    s.metadata[row] = {info.scale, diagonal};
    const float magnitude = math.absolute(diagonal);
    if (!math.finite_float(diagonal) || magnitude < 0x1p-100f || magnitude > 0x1p100f) s.fail(2);
}

template<class Memory>
APXCHOL_GPU_FUNCTION void transpose_count(Memory s, unsigned row) {
    if (row >= s.shape.m) return;
    for (unsigned p = s.ptr[row]; p < s.ptr[row + 1]; ++p) s.increment_count(s.idx[p]);
}

template<class Memory>
APXCHOL_GPU_FUNCTION void transpose_emit(Memory s, unsigned row) {
    if (row >= s.shape.m) return;
    const unsigned begin = s.ptr[row], end = s.ptr[row + 1];
    const unsigned diagonal = s.transpose_ptr[row + 1] - 1;
    s.transpose_idx[diagonal] = row;
    s.transpose_val[diagonal] = s.val[begin];
    for (unsigned p = begin + 1; p < end; ++p) {
        const unsigned target = s.reserve(s.idx[p]);
        s.transpose_idx[target] = row;
        s.transpose_val[target] = s.val[p];
    }
}
}
)
