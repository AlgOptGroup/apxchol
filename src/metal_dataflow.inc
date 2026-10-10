R"apxchol_msl(
struct dataflow_memory {
    using stored_value = float;
    float accumulate(thread const float* v, thread const float* y, float sum) const {
        for (uint k = 0; k < 8; ++k) sum = fma(v[k], y[k], sum);
        return sum;
    }
    device const uint* ptr;
    device const uint* col;
    device const float* val;
    device const uint* words;
    device atomic_uint* tags;
    threadgroup atomic_uint* ticket;
    device const float2* r;
    device float* z;
    uint batches;
    float mu;
    device const uint4* segments;
    uint n;
    dataflow_task describe(uint word) const {
        if (word >> 31) {
            const uint4 s = segments[word & 0x7fffffffu];
            return {s.x & 0x7fffffffu, s.y, s.z, s.w, (s.x >> 31) ? 2u : 1u};
        }
        return {word, begin(word), end(word), word, 0};
    }
    uint width(uint size) const { return 1u << min(5u, 32u - clz((size ? size - 1 : 0u) / 8u)); }
    uint ballot(bool x) const { return uint(ulong(simd_ballot(x))); }
    uint claim() const { return atomic_fetch_add_explicit(ticket, 1u, memory_order_relaxed); }
    uint broadcast(uint x) const { return simd_broadcast(x, 0); }
    bool any(bool x) const { return simd_any(x); }
    float shuffle(float x, uint o) const { return simd_shuffle_xor(x, o); }
    uint begin(uint row) const { return ptr[row] + uint(!kForward); }
    uint end(uint row) const { return ptr[row + 1] - uint(kForward); }
    uint load(uint row) const { return atomic_load_explicit(tags + row, memory_order_relaxed); }
    float as_float(uint x) const { return as_type<float>(x); }
    void prepare(uint row, thread float& rhs, thread float& diagonal) const {
        rhs = kForward ? r[row].x - mu : z[row];
        diagonal = val[kForward ? ptr[row + 1] - 1 : ptr[row]];
    }
    float finish(float rhs, float diagonal, float sum) const {
        return precise::divide(rhs - sum, diagonal);
    }
    void publish(uint row, float x) const {
        if (row < n) z[row] = x;
        atomic_store_explicit(tags + row, isnan(x) ? 0x7fc00000u : as_type<uint>(x), memory_order_relaxed);
    }
};
kernel void dataflow_init(constant Params& P [[buffer(0)]],
                        device const uint* ptr [[buffer(2)]], device const float* val [[buffer(4)]],
                        device const float2* r [[buffer(6)]], device float* z [[buffer(7)]],
                        device const SolveState& state [[buffer(8)]], device atomic_uint* tags [[buffer(10)]],
                        uint row [[thread_position_in_grid]]) {
    if (row >= P.count || !state.active) return;
    dataflow_memory m{ptr, nullptr, val, nullptr, tags, nullptr, r, z, 0, state.mu_r, nullptr, P.m};
    if (row < P.m && m.begin(row) == m.end(row)) {
        float rhs, diagonal;
        m.prepare(row, rhs, diagonal);
        m.publish(row, m.finish(rhs, diagonal, 0));
    }
    else atomic_store_explicit(tags + row, 0x7f800001u, memory_order_relaxed);
}
kernel void dataflow_solve(constant Params& P [[buffer(0)]],
                        device const uint* ptr [[buffer(2)]], device const uint* col [[buffer(3)]],
                        device const float* val [[buffer(4)]], device const float2* r [[buffer(6)]],
                        device float* z [[buffer(7)]], device const SolveState& state [[buffer(8)]],
                        device const uint* words [[buffer(9)]], device atomic_uint* tags [[buffer(10)]],
                        device const uint* blocks [[buffer(11)]], device const uint4* segments [[buffer(12)]], uint group [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    if (!state.active) return;
    threadgroup atomic_uint ticket;
    const uint block = P.offset + group;
    if (tid == 0) atomic_store_explicit(&ticket, blocks[block], memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dataflow_run(dataflow_memory{ptr, col, val, words, tags, &ticket, r, z, blocks[block + 1], state.mu_r, segments, P.m}, lane);
}
)apxchol_msl"
