R"apxchol_msl(
struct dataflow_memory {
    using stored_value = float;
    device const uint* ptr;
    device const uint* col;
    device const float* val;
    device const uint* words;
    device atomic_uint* tags;
    threadgroup atomic_uint* ticket;
    device const df* r;
    device float* z;
    uint batches;
    device const dataflow_segment* segments;
    uint n;
    bool reverse;
    uint width(uint size) const { return 1u << min(5u, 32u - clz((size ? size - 1 : 0u) / 8u)); }
    uint ballot(bool x) const { return uint(ulong(simd_ballot(x))); }
    uint claim() const { return atomic_fetch_add_explicit(ticket, 1u, memory_order_relaxed); }
    uint broadcast(uint x) const { return simd_broadcast(x, 0); }
    bool any(bool x) const { return simd_any(x); }
    float shuffle(float x, uint o) const { return simd_shuffle_xor(x, o); }
    uint load(uint row) const { return atomic_load_explicit(tags + row, memory_order_relaxed); }
    float as_float(uint x) const { return as_type<float>(x); }
    void reset(uint row) const { atomic_store_explicit(tags + row, 0x7f800001u, memory_order_relaxed); }
    void prepare(uint row, thread float& rhs, thread float& diagonal) const {
        rhs = kForward ? r[row].hi : z[row];
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
                        device const uint* ptr [[buffer(13)]], device const float* val [[buffer(15)]],
                        device const df* r [[buffer(2)]], device float* z [[buffer(5)]],
                        device const SolveState& state [[buffer(7)]], device atomic_uint* tags [[buffer(12)]],
                        uint row [[thread_position_in_grid]]) {
    if (row >= P.count || !state.active) return;
    dataflow_initialize(dataflow_memory{ptr, nullptr, val, nullptr, tags, nullptr, r, z, 0, nullptr, P.m, !kForward}, row);
}
kernel void dataflow_solve(constant Params& P [[buffer(0)]],
                        device const uint* ptr [[buffer(13)]], device const uint* col [[buffer(14)]],
                        device const float* val [[buffer(15)]], device const df* r [[buffer(2)]],
                        device float* z [[buffer(5)]], device const SolveState& state [[buffer(7)]],
                        device const uint* words [[buffer(16)]], device atomic_uint* tags [[buffer(12)]],
                        device const uint* blocks [[buffer(17)]], device const dataflow_segment* segments [[buffer(18)]], uint group [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    if (!state.active) return;
    threadgroup atomic_uint ticket;
    const uint block = P.offset + group;
    if (tid == 0) atomic_store_explicit(&ticket, blocks[block], memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dataflow_run(dataflow_memory{ptr, col, val, words, tags, &ticket, r, z, blocks[block + 1], segments, P.m, !kForward}, lane);
}
)apxchol_msl"
