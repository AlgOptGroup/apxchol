// Metal Shading Language source of the Apple GPU PCG (metal_solver.h),
// compiled at run time by src/metal_device.mm (newLibraryWithSource with
// MTLMathModeSafe; "#pragma METAL fp contract(off)" is prepended when the
// compiler accepts it). Checked in as one C++ raw string so that downstream
// builds need no generation step or offline Metal compiler.
R"apxchol_msl(
#include <metal_stdlib>
using namespace metal;
using namespace apxchol::detail::gpu_pcg;
using namespace apxchol::detail::metal_host;

struct Params {
    uint n;         // operator rows
    uint groups;    // reduction groups, ceil(n / 256)
    uint offset;    // first workgroup in this dependency stage
    uint count;     // number of dependency words to initialize
    uint mode;      // finalize mode
    uint iter;      // 1-based PCG iteration
    uint window;    // stagnation window, 0 = off
    uint m;         // factor dimension
};
static_assert(sizeof(Params) == 32, "Params mirrors detail::metal::params");

using SolveState = solve_state<df, float>;
static_assert(sizeof(SolveState) == 64, "SolveState mirrors detail::metal::solve_state");

constant bool kForward [[function_constant(0)]];
constant bool kOperatorDf [[function_constant(1)]];

constant uint kTreeRows = 256;
constant uint kTreeLanes = 16;

inline df fold_lanes(threadgroup const df* sh) {
    df t = df{};
    for (uint q = 0; q < kTreeLanes; ++q) t = df_add(t, sh[q]);
    return t;
}

struct pcg_math {
    using wide = df;
    using scalar = float;
    df add(df a, df b) const { return df_add(a, b); }
    df add_product(df sum, float a, float b) const { return df_add(sum, two_prod(a, b)); }
    df add_product(df sum, df a, float b) const { return df_add(sum, df_mul_f(a, b)); }
    df add_product(df sum, df a, df b) const { return df_add(sum, df_mul(a, b)); }
    df subtract_product(df a, df b, float c) const { return df_add(a, df_neg(df_mul_f(b, c))); }
    float direction(float z, float p, float beta) const { return fma(beta, p, z); }
    df add_entry(df sum, device const float* hi, device const float* lo, uint e, float p) const {
        return df_add(sum, kOperatorDf ? df_mul_f(df{hi[e], lo[e]}, p) : two_prod(hi[e], p));
    }
    df sum_row(df sum) const { return sum; }
};

inline void publish(df sum, device df* partial, threadgroup df* scratch, uint lane, uint group) {
    scratch[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) partial[group] = fold_lanes(scratch);
}

inline pcg_work<uint> tree_work(uint n, uint group, uint lane) {
    return {group * kTreeRows, (group + 1) * kTreeRows, kTreeLanes, lane, n};
}

kernel void spmv_pap(constant Params& P [[buffer(0)]],
                     device const uint* ptr [[buffer(1)]], device const uint* col [[buffer(2)]],
                     device const float* hi [[buffer(3)]], device const float* lo [[buffer(4)]],
                     device const float* p [[buffer(5)]], device df* ap [[buffer(6)]],
                     device df* partial [[buffer(7)]], device const SolveState& state [[buffer(8)]],
                     uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    threadgroup df scratch[kTreeLanes];
    const auto sum = pcg_spmv<1>(pcg_math{}, ptr, col, hi, p, ap, tree_work(P.n, group, lane), 0, state.active, lo);
    publish(sum, partial, scratch, lane, group);
}

kernel void update_xr(constant Params& P [[buffer(0)]],
                      device df* x [[buffer(1)]], device df* r [[buffer(2)]],
                      device const float* p [[buffer(3)]], device const df* ap [[buffer(4)]],
                      device df* partial [[buffer(5)]], device const SolveState& state [[buffer(6)]],
                      uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    threadgroup df scratch[kTreeLanes];
    const auto sums = pcg_update_xr(pcg_math{}, x, p, r, ap, state.alpha, tree_work(P.n, group, lane), state.active);
    publish(sums, partial, scratch, lane, group);
}

kernel void reduce(constant Params& P [[buffer(0)]],
                   device const df* r [[buffer(1)]], device const float* z [[buffer(2)]],
                   device df* partial [[buffer(3)]], device const SolveState& state [[buffer(4)]],
                   uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    threadgroup df scratch[kTreeLanes];
    const auto sum = pcg_reduce(pcg_math{}, r, z, tree_work(P.n, group, lane), kReduceRz, state.active);
    publish(sum, partial, scratch, lane, group);
}

struct recurrence_math {
    bool positive(df x) const { return x.hi > 0 && isfinite(x.hi); }
    bool valid_initial(df x) const { return isfinite(x.hi); }
    bool valid_residual(df x) const { return isfinite(x.hi); }
    bool valid_rz(df x) const { return isfinite(x.hi); }
    bool accept_ratio(float x) const { return isfinite(x); }
    float ratio(df x, df y) const { return df_div(x, y).hi; }
    bool converged(df x, df threshold) const { return df_lt(x, threshold); }
    bool stagnated(df previous, df current) const { return df_lt(df_mul_f(previous, 0.25f), current); }
};

// ---- one threadgroup of 16 lanes: fold the partials, then the PCG scalars ----
kernel void finalize(constant Params& P [[buffer(0)]],
                     device const df* partial [[buffer(1)]],
                     device SolveState& state [[buffer(2)]],
                     uint l [[thread_index_in_threadgroup]]) {
    threadgroup df sh[kTreeLanes];
    const pcg_work<uint> work{0, P.groups, kTreeLanes, l, P.groups};
    sh[l] = pcg_reduce(pcg_math{}, partial, nullptr, work, kReduceSum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (l == 0 && state.active)
        state = advance(recurrence_math{}, state, P.mode, fold_lanes(sh), P.iter, P.window);
}

kernel void p_update(constant Params& P [[buffer(0)]],
                     device float* p [[buffer(1)]], device const float* z [[buffer(2)]],
                     device const SolveState& state [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    pcg_update_p(pcg_math{}, p, z, state.beta, pcg_work<uint>{i, P.n, P.n, 0, P.n}, state.active);
}

// ---- self-test of the error-free transforms (metal_solver::available) ----
inline float4 pack_probe(df a, df b) { return float4(a.hi, a.lo, b.hi, b.lo); }
kernel void df_probe(device const float4* in [[buffer(0)]],
                     device float4* out [[buffer(1)]],
                     constant uint& count [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const float4 v = in[i];
    const df x = df{v.x, v.y}, y = df{v.z, v.w};
    out[4 * i + 0] = pack_probe(two_sum(x.hi, y.hi), two_prod(x.hi, y.hi));
    out[4 * i + 1] = pack_probe(df_add(x, y), df_mul(x, y));
    out[4 * i + 2] = pack_probe(df_mul_f(x, y.hi), df{fma(x.hi, y.hi, x.lo), fma(-x.hi, y.hi, x.lo) * y.lo});
    out[4 * i + 3] = pack_probe(df_div(x, y), df{(x.hi - y.hi) * y.lo, quick_two_sum(x.hi, x.lo).lo});
}
)apxchol_msl"
