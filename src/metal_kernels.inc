// Metal Shading Language source of the Apple GPU PCG (metal_solver.h),
// compiled at run time by src/metal_device.mm (newLibraryWithSource with
// MTLMathModeSafe; "#pragma METAL fp contract(off)" is prepended when the
// compiler accepts it). Checked in as one C++ raw string so that downstream
// builds need no generation step or offline Metal compiler.
R"apxchol_msl(
#include <metal_stdlib>
using namespace metal;
using namespace apxchol::detail::gpu_pcg;

struct Params {
    uint n;         // operator rows
    uint groups;    // reduction groups, ceil(n / 256)
    uint offset;    // first workgroup in this dependency stage
    uint count;     // number of dependency words to initialize
    uint mode;      // finalize mode
    uint iter;      // 1-based PCG iteration
    uint window;    // stagnation window, 0 = off
    uint lap;       // 1 = Laplacian: centre every preconditioner application
    uint m;         // factor dimension
    float inv_n_hi; // 1 / n as double-float
    float inv_n_lo;
};
static_assert(sizeof(Params) == 44, "Params mirrors detail::metal::params");

using SolveState = solve_state<float2, float>;
static_assert(sizeof(SolveState) == 72, "SolveState mirrors detail::metal::solve_state");

constant bool kForward [[function_constant(0)]];
constant bool kOperatorDf [[function_constant(1)]];
constant uint kReduceMode [[function_constant(2)]];

constant uint kTreeRows = 256;
constant uint kTreeLanes = 16;

// ---- double-float error-free transforms (no contraction, no fast math) ----
inline float2 two_sum(float a, float b) {
    float s = a + b;
    float bb = s - a;
    float e = (a - (s - bb)) + (b - bb);
    return float2(s, e);
}

inline float2 quick_two_sum(float a, float b) {
    float s = a + b;
    float e = b - (s - a);
    return float2(s, e);
}

inline float2 df_add(float2 a, float2 b) {
    float2 s = two_sum(a.x, b.x);
    float2 t = two_sum(a.y, b.y);
    s.y += t.x;
    s = quick_two_sum(s.x, s.y);
    s.y += t.y;
    return quick_two_sum(s.x, s.y);
}

inline float2 df_neg(float2 a) { return float2(-a.x, -a.y); }

inline float2 two_prod(float a, float b) {
    float p = a * b;
    float e = fma(a, b, -p);
    return float2(p, e);
}

inline float2 df_mul(float2 a, float2 b) {
    float p = a.x * b.x;
    float e = fma(a.x, b.x, -p);
    e = fma(a.x, b.y, e);
    e = fma(a.y, b.x, e);
    return quick_two_sum(p, e);
}

inline float2 df_mul_f(float2 a, float b) {
    float p = a.x * b;
    float e = fma(a.x, b, -p);
    e = fma(a.y, b, e);
    return quick_two_sum(p, e);
}

inline float2 df_div(float2 a, float2 b) {
    float q1 = a.x / b.x;
    float2 r = df_add(a, df_neg(df_mul_f(b, q1)));
    float q2 = r.x / b.x;
    return quick_two_sum(q1, q2);
}

inline bool df_lt(float2 a, float2 b) { return a.x < b.x || (a.x == b.x && a.y < b.y); }

inline float2 fold_lanes(threadgroup const float2* sh) {
    float2 t = float2(0.0f);
    for (uint q = 0; q < kTreeLanes; ++q) t = df_add(t, sh[q]);
    return t;
}

struct pcg_math {
    using wide = float2;
    using scalar = float;
    float2 add(float2 a, float2 b) const { return df_add(a, b); }
    float2 from_scalar(float a) const { return float2(a, 0); }
    float2 add_product(float2 sum, float a, float b) const { return df_add(sum, two_prod(a, b)); }
    float2 add_product(float2 sum, float2 a, float b) const { return df_add(sum, df_mul_f(a, b)); }
    float2 add_product(float2 sum, float2 a, float2 b) const { return df_add(sum, df_mul(a, b)); }
    float2 subtract_product(float2 a, float2 b, float c) const { return df_add(a, df_neg(df_mul_f(b, c))); }
    float direction(float z, float p, float beta) const { return beta == 0 ? z : fma(beta, p, z); }
    float2 add_entry(float2 sum, device const float* hi, device const float* lo, uint e, float p) const {
        return df_add(sum, kOperatorDf ? df_mul_f(float2(hi[e], lo[e]), p) : two_prod(hi[e], p));
    }
    float2 sum_row(float2 sum) const { return sum; }
};

inline void publish(float2 sum, float2 extra, device float2* partial,
                    threadgroup float2* scratch, uint lane, uint group, uint groups, bool two_sums = false) {
    scratch[lane] = sum;
    if (two_sums) scratch[kTreeLanes + lane] = extra;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        partial[group] = fold_lanes(scratch);
        if (two_sums) partial[groups + group] = fold_lanes(scratch + kTreeLanes);
    }
}

inline pcg_work<uint> tree_work(uint n, uint group, uint lane) {
    return {group * kTreeRows, (group + 1) * kTreeRows, kTreeLanes, lane, n};
}

kernel void spmv_pap(constant Params& P [[buffer(0)]],
                     device const uint* ptr [[buffer(1)]], device const uint* col [[buffer(2)]],
                     device const float* hi [[buffer(3)]], device const float* lo [[buffer(4)]],
                     device const float* p [[buffer(5)]], device float2* ap [[buffer(6)]],
                     device float2* partial [[buffer(7)]], device const SolveState& state [[buffer(8)]],
                     uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    threadgroup float2 scratch[kTreeLanes];
    const auto sum = pcg_spmv<1>(pcg_math{}, ptr, col, hi, p, ap, tree_work(P.n, group, lane), 0, state.active, lo);
    publish(sum, float2(0), partial, scratch, lane, group, P.groups);
}

kernel void update_xr(constant Params& P [[buffer(0)]],
                      device float2* x [[buffer(1)]], device float2* r [[buffer(2)]],
                      device const float* p [[buffer(3)]], device const float2* ap [[buffer(4)]],
                      device float2* partial [[buffer(5)]], device const SolveState& state [[buffer(6)]],
                      uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    threadgroup float2 scratch[2 * kTreeLanes];
    const auto sums = pcg_update_xr(pcg_math{}, x, p, r, ap, state.alpha, tree_work(P.n, group, lane), true, state.active);
    publish(sums.norm, sums.sum, partial, scratch, lane, group, P.groups, true);
}

kernel void reduce(constant Params& P [[buffer(0)]],
                   device const float2* r [[buffer(1)]], device const float* z [[buffer(2)]],
                   device float2* partial [[buffer(3)]], device const SolveState& state [[buffer(4)]],
                   uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    threadgroup float2 scratch[kTreeLanes];
    const auto sum = pcg_reduce(pcg_math{}, r, z, tree_work(P.n, group, lane), kReduceMode, state.mu_z, state.active);
    publish(sum, float2(0), partial, scratch, lane, group, P.groups);
}

struct recurrence_math {
    bool positive(float2 x) const { return x.x > 0 && isfinite(x.x); }
    bool valid_residual(float2 x) const { return isfinite(x.x); }
    bool valid_rz(float2 x) const { return isfinite(x.x); }
    bool accept_ratio(float x) const { return isfinite(x); }
    float ratio(float2 x, float2 y) const { return df_div(x, y).x; }
    bool converged(float2 x, float2 threshold) const { return df_lt(x, threshold); }
    bool stagnated(float2 previous, float2 current) const { return df_lt(df_mul_f(previous, 0.25f), current); }
};

// ---- one threadgroup of 16 lanes: fold the partials, then the PCG scalars ----
kernel void finalize(constant Params& P [[buffer(0)]],
                     device const float2* partial [[buffer(1)]],
                     device SolveState& state [[buffer(2)]],
                     uint l [[thread_index_in_threadgroup]]) {
    threadgroup float2 sh[2 * kTreeLanes];
    const uint G = P.groups;
    const uint slots = P.mode == kFinalizeCheck ? 2 : 1;
    for (uint k = 0; k < slots; ++k) {
        const pcg_work<uint> work{0, G, kTreeLanes, l, G};
        sh[k * kTreeLanes + l] = pcg_reduce(pcg_math{}, partial + k * G, nullptr, work, kReduceSumR);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (l != 0 || !state.active) return;
    const float2 t = fold_lanes(sh), inv_n = float2(P.inv_n_hi, P.inv_n_lo);
    if (P.mode == kFinalizeMuR || P.mode == kFinalizeMuZ) {
        const float mean = P.lap ? df_mul(t, inv_n).x : 0;
        if (P.mode == kFinalizeMuR) state.mu_r = mean;
        else state.mu_z = mean;
    } else if (P.mode == kFinalizeRzInit) {
        if (!isfinite(t.x)) { state.active = 0; state.stop = kStopNonfinite; }
        else { state.rz = t; state.beta = 0; }
    } else {
        state = advance(recurrence_math{}, state, P.mode, t, P.iter, P.window);
        if (P.mode == kFinalizeCheck && state.active && P.lap)
            state.mu_r = df_mul(fold_lanes(sh + kTreeLanes), inv_n).x;
    }
}

kernel void p_update(constant Params& P [[buffer(0)]],
                     device float* p [[buffer(1)]], device const float* z [[buffer(2)]],
                     device const SolveState& state [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    pcg_update_p(pcg_math{}, p, z, state.beta, pcg_work<uint>{i, P.n, P.n, 0, P.n}, state.mu_z, state.active);
}

// ---- self-test of the error-free transforms (metal_solver::available) ----
kernel void df_probe(device const float4* in [[buffer(0)]],
                     device float4* out [[buffer(1)]],
                     constant uint& count [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const float4 v = in[i];
    const float2 x = float2(v.x, v.y), y = float2(v.z, v.w);
    out[4 * i + 0] = float4(two_sum(x.x, y.x), two_prod(x.x, y.x));
    out[4 * i + 1] = float4(df_add(x, y), df_mul(x, y));
    out[4 * i + 2] = float4(df_mul_f(x, y.x), fma(x.x, y.x, x.y), fma(-x.x, y.x, x.y) * y.y);
    out[4 * i + 3] = float4(df_div(x, y), (x.x - y.x) * y.y, quick_two_sum(x.x, x.y).y);
}
)apxchol_msl"
