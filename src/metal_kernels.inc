// Metal Shading Language source of the Apple GPU PCG (metal_solver.h),
// compiled at run time by src/metal_device.mm (newLibraryWithSource with
// MTLMathModeSafe; "#pragma METAL fp contract(off)" is prepended when the
// compiler accepts it). Checked in as one C++ raw string so that downstream
// builds need no generation step or offline Metal compiler.
R"apxchol_msl(
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;         // operator rows
    uint groups;    // reduction groups, ceil(n / 256)
    uint offset;    // step: first slot, or first level of a narrow run
    uint count;     // step: slot count, or end level of a narrow run
    uint mode;      // finalize mode
    uint iter;      // 1-based PCG iteration
    uint window;    // stagnation window, 0 = off
    uint lap;       // 1 = Laplacian: centre every preconditioner application
    uint m;         // factor dimension
    float inv_n_hi; // 1 / n as double-float
    float inv_n_lo;
};
static_assert(sizeof(Params) == 44, "Params mirrors detail::metal::params");

struct SolveState {
    float2 rz;      // r . z of the current direction
    float2 rr;      // recursive r . r at the last check
    float2 thr;     // stop iff rr < thr (host: (tol ||b|| s)^2)
    float2 prev;    // stagnation reference r . r
    float2 pap;     // last p . Ap
    float alpha;
    float beta;
    float mu_r;     // input centring of the preconditioner (0 for SDDM)
    float mu_z;     // output centring of the preconditioner (0 for SDDM)
    uint active;
    uint iters;
    uint stop;
    uint pad;
};
static_assert(sizeof(SolveState) == 72, "SolveState mirrors detail::metal::solve_state");

constant bool kForward [[function_constant(0)]];
constant bool kOperatorDf [[function_constant(1)]];
constant uint kReduceMode [[function_constant(2)]];

constant uint kTreeRows = 256;
constant uint kTreeLanes = 16;
constant uint kTreeSteps = 16;
constant uint kVirtualLanes = 32;

constant uint kStopTolerance = 1;
constant uint kStopBreakdown = 2;
constant uint kStopStagnation = 3;
constant uint kStopNonfinite = 4;

constant uint kFinalizeMuR = 0;
constant uint kFinalizeRzInit = 1;
constant uint kFinalizeMuZ = 2;
constant uint kFinalizeAlpha = 3;
constant uint kFinalizeCheck = 4;
constant uint kFinalizeRz = 5;

constant uint kReduceSumR = 0;
constant uint kReduceSumZ = 1;
constant uint kReduceRz = 2;

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

// ---- operator: Ap = A p in double-float, fused p . Ap over the fixed tree ----
kernel void spmv_pap(constant Params& P [[buffer(0)]],
                     device const uint* ptr [[buffer(1)]],
                     device const uint* col [[buffer(2)]],
                     device const float* ah [[buffer(3)]],
                     device const float* al [[buffer(4)]],
                     device const float* p [[buffer(5)]],
                     device float2* ap [[buffer(6)]],
                     device float2* partial [[buffer(7)]],
                     device const SolveState& state [[buffer(8)]],
                     uint l [[thread_index_in_threadgroup]],
                     uint g [[threadgroup_position_in_grid]]) {
    threadgroup float2 sh[kTreeLanes];
    float2 acc = float2(0.0f);
    if (state.active != 0) {
        for (uint s = 0; s < kTreeSteps; ++s) {
            const uint i = g * kTreeRows + s * kTreeLanes + l;
            if (i >= P.n) break;
            float2 y = float2(0.0f);
            const uint e1 = ptr[i + 1];
            for (uint e = ptr[i]; e < e1; ++e) {
                const float pv = p[col[e]];
                y = df_add(y, kOperatorDf ? df_mul_f(float2(ah[e], al[e]), pv) : two_prod(ah[e], pv));
            }
            ap[i] = y;
            acc = df_add(acc, df_mul_f(y, p[i]));
        }
    }
    sh[l] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (l == 0) partial[g] = fold_lanes(sh);
}

// ---- x += alpha p, r -= alpha Ap; r . r and sum(r) over the fixed tree ----
kernel void update_xr(constant Params& P [[buffer(0)]],
                      device float2* x [[buffer(1)]],
                      device float2* r [[buffer(2)]],
                      device const float* p [[buffer(3)]],
                      device const float2* ap [[buffer(4)]],
                      device float2* partial [[buffer(5)]],
                      device const SolveState& state [[buffer(6)]],
                      uint l [[thread_index_in_threadgroup]],
                      uint g [[threadgroup_position_in_grid]]) {
    threadgroup float2 sh[2 * kTreeLanes];
    float2 rr = float2(0.0f), rs = float2(0.0f);
    if (state.active != 0) {
        const float a = state.alpha;
        for (uint s = 0; s < kTreeSteps; ++s) {
            const uint i = g * kTreeRows + s * kTreeLanes + l;
            if (i >= P.n) break;
            x[i] = df_add(x[i], two_prod(a, p[i]));
            const float2 rv = df_add(r[i], df_neg(df_mul_f(ap[i], a)));
            r[i] = rv;
            rr = df_add(rr, df_mul(rv, rv));
            rs = df_add(rs, rv);
        }
    }
    sh[l] = rr;
    sh[kTreeLanes + l] = rs;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (l == 0) {
        partial[g] = fold_lanes(sh);
        partial[P.groups + g] = fold_lanes(sh + kTreeLanes);
    }
}

// ---- sum(r), sum(z) or r . (z - mu_z) over the fixed tree ----
kernel void reduce(constant Params& P [[buffer(0)]],
                   device const float2* r [[buffer(1)]],
                   device const float* z [[buffer(2)]],
                   device float2* partial [[buffer(3)]],
                   device const SolveState& state [[buffer(4)]],
                   uint l [[thread_index_in_threadgroup]],
                   uint g [[threadgroup_position_in_grid]]) {
    threadgroup float2 sh[kTreeLanes];
    float2 acc = float2(0.0f);
    if (state.active != 0) {
        const float mu = state.mu_z;
        for (uint s = 0; s < kTreeSteps; ++s) {
            const uint i = g * kTreeRows + s * kTreeLanes + l;
            if (i >= P.n) break;
            float2 term;
            if (kReduceMode == kReduceSumR) term = r[i];
            else if (kReduceMode == kReduceSumZ) term = float2(z[i], 0.0f);
            else term = df_mul_f(r[i], z[i] - mu);
            acc = df_add(acc, term);
        }
    }
    sh[l] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (l == 0) partial[g] = fold_lanes(sh);
}

// ---- one threadgroup of 16 lanes: fold the partials, then the PCG scalars ----
kernel void finalize(constant Params& P [[buffer(0)]],
                     device const float2* partial [[buffer(1)]],
                     device SolveState& state [[buffer(2)]],
                     uint l [[thread_index_in_threadgroup]]) {
    threadgroup float2 sh[2 * kTreeLanes];
    const uint G = P.groups;
    const uint slots = P.mode == kFinalizeCheck ? 2 : 1;
    for (uint k = 0; k < slots; ++k) {
        float2 acc = float2(0.0f);
        for (uint g = l; g < G; g += kTreeLanes) acc = df_add(acc, partial[k * G + g]);
        sh[k * kTreeLanes + l] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (l != 0) return;
    SolveState st = state;
    if (st.active == 0) return;
    const float2 t = fold_lanes(sh);
    const float2 inv_n = float2(P.inv_n_hi, P.inv_n_lo);
    if (P.mode == kFinalizeMuR) {
        st.mu_r = P.lap != 0 ? df_mul(t, inv_n).x : 0.0f;
    } else if (P.mode == kFinalizeMuZ) {
        st.mu_z = P.lap != 0 ? df_mul(t, inv_n).x : 0.0f;
    } else if (P.mode == kFinalizeRzInit) {
        if (!isfinite(t.x)) {
            st.active = 0;
            st.stop = kStopNonfinite;
        } else {
            st.rz = t;
            st.beta = 0.0f;
        }
    } else if (P.mode == kFinalizeAlpha) {
        st.pap = t;
        if (!(t.x > 0.0f) || !isfinite(t.x)) {
            // Breakdown: stop before the update; the iteration is not counted.
            st.active = 0;
            st.stop = kStopBreakdown;
        } else {
            const float a = df_div(st.rz, t).x;
            if (!isfinite(a)) {
                st.active = 0;
                st.stop = kStopNonfinite;
            } else {
                st.alpha = a;
            }
        }
    } else if (P.mode == kFinalizeCheck) {
        st.rr = t;
        st.iters = P.iter;
        if (!isfinite(t.x)) {
            st.active = 0;
            st.stop = kStopNonfinite;
        } else if (df_lt(t, st.thr)) {
            st.active = 0;
            st.stop = kStopTolerance;
        } else if (P.window != 0 && P.iter % P.window == 0) {
            if (df_lt(df_mul_f(st.prev, 0.25f), t)) {
                st.active = 0;
                st.stop = kStopStagnation;
            } else {
                st.prev = t;
            }
        }
        if (st.active != 0 && P.lap != 0)
            st.mu_r = df_mul(fold_lanes(sh + kTreeLanes), inv_n).x;
    } else if (P.mode == kFinalizeRz) {
        const float b = isfinite(t.x) ? df_div(t, st.rz).x : t.x;
        if (!isfinite(b)) {
            st.active = 0;
            st.stop = kStopNonfinite;
        } else {
            st.beta = b;
            st.rz = t;
        }
    }
    state = st;
}

// ---- triangular solves: forward rhs = r.hi - mu_r, backward in place on z ----
inline float row_rhs(uint i, device const float2* r, device const float* z, float mu) {
    return kForward ? r[i].x - mu : z[i];
}

inline float light_row(uint t, device const uint* rows, device const uint* ptr,
                       device const uint* col, device const float* val, device const float* dinv,
                       device const float2* r, device const float* z, float mu) {
    const uint i = rows[t];
    float s = row_rhs(i, r, z, mu);
    const uint e1 = ptr[t + 1];
    for (uint e = ptr[t]; e < e1; ++e) s = fma(-val[e], z[col[e]], s);
    return s * dinv[t];
}

// One thread per light row.
kernel void level(constant Params& P [[buffer(0)]],
                  device const uint* rows [[buffer(1)]],
                  device const uint* ptr [[buffer(2)]],
                  device const uint* col [[buffer(3)]],
                  device const float* val [[buffer(4)]],
                  device const float* dinv [[buffer(5)]],
                  device const float2* r [[buffer(6)]],
                  device float* z [[buffer(7)]],
                  device const SolveState& state [[buffer(8)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= P.count) return;
    if (state.active == 0) return;
    const uint t = P.offset + i;
    z[rows[t]] = light_row(t, rows, ptr, col, val, dinv, r, z, state.mu_r);
}

// One 32-thread group per heavy row.
kernel void level_heavy(constant Params& P [[buffer(0)]],
                        device const uint* rows [[buffer(1)]],
                        device const uint* ptr [[buffer(2)]],
                        device const uint* col [[buffer(3)]],
                        device const float* val [[buffer(4)]],
                        device const float* dinv [[buffer(5)]],
                        device const float2* r [[buffer(6)]],
                        device float* z [[buffer(7)]],
                        device const SolveState& state [[buffer(8)]],
                        uint l [[thread_index_in_threadgroup]],
                        uint g [[threadgroup_position_in_grid]]) {
    threadgroup float sh[kVirtualLanes];
    const uint t = P.offset + g;
    const bool act = state.active != 0;
    if (act) {
        const uint b = ptr[t], e1 = ptr[t + 1];
        float acc = 0.0f;
        for (uint e = b + l; e < e1; e += kVirtualLanes)
            acc = fma(val[e], z[col[e]], acc);
        sh[l] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (l == 0 && act) {
        float tot = sh[0];
        for (uint v = 1; v < kVirtualLanes; ++v) tot += sh[v];
        const uint i = rows[t];
        z[i] = (row_rhs(i, r, z, state.mu_r) - tot) * dinv[t];
    }
}

// A run of narrow levels [offset, count) (light rows only) in one
// threadgroup, a device-memory barrier between levels.
kernel void levels_narrow(constant Params& P [[buffer(0)]],
                          device const uint* rows [[buffer(1)]],
                          device const uint* ptr [[buffer(2)]],
                          device const uint* col [[buffer(3)]],
                          device const float* val [[buffer(4)]],
                          device const float* dinv [[buffer(5)]],
                          device const float2* r [[buffer(6)]],
                          device float* z [[buffer(7)]],
                          device const SolveState& state [[buffer(8)]],
                          device const uint* level_ptr [[buffer(9)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint tgs [[threads_per_threadgroup]]) {
    for (uint lv = P.offset; lv < P.count; ++lv) {
        const uint t0 = level_ptr[lv];
        const uint items = level_ptr[lv + 1] - t0;
        for (uint q = tid; q < items; q += tgs) {
            if (state.active == 0) continue;
            const uint t = t0 + q;
            z[rows[t]] = light_row(t, rows, ptr, col, val, dinv, r, z, state.mu_r);
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// ---- p = (z - mu_z) + beta p (p = z - mu_z where beta is 0) ----
kernel void p_update(constant Params& P [[buffer(0)]],
                     device float* p [[buffer(1)]],
                     device const float* z [[buffer(2)]],
                     device const SolveState& state [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= P.n) return;
    if (state.active == 0) return;
    const float zc = z[i] - state.mu_z;
    const float b = state.beta;
    p[i] = b == 0.0f ? zc : fma(b, p[i], zc);
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
