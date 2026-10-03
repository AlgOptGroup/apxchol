// Metal Shading Language source of the Apple GPU block PCG (metal_solver.h),
// compiled at run time by src/metal_device.mm (newLibraryWithSource with
// MTLMathModeSafe; "#pragma METAL fp contract(off)" is prepended when the
// compiler accepts it). Checked in as one C++ raw string so that downstream
// builds need no generation step or offline Metal compiler.
//
// Blocks are node-major: entry (row i, column c) of a block of kc <= 64
// right-hand sides lives at i * kc + c. Everything runs in the factor's
// permuted space. x, r and Ap are double-float (hi + lo fp32 pairs, about 48
// bits) and so is the operator when some value is not fp32-exact; p, z and
// the dropped factor are fp32. A column whose state is inactive is frozen:
// every kernel skips it.
//
// Every per-column reduction uses one fixed tree that depends only on n:
// threadgroup g covers rows [256g, 256g + 256); its thread (c, l), l < 16,
// folds rows 256g + 16s + l for s = 0..15 in sequence; lanes are folded 0..15
// in order into partial[g][c]; the finalize threadgroup's lane l folds
// partials l, l + 16, ... in order, then lanes 0..15. Neither the block width
// kc nor a column's position nor the other columns change a column's bits.
// Heavy triangular rows use 32 fixed virtual lanes for the same reason.
R"apxchol_msl(
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;         // operator rows
    uint kc;        // columns of this batch (the block stride)
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
static_assert(sizeof(Params) == 48, "Params mirrors detail::metal::params");

struct ColState {
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
static_assert(sizeof(ColState) == 72, "ColState mirrors detail::metal::column_state");

constant bool kForward [[function_constant(0)]];
constant bool kOperatorDf [[function_constant(1)]];
constant uint kReduceMode [[function_constant(2)]];

constant uint kTreeRows = 256;
constant uint kTreeLanes = 16;
constant uint kTreeSteps = 16;
constant uint kVirtualLanes = 32;
constant uint kMaxColumns = 64;

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

inline float2 fold_lanes(threadgroup const float2* sh, uint c, uint kc) {
    float2 t = float2(0.0f);
    for (uint q = 0; q < kTreeLanes; ++q) t = df_add(t, sh[q * kc + c]);
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
                     device const ColState* cs [[buffer(8)]],
                     uint2 tid [[thread_position_in_threadgroup]],
                     uint2 tg [[threadgroup_position_in_grid]]) {
    threadgroup float2 sh[kTreeLanes * kMaxColumns];
    const uint c = tid.x, l = tid.y, kc = P.kc, g = tg.x;
    float2 acc = float2(0.0f);
    if (cs[c].active != 0) {
        for (uint s = 0; s < kTreeSteps; ++s) {
            const uint i = g * kTreeRows + s * kTreeLanes + l;
            if (i >= P.n) break;
            float2 y = float2(0.0f);
            const uint e1 = ptr[i + 1];
            for (uint e = ptr[i]; e < e1; ++e) {
                const float pv = p[col[e] * kc + c];
                y = df_add(y, kOperatorDf ? df_mul_f(float2(ah[e], al[e]), pv) : two_prod(ah[e], pv));
            }
            const uint o = i * kc + c;
            ap[o] = y;
            acc = df_add(acc, df_mul_f(y, p[o]));
        }
    }
    sh[l * kc + c] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (l == 0) partial[g * kc + c] = fold_lanes(sh, c, kc);
}

// ---- x += alpha p, r -= alpha Ap; r . r and sum(r) over the fixed tree ----
kernel void update_xr(constant Params& P [[buffer(0)]],
                      device float2* x [[buffer(1)]],
                      device float2* r [[buffer(2)]],
                      device const float* p [[buffer(3)]],
                      device const float2* ap [[buffer(4)]],
                      device float2* partial [[buffer(5)]],
                      device const ColState* cs [[buffer(6)]],
                      uint2 tid [[thread_position_in_threadgroup]],
                      uint2 tg [[threadgroup_position_in_grid]]) {
    threadgroup float2 sh[2 * kTreeLanes * kMaxColumns];
    const uint c = tid.x, l = tid.y, kc = P.kc, g = tg.x;
    float2 rr = float2(0.0f), rs = float2(0.0f);
    if (cs[c].active != 0) {
        const float a = cs[c].alpha;
        for (uint s = 0; s < kTreeSteps; ++s) {
            const uint i = g * kTreeRows + s * kTreeLanes + l;
            if (i >= P.n) break;
            const uint o = i * kc + c;
            x[o] = df_add(x[o], two_prod(a, p[o]));
            const float2 rv = df_add(r[o], df_neg(df_mul_f(ap[o], a)));
            r[o] = rv;
            rr = df_add(rr, df_mul(rv, rv));
            rs = df_add(rs, rv);
        }
    }
    sh[l * kc + c] = rr;
    sh[(kTreeLanes + l) * kc + c] = rs;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (l == 0) {
        partial[g * kc + c] = fold_lanes(sh, c, kc);
        partial[(P.groups + g) * kc + c] = fold_lanes(sh + kTreeLanes * kc, c, kc);
    }
}

// ---- sum(r), sum(z) or r . (z - mu_z) over the fixed tree ----
kernel void reduce(constant Params& P [[buffer(0)]],
                   device const float2* r [[buffer(1)]],
                   device const float* z [[buffer(2)]],
                   device float2* partial [[buffer(3)]],
                   device const ColState* cs [[buffer(4)]],
                   uint2 tid [[thread_position_in_threadgroup]],
                   uint2 tg [[threadgroup_position_in_grid]]) {
    threadgroup float2 sh[kTreeLanes * kMaxColumns];
    const uint c = tid.x, l = tid.y, kc = P.kc, g = tg.x;
    float2 acc = float2(0.0f);
    if (cs[c].active != 0) {
        const float mu = cs[c].mu_z;
        for (uint s = 0; s < kTreeSteps; ++s) {
            const uint i = g * kTreeRows + s * kTreeLanes + l;
            if (i >= P.n) break;
            const uint o = i * kc + c;
            float2 term;
            if (kReduceMode == kReduceSumR) term = r[o];
            else if (kReduceMode == kReduceSumZ) term = float2(z[o], 0.0f);
            else term = df_mul_f(r[o], z[o] - mu);
            acc = df_add(acc, term);
        }
    }
    sh[l * kc + c] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (l == 0) partial[g * kc + c] = fold_lanes(sh, c, kc);
}

// ---- one threadgroup (kc, 16): fold the partials, then the PCG scalars ----
kernel void finalize(constant Params& P [[buffer(0)]],
                     device const float2* partial [[buffer(1)]],
                     device ColState* cs [[buffer(2)]],
                     uint2 tid [[thread_position_in_threadgroup]]) {
    threadgroup float2 sh[2 * kTreeLanes * kMaxColumns];
    const uint c = tid.x, l = tid.y, kc = P.kc, G = P.groups;
    const uint slots = P.mode == kFinalizeCheck ? 2 : 1;
    for (uint k = 0; k < slots; ++k) {
        float2 acc = float2(0.0f);
        for (uint g = l; g < G; g += kTreeLanes) acc = df_add(acc, partial[(k * G + g) * kc + c]);
        sh[(k * kTreeLanes + l) * kc + c] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (l != 0) return;
    ColState st = cs[c];
    if (st.active == 0) return;
    const float2 t = fold_lanes(sh, c, kc);
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
            st.mu_r = df_mul(fold_lanes(sh + kTreeLanes * kc, c, kc), inv_n).x;
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
    cs[c] = st;
}

// ---- triangular solves: forward rhs = r.hi - mu_r, backward in place on z ----
inline float row_rhs(uint i, uint c, uint kc, device const float2* r, device const float* z, float mu) {
    return kForward ? r[i * kc + c].x - mu : z[i * kc + c];
}

inline float light_row(uint t, uint c, uint kc, device const uint* rows, device const uint* ptr,
                       device const uint* col, device const float* val, device const float* dinv,
                       device const float2* r, device const float* z, float mu) {
    const uint i = rows[t];
    float s = row_rhs(i, c, kc, r, z, mu);
    const uint e1 = ptr[t + 1];
    for (uint e = ptr[t]; e < e1; ++e) s = fma(-val[e], z[col[e] * kc + c], s);
    return s * dinv[t];
}

// One level's light rows: thread per (row, column).
kernel void level(constant Params& P [[buffer(0)]],
                  device const uint* rows [[buffer(1)]],
                  device const uint* ptr [[buffer(2)]],
                  device const uint* col [[buffer(3)]],
                  device const float* val [[buffer(4)]],
                  device const float* dinv [[buffer(5)]],
                  device const float2* r [[buffer(6)]],
                  device float* z [[buffer(7)]],
                  device const ColState* cs [[buffer(8)]],
                  uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x, kc = P.kc;
    if (c >= kc || gid.y >= P.count) return;
    if (cs[c].active == 0) return;
    const uint t = P.offset + gid.y;
    z[rows[t] * kc + c] = light_row(t, c, kc, rows, ptr, col, val, dinv, r, z, cs[c].mu_r);
}

// One level's heavy rows (more than 32 dependencies): a threadgroup (kc, L)
// per row. Dependency k of the row goes to virtual lane k mod 32; real lane l
// accumulates virtual lanes l, l + L, ...; lane 0 folds lanes 0..31 in order.
kernel void level_heavy(constant Params& P [[buffer(0)]],
                        device const uint* rows [[buffer(1)]],
                        device const uint* ptr [[buffer(2)]],
                        device const uint* col [[buffer(3)]],
                        device const float* val [[buffer(4)]],
                        device const float* dinv [[buffer(5)]],
                        device const float2* r [[buffer(6)]],
                        device float* z [[buffer(7)]],
                        device const ColState* cs [[buffer(8)]],
                        uint2 tid [[thread_position_in_threadgroup]],
                        uint2 tgs [[threads_per_threadgroup]],
                        uint2 tg [[threadgroup_position_in_grid]]) {
    threadgroup float sh[kVirtualLanes * kMaxColumns];
    const uint c = tid.x, l = tid.y, L = tgs.y, kc = P.kc;
    const uint t = P.offset + tg.x;
    const bool act = cs[c].active != 0;
    if (act) {
        const uint b = ptr[t], e1 = ptr[t + 1];
        for (uint v = l; v < kVirtualLanes; v += L) {
            float s = 0.0f;
            for (uint e = b + v; e < e1; e += kVirtualLanes) s = fma(val[e], z[col[e] * kc + c], s);
            sh[v * kc + c] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (l == 0 && act) {
        float tot = sh[c];
        for (uint v = 1; v < kVirtualLanes; ++v) tot += sh[v * kc + c];
        const uint i = rows[t];
        z[i * kc + c] = (row_rhs(i, c, kc, r, z, cs[c].mu_r) - tot) * dinv[t];
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
                          device const ColState* cs [[buffer(8)]],
                          device const uint* level_ptr [[buffer(9)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint tgs [[threads_per_threadgroup]]) {
    const uint kc = P.kc;
    for (uint lv = P.offset; lv < P.count; ++lv) {
        const uint t0 = level_ptr[lv];
        const uint items = (level_ptr[lv + 1] - t0) * kc;
        for (uint q = tid; q < items; q += tgs) {
            const uint c = q % kc;
            if (cs[c].active == 0) continue;
            const uint t = t0 + q / kc;
            z[rows[t] * kc + c] = light_row(t, c, kc, rows, ptr, col, val, dinv, r, z, cs[c].mu_r);
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// ---- p = (z - mu_z) + beta p (p = z - mu_z where beta is 0) ----
kernel void p_update(constant Params& P [[buffer(0)]],
                     device float* p [[buffer(1)]],
                     device const float* z [[buffer(2)]],
                     device const ColState* cs [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x, i = gid.y, kc = P.kc;
    if (c >= kc || i >= P.n) return;
    if (cs[c].active == 0) return;
    const uint o = i * kc + c;
    const float zc = z[o] - cs[c].mu_z;
    const float b = cs[c].beta;
    p[o] = b == 0.0f ? zc : fma(b, p[o], zc);
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
