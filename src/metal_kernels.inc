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
static_assert(sizeof(SolveState) == 48, "SolveState mirrors detail::metal::solve_state");

constant bool kForward [[function_constant(0)]];
constant bool kOperatorDf [[function_constant(1)]];
constant uint kOperation [[function_constant(2)]];

constant uint kTreeRows = 256;
constant uint kTreeLanes = 16;

struct pcg_math {
    using wide = df;
    using scalar = float;
    df add(df a, df b) const { return df_add(a, b); }
    df shuffle_down(df value, uint offset, uint) const {
        return {simd_shuffle_down(value.hi, offset), simd_shuffle_down(value.lo, offset)};
    }
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

// Workspace slots stay bound for the whole command batch, including SpTRSV.
kernel void pcg_step(constant Params& P [[buffer(0)]],
                     device df* x [[buffer(1)]], device df* r [[buffer(2)]],
                     device df* ap [[buffer(3)]], device float* p [[buffer(4)]],
                     device float* z [[buffer(5)]], device df* partial [[buffer(6)]],
                     device SolveState& state [[buffer(7)]],
                     device const uint* ptr [[buffer(8)]], device const uint* col [[buffer(9)]],
                     device const float* hi [[buffer(10)]], device const float* lo [[buffer(11)]],
                     uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_grid]]) {
    const operation op = operation(kOperation);
    const pcg_work<uint> work = op == operation::direction
        ? pcg_work<uint>{tid, P.n, P.n, 0, P.n}
        : pcg_work<uint>{group * kTreeRows, (group + 1) * kTreeRows, kTreeLanes, lane, P.n};
    const auto sum = step<1>(op, pcg_math{}, workspace<df, float>{x, r, ap, p, z, partial, &state},
                             ptr, col, hi, lo, work, 0);
    if (op != operation::direction) {
        const auto total = sum_lanes<kTreeLanes>(pcg_math{}, sum);
        if (lane == 0) partial[group] = total;
    }
}

struct recurrence_math {
    bool positive(df x) const { return x.hi > 0 && isfinite(x.hi) && isfinite(x.lo); }
    bool valid_residual(df x) const { return !df_lt(x, df{}) && isfinite(x.hi) && isfinite(x.lo); }
    bool accept_ratio(float x) const { return isfinite(x); }
    float ratio(df x, df y) const { return df_div(x, y).hi; }
    bool converged(df x, df threshold) const { return df_lt(x, threshold); }
    bool stagnated(df previous, df current) const { return df_lt(df_mul_f(previous, 0.25f), current); }
};

// ---- one threadgroup of 16 lanes: fold the partials, then the PCG scalars ----
kernel void finalize(constant Params& P [[buffer(0)]],
                     device const df* partial [[buffer(6)]],
                     device SolveState& state [[buffer(7)]],
                     uint l [[thread_index_in_threadgroup]]) {
    const pcg_work<uint> work{0, P.groups, kTreeLanes, l, P.groups};
    const auto sum = sum_lanes<kTreeLanes>(pcg_math{}, sum_partials(pcg_math{}, partial, work));
    if (l == 0 && state.active)
        state = advance(recurrence_math{}, state, P.mode, sum, P.iter, P.window);
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
