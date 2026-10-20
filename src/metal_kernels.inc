// Metal PCG kernels, compiled at run time by metal_solver.mm.
// Embedding the source avoids an offline Metal compiler or generation step.
R"apxchol_msl(
#include <metal_stdlib>
using namespace metal;
using namespace apxchol::detail::gpu_pcg;
using namespace apxchol::detail::metal_host;

static_assert(sizeof(Params) == 40, "Params mirrors the host layout");
static_assert(__builtin_offsetof(Params, preconditioner_scale) == 36, "Params mirrors the host layout");

using SolveState = solve_state<df, float>;
static_assert(sizeof(SolveState) == 48, "SolveState mirrors detail::metal::solve_state");

constant bool kForward [[function_constant(0)]];
constant bool kOperatorDf [[function_constant(1)]];
constant bool kFactorFp16 [[function_constant(3)]];
constant uint kOperation [[function_constant(2)]];
// Distinct compiled pipelines: the scalar specialization has one accumulator.
constant uint kColumns [[function_constant(4)]];

template<unsigned Lanes = 1>
struct pcg_math {
    using wide = df;
    using scalar = float;
    float preconditioner_scale = 1;
    float preconditioned(float z) const { return z * preconditioner_scale; }
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
    df sum_row(df sum) const { return sum_lanes<Lanes>(*this, sum); }
};

template<unsigned Columns>
inline void pcg_step_columns(constant Params& P, workspace<df, float> vectors,
                            device const uint* ptr, device const uint* col,
                            device const float* hi, device const float* lo,
                            uint lane, uint group, uint tid) {
    const operation op = operation(kOperation);
    const pcg_work<uint> work = op == operation::direction
        ? pcg_work<uint>{tid, P.n, P.n, 0, P.n}
        : pcg_work<uint>{group * kTreeRows, (group + 1) * kTreeRows, kTreeLanes, lane, P.n};
    const auto sum = op == operation::multiply && P.spmv_lanes == kTreeLanes
        ? step_columns<kTreeLanes, Columns>(op, pcg_math<kTreeLanes>{P.preconditioner_scale}, vectors, ptr, col, hi, lo,
                          pcg_work<uint>{work.first, work.last, 1, 0, P.n}, lane)
        : step_columns<1, Columns>(op, pcg_math<>{P.preconditioner_scale}, vectors, ptr, col, hi, lo, work, 0);
    if (op != operation::direction) {
        for (unsigned q = 0; q < Columns; ++q) {
            const auto total = sum_lanes<kTreeLanes>(pcg_math<>{}, sum.values[q]);
            if (lane == 0) vectors.partial[group * Columns + q] = total;
        }
    }
}

// Workspace slots stay bound for the whole command batch, including SpTRSV.
kernel void pcg_step(constant Params& P [[buffer(0)]],
                     device df* x [[buffer(1)]], device df* r [[buffer(2)]],
                     device df* ap [[buffer(3)]], device float* p [[buffer(4)]],
                     device float* z [[buffer(5)]], device df* partial [[buffer(6)]],
                     device SolveState* state [[buffer(7)]],
                     device const uint* ptr [[buffer(8)]], device const uint* col [[buffer(9)]],
                     device const float* hi [[buffer(10)]], device const float* lo [[buffer(11)]],
                     uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_grid]]) {
    const auto vectors = workspace<df, float>{x, r, ap, p, z, partial, state};
    if (kColumns == 4) pcg_step_columns<4>(P, vectors, ptr, col, hi, lo, lane, group, tid);
    else if (kColumns == 2) pcg_step_columns<2>(P, vectors, ptr, col, hi, lo, lane, group, tid);
    else pcg_step_columns<1>(P, vectors, ptr, col, hi, lo, lane, group, tid);
}

struct recurrence_math {
    bool positive(df x) const { return x.hi > 0 && isfinite(x.hi) && isfinite(x.lo); }
    bool valid_residual(df x) const { return !df_lt(x, df{}) && isfinite(x.hi) && isfinite(x.lo); }
    bool accept_ratio(float x) const { return isfinite(x); }
    float ratio(df x, df y) const { return df_div(x, y).hi; }
    bool converged(df x, df threshold) const { return df_lt(x, threshold); }
    bool stagnated(df previous, df current) const { return df_lt(df_mul_f(previous, 0.25f), current); }
};

template<unsigned Columns>
inline void finalize_columns(constant Params& P, device const df* partial,
                             device SolveState* state, uint l) {
    const pcg_work<uint> work{0, P.groups, kTreeLanes, l, P.groups};
    const auto partials = sum_partials_columns<Columns>(pcg_math<>{}, partial, work);
    for (unsigned q = 0; q < Columns; ++q) {
        const auto sum = sum_lanes<kTreeLanes>(pcg_math<>{}, partials.values[q]);
        if (l == 0 && state[q].active)
            state[q] = advance(recurrence_math{}, state[q], P.mode, sum, P.iter, P.window);
    }
}

// ---- one threadgroup of 16 lanes: fold the partials, then the PCG scalars ----
kernel void finalize(constant Params& P [[buffer(0)]],
                     device const df* partial [[buffer(6)]],
                     device SolveState* state [[buffer(7)]],
                     uint l [[thread_index_in_threadgroup]]) {
    if (kColumns == 4) finalize_columns<4>(P, partial, state, l);
    else if (kColumns == 2) finalize_columns<2>(P, partial, state, l);
    else finalize_columns<1>(P, partial, state, l);
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
