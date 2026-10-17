#pragma once
#include "layout.h"
inline const char resident_metal_source[] =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "#define APXCHOL_GPU_FUNCTION inline\n"
    "#define APXCHOL_GPU_UINT64 ulong\n"
    "#define APXCHOL_GPU_BUFFER(T) device T*\n"
#define APXCHOL_GPU_CODE(...) #__VA_ARGS__
#include "gks_star.inc"
#include "apxchol/solver/detail/double_float.inc"
#include "math.inc"
#include "setup.inc"
#include "apxchol/solver/detail/residual_sparsify.inc"
#include "sparsify.inc"
#include "view.inc"
#undef APXCHOL_GPU_CODE
R"msl(
using namespace resident_setup;
struct Memory {
    using wide = sparse_round::compensated_math::wide;
    View<wide> data;
    bool failed() const { return atomic_load_explicit(reinterpret_cast<device atomic_uint*>(&data.state->status), memory_order_relaxed) != 0; }
    void fail(unsigned status) const { atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint*>(&data.state->status),status,memory_order_relaxed); }
    void hidden_degree() const { atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(&data.state->prune_hidden),1u,memory_order_relaxed); }
    void add_histogram(unsigned i) const { atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(&data.histogram[i]),1u,memory_order_relaxed); }
    unsigned reserve_active() const { return atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(data.histogram),1u,memory_order_relaxed); }
    unsigned reserve_selected() const { return atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(&data.state->selected),1u,memory_order_relaxed); }
    void count_edge(unsigned node) const { atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(&data.counts[data.candidates[node]].edges),1u,memory_order_relaxed); }
    void remove_edge(unsigned node) const {
        atomic_fetch_sub_explicit(
            reinterpret_cast<device atomic_uint*>(&data.counts[data.candidates[node]].edges),
            1u, memory_order_relaxed);
    }
    void count_update(unsigned node) const { atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(&data.counts[data.candidates[node]].updates),1u,memory_order_relaxed); }
    unsigned reserve_edge(unsigned node) const { return atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(&data.cursor[node]),1u,memory_order_relaxed); }
    unsigned reserve_update(unsigned node) const { return atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(&data.update_cursor[node]),1u,memory_order_relaxed); }
};
struct Group {
    unsigned lane,width,base=0;
    unsigned broadcast(unsigned value,unsigned owner=0) const {return simd_shuffle(value,base+owner);}
    sparse_round::compensated_math::wide broadcast(sparse_round::compensated_math::wide value,unsigned owner) const {
        return {simd_shuffle(value.hi,base+owner),simd_shuffle(value.lo,base+owner)};
    }
    void barrier() const {simdgroup_barrier(mem_flags::mem_device);}
    unsigned prefix(unsigned value) const {
        const unsigned scan=simd_prefix_exclusive_sum(value);
        return scan-simd_shuffle(scan,base);
    }
    unsigned sum(unsigned value) const {return broadcast(prefix(value)+value,width-1);}
};
struct SparseMemory {
    using wide = Memory::wide;
    View<wide> data;
    bool failed() const { return Memory{data}.failed(); }
    void fail(unsigned status) const { Memory{data}.fail(status); }
    unsigned reserve_edge(unsigned i) const { return Memory{data}.reserve_edge(i); }
    device SparseState<wide>* sparse;
    unsigned pass;
    Fill edge(unsigned i) const { return data.fills[i]; }
    unsigned label(unsigned i) const { return data.winners[i]; }
    unsigned bucket(float x) const { return residual_sparsify::bucket(as_type<unsigned>(x)); }
    device atomic_uint* atomic(device unsigned* p) const { return reinterpret_cast<device atomic_uint*>(p); }
    void offer(unsigned i, unsigned bucket, unsigned id) const {
        if (pass == sparse_bucket) atomic_fetch_min_explicit(atomic(data.update_ptr+i), bucket, memory_order_relaxed);
        else if (data.update_ptr[i] == bucket) atomic_fetch_min_explicit(atomic(data.update_cursor+i), id, memory_order_relaxed);
    }
    unsigned choice(unsigned i) const { return data.update_cursor[i]; }
    unsigned parent(unsigned i) const { return atomic_load_explicit(atomic(data.candidates+i), memory_order_relaxed); }
    bool link(unsigned high, unsigned low) const {
        unsigned expected = high;
        return atomic_compare_exchange_weak_explicit(atomic(data.candidates+high), &expected, low,
                                                    memory_order_relaxed, memory_order_relaxed);
    }
    void mark(unsigned i) const {
        if (!atomic_exchange_explicit(atomic(data.update_ids+i), 1u, memory_order_relaxed))
            atomic_fetch_add_explicit(atomic(&sparse->tree), 1u, memory_order_relaxed);
    }
    void retain_parallel() const { atomic_store_explicit(atomic(&sparse->overflow), 1u, memory_order_relaxed); }
    void joined() const { atomic_fetch_add_explicit(atomic(&sparse->joins), 1u, memory_order_relaxed); }
    void count(unsigned i) const { atomic_fetch_add_explicit(atomic(&data.counts[i].edges), 1u, memory_order_relaxed); }
};
constant uint stage [[function_constant(0)]];
kernel void sparsify_stage(device uchar* arena [[buffer(0)]], device Edge* factors [[buffer(1)]],
        constant Layout& layout [[buffer(2)]], device SparseState<Memory::wide>* sparse [[buffer(3)]],
        constant bool& initial [[buffer(4)]], uint i [[thread_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]],
        uint simd_lane [[thread_index_in_simdgroup]], uint width [[threads_per_simdgroup]]) {
    SparseMemory m{view<Memory::wide>(arena, factors, layout), sparse, stage};
    const sparse_round::compensated_math math;
    if (stage == sparse_coalesce) {
        sparse_coalesce_row(m, math, i / width, Group{simd_lane, width});
    } else if (stage == sparse_partials) {
        threadgroup Memory::wide sums[256];
        if (m.failed() || !sparse->enabled || group * 16384 >= sparse->count || (!initial && !sparse->scale.advance)) return;
        auto sum = math.from_float(0);
        const unsigned end = min(sparse->count, (group + 1) * 16384);
        for (unsigned at = group * 16384 + lane; at < end; at += 256) {
            if (m.data.update_ids[at]) continue;
            const auto importance = m.data.prefix[at];
            sum = math.add(sum, initial ? importance :
                residual_sparsify::probability(math, sparse->scale.value, importance));
        }
        sums[lane] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (unsigned stride = 128; stride; stride /= 2) {
            if (lane < stride) sums[lane] = math.add(sums[lane], sums[lane + stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (!lane) m.data.updates[group].value = sums[0];
    } else sparse_step(m, math, stage, i, initial);
}
inline void setup_step(Memory m, unsigned stage, unsigned i, unsigned limit, unsigned width,
                       unsigned lane) {
    const sparse_round::compensated_math math;
    if (stage == stage_count_stars) {
        group_star<false>(m, math, i / width, Group{lane, width});
        return;
    }
    if (stage == stage_emit_stars) {
        group_star<true>(m, math, i / width, Group{lane, width});
        return;
    }
    if (stage == stage_copy_retained) {
        retained(m, i / width, Group{lane % width, width, lane - lane % width});
        return;
    }
    if (stage != stage_initialize && (!m.data.state->active || m.failed()))
        return;
    if (stage == stage_histogram || (stage >= stage_candidates && stage <= stage_commit)) {
        if (i >= m.data.state->active)
            return;
        i = m.data.entry_offsets[i];
    }
    if (stage == stage_initialize)
        initialize(m, math, i);
    if (stage == stage_reset) {
        if (i < 33)
            m.data.histogram[i] = 0;
        if (i == 0) {
            m.data.state->selected = 0;
            m.data.state->prune_hidden = 0;
            m.data.counts[m.data.state->nodes] = {};
        }
    }
    if (stage == stage_histogram)
        histogram(m, i);
    if (stage == stage_threshold && i == 0)
        threshold(m);
    if (stage == stage_candidates)
        candidates(m, i);
    if (stage == stage_decide)
        decide(m, i);
    if (stage == stage_commit)
        commit(m, i);
    if (stage == stage_select)
        select(m, i);
    if (stage == stage_offsets)
        offsets(m, i, limit);
    if (stage == stage_deactivate) {
        deactivate(m, i);
        if (i == 0)
            m.data.histogram[0] = 0;
    }
    if (stage == stage_count_retained)
        count_retained(m, i);
    if (stage == stage_count_incoming)
        incoming<false>(m, i / width, lane, width);
    if (stage == stage_scatter_incoming)
        incoming<true>(m, i / width, lane, width);
    if (stage == stage_update_excess)
        excess(m, math, i);
    if (stage == stage_finish && i == 0)
        finish(m);
    if (stage == stage_validate_compaction && i == 0)
        validate_compact(m);
}
kernel void setup_stage(device uchar* arena [[buffer(0)]],device Edge* factors [[buffer(1)]],
                        constant Layout& layout [[buffer(2)]], constant uint& limit [[buffer(3)]], constant uint& width [[buffer(4)]], uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    setup_step(Memory{view<Memory::wide>(arena,factors,layout)},stage,i,limit,width,lane);
}
struct TailRound {
    Memory m;
    unsigned tid, threads, simd_width, lane;
    void stage(unsigned phase, unsigned count, unsigned limit = 0) const {
        const unsigned width = simd_width;
        const bool grouped = stage_is_grouped(phase);
        for (unsigned i = tid; i < count * (grouped ? width : 1); i += threads)
            setup_step(m, phase, i, limit, width, lane);
        threadgroup_barrier(mem_flags::mem_device);
    }
    void scan(unsigned count) const {
        if (tid == 0) {
            Counts total{};
            for (unsigned i = 0; i < count; ++i) {
                const auto value = m.data.counts[i];
                m.data.counts[i] = total;
                total.edges += value.edges;
                total.updates += value.updates;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
};
kernel void factor_tail(device uchar *arena [[buffer(0)]], device Edge *factors [[buffer(1)]],
                        constant Layout &layout [[buffer(2)]],
                        uint i [[thread_index_in_threadgroup]],
                        uint threads [[threads_per_threadgroup]],
                        uint width [[threads_per_simdgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    Memory m{view<Memory::wide>(arena, factors, layout)};
    if (m.data.state->active > 128 || !width || threads % width || width % min(8u, width)) {
        if (i == 0)
            m.fail(capacity_exceeded);
        return;
    }
    const TailRound dispatch{m, i, threads, width, lane};
    for (unsigned remaining = m.data.state->active;
         remaining && m.data.state->active && !m.failed(); --remaining)
        factor_round(dispatch, m.data.state->active);
    if (i == 0 && m.data.state->active && !m.failed())
        m.fail(no_progress);
}

Counts add_counts(Counts a,Counts b) {return {a.edges+b.edges,a.updates+b.updates};}
kernel void scan_blocks(device Counts* values [[buffer(0)]], device Counts* sums [[buffer(1)]],
                        constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]],
                        uint lane [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]]) {
    threadgroup Counts scratch[256];
    scratch[lane]=i<n ? values[i] : Counts{};
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=1;stride<256;stride*=2) {
        const uint right=(lane+1)*(stride*2)-1;
        if(right<256)scratch[right]=add_counts(scratch[right-stride],scratch[right]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(lane==0){sums[group]=scratch[255];scratch[255]={};}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=128;stride;stride/=2) {
        const uint right=(lane+1)*(stride*2)-1;
        if(right<256){const auto a=scratch[right-stride],b=scratch[right];scratch[right-stride]=b;scratch[right]=add_counts(a,b);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(i<n)values[i]=scratch[lane];
}
kernel void scan_add(device Counts* values [[buffer(0)]],device const Counts* sums [[buffer(1)]],
                     constant uint& n [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i<n)values[i]=add_counts(values[i],sums[i/256]);
}
struct CscInput {
    device const unsigned* ptr;
    device const unsigned* idx;
    device const sparse_round::compensated_math::wide* values;
    unsigned triangles_identical;
    sparse_round::compensated_math::wide canonical(unsigned col,unsigned at) const {
        const unsigned row=idx[at];
        if(triangles_identical || row>=col)return values[at];
        unsigned first=ptr[row],last=ptr[row+1];
        while(first<last){const unsigned mid=first+(last-first)/2;if(idx[mid]<col)first=mid+1;else last=mid;}
        return values[first];
    }
};
kernel void import_count(device uchar* arena [[buffer(0)]],device Edge* factors [[buffer(1)]],
                         constant Layout& layout [[buffer(2)]],device const unsigned* ptr [[buffer(3)]],
                         device const unsigned* idx [[buffer(4)]],device const Memory::wide* values [[buffer(5)]],
                         constant float& regularization [[buffer(6)]],
                         constant unsigned& triangles_identical [[buffer(7)]],
                         uint row [[thread_position_in_grid]]) {
    auto s=view<Memory::wide>(arena,factors,layout);
    if(row>s.state->nodes)return;
    if(row==s.state->nodes){s.counts[row]={};return;}
    const CscInput input{ptr,idx,values,triangles_identical};const sparse_round::compensated_math math;
    auto diagonal=math.from_float(0),degree=diagonal;unsigned count=0;
    for(unsigned at=ptr[row];at<ptr[row+1];++at){const auto v=input.canonical(row,at);
        if(idx[at]==row)diagonal=math.add(diagonal,v);
        else {degree=math.subtract(degree,v);if(math.less(v,math.from_float(0)))++count;}}
    auto excess=math.subtract(diagonal,degree);
    if(!math.less(math.multiply(diagonal,math.from_float(1e-12f)),excess))excess=math.from_float(0);
    const auto ground=math.multiply(diagonal,math.from_float(regularization));
    if(regularization>0 && (!math.finite(ground) || ground.hi<0x1p-126f))
        atomic_fetch_or_explicit((device atomic_uint*)&s.state->status,
                                 1u<<sparse_round::arithmetic_range,memory_order_relaxed);
    else if(math.less(excess,ground))excess=ground;
    s.excess[row]=excess;s.counts[row]={count,0};
    if(math.less(math.from_float(0),excess))atomic_store_explicit((device atomic_uint*)&s.state->has_excess,1u,memory_order_relaxed);
}
kernel void import_emit(device uchar* arena [[buffer(0)]],device Edge* factors [[buffer(1)]],
                        constant Layout& layout [[buffer(2)]],device const unsigned* ptr [[buffer(3)]],
                        device const unsigned* idx [[buffer(4)]],device const Memory::wide* values [[buffer(5)]],
                        constant unsigned& triangles_identical [[buffer(7)]],
                        uint row [[thread_position_in_grid]]) {
    auto s=view<Memory::wide>(arena,factors,layout);
    if(row>s.state->nodes)return;
    s.begin[row]=s.counts[row].edges;
    if(row==s.state->nodes){s.state->edges=s.counts[row].edges;return;}
    s.cursor[row]=s.counts[row+1].edges;
    const CscInput input{ptr,idx,values,triangles_identical};const sparse_round::compensated_math math;
    unsigned target=s.counts[row].edges;
    for(unsigned at=ptr[row];at<ptr[row+1];++at){const auto v=input.canonical(row,at);
        if(idx[at]!=row&&math.less(v,math.from_float(0)))s.graph[0][target++]={idx[at],-math.to_float(v)};}
}
)msl";
