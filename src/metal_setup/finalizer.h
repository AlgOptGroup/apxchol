#pragma once
#include "apxchol/solver/detail/gpu_dataflow_plan.h"
#include "apxchol/solver/sptrsv/factor_drop.h"
#include "engine.h"
#include "../metal_setup.h"
#include <array>
#include <cmath>
#include <mutex>
#define APXCHOL_GPU_CODE(...) __VA_ARGS__
#define APXCHOL_GPU_FUNCTION inline
#include "finalize.inc"
#include "planner.inc"
#undef APXCHOL_GPU_FUNCTION
#undef APXCHOL_GPU_CODE
namespace apxchol::detail::metal_setup {
namespace rf = resident_finalize;
namespace gd = apxchol::detail::gpu_dataflow;
inline const char finalizer_source[] =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "#define APXCHOL_GPU_FUNCTION inline\n#define APXCHOL_GPU_UINT64 ulong\n"
    "#define APXCHOL_GPU_BUFFER(T) device T*\n"
    "#define APXCHOL_GPU_CONSTANT constant\n"
// clang-format off
#define APXCHOL_GPU_CODE(...) #__VA_ARGS__
#include "gks_star.inc"
#include "apxchol/solver/detail/double_float.inc"
#include "math.inc"
#include "setup.inc"
#include "finalize.inc"
#include "planner.inc"
#include "apxchol/solver/detail/gpu_dataflow_pack.inc"
#undef APXCHOL_GPU_CODE
    R"msl(
namespace rf = resident_finalize;
namespace gd = apxchol::detail::gpu_dataflow;
template<class Value> struct StorageMath {
    using wide=sparse_round::compensated_math::wide;
    sparse_round::compensated_math arithmetic;
    wide from_float(float x) const {return arithmetic.from_float(x);}
    float to_float(wide x) const {return arithmetic.to_float(x);}
    bool less(wide x,wide y) const {return arithmetic.less(x,y);}
    wide add(wide x,wide y) const {return arithmetic.add(x,y);}
    wide subtract(wide x,wide y) const {return arithmetic.subtract(x,y);}
    wide multiply(wide x,wide y) const {return arithmetic.multiply(x,y);}
    wide divide(wide x,wide y) const {return arithmetic.divide(x,y);}
    float maximum(float a,float b) const {return max(a,b);}
    float absolute(float a) const {return abs(a);}
    bool finite_float(float a) const {return isfinite(a);}
    bool flushes(float a) const {return abs(a)<0x1p-14f-0x1p-25f;}
    wide drop(rf::Shape shape) const {return {shape.drop_hi,shape.drop_lo};}
    float fraction(float x,thread int& exponent) const {return frexp(x,exponent);}
    float scale_float(float x,int exponent) const {return ldexp(x,exponent);}
    Value store(float a,float scale,bool half_storage) const {
        if(!half_storage)return Value(a);
        a/=scale;return Value(flushes(a)?copysign(0.0f,a):a);
    }
};
template<class Value> struct FinalMemory {
    rf::Shape shape;bool fp16;
    device const resident_setup::Column* columns;
    device resident_setup::Edge* entries;
    device unsigned* perm;
    device rf::Row* rows;
    device unsigned* ptr;
    device unsigned* idx;
    device Value* val;
    device rf::Metadata* metadata;
    device unsigned* transpose_ptr;
    device unsigned* transpose_idx;
    device Value* transpose_val;
    device unsigned* cursor;
    device atomic_uint* status;
    void fail(unsigned value) const {atomic_fetch_or_explicit(status,value,memory_order_relaxed);}
    void increment_count(unsigned row) const {atomic_fetch_add_explicit((device atomic_uint*)(transpose_ptr+row),1u,memory_order_relaxed);}
    unsigned reserve(unsigned row) const {return atomic_fetch_add_explicit((device atomic_uint*)(cursor+row),1u,memory_order_relaxed);}
};
constant unsigned phase [[function_constant(0)]];
constant bool half_storage [[function_constant(1)]];
template <class Value> void run_final(FinalMemory<Value> s, unsigned row) {
    const StorageMath<Value> math;
    if (phase == rf::stage_initialize) {
        rf::permutation(s, row);
        if (row <= s.shape.m) {
            s.ptr[row] = 0;
            s.transpose_ptr[row] = 0;
        }
    }
    if (phase == rf::stage_prepare)
        rf::prepare(s, math, row);
    if (phase == rf::stage_emit)
        rf::emit(s, math, row);
    if (phase == rf::stage_count_transpose)
        rf::transpose_count(s, row);
    if (phase == rf::stage_initialize_cursor && row < s.shape.m)
        s.cursor[row] = s.transpose_ptr[row];
    if (phase == rf::stage_emit_transpose)
        rf::transpose_emit(s, row);
}
kernel void finalize_stage(device const resident_setup::Column* columns [[buffer(0)]],
                          device resident_setup::Edge* entries [[buffer(1)]],
                          device unsigned* perm [[buffer(2)]],
                          device rf::Row* rows [[buffer(3)]],device unsigned* ptr [[buffer(4)]],
                          device unsigned* idx [[buffer(5)]],device uchar* val [[buffer(6)]],
                          device rf::Metadata* metadata [[buffer(7)]],device unsigned* tptr [[buffer(8)]],
                          device unsigned* tidx [[buffer(9)]],device uchar* tval [[buffer(10)]],
                          device unsigned* cursor [[buffer(11)]],
                          device atomic_uint* status [[buffer(12)]],constant rf::Shape& shape [[buffer(13)]],
                          uint row [[thread_position_in_grid]]) {
    if (half_storage)
        run_final(FinalMemory<half>{
            shape,
            true,
            columns,
            entries,
            perm,
            rows,
            ptr,
            idx,
            (device half*)val,
            metadata,
            tptr,
            tidx,
            (device half*)tval,
            cursor,
            status
        }, row);
    else
        run_final(FinalMemory<float>{
            shape,
            false,
            columns,
            entries,
            perm,
            rows,
            ptr,
            idx,
            (device float*)val,
            metadata,
            tptr,
            tidx,
            (device float*)tval,
            cursor,
            status
        }, row);
}
struct PlanMemory {
    unsigned n; bool reverse; unsigned rounds;
    device const unsigned* round_ptr;
    device unsigned* blocks;
    device rf::Step* steps; device rf::PlanInfo* info; device atomic_uint* status;
    void fail(unsigned value) const {atomic_fetch_or_explicit(status,value,memory_order_relaxed);}
};
kernel void plan_structure(device const unsigned* round_ptr [[buffer(0)]],
                           device unsigned* blocks [[buffer(1)]],device rf::Step* steps [[buffer(2)]],
                           device rf::PlanInfo* info [[buffer(3)]],device atomic_uint* status [[buffer(4)]],
                           constant unsigned& n [[buffer(5)]],constant unsigned& reverse [[buffer(6)]],
                           constant unsigned& rounds [[buffer(7)]]) {
    rf::plan(PlanMemory{n,bool(reverse),rounds,round_ptr,blocks,steps,info,status});
}
constant unsigned plan_lanes [[function_constant(2)]];
// Finalization has already checked triangular indices; packing only needs row pointers.
kernel void plan_count(device const unsigned* ptr [[buffer(0)]],
                       device const unsigned* blocks [[buffer(1)]],device const rf::PlanInfo* info [[buffer(2)]],
                       device gd::counts* counts [[buffer(3)]],device atomic_uint* status [[buffer(4)]],
                       constant unsigned& n [[buffer(5)]],constant unsigned& reverse [[buffer(6)]],
                       uint block [[thread_position_in_grid]]) {
    if(block<info->blocks) {
        const gd::row_order order{n,bool(reverse)};
        for(unsigned q=blocks[block];q<blocks[block+1];++q) {
            const unsigned row=order(q);
            if(ptr[row]>=ptr[row+1]) {
                counts[block]={};
                atomic_fetch_or_explicit(status,4u,memory_order_relaxed);
                return;
            }
        }
        counts[block] = plan_lanes == 8
            ? gd::pack_rows<false, 8>(ptr,order,blocks[block],blocks[block+1],gd::chunk_size)
            : gd::pack_rows<false>(ptr,order,blocks[block],blocks[block+1],gd::chunk_size);
    }
}
kernel void plan_emit(device const unsigned* ptr [[buffer(0)]],device const unsigned* blocks [[buffer(1)]],
                      device const rf::PlanInfo* info [[buffer(2)]],device const gd::counts* counts [[buffer(3)]],
                      device unsigned* words [[buffer(4)]],device gd::segment* segments [[buffer(5)]],
                      device unsigned* packed_blocks [[buffer(6)]],constant unsigned& n [[buffer(7)]],
                      constant unsigned& reverse [[buffer(8)]],uint block [[thread_position_in_grid]]) {
    if(block==0)packed_blocks[0]=0;
    if (block >= info->blocks) return;
    const auto first = rf::block_begin(blocks[block], blocks[block+1]), last = blocks[block+1];
    if (plan_lanes == 8)
        gd::pack_rows<true, 8>(ptr,{n,bool(reverse)},first,last,gd::chunk_size,counts[block],words,segments);
    else
        gd::pack_rows<true>(ptr,{n,bool(reverse)},first,last,gd::chunk_size,counts[block],words,segments);
    packed_blocks[block+1] = unsigned(counts[block+1].words / plan_lanes);
}
kernel void scan_unsigned(device unsigned *values [[buffer(0)]],
                          device unsigned *sums [[buffer(1)]], constant unsigned &n [[buffer(2)]],
                          uint i [[thread_position_in_grid]],
                          uint lane [[thread_index_in_threadgroup]],
                          uint group [[threadgroup_position_in_grid]]) {
    threadgroup unsigned scratch[256];
    scratch[lane] = i < n ? values[i] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (unsigned stride = 1; stride < 256; stride *= 2) {
        unsigned right = (lane + 1) * 2 * stride - 1;
        if (right < 256)
            scratch[right] += scratch[right - stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        sums[group] = scratch[255];
        scratch[255] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (unsigned stride = 128; stride; stride /= 2) {
        unsigned right = (lane + 1) * 2 * stride - 1;
        if (right < 256) {
            unsigned a = scratch[right - stride], b = scratch[right];
            scratch[right - stride] = b;
            scratch[right] = a + b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (i < n)
        values[i] = scratch[lane];
}
kernel void scan_unsigned_add(device unsigned *values [[buffer(0)]],
                              device const unsigned *sums [[buffer(1)]],
                              constant unsigned &n [[buffer(2)]],
                              uint i [[thread_position_in_grid]]) {
    if (i < n)
        values[i] += sums[i / 256];
}
kernel void operator_emit(device const unsigned* ptr [[buffer(0)]],device const unsigned* idx [[buffer(1)]],
                          device const apxchol::detail::metal_host::df* values [[buffer(2)]],
                          device const unsigned* perm [[buffer(3)]],
                          device unsigned* outidx [[buffer(4)]],device float* hi [[buffer(5)]],device float* lo [[buffer(6)]],
                          constant unsigned& n [[buffer(7)]],
                          constant unsigned& triangles_identical [[buffer(8)]],
                          constant unsigned& exact [[buffer(9)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]], uint width [[threads_per_simdgroup]]) {
    if(row>=n)return;
    for(unsigned p=ptr[row]+lane;p<ptr[row+1];p+=width){
        const unsigned col=idx[p];unsigned canonical=p;
        if(!triangles_identical && col<row){unsigned first=ptr[col],last=ptr[col+1];
            while(first<last){const unsigned mid=first+(last-first)/2;if(idx[mid]<row)first=mid+1;else last=mid;}
            canonical=first;
        }
        outidx[p]=perm[col];hi[p]=values[canonical].hi;
        if(!exact)lo[p]=values[canonical].lo;
    }
}
)msl";
// clang-format on

struct Finalized {
    triangle forward, backward;
    id<MTLBuffer> metadata = nil, permutation = nil;
    factor_drop_stats drop;
};

class Finalizer {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    std::array<std::array<id<MTLComputePipelineState>, 6>, 2> final_pipes;
    id<MTLComputePipelineState> structure, scan_blocks, scan_add, operator_emit;
    std::array<id<MTLComputePipelineState>, 2> count{}, emit{};
    std::once_flag lane_plan_once;
    id<MTLComputePipelineState> pipeline(NSString *name, unsigned lanes = 0) {
        NSError *error = nil;
        auto constants = [MTLFunctionConstantValues new];
        if (lanes) [constants setConstantValue:&lanes type:MTLDataTypeUInt atIndex:2];
        auto function = lanes ? [library newFunctionWithName:name constantValues:constants error:&error]
                              : [library newFunctionWithName:name];
        if (!function) throw std::runtime_error(describe(error));
        auto result = [device newComputePipelineStateWithFunction:function error:&error];
        if (!result) throw std::runtime_error(describe(error));
        return result;
    }
    id<MTLBuffer> allocate(std::size_t bytes) {
        auto result = [device newBufferWithLength:std::max(bytes, std::size_t(16))
                                          options:MTLResourceStorageModeShared];
        if (result == nil) throw device_memory_error("Metal factor preparation buffer allocation failed");
        return result;
    }
    void finish(id<MTLCommandBuffer> cb, id<MTLBuffer> status = nil) {
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted)
            throw std::runtime_error("Metal factor preparation command buffer failed: " + describe(cb.error));
        const auto failure = status ? *static_cast<unsigned *>(status.contents) : 0u;
        if (failure)
            throw std::runtime_error("Metal factor preparation failed: status " + std::to_string(failure));
    }
    id<MTLComputeCommandEncoder> encoder(id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pipe,
                                         std::initializer_list<id<MTLBuffer>> buffers) {
        auto enc = [cb computeCommandEncoder];
        require(enc != nil, "Metal factor preparation encoder creation failed");
        [enc setComputePipelineState:pipe];
        unsigned index = 0;
        for (auto b : buffers)
            [enc setBuffer:b offset:0 atIndex:index++];
        return enc;
    }
    void dispatch(id<MTLComputeCommandEncoder> enc, unsigned n, unsigned width = 256) {
        [enc dispatchThreads:MTLSizeMake(std::max(n, 1u), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [enc endEncoding];
    }
    void scan(id<MTLCommandBuffer> cb, id<MTLBuffer> values, unsigned n, NSMutableArray *retained) {
        const unsigned groups = (n + 255) / 256;
        auto sums = allocate(std::size_t(groups) * sizeof(unsigned));
        [retained addObject:sums];
        auto enc = encoder(cb, scan_blocks, {values, sums});
        [enc setBytes:&n length:sizeof n atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        if (groups > 1) {
            scan(cb, sums, groups, retained);
            enc = encoder(cb, scan_add, {values, sums});
            [enc setBytes:&n length:sizeof n atIndex:2];
            dispatch(enc, n);
        }
    }

  public:
    Finalizer(id<MTLDevice> dev) : device(dev) {
        NSError *error = nil;
        library = metal::compile_library(device, finalizer_source).library;
        for (unsigned half = 0; half < 2; ++half)
            for (unsigned stage = 0; stage < 6; ++stage) {
                const bool h = half;
                auto constants = [MTLFunctionConstantValues new];
                [constants setConstantValue:&stage type:MTLDataTypeUInt atIndex:0];
                [constants setConstantValue:&h type:MTLDataTypeBool atIndex:1];
                auto fn = [library newFunctionWithName:@"finalize_stage"
                                        constantValues:constants
                                                 error:&error];
                if (!fn)
                    throw std::runtime_error(describe(error));
                final_pipes[half][stage] = [device newComputePipelineStateWithFunction:fn
                                                                                 error:&error];
                if (!final_pipes[half][stage])
                    throw std::runtime_error(describe(error));
            }
        structure = pipeline(@"plan_structure");
        count[0] = pipeline(@"plan_count", gd::lanes);
        emit[0] = pipeline(@"plan_emit", gd::lanes);
        scan_blocks = pipeline(@"scan_unsigned");
        scan_add = pipeline(@"scan_unsigned_add");
        operator_emit = pipeline(@"operator_emit");
        queue = [device newCommandQueue];
        require(queue != nil, "Metal factor preparation command queue creation failed");
    }
    template<unsigned PlanLanes>
    void prepare_plans(triangle& forward, triangle& backward, unsigned m,
                       id<MTLBuffer> round_ptr, std::size_t round_offset, unsigned rounds,
                       id<MTLBuffer> status) {
        static_assert(PlanLanes == 8 || PlanLanes == gd::lanes);
        constexpr unsigned index = PlanLanes == 8;
        if constexpr (index) {
            std::call_once(lane_plan_once, [&] {
                count[1] = pipeline(@"plan_count", 8);
                emit[1] = pipeline(@"plan_emit", 8);
            });
        }
        auto blocks = allocate((m + 1) * sizeof(unsigned)),
             steps = allocate(m * sizeof(rf::Step));
        auto info = allocate(sizeof(rf::PlanInfo)),
             counts = allocate(std::size_t(m + 1) * sizeof(gd::counts));
        for (unsigned reverse = 0; reverse < 2; ++reverse) {
            auto &t = reverse ? backward : forward;
            auto cb = [queue commandBuffer];
            auto enc = encoder(cb, structure, {round_ptr, blocks, steps, info, status});
            [enc setBuffer:round_ptr offset:round_offset atIndex:0];
            [enc setBytes:&m length:sizeof m atIndex:5];
            [enc setBytes:&reverse length:sizeof reverse atIndex:6];
            [enc setBytes:&rounds length:sizeof rounds atIndex:7];
            dispatch(enc, 1, 1);
            enc = encoder(cb, count[index], {t.ptr, blocks, info, counts, status});
            [enc setBytes:&m length:sizeof m atIndex:5];
            [enc setBytes:&reverse length:sizeof reverse atIndex:6];
            dispatch(enc, m);
            finish(cb, status);
            // Counts are complete; compact the shared plan metadata before allocating output.
            struct PlanMetadata { unsigned* blocks; rf::Step* steps; rf::PlanInfo* info; };
            const auto count_data = static_cast<gd::counts*>(counts.contents);
            rf::compact(PlanMetadata{static_cast<unsigned*>(blocks.contents),
                                     static_cast<rf::Step*>(steps.contents),
                                     static_cast<rf::PlanInfo*>(info.contents)},
                        count_data, gd::counts{});
            const auto pi = *static_cast<rf::PlanInfo *>(info.contents);
            const auto total = count_data[pi.blocks];
            require(total.words <= ~0u && total.segments < gd::special_bit &&
                        total.slots + m < gd::special_bit,
                    "Metal packed plan exceeds 32-bit capacity");
            t.levels = pi.levels;
            t.tag_count = m + total.slots;
            const auto step_data = static_cast<rf::Step *>(steps.contents);
            t.stages.reserve(pi.stages);
            for (unsigned i = 0; i < pi.stages; ++i) {
                const auto step = step_data[i];
                // Every retained block has at least one padded batch.
                const auto words = count_data[step.last].words - count_data[step.first].words;
                t.stages.push_back({step.first, step.last,
                    words == std::uint64_t(PlanLanes) * (step.last - step.first)});
            }
            t.words = allocate(total.words * sizeof(unsigned));
            t.blocks = allocate((pi.blocks + 1) * sizeof(unsigned));
            t.segments = allocate(total.segments * sizeof(gd::segment));
            cb = [queue commandBuffer];
            enc = encoder(cb, emit[index],
                          {t.ptr, blocks, info, counts, t.words, t.segments, t.blocks});
            [enc setBytes:&m length:sizeof m atIndex:7];
            [enc setBytes:&reverse length:sizeof reverse atIndex:8];
            dispatch(enc, std::max(pi.blocks, 1u));
            finish(cb, status);
            t.resident_round_plan = true;
        }
    }
    void prepare_operator(id<MTLBuffer> ptr, id<MTLBuffer> idx, id<MTLBuffer> values,
                          id<MTLBuffer> permutation, unsigned n, unsigned nnz,
                          resident_factor &out) {
        // Keep input row order; columns and vector outputs use this permutation.
        out.operator_ptr = ptr;
        out.operator_row_map = permutation;
        out.operator_idx = allocate(std::size_t(nnz) * sizeof(unsigned));
        out.operator_hi = allocate(std::size_t(nnz) * sizeof(float));
        out.operator_lo = out.operator_exact ? nil : allocate(std::size_t(nnz) * sizeof(float));
        auto cb = [queue commandBuffer];
        auto enc = encoder(cb, operator_emit,
                      {ptr, idx, values, permutation, out.operator_idx,
                       out.operator_hi, out.operator_exact ? out.operator_hi : out.operator_lo});
        [enc setBytes:&n length:sizeof n atIndex:7];
        const unsigned triangles_identical = out.operator_triangles_identical;
        [enc setBytes:&triangles_identical length:sizeof triangles_identical atIndex:8];
        const unsigned exact = out.operator_exact;
        [enc setBytes:&exact length:sizeof exact atIndex:9];
        const unsigned width = operator_emit.threadExecutionWidth;
        require(width && width <= operator_emit.maxTotalThreadsPerThreadgroup, "invalid operator SIMD width");
        [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [enc endEncoding];
        finish(cb);
    }
    Finalized run(Factor &input, bool laplacian, bool half, double drop) {
        validate_drop_tolerance(drop);
        const auto state = *input.data().state;
        require(state.status == 0 && state.columns == state.nodes && state.nodes > 0,
                "Metal factor preparation requires a completed factor");
        require(state.nodes + std::uint64_t(state.entries) < gd::special_bit,
                "Metal factor exceeds 31-bit offsets");
        const unsigned n = state.nodes, m = n - unsigned(laplacian), capacity = state.entries + n;
        // Keep the exponent separate: a positive tolerance must not underflow
        // before it is combined with a column's scale.
        int exponent;
        const double mantissa = std::frexp(drop, &exponent);
        const rf::Shape shape{n, m, float(mantissa), float(mantissa - float(mantissa)), exponent};
        Finalized out;
        out.permutation = allocate(n * sizeof(unsigned));
        if (half) out.metadata = allocate(m * sizeof(rf::Metadata));
        auto rows = allocate(m * sizeof(rf::Row));
        auto status = allocate(sizeof(unsigned)), cursor = allocate(m * sizeof(unsigned));
        *static_cast<unsigned *>(status.contents) = 0;
        out.backward.ptr = allocate((m + 1) * sizeof(unsigned));
        out.forward.ptr = allocate((m + 1) * sizeof(unsigned));
        auto placeholder = allocate(16);
        out.backward.idx = out.backward.val = out.forward.idx = out.forward.val = placeholder;
        auto encode = [&](id<MTLCommandBuffer> cb, unsigned phase) {
            auto enc =
                encoder(cb, final_pipes[half][phase],
                        {input.arena, input.factor, out.permutation, rows,
                         out.backward.ptr, out.backward.idx, out.backward.val, half ? out.metadata : placeholder,
                         out.forward.ptr, out.forward.idx, out.forward.val, cursor, status});
            [enc setBuffer:input.arena
                    offset:input.layout.offsets[unsigned(resident_setup::Field::columns)]
                   atIndex:0];
            [enc setBytes:&shape length:sizeof shape atIndex:13];
            dispatch(enc, n + 1);
        };
        NSMutableArray *scan_scratch = [NSMutableArray new];
        auto cb = [queue commandBuffer];
        encode(cb, rf::stage_initialize);
        encode(cb, rf::stage_prepare);
        scan(cb, out.backward.ptr, m + 1, scan_scratch);
        finish(cb, status);
        const unsigned nnz = static_cast<unsigned *>(out.backward.ptr.contents)[m];
        require(nnz <= capacity, "Metal finalized factor exceeds allocated capacity");
        out.drop.rel = drop;
        out.drop.nnz_factor = out.drop.nnz_stored = nnz;
        if (shape.drop_hi > 0) {
            // Without a threshold every entry is kept, so only then are the
            // per-row counts summed.
            const auto *row = static_cast<const rf::Row *>(rows.contents);
            std::uint64_t factored = 0, threshold = 0;
            #pragma omp parallel for schedule(static) reduction(+ : factored, threshold)
            for (unsigned r = 0; r < m; ++r) {
                factored += row[r].factored;
                threshold += row[r].dropped_threshold;
            }
            out.drop.nnz_factor = factored;
            out.drop.dropped = factored - nnz;
            out.drop.dropped_threshold = threshold;
            out.drop.dropped_flush = out.drop.dropped - threshold;
        }
        out.backward.idx = allocate(nnz * sizeof(unsigned));
        out.forward.idx = allocate(nnz * sizeof(unsigned));
        const auto bytes = half ? sizeof(_Float16) : sizeof(float);
        out.backward.val = allocate(nnz * bytes);
        out.forward.val = allocate(nnz * bytes);
        cb = [queue commandBuffer];
        encode(cb, rf::stage_emit);
        encode(cb, rf::stage_count_transpose);
        scan(cb, out.forward.ptr, m + 1, scan_scratch);
        encode(cb, rf::stage_initialize_cursor);
        encode(cb, rf::stage_emit_transpose);
        finish(cb, status);
        rows = nil;
        cursor = nil;
        scan_scratch = nil;
        prepare_plans<gd::lanes>(out.forward, out.backward, m, input.arena,
            input.layout.offsets[unsigned(resident_setup::Field::round_ptr)], state.round, status);
        return out;
    }
};
} // namespace apxchol::detail::metal_setup
