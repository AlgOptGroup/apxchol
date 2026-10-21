#pragma once
#include "source.h"
#include "../metal_compile.h"
#import <Metal/Metal.h>
namespace apxchol::detail::metal_setup {
inline NSMutableArray* sparse_pipelines(id<MTLDevice> device, id<MTLLibrary> library) {
    auto pipelines = [NSMutableArray new];
    for (unsigned phase = 0; phase < resident_setup::sparse_stage_count; ++phase) {
        NSError* error = nil;
        auto constants = [MTLFunctionConstantValues new];
        [constants setConstantValue:&phase type:MTLDataTypeUInt atIndex:0];
        auto function = [library newFunctionWithName:@"sparsify_stage" constantValues:constants error:&error];
        auto pipeline = function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
        if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256)
            throw std::runtime_error("Metal sparsification pipeline: " + metal::describe(error));
        [pipelines addObject:pipeline];
    }
    return pipelines;
}
struct SparseDispatch {
    id<MTLComputeCommandEncoder> encoder;
    NSArray* pipelines;
    id<MTLBuffer> arena, factor;
    resident_setup::Layout layout;
    id<MTLBuffer> state;

    void operator()(unsigned phase, unsigned count, bool initial = false) const {
        using namespace resident_setup;
        id<MTLComputePipelineState> pipeline = pipelines[phase];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:arena offset:0 atIndex:0];
        [encoder setBuffer:factor offset:0 atIndex:1];
        [encoder setBytes:&layout length:sizeof(layout) atIndex:2];
        [encoder setBuffer:state offset:0 atIndex:3];
        [encoder setBytes:&initial length:sizeof(initial) atIndex:4];
        if (phase == sparse_coalesce) {
            [encoder dispatchThreadgroups:MTLSizeMake(std::max(count, 1u), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            return;
        }
        if (phase == sparse_partials) count = ((count + 16383) / 16384) * 256;
        [encoder dispatchThreads:MTLSizeMake(std::max(count, 1u), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
};
template<class Scan>
void encode_sparsify(id<MTLComputeCommandEncoder> encoder, NSArray* pipelines,
        id<MTLBuffer> arena, id<MTLBuffer> factor, resident_setup::Layout layout,
        id<MTLBuffer> state, unsigned active, unsigned nodes, unsigned edge_bound, Scan scan) {
    using namespace resident_setup;
    const SparseDispatch dispatch{encoder, pipelines, arena, factor, layout, state};
    // These arrays are round scratch. Neither factor entries, excess nor the
    // compact active frontier is overwritten while sparsification owns them.
    dispatch(sparse_initialize, nodes + 1);
    dispatch(sparse_coalesce, active);
    scan();
    dispatch(sparse_collect, active);
    dispatch(sparse_gate, 1);
    dispatch(sparse_importance, edge_bound);
    for (unsigned count = active; count > 1; count = (count + 1) / 2) {
        dispatch(sparse_reset, active);
        dispatch(sparse_bucket, edge_bound);
        dispatch(sparse_ordinal, edge_bound);
        dispatch(sparse_contract, active);
        dispatch(sparse_labels, active);
    }
    for (unsigned pass = 0; pass < 7; ++pass) {
        dispatch(sparse_partials, edge_bound, pass == 0);
        dispatch(sparse_fold, 1, pass == 0);
    }
    dispatch(sparse_clear, nodes + 1);
    dispatch(sparse_sample, edge_bound);
    scan();
    dispatch(sparse_offsets, nodes + 1);
    dispatch(sparse_emit, edge_bound);
    dispatch(sparse_finish, 1);
}
}
