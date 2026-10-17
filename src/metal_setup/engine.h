#pragma once
#include "source.h"
#include "sparsify.h"
#include "apxchol/solver/detail/residual_policy.h"
#include "../metal_setup_options.h"
#include <cstdlib>
#include "../metal_compile.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
namespace apxchol::detail::metal_setup {
using namespace resident_setup;
using Wide = apxchol::detail::metal_host::df;
using metal::describe;
inline void require(bool ok, const char *text) {
    if (!ok)
        throw std::runtime_error(text);
}
struct Factor {
    id<MTLBuffer> arena, factor;
    Layout layout;
    View<Wide> data() const {
        return view<Wide>(static_cast<unsigned char *>(arena.contents),
                          static_cast<Edge *>(factor.contents), layout);
    }
};
class Engine {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    NSMutableArray *pipelines, *sparse;

  public:
    Engine(id<MTLDevice> dev) : device(dev) {
        NSError *error = nil;
        require(device != nil, "No Metal device for factorization");
        auto library = metal::compile_library(device, resident_metal_source).library;
        pipelines = [NSMutableArray new];
        for (unsigned stage = 0; stage < 19; ++stage) {
            auto values = [MTLFunctionConstantValues new];
            [values setConstantValue:&stage type:MTLDataTypeUInt atIndex:0];
            auto fn = [library newFunctionWithName:@"setup_stage"
                                    constantValues:values
                                             error:&error];
            if (!fn)
                throw std::runtime_error(describe(error));
            auto pipe = [device newComputePipelineStateWithFunction:fn error:&error];
            if (!pipe)
                throw std::runtime_error(describe(error));
            [pipelines addObject:pipe];
        }
        for (NSString *name in @[ @"scan_blocks", @"scan_add", @"import_count", @"import_emit", @"factor_tail" ]) {
            auto pipe =
                [device newComputePipelineStateWithFunction:[library newFunctionWithName:name]
                                                      error:&error];
            if (!pipe)
                throw std::runtime_error(describe(error));
            require(pipe.maxTotalThreadsPerThreadgroup >= 256, "Metal setup requires 256-thread workgroups");
            [pipelines addObject:pipe];
        }

        sparse = sparse_pipelines(device, library);
        queue = [device newCommandQueue];
        require(queue != nil, "Metal setup command queue creation failed");
    }
    Factor run(unsigned n, unsigned entries, unsigned seed, id<MTLBuffer> input_ptr,
               id<MTLBuffer> input_idx, id<MTLBuffer> input_values, float regularization,
               unsigned triangles_identical, const factor_options& options, std::size_t residual_threshold) {
        @autoreleasepool {
            const unsigned capacity = std::max<unsigned>(entries, 1);
            const auto l = layout(n, capacity);
            auto allocate = [&](std::size_t size) {
                auto b = [device newBufferWithLength:std::max(size, std::size_t(16))
                                             options:MTLResourceStorageModeShared];
                require(b != nil, "Metal setup buffer allocation failed");
                return b;
            };
            auto arena = allocate(l.bytes);
            std::memset(arena.contents, 0, sizeof(State));
            std::size_t factor_capacity = std::max<std::size_t>(4 * std::size_t(capacity), n);
            auto factor = allocate(factor_capacity * sizeof(Edge));
            auto s = view<Wide>(static_cast<unsigned char *>(arena.contents),
                                static_cast<Edge *>(factor.contents), l);
            s.state->nodes = n;
            s.state->edges = entries;
            s.state->edge_capacity = capacity;
            s.state->active = n;
            s.state->seed = seed;
            s.state->prune_gap = s.state->prune_enabled = 1;
            s.state->degree_divisor = metal_degree_divisor(options.partition.degree_quantile);
            NSMutableArray *scratch = [NSMutableArray new];
            for (unsigned count = n + 1;;) {
                count = (count + 255) / 256;
                [scratch addObject:allocate(count * sizeof(Counts))];
                if (count <= 1)
                    break;
            }
            auto finish = [&](id<MTLCommandBuffer> command) {
                [command commit];
                [command waitUntilCompleted];
                if (command.status != MTLCommandBufferStatusCompleted)
                    throw std::runtime_error("Metal setup command buffer failed: " + describe(command.error));
            };
            auto stage = [&](id<MTLComputeCommandEncoder> encoder, unsigned stage, unsigned count,
                             unsigned limit = 0) {
                id<MTLComputePipelineState> pipe = pipelines[stage];
                [encoder setComputePipelineState:pipe];
                [encoder setBuffer:arena offset:0 atIndex:0];
                [encoder setBuffer:factor offset:0 atIndex:1];
                [encoder setBytes:&l length:sizeof(l) atIndex:2];
                [encoder setBytes:&limit length:sizeof(limit) atIndex:3];
                const unsigned simd_width = pipe.threadExecutionWidth;
                const unsigned width = simd_width;
                require(width && simd_width % width == 0, "invalid SIMD subgroup width");
                [encoder setBytes:&width length:sizeof(width) atIndex:4];
                const bool grouped = stage_is_grouped(stage);
                const unsigned threads =
                    grouped ? simd_width : std::min<unsigned>(256, pipe.maxTotalThreadsPerThreadgroup);
                if (grouped) {
                    require(count <= ~0u / width, "Metal grouped dispatch exceeds 32-bit thread indices");
                    count *= width;
                }
                [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            };
            std::function<void(id<MTLComputeCommandEncoder>, id<MTLBuffer>, std::size_t, unsigned,
                               unsigned)>
                scan;
            scan = [&](id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> input, std::size_t offset,
                       unsigned count, unsigned depth) {
                const unsigned groups = (count + 255) / 256;
                id<MTLBuffer> sums = scratch[depth];
                [encoder setComputePipelineState:pipelines[19]];
                [encoder setBuffer:input offset:offset atIndex:0];
                [encoder setBuffer:sums offset:0 atIndex:1];
                [encoder setBytes:&count length:sizeof(count) atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                if (groups > 1) {
                    scan(encoder, sums, 0, groups, depth + 1);
                    [encoder setComputePipelineState:pipelines[20]];
                    [encoder setBuffer:input offset:offset atIndex:0];
                    [encoder setBuffer:sums offset:0 atIndex:1];
                    [encoder setBytes:&count length:sizeof(count) atIndex:2];
                    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
            };
            struct EncoderRound {
                decltype(stage) &dispatch;
                decltype(scan) &prefix;
                id<MTLComputeCommandEncoder> encoder;
                id<MTLBuffer> arena;
                std::size_t offset;
                void stage(unsigned phase, unsigned count, unsigned limit = 0) const {
                    dispatch(encoder, phase, count, limit);
                }
                void scan(unsigned count) const {
                    prefix(encoder, arena, offset, count, 0);
                }
            };
            auto command = [queue commandBuffer];
            auto compute = [command computeCommandEncoder];
            auto import = [&](unsigned phase) {
                auto enc = compute;
                [enc setComputePipelineState:pipelines[21 + phase]];
                [enc setBuffer:arena offset:0 atIndex:0];
                [enc setBuffer:factor offset:0 atIndex:1];
                [enc setBytes:&l length:sizeof l atIndex:2];
                [enc setBuffer:input_ptr offset:0 atIndex:3];
                [enc setBuffer:input_idx offset:0 atIndex:4];
                [enc setBuffer:input_values offset:0 atIndex:5];
                if (!phase)
                    [enc setBytes:&regularization length:sizeof regularization atIndex:6];
                [enc setBytes:&triangles_identical length:sizeof triangles_identical atIndex:7];
                [enc dispatchThreads:MTLSizeMake(n + 1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            };
            import(0);
            scan(compute, arena, l.offsets[unsigned(Field::counts)], n + 1, 0);
            import(1);
            stage(compute, stage_initialize, n);
            [compute endEncoding];
            finish(command);
            const char* sparse_value = std::getenv("APXCHOL_RESIDUAL_SPARSIFY");
            bool sparse_attempted = sparse_value && std::strcmp(sparse_value, "0") == 0;
            id<MTLBuffer> sparse_state = nil;
            while (!s.state->status && s.state->active) {
                require(s.state->round < n, "Metal factorization exceeded its round bound");
                // Independent selected rows share no edge. Across a complete tail,
                // each remaining vertex pair contributes at most one factor entry.
                const bool tail = s.state->active <= 128;
                const std::uint64_t bound = tail
                    ? std::uint64_t(s.state->active) * (s.state->active - 1) / 2
                    : s.state->edges / 2;
                require(bound <= std::numeric_limits<std::size_t>::max() - s.state->entries,
                        "Metal factor reservation exceeds addressable storage");
                const auto needed = std::size_t(s.state->entries) + std::size_t(bound);
                if (needed > factor_capacity) {
                    factor_capacity = tail ? needed : std::max(2 * factor_capacity, needed);
                    auto grown = allocate(factor_capacity * sizeof(Edge));
                    if (s.state->entries) {
                        command = [queue commandBuffer];
                        auto encoder = [command blitCommandEncoder];
                        [encoder copyFromBuffer:factor
                                   sourceOffset:0
                                       toBuffer:grown
                              destinationOffset:0
                                           size:s.state->entries * sizeof(Edge)];
                        [encoder endEncoding];
                        finish(command);
                    }
                    factor = grown;
                    s.factors = static_cast<Edge *>(factor.contents);
                }
                if (tail) {
                    s.state->prune_enabled = 0;
                    command = [queue commandBuffer];
                    compute = [command computeCommandEncoder];
                    id<MTLComputePipelineState> pipe = pipelines[23];
                    [compute setComputePipelineState:pipe];
                    [compute setBuffer:arena offset:0 atIndex:0];
                    [compute setBuffer:factor offset:0 atIndex:1];
                    [compute setBytes:&l length:sizeof(l) atIndex:2];
                    [compute dispatchThreadgroups:MTLSizeMake(1,1,1)
                            threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                    [compute endEncoding];
                    finish(command);
                    continue;
                }
                const unsigned rounds =
                    bound ? std::min<std::size_t>(
                                {2, s.state->active, (factor_capacity - s.state->entries) / bound})
                          : 1;
                const unsigned row_bound = s.state->active;
                const auto before = std::size_t(s.state->active) + s.state->selected;
                const double min_yield = detail::adaptive_is_yield_fraction(options.min_is_fraction,
                    before, before ? double(s.state->edges) / before : 0, residual_threshold);
                const bool handoff = s.state->round &&
                    detail::selection_should_handoff(s.state->selected, s.state->selected,
                        s.state->eligible, before, min_yield, residual_threshold, options.omp_threshold);
                if (handoff) s.state->prune_enabled = 0;
                const bool sparsify = handoff && !sparse_attempted;
                if (sparsify) {
                    sparse_attempted = true;
                    id<MTLComputePipelineState> coalesce_pipe = sparse[sparse_coalesce];
                    require(coalesce_pipe.threadExecutionWidth &&
                            s.state->active <= ~0u / coalesce_pipe.threadExecutionWidth,
                            "Metal sparsification dispatch exceeds 32-bit indices");
                    sparse_state = allocate(sizeof(SparseState<Wide>));
                    auto state = static_cast<SparseState<Wide>*>(sparse_state.contents);
                    *state = {};
                    state->threshold = unsigned(std::min<std::size_t>(residual_threshold, ~0u));
                    state->selected = s.state->selected;
                }
                command = [queue commandBuffer];
                compute = [command computeCommandEncoder];
                if (sparsify)
                    encode_sparsify(compute, sparse, arena, factor, l, sparse_state,
                        s.state->active, n, s.state->edges / 2, [&] {
                            scan(compute, arena, l.offsets[unsigned(Field::counts)], n + 1, 0);
                        });
                for (unsigned round = 0; round < rounds; ++round)
                    factor_round(
                        EncoderRound{stage, scan, compute, arena, l.offsets[unsigned(Field::counts)]},
                        row_bound);
                [compute endEncoding];
                finish(command);
            }

            if (s.state->status || s.state->columns != n)
                throw std::runtime_error("Metal factorization failed: status " + std::to_string(s.state->status) +
                    ", completed columns " + std::to_string(s.state->columns) + "/" + std::to_string(n));
            return {arena, factor, l};
        }
    }
};
} // namespace apxchol::detail::metal_setup
