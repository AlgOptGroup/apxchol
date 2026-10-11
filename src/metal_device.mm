// Device side of the Metal PCG (see src/metal_device.h). Objective-C++
// with ARC; no Eigen, no OpenMP. Every per-call Metal object lives in an
// @autoreleasepool, and every command buffer's status is checked after
// waitUntilCompleted. Errors are collected inside the pool and thrown after
// it, so no C++ exception unwinds through an autorelease pool.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_device.h"

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include "apxchol/solver/detail/metal_schedule.h"
#include <stdexcept>

namespace apxchol::detail::metal {
namespace {

const char kKernelSource[] =
    "\n#include <metal_stdlib>\nusing namespace metal;\n"
    "\n#define APXCHOL_GPU_FUNCTION inline\n"
    "\n#define APXCHOL_GPU_BUFFER(T) device T*\n"
#define APXCHOL_GPU_CODE(...) #__VA_ARGS__
#include "apxchol/solver/detail/gpu_pcg.inc"
#include "apxchol/solver/detail/double_float.inc"
#include "metal_kernels.inc"
    "\n#define APXCHOL_UNROLL _Pragma(\"unroll\")\n"
#include "gpu_dataflow.inc"
#undef APXCHOL_GPU_CODE
#include "metal_dataflow.inc"
    ;

constexpr std::uint32_t kTreeRows = 256;
constexpr std::uint32_t kTreeLanes = 16;
constexpr std::uint32_t kRowThreads = 256;     // threads per threadgroup of the row kernels

using namespace gpu_pcg;

std::string describe(NSError* error) {
    if (error == nil) return "unknown error";
    const char* text = [[error localizedDescription] UTF8String];
    return text ? std::string(text) : std::string("unknown error");
}

struct pipelines {
    id<MTLComputePipelineState> pcg[4][2];  // operation, operator double-float
    id<MTLComputePipelineState> finalize;
    id<MTLComputePipelineState> dataflow_solve[2], dataflow_init[2];
    id<MTLComputePipelineState> probe;
};

struct context {
    device_status status;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> probe_queue = nil;
    pipelines pipes;
};

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                          const char* name, bool forward, bool operator_df,
                                          std::string& error, unsigned op = 0) {
    MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
    [values setConstantValue:&forward type:MTLDataTypeBool atIndex:0];
    [values setConstantValue:&operator_df type:MTLDataTypeBool atIndex:1];
    [values setConstantValue:&op type:MTLDataTypeUInt atIndex:2];
    NSError* err = nil;
    id<MTLFunction> fn = [library newFunctionWithName:[NSString stringWithUTF8String:name]
                                       constantValues:values
                                                error:&err];
    if (fn == nil) {
        error = std::string("function ") + name + ": " + describe(err);
        return nil;
    }
    id<MTLComputePipelineState> pipe = [device newComputePipelineStateWithFunction:fn error:&err];
    if (pipe == nil) error = std::string("pipeline ") + name + ": " + describe(err);
    return pipe;
}

id<MTLLibrary> compile(id<MTLDevice> device, bool contract_pragma, std::string& error) {
    MTLCompileOptions* options = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) {
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        options.languageVersion = MTLLanguageVersion3_2;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    std::string source = contract_pragma ? "#pragma METAL fp contract(off)\n" : "";
    source += kKernelSource;
    NSError* err = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                  options:options
                                                    error:&err];
    if (library == nil) error = describe(err);
    return library;
}

context* make_context() {
    auto* ctx = new context;  // process lifetime: never destroyed
    device_status& st = ctx->status;
    @autoreleasepool {
        // Command-line tools need CoreGraphics linked for this to find the GPU.
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            st.error = "no Metal device";
            return ctx;
        }
        ctx->device = device;
        st.name = [[device name] UTF8String];
        std::string error;
        id<MTLLibrary> library = compile(device, true, error);
        st.contract_pragma = library != nil;
        if (library == nil) {
            std::string retry;
            library = compile(device, false, retry);
            if (library == nil) {
                st.error = "Metal kernel compilation failed: " + retry;
                return ctx;
            }
        }
        pipelines& p = ctx->pipes;
        for (unsigned op = 0; op < 4; ++op)
            for (unsigned df = 0; df < (op == unsigned(operation::multiply) ? 2u : 1u); ++df)
                p.pcg[op][df] = make_pipeline(device, library, "pcg_step", false, df != 0, error, op);
        p.finalize = make_pipeline(device, library, "finalize", false, false, error);
        for (int fwd = 0; fwd < 2; ++fwd) {
            p.dataflow_init[fwd] = make_pipeline(device, library, "dataflow_init", fwd != 0, false, error);
            p.dataflow_solve[fwd] = make_pipeline(device, library, "dataflow_solve", fwd != 0, false, error);
        }
        p.probe = make_pipeline(device, library, "df_probe", false, false, error);
        if (!error.empty()) {
            st.error = "Metal pipeline creation failed: " + error;
            return ctx;
        }
        for (auto pipe : p.dataflow_solve) {
            if ([pipe threadExecutionWidth] != 32 || [pipe maxTotalThreadsPerThreadgroup] < 256) {
                st.error = "Metal dataflow requires 32-thread SIMD groups and 256-thread workgroups";
                return ctx;
            }
        }
        auto threads = [](id<MTLComputePipelineState> pipe) {
            return static_cast<std::uint32_t>([pipe maxTotalThreadsPerThreadgroup]);
        };
        const auto tree_threads = std::min({threads(p.pcg[0][0]), threads(p.pcg[0][1]), threads(p.pcg[1][0]),
                                           threads(p.pcg[2][0]), threads(p.finalize)});
        st.row_threads = std::min({threads(p.pcg[unsigned(operation::direction)][0]),
                                  threads(p.dataflow_init[0]), threads(p.dataflow_init[1])});
        ctx->probe_queue = [device newCommandQueue];
        if (ctx->probe_queue == nil) {
            st.error = "no Metal command queue";
            return ctx;
        }
        if (tree_threads < kTreeLanes || st.row_threads == 0) {
            st.error = "Metal pipeline thread limits are insufficient";
            return ctx;
        }
        st.ok = true;
    }
    return ctx;
}

context& shared() {
    static context* ctx = make_context();
    return *ctx;
}

id<MTLBuffer> upload(id<MTLDevice> device, const void* data, std::size_t bytes) {
    const std::size_t len = std::max<std::size_t>(bytes, 4);
    if (len > [device maxBufferLength]) return nil;
    id<MTLBuffer> b = [device newBufferWithLength:len options:MTLResourceStorageModeShared];
    if (b == nil) return nil;
    if (data != nullptr && bytes != 0) std::memcpy([b contents], data, bytes);
    else std::memset([b contents], 0, len);
    return b;
}

// Waits for `cb` and returns its error text ("" on success).
std::string finish(id<MTLCommandBuffer> cb) {
    [cb commit];
    [cb waitUntilCompleted];
    if ([cb status] != MTLCommandBufferStatusCompleted || [cb error] != nil)
        return "Metal command buffer failed: " + describe([cb error]);
    return {};
}

struct prepared_tri {
    id<MTLBuffer> words = nil, blocks = nil, segments = nil;
    unsigned tag_count = 0;
    std::vector<metal_schedule::detail::step> stages;
    id<MTLBuffer> ptr = nil, col = nil, val = nil;
    std::uint32_t levels = 0;
};

prepared_tri prepare_tri(context &ctx, const tri_arrays &a, std::uint32_t m, bool forward) {
    prepared_tri out;
    auto schedule =
        metal_schedule::build({a.ptr, std::size_t(m) + 1}, {a.col, a.nnz}, {a.val, a.nnz}, !forward);
    auto &packed = schedule.tasks;
    out.tag_count = m + packed.slots;
    // Metal validates the element type even when no task reads the buffer.
    if (packed.segments.empty())
        packed.segments.push_back({});
    out.stages = std::move(schedule.stages);
    out.levels = schedule.levels;
    @autoreleasepool {
        out.ptr = upload(ctx.device, a.ptr, (m + 1) * sizeof(unsigned));
        out.col = upload(ctx.device, a.col, a.nnz * sizeof(unsigned));
        out.val = upload(ctx.device, a.val, a.nnz * sizeof(float));
        out.words = upload(ctx.device, packed.words.data(), packed.words.size() * sizeof(unsigned));
        out.blocks = upload(ctx.device, packed.blocks.data(), packed.blocks.size() * sizeof(unsigned));
        out.segments = upload(ctx.device, packed.segments.data(),
                              packed.segments.size() * sizeof(gpu_dataflow::segment));
    }
    if (out.ptr == nil || out.col == nil || out.val == nil || out.words == nil || out.blocks == nil ||
        out.segments == nil)
        throw device_memory_error("Metal factor allocation failed");
    return out;
}

} // namespace

const device_status &status() noexcept { return shared().status; }

std::vector<float> run_probe(const std::vector<float> &inputs) {
    context &ctx = shared();
    if (!ctx.status.ok) throw std::runtime_error(ctx.status.error);
    const std::uint32_t count = static_cast<std::uint32_t>(inputs.size() / 4);
    std::vector<float> out(static_cast<std::size_t>(count) * 16);
    std::string error;
    @autoreleasepool {
        id<MTLBuffer> in = upload(ctx.device, inputs.data(), inputs.size() * sizeof(float));
        id<MTLBuffer> res = upload(ctx.device, nullptr, out.size() * sizeof(float));
        if (in == nil || res == nil) {
            error = "Metal buffer allocation failed";
        } else {
            id<MTLCommandBuffer> cb = [ctx.probe_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            if (enc == nil) {
                error = "Metal command buffer or encoder creation failed";
            } else {
                [enc setComputePipelineState:ctx.pipes.probe];
                [enc setBuffer:in offset:0 atIndex:0];
                [enc setBuffer:res offset:0 atIndex:1];
                [enc setBytes:&count length:sizeof count atIndex:2];
                const NSUInteger w = 64;
                [enc dispatchThreadgroups:MTLSizeMake((count + w - 1) / w, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
                [enc endEncoding];
                error = finish(cb);
                if (error.empty()) std::memcpy(out.data(), [res contents], out.size() * sizeof(float));
            }
        }
    }
    if (!error.empty()) throw std::runtime_error(error);
    return out;
}

struct engine::impl {
    context* ctx = nullptr;
    id<MTLCommandQueue> queue = nil;
    params arguments{};
    bool operator_df = false;
    id<MTLBuffer> op_ptr = nil, op_col = nil, op_hi = nil, op_lo = nil;
    prepared_tri fwd, bwd;
    id<MTLBuffer> x = nil, r = nil, ap = nil, p = nil, z = nil, partial = nil, state_buffer = nil;
    id<MTLBuffer> tags = nil;

    static void bind(id<MTLComputeCommandEncoder> enc, unsigned index,
                     std::initializer_list<id<MTLBuffer>> buffers) {
        for (auto buffer : buffers) [enc setBuffer:buffer offset:0 atIndex:index++];
    }

    void dispatch(id<MTLComputePipelineState> pipe, const params& prm,
                  unsigned groups, unsigned threads) const {
        [encoder setComputePipelineState:pipe];
        [encoder setBytes:&prm length:sizeof prm atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    }

    void finalize(unsigned mode, unsigned it) const {
        params prm = arguments;
        prm.mode = mode;
        prm.iter = it;
        dispatch(ctx->pipes.finalize, prm, 1, kTreeLanes);
    }

    void encode_tri(const prepared_tri& t, bool forward) const {
        if (!arguments.m) return;
        bind(encoder, 13, {t.ptr, t.col, t.val, t.words, t.blocks, t.segments});
        params prm = arguments;
        prm.count = t.tag_count;
        const unsigned threads = std::min(kRowThreads, ctx->status.row_threads);
        dispatch(ctx->pipes.dataflow_init[forward], prm, (t.tag_count + threads - 1) / threads, threads);
        for (auto stage : t.stages) {
            prm.offset = stage.first;
            dispatch(ctx->pipes.dataflow_solve[forward], prm, stage.last - stage.first, 256);
        }
    }

    void precondition() const {
        encode_tri(fwd, true);
        encode_tri(bwd, false);
    }

    void step(operation op) const {
        const auto pipe = ctx->pipes.pcg[unsigned(op)][op == operation::multiply && operator_df];
        const bool direction = op == operation::direction;
        const unsigned threads = direction ? std::min(kRowThreads, ctx->status.row_threads) : kTreeLanes;
        dispatch(pipe, arguments, direction ? (arguments.n + threads - 1) / threads : arguments.groups, threads);
    }

    id<MTLComputeCommandEncoder> encoder = nil;

    template<class Record>
    bool submit(bool first, Record&& record) {
        std::string error;
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = first ? [cb blitCommandEncoder] : nil;
            if (blit != nil) {
                [blit fillBuffer:x range:NSMakeRange(0, static_cast<NSUInteger>(arguments.n) * sizeof(df32)) value:0];
                [blit endEncoding];
            }
            encoder = cb != nil && (!first || blit != nil) ? [cb computeCommandEncoder] : nil;
            if (encoder == nil) error = "Metal command buffer or encoder creation failed";
            else {
                bind(encoder, 1, {x, r, ap, p, z, partial, state_buffer, op_ptr, op_col, op_hi,
                                  op_lo != nil ? op_lo : op_hi, tags});
                record();
                [encoder endEncoding];
                error = finish(cb);
                encoder = nil;
            }
        }
        if (!error.empty()) throw std::runtime_error(error);
        return static_cast<const solve_state*>([state_buffer contents])->active != 0;
    }
};

engine::engine(const operator_arrays& op, const tri_arrays& fwd, const tri_arrays& bwd,
               std::uint32_t m)
    : impl_(std::make_unique<impl>()) {
    context& ctx = shared();
    if (!ctx.status.ok) throw std::runtime_error(ctx.status.error);
    impl& s = *impl_;
    s.ctx = &ctx;
    s.arguments.n = static_cast<std::uint32_t>(op.n);
    s.arguments.m = m;
    s.arguments.groups = (s.arguments.n + kTreeRows - 1) / kTreeRows;
    s.operator_df = op.lo != nullptr;
    bool ok = true;
    @autoreleasepool {
        id<MTLDevice> dev = ctx.device;
        s.queue = [dev newCommandQueue];
        s.op_ptr = upload(dev, op.ptr, (op.n + 1) * sizeof(std::uint32_t));
        s.op_col = upload(dev, op.col, op.nnz * sizeof(std::uint32_t));
        s.op_hi = upload(dev, op.hi, op.nnz * sizeof(float));
        if (op.lo != nullptr) s.op_lo = upload(dev, op.lo, op.nnz * sizeof(float));
        ok = s.op_ptr != nil && s.op_col != nil && s.op_hi != nil &&
             (op.lo == nullptr || s.op_lo != nil);
    }
    if (s.queue == nil) throw std::runtime_error("Metal command queue creation failed");
    if (!ok) throw device_memory_error("Metal buffer allocation failed for the operator or factor");
    s.fwd = prepare_tri(ctx, fwd, m, true);
    s.bwd = prepare_tri(ctx, bwd, m, false);
    ok = true;
    @autoreleasepool {
        id<MTLDevice> dev = s.ctx->device;
        const std::size_t entries = static_cast<std::size_t>(s.arguments.n);
        s.x = upload(dev, nullptr, entries * sizeof(df32));
        s.r = upload(dev, nullptr, entries * sizeof(df32));
        s.ap = upload(dev, nullptr, entries * sizeof(df32));
        s.p = upload(dev, nullptr, entries * sizeof(float));
        // Sweeps write rows < m; the extra Laplacian coordinate stays zero.
        s.z = upload(dev, nullptr, entries * sizeof(float));
        s.partial = upload(dev, nullptr, static_cast<std::size_t>(s.arguments.groups) * sizeof(df32));
        s.state_buffer = upload(dev, nullptr, sizeof(solve_state));
        s.tags = upload(dev, nullptr, std::max({entries, std::size_t(s.fwd.tag_count),
            std::size_t(s.bwd.tag_count)}) * sizeof(std::uint32_t));
        ok = s.x != nil && s.r != nil && s.ap != nil && s.p != nil && s.z != nil &&
             s.partial != nil && s.state_buffer != nil &&
             s.tags != nil;
    }
    if (!ok) throw device_memory_error("Metal buffer allocation failed for the PCG vectors");
}

engine::~engine() {
    @autoreleasepool {
        impl_.reset();
    }
}

df32* engine::r() noexcept { return static_cast<df32*>([impl_->r contents]); }
df32* engine::x() noexcept { return static_cast<df32*>([impl_->x contents]); }
solve_state& engine::state() noexcept { return *static_cast<solve_state*>([impl_->state_buffer contents]); }

factor_statistics engine::factor_stats() const {
    return {impl_->fwd.levels, impl_->bwd.levels, impl_->fwd.stages.size() + unsigned(impl_->arguments.m != 0), impl_->bwd.stages.size() + unsigned(impl_->arguments.m != 0)};
}

void engine::solve(std::uint32_t max_iter,
                   std::uint32_t window, std::uint32_t check_every) {
    impl& s = *impl_;
    s.arguments.window = window;
    gpu_pcg::solve(s, max_iter, check_every);
}

const float* engine::apply() {
    impl& s = *impl_;
    s.submit(false, [&] { s.precondition(); });
    return static_cast<const float*>([s.z contents]);
}

}  // namespace apxchol::detail::metal
