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
#include "apxchol/solver/detail/metal_schedule.h"
#include <stdexcept>

namespace apxchol::detail::metal {
namespace {

const char kKernelSource[] =
#include "metal_kernels.inc"
    "\n#define APXCHOL_GPU_FUNCTION inline\n"
    "\n#define APXCHOL_UNROLL _Pragma(\"unroll\")\n"
#define APXCHOL_GPU_CODE(...) #__VA_ARGS__
#include "gpu_dataflow.inc"
#undef APXCHOL_GPU_CODE
#include "metal_dataflow.inc"
    ;

constexpr std::uint32_t kTreeRows = 256;
constexpr std::uint32_t kTreeLanes = 16;
constexpr std::uint32_t kRowThreads = 256;     // threads per threadgroup of the row kernels

enum finalize_mode : std::uint32_t {
    kFinalizeMuR = 0,
    kFinalizeRzInit = 1,
    kFinalizeMuZ = 2,
    kFinalizeAlpha = 3,
    kFinalizeCheck = 4,
    kFinalizeRz = 5,
};
enum reduce_mode : std::uint32_t { kReduceSumR = 0, kReduceSumZ = 1, kReduceRz = 2 };

std::string describe(NSError* error) {
    if (error == nil) return "unknown error";
    const char* text = [[error localizedDescription] UTF8String];
    return text ? std::string(text) : std::string("unknown error");
}

struct pipelines {
    id<MTLComputePipelineState> spmv[2];      // [operator double-float]
    id<MTLComputePipelineState> update_xr;
    id<MTLComputePipelineState> reduce[3];    // [reduce_mode]
    id<MTLComputePipelineState> finalize;
    id<MTLComputePipelineState> dataflow_solve[2], dataflow_init[2];
    id<MTLComputePipelineState> p_update;
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
                                          std::uint32_t reduce, std::string& error) {
    MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
    [values setConstantValue:&forward type:MTLDataTypeBool atIndex:0];
    [values setConstantValue:&operator_df type:MTLDataTypeBool atIndex:1];
    [values setConstantValue:&reduce type:MTLDataTypeUInt atIndex:2];
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
        for (int df = 0; df < 2; ++df)
            p.spmv[df] = make_pipeline(device, library, "spmv_pap", false, df != 0, 0, error);
        p.update_xr = make_pipeline(device, library, "update_xr", false, false, 0, error);
        for (std::uint32_t mode = 0; mode < 3; ++mode)
            p.reduce[mode] = make_pipeline(device, library, "reduce", false, false, mode, error);
        p.finalize = make_pipeline(device, library, "finalize", false, false, 0, error);
        for (int fwd = 0; fwd < 2; ++fwd) {
            p.dataflow_init[fwd] = make_pipeline(device, library, "dataflow_init", fwd != 0, false, 0, error);
            p.dataflow_solve[fwd] = make_pipeline(device, library, "dataflow_solve", fwd != 0, false, 0, error);
        }
        p.p_update = make_pipeline(device, library, "p_update", false, false, 0, error);
        p.probe = make_pipeline(device, library, "df_probe", false, false, 0, error);
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
        st.tree_threads = std::min({threads(p.spmv[0]), threads(p.spmv[1]), threads(p.update_xr),
                                    threads(p.reduce[0]), threads(p.reduce[1]),
                                    threads(p.reduce[2]), threads(p.finalize)});
        st.row_threads = threads(p.p_update);
        ctx->probe_queue = [device newCommandQueue];
        if (ctx->probe_queue == nil) {
            st.error = "no Metal command queue";
            return ctx;
        }
        if (st.tree_threads < kTreeLanes || st.row_threads == 0) {
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
    std::uint32_t n = 0, m = 0, groups = 0;
    bool laplacian = false;
    bool operator_df = false;
    float inv_n_hi = 0.0f, inv_n_lo = 0.0f;
    id<MTLBuffer> op_ptr = nil, op_col = nil, op_hi = nil, op_lo = nil;
    prepared_tri fwd, bwd;
    id<MTLBuffer> x = nil, r = nil, ap = nil, p = nil, z = nil, partial = nil, cols = nil;
    id<MTLBuffer> tags = nil;

    params base() const {
        params prm{};
        prm.n = n;
        prm.groups = groups;
        prm.lap = laplacian ? 1u : 0u;
        prm.m = m;
        prm.inv_n_hi = inv_n_hi;
        prm.inv_n_lo = inv_n_lo;
        return prm;
    }

    static void set_params(id<MTLComputeCommandEncoder> enc, const params& prm) {
        [enc setBytes:&prm length:sizeof prm atIndex:0];
    }

    void dispatch_tree(id<MTLComputeCommandEncoder> enc) const {
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kTreeLanes, 1, 1)];
    }

    void dispatch_rows(id<MTLComputeCommandEncoder> enc, std::uint32_t rows) const {
        const std::uint32_t per = std::min(kRowThreads, ctx->status.row_threads);
        [enc dispatchThreadgroups:MTLSizeMake((rows + per - 1) / per, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(per, 1, 1)];
    }

    void encode_finalize(id<MTLComputeCommandEncoder> enc, std::uint32_t mode,
                         std::uint32_t iter, std::uint32_t window) const {
        params prm = base();
        prm.mode = mode;
        prm.iter = iter;
        prm.window = window;
        [enc setComputePipelineState:ctx->pipes.finalize];
        set_params(enc, prm);
        [enc setBuffer:partial offset:0 atIndex:1];
        [enc setBuffer:cols offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kTreeLanes, 1, 1)];
    }

    void encode_reduce(id<MTLComputeCommandEncoder> enc, std::uint32_t mode) const {
        [enc setComputePipelineState:ctx->pipes.reduce[mode]];
        set_params(enc, base());
        [enc setBuffer:r offset:0 atIndex:1];
        [enc setBuffer:z offset:0 atIndex:2];
        [enc setBuffer:partial offset:0 atIndex:3];
        [enc setBuffer:cols offset:0 atIndex:4];
        dispatch_tree(enc);
    }

    void encode_tri(id<MTLComputeCommandEncoder> enc, const prepared_tri& t, bool forward) const {
        if (!m) return;
        const int f = forward ? 1 : 0;
        [enc setBuffer:t.ptr offset:0 atIndex:2];
        [enc setBuffer:t.col offset:0 atIndex:3];
        [enc setBuffer:t.val offset:0 atIndex:4];
        [enc setBuffer:r offset:0 atIndex:6];
        [enc setBuffer:z offset:0 atIndex:7];
        [enc setBuffer:cols offset:0 atIndex:8];
        [enc setBuffer:tags offset:0 atIndex:10];
        params prm = base();
        prm.count = t.tag_count;
        set_params(enc, prm);
        [enc setComputePipelineState:ctx->pipes.dataflow_init[f]];
        dispatch_rows(enc, t.tag_count);
        [enc setBuffer:t.words offset:0 atIndex:9];
        [enc setBuffer:t.blocks offset:0 atIndex:11];
        [enc setBuffer:t.segments offset:0 atIndex:12];
        [enc setComputePipelineState:ctx->pipes.dataflow_solve[f]];
        for (auto step : t.stages) {
            prm.offset = step.first;
            set_params(enc, prm);
            [enc dispatchThreadgroups:MTLSizeMake(step.last - step.first, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    }

    // z = L^-T L^-1 (r - mu_r), then for a Laplacian mu_z = mean(z).
    void encode_precond(id<MTLComputeCommandEncoder> enc) const {
        encode_tri(enc, fwd, true);
        encode_tri(enc, bwd, false);
        if (laplacian) {
            encode_reduce(enc, kReduceSumZ);
            encode_finalize(enc, kFinalizeMuZ, 0, 0);
        }
    }

    void encode_p_update(id<MTLComputeCommandEncoder> enc) const {
        [enc setComputePipelineState:ctx->pipes.p_update];
        set_params(enc, base());
        [enc setBuffer:p offset:0 atIndex:1];
        [enc setBuffer:z offset:0 atIndex:2];
        [enc setBuffer:cols offset:0 atIndex:3];
        dispatch_rows(enc, n);
    }

    void encode_initial_mu(id<MTLComputeCommandEncoder> enc) const {
        if (!laplacian) return;
        encode_reduce(enc, kReduceSumR);
        encode_finalize(enc, kFinalizeMuR, 0, 0);
    }

    void encode_iteration(id<MTLComputeCommandEncoder> enc, std::uint32_t it,
                          std::uint32_t window) const {
        [enc setComputePipelineState:ctx->pipes.spmv[operator_df ? 1 : 0]];
        set_params(enc, base());
        [enc setBuffer:op_ptr offset:0 atIndex:1];
        [enc setBuffer:op_col offset:0 atIndex:2];
        [enc setBuffer:op_hi offset:0 atIndex:3];
        [enc setBuffer:(op_lo != nil ? op_lo : op_hi) offset:0 atIndex:4];
        [enc setBuffer:p offset:0 atIndex:5];
        [enc setBuffer:ap offset:0 atIndex:6];
        [enc setBuffer:partial offset:0 atIndex:7];
        [enc setBuffer:cols offset:0 atIndex:8];
        dispatch_tree(enc);
        encode_finalize(enc, kFinalizeAlpha, it, window);

        [enc setComputePipelineState:ctx->pipes.update_xr];
        set_params(enc, base());
        [enc setBuffer:x offset:0 atIndex:1];
        [enc setBuffer:r offset:0 atIndex:2];
        [enc setBuffer:p offset:0 atIndex:3];
        [enc setBuffer:ap offset:0 atIndex:4];
        [enc setBuffer:partial offset:0 atIndex:5];
        [enc setBuffer:cols offset:0 atIndex:6];
        dispatch_tree(enc);
        encode_finalize(enc, kFinalizeCheck, it, window);

        encode_precond(enc);
        encode_reduce(enc, kReduceRz);
        encode_finalize(enc, kFinalizeRz, it, window);
        encode_p_update(enc);
    }

    bool active() const {
        return static_cast<const solve_state*>([cols contents])->active != 0;
    }

    void zero_grounded_row() const {
        if (laplacian) static_cast<float*>([z contents])[m] = 0.0f;
    }

};

engine::engine(const operator_arrays& op, const tri_arrays& fwd, const tri_arrays& bwd,
               std::uint32_t m, bool laplacian)
    : impl_(std::make_unique<impl>()) {
    context& ctx = shared();
    if (!ctx.status.ok) throw std::runtime_error(ctx.status.error);
    impl& s = *impl_;
    s.ctx = &ctx;
    s.n = static_cast<std::uint32_t>(op.n);
    s.m = m;
    s.groups = (s.n + kTreeRows - 1) / kTreeRows;
    s.laplacian = laplacian;
    s.operator_df = op.lo != nullptr;
    const double inv_n = 1.0 / static_cast<double>(op.n);
    s.inv_n_hi = static_cast<float>(inv_n);
    s.inv_n_lo = static_cast<float>(inv_n - static_cast<double>(s.inv_n_hi));
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
        const std::size_t entries = static_cast<std::size_t>(s.n);
        s.x = upload(dev, nullptr, entries * sizeof(df32));
        s.r = upload(dev, nullptr, entries * sizeof(df32));
        s.ap = upload(dev, nullptr, entries * sizeof(df32));
        s.p = upload(dev, nullptr, entries * sizeof(float));
        s.z = upload(dev, nullptr, entries * sizeof(float));
        s.partial = upload(dev, nullptr, 2 * static_cast<std::size_t>(s.groups) * sizeof(df32));
        s.cols = upload(dev, nullptr, sizeof(solve_state));
        s.tags = upload(dev, nullptr, std::max({entries, std::size_t(s.fwd.tag_count),
            std::size_t(s.bwd.tag_count)}) * sizeof(std::uint32_t));
        ok = s.x != nil && s.r != nil && s.ap != nil && s.p != nil && s.z != nil &&
             s.partial != nil && s.cols != nil &&
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
float* engine::p() noexcept { return static_cast<float*>([impl_->p contents]); }
solve_state& engine::state() noexcept { return *static_cast<solve_state*>([impl_->cols contents]); }

factor_statistics engine::factor_stats() const {
    return {impl_->fwd.levels, impl_->bwd.levels, impl_->fwd.stages.size() + unsigned(impl_->m != 0), impl_->bwd.stages.size() + unsigned(impl_->m != 0)};
}

void engine::solve(std::uint32_t max_iter,
                   std::uint32_t window, std::uint32_t check_every) {
    impl& s = *impl_;
    s.zero_grounded_row();
    std::string error;
    std::uint32_t it = 0;
    bool first = true;
    while (error.empty() && (first || (it < max_iter && s.active()))) {
        const std::uint32_t batch = std::min(std::max<std::uint32_t>(check_every, 1), max_iter - it);
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [s.queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = first ? [cb blitCommandEncoder] : nil;
            if (blit != nil) {
                [blit fillBuffer:s.x range:NSMakeRange(0, static_cast<NSUInteger>(s.n) * sizeof(df32)) value:0];
                [blit endEncoding];
            }
            id<MTLComputeCommandEncoder> enc = cb != nil && (!first || blit != nil) ? [cb computeCommandEncoder] : nil;
            if (enc == nil) {
                error = "Metal command buffer or encoder creation failed";
            } else {
                if (first) {
                    s.encode_initial_mu(enc);
                    s.encode_precond(enc);
                    s.encode_reduce(enc, kReduceRz);
                    s.encode_finalize(enc, kFinalizeRzInit, 0, window);
                    s.encode_p_update(enc);
                }
                for (std::uint32_t b = 0; b < batch; ++b)
                    s.encode_iteration(enc, it + b + 1, window);
                [enc endEncoding];
                error = finish(cb);
            }
        }
        it += batch;
        first = false;
    }
    if (!error.empty()) throw std::runtime_error(error);
}

void engine::apply() {
    impl& s = *impl_;
    s.zero_grounded_row();
    std::string error;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [s.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            error = "Metal command buffer or encoder creation failed";
        } else {
            s.encode_initial_mu(enc);
            s.encode_precond(enc);
            s.encode_p_update(enc);
            [enc endEncoding];
            error = finish(cb);
        }
    }
    if (!error.empty()) throw std::runtime_error(error);
}

}  // namespace apxchol::detail::metal
