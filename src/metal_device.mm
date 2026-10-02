// Device side of the Metal block PCG (see src/metal_device.h). Objective-C++
// with ARC; no Eigen, no OpenMP. Every per-call Metal object lives in an
// @autoreleasepool, and every command buffer's status is checked after
// waitUntilCompleted. Errors are collected inside the pool and thrown after
// it, so no C++ exception unwinds through an autorelease pool.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_device.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace apxchol::detail::metal {
namespace {

const char kKernelSource[] =
#include "metal_kernels.inc"
    ;

constexpr std::uint32_t kTreeRows = 256;
constexpr std::uint32_t kTreeLanes = 16;
constexpr std::uint32_t kVirtualLanes = 32;
constexpr std::uint32_t kRowThreads = 256;     // threads per threadgroup of the row kernels
constexpr std::uint32_t kNarrowThreads = 1024;

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
    id<MTLComputePipelineState> level[2];     // [forward]
    id<MTLComputePipelineState> heavy[2];
    id<MTLComputePipelineState> narrow[2];
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
            p.level[fwd] = make_pipeline(device, library, "level", fwd != 0, false, 0, error);
            p.heavy[fwd] = make_pipeline(device, library, "level_heavy", fwd != 0, false, 0, error);
            p.narrow[fwd] = make_pipeline(device, library, "levels_narrow", fwd != 0, false, 0, error);
        }
        p.p_update = make_pipeline(device, library, "p_update", false, false, 0, error);
        p.probe = make_pipeline(device, library, "df_probe", false, false, 0, error);
        if (!error.empty()) {
            st.error = "Metal pipeline creation failed: " + error;
            return ctx;
        }
        auto threads = [](id<MTLComputePipelineState> pipe) {
            return static_cast<std::uint32_t>([pipe maxTotalThreadsPerThreadgroup]);
        };
        st.tree_threads = std::min({threads(p.spmv[0]), threads(p.spmv[1]), threads(p.update_xr),
                                    threads(p.reduce[0]), threads(p.reduce[1]),
                                    threads(p.reduce[2]), threads(p.finalize)});
        st.row_threads = std::min({threads(p.level[0]), threads(p.level[1]), threads(p.p_update)});
        st.heavy_threads = std::min(threads(p.heavy[0]), threads(p.heavy[1]));
        st.narrow_threads = std::min(threads(p.narrow[0]), threads(p.narrow[1]));
        st.max_buffer_bytes = static_cast<std::uint64_t>([device maxBufferLength]);
        st.working_set_bytes = static_cast<std::uint64_t>([device recommendedMaxWorkingSetSize]);
        ctx->probe_queue = [device newCommandQueue];
        if (ctx->probe_queue == nil) {
            st.error = "no Metal command queue";
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

}  // namespace

const device_status& status() noexcept { return shared().status; }

std::vector<float> run_probe(const std::vector<float>& inputs) {
    context& ctx = shared();
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
    struct tri {
        id<MTLBuffer> level_ptr = nil, rows = nil, ptr = nil, col = nil, val = nil, dinv = nil;
    } fwd, bwd;
    std::uint32_t cap = 0;
    id<MTLBuffer> x = nil, r = nil, ap = nil, p = nil, z = nil, partial = nil, cols = nil;

    params base(std::uint32_t kc) const {
        params prm{};
        prm.n = n;
        prm.kc = kc;
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

    void dispatch_tree(id<MTLComputeCommandEncoder> enc, std::uint32_t kc) const {
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kc, kTreeLanes, 1)];
    }

    void dispatch_rows(id<MTLComputeCommandEncoder> enc, std::uint32_t kc, std::uint32_t rows) const {
        const std::uint32_t per = std::max<std::uint32_t>(
            1, std::min(kRowThreads, ctx->status.row_threads) / kc);
        [enc dispatchThreadgroups:MTLSizeMake(1, (rows + per - 1) / per, 1)
            threadsPerThreadgroup:MTLSizeMake(kc, per, 1)];
    }

    void encode_finalize(id<MTLComputeCommandEncoder> enc, std::uint32_t kc, std::uint32_t mode,
                         std::uint32_t iter, std::uint32_t window) const {
        params prm = base(kc);
        prm.mode = mode;
        prm.iter = iter;
        prm.window = window;
        [enc setComputePipelineState:ctx->pipes.finalize];
        set_params(enc, prm);
        [enc setBuffer:partial offset:0 atIndex:1];
        [enc setBuffer:cols offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kc, kTreeLanes, 1)];
    }

    void encode_reduce(id<MTLComputeCommandEncoder> enc, std::uint32_t kc, std::uint32_t mode) const {
        [enc setComputePipelineState:ctx->pipes.reduce[mode]];
        set_params(enc, base(kc));
        [enc setBuffer:r offset:0 atIndex:1];
        [enc setBuffer:z offset:0 atIndex:2];
        [enc setBuffer:partial offset:0 atIndex:3];
        [enc setBuffer:cols offset:0 atIndex:4];
        dispatch_tree(enc, kc);
    }

    void encode_tri(id<MTLComputeCommandEncoder> enc, std::uint32_t kc, const tri& t,
                    const std::vector<tri_step>& plan, bool forward) const {
        const int f = forward ? 1 : 0;
        [enc setBuffer:t.rows offset:0 atIndex:1];
        [enc setBuffer:t.ptr offset:0 atIndex:2];
        [enc setBuffer:t.col offset:0 atIndex:3];
        [enc setBuffer:t.val offset:0 atIndex:4];
        [enc setBuffer:t.dinv offset:0 atIndex:5];
        [enc setBuffer:r offset:0 atIndex:6];
        [enc setBuffer:z offset:0 atIndex:7];
        [enc setBuffer:cols offset:0 atIndex:8];
        [enc setBuffer:t.level_ptr offset:0 atIndex:9];
        params prm = base(kc);
        // Real lanes of a heavy row: a power of two dividing the 32 virtual
        // lanes, limited by the heavy pipeline's own thread limit.
        std::uint32_t lanes = kVirtualLanes;
        while (lanes > 1 && lanes * kc > ctx->status.heavy_threads) lanes /= 2;
        const std::uint32_t narrow_threads = std::min(kNarrowThreads, ctx->status.narrow_threads);
        for (const tri_step& s : plan) {
            prm.offset = s.first;
            prm.count = s.kind == 2 ? s.last : s.last - s.first;
            if (s.kind == 0) {
                [enc setComputePipelineState:ctx->pipes.level[f]];
                set_params(enc, prm);
                dispatch_rows(enc, kc, prm.count);
            } else if (s.kind == 1) {
                [enc setComputePipelineState:ctx->pipes.heavy[f]];
                set_params(enc, prm);
                [enc dispatchThreadgroups:MTLSizeMake(prm.count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(kc, lanes, 1)];
            } else {
                [enc setComputePipelineState:ctx->pipes.narrow[f]];
                set_params(enc, prm);
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(narrow_threads, 1, 1)];
            }
        }
    }

    // z = L^-T L^-1 (r - mu_r), then for a Laplacian mu_z = mean(z).
    void encode_precond(id<MTLComputeCommandEncoder> enc, std::uint32_t kc,
                        const std::vector<tri_step>& fp, const std::vector<tri_step>& bp) const {
        encode_tri(enc, kc, fwd, fp, true);
        encode_tri(enc, kc, bwd, bp, false);
        if (laplacian) {
            encode_reduce(enc, kc, kReduceSumZ);
            encode_finalize(enc, kc, kFinalizeMuZ, 0, 0);
        }
    }

    void encode_p_update(id<MTLComputeCommandEncoder> enc, std::uint32_t kc) const {
        [enc setComputePipelineState:ctx->pipes.p_update];
        set_params(enc, base(kc));
        [enc setBuffer:p offset:0 atIndex:1];
        [enc setBuffer:z offset:0 atIndex:2];
        [enc setBuffer:cols offset:0 atIndex:3];
        dispatch_rows(enc, kc, n);
    }

    void encode_initial_mu(id<MTLComputeCommandEncoder> enc, std::uint32_t kc) const {
        if (!laplacian) return;
        encode_reduce(enc, kc, kReduceSumR);
        encode_finalize(enc, kc, kFinalizeMuR, 0, 0);
    }

    void encode_iteration(id<MTLComputeCommandEncoder> enc, std::uint32_t kc, std::uint32_t it,
                          std::uint32_t window, const std::vector<tri_step>& fp,
                          const std::vector<tri_step>& bp) const {
        [enc setComputePipelineState:ctx->pipes.spmv[operator_df ? 1 : 0]];
        set_params(enc, base(kc));
        [enc setBuffer:op_ptr offset:0 atIndex:1];
        [enc setBuffer:op_col offset:0 atIndex:2];
        [enc setBuffer:op_hi offset:0 atIndex:3];
        [enc setBuffer:(op_lo != nil ? op_lo : op_hi) offset:0 atIndex:4];
        [enc setBuffer:p offset:0 atIndex:5];
        [enc setBuffer:ap offset:0 atIndex:6];
        [enc setBuffer:partial offset:0 atIndex:7];
        [enc setBuffer:cols offset:0 atIndex:8];
        dispatch_tree(enc, kc);
        encode_finalize(enc, kc, kFinalizeAlpha, it, window);

        [enc setComputePipelineState:ctx->pipes.update_xr];
        set_params(enc, base(kc));
        [enc setBuffer:x offset:0 atIndex:1];
        [enc setBuffer:r offset:0 atIndex:2];
        [enc setBuffer:p offset:0 atIndex:3];
        [enc setBuffer:ap offset:0 atIndex:4];
        [enc setBuffer:partial offset:0 atIndex:5];
        [enc setBuffer:cols offset:0 atIndex:6];
        dispatch_tree(enc, kc);
        encode_finalize(enc, kc, kFinalizeCheck, it, window);

        encode_precond(enc, kc, fp, bp);
        encode_reduce(enc, kc, kReduceRz);
        encode_finalize(enc, kc, kFinalizeRz, it, window);
        encode_p_update(enc, kc);
    }

    bool any_active(std::uint32_t kc) const {
        const auto* cs = static_cast<const column_state*>([cols contents]);
        for (std::uint32_t c = 0; c < kc; ++c)
            if (cs[c].active != 0) return true;
        return false;
    }

    void zero_grounded_row(std::uint32_t kc) const {
        // The Laplacian's grounded unknown m is never solved: z[m] = 0.
        if (!laplacian) return;
        float* zp = static_cast<float*>([z contents]);
        std::fill(zp + static_cast<std::size_t>(m) * kc, zp + static_cast<std::size_t>(m + 1) * kc, 0.0f);
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
        auto load = [&](impl::tri& t, const tri_arrays& a) {
            t.level_ptr = upload(dev, a.level_ptr, (a.levels + 1) * sizeof(std::uint32_t));
            t.rows = upload(dev, a.rows, a.slots * sizeof(std::uint32_t));
            t.ptr = upload(dev, a.ptr, (a.slots + 1) * sizeof(std::uint32_t));
            t.col = upload(dev, a.col, a.deps * sizeof(std::uint32_t));
            t.val = upload(dev, a.val, a.deps * sizeof(float));
            t.dinv = upload(dev, a.dinv, a.slots * sizeof(float));
            return t.level_ptr != nil && t.rows != nil && t.ptr != nil && t.col != nil &&
                   t.val != nil && t.dinv != nil;
        };
        ok = load(s.fwd, fwd) && ok;
        ok = load(s.bwd, bwd) && ok;
    }
    if (s.queue == nil) throw std::runtime_error("Metal command queue creation failed");
    if (!ok) throw device_memory_error("Metal buffer allocation failed for the operator or factor");
}

engine::~engine() {
    @autoreleasepool {
        impl_.reset();
    }
}

void engine::reserve(std::uint32_t kc) {
    impl& s = *impl_;
    if (kc <= s.cap) return;
    bool ok = true;
    @autoreleasepool {
        id<MTLDevice> dev = s.ctx->device;
        const std::size_t entries = static_cast<std::size_t>(s.n) * kc;
        s.x = s.r = s.ap = s.p = s.z = s.partial = s.cols = nil;
        s.x = upload(dev, nullptr, entries * sizeof(df32));
        s.r = upload(dev, nullptr, entries * sizeof(df32));
        s.ap = upload(dev, nullptr, entries * sizeof(df32));
        s.p = upload(dev, nullptr, entries * sizeof(float));
        s.z = upload(dev, nullptr, entries * sizeof(float));
        s.partial = upload(dev, nullptr, 2 * static_cast<std::size_t>(s.groups) * kc * sizeof(df32));
        s.cols = upload(dev, nullptr, kc * sizeof(column_state));
        ok = s.x != nil && s.r != nil && s.ap != nil && s.p != nil && s.z != nil &&
             s.partial != nil && s.cols != nil;
    }
    if (!ok) {
        s.cap = 0;
        throw device_memory_error("Metal buffer allocation failed for the block vectors");
    }
    s.cap = kc;
}

std::uint32_t engine::capacity() const noexcept { return impl_->cap; }
df32* engine::r() noexcept { return static_cast<df32*>([impl_->r contents]); }
df32* engine::x() noexcept { return static_cast<df32*>([impl_->x contents]); }
df32* engine::ap() noexcept { return static_cast<df32*>([impl_->ap contents]); }
float* engine::p() noexcept { return static_cast<float*>([impl_->p contents]); }
float* engine::z() noexcept { return static_cast<float*>([impl_->z contents]); }
column_state* engine::columns() noexcept { return static_cast<column_state*>([impl_->cols contents]); }

void engine::solve(std::uint32_t kc, const std::vector<tri_step>& fwd_plan,
                   const std::vector<tri_step>& bwd_plan, std::uint32_t max_iter,
                   std::uint32_t window, std::uint32_t check_every) {
    impl& s = *impl_;
    if (kc == 0 || kc > s.cap) throw std::logic_error("metal engine: batch exceeds the reserved block");
    s.zero_grounded_row(kc);
    std::string error;
    std::uint32_t it = 0;
    bool first = true;
    while (error.empty() && (first || (it < max_iter && s.any_active(kc)))) {
        const std::uint32_t batch = std::min(std::max<std::uint32_t>(check_every, 1), max_iter - it);
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [s.queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = first ? [cb blitCommandEncoder] : nil;
            if (blit != nil) {
                [blit fillBuffer:s.x range:NSMakeRange(0, static_cast<NSUInteger>(s.n) * kc * sizeof(df32)) value:0];
                [blit endEncoding];
            }
            id<MTLComputeCommandEncoder> enc = cb != nil && (!first || blit != nil) ? [cb computeCommandEncoder] : nil;
            if (enc == nil) {
                error = "Metal command buffer or encoder creation failed";
            } else {
                if (first) {
                    s.encode_initial_mu(enc, kc);
                    s.encode_precond(enc, kc, fwd_plan, bwd_plan);
                    s.encode_reduce(enc, kc, kReduceRz);
                    s.encode_finalize(enc, kc, kFinalizeRzInit, 0, window);
                    s.encode_p_update(enc, kc);
                }
                for (std::uint32_t b = 0; b < batch; ++b)
                    s.encode_iteration(enc, kc, it + b + 1, window, fwd_plan, bwd_plan);
                [enc endEncoding];
                error = finish(cb);
            }
        }
        it += batch;
        first = false;
    }
    if (!error.empty()) throw std::runtime_error(error);
}

void engine::apply(std::uint32_t kc, const std::vector<tri_step>& fwd_plan,
                   const std::vector<tri_step>& bwd_plan) {
    impl& s = *impl_;
    if (kc == 0 || kc > s.cap) throw std::logic_error("metal engine: batch exceeds the reserved block");
    s.zero_grounded_row(kc);
    std::string error;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [s.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) {
            error = "Metal command buffer or encoder creation failed";
        } else {
            s.encode_initial_mu(enc, kc);
            s.encode_precond(enc, kc, fwd_plan, bwd_plan);
            s.encode_p_update(enc, kc);
            [enc endEncoding];
            error = finish(cb);
        }
    }
    if (!error.empty()) throw std::runtime_error(error);
}

}  // namespace apxchol::detail::metal
