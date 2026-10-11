// Metal factor preparation, PCG and command submission share one owner.
// Per-call Objective-C objects are retired inside autorelease pools; command
// errors are thrown after leaving the pool.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "apxchol/solver/metal_solver.h"
#include "apxchol/solver/detail/metal_host.h"
#include "apxchol/solver/detail/gpu_pcg.h"
#include "apxchol/solver/detail/metal_schedule.h"
#include "apxchol/solver/detail/permuted_operator.h"
#include "apxchol/solver/sptrsv/cuda_host.h"
#include "metal_device.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <omp.h>

namespace apxchol::detail::metal {
namespace {

using df32 = metal_host::df;
static_assert(sizeof(df32) == 8);

struct params {
    std::uint32_t n;
    std::uint32_t groups;
    std::uint32_t offset;
    std::uint32_t count;
    std::uint32_t mode;
    std::uint32_t iter;
    std::uint32_t window;
    std::uint32_t m;
};
static_assert(sizeof(params) == 32);
static_assert(offsetof(params, mode) == 16);
static_assert(offsetof(params, m) == 28);

using solve_state = gpu_pcg::solve_state<df32, float>;
static_assert(sizeof(solve_state) == 48);
static_assert(offsetof(solve_state, thr) == 16);
static_assert(offsetof(solve_state, alpha) == 32);
static_assert(offsetof(solve_state, active) == 40);

/// The device cannot hold a buffer. A std::bad_alloc, so callers that map
/// out-of-memory report it as one; what() names the buffer.
class device_memory_error : public std::bad_alloc {
public:
    explicit device_memory_error(std::string what) : what_(std::move(what)) {}
    const char* what() const noexcept override { return what_.c_str(); }

private:
    std::string what_;
};

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

using gpu_pcg::operation;

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

std::vector<float> run_probe(context& ctx, const std::vector<float>& inputs);

// Error-free transforms need round-to-nearest-even, separately rounded
// intermediates and an actual fused multiply-add.
std::string run_arithmetic_probe(context& ctx) {
    const auto out = run_probe(ctx, {
        0x1p0f, 0x1.8p-24f, 0x1.8p-24f, 0.0f, // Sum rounds up, leaving -2^-25.
        0x1p0f, 0.0f, 0x1p-24f, 0.0f,         // Halfway sum rounds to the even mantissa.
        0x1.000002p0f, 0x1p-24f, 0x1.000002p0f, 0.0f,
    });
    // The last quick_two_sum starts with an odd mantissa, so its tie rounds up.
    if (out[0] != 0x1.000002p0f || out[1] != -0x1p-25f || out[15] != -0x1p-25f ||
        out[16] != 0x1p0f || out[17] != 0x1p-24f || out[47] != -0x1p-24f)
        return "Metal double-float transforms require round-to-nearest-even without reassociation";
    // (1 + 2^-23)^2 = 1 + 2^-22 + 2^-46; only FMA retains the product residual.
    if (out[34] != 0x1.000004p0f || out[35] != 0x1p-46f)
        return "Metal double-float transforms require a fused multiply-add";
    return {};
}

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
    try {
        st.error = run_arithmetic_probe(*ctx);
        st.ok = st.error.empty();
    } catch (const std::exception& e) {
        st.error = e.what();
        st.ok = false;
    } catch (...) {
        st.error = "unknown error";
        st.ok = false;
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

prepared_tri prepare_tri(context &ctx, const cuda_host::csr_int<float>& a, bool forward) {
    prepared_tri out;
    const auto m = static_cast<unsigned>(a.m);
    auto schedule =
        metal_schedule::build({reinterpret_cast<const unsigned*>(a.ptr.data()), std::size_t(m) + 1},
                               {reinterpret_cast<const unsigned*>(a.idx.get()), std::size_t(a.nnz)},
                               {a.vals.get(), std::size_t(a.nnz)}, !forward);
    auto &packed = schedule.tasks;
    out.tag_count = m + packed.slots;
    // Metal validates the element type even when no task reads the buffer.
    if (packed.segments.empty())
        packed.segments.push_back({});
    out.stages = std::move(schedule.stages);
    out.levels = schedule.levels;
    @autoreleasepool {
        out.ptr = upload(ctx.device, a.ptr.data(), (m + 1) * sizeof(unsigned));
        out.col = upload(ctx.device, a.idx.get(), a.nnz * sizeof(unsigned));
        out.val = upload(ctx.device, a.vals.get(), a.nnz * sizeof(float));
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

namespace {
std::vector<float> run_probe(context& ctx, const std::vector<float>& inputs) {
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

} // namespace

std::vector<float> run_probe(const std::vector<float>& inputs) {
    context& ctx = shared();
    if (!ctx.status.ok) throw std::runtime_error(ctx.status.error);
    return run_probe(ctx, inputs);
}

} // namespace apxchol::detail::metal

namespace apxchol {

namespace {

namespace mh = detail::metal_host;
namespace ls = detail::metal_schedule;
namespace dm = detail::metal;
using namespace dm;
using detail::gpu_pcg::operation;
using solve_state = dm::solve_state;

void print_banner_once(const metal_solver::statistics& st) {
    static std::once_flag flag;
    std::call_once(flag, [&] {
        if (!std::getenv("APXCHOL_VERBOSE")) return;
        std::fprintf(stderr,
                     "[apxchol] Metal PCG on %s: fp32 factor, "
                     "double-float Krylov vectors, %s operator\n",
                     st.device.c_str(),
                     st.operator_double_float ? "double-float" : "fp32-exact");
    });
}

}  // namespace

bool metal_solver::available() noexcept { return dm::status().ok; }

struct metal_solver::impl {
    solve_options opts;
    factorization F;
    std::size_t n = 0;
    bool laplacian = false;
    // The permuted operator A' = P A P^T (canonical lower values) in fp64:
    // the host residuals run on it.
    std::vector<int> op_ptr;
    std::unique_ptr<int[]> op_col;
    std::unique_ptr<double[]> op_val;
    std::uint32_t check_every = 1;
    statistics stats;

    context* ctx = nullptr;
    id<MTLCommandQueue> queue = nil;
    params arguments{};
    bool operator_df = false;
    id<MTLBuffer> device_op_ptr = nil, device_op_col = nil, op_hi = nil, op_lo = nil;
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
                bind(encoder, 1, {x, r, ap, p, z, partial, state_buffer, device_op_ptr, device_op_col, op_hi,
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

    explicit impl(const solve_options& options) : opts(options) {
        if (opts.backend == solve_backend::cpu)
            throw std::invalid_argument("metal_solver cannot execute an explicit CPU request");
        if (detail::resolve_fp16_storage(opts.factor_opts.factor_storage, false))
            throw std::invalid_argument("Metal requires float32 factor storage");
        if (!metal_solver::available())
            throw std::runtime_error("apxchol::metal_solver: the Metal backend is unavailable: " +
                                     dm::status().error);
    }

    void setup(const Eigen::SparseMatrix<double>& A, checkpoint* cp);
    void prepare_device(const cuda_host::csr_int<float>& forward,
                        const cuda_host::csr_int<float>& backward,
                        const std::vector<float>& hi, const std::vector<float>& lo, unsigned m);

    // y = A' x in fp64, each row summed in storage order; rows are split across
    // threads by stored entries (detail::work_balanced_range).
    void spmv(const double* x, double* y) const {
        const int* ptr = op_ptr.data();
        const int* col = op_col.get();
        const double* val = op_val.get();
        const std::ptrdiff_t rows = static_cast<std::ptrdiff_t>(n);
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num(), nt = omp_get_num_threads();
            const auto [lo, hi] = detail::work_balanced_range(ptr, rows, tid, nt);
            for (std::ptrdiff_t i = lo; i < hi; ++i) {
                double acc = 0.0;
                for (int p = ptr[i]; p < ptr[i + 1]; ++p) acc += val[p] * x[col[p]];
                y[i] = acc;
            }
        }
    }

    double residual_norm(const Eigen::VectorXd& bp, const Eigen::VectorXd& xp,
                         Eigen::VectorXd& work) const {
        spmv(xp.data(), work.data());
        return mh::norm(bp - work);
    }

    void centre(Eigen::VectorXd& x) const { x.array() -= x.mean(); }
};

void metal_solver::impl::setup(const Eigen::SparseMatrix<double>& A, checkpoint* cp) {
    if (A.rows() != A.cols()) throw std::invalid_argument("metal_solver: the operator must be square");
    n = static_cast<std::size_t>(A.rows());
    if (n == 0) throw std::invalid_argument("metal_solver: empty operator");
    if (static_cast<std::size_t>(F.L.rows()) != n || F.perm.size() != n)
        throw std::invalid_argument("metal_solver: factorization dimension mismatch");
    if (F.L.nonZeros() > 0 && F.L.vals_.empty())
        throw std::invalid_argument(
            "metal_solver: factorization values were released; pass a freshly computed factorization");
    // Eigen's int StorageIndex keeps n and the operator's stored count within
    // the device's 32-bit indices; the factor and the two-triangle permuted
    // operator are checked here.
    static_assert(std::is_same_v<Eigen::SparseMatrix<double>::StorageIndex, int>);
    laplacian = !F.sddm;
    const auto m = static_cast<std::uint32_t>(laplacian ? n - 1 : n);
    if (F.L.outerIndexPtr()[m] > static_cast<edge_index>(std::numeric_limits<int>::max()))
        throw std::length_error("metal_solver: the factor exceeds the device's 32-bit offsets");

    if (cp) { cp->descend("setup"); cp->tick(); }
    // Keep compensated dropping in the shared FP64 host implementation.
    // Both backends consume the CSR factor without a level-permuted value copy.
    auto backward = cuda_host::build_L11_csc_int<float>(F.L, m);
    const auto scales = cuda_host::column_scales(backward);
    cuda_host::apply_factor_drop(backward, scales, factor_drop_rel_from_env(), false);
    const auto forward = cuda_host::transpose_csr(backward);
    if (cp) (*cp)("metal_factor_prep");

    Eigen::SparseMatrix<double> compressed;
    const Eigen::SparseMatrix<double>* src = &A;
    if (!A.isCompressed()) {
        compressed = A;
        compressed.makeCompressed();
        src = &compressed;
    }
    // Both triangles are stored: up to twice the stored entries of one triangle.
    if (src->nonZeros() > std::numeric_limits<int>::max() / 2)
        throw std::length_error("metal_solver: operators above 2^30 stored entries exceed the "
                                "device's 32-bit offsets");
    bool exact = true;
    std::int64_t op_nnz = 0;
    detail::build_permuted_full_symmetric_csr(*src, F.perm, op_ptr, op_col, op_val, op_nnz, exact);
    const std::size_t nnz = static_cast<std::size_t>(op_nnz);
    std::vector<float> hi(nnz), lo(exact ? 0 : nnz);
    bool in_range = true;
    #pragma omp parallel for schedule(static) reduction(&& : in_range)
    for (std::ptrdiff_t p = 0; p < static_cast<std::ptrdiff_t>(nnz); ++p) {
        const double v = op_val[p];
        in_range = in_range && ls::in_range(v);
        const mh::df d = mh::split(v);
        hi[p] = d.hi;
        if (!exact) lo[p] = d.lo;
    }
    if (!in_range)
        throw std::domain_error("metal_solver: an operator value is non-finite or outside [2^-100, 2^100]");
    if (cp) (*cp)("metal_operator");

    prepare_device(forward, backward, hi, lo, m);
    // Group four iterations on small systems to amortize command submission;
    // check large solves after each iteration.
    check_every = n > 200000 ? 1 : 4;

    stats.n = static_cast<Eigen::Index>(n);
    stats.levels_forward = fwd.levels;
    stats.levels_backward = bwd.levels;
    stats.steps_forward = fwd.stages.size() + unsigned(m != 0);
    stats.steps_backward = bwd.stages.size() + unsigned(m != 0);
    stats.operator_double_float = !exact;
    stats.device = ctx->status.name;
    if (!opts.keep_factor_values) F.L.release_values();
    if (cp) { (*cp)("metal_upload"); cp->ascend(); }
    print_banner_once(stats);
}

void metal_solver::impl::prepare_device(const cuda_host::csr_int<float>& forward,
                                       const cuda_host::csr_int<float>& backward,
                                       const std::vector<float>& hi, const std::vector<float>& lo,
                                       unsigned m) {
    context& ctx = shared();
    if (!ctx.status.ok) throw std::runtime_error(ctx.status.error);
    impl& s = *this;
    s.ctx = &ctx;
    s.arguments.n = static_cast<std::uint32_t>(n);
    s.arguments.m = m;
    s.arguments.groups = (s.arguments.n + kTreeRows - 1) / kTreeRows;
    const auto nnz = op_ptr.back();
    s.operator_df = !lo.empty();
    bool ok = true;
    @autoreleasepool {
        id<MTLDevice> dev = ctx.device;
        s.queue = [dev newCommandQueue];
        s.device_op_ptr = upload(dev, op_ptr.data(), (n + 1) * sizeof(std::uint32_t));
        s.device_op_col = upload(dev, op_col.get(), nnz * sizeof(std::uint32_t));
        s.op_hi = upload(dev, hi.data(), nnz * sizeof(float));
        if (!lo.empty()) s.op_lo = upload(dev, lo.data(), nnz * sizeof(float));
        ok = s.device_op_ptr != nil && s.device_op_col != nil && s.op_hi != nil &&
             (lo.empty() || s.op_lo != nil);
    }
    if (s.queue == nil) throw std::runtime_error("Metal command queue creation failed");
    if (!ok) throw device_memory_error("Metal buffer allocation failed for the operator or factor");
    s.fwd = prepare_tri(ctx, forward, true);
    s.bwd = prepare_tri(ctx, backward, false);
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

metal_solver::metal_solver(const Eigen::SparseMatrix<double>& A, const solve_options& opts,
                           checkpoint* cp)
    : impl_(new impl(opts)) {
    impl_->F = detail::factorize_for_solver(A, opts.storage, opts.factor_opts, cp, true);
    impl_->setup(A, cp);
}

metal_solver::metal_solver(const Eigen::SparseMatrix<double>& A, factorization F,
                           const solve_options& opts, checkpoint* cp)
    : impl_(new impl(opts)) {
    impl_->F = std::move(F);
    impl_->setup(A, cp);
}

void metal_solver::impl_deleter::operator()(impl* owner) const {
    @autoreleasepool { delete owner; }
}

metal_solver::~metal_solver() = default;
metal_solver::metal_solver(metal_solver&&) noexcept = default;
metal_solver& metal_solver::operator=(metal_solver&&) noexcept = default;

const factorization& metal_solver::factor() const { return impl_->F; }
metal_solver::statistics metal_solver::stats() const { return impl_->stats; }
Eigen::Index metal_solver::rows() const { return static_cast<Eigen::Index>(impl_->n); }

solve_result metal_solver::solve(const Eigen::VectorXd& b, double tol, int max_iter,
                                 const Eigen::VectorXd* x0) const {
    solve_result result;
    solve(b, result, tol, max_iter, x0);
    return result;
}

void metal_solver::solve(const Eigen::VectorXd& b, solve_result& result, double tol,
                         int max_iter, const Eigen::VectorXd* x0) const {
    impl& s = *impl_;
    if (b.size() != rows()) throw std::invalid_argument("metal_solver::solve: RHS length mismatch");
    if (x0 != nullptr && x0->size() != rows())
        throw std::invalid_argument("metal_solver::solve: x0 length mismatch");
    if (!b.allFinite() || (x0 != nullptr && !x0->allFinite()))
        throw std::invalid_argument("metal_solver::solve: non-finite RHS or initial guess");
    if (tol < 0.0) tol = s.opts.tol;
    if (max_iter < 0) max_iter = s.opts.max_iter;
    const auto window = static_cast<std::uint32_t>(std::max(0, s.opts.stagnation_window));

    result.iterations = 0;
    result.residual = 0.0;
    result.backend = solve_backend::gpu;
    result.lumped_offdiag = s.F.lumped_offdiag;
    result.x = Eigen::VectorXd::Zero(rows());
    checkpoint& cp = result.timings;
    cp.descend("pcg");
    cp.tick();
    const double bnorm = mh::norm(b);
    if (bnorm == 0.0) { cp.ascend(); return; }
    if (max_iter == 0 && (x0 == nullptr || x0->isZero(0.0))) {
        result.residual = 1.0;
        cp.ascend();
        return;
    }

    const node_index* perm = s.F.perm.data();
    Eigen::VectorXd bp(s.n), work(s.n);
    auto& xp = result.x;
    mh::scatter(b.data(), perm, s.n, bp.data());
    auto finish = [&] {
        if (s.laplacian) s.centre(xp);
        result.residual = s.residual_norm(bp, xp, work) / bnorm;
        mh::gather(xp.data(), perm, s.n, work.data());
        result.x.swap(work);
        cp("exit_check");
        cp.ascend();
    };

    if (x0 != nullptr) {
        mh::scatter(x0->data(), perm, s.n, xp.data());
        s.spmv(xp.data(), work.data());
        work = bp - work;
    } else {
        work = bp;
    }
    const double initial = x0 ? mh::norm(work) / bnorm : 1.0;
    if (initial < tol || max_iter == 0) { finish(); return; }
    if (s.laplacian) s.centre(work);
    const double max_abs = work.cwiseAbs().maxCoeff();
    if (max_abs == 0.0) { finish(); return; }
    const double scale = mh::pow2_scale(max_abs);
    mh::pack(work.data(), s.n, scale, static_cast<df32*>([s.r contents]));
    auto& state = *static_cast<solve_state*>([s.state_buffer contents]);
    state = {};
    const double reference = bnorm * scale;
    const double threshold = tol * reference;
    state.thr = mh::split_saturated(threshold * threshold);
    state.prev = mh::split_saturated(reference * reference);
    state.active = 1;
    cp("pack");
    s.arguments.window = window;
    detail::gpu_pcg::solve(s, static_cast<std::uint32_t>(max_iter), s.check_every);
    cp("device");
    mh::add_unpacked(static_cast<df32*>([s.x contents]), s.n, scale, xp.data());
    result.iterations = state.iters;
    finish();
}

Eigen::VectorXd metal_solver::apply(const Eigen::VectorXd& r) const {
    impl& s = *impl_;
    if (r.size() != static_cast<Eigen::Index>(s.n))
        throw std::invalid_argument("metal_solver::apply: r length mismatch");
    if (!r.allFinite()) throw std::invalid_argument("metal_solver::apply: r contains a non-finite value");
    const node_index* perm = s.F.perm.data();
    const double mean = s.laplacian ? r.mean() : 0;
    const double max_abs = (r.array() - mean).abs().maxCoeff();
    Eigen::VectorXd z = Eigen::VectorXd::Zero(r.size());
    if (max_abs == 0.0) return z;
    const double scale = mh::pow2_scale(max_abs);
    auto* packed = static_cast<df32*>([s.r contents]);
    #pragma omp parallel for schedule(static)
    for (Eigen::Index i = 0; i < r.size(); ++i)
        packed[perm[i]] = mh::split(scale * (r[i] - mean));
    auto& cs = *static_cast<solve_state*>([s.state_buffer contents]);
    cs = {};
    cs.active = 1;
    s.submit(false, [&] { s.precondition(); });
    const float* applied = static_cast<float*>([s.z contents]);
    const double mean_z = s.laplacian
        ? Eigen::Map<const Eigen::VectorXf>(applied, s.n).cast<double>().mean() : 0;
    #pragma omp parallel for schedule(static)
    for (Eigen::Index i = 0; i < r.size(); ++i)
        z[i] = (double(applied[perm[i]]) - mean_z) / scale;
    return z;
}

}  // namespace apxchol
