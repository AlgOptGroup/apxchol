// Metal factor preparation, PCG and command submission.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "apxchol/solver/metal_solver.h"
#include "apxchol/solver/detail/metal_host.h"
#include "apxchol/solver/detail/gpu_pcg.h"
#include "apxchol/solver/detail/metal_schedule.h"
#include "apxchol/solver/detail/permuted_operator.h"
#include "apxchol/solver/sptrsv/cuda_host.h"
#include "metal_device.h"
#include "metal_compile.h"
#include "metal_setup.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
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

#define APXCHOL_GPU_CODE(...) __VA_ARGS__
#include "metal_params.inc"
#undef APXCHOL_GPU_CODE
static_assert(sizeof(Params) == 40);
static_assert(offsetof(Params, mode) == 16);
static_assert(offsetof(Params, m) == 28);
static_assert(offsetof(Params, spmv_lanes) == 32);
static_assert(offsetof(Params, preconditioner_scale) == 36);

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
#include "apxchol/solver/detail/metal_factor_storage.inc"
#include "metal_params.inc"
#include "metal_kernels.inc"
    "\n#define APXCHOL_UNROLL _Pragma(\"unroll\")\n"
#include "gpu_dataflow.inc"
#undef APXCHOL_GPU_CODE
#include "metal_dataflow.inc"
    ;

constexpr std::uint32_t kRowThreads = 256;     // threads per threadgroup of the row kernels

using gpu_pcg::operation;

struct pipelines {
    id<MTLComputePipelineState> pcg[4][2];  // operation, operator double-float
    id<MTLComputePipelineState> finalize;
    id<MTLComputePipelineState> dataflow_solve[2][3]; // backward, forward, independent forward
    id<MTLComputePipelineState> dataflow_init[2][2];
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
                                          unsigned op = 0, bool factor_fp16 = false, bool independent = false) {
    MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
    [values setConstantValue:&forward type:MTLDataTypeBool atIndex:0];
    [values setConstantValue:&operator_df type:MTLDataTypeBool atIndex:1];
    [values setConstantValue:&op type:MTLDataTypeUInt atIndex:2];
    [values setConstantValue:&factor_fp16 type:MTLDataTypeBool atIndex:3];
    [values setConstantValue:&independent type:MTLDataTypeBool atIndex:4];
    NSError* err = nil;
    id<MTLFunction> fn = [library newFunctionWithName:[NSString stringWithUTF8String:name]
                                       constantValues:values
                                                error:&err];
    if (fn == nil)
        throw std::runtime_error(std::string("Metal function ") + name + ": " + describe(err));
    id<MTLComputePipelineState> pipe = [device newComputePipelineStateWithFunction:fn error:&err];
    if (pipe == nil)
        throw std::runtime_error(std::string("Metal pipeline ") + name + ": " + describe(err));
    return pipe;
}

context* make_context() {
    auto* ctx = new context;  // process lifetime: never destroyed
    device_status& st = ctx->status;
    @autoreleasepool {
        try {
            // Command-line tools need CoreGraphics linked for this to find the GPU.
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil) throw std::runtime_error("no Metal device");
            ctx->device = device;
            st.name = [[device name] UTF8String];
            const auto compiled = compile_library(device, kKernelSource);
            st.contract_pragma = compiled.contract_pragma;
            id<MTLLibrary> library = compiled.library;
            pipelines& p = ctx->pipes;
            for (unsigned op = 0; op < 4; ++op)
                for (unsigned df = 0; df < (op == unsigned(operation::multiply) ? 2u : 1u); ++df)
                    p.pcg[op][df] = make_pipeline(device, library, "pcg_step", false, df != 0, op);
            p.finalize = make_pipeline(device, library, "finalize", false, false);
            for (int half = 0; half < 2; ++half) {
                for (int fwd = 0; fwd < 2; ++fwd)
                    p.dataflow_init[half][fwd] = make_pipeline(device, library, "dataflow_init", fwd != 0, false, 0, half);
                for (unsigned variant = 0; variant < 3; ++variant) {
                    auto pipe = make_pipeline(device, library, "dataflow_solve", variant != 0, false, 0, half, variant == 2);
                    p.dataflow_solve[half][variant] = pipe;
                    if ([pipe threadExecutionWidth] != 32 || [pipe maxTotalThreadsPerThreadgroup] < 256)
                        throw std::runtime_error("Metal dataflow requires 32-thread SIMD groups and 256-thread workgroups");
                }
            }
            p.probe = make_pipeline(device, library, "df_probe", false, false);
            auto threads = [](id<MTLComputePipelineState> pipe) {
                return static_cast<std::uint32_t>([pipe maxTotalThreadsPerThreadgroup]);
            };
            const auto tree_threads = std::min({threads(p.pcg[0][0]), threads(p.pcg[0][1]), threads(p.pcg[1][0]),
                                               threads(p.pcg[2][0]), threads(p.finalize)});
            st.row_threads = std::min({threads(p.pcg[unsigned(operation::direction)][0]),
                                      threads(p.dataflow_init[0][0]), threads(p.dataflow_init[0][1]),
                                      threads(p.dataflow_init[1][0]), threads(p.dataflow_init[1][1])});
            ctx->probe_queue = [device newCommandQueue];
            if (ctx->probe_queue == nil) throw std::runtime_error("no Metal command queue");
            if (tree_threads < kTreeLanes || st.row_threads == 0)
                throw std::runtime_error("Metal pipeline thread limits are insufficient");
            st.error = run_arithmetic_probe(*ctx);
            st.ok = st.error.empty();
        } catch (const std::exception& e) {
            st.error = e.what();
        } catch (...) {
            st.error = "unknown error";
        }
    }
    return ctx;
}

context& shared() {
    static context* ctx = make_context();
    return *ctx;
}

// Exceptions must be caught inside the pool so that it drains before rethrowing.
template<class F>
void with_pool(F&& f) {
    std::exception_ptr failure;
    @autoreleasepool {
        try { f(); }
        catch (...) { failure = std::current_exception(); }
    }
    if (failure) std::rethrow_exception(failure);
}

id<MTLBuffer> upload(id<MTLDevice> device, const void* data, std::size_t bytes, const char* name) {
    const std::size_t len = std::max<std::size_t>(bytes, 4);
    id<MTLBuffer> b = len <= [device maxBufferLength]
        ? [device newBufferWithLength:len options:MTLResourceStorageModeShared] : nil;
    if (b == nil) throw device_memory_error(std::string("Metal buffer allocation failed: ") + name);
    if (data != nullptr && bytes != 0) std::memcpy([b contents], data, bytes);
    else std::memset([b contents], 0, len);
    return b;
}

void finish(id<MTLCommandBuffer> cb) {
    [cb commit];
    [cb waitUntilCompleted];
    if ([cb status] != MTLCommandBufferStatusCompleted || [cb error] != nil)
        throw std::runtime_error("Metal command buffer failed: " + describe([cb error]));
}

struct prepared_tri {
    id<MTLBuffer> words = nil, blocks = nil, segments = nil;
    unsigned tag_count = 0;
    std::vector<metal_schedule::step> stages;
    id<MTLBuffer> ptr = nil, col = nil, val = nil;
    std::uint32_t levels = 0;
    bool resident_round_plan = false;
};

template<class Value>
prepared_tri prepare_tri(context &ctx, const cuda_host::csr_int<Value>& a, bool forward,
                         std::span<const float> diagonals = {}) {
    prepared_tri out;
    const auto m = static_cast<unsigned>(a.m);
    auto schedule =
        metal_schedule::build<Value>({reinterpret_cast<const unsigned*>(a.ptr.data()), std::size_t(m) + 1},
                               {reinterpret_cast<const unsigned*>(a.idx.get()), std::size_t(a.nnz)},
                               {a.vals.get(), std::size_t(a.nnz)}, !forward, diagonals);
    auto &packed = schedule.tasks;
    out.tag_count = m + packed.slots;
    // Metal validates the element type even when no task reads the buffer.
    if (packed.segments.empty())
        packed.segments.push_back({});
    out.stages = std::move(schedule.stages);
    out.levels = schedule.levels;
    with_pool([&] {
        out.ptr = upload(ctx.device, a.ptr.data(), (m + 1) * sizeof(unsigned), "factor pointers");
        out.col = upload(ctx.device, a.idx.get(), a.nnz * sizeof(unsigned), "factor indices");
        out.val = upload(ctx.device, a.vals.get(), a.nnz * sizeof(Value), "factor values");
        out.words = upload(ctx.device, packed.words.data(), packed.words.size() * sizeof(unsigned), "factor task words");
        out.blocks = upload(ctx.device, packed.blocks.data(), packed.blocks.size() * sizeof(unsigned), "factor task blocks");
        out.segments = upload(ctx.device, packed.segments.data(),
                              packed.segments.size() * sizeof(gpu_dataflow::segment), "factor task segments");
    });
    return out;
}

} // namespace

const device_status &status() noexcept { return shared().status; }

namespace {
std::vector<float> run_probe(context& ctx, const std::vector<float>& inputs) {
    const std::uint32_t count = static_cast<std::uint32_t>(inputs.size() / 4);
    std::vector<float> out(static_cast<std::size_t>(count) * 16);
    with_pool([&] {
        id<MTLBuffer> in = upload(ctx.device, inputs.data(), inputs.size() * sizeof(float), "probe input");
        id<MTLBuffer> res = upload(ctx.device, nullptr, out.size() * sizeof(float), "probe result");
        id<MTLCommandBuffer> cb = [ctx.probe_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc == nil) throw std::runtime_error("Metal command buffer or encoder creation failed");
        [enc setComputePipelineState:ctx.pipes.probe];
        [enc setBuffer:in offset:0 atIndex:0];
        [enc setBuffer:res offset:0 atIndex:1];
        [enc setBytes:&count length:sizeof count atIndex:2];
        const NSUInteger w = 64;
        [enc dispatchThreadgroups:MTLSizeMake((count + w - 1) / w, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
        [enc endEncoding];
        finish(cb);
        std::memcpy(out.data(), [res contents], out.size() * sizeof(float));
    });
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

void print_banner_once(const metal_solver::statistics& st, bool factor_fp16) {
    static std::once_flag flag;
    std::call_once(flag, [&] {
        if (!std::getenv("APXCHOL_VERBOSE")) return;
        std::fprintf(stderr,
                     "[apxchol] Metal PCG on %s: %s factor, "
                     "double-float Krylov vectors, %s operator\n",
                     st.device.c_str(), factor_fp16 ? "fp16" : "fp32",
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
    detail::permuted_operator_csr op;
    Eigen::SparseMatrix<double> original_operator;
    std::uint32_t check_every = 1;
    statistics stats;

    context* ctx = nullptr;
    id<MTLCommandQueue> queue = nil;
    Params arguments{};
    bool operator_df = false, factor_fp16 = false;
    id<MTLBuffer> factor_metadata = nil;
    id<MTLBuffer> device_op_ptr = nil, device_op_col = nil, op_hi = nil, op_lo = nil;
    prepared_tri fwd, bwd;
    id<MTLBuffer> x = nil, r = nil, ap = nil, p = nil, z = nil, partial = nil, state_buffer = nil;
    id<MTLBuffer> tags = nil;

    static void bind(id<MTLComputeCommandEncoder> enc, unsigned index,
                     std::initializer_list<id<MTLBuffer>> buffers) {
        for (auto buffer : buffers) [enc setBuffer:buffer offset:0 atIndex:index++];
    }

    void dispatch(id<MTLComputePipelineState> pipe, const Params& prm,
                  unsigned groups, unsigned threads) const {
        [encoder setComputePipelineState:pipe];
        [encoder setBytes:&prm length:sizeof prm atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    }

    void finalize(unsigned mode, unsigned it) const {
        Params prm = arguments;
        prm.mode = mode;
        prm.iter = it;
        dispatch(ctx->pipes.finalize, prm, 1, kTreeLanes);
    }

    void encode_tri(const prepared_tri& t, bool forward) const {
        if (!arguments.m) return;
        bind(encoder, 13, {t.ptr, t.col, t.val, t.words, t.blocks, t.segments});
        if (factor_fp16) bind(encoder, 19, {factor_metadata});
        Params prm = arguments;
        prm.count = t.tag_count;
        const unsigned threads = std::min(kRowThreads, ctx->status.row_threads);
        dispatch(ctx->pipes.dataflow_init[factor_fp16][forward], prm, (t.tag_count + threads - 1) / threads, threads);
        for (auto stage : t.stages) {
            prm.offset = stage.first;
            const bool independent = forward && t.resident_round_plan && stage.last - stage.first > 1;
            dispatch(ctx->pipes.dataflow_solve[factor_fp16][independent ? 2 : unsigned(forward)], prm,
                     stage.last - stage.first, independent && stage.one_batch_per_block ? 32 : 256);
        }
    }

    void precondition() const {
        encode_tri(fwd, true);
        encode_tri(bwd, false);
    }

    void step(operation op) const {
        const auto pipe = ctx->pipes.pcg[unsigned(op)][op == operation::multiply && operator_df];
        const bool direction = op == operation::direction;
        const unsigned threads = direction ? std::min(kRowThreads, ctx->status.row_threads) : unsigned(kTreeLanes);
        dispatch(pipe, arguments, direction ? (arguments.n + threads - 1) / threads : arguments.groups, threads);
    }

    id<MTLComputeCommandEncoder> encoder = nil;

    template<class Record>
    bool submit(bool first, Record&& record) {
        with_pool([&] {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            if (first) {
                id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                if (blit == nil) throw std::runtime_error("Metal command buffer or encoder creation failed");
                [blit fillBuffer:x range:NSMakeRange(0, static_cast<NSUInteger>(arguments.n) * sizeof(df32)) value:0];
                [blit endEncoding];
            }
            encoder = [cb computeCommandEncoder];
            if (encoder == nil) throw std::runtime_error("Metal command buffer or encoder creation failed");
            bind(encoder, 1, {x, r, ap, p, z, partial, state_buffer, device_op_ptr, device_op_col, op_hi,
                              op_lo != nil ? op_lo : op_hi, tags});
            record();
            [encoder endEncoding];
            encoder = nil;
            finish(cb);
        });
        return static_cast<const solve_state*>([state_buffer contents])->active != 0;
    }

    explicit impl(const solve_options& options) : opts(options) {
        if (opts.backend == solve_backend::cpu)
            throw std::invalid_argument("metal_solver cannot execute an explicit CPU request");
        factor_fp16 = detail::resolve_fp16_storage(opts.factor_opts.factor_storage, false);
        if (!metal_solver::available())
            throw std::runtime_error("apxchol::metal_solver: the Metal backend is unavailable: " +
                                     dm::status().error);
    }

    void setup(const Eigen::SparseMatrix<double>& A, checkpoint* cp,
               detail::metal_setup::resident_factor* resident = nullptr);
    void prepare_device(cuda_host::csr_int<float> backward,
                        const std::vector<float>& scales,
                        const std::vector<float>& hi, const std::vector<float>& lo, unsigned m,
                        detail::metal_setup::resident_factor* resident = nullptr);

    // y = A' x in fp64, each row summed in storage order; rows are split across
    // threads by stored entries (detail::work_balanced_range).
    void spmv(const Eigen::VectorXd& x, Eigen::VectorXd& y) const {
        if (original_operator.rows() != 0) {
            const auto& A = original_operator;
            #pragma omp parallel
            {
                const auto [first, last] = detail::work_balanced_range(A.outerIndexPtr(), A.rows(),
                    omp_get_thread_num(), omp_get_num_threads());
                for (auto row = first; row < last; ++row) {
                    double sum = 0;
                    for (int p = A.outerIndexPtr()[row]; p < A.outerIndexPtr()[row + 1]; ++p)
                        sum += A.valuePtr()[p] * x[F.perm[A.innerIndexPtr()[p]]];
                    y[F.perm[row]] = sum;
                }
            }
            return;
        }
        const int* ptr = op.ptr.data();
        const int* col = op.idx.get();
        const double* val = op.vals.get();
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
        spmv(xp, work);
        return mh::norm(bp - work);
    }

    void centre(Eigen::VectorXd& x) const { x.array() -= x.mean(); }
};

void metal_solver::impl::setup(const Eigen::SparseMatrix<double>& A, checkpoint* cp,
                               detail::metal_setup::resident_factor* resident) {
    if (A.rows() != A.cols()) throw std::invalid_argument("metal_solver: the operator must be square");
    n = static_cast<std::size_t>(A.rows());
    if (n == 0) throw std::invalid_argument("metal_solver: empty operator");
    if (static_cast<std::size_t>(F.L.rows()) != n || F.perm.size() != n)
        throw std::invalid_argument("metal_solver: factorization dimension mismatch");
    if (!resident && F.L.nonZeros() > 0 && F.L.vals_.empty())
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
    cuda_host::csr_int<float> backward;
    std::vector<float> scales, hi, lo;
    bool exact;
    if (resident) {
        original_operator = A;
        if (!resident->operator_triangles_identical) {
            #pragma omp parallel
            {
                const auto [first, last] = detail::work_balanced_range(A.outerIndexPtr(), A.rows(),
                    omp_get_thread_num(), omp_get_num_threads());
                bool paired = true;
                for (auto row = first; row < last; ++row)
                    for (int p = A.outerIndexPtr()[row]; p < A.outerIndexPtr()[row + 1]; ++p)
                        original_operator.valuePtr()[p] = detail::canonical_lower_value(
                            A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr(), int(row),
                            A.innerIndexPtr()[p], p, paired);
            }
        }
        exact = resident->operator_exact;
        arguments.spmv_lanes = resident->max_operator_row > 256 ? kTreeLanes : 1;
        arguments.preconditioner_scale = resident->preconditioner_scale;
    } else {
        // Keep compensated dropping in the shared FP64 host implementation.
        // Both backends consume the CSR factor without a level-permuted value copy.
        backward = cuda_host::build_L11_csc_int<float>(F.L, m);
        if (!opts.keep_factor_values) F.L.release_values();
        scales = cuda_host::column_scales(backward);
        if (factor_fp16) cuda_host::normalize_fp16_scales(backward, scales);
        cuda_host::apply_factor_drop(backward, scales, factor_drop_rel_from_env(), factor_fp16);
        if (cp) (*cp)("metal_factor_prep");

        Eigen::SparseMatrix<double> compressed;
        if (!A.isCompressed()) {
            compressed = A;
            compressed.makeCompressed();
        }
        const auto& src = A.isCompressed() ? A : compressed;
        // Both triangles are stored: up to twice the stored entries of one triangle.
        if (src.nonZeros() > std::numeric_limits<int>::max() / 2)
            throw std::length_error("metal_solver: operators with 2^30 or more stored entries exceed the "
                                    "device's 32-bit offsets");
        op = detail::build_permuted_full_symmetric_csr(src, F.perm);
        int max_row = 0;
        double min_diagonal = std::numeric_limits<double>::infinity(), max_diagonal = 0;
        #pragma omp parallel for schedule(static) reduction(min : min_diagonal) reduction(max : max_diagonal, max_row)
        for (std::size_t row = 0; row < n; ++row) {
            max_row = std::max(max_row, op.ptr[row + 1] - op.ptr[row]);
            const auto end = op.idx.get() + op.ptr[row + 1];
            auto p = std::lower_bound(op.idx.get() + op.ptr[row], end, int(row));
            double diagonal = 0;
            for (; p != end && *p == int(row); ++p) diagonal += op.vals[p - op.idx.get()];
            if (diagonal > 0) {
                min_diagonal = std::min(min_diagonal, diagonal);
                max_diagonal = std::max(max_diagonal, diagonal);
            }
        }
        arguments.spmv_lanes = max_row > 256 ? kTreeLanes : 1;
        exact = op.fp32_exact;
        const std::size_t nnz = static_cast<std::size_t>(op.ptr.back());
        hi.resize(nnz); lo.resize(exact ? 0 : nnz);
        bool in_range = true;
        #pragma omp parallel for schedule(static) reduction(&& : in_range)
        for (std::ptrdiff_t p = 0; p < static_cast<std::ptrdiff_t>(nnz); ++p) {
            const double v = op.vals[p];
            in_range = in_range && ls::in_range(v);
            const mh::df d = mh::split(v);
            hi[p] = d.hi;
            if (!exact) lo[p] = d.lo;
        }
        if (!in_range)
            throw std::domain_error("metal_solver: an operator value is non-finite or outside [2^-100, 2^100]");
        // A fixed positive PCG preconditioner scale preserves x while balancing
        // rho and curvature exponents; diagonal extremes include mixed scales.
        arguments.preconditioner_scale = 1;
        if (max_diagonal > 0) {
            const int exponent_sum = std::ilogb(min_diagonal) + std::ilogb(max_diagonal);
            const int shift = static_cast<int>(std::floor(exponent_sum / 4.0));
            arguments.preconditioner_scale = std::scalbn(1.0f, shift);
        }
        if (cp) (*cp)("metal_operator");

    }
    prepare_device(std::move(backward), scales, hi, lo, m, resident);
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
    if (cp) { (*cp)("metal_upload"); cp->ascend(); }
    print_banner_once(stats, factor_fp16);
}

void metal_solver::impl::prepare_device(cuda_host::csr_int<float> backward,
                                       const std::vector<float>& scales,
                                       const std::vector<float>& hi, const std::vector<float>& lo,
                                       unsigned m, detail::metal_setup::resident_factor* resident) {
    context& ctx = shared();
    this->ctx = &ctx;
    arguments.n = static_cast<std::uint32_t>(n);
    arguments.m = m;
    arguments.groups = (arguments.n + kTreeRows - 1) / kTreeRows;
    const auto nnz = resident ? 0 : op.ptr.back();
    operator_df = resident ? !resident->operator_exact : !lo.empty();
    with_pool([&] {
        id<MTLDevice> dev = ctx.device;
        queue = [dev newCommandQueue];
        if (queue == nil) throw std::runtime_error("Metal command queue creation failed");
        if (resident) {
            device_op_ptr = resident->operator_ptr;
            device_op_col = resident->operator_idx;
            op_hi = resident->operator_hi;
            op_lo = resident->operator_lo;
        } else {
        device_op_ptr = upload(dev, op.ptr.data(), (n + 1) * sizeof(std::uint32_t), "operator pointers");
        device_op_col = upload(dev, op.idx.get(), nnz * sizeof(std::uint32_t), "operator indices");
        op_hi = upload(dev, hi.data(), nnz * sizeof(float), "operator high values");
        if (!lo.empty()) op_lo = upload(dev, lo.data(), nnz * sizeof(float), "operator low values");

        }
    });
    if (resident) {
        auto install = [](detail::metal_setup::triangle t) {
            prepared_tri result;
            result.ptr = t.ptr; result.col = t.idx; result.val = t.val;
            result.words = t.words; result.blocks = t.blocks; result.segments = t.segments;
            result.stages = std::move(t.stages);
            result.levels = t.levels; result.tag_count = t.tag_count;
            result.resident_round_plan = true;
            return result;
        };
        fwd = install(std::move(resident->forward));
        bwd = install(std::move(resident->backward));
        factor_metadata = resident->metadata;
    } else {
    auto install = [&]<class Value>(const cuda_host::csr_int<Value>& bt, std::span<const float> diag = {}) {
        const auto ft = cuda_host::transpose_csr(bt);
        fwd = prepare_tri(ctx, ft, true, diag);
        bwd = prepare_tri(ctx, bt, false, diag);
    };
    if (factor_fp16) {
        auto narrow = cuda_host::narrow_fp16_scaled(backward, scales);
        std::vector<mh::factor_metadata> metadata(m);
        for (unsigned j = 0; j < m; ++j) {
            metadata[j] = mh::pack_factor_metadata(scales[j], narrow.diag[j]);
            narrow.diag[j] = metadata[j].diagonal;
        }
        with_pool([&] {
            factor_metadata = upload(ctx.device, metadata.data(), metadata.size() * sizeof(metadata[0]), "factor scales");
        });
        cuda_host::csr_int<_Float16> half{backward.m, backward.nnz, std::move(backward.ptr),
                                        std::move(backward.idx), std::move(narrow.vals)};
        backward.vals.reset();
        install(half, narrow.diag);
    } else install(backward);
    }
    with_pool([&] {
        id<MTLDevice> dev = ctx.device;
        x = upload(dev, nullptr, n * sizeof(df32), "solution");
        r = upload(dev, nullptr, n * sizeof(df32), "residual");
        ap = upload(dev, nullptr, n * sizeof(df32), "operator product");
        p = upload(dev, nullptr, n * sizeof(float), "search direction");
        // Sweeps write rows < m; the extra Laplacian coordinate stays zero.
        z = upload(dev, nullptr, n * sizeof(float), "preconditioner result");
        partial = upload(dev, nullptr, static_cast<std::size_t>(arguments.groups) * sizeof(df32), "partial sums");
        state_buffer = upload(dev, nullptr, sizeof(solve_state), "solve state");
        tags = upload(dev, nullptr, std::max({n, std::size_t(fwd.tag_count),
            std::size_t(bwd.tag_count)}) * sizeof(std::uint32_t), "dataflow tags");
    });
}

metal_solver::metal_solver(const Eigen::SparseMatrix<double>& A, const solve_options& opts,
                           checkpoint* cp)
    : metal_solver(A, opts, cp, false) {}

metal_solver::metal_solver(const Eigen::SparseMatrix<double>& A, const solve_options& opts,
                           checkpoint* cp, bool input_prechecked)
    : impl_(new impl(opts)) {
    auto resident = detail::metal_setup::factorize(shared().device, A, opts, cp, input_prechecked);
    impl_->F = std::move(resident.factor);
    impl_->setup(A, cp, &resident);
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

    const auto& perm = s.F.perm;
    Eigen::VectorXd bp(s.n), work(s.n);
    auto& xp = result.x;
    mh::scatter({b.data(), s.n}, perm, {bp.data(), s.n});
    auto finish = [&] {
        if (s.laplacian) s.centre(xp);
        result.residual = s.residual_norm(bp, xp, work) / bnorm;
        mh::gather({xp.data(), s.n}, perm, {work.data(), s.n});
        result.x.swap(work);
        cp("exit_check");
        cp.ascend();
    };

    if (x0 != nullptr) {
        mh::scatter({x0->data(), s.n}, perm, {xp.data(), s.n});
        s.spmv(xp, work);
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
    mh::pack({work.data(), s.n}, scale, {static_cast<df32*>([s.r contents]), s.n});
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
    cp("gpu_pcg_loop");
    mh::add_unpacked({static_cast<df32*>([s.x contents]), s.n}, scale, {xp.data(), s.n});
    result.iterations = state.iters;
    finish();
}

factor_storage_type metal_solver::factor_storage() const {
    return impl_->factor_fp16 ? factor_storage_type::fp16 : factor_storage_type::fp32;
}

Eigen::VectorXd metal_solver::apply(const Eigen::VectorXd& r) const {
    impl& s = *impl_;
    if (r.size() != static_cast<Eigen::Index>(s.n))
        throw std::invalid_argument("metal_solver::apply: r length mismatch");
    if (!r.allFinite()) throw std::invalid_argument("metal_solver::apply: r contains a non-finite value");
    const auto& perm = s.F.perm;
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
