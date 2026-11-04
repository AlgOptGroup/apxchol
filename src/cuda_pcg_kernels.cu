#include "cuda_buffer.cuh"
#include "apxchol/solver/pcg_cuda_kernels.h"
#include "apxchol/solver/detail/gpu_pcg.h"
#include "apxchol/solver/detail/cuda_error.h"
#include "apxchol/csc_work.h"

#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

// The GPU-resident PCG's own kernels (see pcg_cuda_kernels.h for the contract):
// CSR SpMV with p.Ap folded in, the fused vector passes, and DETERMINISTIC
// reductions -- fixed grid, fixed per-thread strided assignment, fixed-order
// warp-shuffle + shared-memory tree per block, one partial per block, fixed-
// order single-block final reduce. No floating-point atomics anywhere.
namespace apxchol::detail::cuda_pcg_kernels {

namespace {

using namespace apxchol::detail::gpu_pcg;

constexpr int kWarps = kBlockThreads / 32;

template<int Lanes = 1>
struct pcg_math {
    using wide = double;
    using scalar = double;
    __device__ double preconditioned(double z) const { return z; }
    __device__ double add(double a, double b) const { return a + b; }
    __device__ double shuffle_down(double value, unsigned offset, unsigned width) const {
        return __shfl_down_sync(0xffffffffu, value, offset, width);
    }
    __device__ double add_product(double sum, double a, double b) const { return sum + a * b; }
    __device__ double update_solution(double x, double alpha, double p) const { return add_product(x, alpha, p); }
    __device__ double subtract_product(double a, double b, double c) const { return a - b * c; }
    __device__ double direction(double z, double p, double beta) const { return z + beta * p; }
    template<class Value>
    __device__ double add_entry(double sum, const Value* hi, const Value*, unsigned e, double p) const {
        return sum + double(hi[e]) * p;
    }
    __device__ double sum_row(double sum) const {
        return sum_lanes<Lanes>(*this, sum);
    }
};

// Every thread participates; the first warp folds one partial per warp.
template<class Math = pcg_math<>>
__device__ __forceinline__ double block_reduce_sum(double v, double* sh, Math math = {}) {
    v = sum_lanes<32>(math, v);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    return warp == 0 ? sum_lanes<32>(math, lane < kWarps ? sh[lane] : 0.0) : 0.0;
}

template<operation Op, int Lanes = 1, unsigned Columns = 1, class Value = double>
__global__ void __launch_bounds__(kBlockThreads)
step_kernel(std::int64_t n, workspace v, const int* ptr = nullptr,
            const int* col = nullptr, const Value* val = nullptr,
            const int* row_map = nullptr) {
    // Iterations after every column stopped run only to fill the batch;
    // finalize ignores their partials.
    if (!any_active<Columns>(v.state)) return;
    constexpr int rows = kBlockThreads / Lanes;
    const pcg_work<std::int64_t> work{static_cast<std::int64_t>(blockIdx.x) * rows, n,
        static_cast<std::int64_t>(gridDim.x) * rows, threadIdx.x / Lanes, n};
    const auto sums = detail::gpu_pcg::step_columns<Lanes, Columns>(
        Op, pcg_math<Lanes>{}, v, ptr, col, val, static_cast<const Value*>(nullptr),
        work, threadIdx.x % Lanes, row_map);
    if constexpr (Op != operation::direction) {
        __shared__ double scratch[Columns][kWarps];
        for (unsigned q = 0; q < Columns; ++q) {
            const double total = block_reduce_sum(sums.values[q], scratch[q]);
            if (threadIdx.x == 0) v.partial[blockIdx.x * Columns + q] = total;
        }
    }
}

struct recurrence_math {
    double bnorm;
    __device__ bool positive(double x) const { return x > 0 && isfinite(x); }
    __device__ bool valid_residual(double x) const { return isfinite(sqrt(x) / bnorm); }
    __device__ bool accept_ratio(double x) const { return isfinite(x); }
    __device__ double ratio(double x, double y) const { return x / y; }
    __device__ bool converged(double x, double threshold) const { return sqrt(x) / bnorm < threshold; }
};

template<unsigned Columns>
__global__ void __launch_bounds__(kBlockThreads)
finalize_kernel(const double* part, int blocks, solve_state* state, unsigned mode,
                unsigned iteration, column_sums<double, Columns> bnorm) {
    if (!any_active<Columns>(state)) return;
    __shared__ double scratch[Columns][kWarps];
    const pcg_work<int> work{0, blocks, kBlockThreads, static_cast<int>(threadIdx.x), blocks};
    const auto sums = sum_partials_columns<Columns>(pcg_math<>{}, part, work);
    for (unsigned q = 0; q < Columns; ++q) {
        const double sum = block_reduce_sum(sums.values[q], scratch[q]);
        // The iteration-zero check belongs only to warm columns. Cold columns
        // have alpha=0 and retain the scalar entry behavior, including tol>1.
        if (threadIdx.x == 0 && state[q].active &&
            (mode != kFinalizeCheck || iteration != 0 || state[q].alpha != 0))
            state[q] = advance(recurrence_math{bnorm.values[q]}, state[q], mode, sum, iteration);
    }
}

template<unsigned Columns, class Value>
void multiply(cudaStream_t stream, std::int64_t n, workspace v,
              const operator_csr& a, const Value* values, int lanes) {
    const int grid = pcg_blocks(n * lanes);
    auto launch = [&]<int Lanes>() {
        step_kernel<operation::multiply, Lanes, Columns><<<grid, kBlockThreads, 0, stream>>>(
            n, v, a.row_ptr, a.col_idx, values, a.row_map);
    };
    switch (lanes) {
        case 1: launch.template operator()<1>(); break;
        case 2: launch.template operator()<2>(); break;
        case 4: launch.template operator()<4>(); break;
        case 8: launch.template operator()<8>(); break;
        case 16: launch.template operator()<16>(); break;
        default: launch.template operator()<32>(); break;
    }
}

enum class exit_operation { center, maximum, norm };

template<unsigned Columns>
__global__ void __launch_bounds__(kBlockThreads)
initialize_kernel(std::int64_t n, workspace v, column_sums<double, Columns> scales) {
    for (std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < n; row += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        for (unsigned q = 0; q < Columns; ++q) {
            const auto i = row * Columns + q;
            v.r[i] = v.rhs[i] * scales.values[q];
            v.x[i] = 0;
        }
    }
}

struct maximum_math : pcg_math<> {
    __device__ double add(double a, double b) const {
        return isfinite(a) && isfinite(b) ? fmax(a, b) : INFINITY;
    }
};

template<exit_operation Op, unsigned Columns>
__global__ void __launch_bounds__(kBlockThreads)
exit_partials_kernel(std::int64_t n, workspace v, column_sums<double, Columns> scales) {
    const std::conditional_t<Op == exit_operation::maximum, maximum_math, pcg_math<>> math{};
    double sum[Columns]{};
    for (std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < n; row += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        for (unsigned q = 0; q < Columns; ++q) {
            const auto i = row * Columns + q;
            if constexpr (Op == exit_operation::center) sum[q] += v.x[i] / double(n);
            else if constexpr (Op == exit_operation::maximum) {
                v.r[i] = v.rhs[i] * fmax(1.0, scales.values[q]) - v.ap[i];
                sum[q] = math.add(sum[q], fabs(v.r[i]));
            } else {
                const double maximum = v.state[q].beta;
                const double normalized = maximum > 0 ? v.r[i] / maximum : 0;
                sum[q] += normalized * normalized;
            }
        }
    }
    __shared__ double scratch[Columns][kWarps];
    for (unsigned q = 0; q < Columns; ++q) {
        const double total = block_reduce_sum(sum[q], scratch[q], math);
        if (threadIdx.x == 0) v.partial[blockIdx.x * Columns + q] = total;
    }
}

template<exit_operation Op, unsigned Columns>
__global__ void __launch_bounds__(kBlockThreads)
exit_reduce_kernel(workspace v, int blocks, column_sums<double, Columns> bnorm,
                   column_sums<double, Columns> scales) {
    const std::conditional_t<Op == exit_operation::maximum, maximum_math, pcg_math<>> math{};
    __shared__ double scratch[Columns][kWarps];
    const pcg_work<int> work{0, blocks, kBlockThreads, static_cast<int>(threadIdx.x), blocks};
    const auto sums = sum_partials_columns<Columns>(math, v.partial, work);
    for (unsigned q = 0; q < Columns; ++q) {
        const double sum = block_reduce_sum(sums.values[q], scratch[q], math);
        if (threadIdx.x == 0) {
            if constexpr (Op == exit_operation::center) {
                v.state[q].beta = sum;
                v.state[q].active = kVerifySolution; // Include stopped columns.
            } else if constexpr (Op == exit_operation::maximum) v.state[q].beta = sum;
            else {
                const double maximum = v.state[q].beta;
                if (!isfinite(maximum)) v.state[q].rr = INFINITY;
                else {
                    int exponent;
                    const double mantissa = frexp(maximum, &exponent);
                    // solve_scale / verification_scale is min(1, solve_scale).
                    if (scales.values[q] < 1) exponent += ilogb(scales.values[q]);
                    v.state[q].rr = scalbn(mantissa * (sqrt(sum) / bnorm.values[q]), exponent);
                }
            }
        }
    }
}

template<unsigned Columns>
__global__ void __launch_bounds__(kBlockThreads)
restore_solution_kernel(std::int64_t n, workspace v, column_sums<double, Columns> scales) {
    unsigned errors[Columns]{};
    for (std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < n; row += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        for (unsigned q = 0; q < Columns; ++q) {
            const auto i = row * Columns + q;
            const double centered = v.x[i] - v.state[q].beta;
            const double output = centered / scales.values[q];
            if (!isfinite(output)) errors[q] |= kSolutionOverflow;
            if (centered != 0 && output == 0) errors[q] |= kSolutionUnderflow;
            v.x[i] = output;
            v.p[i] = output * fmax(1.0, scales.values[q]);
        }
    }
    for (unsigned q = 0; q < Columns; ++q)
        if (errors[q]) atomicOr(&v.state[q].active, errors[q]);
}

} // namespace

template<unsigned Columns>
void initialize(cudaStream_t stream, std::int64_t n, workspace v,
                std::array<double, Columns> scales) {
    column_sums<double, Columns> scaling{};
    for (unsigned q = 0; q < Columns; ++q) scaling.values[q] = scales[q];
    initialize_kernel<Columns><<<pcg_blocks(n), kBlockThreads, 0, stream>>>(n, v, scaling);
}

template<unsigned Columns>
void step(cudaStream_t stream, operation op, std::int64_t n, workspace v,
          const operator_csr& matrix, int lanes) {
    const int grid = pcg_blocks(n);
    switch (op) {
        case operation::multiply:
            if (matrix.fp32) multiply<Columns>(stream, n, v, matrix, matrix.values_f32, lanes);
            else multiply<Columns>(stream, n, v, matrix, matrix.values_f64, lanes);
            break;
        case operation::residual:
            step_kernel<operation::residual, 1, Columns><<<grid, kBlockThreads, 0, stream>>>(n, v);
            break;
        case operation::dot:
            step_kernel<operation::dot, 1, Columns><<<grid, kBlockThreads, 0, stream>>>(n, v);
            break;
        case operation::direction:
            step_kernel<operation::direction, 1, Columns><<<grid, kBlockThreads, 0, stream>>>(n, v);
            break;
    }
}

template<unsigned Columns>
void finalize(cudaStream_t stream, const double* part, int blocks, solve_state* state,
              unsigned mode, unsigned iteration, std::array<double, Columns> bnorm) {
    column_sums<double, Columns> norms{};
    for (unsigned q = 0; q < Columns; ++q) norms.values[q] = bnorm[q];
    finalize_kernel<Columns><<<1, kBlockThreads, 0, stream>>>(part, blocks, state, mode, iteration, norms);
}

template<unsigned Columns>
void finish(cudaStream_t stream, std::int64_t n, workspace v,
            const operator_csr& matrix, int lanes, bool laplacian,
            std::array<double, Columns> scales, std::array<double, Columns> bnorm) {
    column_sums<double, Columns> norms{}, scaling{};
    for (unsigned q = 0; q < Columns; ++q) {
        norms.values[q] = bnorm[q];
        scaling.values[q] = scales[q];
    }
    const int blocks = pcg_blocks(n);
    if (laplacian)
        exit_partials_kernel<exit_operation::center, Columns><<<blocks, kBlockThreads, 0, stream>>>(n, v, scaling);
    exit_reduce_kernel<exit_operation::center, Columns><<<1, kBlockThreads, 0, stream>>>(v, laplacian ? blocks : 0, norms, scaling);
    restore_solution_kernel<Columns><<<blocks, kBlockThreads, 0, stream>>>(n, v, scaling);
    step<Columns>(stream, operation::multiply, n, v, matrix, lanes);
    exit_partials_kernel<exit_operation::maximum, Columns><<<blocks, kBlockThreads, 0, stream>>>(n, v, scaling);
    exit_reduce_kernel<exit_operation::maximum, Columns><<<1, kBlockThreads, 0, stream>>>(v, blocks, norms, scaling);
    exit_partials_kernel<exit_operation::norm, Columns><<<blocks, kBlockThreads, 0, stream>>>(n, v, scaling);
    exit_reduce_kernel<exit_operation::norm, Columns><<<1, kBlockThreads, 0, stream>>>(v, blocks, norms, scaling);
}

template void initialize<1>(cudaStream_t, std::int64_t, workspace, std::array<double, 1>);
template void initialize<2>(cudaStream_t, std::int64_t, workspace, std::array<double, 2>);
template void initialize<4>(cudaStream_t, std::int64_t, workspace, std::array<double, 4>);
template void step<1>(cudaStream_t, operation, std::int64_t, workspace, const operator_csr&, int);
template void step<2>(cudaStream_t, operation, std::int64_t, workspace, const operator_csr&, int);
template void step<4>(cudaStream_t, operation, std::int64_t, workspace, const operator_csr&, int);
template void finalize<1>(cudaStream_t, const double*, int, solve_state*, unsigned, unsigned, std::array<double, 1>);
template void finalize<2>(cudaStream_t, const double*, int, solve_state*, unsigned, unsigned, std::array<double, 2>);
template void finalize<4>(cudaStream_t, const double*, int, solve_state*, unsigned, unsigned, std::array<double, 4>);
template void finish<1>(cudaStream_t, std::int64_t, workspace, const operator_csr&, int, bool, std::array<double, 1>, std::array<double, 1>);
template void finish<2>(cudaStream_t, std::int64_t, workspace, const operator_csr&, int, bool, std::array<double, 2>, std::array<double, 2>);
template void finish<4>(cudaStream_t, std::int64_t, workspace, const operator_csr&, int, bool, std::array<double, 4>, std::array<double, 4>);

namespace {

void operator_check(cudaError_t status, const char* = nullptr) {
    detail::check_cuda(status, "GPU operator CSR");
}

struct operator_allocation : detail::cuda_untracked_allocation {
    static std::size_t size_bytes(std::size_t count, std::size_t width, const char*) {
        if (count > std::numeric_limits<std::size_t>::max() / width)
            throw std::overflow_error("GPU operator CSR allocation overflow");
        return count * width;
    }
};
template<class T> using operator_buffer = detail::cuda_buffer<T, operator_check, operator_allocation>;

struct operator_status {
    // Each count is <= nnz <= INT_MAX: adding packed counts cannot carry from
    // the low (lower triangle) half into the high (upper triangle) half.
    unsigned long long triangles;
    unsigned flags; // bit 0: unsupported layout; bit 1: inexact fp32 lower value
};

__global__ void __launch_bounds__(kBlockThreads)
operator_count_columns(int n, int nnz, const int* outer, const int* inner,
                       const double* values, operator_status* status) {
    const int lane = threadIdx.x & 31;
    const std::int64_t warp = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x / 32;
    unsigned long long triangles = 0;
    bool bad = outer[0] != 0 || outer[n] != nnz, inexact = false;
    for (std::int64_t col = warp; col < n; col += stride) {
        const int begin = outer[col], end = outer[col + 1];
        if (begin < 0 || end < begin || end > nnz) { bad = true; continue; }
        for (std::int64_t p = static_cast<std::int64_t>(begin) + lane; p < end; p += 32) {
            const int row = inner[p];
            if (row < 0 || row >= n || (p > begin && inner[p - 1] >= row)) bad = true;
            if (row < col) triangles += 1ULL << 32;
            else {
                if (row > col) ++triangles;
                if (static_cast<double>(__double2float_rn(values[p])) != values[p]) inexact = true;
            }
        }
    }
    // All 256 lanes participate, including physical tail warps with no column.
    const int any_bad = __syncthreads_or(bad);
    const int any_inexact = __syncthreads_or(inexact);
    using reduce = cub::BlockReduce<unsigned long long, kBlockThreads>;
    __shared__ typename reduce::TempStorage scratch;
    const unsigned long long sum = reduce(scratch).Sum(triangles);
    if (threadIdx.x == 0) {
        atomicAdd(&status->triangles, sum);
        if (any_bad || any_inexact)
            atomicOr(&status->flags, static_cast<unsigned>(any_bad | (any_inexact << 1)));
    }
}

// Rows stay in input order: the symmetric CSC arrays already are its CSR, so
// only columns are renumbered into factor order. multiply() maps each stored
// row to its factor-order vector row through row_map (= perm).
__global__ void __launch_bounds__(kBlockThreads)
operator_fill_columns(int n, const int* outer, const int* inner,
                      const double* input, const std::uint32_t* perm,
                      int* columns, double* wide, float* narrow,
                      int* row_map, operator_status* status) {
    const int lane = threadIdx.x & 31;
    const std::int64_t warp = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x / 32;
    bool paired = true;
    for (std::int64_t col = warp; col < n; col += stride) {
        if (lane == 0) row_map[col] = static_cast<int>(perm[col]);
        for (std::int64_t p = static_cast<std::int64_t>(outer[col]) + lane;
             p < outer[col + 1]; p += 32) {
            const int row = inner[p];
            const double value = input[detail::canonical_lower_value(
                outer, inner, thrust::counting_iterator<int>{0}, col, row, p, paired)];
            columns[p] = static_cast<int>(perm[row]);
            if (narrow) narrow[p] = __double2float_rn(value);
            else wide[p] = value;
        }
    }
    if (__syncthreads_or(!paired) && threadIdx.x == 0) atomicOr(&status->flags, 1U);
}

} // namespace

bool try_build_permuted_operator_csr(
        int n, int nnz, const int* host_outer, const int* host_inner,
        const double* host_values, const std::uint32_t* host_perm,
        int precision, operator_csr& output) {
    if (n < 0 || nnz < 0 || precision < -1 || precision > 1 ||
        !host_outer || (nnz && (!host_inner || !host_values)) || (n && !host_perm))
        return false;
    if (output.row_ptr || output.col_idx || output.values_f64 || output.values_f32 || output.row_map)
        throw std::invalid_argument("GPU operator CSR output must not own allocations");
    operator_buffer<int> outer(static_cast<std::size_t>(n) + 1), inner(nnz);
    operator_buffer<double> input(nnz);
    operator_buffer<std::uint32_t> perm(n);
    operator_buffer<operator_status> status(1);
    operator_check(cudaMemcpy(outer.get(), host_outer, (static_cast<std::size_t>(n) + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (nnz) {
        operator_check(cudaMemcpy(inner.get(), host_inner, static_cast<std::size_t>(nnz) * sizeof(int), cudaMemcpyHostToDevice));
        operator_check(cudaMemcpy(input.get(), host_values, static_cast<std::size_t>(nnz) * sizeof(double), cudaMemcpyHostToDevice));
    }
    if (n) operator_check(cudaMemcpy(perm.get(), host_perm, static_cast<std::size_t>(n) * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
    operator_check(cudaMemset(status.get(), 0, sizeof(operator_status)));
    const int column_blocks = pcg_blocks(static_cast<std::int64_t>(n) * 32);
    operator_count_columns<<<column_blocks, kBlockThreads>>>(n, nnz, outer.get(), inner.get(), input.get(), status.get());
    operator_check(cudaGetLastError());
    operator_status observed{};
    operator_check(cudaMemcpy(&observed, status.get(), sizeof(observed), cudaMemcpyDeviceToHost));
    if ((observed.flags & 1U) || (observed.triangles & 0xffffffffULL) != (observed.triangles >> 32))
        return false;
    if (precision == 1 && (observed.flags & 2U))
        throw std::invalid_argument("FP32 operator storage requires exactly representable coefficients");
    const bool fp32 = precision == 1 || (precision == -1 && !(observed.flags & 2U));

    operator_buffer<int> columns(nnz), row_map(n);
    operator_buffer<double> wide(fp32 ? 0 : nnz);
    operator_buffer<float> narrow(fp32 ? nnz : 0);
    if (n) {
        operator_fill_columns<<<column_blocks, kBlockThreads>>>(n, outer.get(), inner.get(), input.get(), perm.get(),
            columns.get(), wide.get(), narrow.get(), row_map.get(), status.get());
        operator_check(cudaGetLastError());
        operator_check(cudaMemcpy(&observed, status.get(), sizeof(observed), cudaMemcpyDeviceToHost));
        if (observed.flags & 1U) return false;
    }
    // Publication cannot throw; the caller now owns exactly the final arrays.
    output.row_ptr = outer.release();
    output.col_idx = columns.release();
    output.row_map = row_map.release();
    if (fp32) output.values_f32 = narrow.release();
    else output.values_f64 = wide.release();
    output.nnz = nnz;
    output.fp32 = fp32;
    return true;
}

} // namespace apxchol::detail::cuda_pcg_kernels
