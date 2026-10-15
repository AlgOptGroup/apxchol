#include "cuda_buffer.cuh"
#include "apxchol/solver/pcg_cuda_kernels.h"
#include "apxchol/solver/detail/gpu_pcg.h"
#include "apxchol/csc_work.h"

#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

// The GPU-resident PCG's own kernels (see pcg_cuda_kernels.h for the contract):
// CSR SpMV with p.Ap folded in, the fused vector passes, and DETERMINISTIC
// reductions -- fixed grid, fixed per-thread strided assignment, fixed-order
// warp-shuffle + shared-memory tree per block, one partial per block, fixed-
// order single-block final reduce. No floating-point atomics anywhere.
namespace apxchol::pcg_cuda {

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
__device__ __forceinline__ double block_reduce_sum(double v, double* sh) {
    v = sum_lanes<32>(pcg_math<>{}, v);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    return warp == 0 ? sum_lanes<32>(pcg_math<>{}, lane < kWarps ? sh[lane] : 0.0) : 0.0;
}

template<operation Op, int Lanes = 1, unsigned Columns = 1, class Value = double>
__global__ void __launch_bounds__(kBlockThreads)
step_kernel(std::int64_t n, workspace v, const int* ptr = nullptr,
            const int* col = nullptr, const Value* val = nullptr) {
    constexpr int rows = kBlockThreads / Lanes;
    const pcg_work<std::int64_t> work{static_cast<std::int64_t>(blockIdx.x) * rows, n,
        static_cast<std::int64_t>(gridDim.x) * rows, threadIdx.x / Lanes, n};
    const auto sums = detail::gpu_pcg::step_columns<Lanes, Columns>(
        Op, pcg_math<Lanes>{}, v, ptr, col, val, static_cast<const Value*>(nullptr),
        work, threadIdx.x % Lanes);
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
    __device__ bool stagnated(double previous, double current) const { return previous * 0.25 < current; }
};

template<unsigned Columns>
__global__ void __launch_bounds__(kBlockThreads)
finalize_kernel(const double* part, int blocks, solve_state* state, unsigned mode,
                unsigned iteration, column_sums<double, Columns> bnorm, unsigned window) {
    __shared__ double scratch[Columns][kWarps];
    const pcg_work<int> work{0, blocks, kBlockThreads, static_cast<int>(threadIdx.x), blocks};
    const auto sums = sum_partials_columns<Columns>(pcg_math<>{}, part, work);
    for (unsigned q = 0; q < Columns; ++q) {
        const double sum = block_reduce_sum(sums.values[q], scratch[q]);
        // The iteration-zero check belongs only to warm columns. Cold columns
        // have alpha=0 and retain the scalar entry behavior, including tol>1.
        if (threadIdx.x == 0 && state[q].active &&
            (mode != kFinalizeCheck || iteration != 0 || state[q].alpha != 0))
            state[q] = advance(recurrence_math{bnorm.values[q]}, state[q], mode, sum, iteration, window);
    }
}

template<unsigned Columns, class Value>
void multiply(cudaStream_t stream, std::int64_t n, workspace v,
              const operator_csr& a, const Value* values, int lanes) {
    const int grid = pcg_blocks(n * lanes);
    auto launch = [&]<int Lanes>() {
        step_kernel<operation::multiply, Lanes, Columns><<<grid, kBlockThreads, 0, stream>>>(
            n, v, a.row_ptr, a.col_idx, values);
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

} // namespace

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
              unsigned mode, unsigned iteration, std::array<double, Columns> bnorm, unsigned window) {
    column_sums<double, Columns> norms{};
    for (unsigned q = 0; q < Columns; ++q) norms.values[q] = bnorm[q];
    finalize_kernel<Columns><<<1, kBlockThreads, 0, stream>>>(part, blocks, state, mode, iteration, norms, window);
}

template void step<1>(cudaStream_t, operation, std::int64_t, workspace, const operator_csr&, int);
template void step<2>(cudaStream_t, operation, std::int64_t, workspace, const operator_csr&, int);
template void finalize<1>(cudaStream_t, const double*, int, solve_state*, unsigned, unsigned, std::array<double, 1>, unsigned);
template void finalize<2>(cudaStream_t, const double*, int, solve_state*, unsigned, unsigned, std::array<double, 2>, unsigned);

namespace {

void operator_check(cudaError_t status, const char* = nullptr) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("GPU operator CSR: ") + cudaGetErrorString(status));
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
                       const double* values, const std::uint32_t* perm,
                       int* counts, operator_status* status) {
    const int lane = threadIdx.x & 31;
    const std::int64_t warp = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x / 32;
    unsigned long long triangles = 0;
    bool bad = outer[0] != 0 || outer[n] != nnz, inexact = false;
    for (std::int64_t col = warp; col < n; col += stride) {
        const int begin = outer[col], end = outer[col + 1];
        if (begin < 0 || end < begin || end > nnz) { bad = true; continue; }
        if (lane == 0) counts[perm[col]] = end - begin;
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

__global__ void __launch_bounds__(kBlockThreads)
operator_fill_columns(int n, const int* outer, const int* inner,
                      const double* input, const std::uint32_t* perm,
                      std::uint64_t* keys, double* values, operator_status* status) {
    const int lane = threadIdx.x & 31;
    const std::int64_t warp = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x / 32;
    bool paired = true;
    for (std::int64_t col = warp; col < n; col += stride) {
        for (std::int64_t p = static_cast<std::int64_t>(outer[col]) + lane;
             p < outer[col + 1]; p += 32) {
            const int row = inner[p];
            const double value = input[detail::canonical_lower_value(
                outer, inner, thrust::counting_iterator<int>{0}, col, row, p, paired)];
            // Input ordinal p is a unique fixed slot. Sorting this integer key
            // orders CSR rows and their columns without any floating arithmetic.
            keys[p] = (static_cast<std::uint64_t>(perm[col]) << 32) | perm[row];
            values[p] = value;
        }
    }
    if (__syncthreads_or(!paired) && threadIdx.x == 0) atomicOr(&status->flags, 1U);
}

__global__ void __launch_bounds__(kBlockThreads)
operator_decode_entries(int nnz, const std::uint64_t* keys, int* columns,
                        const double* values, float* narrow) {
    for (std::int64_t p = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         p < nnz; p += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        columns[p] = static_cast<int>(keys[p] & 0xffffffffULL);
        if (narrow) narrow[p] = __double2float_rn(values[p]);
    }
}

} // namespace

bool try_build_permuted_operator_csr(
        int n, int nnz, const int* host_outer, const int* host_inner,
        const double* host_values, const std::uint32_t* host_perm,
        int precision, operator_csr& output) {
    if (n < 0 || nnz < 0 || precision < -1 || precision > 1 ||
        !host_outer || (nnz && (!host_inner || !host_values)) || (n && !host_perm))
        return false;
    if (output.row_ptr || output.col_idx || output.values_f64 || output.values_f32)
        throw std::invalid_argument("GPU operator CSR output must not own allocations");
    operator_buffer<int> outer(static_cast<std::size_t>(n) + 1), inner(nnz), counts(n);
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
    operator_count_columns<<<column_blocks, kBlockThreads>>>(n, nnz, outer.get(), inner.get(), input.get(), perm.get(), counts.get(), status.get());
    operator_check(cudaGetLastError());
    operator_status observed{};
    operator_check(cudaMemcpy(&observed, status.get(), sizeof(observed), cudaMemcpyDeviceToHost));
    if ((observed.flags & 1U) || (observed.triangles & 0xffffffffULL) != (observed.triangles >> 32))
        return false;
    const bool fp32 = precision == 1 || (precision == -1 && !(observed.flags & 2U));

    operator_buffer<std::uint64_t> keys_a(nnz);
    operator_buffer<double> values_a(nnz);
    if (nnz) {
        operator_fill_columns<<<column_blocks, kBlockThreads>>>(n, outer.get(), inner.get(), input.get(), perm.get(), keys_a.get(), values_a.get(), status.get());
        operator_check(cudaGetLastError());
        operator_check(cudaMemcpy(&observed, status.get(), sizeof(observed), cudaMemcpyDeviceToHost));
        if (observed.flags & 1U) return false;
    }
    // The two completed status copies precede retirement of input storage.
    // Do not overlap the full CSC input with both radix-sort key/value buffers.
    outer.reset(); inner.reset(); input.reset(); perm.reset(); status.reset();
    operator_buffer<int> rows(static_cast<std::size_t>(n) + 1);
    if (n) {
        std::size_t bytes = 0;
        operator_check(cub::DeviceScan::ExclusiveSum(nullptr, bytes, counts.get(), rows.get(), n));
        operator_buffer<unsigned char> scratch(bytes);
        operator_check(cub::DeviceScan::ExclusiveSum(scratch.get(), bytes, counts.get(), rows.get(), n));
    }
    operator_check(cudaMemcpy(rows.get() + n, &nnz, sizeof(int), cudaMemcpyHostToDevice));
    counts.reset();
    operator_buffer<std::uint64_t> keys_b(nnz);
    operator_buffer<double> values_b(nnz);
    if (nnz) {
        std::size_t bytes = 0;
        operator_check(cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys_a.get(), keys_b.get(), values_a.get(), values_b.get(), nnz));
        operator_buffer<unsigned char> scratch(bytes);
        operator_check(cub::DeviceRadixSort::SortPairs(scratch.get(), bytes, keys_a.get(), keys_b.get(), values_a.get(), values_b.get(), nnz));
    }
    keys_a.reset(); values_a.reset();
    operator_buffer<int> columns(nnz);
    operator_buffer<float> narrow(fp32 ? nnz : 0);
    if (nnz) {
        operator_decode_entries<<<pcg_blocks(nnz), kBlockThreads>>>(
            nnz, keys_b.get(), columns.get(), values_b.get(), narrow.get());
        operator_check(cudaGetLastError());
        operator_check(cudaStreamSynchronize(nullptr));
    }
    // Publication cannot throw; the caller now owns exactly the final arrays.
    output.row_ptr = rows.release();
    output.col_idx = columns.release();
    if (fp32) output.values_f32 = narrow.release();
    else output.values_f64 = values_b.release();
    output.nnz = nnz;
    output.fp32 = fp32;
    return true;
}

} // namespace apxchol::pcg_cuda
