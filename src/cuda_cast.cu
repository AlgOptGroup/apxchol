#include "apxchol/solver/sptrsv/cuda_cast.h"

// Elementwise precision casts on the device. One-pass, fully memory-bound; the
// per-iter cost is ~m loads + m stores at half/double width, negligible next to
// the SpSV it brackets. Used only by the GPU-resident PCG's fp32 preconditioner
// boundary (host-side casts cover the host-PCG path).
namespace apxchol {

namespace {

template <typename Src, typename Dst>
__global__ void cast_kernel(const Src* __restrict__ src, Dst* __restrict__ dst,
                            int64_t n) {
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i < n) dst[i] = static_cast<Dst>(src[i]);
}

inline int grid_for(int64_t n, int block) {
    return static_cast<int>((n + block - 1) / block);
}

} // namespace

void cast_f64_to_f32(const double* src, float* dst, int64_t n, cudaStream_t stream) {
    if (n <= 0) return;
    constexpr int block = 256;
    cast_kernel<double, float><<<grid_for(n, block), block, 0, stream>>>(src, dst, n);
}

void cast_f32_to_f64(const float* src, double* dst, int64_t n, cudaStream_t stream) {
    if (n <= 0) return;
    constexpr int block = 256;
    cast_kernel<float, double><<<grid_for(n, block), block, 0, stream>>>(src, dst, n);
}

} // namespace apxchol
