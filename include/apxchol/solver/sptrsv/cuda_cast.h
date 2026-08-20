#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Device-side precision casts for the mixed-precision GPU SpTRSV. The factor +
// triangular solve run in sptrsv_value_t (fp32) while the GPU-resident PCG
// keeps its vectors in fp64, so the preconditioner boundary needs a narrow
// (fp64->fp32) on the way in and a widen (fp32->fp64) on the way out. These
// wrap a trivial elementwise kernel; defined in src/cuda_cast.cu (compiled by
// nvcc -- the .cpp TUs are built by the host compiler and can't host
// __global__).
namespace apxchol {

void cast_f64_to_f32(const double* src, float* dst, int64_t n, cudaStream_t stream);
void cast_f32_to_f64(const float* src, double* dst, int64_t n, cudaStream_t stream);

} // namespace apxchol
