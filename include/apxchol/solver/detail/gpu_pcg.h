#pragma once
#include <algorithm>
#define APXCHOL_GPU_CODE(...) __VA_ARGS__
#ifdef __CUDACC__
#define APXCHOL_GPU_FUNCTION __device__ __forceinline__
#else
#define APXCHOL_GPU_FUNCTION inline
#endif
#define APXCHOL_GPU_BUFFER(T) T*
#include "gpu_pcg.inc"
#undef APXCHOL_GPU_BUFFER
#undef APXCHOL_GPU_FUNCTION
#undef APXCHOL_GPU_CODE

namespace apxchol::detail::gpu_pcg {

// Iterations enqueued per submit. Iterations after convergence are
// predicated off on the device, so the batch only bounds how often the host
// waits; one wait per iteration leaves the GPU idle between iterations.
inline constexpr unsigned kBatchIterations = 4;

// Both backends enqueue the same device-predicated recurrence. submit owns
// the stream/command-buffer lifetime and reads the completed state once per batch.
template<class Backend>
void solve(Backend& gpu, unsigned max_iter) {
    constexpr unsigned batch_size = kBatchIterations;
    unsigned first = 0;
    do {
        const unsigned last = first + std::min(batch_size, max_iter - first);
        if (!gpu.submit(first == 0, [&] {
            for (unsigned it = first; it < last; ++it) {
                gpu.precondition();
                gpu.step(operation::dot);
                gpu.finalize(it ? kFinalizeRz : kFinalizeRzInit, it);
                gpu.step(operation::direction);
                gpu.step(operation::multiply);
                gpu.finalize(kFinalizeAlpha, it + 1);
                gpu.step(operation::residual);
                gpu.finalize(kFinalizeCheck, it + 1);
            }
        })) break;
        first = last;
    } while (first < max_iter);
}

} // namespace apxchol::detail::gpu_pcg
