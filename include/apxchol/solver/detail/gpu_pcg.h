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

// Both backends enqueue the same device-predicated recurrence. submit owns
// the stream/command-buffer lifetime and reads the completed state once per batch.
template<class Backend>
void solve(Backend& gpu, unsigned max_iter, unsigned batch_size) {
    batch_size = std::max(batch_size, 1u);
    auto direction = [&](unsigned it) {
        gpu.step(operation::dot);
        gpu.finalize(it ? kFinalizeRz : kFinalizeRzInit, it);
        gpu.step(operation::direction);
    };
    unsigned first = 0;
    do {
        const unsigned last = first + std::min(batch_size, max_iter - first);
        if (!gpu.submit(first == 0, [&] {
            if (first == 0) {
                gpu.precondition();
                direction(0);
            }
            for (unsigned it = first; it < last; ++it) {
                gpu.step(operation::multiply);
                gpu.finalize(kFinalizeAlpha, it + 1);
                gpu.step(operation::residual);
                gpu.finalize(kFinalizeCheck, it + 1);
                gpu.precondition();
                direction(it + 1);
            }
        })) break;
        first = last;
    } while (first < max_iter);
}

} // namespace apxchol::detail::gpu_pcg
