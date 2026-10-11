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

enum class operation { precondition, initial_direction, multiply, update_residual, update_direction };

// submit records/executes a batch and returns its completed active state.
// run may stop submission immediately (CUDA), or enqueue device-predicated
// operations until the batch completes (Metal). Neither requires an extra wait.
template<class Submit>
void solve(unsigned max_iter, unsigned batch_size, Submit&& submit) {
    batch_size = std::max(batch_size, 1u);
    unsigned first = 0;
    do {
        const unsigned last = first + std::min(batch_size, max_iter - first);
        if (!submit(first == 0, [&](auto&& run) {
            if (first == 0 &&
                (!run(operation::precondition, 0) || !run(operation::initial_direction, 0)))
                return;
            for (unsigned it = first; it < last; ++it) {
                if (!run(operation::multiply, it + 1) ||
                    !run(operation::update_residual, it + 1) ||
                    !run(operation::precondition, it + 1) ||
                    !run(operation::update_direction, it + 1))
                    break;
            }
        })) break;
        first = last;
    } while (first < max_iter);
}

} // namespace apxchol::detail::gpu_pcg
