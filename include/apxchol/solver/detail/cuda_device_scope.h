#pragma once
#include <cuda_runtime.h>

namespace apxchol::detail {

// Cleanup is best-effort and must not throw. Disabled scopes make no CUDA
// calls, preserving lazy context initialization for host-only lifetimes.
class cuda_device_scope {
public:
    explicit cuda_device_scope(int device, bool enabled = true) noexcept {
        restore_ = enabled &&
            cudaGetDevice(&saved_device_) == cudaSuccess &&
            saved_device_ != device &&
            cudaSetDevice(device) == cudaSuccess;
    }
    ~cuda_device_scope() {
        if (restore_) (void)cudaSetDevice(saved_device_);
    }

    cuda_device_scope(const cuda_device_scope&) = delete;
    cuda_device_scope& operator=(const cuda_device_scope&) = delete;

private:
    int saved_device_ = -1;
    bool restore_ = false;
};

} // namespace apxchol::detail
