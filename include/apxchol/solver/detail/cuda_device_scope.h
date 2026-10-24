#pragma once
#include "cuda_error.h"

namespace apxchol::detail {

// Active operations use checked{}; cleanup is best-effort and never throws.
// Disabled cleanup scopes make no CUDA calls, preserving host-only lifetimes.
class cuda_device_scope {
public:
    explicit cuda_device_scope(int device, bool enabled = true) noexcept {
        restore_ = enabled &&
            cudaGetDevice(&saved_device_) == cudaSuccess &&
            saved_device_ != device &&
            cudaSetDevice(device) == cudaSuccess;
    }
    struct checked {};
    cuda_device_scope(int device, checked) {
        check_cuda(cudaGetDevice(&saved_device_), "cudaGetDevice");
        if (saved_device_ != device) {
            check_cuda(cudaSetDevice(device), "cudaSetDevice");
            restore_ = true;
        }
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
