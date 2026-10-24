#pragma once
#include "device_memory_error.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace apxchol::detail {
inline void check_cuda(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return;
    auto message = std::string(operation) + ": " + cudaGetErrorString(error);
    if (error == cudaErrorMemoryAllocation)
        throw device_memory_error(std::move(message));
    throw std::runtime_error(std::move(message));
}
} // namespace apxchol::detail
