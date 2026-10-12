#pragma once

#include <cuda_runtime.h>

namespace apxchol::detail {

// Every thread participates; inactive lanes contribute zero. XOR and unsigned
// sum preserve exact audit hashes under this warp-to-block reduction.
template<unsigned BlockSize, class Digest>
__device__ Digest reduce_block_digest(Digest value, Digest* warp_values) {
    const unsigned lane = threadIdx.x % 32;
    const unsigned warp = threadIdx.x / 32;
    for (unsigned offset = 16; offset; offset >>= 1) {
        value.xor_hash ^= __shfl_down_sync(0xffffffffU, value.xor_hash, offset);
        value.sum_hash += __shfl_down_sync(0xffffffffU, value.sum_hash, offset);
    }
    if (!lane) warp_values[warp] = value;
    __syncthreads();
    if (!warp) {
        value = lane < BlockSize / 32 ? warp_values[lane] : Digest{};
        for (unsigned offset = 16; offset; offset >>= 1) {
            value.xor_hash ^= __shfl_down_sync(0xffffffffU, value.xor_hash, offset);
            value.sum_hash += __shfl_down_sync(0xffffffffU, value.sum_hash, offset);
        }
    }
    __syncthreads();
    return value;  // Only thread zero contains the complete result.
}

template<unsigned BlockSize, class Digest>
__device__ void digest_add_block(Digest value, Digest* target, Digest* warp_values) {
    value = reduce_block_digest<BlockSize>(value, warp_values);
    if (!threadIdx.x) {
        atomicXor(&target->xor_hash, value.xor_hash);
        atomicAdd(&target->sum_hash, value.sum_hash);
    }
}

} // namespace apxchol::detail
