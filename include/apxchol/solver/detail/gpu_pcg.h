#pragma once
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
