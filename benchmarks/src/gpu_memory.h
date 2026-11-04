#pragma once
// Per-process device memory from NVML, shared by benchmark.cpp and amgcl_cuda.cu.
//
// Every GPU solver is measured the same way: this process's `usedGpuMemory` on
// the current CUDA device, minus a baseline read right after CUDA context
// creation. Other processes on the device are excluded by construction and are
// only counted, so a cell can be flagged as sharing the GPU. A value that NVML
// cannot attribute to this process (no NVML, MIG, unsupported, PID not listed)
// is reported as unavailable, never as a device-wide number.
#include <string>

struct gpu_memory_baseline {
    std::string bus_id;          // NVML PCI bus id of the measured device
    std::string unavailable;     // empty when self_mib is valid, else the reason
    double self_mib = -1;        // this process after context creation
    int other_processes = -1;    // other compute processes on the device; -1 = unknown
    double other_mib = -1;       // their memory, where NVML reports it; -1 = unknown
};

// Read and keep the baseline. Call once, after the CUDA context exists and
// before any solver allocates device memory.
const gpu_memory_baseline& record_gpu_memory_baseline();

// This process's device memory above the baseline in MiB, or -1 when either
// reading is unavailable or no baseline was recorded.
double gpu_memory_above_baseline_mib();
