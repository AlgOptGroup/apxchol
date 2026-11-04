#pragma once
// Shared between benchmark.cpp and CUDA adapter files (amgcl_cuda.cu etc.)
#include <string>

struct BenchResult {
    std::string solver_name;
    std::string graph_name;
    std::string execution_route; // v1: complete cpu or gpu setup/solve route; empty for other solvers
    int n = 0;
    int nnz = 0;
    double setup_time = 0;
    double solve_time = 0;
    double total_time = 0;
    int iterations = 0;
    double rel_residual = 0;
    double fillin = 0;
    long long factor_offdiag = -1; // measured v1 owner; -1 = unavailable
    double us_per_nnz = 0;
    double solve_rss_mb = 0;   // host VmRSS held during the solve phase (factor+operator+
                               // vectors, after the setup pool is freed). Peak RSS comes
                               // from /usr/bin/time externally; peak - this = setup transient.
    // GPU device memory (MiB) this process holds above its post-context baseline, read
    // via NVML for this PID (gpu_memory.h); -1 = unmeasured (CPU solvers, no CUDA, or NVML
    // cannot attribute it). setup: after setup returns; solve: at solve end with operator,
    // factor/hierarchy and PCG vectors still resident. Whole-run PEAK is sampled by the
    // runner's untimed nvidia-smi pass for the same PID.
    double setup_vram_mb = -1;
    double solve_vram_mb = -1;
    int retained_repeats = 1;
    int representative_repeat = 1;
    double max_repeat_rel_residual = -1;
    int solve_passes = 0;
    double stop_check_seconds = 0; // included in solve_time, never subtracted
};
