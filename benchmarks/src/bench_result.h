#pragma once
// Shared between benchmark.cpp and CUDA adapter files (amgcl_cuda.cu etc.)
#include <string>

struct BenchResult {
    std::string solver_name;
    std::string graph_name;
    int n = 0;
    int nnz = 0;
    double setup_time = 0;
    double solve_time = 0;
    double total_time = 0;
    int iterations = 0;
    double rel_residual = 0;
    double fillin = 0;
    double us_per_nnz = 0;
    double solve_rss_mb = 0;   // host VmRSS held during the solve phase (factor+operator+
                               // vectors, after the setup pool is freed). Peak RSS comes
                               // from /usr/bin/time externally; peak - this = setup transient.
    // GPU device memory (MB), -1 = unmeasured (CPU solvers, or no CUDA). The device analog
    // of solve_rss_mb: VRAM in use at the end of the solve phase (operator + factor + SpSV
    // buffers + PCG vectors still resident), read via cudaMemGetInfo (total-free). Whole-run
    // PEAK VRAM is sampled externally by the sweep (nvidia-smi --query-compute-apps), the
    // device analog of /usr/bin/time -%M; peak - solve = the setup/analysis transient.
    double solve_vram_mb = -1;
};
