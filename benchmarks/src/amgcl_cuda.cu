// AMGCL CUDA backend adapter for the bench.
// Compiled with nvcc because amgcl::backend::cuda uses thrust + cuSPARSE.
//
// Exposes a C++-callable function with C linkage so benchmark.cpp can
// invoke it without itself being compiled by nvcc.

#include "bench_result.h"

#include <cuda_runtime.h>
#include <cusparse.h>

// AMGCL 1.4.4's cuda backend uses thrust::make_tuple/get, which CUDA 13.x's
// thrust no longer includes transitively. Bring it in explicitly.
#include <thrust/tuple.h>

#include <amgcl/backend/cuda.hpp>
#include <amgcl/adapter/zero_copy.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>

#include <chrono>
#include <vector>
#include <string>
#include <cmath>

namespace {
inline double now_s() {
    using clk = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}
}

// Called from benchmark.cpp. Caller passes the mandatory solve operator as CRS,
// the RHS, and output storage. Validation stays in benchmark.cpp so this adapter
// neither materializes nor traverses a second, benchmark-only operator.
//
//   m        — solve-operator dimension
//   row_ptr  — CSR row pointers, size m + 1
//   col_idx  — CSR column indices, size row_ptr[m]
//   vals     — CSR values, size row_ptr[m]
//   bsub     — RHS, size m
//   tol/maxiter — PCG params
// On exit, populates *r (already has solver_name + graph_name + n/nnz set).
extern "C" void run_amgcl_cuda_impl(
    BenchResult* r,
    double                host_prep_seconds,
    int m,
    const std::ptrdiff_t* row_ptr,   // size m+1
    const std::ptrdiff_t* col_idx,   // size row_ptr[m]
    const double*         vals,      // size row_ptr[m]
    const double*         bsub,      // size m
    double*               solution,  // size m
    double                tol,
    int                   maxiter,
    int                   relax_coarse)
{
    using Backend = amgcl::backend::cuda<double>;

    // The C++ wrapper has already spent host_prep_seconds grounding the
    // Laplacian and materializing the solve/residual CSR arrays.  Keep the
    // remaining CUDA-backend preparation in the same setup interval.  This
    // must start before handle creation and before the module-local copies
    // below; neither is shared benchmark input preparation.
    double t0 = now_s();

    Backend::params bprm;
    cusparseHandle_t cusparse_handle = nullptr;
    cusparseCreate(&cusparse_handle);
    bprm.cusparse_handle = cusparse_handle;

    using Solver = amgcl::make_solver<
        amgcl::amg<Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0>,
        amgcl::solver::cg<Backend>
    >;

    Solver::params prm;
    prm.solver.tol = tol;
    prm.solver.maxiter = maxiter;
    // ground = coarse (relax_coarse=1): relax the coarsest grid instead of the
    // default skyline_lu DIRECT solve, which throws "Zero sum in skyline_lu" on a
    // singular coarse operator. ground = pin (relax_coarse=0): the matrix is SPD, so
    // keep AMGCL's DEFAULT direct coarse solve. (is_laplacian still gates residual
    // mean-centering below, since the ORIGINAL L is a Laplacian either way.)
    if (relax_coarse) prm.precond.direct_coarse = false;

    // Use AMGCL's supplied non-owning adapter.  AMGCL copies this build matrix to
    // its CUDA backend as part of Solver construction; no module-local host copy
    // is needed.
    auto A = amgcl::adapter::zero_copy(m, row_ptr, col_idx, vals);

    Solver solve(A, prm, bprm);
    cudaDeviceSynchronize();
    r->setup_time = host_prep_seconds + (now_s() - t0);

    // Device vectors are per-RHS work, so charge their allocation/upload to
    // solve.  They were previously outside both setup_time and solve_time,
    // making total_time smaller than the work actually performed.
    t0 = now_s();
    thrust::device_vector<double> d_b(bsub, bsub + m);
    thrust::device_vector<double> d_x(m, 0.0);

    auto [iters, error] = solve(d_b, d_x);
    thrust::copy(d_x.begin(), d_x.end(), solution);
    r->solve_time = now_s() - t0;
    r->total_time = r->setup_time + r->solve_time;
    r->iterations = static_cast<int>(iters);

    // Device VRAM held at solve end (AMG hierarchy + CSR operator + PCG vectors
    // still resident), total-free via cudaMemGetInfo — the device analog of
    // solve_rss_mb, sampled before `solve`/device vectors go out of scope.
    { size_t mf = 0, mt = 0;
      if (cudaMemGetInfo(&mf, &mt) == cudaSuccess && mt >= mf)
          r->solve_vram_mb = (mt - mf) / (1024.0 * 1024.0); }

    r->fillin = 0;
    r->us_per_nnz = r->total_time / r->nnz * 1e6;

    cusparseDestroy(cusparse_handle);
}
