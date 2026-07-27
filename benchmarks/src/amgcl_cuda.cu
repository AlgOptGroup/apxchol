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
#include <amgcl/adapter/crs_tuple.hpp>
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

// Called from benchmark.cpp. Caller passes CSR-style arrays for L11 (the
// pinned (n-1) x (n-1) submatrix), the RHS vector bsub, and the original
// (full-size) data needed to recompute residual the same way the CPU
// run_amgcl path does (so the printed RelRes is apples-to-apples).
//
//   m        — submatrix dimension (n - 1)
//   row_ptr  — CSR row pointers, size m + 1
//   col_idx  — CSR column indices, size row_ptr[m]
//   vals     — CSR values, size row_ptr[m]
//   bsub     — RHS, size m
//   tol/maxiter — PCG params
//   full_n   — original matrix dimension n (for residual recomputation)
//   full_b   — original full-size RHS, size full_n (for residual)
//   full_csr_row/col/val — original full L CSR (for residual computation)
//   nnz_full — nnz of full L
//
// On exit, populates *r (already has solver_name + graph_name + n/nnz set).
extern "C" void run_amgcl_cuda_impl(
    BenchResult* r,
    int m,
    const std::ptrdiff_t* row_ptr,   // size m+1
    const std::ptrdiff_t* col_idx,   // size row_ptr[m]
    const double*         vals,      // size row_ptr[m]
    const double*         bsub,      // size m
    double                tol,
    int                   maxiter,
    int                   full_n,
    const double*         full_b,
    const std::ptrdiff_t* full_row_ptr,
    const std::ptrdiff_t* full_col_idx,
    const double*         full_vals,
    int                   is_laplacian,
    int                   relax_coarse)
{
    using Backend = amgcl::backend::cuda<double>;

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

    // Build CRS tuple from host arrays. AMGCL copies these to device.
    std::vector<std::ptrdiff_t> ptr(row_ptr, row_ptr + m + 1);
    std::vector<std::ptrdiff_t> col(col_idx, col_idx + row_ptr[m]);
    std::vector<double>          val(vals,    vals    + row_ptr[m]);
    auto A = std::tie(m, ptr, col, val);

    double t0 = now_s();
    Solver solve(A, prm, bprm);
    r->setup_time = now_s() - t0;

    // Device vectors for RHS and solution.
    thrust::device_vector<double> d_b(bsub, bsub + m);
    thrust::device_vector<double> d_x(m, 0.0);

    t0 = now_s();
    auto [iters, error] = solve(d_b, d_x);
    cudaDeviceSynchronize();
    r->solve_time = now_s() - t0;
    r->total_time = r->setup_time + r->solve_time;
    r->iterations = static_cast<int>(iters);

    // Device VRAM held at solve end (AMG hierarchy + CSR operator + PCG vectors
    // still resident), total-free via cudaMemGetInfo — the device analog of
    // solve_rss_mb, sampled before `solve`/device vectors go out of scope.
    { size_t mf = 0, mt = 0;
      if (cudaMemGetInfo(&mf, &mt) == cudaSuccess && mt >= mf)
          r->solve_vram_mb = (mt - mf) / (1024.0 * 1024.0); }

    // Copy solution back to host and compute the full-system residual via
    // the same recipe as the CPU run_amgcl path (mean-centered).
    std::vector<double> sol(m, 0.0);
    thrust::copy(d_x.begin(), d_x.end(), sol.begin());

    // x = [sol; 0]. For a singular Laplacian, re-center into the range space
    // (subtract the mean). For SDDM (is_laplacian==0, m==full_n) the solve is
    // unique — no pin, no centering — matching the CPU run_amgcl path.
    std::vector<double> x(full_n, 0.0);
    double xs = 0.0;
    for (int i = 0; i < m; ++i) { x[i] = sol[i]; xs += sol[i]; }
    if (is_laplacian) {
        const double xmean = xs / full_n;
        for (int i = 0; i < full_n; ++i) x[i] -= xmean;
    }

    // res = b - L*x ; res -= res.mean()
    std::vector<double> res(full_n);
    for (int i = 0; i < full_n; ++i) res[i] = full_b[i];
    for (int i = 0; i < full_n; ++i) {
        double acc = 0.0;
        const std::ptrdiff_t b0 = full_row_ptr[i];
        const std::ptrdiff_t b1 = full_row_ptr[i + 1];
        for (std::ptrdiff_t p = b0; p < b1; ++p)
            acc += full_vals[p] * x[full_col_idx[p]];
        res[i] -= acc;
    }
    double rs = 0.0;
    for (int i = 0; i < full_n; ++i) rs += res[i];
    const double rmean = is_laplacian ? rs / full_n : 0.0;
    double rnorm2 = 0.0, bnorm2 = 0.0;
    for (int i = 0; i < full_n; ++i) {
        double rr = res[i] - rmean;
        rnorm2 += rr * rr;
        bnorm2 += full_b[i] * full_b[i];
    }
    const double bnorm = std::sqrt(bnorm2);
    r->rel_residual = std::sqrt(rnorm2) / (bnorm > 0 ? bnorm : 1.0);
    r->fillin = 0;
    r->us_per_nnz = r->total_time / r->nnz * 1e6;

    cusparseDestroy(cusparse_handle);
}
