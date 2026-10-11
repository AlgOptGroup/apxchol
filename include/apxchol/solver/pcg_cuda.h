#pragma once
/// GPU-resident PCG loop -- our own kernels only (pcg_cuda_kernels.h): no
/// cuSPARSE, no cuBLAS. The CUDA library build links cudart alone.
///
/// Operator construction and all iteration arithmetic use the device. CPU
/// solves have their own complete host route; there is no host-operator
/// construction fallback in this owner.
///
/// `cuda_pcg` keeps the input matrix A as a full-symmetric CSR on the
/// device once, allocates all 5 PCG vectors (x, r, p, z, Ap) on device,
/// and runs every iteration entirely on the GPU:
///   - SpMV + p.Ap: our CSR kernel (pcg_cuda::spmv_pAp; LANES threads per
///     row picked from the average nnz/row, fp64 or fp32-exact operator
///     values promoted per product, fp64 accumulate)
///   - the fused vector passes mirroring the CPU loop (src/solve.cpp):
///     update_xr (x += alpha p, r -= alpha Ap, r.r in one pass), r.z,
///     update_p (p = z + beta p)
///   - precond.solve via the existing cuda_sptrsv::solve_LLt_dev
/// Every reduction is DETERMINISTIC (fixed grid, per-block partials, fixed-
/// order final reduce -- no floating-point atomics; see pcg_cuda_kernels.h),
/// so on our dataflow SpTRSV kernel the whole solve is bit-
/// identical run to run. Only the initial b is H2D'd, three 8-byte scalars
/// per iteration (p.Ap, r.r, r.z) come back to the host, only the final x
/// is D2H'd.
///
/// Env: APXCHOL_GPU_SPMV_LANES=1|2|4|8|16|32 overrides the SpMV's threads-
/// per-row choice (A/B only; read at setup; spmv_lanes() reports it).
/// APXCHOL_GPU_FP32_OPERATOR=0|1 overrides the operator storage precision
/// (default AUTO: fp32 iff every value round-trips fp32).

#include <Eigen/Sparse>
#include <Eigen/Core>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// check_cuda lives in apxchol/solver/sptrsv/cuda.h (already included
// transitively via preconditioner.h).
#include "apxchol/solver/pcg_cuda_kernels.h"
#include "apxchol/solver/detail/gpu_pcg.h"
#include "apxchol/solver/detail/gpu_diagnostics.h"
#include "apxchol/solver/sptrsv/cuda.h"

namespace apxchol {

namespace detail {
struct cuda_pcg_math {
    double bnorm;
    bool valid_initial(double) const { return true; }
    bool positive(double x) const { return x > 0; }
    bool valid_residual(double rr) const { return std::isfinite(std::sqrt(rr) / bnorm); }
    bool valid_rz(double x) const { return std::isfinite(x) && x != 0; }
    bool accept_ratio(double) const { return true; }
    double ratio(double x, double y) const { return x / y; }
    double mean(double) const { return 0; }
    bool converged(double rr, double tolerance) const { return std::sqrt(rr) / bnorm < tolerance; }
    bool stagnated(double previous, double current) const { return previous * 0.25 < current; }
};
} // namespace detail

#define APXCHOL_PCG_CUDA_CHECK(c)      apxchol::detail::check_cuda((c), #c)

/// All-on-device PCG. Construct once per matrix; reusable across solves.
class cuda_pcg {
public:
    cuda_pcg() = default;
    cuda_pcg(const cuda_pcg&) = delete;
    cuda_pcg& operator=(const cuda_pcg&) = delete;
    ~cuda_pcg() { destroy(); }

    /// Build the device-side full-symmetric CSR of the PERMUTED input
    /// (A_perm = P L P^T). The apxchol factor was built on A_perm, so the
    /// preconditioner's trsv_.solve_LLt_dev expects permuted input and gives
    /// permuted output. Running PCG entirely in permuted space lets us avoid
    /// per-iter permute/unpermute on device.
    void setup(const Eigen::SparseMatrix<double>& L,
               const std::vector<node_index>& perm)
    {
        destroy();
        n_ = static_cast<int64_t>(L.rows());

        // This owner has one operator-construction route. Unsupported stored
        // formats fail before PCG state is allocated; no host CSR is built.
        if constexpr (sizeof(node_index) != sizeof(std::uint32_t))
            throw std::invalid_argument("GPU PCG requires 32-bit node indices; request CPU explicitly");
        if (!L.isCompressed() || L.rows() != L.cols() ||
            L.rows() > std::numeric_limits<int>::max() ||
            L.nonZeros() > std::numeric_limits<int>::max() ||
            perm.size() < static_cast<std::size_t>(n_))
            throw std::invalid_argument("GPU PCG requires a compressed square int32-sized operator and a complete permutation");

        int precision = -1;
        if (const char* e = std::getenv("APXCHOL_GPU_FP32_OPERATOR")) {
            if (std::string(e) == "0") precision = 0;
            else if (*e) precision = 1;
        }
        // Allocate the permutation before accepting device ownership. The
        // factor supplied this validated bijection; the builder checks the CSC.
        h_perm_.assign(perm.begin(), perm.begin() + n_);
        pcg_cuda::operator_csr prepared;
        if (!pcg_cuda::try_build_permuted_operator_csr(
                static_cast<int>(n_), static_cast<int>(L.nonZeros()),
                L.outerIndexPtr(), L.innerIndexPtr(), L.valuePtr(),
                reinterpret_cast<const std::uint32_t*>(perm.data()), precision, prepared))
            throw std::invalid_argument("GPU PCG requires sorted, unique, fully paired CSC storage; request CPU explicitly");
        d_row_ptr_ = prepared.row_ptr;
        d_col_idx_ = prepared.col_idx;
        d_vals_ = prepared.values_f64;
        d_vals_f32_ = prepared.values_f32;
        nnz_ = prepared.nnz;
        fp32_op_ = prepared.fp32;

        // SpMV row mapping: threads per row from the average nnz/row
        // (pcg_cuda::spmv_lanes_for), env APXCHOL_GPU_SPMV_LANES overrides.
        spmv_lanes_ = pcg_cuda::spmv_lanes_for(n_ > 0 ? static_cast<double>(nnz_) / static_cast<double>(n_) : 1.0);
        if (const char* e = std::getenv("APXCHOL_GPU_SPMV_LANES")) {
            const int v = std::atoi(e);
            if (v == 1 || v == 2 || v == 4 || v == 8 || v == 16 || v == 32) spmv_lanes_ = v;
            else if (*e) std::fprintf(stderr, "[apxchol] APXCHOL_GPU_SPMV_LANES=%s ignored (expected 1|2|4|8|16|32); using %d\n", e, spmv_lanes_);
        }
        spmv_blocks_ = pcg_cuda::pcg_blocks(n_ * spmv_lanes_);
        vec_blocks_  = pcg_cuda::pcg_blocks(n_);

        // PCG iterate vectors, the reduction partials (one double per block of
        // the fixed grid) and the device / pinned-host scalar the fixed-order
        // final reduce lands in.
        for (double** p : {&d_b_, &d_x_, &d_r_, &d_p_, &d_z_, &d_Ap_})
            APXCHOL_PCG_CUDA_CHECK(cudaMalloc(p, n_ * sizeof(double)));
        APXCHOL_PCG_CUDA_CHECK(cudaMalloc(&d_part_, pcg_cuda::kMaxBlocks * sizeof(double)));
        APXCHOL_PCG_CUDA_CHECK(cudaMalloc(&d_scalar_, sizeof(double)));
        APXCHOL_PCG_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_scalar_), sizeof(double), cudaHostAllocDefault));

        // Complete owned setup before returning, including direct cuda_pcg
        // callers. The input CSC storage may end immediately after this call.
        APXCHOL_PCG_CUDA_CHECK(cudaDeviceSynchronize());
        if (detail::gpu_setup_diagnostics())
            std::fprintf(stderr, "[gpu-operator-builder] route=device completion_wait=1\n");
        ready_ = true;
    }

    /// Solve A*x = b via PCG using a device-capable preconditioner. The
    /// preconditioner's `solve_LLt_dev(d_in, d_out)` device entry point is
    /// invoked per iter. Returns iteration count and final relative residual.
    template<class Precond>
    void solve(const Precond& precond,
               const Eigen::VectorXd& b_host,
               Eigen::VectorXd& x_host,
               double tol,
               int max_iter,
               int& iters_out,
               double& residual_out,
               bool laplacian)
    {
        if (!ready_) throw std::runtime_error("cuda_pcg::solve: setup() not called");

        // Permute b -> b_perm.  perm.indices()[orig_v] = new_idx, so
        // b_perm[new_idx] = b[orig_v].
        Eigen::VectorXd b_perm(n_);
        for (int64_t v = 0; v < n_; ++v)
            b_perm[h_perm_[v]] = b_host[v];

        // Center b_perm for Laplacian path (rank n-1).
        if (laplacian)
            b_perm.array() -= b_perm.mean();

        const double bnorm = b_perm.norm();
        if (bnorm == 0.0) {
            x_host = Eigen::VectorXd::Zero(n_);
            iters_out = 0;
            residual_out = 0.0;
            return;
        }

        // H2D b_perm into d_b_. d_x_ starts at 0. d_r_ = d_b_.
        APXCHOL_PCG_CUDA_CHECK(cudaMemcpy(d_b_, b_perm.data(), n_ * sizeof(double),
                                          cudaMemcpyHostToDevice));
        APXCHOL_PCG_CUDA_CHECK(cudaMemset(d_x_, 0, n_ * sizeof(double)));
        APXCHOL_PCG_CUDA_CHECK(cudaMemcpyAsync(d_r_, d_b_, n_ * sizeof(double),
                                                cudaMemcpyDeviceToDevice, 0));

        // z = M^{-1} r ; p = z ; rz = r·z
        // The factor was built with m = laplacian ? n-1 : n. For the Laplacian
        // path, solve_LLt_dev only writes z[0..n-2]; z[n-1] is left untouched
        // (would be uninitialized on first iter, stale on later iters). Zero
        // it to keep p[n-1] = 0 throughout the loop (matches the CPU path
        // which sets x(m)=0 inside _solve_impl).
        precond.trsv().solve_LLt_dev(d_r_, d_z_);
        if (laplacian)
            APXCHOL_PCG_CUDA_CHECK(cudaMemset(d_z_ + (n_ - 1), 0, sizeof(double)));
        APXCHOL_PCG_CUDA_CHECK(cudaMemcpyAsync(d_p_, d_z_, n_ * sizeof(double),
                                                cudaMemcpyDeviceToDevice, 0));
        pcg_cuda::dot(0, n_, d_r_, d_z_, d_part_);
        namespace pcg = detail::gpu_pcg;
        pcg::solve_state<double, double> state{};
        state.rz = reduce(vec_blocks_);
        state.thr = tol;
        state.active = 1;
        const detail::cuda_pcg_math math{bnorm};
        for (int it = 0; it < max_iter; ++it) {
            if (fp32_op_) pcg_cuda::spmv_pAp(0, static_cast<int>(n_), d_row_ptr_, d_col_idx_, d_vals_f32_, d_p_, d_Ap_, d_part_, spmv_lanes_);
            else          pcg_cuda::spmv_pAp(0, static_cast<int>(n_), d_row_ptr_, d_col_idx_, d_vals_,     d_p_, d_Ap_, d_part_, spmv_lanes_);
            state = pcg::advance(math, state, pcg::kFinalizeAlpha, reduce(spmv_blocks_), 0.0, it + 1, 0, false);
            if (!state.active) break;

            pcg_cuda::update_xr(0, n_, d_x_, d_p_, d_r_, d_Ap_, state.alpha, d_part_);
            state = pcg::advance(math, state, pcg::kFinalizeCheck, reduce(vec_blocks_), 0.0, it + 1, 0, false);
            if (!state.active) break;

            precond.trsv().solve_LLt_dev(d_r_, d_z_);
            if (laplacian)
                APXCHOL_PCG_CUDA_CHECK(cudaMemset(d_z_ + (n_ - 1), 0, sizeof(double)));
            pcg_cuda::dot(0, n_, d_r_, d_z_, d_part_);
            state = pcg::advance(math, state, pcg::kFinalizeRz, reduce(vec_blocks_), 0.0, it + 1, 0, false);
            if (!state.active) break;
            pcg_cuda::update_p(0, n_, d_p_, d_z_, state.beta);
        }

        // The uploaded RHS is no longer needed. Reuse its touched host buffer
        // for D2H, then un-permute and (for Laplacian) re-center.
        APXCHOL_PCG_CUDA_CHECK(cudaMemcpy(b_perm.data(), d_x_,
                                          n_ * sizeof(double),
                                          cudaMemcpyDeviceToHost));
        x_host.resize(n_);
        // x[orig_v] = b_perm[new_idx] where new_idx = h_perm_[orig_v].
        for (int64_t v = 0; v < n_; ++v)
            x_host[v] = b_perm[h_perm_[v]];
        if (laplacian)
            x_host.array() -= x_host.mean();

        iters_out = state.iters;
        residual_out = state.iters ? std::sqrt(state.rr) / bnorm : 1.0;
    }

    bool ready() const { return ready_; }
    int64_t n() const { return n_; }
    int64_t nnz() const { return nnz_; }
    /// Operator storage precision resolved at setup (fp32 iff exact, or the env).
    bool fp32_operator() const { return fp32_op_; }
    /// Threads per row of the SpMV resolved at setup (spmv_lanes_for / env).
    int spmv_lanes() const { return spmv_lanes_; }

private:
    // Fixed-order final reduce of the first `blocks` per-block partials in
    // d_part_ (written by the kernel that just ran on stream 0) into the device
    // scalar, then 8 bytes back into pinned host memory. Synchronises stream 0.
    double reduce(int blocks) const {
        pcg_cuda::reduce_partials(0, d_part_, blocks, d_scalar_);
        APXCHOL_PCG_CUDA_CHECK(cudaMemcpyAsync(h_scalar_, d_scalar_, sizeof(double), cudaMemcpyDeviceToHost, 0));
        APXCHOL_PCG_CUDA_CHECK(cudaStreamSynchronize(0));
        return *h_scalar_;
    }

    void destroy() {
        // A setup failure can own operator arrays before ready_ becomes true.
        // Non-null ownership, not successful completion, controls retirement.
        for (double** p : {&d_b_, &d_x_, &d_r_, &d_p_, &d_z_, &d_Ap_, &d_part_, &d_scalar_}) {
            if (*p) { cudaFree(*p); *p = nullptr; }
        }
        if (h_scalar_) { cudaFreeHost(h_scalar_); h_scalar_ = nullptr; }
        if (d_row_ptr_) { cudaFree(d_row_ptr_); d_row_ptr_ = nullptr; }
        if (d_col_idx_) { cudaFree(d_col_idx_); d_col_idx_ = nullptr; }
        if (d_vals_)     { cudaFree(d_vals_);     d_vals_     = nullptr; }
        if (d_vals_f32_) { cudaFree(d_vals_f32_); d_vals_f32_ = nullptr; }
        fp32_op_ = false;
        spmv_lanes_ = 1; spmv_blocks_ = vec_blocks_ = 0;
        ready_ = false;
    }

    int64_t n_ = 0;
    int64_t nnz_ = 0;
    bool ready_ = false;

    // Device A_perm in CSR.
    int*    d_row_ptr_ = nullptr;
    int*    d_col_idx_ = nullptr;
    double* d_vals_    = nullptr;
    // fp32-operator path (APXCHOL_GPU_FP32_OPERATOR): operator values stored fp32,
    // SpMV promotes per product (Krylov vectors stay fp64).
    bool    fp32_op_    = false;
    float*  d_vals_f32_ = nullptr;
    // SpMV row mapping (threads per row) and the two fixed grids.
    int     spmv_lanes_  = 1;
    int     spmv_blocks_ = 0;
    int     vec_blocks_  = 0;

    // Host-side permutation map (perm.indices()) for one-time use in solve()
    // to permute b -> b_perm and x_perm -> x.
    std::vector<node_index> h_perm_;

    // Device PCG vectors + reduction scratch.
    double* d_b_  = nullptr;
    double* d_x_  = nullptr;
    double* d_r_  = nullptr;
    double* d_p_  = nullptr;
    double* d_z_  = nullptr;
    double* d_Ap_ = nullptr;
    double* d_part_   = nullptr;   // kMaxBlocks per-block partials
    double* d_scalar_ = nullptr;   // the final reduce's result
    double* h_scalar_ = nullptr;   // pinned host mirror of d_scalar_
};

#undef APXCHOL_PCG_CUDA_CHECK

}  // namespace apxchol
