#pragma once
/// GPU-resident PCG loop -- our own kernels only (pcg_cuda_kernels.h): no
/// cuSPARSE, no cuBLAS. The CUDA library build links cudart alone.
///
/// Why this exists: the existing CPU PCG path in src/solve.cpp issues
/// `precond.solve(r)` per iter, which on the CUDA build copies r → device,
/// runs the GPU SpTRSV, copies result → host. ~10 ms/iter is spent on
/// CPU↔GPU transfers alone. The SpMV `y = A*x` runs on the CPU even in
/// the CUDA build, paying further bandwidth cost and missing GPU SpMV
/// throughput (~10× the CPU rate on this hardware).
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
/// so on our SpTRSV kernels (dataflow / level-set) the whole solve is bit-
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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// check_cuda lives in apxchol/solver/sptrsv/cuda.h (already included
// transitively via preconditioner.h).
#include "apxchol/solver/pcg_cuda_kernels.h"
#include "apxchol/solver/sptrsv/cuda.h"

namespace apxchol {

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

        // Permute and build full-symmetric CSR in one go (host side, once).
        // A_perm[i,j] = L[iperm[i], iperm[j]] where iperm is perm.inverse().
        std::vector<int> h_row_ptr;
        std::vector<int> h_col_idx;
        std::vector<double> h_vals;
        bool op_fp32_exact = false;   // set by the builder: A is exactly fp32-representable
        build_permuted_full_symmetric_csr(L, perm, h_row_ptr, h_col_idx, h_vals, op_fp32_exact);
        nnz_ = static_cast<int64_t>(h_col_idx.size());

        // Save the permutation indices (host-side) for use in solve():
        // input b -> b_perm, output x_perm -> x.
        h_perm_.assign(perm.begin(), perm.begin() + n_);

        // Upload matrix to device.
        if (std::getenv("APXCHOL_GPU_MEM_DEBUG")) { size_t mf=0, mt=0; cudaMemGetInfo(&mf,&mt);
          fprintf(stderr,"[mem] PCG operator A_perm: nnz=%lld colidx=%.2fGB vals(fp64)=%.2fGB rowptr=%.2fGB"
                  " | GPU free=%.2f / total=%.2f GB BEFORE operator alloc\n",
                  (long long)nnz_, nnz_*4.0/1e9, nnz_*8.0/1e9, (n_+1)*4.0/1e9, mf/1e9, mt/1e9); }
        APXCHOL_PCG_CUDA_CHECK(cudaMalloc(&d_row_ptr_, (n_ + 1) * sizeof(int)));
        APXCHOL_PCG_CUDA_CHECK(cudaMalloc(&d_col_idx_, nnz_ * sizeof(int)));
        APXCHOL_PCG_CUDA_CHECK(cudaMemcpy(d_row_ptr_, h_row_ptr.data(),
                                          (n_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
        APXCHOL_PCG_CUDA_CHECK(cudaMemcpy(d_col_idx_, h_col_idx.data(),
                                          nnz_ * sizeof(int), cudaMemcpyHostToDevice));
        // Operator A_perm storage precision. fp32 is LOSSLESS only when every value
        // round-trips fp32 (op_fp32_exact, detected for free during the build above);
        // then it halves the operator footprint -- the lever that lets the giant social
        // factors (com-Orkut) fit 16GB -- at fp64-accurate compute (the SpMV promotes
        // each value to fp64; Krylov vectors stay fp64, so the 1e-8 floor is preserved).
        // Default = AUTO: fp32 iff exact. APXCHOL_GPU_FP32_OPERATOR overrides -- "0"
        // forces fp64; any other value forces fp32 (testing; floors if A is inexact).
        // Same rule as the CPU's op_fp32_ (src/solve.cpp).
        { const char* e = std::getenv("APXCHOL_GPU_FP32_OPERATOR");
          if (e && std::string(e) == "0")   fp32_op_ = false;
          else if (e && *e != '\0')         fp32_op_ = true;
          else                              fp32_op_ = op_fp32_exact; }
        if (fp32_op_) {
            APXCHOL_PCG_CUDA_CHECK(cudaMalloc(&d_vals_f32_, nnz_ * sizeof(float)));
            // Parallel, no-init cast (make_unique_for_overwrite avoids the O(nnz) zero
            // fill); then drop the fp64 host copy so the peak host footprint is fp32-only.
            auto h_vals_f = std::make_unique_for_overwrite<float[]>(static_cast<size_t>(nnz_));
            #pragma omp parallel for schedule(static)
            for (int64_t k = 0; k < nnz_; ++k) h_vals_f[k] = static_cast<float>(h_vals[k]);
            APXCHOL_PCG_CUDA_CHECK(cudaMemcpy(d_vals_f32_, h_vals_f.get(),
                                              nnz_ * sizeof(float), cudaMemcpyHostToDevice));
            std::vector<double>().swap(h_vals);
        } else {
            APXCHOL_PCG_CUDA_CHECK(cudaMalloc(&d_vals_, nnz_ * sizeof(double)));
            APXCHOL_PCG_CUDA_CHECK(cudaMemcpy(d_vals_, h_vals.data(),
                                              nnz_ * sizeof(double), cudaMemcpyHostToDevice));
        }
        if (std::getenv("APXCHOL_GPU_MEM_DEBUG"))
            fprintf(stderr, "[fp32op] operator stored %s (fp32-exact=%d)\n",
                    fp32_op_ ? "fp32" : "fp64", static_cast<int>(op_fp32_exact));

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

        ready_ = true;
    }

    /// Solve A*x = b via PCG using the host-side preconditioner. The
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
        double rz = reduce(vec_blocks_);

        double rnorm = bnorm;
        int it;
        for (it = 0; it < max_iter; ++it) {
            // Ap = A * p with pAp = p·Ap folded into the row loop (fp32-operator:
            // fp32 loads promoted per product, fp64 accumulate -- Ap stays
            // fp64-accurate either way). !(pAp > 0) catches NaN and <= 0.
            if (fp32_op_) pcg_cuda::spmv_pAp(0, static_cast<int>(n_), d_row_ptr_, d_col_idx_, d_vals_f32_, d_p_, d_Ap_, d_part_, spmv_lanes_);
            else          pcg_cuda::spmv_pAp(0, static_cast<int>(n_), d_row_ptr_, d_col_idx_, d_vals_,     d_p_, d_Ap_, d_part_, spmv_lanes_);
            const double pAp = reduce(spmv_blocks_);
            if (!(pAp > 0.0)) break;

            const double alpha = rz / pAp;

            // x += alpha * p ; r -= alpha * Ap ; rr = r·r  -- one pass.
            pcg_cuda::update_xr(0, n_, d_x_, d_p_, d_r_, d_Ap_, alpha, d_part_);
            const double rr = reduce(vec_blocks_);

            // rnorm = ||r|| / bnorm. NaN propagates through the reductions;
            // !(rnorm < tol) would re-enter the loop forever on NaN, so guard
            // with an isfinite check. ++it before break to count the just-
            // completed iter (matches apxchol_v1's `res.iterations = i + 1`).
            rnorm = std::sqrt(rr) / bnorm;
            if (!std::isfinite(rnorm)) { ++it; break; }
            if (rnorm < tol) { ++it; break; }

            // z = M^{-1} r ; rz_new = r·z
            precond.trsv().solve_LLt_dev(d_r_, d_z_);
            if (laplacian)
                APXCHOL_PCG_CUDA_CHECK(cudaMemset(d_z_ + (n_ - 1), 0, sizeof(double)));
            pcg_cuda::dot(0, n_, d_r_, d_z_, d_part_);
            const double rz_new = reduce(vec_blocks_);
            if (!std::isfinite(rz_new) || rz_new == 0.0) { ++it; break; }
            const double beta = rz_new / rz;
            rz = rz_new;

            // p = z + beta * p
            pcg_cuda::update_p(0, n_, d_p_, d_z_, beta);
        }

        // D2H x_perm, then un-permute and (for Laplacian) re-center.
        Eigen::VectorXd x_perm(n_);
        APXCHOL_PCG_CUDA_CHECK(cudaMemcpy(x_perm.data(), d_x_,
                                          n_ * sizeof(double),
                                          cudaMemcpyDeviceToHost));
        x_host.resize(n_);
        // x[orig_v] = x_perm[new_idx] where new_idx = h_perm_[orig_v].
        for (int64_t v = 0; v < n_; ++v)
            x_host[v] = x_perm[h_perm_[v]];
        if (laplacian)
            x_host.array() -= x_host.mean();

        iters_out = it;
        residual_out = rnorm;
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

    // Build full-symmetric CSR of A_perm = P L P^T from a (lower-half-stored)
    // symmetric matrix L and its permutation P. The factor F_.L was built on
    // A_perm, so running PCG in permuted space matches what trsv_.solve_LLt_dev
    // expects per iter.
    //
    // perm.indices()[orig_v] = new_idx ⇒  A_perm[i,j] = L[iperm(i), iperm(j)]
    // where iperm = P^{-1}. The permutation acts on BOTH row and col of L.
    // Output: row_ptr/col_idx/vals = CSR of A_perm (full symmetric, sorted).
    //
    // Parallel build: PASS 1 uses atomic-fetch-add on shared row_ptr counts;
    // PASS 2 uses atomic-fetch-add on shared row_pos to claim slots; per-row
    // sort runs as a parallel-for. Identical cache behavior pattern to our
    // make_graph/csc_to_csr parallel builds — works well here because n_perm
    // is large and the atomic-counter cache lines spread thin.
    // fp32_exact (out) := every operator value round-trips fp32 (v == double(float(v))),
    // so storing A in fp32 is LOSSLESS. Computed FOR FREE as an OMP reduction in PASS 2's
    // existing value loop -- no separate scan. (A is symmetric; PASS 2 visits the upper
    // triangle incl. diagonal = every distinct value.) This is the "detect at input"
    // gate that lets exact matrices use the half-size fp32 operator while Krylov compute
    // stays fp64 (so the 1e-8 residual floor is preserved).
    static void build_permuted_full_symmetric_csr(
        const Eigen::SparseMatrix<double>& L,
        const std::vector<node_index>& perm,
        std::vector<int>& row_ptr,
        std::vector<int>& col_idx,
        std::vector<double>& vals,
        bool& fp32_exact)
    {
        const int n = static_cast<int>(L.rows());
        const int* L_outer = L.outerIndexPtr();
        const int* L_inner = L.innerIndexPtr();
        const double* L_vals = L.valuePtr();
        // perm_[v] = new_idx for original vertex v.
        const node_index* p_idx = perm.data();

        // PASS 1 (parallel): atomic count per-row of A_perm.
        row_ptr.assign(n + 1, 0);
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n; ++k) {
            const int pk = p_idx[k];
            for (int p = L_outer[k]; p < L_outer[k + 1]; ++p) {
                const int row = L_inner[p];
                if (row < k) continue;
                const int pr = p_idx[row];
                __atomic_fetch_add(&row_ptr[pr + 1], 1, __ATOMIC_RELAXED);
                if (row != k)
                    __atomic_fetch_add(&row_ptr[pk + 1], 1, __ATOMIC_RELAXED);
            }
        }
        // Prefix sum (serial, m+1 entries — sub-ms even for n=4M).
        for (int i = 0; i < n; ++i)
            row_ptr[i + 1] += row_ptr[i];
        const int total = row_ptr[n];
        col_idx.assign(total, 0);
        vals.assign(total, 0.0);

        // PASS 2 (parallel): atomic-claim slot, scatter. Non-deterministic
        // per-row order across threads; restored by per-row sort below. The fp32
        // exactness reduction rides along for free (every value v is read here anyway).
        std::vector<int> pos(row_ptr.begin(), row_ptr.begin() + n);
        bool exact = true;
        #pragma omp parallel for schedule(static) reduction(&&:exact)
        for (int k = 0; k < n; ++k) {
            const int pk = p_idx[k];
            for (int p = L_outer[k]; p < L_outer[k + 1]; ++p) {
                const int row = L_inner[p];
                if (row < k) continue;
                const double v = L_vals[p];
                if (static_cast<double>(static_cast<float>(v)) != v) exact = false;  // lossless-fp32 check
                const int pr = p_idx[row];
                // A_perm[pr, pk] = v
                const int slot_pr = __atomic_fetch_add(&pos[pr], 1, __ATOMIC_RELAXED);
                col_idx[slot_pr] = pk;
                vals[slot_pr]    = v;
                if (row != k) {
                    // A_perm[pk, pr] = v
                    const int slot_pk = __atomic_fetch_add(&pos[pk], 1, __ATOMIC_RELAXED);
                    col_idx[slot_pk] = pr;
                    vals[slot_pk]    = v;
                }
            }
        }
        fp32_exact = exact;

        // Sort each row's (col, val) ascending: sorted CSR gives the SpMV its
        // best locality on the x gathers. Per-thread kv buffer reused across
        // rows (avoids n tiny mallocs).
        #pragma omp parallel
        {
            std::vector<std::pair<int, double>> kv;
            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                const int rs = row_ptr[i], re = row_ptr[i + 1];
                if (re - rs < 2) continue;
                kv.clear();
                kv.reserve(re - rs);
                for (int p = rs; p < re; ++p)
                    kv.emplace_back(col_idx[p], vals[p]);
                std::sort(kv.begin(), kv.end(),
                          [](const auto& a, const auto& b){ return a.first < b.first; });
                for (int p = rs; p < re; ++p) {
                    col_idx[p] = kv[p - rs].first;
                    vals[p]    = kv[p - rs].second;
                }
            }
        }
    }

    void destroy() {
        if (!ready_) return;
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
