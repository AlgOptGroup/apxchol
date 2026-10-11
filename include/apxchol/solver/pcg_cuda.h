#pragma once
// GPU-owned operator and reusable PCG workspace. Kernels use native FP64,
// fixed reductions and the shared device recurrence; only the completed state
// is read by the host each iteration. The core links cudart alone.

#include <Eigen/Sparse>
#include <Eigen/Core>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <limits>
#include <initializer_list>
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
        if (!pcg_cuda::try_build_permuted_operator_csr(
                static_cast<int>(n_), static_cast<int>(L.nonZeros()),
                L.outerIndexPtr(), L.innerIndexPtr(), L.valuePtr(),
                reinterpret_cast<const std::uint32_t*>(perm.data()), precision, matrix_))
            throw std::invalid_argument("GPU PCG requires sorted, unique, fully paired CSC storage; request CPU explicitly");

        // SpMV row mapping: threads per row from the average nnz/row
        // (pcg_cuda::spmv_lanes_for), env APXCHOL_GPU_SPMV_LANES overrides.
        spmv_lanes_ = pcg_cuda::spmv_lanes_for(n_ > 0 ? static_cast<double>(matrix_.nnz) / static_cast<double>(n_) : 1.0);
        if (const char* e = std::getenv("APXCHOL_GPU_SPMV_LANES")) {
            const int v = std::atoi(e);
            if (v == 1 || v == 2 || v == 4 || v == 8 || v == 16 || v == 32) spmv_lanes_ = v;
            else if (*e) std::fprintf(stderr, "[apxchol] APXCHOL_GPU_SPMV_LANES=%s ignored (expected 1|2|4|8|16|32); using %d\n", e, spmv_lanes_);
        }
        spmv_blocks_ = pcg_cuda::pcg_blocks(n_ * spmv_lanes_);
        vec_blocks_  = pcg_cuda::pcg_blocks(n_);

        // Vectors, fixed reduction scratch and the completed device state.
        for (double** p : {&vectors_.x, &vectors_.r, &vectors_.p, &vectors_.z, &vectors_.ap})
            APXCHOL_PCG_CUDA_CHECK(cudaMalloc(p, n_ * sizeof(double)));
        APXCHOL_PCG_CUDA_CHECK(cudaMalloc(&vectors_.partial, pcg_cuda::kMaxBlocks * sizeof(double)));
        APXCHOL_PCG_CUDA_CHECK(cudaMalloc(&vectors_.state, sizeof(pcg_cuda::solve_state)));
        APXCHOL_PCG_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_state_), sizeof(pcg_cuda::solve_state), cudaHostAllocDefault));

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

        APXCHOL_PCG_CUDA_CHECK(cudaMemcpy(vectors_.r, b_perm.data(), n_ * sizeof(double), cudaMemcpyHostToDevice));
        APXCHOL_PCG_CUDA_CHECK(cudaMemset(vectors_.x, 0, n_ * sizeof(double)));

        *h_state_ = {};
        h_state_->thr = tol;
        h_state_->active = 1;
        APXCHOL_PCG_CUDA_CHECK(cudaMemcpy(vectors_.state, h_state_, sizeof(*vectors_.state), cudaMemcpyHostToDevice));
        if (laplacian)
            APXCHOL_PCG_CUDA_CHECK(cudaMemset(vectors_.z + (n_ - 1), 0, sizeof(double)));
        iteration commands{*this, precond, bnorm};
        detail::gpu_pcg::solve(commands, std::max(max_iter, 0), 1);

        // The uploaded RHS is no longer needed. Reuse its touched host buffer
        // for D2H, then un-permute and (for Laplacian) re-center.
        APXCHOL_PCG_CUDA_CHECK(cudaMemcpy(b_perm.data(), vectors_.x,
                                          n_ * sizeof(double),
                                          cudaMemcpyDeviceToHost));
        x_host.resize(n_);
        // x[orig_v] = b_perm[new_idx] where new_idx = h_perm_[orig_v].
        for (int64_t v = 0; v < n_; ++v)
            x_host[v] = b_perm[h_perm_[v]];
        if (laplacian)
            x_host.array() -= x_host.mean();

        iters_out = h_state_->iters;
        residual_out = h_state_->iters ? std::sqrt(h_state_->rr) / bnorm : 1.0;
    }

    bool ready() const { return ready_; }
    int64_t n() const { return n_; }
    int64_t nnz() const { return matrix_.nnz; }
    /// Operator storage precision resolved at setup (fp32 iff exact, or the env).
    bool fp32_operator() const { return matrix_.fp32; }
    /// Threads per row of the SpMV resolved at setup (spmv_lanes_for / env).
    int spmv_lanes() const { return spmv_lanes_; }

private:
    template<class Precond>
    struct iteration {
        cuda_pcg& s;
        const Precond& precond;
        double bnorm;
        void precondition() const { precond.trsv().solve_LLt_dev(s.vectors_.r, s.vectors_.z, &s.vectors_.state->active); }
        void finalize(unsigned mode, unsigned it) const {
            const int blocks = mode == detail::gpu_pcg::kFinalizeAlpha ? s.spmv_blocks_ : s.vec_blocks_;
            pcg_cuda::finalize(0, s.vectors_.partial, blocks, s.vectors_.state, mode, it, bnorm);
        }
        void step(detail::gpu_pcg::operation op) const {
            pcg_cuda::step(0, op, s.n_, s.vectors_, s.matrix_, s.spmv_lanes_);
        }
        template<class Record>
        bool submit(bool, Record&& record) const {
            record();
            APXCHOL_PCG_CUDA_CHECK(cudaMemcpyAsync(s.h_state_, s.vectors_.state, sizeof(*s.vectors_.state), cudaMemcpyDeviceToHost, 0));
            APXCHOL_PCG_CUDA_CHECK(cudaStreamSynchronize(0));
            return s.h_state_->active;
        }
    };

    void destroy() {
        // A setup failure can own operator arrays before ready_ becomes true.
        // Non-null ownership, not successful completion, controls retirement.
        for (void* p : std::initializer_list<void*>{vectors_.x, vectors_.r, vectors_.ap,
                 vectors_.p, vectors_.z, vectors_.partial, vectors_.state,
                 matrix_.row_ptr, matrix_.col_idx, matrix_.values_f64, matrix_.values_f32})
            if (p) cudaFree(p);
        vectors_ = {};
        matrix_ = {};
        if (h_state_) { cudaFreeHost(h_state_); h_state_ = nullptr; }
        spmv_lanes_ = 1; spmv_blocks_ = vec_blocks_ = 0;
        ready_ = false;
    }

    int64_t n_ = 0;
    bool ready_ = false;
    pcg_cuda::operator_csr matrix_;
    pcg_cuda::workspace vectors_{};
    pcg_cuda::solve_state* h_state_ = nullptr;
    // SpMV row mapping (threads per row) and the two fixed grids.
    int     spmv_lanes_  = 1;
    int     spmv_blocks_ = 0;
    int     vec_blocks_  = 0;

    // Host-side permutation map (perm.indices()) for one-time use in solve()
    // to permute b -> b_perm and x_perm -> x.
    std::vector<node_index> h_perm_;


};

#undef APXCHOL_PCG_CUDA_CHECK

}  // namespace apxchol
