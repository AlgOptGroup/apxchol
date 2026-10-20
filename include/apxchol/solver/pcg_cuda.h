#pragma once
// GPU-owned operator and reusable PCG workspace. Kernels use native FP64,
// fixed reductions and the shared device recurrence; only the completed state
// is read by the host each iteration. The core links cudart alone.

#include <Eigen/Sparse>
#include <Eigen/Core>
#include <cmath>
#include <array>
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
#include "apxchol/solver/detail/gpu_scaling.h"
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
        // Scalar-only sessions retain one column of storage. The first panel
        // grows this same workspace transactionally, then reuses it.
        reserve_columns(1);

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
               bool laplacian,
               const Eigen::VectorXd* x0 = nullptr,
               int stagnation_window = 0)
    {
        if (!ready_) throw std::runtime_error("cuda_pcg::solve: setup() not called");
        if (b_host.size() != n_ || (x0 && x0->size() != n_))
            throw std::invalid_argument("cuda_pcg::solve: vector length mismatch");
        std::array<int, 1> iterations{};
        std::array<double, 1> residuals{};
        x_host.resize(n_);
        solve_columns<1>(precond, b_host, x_host, tol, max_iter, iterations,
                         residuals, laplacian, x0, stagnation_window);
        iters_out = iterations[0];
        residual_out = residuals[0];
    }

    template<class Precond, class Input, class Output>
    void solve_pair(const Precond& precond, const Input& b_host, Output& x_host,
                    double tol, int max_iter, std::array<int, 2>& iterations,
                    std::array<double, 2>& residuals, bool laplacian,
                    const Input* x0 = nullptr, int stagnation_window = 0) {
        solve_columns<2>(precond, b_host, x_host, tol, max_iter, iterations,
                         residuals, laplacian, x0, stagnation_window);
    }

    template<unsigned Columns, class Precond, class Input, class Output>
    void solve_panel(const Precond& precond, const Input& b_host, Output& x_host,
                    double tol, int max_iter, std::array<int, Columns>& iterations,
                    std::array<double, Columns>& residuals, bool laplacian,
                    const Input* x0 = nullptr, int stagnation_window = 0) {
        solve_columns<Columns>(precond, b_host, x_host, tol, max_iter, iterations,
                               residuals, laplacian, x0, stagnation_window);
    }

    bool ready() const { return ready_; }
    int64_t n() const { return n_; }
    int64_t nnz() const { return matrix_.nnz; }
    /// Operator storage precision resolved at setup (FP32 only when exact).
    bool fp32_operator() const { return matrix_.fp32; }
    /// Threads per row of the SpMV resolved at setup (spmv_lanes_for / env).
    int spmv_lanes() const { return spmv_lanes_; }

private:
    template<unsigned Columns, class Precond, class Input, class Output>
    void solve_columns(const Precond& precond, const Input& b_host, Output& x_host,
                       double tol, int max_iter, std::array<int, Columns>& iterations,
                       std::array<double, Columns>& residuals, bool laplacian,
                       const Input* x0, int stagnation_window) {
        static_assert(Columns == 1 || Columns == 2 || Columns == 4);
        if (std::uint64_t(n_) > std::numeric_limits<unsigned>::max() / Columns)
            throw std::overflow_error("GPU RHS panel exceeds packed index range");
        if (!ready_) throw std::runtime_error("cuda_pcg::solve: setup() not called");
        if (b_host.rows() != n_ || b_host.cols() != Columns ||
            x_host.rows() != n_ || x_host.cols() != Columns ||
            (x0 && (x0->rows() != n_ || x0->cols() != Columns)))
            throw std::invalid_argument("cuda_pcg::solve: dimensions mismatch");
        reserve_columns(Columns);
        using host_panel = Eigen::Matrix<double, Eigen::Dynamic, Columns,
            Columns == 1 ? Eigen::ColMajor : Eigen::RowMajor>;
        host_panel packed(n_, Columns);
        std::array<double, Columns> scales{}, bnorm{}, max_b{}, max_x0{};
        bool any_active = false, any_warm = false;
        for (unsigned q = 0; q < Columns; ++q) {
            h_state_[q] = {};
            for (int64_t v = 0; v < n_; ++v) {
                const double rhs = b_host(v, q);
                if (!std::isfinite(rhs))
                    throw std::invalid_argument("cuda_pcg::solve: RHS must be finite");
                packed(h_perm_[v], q) = rhs;
                max_b[q] = std::max(max_b[q], std::abs(rhs));
                if (x0) {
                    const double guess = (*x0)(v, q);
                    if (!std::isfinite(guess))
                        throw std::invalid_argument("cuda_pcg::solve: x0 must be finite");
                    max_x0[q] = std::max(max_x0[q], std::abs(guess));
                }
            }
            any_active |= max_b[q] != 0;
        }
        if (!any_active) {
            x_host.setZero();
            iterations.fill(0);
            residuals.fill(0);
            return;
        }
        const std::size_t vector_bytes = n_ * Columns * sizeof(double);
        APXCHOL_PCG_CUDA_CHECK(cudaMemcpy(vectors_.rhs, packed.data(), vector_bytes, cudaMemcpyHostToDevice));
        for (unsigned q = 0; q < Columns; ++q) {
            scales[q] = 1;
            bnorm[q] = 1; // Inactive zero columns never divide by a zero norm.
            if (max_b[q] == 0) continue;
            scales[q] = detail::gpu_scaling::fp32_scale(std::max(max_b[q], max_x0[q]));
            if (scales[q] != 1) packed.col(q) *= scales[q];
            if constexpr (Columns == 1) bnorm[q] = detail::gpu_scaling::norm(packed);
            else {
                // Retain the contiguous scalar norm fold for each packed column.
                const Eigen::VectorXd rhs_column = packed.col(q);
                bnorm[q] = detail::gpu_scaling::norm(rhs_column);
            }
            if (!std::isnormal(bnorm[q] * bnorm[q]))
                throw std::invalid_argument("CUDA solve: RHS and initial guess exceed the common scaling range");
            auto& st = h_state_[q];
            st.thr = tol;
            st.prev = bnorm[q] * bnorm[q];
            st.active = 1;
            if (max_x0[q] != 0) { st.alpha = 1; any_warm = true; }
        }
        pcg_cuda::initialize<Columns>(0, n_, vectors_, scales);
        APXCHOL_PCG_CUDA_CHECK(cudaMemcpy(vectors_.state, h_state_, Columns * sizeof(*vectors_.state), cudaMemcpyHostToDevice));
        // Width changes reinterpret the reused storage; clear every grounded
        // coordinate on entry rather than inheriting a prior width's contents.
        if (laplacian && n_)
            APXCHOL_PCG_CUDA_CHECK(cudaMemset(vectors_.z + (n_ - 1) * Columns, 0, Columns * sizeof(double)));
        iteration<Precond, Columns> commands{*this, precond, bnorm,
            static_cast<unsigned>(std::max(stagnation_window, 0))};
        if (any_warm) {
            packed.setZero();
            for (unsigned q = 0; q < Columns; ++q) {
                if (!h_state_[q].alpha) continue;
                for (int64_t v = 0; v < n_; ++v)
                    packed(h_perm_[v], q) = (*x0)(v, q) * scales[q];
                if (laplacian) packed.col(q).array() -= packed.col(q).mean();
            }
            APXCHOL_PCG_CUDA_CHECK(cudaMemcpy(vectors_.p, packed.data(), vector_bytes, cudaMemcpyHostToDevice));
            // alpha=1 only for warm columns; cold p=0 and alpha=0. Their
            // iteration-zero convergence check is suppressed by finalize.
            commands.step(detail::gpu_pcg::operation::multiply);
            commands.step(detail::gpu_pcg::operation::residual);
            commands.finalize(detail::gpu_pcg::kFinalizeCheck, 0);
        }
        detail::gpu_pcg::solve(commands, std::max(max_iter, 0), 1);
        pcg_cuda::finish<Columns>(0, n_, vectors_, matrix_, spmv_lanes_, laplacian, scales, bnorm);
        APXCHOL_PCG_CUDA_CHECK(cudaMemcpyAsync(h_state_, vectors_.state,
            Columns * sizeof(*vectors_.state), cudaMemcpyDeviceToHost, 0));
        APXCHOL_PCG_CUDA_CHECK(cudaMemcpy(packed.data(), vectors_.x, vector_bytes, cudaMemcpyDeviceToHost));
        for (unsigned q = 0; q < Columns; ++q) {
            if (h_state_[q].active & pcg_cuda::kSolutionOverflow)
                throw std::overflow_error("CUDA solve: solution exceeds the finite range");
            if (h_state_[q].active & pcg_cuda::kSolutionUnderflow)
                throw std::underflow_error("CUDA solve: solution underflows to zero");
        }
        for (unsigned q = 0; q < Columns; ++q) {
            auto column = x_host.col(q);
            for (int64_t v = 0; v < n_; ++v) column[v] = packed(h_perm_[v], q);
            iterations[q] = h_state_[q].iters;
            residuals[q] = h_state_[q].rr;
        }
    }

    template<class Precond, unsigned Columns>
    struct iteration {
        cuda_pcg& s;
        const Precond& precond;
        std::array<double, Columns> bnorm;
        unsigned window;
        void precondition() const {
            static_assert(sizeof(pcg_cuda::solve_state) % sizeof(unsigned) == 0);
            precond.trsv().template solve_LLt_dev<Columns>(s.vectors_.r, s.vectors_.z,
                &s.vectors_.state->active, sizeof(pcg_cuda::solve_state) / sizeof(unsigned));
        }
        void finalize(unsigned mode, unsigned it) const {
            const int lanes = mode == detail::gpu_pcg::kFinalizeAlpha ? s.spmv_lanes_ : 1;
            const int blocks = pcg_cuda::pcg_blocks(s.n_ * lanes);
            pcg_cuda::finalize<Columns>(0, s.vectors_.partial, blocks, s.vectors_.state, mode, it, bnorm,
                               it ? window : 0);
        }
        void step(detail::gpu_pcg::operation op) const {
            pcg_cuda::step<Columns>(0, op, s.n_, s.vectors_, s.matrix_, s.spmv_lanes_);
        }
        template<class Record>
        bool submit(bool, Record&& record) const {
            record();
            APXCHOL_PCG_CUDA_CHECK(cudaMemcpyAsync(s.h_state_, s.vectors_.state, Columns * sizeof(*s.vectors_.state), cudaMemcpyDeviceToHost, 0));
            APXCHOL_PCG_CUDA_CHECK(cudaStreamSynchronize(0));
            bool active = false;
            for (unsigned q = 0; q < Columns; ++q) active |= s.h_state_[q].active != 0;
            return active;
        }
    };

    static void release_workspace(pcg_cuda::workspace& vectors, pcg_cuda::solve_state*& host) noexcept {
        for (void* pointer : std::initializer_list<void*>{vectors.x, vectors.r, vectors.ap,
                vectors.p, vectors.z, vectors.rhs, vectors.partial, vectors.state})
            if (pointer) (void)cudaFree(pointer);
        vectors = {};
        if (host) { (void)cudaFreeHost(host); host = nullptr; }
    }

    void reserve_columns(unsigned columns) {
        if (columns <= capacity_) return;
        pcg_cuda::workspace next{};
        pcg_cuda::solve_state* next_host = nullptr;
        try {
            for (double** pointer : {&next.x, &next.r, &next.p, &next.z, &next.ap, &next.rhs})
                APXCHOL_PCG_CUDA_CHECK(cudaMalloc(pointer, columns * n_ * sizeof(double)));
            APXCHOL_PCG_CUDA_CHECK(cudaMalloc(&next.partial, columns * pcg_cuda::kMaxBlocks * sizeof(double)));
            APXCHOL_PCG_CUDA_CHECK(cudaMalloc(&next.state, columns * sizeof(pcg_cuda::solve_state)));
            APXCHOL_PCG_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&next_host),
                columns * sizeof(pcg_cuda::solve_state), cudaHostAllocDefault));
        } catch (...) {
            release_workspace(next, next_host);
            throw;
        }
        // Every solve reinitializes its inputs. Publish only after all new
        // allocations succeed; an allocation failure keeps the old owner usable.
        release_workspace(vectors_, h_state_);
        vectors_ = next;
        h_state_ = next_host;
        capacity_ = columns;
    }

    void destroy() {
        // A failed setup may own arrays before ready_ becomes true.
        release_workspace(vectors_, h_state_);
        for (void* pointer : std::initializer_list<void*>{matrix_.row_ptr, matrix_.col_idx,
                matrix_.values_f64, matrix_.values_f32})
            if (pointer) (void)cudaFree(pointer);
        matrix_ = {};
        capacity_ = 0;
        spmv_lanes_ = 1;
        ready_ = false;
    }

    int64_t n_ = 0;
    bool ready_ = false;
    unsigned capacity_ = 0;
    pcg_cuda::operator_csr matrix_;
    pcg_cuda::workspace vectors_{};
    pcg_cuda::solve_state* h_state_ = nullptr;
    // SpMV row mapping (threads per row).
    int     spmv_lanes_  = 1;

    // Host-side permutation map (perm.indices()) for one-time use in solve()
    // to permute b -> b_perm and x_perm -> x.
    std::vector<node_index> h_perm_;


};

#undef APXCHOL_PCG_CUDA_CHECK

}  // namespace apxchol
