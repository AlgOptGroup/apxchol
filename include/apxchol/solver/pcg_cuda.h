#pragma once
// GPU-owned operator and reusable PCG workspace. Kernels use native FP64,
// fixed reductions and the shared device recurrence; the host reads only the
// completed state, once per batch of iterations. The core links cudart alone.

#include <Eigen/Sparse>
#include <Eigen/Core>
#include <cmath>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "apxchol/solver/pcg_cuda_kernels.h"
#include "apxchol/solver/detail/gpu_pcg.h"
#include "apxchol/solver/detail/pcg_restart.h"
#include "apxchol/solver/detail/gpu_scaling.h"
#include "apxchol/solver/detail/gpu_diagnostics.h"
#include "apxchol/solver/sptrsv/cuda.h"

namespace apxchol::detail {
class cuda_route;

#define APXCHOL_PCG_CUDA_CHECK(c)      apxchol::detail::check_cuda((c), #c)

/// All-on-device PCG. Construct once per matrix; reusable across solves.
class cuda_pcg {
    friend class cuda_route;
public:
    cuda_pcg() = default;
    /// Shares the device operator, which is read-only after setup, and
    /// allocates its own vectors and stream, so copies solve concurrently.
    cuda_pcg(const cuda_pcg& other)
        : n_(other.n_), op_(other.op_), spmv_lanes_(other.spmv_lanes_) {
        if (!other.ready_) return;
        device_ = other.device_;
        const detail::cuda_device_scope device(device_, detail::cuda_device_scope::checked{});
        try {
            APXCHOL_PCG_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
            reserve_columns(1);
        } catch (...) {
            destroy();
            throw;
        }
        ready_ = true;
    }
    cuda_pcg& operator=(const cuda_pcg&) = delete;
    ~cuda_pcg() { destroy(); }

    /// Build the device-side full-symmetric CSR of the PERMUTED input
    /// (A_perm = P L P^T). The apxchol factor was built on A_perm, so the
    /// preconditioner's trsv_.solve_LLt_dev expects permuted input and gives
    /// permuted output. Running PCG entirely in permuted space lets us avoid
    /// per-iter permute/unpermute on device.
    void setup(const Eigen::SparseMatrix<double>& L,
               const std::vector<node_index>& perm, bool verbose = false)
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

        // Allocate the permutation before accepting device ownership. The
        // factor supplied this validated bijection; the builder checks the CSC.
        auto op = std::make_shared<operator_storage>();
        op->perm.assign(perm.begin(), perm.begin() + n_);
        APXCHOL_PCG_CUDA_CHECK(cudaGetDevice(&device_));
        op->device = device_;
        if (!cuda_pcg_kernels::try_build_permuted_operator_csr(
                static_cast<int>(n_), static_cast<int>(L.nonZeros()),
                L.outerIndexPtr(), L.innerIndexPtr(), L.valuePtr(),
                reinterpret_cast<const std::uint32_t*>(perm.data()), -1, op->matrix))
            throw std::invalid_argument("GPU PCG requires sorted, unique, fully paired CSC storage; request CPU explicitly");
        op_ = std::move(op);
        APXCHOL_PCG_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

        spmv_lanes_ = cuda_pcg_kernels::spmv_lanes_for(n_ > 0 ? static_cast<double>(op_->matrix.nnz) / static_cast<double>(n_) : 1.0);
        // Scalar-only sessions retain one column of storage. The first panel
        // grows this same workspace transactionally, then reuses it.
        reserve_columns(1);

        // Complete owned setup before returning, including direct cuda_pcg
        // callers. The input CSC storage may end immediately after this call.
        APXCHOL_PCG_CUDA_CHECK(cudaDeviceSynchronize());
        if (verbose)
            std::fprintf(stderr, "[gpu-operator-builder] route=device completion_wait=1\n");
        ready_ = true;
    }

    /// Solve A*X = B for one, two or four columns via PCG using a
    /// device-capable preconditioner, whose `trsv().solve_LLt_dev` runs per
    /// iteration. Reports iteration counts and final relative residuals.
    template<unsigned Columns, class Precond, class Input, class Output>
    void solve_panel(const Precond& precond, const Input& b_host, Output& x_host,
                    double tol, int max_iter, std::array<int, Columns>& iterations,
                    std::array<double, Columns>& residuals, bool laplacian,
                    const Input* x0 = nullptr) {
        solve_columns<Columns>(precond, b_host, x_host, tol, max_iter, iterations,
                               residuals, laplacian, x0);
    }

    bool ready() const { return ready_; }
    int64_t n() const { return n_; }
    int64_t nnz() const { return op_ ? op_->matrix.nnz : 0; }
    /// Operator storage precision resolved at setup (FP32 only when exact).
    bool fp32_operator() const { return op_ && op_->matrix.fp32; }
    /// This copy's stream; every solve is ordered on it.
    cudaStream_t stream() const { return stream_; }
    /// Device operator values; copies share them.
    const void* device_operator_values() const {
        return !op_ ? nullptr : op_->matrix.fp32 ? static_cast<const void*>(op_->matrix.values_f32)
                                                 : static_cast<const void*>(op_->matrix.values_f64);
    }
    /// Threads per row of the SpMV resolved from the average row length.
    int spmv_lanes() const { return spmv_lanes_; }
    double device_memory_used_mb() const {
        const detail::cuda_device_scope device(device_, detail::cuda_device_scope::checked{});
        size_t free = 0, total = 0;
        return cudaMemGetInfo(&free, &total) == cudaSuccess && total >= free
            ? (total - free) / (1024.0 * 1024.0) : -1;
    }

private:
    template<unsigned Columns, class Precond, class Input, class Output>
    void solve_columns(const Precond& precond, const Input& b_host, Output& x_host,
                       double tol, int max_iter, std::array<int, Columns>& iterations,
                       std::array<double, Columns>& residuals, bool laplacian,
                       const Input* x0) {
        auto stopped = solve_columns_once<Columns>(precond, b_host, x_host, tol, max_iter,
                                                    iterations, residuals, laplacian, x0);
        for (unsigned q = 0; q < Columns; ++q) {
            int progress = iterations[q];
            const auto restart = [&] {
                return detail::pcg_restart(stopped[q], residuals[q] < tol, residuals[q],
                                           progress, max_iter - iterations[q]);
            };
            if (!restart()) continue;
            Eigen::VectorXd solution = x_host.col(q);
            using column_view = Eigen::Ref<const Eigen::VectorXd, 0,
                                           Eigen::InnerStride<Eigen::Dynamic>>;
            const column_view rhs(b_host.col(q)), initial(solution);
            while (restart()) {
                std::array<int, 1> count{};
                std::array<double, 1> residual{};
                const auto next = solve_columns_once<1>(precond, rhs, solution, tol,
                    max_iter - iterations[q], count, residual, laplacian, &initial);
                progress = count[0];
                iterations[q] += progress;
                residuals[q] = residual[0];
                stopped[q] = next[0];
            }
            x_host.col(q) = solution;
        }
    }

    template<unsigned Columns, class Precond, class Input, class Output>
    std::array<bool, Columns> solve_columns_once(const Precond& precond,
                       const Input& b_host, Output& x_host, double tol, int max_iter,
                       std::array<int, Columns>& iterations,
                       std::array<double, Columns>& residuals, bool laplacian,
                       const Input* x0) {
        static_assert(Columns == 1 || Columns == 2 || Columns == 4);
        if (std::uint64_t(n_) > std::numeric_limits<unsigned>::max() / Columns)
            throw std::overflow_error("GPU RHS panel exceeds packed index range");
        if (!ready_) throw std::runtime_error("cuda_pcg::solve: setup() not called");
        if (b_host.rows() != n_ || b_host.cols() != Columns ||
            x_host.rows() != n_ || x_host.cols() != Columns ||
            (x0 && (x0->rows() != n_ || x0->cols() != Columns)))
            throw std::invalid_argument("cuda_pcg::solve: dimensions mismatch");
        const detail::cuda_device_scope device(device_, detail::cuda_device_scope::checked{});
        reserve_columns(Columns);
        const auto& perm = op_->perm;
        using host_panel = Eigen::Matrix<double, Eigen::Dynamic, Columns,
            Columns == 1 ? Eigen::ColMajor : Eigen::RowMajor>;
        host_panel packed(n_, Columns);
        std::array<double, Columns> scales{}, bnorm{}, max_b{}, max_x0{};
        bool any_active = false, any_warm = false;
        std::fill_n(h_state_, Columns, cuda_pcg_kernels::solve_state{});
        for (int64_t v = 0; v < n_; ++v) {
            for (unsigned q = 0; q < Columns; ++q) {
                const double rhs = b_host(v, q);
                packed(perm[v], q) = rhs;
                max_b[q] = std::max(max_b[q], std::abs(rhs));
                if (x0) max_x0[q] = std::max(max_x0[q], std::abs((*x0)(v, q)));
            }
        }
        for (double maximum : max_b) any_active |= maximum != 0;
        if (!any_active) {
            x_host.setZero();
            iterations.fill(0);
            residuals.fill(0);
            return {};
        }
        const std::size_t vector_bytes = n_ * Columns * sizeof(double);
        APXCHOL_PCG_CUDA_CHECK(cudaMemcpyAsync(vectors_.rhs, packed.data(), vector_bytes, cudaMemcpyHostToDevice, stream_));
        for (unsigned q = 0; q < Columns; ++q) {
            scales[q] = 1;
            bnorm[q] = 1; // Inactive zero columns never divide by a zero norm.
            if (max_b[q] == 0) continue;
            scales[q] = detail::gpu_scaling::fp32_scale(std::max(max_b[q], max_x0[q]));
            // packed stays unchanged until the stream syncs: the upload above
            // may still read it. Fold the scaled column's norm contiguously.
            const Eigen::VectorXd rhs_column = packed.col(q) * scales[q];
            bnorm[q] = detail::gpu_scaling::norm(rhs_column);
            if (!std::isnormal(bnorm[q] * bnorm[q]))
                throw std::invalid_argument("CUDA solve: RHS and initial guess exceed the common scaling range");
            auto& st = h_state_[q];
            st.thr = tol;
            st.active = 1;
            if (max_x0[q] != 0) { st.alpha = 1; any_warm = true; }
        }
        cuda_pcg_kernels::initialize<Columns>(stream_, n_, vectors_, scales);
        APXCHOL_PCG_CUDA_CHECK(cudaMemcpyAsync(vectors_.state, h_state_, Columns * sizeof(*vectors_.state), cudaMemcpyHostToDevice, stream_));
        // Width changes reinterpret the reused storage; clear every grounded
        // coordinate on entry rather than inheriting a prior width's contents.
        if (laplacian && n_)
            APXCHOL_PCG_CUDA_CHECK(cudaMemsetAsync(vectors_.z + (n_ - 1) * Columns, 0, Columns * sizeof(double), stream_));
        iteration<Precond, Columns> commands{*this, precond, bnorm};
        host_panel warm;
        if (any_warm) {
            warm.resize(n_, Columns);
            for (int64_t v = 0; v < n_; ++v)
                for (unsigned q = 0; q < Columns; ++q)
                    warm(perm[v], q) = h_state_[q].alpha ? (*x0)(v, q) * scales[q] : 0;
            if (laplacian)
                for (unsigned q = 0; q < Columns; ++q)
                    if (h_state_[q].alpha) warm.col(q).array() -= warm.col(q).mean();
            APXCHOL_PCG_CUDA_CHECK(cudaMemcpyAsync(vectors_.p, warm.data(), vector_bytes, cudaMemcpyHostToDevice, stream_));
            // alpha=1 only for warm columns; cold p=0 and alpha=0. Their
            // iteration-zero convergence check is suppressed by finalize.
            commands.step(detail::gpu_pcg::operation::multiply);
            commands.step(detail::gpu_pcg::operation::residual);
            commands.finalize(detail::gpu_pcg::kFinalizeCheck, 0);
        }
        detail::gpu_pcg::solve(commands, std::max(max_iter, 0));
        cuda_pcg_kernels::finish<Columns>(stream_, n_, vectors_, op_->matrix, spmv_lanes_, laplacian, scales, bnorm);
        APXCHOL_PCG_CUDA_CHECK(cudaMemcpyAsync(h_state_, vectors_.state,
            Columns * sizeof(*vectors_.state), cudaMemcpyDeviceToHost, stream_));
        APXCHOL_PCG_CUDA_CHECK(cudaMemcpyAsync(packed.data(), vectors_.x, vector_bytes, cudaMemcpyDeviceToHost, stream_));
        APXCHOL_PCG_CUDA_CHECK(cudaStreamSynchronize(stream_));
        for (unsigned q = 0; q < Columns; ++q) {
            if (h_state_[q].active & cuda_pcg_kernels::kSolutionOverflow)
                throw std::overflow_error("CUDA solve: solution exceeds the finite range");
            if (h_state_[q].active & cuda_pcg_kernels::kSolutionUnderflow)
                throw std::underflow_error("CUDA solve: solution underflows to zero");
        }
        for (int64_t v = 0; v < n_; ++v)
            for (unsigned q = 0; q < Columns; ++q)
                x_host(v, q) = packed(perm[v], q);
        std::array<bool, Columns> stopped{};
        for (unsigned q = 0; q < Columns; ++q) {
            iterations[q] = h_state_[q].iters;
            residuals[q] = h_state_[q].rr;
            stopped[q] = h_state_[q].tolerance_stop != 0;
        }
        return stopped;
    }

    template<class Precond, unsigned Columns>
    struct iteration {
        cuda_pcg& s;
        const Precond& precond;
        std::array<double, Columns> bnorm;
        void precondition() const {
            static_assert(sizeof(cuda_pcg_kernels::solve_state) % sizeof(unsigned) == 0);
            precond.trsv().template solve_LLt_dev<Columns>(s.vectors_.r, s.vectors_.z,
                &s.vectors_.state->active, sizeof(cuda_pcg_kernels::solve_state) / sizeof(unsigned),
                s.stream_);
        }
        void finalize(unsigned mode, unsigned it) const {
            const int lanes = mode == detail::gpu_pcg::kFinalizeAlpha ? s.spmv_lanes_ : 1;
            const int blocks = cuda_pcg_kernels::pcg_blocks(s.n_ * lanes);
            cuda_pcg_kernels::finalize<Columns>(s.stream_, s.vectors_.partial, blocks, s.vectors_.state, mode, it, bnorm);
        }
        void step(detail::gpu_pcg::operation op) const {
            cuda_pcg_kernels::step<Columns>(s.stream_, op, s.n_, s.vectors_, s.op_->matrix, s.spmv_lanes_);
        }
        template<class Record>
        bool submit(bool, Record&& record) const {
            record();
            APXCHOL_PCG_CUDA_CHECK(cudaMemcpyAsync(s.h_state_, s.vectors_.state, Columns * sizeof(*s.vectors_.state), cudaMemcpyDeviceToHost, s.stream_));
            APXCHOL_PCG_CUDA_CHECK(cudaStreamSynchronize(s.stream_));
            bool active = false;
            for (unsigned q = 0; q < Columns; ++q) active |= s.h_state_[q].active != 0;
            return active;
        }
    };

    static void release_workspace(cuda_pcg_kernels::workspace& vectors, cuda_pcg_kernels::solve_state*& host) noexcept {
        for (void* pointer : std::initializer_list<void*>{vectors.x, vectors.r, vectors.ap,
                vectors.p, vectors.z, vectors.rhs, vectors.partial, vectors.state})
            if (pointer) (void)cudaFree(pointer);
        vectors = {};
        if (host) { (void)cudaFreeHost(host); host = nullptr; }
    }

    void reserve_columns(unsigned columns) {
        if (columns <= capacity_) return;
        cuda_pcg_kernels::workspace next{};
        cuda_pcg_kernels::solve_state* next_host = nullptr;
        try {
            for (double** pointer : {&next.x, &next.r, &next.p, &next.z, &next.ap, &next.rhs})
                APXCHOL_PCG_CUDA_CHECK(cudaMalloc(pointer, columns * n_ * sizeof(double)));
            APXCHOL_PCG_CUDA_CHECK(cudaMalloc(&next.partial, columns * cuda_pcg_kernels::kMaxBlocks * sizeof(double)));
            APXCHOL_PCG_CUDA_CHECK(cudaMalloc(&next.state, columns * sizeof(cuda_pcg_kernels::solve_state)));
            APXCHOL_PCG_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&next_host),
                columns * sizeof(cuda_pcg_kernels::solve_state), cudaHostAllocDefault));
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

    void destroy() noexcept {
        // A failed setup may own arrays before ready_ becomes true.
        const detail::cuda_device_scope device(device_, device_ >= 0);
        release_workspace(vectors_, h_state_);
        if (stream_) (void)cudaStreamDestroy(stream_);
        stream_ = nullptr;
        op_.reset();
        capacity_ = 0;
        spmv_lanes_ = 1;
        ready_ = false;
        device_ = -1;
    }

    // Shared by copies and freed by the last one: the permuted operator and
    // the host-side permutation map (perm.indices()) that permutes
    // b -> b_perm and x_perm -> x.
    struct operator_storage {
        int device = -1;
        cuda_pcg_kernels::operator_csr matrix;
        std::vector<node_index> perm;
        operator_storage() = default;
        operator_storage(const operator_storage&) = delete;
        operator_storage& operator=(const operator_storage&) = delete;
        ~operator_storage() {
            const detail::cuda_device_scope scope(device, device >= 0);
            for (void* pointer : std::initializer_list<void*>{matrix.row_ptr, matrix.col_idx,
                    matrix.values_f64, matrix.values_f32, matrix.row_map})
                if (pointer) (void)cudaFree(pointer);
        }
    };

    int64_t n_ = 0;
    int device_ = -1;
    bool ready_ = false;
    unsigned capacity_ = 0;
    std::shared_ptr<const operator_storage> op_;
    // Per-copy: the solve stream, vectors and host-visible exit state.
    cudaStream_t stream_ = nullptr;
    cuda_pcg_kernels::workspace vectors_{};
    cuda_pcg_kernels::solve_state* h_state_ = nullptr;
    // SpMV row mapping (threads per row).
    int     spmv_lanes_  = 1;

};

#undef APXCHOL_PCG_CUDA_CHECK

}  // namespace apxchol::detail
