#pragma once
// CUDA route of solver: device-owned factor, triangular solves and PCG.
#include "apxchol/solver/solve.h"
#include "apxchol/detail/checkpoint.h"
#include "apxchol/solver/cuda_context.h"
#include "apxchol/solver/pcg_cuda.h"
#include "apxchol/solver/detail/gpu_diagnostics.h"
#include "apxchol/solver/detail/pcg_reduction.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace apxchol::detail {
// solver validates options and arguments, selects the route and rejects
// overlapping calls; the route trusts its arguments. Copies share the
// factor and operator and own their device workspace and stream.
class cuda_route {
public:
    // Panel widths solve_columns and apply_columns accept, widest first.
    static constexpr std::array<int, 3> solve_widths{4, 2, 1};
    static constexpr std::array<int, 1> apply_widths{1};

    /// Factorizes A on the device, or uploads the host factor F when it is
    /// not null. cp receives setup timings and is not retained.
    cuda_route(const Eigen::SparseMatrix<double>& A, const solve_options& opts,
                      checkpoint* cp, factorization* F)
        : opts_(opts.factor_opts), keep_factor_(opts.keep_factor) {
        if (F) {
            if (factorization_access::device_factor(*F))
                throw std::invalid_argument("GPU solver: supply a host factor without a device capsule");
            if (!F->sddm && F->L.rows() == 1)
                throw std::invalid_argument("CUDA cannot adopt a scalar Laplacian factor");
            *F_ = std::move(*F);
            install_factor(cp);
        } else {
            factorize(A, cp);
        }
        // Lumping changes only the factorization; the GPU operator uses A itself.
        if (cp) { cp->descend("setup"); cp->tick(); }
        const auto operator_setup_begin = gpu_setup_diagnostic_clock::now(opts_.verbose);
        gpcg_.setup(A, F_->perm, opts_.verbose);
        if (cp) { (*cp)("gpu_pcg_setup"); cp->ascend(); }
        if (opts_.verbose) {
            const double operator_setup_wall_s = std::chrono::duration<double>(
                gpu_setup_diagnostic_clock::now(opts_.verbose) - operator_setup_begin).count();
            // Host API interval, disjoint from factorize/install. The device
            // builder completes inside setup. Common benchmark timing adds its
            // own completion boundary.
            std::fprintf(stderr,
                "[gpu-operator-setup-receipt] operator_setup_wall_s=%.17g\n",
                operator_setup_wall_s);
        }
    }

    /// Solves the columns of B, a panel of one of solve_widths, into X.
    /// timings, when not null, receives the PCG checkpoints.
    void solve_columns(const_matrix_view B, matrix_view X, const const_matrix_view* X0,
                       std::span<solve_result> results, checkpoint* timings,
                       double tol, int max_iter) {
        if (timings) { timings->descend("pcg"); timings->tick(); }
        const auto run = [&]<unsigned Columns>() {
            std::array<int, Columns> iterations{};
            std::array<double, Columns> residuals{};
            gpcg_.solve_panel<Columns>(*this, B, X, tol, max_iter, iterations, residuals,
                                       !F_->sddm, X0);
            for (unsigned q = 0; q < Columns; ++q) {
                results[q].iterations = iterations[q];
                results[q].residual = residuals[q];
                results[q].converged = residuals[q] < tol;
            }
        };
        if (B.cols() == 4) run.template operator()<4>();
        else if (B.cols() == 2) run.template operator()<2>();
        else run.template operator()<1>();
        if (timings) { (*timings)("gpu_pcg_loop"); timings->ascend(); }
        // Solve-held device VRAM (the operator, vectors and factor are still
        // resident here; shared arrays are freed with the last copy).
        const double vram_mb = gpcg_.device_memory_used_mb();
        for (auto& result : results) {
            result.backend = solve_backend::gpu;
            result.solve_vram_mb = vram_mb;
        }
    }

    /// z = M^{-1} r for one column, through host-side permutation and centering:
    /// Laplacian center → P → L^{-1} → L^{-T} → P^{-1} → center; SDDM skips centering.
    void apply_columns(const_matrix_view R, matrix_view Z) {
        const Eigen::Index n = R.rows();
        const Eigen::VectorXd b = R.col(0);
        Eigen::VectorXd z(n);
        const node_index* P = F_->perm.data();
        const bool center = !F_->sddm;
        const int workers = cpu_solve_threads(n);
        const std::size_t need = part_capacity();
        if (part_.size() < need) part_.resize(need);
        scratch_.resize(n);
        const double mean = center
            ? det_sum_columns<1>(b.data(), n, part_.data())[0] / static_cast<double>(n) : 0.0;
        #pragma omp parallel for schedule(static) if(workers > 1) num_threads(workers)
        for (Eigen::Index v = 0; v < n; ++v)
            z[P[v]] = b[v] - (center ? mean : 0.0);
        trsv_.solve_LLt(z.data(), scratch_.data(), gpcg_.stream());
        double* out = scratch_.data();
        if (!F_->sddm) out[n - 1] = 0.0;
        const double shift = center
            ? det_sum_columns<1>(out, n, part_.data())[0] / static_cast<double>(n) : 0.0;
        #pragma omp parallel for schedule(static) if(workers > 1) num_threads(workers)
        for (Eigen::Index v = 0; v < n; ++v)
            z[v] = out[P[v]] - shift;
        Z.col(0) = z;
    }

    /// Shares the factor, the device triangular-solve storage and the device
    /// operator; allocates this copy's device workspace and stream.
    cuda_route(const cuda_route& other)
        : opts_(other.opts_), keep_factor_(other.keep_factor_), F_(other.F_),
          trsv_(other.trsv_), gpcg_(other.gpcg_) {}

    const factorization& factor() const { return *F_; }
    factor_storage_type factor_storage() const {
        return trsv_.fp16() ? factor_storage_type::fp16 : factor_storage_type::fp32;
    }
    /// The device triangular solves; cuda_pcg calls solve_LLt_dev through it.
    const cuda_sptrsv& trsv() const { return trsv_; }
    const cuda_pcg& pcg() const { return gpcg_; }
    /// The original operator in FP64 CSC, downloaded from the device.
    Eigen::SparseMatrix<double> original_operator() const;

private:
    void factorize(const Eigen::SparseMatrix<double>& A, checkpoint* cp) {
        [[maybe_unused]] const auto factorize_install_begin =
            gpu_setup_diagnostic_clock::now(opts_.verbose);
        cuda_context::prewarm();
        *F_ = factorize_for_solver(A, opts_, cp, setup_route::gpu, keep_factor_);
        // Reject inadmissible factors before installation can initialize
        // CUDA or upload host arrays.
        const auto& device = factorization_access::device_factor(*F_);
        if (!device || device.use_count() != 1 || device->empty())
            throw std::invalid_argument("GPU route requires a unique nonempty device-owned factor");
        install_factor(cp);
        // Complete the adopted factor's queued setup work before returning.
        const auto setup_completion = cudaDeviceSynchronize();
        if (setup_completion != cudaSuccess)
            throw std::runtime_error(std::string("GPU setup completion: ") +
                                     cudaGetErrorString(setup_completion));
        if (cp) { cp->descend("setup"); (*cp)("gpu_setup_complete"); cp->ascend(); }
        if (opts_.verbose) {
            // Host API duration; device-adopted setup additionally completed its work.
            const double factorize_install_wall_s = std::chrono::duration<double>(
                gpu_setup_diagnostic_clock::now(opts_.verbose) - factorize_install_begin).count();
            const auto& drop = trsv_.drop_stats();
            std::fprintf(stderr,
                "[gpu-setup-receipt] factorize_install_wall_s=%.17g "
                "n=%lld rounds=%zu raw_factor_nnz=%zu stored_nnz=%llu adopted=%d "
                "fp16=%d host_factor_array_bytes=%zu adoption_download_bytes=%zu "
                "factor_drop_rel=%.17g dropped_threshold=%llu dropped_flush=%llu\n",
                factorize_install_wall_s, static_cast<long long>(F_->perm.size()),
                F_->rounds.size(), static_cast<std::size_t>(F_->L.nonZeros()),
                static_cast<unsigned long long>(trsv_.stored_nnz()),
                trsv_.adopted_device_factor() ? 1 : 0, trsv_.fp16() ? 1 : 0,
                F_->L.inner_.size() * sizeof(node_index) + F_->L.vals_.size() * sizeof(factor_value_t),
                trsv_.adoption_host_download_bytes(), drop.rel,
                static_cast<unsigned long long>(drop.dropped_threshold),
                static_cast<unsigned long long>(drop.dropped_flush));
        }
    }

    /// Device triangular-solve setup from F_: adopts its device capsule, or
    /// uploads its host arrays.
    void install_factor(checkpoint* cp) {
        // SpTRSV setup is setup work: record it under "setup" so the
        // setup/solve split stays complete.
        if (cp) { cp->descend("setup"); cp->tick(); }
        // Device init, on its OWN checkpoint label. CUDA creates the primary
        // context lazily inside the first CUDA call. factorize() prewarms it
        // on a helper thread; pay only the REMAINING wait here, under
        // "cuda_init", so sptrsv_setup below reports its own work only.
        const double cuda_init_s = cuda_context::ensure_context();
        if (cp) (*cp)("cuda_init");
        if (opts_.verbose)
            std::fprintf(stderr, "[sptrsv-setup gpu] %-22s %8.2f ms  (context creation %.2f ms)\n",
                         "cuda_init", cuda_init_s * 1e3, cuda_context::context_seconds() * 1e3);
        auto& device = factorization_access::device_factor(*F_);
        if (device) {
            trsv_.setup_adopting_device_factor(std::move(*device), opts_);
            device.reset();
        } else {
            const auto n = static_cast<node_index>(F_->L.rows());
            trsv_.setup(F_->L, F_->sddm ? n : n - 1, opts_);
        }
        if (cp) { (*cp)("sptrsv_setup"); cp->ascend(); }
        // The device holds its own copy; keep_factor=true keeps the host
        // row/value arrays for callers that export or adopt L.
        if (!keep_factor_) F_->L.release_values();
    }

    factor_options opts_;
    bool keep_factor_ = false;
    // Shared by copies; written only during construction.
    std::shared_ptr<factorization> F_ = std::make_shared<factorization>();
    cuda_sptrsv trsv_;
    cuda_pcg gpcg_;
    // Host workspace of apply_columns.
    Eigen::VectorXd scratch_;
    std::vector<double> part_;
};
} // namespace apxchol::detail
