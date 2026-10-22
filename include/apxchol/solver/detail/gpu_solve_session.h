#pragma once
// Shared GPU setup lifetime for reusable solves and benchmark retries.
#include "apxchol/solver/solve.h"
#include "apxchol/solver/pcg_cuda.h"
#include "apxchol/solver/detail/solve_backend.h"

namespace apxchol::detail {
class gpu_solve_session {
    gpu_preconditioner precond;
    cuda_pcg gpcg;
    const bool use_cp;
    const int stagnation_window;
public:
    gpu_solve_session(const Eigen::SparseMatrix<double>& L,
                      const solve_options& opts, solve_result& res)
        : gpu_solve_session(L, opts, &res.timings) {
        res.lumped_offdiag = precond.factor().lumped_offdiag;
    }
    gpu_solve_session(const Eigen::SparseMatrix<double>& L,
                      const solve_options& opts, checkpoint* cp)
        : use_cp(opts.collect_timings), stagnation_window(opts.stagnation_window) {
        validate_solve_options(opts);
        if (opts.backend == solve_backend::cpu)
            throw std::invalid_argument("GPU session cannot execute an explicit CPU request");
        auto required = opts;
        required.backend = solve_backend::gpu;
        (void)select_solve_backend(required);
        precond.set_options(opts.factor_opts);
        precond.set_keep_factor(opts.keep_factor_values);
        if (use_cp) precond.set_checkpoint(cp);
        precond.compute(L);
        precond.set_checkpoint(nullptr);
        if (!precond.trsv().adopted_device_factor())
            throw std::runtime_error("GPU route did not produce a device-owned factor");
        // Lumping (if any) happened inside compute(), on a private copy; the
        // GPU operator below is built from L itself.
        if (cp && use_cp) { cp->descend("setup"); cp->tick(); }
        const auto operator_setup_begin = detail::gpu_setup_diagnostic_clock::now(opts.factor_opts.verbose);
        gpcg.setup(L, precond.factor().perm, opts.factor_opts.verbose);
        if (cp && use_cp) { (*cp)("gpu_pcg_setup"); cp->ascend(); }
        if (opts.factor_opts.verbose) {
            const double operator_setup_wall_s = std::chrono::duration<double>(
                detail::gpu_setup_diagnostic_clock::now(opts.factor_opts.verbose) - operator_setup_begin).count();
            // Host API interval, disjoint from factorize/install. The device
            // builder completes inside setup. Common benchmark timing adds its
            // own completion boundary.
            std::fprintf(stderr,
                "[gpu-operator-setup-receipt] operator_setup_wall_s=%.17g\n",
                operator_setup_wall_s);
        }
    }
    const gpu_preconditioner& preconditioner() const { return precond; }

    void solve(const Eigen::VectorXd& b, solve_result& res, double tol, int max_iter,
               const Eigen::VectorXd* x0 = nullptr) {
        validate_solve_parameters(tol, max_iter);
        if (b.size() != precond.rows())
            throw std::invalid_argument("GPU solve: RHS length mismatch");
        if (x0 && (x0->size() != b.size() || !x0->allFinite()))
            throw std::invalid_argument("GPU solve: x0 must be finite and match the RHS length");
        res.backend = solve_backend::gpu;
        res.lumped_offdiag = precond.factor().lumped_offdiag;
        if (use_cp) { res.timings.descend("pcg"); res.timings.tick(); }
        int iters = 0;
        double rnorm = 1.0;
        gpcg.solve(precond, b, res.x,
                   tol, max_iter,
                   iters, rnorm,
                   /*laplacian=*/ !precond.factor().sddm, x0, stagnation_window);
        if (use_cp) { res.timings("gpu_pcg_loop"); res.timings.ascend(); }
        res.iterations = iters;
        res.residual = rnorm;
        res.converged = rnorm < tol;
        // Solve-held device VRAM (gpcg's operator+vectors + the SpTRSV factor are
        // still resident here; everything is freed when gpcg/precond destruct).
        { size_t mf = 0, mt = 0;
          if (cudaMemGetInfo(&mf, &mt) == cudaSuccess && mt >= mf)
              res.solve_vram_mb = (mt - mf) / (1024.0 * 1024.0); }
    }

    batch_solve_result solve_panel(const_matrix_view B, matrix_view X, double tol,
                                 int max_iter, const const_matrix_view* X0 = nullptr) {
        validate_solve_parameters(tol, max_iter);
        if (B.rows() != precond.rows() || (B.cols() != 2 && B.cols() != 4) ||
            X.rows() != B.rows() || X.cols() != B.cols() ||
            (X0 && (X0->rows() != B.rows() || X0->cols() != B.cols())))
            throw std::invalid_argument("GPU solve panel: dimensions mismatch");
        batch_solve_result result;
        result.columns.resize(B.cols());
        if (use_cp) { result.timings.descend("pcg"); result.timings.tick(); }
        const auto run = [&]<unsigned Columns>() {
            std::array<int, Columns> iterations{};
            std::array<double, Columns> residuals{};
            gpcg.solve_panel<Columns>(precond, B, X, tol, max_iter, iterations, residuals,
                                     !precond.factor().sddm, X0, stagnation_window);
            for (unsigned q = 0; q < Columns; ++q) {
                result.columns[q].iterations = iterations[q];
                result.columns[q].residual = residuals[q];
                result.columns[q].converged = residuals[q] < tol;
            }
        };
        if (B.cols() == 4) run.template operator()<4>();
        else run.template operator()<2>();
        if (use_cp) { result.timings("gpu_pcg_loop"); result.timings.ascend(); }
        double vram_mb = -1;
        size_t free = 0, total = 0;
        if (cudaMemGetInfo(&free, &total) == cudaSuccess && total >= free)
            vram_mb = (total - free) / (1024.0 * 1024.0);
        for (auto& column : result.columns) {
            column.backend = solve_backend::gpu;
            column.lumped_offdiag = precond.factor().lumped_offdiag;
            column.solve_vram_mb = vram_mb;
        }
        return result;
    }

};
} // namespace apxchol::detail
