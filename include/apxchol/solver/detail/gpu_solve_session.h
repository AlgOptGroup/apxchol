#pragma once
// Internal shared lifetime for the public one-shot GPU solve and the benchmark's
// native-solve retries. No change to GPU kernels, storage or native stop test.
#include "apxchol/solver/solve.h"
#include "apxchol/solver/pcg_cuda.h"

namespace apxchol::detail {
class gpu_solve_session {
    apx_cholesky precond;
    cuda_pcg gpcg;
    const bool use_cp = !std::getenv("APXCHOL_NO_CHECKPOINT");
public:
    gpu_solve_session(const Eigen::SparseMatrix<double>& L,
                      const solve_options& opts, solve_result& res) {
        precond.set_options(opts.factor_opts);
        precond.set_storage(opts.storage);
        if (!std::getenv("APXCHOL_NO_CHECKPOINT"))
            precond.set_checkpoint(&res.timings);
        precond.compute(L);
        // Lumping (if any) happened inside compute(), on a private copy; the
        // GPU operator below is built from L itself.
        res.lumped_offdiag = precond.factor().lumped_offdiag;

        if (use_cp) { res.timings.descend("setup"); res.timings.tick(); }
        const auto operator_setup_begin = detail::gpu_setup_diagnostic_clock::now();
        gpcg.setup(L, precond.factor().perm);
        if (use_cp) { res.timings("gpu_pcg_setup"); res.timings.ascend(); }
        if (detail::gpu_setup_diagnostics()) {
            const double operator_setup_wall_s = std::chrono::duration<double>(
                detail::gpu_setup_diagnostic_clock::now() - operator_setup_begin).count();
            // Host API interval, disjoint from factorize/install. The device
            // builder completes inside setup; host fallback retains its prior
            // behavior. Common benchmark timing adds its own completion boundary.
            std::fprintf(stderr,
                "[gpu-operator-setup-receipt] diagnostics=%s operator_setup_wall_s=%.17g\n",
                detail::gpu_setup_diagnostics() ? "enabled" : "disabled", operator_setup_wall_s);
        }
    }
    void solve(const Eigen::VectorXd& b, solve_result& res, double tol, int max_iter) {
        if (use_cp) { res.timings.descend("pcg"); res.timings.tick(); }
        int iters = 0;
        double rnorm = 1.0;
        gpcg.solve(precond, b, res.x,
                   tol, max_iter,
                   iters, rnorm,
                   /*laplacian=*/ !precond.factor().sddm);
        if (use_cp) { res.timings("gpu_pcg_loop"); res.timings.ascend(); }
        res.iterations = iters;
        res.residual = rnorm;
        // Solve-held device VRAM (gpcg's operator+vectors + the SpTRSV factor are
        // still resident here; everything is freed when gpcg/precond destruct).
        { size_t mf = 0, mt = 0;
          if (cudaMemGetInfo(&mf, &mt) == cudaSuccess && mt >= mf)
              res.solve_vram_mb = (mt - mf) / (1024.0 * 1024.0); }
    }
};
} // namespace apxchol::detail
