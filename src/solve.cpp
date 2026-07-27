#include "apxchol/solver/solve.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(APXCHOL_USE_CUDA)
#include "apxchol/solver/pcg_cuda.h"
#endif

namespace apxchol {

namespace {
// Eigen's parallel SpMV requires Eigen::initParallel() to be called
// once per process before any threaded operation.  Calling it from a
// function-local static keeps it lazy and thread-safe.
inline void ensure_eigen_parallel() {
    static const bool dummy = []{ Eigen::initParallel(); return true; }();
    (void)dummy;
}

// One-time build-flag report straight from the LIBRARY TU: the width is
// sizeof() of the type solve.cpp actually compiled, so 4 == this .so/.a was
// built with -DAPXCHOL_SPTRSV_FP32, 8 == fp64. Reports whichever SpTRSV
// backend (CPU omp / GPU cuda) this build compiled. Opt-in via
// APXCHOL_VERBOSE — a library should be silent on stderr by default.
inline void print_sptrsv_banner_once() {
    static const bool printed = [] {
        if (!std::getenv("APXCHOL_VERBOSE")) return true;
#if defined(APXCHOL_USE_CUDA)
        const char* backend = "GPU/cuSPARSE";
        const char* vname   = apxchol::cuda_sptrsv::value_name;
        const std::size_t vbytes = apxchol::cuda_sptrsv::value_bytes;
#else
        const char* backend = "CPU/omp";
        const char* vname   = apxchol::omp_sptrsv::value_name;
        const std::size_t vbytes = apxchol::omp_sptrsv::value_bytes;
#endif
        std::fprintf(stderr, "[apxchol] SpTRSV (%s) factor values: %s, %zu bytes/elem"
#ifdef APXCHOL_SPTRSV_FP32
                             " (APXCHOL_SPTRSV_FP32 defined)\n",
#else
                             " (APXCHOL_SPTRSV_FP32 NOT defined)\n",
#endif
                     backend, vname, vbytes);
        return true;
    }();
    (void)printed;
}

// Parallel symmetric SpMV: y = L * x where L is stored as a row-major
// FULL symmetric matrix (NOT just lower triangle). Each thread owns a
// row range and computes y[i] = Σ_k val[k] * x[col[k]] without races.
//
// Required because Eigen's selfadjointView * x and plain (L * x) for
// sparse L are SEQUENTIAL — neither path uses Eigen::nbThreads() for
// the SpMV kernel even when EIGEN_HAS_OPENMP is defined. Empirical
// on a 7.9M-nnz LP-IPM matrix:
//   Eigen selfadjoint SpMV:   6.4 ms @ any T  (10 GB/s)
//   This parallel CSR SpMV:   2.4 ms @ T=16   (26 GB/s, 2.7x faster)
// True iff every operator value round-trips fp32 (v == double(float(v))), so the
// operator A can be stored fp32 losslessly. O(nnz) parallel scan (a few ms even at
// 24M nnz -- negligible vs the multi-second setup); mirrors the GPU operator check.
inline bool operator_is_fp32_exact(const Eigen::SparseMatrix<double>& L) {
    const double* v = L.valuePtr();
    const Eigen::Index nnz = L.nonZeros();
    bool exact = true;
    #pragma omp parallel for schedule(static) reduction(&&:exact)
    for (Eigen::Index p = 0; p < nnz; ++p)
        if (static_cast<double>(static_cast<float>(v[p])) != v[p]) exact = false;
    return exact;
}

// y = Lrm * x. Templated on the operator's stored scalar S: when A is fp32-exact we
// hold Lrm as SparseMatrix<float> (half the value bytes), and S(float) * x(double)
// promotes to double so the accumulation -- and the PCG recurrence -- stay fp64.
template<class S>
inline void parallel_spmv_csr(const Eigen::SparseMatrix<S, Eigen::RowMajor>& Lrm,
                              const Eigen::VectorXd& x, Eigen::VectorXd& y) {
    const int n = static_cast<int>(Lrm.rows());
    const auto* outer = Lrm.outerIndexPtr();
    const auto* inner = Lrm.innerIndexPtr();
    const S*    val   = Lrm.valuePtr();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        // 4-way accumulator split: breaks the serial FMA dep chain into 4
        // independent chains so the OoO core can issue ~4 FMAs/cycle
        // overlapped with the gather loads x[inner[k]]. Compiler doesn't
        // auto-unroll FP reductions (addition not associative); the rounding
        // -order change here is accepted.
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        const int row_start = outer[i];
        const int row_end   = outer[i + 1];
        int k = row_start;
        for (; k + 4 <= row_end; k += 4) {
            s0 += val[k + 0] * x[inner[k + 0]];
            s1 += val[k + 1] * x[inner[k + 1]];
            s2 += val[k + 2] * x[inner[k + 2]];
            s3 += val[k + 3] * x[inner[k + 3]];
        }
        double sum = (s0 + s1) + (s2 + s3);
        for (; k < row_end; ++k)
            sum += val[k] * x[inner[k]];
        y[i] = sum;
    }
}

} // namespace

// ── cpu_solver: factor + operator built once, PCG-solve many b ──────────────────
// Checkpoint placement here defines the setup/solve timing split used by the
// benchmark suite; keep the operation order and checkpoint labels stable.

cpu_solver::cpu_solver(const Eigen::SparseMatrix<double>& L,
                       const solve_options& opts, checkpoint* cp)
    : opts_(opts), n_(L.rows()) {
    ensure_eigen_parallel();
    print_sptrsv_banner_once();

    // Build preconditioner.
    precond_.set_options(opts_.factor_opts);
    precond_.set_storage(opts_.storage);
    precond_.set_keep_factor(opts_.keep_factor_values);
    if (cp) precond_.set_checkpoint(cp);
    const bool _pcg_trace = std::getenv("APXCHOL_PCG_TRACE") != nullptr;
    const auto _t_compute_start = std::chrono::high_resolution_clock::now();
    precond_.compute(L);
    if (_pcg_trace) {
        const double dt = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - _t_compute_start).count();
        const double dt_factorize = cp ? cp->total("setup") : 0.0;
        std::fprintf(stderr, "[pcg] precond.compute wall=%.0f ms (factorize=%.0f ms, sptrsv-setup+misc=%.0f ms)\n",
                     dt*1000, dt_factorize*1000, (dt - dt_factorize)*1000);
    }
    build_operator(L, cp);
}

cpu_solver::cpu_solver(const Eigen::SparseMatrix<double>& L,
                       factorization F,
                       const solve_options& opts, checkpoint* cp)
    : opts_(opts), n_(L.rows()) {
    ensure_eigen_parallel();
    print_sptrsv_banner_once();

    if (static_cast<Eigen::Index>(F.L.rows()) != n_)
        throw std::invalid_argument("cpu_solver: factorization dimension mismatch");
    if (F.L.nonZeros() > 0 && F.L.vals_.empty())
        throw std::invalid_argument(
            "cpu_solver: factorization values were released; pass a freshly "
            "computed factorization");
    precond_.set_keep_factor(opts_.keep_factor_values);
    if (cp) precond_.set_checkpoint(cp);
    precond_.set_factor(std::move(F));
    build_operator(L, cp);
}

void cpu_solver::build_operator(const Eigen::SparseMatrix<double>& L,
                                checkpoint* cp) {
    // Build a row-major FULL symmetric copy of L for our parallel SpMV.
    // L is the lower-triangle CSC; selfadjointView reflects across the
    // diagonal. We convert once before PCG starts so per-iter SpMV is
    // just a CSR row-scan.
    const bool pcg_trace_outer = std::getenv("APXCHOL_PCG_TRACE") != nullptr;
    {
        // Lrm conversion is one-time SETUP work for the parallel SpMV;
        // checkpoint it under "setup" so bench's setup_time/solve_time
        // split is accurate.
        const auto t0 = std::chrono::high_resolution_clock::now();
        if (cp) { cp->descend("setup"); cp->tick(); }
        // Manual parallel selfadjoint(Lower) → full symmetric CSR build.
        // Eigen's `Lrm = L.selfadjointView<Lower>()` triggers a serial
        // conversion (~45 ms on IPM iter40). We walk L's CSC twice in
        // parallel: PASS 1 counts row_nnz contributions, PASS 2 scatters.
        // Memory cost: O(n) — one shared row_nnz array (also reused as
        // row_pos for scatter cursor), no per-thread histograms.
        const Eigen::Index n_eig = L.rows();
        {
        const int* L_outer = L.outerIndexPtr();  // CSC col ptrs, n_eig+1
        const int* L_inner = L.innerIndexPtr();  // row indices
        const double* L_vals = L.valuePtr();
        std::vector<int> row_nnz(static_cast<size_t>(n_eig), 0);
        // PASS 1: per column k, each LOWER-tri entry (row >= k) contributes
        // +1 to row_nnz[row] (original lower) and, if row > k, +1 to row_nnz[k]
        // (transpose mirror). selfadjointView<Lower> ignores upper entries
        // (row < k) so we must filter to row >= k — the input L may store
        // either only the lower triangle or both halves.
        #pragma omp parallel for schedule(static)
        for (Eigen::Index k = 0; k < n_eig; ++k) {
            for (int p = L_outer[k]; p < L_outer[k + 1]; ++p) {
                const int row = L_inner[p];
                if (row < k) continue;  // upper-tri entry, ignore
                __atomic_fetch_add(&row_nnz[row], 1, __ATOMIC_RELAXED);
                if (row != k)
                    __atomic_fetch_add(&row_nnz[k], 1, __ATOMIC_RELAXED);
            }
        }
        Lrm_.resize(n_eig, n_eig);
        // Build row_ptr (cumulative). Eigen's outerIndexPtr is the row_ptr in
        // RowMajor mode. Serial prefix sum, ~ms.
        int* Lrm_outer = Lrm_.outerIndexPtr();
        Lrm_outer[0] = 0;
        for (Eigen::Index i = 0; i < n_eig; ++i)
            Lrm_outer[i + 1] = Lrm_outer[i] + row_nnz[i];
        const int total_nnz = Lrm_outer[n_eig];
        Lrm_.resizeNonZeros(total_nnz);
        int* Lrm_inner = Lrm_.innerIndexPtr();
        double* Lrm_vals = Lrm_.valuePtr();
        // Per-row write cursor (atomic claim).
        std::vector<int> row_pos(static_cast<size_t>(n_eig));
        std::copy(Lrm_outer, Lrm_outer + n_eig, row_pos.begin());
        // PASS 2: scatter entries to rows. Same filter as PASS 1 — skip
        // upper-tri (row < k) since selfadjointView<Lower> ignores them.
        #pragma omp parallel for schedule(static)
        for (Eigen::Index k = 0; k < n_eig; ++k) {
            for (int p = L_outer[k]; p < L_outer[k + 1]; ++p) {
                const int row = L_inner[p];
                if (row < k) continue;
                const double v = L_vals[p];
                // Lower entry: M(row, k) = v
                const int slot = __atomic_fetch_add(&row_pos[row], 1, __ATOMIC_RELAXED);
                Lrm_inner[slot] = static_cast<int>(k);
                Lrm_vals[slot]  = v;
                if (row != k) {
                    // Upper transpose: M(k, row) = v
                    const int slot2 = __atomic_fetch_add(&row_pos[k], 1, __ATOMIC_RELAXED);
                    Lrm_inner[slot2] = row;
                    Lrm_vals[slot2]  = v;
                }
            }
        }
        // Sort each row's columns in ascending order (Eigen expects sorted CSR).
        // Per-thread reused kv buffer — avoids n_eig tiny allocations that
        // dominated the previous version on grid_2000 (n=4M → 4M mallocs).
        #pragma omp parallel
        {
            std::vector<std::pair<int, double>> kv;
            #pragma omp for schedule(static)
            for (Eigen::Index i = 0; i < n_eig; ++i) {
                const int rs = Lrm_outer[i], re = Lrm_outer[i + 1];
                if (re - rs < 2) continue;
                kv.clear();
                kv.reserve(re - rs);
                for (int p = rs; p < re; ++p)
                    kv.emplace_back(Lrm_inner[p], Lrm_vals[p]);
                std::sort(kv.begin(), kv.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });
                for (int p = rs; p < re; ++p) {
                    Lrm_inner[p] = kv[p - rs].first;
                    Lrm_vals[p]  = kv[p - rs].second;
                }
            }
        }
        Lrm_.makeCompressed();
        if (cp) { (*cp)("spmv_lrm_build"); cp->ascend(); }
        if (pcg_trace_outer) {
            const auto dt = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t0).count();
            std::fprintf(stderr, "[pcg] Lrm conversion: %.0f ms\n", dt*1000);
        }
        } // end parallel-build block
    }

    // Operator A storage precision: fp32 when every value round-trips fp32 (lossless),
    // else fp64. fp32 halves Lrm's value array for the whole solve and feeds the same
    // fp32-load/fp64-accumulate SpMV; the Krylov recurrence stays fp64, so the 1e-8
    // floor is preserved (same lever as the GPU operator). APXCHOL_FP32_OPERATOR
    // overrides: "0" forces fp64, any other value forces fp32 (may floor if inexact).
    { const char* e = std::getenv("APXCHOL_FP32_OPERATOR");
      if (e && std::string(e) == "0")   op_fp32_ = false;
      else if (e && *e != '\0')         op_fp32_ = true;
      else                              op_fp32_ = operator_is_fp32_exact(L); }
    if (op_fp32_) {
        Lrm_f_ = Lrm_.cast<float>();
        Lrm_ = Eigen::SparseMatrix<double, Eigen::RowMajor>();   // free the fp64 copy (steady-state fp32)
    }

    // Memory breakdown of the major live arrays just before PCG (env-gated).
    // The graph pool is already freed here; what remains is the factor held
    // THREE times (F_.L + SpTRSV CSR + SpTRSV CSC) + Lrm (full-symmetric SpMV
    // copy of the input) + the input + PCG vectors. VmHWM/VmRSS from /proc.
    if (std::getenv("APXCHOL_MEM_BREAKDOWN")) {
        const double MB = 1.0 / (1024.0 * 1024.0);
        auto proc_kb = [](const char* key) -> long {
            std::ifstream f("/proc/self/status"); std::string ln;
            const std::size_t klen = std::char_traits<char>::length(key);
            while (std::getline(f, ln))
                if (ln.rfind(key, 0) == 0) return std::stol(ln.substr(klen));
            return -1;
        };
        const long in_nnz  = static_cast<long>(L.nonZeros());
        const long fac_nnz = static_cast<long>(precond_.factor().L.nonZeros());
        const long lrm_nnz = static_cast<long>(op_fp32_ ? Lrm_f_.nonZeros() : Lrm_.nonZeros());
        const long N       = static_cast<long>(L.rows());
        std::fprintf(stderr,
            "[breakdown] before PCG (factor held 3x; graph pool already freed):\n"
            "  input Eigen L  : nnz=%-11ld ~%6.0f MB\n"
            "  factor F_.L    : nnz=%-11ld ~%6.0f MB\n"
            "  SpTRSV CSR+CSC : ~2x factor       ~%6.0f MB\n"
            "  Lrm (SpMV,%s): nnz=%-11ld ~%6.0f MB\n"
            "  PCG vectors    : 6 x n=%-9ld ~%6.0f MB\n"
            "  >>> VmRSS now=%.0f MB   VmHWM(peak)=%.0f MB\n",
            in_nnz,  in_nnz  * 12.0 * MB,
            fac_nnz, fac_nnz * 12.0 * MB,
            fac_nnz * 12.0 * 2 * MB,
            op_fp32_ ? "fp32" : "fp64", lrm_nnz, lrm_nnz * (op_fp32_ ? 8.0 : 12.0) * MB,
            N, N * 8.0 * 6 * MB,
            proc_kb("VmRSS:") / 1024.0, proc_kb("VmHWM:") / 1024.0);
    }
}

void cpu_solver::solve_impl(const Eigen::VectorXd& b, Eigen::Ref<Eigen::VectorXd> x,
                            solve_result& res, double tol, int max_iter,
                            const Eigen::VectorXd* x0) const {
    if (tol < 0.0)    tol = opts_.tol;
    if (max_iter < 0) max_iter = opts_.max_iter;

    // Preconditioned CG with stagnation detection.
    const Eigen::Index n = n_;
    const double bnorm = b.norm();
    if (bnorm == 0.0) { x.setZero(); return; }

    if (x0 != nullptr && x0->size() != n)
        throw std::invalid_argument("cpu_solver::solve: x0 length mismatch");

    // Reused workspace: repeated solves allocate nothing after the first call.
    if (r_.size() != n) { r_.resize(n); z_.resize(n); p_.resize(n); Ap_.resize(n); }
    Eigen::VectorXd& r = r_;
    Eigen::VectorXd& z = z_;
    Eigen::VectorXd& p = p_;
    Eigen::VectorXd& Ap = Ap_;
    if (x0 != nullptr && !x0->isZero(0.0)) {
        x = *x0;
        if (op_fp32_) parallel_spmv_csr(Lrm_f_, x, r);
        else          parallel_spmv_csr(Lrm_,   x, r);
        r = b - r;                                 // r = b - L*x0
        res.iterations = 0;
        res.residual = r.norm() / bnorm;
        if (res.residual < tol) return;
    } else {
        x.setZero();
        r = b;                                     // r = b - L*0 = b
        // Honest pre-loop state: the relative residual of x = 0 is exactly 1.
        // Without this, an exit before the first PCG update (max_iter = 0, or
        // a pAp <= 0 breakdown on iteration 1) would report the field's 0.0
        // default — i.e. claim convergence for a solve that never ran.
        res.iterations = 0;
        res.residual = 1.0;
    }

    // Descend into "pcg" so all per-iter PCG stages — including precond.solve()
    // (which internally descends into "solve" → forward/back/permute) — are
    // grouped under pcg.*. Without this, the bench's wall solve_time (wall -
    // setup) included un-checkpointed PCG ops (SpMV/dots/axpys/norm) that
    // dominated the gap between checkpoint "solve" (444 ms) and bench's wall
    // solve (~1.1 s on IPM iter40 T=16).
    const bool use_cp = !std::getenv("APXCHOL_NO_CHECKPOINT");
    if (use_cp) { res.timings.descend("pcg"); res.timings.tick(); }

    z = precond_.solve(r);                 // recorded as pcg.solve.{forward,back,…}
    p = z;
    double rz = r.dot(z);
    if (use_cp) res.timings("dot_init");

    double prev_check_residual = 1.0;
    const int check_interval = opts_.stagnation_window;

    // Per-component timing (set APXCHOL_PCG_TRACE=1).
    const bool pcg_trace = std::getenv("APXCHOL_PCG_TRACE") != nullptr;
    double t_spmv = 0, t_dots = 0, t_axpys = 0, t_norm = 0, t_precond = 0;
    auto now_us = []() {
        return std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    double t_pcg_start = pcg_trace ? now_us() : 0;

    for (int i = 0; i < max_iter; ++i) {
        double t0 = pcg_trace ? now_us() : 0;
        if (op_fp32_) parallel_spmv_csr(Lrm_f_, p, Ap);   // fp32 operator, fp64 accumulate
        else          parallel_spmv_csr(Lrm_,   p, Ap);
        if (pcg_trace) t_spmv += now_us() - t0;
        if (use_cp) res.timings("spmv");
        if (pcg_trace) t0 = now_us();
        double pAp = p.dot(Ap);
        if (pcg_trace) t_dots += now_us() - t0;
        if (use_cp) res.timings("dot_pAp");
        if (pAp <= 0.0) break;
        double alpha = rz / pAp;

        double t0a = pcg_trace ? now_us() : 0;
        x += alpha * p;
        r -= alpha * Ap;
        if (pcg_trace) t_axpys += now_us() - t0a;
        if (use_cp) res.timings("axpy_xr");
        double t0n = pcg_trace ? now_us() : 0;
        double rnorm = r.norm() / bnorm;
        if (pcg_trace) t_norm += now_us() - t0n;
        if (use_cp) res.timings("norm");
        res.iterations = i + 1;
        res.residual = rnorm;

        if (rnorm < tol) break;

        // Stagnation detection: if residual hasn't improved sufficiently
        // over the last check_interval iterations, stop early.
        if (check_interval > 0 && (i + 1) % check_interval == 0) {
            if (rnorm > prev_check_residual * 0.5) break;
            prev_check_residual = rnorm;
        }

        double t0p = pcg_trace ? now_us() : 0;
        z = precond_.solve(r);                 // recorded as pcg.solve.{forward,back,…}
        if (pcg_trace) t_precond += now_us() - t0p;

        double t0d = pcg_trace ? now_us() : 0;
        double rz_new = r.dot(z);
        if (pcg_trace) t_dots += now_us() - t0d;
        if (use_cp) res.timings("dot_rz");
        double beta = rz_new / rz;
        double t0a2 = pcg_trace ? now_us() : 0;
        p = z + beta * p;
        if (pcg_trace) t_axpys += now_us() - t0a2;
        if (use_cp) res.timings("axpy_p");
        rz = rz_new;
    }
    if (use_cp) res.timings.ascend();

    if (pcg_trace) {
        const double total_us = now_us() - t_pcg_start;
        const int nit = res.iterations;
        std::fprintf(stderr,
            "[pcg] iters=%d total=%.0fms\n"
            "      spmv=%.0fms (%.1fms/iter)\n"
            "      precond.solve=%.0fms (%.1fms/iter)\n"
            "      dots=%.0fms (%.1fms/iter)\n"
            "      axpys=%.0fms (%.1fms/iter)\n"
            "      norm=%.0fms (%.1fms/iter)\n"
            "      unaccounted=%.0fms\n",
            nit, total_us/1000,
            t_spmv/1000, t_spmv/1000/nit,
            t_precond/1000, t_precond/1000/nit,
            t_dots/1000, t_dots/1000/nit,
            t_axpys/1000, t_axpys/1000/nit,
            t_norm/1000, t_norm/1000/nit,
            (total_us - t_spmv - t_precond - t_dots - t_axpys - t_norm)/1000);
    }
}

void cpu_solver::solve(const Eigen::VectorXd& b, solve_result& res,
                       double tol, int max_iter,
                       const Eigen::VectorXd* x0) const {
    res.x.resize(n_);
    solve_impl(b, res.x, res, tol, max_iter, x0);
}

solve_result cpu_solver::solve(const Eigen::VectorXd& b, double tol, int max_iter,
                               const Eigen::VectorXd* x0) const {
    solve_result res;
    solve(b, res, tol, max_iter, x0);
    return res;
}

solve_result cpu_solver::solve(const Eigen::VectorXd& b, Eigen::Ref<Eigen::VectorXd> x,
                               double tol, int max_iter,
                               const Eigen::VectorXd* x0) const {
    if (x.size() != n_)
        throw std::invalid_argument("cpu_solver::solve: output x length mismatch");
    solve_result res;
    solve_impl(b, x, res, tol, max_iter, x0);
    return res;
}

// ── one-shot solve ──────────────────────────────────────────────────────────────
solve_result solve(const Eigen::SparseMatrix<double>& L,
                   const Eigen::VectorXd& b,
                   const solve_options& opts) {
    ensure_eigen_parallel();
    print_sptrsv_banner_once();

#if defined(APXCHOL_USE_CUDA)
    // GPU-resident ("native") PCG: keep A + all 5 PCG vectors on device, use
    // cuBLAS axpy/dot/nrm2 + cuSPARSE SpMV, so nothing crosses the bus per
    // iteration. (A host-PCG-with-GPU-SpTRSV path is transfer-bound and
    // pointless; construct a cpu_solver directly if you really want it.)
    {
        apx_cholesky precond;
        solve_result res;
        precond.set_options(opts.factor_opts);
        precond.set_storage(opts.storage);
        if (!std::getenv("APXCHOL_NO_CHECKPOINT"))
            precond.set_checkpoint(&res.timings);
        precond.compute(L);

        const bool use_cp = !std::getenv("APXCHOL_NO_CHECKPOINT");
        if (use_cp) { res.timings.descend("setup"); res.timings.tick(); }
        cuda_pcg gpcg;
        gpcg.setup(L, precond.factor().perm);
        if (use_cp) { res.timings("gpu_pcg_setup"); res.timings.ascend(); }

        if (use_cp) { res.timings.descend("pcg"); res.timings.tick(); }
        int iters = 0;
        double rnorm = 1.0;
        gpcg.solve(precond, b, res.x,
                   opts.tol, opts.max_iter,
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
        return res;
    }
#else
    solve_result res;
    const cpu_solver slv(L, opts,
                         std::getenv("APXCHOL_NO_CHECKPOINT") ? nullptr : &res.timings);
    slv.solve(b, res);
    return res;
#endif
}

Eigen::VectorXd generate_test_rhs(Eigen::Index n) {
    Eigen::VectorXd b = Eigen::VectorXd::Random(n);
    b.array() -= b.mean();
    return b.normalized();
}

} // namespace apxchol
