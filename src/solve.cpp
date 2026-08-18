#include "apxchol/solver/solve.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
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
#if defined(__x86_64__) || defined(__i386__)
#include <pmmintrin.h>   // _MM_SET_DENORMALS_ZERO_MODE (DAZ)
#include <xmmintrin.h>   // _MM_SET_FLUSH_ZERO_MODE    (FTZ)
#endif

namespace apxchol {

namespace {

// APXCHOL_FTZ=1 (env, read at every PCG entry): set the x86 MXCSR FTZ bit
// (results that would be subnormal become zero) and DAZ bit (subnormal INPUTS
// are treated as zero) on the calling thread and on every thread of the
// OpenMP team. MXCSR is per-thread state, so it is set inside a parallel region
// -- libgomp keeps its pooled worker threads across regions, so the SpMV /
// SpTRSV / axpy regions of the PCG loop then run on threads that have it set
// -- and once more on the master (Eigen's dots / norms run there). Opt-in;
// the default leaves MXCSR alone. Sticky for the threads' lifetime (not
// restored on return; every subsequent parallel region in the process
// inherits it). x86 only: a no-op elsewhere. Numerics: subnormals in the
// factor / operator / PCG vectors are treated as zero -- the factor census
// (omp_sptrsv::lowprec_stats().factor_subnormal, APXCHOL_VERBOSE) says whether
// there are any to matter.
inline void maybe_enable_ftz_daz() {
    const char* e = std::getenv("APXCHOL_FTZ");
    if (!(e && std::atoi(e) != 0)) return;
#if defined(__x86_64__) || defined(__i386__)
    auto set_ftz_daz = [] {
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    };
    int nt = 1;
    #pragma omp parallel
    {
        set_ftz_daz();
        #pragma omp single
        {
#ifdef _OPENMP
            nt = omp_get_num_threads();
#endif
        }
    }
    set_ftz_daz();   // the master (also a team member above; harmless twice)
    static const bool printed = [nt] {
        if (std::getenv("APXCHOL_VERBOSE"))
            std::fprintf(stderr, "[apxchol] APXCHOL_FTZ=1: MXCSR FTZ+DAZ set on the master and on the"
                                 " %d-thread OpenMP team at PCG entry (sticky)\n", nt);
        return true;
    }();
    (void)printed;
#else
    static const bool warned = [] {
        std::fprintf(stderr, "[apxchol] APXCHOL_FTZ=1 ignored: not an x86 build\n");
        return true;
    }();
    (void)warned;
#endif
}
// Eigen's parallel SpMV requires Eigen::initParallel() to be called
// once per process before any threaded operation.  Calling it from a
// function-local static keeps it lazy and thread-safe.
inline void ensure_eigen_parallel() {
    static const bool dummy = []{ Eigen::initParallel(); return true; }();
    (void)dummy;
}

// One-time build-flag report straight from the LIBRARY TU: the width is
// sizeof() of the type solve.cpp actually compiled, so 2 == this .so/.a was
// built with APXCHOL_SPTRSV_LOWPREC = FP16_SCALED, 4 == -DAPXCHOL_SPTRSV_FP32,
// 8 == fp64. Reports whichever SpTRSV
// backend (CPU omp / GPU cuda) this build compiled. Opt-in via
// APXCHOL_VERBOSE — a library should be silent on stderr by default.
inline void print_sptrsv_banner_once() {
    static const bool printed = [] {
        if (!std::getenv("APXCHOL_VERBOSE")) return true;
#if defined(APXCHOL_USE_CUDA)
        // Runtime backend / storage modes of the GPU SpTRSV (env, resolved
        // per setup by cuda_sptrsv; the banner is one-shot, so it reports the
        // value at first solve): APXCHOL_GPU_SPTRSV=dataflow|cusparse|levelset
        // (unset = AUTO: the dataflow backend on the fp32 build; on fp64
        // cuSPARSE with the level-set fallback where cuSPARSE is compiled in
        // -- CMake APXCHOL_CUDA_WITH_CUSPARSE -- else the level-set) and the
        // kernel backends' opt-in fp16 storage APXCHOL_GPU_SPTRSV_FP16=1.
        const bool gpu_fp16 = apxchol::cuda_sptrsv::fp16_from_env();
        const int  gpu_be   = apxchol::cuda_sptrsv::backend_from_env();
        const bool gpu_cus  = apxchol::cuda_sptrsv::cusparse_available();
        const char* backend = gpu_be == 2 ? "GPU/dataflow (APXCHOL_GPU_SPTRSV=dataflow)"
                            : gpu_be > 0 ? "GPU/levelset (APXCHOL_GPU_SPTRSV=levelset)"
                            : gpu_be < 0 ? "GPU/cuSPARSE (APXCHOL_GPU_SPTRSV=cusparse)"
                            : apxchol::cuda_sptrsv::auto_prefers_dataflow()
                                ? (gpu_cus ? "GPU/auto: dataflow (fp32 build default; APXCHOL_GPU_SPTRSV=cusparse|levelset overrides)"
                                           : "GPU/auto: dataflow (fp32 build default; APXCHOL_GPU_SPTRSV=levelset overrides; cuSPARSE not compiled in)")
                                : gpu_fp16 ? "GPU/auto: level-set (fp16 storage on the fp64 build)"
                                : gpu_cus  ? "GPU/auto: cuSPARSE, level-set if its SpSV analysis buffers do not fit (fp64 build)"
                                           : "GPU/auto: level-set (fp64 build, cuSPARSE not compiled in)";
        const char* vname   = gpu_fp16 ? "fp16 (per-column scaled, diagonal fp32; APXCHOL_GPU_SPTRSV_FP16=1)"
                                       : apxchol::cuda_sptrsv::value_name;
        const std::size_t vbytes = gpu_fp16 ? 2 : apxchol::cuda_sptrsv::value_bytes;
#else
        const char* backend = "CPU/omp";
        const char* vname   = apxchol::omp_sptrsv::value_name;
        const std::size_t vbytes = apxchol::omp_sptrsv::value_bytes;
#endif
        std::fprintf(stderr, "[apxchol] SpTRSV (%s) factor values: %s, %zu bytes/elem"
#if defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED)
                             " (APXCHOL_SPTRSV_LOWPREC=%s; off-diagonals only, diagonal fp32; rounding=RNE)\n",
#elif defined(APXCHOL_SPTRSV_FP32)
                             " (APXCHOL_SPTRSV_FP32 defined)\n",
#else
                             " (APXCHOL_SPTRSV_FP32 NOT defined)\n",
#endif
                     backend, vname, vbytes
#if defined(APXCHOL_SPTRSV_LOWPREC_FP16_SCALED)
                     , apxchol::omp_sptrsv::lowprec_variant
#endif
                     );
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

// ── Fused PCG vector kernels ─────────────────────────────────────────────────
// The outer PCG loop used to be a chain of plain Eigen expressions -- p.dot(Ap),
// x += alpha*p, r -= alpha*Ap, r.norm(), r.dot(z), p = z + beta*p -- every one
// of them a separate single-threaded full-n pass, ~14 n-vector streams per
// iteration outside the SpMV/SpTRSV kernels. The kernels below fuse them so
// each vector is streamed ONCE per group of operations, and run the passes
// OpenMP-parallel with DETERMINISTIC reductions: per-thread partials over the
// fixed schedule(static) chunk partition, summed serially in thread order
// (detail::static_chunk / omp_ids / reduce_parts in preconditioner.h; never a
// reduction() clause). Bit-identical run-to-run for a fixed thread count.
using detail::omp_ids;

// y = Lrm * x, and returns x·y (the PCG pAp) folded into the row loop:
// row i's dot product is finished right there, so the p·Ap reduction costs
// one extra FMA per row instead of a second 2-stream pass over (p, Ap).
// Templated on the operator's stored scalar S: when A is fp32-exact we hold
// Lrm as SparseMatrix<float> (half the value bytes), and S(float) * x(double)
// promotes to double so the accumulation -- and the PCG recurrence -- stay
// fp64. `part` = per-thread partial buffer (detail::part_capacity() doubles).
template<class S>
inline double parallel_spmv_csr(const Eigen::SparseMatrix<S, Eigen::RowMajor>& Lrm,
                                const double* xp, double* yp, double* part) {
    const Eigen::Index n = Lrm.rows();
    const auto* outer = Lrm.outerIndexPtr();
    const auto* inner = Lrm.innerIndexPtr();
    const S*    val   = Lrm.valuePtr();
    int nt_used = 1;
    #pragma omp parallel if(n > detail::kFusedOmpMin)
    {
        int tid, nt; omp_ids(tid, nt);
        if (tid == 0) nt_used = nt;
        const auto [lo, hi] = detail::static_chunk(n, tid, nt);
        double xy = 0.0;
        for (Eigen::Index i = lo; i < hi; ++i) {
            // 4-way accumulator split: breaks the serial FMA dep chain into 4
            // independent chains so the OoO core can issue ~4 FMAs/cycle
            // overlapped with the gather loads x[inner[k]]. Compiler doesn't
            // auto-unroll FP reductions (addition not associative); the rounding
            // -order change here is accepted.
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            const auto row_start = outer[i];
            const auto row_end   = outer[i + 1];
            auto k = row_start;
            for (; k + 4 <= row_end; k += 4) {
                s0 += val[k + 0] * xp[inner[k + 0]];
                s1 += val[k + 1] * xp[inner[k + 1]];
                s2 += val[k + 2] * xp[inner[k + 2]];
                s3 += val[k + 3] * xp[inner[k + 3]];
            }
            double sum = (s0 + s1) + (s2 + s3);
            for (; k < row_end; ++k)
                sum += val[k] * xp[inner[k]];
            yp[i] = sum;
            xy += xp[i] * sum;
        }
        part[static_cast<std::size_t>(tid) * detail::kPartStride] = xy;
    }
    return detail::reduce_parts(part, nt_used);
}

// One pass over (x, p, r, Ap):  x += alpha*p ; r -= alpha*Ap ; returns
// rr = r·r and rs = Σ r (both on the UPDATED r). rr feeds the residual norm,
// rs feeds the Laplacian centering of the next preconditioner application
// (apx_cholesky::apply_fused takes it, saving its own mean pass over r).
// Two independent 4-way accumulator sets so neither reduction chain stalls
// the streaming updates.
inline void update_xr(double* x, const double* p, double* r, const double* Ap,
                      double alpha, Eigen::Index n, double* part,
                      double& rr, double& rs) {
    int nt_used = 1;
    #pragma omp parallel if(n > detail::kFusedOmpMin)
    {
        int tid, nt; omp_ids(tid, nt);
        if (tid == 0) nt_used = nt;
        const auto [lo, hi] = detail::static_chunk(n, tid, nt);
        double q0 = 0.0, q1 = 0.0, q2 = 0.0, q3 = 0.0;   // r·r
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;   // Σ r
        Eigen::Index i = lo;
        for (; i + 4 <= hi; i += 4) {
            x[i + 0] += alpha * p[i + 0];
            x[i + 1] += alpha * p[i + 1];
            x[i + 2] += alpha * p[i + 2];
            x[i + 3] += alpha * p[i + 3];
            const double r0 = r[i + 0] - alpha * Ap[i + 0];
            const double r1 = r[i + 1] - alpha * Ap[i + 1];
            const double r2 = r[i + 2] - alpha * Ap[i + 2];
            const double r3 = r[i + 3] - alpha * Ap[i + 3];
            r[i + 0] = r0; r[i + 1] = r1; r[i + 2] = r2; r[i + 3] = r3;
            q0 += r0 * r0; q1 += r1 * r1; q2 += r2 * r2; q3 += r3 * r3;
            s0 += r0;      s1 += r1;      s2 += r2;      s3 += r3;
        }
        double q = (q0 + q1) + (q2 + q3);
        double s = (s0 + s1) + (s2 + s3);
        for (; i < hi; ++i) {
            x[i] += alpha * p[i];
            const double ri = r[i] - alpha * Ap[i];
            r[i] = ri;
            q += ri * ri;
            s += ri;
        }
        part[static_cast<std::size_t>(tid) * detail::kPartStride + 0] = q;
        part[static_cast<std::size_t>(tid) * detail::kPartStride + 1] = s;
    }
    rr = detail::reduce_parts(part, nt_used, 0);
    rs = detail::reduce_parts(part, nt_used, 1);
}

// p = z + beta*p in one parallel pass (no reduction).
inline void update_p(double* p, const double* z, double beta, Eigen::Index n) {
    #pragma omp parallel for schedule(static) if(n > detail::kFusedOmpMin)
    for (Eigen::Index i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
}

// Initial residual in one pass: r = b - Ax0 (Ax0 == nullptr -> r = b),
// returning rr = r·r and rs = Σ r. Same deterministic scheme.
inline void init_residual(double* r, const double* b, const double* Ax0,
                          Eigen::Index n, double* part, double& rr, double& rs) {
    int nt_used = 1;
    #pragma omp parallel if(n > detail::kFusedOmpMin)
    {
        int tid, nt; omp_ids(tid, nt);
        if (tid == 0) nt_used = nt;
        const auto [lo, hi] = detail::static_chunk(n, tid, nt);
        double q0 = 0.0, q1 = 0.0, q2 = 0.0, q3 = 0.0;
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        Eigen::Index i = lo;
        if (Ax0) {
            for (; i + 4 <= hi; i += 4) {
                const double r0 = b[i + 0] - Ax0[i + 0];
                const double r1 = b[i + 1] - Ax0[i + 1];
                const double r2 = b[i + 2] - Ax0[i + 2];
                const double r3 = b[i + 3] - Ax0[i + 3];
                r[i + 0] = r0; r[i + 1] = r1; r[i + 2] = r2; r[i + 3] = r3;
                q0 += r0 * r0; q1 += r1 * r1; q2 += r2 * r2; q3 += r3 * r3;
                s0 += r0;      s1 += r1;      s2 += r2;      s3 += r3;
            }
        } else {
            for (; i + 4 <= hi; i += 4) {
                const double r0 = b[i + 0], r1 = b[i + 1], r2 = b[i + 2], r3 = b[i + 3];
                r[i + 0] = r0; r[i + 1] = r1; r[i + 2] = r2; r[i + 3] = r3;
                q0 += r0 * r0; q1 += r1 * r1; q2 += r2 * r2; q3 += r3 * r3;
                s0 += r0;      s1 += r1;      s2 += r2;      s3 += r3;
            }
        }
        double q = (q0 + q1) + (q2 + q3);
        double s = (s0 + s1) + (s2 + s3);
        for (; i < hi; ++i) {
            const double ri = Ax0 ? b[i] - Ax0[i] : b[i];
            r[i] = ri;
            q += ri * ri;
            s += ri;
        }
        part[static_cast<std::size_t>(tid) * detail::kPartStride + 0] = q;
        part[static_cast<std::size_t>(tid) * detail::kPartStride + 1] = s;
    }
    rr = detail::reduce_parts(part, nt_used, 0);
    rs = detail::reduce_parts(part, nt_used, 1);
}

// x -= mean(x): the once-per-solve min-norm centring of a Laplacian
// solution (see cpu_solver::solve_impl). One deterministic reduction
// (detail::det_sum) + one parallel subtract pass.
inline void center_x(double* x, Eigen::Index n, double* part) {
    const double mean = detail::det_sum(x, n, part) / static_cast<double>(n);
    #pragma omp parallel for schedule(static) if(n > detail::kFusedOmpMin)
    for (Eigen::Index i = 0; i < n; ++i) x[i] -= mean;
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
        // Free the fp64 copy (steady-state fp32). Swap-with-empty: assigning an
        // empty SparseMatrix only resets the sizes and KEEPS the value/index
        // buffers allocated (Eigen's CompressedStorage never shrinks), i.e.
        // `Lrm_ = SparseMatrix()` left the full fp64 operator resident.
        Eigen::SparseMatrix<double, Eigen::RowMajor>().swap(Lrm_);
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
    maybe_enable_ftz_daz();   // opt-in APXCHOL_FTZ=1, per-thread MXCSR (see above)

    // Preconditioned CG with stagnation detection.
    const Eigen::Index n = n_;

    if (x0 != nullptr && x0->size() != n)
        throw std::invalid_argument("cpu_solver::solve: x0 length mismatch");

    // Reused workspace: repeated solves allocate nothing after the first call
    // (part_ holds the per-thread partials of the deterministic reductions;
    // sized to the current max team, which the caller may change between
    // solves via omp_set_num_threads).
    if (r_.size() != n) { r_.resize(n); z_.resize(n); p_.resize(n); Ap_.resize(n); }
    { const std::size_t need = detail::part_capacity();
      if (part_.size() < need) part_.resize(need); }
    Eigen::VectorXd& r = r_;
    Eigen::VectorXd& z = z_;
    Eigen::VectorXd& p = p_;
    Eigen::VectorXd& Ap = Ap_;
    double* part = part_.data();

    // Laplacian (rank n-1) case: the returned x is the MIN-NORM solution --
    // mean(x) = 0 -- for every exit below (SDDM: full rank, nothing to do).
    // Under the center-k schedule (APXCHOL_GROUND=center-k, the default) most
    // preconditioner applications skip their output re-centring, so x
    // accumulates a component along the null space 1 (invisible to the
    // residual, since A·1 = 0); one deterministic mean subtraction at the end
    // (center_x) restores exactly what per-application centring (K = 1) and
    // a zero start would have produced. A warm start x0 gets the same
    // treatment, so the answer never depends on x0's constant.
    const bool laplacian = !precond_.factor().sddm;

    // Every vector pass of the loop below is a fused, OpenMP-parallel kernel
    // (see the anonymous namespace at the top of this file) whose reductions
    // are deterministic for a fixed thread count. Scalars carried between
    // passes: rr = r·r (norm), rs = Σ r (Laplacian centering inside the
    // preconditioner), rz = r·z, pAp = p·Ap.
    double bnorm, rr, rs;
    if (x0 != nullptr && !x0->isZero(0.0)) {
        bnorm = b.norm();
        if (bnorm == 0.0) { x.setZero(); return; }
        x = *x0;
        // r = b - L*x0: SpMV into Ap (scratch here), then one pass forms r
        // and its reductions. The SpMV's fused x·Ax0 is not needed.
        if (op_fp32_) (void)parallel_spmv_csr(Lrm_f_, x.data(), Ap.data(), part);
        else          (void)parallel_spmv_csr(Lrm_,   x.data(), Ap.data(), part);
        init_residual(r.data(), b.data(), Ap.data(), n, part, rr, rs);
        res.iterations = 0;
        res.residual = std::sqrt(rr) / bnorm;
        if (res.residual < tol) {
            if (laplacian) center_x(x.data(), n, part);
            return;
        }
    } else {
        // r = b - L*0 = b, copied in the same pass that produces b·b (= bnorm²)
        // and Σ b.
        init_residual(r.data(), b.data(), nullptr, n, part, rr, rs);
        bnorm = std::sqrt(rr);
        if (bnorm == 0.0) { x.setZero(); return; }
        x.setZero();
        // Honest pre-loop state: the relative residual of x = 0 is exactly 1.
        // Without this, an exit before the first PCG update (max_iter = 0, or
        // a pAp <= 0 breakdown on iteration 1) would report the field's 0.0
        // default — i.e. claim convergence for a solve that never ran.
        res.iterations = 0;
        res.residual = 1.0;
    }

    // Descend into "pcg" so all per-iter PCG stages — including the
    // preconditioner application (which internally descends into "solve" →
    // permute/forward/back/unpermute+rz) — are grouped under pcg.*. Without
    // this, the bench's wall solve_time (wall - setup) included
    // un-checkpointed PCG ops that dominated the gap between checkpoint
    // "solve" and bench's wall solve.
    const bool use_cp = !std::getenv("APXCHOL_NO_CHECKPOINT");
    if (use_cp) { res.timings.descend("pcg"); res.timings.tick(); }

    // New solve: restart the center-k application counter (APXCHOL_GROUND=
    // center-k, see env_knobs.h) so the centring schedule -- every K-th
    // preconditioner application centres -- is the same for every solve on
    // this factor (repeated solves stay bit-identical).
    precond_.reset_apply_count();
    // z = M^{-1} r with r·z fused into the pass that writes z (recorded as
    // pcg.solve.{permute,forward,back,unpermute+rz}); p = z is one copy pass.
    double rz = precond_.apply_fused(r.data(), rs, z.data());
    p = z;
    if (use_cp) res.timings("copy_p");

    double prev_check_residual = 1.0;
    const int check_interval = opts_.stagnation_window;

    // Per-component timing (set APXCHOL_PCG_TRACE=1). Buckets follow the
    // fused kernels: spmv (+pAp), precond (+rz), update_xr (+norm), update_p.
    const bool pcg_trace = std::getenv("APXCHOL_PCG_TRACE") != nullptr;
    double t_spmv = 0, t_update_xr = 0, t_update_p = 0, t_precond = 0;
    auto now_us = []() {
        return std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    double t_pcg_start = pcg_trace ? now_us() : 0;

    for (int i = 0; i < max_iter; ++i) {
        // Ap = L*p with pAp = p·Ap folded into the row loop.
        double t0 = pcg_trace ? now_us() : 0;
        double pAp;
        if (op_fp32_) pAp = parallel_spmv_csr(Lrm_f_, p.data(), Ap.data(), part);   // fp32 operator, fp64 accumulate
        else          pAp = parallel_spmv_csr(Lrm_,   p.data(), Ap.data(), part);
        if (pcg_trace) t_spmv += now_us() - t0;
        if (use_cp) res.timings("spmv+pAp");
        if (pAp <= 0.0) break;
        double alpha = rz / pAp;

        // x += alpha*p ; r -= alpha*Ap ; rr = r·r ; rs = Σ r  -- one pass.
        double t0a = pcg_trace ? now_us() : 0;
        update_xr(x.data(), p.data(), r.data(), Ap.data(), alpha, n, part, rr, rs);
        if (pcg_trace) t_update_xr += now_us() - t0a;
        if (use_cp) res.timings("update_xr_norm");
        double rnorm = std::sqrt(rr) / bnorm;
        res.iterations = i + 1;
        res.residual = rnorm;

        if (rnorm < tol) break;

        // Stagnation detection: if residual hasn't improved sufficiently
        // over the last check_interval iterations, stop early.
        if (check_interval > 0 && (i + 1) % check_interval == 0) {
            if (rnorm > prev_check_residual * 0.5) break;
            prev_check_residual = rnorm;
        }

        // z = M^{-1} r, rz_new = r·z fused into the pass that writes z
        // (recorded as pcg.solve.{permute,forward,back,unpermute+rz}).
        double t0p = pcg_trace ? now_us() : 0;
        double rz_new = precond_.apply_fused(r.data(), rs, z.data());
        if (pcg_trace) t_precond += now_us() - t0p;

        double beta = rz_new / rz;
        double t0a2 = pcg_trace ? now_us() : 0;
        update_p(p.data(), z.data(), beta, n);      // p = z + beta*p
        if (pcg_trace) t_update_p += now_us() - t0a2;
        if (use_cp) res.timings("update_p");
        rz = rz_new;
    }
    // Min-norm solution for a Laplacian (see the note above the loop).
    if (laplacian) {
        center_x(x.data(), n, part);
        if (use_cp) res.timings("center_x");
    }
    if (use_cp) res.timings.ascend();

    if (pcg_trace) {
        const double total_us = now_us() - t_pcg_start;
        const int nit = res.iterations;
        std::fprintf(stderr,
            "[pcg] iters=%d total=%.0fms\n"
            "      spmv+pAp=%.0fms (%.1fms/iter)\n"
            "      precond+rz=%.0fms (%.1fms/iter)\n"
            "      update_xr+norm=%.0fms (%.1fms/iter)\n"
            "      update_p=%.0fms (%.1fms/iter)\n"
            "      unaccounted=%.0fms\n",
            nit, total_us/1000,
            t_spmv/1000, t_spmv/1000/nit,
            t_precond/1000, t_precond/1000/nit,
            t_update_xr/1000, t_update_xr/1000/nit,
            t_update_p/1000, t_update_p/1000/nit,
            (total_us - t_spmv - t_precond - t_update_xr - t_update_p)/1000);
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
    // GPU-resident ("native") PCG: keep A + all 5 PCG vectors on device and
    // run our own SpMV / fused vector kernels with deterministic reductions
    // (pcg_cuda.h; no cuSPARSE / cuBLAS), so nothing but three 8-byte scalars
    // per iteration crosses the bus. (A host-PCG-with-GPU-SpTRSV path is
    // transfer-bound and pointless; construct a cpu_solver directly if you
    // really want it.)
    {
        apx_cholesky precond;
        solve_result res;
        precond.set_options(opts.factor_opts);
        precond.set_storage(opts.storage);
        if (!std::getenv("APXCHOL_NO_CHECKPOINT"))
            precond.set_checkpoint(&res.timings);
        // The GPU SpTRSV's AUTO backend decision (cuda.h: cuSPARSE unless its
        // O(nnz) SpSV analysis buffers do not fit) must leave room for what
        // cuda_pcg::setup allocates AFTER it: the full-symmetric operator
        // (nnz_full = 2*(entries with row >= col) - n; int32 col idx + fp64
        // values -- an upper bound, fp32-exact operators store fp32) + row
        // pointers + the 6 fp64 PCG vectors.
        {
            const Eigen::Index n = L.rows();
            const int* Lo = L.outerIndexPtr();
            const int* Li = L.innerIndexPtr();
            std::int64_t lower = 0;
            #pragma omp parallel for schedule(static) reduction(+ : lower)
            for (Eigen::Index k = 0; k < n; ++k)
                for (int p = Lo[k]; p < Lo[k + 1]; ++p) lower += Li[p] >= k;
            const std::int64_t nnz_full = 2 * lower - n;
            precond.trsv().set_reserve_bytes(
                static_cast<std::size_t>(nnz_full) * (sizeof(int) + sizeof(double)) +
                static_cast<std::size_t>(n + 1) * sizeof(int) +
                6 * static_cast<std::size_t>(n) * sizeof(double));
        }
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
