#include "apxchol/solver/solve.h"
#include "apxchol/solver/detail/solve_backend.h"
#include "apxchol/solver/detail/thread_budget.h"
#include "apxchol/solver/detail/lower_triangle_csr.h"
#include "apxchol/csc_work.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <variant>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(APXCHOL_USE_CUDA)
#include "apxchol/solver/detail/gpu_solve_session.h"
#elif defined(APXCHOL_USE_METAL)
#include "apxchol/solver/metal_solver.h"
#include "apxchol/solver/detail/metal_schedule.h"
#endif

namespace apxchol {

namespace {

// (APXCHOL_FTZ=1 -- setting the x86 MXCSR FTZ + DAZ bits on the master and on
// every OpenMP team thread at PCG entry -- lived here until 2026-08-20. It
// was REFUTED end to end: no subnormal ever enters this computation, so the
// bits are a no-op that costs one extra fork-join per solve. Evidence, T=1
// so the whole pipeline is deterministic: FTZ on vs off gives a BYTE-IDENTICAL
// solution vector on iter0040 (same md5, residual 8.074599861229295e-09 to
// all 16 digits, 57 iterations) and identical iterations/residual on
// grid_2000 (49, 7.813e-09) -- a flushed or treated-as-zero subnormal would
// have to change a bit. The factor census agrees: factor_subnormal = 0 of
// 4.6M / 21.9M stored fp32 values on the two. See "Retired knobs" in
// AGENTS.md for the timing A/B.)
// Eigen's parallel SpMV requires Eigen::initParallel() to be called
// once per process before any threaded operation.  Calling it from a
// function-local static keeps it lazy and thread-safe.
inline void ensure_eigen_parallel() {
    static const bool dummy = []{ Eigen::initParallel(); return true; }();
    (void)dummy;
}

template<solve_backend Route>
void print_sptrsv_banner(bool fp16) {
    constexpr const char* backend = Route == solve_backend::gpu ? "GPU/dataflow" : "CPU/omp";
    if (std::getenv("APXCHOL_VERBOSE"))
        std::fprintf(stderr, "[apxchol] SpTRSV (%s) factor values: %s, %zu bytes/elem\n",
                     backend, fp16 ? "fp16 (per-column scaled, diagonal fp32)" : "float (fp32)",
                     fp16 ? std::size_t(2) : sizeof(sptrsv_value_t));
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
// One O(nnz) parallel storage scan resolves both direct-copy prerequisites.
// Besides the fp32 test, explicitly check the nondecreasing-inner-index
// property used by equal_range below. Eigen's normal insertion paths preserve
// it, but its low-level unordered compressed API does not, and makeCompressed()
// is a no-op for an already-compressed matrix.
struct operator_storage_properties {
    bool fp32_exact = true;
    bool inner_indices_sorted = true;
};

inline operator_storage_properties scan_operator_storage(
        const Eigen::SparseMatrix<double>& L) {
    const double* v = L.valuePtr();
    const int* outer = L.outerIndexPtr();
    const int* inner = L.innerIndexPtr();
    bool exact = true;
    bool sorted = true;
    #pragma omp parallel for schedule(static) reduction(&& : exact, sorted)
    for (Eigen::Index col = 0; col < L.outerSize(); ++col) {
        const int begin = outer[col];
        const int end = L.isCompressed()
            ? outer[col + 1]
            : begin + L.innerNonZeroPtr()[col];
        for (int p = begin; p < end; ++p) {
            if (static_cast<double>(static_cast<float>(v[p])) != v[p])
                exact = false;
            if (p > begin && inner[p - 1] > inner[p])
                sorted = false;
        }
    }
    return {exact, sorted};
}

// A compressed column-major matrix and a row-major matrix have the same three
// array layout when the sparsity pattern is symmetric: source column i is
// destination row i, and both keep inner indices sorted. Copy that layout
// directly while retaining the old selfadjointView<Lower>() value contract.
// In particular, an accepted matrix may differ from its transpose within the
// symmetry tolerance; an upper slot therefore takes the value of its lower
// partner instead of blindly copying the caller's upper value.
//
// The destination owns all three arrays. cpu_solver is reusable and may
// outlive (or be used after mutation of) the constructor's L, so a persistent
// Eigen::Map over the caller's storage would be a lifetime bug.
//
// Returns false when the full stored pattern is not symmetric. The caller then
// rebuilds from the lower triangle, preserving the custom-factor constructor's
// historical support for one-triangle operators. A successful return produces
// exactly the sorted CSR structure/value sequence of that general rebuild.
template<class S>
bool copy_symmetric_csc_as_owned_csr(
        const Eigen::SparseMatrix<double>& L,
        Eigen::SparseMatrix<S, Eigen::RowMajor>& Lrm,
        bool inner_indices_sorted) {
    if (!L.isCompressed() || L.rows() != L.cols() || !inner_indices_sorted)
        return false;

    const Eigen::Index n = L.rows();
    const Eigen::Index nnz = L.nonZeros();
    const int* src_outer = L.outerIndexPtr();
    const int* src_inner = L.innerIndexPtr();
    const double* src_vals = L.valuePtr();

    Lrm.resize(n, n);
    Lrm.resizeNonZeros(nnz);
    int* dst_outer = Lrm.outerIndexPtr();
    int* dst_inner = Lrm.innerIndexPtr();
    S* dst_vals = Lrm.valuePtr();

    // Fast path. The pairing loop below searches the transpose partner of every
    // upper entry so that the copy stores the canonical LOWER value: about
    // log2(column length) cache misses per entry, landing in the hub columns of
    // power-law and IPM operators (6.6 ns per stored entry on iter0040, 6.3 on
    // as-Skitter, against 1.1 on a grid). When the two triangles are bit-identical
    // that value is the entry itself. So: copy straight through while summing an
    // order-independent fingerprint of (unordered coordinates, position inside a
    // duplicate run, value bits) per triangle. Equal sums and counts prove the
    // plain copy is byte for byte what the pairing loop would have written --
    // duplicate runs in partner order, signed and explicit zeros included. Any
    // difference falls through to the pairing loop, which overwrites every entry.
    {
        std::uint64_t hash_lower = 0, hash_upper = 0;
        Eigen::Index direct_lower = 0, direct_upper = 0;
        #pragma omp parallel reduction(+ : hash_lower, hash_upper, direct_lower, direct_upper)
        {
            const auto [lo, hi] = detail::work_balanced_range(src_outer, n);
            for (Eigen::Index col = lo; col < hi; ++col) {
                dst_outer[col] = src_outer[col];
                const int col_i = static_cast<int>(col);
                int previous_row = -1;
                std::uint64_t run = 0;
                for (int p = src_outer[col]; p < src_outer[col + 1]; ++p) {
                    const int row = src_inner[p];
                    run = row == previous_row ? run + 1 : 0;
                    previous_row = row;
                    dst_inner[p] = row;
                    dst_vals[p] = static_cast<S>(src_vals[p]);
                    if (row < col_i) {
                        ++direct_upper;
                        hash_upper += detail::symmetric_entry_hash(
                            static_cast<std::uint64_t>(row),
                            static_cast<std::uint64_t>(col_i), run, src_vals[p]);
                    } else if (row > col_i) {
                        ++direct_lower;
                        hash_lower += detail::symmetric_entry_hash(
                            static_cast<std::uint64_t>(col_i),
                            static_cast<std::uint64_t>(row), run, src_vals[p]);
                    }
                }
            }
        }
        dst_outer[n] = src_outer[n];
        if (hash_lower == hash_upper && direct_lower == direct_upper) return true;
    }

    Eigen::Index lower_nnz = 0;
    Eigen::Index upper_nnz = 0;
    bool upper_is_paired = true;
    #pragma omp parallel for schedule(static) \
        reduction(+ : lower_nnz, upper_nnz) reduction(&& : upper_is_paired)
    for (Eigen::Index col = 0; col < n; ++col) {
        dst_outer[col] = src_outer[col];
        const int col_i = static_cast<int>(col);
        for (int p = src_outer[col]; p < src_outer[col + 1];) {
            const int row = src_inner[p];
            int run_end = p + 1;
            while (run_end < src_outer[col + 1] &&
                   src_inner[run_end] == row)
                ++run_end;
            const int run_size = run_end - p;

            for (int q = p; q < run_end; ++q) {
                dst_inner[q] = row;
                dst_vals[q] = static_cast<S>(src_vals[q]);
            }
            if (row < col_i) {
                upper_nnz += run_size;
                // Destination CSR(row=col, col=row) must use the canonical
                // lower values L(col,row), found in source CSC column `row`.
                // Compressed Eigen matrices may contain duplicate sorted
                // inner indices, so require and copy the whole partner run
                // one-for-one. A mere lower_bound hit plus globally balanced
                // triangle counts can accept opposite multiplicity errors on
                // two coordinates and silently change the operator.
                const int begin = src_outer[row];
                const int end = src_outer[row + 1];
                const auto partners = std::equal_range(
                    src_inner + begin, src_inner + end, col_i);
                const int partner_begin = static_cast<int>(
                    partners.first - src_inner);
                const int partner_size = static_cast<int>(
                    partners.second - partners.first);
                if (partner_size == run_size) {
                    for (int offset = 0; offset < run_size; ++offset)
                        dst_vals[p + offset] = static_cast<S>(
                            src_vals[partner_begin + offset]);
                } else {
                    upper_is_paired = false;
                }
            } else if (row > col_i) {
                lower_nnz += run_size;
            }
            p = run_end;
        }
    }
    dst_outer[n] = src_outer[n];

    // Per-coordinate duplicate multiplicities now match for every upper run.
    // Equal global triangle counts additionally exclude lower-only runs.
    // Explicit-zero asymmetries take the general fallback so they cannot
    // perturb the SpMV's four-accumulator association through changed row
    // lengths.
    return upper_is_paired && upper_nnz == lower_nnz;
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
    const int workers = detail::cpu_spmv_threads(n, outer[n] - outer[0]);
    #pragma omp parallel if(workers > 1) num_threads(workers)
    {
        int tid, nt; omp_ids(tid, nt);
        if (tid == 0) nt_used = nt;
        // Rows split by stored entries, not by count: a row costs its length,
        // and with hub rows the heaviest equal-count chunk carries 2-4x the
        // mean (see detail::work_balanced_range). The bounds depend only on
        // the row pointers and the team size, and the partials below are still
        // summed in thread order, so the result stays bit-identical run to run
        // for a fixed thread count.
        const auto [lo, hi] = detail::work_balanced_range(outer, n, tid, nt);
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
    const int workers = detail::cpu_solve_threads(n);
    #pragma omp parallel if(workers > 1) num_threads(workers)
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
    const int workers = detail::cpu_solve_threads(n);
    #pragma omp parallel for schedule(static) if(workers > 1) num_threads(workers)
    for (Eigen::Index i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
}

// Initial residual in one pass: r = b - Ax0 (Ax0 == nullptr -> r = b),
// returning rr = r·r and rs = Σ r. Same deterministic scheme.
inline void init_residual(double* r, const double* b, const double* Ax0,
                          Eigen::Index n, double* part, double& rr, double& rs) {
    int nt_used = 1;
    const int workers = detail::cpu_solve_threads(n);
    #pragma omp parallel if(workers > 1) num_threads(workers)
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
    const int workers = detail::cpu_solve_threads(n);
    #pragma omp parallel for schedule(static) if(workers > 1) num_threads(workers)
    for (Eigen::Index i = 0; i < n; ++i) x[i] -= mean;
}

} // namespace

// ── cpu_solver: factor + operator built once, PCG-solve many b ──────────────────
// Checkpoint placement here defines the setup/solve timing split used by the
// benchmark suite; keep the operation order and checkpoint labels stable.

cpu_solver::cpu_solver(const Eigen::SparseMatrix<double>& L,
                       const solve_options& opts, checkpoint* cp)
    : opts_(opts), n_(L.rows()) {
    const detail::thread_budget threads(opts_.num_threads);
    if (opts.backend != solve_backend::automatic && opts.backend != solve_backend::cpu)
        throw std::invalid_argument("cpu_solver requires the CPU backend");
    ensure_eigen_parallel();

    // Build preconditioner.
    precond_.set_options(opts_.factor_opts);
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
    print_sptrsv_banner<solve_backend::cpu>(precond_.trsv().fp16());
    build_operator(L, cp);
    precond_.set_checkpoint(nullptr);
}

cpu_solver::cpu_solver(const Eigen::SparseMatrix<double>& L,
                       factorization F,
                       const solve_options& opts, checkpoint* cp)
    : opts_(opts), n_(L.rows()) {
    const detail::thread_budget threads(opts_.num_threads);
    if (opts.backend != solve_backend::automatic && opts.backend != solve_backend::cpu)
        throw std::invalid_argument("cpu_solver requires the CPU backend");
    ensure_eigen_parallel();

    if (static_cast<Eigen::Index>(F.L.rows()) != n_)
        throw std::invalid_argument("cpu_solver: factorization dimension mismatch");
    if (F.L.nonZeros() > 0 && F.L.vals_.empty())
        throw std::invalid_argument(
            "cpu_solver: factorization values were released; pass a freshly "
            "computed factorization");
    precond_.set_keep_factor(opts_.keep_factor_values);
    if (cp) precond_.set_checkpoint(cp);
    precond_.set_options(opts_.factor_opts);
    precond_.set_factor(std::move(F));
    print_sptrsv_banner<solve_backend::cpu>(precond_.trsv().fp16());
    build_operator(L, cp);
    precond_.set_checkpoint(nullptr);
}

void cpu_solver::build_operator(const Eigen::SparseMatrix<double>& L,
                                checkpoint* cp) {
    // Build an owned row-major FULL symmetric operator for parallel SpMV.
    // The common full-symmetric CSC needs no histogram/scatter/sort: its three
    // arrays already have the CSR layout. A lower-triangle/custom input takes
    // the retained general selfadjoint(Lower) reconstruction below.
    const bool pcg_trace_outer = std::getenv("APXCHOL_PCG_TRACE") != nullptr;
    {
        // Direct array access needs compressed storage. Standard inputs already
        // are compressed; keep the uncommon conversion local to setup so the
        // solver still owns only its final CSR after this scope.
        Eigen::SparseMatrix<double> compressed_L;
        const Eigen::SparseMatrix<double>* src = &L;
        if (!L.isCompressed()) {
            compressed_L = L;
            compressed_L.makeCompressed();
            src = &compressed_L;
        }
        const Eigen::SparseMatrix<double>& Lcsc = *src;
        const operator_storage_properties storage = scan_operator_storage(Lcsc);

        // Operator A storage precision: fp32 when every caller value
        // round-trips fp32 (lossless), else fp64. Resolve this BEFORE building
        // so the direct path writes the final scalar width and never creates a
        // transient fp64 Lrm merely to cast it. APXCHOL_FP32_OPERATOR
        // overrides: "0" forces fp64, any other value forces fp32 (may floor
        // if inexact).
        { const char* e = std::getenv("APXCHOL_FP32_OPERATOR");
          if (e && std::string(e) == "0")   op_fp32_ = false;
          else if (e && *e != '\0')         op_fp32_ = true;
          else                              op_fp32_ = storage.fp32_exact; }

        // Lrm conversion is one-time SETUP work for the parallel SpMV;
        // checkpoint it under "setup" so bench's setup_time/solve_time
        // split is accurate.
        const auto t0 = std::chrono::high_resolution_clock::now();
        if (cp) { cp->descend("setup"); cp->tick(); }
        const bool copied = op_fp32_
            ? copy_symmetric_csc_as_owned_csr(
                  Lcsc, Lrm_f_, storage.inner_indices_sorted)
            : copy_symmetric_csc_as_owned_csr(
                  Lcsc, Lrm_, storage.inner_indices_sorted);
        bool fallback_needs_fp32_cast = false;

        if (!copied) {
            // General parallel selfadjoint(Lower) → full symmetric CSR
            // fallback. This preserves support for the adopting constructor's
            // one-triangle operator without taxing the validated/full common
            // path with atomics, a prefix sum, or per-row sorting.
            if (op_fp32_)
                Eigen::SparseMatrix<float, Eigen::RowMajor>().swap(Lrm_f_);

            Lrm_.resize(Lcsc.rows(), Lcsc.cols());
            detail::reflect_lower_to_csr(Lcsc, {Lrm_.outerIndexPtr(), std::size_t(Lcsc.rows()) + 1},
                [](Eigen::Index i) { return static_cast<int>(i); },
                [&](int nnz) {
                    Lrm_.resizeNonZeros(nnz);
                    return std::pair{Lrm_.innerIndexPtr(), Lrm_.valuePtr()};
                },
                [](const auto& a, const auto& b) { return a.first < b.first; });
            Lrm_.makeCompressed();
            fallback_needs_fp32_cast = op_fp32_;
        }

        if (cp) { (*cp)("spmv_lrm_build"); cp->ascend(); }
        if (pcg_trace_outer) {
            const auto dt = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t0).count();
            std::fprintf(stderr, "[pcg] Lrm conversion: %.0f ms\n", dt*1000);
        }
        // Preserve the historical checkpoint/trace boundary on the uncommon
        // fallback: its fp32 cast was never part of spmv_lrm_build. The direct
        // path has no cast or transient fp64 matrix at all.
        if (fallback_needs_fp32_cast) {
            Lrm_f_ = Lrm_.cast<float>();
            Eigen::SparseMatrix<double, Eigen::RowMajor>().swap(Lrm_);
        }
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
    const detail::thread_budget threads(opts_.num_threads);
    if (tol < 0.0)    tol = opts_.tol;
    if (max_iter < 0) max_iter = opts_.max_iter;

    // Report what the preconditioner build had to lump. The PCG below applies
    // Lrm_/Lrm_f_, built from the caller's matrix, so the residual it reports
    // is against the true operator whether anything was lumped or not.
    res.lumped_offdiag = precond_.factor().lumped_offdiag;
    res.backend = solve_backend::cpu;
    res.solve_vram_mb = -1.0;

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
        if (bnorm == 0.0) {
            x.setZero();
            res.iterations = 0;
            res.residual = 0.0;
            return;
        }
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
        if (bnorm == 0.0) {
            x.setZero();
            res.iterations = 0;
            res.residual = 0.0;
            return;
        }
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
    precond_.set_checkpoint(use_cp ? &res.timings : nullptr);
    struct clear_checkpoint {
        const apx_cholesky& preconditioner;
        ~clear_checkpoint() { preconditioner.set_checkpoint(nullptr); }
    } clear{precond_};

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

Eigen::VectorXd cpu_solver::apply(const Eigen::VectorXd& r) const {
    const detail::thread_budget threads(opts_.num_threads);
    return precond_.solve(r);
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

solve_backend detail::select_solve_backend(const solve_options& opts,
                                          const Eigen::SparseMatrix<double>* A) {
    if (opts.backend == solve_backend::cpu) return solve_backend::cpu;
    if (opts.backend != solve_backend::automatic && opts.backend != solve_backend::gpu)
        throw std::invalid_argument("invalid solve backend");
    // Keep small solves on CPU unless it cannot honor the requested storage.
#if defined(APXCHOL_USE_METAL)
    constexpr Eigen::Index cpu_limit = 32 * 1024;
#else
    constexpr Eigen::Index cpu_limit = 1;
#endif
    if (opts.backend == solve_backend::automatic && A && A->rows() <= cpu_limit) {
        if (A->rows() == 1 || omp_sptrsv::fp16_supported() ||
            !detail::resolve_fp16_storage(opts.factor_opts.factor_storage, false))
            return solve_backend::cpu;
    }
#if defined(APXCHOL_USE_CUDA)
    bool compatible = sizeof(node_index) == sizeof(std::uint32_t) &&
        opts.factor_opts.is_select == "block_greedy" &&
        opts.factor_opts.residual_peel == residual_peel_strategy::natural &&
        opts.factor_opts.exact_clique_max_degree == 0 &&
        exact_core_or_off(opts.factor_opts.exact_core_max_h) == 0 &&
        opts.factor_opts.double_cycle_min_h == 0;
    if (compatible && opts.backend == solve_backend::automatic && A) {
        std::size_t offdiagonals = 0;
        bool zeros = false;
        compatible = detail::gpu_owned_csc_layout_supported(*A, offdiagonals, &zeros);
        // Numerical validation proves pairing for nonzero entries. Stored zeros
        // also need structural mates for the device operator builder.
        if (compatible && zeros) compatible = detail::gpu_owned_csc_paired(*A);
    }
    if (compatible && (opts.backend == solve_backend::gpu ||
                       detail::gpu_round_shadow_runtime_available()))
        return solve_backend::gpu;
    if (opts.backend == solve_backend::gpu)
        throw std::invalid_argument("GPU route requires 32-bit nodes, block_greedy, vec_pool_aos, "
            "natural residual peeling and supported sampler options; request CPU explicitly");
    return solve_backend::cpu;
#elif defined(APXCHOL_USE_METAL)
    if (opts.backend == solve_backend::automatic && A) {
        bool in_range = true;
        #pragma omp parallel for schedule(static) reduction(&& : in_range)
        for (Eigen::Index col = 0; col < A->outerSize(); ++col)
            for (Eigen::SparseMatrix<double>::InnerIterator entry(*A, col); entry; ++entry)
                in_range = in_range && detail::metal_schedule::in_range(entry.value());
        if (!in_range) return solve_backend::cpu;
    }
    return opts.backend == solve_backend::gpu || metal_solver::available()
        ? solve_backend::gpu : solve_backend::cpu;
#else
    if (opts.backend == solve_backend::gpu)
        throw std::invalid_argument("GPU route requested but GPU support is not built");
    return solve_backend::cpu;
#endif
}

struct solver::impl {
    using owner_type = std::variant<cpu_solver
#if defined(APXCHOL_USE_CUDA)
        , detail::gpu_solve_session
#elif defined(APXCHOL_USE_METAL)
        , metal_solver
#endif
        >;
    owner_type owner;
    double tol;
    int max_iter;
    int num_threads;

    static owner_type make_owner(const Eigen::SparseMatrix<double>& A,
                                 const solve_options& opts, checkpoint* cp,
                                 factorization* F) {
        const detail::thread_budget threads(opts.num_threads);
        if (A.rows() == 0) throw std::invalid_argument("solver: empty operator");
        if (cp) { cp->descend("setup"); cp->tick(); }
        const auto route = F && opts.backend == solve_backend::automatic
            ? solve_backend::cpu : detail::select_solve_backend(opts, &A);
        if (cp) { (*cp)("backend_selection"); cp->ascend(); }
#if defined(APXCHOL_USE_CUDA)
        if (route == solve_backend::gpu) {
            if (F) throw std::invalid_argument("CUDA solver cannot adopt a host factor; request CPU explicitly");
            return owner_type(std::in_place_type<detail::gpu_solve_session>, A, opts, cp);
        }
#elif defined(APXCHOL_USE_METAL)
        if (route == solve_backend::gpu) {
            if (F) return owner_type(std::in_place_type<metal_solver>, A, std::move(*F), opts, cp);
            return owner_type(std::in_place_type<metal_solver>, A, opts, cp);
        }
#else
        (void)route;
#endif
        if (F) return owner_type(std::in_place_type<cpu_solver>, A, std::move(*F), opts, cp);
        return owner_type(std::in_place_type<cpu_solver>, A, opts, cp);
    }

    impl(const Eigen::SparseMatrix<double>& A, const solve_options& opts,
         checkpoint* cp, factorization* F = nullptr)
        : owner(make_owner(A, opts, cp, F)), tol(opts.tol), max_iter(opts.max_iter),
          num_threads(opts.num_threads) {
#if defined(APXCHOL_USE_CUDA)
        if (const auto* gpu = std::get_if<detail::gpu_solve_session>(&owner))
            print_sptrsv_banner<solve_backend::gpu>(gpu->preconditioner().trsv().fp16());
#endif
    }
};

solver::solver(const Eigen::SparseMatrix<double>& A, const solve_options& opts, checkpoint* cp)
    : impl_(std::make_unique<impl>(A, opts, cp)) {}
solver::solver(const Eigen::SparseMatrix<double>& A, factorization F,
               const solve_options& opts, checkpoint* cp)
    : impl_(std::make_unique<impl>(A, opts, cp, &F)) {}
solver::~solver() = default;
solver::solver(solver&&) noexcept = default;
solver& solver::operator=(solver&&) noexcept = default;

void solver::solve(const Eigen::VectorXd& b, solve_result& result, double tol,
                   int max_iter, const Eigen::VectorXd* x0) const {
    const detail::thread_budget threads(impl_->num_threads);
    if (b.size() != rows()) throw std::invalid_argument("solver::solve: RHS length mismatch");
    std::visit([&](auto& owner) {
        owner.solve(b, result, tol < 0 ? impl_->tol : tol,
                    max_iter < 0 ? impl_->max_iter : max_iter, x0);
    }, impl_->owner);
}

solve_result solver::solve(const Eigen::VectorXd& b, double tol, int max_iter,
                           const Eigen::VectorXd* x0) const {
    solve_result result;
    solve(b, result, tol, max_iter, x0);
    return result;
}

solve_result solver::solve(const Eigen::VectorXd& b, Eigen::Ref<Eigen::VectorXd> x,
                           double tol, int max_iter, const Eigen::VectorXd* x0) const {
    if (b.size() != rows() || x.size() != rows())
        throw std::invalid_argument("solver::solve: vector length mismatch");
    if (const auto* cpu = std::get_if<cpu_solver>(&impl_->owner))
        return cpu->solve(b, x, tol, max_iter, x0);
    auto result = solve(b, tol, max_iter, x0);
    x = result.x;
    result.x.resize(0);
    return result;
}

Eigen::VectorXd solver::apply(const Eigen::VectorXd& r) const {
    const detail::thread_budget threads(impl_->num_threads);
    if (r.size() != rows()) throw std::invalid_argument("solver::apply: RHS length mismatch");
    return std::visit([&](auto& owner) -> Eigen::VectorXd {
        if constexpr (requires { owner.apply(r); }) return owner.apply(r);
        else return owner.preconditioner().solve(r);
    }, impl_->owner);
}
const factorization& solver::factor() const {
    return std::visit([](const auto& owner) -> const factorization& {
        if constexpr (requires { owner.factor(); }) return owner.factor();
        else return owner.preconditioner().factor();
    }, impl_->owner);
}
Eigen::Index solver::rows() const { return static_cast<Eigen::Index>(factor().perm.size()); }
solve_backend solver::backend() const {
    return std::holds_alternative<cpu_solver>(impl_->owner)
        ? solve_backend::cpu : solve_backend::gpu;
}
factor_storage_type solver::factor_storage() const {
    return std::visit([](const auto& owner) {
        if constexpr (requires { owner.factor_storage(); }) return owner.factor_storage();
        else return owner.preconditioner().trsv().fp16()
            ? factor_storage_type::fp16 : factor_storage_type::fp32;
    }, impl_->owner);
}

solve_result solve(const Eigen::SparseMatrix<double>& L, const Eigen::VectorXd& b,
                   const solve_options& opts) {
    ensure_eigen_parallel();
    solve_result result;
    const solver owner(L, opts,
        std::getenv("APXCHOL_NO_CHECKPOINT") ? nullptr : &result.timings);
    owner.solve(b, result);
    return result;
}

Eigen::VectorXd generate_test_rhs(Eigen::Index n) {
    Eigen::VectorXd b = Eigen::VectorXd::Random(n);
    b.array() -= b.mean();
    return b.normalized();
}

} // namespace apxchol
