#include "apxchol/solver/solve.h"
#include "apxchol/solver/detail/solver_route.h"
#include "apxchol/solver/detail/pcg_reduction.h"
#include "apxchol/solver/detail/solve_backend.h"
#include "apxchol/solver/detail/factor_validation.h"
#include "apxchol/detail/call_guard.h"
#include "apxchol/solver/detail/thread_budget.h"
#include "apxchol/solver/detail/lower_triangle_csr.h"
#include "apxchol/solver/detail/gpu_scaling.h"
#include "apxchol/solver/detail/pcg_restart.h"
#include "apxchol/csc_work.h"
#include "apxchol/detail/dense.h"
#include "apxchol/detail/csc_input.h"
#include "apxchol/operator_class.h"
#include "host_spmv.h"
#include "solver_archive_internal.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <omp.h>
#if defined(APXCHOL_USE_METAL)
#include "metal_device.h"
#include "metal_setup_options.h"
#endif

namespace apxchol {

bool backend_available(solve_backend backend) noexcept {
    if (backend == solve_backend::cpu || backend == solve_backend::automatic) return true;
    if (backend != solve_backend::gpu) return false;
    try {
#if defined(APXCHOL_USE_CUDA)
        return detail::gpu_round_shadow_runtime_available();
#elif defined(APXCHOL_USE_METAL)
        return detail::metal::status().ok;
#else
        return false;
#endif
    } catch (...) {
        return false;
    }
}

void detail::validate_solve_parameters(double tol, int max_iter) {
    if (!(tol > 0) || !std::isfinite(tol))
        throw std::invalid_argument("solve tolerance must be finite and positive");
    if (max_iter < 0)
        throw std::invalid_argument("max_iter must be nonnegative");
}

void detail::validate_solve_options(const solve_options& opts) {
    if (opts.num_threads < 0)
        throw std::invalid_argument("num_threads must be nonnegative");
    if (opts.backend != solve_backend::automatic && opts.backend != solve_backend::cpu &&
        opts.backend != solve_backend::gpu)
        throw std::invalid_argument("invalid solve backend");
    validate_factor_options(opts.factor_opts);
}

namespace {

// Omitted values take the library defaults; explicit values must be valid.
std::pair<double, int> resolve_solve_parameters(std::optional<double> tol,
                                                std::optional<int> max_iter) {
    const std::pair resolved{tol.value_or(default_tol), max_iter.value_or(default_max_iter)};
    detail::validate_solve_parameters(resolved.first, resolved.second);
    return resolved;
}

using detail::const_matrix_ref;
using detail::matrix_ref;

void check_matrix_views(const_matrix_ref B, matrix_ref X, std::int64_t rows,
                        const const_matrix_ref* X0 = nullptr) {
    if (B.rows() != rows || X.rows() != rows || X.cols() != B.cols() ||
        (X0 && (X0->rows() != rows || X0->cols() != B.cols())))
        throw std::invalid_argument("apxchol: RHS, output and warm-start shapes must match");
    const auto extent = [](const auto &value) {
        if (!value.size())
            return std::pair<std::uintptr_t, std::uintptr_t>{0, 0};
        const auto row = (value.rows() - 1) * value.row_stride();
        const auto col = (value.cols() - 1) * value.col_stride();
        const auto first =
            value.data() + std::min<std::int64_t>(0, row) + std::min<std::int64_t>(0, col);
        const auto last =
            value.data() + std::max<std::int64_t>(0, row) + std::max<std::int64_t>(0, col) + 1;
        return std::pair{reinterpret_cast<std::uintptr_t>(first),
                         reinterpret_cast<std::uintptr_t>(last)};
    };
    const auto output_extent = extent(X);
    const auto overlaps = [&](const auto &value) {
        const auto [lo, hi] = extent(value);
        return output_extent.first < hi && lo < output_extent.second;
    };
    if (overlaps(B) || (X0 && overlaps(*X0)))
        throw std::invalid_argument("apxchol: output must not overlap RHS or warm start");
}

template<solve_backend Route>
void print_sptrsv_banner(bool fp16, bool verbose) {
    constexpr const char* backend = Route == solve_backend::gpu ? "GPU/dataflow" : "CPU/omp";
    if (verbose)
        std::fprintf(stderr, "[apxchol] SpTRSV (%s) factor values: %s, %zu bytes/elem\n",
                     backend, fp16 ? "fp16 (per-column scaled, diagonal fp32)" : "float (fp32)",
                     fp16 ? std::size_t(2) : sizeof(detail::sptrsv_value_t));
}

// One O(nnz) parallel storage scan resolves both direct-copy prerequisites:
// lossless fp32 values and nondecreasing row indices inside each column (used
// by equal_range below; callers' compressed arrays need not keep them).
struct operator_storage_properties {
    bool fp32_exact = true;
    bool inner_indices_sorted = true;
};

inline operator_storage_properties scan_operator_storage(detail::csc_ref<double> L) {
    const double* v = L.val;
    const edge_index* outer = L.ptr;
    const node_index* inner = L.idx;
    bool exact = true;
    bool sorted = true;
    const std::int64_t n = L.n;
    #pragma omp parallel reduction(&& : exact, sorted)
    {
        const auto [lo, hi] = detail::work_balanced_range(outer, n);
        for (std::int64_t col = lo; col < hi; ++col) {
            const edge_index begin = outer[col];
            const edge_index end = outer[col + 1];
            for (edge_index p = begin; p < end; ++p) {
                if (static_cast<double>(static_cast<float>(v[p])) != v[p])
                    exact = false;
                if (p > begin && inner[p - 1] > inner[p])
                    sorted = false;
            }
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
// The destination owns all three arrays. The CPU route is reusable and may
// outlive (or be used after mutation of) the constructor's L, so a persistent
// view of the caller's storage would be a lifetime bug.
//
// Returns false when the full stored pattern is not symmetric. The caller then
// rebuilds from the lower triangle, e.g. for explicit zeros stored in one
// triangle only or for duplicate or unsorted entries. A successful return
// produces exactly the sorted CSR structure/value sequence of that general
// rebuild.
template<class S>
bool copy_symmetric_csc_as_owned_csr(
        detail::csc_ref<double> L,
        detail::csc_matrix<S>& Lrm,
        bool inner_indices_sorted) {
    if (!inner_indices_sorted)
        return false;

    const std::int64_t n = L.n;
    const edge_index* src_outer = L.ptr;
    const node_index* src_inner = L.idx;
    const double* src_vals = L.val;

    Lrm.resize(L.n, L.nnz());
    edge_index* dst_outer = Lrm.ptr();
    node_index* dst_inner = Lrm.idx();
    S* dst_vals = Lrm.val();

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
        std::int64_t direct_lower = 0, direct_upper = 0;
        #pragma omp parallel reduction(+ : hash_lower, hash_upper, direct_lower, direct_upper)
        {
            const auto [lo, hi] = detail::work_balanced_range(src_outer, n);
            for (std::int64_t col = lo; col < hi; ++col) {
                dst_outer[col] = src_outer[col];
                const auto col_i = static_cast<node_index>(col);
                std::uint64_t run = 0;
                for (edge_index p = src_outer[col]; p < src_outer[col + 1]; ++p) {
                    const node_index row = src_inner[p];
                    run = p > src_outer[col] && row == src_inner[p - 1] ? run + 1 : 0;
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

    std::int64_t lower_nnz = 0;
    std::int64_t upper_nnz = 0;
    bool upper_is_paired = true;
    #pragma omp parallel reduction(+ : lower_nnz, upper_nnz) \
        reduction(&& : upper_is_paired)
    {
        const auto [lo, hi] = detail::work_balanced_range(src_outer, n);
        for (std::int64_t col = lo; col < hi; ++col) {
            dst_outer[col] = src_outer[col];
            const auto col_i = static_cast<node_index>(col);
            for (edge_index p = src_outer[col]; p < src_outer[col + 1];) {
                const node_index row = src_inner[p];
                edge_index run_end = p + 1;
                while (run_end < src_outer[col + 1] &&
                       src_inner[run_end] == row)
                    ++run_end;
                const edge_index run_size = run_end - p;

                for (edge_index q = p; q < run_end; ++q) {
                    dst_inner[q] = row;
                    dst_vals[q] = static_cast<S>(src_vals[q]);
                }
                if (row < col_i) {
                    upper_nnz += run_size;
                    // Destination CSR(row=col, col=row) must use the canonical
                    // lower values L(col,row), found in source CSC column `row`.
                    // Compressed input may contain duplicate sorted inner
                    // indices, so require and copy the whole partner run
                    // one-for-one. A mere lower_bound hit plus globally balanced
                    // triangle counts can accept opposite multiplicity errors on
                    // two coordinates and silently change the operator.
                    const edge_index begin = src_outer[row];
                    const edge_index end = src_outer[row + 1];
                    const auto partners = std::equal_range(
                        src_inner + begin, src_inner + end, col_i);
                    const auto partner_begin = static_cast<edge_index>(
                        partners.first - src_inner);
                    const auto partner_size = static_cast<edge_index>(
                        partners.second - partners.first);
                    if (partner_size == run_size) {
                        for (edge_index offset = 0; offset < run_size; ++offset)
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
// The outer PCG loop's vector operations -- p.dot(Ap), x += alpha*p,
// r -= alpha*Ap, r.norm(), r.dot(z), p = z + beta*p -- would each be a separate
// full-n pass, ~14 n-vector streams per iteration outside the SpMV/SpTRSV
// kernels. The kernels below fuse them so
// each vector is streamed ONCE per group of operations, and run the passes
// OpenMP-parallel with DETERMINISTIC reductions: per-thread partials over the
// fixed schedule(static) chunk partition, summed serially in thread order
// (detail::static_chunk / omp_ids / reduce_parts in pcg_reduction.h; never a
// reduction() clause). Bit-identical run-to-run for a fixed thread count.
using detail::omp_ids;

// RHS lanes are contiguous within a row. Scalar calls use the same loops with C=1.
template <int C, int R, class S, class Entry>
std::array<double, R> panel_spmv_reduce(const detail::csc_matrix<S> &A,
                                      const double *x, double *y, double *part, Entry entry) {
    constexpr int stride = std::max(R, detail::kPartStride);
    const std::int64_t n = A.rows();
    const edge_index *ptr = A.ptr();
    const node_index *col = A.idx();
    const S *val = A.val();
    int nt_used = 1;
    const int workers = detail::cpu_spmv_threads(n, ptr[n] - ptr[0]);
#pragma omp parallel if (workers > 1) num_threads(workers) firstprivate(entry)
    {
        int tid, nt;
        detail::omp_ids(tid, nt);
        if (tid == 0)
            nt_used = nt;
        const auto [lo, hi] = detail::work_balanced_range(ptr, n, tid, nt);
        std::array<double, R> reduced{};
        for (std::int64_t i = lo; i < hi; i++) {
            // 64-bit entry offsets let the unrolled loads index without
            // 32-bit wraparound arithmetic.
            const auto sum = detail::sparse_row_product<C>(col, val,
                static_cast<std::int64_t>(ptr[i]), static_cast<std::int64_t>(ptr[i + 1]),
                [x](node_index row, int q) { return x[std::int64_t(row) * C + q]; });
            for (int q = 0; q < C; q++)
                y[i * C + q] = sum[q];
            entry(i, x + i * C, sum, reduced);
        }
        for (int q = 0; q < R; q++)
            part[std::size_t(tid) * stride + q] = reduced[q];
    }
    std::array<double, R> reduced{};
    for (int q = 0; q < R; q++)
        reduced[q] = detail::reduce_parts(part, nt_used, q, stride);
    return reduced;
}

template <int C, class S>
std::array<double, C> panel_spmv(const detail::csc_matrix<S> &A, const double *x,
                                 double *y, double *part) {
    return panel_spmv_reduce<C, C>(A, x, y, part,
        [](std::int64_t, const double* input, const std::array<double, C>& product,
           std::array<double, C>& xy) {
            for (int q = 0; q < C; q++) xy[q] += input[q] * product[q];
        });
}

// Transform four rows together, then accumulate their squared norms and sums.
// The four independent chains preserve the scalar reduction order.
template <int C, class Entry>
void panel_residual(std::int64_t n, double *part, Entry entry, std::array<double, C> &rr,
                    std::array<double, C> &rs) {
    static_assert(C > 0 && C <= 8);
    constexpr int stride = std::max(2 * C, detail::kPartStride);
    int nt_used = 1;
    const int workers = detail::cpu_solve_threads(n);
#pragma omp parallel if (workers > 1) num_threads(workers) firstprivate(entry)
    {
        int tid, nt;
        detail::omp_ids(tid, nt);
        if (tid == 0)
            nt_used = nt;
        const auto [lo, hi] = detail::static_chunk(n, tid, nt);
        std::array<std::array<double, C>, 4> norms{}, sums{};
        std::int64_t i = lo;
        for (; i + 4 <= hi; i += 4) {
            const auto residuals = entry.template operator()<4>(i);
            for (int k = 0; k < 4; k++)
                for (int q = 0; q < C; q++) {
                    const double r = residuals[k][q];
                    norms[k][q] += r * r;
                    sums[k][q] += r;
                }
        }
        std::array<double, C> norm{}, sum{};
        for (int q = 0; q < C; q++) {
            norm[q] = (norms[0][q] + norms[1][q]) + (norms[2][q] + norms[3][q]);
            sum[q] = (sums[0][q] + sums[1][q]) + (sums[2][q] + sums[3][q]);
        }
        for (; i < hi; i++) {
            const auto residuals = entry.template operator()<1>(i);
            for (int q = 0; q < C; q++) {
                const double r = residuals[0][q];
                norm[q] += r * r;
                sum[q] += r;
            }
        }
        for (int q = 0; q < C; q++) {
            part[std::size_t(tid) * stride + 2 * q] = norm[q];
            part[std::size_t(tid) * stride + 2 * q + 1] = sum[q];
        }
    }
    for (int q = 0; q < C; q++) {
        rr[q] = detail::reduce_parts(part, nt_used, 2 * q, stride);
        rs[q] = detail::reduce_parts(part, nt_used, 2 * q + 1, stride);
    }
}

template <int C>
void panel_update_xr(double *x, const double *p, double *r, const double *Ap,
                     const std::array<double, C> &alpha, const std::array<bool, C> &active,
                     std::int64_t n, double *part, std::array<double, C> &rr,
                     std::array<double, C> &rs) {
    panel_residual<C>(
        n, part,
        [=]<int Rows>(std::int64_t first) {
            for (int k = 0; k < Rows; ++k)
                for (int c = 0; c < C; ++c)
                    if (C == 1 || active[c]) {
                        const auto i = (first + k) * C + c;
                        x[i] += alpha[c] * p[i];
                    }
            std::array<std::array<double, C>, Rows> residuals{};
            for (int k = 0; k < Rows; ++k)
                for (int c = 0; c < C; ++c) {
                    const auto i = (first + k) * C + c;
                    residuals[k][c] = (C == 1 || active[c]) ? r[i] - alpha[c] * Ap[i] : r[i];
                }
            for (int k = 0; k < Rows; ++k)
                for (int c = 0; c < C; ++c)
                    r[(first + k) * C + c] = residuals[k][c];
            return residuals;
        },
        rr, rs);
}

template <int C>
void panel_init_residual(double *r, const double *b, const double *Ax0, std::int64_t n,
                         double *part, std::array<double, C> &rr, std::array<double, C> &rs) {
    panel_residual<C>(n, part, [=]<int Rows>(std::int64_t first) {
        std::array<std::array<double, C>, Rows> residuals{};
        for (int k = 0; k < Rows; ++k)
            for (int c = 0; c < C; ++c) {
                const auto i = (first + k) * C + c;
                residuals[k][c] = Ax0 ? b[i] - Ax0[i] : b[i];
            }
        for (int k = 0; k < Rows; ++k)
            for (int c = 0; c < C; ++c)
                r[(first + k) * C + c] = residuals[k][c];
        return residuals;
    }, rr, rs);
}

template <int C>
void panel_update_p(double *p, const double *z, const std::array<double, C> &beta,
                    const std::array<bool, C> &active, std::int64_t n) {
    const int workers = detail::cpu_solve_threads(n);
#pragma omp parallel for schedule(static) if (workers > 1) num_threads(workers)
    for (std::int64_t i = 0; i < n; i++)
        for (int c = 0; c < C; c++)
            p[i * C + c] = active[c] ? z[i * C + c] + beta[c] * p[i * C + c] : 0.0;
}


} // namespace

// ── cpu_route: factor + operator built once, PCG-solve many b ───────────────────
// Checkpoint placement here defines the setup/solve timing split used by the
// benchmark suite; keep the operation order and checkpoint labels stable.

detail::cpu_route::cpu_route(csc_ref<double> A,
                             const solve_options& opts, checkpoint* cp, factorization* F)
    : opts_(opts), n_(A.n) {
    if (F) {
        install_factor(std::move(*F), cp);
    } else {
        const bool _pcg_trace = opts_.factor_opts.verbose;
        const auto _t_compute_start = std::chrono::high_resolution_clock::now();
        install_factor(detail::factorize_for_solver(A, opts_.factor_opts, cp, setup_route::cpu), cp);
        if (_pcg_trace) {
            const double dt = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - _t_compute_start).count();
            const double dt_factorize = cp ? cp->total("setup") : 0.0;
            std::fprintf(stderr, "[pcg] precond.compute wall=%.0f ms (factorize=%.0f ms, sptrsv-setup+misc=%.0f ms)\n",
                         dt*1000, dt_factorize*1000, (dt - dt_factorize)*1000);
        }
    }
    print_sptrsv_banner<solve_backend::cpu>(trsv().fp16(), opts_.factor_opts.verbose);
    build_operator(A, cp);
}

void detail::cpu_route::install_factor(factorization F, checkpoint* cp) {
    auto next = std::make_shared<installed>();
    next->F = std::move(F);
    auto& factor = next->F;
    auto& trsv = next->trsv;
    // SpTRSV setup is setup work: record it under "setup" so the setup/solve
    // split stays complete.
    if (cp) { cp->descend("setup"); cp->tick(); }
    // Laplacians leave the grounded last row/column unused.
    const node_index factor_dim = static_cast<node_index>(factor.sddm ? n_ : n_ - 1);
    // Round-as-level: hand the SpTRSV the per-round IS-column boundaries so it
    // builds level sets straight off the elimination rounds (same-round IS
    // columns are mutually independent -> one level) instead of two O(nnz)
    // topological depth scans. Each region is one factor column placed in round
    // order; trailing residual-peel columns get sequential levels inside the
    // SpTRSV. Missing or invalid round metadata uses the topological scan.
    if (!factor.rounds.empty()) {
        std::vector<node_index> bounds;
        bounds.reserve(factor.rounds.size() + 1);
        bounds.push_back(0);
        node_index acc = 0;
        for (const auto& r : factor.rounds) {
            acc += static_cast<node_index>(r.is_size);
            if (acc > factor_dim) acc = static_cast<node_index>(factor_dim);
            bounds.push_back(acc);
        }
        trsv.set_round_bounds(std::move(bounds));
    }
    // Hand the factor over: the SpTRSV releases F.L's row/value arrays at the
    // first point it no longer reads them (Laplacian path: right after its L11
    // copy), so its setup transient does not sit on a dead copy of the
    // factor. An installed factor is reinstalled exactly; installed_factor()
    // exports it again from the SpTRSV.
    if (factor.storage == factor_storage_type::automatic)
        trsv.setup_consuming(factor.L, factor_dim, opts_.factor_opts);
    else
        trsv.install_exact(factor.L, factor_dim, factor.storage == factor_storage_type::fp16,
                           factor.scale.data(), opts_.factor_opts);
    if (cp) { (*cp)("sptrsv_setup"); cp->ascend(); }
    // The PCG loop only uses trsv, never F.L. Release whatever row/value
    // arrays setup_consuming left (idempotent); column pointers remain.
    detail::sparse_csc_access::release_values(factor.L);
    std::vector<float>().swap(factor.scale);
    scratch_.resize(n_);
    scratch2_.resize(n_);
    installed_ = std::move(next);
    barrier_ = installed_->trsv.make_workspace();
}

void detail::cpu_route::build_operator(csc_ref<double> L, checkpoint* cp) {
    // Build an owned row-major FULL symmetric operator for parallel SpMV.
    // The common full-symmetric CSC needs no histogram/scatter/sort: its three
    // arrays already have the CSR layout. Other stored patterns take the
    // general selfadjoint(Lower) reconstruction below.
    const bool pcg_trace_outer = opts_.factor_opts.verbose;
    auto next = std::make_shared<pcg_operator>();
    auto& op = *next;
    {
        const operator_storage_properties storage = scan_operator_storage(L);

        // Operator A storage precision: fp32 when every caller value
        // round-trips fp32 (lossless), else fp64. Resolve this BEFORE building
        // so the direct path writes the final scalar width and never creates a
        // transient fp64 Lrm merely to cast it.
        op.fp32 = storage.fp32_exact;

        // Lrm conversion is one-time SETUP work for the parallel SpMV;
        // checkpoint it under "setup" so bench's setup_time/solve_time
        // split is accurate.
        const auto t0 = std::chrono::high_resolution_clock::now();
        if (cp) { cp->descend("setup"); cp->tick(); }
        const bool copied = op.fp32
            ? copy_symmetric_csc_as_owned_csr(L, op.Lrm_f, storage.inner_indices_sorted)
            : copy_symmetric_csc_as_owned_csr(L, op.Lrm, storage.inner_indices_sorted);
        bool fallback_needs_fp32_cast = false;

        if (!copied) {
            // General parallel selfadjoint(Lower) → full symmetric CSR
            // fallback. It handles the uncommon stored patterns without
            // taxing the full symmetric common path with atomics, a prefix
            // sum, or per-row sorting.
            if (op.fp32) op.Lrm_f.clear();

            op.Lrm.resize(L.n, 0);
            detail::reflect_lower_to_csr(L,
                std::span<edge_index>(op.Lrm.ptr(), static_cast<std::size_t>(L.n) + 1),
                [](std::int64_t i) { return static_cast<node_index>(i); },
                [&](edge_index nnz) {
                    op.Lrm.resize_nonzeros(nnz);
                    return std::pair{op.Lrm.idx(), op.Lrm.val()};
                });
            fallback_needs_fp32_cast = op.fp32;
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
            op.Lrm_f = csc_cast<float>(op.Lrm.ref());
            op.Lrm.clear();
        }
    }
    op_ = std::move(next);
}

detail::cpu_route::cpu_route(const cpu_route& other)
    : opts_(other.opts_), installed_(other.installed_), op_(other.op_), n_(other.n_),
      barrier_(installed_->trsv.make_workspace()) {}

factor_storage_type detail::cpu_route::factor_storage() const {
    return trsv().fp16() ? factor_storage_type::fp16 : factor_storage_type::fp32;
}

factorization detail::cpu_route::installed_factor() const {
    const auto& source = factor();
    factorization F;
    F.perm = source.perm;
    F.sddm = source.sddm;
    F.lumped_offdiag = source.lumped_offdiag;
    F.rounds = source.rounds;
    F.storage = factor_storage();
    trsv().export_installed(F.L, static_cast<node_index>(n_), F.scale);
    return F;
}

template<int C>
void detail::cpu_route::check_solution(const_matrix_ref B, const double* x,
                                       solve_result* results, checkpoint* timings,
                                       double tol) const {
    if (timings) { timings->descend("pcg"); timings->tick(); }
    double* product = Ap_.data();
    double* residual = r_.data();
    // Recompute the original-system residual and both norms during SpMV.
    // Retain Ax and r for the stable fallback when squared norms under/overflow.
    const auto entry = [&](std::int64_t i, const double*, const std::array<double, C>& row,
                           std::array<double, 2 * C>& squares) {
        for (int c = 0; c < C; ++c) {
            const double b = B(i, c), r = b - row[c];
            residual[i * C + c] = r;
            squares[2 * c] += b * b;
            squares[2 * c + 1] += r * r;
        }
    };
    const auto squares = op_->fp32
        ? panel_spmv_reduce<C, 2 * C>(op_->Lrm_f, x, product, part_.data(), entry)
        : panel_spmv_reduce<C, 2 * C>(op_->Lrm, x, product, part_.data(), entry);
    for (int c = 0; c < C; ++c) {
        if (std::isnormal(squares[2 * c]) && std::isnormal(squares[2 * c + 1])) {
            results[c].residual = std::sqrt(squares[2 * c + 1]) / std::sqrt(squares[2 * c]);
            results[c].converged = results[c].residual < tol;
            continue;
        }
        // Column c of the row-interleaved residual and product panels.
        const const_matrix_ref r_col(residual + c, n_, 1, C, n_ * C);
        const const_matrix_ref ax_col(product + c, n_, 1, C, n_ * C);
        const auto b_col = B.col(c);
        const detail::gpu_scaling::norm_reference reference(b_col);
        double relative = reference.value == 0
            ? (vec::is_zero(r_col) ? 0.0 : std::numeric_limits<double>::infinity())
            : reference.relative(r_col);
        // Form an overflowing difference in scaled coordinates before subtracting.
        if (!std::isfinite(relative) && reference.value > 0 && vec::all_finite(ax_col)) {
            const double scale = detail::gpu_scaling::pow2_scale(std::max(
                vec::max_abs(b_col), vec::max_abs(ax_col)));
            buffer<double> difference(static_cast<std::size_t>(n_));
            for (std::int64_t i = 0; i < n_; ++i)
                difference[i] = b_col[i] * scale - ax_col[i] * scale;
            relative = detail::gpu_scaling::norm(const_matrix_ref(difference.data(), n_))
                     / detail::gpu_scaling::scaled_norm(b_col, scale);
        }
        results[c].residual = relative;
        results[c].converged = relative < tol;
    }
    if (timings) { (*timings)("exit_check"); timings->ascend(); }
}

template<int C>
void detail::cpu_route::finish_solution(const_matrix_ref B, double* x, solve_result* results,
                                        checkpoint* timings, double tol, int max_iter) const {
    std::array<bool, C> tolerance_stops;
    for (int c = 0; c < C; ++c)
        tolerance_stops[c] = std::isfinite(results[c].residual) && results[c].residual < tol;
    check_solution<C>(B, x, results, timings, tol);
    for (int c = 0; c < C; ++c) {
        auto& result = results[c];
        auto progress = result.iterations;
        const auto restart = [&] {
            return detail::pcg_restart(tolerance_stops[c], result.converged,
                result.residual, progress, max_iter - result.iterations);
        };
        if (!restart()) continue;
        buffer<double> current(static_cast<std::size_t>(n_));
        for (std::int64_t i = 0; i < n_; ++i) current[i] = x[i * C + c];
        const auto rhs = B.col(c);
        while (restart()) {
            solve_result attempt;
            z_.resize(static_cast<std::size_t>(n_));
            for (std::int64_t i = 0; i < n_; ++i) z_[i] = rhs[i];
            solve_panel<1>(z_.data(), current.data(), &attempt, timings, tol,
                           max_iter - result.iterations, current.data());
            tolerance_stops[c] = std::isfinite(attempt.residual) && attempt.residual < tol;
            check_solution<1>(rhs, current.data(), &attempt, timings, tol);
            progress = attempt.iterations;
            result.iterations += progress;
            result.residual = attempt.residual;
            result.converged = attempt.converged;
        }
        for (std::int64_t i = 0; i < n_; ++i) x[i * C + c] = current[i];
    }
}

template <int C>
void detail::cpu_route::solve_panel(const double *b, double *x, solve_result *results, checkpoint* timings, double tol,
                                    int max_iter, const double *x0) const {
    static_assert(C > 0 && C <= 8);
    const std::int64_t n = n_;
    for (int c = 0; c < C; c++) {
        results[c].backend = solve_backend::cpu;
        results[c].iterations = 0;
    }
    const auto panel = static_cast<std::size_t>(n * C);
    r_.resize(panel);
    z_.resize(panel);
    p_.resize(panel);
    Ap_.resize(panel);
    const auto part_size = detail::part_capacity() * (C > 4 ? 2 : 1);
    if (part_.size() < part_size) part_.resize(part_size);
    double *r = r_.data(), *z = z_.data(), *p = p_.data(), *Ap = Ap_.data(), *part = part_.data();
    const bool laplacian = !factor().sddm;
    std::array<double, C> norm{}, rr{}, rs{}, rz{};
    std::array<bool, C> active{}, warm{};
    // Column c of a row-interleaved panel.
    const auto column = [n](const double* panel_data, int c) {
        return const_matrix_ref(panel_data + c, n, 1, C, n * C);
    };
    if (x0) {
        for (int c = 0; c < C; c++)
            warm[c] = !vec::is_zero(column(x0, c));
    }
    const bool any_warm = std::any_of(warm.begin(), warm.end(), [](bool v) { return v; });
    if (any_warm) {
        if (x0 != x) std::copy_n(x0, n * C, x);
        if (op_->fp32)
            (void)panel_spmv<C>(op_->Lrm_f, x, Ap, part);
        else
            (void)panel_spmv<C>(op_->Lrm, x, Ap, part);
    }
    panel_init_residual<C>(r, b, any_warm ? Ap : nullptr, n, part, rr, rs);
    if (!any_warm) std::fill_n(x, n * C, 0.0);
    for (int c = 0; c < C; c++)
        norm[c] = warm[c] ? std::sqrt(vec::squared_norm(column(b, c))) : std::sqrt(rr[c]);
    // Only exceptional columns need scaling; ordinary panels reuse their workspace.
    std::array<double, C> scales;
    scales.fill(1.0);
    bool scaled = false;
    for (int c = 0; c < C; ++c) {
        const double squared = warm[c] ? norm[c] * norm[c] : rr[c];
        if ((std::isnormal(squared) && std::isfinite(rr[c])) || n == 0) continue;
        const auto rhs = column(b, c);
        double maximum = vec::max_abs(rhs);
        if (maximum == 0) continue;
        if (x0) maximum = std::max(maximum, vec::max_abs(column(x0, c)));
        scales[c] = detail::gpu_scaling::pow2_scale(maximum);
        if (scales[c] == 1.0 || !std::isnormal(vec::scaled_squared_norm(rhs, scales[c])))
            throw std::invalid_argument("CPU solve: RHS and initial guess exceed the common scaling range");
        scaled = true;
    }
    if (scaled) {
        buffer<double> scaled_b(panel), scaled_x0(x0 ? panel : 0);
        for (std::int64_t i = 0; i < n; ++i)
            for (int c = 0; c < C; ++c) {
                scaled_b[i * C + c] = b[i * C + c] * scales[c];
                if (x0) scaled_x0[i * C + c] = x0[i * C + c] * scales[c];
            }
        solve_panel<C>(scaled_b.data(), x, results, timings, tol, max_iter,
                        x0 ? scaled_x0.data() : nullptr);
        for (std::int64_t i = 0; i < n; ++i)
            for (int c = 0; c < C; ++c) {
                double& value = x[i * C + c];
                const double restored = value / scales[c];
                if (!std::isfinite(restored))
                    throw std::overflow_error("CPU solve: solution exceeds the finite range");
                if (value != 0 && restored == 0)
                    throw std::underflow_error("CPU solve: solution underflows to zero");
                value = restored;
            }
        return;
    }
    for (int c = 0; c < C; c++) {
        results[c].residual = norm[c] == 0 ? 0 : (warm[c] ? std::sqrt(rr[c]) / norm[c] : 1);
        active[c] = norm[c] != 0 && !(warm[c] && results[c].residual < tol);
        if (norm[c] == 0)
            for (std::int64_t i = 0; i < n; i++) {
                x[i * C + c] = 0;
                r[i * C + c] = 0;
            }
    }
    const auto any_active = [&] {
        return std::any_of(active.begin(), active.end(), [](bool v) { return v; });
    };
    const auto center = [&] {
        if (!laplacian)
            return;
        auto mean = detail::det_sum_columns<C>(x, n, part);
        for (double &v : mean)
            v /= static_cast<double>(n);
        const int workers = detail::cpu_solve_threads(n);
#pragma omp parallel for schedule(static) if (workers > 1) num_threads(workers)
        for (std::int64_t i = 0; i < n; i++)
            for (int c = 0; c < C; c++)
                x[i * C + c] -= mean[c];
    };
    if (!any_active()) {
        center();
        return;
    }

    const bool use_cp = timings != nullptr;
    if (use_cp) {
        timings->descend("pcg");
        timings->tick();
    }
    apply_count_ = 0;
    rz = apply_fused<C>(r, rs, z, timings);
    if constexpr (C == 1) {
        std::copy_n(z, n, p);
    } else {
        for (std::int64_t i = 0; i < n; i++)
            for (int c = 0; c < C; c++)
                p[i * C + c] = active[c] ? z[i * C + c] : 0;
    }
    if (use_cp)
        (*timings)("copy_p");

    const bool trace = opts_.factor_opts.verbose;
    const auto now = [] {
        return std::chrono::duration<double, std::micro>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };
    double t_spmv = 0, t_update_xr = 0, t_update_p = 0, t_precond = 0;
    const double start = trace ? now() : 0;
    int iterations = 0;
    for (int i = 0; i < max_iter; i++) {
        double t = trace ? now() : 0;
        const auto pap =
            op_->fp32 ? panel_spmv<C>(op_->Lrm_f, p, Ap, part) : panel_spmv<C>(op_->Lrm, p, Ap, part);
        if (trace)
            t_spmv += now() - t;
        if (use_cp)
            (*timings)("spmv+pAp");
        std::array<double, C> alpha{};
        for (int c = 0; c < C; c++) {
            active[c] = active[c] && rz[c] > 0 && std::isfinite(rz[c])
                && pap[c] > 0 && std::isfinite(pap[c]);
            if (active[c]) {
                alpha[c] = rz[c] / pap[c];
                active[c] = std::isfinite(alpha[c]);
            }
        }
        if (!any_active())
            break;
        t = trace ? now() : 0;
        panel_update_xr<C>(x, p, r, Ap, alpha, active, n, part, rr, rs);
        if (trace)
            t_update_xr += now() - t;
        if (use_cp)
            (*timings)("update_xr_norm");
        iterations = i + 1;
        for (int c = 0; c < C; c++) {
            if (!active[c])
                continue;
            results[c].iterations = i + 1;
            results[c].residual = std::sqrt(rr[c]) / norm[c];
            if (!std::isfinite(results[c].residual) || results[c].residual < tol)
                active[c] = false;
        }
        if (!any_active())
            break;
        t = trace ? now() : 0;
        const auto next = apply_fused<C>(r, rs, z, timings);
        if (trace)
            t_precond += now() - t;
        std::array<double, C> beta{};
        for (int c = 0; c < C; c++) {
            active[c] = active[c] && next[c] > 0 && std::isfinite(next[c]);
            if (active[c]) {
                beta[c] = next[c] / rz[c];
                active[c] = std::isfinite(beta[c]);
            }
        }
        if (!any_active())
            break;
        t = trace ? now() : 0;
        panel_update_p<C>(p, z, beta, active, n);
        if (trace)
            t_update_p += now() - t;
        if (use_cp)
            (*timings)("update_p");
        rz = next;
    }
    center();
    if (use_cp) {
        if (laplacian)
            (*timings)("center_x");
        timings->ascend();
    }
    if (trace) {
        const double total = now() - start;
        std::fprintf(stderr,
                     "[pcg] iters=%d total=%.0fms\n"
                     "      spmv+pAp=%.0fms (%.1fms/iter)\n"
                     "      precond+rz=%.0fms (%.1fms/iter)\n"
                     "      update_xr+norm=%.0fms (%.1fms/iter)\n"
                     "      update_p=%.0fms (%.1fms/iter)\n"
                     "      unaccounted=%.0fms\n",
                     iterations, total / 1000, t_spmv / 1000, t_spmv / 1000 / iterations,
                     t_precond / 1000, t_precond / 1000 / iterations, t_update_xr / 1000,
                     t_update_xr / 1000 / iterations, t_update_p / 1000,
                     t_update_p / 1000 / iterations,
                     (total - t_spmv - t_precond - t_update_xr - t_update_p) / 1000);
    }
}

// Runs one panel on contiguous row-interleaved RHS lanes. A unit-stride single
// column runs in place; wider or strided panels are packed into z_/panel_x_.
// Packed RHS values in z_ are consumed before the first preconditioner
// application overwrites them.
template<int C, class Run>
void detail::cpu_route::with_packed_panel(const_matrix_ref B, matrix_ref X,
                                          const const_matrix_ref* X0,
                                          checkpoint* timings, Run run) const {
    if constexpr (C == 1) {
        if (B.row_stride() == 1 && X.row_stride() == 1 && (!X0 || X0->row_stride() == 1)) {
            run(B.data(), X.data(), X0 ? X0->data() : nullptr);
            return;
        }
    }
    const std::int64_t n = B.rows();
    const int workers = detail::cpu_solve_threads(n);
    z_.resize(static_cast<std::size_t>(n * C));
    panel_x_.resize(static_cast<std::size_t>(n * C));
#pragma omp parallel for schedule(static) if (workers > 1) num_threads(workers)
    for (std::int64_t i = 0; i < n; ++i)
        for (int c = 0; c < C; ++c) {
            z_[i * C + c] = B(i, c);
            if (X0)
                panel_x_[i * C + c] = (*X0)(i, c);
        }
    if (timings)
        (*timings)("pack_rhs");
    run(z_.data(), panel_x_.data(), X0 ? panel_x_.data() : nullptr);
#pragma omp parallel for schedule(static) if (workers > 1) num_threads(workers)
    for (std::int64_t i = 0; i < n; ++i)
        for (int c = 0; c < C; ++c)
            X(i, c) = panel_x_[i * C + c];
    if (timings)
        (*timings)("unpack_x");
}

void detail::cpu_route::solve_columns(const_matrix_ref B, matrix_ref X,
                                      const const_matrix_ref* X0,
                                      std::span<solve_result> results, checkpoint* timings,
                                      double tol, int max_iter) const {
    const auto run = [&]<int C>() {
        with_packed_panel<C>(B, X, X0, timings,
            [&](const double* b, double* x, const double* x0) {
                solve_panel<C>(b, x, results.data(), timings, tol, max_iter, x0);
                finish_solution<C>(B, x, results.data(), timings, tol, max_iter);
            });
    };
    switch (B.cols()) {
        case 8: return run.template operator()<8>();
        case 4: return run.template operator()<4>();
        case 2: return run.template operator()<2>();
        default: return run.template operator()<1>();
    }
}

void detail::cpu_route::apply_columns(const_matrix_ref R, matrix_ref Z) const {
    const auto run = [&]<int C>() {
        with_packed_panel<C>(R, Z, nullptr, nullptr,
            [&](const double* r, double* z, const double*) { apply_panel<C>(r, z); });
    };
    switch (R.cols()) {
        case 8: return run.template operator()<8>();
        case 4: return run.template operator()<4>();
        case 2: return run.template operator()<2>();
        default: return run.template operator()<1>();
    }
}

template<int C>
std::array<double, C> detail::cpu_route::apply_fused(
        const double* r, const std::array<double, C>& r_sum, double* z, checkpoint* cp) const {
    // PCG stays in the centered subspace in exact arithmetic; correct drift
    // every ten preconditioner applications.
    const bool center = !factor().sddm && (++apply_count_ % 10) == 0;
    std::array<double, C> rz{};
    apply_core<C>(r, r_sum, center, z, &rz, cp);
    return rz;
}

template<int C>
void detail::cpu_route::apply_panel(const double* b, double* z) const {
    const auto need = detail::part_capacity();
    if (part_.size() < need) part_.resize(need);
    const bool center = !factor().sddm;
    const auto sums = center
        ? detail::det_sum_columns<C>(b, n_, part_.data())
        : std::array<double, C>{};
    apply_core<C>(b, sums, center, z, nullptr, nullptr);
}

/// `b_sum` = Σ b and `center` selects whether to perform the two Laplacian
/// mean passes (both ignored for SDDM). If `rz_out` is non-null the final
/// unpermute gather also accumulates b·z into it.
template<int C>
void detail::cpu_route::apply_core(const double* b, const std::array<double, C>& b_sum,
                                   bool center, double* z, std::array<double, C>* rz_out,
                                   checkpoint* cp) const {
    static_assert(C > 0 && C <= detail::kPartStride);
    if (cp) { cp->descend("solve"); cp->tick(); }
    const auto& F = factor();
    const node_index* P = F.perm.data();
    const std::int64_t n = n_;
    const int workers = detail::cpu_solve_threads(n);
    double* part = part_.data();
    scratch_.resize(static_cast<std::size_t>(n * C));
    scratch2_.resize(static_cast<std::size_t>(n * C));

    std::array<double, C> mean{};
    for (int c = 0; c < C; ++c)
        if (center && n > 0)
            mean[c] = b_sum[c] / static_cast<double>(n);

    // perm[original] is the position in the factor; RHS columns are contiguous.
    #pragma omp parallel for schedule(static) if(workers > 1) num_threads(workers)
    for (std::int64_t v = 0; v < n; ++v)
        for (int c = 0; c < C; ++c)
            z[std::size_t(P[v]) * C + c] = b[v * C + c] - (center ? mean[c] : 0.0);
    if (cp) (*cp)("permute");

    trsv().template forward_solve<C>(z, scratch_.data(), barrier_);
    if (cp) (*cp)("forward");
    trsv().template transpose_solve<C>(scratch_.data(), scratch2_.data(), barrier_);
    double* out = scratch2_.data();
    if (cp) (*cp)("back");
    if (!F.sddm)
        std::fill_n(out + (n - 1) * C, C, 0.0);

    std::array<double, C> shift{};
    if (center) {
        shift = detail::det_sum_columns<C>(out, n, part);
        for (int c = 0; c < C; ++c)
            shift[c] /= static_cast<double>(n);
    }
    int nt_used = 1;
    #pragma omp parallel if(workers > 1) num_threads(workers)
    {
        int tid, nt; detail::omp_ids(tid, nt);
        if (tid == 0) nt_used = nt;
        const auto [lo, hi] = detail::static_chunk(n, tid, nt);
        std::array<std::array<double, C>, 4> sums{};
        auto gather = [&](std::int64_t v, auto& sum) {
            for (int c = 0; c < C; ++c) {
                const double value = out[std::size_t(P[v]) * C + c] - shift[c];
                z[v * C + c] = value;
                sum[c] += b[v * C + c] * value;
            }
        };
        std::int64_t v = lo;
        for (; v + 4 <= hi; v += 4) {
            std::array<std::array<double, C>, 4> values{};
            for (int k = 0; k < 4; ++k)
                for (int c = 0; c < C; ++c)
                    values[k][c] = out[std::size_t(P[v + k]) * C + c] - shift[c];
            for (int k = 0; k < 4; ++k)
                for (int c = 0; c < C; ++c)
                    z[(v + k) * C + c] = values[k][c];
            for (int k = 0; k < 4; ++k)
                for (int c = 0; c < C; ++c)
                    sums[k][c] += b[(v + k) * C + c] * values[k][c];
        }
        std::array<double, C> sum{};
        for (int c = 0; c < C; ++c)
            sum[c] = (sums[0][c] + sums[1][c]) + (sums[2][c] + sums[3][c]);
        for (; v < hi; ++v) gather(v, sum);
        for (int c = 0; c < C; ++c)
            part[static_cast<std::size_t>(tid) * detail::kPartStride + c] = sum[c];
    }
    if (rz_out)
        for (int c = 0; c < C; ++c)
            (*rz_out)[c] = detail::reduce_parts(part, nt_used, c);
    if (cp) { (*cp)(rz_out ? "unpermute+rz" : "unpermute"); cp->ascend(); }
}

solve_backend detail::select_solve_backend(const solve_options& opts,
                                          const csc_ref<double>* A) {
    if (opts.backend == solve_backend::cpu) return solve_backend::cpu;
    if (opts.backend != solve_backend::automatic && opts.backend != solve_backend::gpu)
        throw std::invalid_argument("invalid solve backend");
    // Keep small solves on CPU unless it cannot honor the requested storage.
#if defined(APXCHOL_USE_METAL)
    constexpr node_index cpu_limit = 32 * 1024;
#else
    constexpr node_index cpu_limit = 1;
#endif
    if (opts.backend == solve_backend::automatic && A && A->n <= cpu_limit) {
        if (A->n == 1 || omp_sptrsv::fp16_supported() ||
            !detail::resolve_fp16_storage(opts.factor_opts.factor_storage, false))
            return solve_backend::cpu;
    }
#if defined(APXCHOL_USE_CUDA)
    bool compatible = sizeof(node_index) == sizeof(std::uint32_t) &&
        opts.factor_opts.partitioner == partitioner_kind::block_greedy &&
        opts.factor_opts.experimental.cpu_schedule == cpu_sptrsv_schedule::automatic &&
        opts.factor_opts.experimental.exact_clique_max_degree == 0 &&
        opts.factor_opts.experimental.exact_core_max_h.value_or(0) == 0 &&
        opts.factor_opts.experimental.double_cycle_min_h == 0;
    if (compatible && opts.backend == solve_backend::automatic && A) {
        std::size_t offdiagonals = 0;
        bool zeros = false;
        compatible = detail::gpu_owned_csc_layout_supported(*A, offdiagonals, &zeros);
        // Numerical validation proves pairing for nonzero entries. Stored zeros
        // also need structural mates for the device operator builder.
        if (compatible && zeros) compatible = detail::gpu_owned_csc_paired(*A);
    }
    if (compatible && (opts.backend == solve_backend::gpu ||
                       backend_available(solve_backend::gpu)))
        return solve_backend::gpu;
    if (opts.backend == solve_backend::gpu)
        throw std::invalid_argument("GPU setup does not support the requested factor options "
            "or node-index width; request CPU explicitly");
    return solve_backend::cpu;
#elif defined(APXCHOL_USE_METAL)
    bool compatible = detail::metal_setup_options_supported(opts.factor_opts);
    if (compatible && opts.backend == solve_backend::automatic) {
        if (!detail::metal::status().ok) return solve_backend::cpu;
        compatible = !A || detail::metal_setup_input_supported(*A);
    }
    if (!compatible) {
        if (opts.backend == solve_backend::gpu)
            throw std::invalid_argument("Metal GPU setup requires supported GKS options and compressed, sorted, paired CSC input");
        return solve_backend::cpu;
    }
    if (!detail::metal_setup_regularization_supported(opts.factor_opts.regularization,
            opts.backend == solve_backend::automatic ? A : nullptr)) {
        if (opts.backend == solve_backend::gpu)
            throw std::invalid_argument(detail::metal_regularization_error);
        return solve_backend::cpu;
    }
    return solve_backend::gpu;
#else
    if (opts.backend == solve_backend::gpu)
        throw std::invalid_argument("GPU route requested but GPU support is not built");
    return solve_backend::cpu;
#endif
}

namespace {
// Splits `columns` into panels of the widest width that fits, widest first;
// every width list ends in 1.
template<std::size_t K, class Run>
void for_panels(std::int64_t columns, const std::array<int, K>& widths, Run run) {
    for (std::int64_t first = 0; first < columns;) {
        const int width = *std::find_if(widths.begin(), widths.end(),
                                        [&](int w) { return w <= columns - first; });
        run(first, width);
        first += width;
    }
}
} // namespace

struct solver::impl {
    detail::solver_route route;
    std::int64_t operator_entries;
    int num_threads;
    // Internal timing tree attached through solver_access; copies start without one.
    detail::checkpoint* timings = nullptr;
    double setup_seconds = 0.0;
    // Rejects overlapping calls, which would share the route's workspace.
    // A copy starts with its own guard.
    detail::call_guard calls;

    static detail::solver_route copy_route(const detail::solver_route& route) {
        return std::visit([](const auto& owner) {
            using route_type = std::decay_t<decltype(owner)>;
            return detail::solver_route(std::in_place_type<route_type>, owner);
        }, route);
    }

    static detail::solver_route make_route(detail::csc_ref<double> A,
                                           const solve_options& opts, detail::checkpoint* cp,
                                           factorization* F) {
        detail::validate_solve_options(opts);
        const detail::thread_budget threads(opts.num_threads);
        if (A.n == 0) throw std::invalid_argument("apxchol: empty operator");
        if (F) {
            detail::validate_adopted_operator(A, *F, opts.factor_opts.lump_max_mass);
            detail::validate_adopted_factor(*F);
        }
        if (cp) { cp->descend("setup"); cp->tick(); }
        const auto route = F && opts.backend == solve_backend::automatic
            ? solve_backend::cpu
#if defined(APXCHOL_USE_CUDA) || defined(APXCHOL_USE_METAL)
            : F && opts.backend == solve_backend::gpu ? solve_backend::gpu
#endif
            : detail::select_solve_backend(opts, &A);
        if (cp) { (*cp)("backend_selection"); cp->ascend(); }
#if defined(APXCHOL_USE_CUDA)
        if (route == solve_backend::gpu)
            return detail::solver_route(std::in_place_type<detail::cuda_route>, A, opts, cp, F);
#elif defined(APXCHOL_USE_METAL)
        if (route == solve_backend::gpu) {
            const auto& device = detail::metal::status();
            if (!device.ok)
                throw std::runtime_error("apxchol: the Metal backend is unavailable: " + device.error);
            if (F && opts.factor_opts.experimental.cpu_schedule != cpu_sptrsv_schedule::automatic)
                throw std::invalid_argument("Metal cannot use a CPU triangular-solve schedule");
            return detail::solver_route(std::in_place_type<detail::metal_route>, A, opts, cp, F);
        }
#else
        (void)route;
#endif
        return detail::solver_route(std::in_place_type<detail::cpu_route>, A, opts, cp, F);
    }

    impl(detail::csc_ref<double> A, const solve_options& opts,
         detail::checkpoint* cp, factorization* F)
        : route(make_route(A, opts, cp, F)), operator_entries(static_cast<std::int64_t>(A.nnz())),
          num_threads(opts.num_threads), timings(cp) {
#if defined(APXCHOL_USE_CUDA)
        if (const auto* gpu = std::get_if<detail::cuda_route>(&route))
            print_sptrsv_banner<solve_backend::gpu>(gpu->trsv().fp16(), opts.factor_opts.verbose);
#endif
    }

    impl(const impl& other)
        : route(copy_route(other.route)), operator_entries(other.operator_entries),
          num_threads(other.num_threads),
          setup_seconds(other.setup_seconds) {}

    std::int64_t rows() const {
        return std::visit([](const auto& owner) {
            return static_cast<std::int64_t>(owner.factor().perm.size());
        }, route);
    }

    // The only argument checks and panel loop of every solve: the routes
    // trust what reaches them. tol and max_iter are resolved. Returns the
    // wall-clock seconds since start, taken when the public call began.
    double solve(detail::const_matrix_ref B, detail::matrix_ref X,
                 const detail::const_matrix_ref* X0,
                 std::span<solve_result> results, double tol, int max_iter,
                 std::chrono::steady_clock::time_point start) {
        const auto scope = calls.enter();
        const detail::thread_budget threads(num_threads);
        check_matrix_views(B, X, rows(), X0);
        if (!detail::vec::all_finite_matrix(B) || (X0 && !detail::vec::all_finite_matrix(*X0)))
            throw std::invalid_argument("apxchol: RHS and initial guess must be finite");
        std::visit([&](auto& owner) {
            for_panels(B.cols(), owner.solve_widths, [&](std::int64_t first, int width) {
                std::optional<detail::const_matrix_ref> initial;
                if (X0) initial.emplace(X0->middle_cols(first, width));
                owner.solve_columns(B.middle_cols(first, width), X.middle_cols(first, width),
                                    initial ? &*initial : nullptr,
                                    results.subspan(first, width), timings, tol, max_iter);
            });
        }, route);
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    void apply(detail::const_matrix_ref R, detail::matrix_ref Z) {
        const auto scope = calls.enter();
        const detail::thread_budget threads(num_threads);
        check_matrix_views(R, Z, rows());
        if (!detail::vec::all_finite_matrix(R))
            throw std::invalid_argument("apxchol: preconditioner input must be finite");
        std::visit([&](auto& owner) {
            for_panels(R.cols(), owner.apply_widths, [&](std::int64_t first, int width) {
                owner.apply_columns(R.middle_cols(first, width), Z.middle_cols(first, width));
            });
        }, route);
    }
};

const detail::solver_route& detail::solver_access::route(const solver& owner) {
    return owner.impl_->route;
}

void detail::solver_access::save(const std::filesystem::path& path, const solver& owner) {
    const thread_budget threads(owner.impl_->num_threads);
    const auto F = std::visit([](const auto& backend) { return backend.installed_factor(); },
                              owner.impl_->route);
    const auto A = std::visit([](const auto& backend) { return backend.original_operator(); },
                              owner.impl_->route);
    save_solver_data(path, A.ref(), F, owner.impl_->operator_entries);
}

solver detail::solver_access::make(csc_ref<double> A, const solve_options& opts,
                                   checkpoint* timings) {
    return solver(A, nullptr, opts, timings);
}

solver detail::solver_access::make(csc_ref<double> A, factorization F, const solve_options& opts) {
    return solver(A, &F, opts, nullptr);
}

void detail::solver_access::finish_load(solver& owner, std::int64_t entries,
                                        std::chrono::steady_clock::time_point started) {
    owner.impl_->operator_entries = entries;
    set_setup_start(owner, started);
}

void detail::solver_access::set_setup_start(solver& owner,
                                            std::chrono::steady_clock::time_point started) {
    owner.impl_->setup_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

detail::csc_matrix<double> detail::cpu_route::original_operator() const {
    if (op_->fp32) return archive_operator_csc(op_->Lrm_f.ref());
    return archive_operator_csc(op_->Lrm.ref());
}

#if defined(APXCHOL_USE_CUDA)
detail::csc_matrix<double> detail::cuda_route::original_operator() const {
    static_assert(sizeof(node_index) == sizeof(int), "GPU routes require 32-bit vertices");
    const auto& pcg = gpcg_;
    const cuda_device_scope device(pcg.device_, cuda_device_scope::checked{});
    const auto& op = *pcg.op_;
    const int n = static_cast<int>(pcg.n_), nnz = op.matrix.nnz;
    csc_matrix<double> stored(static_cast<node_index>(n), static_cast<edge_index>(nnz));
    std::vector<node_index> inverse(n);
    // The device keeps 32-bit offsets; 64_EDGES builds widen them here.
    std::vector<int> offsets(static_cast<std::size_t>(n) + 1);
    check_cuda(cudaMemcpy(offsets.data(), op.matrix.row_ptr, offsets.size() * sizeof(int), cudaMemcpyDeviceToHost), "download archive offsets");
    std::copy(offsets.begin(), offsets.end(), stored.ptr());
    check_cuda(cudaMemcpy(stored.idx(), op.matrix.col_idx, static_cast<std::size_t>(nnz) * sizeof(int), cudaMemcpyDeviceToHost), "download archive indices");
    if (op.matrix.fp32) {
        std::vector<float> narrow(nnz);
        check_cuda(cudaMemcpy(narrow.data(), op.matrix.values_f32, narrow.size() * sizeof(float), cudaMemcpyDeviceToHost), "download archive values");
        std::copy(narrow.begin(), narrow.end(), stored.val());
    } else {
        check_cuda(cudaMemcpy(stored.val(), op.matrix.values_f64, static_cast<std::size_t>(nnz) * sizeof(double), cudaMemcpyDeviceToHost), "download archive values");
    }
    // CUDA stores rows in original order and only columns in factor order.
    for (int i = 0; i < n; ++i) inverse[op.perm[i]] = static_cast<node_index>(i);
    for (edge_index p = 0; p < stored.nnz(); ++p) stored.idx()[p] = inverse[stored.idx()[p]];
    return archive_operator_csc(stored.ref());
}
#endif

namespace {

// A solver for a public view; setup_seconds includes checking the view.
template<class Index>
solver from_view(basic_csc_view<Index> A, factorization* F, const solve_options& opts) {
    const auto start = std::chrono::steady_clock::now();
    const detail::csc_input input(A);
    solver owner = F ? detail::solver_access::make(input, std::move(*F), opts)
                     : detail::solver_access::make(input, opts);
    detail::solver_access::set_setup_start(owner, start);
    return owner;
}

} // namespace

solver::solver(csc_view A, const solve_options& opts) : solver(from_view(A, nullptr, opts)) {}
solver::solver(csc_view64 A, const solve_options& opts) : solver(from_view(A, nullptr, opts)) {}
solver::solver(csc_view A, factorization F, const solve_options& opts)
    : solver(from_view(A, &F, opts)) {}
solver::solver(csc_view64 A, factorization F, const solve_options& opts)
    : solver(from_view(A, &F, opts)) {}
solver::solver(const detail::csc_ref<double>& A, factorization* F,
               const solve_options& opts, detail::checkpoint* timings) {
    const auto start = std::chrono::steady_clock::now();
    impl_ = std::make_unique<impl>(A, opts, timings, F);
    impl_->setup_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
solver::~solver() = default;
solver::solver(solver&&) noexcept = default;
solver& solver::operator=(solver&&) noexcept = default;
solver::solver(const solver& other) : impl_(std::make_unique<impl>(*other.impl_)) {}
solver& solver::operator=(const solver& other) {
    if (this != &other) impl_ = std::make_unique<impl>(*other.impl_);
    return *this;
}

namespace {

// A vector of n elements as a one-column view.
detail::const_matrix_ref column(std::span<const double> v) {
    return {v.data(), static_cast<std::int64_t>(v.size())};
}
detail::matrix_ref column(std::span<double> v) {
    return {v.data(), static_cast<std::int64_t>(v.size())};
}

void check_rhs_length(std::span<const double> b, std::int64_t rows) {
    if (std::cmp_not_equal(b.size(), rows))
        throw std::invalid_argument("apxchol: RHS length does not match the operator");
}

} // namespace

solve_result solver::solve(std::span<const double> b, std::optional<double> requested_tol,
                           std::optional<int> requested_max_iter, std::span<const double> x0) {
    const auto start = std::chrono::steady_clock::now();
    const auto [tol, max_iter] = resolve_solve_parameters(requested_tol, requested_max_iter);
    check_rhs_length(b, rows());
    solve_result result;
    result.x.resize(static_cast<std::size_t>(rows()));
    const auto initial = column(x0);
    result.solve_seconds = impl_->solve(column(b), column(std::span<double>(result.x)),
                                        x0.empty() ? nullptr : &initial, {&result, 1},
                                        tol, max_iter, start);
    return result;
}

solve_result solver::solve(std::span<const double> b, std::span<double> x,
                           std::optional<double> requested_tol, std::optional<int> requested_max_iter,
                           std::span<const double> x0) {
    const auto start = std::chrono::steady_clock::now();
    const auto [tol, max_iter] = resolve_solve_parameters(requested_tol, requested_max_iter);
    solve_result result;
    const auto initial = column(x0);
    result.solve_seconds = impl_->solve(column(b), column(x),
                                        x0.empty() ? nullptr : &initial, {&result, 1},
                                        tol, max_iter, start);
    return result;
}

batch_solve_result solver::solve(const_matrix_view B, matrix_view X,
                                 std::optional<double> requested_tol,
                                 std::optional<int> requested_max_iter,
                                 const_matrix_view X0) {
    const auto start = std::chrono::steady_clock::now();
    const auto [tol, max_iter] = resolve_solve_parameters(requested_tol, requested_max_iter);
    const auto rhs = detail::dense_input(B);
    const auto out = detail::dense_input(X);
    std::optional<detail::const_matrix_ref> initial;
    if (X0.data) initial.emplace(detail::dense_input(X0));
    batch_solve_result result;
    result.columns.resize(static_cast<std::size_t>(rhs.cols()));
    result.solve_seconds = impl_->solve(rhs, out, initial ? &*initial : nullptr, result.columns,
                                        tol, max_iter, start);
    return result;
}

std::vector<double> solver::apply(std::span<const double> r) {
    if (std::cmp_not_equal(r.size(), rows()))
        throw std::invalid_argument("apxchol: preconditioner input length does not match the operator");
    std::vector<double> z(r.size());
    impl_->apply(column(r), column(std::span<double>(z)));
    return z;
}

void solver::apply(const_matrix_view R, matrix_view Z) {
    impl_->apply(detail::dense_input(R), detail::dense_input(Z));
}

factorization solver::factor() const {
    const detail::thread_budget threads(impl_->num_threads);
    return std::visit([](const auto& owner) { return owner.installed_factor(); }, impl_->route);
}
std::int64_t solver::rows() const { return impl_->rows(); }
std::int64_t solver::operator_nnz() const { return impl_->operator_entries; }
solve_backend solver::backend() const {
    return std::holds_alternative<detail::cpu_route>(impl_->route)
        ? solve_backend::cpu : solve_backend::gpu;
}
factor_storage_type solver::factor_storage() const {
    return std::visit([](const auto& owner) { return owner.factor_storage(); }, impl_->route);
}
double solver::setup_seconds() const { return impl_->setup_seconds; }

namespace {

template<class Index>
solve_result solve_once(basic_csc_view<Index> A, std::span<const double> b,
                        std::optional<double> tol, std::optional<int> max_iter,
                        std::span<const double> x0, const solve_options& opts) {
    // Reject invalid limits and lengths before paying for setup.
    resolve_solve_parameters(tol, max_iter);
    check_rhs_length(b, A.n);
    if (!x0.empty() && std::cmp_not_equal(x0.size(), A.n))
        throw std::invalid_argument("apxchol: initial guess length does not match the operator");
    solver owner(A, opts);
    return owner.solve(b, tol, max_iter, x0);
}

} // namespace

solve_result solve(csc_view A, std::span<const double> b, std::optional<double> tol,
                   std::optional<int> max_iter, std::span<const double> x0,
                   const solve_options& opts) {
    return solve_once(A, b, tol, max_iter, x0, opts);
}

solve_result solve(csc_view64 A, std::span<const double> b, std::optional<double> tol,
                   std::optional<int> max_iter, std::span<const double> x0,
                   const solve_options& opts) {
    return solve_once(A, b, tol, max_iter, x0, opts);
}

} // namespace apxchol
