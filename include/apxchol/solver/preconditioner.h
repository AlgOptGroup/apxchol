#pragma once
#include "apxchol/checkpoint.h"
#include "apxchol/solver/detail/gpu_diagnostics.h"
#include "apxchol/env_knobs.h"
#include "apxchol/solver/cuda_context.h"
#include "apxchol/solver/factorization.h"
#include "apxchol/solver/detail/setup_route.h"
#include "apxchol/solver/sptrsv/omp.h"
#include <type_traits>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(APXCHOL_USE_CUDA)
#include "apxchol/solver/sptrsv/cuda.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace apxchol {

// ── Deterministic parallel-reduction scaffolding ─────────────────────────────
// Shared by the fused vector passes of the preconditioner (below) and of the
// PCG loop (src/solve.cpp). Every reduction those passes perform follows one
// scheme so its fp summation order NEVER depends on scheduling:
//   * the index range is cut into the contiguous per-thread chunks that
//     `schedule(static)` (no chunk size) would produce -- a fixed partition
//     for a given (n, thread count);
//   * each thread accumulates its chunk into private registers (a 4-way
//     accumulator split where the pass has no other work to hide the FMA
//     dependency chain), then writes ONE partial into `part[tid * kStride]`;
//   * the partials are summed serially in thread order 0..nt-1.
// No `reduction(+:...)` clause anywhere: libgomp combines those in thread
// COMPLETION order, which varies run to run. Results are therefore
// bit-identical run-to-run for a fixed thread count (they do differ across
// thread counts -- the chunk boundaries move -- exactly like every other
// parallel pass in the solver).
namespace detail {

/// Per-thread stride (in doubles) of the partial-sum buffer: one cache line
/// per thread, so the single end-of-chunk write never false-shares.
inline constexpr int kPartStride = 8;

/// Contiguous chunk of [0, n) owned by thread `tid` of `nt`: the first
/// (n mod nt) threads get one extra element, exactly like schedule(static).
inline std::pair<Eigen::Index, Eigen::Index>
static_chunk(Eigen::Index n, int tid, int nt) noexcept {
    const Eigen::Index base = n / nt, rem = n % nt;
    const Eigen::Index lo = static_cast<Eigen::Index>(tid) * base
                          + std::min<Eigen::Index>(tid, rem);
    return {lo, lo + base + (tid < rem ? 1 : 0)};
}

/// Number of doubles the partial buffer must hold: omp_get_max_threads() is,
/// per the OpenMP spec, an upper bound on the team size of any parallel
/// region encountered next using cpu_solve_threads -- so `kPartStride`
/// doubles per that many threads always suffices. Callers keep the buffer
/// as reusable workspace so repeated solves stay allocation-free.
inline std::size_t part_capacity() noexcept {
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
#endif
    return static_cast<std::size_t>(nt) * kPartStride;
}

/// Thread id / team size inside a parallel region (0 / 1 for a serial team).
inline void omp_ids(int& tid, int& nt) noexcept {
    tid = 0; nt = 1;
#ifdef _OPENMP
    tid = omp_get_thread_num(); nt = omp_get_num_threads();
#endif
}

/// Serial thread-order sum of `nt` partials at slot `slot` (< stride).
inline double reduce_parts(const double* part, int nt, int slot = 0, int stride = kPartStride) noexcept {
    double s = 0.0;
    for (int t = 0; t < nt; ++t) s += part[static_cast<std::size_t>(t) * stride + slot];
    return s;
}

/// Deterministic parallel Σ v[i], i in [0, n). `part` must hold
/// part_capacity() doubles. Fixed 4-way accumulator split per chunk.
template<int Columns>
std::array<double, Columns> det_sum_columns(const double* v, Eigen::Index n, double* part) {
    static_assert(Columns > 0 && Columns <= kPartStride);
    int nt_used = 1;
    const int workers = cpu_solve_threads(n);
    #pragma omp parallel if(workers > 1) num_threads(workers)
    {
        int tid, nt; omp_ids(tid, nt);
        if (tid == 0) nt_used = nt;
        const auto [lo, hi] = static_chunk(n, tid, nt);
        std::array<double, Columns> s0{}, s1{}, s2{}, s3{};
        Eigen::Index i = lo;
        for (; i + 4 <= hi; i += 4)
            for (int c = 0; c < Columns; ++c) {
                s0[c] += v[(i + 0) * Columns + c];
                s1[c] += v[(i + 1) * Columns + c];
                s2[c] += v[(i + 2) * Columns + c];
                s3[c] += v[(i + 3) * Columns + c];
            }
        std::array<double, Columns> sum{};
        for (int c = 0; c < Columns; ++c)
            sum[c] = (s0[c] + s1[c]) + (s2[c] + s3[c]);
        for (; i < hi; ++i)
            for (int c = 0; c < Columns; ++c) sum[c] += v[i * Columns + c];
        for (int c = 0; c < Columns; ++c)
            part[static_cast<std::size_t>(tid) * kPartStride + c] = sum[c];
    }
    std::array<double, Columns> sum{};
    for (int c = 0; c < Columns; ++c) sum[c] = reduce_parts(part, nt_used, c);
    return sum;
}

} // namespace detail

/// Eigen-compatible preconditioner using approximate Cholesky factorization.
///
/// Inherits SparseSolverBase for the standard solve() / _solve_impl() pattern.
/// Use with Eigen's PCG:
///   Eigen::ConjugateGradient<SpMat, Eigen::Lower, apxchol::apx_cholesky> cg;
///   cg.compute(L);   // triggers apx_cholesky::compute(L) internally
///   x = cg.solve(b);
///
namespace detail {
template<class Trsv, setup_route Route>
class basic_apx_cholesky : public Eigen::SparseSolverBase<basic_apx_cholesky<Trsv, Route>> {
    using Base = Eigen::SparseSolverBase<basic_apx_cholesky<Trsv, Route>>;
#if defined(APXCHOL_USE_CUDA)
    static constexpr bool on_gpu = std::is_same_v<Trsv, cuda_sptrsv>;
#else
    static constexpr bool on_gpu = false;
#endif
    static_assert(on_gpu == (Route == setup_route::gpu));

protected:
    using Base::m_isInitialized;

public:
    using Scalar      = double;
    using RealScalar  = double;
    // Eigen-facing index type for the SolverBase interface. Kept signed `int`
    // (Eigen requires a signed StorageIndex); independent of apxchol's internal
    // unsigned node_index/edge_index, which the factor now owns directly.
    using StorageIndex = int;
    enum {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic
    };

    basic_apx_cholesky() = default;

    const factorization& factor() const { return F_; }
    const factor_options& options() const { return opts_; }
    void set_options(const factor_options& opts) { opts_ = opts; }
    void set_checkpoint(checkpoint* cp) const { cp_ = cp; }
    /// Keep the factor's row/value arrays alive after SpTRSV setup (they are
    /// otherwise freed to save one factor-sized copy).  Needed by callers that
    /// want to read factor().L afterwards, e.g. the Python binding's L/D/P
    /// export, or to pass the factor on to set_factor() elsewhere.  Must be
    /// set before factorize()/compute()/set_factor().
    void set_keep_factor(bool keep) { keep_factor_ = keep; }

    /// No-op: approximate Cholesky has no separate symbolic analysis phase.
    basic_apx_cholesky& analyzePattern(const auto&) {
        m_analysisIsOk = true;
        m_isInitialized = true;
        m_info = Eigen::Success;
        return *this;
    }

    /// Compute the approximate Cholesky factorization of A.
    basic_apx_cholesky& factorize(const auto& A) {
        eigen_assert(m_analysisIsOk && "analyzePattern() should be called first");
        // Only the internal device specialization may initialize CUDA. CPU
        // factorization and Eigen/CPU callers never touch its context, even in
        // a CUDA-enabled build. Preserve the device setup's existing prewarm.
#if defined(APXCHOL_USE_CUDA)
        [[maybe_unused]] const auto factorize_install_begin = detail::gpu_setup_diagnostic_clock::now();
#endif
        if constexpr (on_gpu) cuda_ctx::prewarm();
        n_ = A.rows();
        F_ = detail::factorize_for_solver(A, opts_, cp_, Route, keep_factor_);
        install_factor();
#if defined(APXCHOL_USE_CUDA)
        if constexpr (on_gpu) {
            // Complete the adopted factor's queued setup work before returning.
            const auto setup_completion = cudaDeviceSynchronize();
            if (setup_completion != cudaSuccess)
                throw std::runtime_error(std::string("GPU setup completion: ") +
                                         cudaGetErrorString(setup_completion));
            if (cp_) { cp_->descend("setup"); (*cp_)("gpu_setup_complete"); cp_->ascend(); }
            if (detail::gpu_setup_diagnostics()) {
                // Host API duration; device-adopted setup additionally completed its work.
                const double factorize_install_wall_s = std::chrono::duration<double>(
                    detail::gpu_setup_diagnostic_clock::now() - factorize_install_begin).count();
                const auto& drop = trsv_.drop_stats();
                std::fprintf(stderr,
                    "[gpu-setup-receipt] diagnostics=%s factorize_install_wall_s=%.17g "
                    "n=%lld rounds=%zu raw_factor_nnz=%zu stored_nnz=%llu adopted=%d "
                    "fp16=%d host_factor_array_bytes=%zu adoption_download_bytes=%zu "
                    "factor_drop_rel=%.17g dropped_threshold=%llu dropped_flush=%llu\n",
                    detail::gpu_setup_diagnostics() ? "enabled" : "disabled",
                    factorize_install_wall_s, static_cast<long long>(n_), F_.rounds.size(),
                    static_cast<std::size_t>(F_.L.nonZeros()),
                    static_cast<unsigned long long>(trsv_.stored_nnz()),
                    trsv_.adopted_device_factor() ? 1 : 0, trsv_.fp16() ? 1 : 0,
                    F_.L.inner_.size() * sizeof(node_index) + F_.L.vals_.size() * sizeof(factor_value_t),
                    trsv_.adoption_host_download_bytes(), drop.rel,
                    static_cast<unsigned long long>(drop.dropped_threshold),
                    static_cast<unsigned long long>(drop.dropped_flush));
            }
        }
#endif
        return *this;
    }

    /// Install an externally computed factorization (e.g. from a factorize()
    /// call with a custom eliminator or partitioner) and run the SpTRSV setup.
    /// After this the preconditioner is ready for _solve_impl / Eigen CG.
    /// The factorization must still own its row/value arrays — a factor read
    /// back from a default-configured solver has had them released (see
    /// set_keep_factor); pass a freshly computed factorization instead.
    basic_apx_cholesky& set_factor(factorization F) {
        if (F.L.nonZeros() > 0 && F.L.vals_.empty())
            throw std::invalid_argument(
                "apx_cholesky::set_factor: factorization values were released; "
                "pass a freshly computed factorization");
        n_ = static_cast<Eigen::Index>(F.L.rows());
        F_ = std::move(F);
        m_analysisIsOk = true;
        m_isInitialized = true;
        install_factor();
        return *this;
    }

private:
    /// Shared tail of factorize()/set_factor(): SpTRSV setup from F_.
    void install_factor() {
#if defined(APXCHOL_USE_CUDA)
        if constexpr (Route == setup_route::gpu) {
            // Reject inadmissible factors before installation can initialize
            // CUDA or upload host arrays.
            if (!F_.research_device_factor ||
                F_.research_device_factor.use_count() != 1 ||
                F_.research_device_factor->empty())
                throw std::invalid_argument(
                    "GPU route requires a unique nonempty device-owned factor");
        }
#endif
        scratch_.resize(n_);
        if constexpr (!on_gpu) {
            // Second scratch buffer (CPU path only, +8n bytes RSS): receives the
            // transpose-solve output so the final unpermute gathers straight from
            // it, eliminating the serial full-n `scratch_ = x` staging copy that
            // _solve_impl used to pay per PCG iteration.
            scratch2_.resize(n_);
        }

        // sptrsv setup (CPU schedules / GPU dataflow preparation) is a real ~900ms
        // cost on IPM-scale matrices. Track it as part of "setup" so the
        // bench's setup_time / solve_time split is honest.
        if (cp_) { cp_->descend("setup"); cp_->tick(); }
        if constexpr (!on_gpu) {
            // Laplacians leave the grounded last row/column unused.
            const node_index factor_dim = static_cast<node_index>(F_.sddm ? n_ : n_ - 1);
            // Round-as-level: hand the SpTRSV the per-round IS-column boundaries so it
            // builds level sets straight off the elimination rounds (same-round IS
            // columns are mutually independent -> one level) instead of two O(nnz)
            // topological depth scans. Each region is one factor column placed in round
            // order; trailing residual-peel columns get sequential levels inside the
            // SpTRSV. The backend uses these by default (round-as-level is a -6..-18%
            // solve win); APXCHOL_ROUND_LEVELS=0 forces the topological scan instead.
            if (!F_.rounds.empty()) {
                std::vector<node_index> bounds;
                bounds.reserve(F_.rounds.size() + 1);
                bounds.push_back(0);
                node_index acc = 0;
                for (const auto& r : F_.rounds) {
                    acc += static_cast<node_index>(r.is_size);
                    if (acc > factor_dim) acc = static_cast<node_index>(factor_dim);
                    bounds.push_back(acc);
                }
                trsv_.set_round_bounds(std::move(bounds));
            }
            // Hand the factor over unless the caller wants to read it afterwards:
            // the CPU SpTRSV then releases F_.L's row/value arrays at the first
            // point it no longer reads them (Laplacian path: right after its L11
            // copy), so its setup transient does not sit on a dead copy of the
            // factor. The release below then finds nothing left to free.
            if (keep_factor_) trsv_.setup(F_.L, factor_dim, opts_.factor_storage);
            else              trsv_.setup_consuming(F_.L, factor_dim, opts_.factor_storage);
        }
#if defined(APXCHOL_USE_CUDA)
        else {
            // Device init, on its OWN checkpoint label. CUDA creates the primary
            // context lazily inside the first CUDA call, which is trsv_.setup()
            // below: until 2026-08-20 that fixed per-process cost (~100-135 ms on
            // an RTX 4090 Laptop, ~715 ms on a GH200) was silently counted as
            // sptrsv_setup, inflating every published GPU setup number -- worst at
            // small n, where it was the majority of the reported time. Now
            // factorize() prewarms it on a helper thread and we pay only the
            // REMAINING wait here, under "cuda_init"; sptrsv_setup below reports
            // its own work only. NOTE: this changes the meaning of published GPU
            // setup numbers -- the benchmark cells need regenerating.
            const double cuda_init_s = cuda_ctx::ensure_context();
            if (cp_) (*cp_)("cuda_init");
            if (detail::gpu_setup_diagnostics() && std::getenv("APXCHOL_SPTRSV_SETUP_TRACE"))   // same knob as the stage trace below
                std::fprintf(stderr, "[sptrsv-setup gpu] %-22s %8.2f ms  (context creation %.2f ms)\n",
                             "cuda_init", cuda_init_s * 1e3, cuda_ctx::context_seconds() * 1e3);
            trsv_.setup_adopting_device_factor_for_research(
                std::move(*F_.research_device_factor), opts_.factor_storage);
            F_.research_device_factor.reset();
        }
#endif
        if (cp_) { (*cp_)("sptrsv_setup"); cp_->ascend(); }

        // The SpTRSV (and the CUDA path) have copied the factor into their own
        // CSR/CSC; the PCG loop only uses trsv_, never F_.L. Free the factor's
        // big row/value arrays -> one fewer full copy of the factor held during
        // the solve (~nnz*12 B; e.g. ~1.4 GB on com-LiveJournal). nonZeros() (the
        // fill stat) still works via the retained column pointers. (On the CPU
        // path setup_consuming has normally done this already; idempotent.)
        // set_keep_factor(true) skips the free for callers that export L.
        if (!keep_factor_)
            F_.L.release_values();

        m_factorizationIsOk = true;
        m_info = Eigen::Success;
    }

public:

    /// Shortcut for analyzePattern() + factorize().
    basic_apx_cholesky& compute(const auto& A) {
        analyzePattern(A);
        factorize(A);
        return *this;
    }

    /// Restart the per-solve application counter of the center-k centring
    /// schedule (APXCHOL_GROUND=center-k, the default; see env_knobs.h). Call
    /// at the start of each PCG solve so the schedule (every K-th application
    /// centres) is identical for every solve on this factor -- repeated
    /// solves then reproduce bit-identical results. Irrelevant for K = 1
    /// (every application centres) and on the SDDM path.
    void reset_apply_count() const noexcept { apply_count_ = 0; }

    Eigen::Index rows() const { return n_; }
    Eigen::Index cols() const { return n_; }
    Eigen::ComputationInfo info() const { return m_info; }

    Trsv& trsv() { return trsv_; }
    const Trsv& trsv() const { return trsv_; }

    /// Apply M^{-1} to rhs via approximate Cholesky (Eigen entry point; also
    /// what apply()/Eigen::ConjugateGradient hit).
    ///
    /// Laplacian path (rank n-1): center → P → L^{-1} → L^{-T} → P^{-1} → center,
    ///   where the two centring passes run on every K-th application only
    ///   (APXCHOL_GROUND=center-k, K = APXCHOL_CENTER_K, default 10; see
    ///   env_knobs.h and take_center_()). K = 1 centres every application.
    /// SDDM path     (rank n):   P → L^{-1} → L^{-T} → P^{-1}.
    ///
    /// `b` may be any dense column expression (a plain VectorXd binds without a
    /// copy); `x` must be a contiguous dense vector (its .data() is handed to
    /// the SpTRSV, as before).
    void _solve_impl(const auto& b, auto& x) const {
        eigen_assert(m_factorizationIsOk && "factorize() should be called first");
        const Eigen::Ref<const Eigen::VectorXd> bv(b);
        apply_columns<1>(bv.data(), x.data());
    }

    /// PCG-loop application: z = M^{-1} r, with the reductions the outer loop
    /// needs FUSED into the passes that already stream the vectors:
    ///   * r·z is accumulated in the final unpermute gather (the pass that
    ///     writes z) -- returned;
    ///   * Laplacian path only: the input centering uses `r_sum` (= Σ r,
    ///     which the PCG update pass produces for free) instead of a mean
    ///     pass over r, and the output re-centering is one deterministic
    ///     reduction over the permuted solve output followed by the same
    ///     gather (subtract-mean + write z + r·z in one pass) -- so z is the
    ///     centered vector exactly as _solve_impl produces it.
    /// `r_sum` is ignored on the SDDM path and on the Laplacian path's
    /// non-centring applications (center-k schedule: only every K-th
    /// application of a solve centres, the others skip both mean passes; see
    /// take_center_()). Reductions are deterministic for a fixed thread count
    /// (see detail:: scaffolding at the top of the file).
    double apply_fused(const double* r, double r_sum, double* z) const {
        return apply_fused<1>(r, {r_sum}, z)[0];
    }

    template<int Columns>
    std::array<double, Columns> apply_fused(
            const double* r, const std::array<double, Columns>& r_sum, double* z) const {
        eigen_assert(m_factorizationIsOk && "factorize() should be called first");
        ensure_part_();
        const bool center = !F_.sddm && take_center_();
        std::array<double, Columns> rz{};
        std::array<bool, Columns> centers{};
        centers.fill(center);
        apply_core<Columns>(r, r_sum, centers, z, &rz);
        return rz;
    }

    template<int Columns>
    void apply_columns(const double* b, double* z) const {
        ensure_part_();
        std::array<bool, Columns> centers{};
        for (bool& center : centers) center = !F_.sddm && take_center_();
        const auto sums = std::any_of(centers.begin(), centers.end(), [](bool c) { return c; })
            ? detail::det_sum_columns<Columns>(b, n_, part_.data())
            : std::array<double, Columns>{};
        apply_core<Columns>(b, sums, centers, z, nullptr);
    }

private:
    void ensure_part_() const {
        const std::size_t need = detail::part_capacity();
        if (part_.size() < need) part_.resize(need);
    }

    /// Laplacian-path centring schedule (APXCHOL_GROUND=center-k, see
    /// env_knobs.h): advances the per-solve application counter and returns
    /// whether THIS application centres -- applications K, 2K, ... (1-based
    /// since the last reset_apply_count()) do, the others skip both mean
    /// passes; K = 1 centres every application. In any other grounding mode
    /// (reg: make_graph grounds the matrix, so it classifies SDDM and never
    /// reaches this path) a factor that is nevertheless Laplacian centres on
    /// every application. Call exactly once per Laplacian-path application.
    bool take_center_() const noexcept {
        const auto& k = detail::env_knobs::get();
        if (k.ground != detail::grounding_kind::center_k) return true;
        return (++apply_count_ % static_cast<std::uint64_t>(k.center_k)) == 0;
    }

    /// Shared body of _solve_impl / apply_fused: z = M^{-1} b.
    /// `b_sum` = Σ b and `centers` selects columns that need the two
    /// Laplacian mean passes (both ignored for SDDM). If `rz_out` is non-null
    /// the final unpermute gather also accumulates b·z into it.
    template<int Columns>
    void apply_core(const double* b, const std::array<double, Columns>& b_sum,
                    const std::array<bool, Columns>& centers, double* z, std::array<double, Columns>* rz_out) const {
        static_assert(Columns > 0 && Columns <= detail::kPartStride);
        static_assert(!on_gpu || Columns == 1);
        if (cp_) { cp_->descend("solve"); cp_->tick(); }
        const node_index* P = F_.perm.data();
        const Eigen::Index n = n_;
        const int workers = detail::cpu_solve_threads(n);
        double* part = part_.data();
        scratch_.resize(n * Columns);
        if constexpr (!on_gpu) scratch2_.resize(n * Columns);

        std::array<double, Columns> mean{};
        for (int c = 0; c < Columns; ++c)
            if (centers[c] && n > 0)
                mean[c] = b_sum[c] / static_cast<double>(n);

        // perm[original] is the position in the factor; RHS columns are contiguous.
        #pragma omp parallel for schedule(static) if(workers > 1) num_threads(workers)
        for (Eigen::Index v = 0; v < n; ++v)
            for (int c = 0; c < Columns; ++c)
                z[std::size_t(P[v]) * Columns + c] = b[v * Columns + c] - (centers[c] ? mean[c] : 0.0);
        if (cp_) (*cp_)("permute");

        double* out;
#if defined(APXCHOL_USE_CUDA)
        if constexpr (on_gpu) {
            trsv_.solve_LLt(z, scratch_.data());
            out = scratch_.data();
        } else
#endif
        {
            trsv_.template forward_solve<Columns>(z, scratch_.data());
            if (cp_) (*cp_)("forward");
            trsv_.template transpose_solve<Columns>(scratch_.data(), scratch2_.data());
            out = scratch2_.data();
        }
        if (cp_) (*cp_)("back");
        if (!F_.sddm)
            std::fill_n(out + (n - 1) * Columns, Columns, 0.0);

        std::array<double, Columns> shift{};
        if (std::any_of(centers.begin(), centers.end(), [](bool c) { return c; })) {
            shift = detail::det_sum_columns<Columns>(out, n, part);
            for (int c = 0; c < Columns; ++c)
                shift[c] = centers[c] ? shift[c] / static_cast<double>(n) : 0.0;
        }
        int nt_used = 1;
        #pragma omp parallel if(workers > 1) num_threads(workers)
        {
            int tid, nt; detail::omp_ids(tid, nt);
            if (tid == 0) nt_used = nt;
            const auto [lo, hi] = detail::static_chunk(n, tid, nt);
            std::array<std::array<double, Columns>, 4> sums{};
            auto gather = [&](Eigen::Index v, auto& sum) {
                for (int c = 0; c < Columns; ++c) {
                    const double value = out[std::size_t(P[v]) * Columns + c] - shift[c];
                    z[v * Columns + c] = value;
                    sum[c] += b[v * Columns + c] * value;
                }
            };
            Eigen::Index v = lo;
            for (; v + 4 <= hi; v += 4) {
                std::array<std::array<double, Columns>, 4> values{};
                for (int k = 0; k < 4; ++k)
                    for (int c = 0; c < Columns; ++c)
                        values[k][c] = out[std::size_t(P[v + k]) * Columns + c] - shift[c];
                for (int k = 0; k < 4; ++k)
                    for (int c = 0; c < Columns; ++c)
                        z[(v + k) * Columns + c] = values[k][c];
                for (int k = 0; k < 4; ++k)
                    for (int c = 0; c < Columns; ++c)
                        sums[k][c] += b[(v + k) * Columns + c] * values[k][c];
            }
            std::array<double, Columns> sum{};
            for (int c = 0; c < Columns; ++c)
                sum[c] = (sums[0][c] + sums[1][c]) + (sums[2][c] + sums[3][c]);
            for (; v < hi; ++v) gather(v, sum);
            for (int c = 0; c < Columns; ++c)
                part[static_cast<std::size_t>(tid) * detail::kPartStride + c] = sum[c];
        }
        if (rz_out)
            for (int c = 0; c < Columns; ++c)
                (*rz_out)[c] = detail::reduce_parts(part, nt_used, c);
        if (cp_) { (*cp_)(rz_out ? "unpermute+rz" : "unpermute"); cp_->ascend(); }
    }

    struct factorization F_;
    factor_options opts_;
    mutable checkpoint* cp_ = nullptr;
    bool keep_factor_ = false;
    // Laplacian-path application counter of the center-k centring schedule
    // (counts _solve_impl / apply_fused calls on the Laplacian branch since
    // the last reset_apply_count(); see take_center_()).
    mutable std::uint64_t apply_count_ = 0;
    mutable Eigen::VectorXd scratch_;
    // Per-thread partial sums for the deterministic reductions in apply_core
    // (sized to the max team on first use; reusable, so solves stay
    // allocation-free after the first call).
    mutable std::vector<double> part_;
    // Transpose-solve output staging for the CPU _solve_impl (see
    // install_factor); unused (never resized) by the device specialization.
    mutable Eigen::VectorXd scratch2_;
    Eigen::Index n_ = 0;
    Eigen::ComputationInfo m_info = Eigen::Success;
    bool m_analysisIsOk = false;
    bool m_factorizationIsOk = false;

    mutable Trsv trsv_;
};

#if defined(APXCHOL_USE_CUDA)
using gpu_preconditioner = basic_apx_cholesky<cuda_sptrsv, setup_route::gpu>;
#endif
} // namespace detail

// Eigen and cpu_solver always execute a complete CPU route, including in a
// CUDA-enabled build. GPU PCG owns its internal device preconditioner separately.
using apx_cholesky = detail::basic_apx_cholesky<omp_sptrsv, detail::setup_route::cpu>;

} // namespace apxchol
