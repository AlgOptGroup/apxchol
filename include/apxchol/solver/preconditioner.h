#pragma once
#include "apxchol/checkpoint.h"
#include "apxchol/env_knobs.h"
#include "apxchol/solver/cuda_context.h"
#include "apxchol/solver/factorization.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(APXCHOL_USE_CUDA)
#include "apxchol/solver/sptrsv/cuda.h"
#else
#include "apxchol/solver/sptrsv/omp.h"
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

/// Minimum n before a fused vector pass engages OpenMP. At or below it the
/// SAME loop runs on the encountering thread (the `if` clause disables the
/// team, omp_ids reports tid 0 / nt 1), so the sub-threshold path is
/// bit-identical to a T=1 run -- one code path, no second kernel.
///
/// 2000, the value the passes shipped with, and a CONSTANT: the
/// APXCHOL_FUSED_PARALLEL_MIN override was removed 2026-08-20 after raising
/// it was REFUTED by measurement (2026-08-19, T=16, Zen 4). At n = 299k/335k
/// (coAuthorsDBLP / com-Amazon -- the vectors are NOT cache-resident, 6 fp64
/// n-vectors alone are ~16 MB) every gated pass is faster PARALLEL --
/// update_xr 0.086 vs 0.396 ms/iter serialized, permute 0.089 vs 0.171,
/// unpermute+rz 0.096 vs 0.236, spmv+pAp 0.62 vs 4.32 -- and serializing them
/// also starves the level-set SpTRSV in between (sleeping team, forward
/// 130 -> 226 ms). Do NOT raise it to make mid-size problems "cache-friendly
/// serial".
inline constexpr Eigen::Index fused_omp_min() noexcept { return 2000; }

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
/// region encountered next without a num_threads clause -- so `kPartStride`
/// doubles per that many threads always suffices. Callers keep the buffer
/// as reusable workspace so repeated solves stay allocation-free.
inline std::size_t part_capacity() noexcept {
    int nt = 1;
#ifdef _OPENMP
    nt = omp_get_max_threads();
#endif
    return static_cast<std::size_t>(nt) * kPartStride;
}

/// Thread id / team size inside a parallel region (0 / 1 when the `if`
/// clause disabled the team, or without OpenMP).
inline void omp_ids(int& tid, int& nt) noexcept {
    tid = 0; nt = 1;
#ifdef _OPENMP
    tid = omp_get_thread_num(); nt = omp_get_num_threads();
#endif
}

/// Serial thread-order sum of `nt` partials at slot `slot` (< kPartStride).
inline double reduce_parts(const double* part, int nt, int slot = 0) noexcept {
    double s = 0.0;
    for (int t = 0; t < nt; ++t) s += part[static_cast<std::size_t>(t) * kPartStride + slot];
    return s;
}

/// Deterministic parallel Σ v[i], i in [0, n). `part` must hold
/// part_capacity() doubles. Fixed 4-way accumulator split per chunk.
inline double det_sum(const double* v, Eigen::Index n, double* part) {
    int nt_used = 1;
    #pragma omp parallel if(n > fused_omp_min())
    {
        int tid, nt; omp_ids(tid, nt);
        if (tid == 0) nt_used = nt;
        const auto [lo, hi] = static_chunk(n, tid, nt);
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        Eigen::Index i = lo;
        for (; i + 4 <= hi; i += 4) {
            s0 += v[i + 0]; s1 += v[i + 1]; s2 += v[i + 2]; s3 += v[i + 3];
        }
        double s = (s0 + s1) + (s2 + s3);
        for (; i < hi; ++i) s += v[i];
        part[static_cast<std::size_t>(tid) * kPartStride] = s;
    }
    return reduce_parts(part, nt_used);
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
/// Triangular solve backend selected at compile time:
///   - Default: OpenMP level-set parallel SpTRSV (CPU)
///   - APXCHOL_USE_CUDA: cuSPARSE GPU-accelerated SpSV
class apx_cholesky : public Eigen::SparseSolverBase<apx_cholesky> {
    using Base = Eigen::SparseSolverBase<apx_cholesky>;

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

    apx_cholesky() = default;

    const factorization& factor() const { return F_; }
    const factor_options& options() const { return opts_; }
    void set_options(const factor_options& opts) { opts_ = opts; }
    void set_storage(graph_storage s) { storage_ = s; }
    void set_checkpoint(checkpoint* cp) { cp_ = cp; }
    /// Keep the factor's row/value arrays alive after SpTRSV setup (they are
    /// otherwise freed to save one factor-sized copy).  Needed by callers that
    /// want to read factor().L afterwards, e.g. the Python binding's L/D/P
    /// export, or to pass the factor on to set_factor() elsewhere.  Must be
    /// set before factorize()/compute()/set_factor().
    void set_keep_factor(bool keep) { keep_factor_ = keep; }

    /// No-op: approximate Cholesky has no separate symbolic analysis phase.
    apx_cholesky& analyzePattern(const auto&) {
        m_analysisIsOk = true;
        m_isInitialized = true;
        m_info = Eigen::Success;
        return *this;
    }

    /// Compute the approximate Cholesky factorization of A.
    apx_cholesky& factorize(const auto& A) {
        eigen_assert(m_analysisIsOk && "analyzePattern() should be called first");
        // Kick CUDA device init onto a helper thread so it runs concurrently
        // with the elimination below (cuda_context.h; a no-op without CUDA).
        // HERE and not in a solver constructor: this is the earliest point that
        // is common to every path which then does host work before touching the
        // GPU -- cpu_solver's ctor, Eigen's cg.compute(), the Python/Octave
        // bindings -- and the whole factorization (make_graph, find_partition,
        // eliminate, assembly: ~0.5 s on the suite's big matrices) sits between
        // it and install_factor()'s first CUDA call. Starting it earlier could
        // only buy the few microseconds of set_options/analyzePattern; starting
        // it in a constructor would also spin up a context for callers that
        // construct a solver and never factor.
        cuda_ctx::prewarm();
        n_ = A.rows();
        F_ = apxchol::factorize(A, storage_, opts_, cp_);
        install_factor();
        return *this;
    }

    /// Install an externally computed factorization (e.g. from a factorize()
    /// call with a custom eliminator or partitioner) and run the SpTRSV setup.
    /// After this the preconditioner is ready for _solve_impl / Eigen CG.
    /// The factorization must still own its row/value arrays — a factor read
    /// back from a default-configured solver has had them released (see
    /// set_keep_factor); pass a freshly computed factorization instead.
    apx_cholesky& set_factor(factorization F) {
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
        scratch_.resize(n_);
#if !defined(APXCHOL_USE_CUDA)
        // Second scratch buffer (CPU path only, +8n bytes RSS): receives the
        // transpose-solve output so the final unpermute gathers straight from
        // it, eliminating the serial full-n `scratch_ = x` staging copy that
        // _solve_impl used to pay per PCG iteration.
        scratch2_.resize(n_);
#endif

        // For Laplacians (rank n-1) the last row/col is unused.
        // For SDDM (full rank) we use the complete n×n factor.
        const node_index factor_dim = static_cast<node_index>(F_.sddm ? n_ : n_ - 1);
        // sptrsv setup (level sets / cuSPARSE analysis) is a real ~900ms
        // cost on IPM-scale matrices. Track it as part of "setup" so the
        // bench's setup_time / solve_time split is honest.
        if (cp_) { cp_->descend("setup"); cp_->tick(); }
#if !defined(APXCHOL_USE_CUDA)
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
#endif
#if !defined(APXCHOL_USE_CUDA)
        // Hand the factor over unless the caller wants to read it afterwards:
        // the CPU SpTRSV then releases F_.L's row/value arrays at the first
        // point it no longer reads them (Laplacian path: right after its L11
        // copy), so its setup transient does not sit on a dead copy of the
        // factor. The release below then finds nothing left to free.
        if (keep_factor_) trsv_.setup(F_.L, factor_dim);
        else              trsv_.setup_consuming(F_.L, factor_dim);
#else
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
        if (std::getenv("APXCHOL_SPTRSV_SETUP_TRACE"))   // same knob as the stage trace below
            std::fprintf(stderr, "[sptrsv-setup gpu] %-22s %8.2f ms  (context creation %.2f ms)\n",
                         "cuda_init", cuda_init_s * 1e3, cuda_ctx::context_seconds() * 1e3);
        trsv_.setup(F_.L, factor_dim);
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
    apx_cholesky& compute(const auto& A) {
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

#if defined(APXCHOL_USE_CUDA)
    /// Device-resident SpTRSV state (cuda_pcg uses solve_LLt_dev directly).
    cuda_sptrsv& trsv() { return trsv_; }
    const cuda_sptrsv& trsv() const { return trsv_; }
#else
    omp_sptrsv& trsv() { return trsv_; }
    const omp_sptrsv& trsv() const { return trsv_; }
#endif

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
        ensure_part_();
        // Σb feeds the Laplacian input centring, so it is only computed on a
        // centring application (center-k schedule).
        const bool center = !F_.sddm && take_center_();
        const double b_sum = center ? detail::det_sum(bv.data(), n_, part_.data()) : 0.0;
        apply_core(bv.data(), b_sum, center, x.data(), nullptr);
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
        eigen_assert(m_factorizationIsOk && "factorize() should be called first");
        ensure_part_();
        const bool center = !F_.sddm && take_center_();
        double rz = 0.0;
        apply_core(r, r_sum, center, z, &rz);
        return rz;
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
    /// `b_sum` = Σ b and `center` = whether this application runs the two
    /// Laplacian mean passes (both ignored for SDDM). If `rz_out` is non-null
    /// the final unpermute gather also accumulates b·z into it.
    void apply_core(const double* b, double b_sum, bool center,
                    double* z, double* rz_out) const {
        if (cp_) { cp_->descend("solve"); cp_->tick(); }

        // Permutation as explicit gather/scatter (replaces Eigen::PermutationMatrix
        // products). perm[v] = new position of original vertex v.
        //   to permuted   : dst[perm[v]] = src[v]   (scatter)
        //   from permuted : dst[v] = src[perm[v]]   (gather)
        // Both are bijective writes -> trivially parallel. The gather is the
        // last pass that writes z, so the b·z reduction (and, on the
        // Laplacian path, the output re-centering) ride on it: `shift` is
        // subtracted from every gathered value, and the per-thread partial
        // b·z products are summed in thread order (deterministic).
        const node_index* P = F_.perm.data();
        const Eigen::Index n = n_;
        double* part = part_.data();
        auto gather_out = [&](const double* src, double shift) {
            int nt_used = 1;
            #pragma omp parallel if(n > detail::fused_omp_min())
            {
                int tid, nt; detail::omp_ids(tid, nt);
                if (tid == 0) nt_used = nt;
                const auto [lo, hi] = detail::static_chunk(n, tid, nt);
                double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                Eigen::Index v = lo;
                for (; v + 4 <= hi; v += 4) {
                    const double z0 = src[P[v + 0]] - shift;
                    const double z1 = src[P[v + 1]] - shift;
                    const double z2 = src[P[v + 2]] - shift;
                    const double z3 = src[P[v + 3]] - shift;
                    z[v + 0] = z0; z[v + 1] = z1; z[v + 2] = z2; z[v + 3] = z3;
                    s0 += b[v + 0] * z0; s1 += b[v + 1] * z1;
                    s2 += b[v + 2] * z2; s3 += b[v + 3] * z3;
                }
                double s = (s0 + s1) + (s2 + s3);
                for (; v < hi; ++v) {
                    const double zv = src[P[v]] - shift;
                    z[v] = zv;
                    s += b[v] * zv;
                }
                part[static_cast<std::size_t>(tid) * detail::kPartStride] = s;
            }
            if (rz_out) *rz_out = detail::reduce_parts(part, nt_used);
        };
        // Label of the final pass: says whether b·z was fused in, so a
        // profile shows which application path ran.
        const char* unpermute_label = rz_out ? "unpermute+rz" : "unpermute";

        if (F_.sddm) {
            // ── SDDM path: full-rank factor, no centering ──
            #pragma omp parallel for schedule(static) if(n > detail::fused_omp_min())
            for (Eigen::Index v = 0; v < n; ++v) z[P[v]] = b[v];
            if (cp_) (*cp_)("permute");

#if defined(APXCHOL_USE_CUDA)
            // solve_LLt reads z[0..n) and writes scratch_[0..n); the gather
            // below reads scratch_ straight (the old x = scratch_; scratch_ = x
            // round trip was a no-op pair of serial copies).
            trsv_.solve_LLt(z, scratch_.data());
            if (cp_) (*cp_)("back");
            const double* out = scratch_.data();
#else
            trsv_.forward_solve(z, scratch_.data());
            if (cp_) (*cp_)("forward");
            // The transpose solve writes scratch2_ (not z) so the unpermute
            // below can gather straight from it — no serial staging copy.
            trsv_.transpose_solve(scratch_.data(), scratch2_.data());
            if (cp_) (*cp_)("back");
            const double* out = scratch2_.data();
#endif
            gather_out(out, 0.0);
            if (cp_) (*cp_)(unpermute_label);
        } else {
            // ── Laplacian path: rank-(n-1) factor with centering ──
            const Eigen::Index m = n_ - 1;

            // APXCHOL_GROUND=center-k (see env_knobs.h): the two mean-
            // subtraction passes run only on centring applications (every
            // K-th one -- `center` was decided by the caller via
            // take_center_()); the others skip both. In exact arithmetic PCG
            // keeps r in the centred subspace (A·1 = 0 => 1ᵀ(b − A x) = 1ᵀb
            // = 0), so the pre-centring is a no-op there and the post-
            // centring only removes an x-component along 1 (null space --
            // invisible to the residual). Centring every K-th application
            // bounds the fp drift of both; K = 1 centres every application.
            if (center) {
                // Input centering fused into the permute scatter:
                // z[P[v]] = b[v] - mean(b), mean from the caller's Σb.
                const double b_mean = b_sum / static_cast<double>(n);
                #pragma omp parallel for schedule(static) if(n > detail::fused_omp_min())
                for (Eigen::Index v = 0; v < n; ++v) z[P[v]] = b[v] - b_mean;
            } else {
                // Skipped application: plain scatter (b_sum unused).
                #pragma omp parallel for schedule(static) if(n > detail::fused_omp_min())
                for (Eigen::Index v = 0; v < n; ++v) z[P[v]] = b[v];
            }
            if (cp_) (*cp_)("permute");

#if defined(APXCHOL_USE_CUDA)
            // solve_LLt reads z[0..m) and writes scratch_[0..m); entry m (the
            // grounded vertex) is 0.
            trsv_.solve_LLt(z, scratch_.data());
            if (cp_) (*cp_)("back");
            scratch_(m) = 0.0;
            const double* out = scratch_.data();
#else
            trsv_.forward_solve(z, scratch_.data());
            if (cp_) (*cp_)("forward");
            trsv_.transpose_solve(scratch_.data(), scratch2_.data());
            if (cp_) (*cp_)("back");
            scratch2_(m) = 0.0;
            const double* out = scratch2_.data();
#endif
            // Output re-centering (centring applications only): the
            // permutation preserves the multiset, so mean(z) == mean(out) --
            // one deterministic reduction over the permuted solve output,
            // then the gather subtracts it while writing z (and accumulates
            // b·z if asked). Replaces the gather + z.mean() + z -= mean
            // triple pass. A skipped application gathers with shift 0.
            const double z_mean = center
                ? detail::det_sum(out, n, part) / static_cast<double>(n) : 0.0;
            gather_out(out, z_mean);
            if (cp_) (*cp_)(unpermute_label);
        }

        if (cp_) cp_->ascend();
    }

    struct factorization F_;
    factor_options opts_;
    graph_storage storage_ = graph_storage::vec_pool;
    checkpoint* cp_ = nullptr;
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
#if !defined(APXCHOL_USE_CUDA)
    // Transpose-solve output staging for the CPU _solve_impl (see
    // install_factor); unused (never resized) under CUDA.
    mutable Eigen::VectorXd scratch2_;
#endif
    Eigen::Index n_ = 0;
    Eigen::ComputationInfo m_info = Eigen::Success;
    bool m_analysisIsOk = false;
    bool m_factorizationIsOk = false;

#if defined(APXCHOL_USE_CUDA)
    mutable cuda_sptrsv trsv_;
#else
    omp_sptrsv trsv_;
#endif
};

} // namespace apxchol
