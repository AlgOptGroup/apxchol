#pragma once
#include "apxchol/sparse_csc.h"
#include "apxchol/solver/sptrsv/cuda_cast.h"
#include "apxchol/solver/sptrsv/cuda_dataflow.h"
#include "apxchol/solver/sptrsv/cuda_host.h"
#include "apxchol/solver/sptrsv/factor_drop.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace apxchol {

// GPU SpTRSV value type = the shared sptrsv_value_t (fp32): the factor
// (d_vals_) AND the internal solve vectors (d_x_/d_y_) all use this; the
// PCG-facing interface stays fp64 and casts at the boundary.
using cuda_value_t = sptrsv_value_t;
static_assert(sizeof(cuda_value_t) == 4, "the GPU SpTRSV runs in fp32");

namespace detail {

inline void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
}

} // namespace detail

#define APXCHOL_CUDA_CHECK(call)    apxchol::detail::check_cuda(call, #call)

/// GPU sparse triangular solve using our persistent dataflow kernel.
/// The CUDA library links cudart only. There is no cuSPARSE backend or
/// backend-selection policy. APXCHOL_GPU_SPTRSV=dataflow is accepted for
/// compatibility; any other nonempty value is rejected before setup.
///
/// CSR of L and L^T, O(n) warp-batch tables and epoch-tagged {value, epoch}
/// words support one persistent launch per sweep, without analysis buffers
/// or level schedules. Results are bit-deterministic across repeated
/// launches and grid sizes. See cuda_dataflow.h for the kernel design;
/// tests/test_sptrsv_drop.cpp compares it with independent CPU references.
///
/// COMPACTING FACTOR DROP: setup runs THE compacting drop
/// (factor_drop.h -- the very implementation omp_sptrsv::setup runs on the
/// CPU; same env knob, same default: APXCHOL_FACTOR_DROP=<rel>, default 1e-4,
/// <= 0 = off; the column-sum compensation is unconditional) on the HOST
/// L11 arrays before anything is uploaded: off-diagonals below rel * (column
/// max |off-diagonal|) are removed, the diagonal always kept, each column's
/// dropped mass folded back into its kept off-diagonals so column sums are
/// preserved. Everything downstream -- the upload, dataflow transpose and
/// batch tables -- sees only the compacted factor, so device factor bytes
/// and per-row dependency work shrink by the dropped fraction (iter0040: 52%).
/// drop_stats() / stored_nnz() report it (the benchmark's FILL line prints
/// stored_nnz= on this backend too); the CPU unit tests state that the
/// arrays this backend uploads are the arrays omp_sptrsv stores.
///
/// FP16 STORAGE (env APXCHOL_SPTRSV_FP16, shared with the CPU backend --
/// lowprec.h; default ON on the GPU): the storage is the
/// CPU's FP16_SCALED contract (cuda_host.h file header; omp.h "FOLDED INTO
/// THE VECTORS"): the off-diagonals hold binary16 of the column-scaled L~ =
/// L D^-1 (s_j = column max |off-diagonal|), the diagonal is a separate fp32
/// diag[j] = fp32(L_jj) / s_j -- plus the column's rounding residual, always
/// (the same contract the CPU SpTRSV applies; without it the Laplacian path
/// pays iter0040 45 -> 64 iterations) -- and inv_scale[j] = fp32(1 / s_j); the forward solve on
/// L~ returns y' = D y and the back solve reads its input times inv_scale^2
/// (r_j^2 kept in DOUBLE on the device and multiplied in double, exactly the
/// CPU FP16_SCALED contract: pre-squaring into fp32 overflows to Inf for
/// s_j < ~5.4e-20 -- a column whose off-diagonal max is that small does occur
/// on real factors -- and the Inf turned the whole solve into silent NaN),
/// so the pair applies (L_s L_s^T)^-1 for the stored L_s = L~ D. Products
/// and sums are otherwise formed in cuda_value_t (float) after the half ->
/// float widen. Columns whose scale s_j cannot be represented at all (fp32
/// 1/s_j overflows, s_j < ~3e-39, or fp32(L_jj)/s_j overflows, off-diagonals
/// >= ~1e38x below the diagonal) fall back to s_j = 1: their off-diagonals
/// are dropped/flushed -- below anything an fp32 sweep could see next to
/// that diagonal anyway. setup() then verifies
/// every device diag / inv_scale^2 is finite (diag also nonzero) and THROWS
/// otherwise: a factor the fp16 storage cannot represent must fail loudly,
/// never dissolve into NaN.
/// Device factor bytes per stored entry: 4 (colidx) + 2 (value), times two
/// CSRs, plus 8 B/row -- vs 4 + 4 with the fp32 storage.
/// fp16 subnormals are always flushed to zero at storage and the drop's keep
/// predicate also drops what fp16 flushes, exactly as on the CPU.
class cuda_sptrsv {
public:
    cuda_sptrsv() = default;

    // Width of the on-device factor values (mirrors omp_sptrsv): 4, fp32.
    // Printed at startup, same as the CPU backend. (The runtime fp16 storage is
    // a runtime mode: fp16() / value_bytes_effective() report it.)
    static constexpr std::size_t value_bytes = sizeof(cuda_value_t);
    static constexpr const char* value_name = "float (fp32)";

    cuda_sptrsv(const cuda_sptrsv&) = delete;
    cuda_sptrsv& operator=(const cuda_sptrsv&) = delete;

    // Move not supported (complex GPU state) — use via pointer or unique_ptr.
    cuda_sptrsv(cuda_sptrsv&&) = delete;
    cuda_sptrsv& operator=(cuda_sptrsv&&) = delete;

    ~cuda_sptrsv() { destroy(); }

    // Read factor storage at every setup. The GPU defaults to fp16;
    // the CPU reads the same storage variable with the opposite default.
    static int fp16_env_tristate() { return sptrsv_fp16_env_tristate(); }
    static bool fp16_from_env() { return fp16_env_tristate() == 1; }
    static bool fp16_resolved() { return fp16_env_tristate() != 0; }
    /// Setup: build L11 on the host, run the compacting drop, (fp16: narrow),
    /// copy to device, build the dataflow tables. Env APXCHOL_SPTRSV_SETUP_TRACE=1 (the
    /// CPU backend's knob, same name) prints the per-stage wall times of one
    /// setup to stderr (diagnostic). Since 2026-08-20 the per-process CUDA
    /// context creation is NOT part of any stage here: the caller establishes
    /// it before this call and reports it as "cuda_init"
    /// (solver/cuda_context.h, apx_cholesky::install_factor); it used to be
    /// charged to "build_L11", ~100-135 ms on this machine, ~715 ms on GH200.
    void setup(const sparse_csc& L, node_index m) {
        validate_backend_env();
        const bool fp16 = fp16_resolved();
        destroy();
        m_ = static_cast<int64_t>(m);
        const bool trace = std::getenv("APXCHOL_SPTRSV_SETUP_TRACE") != nullptr;
        auto t_prev = std::chrono::steady_clock::now();
        auto mark = [&](const char* what) {
            if (!trace) return;
            cudaDeviceSynchronize();
            const auto now = std::chrono::steady_clock::now();
            std::fprintf(stderr, "[sptrsv-setup gpu] %-22s %8.2f ms\n", what,
                         std::chrono::duration<double, std::milli>(now - t_prev).count());
            t_prev = std::chrono::steady_clock::now();
        };

        fp16_ = fp16;
        if (std::getenv("APXCHOL_VERBOSE"))
            std::fprintf(stderr, "[apxchol] GPU SpTRSV backend: %s (%s storage)\n",
                         backend_name(), fp16_ ? "fp16" : "fp32");
        const double factor_drop_rel = factor_drop_rel_from_env();

        // Build L11 = top-left m×m block of the factor in CSC, with checked
        // 32-bit indices for the dataflow kernels.
        // The same three arrays are CSR of L^T (the back solve's operand).
        cuda_host::csr_int<cuda_value_t> LT = cuda_host::build_L11_csc_int<cuda_value_t>(L, m_);
        mark("build_L11");

        // Per-column scales (the drop's threshold reference; the fp16 storage's
        // scale), from the factor BEFORE the drop.
        std::vector<float> col_scale;
        if (factor_drop_rel > 0.0 || fp16_) col_scale = cuda_host::column_scales(LT);
        // fp16 storage only: a column whose scale cannot be represented --
        // fp32 1/s_j (s_j < ~3e-39) or the scaled diagonal fp32(L_jj)/s_j
        // (off-diagonals >= ~1e38x below the diagonal) would overflow --
        // falls back to s_j = 1 (the "no off-diagonals" convention): its
        // off-diagonals are below anything an fp32 sweep could see next to
        // that diagonal, and the drop/flush below removes them. Applied
        // BEFORE the drop so the threshold and the storage agree on the
        // scale.
        if (fp16_)
            for (int64_t j = 0; j < m_; ++j) {
                const float s = col_scale[static_cast<std::size_t>(j)];
                const float d = static_cast<float>(LT.vals[LT.ptr[static_cast<std::size_t>(j)]]);
                if (!std::isfinite(1.0f / s) || !std::isfinite(d / s))
                    col_scale[static_cast<std::size_t>(j)] = 1.0f;
            }

        // ── Compacting drop (factor_drop.h; see the class comment) ──
        stats_ = cuda_host::apply_factor_drop(LT, col_scale, factor_drop_rel, fp16_);
        nnz_ = LT.nnz;
        mark("col_scale+drop");

        size_t free_before = 0, total = 0;
        APXCHOL_CUDA_CHECK(cudaMemGetInfo(&free_before, &total));
        factor_bytes_ = 0;
        auto dev_alloc = [&](void** p, std::size_t bytes) {
            APXCHOL_CUDA_CHECK(cudaMalloc(p, bytes));
            factor_bytes_ += bytes;
        };

        // Structure of CSR(L^T) (every storage).
        dev_alloc(reinterpret_cast<void**>(&d_rowPtr_), (m_ + 1) * sizeof(int));
        dev_alloc(reinterpret_cast<void**>(&d_colIdx_), nnz_ * sizeof(int));
        APXCHOL_CUDA_CHECK(cudaMemcpy(d_rowPtr_, LT.ptr.data(), (m_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
        APXCHOL_CUDA_CHECK(cudaMemcpy(d_colIdx_, LT.idx.get(),  nnz_ * sizeof(int),     cudaMemcpyHostToDevice));

        // Allocate device vectors (solve runs at cuda_value_t width).
        APXCHOL_CUDA_CHECK(cudaMalloc(&d_x_, m_ * sizeof(cuda_value_t)));
        APXCHOL_CUDA_CHECK(cudaMalloc(&d_y_, m_ * sizeof(cuda_value_t)));
        h_stage_.resize(static_cast<size_t>(m_));
        mark("upload_LT_struct+vecs");

        if (fp16_) {
            // FP16 per-column-scaled storage (cuda_host.h contract): values of
            // CSR(L^T) narrowed on the host, fp32 diag / inv_scale^2 alongside,
            // then the transpose (carrying the same 16-bit patterns) for CSR(L).
            cuda_host::fp16_scaled_arrays h16 = cuda_host::narrow_fp16_scaled(LT, col_scale);
            fp16_flushed_ = h16.flushed; fp16_subnormal_ = h16.subnormal;
            // r_j^2 in DOUBLE (exact; the kernels multiply rhs * r_j^2 in
            // double and narrow once -- pre-squaring into fp32 overflows for
            // s_j < ~5.4e-20 and the Inf became silent NaN downstream).
            std::vector<double> inv_scale2(static_cast<std::size_t>(m_));
            for (int64_t j = 0; j < m_; ++j) {
                const double r = static_cast<double>(h16.inv_scale[j]);
                inv_scale2[j] = r * r;
            }
            // Fail LOUDLY on a factor the fp16 storage cannot represent
            // (unreachable after the scale fallback above -- a guard, not a
            // policy): a non-finite diag/inv_scale would dissolve the solve
            // into NaN with no error anywhere.
            for (int64_t j = 0; j < m_; ++j)
                if (!(std::isfinite(h16.diag[j]) && h16.diag[j] != 0.0f && std::isfinite(inv_scale2[j])))
                    throw std::runtime_error(
                        "apxchol cuda_sptrsv: fp16 factor storage cannot represent column " + std::to_string(j) +
                        " (diag=" + std::to_string(h16.diag[j]) + ", inv_scale^2=" + std::to_string(inv_scale2[j]) +
                        "); set APXCHOL_SPTRSV_FP16=0");
            dev_alloc(reinterpret_cast<void**>(&d_vals16_),     nnz_ * sizeof(std::uint16_t));
            dev_alloc(reinterpret_cast<void**>(&d_diag_),       m_ * sizeof(float));
            dev_alloc(reinterpret_cast<void**>(&d_inv_scale2_), m_ * sizeof(double));
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_vals16_, h16.vals.get(), nnz_ * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_diag_, h16.diag.data(), m_ * sizeof(float), cudaMemcpyHostToDevice));
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_inv_scale2_, inv_scale2.data(), m_ * sizeof(double), cudaMemcpyHostToDevice));
            // CSR of L for the forward solve: transpose of (structure of LT,
            // fp16 values).
            cuda_host::csr_int<std::uint16_t> LT16;
            LT16.m = LT.m; LT16.nnz = LT.nnz; LT16.ptr = LT.ptr;
            LT16.idx  = std::move(LT.idx);
            LT16.vals = std::move(h16.vals);
            cuda_host::csr_int<std::uint16_t> L16 = cuda_host::transpose_csr(LT16);
            mark("fp16_narrow+transpose");
            dev_alloc(reinterpret_cast<void**>(&d_L_rowptr_), (m_ + 1) * sizeof(int));
            dev_alloc(reinterpret_cast<void**>(&d_L_colidx_), nnz_ * sizeof(int));
            dev_alloc(reinterpret_cast<void**>(&d_L_vals16_), nnz_ * sizeof(std::uint16_t));
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_L_rowptr_, L16.ptr.data(), (m_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_L_colidx_, L16.idx.get(),  nnz_ * sizeof(int), cudaMemcpyHostToDevice));
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_L_vals16_, L16.vals.get(), nnz_ * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
            mark("upload_L");
            setup_kernel_backend(LT16.ptr, L16.ptr, LT16.idx.get(), L16.idx.get());
            mark("kernel_backend_tables");
        } else {
            dev_alloc(reinterpret_cast<void**>(&d_vals_), nnz_ * sizeof(cuda_value_t));
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_vals_, LT.vals.get(), nnz_ * sizeof(cuda_value_t), cudaMemcpyHostToDevice));
            mark("upload_LT_vals");
            // Forward solve L y = x gathers over rows of L -> needs CSR of L.
            // Transpose the stored CSR of L^T for the forward sweep.
            cuda_host::csr_int<cuda_value_t> Lc = cuda_host::transpose_csr(LT);
            mark("transpose");
            dev_alloc(reinterpret_cast<void**>(&d_L_rowptr_), (m_ + 1) * sizeof(int));
            dev_alloc(reinterpret_cast<void**>(&d_L_colidx_), nnz_ * sizeof(int));
            dev_alloc(reinterpret_cast<void**>(&d_L_vals_),   nnz_ * sizeof(cuda_value_t));
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_L_rowptr_, Lc.ptr.data(), (m_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_L_colidx_, Lc.idx.get(),  nnz_ * sizeof(int), cudaMemcpyHostToDevice));
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_L_vals_,   Lc.vals.get(), nnz_ * sizeof(cuda_value_t), cudaMemcpyHostToDevice));
            mark("upload_L");
            setup_kernel_backend(LT.ptr, Lc.ptr, LT.idx.get(), Lc.idx.get());
            mark("kernel_backend_tables");
        }
        size_t free_after = 0;
        APXCHOL_CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
        device_delta_bytes_ = free_before >= free_after ? free_before - free_after : 0;

        if (std::getenv("APXCHOL_VERBOSE") || std::getenv("APXCHOL_GPU_MEM_DEBUG")) {
            const std::uint64_t off = stats_.nnz_stored - static_cast<std::uint64_t>(m_);
            const std::uint64_t off0 = off + stats_.dropped;
            std::fprintf(stderr,
                "[apxchol] sptrsv storage GPU/%s %s: stored_nnz=%llu offdiag=%llu%s"
                " | factor device bytes=%.1f MB (cudaMemGetInfo delta %.1f MB; free %.2f -> %.2f of %.2f GB)\n",
                backend_name(),
                fp16_ ? "fp16 (per-column scaled, diag fp32 + rounding residual)" : value_name,
                static_cast<unsigned long long>(stats_.nnz_stored), static_cast<unsigned long long>(off),
                fp16_ ? (" flushed_to_zero=" + std::to_string(fp16_flushed_) + " subnormal=" + std::to_string(fp16_subnormal_)).c_str() : "",
                factor_bytes_ / 1e6, device_delta_bytes_ / 1e6, free_before / 1e9, free_after / 1e9, total / 1e9);
            if (stats_.rel > 0.0)
                std::fprintf(stderr,
                    "[apxchol] factor drop (APXCHOL_FACTOR_DROP=%g, column sums preserved): dropped=%llu"
                    " (%.4f%% of %llu off-diagonals; threshold=%llu, format_zero=%llu)"
                    " stored_nnz %llu -> %llu (%.4f%% of factor)\n",
                    stats_.rel,
                    static_cast<unsigned long long>(stats_.dropped),
                    100.0 * static_cast<double>(stats_.dropped) / static_cast<double>(off0 ? off0 : 1),
                    static_cast<unsigned long long>(off0),
                    static_cast<unsigned long long>(stats_.dropped_threshold),
                    static_cast<unsigned long long>(stats_.dropped_flush),
                    static_cast<unsigned long long>(stats_.nnz_factor),
                    static_cast<unsigned long long>(stats_.nnz_stored),
                    100.0 * static_cast<double>(stats_.nnz_stored) / static_cast<double>(stats_.nnz_factor ? stats_.nnz_factor : 1));
        }
        ready_ = true;
    }

    /// Combined forward + back solve on GPU: computes L^{-T} L^{-1} x.
    /// Reads x_in[0..m-1] from host, writes result to x_out[0..m-1] on host.
    /// x_in and x_out may alias (ping-pong stays on GPU).
    void solve_LLt(const double* x_in, double* x_out) const {
        // Narrow on the host into the staging buffer, ship half the bytes, solve
        // in fp32, ship back, widen. The host casts are O(m) and dwarfed by the
        // (now-halved) PCIe transfer they bracket.
        for (int64_t i = 0; i < m_; ++i) h_stage_[i] = static_cast<cuda_value_t>(x_in[i]);
        APXCHOL_CUDA_CHECK(cudaMemcpy(d_x_, h_stage_.data(), m_ * sizeof(cuda_value_t), cudaMemcpyHostToDevice));
        solve_LLt_dev_impl();
        APXCHOL_CUDA_CHECK(cudaMemcpy(h_stage_.data(), d_x_, m_ * sizeof(cuda_value_t), cudaMemcpyDeviceToHost));
        for (int64_t i = 0; i < m_; ++i) x_out[i] = static_cast<double>(h_stage_[i]);
    }

    /// GPU-resident variant: input/output are DEVICE pointers. Avoids
    /// the two host↔device transfers per call. Used by GPU-resident PCG
    /// loops so vectors stay on device between iters. d_in and d_out
    /// may alias (we copy in → internal d_x, ping-pong on device, then
    /// copy d_x → out).
    void solve_LLt_dev(const double* d_in, double* d_out) const {
        // GPU-resident PCG keeps its vectors fp64; narrow into d_x_ (fp32),
        // solve in fp32, widen back into d_out. Two elementwise device passes,
        // on the SpSV's stream.
        cast_f64_to_f32(d_in, d_x_, m_, 0);
        solve_LLt_dev_impl();
        cast_f32_to_f64(d_x_, d_out, m_, 0);
    }

    bool ready() const { return ready_; }
    /// What the compacting drop did at the last setup() (factor_drop.h): rel /
    /// nnz_factor / nnz_stored / dropped*.
    const factor_drop_stats& drop_stats() const { return stats_; }
    /// nnz the device CSR(s) hold after the last setup() (after the drop).
    std::uint64_t stored_nnz() const { return stats_.nnz_stored; }
    /// Backend name for benchmark metadata.
    const char* backend_name() const { return "dataflow"; }
    /// Blocks of the persistent grid chosen at setup.
    int dataflow_grid() const { return df_grid_; }
    bool fp16() const { return fp16_; }
    /// Bytes per stored off-diagonal value on the device (2 under fp16).
    std::size_t value_bytes_effective() const { return fp16_ ? 2 : value_bytes; }
    /// Device bytes of the factor arrays cudaMalloc'd by the last setup()
    /// (structure + values of every CSR copy, fp16's diag / inv_scale^2 and the
    /// dataflow batch tables; NOT the solve vectors) and the cudaMemGetInfo
    /// free-memory delta across the whole setup, at allocation granularity.
    std::size_t factor_device_bytes() const { return factor_bytes_; }
    std::size_t device_bytes_delta() const { return device_delta_bytes_; }
private:
    static void validate_backend_env() {
        const char* value = std::getenv("APXCHOL_GPU_SPTRSV");
        if (value && *value && std::string(value) != "dataflow")
            throw std::invalid_argument(
                std::string("APXCHOL_GPU_SPTRSV='") + value +
                "' is unsupported; the GPU backend is dataflow");
    }

    // The dataflow backend's host-side tables (cuda_dataflow.h): no levels at
    // all -- the warp batch tables of both directions (from the CSR row
    // lengths), the m tagged words, the control ints, the resident grid.
    void setup_kernel_backend(const std::vector<int>& LT_ptr, const std::vector<int>& L_ptr,
                              const int* LT_idx, const int* L_idx) {
        const int mi = static_cast<int>(m_);
        const std::vector<int> len_L  = cuda_host::csr_row_lengths(mi, L_ptr.data());
        const std::vector<int> len_LT = cuda_host::csr_row_lengths(mi, LT_ptr.data());
        const int pre = dataflow_prefetch_depth();
        // Row segmentation: each direction gets its own threshold, from its
        // own row-length histogram (the two differ a lot -- an elimination
        // factor's hub rows land in CSR of L, its columns stay bounded).
        const cuda_host::dataflow_len_stats st_L  = cuda_host::dataflow_row_stats(mi, len_L.data(), pre);
        const cuda_host::dataflow_len_stats st_LT = cuda_host::dataflow_row_stats(mi, len_LT.data(), pre);
        const cuda_host::dataflow_seg_params sp_f = dataflow_seg_params_for(pre, st_L);
        const cuda_host::dataflow_seg_params sp_b = dataflow_seg_params_for(pre, st_LT);
        const cuda_host::dataflow_plan fwd = cuda_host::dataflow_build_plan(
            mi, false, L_ptr.data(), L_idx, len_L.data(), pre, sp_f);
        const cuda_host::dataflow_plan bck = cuda_host::dataflow_build_plan(
            mi, true, LT_ptr.data(), LT_idx, len_LT.data(), pre, sp_b);
        // The kernel relies on the plan's warp-sharing property implicitly
        // (see cuda_host.h dataflow_plan_check); O(m), so just check it.
        for (int dir = 0; dir < 2; ++dir) {
            const std::string why = cuda_host::dataflow_plan_check(
                dir ? bck : fwd, mi, dir != 0, (dir ? len_LT : len_L).data(), pre);
            if (!why.empty())
                throw std::runtime_error("apxchol cuda_sptrsv: bad dataflow plan (" +
                                         std::string(dir ? "back" : "forward") + "): " + why);
        }
        fwd_batches_ = fwd.n_batches();
        bck_batches_ = bck.n_batches();
        df_slots_ = std::max(fwd.n_slots, bck.n_slots);   // the two sweeps reuse the same words
        df_split_rows_ = fwd.n_split + bck.n_split;
        if (static_cast<std::int64_t>(m_) + df_slots_ > static_cast<std::int64_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("apxchol cuda_sptrsv: m + dataflow partial slots exceeds the 32-bit index range");
        auto up = [&](int** d, const std::vector<int>& h) {
            APXCHOL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(d), h.size() * sizeof(int)));
            factor_bytes_ += h.size() * sizeof(int);
            APXCHOL_CUDA_CHECK(cudaMemcpy(*d, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice));
        };
        auto up_spec = [&](int4** d, const std::vector<cuda_host::dataflow_spec>& h) {
            if (h.empty()) return;
            static_assert(sizeof(cuda_host::dataflow_spec) == sizeof(int4), "spec items upload as int4");
            APXCHOL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(d), h.size() * sizeof(int4)));
            factor_bytes_ += h.size() * sizeof(int4);
            APXCHOL_CUDA_CHECK(cudaMemcpy(*d, h.data(), h.size() * sizeof(int4), cudaMemcpyHostToDevice));
        };
        up(&d_fwd_batches_, fwd.batch_start); up(&d_bck_batches_, bck.batch_start);
        up(&d_fwd_spec_sel_, fwd.batch_spec); up(&d_bck_spec_sel_, bck.batch_spec);
        up_spec(&d_fwd_spec_, fwd.spec);      up_spec(&d_bck_spec_, bck.spec);
        if (std::getenv("APXCHOL_GPU_SPTRSV_STATS"))
            dataflow_print_stats(st_L, st_LT, sp_f, sp_b, fwd, bck);
        const std::size_t tag_words = static_cast<std::size_t>(m_) + static_cast<std::size_t>(df_slots_);
        APXCHOL_CUDA_CHECK(cudaMalloc(&d_df_tag_, tag_words * sizeof(unsigned long long)));
        APXCHOL_CUDA_CHECK(cudaMemset(d_df_tag_, 0, tag_words * sizeof(unsigned long long)));
        APXCHOL_CUDA_CHECK(cudaMalloc(&d_df_ctrl_, 4 * sizeof(int)));
        APXCHOL_CUDA_CHECK(cudaMemset(d_df_ctrl_, 0, 4 * sizeof(int)));
        factor_bytes_ += tag_words * sizeof(unsigned long long) + 4 * sizeof(int);
        df_epoch_ = 0;
        df_grid_ = dataflow_grid_size(fp16_);

        if (std::getenv("APXCHOL_GPU_MEM_DEBUG")) { size_t mf=0, mt=0; cudaMemGetInfo(&mf,&mt);
            fprintf(stderr,"[mem] SpTRSV dataflow: 2x factor (CSR L+L^T)=%.2fGB (%zuB/value) "
                    "fwd_batches=%d bck_batches=%d grid=%dx%d tag=%.1fMB slots=%d split_rows=%d | GPU free=%.2f/%.2f GB\n",
                    2.0*nnz_*(4.0+(double)value_bytes_effective())/1e9, value_bytes_effective(),
                    fwd_batches_, bck_batches_, df_grid_, kDataflowBlock, tag_words * 8.0 / 1e6,
                    df_slots_, df_split_rows_, mf/1e9, mt/1e9); }
    }

    /// The segmentation parameters of ONE sweep direction. ON by default; the
    /// threshold comes from that direction's own row-length histogram
    /// (cuda_host::dataflow_split_threshold -- a pure function of row lengths,
    /// so grid size cannot enter the result). ONE knob:
    /// APXCHOL_GPU_DF_SPLIT=<off-diagonals> pins the threshold for both
    /// directions, =0 turns segmentation off (the rollback switch: the plan,
    /// and the output, become bit-identical to the unsplit kernel).
    ///
    /// The segment LENGTH is not a knob: it is ONE CHUNK, 32 * pre entries.
    /// Measured on kron_g500-logn16 with the threshold pinned at 256
    /// (gpu_pcg_loop ms/iteration, medians of 4, interleaved): 256 entries
    /// 8.87, 512 10.34, 1024 11.62, unsplit 17.81; on com-LiveJournal 50.49 /
    /// 60.60 / 112.97 against 156.59 -- the depth model's sqrt(len) optimum is
    /// swamped by the extra round trip every additional chunk of a segment
    /// costs on the critical path, so the shortest segment the chunk walk can
    /// express wins. The planner keeps `seg` general; only this default is
    /// fixed. max_seg only bounds the ticket window one giant row can occupy
    /// and never binds in practice (the longest measured factor row,
    /// com-Orkut's 134368, gives S = 525 against the 4096 cap).
    static cuda_host::dataflow_seg_params dataflow_seg_params_for(
            int pre, const cuda_host::dataflow_len_stats& st) {
        cuda_host::dataflow_seg_params p;
        const char* e = std::getenv("APXCHOL_GPU_DF_SPLIT");
        const int   v = e ? std::atoi(e) : -1;
        if (e && v <= 0) return p;                       // seg = 0: off
        p.seg = 32 * pre;
        p.split_min = e ? v : cuda_host::dataflow_split_threshold(st);
        return p;
    }

    // APXCHOL_GPU_SPTRSV_STATS=1: the row-length histogram of both sweep
    // directions -- the evidence behind the row-segmentation threshold
    // (cuda_host.h dataflow_row_stats). Bucket k = the rows an unsplit walk
    // chews in more than 2^k chunks of C = 32 * pre entries, and the share of
    // the factor they hold: on a hub factor those few rows ARE the critical
    // path, on a grid / IPM factor the histogram is empty past bucket 0.
    void dataflow_print_stats(const cuda_host::dataflow_len_stats& st_L,
                              const cuda_host::dataflow_len_stats& st_LT,
                              const cuda_host::dataflow_seg_params& sp_f,
                              const cuda_host::dataflow_seg_params& sp_b,
                              const cuda_host::dataflow_plan& fwd,
                              const cuda_host::dataflow_plan& bck) const {
        for (int dir = 0; dir < 2; ++dir) {
            const cuda_host::dataflow_len_stats& st = dir ? st_LT : st_L;
            const cuda_host::dataflow_seg_params& sp = dir ? sp_b : sp_f;
            std::fprintf(stderr, "[df-stats] %s: m=%d nnz=%lld mean_len=%.2f max_len=%d C=%d\n",
                         dir ? "back (CSR L^T)" : "fwd (CSR L)", st.m,
                         static_cast<long long>(st.nnz), st.mean_len, st.max_len, st.base);
            for (int k = 0; k < cuda_host::dataflow_len_stats::kBuckets; ++k) {
                if (!st.rows_ge[k]) break;
                std::fprintf(stderr, "[df-stats]   len >= %8lld : rows %10lld (%6.3f%%)  entries %12lld (%6.2f%% of nnz)\n",
                             static_cast<long long>(st.base) << k,
                             static_cast<long long>(st.rows_ge[k]),
                             100.0 * static_cast<double>(st.rows_ge[k]) / (st.m ? st.m : 1),
                             static_cast<long long>(st.nnz_ge[k]),
                             100.0 * static_cast<double>(st.nnz_ge[k]) / static_cast<double>(st.nnz ? st.nnz : 1));
            }
            const cuda_host::dataflow_plan& pl = dir ? bck : fwd;
            std::fprintf(stderr, "[df-stats]   long-row nnz share %.3f -> seg=%d split_min=%d max_seg=%d"
                                 " -> batches=%d split_rows=%d slots=%d spec=%zu (tables %.2f MB)\n",
                         st.nnz ? static_cast<double>(st.nnz_ge[0]) / static_cast<double>(st.nnz) : 0.0,
                         sp.seg, sp.split_min, sp.max_seg, pl.n_batches(), pl.n_split, pl.n_slots,
                         pl.spec.size(),
                         (pl.batch_start.size() * 4.0 + pl.batch_spec.size() * 4.0 + pl.spec.size() * 16.0) / 1e6);
        }
    }


    void solve_LLt_dev_impl() const {
        // One persistent launch per sweep (cuda_dataflow.h). Forward on
        // CSR of L with CSR of L^T as the dependent lists, back the
        // reverse; the level orders are the topological claim orders.
        const int m = static_cast<int>(m_);
        // Epochs: the forward sweep tags its words with e, the back sweep
        // with e+1 (one shared tag array; a consumer only accepts the
        // current sweep's epoch). Long before the 32-bit epoch could wrap
        // onto a stale word the array is cleared and the count restarted.
        if (df_epoch_ >= 0xFFFFFF00u) {
            APXCHOL_CUDA_CHECK(cudaMemsetAsync(d_df_tag_, 0,
                (static_cast<std::size_t>(m_) + static_cast<std::size_t>(df_slots_)) * sizeof(unsigned long long), 0));
            df_epoch_ = 0;
        }
        const unsigned e_fwd = ++df_epoch_;
        const unsigned e_bck = ++df_epoch_;
        if (fp16_) {
            dataflow_solve_fp16(0, m, false, d_L_rowptr_, d_L_colidx_, d_L_vals16_, d_diag_, nullptr,
                                d_fwd_batches_, d_fwd_spec_sel_, d_fwd_spec_, fwd_batches_,
                                d_df_tag_, e_fwd, d_df_ctrl_, df_grid_, d_x_, d_y_);
            dataflow_solve_fp16(0, m, true, d_rowPtr_, d_colIdx_, d_vals16_, d_diag_, d_inv_scale2_,
                                d_bck_batches_, d_bck_spec_sel_, d_bck_spec_, bck_batches_,
                                d_df_tag_, e_bck, d_df_ctrl_ + 2, df_grid_, d_y_, d_x_);
        } else {
            dataflow_solve(0, m, false, d_L_rowptr_, d_L_colidx_, d_L_vals_,
                           d_fwd_batches_, d_fwd_spec_sel_, d_fwd_spec_, fwd_batches_,
                           d_df_tag_, e_fwd, d_df_ctrl_, df_grid_, d_x_, d_y_);
            dataflow_solve(0, m, true, d_rowPtr_, d_colIdx_, d_vals_,
                           d_bck_batches_, d_bck_spec_sel_, d_bck_spec_, bck_batches_,
                           d_df_tag_, e_bck, d_df_ctrl_ + 2, df_grid_, d_y_, d_x_);
        }
    }

    void destroy() {
        if (!ready_) return;
        cudaFree(d_L_rowptr_); cudaFree(d_L_colidx_); cudaFree(d_L_vals_); cudaFree(d_L_vals16_);
        cudaFree(d_vals16_); cudaFree(d_diag_); cudaFree(d_inv_scale2_);
        cudaFree(d_df_tag_); cudaFree(d_df_ctrl_);
        cudaFree(d_fwd_batches_); cudaFree(d_bck_batches_);
        cudaFree(d_fwd_spec_sel_); cudaFree(d_bck_spec_sel_);
        cudaFree(d_fwd_spec_); cudaFree(d_bck_spec_);
        d_df_tag_ = nullptr; d_df_ctrl_ = nullptr;
        d_fwd_batches_ = d_bck_batches_ = nullptr; fwd_batches_ = bck_batches_ = 0;
        d_fwd_spec_sel_ = d_bck_spec_sel_ = nullptr;
        d_fwd_spec_ = d_bck_spec_ = nullptr;
        df_slots_ = df_split_rows_ = 0;
        df_epoch_ = 0;
        d_L_rowptr_ = d_L_colidx_ = nullptr;
        d_L_vals_ = nullptr; d_L_vals16_ = nullptr;
        d_vals16_ = nullptr; d_diag_ = nullptr; d_inv_scale2_ = nullptr;
        df_grid_ = 0;
        cudaFree(d_rowPtr_);
        cudaFree(d_colIdx_);
        cudaFree(d_vals_);
        cudaFree(d_x_);
        cudaFree(d_y_);
        d_rowPtr_ = d_colIdx_ = nullptr;
        d_vals_ = nullptr;
        d_x_    = nullptr;
        d_y_    = nullptr;
        ready_ = false;
        fp16_ = false;
        factor_bytes_ = device_delta_bytes_ = 0;
        fp16_flushed_ = fp16_subnormal_ = 0;
        stats_ = factor_drop_stats{};
    }

    int64_t m_ = 0;
    int64_t nnz_ = 0;
    bool ready_ = false;

    // What the compacting drop did (factor_drop.h) and the device footprint.
    factor_drop_stats stats_;
    std::size_t factor_bytes_ = 0;         // cudaMalloc'd factor arrays (see factor_device_bytes)
    std::size_t device_delta_bytes_ = 0;   // cudaMemGetInfo free delta over setup
    std::uint64_t fp16_flushed_ = 0, fp16_subnormal_ = 0;

    // Device memory (values + solve vectors at cuda_value_t width).
    int*          d_rowPtr_ = nullptr;
    int*          d_colIdx_ = nullptr;
    cuda_value_t* d_vals_   = nullptr;
    cuda_value_t* d_x_      = nullptr;
    cuda_value_t* d_y_      = nullptr;
    // Host staging for the fp32 host-PCG boundary (narrow/widen).
    mutable std::vector<cuda_value_t> h_stage_;

    // Our dataflow backend. The stored d_* arrays are CSR
    // of L^T (back solve); the forward solve needs CSR of L (d_L_*).
    int*          d_L_rowptr_     = nullptr;
    int*          d_L_colidx_     = nullptr;
    cuda_value_t* d_L_vals_       = nullptr;
    // Dataflow backend (APXCHOL_GPU_SPTRSV=dataflow; cuda_dataflow.h): the
    // warp batch tables of both directions, the m tagged words {value, epoch}
    // both sweeps publish into (mutable state, 8 B/row), the sweep epoch, 2+2
    // control ints and the persistent grid.
    int*          d_fwd_batches_   = nullptr;   // cuda_host::dataflow_build_plan
    int*          d_bck_batches_   = nullptr;
    int*          d_fwd_spec_sel_  = nullptr;   // batch_spec: -1 = plain, else index into d_*_spec_
    int*          d_bck_spec_sel_  = nullptr;
    int4*         d_fwd_spec_      = nullptr;   // the row-segmentation items
    int4*         d_bck_spec_      = nullptr;
    int           fwd_batches_ = 0, bck_batches_ = 0;
    int           df_slots_ = 0;                // extra tagged words for the segment partials
    int           df_split_rows_ = 0;           // rows the plan split (both directions summed)
    unsigned long long* d_df_tag_  = nullptr;
    mutable unsigned    df_epoch_  = 0;         // last epoch handed out (0 = none)
    int*          d_df_ctrl_       = nullptr;   // {fwd ticket, fwd finished, bck ticket, bck finished}
    int           df_grid_         = 0;
    // fp16 storage (APXCHOL_SPTRSV_FP16): binary16
    // values of CSR(L~^T) (d_vals16_, replaces d_vals_) and CSR(L~)
    // (d_L_vals16_, replaces d_L_vals_), the fp32 scaled diagonal and the
    // back solve's per-row input scale inv_scale^2 -- held in DOUBLE (r_j^2
    // exact; fp32 overflows for tiny scales, see the class comment).
    bool          fp16_           = false;
    std::uint16_t* d_vals16_      = nullptr;
    std::uint16_t* d_L_vals16_    = nullptr;
    float*        d_diag_         = nullptr;
    double*       d_inv_scale2_   = nullptr;
};

#undef APXCHOL_CUDA_CHECK

} // namespace apxchol
