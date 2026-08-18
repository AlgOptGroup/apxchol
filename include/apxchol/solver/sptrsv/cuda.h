#pragma once
#include "apxchol/sparse_csc.h"
#include "apxchol/solver/sptrsv/cuda_cast.h"
#include "apxchol/solver/sptrsv/cuda_dataflow.h"
#include "apxchol/solver/sptrsv/cuda_host.h"
#include "apxchol/solver/sptrsv/cuda_levelset.h"
#include "apxchol/solver/sptrsv/factor_drop.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace apxchol {

// GPU SpTRSV value type = the shared sptrsv_value_t (fp32 under
// -DAPXCHOL_SPTRSV_FP32). cuSPARSE SpSV requires matrix/vector/compute types to
// match, so the factor (d_vals_) AND the internal solve vectors (d_x_/d_y_) all
// use this; the PCG-facing interface stays fp64 and casts at the boundary.
using cuda_value_t = sptrsv_value_t;
inline constexpr cudaDataType_t cuda_value_dtype =
    sizeof(cuda_value_t) == 4 ? CUDA_R_32F : CUDA_R_64F;

namespace detail {

inline void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
}

inline void check_cusparse(cusparseStatus_t err, const char* msg) {
    if (err != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error(std::string(msg) + ": cusparse error " + std::to_string(err));
}

} // namespace detail

#define APXCHOL_CUDA_CHECK(call)    apxchol::detail::check_cuda(call, #call)
#define APXCHOL_CUSPARSE_CHECK(call) apxchol::detail::check_cusparse(call, #call)

/// RAII wrapper for the GPU sparse triangular solve of the factor: three
/// backends behind one interface.
///   * Our sync-free DATAFLOW kernel (the DEFAULT on the fp32 build -- the
///     AUTO choice; env APXCHOL_GPU_SPTRSV=dataflow names it explicitly;
///     cuda_dataflow.h): CSR of L (forward) + CSR of L^T (back) on device
///     (no level schedules) plus 8 B/row of epoch-tagged {value, epoch}
///     words, two O(n) warp-batch tables and four control ints -- ONE
///     persistent launch per sweep, rows claimed in natural (forward) /
///     reverse (back) order, consumers poll their neighbours' tagged words.
///     No analysis buffer, O(n) state. Faster than cuSPARSE on grid_2000 /
///     iter0040 (measured, gpu_pcg_loop ms/iter: 1.08 -> 0.98 iter0040,
///     3.54 -> 3.20 grid_2000), bit-deterministic run to run and across
///     grid sizes. Same fp16 storage option as the level-set backend; fp32
///     build only (dataflow_supported()).
///   * cuSPARSE SpSV (env APXCHOL_GPU_SPTRSV=cusparse; the AUTO choice on the
///     fp64 build, where the dataflow kernel does not exist): copies L11 to
///     device once, runs the analysis once (O(nnz) scratch), then
///     GPU-parallel forward/back solves. Not deterministic run to run.
///   * Our O(n)-schedule level-set kernels (env APXCHOL_GPU_SPTRSV=levelset,
///     cuda_levelset.h): the same two CSRs, one launch per level --
///     memory-frugal (no O(nnz) analysis buffer), deterministic, with the
///     opt-in FP16 factor storage (APXCHOL_GPU_SPTRSV_FP16=1, below); the
///     fp64 build's AUTO fallback when cuSPARSE's buffers do not fit.
///
/// cuSPARSE SpSV requires CSR (not CSC).  Eigen stores column-major (CSC).
/// We exploit the identity  CSC(L) == CSR(L^T):  the same three arrays
/// (outerIndexPtr, innerIndexPtr, valuePtr) describe L in CSC or L^T in CSR.
/// We create L^T (upper triangular) in CSR, then:
///   - Forward  solve  L  * y = x  ↔  TRANSPOSE    on L^T
///   - Back     solve  L^T* z = y  ↔  NON_TRANSPOSE on L^T
///
/// BACKEND SELECTION (per setup): env APXCHOL_GPU_SPTRSV=dataflow|cusparse|
/// levelset is an explicit override -- exactly that backend, no fallback
/// (=cusparse together with APXCHOL_GPU_SPTRSV_FP16=1 is impossible, cuSPARSE
/// has no fp16 SpSV: a stderr note, then the AUTO choice). Unset (or any
/// other value) = AUTO:
///   * fp32 SpTRSV build (APXCHOL_SPTRSV_FP32=ON, the default): the DATAFLOW
///     backend, unconditionally -- it has no analysis buffer (its device
///     state is the two CSRs the level-set backend needs anyway plus 8 B/row
///     + O(n) batch tables), so nothing to fit and nothing to decide; with
///     APXCHOL_GPU_SPTRSV_FP16=1 the fp16 dataflow kernel. Faster than
///     cuSPARSE on every measured workload and deterministic.
///   * fp64 SpTRSV build (APXCHOL_SPTRSV_FP32=OFF; dataflow_supported() is
///     false): the pre-dataflow OOM-aware rule -- cuSPARSE, unless its two
///     SpSV analyses' O(nnz) device scratch (~23 B/nnz each on this toolkit
///     -- 128 MB on iter0040, 766 MB on grid_2000, several GB on the giant
///     social factors, which is exactly what OOMs com-Orkut on a 16 GB card)
///     does not fit. Setup, once the factor is on the device, asks cuSPARSE
///     for both buffer sizes (cusparseSpSV_bufferSize, no allocation), and
///     picks cuSPARSE only if bufFwd + bufBck fits into the free device
///     memory minus a 10%-of-total safety margin minus what the caller says
///     it still needs to allocate afterwards (set_reserve_bytes():
///     apxchol::solve passes the GPU-resident PCG's operator + vectors,
///     which are allocated AFTER this setup); else the O(n)-schedule
///     level-set backend (its extra cost is one more CSR copy of the factor,
///     nnz*(4+value) bytes, plus 8 B/row). With APXCHOL_GPU_SPTRSV_FP16=1
///     the level-set backend directly (the only fp16-capable one there).
/// The decision is printed under APXCHOL_VERBOSE ("[apxchol] GPU SpTRSV
/// backend: ..."; backend_reason() / backend_forced() / backend_name()).
/// The OOM-aware rule is kept ONLY for the fp64 build's AUTO: on the fp32
/// build an explicit =cusparse gets cuSPARSE with no fitting check (as
/// before), and AUTO never asks cuSPARSE for anything. benchmarks/
/// sweep_fair.py's "retry with APXCHOL_GPU_SPTRSV=levelset after an OOM" is
/// redundant on both builds.
///
/// COMPACTING FACTOR DROP (both backends): setup runs THE compacting drop
/// (factor_drop.h -- the very implementation omp_sptrsv::setup runs on the
/// CPU; same env knobs, same default: APXCHOL_FACTOR_DROP=<rel>, default 1e-4,
/// <= 0 = off; APXCHOL_FACTOR_DROP_COMPENSATE=0 = plain removal) on the HOST
/// L11 arrays before anything is uploaded: off-diagonals below rel * (column
/// max |off-diagonal|) are removed, the diagonal always kept, each column's
/// dropped mass folded back into its kept off-diagonals so column sums are
/// preserved. Everything downstream -- the upload, cuSPARSE's analysis, the
/// level-set transpose and schedules -- sees only the compacted factor, so
/// the device factor bytes, the SpSV analysis buffers and the per-level
/// dependency work all shrink by the dropped fraction (iter0040: 52%).
/// drop_stats() / stored_nnz() report it (the benchmark's FILL line prints
/// stored_nnz= on this backend too); the CPU unit tests state that the
/// arrays this backend uploads are the arrays omp_sptrsv stores.
///
/// FP16 STORAGE (env APXCHOL_GPU_SPTRSV_FP16=1, default off; our two kernel
/// backends only -- without APXCHOL_GPU_SPTRSV=dataflow it forces levelset,
/// with a stderr note if cuSPARSE was asked for explicitly): cuSPARSE 12.8's SpSV REJECTS
/// CUDA_R_16F matrix values (cusparseSpSV_bufferSize returns
/// CUSPARSE_STATUS_NOT_SUPPORTED, "value type of matA (CUDA_R_16F) is not
/// supported", for every A/vector/compute combination -- probed on this
/// toolkit), so the fp16 factor lives in OUR kernels. The storage is the
/// CPU's FP16_SCALED contract (cuda_host.h file header; omp.h "FOLDED INTO
/// THE VECTORS"): the off-diagonals hold binary16 of the column-scaled L~ =
/// L D^-1 (s_j = column max |off-diagonal|), the diagonal is a separate fp32
/// diag[j] = fp32(L_jj) / s_j -- plus the column's rounding residual by
/// default (the CPU's APXCHOL_LOWPREC_DIAG_COMP=1 semantics, ON here:
/// =0 turns it off for A/B; without it the Laplacian path pays iter0040 45 ->
/// 64 iterations) -- and inv_scale[j] = fp32(1 / s_j); the forward solve on
/// L~ returns y' = D y and the back solve reads its input times inv_scale^2,
/// so the pair applies (L_s L_s^T)^-1 for the stored L_s = L~ D. Products
/// and sums are formed in cuda_value_t (float) after the half -> float widen.
/// Device factor bytes per stored entry: 4 (colidx) + 2 (value), times two
/// CSRs, plus 8 B/row -- vs 4 + 4 fp32 (times two on the level-set backend).
/// fp16 subnormals are flushed to zero at storage (APXCHOL_FP16_KEEP_SUBNORMAL=1
/// keeps them) and the drop's keep predicate also drops what fp16 flushes,
/// exactly as on the CPU FP16_SCALED build.
class cuda_sptrsv {
public:
    cuda_sptrsv() = default;

    // Compiled width of the on-device factor values (mirrors omp_sptrsv): 4 ==
    // -DAPXCHOL_SPTRSV_FP32, 8 == fp64. Printed at startup so the GPU build flag
    // is observable at runtime, same as the CPU backend. (The opt-in fp16
    // storage is a runtime mode: fp16() / value_bytes_effective() report it.)
    static constexpr std::size_t value_bytes = sizeof(cuda_value_t);
    static constexpr const char* value_name =
        sizeof(cuda_value_t) == 4 ? "float (fp32)" : "double (fp64)";

    cuda_sptrsv(const cuda_sptrsv&) = delete;
    cuda_sptrsv& operator=(const cuda_sptrsv&) = delete;

    // Move not supported (complex GPU state) — use via pointer or unique_ptr.
    cuda_sptrsv(cuda_sptrsv&&) = delete;
    cuda_sptrsv& operator=(cuda_sptrsv&&) = delete;

    ~cuda_sptrsv() { destroy(); }

    // Env resolution of the runtime modes (read at every setup()):
    //   APXCHOL_GPU_SPTRSV=dataflow|cusparse|levelset (unset = AUTO: dataflow
    //   on the fp32 build, the OOM-aware cuSPARSE/level-set rule on fp64),
    //   APXCHOL_GPU_SPTRSV_FP16=1 (default off; kernel backends only -- with
    //   =cusparse it is a stderr note + AUTO).
    static bool fp16_from_env() {
        const char* e = std::getenv("APXCHOL_GPU_SPTRSV_FP16");
        return e && std::atoi(e) != 0;
    }
    // The env's explicit choice: +2 = dataflow forced, +1 = levelset forced,
    // -1 = cusparse forced (fp16 off), 0 = unset / anything else = AUTO (see
    // the class comment; auto_prefers_dataflow() says which AUTO).
    static int backend_from_env() {
        const char* be = std::getenv("APXCHOL_GPU_SPTRSV");
        const std::string s(be ? be : "");
        if (s == "dataflow") return +2;
        if (s == "levelset") return +1;
        if (s == "cusparse") return fp16_from_env() ? 0 : -1;   // no fp16 SpSV in cuSPARSE: AUTO
        return 0;
    }
    static bool levelset_from_env() { return backend_from_env() == 1; }
    static bool dataflow_from_env() { return backend_from_env() == 2; }
    /// What AUTO resolves to on this build: true = the dataflow backend (fp32
    /// SpTRSV build), false = the fp64 build's cuSPARSE-if-it-fits rule.
    static bool auto_prefers_dataflow() { return dataflow_supported(); }
    // Safety margin of the fp64 build's AUTO decision: cuSPARSE is chosen only
    // if its analysis buffers fit into free - kAutoMarginFraction * total -
    // reserve.
    static constexpr double kAutoMarginFraction = 0.10;
    /// Device bytes the caller will allocate AFTER setup() (the GPU-resident
    /// PCG's operator + vectors); the fp64 build's AUTO backend decision keeps
    /// them free (the dataflow AUTO of the fp32 build has nothing to fit and
    /// ignores it). Survives setup()/destroy(); default 0.
    void set_reserve_bytes(std::size_t bytes) { reserve_bytes_ = bytes; }
    std::size_t reserve_bytes() const { return reserve_bytes_; }
    // APXCHOL_LOWPREC_DIAG_COMP for the fp16 storage: ON unless "=0" (the CPU
    // build reads the same variable, default OFF there -- see the class
    // comment for why the default differs).
    static bool fp16_diag_comp_from_env() {
        const char* e = std::getenv("APXCHOL_LOWPREC_DIAG_COMP");
        return !(e && *e && std::atoi(e) == 0);
    }
    // APXCHOL_FP16_KEEP_SUBNORMAL=1 keeps fp16 subnormals (default: flush).
    static bool fp16_flush_subnormal_from_env() {
        const char* e = std::getenv("APXCHOL_FP16_KEEP_SUBNORMAL");
        return !(e && std::atoi(e) != 0);
    }

    /// Setup: build L11 on the host, run the compacting drop, (fp16: narrow),
    /// copy to device, create cuSPARSE descriptors / level schedules.
    void setup(const sparse_csc& L, node_index m) {
        destroy();
        m_ = static_cast<int64_t>(m);

        // Runtime modes (env, per setup). AUTO (be_env == 0) is the dataflow
        // backend on the fp32 build (nothing to fit, nothing to decide); on
        // the fp64 build it starts on cuSPARSE and is decided below (fp16:
        // straight to the level-set, the only fp16-capable backend there).
        // use_dataflow_ implies use_levelset_ (the dataflow backend shares
        // the level-set's device arrays).
        fp16_ = fp16_from_env();
        const int be_env = backend_from_env();
        backend_forced_ = be_env != 0;
        use_dataflow_   = be_env == 2 || (be_env == 0 && auto_prefers_dataflow());
        use_levelset_   = be_env > 0 || use_dataflow_ || (be_env == 0 && fp16_);
        {
            const char* be = std::getenv("APXCHOL_GPU_SPTRSV");
            if (fp16_ && be && std::string(be) == "cusparse")
                std::fprintf(stderr, "[apxchol] APXCHOL_GPU_SPTRSV_FP16=1: cuSPARSE SpSV has no fp16 value type"
                                     " (CUSPARSE_STATUS_NOT_SUPPORTED); using the AUTO backend (%s)\n",
                                     use_dataflow_ ? "dataflow" : "level-set");
        }
        backend_reason_.clear();
        const bool   fp16_flush_subnormal = fp16_flush_subnormal_from_env();
        diag_comp_ = fp16_ && fp16_diag_comp_from_env();
        const double factor_drop_rel = factor_drop_rel_from_env();
        const bool   drop_compensate = factor_drop_compensate_from_env();

        // Build L11 = top-left m×m block of the factor in CSC, as int arrays
        // (cuSPARSE CUSPARSE_INDEX_32I). The factor on GPU-tested matrices fits
        // int32; a factor exceeding it cannot use cuSPARSE's 32-bit index API.
        // The same three arrays are CSR of L^T (the back solve's operand).
        cuda_host::csr_int<cuda_value_t> LT = cuda_host::build_L11_csc_int<cuda_value_t>(L, m_);

        // Per-column scales (the drop's threshold reference; the fp16 storage's
        // scale), from the factor BEFORE the drop.
        std::vector<float> col_scale;
        if (factor_drop_rel > 0.0 || fp16_) col_scale = cuda_host::column_scales(LT);

        // ── Compacting drop (factor_drop.h; see the class comment) ──
        stats_ = cuda_host::apply_factor_drop(LT, col_scale, factor_drop_rel, drop_compensate,
                                              fp16_, fp16_flush_subnormal);
        nnz_ = LT.nnz;

        size_t free_before = 0, total = 0;
        APXCHOL_CUDA_CHECK(cudaMemGetInfo(&free_before, &total));
        factor_bytes_ = 0;
        auto dev_alloc = [&](void** p, std::size_t bytes) {
            APXCHOL_CUDA_CHECK(cudaMalloc(p, bytes));
            factor_bytes_ += bytes;
        };

        // Structure of CSR(L^T) (both backends, every storage).
        dev_alloc(reinterpret_cast<void**>(&d_rowPtr_), (m_ + 1) * sizeof(int));
        dev_alloc(reinterpret_cast<void**>(&d_colIdx_), nnz_ * sizeof(int));
        APXCHOL_CUDA_CHECK(cudaMemcpy(d_rowPtr_, LT.ptr.data(), (m_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
        APXCHOL_CUDA_CHECK(cudaMemcpy(d_colIdx_, LT.idx.get(),  nnz_ * sizeof(int),     cudaMemcpyHostToDevice));

        // Allocate device vectors (solve runs at cuda_value_t width).
        APXCHOL_CUDA_CHECK(cudaMalloc(&d_x_, m_ * sizeof(cuda_value_t)));
        APXCHOL_CUDA_CHECK(cudaMalloc(&d_y_, m_ * sizeof(cuda_value_t)));
        h_stage_.resize(static_cast<size_t>(m_));

        if (fp16_) {
            backend_reason_ = use_dataflow_
                ? (backend_forced_ ? "dataflow (APXCHOL_GPU_SPTRSV=dataflow, APXCHOL_GPU_SPTRSV_FP16=1)"
                                   : "dataflow (auto: the fp32 build's default; APXCHOL_GPU_SPTRSV_FP16=1 storage)")
                : (backend_forced_ ? "level-set (APXCHOL_GPU_SPTRSV=levelset, APXCHOL_GPU_SPTRSV_FP16=1)"
                                   : "level-set (auto: APXCHOL_GPU_SPTRSV_FP16=1 on the fp64 build -- cuSPARSE has no fp16 SpSV,"
                                     " the dataflow kernel needs the fp32 build)");
            if (std::getenv("APXCHOL_VERBOSE"))
                std::fprintf(stderr, "[apxchol] GPU SpTRSV backend: %s\n", backend_reason_.c_str());
            // FP16 per-column-scaled storage (cuda_host.h contract): values of
            // CSR(L^T) narrowed on the host, fp32 diag / inv_scale^2 alongside,
            // then the transpose (carrying the same 16-bit patterns) for CSR(L).
            cuda_host::fp16_scaled_arrays h16 =
                cuda_host::narrow_fp16_scaled(LT, col_scale, fp16_flush_subnormal, diag_comp_);
            fp16_flushed_ = h16.flushed; fp16_subnormal_ = h16.subnormal;
            std::vector<float> inv_scale2(static_cast<std::size_t>(m_));
            for (int64_t j = 0; j < m_; ++j) {
                const double r = static_cast<double>(h16.inv_scale[j]);
                inv_scale2[j] = static_cast<float>(r * r);   // r_j^2 (exact in double, one fp32 rounding)
            }
            dev_alloc(reinterpret_cast<void**>(&d_vals16_),     nnz_ * sizeof(std::uint16_t));
            dev_alloc(reinterpret_cast<void**>(&d_diag_),       m_ * sizeof(float));
            dev_alloc(reinterpret_cast<void**>(&d_inv_scale2_), m_ * sizeof(float));
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_vals16_, h16.vals.get(), nnz_ * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_diag_, h16.diag.data(), m_ * sizeof(float), cudaMemcpyHostToDevice));
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_inv_scale2_, inv_scale2.data(), m_ * sizeof(float), cudaMemcpyHostToDevice));
            // CSR of L for the forward solve: transpose of (structure of LT,
            // fp16 values).
            cuda_host::csr_int<std::uint16_t> LT16;
            LT16.m = LT.m; LT16.nnz = LT.nnz; LT16.ptr = LT.ptr;
            LT16.idx  = std::move(LT.idx);
            LT16.vals = std::move(h16.vals);
            cuda_host::csr_int<std::uint16_t> L16 = cuda_host::transpose_csr(LT16);
            dev_alloc(reinterpret_cast<void**>(&d_L_rowptr_), (m_ + 1) * sizeof(int));
            dev_alloc(reinterpret_cast<void**>(&d_L_colidx_), nnz_ * sizeof(int));
            dev_alloc(reinterpret_cast<void**>(&d_L_vals16_), nnz_ * sizeof(std::uint16_t));
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_L_rowptr_, L16.ptr.data(), (m_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_L_colidx_, L16.idx.get(),  nnz_ * sizeof(int), cudaMemcpyHostToDevice));
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_L_vals16_, L16.vals.get(), nnz_ * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
            setup_kernel_backend(LT16.ptr, LT16.idx.get(), L16.ptr, L16.idx.get());
        } else {
            dev_alloc(reinterpret_cast<void**>(&d_vals_), nnz_ * sizeof(cuda_value_t));
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_vals_, LT.vals.get(), nnz_ * sizeof(cuda_value_t), cudaMemcpyHostToDevice));
            if (!use_levelset_) {
                // cuSPARSE descriptors + the two SpSV buffer-size queries (no
                // allocation yet). AUTO (fp64 build only -- the fp32 build's
                // AUTO is the dataflow backend and never gets here): keep
                // cuSPARSE only if its analysis buffers fit next to what is
                // already resident, with the margin.
                cusparse_prepare();
                if (!backend_forced_) {
                    size_t mf = 0, mt = 0;
                    APXCHOL_CUDA_CHECK(cudaMemGetInfo(&mf, &mt));
                    const double margin = kAutoMarginFraction * static_cast<double>(mt);
                    const double need   = static_cast<double>(cusparse_buf_bytes_);
                    const double avail  = static_cast<double>(mf) - margin - static_cast<double>(reserve_bytes_);
                    const double ls_need = static_cast<double>(nnz_) * (sizeof(int) + sizeof(cuda_value_t))
                                         + 2.0 * static_cast<double>(m_) * sizeof(int);
                    char buf[512];
                    if (need <= avail) {
                        std::snprintf(buf, sizeof(buf),
                            "cuSPARSE (auto: SpSV analysis buffers %.1f MB fit: free %.1f MB - %.0f%% margin %.1f MB"
                            " - caller reserve %.1f MB = %.1f MB; level-set would need %.1f MB)",
                            need / 1e6, mf / 1e6, 100.0 * kAutoMarginFraction, margin / 1e6, reserve_bytes_ / 1e6,
                            avail / 1e6, ls_need / 1e6);
                    } else {
                        std::snprintf(buf, sizeof(buf),
                            "level-set (auto: cuSPARSE SpSV analysis buffers %.1f MB do NOT fit: free %.1f MB - %.0f%% margin %.1f MB"
                            " - caller reserve %.1f MB = %.1f MB; level-set needs %.1f MB)",
                            need / 1e6, mf / 1e6, 100.0 * kAutoMarginFraction, margin / 1e6, reserve_bytes_ / 1e6,
                            avail / 1e6, ls_need / 1e6);
                        cusparse_teardown();
                        use_levelset_ = true;
                    }
                    backend_reason_ = buf;
                } else {
                    backend_reason_ = "cuSPARSE (APXCHOL_GPU_SPTRSV=cusparse)";
                }
            } else {
                backend_reason_ = use_dataflow_
                    ? (backend_forced_ ? "dataflow (APXCHOL_GPU_SPTRSV=dataflow)"
                                       : "dataflow (auto: the fp32 build's default -- no analysis buffer, O(n) state;"
                                         " APXCHOL_GPU_SPTRSV=cusparse|levelset overrides)")
                    : "level-set (APXCHOL_GPU_SPTRSV=levelset)";
            }
            if (std::getenv("APXCHOL_VERBOSE"))
                std::fprintf(stderr, "[apxchol] GPU SpTRSV backend: %s\n", backend_reason_.c_str());
            if (use_levelset_) {
                // Forward solve L y = x gathers over rows of L -> needs CSR of L.
                // Transpose the stored CSR of L^T (2x factor storage -- still far
                // under cuSPARSE's factor + O(nnz) SpSV scratch).
                cuda_host::csr_int<cuda_value_t> Lc = cuda_host::transpose_csr(LT);
                dev_alloc(reinterpret_cast<void**>(&d_L_rowptr_), (m_ + 1) * sizeof(int));
                dev_alloc(reinterpret_cast<void**>(&d_L_colidx_), nnz_ * sizeof(int));
                dev_alloc(reinterpret_cast<void**>(&d_L_vals_),   nnz_ * sizeof(cuda_value_t));
                APXCHOL_CUDA_CHECK(cudaMemcpy(d_L_rowptr_, Lc.ptr.data(), (m_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
                APXCHOL_CUDA_CHECK(cudaMemcpy(d_L_colidx_, Lc.idx.get(),  nnz_ * sizeof(int), cudaMemcpyHostToDevice));
                APXCHOL_CUDA_CHECK(cudaMemcpy(d_L_vals_,   Lc.vals.get(), nnz_ * sizeof(cuda_value_t), cudaMemcpyHostToDevice));
                setup_kernel_backend(LT.ptr, LT.idx.get(), Lc.ptr, Lc.idx.get());
            } else {
                cusparse_analyze();
            }
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
                fp16_ ? (diag_comp_ ? "fp16 (per-column scaled, diag fp32 + rounding residual)"
                                    : "fp16 (per-column scaled, diag fp32)") : value_name,
                static_cast<unsigned long long>(stats_.nnz_stored), static_cast<unsigned long long>(off),
                fp16_ ? (" flushed_to_zero=" + std::to_string(fp16_flushed_) + " subnormal=" + std::to_string(fp16_subnormal_)).c_str() : "",
                factor_bytes_ / 1e6, device_delta_bytes_ / 1e6, free_before / 1e9, free_after / 1e9, total / 1e9);
            if (stats_.rel > 0.0)
                std::fprintf(stderr,
                    "[apxchol] factor drop (APXCHOL_FACTOR_DROP=%g%s): dropped=%llu (%.4f%% of %llu off-diagonals;"
                    " threshold=%llu, format_zero=%llu) stored_nnz %llu -> %llu (%.4f%% of factor)\n",
                    stats_.rel, stats_.compensate ? ", column sums preserved" : ", plain removal (COMPENSATE=0)",
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
        // #ifdef (not `if constexpr`): these aren't templates, so a discarded
        // `if constexpr` branch is still type-checked -- and the fp64 d_x_ is
        // double*, which the fp32 cast/stage calls would reject. The
        // preprocessor genuinely removes the other path.
#ifdef APXCHOL_SPTRSV_FP32
        // Narrow on the host into the staging buffer, ship half the bytes, solve
        // in fp32, ship back, widen. The host casts are O(m) and dwarfed by the
        // (now-halved) PCIe transfer they bracket.
        for (int64_t i = 0; i < m_; ++i) h_stage_[i] = static_cast<cuda_value_t>(x_in[i]);
        APXCHOL_CUDA_CHECK(cudaMemcpy(d_x_, h_stage_.data(), m_ * sizeof(cuda_value_t), cudaMemcpyHostToDevice));
        solve_LLt_dev_impl();
        APXCHOL_CUDA_CHECK(cudaMemcpy(h_stage_.data(), d_x_, m_ * sizeof(cuda_value_t), cudaMemcpyDeviceToHost));
        for (int64_t i = 0; i < m_; ++i) x_out[i] = static_cast<double>(h_stage_[i]);
#else
        APXCHOL_CUDA_CHECK(cudaMemcpy(d_x_, x_in, m_ * sizeof(double), cudaMemcpyHostToDevice));
        solve_LLt_dev_impl();
        APXCHOL_CUDA_CHECK(cudaMemcpy(x_out, d_x_, m_ * sizeof(double), cudaMemcpyDeviceToHost));
#endif
    }

    /// GPU-resident variant: input/output are DEVICE pointers. Avoids
    /// the two host↔device transfers per call. Used by GPU-resident PCG
    /// loops so vectors stay on device between iters. d_in and d_out
    /// may alias (we copy in → internal d_x, ping-pong on device, then
    /// copy d_x → out).
    void solve_LLt_dev(const double* d_in, double* d_out) const {
#ifdef APXCHOL_SPTRSV_FP32
        // GPU-resident PCG keeps its vectors fp64; narrow into d_x_ (fp32),
        // solve in fp32, widen back into d_out. Two elementwise device passes,
        // on the SpSV's stream.
        cast_f64_to_f32(d_in, d_x_, m_, 0);
        solve_LLt_dev_impl();
        cast_f32_to_f64(d_x_, d_out, m_, 0);
#else
        APXCHOL_CUDA_CHECK(cudaMemcpyAsync(d_x_, d_in, m_ * sizeof(double),
                                           cudaMemcpyDeviceToDevice, 0));
        solve_LLt_dev_impl();
        APXCHOL_CUDA_CHECK(cudaMemcpyAsync(d_out, d_x_, m_ * sizeof(double),
                                           cudaMemcpyDeviceToDevice, 0));
#endif
    }

    bool ready() const { return ready_; }
    /// What the compacting drop did at the last setup() (factor_drop.h): rel /
    /// compensate / nnz_factor / nnz_stored / dropped*.
    const factor_drop_stats& drop_stats() const { return stats_; }
    /// nnz the device CSR(s) hold after the last setup() (after the drop).
    std::uint64_t stored_nnz() const { return stats_.nnz_stored; }
    /// Runtime modes resolved at the last setup(): the backend in use
    /// (levelset() is true for BOTH kernel backends -- they share the device
    /// arrays; dataflow() singles out the sync-free one; backend_name() names
    /// it, in the env spelling: "cusparse" | "levelset" | "dataflow"), whether
    /// the env forced it (else the AUTO decision: dataflow on the fp32 build),
    /// and the printed reason.
    bool levelset() const { return use_levelset_; }
    bool dataflow() const { return use_dataflow_; }
    const char* backend_name() const { return use_dataflow_ ? "dataflow" : use_levelset_ ? "levelset" : "cusparse"; }
    /// Dataflow backend: blocks of the persistent grid (0 on the others).
    int dataflow_grid() const { return df_grid_; }
    bool backend_forced() const { return backend_forced_; }
    const std::string& backend_reason() const { return backend_reason_; }
    bool fp16() const { return fp16_; }
    /// fp16 storage: whether the column rounding residual was folded into diag[].
    bool fp16_diag_comp() const { return diag_comp_; }
    /// Bytes per stored off-diagonal value on the device (2 under fp16).
    std::size_t value_bytes_effective() const { return fp16_ ? 2 : value_bytes; }
    /// Device bytes of the factor arrays cudaMalloc'd by the last setup()
    /// (structure + values of every CSR copy, fp16's diag / inv_scale^2 and the
    /// level-set schedules; NOT the solve vectors, NOT cuSPARSE's analysis
    /// buffers) and the cudaMemGetInfo free-memory delta across the whole
    /// setup (which includes those, at allocation granularity).
    std::size_t factor_device_bytes() const { return factor_bytes_; }
    std::size_t device_bytes_delta() const { return device_delta_bytes_; }
    /// cuSPARSE SpSV analysis buffers (0 on the level-set backend).
    std::size_t cusparse_buffer_bytes() const { return cusparse_buf_bytes_; }

private:
    // Level schedules for the level-set backend: forward from CSR of L
    // (ascending), back from CSR of L^T (descending). Serial O(nnz).
    // The two kernel backends' schedules. Level-set: the level orders of both
    // directions (forward from CSR of L ascending, back from CSR of L^T
    // descending; serial O(nnz) host work) + host level boundaries. Dataflow
    // (cuda_dataflow.h): no levels at all -- the warp batch tables of both
    // directions (from the CSR row lengths), the m tagged words, the control
    // ints, the resident grid.
    void setup_kernel_backend(const std::vector<int>& LT_ptr, const int* LT_idx,
                              const std::vector<int>& L_ptr,  const int* L_idx) {
        const int mi = static_cast<int>(m_);
        if (!use_dataflow_) {
            std::vector<int> fwd_order, bck_order;
            cuda_host::compute_levels(mi, L_ptr.data(),  L_idx,  true,  fwd_order, fwd_level_ptr_);
            cuda_host::compute_levels(mi, LT_ptr.data(), LT_idx, false, bck_order, bck_level_ptr_);
            APXCHOL_CUDA_CHECK(cudaMalloc(&d_fwd_order_, m_ * sizeof(int)));
            APXCHOL_CUDA_CHECK(cudaMalloc(&d_bck_order_, m_ * sizeof(int)));
            factor_bytes_ += 2 * m_ * sizeof(int);
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_fwd_order_, fwd_order.data(), m_ * sizeof(int), cudaMemcpyHostToDevice));
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_bck_order_, bck_order.data(), m_ * sizeof(int), cudaMemcpyHostToDevice));
        } else {
            if (!dataflow_supported())
                throw std::runtime_error("apxchol: APXCHOL_GPU_SPTRSV=dataflow needs the fp32 SpTRSV build (APXCHOL_SPTRSV_FP32=ON)");
            const std::vector<int> len_L  = cuda_host::csr_row_lengths(mi, L_ptr.data());
            const std::vector<int> len_LT = cuda_host::csr_row_lengths(mi, LT_ptr.data());
            const int pre = dataflow_prefetch_depth();
            const std::vector<int> fwd_b = cuda_host::dataflow_batches(mi, false, len_L.data(),  pre);
            const std::vector<int> bck_b = cuda_host::dataflow_batches(mi, true,  len_LT.data(), pre);
            fwd_batches_ = static_cast<int>(fwd_b.size()) - 1;
            bck_batches_ = static_cast<int>(bck_b.size()) - 1;
            auto up = [&](int** d, const std::vector<int>& h) {
                APXCHOL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(d), h.size() * sizeof(int)));
                factor_bytes_ += h.size() * sizeof(int);
                APXCHOL_CUDA_CHECK(cudaMemcpy(*d, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice));
            };
            up(&d_fwd_batches_, fwd_b); up(&d_bck_batches_, bck_b);
            APXCHOL_CUDA_CHECK(cudaMalloc(&d_df_tag_, m_ * sizeof(unsigned long long)));
            APXCHOL_CUDA_CHECK(cudaMemset(d_df_tag_, 0, m_ * sizeof(unsigned long long)));
            APXCHOL_CUDA_CHECK(cudaMalloc(&d_df_ctrl_, 4 * sizeof(int)));
            APXCHOL_CUDA_CHECK(cudaMemset(d_df_ctrl_, 0, 4 * sizeof(int)));
            factor_bytes_ += m_ * sizeof(unsigned long long) + 4 * sizeof(int);
            df_epoch_ = 0;
            df_grid_ = dataflow_grid_size(fp16_);
        }

        if (std::getenv("APXCHOL_GPU_MEM_DEBUG")) { size_t mf=0, mt=0; cudaMemGetInfo(&mf,&mt);
          if (use_dataflow_)
            fprintf(stderr,"[mem] SpTRSV dataflow: 2x factor (CSR L+L^T)=%.2fGB (%zuB/value) "
                    "fwd_batches=%d bck_batches=%d grid=%dx%d tag=%.1fMB | GPU free=%.2f/%.2f GB\n",
                    2.0*nnz_*(4.0+(double)value_bytes_effective())/1e9, value_bytes_effective(),
                    fwd_batches_, bck_batches_, df_grid_, kDataflowBlock, m_ * 8.0 / 1e6, mf/1e9, mt/1e9);
          else
            fprintf(stderr,"[mem] SpTRSV level-set: 2x factor (CSR L+L^T)=%.2fGB (%zuB/value) "
                    "fwd_levels=%zu bck_levels=%zu | GPU free=%.2f/%.2f GB\n",
                    2.0*nnz_*(4.0+(double)value_bytes_effective())/1e9, value_bytes_effective(),
                    fwd_level_ptr_.size()-1, bck_level_ptr_.size()-1, mf/1e9, mt/1e9); }
    }

    // cuSPARSE backend, phase 1: handle, matrix / vector descriptors, the two
    // SpSV descriptors and their buffer SIZES (cusparseSpSV_bufferSize; no
    // device allocation) -- what the AUTO decision needs. Requires d_rowPtr_ /
    // d_colIdx_ / d_vals_ / d_x_ / d_y_ on the device.
    void cusparse_prepare() {
        APXCHOL_CUSPARSE_CHECK(cusparseCreate(&cusparse_));

        // Create L^T in CSR format (upper triangular, non-unit diagonal).
        APXCHOL_CUSPARSE_CHECK(cusparseCreateCsr(
            &matLT_, m_, m_, nnz_,
            d_rowPtr_, d_colIdx_, d_vals_,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, cuda_value_dtype));

        cusparseFillMode_t fill = CUSPARSE_FILL_MODE_UPPER;
        cusparseDiagType_t diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
        APXCHOL_CUSPARSE_CHECK(cusparseSpMatSetAttribute(matLT_, CUSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill)));
        APXCHOL_CUSPARSE_CHECK(cusparseSpMatSetAttribute(matLT_, CUSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag)));

        // Create dense vector descriptors.
        APXCHOL_CUSPARSE_CHECK(cusparseCreateDnVec(&vecX_, m_, d_x_, cuda_value_dtype));
        APXCHOL_CUSPARSE_CHECK(cusparseCreateDnVec(&vecY_, m_, d_y_, cuda_value_dtype));

        const cuda_value_t alpha = cuda_value_t(1);
        // Forward:  L * y = x  ↔  TRANSPOSE on L^T.  vecX → vecY.
        APXCHOL_CUSPARSE_CHECK(cusparseSpSV_createDescr(&descrFwd_));
        APXCHOL_CUSPARSE_CHECK(cusparseSpSV_bufferSize(
            cusparse_, CUSPARSE_OPERATION_TRANSPOSE, &alpha,
            matLT_, vecX_, vecY_, cuda_value_dtype,
            CUSPARSE_SPSV_ALG_DEFAULT, descrFwd_, &bufSizeFwd_));
        // Back:  L^T * z = y  ↔  NON_TRANSPOSE on L^T.  vecY → vecX.
        APXCHOL_CUSPARSE_CHECK(cusparseSpSV_createDescr(&descrBck_));
        APXCHOL_CUSPARSE_CHECK(cusparseSpSV_bufferSize(
            cusparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            matLT_, vecY_, vecX_, cuda_value_dtype,
            CUSPARSE_SPSV_ALG_DEFAULT, descrBck_, &bufSizeBck_));
        cusparse_buf_bytes_ = bufSizeFwd_ + bufSizeBck_;
    }

    // cuSPARSE backend, phase 2: allocate the analysis buffers and run both
    // analyses (this is the O(nnz) device allocation the AUTO decision guards).
    void cusparse_analyze() {
        const cuda_value_t alpha = cuda_value_t(1);
        APXCHOL_CUDA_CHECK(cudaMalloc(&bufFwd_, bufSizeFwd_));
        APXCHOL_CUSPARSE_CHECK(cusparseSpSV_analysis(
            cusparse_, CUSPARSE_OPERATION_TRANSPOSE, &alpha,
            matLT_, vecX_, vecY_, cuda_value_dtype,
            CUSPARSE_SPSV_ALG_DEFAULT, descrFwd_, bufFwd_));
        APXCHOL_CUDA_CHECK(cudaMalloc(&bufBck_, bufSizeBck_));
        APXCHOL_CUSPARSE_CHECK(cusparseSpSV_analysis(
            cusparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            matLT_, vecY_, vecX_, cuda_value_dtype,
            CUSPARSE_SPSV_ALG_DEFAULT, descrBck_, bufBck_));

        if (std::getenv("APXCHOL_GPU_MEM_DEBUG")) { size_t mf=0, mt=0; cudaMemGetInfo(&mf,&mt);
          fprintf(stderr,"[mem] SpTRSV factor: nnz=%lld colidx=%.2fGB vals=%.2fGB (%zuB/elem) "
                  "SpSV bufFwd=%.2fGB bufBck=%.2fGB | GPU free=%.2f / total=%.2f GB after SpTRSV setup\n",
                  (long long)nnz_, nnz_*4.0/1e9, nnz_*(double)sizeof(cuda_value_t)/1e9, sizeof(cuda_value_t),
                  bufSizeFwd_/1e9, bufSizeBck_/1e9, mf/1e9, mt/1e9); }
    }

    // Release everything cusparse_prepare / cusparse_analyze created (the AUTO
    // fallback calls it before switching to the level-set backend; destroy()
    // calls it on the cuSPARSE backend).
    void cusparse_teardown() {
        if (descrFwd_) cusparseSpSV_destroyDescr(descrFwd_);
        if (descrBck_) cusparseSpSV_destroyDescr(descrBck_);
        if (vecX_)     cusparseDestroyDnVec(vecX_);
        if (vecY_)     cusparseDestroyDnVec(vecY_);
        if (matLT_)    cusparseDestroySpMat(matLT_);
        if (cusparse_) cusparseDestroy(cusparse_);
        cudaFree(bufFwd_);
        cudaFree(bufBck_);
        descrFwd_ = descrBck_ = nullptr; vecX_ = vecY_ = nullptr; matLT_ = nullptr; cusparse_ = nullptr;
        bufFwd_ = bufBck_ = nullptr;
        bufSizeFwd_ = bufSizeBck_ = 0;
        cusparse_buf_bytes_ = 0;
    }

    void solve_LLt_dev_impl() const {
        if (use_dataflow_) {
            // One persistent launch per sweep (cuda_dataflow.h). Forward on
            // CSR of L with CSR of L^T as the dependent lists, back the
            // reverse; the level orders are the topological claim orders.
            const int m = static_cast<int>(m_);
            // Epochs: the forward sweep tags its words with e, the back sweep
            // with e+1 (one shared tag array; a consumer only accepts the
            // current sweep's epoch). Long before the 32-bit epoch could wrap
            // onto a stale word the array is cleared and the count restarted.
            if (df_epoch_ >= 0xFFFFFF00u) {
                APXCHOL_CUDA_CHECK(cudaMemsetAsync(d_df_tag_, 0, m_ * sizeof(unsigned long long), 0));
                df_epoch_ = 0;
            }
            const unsigned e_fwd = ++df_epoch_;
            const unsigned e_bck = ++df_epoch_;
            if (fp16_) {
                dataflow_solve_fp16(0, m, false, d_L_rowptr_, d_L_colidx_, d_L_vals16_, d_diag_, nullptr,
                                    d_fwd_batches_, fwd_batches_, d_df_tag_, e_fwd, d_df_ctrl_, df_grid_, d_x_, d_y_);
                dataflow_solve_fp16(0, m, true, d_rowPtr_, d_colIdx_, d_vals16_, d_diag_, d_inv_scale2_,
                                    d_bck_batches_, bck_batches_, d_df_tag_, e_bck, d_df_ctrl_ + 2, df_grid_, d_y_, d_x_);
            } else {
                dataflow_solve(0, m, false, d_L_rowptr_, d_L_colidx_, d_L_vals_,
                               d_fwd_batches_, fwd_batches_, d_df_tag_, e_fwd, d_df_ctrl_, df_grid_, d_x_, d_y_);
                dataflow_solve(0, m, true, d_rowPtr_, d_colIdx_, d_vals_,
                               d_bck_batches_, bck_batches_, d_df_tag_, e_bck, d_df_ctrl_ + 2, df_grid_, d_y_, d_x_);
            }
            return;
        }
        if (fp16_) {
            // Forward  L~ y' = x  (CSR of L~, forward schedule; no input scale):  d_x_ -> d_y_.
            levelset_solve_fp16(0, d_L_rowptr_, d_L_colidx_, d_L_vals16_, d_diag_, nullptr,
                                d_fwd_order_, fwd_level_ptr_.data(),
                                static_cast<int>(fwd_level_ptr_.size()) - 1, d_x_, d_y_);
            // Back  L~^T z = D^-2 y'  (CSR of L~^T, back schedule; input scaled by inv_scale^2):  d_y_ -> d_x_.
            levelset_solve_fp16(0, d_rowPtr_, d_colIdx_, d_vals16_, d_diag_, d_inv_scale2_,
                                d_bck_order_, bck_level_ptr_.data(),
                                static_cast<int>(bck_level_ptr_.size()) - 1, d_y_, d_x_);
            return;
        }
        if (use_levelset_) {
            // Forward  L y = x  (CSR of L, forward schedule):  d_x_ -> d_y_.
            levelset_solve(0, d_L_rowptr_, d_L_colidx_, d_L_vals_,
                           d_fwd_order_, fwd_level_ptr_.data(),
                           static_cast<int>(fwd_level_ptr_.size()) - 1, d_x_, d_y_);
            // Back  L^T z = y  (CSR of L^T, back schedule):  d_y_ -> d_x_.
            levelset_solve(0, d_rowPtr_, d_colIdx_, d_vals_,
                           d_bck_order_, bck_level_ptr_.data(),
                           static_cast<int>(bck_level_ptr_.size()) - 1, d_y_, d_x_);
            return;
        }
        const cuda_value_t alpha = cuda_value_t(1);
        // Forward:  L * d_y = d_x  ↔  TRANSPOSE on L^T.
        APXCHOL_CUSPARSE_CHECK(cusparseSpSV_solve(
            cusparse_, CUSPARSE_OPERATION_TRANSPOSE, &alpha,
            matLT_, vecX_, vecY_, cuda_value_dtype,
            CUSPARSE_SPSV_ALG_DEFAULT, descrFwd_));
        // Back:  L^T * d_x = d_y  ↔  NON_TRANSPOSE on L^T.
        APXCHOL_CUSPARSE_CHECK(cusparseSpSV_solve(
            cusparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            matLT_, vecY_, vecX_, cuda_value_dtype,
            CUSPARSE_SPSV_ALG_DEFAULT, descrBck_));
    }

    void destroy() {
        if (!ready_) return;
        if (use_levelset_) {
            cudaFree(d_L_rowptr_); cudaFree(d_L_colidx_); cudaFree(d_L_vals_); cudaFree(d_L_vals16_);
            cudaFree(d_fwd_order_); cudaFree(d_bck_order_);
            cudaFree(d_vals16_); cudaFree(d_diag_); cudaFree(d_inv_scale2_);
            cudaFree(d_df_tag_); cudaFree(d_df_ctrl_);
            cudaFree(d_fwd_batches_); cudaFree(d_bck_batches_);
            d_df_tag_ = nullptr; d_df_ctrl_ = nullptr;
            d_fwd_batches_ = d_bck_batches_ = nullptr; fwd_batches_ = bck_batches_ = 0;
            df_epoch_ = 0;
            d_L_rowptr_ = d_L_colidx_ = d_fwd_order_ = d_bck_order_ = nullptr;
            d_L_vals_ = nullptr; d_L_vals16_ = nullptr;
            d_vals16_ = nullptr; d_diag_ = d_inv_scale2_ = nullptr;
            df_grid_ = 0;
            fwd_level_ptr_.clear(); bck_level_ptr_.clear();
        } else {
            cusparse_teardown();
        }
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
        use_levelset_ = false;
        use_dataflow_ = false;
        backend_forced_ = false;
        backend_reason_.clear();
        fp16_ = false;
        diag_comp_ = false;
        factor_bytes_ = device_delta_bytes_ = cusparse_buf_bytes_ = 0;
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
    std::size_t cusparse_buf_bytes_ = 0;   // SpSV analysis buffers
    std::uint64_t fp16_flushed_ = 0, fp16_subnormal_ = 0;

    // Device memory (values + solve vectors at cuda_value_t width).
    int*          d_rowPtr_ = nullptr;
    int*          d_colIdx_ = nullptr;
    cuda_value_t* d_vals_   = nullptr;
    cuda_value_t* d_x_      = nullptr;
    cuda_value_t* d_y_      = nullptr;
    // Host staging for the fp32 host-PCG boundary (narrow/widen); unused at fp64.
    mutable std::vector<cuda_value_t> h_stage_;

    // Custom level-set backend (APXCHOL_GPU_SPTRSV=levelset). The stored d_* arrays
    // are CSR of L^T (back solve); the forward solve needs CSR of L (d_L_*). Each
    // direction has a device row-order (rows by level) + host level boundaries
    // (read to size the per-level launches).
    bool          use_levelset_  = false;
    bool          use_dataflow_  = false;   // implies use_levelset_ (shares its arrays); the fp32 build's AUTO
    bool          backend_forced_ = false;
    std::string   backend_reason_;
    std::size_t   reserve_bytes_  = 0;     // set_reserve_bytes(); NOT reset by destroy()
    int*          d_L_rowptr_     = nullptr;
    int*          d_L_colidx_     = nullptr;
    cuda_value_t* d_L_vals_       = nullptr;
    int*          d_fwd_order_    = nullptr;   // level-set schedules
    int*          d_bck_order_    = nullptr;
    std::vector<int> fwd_level_ptr_;
    std::vector<int> bck_level_ptr_;
    // Dataflow backend (APXCHOL_GPU_SPTRSV=dataflow; cuda_dataflow.h): the
    // warp batch tables of both directions, the m tagged words {value, epoch}
    // both sweeps publish into (mutable state, 8 B/row), the sweep epoch, 2+2
    // control ints and the persistent grid.
    int*          d_fwd_batches_   = nullptr;   // cuda_host::dataflow_batches
    int*          d_bck_batches_   = nullptr;
    int           fwd_batches_ = 0, bck_batches_ = 0;
    unsigned long long* d_df_tag_  = nullptr;
    mutable unsigned    df_epoch_  = 0;         // last epoch handed out (0 = none)
    int*          d_df_ctrl_       = nullptr;   // {fwd ticket, fwd finished, bck ticket, bck finished}
    int           df_grid_         = 0;
    // fp16 storage (APXCHOL_GPU_SPTRSV_FP16=1; level-set only): binary16 values
    // of CSR(L~^T) (d_vals16_, replaces d_vals_) and CSR(L~) (d_L_vals16_,
    // replaces d_L_vals_), the fp32 scaled diagonal and the back solve's
    // per-row input scale inv_scale^2.
    bool          fp16_           = false;
    bool          diag_comp_      = false;
    std::uint16_t* d_vals16_      = nullptr;
    std::uint16_t* d_L_vals16_    = nullptr;
    float*        d_diag_         = nullptr;
    float*        d_inv_scale2_   = nullptr;

    // cuSPARSE state.
    cusparseHandle_t      cusparse_ = nullptr;
    cusparseSpMatDescr_t  matLT_    = nullptr;
    cusparseDnVecDescr_t  vecX_     = nullptr;
    cusparseDnVecDescr_t  vecY_     = nullptr;
    cusparseSpSVDescr_t   descrFwd_ = nullptr;
    cusparseSpSVDescr_t   descrBck_ = nullptr;
    void*                 bufFwd_   = nullptr;
    void*                 bufBck_   = nullptr;
    size_t                bufSizeFwd_ = 0;
    size_t                bufSizeBck_ = 0;
};

#undef APXCHOL_CUDA_CHECK
#undef APXCHOL_CUSPARSE_CHECK

} // namespace apxchol
