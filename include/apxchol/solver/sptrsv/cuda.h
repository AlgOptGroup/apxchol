#pragma once
#include "apxchol/sparse_csc.h"
#include "apxchol/solver/sptrsv/cuda_cast.h"
#include "apxchol/solver/sptrsv/cuda_dataflow.h"
#include "apxchol/solver/sptrsv/cuda_host.h"
#include "apxchol/solver/sptrsv/factor_drop.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace apxchol {
namespace detail { class gpu_round_shadow_device_state; }

// GPU SpTRSV value type = the shared sptrsv_value_t (fp32): the factor
// (d_vals_) AND the internal solve vectors (d_x_/d_y_) all use this; the
// PCG-facing interface stays fp64 and casts at the boundary.
using cuda_value_t = sptrsv_value_t;
static_assert(sizeof(cuda_value_t) == 4, "the GPU SpTRSV runs in fp32");

/// Non-owning allocation descriptions used to spell the ownership handoff
/// below. `bytes` is the complete cudaMalloc allocation capacity, not only its
/// logical prefix, so geometrically grown R2b buffers remain honestly
/// accounted after transfer.
template<class T>
struct cuda_sptrsv_device_allocation {
    T* data = nullptr;
    std::size_t bytes = 0;
};

/// Non-owning allocation triples used only to spell the ownership handoff
/// below without six positional `void*` arguments. Every member must describe
/// the base of a distinct cudaMalloc allocation on `cuda_device`.
struct cuda_sptrsv_device_csr_fp32 {
    cuda_sptrsv_device_allocation<int> row_ptr;
    cuda_sptrsv_device_allocation<int> col_idx;
    cuda_sptrsv_device_allocation<cuda_value_t> values;
};

struct cuda_sptrsv_device_csr_fp16 {
    cuda_sptrsv_device_allocation<int> row_ptr;
    cuda_sptrsv_device_allocation<int> col_idx;
    cuda_sptrsv_device_allocation<std::uint16_t> values;
};

/// Move-only ownership capsule for the research-only direct CUDA SpTRSV
/// installation boundary. A successful own_*() call takes immediate ownership
/// of every supplied cudaMalloc base pointer. Destruction frees them on the
/// recorded CUDA device; moving leaves the source empty.
///
/// The arrays are already-final SpTRSV storage, not a raw elimination factor:
/// compacting drop, transpose construction, and (for fp16) scaled narrowing
/// have happened before this boundary. `stats` is mandatory provenance for
/// that preprocessing. own_*() validates only host metadata and pointer
/// uniqueness before ownership transfers; cuda_sptrsv's adoption call validates
/// the live allocations and downloaded CSR structures.
class cuda_sptrsv_device_factor {
public:
    enum class storage { fp32, fp16_scaled };

    cuda_sptrsv_device_factor() = default;
    cuda_sptrsv_device_factor(const cuda_sptrsv_device_factor&) = delete;
    cuda_sptrsv_device_factor& operator=(const cuda_sptrsv_device_factor&) = delete;

    cuda_sptrsv_device_factor(cuda_sptrsv_device_factor&& other) noexcept {
        move_from(other);
    }
    cuda_sptrsv_device_factor& operator=(cuda_sptrsv_device_factor&& other) noexcept {
        if (this != &other) {
            reset();
            move_from(other);
        }
        return *this;
    }

    ~cuda_sptrsv_device_factor() { reset(); }

    /// On success the returned capsule owns all six allocations. If metadata
    /// validation throws, ownership has NOT transferred and the caller still
    /// owns the raw pointers.
    static cuda_sptrsv_device_factor own_fp32(
            int cuda_device, std::int64_t m, std::int64_t nnz,
            cuda_sptrsv_device_csr_fp32 L,
            cuda_sptrsv_device_csr_fp32 LT,
            factor_drop_stats stats) {
        const std::size_t rows = static_cast<std::size_t>(m);
        const std::size_t entries = static_cast<std::size_t>(nnz);
        validate_metadata(cuda_device, m, nnz, stats,
                          {{L.row_ptr.data, L.row_ptr.bytes, (rows + 1) * sizeof(int)},
                           {L.col_idx.data, L.col_idx.bytes, entries * sizeof(int)},
                           {L.values.data, L.values.bytes, entries * sizeof(cuda_value_t)},
                           {LT.row_ptr.data, LT.row_ptr.bytes, (rows + 1) * sizeof(int)},
                           {LT.col_idx.data, LT.col_idx.bytes, entries * sizeof(int)},
                           {LT.values.data, LT.values.bytes, entries * sizeof(cuda_value_t)}});
        cuda_sptrsv_device_factor out;
        out.cuda_device_ = cuda_device;
        out.m_ = m;
        out.nnz_ = nnz;
        out.storage_ = storage::fp32;
        out.stats_ = stats;
        out.L_row_ptr_ = L.row_ptr.data;
        out.L_col_idx_ = L.col_idx.data;
        out.L_values_ = L.values.data;
        out.LT_row_ptr_ = LT.row_ptr.data;
        out.LT_col_idx_ = LT.col_idx.data;
        out.LT_values_ = LT.values.data;
        out.allocation_bytes_ = {L.row_ptr.bytes, L.col_idx.bytes, L.values.bytes,
                                 LT.row_ptr.bytes, LT.col_idx.bytes, LT.values.bytes,
                                 0, 0};
        return out;
    }

    /// FP16_SCALED form of the same handoff. `diag` and `inv_scale2` are
    /// separate cudaMalloc base allocations of m floats / m doubles. The two
    /// counters are the narrowing diagnostics reported by the upload path.
    static cuda_sptrsv_device_factor own_fp16_scaled(
            int cuda_device, std::int64_t m, std::int64_t nnz,
            cuda_sptrsv_device_csr_fp16 L,
            cuda_sptrsv_device_csr_fp16 LT,
            cuda_sptrsv_device_allocation<float> diag,
            cuda_sptrsv_device_allocation<double> inv_scale2,
            factor_drop_stats stats,
            std::uint64_t flushed, std::uint64_t subnormal) {
        const std::size_t rows = static_cast<std::size_t>(m);
        const std::size_t entries = static_cast<std::size_t>(nnz);
        validate_metadata(cuda_device, m, nnz, stats,
                          {{L.row_ptr.data, L.row_ptr.bytes, (rows + 1) * sizeof(int)},
                           {L.col_idx.data, L.col_idx.bytes, entries * sizeof(int)},
                           {L.values.data, L.values.bytes, entries * sizeof(std::uint16_t)},
                           {LT.row_ptr.data, LT.row_ptr.bytes, (rows + 1) * sizeof(int)},
                           {LT.col_idx.data, LT.col_idx.bytes, entries * sizeof(int)},
                           {LT.values.data, LT.values.bytes, entries * sizeof(std::uint16_t)},
                           {diag.data, diag.bytes, rows * sizeof(float)},
                           {inv_scale2.data, inv_scale2.bytes, rows * sizeof(double)}});
        cuda_sptrsv_device_factor out;
        out.cuda_device_ = cuda_device;
        out.m_ = m;
        out.nnz_ = nnz;
        out.storage_ = storage::fp16_scaled;
        out.stats_ = stats;
        out.fp16_flushed_ = flushed;
        out.fp16_subnormal_ = subnormal;
        out.L_row_ptr_ = L.row_ptr.data;
        out.L_col_idx_ = L.col_idx.data;
        out.L_values_ = L.values.data;
        out.LT_row_ptr_ = LT.row_ptr.data;
        out.LT_col_idx_ = LT.col_idx.data;
        out.LT_values_ = LT.values.data;
        out.diag_ = diag.data;
        out.inv_scale2_ = inv_scale2.data;
        out.allocation_bytes_ = {L.row_ptr.bytes, L.col_idx.bytes, L.values.bytes,
                                 LT.row_ptr.bytes, LT.col_idx.bytes, LT.values.bytes,
                                 diag.bytes, inv_scale2.bytes};
        return out;
    }

    bool empty() const noexcept { return L_row_ptr_ == nullptr; }
    int cuda_device() const noexcept { return cuda_device_; }
    std::int64_t rows() const noexcept { return m_; }
    std::int64_t nonzeros() const noexcept { return nnz_; }
    storage value_storage() const noexcept { return storage_; }

private:
    friend class cuda_sptrsv;
    friend class detail::gpu_round_shadow_device_state;

    struct allocation_metadata {
        const void* data;
        std::size_t bytes;
        std::size_t required;
    };

    static void validate_metadata(
            int cuda_device, std::int64_t m, std::int64_t nnz,
            const factor_drop_stats& stats,
            std::initializer_list<allocation_metadata> allocations) {
        if (cuda_device < 0)
            throw std::invalid_argument(
                "apxchol cuda_sptrsv adoption: CUDA device must be nonnegative");
        if (m <= 0 || m > std::numeric_limits<int>::max())
            throw std::invalid_argument(
                "apxchol cuda_sptrsv adoption: dimension must be positive and fit int32");
        if (nnz < m || nnz > std::numeric_limits<int>::max())
            throw std::invalid_argument(
                "apxchol cuda_sptrsv adoption: nnz must contain every diagonal and fit int32");
        if (!std::isfinite(stats.rel) || stats.rel < 0.0 ||
            stats.nnz_stored != static_cast<std::uint64_t>(nnz) ||
            stats.nnz_factor < stats.nnz_stored ||
            stats.dropped != stats.dropped_threshold + stats.dropped_flush ||
            stats.nnz_factor - stats.nnz_stored != stats.dropped)
            throw std::invalid_argument(
                "apxchol cuda_sptrsv adoption: inconsistent factor-drop provenance");
        std::vector<const void*> sorted;
        sorted.reserve(allocations.size());
        for (const allocation_metadata& allocation : allocations) {
            if (!allocation.data)
                throw std::invalid_argument(
                    "apxchol cuda_sptrsv adoption: every device allocation is required");
            if (allocation.bytes < allocation.required)
                throw std::invalid_argument(
                    "apxchol cuda_sptrsv adoption: a device allocation is smaller than its logical array");
            sorted.push_back(allocation.data);
        }
        std::sort(sorted.begin(), sorted.end(), std::less<const void*>{});
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
            throw std::invalid_argument(
                "apxchol cuda_sptrsv adoption: device allocations must be distinct bases");
    }

    void clear() noexcept {
        cuda_device_ = -1;
        m_ = nnz_ = 0;
        storage_ = storage::fp32;
        finalized_sorted_triangular_ = false;
        stats_ = factor_drop_stats{};
        fp16_flushed_ = fp16_subnormal_ = 0;
        L_row_ptr_ = LT_row_ptr_ = nullptr;
        L_col_idx_ = LT_col_idx_ = nullptr;
        L_values_ = LT_values_ = nullptr;
        diag_ = nullptr;
        inv_scale2_ = nullptr;
        allocation_bytes_.fill(0);
    }

    void move_from(cuda_sptrsv_device_factor& other) noexcept {
        cuda_device_ = other.cuda_device_;
        m_ = other.m_;
        nnz_ = other.nnz_;
        storage_ = other.storage_;
        finalized_sorted_triangular_ = other.finalized_sorted_triangular_;
        stats_ = other.stats_;
        fp16_flushed_ = other.fp16_flushed_;
        fp16_subnormal_ = other.fp16_subnormal_;
        L_row_ptr_ = other.L_row_ptr_;
        L_col_idx_ = other.L_col_idx_;
        L_values_ = other.L_values_;
        LT_row_ptr_ = other.LT_row_ptr_;
        LT_col_idx_ = other.LT_col_idx_;
        LT_values_ = other.LT_values_;
        diag_ = other.diag_;
        inv_scale2_ = other.inv_scale2_;
        allocation_bytes_ = other.allocation_bytes_;
        other.clear();
    }

    void reset() noexcept {
        if (empty()) return;
        int saved_device = -1;
        bool restore = false;
        if (cudaGetDevice(&saved_device) == cudaSuccess &&
            saved_device != cuda_device_ &&
            cudaSetDevice(cuda_device_) == cudaSuccess)
            restore = true;
        for (void* p : {static_cast<void*>(L_row_ptr_),
                        static_cast<void*>(L_col_idx_), L_values_,
                        static_cast<void*>(LT_row_ptr_),
                        static_cast<void*>(LT_col_idx_), LT_values_,
                        static_cast<void*>(diag_),
                        static_cast<void*>(inv_scale2_)})
            if (p) (void)cudaFree(p);
        clear();
        if (restore) (void)cudaSetDevice(saved_device);
    }

    int cuda_device_ = -1;
    std::int64_t m_ = 0;
    std::int64_t nnz_ = 0;
    storage storage_ = storage::fp32;
    bool finalized_sorted_triangular_ = false;
    factor_drop_stats stats_;
    std::uint64_t fp16_flushed_ = 0;
    std::uint64_t fp16_subnormal_ = 0;
    int* L_row_ptr_ = nullptr;
    int* L_col_idx_ = nullptr;
    void* L_values_ = nullptr;
    int* LT_row_ptr_ = nullptr;
    int* LT_col_idx_ = nullptr;
    void* LT_values_ = nullptr;
    float* diag_ = nullptr;
    double* inv_scale2_ = nullptr;
    std::array<std::size_t, 8> allocation_bytes_{};
};


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
        cleanup_needed_ = true;
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

    /// Research-only installation boundary for a factor whose final CSR(L)
    /// and CSR(L^T) storage already lives on the active CUDA device. This call
    /// is used by the FORCE-only resident-finalizer prototype. The ordinary
    /// production path remains the default and owns host preprocessing.
    ///
    /// Ownership and rollback:
    ///   * pass a move-only cuda_sptrsv_device_factor by value;
    ///   * existing factor state is destroyed, as in ordinary setup();
    ///   * ownership is consumed on entry even when validation/setup throws;
    ///   * before the final commit the capsule owns the factor allocations and
    ///     this object owns only newly allocated solve/schedule state;
    ///   * on failure both owners free their respective allocations and this
    ///     object is reset to empty; on success every factor allocation moves
    ///     here and lives until this cuda_sptrsv is destroyed.
    ///
    /// This first boundary still downloads both int32 CSR structures to build
    /// and validate the existing HOST dataflow plan (plus fp16 diag/inv-scale
    /// metadata when applicable), then uploads the O(m) plan tables. It never
    /// downloads factor values and never uploads any CSR structure or value.
    /// Capsules made by the private resident finalizer additionally avoid all
    /// index downloads: sorted triangular diagonals are known by construction,
    /// and the host planner needs only the two O(m) row-pointer arrays.
    void setup_adopting_device_factor_for_research(
            cuda_sptrsv_device_factor factor) {
        // A prior failed ordinary setup may have left partial allocations. The
        // ordinary setup calls destroy() before reuse too; make the fresh-state
        // precondition concrete before adopting another owner.
        destroy();
        if (factor.empty())
            throw std::invalid_argument(
                "apxchol cuda_sptrsv adoption: factor ownership capsule is empty");

        int current_device = -1;
        APXCHOL_CUDA_CHECK(cudaGetDevice(&current_device));
        if (current_device != factor.cuda_device_)
            throw std::invalid_argument(
                "apxchol cuda_sptrsv adoption: active CUDA device differs from the factor owner");
        validate_backend_env();
        const bool requested_fp16 = fp16_resolved();
        const bool factor_fp16 =
            factor.storage_ == cuda_sptrsv_device_factor::storage::fp16_scaled;
        if (requested_fp16 != factor_fp16)
            throw std::invalid_argument(
                "apxchol cuda_sptrsv adoption: device storage does not match APXCHOL_SPTRSV_FP16 resolution");
        const double requested_drop = factor_drop_rel_from_env();
        if (factor.stats_.rel != requested_drop)
            throw std::invalid_argument(
                "apxchol cuda_sptrsv adoption: preprocessed drop provenance does not match APXCHOL_FACTOR_DROP");

        const int m = static_cast<int>(factor.m_);
        const std::size_t rows = static_cast<std::size_t>(m);
        const std::size_t nnz = static_cast<std::size_t>(factor.nnz_);
        const std::size_t ptr_bytes = (rows + 1) * sizeof(int);
        const std::size_t idx_bytes = nnz * sizeof(int);
        const std::size_t val_bytes = nnz * (factor_fp16
            ? sizeof(std::uint16_t) : sizeof(cuda_value_t));

        std::size_t adopted_bytes = 0;
        std::size_t allocation_index = 0;
        auto account = [&](const void* pointer, std::size_t required,
                           const char* name) {
            const std::size_t bytes =
                factor.allocation_bytes_[allocation_index++];
            if (bytes < required)
                throw std::logic_error(
                    "apxchol cuda_sptrsv adoption: ownership metadata lost its validated capacity");
            validate_device_allocation(pointer, current_device, name);
            if (bytes > std::numeric_limits<std::size_t>::max() - adopted_bytes)
                throw std::overflow_error(
                    "apxchol cuda_sptrsv adoption: device allocation byte total overflows size_t");
            adopted_bytes += bytes;
        };
        account(factor.L_row_ptr_, ptr_bytes, "CSR(L) row pointers");
        account(factor.L_col_idx_, idx_bytes, "CSR(L) column indices");
        account(factor.L_values_, val_bytes, "CSR(L) values");
        account(factor.LT_row_ptr_, ptr_bytes, "CSR(L^T) row pointers");
        account(factor.LT_col_idx_, idx_bytes, "CSR(L^T) column indices");
        account(factor.LT_values_, val_bytes, "CSR(L^T) values");
        if (factor_fp16) {
            account(factor.diag_, rows * sizeof(float), "fp16 scaled diagonal");
            account(factor.inv_scale2_, rows * sizeof(double), "fp16 inverse scale squared");
        }

        const bool trace = std::getenv("APXCHOL_SPTRSV_SETUP_TRACE") != nullptr;
        auto t_prev = std::chrono::steady_clock::now();
        auto mark = [&](const char* what) {
            if (!trace) return;
            APXCHOL_CUDA_CHECK(cudaDeviceSynchronize());
            const auto now = std::chrono::steady_clock::now();
            std::fprintf(stderr, "[sptrsv-adopt gpu] %-22s %8.2f ms\n", what,
                         std::chrono::duration<double, std::milli>(now - t_prev).count());
            t_prev = std::chrono::steady_clock::now();
        };

        std::vector<int> L_ptr(rows + 1), LT_ptr(rows + 1);
        std::vector<int> L_idx, LT_idx;
        APXCHOL_CUDA_CHECK(cudaMemcpy(
            L_ptr.data(), factor.L_row_ptr_, ptr_bytes, cudaMemcpyDeviceToHost));
        APXCHOL_CUDA_CHECK(cudaMemcpy(
            LT_ptr.data(), factor.LT_row_ptr_, ptr_bytes, cudaMemcpyDeviceToHost));
        std::size_t adoption_download_bytes = 2 * ptr_bytes;
        if (factor.finalized_sorted_triangular_) {
            // The private finalizer constructed both sorted triangular CSRs
            // from one validated coordinate stream and checked duplicates on
            // device. Only O(m) row pointers are needed for host plan packing.
            for (const auto* ptr : {&L_ptr, &LT_ptr}) {
                if (ptr->front() != 0 || ptr->back() != factor.nnz_)
                    throw std::invalid_argument("apxchol cuda_sptrsv adoption: invalid finalized pointer endpoints");
                for (int row = 0; row < m; ++row)
                    if ((*ptr)[row] >= (*ptr)[row + 1])
                        throw std::invalid_argument("apxchol cuda_sptrsv adoption: empty or unordered finalized row");
            }
            mark("download row pointers");
        } else {
            L_idx.resize(nnz); LT_idx.resize(nnz);
            APXCHOL_CUDA_CHECK(cudaMemcpy(
                L_idx.data(), factor.L_col_idx_, idx_bytes, cudaMemcpyDeviceToHost));
            APXCHOL_CUDA_CHECK(cudaMemcpy(
                LT_idx.data(), factor.LT_col_idx_, idx_bytes, cudaMemcpyDeviceToHost));
            adoption_download_bytes += 2 * idx_bytes;
            mark("download CSR structure");
            const std::string structure_error =
                cuda_host::dataflow_factor_structure_check(
                    m, factor.nnz_, L_ptr.data(), L_idx.data(),
                    LT_ptr.data(), LT_idx.data());
            if (!structure_error.empty())
                throw std::invalid_argument("apxchol cuda_sptrsv adoption: " + structure_error);
        }

        if (factor_fp16) {
            std::vector<float> diag(rows);
            std::vector<double> inv_scale2(rows);
            APXCHOL_CUDA_CHECK(cudaMemcpy(
                diag.data(), factor.diag_, rows * sizeof(float),
                cudaMemcpyDeviceToHost));
            APXCHOL_CUDA_CHECK(cudaMemcpy(
                inv_scale2.data(), factor.inv_scale2_, rows * sizeof(double),
                cudaMemcpyDeviceToHost));
            adoption_download_bytes +=
                rows * (sizeof(float) + sizeof(double));
            for (int row = 0; row < m; ++row)
                if (!(std::isfinite(diag[static_cast<std::size_t>(row)]) &&
                      diag[static_cast<std::size_t>(row)] != 0.0f &&
                      std::isfinite(inv_scale2[static_cast<std::size_t>(row)])))
                    throw std::invalid_argument(
                        "apxchol cuda_sptrsv adoption: invalid fp16 scaled metadata at row " +
                        std::to_string(row));
        }
        mark("validate adoption input");

        size_t free_before = 0, total = 0;
        APXCHOL_CUDA_CHECK(cudaMemGetInfo(&free_before, &total));
        try {
            cleanup_needed_ = true;
            m_ = factor.m_;
            nnz_ = factor.nnz_;
            stats_ = factor.stats_;
            fp16_ = factor_fp16;
            fp16_flushed_ = factor.fp16_flushed_;
            fp16_subnormal_ = factor.fp16_subnormal_;
            adopted_device_factor_ = true;
            adopted_cuda_device_ = current_device;
            adoption_host_download_bytes_ = adoption_download_bytes;
            factor_bytes_ = adopted_bytes;

            APXCHOL_CUDA_CHECK(cudaMalloc(&d_x_, rows * sizeof(cuda_value_t)));
            APXCHOL_CUDA_CHECK(cudaMalloc(&d_y_, rows * sizeof(cuda_value_t)));
            h_stage_.resize(rows);
            mark("allocate solve vectors");
            setup_kernel_backend(LT_ptr, L_ptr, LT_idx.data(), L_idx.data(),
                                 factor.finalized_sorted_triangular_);
            mark("kernel backend tables");

            size_t free_after = 0;
            APXCHOL_CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
            device_delta_bytes_ =
                free_before >= free_after ? free_before - free_after : 0;

            // Final no-throw ownership commit. Until here `factor` remained the
            // sole owner of every input allocation.
            d_L_rowptr_ = factor.L_row_ptr_;
            d_L_colidx_ = factor.L_col_idx_;
            d_rowPtr_ = factor.LT_row_ptr_;
            d_colIdx_ = factor.LT_col_idx_;
            if (fp16_) {
                d_L_vals16_ = static_cast<std::uint16_t*>(factor.L_values_);
                d_vals16_ = static_cast<std::uint16_t*>(factor.LT_values_);
                d_diag_ = factor.diag_;
                d_inv_scale2_ = factor.inv_scale2_;
            } else {
                d_L_vals_ = static_cast<cuda_value_t*>(factor.L_values_);
                d_vals_ = static_cast<cuda_value_t*>(factor.LT_values_);
            }
            factor.clear();
            ready_ = true;

            if (std::getenv("APXCHOL_VERBOSE") ||
                std::getenv("APXCHOL_GPU_MEM_DEBUG"))
                std::fprintf(stderr,
                    "[apxchol] sptrsv storage GPU/dataflow adopted %s: "
                    "stored_nnz=%llu factor_device_bytes=%.1f MB "
                    "host_structure_download=%.1f MB install_delta=%.1f MB\n",
                    fp16_ ? "fp16" : value_name,
                    static_cast<unsigned long long>(stats_.nnz_stored),
                    factor_bytes_ / 1e6,
                    adoption_host_download_bytes_ / 1e6,
                    device_delta_bytes_ / 1e6);
        } catch (...) {
            // Factor pointers have not moved before the no-throw commit above;
            // destroy() owns only partial solve/plan allocations here.
            destroy();
            throw;
        }
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
    /// True only for setup_adopting_device_factor_for_research(). The ordinary
    /// production upload path always reports false / zero.
    bool adopted_device_factor() const { return adopted_device_factor_; }
    /// Exact D2H bytes: row pointers for trusted finalized factors; complete
    /// int32 structures and fp16 metadata for externally supplied capsules.
    std::size_t adoption_host_download_bytes() const {
        return adoption_host_download_bytes_;
    }
private:
    static void validate_backend_env() {
        const char* value = std::getenv("APXCHOL_GPU_SPTRSV");
        if (value && *value && std::string(value) != "dataflow")
            throw std::invalid_argument(
                std::string("APXCHOL_GPU_SPTRSV='") + value +
                "' is unsupported; the GPU backend is dataflow");
    }

    static void validate_device_allocation(
            const void* pointer, int expected_device,
            const char* name) {
        cudaPointerAttributes attributes{};
        const cudaError_t attr_status =
            cudaPointerGetAttributes(&attributes, pointer);
        if (attr_status != cudaSuccess)
            throw std::invalid_argument(
                std::string("apxchol cuda_sptrsv adoption: cannot inspect ") +
                name + ": " + cudaGetErrorString(attr_status));
#if CUDART_VERSION >= 10000
        const cudaMemoryType memory_type = attributes.type;
#else
        const cudaMemoryType memory_type = attributes.memoryType;
#endif
        if (memory_type != cudaMemoryTypeDevice ||
            attributes.device != expected_device)
            throw std::invalid_argument(
                std::string("apxchol cuda_sptrsv adoption: ") + name +
                " is not device memory on the recorded CUDA device");
    }

    // The dataflow backend's host-side tables (cuda_dataflow.h): no levels at
    // all -- the warp batch tables of both directions (from the CSR row
    // lengths), the m tagged words, the control ints, the resident grid.
    void setup_kernel_backend(const std::vector<int>& LT_ptr, const std::vector<int>& L_ptr,
                              const int* LT_idx, const int* L_idx, bool sorted_triangular = false) {
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
            mi, false, L_ptr.data(), L_idx, len_L.data(), pre, sp_f, sorted_triangular);
        const cuda_host::dataflow_plan bck = cuda_host::dataflow_build_plan(
            mi, true, LT_ptr.data(), LT_idx, len_LT.data(), pre, sp_b, sorted_triangular);
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

    void destroy() noexcept {
        // setup() calls destroy() on a pristine object. Do not turn that host-
        // only no-op into cudaFree(nullptr): it would be eligible to create the
        // lazy CUDA context inside the timed production setup boundary.
        if (!cleanup_needed_) return;
        int saved_device = -1;
        bool restore_device = false;
        if (adopted_device_factor_ && adopted_cuda_device_ >= 0 &&
            cudaGetDevice(&saved_device) == cudaSuccess &&
            saved_device != adopted_cuda_device_ &&
            cudaSetDevice(adopted_cuda_device_) == cudaSuccess)
            restore_device = true;

        // Free by pointer presence, not by ready_: setup/adoption exceptions can
        // leave a valid partial allocation graph before the final ready commit.
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
        adopted_device_factor_ = false;
        adopted_cuda_device_ = -1;
        adoption_host_download_bytes_ = 0;
        m_ = nnz_ = 0;
        cleanup_needed_ = false;
        if (restore_device) (void)cudaSetDevice(saved_device);
    }

    int64_t m_ = 0;
    int64_t nnz_ = 0;
    bool ready_ = false;
    bool cleanup_needed_ = false;

    // What the compacting drop did (factor_drop.h) and the device footprint.
    factor_drop_stats stats_;
    std::size_t factor_bytes_ = 0;         // cudaMalloc'd factor arrays (see factor_device_bytes)
    std::size_t device_delta_bytes_ = 0;   // cudaMemGetInfo free delta over setup
    std::uint64_t fp16_flushed_ = 0, fp16_subnormal_ = 0;
    bool adopted_device_factor_ = false;
    int adopted_cuda_device_ = -1;
    std::size_t adoption_host_download_bytes_ = 0;

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
