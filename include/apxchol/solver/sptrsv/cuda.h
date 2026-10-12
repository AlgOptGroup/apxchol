#pragma once
#include "apxchol/solver/detail/gpu_diagnostics.h"
#include "apxchol/solver/detail/cuda_device_scope.h"
#include "apxchol/sparse_csc.h"
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
#include <span>
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
// Host _Float16 and device __half buffers share the binary16 representation.
static_assert(sizeof(_Float16) == sizeof(__half));

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
    cuda_sptrsv_device_allocation<__half> values;
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

    cuda_sptrsv_device_factor(cuda_sptrsv_device_factor&& other) noexcept
        : data_(std::exchange(other.data_, {})) {}
    cuda_sptrsv_device_factor& operator=(cuda_sptrsv_device_factor&& other) noexcept {
        if (this != &other) {
            reset();
            data_ = std::exchange(other.data_, {});
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
        out.data_.cuda_device = cuda_device;
        out.data_.m = m;
        out.data_.nnz = nnz;
        out.data_.format = storage::fp32;
        out.data_.stats = stats;
        out.data_.L_row_ptr = L.row_ptr.data;
        out.data_.L_col_idx = L.col_idx.data;
        out.data_.L_values = L.values.data;
        out.data_.LT_row_ptr = LT.row_ptr.data;
        out.data_.LT_col_idx = LT.col_idx.data;
        out.data_.LT_values = LT.values.data;
        out.data_.allocation_bytes = {L.row_ptr.bytes, L.col_idx.bytes, L.values.bytes,
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
                           {L.values.data, L.values.bytes, entries * sizeof(__half)},
                           {LT.row_ptr.data, LT.row_ptr.bytes, (rows + 1) * sizeof(int)},
                           {LT.col_idx.data, LT.col_idx.bytes, entries * sizeof(int)},
                           {LT.values.data, LT.values.bytes, entries * sizeof(__half)},
                           {diag.data, diag.bytes, rows * sizeof(float)},
                           {inv_scale2.data, inv_scale2.bytes, rows * sizeof(double)}});
        cuda_sptrsv_device_factor out;
        out.data_.cuda_device = cuda_device;
        out.data_.m = m;
        out.data_.nnz = nnz;
        out.data_.format = storage::fp16_scaled;
        out.data_.stats = stats;
        out.data_.fp16_flushed = flushed;
        out.data_.fp16_subnormal = subnormal;
        out.data_.L_row_ptr = L.row_ptr.data;
        out.data_.L_col_idx = L.col_idx.data;
        out.data_.L_values = L.values.data;
        out.data_.LT_row_ptr = LT.row_ptr.data;
        out.data_.LT_col_idx = LT.col_idx.data;
        out.data_.LT_values = LT.values.data;
        out.data_.diag = diag.data;
        out.data_.inv_scale2 = inv_scale2.data;
        out.data_.allocation_bytes = {L.row_ptr.bytes, L.col_idx.bytes, L.values.bytes,
                                 LT.row_ptr.bytes, LT.col_idx.bytes, LT.values.bytes,
                                 diag.bytes, inv_scale2.bytes};
        return out;
    }

    bool empty() const noexcept { return data_.L_row_ptr == nullptr; }
    int cuda_device() const noexcept { return data_.cuda_device; }
    std::int64_t rows() const noexcept { return data_.m; }
    std::int64_t nonzeros() const noexcept { return data_.nnz; }
    storage value_storage() const noexcept { return data_.format; }

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

    void clear() noexcept { data_ = {}; }

    void reset() noexcept {
        if (empty()) return;
        const detail::cuda_device_scope device(data_.cuda_device);
        for (void* p : {static_cast<void*>(data_.L_row_ptr),
                        static_cast<void*>(data_.L_col_idx), data_.L_values,
                        static_cast<void*>(data_.LT_row_ptr),
                        static_cast<void*>(data_.LT_col_idx), data_.LT_values,
                        static_cast<void*>(data_.diag),
                        static_cast<void*>(data_.inv_scale2)})
            if (p) (void)cudaFree(p);
        clear();
    }

    struct state {
        int cuda_device = -1;
        std::int64_t m = 0;
        std::int64_t nnz = 0;
        storage format = storage::fp32;
        bool finalized_sorted_triangular = false;
        factor_drop_stats stats;
        std::uint64_t fp16_flushed = 0;
        std::uint64_t fp16_subnormal = 0;
        int* L_row_ptr = nullptr;
        int* L_col_idx = nullptr;
        void* L_values = nullptr;
        int* LT_row_ptr = nullptr;
        int* LT_col_idx = nullptr;
        void* LT_values = nullptr;
        float* diag = nullptr;
        double* inv_scale2 = nullptr;
        std::array<std::size_t, 8> allocation_bytes{};
    } data_;
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
/// CSR of L and L^T, packed task tables and atomic FP32 dependency values
/// support an initialization launch followed by one persistent solve launch,
/// without level schedules. Results are bit-deterministic across repeated
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
/// FP16 STORAGE (env APXCHOL_FACTOR_STORAGE, shared with the CPU backend --
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
    static bool fp16_resolved(factor_storage_type storage = factor_storage_type::automatic) {
        return detail::resolve_fp16_storage(storage, true);
    }
    /// Setup: build L11 on the host, run the compacting drop, (fp16: narrow),
    /// copy to device, build the dataflow tables. Env APXCHOL_SPTRSV_SETUP_TRACE=1 (the
    /// CPU backend's knob, same name) prints the per-stage wall times of one
    /// setup to stderr (diagnostic). Since 2026-08-20 the per-process CUDA
    /// context creation is NOT part of any stage here: the caller establishes
    /// it before this call and reports it as "cuda_init"
    /// (solver/cuda_context.h, apx_cholesky::install_factor); it used to be
    /// charged to "build_L11", ~100-135 ms on this machine, ~715 ms on GH200.
    void setup(const sparse_csc& L, node_index m,
               factor_storage_type storage = factor_storage_type::automatic) {
        validate_backend_env();
        const bool fp16 = fp16_resolved(storage);
        destroy();
        data_.m = static_cast<int64_t>(m);
        const bool trace = detail::gpu_setup_diagnostics() &&
            std::getenv("APXCHOL_SPTRSV_SETUP_TRACE") != nullptr;
        auto t_prev = detail::gpu_setup_diagnostic_clock::now();
        auto mark = [&](const char* what) {
            if (!trace) return;
            cudaDeviceSynchronize();
            const auto now = std::chrono::steady_clock::now();
            std::fprintf(stderr, "[sptrsv-setup gpu] %-22s %8.2f ms\n", what,
                         std::chrono::duration<double, std::milli>(now - t_prev).count());
            t_prev = std::chrono::steady_clock::now();
        };

        data_.fp16 = fp16;
        if (detail::gpu_setup_diagnostics() && std::getenv("APXCHOL_VERBOSE"))
            std::fprintf(stderr, "[apxchol] GPU SpTRSV backend: %s (%s storage)\n",
                         backend_name(), data_.fp16 ? "fp16" : "fp32");
        const double factor_drop_rel = factor_drop_rel_from_env();

        // Build L11 = top-left m×m block of the factor in CSC, with checked
        // 32-bit indices for the dataflow kernels.
        // The same three arrays are CSR of L^T (the back solve's operand).
        cuda_host::csr_int<cuda_value_t> LT = cuda_host::build_L11_csc_int<cuda_value_t>(L, data_.m);
        mark("build_L11");

        // Per-column scales (the drop's threshold reference; the fp16 storage's
        // scale), from the factor BEFORE the drop.
        std::vector<float> col_scale;
        if (factor_drop_rel > 0.0 || data_.fp16) col_scale = cuda_host::column_scales(LT);
        // fp16 storage only: a column whose scale cannot be represented --
        // fp32 1/s_j (s_j < ~3e-39) or the scaled diagonal fp32(L_jj)/s_j
        // (off-diagonals >= ~1e38x below the diagonal) would overflow --
        // falls back to s_j = 1 (the "no off-diagonals" convention): its
        // off-diagonals are below anything an fp32 sweep could see next to
        // that diagonal, and the drop/flush below removes them. Applied
        // BEFORE the drop so the threshold and the storage agree on the
        // scale.
        if (data_.fp16)
            for (int64_t j = 0; j < data_.m; ++j) {
                const float s = col_scale[static_cast<std::size_t>(j)];
                const float d = static_cast<float>(LT.vals[LT.ptr[static_cast<std::size_t>(j)]]);
                if (!std::isfinite(1.0f / s) || !std::isfinite(d / s))
                    col_scale[static_cast<std::size_t>(j)] = 1.0f;
            }

        // ── Compacting drop (factor_drop.h; see the class comment) ──
        data_.stats = cuda_host::apply_factor_drop(LT, col_scale, factor_drop_rel, data_.fp16);
        data_.nnz = LT.nnz;
        mark("col_scale+drop");

        size_t free_before = 0, total = 0;
        APXCHOL_CUDA_CHECK(cudaMemGetInfo(&free_before, &total));
        data_.cleanup_needed = true;
        data_.factor_bytes = 0;
        // Both storage formats use the same transpose, upload and plan construction.
        auto upload_transpose = [&](const auto& back, auto*& values, const char* stage) {
            const auto forward = cuda_host::transpose_csr(back);
            mark(stage);
            upload_factor(data_.d_L_rowptr, forward.ptr.data(), data_.m + 1);
            upload_factor(data_.d_L_colidx, forward.idx.get(), data_.nnz);
            upload_factor(values, forward.vals.get(), data_.nnz);
            mark("upload_L");
            setup_kernel_backend(back.ptr, forward.ptr);
            mark("kernel_backend_tables");
        };

        // Structure of CSR(L^T) (every storage).
        upload_factor(data_.d_rowPtr, LT.ptr.data(), data_.m + 1);
        upload_factor(data_.d_colIdx, LT.idx.get(), data_.nnz);

        // Allocate device vectors (solve runs at cuda_value_t width).
        APXCHOL_CUDA_CHECK(cudaMalloc(&data_.d_x, data_.m * sizeof(cuda_value_t)));
        APXCHOL_CUDA_CHECK(cudaMalloc(&data_.d_y, data_.m * sizeof(cuda_value_t)));
        h_stage_.resize(static_cast<size_t>(data_.m));
        mark("upload_LT_struct+vecs");

        if (data_.fp16) {
            // FP16 per-column-scaled storage (cuda_host.h contract): values of
            // CSR(L^T) narrowed on the host, fp32 diag / inv_scale^2 alongside,
            // then the transpose (carrying the same 16-bit patterns) for CSR(L).
            cuda_host::fp16_scaled_arrays h16 = cuda_host::narrow_fp16_scaled(LT, col_scale);
            mark("  fp16_narrow");
            data_.fp16_flushed = h16.flushed; data_.fp16_subnormal = h16.subnormal;
            // r_j^2 in DOUBLE (exact; the kernels multiply rhs * r_j^2 in
            // double and narrow once -- pre-squaring into fp32 overflows for
            // s_j < ~5.4e-20 and the Inf became silent NaN downstream).
            std::vector<double> inv_scale2(static_cast<std::size_t>(data_.m));
            for (int64_t j = 0; j < data_.m; ++j) {
                const double r = static_cast<double>(h16.inv_scale[j]);
                inv_scale2[j] = r * r;
            }
            // Fail LOUDLY on a factor the fp16 storage cannot represent
            // (unreachable after the scale fallback above -- a guard, not a
            // policy): a non-finite diag/inv_scale would dissolve the solve
            // into NaN with no error anywhere.
            for (int64_t j = 0; j < data_.m; ++j)
                if (!(std::isfinite(h16.diag[j]) && h16.diag[j] != 0.0f && std::isfinite(inv_scale2[j])))
                    throw std::runtime_error(
                        "apxchol cuda_sptrsv: fp16 factor storage cannot represent column " + std::to_string(j) +
                        " (diag=" + std::to_string(h16.diag[j]) + ", inv_scale^2=" + std::to_string(inv_scale2[j]) +
                        "); set APXCHOL_FACTOR_STORAGE=float32");
            upload_factor(data_.d_vals16, h16.vals.get(), data_.nnz);
            upload_factor(data_.d_diag, h16.diag.data(), data_.m);
            upload_factor(data_.d_inv_scale2, inv_scale2.data(), data_.m);
            // CSR of L for the forward solve: transpose of (structure of LT,
            // fp16 values).
            cuda_host::csr_int<_Float16> LT16;
            LT16.m = LT.m; LT16.nnz = LT.nnz; LT16.ptr = LT.ptr;
            LT16.idx  = std::move(LT.idx);
            LT16.vals = std::move(h16.vals);
            mark("  fp16_diag+uploads");
            upload_transpose(LT16, data_.d_L_vals16, "  fp16_transpose");
        } else {
            upload_factor(data_.d_vals, LT.vals.get(), data_.nnz);
            mark("upload_LT_vals");
            upload_transpose(LT, data_.d_L_vals, "transpose");
        }
        size_t free_after = 0;
        APXCHOL_CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
        data_.device_delta_bytes = free_before >= free_after ? free_before - free_after : 0;

        if (std::getenv("APXCHOL_VERBOSE") || std::getenv("APXCHOL_GPU_MEM_DEBUG")) {
            const std::uint64_t off = data_.stats.nnz_stored - static_cast<std::uint64_t>(data_.m);
            const std::uint64_t off0 = off + data_.stats.dropped;
            std::fprintf(stderr,
                "[apxchol] sptrsv storage GPU/%s %s: stored_nnz=%llu offdiag=%llu%s"
                " | factor device bytes=%.1f MB (cudaMemGetInfo delta %.1f MB; free %.2f -> %.2f of %.2f GB)\n",
                backend_name(),
                data_.fp16 ? "fp16 (per-column scaled, diag fp32 + rounding residual)" : value_name,
                static_cast<unsigned long long>(data_.stats.nnz_stored), static_cast<unsigned long long>(off),
                data_.fp16 ? (" flushed_to_zero=" + std::to_string(data_.fp16_flushed) + " subnormal=" + std::to_string(data_.fp16_subnormal)).c_str() : "",
                data_.factor_bytes / 1e6, data_.device_delta_bytes / 1e6, free_before / 1e9, free_after / 1e9, total / 1e9);
            if (data_.stats.rel > 0.0)
                std::fprintf(stderr,
                    "[apxchol] factor drop (APXCHOL_FACTOR_DROP=%g, column sums preserved): dropped=%llu"
                    " (%.4f%% of %llu off-diagonals; threshold=%llu, format_zero=%llu)"
                    " stored_nnz %llu -> %llu (%.4f%% of factor)\n",
                    data_.stats.rel,
                    static_cast<unsigned long long>(data_.stats.dropped),
                    100.0 * static_cast<double>(data_.stats.dropped) / static_cast<double>(off0 ? off0 : 1),
                    static_cast<unsigned long long>(off0),
                    static_cast<unsigned long long>(data_.stats.dropped_threshold),
                    static_cast<unsigned long long>(data_.stats.dropped_flush),
                    static_cast<unsigned long long>(data_.stats.nnz_factor),
                    static_cast<unsigned long long>(data_.stats.nnz_stored),
                    100.0 * static_cast<double>(data_.stats.nnz_stored) / static_cast<double>(data_.stats.nnz_factor ? data_.stats.nnz_factor : 1));
        }
        data_.ready = true;
    }

    /// Install final CSR(L) and CSR(L^T) storage on the active CUDA device.
    /// Production uses the private resident finalizer; diagnostic callers may
    /// supply generic ownership capsules, which receive full validation.
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
    /// Generic capsules download and validate both CSR structures before host
    /// planning (plus fp16 scaling metadata). Capsules made by the private
    /// resident finalizer instead validate row pointers and build the schedule
    /// on device: only fixed-size statistics/counts cross to the host. Neither
    /// route downloads factor values or uploads a factor CSR again.
    void setup_adopting_device_factor_for_research(
            cuda_sptrsv_device_factor factor,
            factor_storage_type storage = factor_storage_type::automatic) {
        // A prior failed ordinary setup may have left partial allocations. The
        // ordinary setup calls destroy() before reuse too; make the fresh-state
        // precondition concrete before adopting another owner.
        destroy();
        if (factor.empty())
            throw std::invalid_argument(
                "apxchol cuda_sptrsv adoption: factor ownership capsule is empty");

        int current_device = -1;
        APXCHOL_CUDA_CHECK(cudaGetDevice(&current_device));
        if (current_device != factor.data_.cuda_device)
            throw std::invalid_argument(
                "apxchol cuda_sptrsv adoption: active CUDA device differs from the factor owner");
        validate_backend_env();
        const bool requested_fp16 = fp16_resolved(storage);
        const bool factor_fp16 =
            factor.data_.format == cuda_sptrsv_device_factor::storage::fp16_scaled;
        if (requested_fp16 != factor_fp16)
            throw std::invalid_argument(
                "apxchol cuda_sptrsv adoption: device storage does not match requested factor storage");
        const double requested_drop = factor_drop_rel_from_env();
        if (factor.data_.stats.rel != requested_drop)
            throw std::invalid_argument(
                "apxchol cuda_sptrsv adoption: preprocessed drop provenance does not match APXCHOL_FACTOR_DROP");

        const int m = static_cast<int>(factor.data_.m);
        const std::size_t rows = static_cast<std::size_t>(m);
        const std::size_t nnz = static_cast<std::size_t>(factor.data_.nnz);
        const std::size_t ptr_bytes = (rows + 1) * sizeof(int);
        const std::size_t idx_bytes = nnz * sizeof(int);
        const std::size_t val_bytes = nnz * (factor_fp16
            ? sizeof(__half) : sizeof(cuda_value_t));

        std::size_t adopted_bytes = 0;
        std::size_t allocation_index = 0;
        auto account = [&](const void* pointer, std::size_t required,
                           const char* name) {
            const std::size_t bytes =
                factor.data_.allocation_bytes[allocation_index++];
            if (bytes < required)
                throw std::logic_error(
                    "apxchol cuda_sptrsv adoption: ownership metadata lost its validated capacity");
            validate_device_allocation(pointer, current_device, name);
            if (bytes > std::numeric_limits<std::size_t>::max() - adopted_bytes)
                throw std::overflow_error(
                    "apxchol cuda_sptrsv adoption: device allocation byte total overflows size_t");
            adopted_bytes += bytes;
        };
        account(factor.data_.L_row_ptr, ptr_bytes, "CSR(L) row pointers");
        account(factor.data_.L_col_idx, idx_bytes, "CSR(L) column indices");
        account(factor.data_.L_values, val_bytes, "CSR(L) values");
        account(factor.data_.LT_row_ptr, ptr_bytes, "CSR(L^T) row pointers");
        account(factor.data_.LT_col_idx, idx_bytes, "CSR(L^T) column indices");
        account(factor.data_.LT_values, val_bytes, "CSR(L^T) values");
        if (factor_fp16) {
            account(factor.data_.diag, rows * sizeof(float), "fp16 scaled diagonal");
            account(factor.data_.inv_scale2, rows * sizeof(double), "fp16 inverse scale squared");
        }

        const bool trace = detail::gpu_setup_diagnostics() &&
            std::getenv("APXCHOL_SPTRSV_SETUP_TRACE") != nullptr;
        auto t_prev = detail::gpu_setup_diagnostic_clock::now();
        auto mark = [&](const char* what) {
            if (!trace) return;
            APXCHOL_CUDA_CHECK(cudaDeviceSynchronize());
            const auto now = std::chrono::steady_clock::now();
            std::fprintf(stderr, "[sptrsv-adopt gpu] %-22s %8.2f ms\n", what,
                         std::chrono::duration<double, std::milli>(now - t_prev).count());
            t_prev = std::chrono::steady_clock::now();
        };

        std::vector<int> L_ptr, LT_ptr, L_idx, LT_idx;
        std::size_t adoption_download_bytes = 0;
        if (!factor.data_.finalized_sorted_triangular) {
            L_ptr.resize(rows + 1); LT_ptr.resize(rows + 1);
            L_idx.resize(nnz); LT_idx.resize(nnz);
            APXCHOL_CUDA_CHECK(cudaMemcpy(
                L_ptr.data(), factor.data_.L_row_ptr, ptr_bytes, cudaMemcpyDeviceToHost));
            APXCHOL_CUDA_CHECK(cudaMemcpy(
                LT_ptr.data(), factor.data_.LT_row_ptr, ptr_bytes, cudaMemcpyDeviceToHost));
            APXCHOL_CUDA_CHECK(cudaMemcpy(
                L_idx.data(), factor.data_.L_col_idx, idx_bytes, cudaMemcpyDeviceToHost));
            APXCHOL_CUDA_CHECK(cudaMemcpy(
                LT_idx.data(), factor.data_.LT_col_idx, idx_bytes, cudaMemcpyDeviceToHost));
            adoption_download_bytes = 2 * (ptr_bytes + idx_bytes);
            mark("download CSR structure");
            const std::string structure_error =
                cuda_host::dataflow_factor_structure_check(
                    m, factor.data_.nnz, L_ptr.data(), L_idx.data(),
                    LT_ptr.data(), LT_idx.data());
            if (!structure_error.empty())
                throw std::invalid_argument("apxchol cuda_sptrsv adoption: " + structure_error);
        }

        if (factor_fp16 && !factor.data_.finalized_sorted_triangular) {
            std::vector<float> diag(rows);
            std::vector<double> inv_scale2(rows);
            APXCHOL_CUDA_CHECK(cudaMemcpy(
                diag.data(), factor.data_.diag, rows * sizeof(float),
                cudaMemcpyDeviceToHost));
            APXCHOL_CUDA_CHECK(cudaMemcpy(
                inv_scale2.data(), factor.data_.inv_scale2, rows * sizeof(double),
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
            data_.cleanup_needed = true;
            data_.m = factor.data_.m;
            data_.nnz = factor.data_.nnz;
            data_.stats = factor.data_.stats;
            data_.fp16 = factor_fp16;
            data_.fp16_flushed = factor.data_.fp16_flushed;
            data_.fp16_subnormal = factor.data_.fp16_subnormal;
            data_.adopted_device_factor = true;
            data_.adopted_cuda_device = current_device;
            data_.adoption_host_download_bytes = adoption_download_bytes;
            data_.factor_bytes = adopted_bytes;

            APXCHOL_CUDA_CHECK(cudaMalloc(&data_.d_x, rows * sizeof(cuda_value_t)));
            APXCHOL_CUDA_CHECK(cudaMalloc(&data_.d_y, rows * sizeof(cuda_value_t)));
            h_stage_.resize(rows);
            mark("allocate solve vectors");
            if (factor.data_.finalized_sorted_triangular)
                setup_device_kernel_backend(factor);
            else
                setup_kernel_backend(LT_ptr, L_ptr);
            mark("kernel backend tables");

            size_t free_after = 0;
            APXCHOL_CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
            data_.device_delta_bytes =
                free_before >= free_after ? free_before - free_after : 0;

            // Final no-throw ownership commit. Until here `factor` remained the
            // sole owner of every input allocation.
            data_.d_L_rowptr = factor.data_.L_row_ptr;
            data_.d_L_colidx = factor.data_.L_col_idx;
            data_.d_rowPtr = factor.data_.LT_row_ptr;
            data_.d_colIdx = factor.data_.LT_col_idx;
            if (data_.fp16) {
                data_.d_L_vals16 = static_cast<__half*>(factor.data_.L_values);
                data_.d_vals16 = static_cast<__half*>(factor.data_.LT_values);
                data_.d_diag = factor.data_.diag;
                data_.d_inv_scale2 = factor.data_.inv_scale2;
            } else {
                data_.d_L_vals = static_cast<cuda_value_t*>(factor.data_.L_values);
                data_.d_vals = static_cast<cuda_value_t*>(factor.data_.LT_values);
            }
            factor.clear();
            data_.ready = true;

            if (std::getenv("APXCHOL_VERBOSE") ||
                std::getenv("APXCHOL_GPU_MEM_DEBUG"))
                std::fprintf(stderr,
                    "[apxchol] sptrsv storage GPU/dataflow adopted %s: "
                    "stored_nnz=%llu factor_device_bytes=%.1f MB "
                    "host_structure_download=%.1f MB install_delta=%.1f MB\n",
                    data_.fp16 ? "fp16" : value_name,
                    static_cast<unsigned long long>(data_.stats.nnz_stored),
                    data_.factor_bytes / 1e6,
                    data_.adoption_host_download_bytes / 1e6,
                    data_.device_delta_bytes / 1e6);
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
        for (int64_t i = 0; i < data_.m; ++i) h_stage_[i] = static_cast<cuda_value_t>(x_in[i]);
        APXCHOL_CUDA_CHECK(cudaMemcpy(data_.d_x, h_stage_.data(), data_.m * sizeof(cuda_value_t), cudaMemcpyHostToDevice));
        solve_LLt_dev_impl();
        APXCHOL_CUDA_CHECK(cudaMemcpy(h_stage_.data(), data_.d_x, data_.m * sizeof(cuda_value_t), cudaMemcpyDeviceToHost));
        for (int64_t i = 0; i < data_.m; ++i) x_out[i] = static_cast<double>(h_stage_[i]);
    }

    /// GPU-resident variant: input/output are DEVICE pointers. Avoids
    /// the two host↔device transfers per call. Used by GPU-resident PCG
    /// loops so vectors stay on device between iterations. Narrowing input and
    /// widening output are fused into the sweeps; dependencies remain FP32.
    void solve_LLt_dev(const double* d_in, double* d_out, const unsigned* active = nullptr) const {
        solve_LLt_dev_impl(active, d_in, d_out);
    }

    bool ready() const { return data_.ready; }
    /// What the compacting drop did at the last setup() (factor_drop.h): rel /
    /// nnz_factor / nnz_stored / dropped*.
    const factor_drop_stats& drop_stats() const { return data_.stats; }
    /// nnz the device CSR(s) hold after the last setup() (after the drop).
    std::uint64_t stored_nnz() const { return data_.stats.nnz_stored; }
    /// Backend name for benchmark metadata.
    const char* backend_name() const { return "dataflow"; }
    /// Blocks of the persistent grid chosen at setup.
    int dataflow_grid() const { return data_.df_grid; }
    bool fp16() const { return data_.fp16; }
    /// Bytes per stored off-diagonal value on the device (2 under fp16).
    std::size_t value_bytes_effective() const { return data_.fp16 ? 2 : value_bytes; }
    /// Device bytes of the factor arrays cudaMalloc'd by the last setup()
    /// (structure + values of every CSR copy, fp16's diag / inv_scale^2 and the
    /// dataflow batch tables; NOT the solve vectors) and the cudaMemGetInfo
    /// free-memory delta across the whole setup, at allocation granularity.
    std::size_t factor_device_bytes() const { return data_.factor_bytes; }
    std::size_t device_bytes_delta() const { return data_.device_delta_bytes; }
    /// True for device-factor adoption; false for host preprocessing/upload.
    bool adopted_device_factor() const { return data_.adopted_device_factor; }
    /// Exact D2H bytes: fixed-size statistics for trusted finalized factors; complete
    /// int32 structures and fp16 metadata for externally supplied capsules.
    std::size_t adoption_host_download_bytes() const {
        return data_.adoption_host_download_bytes;
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
        if (attributes.type != cudaMemoryTypeDevice ||
            attributes.device != expected_device)
            throw std::invalid_argument(
                std::string("apxchol cuda_sptrsv adoption: ") + name +
                " is not device memory on the recorded CUDA device");
    }

    template<class Device, class Host>
    void upload_factor(Device*& device, const Host* host, std::size_t count) {
        static_assert(sizeof(Device) == sizeof(Host));
        const auto bytes = count * sizeof(Device);
        APXCHOL_CUDA_CHECK(cudaMalloc(&device, bytes));
        data_.factor_bytes += bytes;
        APXCHOL_CUDA_CHECK(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
    }

    // The dataflow backend's host-side tables (cuda_dataflow.h): no levels at
    // all -- the warp batch tables of both directions (from the CSR row
    // lengths), the m tagged words, the control ints, the resident grid.
    void setup_kernel_backend(std::span<const int> LT_ptr, std::span<const int> L_ptr) {
        const int mi = static_cast<int>(data_.m);
        const int pre = dataflow_prefetch_depth();
        // Row segmentation: each direction gets its own threshold, from its
        // own row-length histogram (the two differ a lot -- an elimination
        // factor's hub rows land in CSR of L, its columns stay bounded).
        const auto st_L = cuda_host::dataflow_row_stats(L_ptr, pre);
        const auto st_LT = cuda_host::dataflow_row_stats(LT_ptr, pre);
        const unsigned split_fwd = dataflow_split_threshold_for(st_L);
        const unsigned split_bck = dataflow_split_threshold_for(st_LT);
        namespace df = detail::gpu_dataflow;
        const std::array blocks{0u, unsigned(mi)};
        const auto fwd = df::pack(L_ptr.data(), {unsigned(mi), false}, blocks, split_fwd);
        const auto bck = df::pack(LT_ptr.data(), {unsigned(mi), true}, blocks, split_bck);
        data_.fwd_batches = fwd.words.size() / df::lanes;
        data_.bck_batches = bck.words.size() / df::lanes;
        data_.df_slots = std::max(fwd.slots, bck.slots);
        data_.df_split_rows = fwd.segments.size() + bck.segments.size() - fwd.slots - bck.slots;
        auto upload = [&](auto*& device, const auto& host) {
            if (!host.empty()) upload_factor(device, host.data(), host.size());
        };
        upload(data_.d_fwd_words, fwd.words);
        upload(data_.d_bck_words, bck.words);
        upload(data_.d_fwd_segments, fwd.segments);
        upload(data_.d_bck_segments, bck.segments);
        if (std::getenv("APXCHOL_GPU_SPTRSV_STATS")) {
            dataflow_print_direction_stats(st_L, split_fwd, false, data_.fwd_batches, fwd.slots, fwd.segments.size() - fwd.slots);
            dataflow_print_direction_stats(st_LT, split_bck, true, data_.bck_batches, bck.slots, bck.segments.size() - bck.slots);
        }
        initialize_dataflow_state();
    }

    // Only the private finalizer supplies sorted triangular CSR ownership.
    // Generic capsules must retain full structure validation in the host route.
    void setup_device_kernel_backend(const cuda_sptrsv_device_factor& factor) {
        const char* e = std::getenv("APXCHOL_GPU_DF_SPLIT");
        const int split_min = e ? std::max(0, std::atoi(e)) : -1;
        for (int direction = 0; direction < 2; ++direction) {
            const bool reverse = direction != 0;
            const auto plan = detail::build_dataflow_device_plan(
                static_cast<int>(data_.m), reverse, static_cast<int>(data_.nnz),
                reverse ? factor.data_.LT_row_ptr : factor.data_.L_row_ptr, split_min,
                !reverse && data_.fp16 ? factor.data_.diag : nullptr,
                !reverse && data_.fp16 ? factor.data_.inv_scale2 : nullptr);
            // Consume allocations immediately, before any operation can throw.
            // The outer setup catch then owns every partial plan via destroy().
            (reverse ? data_.d_bck_words : data_.d_fwd_words) = plan.words;
            (reverse ? data_.d_bck_segments : data_.d_fwd_segments) = plan.segments;
            (reverse ? data_.bck_batches : data_.fwd_batches) = plan.n_batches;
            data_.df_slots = std::max(data_.df_slots, plan.n_slots);
            data_.df_split_rows += plan.n_split;
            data_.factor_bytes += plan.bytes;
            data_.adoption_host_download_bytes += plan.host_download_bytes;
            if (std::getenv("APXCHOL_GPU_SPTRSV_STATS")) {
                cuda_host::dataflow_len_stats st;
                st.m = static_cast<int>(data_.m); st.nnz = data_.nnz;
                st.base = 32 * dataflow_prefetch_depth();
                st.max_len = plan.max_len;
                st.mean_len = static_cast<double>(data_.nnz) / static_cast<double>(data_.m);
                for (int k = 0; k < cuda_host::dataflow_len_stats::kBuckets; ++k) {
                    st.rows_ge[k] = plan.rows_ge[k]; st.nnz_ge[k] = plan.nnz_ge[k];
                }
                dataflow_print_direction_stats(st, dataflow_split_threshold_for(st),
                    reverse, plan.n_batches, plan.n_slots, plan.n_split);
            }
        }
        initialize_dataflow_state();
    }

    void initialize_dataflow_state() {
        const std::size_t tag_words = static_cast<std::size_t>(data_.m) + static_cast<std::size_t>(data_.df_slots);
        APXCHOL_CUDA_CHECK(cudaMalloc(&data_.d_df_tag, tag_words * sizeof(unsigned)));
        APXCHOL_CUDA_CHECK(cudaMalloc(&data_.d_df_ticket, sizeof(unsigned)));
        data_.factor_bytes += tag_words * sizeof(unsigned) + sizeof(unsigned);
        data_.df_grid = dataflow_grid_size(data_.fp16);

        if (std::getenv("APXCHOL_GPU_MEM_DEBUG")) { size_t mf=0, mt=0; cudaMemGetInfo(&mf,&mt);
            fprintf(stderr,"[mem] SpTRSV dataflow: 2x factor (CSR L+L^T)=%.2fGB (%zuB/value) "
                    "fwd_batches=%d bck_batches=%d grid=%dx%d tag=%.1fMB slots=%d split_rows=%d | GPU free=%.2f/%.2f GB\n",
                    2.0*data_.nnz*(4.0+(double)value_bytes_effective())/1e9, value_bytes_effective(),
                    data_.fwd_batches, data_.bck_batches, data_.df_grid, kDataflowBlock, tag_words * 4.0 / 1e6,
                    data_.df_slots, data_.df_split_rows, mf/1e9, mt/1e9); }
    }

    static unsigned dataflow_split_threshold_for(const cuda_host::dataflow_len_stats& stats) {
        const char* value = std::getenv("APXCHOL_GPU_DF_SPLIT");
        return value ? std::max(0, std::atoi(value)) : cuda_host::dataflow_split_threshold(stats);
    }

    static void dataflow_print_direction_stats(const cuda_host::dataflow_len_stats& st,
            unsigned split_min, bool reverse,
            int batches, int slots, int splits) {
        std::fprintf(stderr, "[df-stats] %s: m=%d nnz=%lld mean_len=%.2f max_len=%d C=%d\n",
                     reverse ? "back (CSR L^T)" : "fwd (CSR L)", st.m,
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
        std::fprintf(stderr, "[df-stats]   long-row nnz share %.3f -> split_min=%u"
                             " -> batches=%d split_rows=%d slots=%d spec=%d (tables %.2f MB)\n",
                     st.nnz ? static_cast<double>(st.nnz_ge[0]) / static_cast<double>(st.nnz) : 0.0,
                     split_min, batches, splits, slots, slots + splits,
                     (static_cast<double>(batches) * 32 * sizeof(unsigned) +
                      (static_cast<double>(slots) + splits) * sizeof(int4)) / 1e6);
    }



    void solve_LLt_dev_impl(const unsigned* active = nullptr,
                            const double* rhs64 = nullptr, double* out64 = nullptr) const {
        // Initialize roots, then solve each direction with the shared dataflow worker.
        const int m = static_cast<int>(data_.m);
        const unsigned tag_count = unsigned(data_.m) + unsigned(data_.df_slots);
        if (data_.fp16) {
            dataflow_solve_fp16(0, m, false, data_.d_L_rowptr, data_.d_L_colidx, data_.d_L_vals16, data_.d_diag, nullptr,
                                data_.d_fwd_words, data_.d_fwd_segments, data_.fwd_batches,
                                data_.d_df_tag, tag_count, data_.d_df_ticket, data_.df_grid, data_.d_x, data_.d_y, active, rhs64, nullptr);
            dataflow_solve_fp16(0, m, true, data_.d_rowPtr, data_.d_colIdx, data_.d_vals16, data_.d_diag, data_.d_inv_scale2,
                                data_.d_bck_words, data_.d_bck_segments, data_.bck_batches,
                                data_.d_df_tag, tag_count, data_.d_df_ticket, data_.df_grid, data_.d_y, data_.d_x, active, nullptr, out64);
        } else {
            dataflow_solve(0, m, false, data_.d_L_rowptr, data_.d_L_colidx, data_.d_L_vals,
                           data_.d_fwd_words, data_.d_fwd_segments, data_.fwd_batches,
                           data_.d_df_tag, tag_count, data_.d_df_ticket, data_.df_grid, data_.d_x, data_.d_y, active, rhs64, nullptr);
            dataflow_solve(0, m, true, data_.d_rowPtr, data_.d_colIdx, data_.d_vals,
                           data_.d_bck_words, data_.d_bck_segments, data_.bck_batches,
                           data_.d_df_tag, tag_count, data_.d_df_ticket, data_.df_grid, data_.d_y, data_.d_x, active, nullptr, out64);
        }
    }

    void destroy() noexcept {
        // Pristine and pre-allocation lifetimes must not initialize CUDA.
        if (!data_.cleanup_needed) return;
        const detail::cuda_device_scope device(
            data_.adopted_cuda_device, data_.adopted_device_factor && data_.adopted_cuda_device >= 0);
        // Include partial setup allocations in their existing release order.
        void* const allocations[] = {
            data_.d_L_rowptr, data_.d_L_colidx, data_.d_L_vals, data_.d_L_vals16,
            data_.d_vals16, data_.d_diag, data_.d_inv_scale2,
            data_.d_df_tag, data_.d_df_ticket, data_.d_fwd_words, data_.d_bck_words,
            data_.d_fwd_segments, data_.d_bck_segments,
            data_.d_rowPtr, data_.d_colIdx, data_.d_vals, data_.d_x, data_.d_y};
        for (void* allocation : allocations) (void)cudaFree(allocation);
        data_ = {};
    }

    struct state {
        int64_t m = 0;
        int64_t nnz = 0;
        bool ready = false;
        bool cleanup_needed = false;

        // What the compacting drop did (factor_drop.h) and the device footprint.
        factor_drop_stats stats;
        std::size_t factor_bytes = 0;         // cudaMalloc'd factor arrays (see factor_device_bytes)
        std::size_t device_delta_bytes = 0;   // cudaMemGetInfo free delta over setup
        std::uint64_t fp16_flushed = 0, fp16_subnormal = 0;
        bool adopted_device_factor = false;
        int adopted_cuda_device = -1;
        std::size_t adoption_host_download_bytes = 0;

        // Device memory (values + solve vectors at cuda_value_t width).
        int*          d_rowPtr = nullptr;
        int*          d_colIdx = nullptr;
        cuda_value_t* d_vals   = nullptr;
        cuda_value_t* d_x      = nullptr;
        cuda_value_t* d_y      = nullptr;

        // Our dataflow backend. The stored d_* arrays are CSR
        // of L^T (back solve); the forward solve needs CSR of L (d_L_*).
        int*          d_L_rowptr     = nullptr;
        int*          d_L_colidx     = nullptr;
        cuda_value_t* d_L_vals       = nullptr;
        // Both directions share the reset-on-sweep value tags and ticket counter.
        unsigned* d_fwd_words = nullptr;
        unsigned* d_bck_words = nullptr;
        detail::gpu_dataflow::segment* d_fwd_segments = nullptr;
        detail::gpu_dataflow::segment* d_bck_segments = nullptr;
        int           fwd_batches = 0, bck_batches = 0;
        int           df_slots = 0;                // extra tagged words for the segment partials
        int           df_split_rows = 0;           // rows the plan split (both directions summed)
        unsigned*     d_df_tag  = nullptr;
        unsigned*     d_df_ticket       = nullptr;   // shared ticket counter
        int           df_grid         = 0;
        // fp16 storage (APXCHOL_FACTOR_STORAGE): binary16
        // values of CSR(L~^T) (d_vals16, replaces d_vals) and CSR(L~)
        // (d_L_vals16, replaces d_L_vals), the fp32 scaled diagonal and the
        // back solve's per-row input scale inv_scale^2 -- held in DOUBLE (r_j^2
        // exact; fp32 overflows for tiny scales, see the class comment).
        bool          fp16           = false;
        __half*       d_vals16       = nullptr;
        __half*       d_L_vals16     = nullptr;
        float*        d_diag         = nullptr;
        double*       d_inv_scale2   = nullptr;
    } data_;

    // Host staging for the fp32 host-PCG boundary (narrow/widen).
    mutable std::vector<cuda_value_t> h_stage_;
};

#undef APXCHOL_CUDA_CHECK

} // namespace apxchol
