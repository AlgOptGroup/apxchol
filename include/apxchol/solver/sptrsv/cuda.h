#pragma once
#include "apxchol/sparse_csc.h"
#include "apxchol/solver/sptrsv/cuda_cast.h"
#include "apxchol/solver/sptrsv/cuda_levelset.h"
#include <cuda_runtime.h>
#include <cusparse.h>
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

/// RAII wrapper for cuSPARSE SpSV (sparse triangular solve on GPU).
/// Copies L11 to device once, runs analysis once, then provides
/// fast GPU-parallel forward/back solves with per-call vector transfers.
///
/// cuSPARSE SpSV requires CSR (not CSC).  Eigen stores column-major (CSC).
/// We exploit the identity  CSC(L) == CSR(L^T):  the same three arrays
/// (outerIndexPtr, innerIndexPtr, valuePtr) describe L in CSC or L^T in CSR.
/// We create L^T (upper triangular) in CSR, then:
///   - Forward  solve  L  * y = x  ↔  TRANSPOSE    on L^T
///   - Back     solve  L^T* z = y  ↔  NON_TRANSPOSE on L^T
class cuda_sptrsv {
public:
    cuda_sptrsv() = default;

    // Compiled width of the on-device factor values (mirrors omp_sptrsv): 4 ==
    // -DAPXCHOL_SPTRSV_FP32, 8 == fp64. Printed at startup so the GPU build flag
    // is observable at runtime, same as the CPU backend.
    static constexpr std::size_t value_bytes = sizeof(cuda_value_t);
    static constexpr const char* value_name =
        sizeof(cuda_value_t) == 4 ? "float (fp32)" : "double (fp64)";

    cuda_sptrsv(const cuda_sptrsv&) = delete;
    cuda_sptrsv& operator=(const cuda_sptrsv&) = delete;

    // Move not supported (complex GPU state) — use via pointer or unique_ptr.
    cuda_sptrsv(cuda_sptrsv&&) = delete;
    cuda_sptrsv& operator=(cuda_sptrsv&&) = delete;

    ~cuda_sptrsv() { destroy(); }

    /// Setup: copy L11 to device, create cuSPARSE descriptors, run analysis.
    void setup(const sparse_csc& L, node_index m) {
        destroy();
        m_ = static_cast<int64_t>(m);

        // Build L11 = top-left m×m block of the factor in CSC, as int arrays
        // (cuSPARSE CUSPARSE_INDEX_32I). The factor on GPU-tested matrices fits
        // int32; a factor exceeding it cannot use cuSPARSE's 32-bit index API.
        std::vector<int>          h_rowPtr_v;   // m+1 (CSC col ptrs → CSR row ptrs of L^T)
        std::vector<int>          h_colIdx_v;   // nnz (CSC row idxs → CSR col idxs of L^T)
        std::vector<cuda_value_t> h_vals_v;     // nnz, narrowed to fp32 under the flag
        build_L11_csc_int(L, m_, h_rowPtr_v, h_colIdx_v, h_vals_v);

        nnz_ = static_cast<int64_t>(h_colIdx_v.size());
        const int* h_rowPtr = h_rowPtr_v.data();
        const int* h_colIdx = h_colIdx_v.data();
        const cuda_value_t* h_vals = h_vals_v.data();

        // Allocate and copy matrix to device (values at cuda_value_t width).
        APXCHOL_CUDA_CHECK(cudaMalloc(&d_rowPtr_, (m_ + 1) * sizeof(int)));
        APXCHOL_CUDA_CHECK(cudaMalloc(&d_colIdx_, nnz_ * sizeof(int)));
        APXCHOL_CUDA_CHECK(cudaMalloc(&d_vals_,   nnz_ * sizeof(cuda_value_t)));

        APXCHOL_CUDA_CHECK(cudaMemcpy(d_rowPtr_, h_rowPtr, (m_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
        APXCHOL_CUDA_CHECK(cudaMemcpy(d_colIdx_, h_colIdx, nnz_ * sizeof(int),     cudaMemcpyHostToDevice));
        APXCHOL_CUDA_CHECK(cudaMemcpy(d_vals_,   h_vals,   nnz_ * sizeof(cuda_value_t), cudaMemcpyHostToDevice));

        // Allocate device vectors (solve runs at cuda_value_t width).
        APXCHOL_CUDA_CHECK(cudaMalloc(&d_x_, m_ * sizeof(cuda_value_t)));
        APXCHOL_CUDA_CHECK(cudaMalloc(&d_y_, m_ * sizeof(cuda_value_t)));
        h_stage_.resize(static_cast<size_t>(m_));

        // Backend select: APXCHOL_GPU_SPTRSV = levelset (our O(n) vertex level-set -- a
        // memory-frugal alternative to cuSPARSE's O(nnz) analysis buffer, so it fits the
        // giant social factors that OOM cuSPARSE) | cuSPARSE (default). The level-set
        // needs CSR of L (forward); the stored d_* arrays are CSR of L^T.
        const char* be = std::getenv("APXCHOL_GPU_SPTRSV");
        const std::string bes = be ? be : "";
        use_levelset_ = (bes == "levelset");

        if (use_levelset_) {
            // Forward solve L y = x gathers over rows of L -> needs CSR of L. Transpose
            // the stored CSR of L^T (2x factor storage -- still far under cuSPARSE's
            // factor + O(nnz) SpSV scratch).
            std::vector<int> h_Lrp, h_Lci;
            std::vector<cuda_value_t> h_Lv;
            transpose_csr(static_cast<int>(m_), h_rowPtr_v, h_colIdx_v, h_vals_v, h_Lrp, h_Lci, h_Lv);
            APXCHOL_CUDA_CHECK(cudaMalloc(&d_L_rowptr_, (m_ + 1) * sizeof(int)));
            APXCHOL_CUDA_CHECK(cudaMalloc(&d_L_colidx_, nnz_ * sizeof(int)));
            APXCHOL_CUDA_CHECK(cudaMalloc(&d_L_vals_,   nnz_ * sizeof(cuda_value_t)));
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_L_rowptr_, h_Lrp.data(), (m_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_L_colidx_, h_Lci.data(), nnz_ * sizeof(int), cudaMemcpyHostToDevice));
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_L_vals_,   h_Lv.data(),  nnz_ * sizeof(cuda_value_t), cudaMemcpyHostToDevice));

            // Level schedules: fwd from CSR of L (ascending), back from CSR of L^T
            // (descending). Serial O(nnz).
            std::vector<int> fwd_order, bck_order;
            compute_levels(static_cast<int>(m_), h_Lrp,      h_Lci,      true,  fwd_order, fwd_level_ptr_);
            compute_levels(static_cast<int>(m_), h_rowPtr_v, h_colIdx_v, false, bck_order, bck_level_ptr_);
            APXCHOL_CUDA_CHECK(cudaMalloc(&d_fwd_order_, m_ * sizeof(int)));
            APXCHOL_CUDA_CHECK(cudaMalloc(&d_bck_order_, m_ * sizeof(int)));
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_fwd_order_, fwd_order.data(), m_ * sizeof(int), cudaMemcpyHostToDevice));
            APXCHOL_CUDA_CHECK(cudaMemcpy(d_bck_order_, bck_order.data(), m_ * sizeof(int), cudaMemcpyHostToDevice));

            if (std::getenv("APXCHOL_GPU_MEM_DEBUG")) { size_t mf=0, mt=0; cudaMemGetInfo(&mf,&mt);
              fprintf(stderr,"[mem] SpTRSV level-set: 2x factor (CSR L+L^T)=%.2fGB "
                      "fwd_levels=%zu | GPU free=%.2f/%.2f GB\n",
                      2.0*nnz_*(4.0+(double)sizeof(cuda_value_t))/1e9,
                      fwd_level_ptr_.size()-1, mf/1e9, mt/1e9); }
        } else {
            // Create cuSPARSE handle.
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

            // Forward analysis:  L * y = x  ↔  TRANSPOSE on L^T.  vecX → vecY.
            APXCHOL_CUSPARSE_CHECK(cusparseSpSV_createDescr(&descrFwd_));
            size_t bufSizeFwd = 0;
            APXCHOL_CUSPARSE_CHECK(cusparseSpSV_bufferSize(
                cusparse_, CUSPARSE_OPERATION_TRANSPOSE, &alpha,
                matLT_, vecX_, vecY_, cuda_value_dtype,
                CUSPARSE_SPSV_ALG_DEFAULT, descrFwd_, &bufSizeFwd));
            APXCHOL_CUDA_CHECK(cudaMalloc(&bufFwd_, bufSizeFwd));
            APXCHOL_CUSPARSE_CHECK(cusparseSpSV_analysis(
                cusparse_, CUSPARSE_OPERATION_TRANSPOSE, &alpha,
                matLT_, vecX_, vecY_, cuda_value_dtype,
                CUSPARSE_SPSV_ALG_DEFAULT, descrFwd_, bufFwd_));

            // Back analysis:  L^T * z = y  ↔  NON_TRANSPOSE on L^T.  vecY → vecX.
            APXCHOL_CUSPARSE_CHECK(cusparseSpSV_createDescr(&descrBck_));
            size_t bufSizeBck = 0;
            APXCHOL_CUSPARSE_CHECK(cusparseSpSV_bufferSize(
                cusparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                matLT_, vecY_, vecX_, cuda_value_dtype,
                CUSPARSE_SPSV_ALG_DEFAULT, descrBck_, &bufSizeBck));
            APXCHOL_CUDA_CHECK(cudaMalloc(&bufBck_, bufSizeBck));
            APXCHOL_CUSPARSE_CHECK(cusparseSpSV_analysis(
                cusparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                matLT_, vecY_, vecX_, cuda_value_dtype,
                CUSPARSE_SPSV_ALG_DEFAULT, descrBck_, bufBck_));

            if (std::getenv("APXCHOL_GPU_MEM_DEBUG")) { size_t mf=0, mt=0; cudaMemGetInfo(&mf,&mt);
              fprintf(stderr,"[mem] SpTRSV factor: nnz=%lld colidx=%.2fGB vals=%.2fGB (%zuB/elem) "
                      "SpSV bufFwd=%.2fGB bufBck=%.2fGB | GPU free=%.2f / total=%.2f GB after SpTRSV setup\n",
                      (long long)nnz_, nnz_*4.0/1e9, nnz_*(double)sizeof(cuda_value_t)/1e9, sizeof(cuda_value_t),
                      bufSizeFwd/1e9, bufSizeBck/1e9, mf/1e9, mt/1e9); }
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

private:
    // Build the top-left m×m block of the sparse_csc factor L as int CSC arrays
    // (col_ptr[m+1], row_idx[nnz], vals[nnz]). For the Laplacian case (m < n)
    // entries in the dropped last row (n-1) are filtered out. Serial — setup runs
    // once. Offsets are read as edge_index; emitted as int for cuSPARSE.
    static void build_L11_csc_int(const sparse_csc& L, int64_t m,
                                  std::vector<int>& col_ptr,
                                  std::vector<int>& row_idx,
                                  std::vector<cuda_value_t>& vals) {
        const edge_index* Lo = L.outerIndexPtr();
        const node_index* Li = L.innerIndexPtr();
        const auto*       Lv = L.valuePtr();   // sptrsv_value_t* (fp32 under flag)
        const int64_t n = static_cast<int64_t>(L.rows());
        col_ptr.assign(static_cast<size_t>(m) + 1, 0);
        if (m == n) {
            for (int64_t j = 0; j < m; ++j)
                col_ptr[j + 1] = col_ptr[j] +
                    static_cast<int>(Lo[j + 1] - Lo[j]);
            const edge_index nnz = Lo[m];
            row_idx.resize(static_cast<size_t>(nnz));
            vals.resize(static_cast<size_t>(nnz));
            for (edge_index p = 0; p < nnz; ++p) {
                row_idx[p] = static_cast<int>(Li[p]);
                vals[p]    = static_cast<cuda_value_t>(Lv[p]);
            }
        } else {
            const node_index drop = static_cast<node_index>(n - 1);
            for (int64_t j = 0; j < m; ++j) {
                int kept = 0;
                for (edge_index p = Lo[j]; p < Lo[j + 1]; ++p)
                    if (Li[p] != drop) ++kept;
                col_ptr[j + 1] = col_ptr[j] + kept;
            }
            row_idx.resize(static_cast<size_t>(col_ptr[static_cast<size_t>(m)]));
            vals.resize(row_idx.size());
            int out = 0;
            for (int64_t j = 0; j < m; ++j)
                for (edge_index p = Lo[j]; p < Lo[j + 1]; ++p) {
                    if (Li[p] == drop) continue;
                    row_idx[out] = static_cast<int>(Li[p]);
                    vals[out]    = static_cast<cuda_value_t>(Lv[p]);
                    ++out;
                }
        }
    }

    // Transpose a square m×m CSR. The level-set forward solve needs CSR of L (row
    // access), but setup builds CSR of L^T; this produces CSR of L from it. O(nnz).
    static void transpose_csr(int m, const std::vector<int>& irp, const std::vector<int>& ici,
                              const std::vector<cuda_value_t>& iv, std::vector<int>& orp,
                              std::vector<int>& oci, std::vector<cuda_value_t>& ov) {
        const std::size_t nnz = ici.size();
        orp.assign(static_cast<std::size_t>(m) + 1, 0);
        for (std::size_t p = 0; p < nnz; ++p) orp[ici[p] + 1]++;
        for (int i = 0; i < m; ++i) orp[i + 1] += orp[i];
        oci.resize(nnz); ov.resize(nnz);
        std::vector<int> pos(orp.begin(), orp.end());
        for (int i = 0; i < m; ++i)
            for (int p = irp[i]; p < irp[i + 1]; ++p) {
                const int d = pos[ici[p]]++;
                oci[d] = i; ov[d] = iv[p];
            }
    }

    // Topological level partition of a triangular CSR. `ascending` = lower-tri
    // (forward: off-diagonal deps have smaller index, sweep 0..m-1); !ascending =
    // upper-tri (back: deps larger, sweep m-1..0). The diagonal is skipped. Returns
    // row_order (rows bucketed by level) + level_ptr (boundaries, size num_levels+1).
    static void compute_levels(int m, const std::vector<int>& rowptr, const std::vector<int>& colidx,
                               bool ascending, std::vector<int>& row_order, std::vector<int>& level_ptr) {
        std::vector<int> level(static_cast<std::size_t>(m), 0);
        int maxlev = 0;
        auto proc = [&](int i) {
            int lv = 0;
            for (int p = rowptr[i]; p < rowptr[i + 1]; ++p) {
                const int j = colidx[p];
                if (j == i) continue;
                if (level[j] + 1 > lv) lv = level[j] + 1;
            }
            level[i] = lv; if (lv > maxlev) maxlev = lv;
        };
        if (ascending) for (int i = 0; i < m; ++i) proc(i);
        else           for (int i = m - 1; i >= 0; --i) proc(i);
        const int nlev = maxlev + 1;
        level_ptr.assign(static_cast<std::size_t>(nlev) + 1, 0);
        for (int i = 0; i < m; ++i) level_ptr[level[i] + 1]++;
        for (int l = 0; l < nlev; ++l) level_ptr[l + 1] += level_ptr[l];
        row_order.resize(static_cast<std::size_t>(m));
        std::vector<int> pos(level_ptr.begin(), level_ptr.end());
        for (int i = 0; i < m; ++i) row_order[pos[level[i]]++] = i;
    }

    void solve_LLt_dev_impl() const {
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
public:

    bool ready() const { return ready_; }

private:
    void destroy() {
        if (!ready_) return;
        if (use_levelset_) {
            cudaFree(d_L_rowptr_); cudaFree(d_L_colidx_); cudaFree(d_L_vals_);
            cudaFree(d_fwd_order_); cudaFree(d_bck_order_);
            d_L_rowptr_ = d_L_colidx_ = d_fwd_order_ = d_bck_order_ = nullptr;
            d_L_vals_ = nullptr;
            fwd_level_ptr_.clear(); bck_level_ptr_.clear();
        } else {
            cusparseSpSV_destroyDescr(descrFwd_);
            cusparseSpSV_destroyDescr(descrBck_);
            cusparseDestroyDnVec(vecX_);
            cusparseDestroyDnVec(vecY_);
            cusparseDestroySpMat(matLT_);
            cusparseDestroy(cusparse_);
            cudaFree(bufFwd_);
            cudaFree(bufBck_);
            bufFwd_ = bufBck_ = nullptr;
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
    }

    int64_t m_ = 0;
    int64_t nnz_ = 0;
    bool ready_ = false;

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
    int*          d_L_rowptr_     = nullptr;
    int*          d_L_colidx_     = nullptr;
    cuda_value_t* d_L_vals_       = nullptr;
    int*          d_fwd_order_    = nullptr;   // level-set schedules
    int*          d_bck_order_    = nullptr;
    std::vector<int> fwd_level_ptr_;
    std::vector<int> bck_level_ptr_;

    // cuSPARSE state.
    cusparseHandle_t      cusparse_ = nullptr;
    cusparseSpMatDescr_t  matLT_    = nullptr;
    cusparseDnVecDescr_t  vecX_     = nullptr;
    cusparseDnVecDescr_t  vecY_     = nullptr;
    cusparseSpSVDescr_t   descrFwd_ = nullptr;
    cusparseSpSVDescr_t   descrBck_ = nullptr;
    void*                 bufFwd_   = nullptr;
    void*                 bufBck_   = nullptr;
};

#undef APXCHOL_CUDA_CHECK
#undef APXCHOL_CUSPARSE_CHECK

} // namespace apxchol
