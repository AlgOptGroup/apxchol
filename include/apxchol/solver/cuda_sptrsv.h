#pragma once
#include <Eigen/Sparse>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdexcept>
#include <string>

namespace apxchol {

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

    cuda_sptrsv(const cuda_sptrsv&) = delete;
    cuda_sptrsv& operator=(const cuda_sptrsv&) = delete;

    // Move not supported (complex GPU state) — use via pointer or unique_ptr.
    cuda_sptrsv(cuda_sptrsv&&) = delete;
    cuda_sptrsv& operator=(cuda_sptrsv&&) = delete;

    ~cuda_sptrsv() { destroy(); }

    /// Setup: copy L11 to device, create cuSPARSE descriptors, run analysis.
    void setup(const Eigen::SparseMatrix<double>& L, Eigen::Index m) {
        destroy();
        m_ = m;

        // Materialize L11 and get its CSC arrays.
        Eigen::SparseMatrix<double> L11 = L.topLeftCorner(m, m);
        L11.makeCompressed();

        nnz_ = L11.nonZeros();
        // CSC arrays of L11:  outerIndexPtr = column pointers, innerIndexPtr = row indices.
        // Reinterpreted as CSR of L11^T: outerIndexPtr = row pointers, innerIndexPtr = col indices.
        const int* h_rowPtr = L11.outerIndexPtr();        // m+1 ints  (CSC col ptrs → CSR row ptrs of L^T)
        const int* h_colIdx = L11.innerIndexPtr();        // nnz ints  (CSC row idxs → CSR col idxs of L^T)
        const double* h_vals = L11.valuePtr();            // nnz doubles

        // Allocate and copy matrix to device.
        APXCHOL_CUDA_CHECK(cudaMalloc(&d_rowPtr_, (m_ + 1) * sizeof(int)));
        APXCHOL_CUDA_CHECK(cudaMalloc(&d_colIdx_, nnz_ * sizeof(int)));
        APXCHOL_CUDA_CHECK(cudaMalloc(&d_vals_,   nnz_ * sizeof(double)));

        APXCHOL_CUDA_CHECK(cudaMemcpy(d_rowPtr_, h_rowPtr, (m_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
        APXCHOL_CUDA_CHECK(cudaMemcpy(d_colIdx_, h_colIdx, nnz_ * sizeof(int),     cudaMemcpyHostToDevice));
        APXCHOL_CUDA_CHECK(cudaMemcpy(d_vals_,   h_vals,   nnz_ * sizeof(double),  cudaMemcpyHostToDevice));

        // Allocate device vectors (ping-pong: d_x and d_y).
        APXCHOL_CUDA_CHECK(cudaMalloc(&d_x_, m_ * sizeof(double)));
        APXCHOL_CUDA_CHECK(cudaMalloc(&d_y_, m_ * sizeof(double)));

        // Create cuSPARSE handle.
        APXCHOL_CUSPARSE_CHECK(cusparseCreate(&cusparse_));

        // Create L^T in CSR format (upper triangular, non-unit diagonal).
        APXCHOL_CUSPARSE_CHECK(cusparseCreateCsr(
            &matLT_, m_, m_, nnz_,
            d_rowPtr_, d_colIdx_, d_vals_,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

        cusparseFillMode_t fill = CUSPARSE_FILL_MODE_UPPER;
        cusparseDiagType_t diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
        APXCHOL_CUSPARSE_CHECK(cusparseSpMatSetAttribute(matLT_, CUSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill)));
        APXCHOL_CUSPARSE_CHECK(cusparseSpMatSetAttribute(matLT_, CUSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag)));

        // Create dense vector descriptors.
        APXCHOL_CUSPARSE_CHECK(cusparseCreateDnVec(&vecX_, m_, d_x_, CUDA_R_64F));
        APXCHOL_CUSPARSE_CHECK(cusparseCreateDnVec(&vecY_, m_, d_y_, CUDA_R_64F));

        double alpha = 1.0;

        // Forward analysis:  L * y = x  ↔  TRANSPOSE on L^T.  vecX → vecY.
        APXCHOL_CUSPARSE_CHECK(cusparseSpSV_createDescr(&descrFwd_));
        size_t bufSizeFwd = 0;
        APXCHOL_CUSPARSE_CHECK(cusparseSpSV_bufferSize(
            cusparse_, CUSPARSE_OPERATION_TRANSPOSE, &alpha,
            matLT_, vecX_, vecY_, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, descrFwd_, &bufSizeFwd));
        APXCHOL_CUDA_CHECK(cudaMalloc(&bufFwd_, bufSizeFwd));
        APXCHOL_CUSPARSE_CHECK(cusparseSpSV_analysis(
            cusparse_, CUSPARSE_OPERATION_TRANSPOSE, &alpha,
            matLT_, vecX_, vecY_, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, descrFwd_, bufFwd_));

        // Back analysis:  L^T * z = y  ↔  NON_TRANSPOSE on L^T.  vecY → vecX.
        APXCHOL_CUSPARSE_CHECK(cusparseSpSV_createDescr(&descrBck_));
        size_t bufSizeBck = 0;
        APXCHOL_CUSPARSE_CHECK(cusparseSpSV_bufferSize(
            cusparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            matLT_, vecY_, vecX_, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, descrBck_, &bufSizeBck));
        APXCHOL_CUDA_CHECK(cudaMalloc(&bufBck_, bufSizeBck));
        APXCHOL_CUSPARSE_CHECK(cusparseSpSV_analysis(
            cusparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            matLT_, vecY_, vecX_, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, descrBck_, bufBck_));

        ready_ = true;
    }

    /// Combined forward + back solve on GPU: computes L^{-T} L^{-1} x.
    /// Reads x_in[0..m-1] from host, writes result to x_out[0..m-1] on host.
    /// x_in and x_out may alias (ping-pong stays on GPU).
    void solve_LLt(const double* x_in, double* x_out) const {
        double alpha = 1.0;

        // H2D: copy input to d_x.
        APXCHOL_CUDA_CHECK(cudaMemcpy(d_x_, x_in, m_ * sizeof(double), cudaMemcpyHostToDevice));

        // Forward:  L * d_y = d_x  ↔  TRANSPOSE on L^T.
        APXCHOL_CUSPARSE_CHECK(cusparseSpSV_solve(
            cusparse_, CUSPARSE_OPERATION_TRANSPOSE, &alpha,
            matLT_, vecX_, vecY_, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, descrFwd_));

        // Back:  L^T * d_x = d_y  ↔  NON_TRANSPOSE on L^T.
        APXCHOL_CUSPARSE_CHECK(cusparseSpSV_solve(
            cusparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            matLT_, vecY_, vecX_, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT, descrBck_));

        // D2H: copy result from d_x.
        APXCHOL_CUDA_CHECK(cudaMemcpy(x_out, d_x_, m_ * sizeof(double), cudaMemcpyDeviceToHost));
    }

    bool ready() const { return ready_; }

private:
    void destroy() {
        if (!ready_) return;
        cusparseSpSV_destroyDescr(descrFwd_);
        cusparseSpSV_destroyDescr(descrBck_);
        cusparseDestroyDnVec(vecX_);
        cusparseDestroyDnVec(vecY_);
        cusparseDestroySpMat(matLT_);
        cusparseDestroy(cusparse_);
        cudaFree(d_rowPtr_);
        cudaFree(d_colIdx_);
        cudaFree(d_vals_);
        cudaFree(d_x_);
        cudaFree(d_y_);
        cudaFree(bufFwd_);
        cudaFree(bufBck_);
        ready_ = false;
    }

    int64_t m_ = 0;
    int64_t nnz_ = 0;
    bool ready_ = false;

    // Device memory.
    int*    d_rowPtr_ = nullptr;
    int*    d_colIdx_ = nullptr;
    double* d_vals_   = nullptr;
    double* d_x_      = nullptr;
    double* d_y_      = nullptr;

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
