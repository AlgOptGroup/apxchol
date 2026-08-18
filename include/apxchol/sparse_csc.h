#pragma once
#include "apxchol/types.h"
#include "apxchol/bf16.h"
#include <vector>
#include <cstddef>
#include <cassert>
#include <limits>

namespace apxchol {

// Value type for the SpTRSV's off-diagonal factor entries, shared by BOTH the
// OpenMP (omp.h) and CUDA (cuda.h) backends so the build flag means the same
// thing on CPU and GPU. FP32 (-DAPXCHOL_SPTRSV_FP32, ON by default; pass =OFF
// for an fp64 baseline) halves the two largest factor copies (csr/csc on CPU,
// d_vals on GPU), cutting memory and the bandwidth on the bandwidth-bound
// triangular solve. The outer PCG (vectors, dots, SpMV/residual) stays fp64,
// so it still converges to 1e-8: the preconditioner is approximate, fp32 only
// costs a few extra PCG iters ("changes iters, not the residual floor").
//
// BF16 (-DAPXCHOL_SPTRSV_BF16, OFF by default; CPU/omp backend only) goes one
// step further: 16-bit storage (bf16_t, see bf16.h) for the SpTRSV's own
// CSR/CSC value arrays -- 2 B/nnz of value stream instead of 4, i.e. the
// 8 B/nnz (idx + val) factor stream drops to 6 B/nnz. Compute is unchanged:
// every read widens bf16 -> fp32 -> double in registers via widen(); the
// rounding happens once, when omp_sptrsv::setup copies the factor into its
// CSR/CSC (RNE on the fp32 pattern by default; APXCHOL_BF16_STOCHASTIC=1
// selects unbiased stochastic rounding). ONLY the off-diagonals are bf16: the
// diagonal is kept exact-fp32 in a separate omp_sptrsv::diag_ array (rounding
// the diagonal to 8 bits was measured to be the dominant iteration-count
// damage). Precision is 2^-8 relative per off-diagonal entry, so this is a
// preconditioner-quality knob: PCG still converges to tol, possibly in more
// iterations. It takes precedence over APXCHOL_SPTRSV_FP32 when both are
// defined.
//
// Two value types, deliberately:
//   - factor_value_t : what the FACTOR (sparse_csc::vals_, the assembler's
//     output) stores. fp32 under FP32 *and* BF16 (the bf16 build needs the
//     exact fp32 diagonal at SpTRSV setup, and its factor is released right
//     after setup anyway), fp64 otherwise.
//   - sptrsv_value_t : what the SpTRSV kernels' CSR/CSC value arrays store
//     (and the GPU backend's d_vals). bf16 under BF16, else == factor_value_t.
// They coincide on the fp32/fp64 builds, so those are unchanged byte-for-byte.
#if defined(APXCHOL_SPTRSV_BF16)
using factor_value_t = float;
using sptrsv_value_t = bf16_t;
#elif defined(APXCHOL_SPTRSV_FP32)
using factor_value_t = float;
using sptrsv_value_t = float;
#else
using factor_value_t = double;
using sptrsv_value_t = double;
#endif

/// Owning Compressed-Sparse-Column matrix that replaces Eigen::SparseMatrix for
/// the factor L. We own the storage so the index width is OURS, not forced
/// signed-`int` by Eigen's StorageIndex static_assert.
///
/// Two index roles, deliberately different (see types.h) -- BOTH unsigned:
///   - column pointers (`outer_`)  -> edge_index : a cumulative NON-NEGATIVE
///     offset that can reach billions, so it is the widenable one.
///   - row indices     (`inner_`)  -> node_index : a vertex id, < 2^31 forever,
///     so it stays 32-bit even under APXCHOL_64BIT_INDICES.
///
/// Overflow safety: the only place a value can exceed its type is the cumulative
/// column pointer; `set_outer`/`finish` assert it stays <= edge_index max. We
/// never form `(a+b)/2` or `a-b` on offsets where the result could be negative.
///
/// The accessor names mirror the subset of the Eigen::SparseMatrix interface the
/// factor consumers (sptrsv, cuda pcg, assembler) actually use, so their bodies
/// stay intact -- only the pointer *types* change (int* -> edge_index*/node_index*).
struct sparse_csc {
    static constexpr edge_index kEdgeMax = std::numeric_limits<edge_index>::max();

    node_index                  n_ = 0; // rows == cols (square factor)
    std::vector<edge_index>     outer_; // n_+1 column pointers (cumulative nnz)
    std::vector<node_index>     inner_; // row indices
    // Off-diagonal AND diagonal factor values, fp32 under -DAPXCHOL_SPTRSV_FP32
    // and -DAPXCHOL_SPTRSV_BF16 (saves ~nnz*4 B at the assembly peak; the bf16
    // narrowing of the off-diagonals happens later, in omp_sptrsv::setup, so
    // the diagonal reaches the SpTRSV at full fp32). Reads should widen(). The
    // diagonal can ride along in fp32: it's L(i,i) = sqrt(weighted degree), a
    // benign well-scaled positive scalar, and fp32 diag converges to 1e-8 just
    // like fp64 (verified on IPM/social/grid; the GPU backend has always stored
    // the diagonal in fp32 too). The fp64 build leaves vals_ as double ->
    // byte-identical.
    std::vector<factor_value_t> vals_;

    sparse_csc() = default;

    // Eigen-compatible surface ---------------------------------------------
    node_index rows() const { return n_; }
    node_index cols() const { return n_; }
    node_index outerSize() const { return n_; }
    edge_index nonZeros() const { return outer_.empty() ? edge_index(0) : outer_.back(); }

    edge_index*       outerIndexPtr()       { return outer_.data(); }
    const edge_index* outerIndexPtr() const { return outer_.data(); }
    node_index*       innerIndexPtr()       { return inner_.data(); }
    const node_index* innerIndexPtr() const { return inner_.data(); }
    factor_value_t*       valuePtr()            { return vals_.data(); }
    const factor_value_t* valuePtr()      const { return vals_.data(); }

    // Build surface (mirrors the resize/reserve/resizeNonZeros/makeCompressed
    // sequence assemble_csc used on the Eigen matrix) ----------------------
    void resize(node_index rows, node_index /*cols*/) {
        n_ = rows;
        outer_.assign(static_cast<std::size_t>(rows) + 1, edge_index(0));
    }
    void reserve(edge_index /*nnz*/) {}             // outer_ already sized; no-op
    void resizeNonZeros(edge_index nnz) {
        inner_.resize(static_cast<std::size_t>(nnz));
        vals_.resize(static_cast<std::size_t>(nnz));
    }
    void makeCompressed() {}                        // compressed by construction

    /// Free the large row-index + value arrays, keeping the column pointers so
    /// nonZeros()/rows()/cols() still work. Called once the factor has been
    /// consumed by the SpTRSV setup (which copies it into its own CSR/CSC): the
    /// PCG loop never touches F_.L afterwards, so this drops one full copy of the
    /// factor (~nnz*12 B) for the entire solve. swap-with-empty actually frees.
    void release_values() {
        std::vector<node_index>().swap(inner_);
        std::vector<factor_value_t>().swap(vals_);
    }

    /// Assert the just-built cumulative column pointers never overflowed
    /// edge_index. Call once after the prefix sum (assemble_csc).
    void assert_no_offset_overflow() const {
        // outer_ is monotone non-decreasing iff no wrap occurred; a wrapped
        // (unsigned) offset would appear SMALLER than its predecessor.
        for (std::size_t i = 1; i < outer_.size(); ++i)
            assert(outer_[i] >= outer_[i - 1] &&
                   "edge_index overflow: factor nnz exceeds the index type "
                   "(rebuild with APXCHOL_64BIT_INDICES=ON)");
    }
};

} // namespace apxchol
