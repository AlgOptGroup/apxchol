#pragma once
#include "apxchol/types.h"
#include "apxchol/lowprec.h"   // fp16_t, widen(), the APXCHOL_SPTRSV_FP16 reader
#include <vector>
#include <cstddef>
#include <limits>

namespace apxchol {

// Value types of the factor and of the SpTRSV's own copies of it, shared by
// BOTH the OpenMP (omp.h) and CUDA (cuda.h) backends so they mean the same
// thing on CPU and GPU. Both are fp32, UNCONDITIONALLY: the fp64 baseline
// (CMake APXCHOL_SPTRSV_FP32=OFF) was removed 2026-08-20 -- fp32 halves the
// two largest factor copies (csr/csc on CPU, d_vals on GPU), cutting memory
// and the bandwidth on the bandwidth-bound triangular solve, and it costs
// only a few extra PCG iterations ("changes iters, not the residual floor"):
// the outer PCG (vectors, dots, SpMV/residual) stays fp64 and still converges
// to 1e-8. Nothing in the tree ever shipped or measured against the fp64
// storage; the GPU's dataflow backend never supported it at all.
//
//   - factor_value_t : what the FACTOR (sparse_csc::vals_, the assembler's
//     output) stores. The SpTRSV's setup reads the exact fp32 diagonal and
//     the per-column scales off it, and the factor is released right after
//     setup on the consuming path.
//   - sptrsv_value_t : the DEFAULT width of the SpTRSV kernels' CSR/CSC value
//     arrays (and of the GPU backend's d_vals): fp32. Both backends can narrow
//     those arrays to fp16 at RUNTIME instead (APXCHOL_SPTRSV_FP16=1,
//     lowprec.h) -- that is a per-setup choice of storage type, never a
//     compile-time typedef.
using factor_value_t = float;
using sptrsv_value_t = float;

/// Owning Compressed-Sparse-Column matrix that replaces Eigen::SparseMatrix for
/// the factor L. We own the storage so the index width is OURS, not forced
/// signed-`int` by Eigen's StorageIndex static_assert.
///
/// Two index roles, deliberately different (see types.h) -- BOTH unsigned:
///   - column pointers (`outer_`)  -> edge_index : a cumulative NON-NEGATIVE
///     offset that can reach billions, so it is the widenable one.
///   - row indices     (`inner_`)  -> node_index : a vertex id, < 2^32 by default,
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
    // Off-diagonal AND diagonal factor values, fp32 (saves ~nnz*4 B at the
    // assembly peak vs fp64; any further narrowing of the off-diagonals
    // happens later, in omp_sptrsv::setup, so the diagonal reaches the SpTRSV
    // at full fp32). Reads should widen(). The
    // diagonal can ride along in fp32: it's L(i,i) = sqrt(weighted degree), a
    // benign well-scaled positive scalar, and fp32 diag converges to 1e-8 just
    // like fp64 (verified on IPM/social/grid; the GPU backend has always stored
    // the diagonal in fp32 too).
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
};

} // namespace apxchol
