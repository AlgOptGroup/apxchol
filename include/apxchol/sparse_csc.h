#pragma once
#include "apxchol/types.h"
#include "apxchol/lowprec.h"
#include <span>
#include <vector>
#include <cstddef>

namespace apxchol {

// Value types of the factor and of the SpTRSV's own copies of it, shared by
// BOTH the OpenMP (omp.h) and CUDA (cuda.h) backends so they mean the same
// thing on CPU and GPU. Both are fp32, UNCONDITIONALLY: fp32 halves the two
// largest factor copies (csr/csc on CPU, d_vals on GPU), cutting memory and
// the bandwidth of the bandwidth-bound triangular solve, and costs only a few
// extra PCG iterations: the outer PCG (vectors, dots, SpMV/residual) stays
// fp64 and still converges to 1e-8.
//
//   - factor_value_t : what the FACTOR (sparse_csc::values(), the assembler's
//     output) stores. The SpTRSV's setup reads the exact fp32 diagonal and
//     the per-column scales off it, and the factor is released right after
//     setup on the consuming path.
//   - sptrsv_value_t : the DEFAULT width of the SpTRSV kernels' CSR/CSC value
//     arrays (and of the GPU backend's d_vals): fp32. Both backends can narrow
//     those arrays to fp16 at setup instead (factor_options.factor_storage,
//     lowprec.h) -- that is a per-setup choice of storage type, never a
//     compile-time typedef.
using factor_value_t = float;
namespace detail { using sptrsv_value_t = float; }

namespace detail { struct sparse_csc_access; }

/// Owning Compressed-Sparse-Column matrix for the factor L, with the build's
/// own index widths.
///
/// Two index roles, deliberately different (see types.h) -- BOTH unsigned:
///   - column pointers (`col_ptrs()`)    -> edge_index : a cumulative NON-NEGATIVE
///     offset that can reach billions, so it is the widenable one.
///   - row indices     (`row_indices()`) -> node_index : a vertex id, < 2^32 by
///     default, so it stays 32-bit with APXCHOL_INDEX_WIDTH=64_EDGES.
///
/// The public surface is a read-only view. The assembler, archive and GPU
/// finalize code build and release the storage through detail::sparse_csc_access.
class sparse_csc {
public:
    sparse_csc() = default;

    node_index rows() const { return n_; }
    node_index cols() const { return n_; }
    edge_index nonZeros() const { return outer_.empty() ? edge_index(0) : outer_.back(); }
    /// False once a solver released the row indices and values after its
    /// SpTRSV setup; the column pointers stay.
    bool has_values() const { return !vals_.empty(); }

    /// rows()+1 cumulative column pointers.
    std::span<const edge_index>     col_ptrs() const { return outer_; }
    /// nonZeros() row indices; each column lists its diagonal first.
    std::span<const node_index>     row_indices() const { return inner_; }
    /// nonZeros() values, parallel to row_indices().
    std::span<const factor_value_t> values() const { return vals_; }

private:
    friend struct detail::sparse_csc_access;

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
};

namespace detail {

/// Write access to sparse_csc storage for code that builds, loads or releases a
/// factor. Not part of the public API.
struct sparse_csc_access {
    static node_index& n(sparse_csc& L) { return L.n_; }
    static std::vector<edge_index>&     outer(sparse_csc& L) { return L.outer_; }
    static std::vector<node_index>&     inner(sparse_csc& L) { return L.inner_; }
    static std::vector<factor_value_t>& vals(sparse_csc& L) { return L.vals_; }
    static const std::vector<edge_index>&     outer(const sparse_csc& L) { return L.outer_; }
    static const std::vector<node_index>&     inner(const sparse_csc& L) { return L.inner_; }
    static const std::vector<factor_value_t>& vals(const sparse_csc& L) { return L.vals_; }

    /// Size an n x n factor with zeroed column pointers.
    static void resize(sparse_csc& L, node_index n) {
        L.n_ = n;
        L.outer_.assign(static_cast<std::size_t>(n) + 1, edge_index(0));
    }
    static void resize_nonzeros(sparse_csc& L, edge_index nnz) {
        L.inner_.resize(static_cast<std::size_t>(nnz));
        L.vals_.resize(static_cast<std::size_t>(nnz));
    }

    /// Free the large row-index + value arrays, keeping the column pointers so
    /// nonZeros()/rows()/cols() still work. Called once the factor has been
    /// consumed by the SpTRSV setup (which copies it into its own CSR/CSC): the
    /// PCG loop never touches F_.L afterwards, so this drops one full copy of the
    /// factor: nnz * (sizeof(node_index) + sizeof(factor_value_t)) bytes.
    /// swap-with-empty releases the arrays' allocations.
    static void release_values(sparse_csc& L) {
        std::vector<node_index>().swap(L.inner_);
        std::vector<factor_value_t>().swap(L.vals_);
    }
};

} // namespace detail

} // namespace apxchol
