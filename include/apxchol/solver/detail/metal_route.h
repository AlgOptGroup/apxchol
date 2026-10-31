#pragma once
// Metal route of solver: PCG with reusable factor, operator and workspace.
// solver validates options and arguments, selects the route and rejects
// overlapping calls; the route trusts its arguments. It cannot be copied.
#include "apxchol/detail/checkpoint.h"
#include "apxchol/solver/factorization.h"
#include "apxchol/solver/solve.h"
#include "apxchol/solver/sptrsv/factor_drop.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace apxchol::detail {

class metal_route {
public:
    // Panel widths solve_columns and apply_columns accept, widest first.
    static constexpr std::array<int, 3> solve_widths{4, 2, 1};
    static constexpr std::array<int, 1> apply_widths{1};

    struct statistics {
        Eigen::Index n = 0;
        std::size_t levels_forward = 0;
        std::size_t levels_backward = 0;
        std::size_t steps_forward = 0;   // dispatch steps per sweep
        std::size_t steps_backward = 0;
        bool operator_double_float = false;  // some operator value is not fp32-exact
        std::string device;
        factor_drop_stats drop;  // what the factor drop removed before upload
    };

    /// Factorizes A on the device, or adopts F when it is not null (exactly,
    /// if F is an installed factor). Construction options (including
    /// regularization/lumping) are not reapplied to F. Factorization requires supported GKS options and
    /// compressed, sorted, paired CSC input satisfying operator_class.h;
    /// numerical limits are in docs/precision.md.
    metal_route(const Eigen::SparseMatrix<double>& A, const solve_options& opts,
                checkpoint* cp, factorization* F);
    ~metal_route();
    metal_route(metal_route&&) noexcept;
    metal_route& operator=(metal_route&&) noexcept;
    metal_route(const metal_route&) = delete;
    metal_route& operator=(const metal_route&) = delete;

    /// Solves the columns of B, a panel of one of solve_widths, into X.
    /// timings, when not null, receives the PCG checkpoints.
    void solve_columns(const_matrix_view B, matrix_view X, const const_matrix_view* X0,
                       std::span<solve_result> results, checkpoint* timings,
                       double tol, int max_iter) const;
    /// One device preconditioner application z = M^{-1} r (centred for a
    /// Laplacian).
    void apply_columns(const_matrix_view R, matrix_view Z) const;

    /// Permutation, rounds and flags; L keeps only its column pointers.
    const factorization& factor() const;
    /// The installed preconditioner (factorization.h), read from the
    /// unified-memory triangles.
    factorization installed_factor() const;
    /// Entries of installed_factor().L.
    std::uint64_t factor_nnz() const;
    factor_storage_type factor_storage() const;
    statistics stats() const;
    /// The original operator in FP64 CSC.
    Eigen::SparseMatrix<double> original_operator() const;

private:
    struct impl;
    struct impl_deleter {
        void operator()(impl*) const;
    };
    std::unique_ptr<impl, impl_deleter> impl_;
};

}  // namespace apxchol::detail
