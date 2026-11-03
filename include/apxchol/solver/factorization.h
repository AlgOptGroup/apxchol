#pragma once
#include "apxchol/graph/conversions.h"
#include "apxchol/operator_class.h"
#include "apxchol/solver/elimination/elimination.h"
#include "apxchol/solver/factor_options.h"
#include "apxchol/solver/detail/setup_route.h"
#include "apxchol/solver/detail/factor_entry.h"
#include "apxchol/solver/partitioner.h"
#include "apxchol/sparse_csc.h"
#include "apxchol/detail/csc_input.h"
#include "apxchol/views.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace apxchol {

namespace detail {
struct checkpoint;
#if defined(APXCHOL_USE_CUDA)
class cuda_sptrsv_device_factor;
struct factorization_access;
#endif
}

/// Approximate Cholesky preconditioner. With S = diag(scale), or S = I when
/// scale is empty, it is
///
///     M = P^T L S^2 L^T P,
///
/// which approximates the operator used for construction, including any
/// lumping or regularization. P is the permutation matrix of perm. For a
/// Laplacian, the solve uses only the leading (n-1) x (n-1) blocks of L and S
/// and centers its input and output.
///
/// `storage` says which of two forms L is in:
///   - automatic: a construction factor from factorize(). S = I, and for a
///     Laplacian L's last column holds a placeholder diagonal. A solver that
///     adopts it applies its own drop_tolerance and factor_storage.
///   - fp32 or fp16: the installed preconditioner, as solver::factor()
///     returns it after dropping and storage rounding. For a Laplacian, L's
///     last row and column are empty. A solver adopts it exactly and ignores
///     its drop_tolerance and factor_storage options.
/// fp32: L holds the FP32 values the triangular solves use, and scale is
/// empty. fp16: every off-diagonal of L is a binary16 value (widened exactly
/// to FP32), the diagonal L_jj = d_j is FP32 and scale holds the positive FP32
/// column scales s_j. CPU and CUDA apply S^{-2} as fp32(1 / s_j)^2; Metal
/// forms fp32(L_ij s_j) and fp32(d_j s_j).
struct factorization {
    sparse_csc L;                                           // lower-triangular factor (owned CSC)
    // Elimination-order permutation: perm[original_vertex] = new_position.
    //   to permuted space   : x_perm[perm[v]] = b[v]   (scatter)
    //   from permuted space : x[v] = x_perm[perm[v]]    (gather)
    std::vector<node_index> perm;

    // True if any vertex has positive diagonal excess; false for a pure Laplacian.
    // Affects preconditioner rank and centering.
    bool sddm = false;

    // Stored positive off-diagonal entries removed by M-matrix lumping (two
    // per pair). Only the preconditioner changes; the solve operator is untouched.
    std::int64_t lumped_offdiag = 0;

    // Per-round elimination metadata. is_size drives the SpTRSV round-as-level path.
    struct round_stats {
        size_t active;     // active vertices at start of round
        size_t is_size;    // vertices eliminated this round
    };
    std::vector<round_stats> rounds;

    /// Form of L: automatic for a construction factor, otherwise the
    /// installed triangular-solve storage (see above).
    factor_storage_type storage = factor_storage_type::automatic;
    /// fp16 storage only: column scales s_j, one per column of L (1 for an
    /// empty Laplacian last column). Empty otherwise.
    std::vector<float> scale;

#if defined(APXCHOL_USE_CUDA)
private:
    friend struct detail::factorization_access;
    // CUDA ownership capsule of a factor that stays on the device. Exported
    // factors retain host L values; consuming solves may retain only L's
    // dimensions and column pointers.
    std::shared_ptr<detail::cuda_sptrsv_device_factor> device_factor_;
#endif
};

#if defined(APXCHOL_USE_CUDA)
namespace detail {
struct factorization_access {
    static std::shared_ptr<cuda_sptrsv_device_factor>& device_factor(factorization& F) {
        return F.device_factor_;
    }
    static const std::shared_ptr<cuda_sptrsv_device_factor>& device_factor(const factorization& F) {
        return F.device_factor_;
    }
};
} // namespace detail
#endif

namespace detail {

/// Shared matrix-input path: validate, lump if needed, and build the graph with
/// regularization. The caller's operator remains unchanged for PCG.
template<typename Build>
factorization factorize_operator(csc_ref<double> L,
                                 const factor_options& options, Build&& build) {
    validate_factor_options(options);
    const operator_view op(L, options.lump_max_mass);
    // The graph builder applies these matrix-only options.
    auto graph_options = options;
    graph_options.regularization = 0;
    graph_options.lump_max_mass = kDefaultLumpMassCeiling;
    // The scan's bit-identical-triangles proof holds for L, i.e. only when
    // nothing was lumped.
    factorization F = build(make_graph_from_operator(op.matrix(),
        op.lumped() == 0 && op.scan().triangles_bit_identical, options.regularization), graph_options);
    F.lumped_offdiag = op.lumped();
    return F;
}

} // namespace detail

/// Extension overloads validate the operator, apply permitted lumping and
/// regularization, then run the chosen partitioner and eliminator. A passed
/// partitioner instance replaces `opts.partitioner`; otherwise
/// `opts.partitioner` names the built-in one.
///
/// Custom-eliminator overload: substitute your own star-vertex elimination
/// rule (see elimination/elimination.h for the `eliminator` contract).
/// Overloads without an eliminator build the built-in rule from `opts`.
/// Every overload takes a csc_view or csc_view64 (views.h); invalid views
/// throw std::invalid_argument.
template<class Index, eliminator E>
factorization factorize(basic_csc_view<Index> L, const E& elim,
                        const factor_options& opts = {});

/// Partitioner-instance overloads: pass a configured (possibly stateful)
/// partitioner — the seam for custom independent-set selection and custom
/// elimination order (see solver/partitioner.h for the concept and contract).
/// The instance is taken by value; state mutated during the run is not
/// observable afterwards — reference it through a member pointer if you need
/// to inspect it post-run.
template<class Index, partitioner P>
factorization factorize(basic_csc_view<Index> L, P part,
                        const factor_options& opts = {});

template<class Index, partitioner P, eliminator E>
factorization factorize(basic_csc_view<Index> L, P part,
                        const E& elim,
                        const factor_options& opts = {});

/// Matrix-level factorization using the built-in AoS graph and named partitioner.
template<class Index>
factorization factorize(basic_csc_view<Index> L,
                        const factor_options& opts = {});

namespace detail {

// Internal consuming route used only by the solver routes. Public factorize()
// always retains exportable host arrays; the device-resident route omits them.
factorization factorize_for_solver(csc_ref<double> L,
                                  const factor_options& opts,
                                  checkpoint* cp,
                                  setup_route route = setup_route::cpu);

struct factor_col {
    node_index vertex;
    // sqrt(total degree including SDDM excess), computed in FP64 and stored
    // at the factor's precision.
    factor_value_t diag = 0;
    // Off-diagonal entries borrow per-thread arena storage until CSC assembly.
    // Device-resident consuming factors may omit them while retaining entry_count.
    factor_entry* entries = nullptr;
    node_index entry_count = 0;
};

// Build elimination-order permutation and assemble L in CSC format.
// Precondition: factor_cols contains exactly one entry per vertex (all
// n vertices have been eliminated before this call).
void build_csc(factorization& result,
               const std::vector<factor_col>& factor_cols,
               node_index n,
               checkpoint* cp, bool build_values = true);

} // namespace detail

} // namespace apxchol

#include "apxchol/solver/factorization_impl.h"
