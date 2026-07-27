/// Customization seam A: a user-defined star-vertex elimination rule.
///
/// When a vertex is eliminated, the eliminator decides which (sparsified)
/// Schur-complement clique edges are added to the remaining graph.  This
/// example plugs in the exact clique — every neighbor pair, weight
/// w_i·w_j/deg — i.e. zero sampling variance at the cost of more fill-in.
/// See include/apxchol/solver/elimination/elimination.h for the contract.
#include "apxchol.h"
#include "example_grid.h"
#include <cstdio>

struct exact_clique_eliminator {
    void sample_clique(std::span<apxchol::weighted_neighbor> neighbors,
                       double deg,
                       std::uint64_t /*seed: deterministic rule, unused*/,
                       apxchol::edge_emitter out) const {
        for (const auto& [u, wu] : neighbors)
            for (const auto& [v, wv] : neighbors) {
                if (u == v) break;                 // each unordered pair once
                out(u, v, wu * wv / deg);
            }
    }
};
static_assert(apxchol::eliminator<exact_clique_eliminator>);

int main() {
    auto L = grid_laplacian(60);
    Eigen::VectorXd b = apxchol::generate_test_rhs(L.rows());
    apxchol::factor_options opts;

    // Reference: the default tree-sampling eliminator (default storage).
    auto F_tree = apxchol::factorize(L, opts);

    // Custom rule: factorize with our eliminator (the graph is built
    // internally), then hand the factorization to cpu_solver (or
    // apx_cholesky::set_factor) for PCG.
    auto F = apxchol::factorize(L, exact_clique_eliminator{}, opts);
    std::printf("nnz(L): tree-sampled %zu, exact-clique %zu\n",
                size_t(F_tree.L.nonZeros()), size_t(F.L.nonZeros()));

    apxchol::cpu_solver slv(L, std::move(F));
    auto res = slv.solve(b);
    std::printf("exact-clique factor: %ld iterations, residual %.2e\n",
                long(res.iterations), res.residual);

    return res.residual < 1e-8 ? 0 : 1;
}
