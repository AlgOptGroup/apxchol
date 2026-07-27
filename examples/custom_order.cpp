/// Customization seams B + C: independent-set selection and elimination order.
///
/// The partitioner selects the vertices eliminated each round, and the global
/// elimination order (the permutation P) is exactly the concatenation of those
/// rounds.  This partitioner honors a user-supplied per-vertex priority: each
/// round it scans vertices in ascending priority and greedily keeps an
/// independent set, so low-priority vertices are eliminated first.
/// See include/apxchol/solver/partitioner.h for the full contract.
#include "apxchol.h"
#include "apxchol/solver/partitioner_helpers.h"
#include "example_grid.h"
#include <algorithm>
#include <cstdio>
#include <span>
#include <vector>

struct priority_partitioner {
    static constexpr std::string_view name = "priority";

    /// User-supplied priority per vertex (lower = eliminated earlier).
    std::vector<double> priority;

    template<apxchol::incidence_storage Incidence>
    void find_partition(apxchol::graph<Incidence>& G,
                        std::span<const apxchol::node_index> active,
                        const apxchol::partition_context& /*ctx*/,
                        apxchol::selection& out) {
        // Scan vertices in priority order; greedily keep an independent set
        // (the selection answers the membership queries).
        order_.assign(active.begin(), active.end());
        std::sort(order_.begin(), order_.end(),
                  [&](auto a, auto b) { return priority[a] < priority[b]; });
        for (auto v : order_) {
            bool independent = true;
            for (auto idx : G.adj(v)) {
                auto u = G.edge_target(idx, v);
                if (G.is_active(u) && out.contains(u)) { independent = false; break; }
            }
            if (independent) out.add(v);
        }
    }

private:
    std::vector<apxchol::node_index> order_;       // sorted candidate list
};
static_assert(apxchol::partitioner<priority_partitioner>);

int main() {
    const int k = 60;
    auto L = grid_laplacian(k);
    Eigen::VectorXd b = apxchol::generate_test_rhs(L.rows());

    // Eliminate the grid interior first, boundary last (an arbitrary
    // demonstration order — plug in ND/AMD/domain knowledge here).
    priority_partitioner part;
    part.priority.resize(size_t(k) * k);
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) {
            const bool boundary = i == 0 || j == 0 || i == k - 1 || j == k - 1;
            part.priority[size_t(i) * k + j] = boundary ? 1.0 : 0.0;
        }

    auto F = apxchol::factorize(L, std::move(part));
    std::printf("custom-order factor: nnz %zu, %zu rounds\n",
                size_t(F.L.nonZeros()), F.rounds.size());

    apxchol::cpu_solver slv(L, std::move(F));
    auto res = slv.solve(b);
    std::printf("custom-order solve : %ld iterations, residual %.2e\n",
                long(res.iterations), res.residual);

    return res.residual < 1e-8 ? 0 : 1;
}
