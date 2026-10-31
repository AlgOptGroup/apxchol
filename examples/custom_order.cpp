/// Customization seams B + C: independent-set selection and elimination order.
///
/// The partitioner selects the vertices eliminated each round, and the global
/// elimination order (the permutation P) is exactly the concatenation of those
/// rounds.  This partitioner honors a user-supplied per-vertex priority: each
/// round it scans vertices in ascending priority and greedily keeps an
/// independent set, so low-priority vertices are eliminated first.
/// See include/apxchol/solver/partitioner.h for the full contract.
#include "apxchol.h"
#include "example_grid.h"
#include <algorithm>
#include <cstdio>
#include <span>
#include <vector>

struct priority_partitioner {
    /// User-supplied priority per vertex (lower = eliminated earlier).
    std::vector<double> priority;

    void find_partition(const apxchol::partition_graph& G,
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
            G.for_each_neighbor(v, [&](auto u) {
                if (out.contains(u)) independent = false;
            });
            if (independent) out.add(v);
        }
    }

private:
    std::vector<apxchol::node_index> order_;       // sorted candidate list
};
static_assert(apxchol::partitioner<priority_partitioner>);

int main() {
    const int k = 60;
    const auto grid = grid_laplacian(k);
    const apxchol::csc_view L = grid.view();
    std::vector<double> b(static_cast<std::size_t>(L.n), 0.0);   // unit source and sink
    b.front() = 1.0;
    b.back() = -1.0;

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

    apxchol::solver slv(L, std::move(F));
    auto res = slv.solve(b);
    std::printf("custom-order solve : %ld iterations, residual %.2e\n",
                long(res.iterations), res.residual);

    return res.residual < 1e-8 ? 0 : 1;
}
