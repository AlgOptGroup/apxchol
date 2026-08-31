#pragma once
/// BKZ26 Algorithm 1 weighted-Pruefer clique sampling embedded in apxchol.
///
/// This header implements only the product-clique spanning-tree sampler from
/// Algorithm 1 of Baumann--Kyng--Zoecklein (2026).  It does not implement the
/// paper's Algorithm 3: apxchol keeps its own pivot selection, parallel rounds,
/// graph representation, factor assembly, and solve path.

#include "apxchol/solver/elimination/elimination.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace apxchol {

namespace detail {

/// Exact inverse CDF used for one weighted Pruefer symbol. `prefix` is a
/// positive, finite cumulative sum and target is in [0, prefix.back()).
inline size_t bkz26_inverse_cdf(std::span<const double> prefix,
                                double target) {
    const auto it = std::upper_bound(prefix.begin(), prefix.end(), target);
    return it == prefix.end()
        ? prefix.size() - 1
        : static_cast<size_t>(it - prefix.begin());
}

/// Linear-time decoding of a valid Pruefer code. `degrees` must contain
/// one plus each symbol's multiplicity. Edges are emitted in the standard
/// smallest-leaf order, which fixes output order for a fixed code.
template<typename Emit>
inline void bkz26_decode_prufer(std::span<const node_index> code,
                                std::span<node_index> degrees,
                                Emit&& emit) {
    const size_t n = degrees.size();
    if (n < 2) return;

    size_t pointer = 0;
    while (degrees[pointer] != 1) ++pointer;
    size_t leaf = pointer;

    for (const node_index encoded : code) {
        const size_t symbol = static_cast<size_t>(encoded);
        emit(leaf, symbol);
        if (--degrees[symbol] == 1 && symbol < pointer) {
            leaf = symbol;
        } else {
            ++pointer;
            while (pointer < n && degrees[pointer] != 1) ++pointer;
            leaf = pointer;
        }
    }
    emit(leaf, n - 1);
}

} // namespace detail

/// BKZ26 Algorithm 1 clique sampler embedded in apxchol.
///
/// For active-neighbor conductances a_i, W = sum_i a_i, and pivot diagonal
/// `deg`, draw d-2 independent Pruefer symbols with probability a_i / W and
/// decode them into a weighted uniform spanning tree of the product clique.
///
/// On a pure graph Laplacian, deg == W and a sampled edge {i,j} receives
/// conductance a_i*a_j/(a_i+a_j), exactly Algorithm 1 after expressing the
/// Schur clique as a product clique.  apxchol also accepts SDDM pivots where
/// deg > W because diagonal excess is an implicit edge to ground.  For that
/// extension, multiplying by W/deg keeps each emitted edge unbiased for the
/// Schur-clique conductance a_i*a_j/deg.
///
/// Each sampled pivot emits exactly d-1 clique edges, but this local bound does
/// not imply equal final factor fill: the sampled topology and weights change
/// later residual graphs and therefore later pivot neighborhoods.
struct volume_tree_elimination {
    size_t exact_clique_max_degree = 0;

    size_t max_clique_edges(node_index degree) const {
        const size_t d = static_cast<size_t>(degree);
        if (exact_clique_max_degree > 0 && d <= exact_clique_max_degree)
            return d * (d - 1) / 2;
        return d == 0 ? 0 : d - 1;
    }

    void sample_clique(std::span<weighted_neighbor> neighbors,
                       double deg,
                       std::uint64_t seed,
                       edge_emitter out) const {
        const size_t d = neighbors.size();
        if (d < 2 || deg <= 0.0) return;

        // Algorithm 1 requires positive finite product-clique weights. Keep
        // apxchol's established GKS behavior outside that premise.
        for (const auto& neighbor : neighbors) {
            if (!std::isfinite(neighbor.weight) || neighbor.weight <= 0.0) {
                tree_elimination{
                    .exact_clique_max_degree = exact_clique_max_degree}
                    .sample_clique(neighbors, deg, seed, out);
                return;
            }
        }

        if (exact_clique_max_degree > 0 && d <= exact_clique_max_degree) {
            for (const auto& [u, wu] : neighbors)
                for (const auto& [v, wv] : neighbors) {
                    if (u == v) break;
                    out(u, v, wu * wv / deg);
                }
            return;
        }

        // The law is invariant under relabeling. Canonicalization makes the
        // code and ordered edge stream independent of adjacency arrival order.
        if (!detail::radix_sort_neighbors(neighbors)) {
            std::sort(neighbors.begin(), neighbors.end(),
                      [](const auto& a, const auto& b) {
                          return a.weight != b.weight ? a.weight < b.weight
                                                      : a.vertex < b.vertex;
                      });
        }

        static thread_local std::vector<double> prefix;
        prefix.resize(d);
        prefix[0] = neighbors[0].weight;
        for (size_t i = 1; i < d; ++i)
            prefix[i] = prefix[i - 1] + neighbors[i].weight;
        const double W = prefix.back();
        if (!std::isfinite(W) || W <= 0.0) {
            tree_elimination{
                .exact_clique_max_degree = exact_clique_max_degree}
                .sample_clique(neighbors, deg, seed, out);
            return;
        }

        static thread_local std::vector<node_index> code;
        static thread_local std::vector<node_index> degrees;
        code.resize(d - 2);
        degrees.assign(d, node_index{1});

        random_stream random{seed};
        for (size_t k = 0; k + 2 < d; ++k) {
            const size_t symbol = detail::bkz26_inverse_cdf(
                prefix, random.next_unit() * W);
            code[k] = static_cast<node_index>(symbol);
            ++degrees[symbol];
        }

        out.reserve(d - 1);
        const double scale = W / deg;
        detail::bkz26_decode_prufer(
            code, degrees,
            [&](size_t i, size_t j) {
                const double wi = neighbors[i].weight;
                const double wj = neighbors[j].weight;
                out(neighbors[i].vertex, neighbors[j].vertex,
                    scale * wi * wj / (wi + wj));
            });
    }
};

static_assert(eliminator<volume_tree_elimination>);

} // namespace apxchol
