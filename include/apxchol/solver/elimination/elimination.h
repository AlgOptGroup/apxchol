#pragma once
/// Modular elimination (clique sampling) strategies for approximate Cholesky.
///
/// When a vertex v is eliminated from a Laplacian graph, the exact Schur
/// complement adds a weighted clique on v's neighbors.  To control fill-in,
/// we sample a sparse approximation of this clique.  Different sampling
/// strategies offer different trade-offs between approximation quality
/// (fewer rounds needed) and sparsity (less fill-in per round).
///
/// Each eliminator is a stateless struct satisfying a duck-typed interface:
///
///   struct SomeEliminator {
///       // Sample clique edges for the elimination of one vertex.
///       // Pre: `valid` contains (neighbor, weight) pairs sorted by weight.
///       //       `deg` is the weighted degree sum(w).
///       // The implementation appends sampled edges to `edges_out`.
///       void sample_clique(std::span<const std::pair<node_index, double>> valid,
///                          double deg,
///                          std::mt19937& gen,
///                          std::vector<deferred_edge>& edges_out,
///                          std::vector<double>& scratch) const;
///   };
///
/// The factor column (L entries) is always constructed identically —
/// entries[j] = (neighbor_j, w_j / sqrt(deg)) — and is handled by the
/// caller.  The eliminator only controls which clique edges are generated.
///
/// Available strategies:
///   - tree_elimination    — Random spanning tree of the clique (d-1 edges).
///                           Current default.  Each neighbor i is paired with
///                           one neighbor j>i sampled proportional to weight
///                           from the suffix, producing harmonic-mean edges.
///
///
///   - star_elimination    — Pick the heaviest-edge neighbor as center, connect
///                           all other d-1 neighbors to it with harmonic-mean
///                           weights.  Very sparse (d-1 edges, star topology).
///
///   - clique_elimination  — Exact Schur complement: add all d(d-1)/2 clique
///                           edges.  No approximation error, but O(d²) fill-in.
///
///   - single_elimination  — Ultra-sparse: emit exactly one random clique edge
///                           with weight scaled by (d-1).  Risk: may disconnect
///                           the graph, weakening the preconditioner.

#include "apxchol/types.h"
#include "apxchol/solver/factor_options.h"
#include <algorithm>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace apxchol::detail {

/// Deferred clique edge for batched application after parallel elimination.
struct deferred_edge { node_index u, v; double w; };

/// Tree elimination: spanning tree of the clique.
///
/// For each neighbor i = 0..d-2, sample one neighbor j from the suffix
/// {i+1, ..., d-1} with probability proportional to edge weight.
/// The resulting d-1 edges form a random spanning tree of the clique
/// on v's neighbors.  Edge weight = w_i·w_j / (w_i + w_j) (harmonic mean).
///
    struct tree_elimination {
        void sample_clique(std::span<const std::pair<node_index, double>> valid,
                           double deg,
                           std::mt19937& gen,
                           std::vector<deferred_edge>& edges_out,
                           std::vector<double>& prefix) const {
            if (valid.size() < 2 || deg <= 0.0) return;

            prefix.resize(valid.size());
            prefix[0] = valid[0].second;
            for (size_t i = 1; i < valid.size(); ++i)
                prefix[i] = prefix[i - 1] + valid[i].second;

            for (size_t i = 0; i + 1 < valid.size(); ++i) {
                const double suffix_sum = prefix.back() - prefix[i];
                if (suffix_sum <= 0.0) continue;

                std::uniform_real_distribution<double> U(0.0, suffix_sum);
                const double r = U(gen);

                auto it = std::upper_bound(
                    prefix.begin() + static_cast<std::ptrdiff_t>(i) + 1,
                    prefix.end(),
                    prefix[i] + r);

                size_t j = static_cast<size_t>(it - prefix.begin());
                if (j >= valid.size()) j = valid.size() - 1;

                auto [va, wa] = valid[i];
                auto [vb, wb] = valid[j];
                (void)wb;

                const double w = wa * suffix_sum / deg;
                edges_out.push_back({va, vb, w});
            }
        }
    };

/// IID elimination: Kyng-Sachdeva Algorithm 2 (CliqueSample).
///
/// Draw deg(v) independent samples.  Each sample:
///   1. e1 ~ edges incident on v, proportional to weight.
///   2. e2 ~ edges incident on v, uniform.
///   3. If e1, e2 have different non-v endpoints u1, u2:
///      emit edge (u1, u2) with weight w(e1)·w(e2) / (w(e1) + w(e2)).
///
/// In expectation, the sum equals the exact clique C_v(S).
/// Produces ~d/2 edges in expectation (half the pairs collide → 0).
struct iid_elimination {
    void sample_clique(std::span<const std::pair<node_index, double>> valid,
                       double /*deg*/,
                       std::mt19937& gen,
                       std::vector<deferred_edge>& edges_out,
                       std::vector<double>& prefix) const {
        if (valid.size() < 2) return;

        const size_t d = valid.size();

        // Build prefix-sum array for proportional sampling of e1.
        prefix.resize(d);
        prefix[0] = valid[0].second;
        for (size_t i = 1; i < d; ++i)
            prefix[i] = prefix[i - 1] + valid[i].second;

        std::uniform_int_distribution<size_t> unif_idx(0, d - 1);

        for (size_t s = 0; s < d; ++s) {
            // Sample e1 proportional to weight.
            std::uniform_real_distribution<double> U(0.0, prefix.back());
            double r1 = U(gen);
            size_t i = static_cast<size_t>(
                std::lower_bound(prefix.begin(), prefix.end(), r1) - prefix.begin());
            if (i >= d) i = d - 1;

            // Sample e2 uniformly.
            size_t j = unif_idx(gen);

            if (i == j) continue;  // same neighbor → Y = 0

            auto [va, wa] = valid[i];
            auto [vb, wb] = valid[j];
            double w = wa * wb / (wa + wb);
            edges_out.push_back({va, vb, w});
        }
    }
};

/// Star elimination: connect all neighbors to the heaviest-edge neighbor.
///
/// Picks the neighbor with the largest edge weight as the "center" and
/// creates d-1 edges connecting every other neighbor to the center.
/// Each edge has weight w_center·w_i / (w_center + w_i) (harmonic mean).
///
/// Advantages: very simple, deterministic, d-1 edges like tree.
/// Disadvantage: less uniform approximation of the clique — effective
/// resistance estimates for non-center pairs may be worse.
struct star_elimination {
    void sample_clique(std::span<const std::pair<node_index, double>> valid,
                       double /*deg*/,
                       std::mt19937& /*gen*/,
                       std::vector<deferred_edge>& edges_out,
                       std::vector<double>& /*scratch*/) const {
        if (valid.size() < 2) return;

        // valid is sorted by weight ascending; last element is heaviest.
        auto [center, wc] = valid.back();

        for (size_t i = 0; i + 1 < valid.size(); ++i) {
            auto [vi, wi] = valid[i];
            double w = wc * wi / (wc + wi);
            edges_out.push_back({vi, center, w});
        }
    }
};

/// Clique elimination: exact Schur complement (no sparsification).
///
/// Adds all d(d-1)/2 clique edges with exact weights:
///   w_{ij} = w_i · w_j / sum(w_k for all neighbors k of v)
///
/// No approximation error, so fewer PCG iterations may be needed.
/// However, O(d²) fill-in per eliminated vertex can make the graph
/// much denser in later rounds.
struct clique_elimination {
    void sample_clique(std::span<const std::pair<node_index, double>> valid,
                       double deg,
                       std::mt19937& /*gen*/,
                       std::vector<deferred_edge>& edges_out,
                       std::vector<double>& /*scratch*/) const {
        if (valid.size() < 2) return;

        for (size_t i = 0; i < valid.size(); ++i) {
            auto [vi, wi] = valid[i];
            for (size_t j = i + 1; j < valid.size(); ++j) {
                auto [vj, wj] = valid[j];
                double w = wi * wj / deg;
                edges_out.push_back({vi, vj, w});
            }
        }
    }
};

} // namespace apxchol::detail
