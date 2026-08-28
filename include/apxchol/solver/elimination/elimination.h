#pragma once
/// Elimination (clique sampling) strategies for approximate Cholesky — the
/// star-vertex elimination rule, and the customization seam for replacing it.
///
/// When a vertex v is eliminated from a Laplacian graph, the exact Schur
/// complement adds a weighted clique on v's neighbors.  To control fill-in,
/// an eliminator samples a sparse approximation of this clique.  Different
/// sampling rules trade approximation quality (fewer PCG iterations) against
/// sparsity (less fill-in).
///
/// ── Writing your own eliminator ──
///
/// An eliminator is a struct satisfying the `eliminator` concept below:
///
///   struct my_eliminator {
///       // Sample clique edges for the elimination of one vertex v.
///       // Pre:  `neighbors` holds v's ACTIVE neighbors with parallel edges
///       //       already merged (each vertex appears once, weights summed).
///       //       Order is unspecified; the span is mutable and the library
///       //       never reads it again after your call — permute it, or
///       //       overwrite it entirely, as suits your rule.
///       //       `deg` is v's total weighted degree. NOTE: it is NOT
///       //       necessarily the sum of the `neighbors` weights — on SDDM
///       //       inputs the diagonal excess (an implicit edge to ground)
///       //       contributes to `deg` but has no neighbor entry. Always use
///       //       the passed value; never recompute it.
///       // Post: sampled edges handed to `out`.  For an unbiased
///       //       preconditioner, E[sum of emitted edges] should equal the
///       //       exact Schur-complement clique (weight w_i*w_j/deg per pair).
///       // `seed` is this elimination's random seed — a pure function of
///       //  (factor seed, vertex), so the seed/draw sequence is identical
///       //  across runs, thread counts, and schedules. The resulting factor
///       //  is bit-identical at a fixed thread count; at T>1 merged
///       //  parallel-edge weights can differ in the final ulps (fp
///       //  accumulation order). Deterministic rules simply ignore it;
///       //  randomized rules derive their draws from it (the random_stream
///       //  helper below, or seed any generator you like). ONE eliminator
///       //  instance is shared by ALL threads (passed by const&), so it must
///       //  be const-callable and hold no mutable members; for per-thread
///       //  working memory use `static thread_local` locals, as the built-in
///       //  rule does for its prefix sums. Each call must itself stay
///       //  single-threaded — the parallelism is outside (one eliminated
///       //  vertex per task).
///       void sample_clique(std::span<weighted_neighbor> neighbors,
///                          double deg,
///                          std::uint64_t seed,
///                          edge_emitter out) const;
///   };
///
///   auto F = apxchol::factorize(A, my_eliminator{}, opts);
///
/// A lambda (or any callable with the same argument list) can be adapted with
/// `as_eliminator`:
///
///   auto F = apxchol::factorize(A, apxchol::as_eliminator(
///       [](std::span<apxchol::weighted_neighbor> nb, double deg,
///          std::uint64_t seed, apxchol::edge_emitter out) { ... }));
///
/// The factor column (L entries) is always constructed identically —
/// entries[j] = (neighbor_j, w_j / sqrt(deg)) — and is handled by the
/// caller.  The eliminator only controls which clique edges are generated.
///
/// `tree_elimination` is the built-in rule: a random spanning tree of the
/// clique (d-1 edges), with an optional exact-clique mode below a degree
/// threshold.

#include "apxchol/types.h"
#include "apxchol/solver/factor_options.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace apxchol {

/// One (deduplicated, active) neighbor of the vertex being eliminated.
struct weighted_neighbor {
    node_index vertex;
    double     weight;
};

/// A sampled clique edge. "Deferred" because the elimination round runs in
/// parallel: edges are collected in per-thread buffers and applied to the
/// shared graph in a batched second phase, not inserted immediately.
struct deferred_edge { node_index u, v; double w; };

/// Append-only sink for sampled clique edges: eliminators can add edges but
/// not inspect or remove previously emitted ones. Cheap to copy (wraps a
/// pointer to the per-thread buffer).
class edge_emitter {
public:
    explicit edge_emitter(std::vector<deferred_edge>& buf) : buf_(&buf) {}
    void operator()(deferred_edge e) const { buf_->push_back(e); }
    void operator()(node_index u, node_index v, double w) const {
        buf_->push_back({u, v, w});
    }
    /// Bulk append (e.g. a whole precomputed tree at once).
    void operator()(std::span<const deferred_edge> es) const {
        buf_->insert(buf_->end(), es.begin(), es.end());
    }
    /// Reserve capacity for `n` additional edges.
    void reserve(size_t n) const { buf_->reserve(buf_->size() + n); }

private:
    std::vector<deferred_edge>* buf_;
};

/// Minimal counter-based random stream for eliminators: construct from the
/// per-elimination seed, then draw.  Cheap (a few arithmetic ops per draw,
/// no large state) and deterministic at any thread count.  splitmix64 core.
struct random_stream {
    std::uint64_t state;

    std::uint64_t next() {
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    /// Uniform double in [0, 1).
    double next_unit() { return double(next() >> 11) * 0x1.0p-53; }
};

/// Concept for a star-vertex elimination rule (see the header comment for the
/// full contract).  Models: tree_elimination (built-in), user-defined rules,
/// as_eliminator(lambda).
template<typename T>
concept eliminator = requires(const T& t,
        std::span<weighted_neighbor> neighbors,
        double deg,
        std::uint64_t seed,
        edge_emitter out) {
    t.sample_clique(neighbors, deg, seed, out);
};

/// Adapt a callable (e.g. a lambda) with the sample_clique argument list into
/// an eliminator.
template<typename F>
struct callable_eliminator {
    F fn;
    void sample_clique(std::span<weighted_neighbor> neighbors, double deg,
                       std::uint64_t seed, edge_emitter out) const {
        fn(neighbors, deg, seed, out);
    }
};

template<typename F>
callable_eliminator<std::decay_t<F>> as_eliminator(F&& fn) {
    return {std::forward<F>(fn)};
}

namespace detail {

/// Reproduce the canonical (weight, vertex) order with a stable weight radix
/// followed by a vertex sort inside each equal-weight run.  Comparison sort
/// remains faster below the measured crossover; larger cliques repay the six
/// linear passes of the exact 11-bit radix.
inline bool radix_sort_neighbors(std::span<weighted_neighbor> values) {
    constexpr size_t kMinDegree = 512;
    if (values.size() < kMinDegree) return false;
    bool all_equal = true;
    for (const auto& value : values) {
        if (!std::isfinite(value.weight) || value.weight < 0.0)
            return false;
        all_equal = all_equal && value.weight == values.front().weight;
    }
    if (all_equal) {
        std::sort(values.begin(), values.end(),
                  [](const auto& a, const auto& b) {
                      return a.vertex < b.vertex;
                  });
        return true;
    }

    constexpr unsigned kBits = 11;
    constexpr size_t kBuckets = size_t{1} << kBits;
    constexpr std::uint64_t kMask = kBuckets - 1;
    static thread_local std::vector<weighted_neighbor> scratch;
    static thread_local std::array<size_t, kBuckets> counts;
    scratch.resize(values.size());

    auto pass = [&](std::span<const weighted_neighbor> source,
                    std::span<weighted_neighbor> destination,
                    unsigned shift) {
        std::fill(counts.begin(), counts.end(), size_t{0});
        auto key = [](const weighted_neighbor& value) -> std::uint64_t {
            // Numeric order and IEEE bit order agree for finite nonnegative
            // doubles. Normalize signed zero because the comparator treats
            // both zero encodings as equal and then breaks the tie by vertex.
            return value.weight == 0.0 ? 0
                : std::bit_cast<std::uint64_t>(value.weight);
        };
        for (const auto& value : source)
            ++counts[(key(value) >> shift) & kMask];
        size_t offset = 0;
        for (size_t& count : counts) {
            const size_t next = offset + count;
            count = offset;
            offset = next;
        }
        for (const auto& value : source)
            destination[counts[(key(value) >> shift) & kMask]++] = value;
    };
    for (unsigned shift = 0; shift < 64; shift += 2 * kBits) {
        pass(values, scratch, shift);
        pass(scratch, values, shift + kBits);
    }
    for (size_t first = 0; first < values.size();) {
        size_t last = first + 1;
        while (last < values.size() &&
               values[last].weight == values[first].weight)
            ++last;
        if (last - first > 1) {
            std::sort(values.begin() + static_cast<std::ptrdiff_t>(first),
                      values.begin() + static_cast<std::ptrdiff_t>(last),
                      [](const auto& a, const auto& b) {
                          return a.vertex < b.vertex;
                      });
        }
        first = last;
    }
    return true;
}

} // namespace detail

/// Tree elimination: spanning tree of the clique (CliqueTreeSample).
/// The built-in (and default) eliminator; configured from factor_options
/// (exact_clique_max_degree).
///
/// Implements Algorithm 3 / Equation (7) of Gao, Kyng, Spielman (2023),
/// "Robust and Practical Solution of Laplacian Equations by Approximate
/// Elimination" (arXiv:2303.00709).
///
/// For each neighbor i = 0..d-2 (in ascending-weight order):
///   - sample one neighbor j from the suffix {i+1,...,d-1} with
///     probability  p_i(j) = a(j) / (1^T a_i),
///   - emit edge (i,j) with weight  w̃(i,j) = a(i) · (1^T a_i) / d,
/// where (1^T a_i) is the suffix sum w_{i+1}+...+w_{d-1} and d = S(v,v)
/// is the total weighted degree.  Unbiased per clique edge: Claim 4.3
/// shows E[Σ_i S̃_i] = Clique[L]_v.
///
/// The estimator is unbiased for ANY fixed neighbor order; sorting ascending
/// by weight is the GKS variance-reduction choice — every light source pairs
/// into its heavier suffix (its edge estimate is dominated by the reliable
/// heavy mass), and only the heavy-heavy tail is sampled among itself.
///
/// Earlier code used the iid pairwise harmonic mean w_i·w_j/(w_i+w_j)
/// here; that is the Kyng-Sachdeva CliqueSample weight (BK SPAA'24
/// Algorithm 4 / iid_elimination), not the tree-sample weight.  The two
/// agree only on degree-2 suffixes (where d = w_i+w_j).
struct tree_elimination {
    /// Exact-clique-at-low-degree: when the eliminated vertex's degree is at
    /// most this value, emit the FULL exact Schur-complement clique
    /// (all d(d-1)/2 edges, weight w_i·w_j/deg) instead of the d-1 sampled
    /// edges. Zero-variance where it is cheap (quadratic in d). 0 = off.
    size_t exact_clique_max_degree = 0;

    /// Upper bound on the number of edges one sample_clique call emits, for a
    /// vertex of the given degree (used by tests / capacity reservations).
    size_t max_clique_edges(node_index deg) const {
        const size_t d = static_cast<size_t>(deg);
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

        // Exact clique at low degree: emit the full Schur-complement update
        // (every neighbor pair, weight w_i·w_j/deg) with zero sampling
        // variance. Quadratic in d, so gated by exact_clique_max_degree.
        if (exact_clique_max_degree > 0 && d <= exact_clique_max_degree) {
            for (const auto& [u, wu] : neighbors)
                for (const auto& [v, wv] : neighbors) {
                    if (u == v) break;                 // each unordered pair once
                    out(u, v, wu * wv / deg);
                }
            return;
        }

        // Canonical order for the suffix sampler (see the header comment).
        // Vertex id breaks weight ties so the order — and therefore the
        // sampling — does not depend on the schedule-dependent arrival order.
        if (!detail::radix_sort_neighbors(neighbors)) {
            std::sort(neighbors.begin(), neighbors.end(),
                      [](const auto& a, const auto& b) {
                          return a.weight != b.weight ? a.weight < b.weight
                                                      : a.vertex < b.vertex;
                      });
        }

        // Prefix sums of the sorted weights; per-thread reusable buffer.
        static thread_local std::vector<double> prefix;
        prefix.resize(d);
        prefix[0] = neighbors[0].weight;
        for (size_t i = 1; i < d; ++i)
            prefix[i] = prefix[i - 1] + neighbors[i].weight;

        random_stream rs{seed};
        // A coarse inverse-CDF directory narrows each exact upper_bound to one
        // cumulative-mass bucket. The final search and RNG are unchanged, so
        // emitted edges remain byte-identical. Building the directory loses on
        // small cliques; degree 512 is the conservative end-to-end validated
        // cutoff on both x86-64 and Grace.
        static thread_local std::vector<size_t> inverse_cdf;
        constexpr size_t kDirectoryMinDegree = 512;
        const bool use_buckets = d >= kDirectoryMinDegree &&
                                 std::isfinite(prefix.back()) &&
                                 prefix.back() > 0.0;
        size_t bucket_count = 0;
        double bucket_scale = 0.0;
        if (use_buckets) {
            bucket_count = std::min<size_t>(65'536, d / 8);
            inverse_cdf.resize(bucket_count + 1);
            bucket_scale = static_cast<double>(bucket_count) / prefix.back();
            auto bucket_of = [&](double value) {
                const double scaled = value * bucket_scale;
                if (!(scaled > 0.0)) return size_t{0};
                return std::min(bucket_count - 1,
                                static_cast<size_t>(scaled));
            };
            size_t pos = 0;
            inverse_cdf[0] = 0;
            for (size_t bucket = 1; bucket < bucket_count; ++bucket) {
                while (pos < d && bucket_of(prefix[pos]) < bucket) ++pos;
                inverse_cdf[bucket] = pos;
            }
            inverse_cdf[bucket_count] = d;
        }
        for (size_t i = 0; i + 1 < d; ++i) {
            const double suffix_sum = prefix.back() - prefix[i];
            if (suffix_sum <= 0.0) continue;

            const double r = rs.next_unit() * suffix_sum;

            const double target = prefix[i] + r;
            auto first = prefix.begin() + static_cast<std::ptrdiff_t>(i) + 1;
            auto last = prefix.end();
            if (use_buckets) {
                const double scaled = target * bucket_scale;
                const size_t bucket = !(scaled > 0.0) ? 0 :
                    std::min(bucket_count - 1, static_cast<size_t>(scaled));
                first = prefix.begin() + static_cast<std::ptrdiff_t>(
                    std::max(i + 1, inverse_cdf[bucket]));
                last = prefix.begin() + static_cast<std::ptrdiff_t>(
                    inverse_cdf[bucket + 1]);
            }
            auto it = std::upper_bound(first, last, target);

            size_t j = static_cast<size_t>(it - prefix.begin());
            if (j >= d) j = d - 1;

            out(neighbors[i].vertex, neighbors[j].vertex,
                neighbors[i].weight * suffix_sum / deg);
        }
    }
};

static_assert(eliminator<tree_elimination>);

/// Back-compat aliases: these types predate the public eliminator seam.
namespace detail {
using apxchol::deferred_edge;
using apxchol::tree_elimination;
} // namespace detail

} // namespace apxchol
