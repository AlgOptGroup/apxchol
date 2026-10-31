#pragma once
#include "apxchol/csc_work.h"
#include "apxchol/detail/parallel_exception.h"
#include "apxchol/graph/incidence_list.h"
#include "apxchol/graph/residual_normalization.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <ranges>
#include <span>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace apxchol {

namespace detail {
struct residual_coalescer;
}

// Residual-pool edge weight storage type. fp32 under -DAPXCHOL_POOL_FP32 (ON by
// default; pass =OFF for fp64). The residual graph is a MULTIGRAPH: adjacency writes
// APPEND new edges and weights are never re-summed in place (the per-vertex
// dedup/weighted-degree aggregation is done transiently in fp64 at read), so each
// stored weight is a TERMINAL write -- fp32 is safe (verified: PCG iters/residual
// unchanged across IPM incl. high-DR, social, grid). fp32 halves the 16 B edge
// {node_index + double, alignment-padded} to 8 B {node_index + float}, ~halving
// graph payload. The
// main elimination remains append-only: maintaining one physical edge per
// active endpoint pair would require a concurrent dynamic pair map, while
// physically merging only the pivot's adjacency was measured slower than the
// current transient fp64 aggregation in process_vertex(). A governed one-shot
// rebuild may coalesce the late residual; the degree heuristic then counts
// distinct neighbors rather than retaining duplicate-edge multiplicity.
/// Mutable incidence-list graph.
///
class graph {
public:
    graph() = default;

    explicit graph(node_index n)
        : n_(n), m_(0), num_active_(n),
          active_(active_word_count(n), ~active_word{0}),
          excess_(n, 0.0) {
        adj_.init(n);
    }

    node_index n() const { return n_; }   // vertex count
    /// Edges EVER added (can exceed 2^31). MONOTONE: elimination prunes edges
    /// but never decrements this, so mid-elimination `m()` is not the live edge
    /// count and `2*m()/active` is not the live average degree — it is inflated
    /// by every edge the eliminated prefix consumed. Anything needing a live
    /// degree must measure one (prune_and_degrees) or be handed one.
    edge_index m() const { return m_; }

    // Yields {to, w} with w PROMOTED to double, so all consumers (elimination,
    // weighted-degree, conversions) compute in fp64 regardless of pool storage.
    auto neighbors(node_index v) const {
        return adj_[v] | std::views::transform(
            [](const detail::directed_pool_edge& i) {
                struct nbr { node_index to; double w; };
                return nbr{i.to, static_cast<double>(i.w)};
            });
    }

    // ── mutation ──

    void add_edge(node_index u, node_index v, double w) {
        adj_.push(u, {v, static_cast<detail::pool_value_t>(w)});
        adj_.push(v, {u, static_cast<detail::pool_value_t>(w)});
        ++m_;
    }

    void deactivate(node_index v) {
        if (!clear_active_bit(v)) return;
        --num_active_;
        adj_.clear(v);
    }

    /// Directed-inline vec_pool: account for a bulk set of undirected edges
    /// whose two incidences are written directly, without an edge-pool stage.
    void record_edges_added(edge_index count) {
        m_ += count;
    }

    /// Directed-inline vec_pool: materialize one endpoint incidence directly
    /// into a slab already sized by adj_bulk_reserve_parallel().
    void adj_atomic_push_directed(node_index from, node_index to, double w) {
        adj_.atomic_push_reserved(
            from, detail::directed_pool_edge{to, static_cast<detail::pool_value_t>(w)});
    }

    /// Directed-inline vec_pool: write an incidence at a caller-assigned
    /// offset in the pre-reserved suffix without an atomic slot claim.
    void adj_write_reserved_directed_at(node_index from, node_index offset,
                                        node_index to, double w) {
        adj_.write_reserved_at(
            from, offset,
            detail::directed_pool_edge{to, static_cast<detail::pool_value_t>(w)});
    }

    /// Publish all preassigned suffix slots for one vec_pool vertex.
    void adj_commit_reserved_directed(node_index v, node_index count) {
        adj_.commit_reserved(v, count);
    }

    /// pre-reserve cap_[v] >= need before a parallel
    /// adj_atomic_push_reserved phase.  NOT thread-safe.
    void adj_reserve_for(node_index v, node_index need) {
        adj_.reserve_for(v, need);
    }

    /// serial compaction decision for the imminent parallel
    /// reserve. The caller provides the surrounding OpenMP single/barrier.
    void adj_compact_for_round_if_needed(
            const std::vector<node_index>& incoming) {
        adj_.compact_for_round_if_needed(incoming);
    }

    /// bulk parallel reserve_for. Must be called from inside an
    /// OpenMP parallel region after the caller's shared compaction decision;
    /// the caller synchronizes the team before publishing the reserved slabs.
    template<typename TouchedIter>
    void adj_bulk_reserve_parallel(TouchedIter begin, TouchedIter end,
                                   const std::vector<node_index>& incoming) {
        adj_.bulk_reserve_parallel(begin, end, incoming);
    }

    /// current adjacency count for vertex v (used for
    /// reserve_for pre-pass).
    node_index adj_count(node_index v) const {
        return static_cast<node_index>(adj_[v].size());
    }

    /// Sort vertex v's records by (target, weight), restoring deterministic
    /// order after a parallel atomic-push phase.
    void adj_sort_slab(node_index v) {
        adj_.sort_slab(v);
    }

    /// live fraction (1 - abandoned/pool_size).
    double adj_live_fraction() const { return adj_.live_fraction(); }

    /// pool diagnostics (grow/compact counts, worst fragmentation).
    size_t adj_grow_count() const { return adj_.grow_count(); }
    size_t adj_compact_count() const { return adj_.compact_count(); }
    double adj_min_live_fraction() const { return adj_.min_live_fraction(); }
    void adj_note_live_fraction() { adj_.note_live_fraction(); }

    /// Atomically add `delta` to excess[v]; safe under data races on v.
    void atomic_add_excess(node_index v, double delta) {
        #pragma omp atomic
        excess_[v] += delta;
    }

    /// Mark vertex inactive without decrementing num_active_; the
    /// caller must invoke `bulk_decrement_active(k)` once after the
    /// parallel batch. The vertex's adjacency list is cleared.
    void set_inactive_unchecked(node_index v) {
        const bool was_active = clear_active_bit(v);
        assert(was_active);
        (void)was_active;
        adj_.clear(v);
    }

    /// Apply a bulk decrement to `num_active_` after a parallel batch
    /// of `set_inactive_unchecked` calls.
    void bulk_decrement_active(node_index k) { num_active_ -= k; }

    /// Prune dead edges and (optionally) visit each survivor.  Returns
    /// the surviving (active) degree count.  The default no-op visitor
    /// makes this serve as a pure degree count too.
    template<typename Visitor>
    node_index prune_and_visit(node_index v, Visitor&& visit) {
        node_index count = 0;
        adj_.filter(v, [&](const auto& idx) {
            auto u = idx.to;
            if (!is_active(u)) return false;
            visit(u);
            ++count;
            return true;
        });
        return count;
    }

    /// Prune + count, no visitor.
    node_index prune_and_degree(node_index v) {
        return prune_and_visit(v, [](node_index) {});
    }

    // ── active-aware queries ──

    bool is_active(node_index v) const {
        const active_word word = __atomic_load_n(
            &active_[static_cast<size_t>(v) / kActiveWordBits],
            __ATOMIC_RELAXED);
        return (word & active_mask(v)) != 0;
    }
    node_index num_active() const { return num_active_; }

    /// Raw access to the adjacency records (for tight inner loops).
    auto adj(node_index v) const { return adj_[v]; }

    /// Per-vertex excess diagonal (for SDDM matrices).
    /// For a pure Laplacian this is zero everywhere.
    /// For SDDM,  excess[v] = M[v,v] - sum of incident edge weights.
    double  excess(node_index v) const { return excess_[v]; }
    double& excess(node_index v)       { return excess_[v]; }

    /// Approximate heap memory usage in bytes.
    std::size_t memory_bytes() const {
        return adj_.memory_bytes()
             + active_.capacity() * sizeof(active_word)
             + excess_.capacity() * sizeof(double);
    }

private:
    friend struct detail::residual_coalescer;

    struct coalesce_stats {
        std::size_t multi_edges = 0;
        std::size_t distinct_edges = 0;
        std::size_t bytes_before = 0;
        std::size_t bytes_after = 0;
    };

    struct sparsify_stats {
        std::size_t physical_before = 0;
        std::size_t distinct_before = 0;
        std::size_t kept_edges = 0;
        std::size_t backbone_edges = 0;
        std::size_t bytes_before = 0;
        std::size_t bytes_after = 0;
        std::size_t forest_candidates = 0;
        std::size_t forest_shards = 1;
        std::size_t rebuild_threads = 1;
        double coalesce_ms = 0.0;
        double collect_ms = 0.0;
        double order_ms = 0.0;
        double forest_ms = 0.0;
        double rebuild_ms = 0.0;
        double total_weight = 0.0;
        double backbone_weight = 0.0;
        double expected_kept_edges = 0.0;
        double min_offtree_probability = 1.0;
        double max_inverse_probability = 1.0;
    };

    struct active_sparsify_sample {
        double duplicate_ratio = 1.0;
        double strongest_weight_share = 0.0;
        double avg_distinct_degree = 0.0;
        std::size_t sampled_vertices = 0;
    };

    active_sparsify_sample estimate_active_sparsify_sample(
        std::span<const node_index> active,
        std::size_t sample_vertices = 256) const {
        const std::size_t samples = std::min(sample_vertices, active.size());
        if (samples == 0) return {};
        unsigned long long multi_total = 0;
        unsigned long long distinct_total = 0;
        // long double sums go to per-thread slots summed in thread order: an
        // OpenMP reduction on them needs 16-byte atomics (libatomic).
        std::vector<std::array<long double, 2>> totals;
        detail::parallel_exception error;
        #pragma omp parallel reduction(+:multi_total,distinct_total)
        {
            #pragma omp single
            totals.assign(static_cast<std::size_t>(omp_get_num_threads()), {});
            long double& weight_total = totals[static_cast<std::size_t>(omp_get_thread_num())][0];
            long double& strongest_total = totals[static_cast<std::size_t>(omp_get_thread_num())][1];
            std::vector<std::pair<node_index, double>> neighbors;
            #pragma omp for schedule(static)
            for (std::size_t s = 0; s < samples; ++s) error.run([&] {
                const node_index u = active[s * active.size() / samples];
                neighbors.clear();
                for (const auto& entry : adj_[u]) {
                    const node_index v = entry.to;
                    if (!is_active(v)) continue;
                    neighbors.emplace_back(
                        v, static_cast<double>(entry.w));
                    ++multi_total;
                }
                std::ranges::sort(neighbors, {},
                    [](const auto& pair) { return pair.first; });
                double local_total = 0.0;
                double local_strongest = 0.0;
                for (std::size_t i = 0; i < neighbors.size();) {
                    const node_index v = neighbors[i].first;
                    double weight = 0.0;
                    do {
                        weight += neighbors[i].second;
                        ++i;
                    } while (i < neighbors.size() && neighbors[i].first == v);
                    ++distinct_total;
                    local_total += weight;
                    local_strongest = std::max(local_strongest, weight);
                }
                weight_total += local_total;
                strongest_total += local_strongest;
            });
        }
        error.rethrow();
        long double weight_total = 0.0L, strongest_total = 0.0L;
        for (const auto& [weight, strongest] : totals) {
            weight_total += weight;
            strongest_total += strongest;
        }
        active_sparsify_sample result;
        result.sampled_vertices = samples;
        result.duplicate_ratio = distinct_total
            ? static_cast<double>(multi_total) / distinct_total : 1.0;
        result.avg_distinct_degree =
            static_cast<double>(distinct_total) / samples;
        result.strongest_weight_share = weight_total > 0.0L
            ? static_cast<double>(strongest_total / weight_total) : 0.0;
        return result;
    }

    /// Connectivity-safe late-residual sparsification for directed AoS.
    /// Parallel edges are combined first and a spanning forest is retained
    /// exactly. Off-tree weights use Bernoulli/Horvitz--Thompson estimates;
    /// degrees count distinct neighbours.
    sparsify_stats sparsify_active(std::span<const node_index> active,
                                   double keep_probability,
                                   std::uint64_t seed) {
        using clock = std::chrono::steady_clock;
        auto stage = clock::now();
        keep_probability = std::clamp(keep_probability, 1e-6, 1.0);
        const coalesce_stats coalesced = coalesce_active(active);
        const double coalesce_ms = std::chrono::duration<double, std::milli>(
            clock::now() - stage).count();
        stage = clock::now();
        struct item {
            node_index u, v;
            double weight;
            node_index multiplicity;
            bool backbone = false;
            bool kept = false;
        };
        std::vector<item> items;
        items.reserve(coalesced.distinct_edges);
        for (node_index u : active)
            for (const auto& entry : adj_[u]) {
                const node_index v = entry.to;
                if (v > u && is_active(v))
                    items.push_back({u, v, static_cast<double>(entry.w),
                                     1, false, false});
            }
        const double collect_ms = std::chrono::duration<double, std::milli>(
            clock::now() - stage).count();
        stage = clock::now();

        const node_index npos = node_index(-1);
        std::vector<node_index> parent(static_cast<std::size_t>(n_), npos);
        std::vector<unsigned char> rank(static_cast<std::size_t>(n_), 0);
        for (node_index v : active) parent[v] = v;
        auto find = [&](node_index v) {
            node_index root = v;
            while (parent[root] != root) root = parent[root];
            while (parent[v] != v) {
                const node_index next = parent[v];
                parent[v] = root;
                v = next;
            }
            return root;
        };
        auto unite = [&](node_index u, node_index v) {
            u = find(u);
            v = find(v);
            if (u == v) return false;
            if (rank[u] < rank[v]) std::swap(u, v);
            parent[v] = u;
            if (rank[u] == rank[v]) ++rank[u];
            return true;
        };
        std::vector<std::size_t> order(items.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::vector<std::size_t> scratch(order.size());
        // Heaviest first by the inverted exponent and top three mantissa
        // bits of the FP32 weight, as in residual_sparsify::bucket.
        auto bucket = [&](std::size_t index) -> std::size_t {
            return ~std::bit_cast<std::uint32_t>(
                static_cast<float>(items[index].weight)) >> 20;
        };
        std::array<std::size_t, std::size_t{1} << 12> counts{};
        for (std::size_t index : order) ++counts[bucket(index)];
        std::size_t offset = 0;
        for (std::size_t& count : counts) {
            const std::size_t next = offset + count;
            count = offset;
            offset = next;
        }
        for (std::size_t index : order) scratch[counts[bucket(index)]++] = index;
        order.swap(scratch);
        // The radix scratch now contains only the discarded iota order.
        // Release it before allocating an equally wide double cache so the
        // optimization does not add another O(m) live allocation.
        std::vector<std::size_t>().swap(scratch);
        const double order_ms = std::chrono::duration<double, std::milli>(
            clock::now() - stage).count();
        stage = clock::now();
        std::vector<std::size_t> filtered_order;
        const std::vector<std::size_t>* forest_order = &order;
        std::size_t forest_shards = 1;
        if (!active.empty() && !items.empty()) {
            std::size_t available_threads = 1;
#ifdef _OPENMP
            available_threads = static_cast<std::size_t>(
                std::max(1, omp_get_max_threads()));
#endif
            const long double edge_density =
                static_cast<long double>(items.size()) /
                static_cast<long double>(active.size());
            forest_shards = std::min<std::size_t>(
                available_threads,
                std::max<std::size_t>(1, static_cast<std::size_t>(
                    std::floor(std::sqrt(edge_density)))));
            if (forest_shards > 1) {
                // Exact Kruskal filtering. Each shard is a contiguous slice of
                // the already ordered edge stream. If its local forest drops
                // edge e, earlier edges in the same shard already connect e's
                // endpoints; the global ordered scan would therefore drop e
                // as well. Concatenating shard forests in shard order retains
                // every edge that can affect the original global decisions.
                const node_index dense_npos = node_index(-1);
                std::vector<node_index> dense_of(
                    static_cast<std::size_t>(n_), dense_npos);
#pragma omp parallel for schedule(static)
                for (std::size_t i = 0; i < active.size(); ++i)
                    dense_of[active[i]] = static_cast<node_index>(i);

                std::vector<std::vector<std::size_t>> local_forests(
                    forest_shards);
                detail::parallel_exception forest_error;
#pragma omp parallel for num_threads(forest_shards) schedule(static, 1)
                for (std::size_t shard = 0; shard < forest_shards; ++shard) forest_error.run([&] {
                    std::vector<node_index> local_parent(active.size());
                    std::iota(local_parent.begin(), local_parent.end(),
                              node_index{0});
                    std::vector<unsigned char> local_rank(active.size(), 0);
                    auto local_find = [&](node_index v) {
                        node_index root = v;
                        while (local_parent[root] != root)
                            root = local_parent[root];
                        while (local_parent[v] != v) {
                            const node_index next = local_parent[v];
                            local_parent[v] = root;
                            v = next;
                        }
                        return root;
                    };
                    auto local_unite = [&](node_index u, node_index v) {
                        u = local_find(u);
                        v = local_find(v);
                        if (u == v) return false;
                        if (local_rank[u] < local_rank[v]) std::swap(u, v);
                        local_parent[v] = u;
                        if (local_rank[u] == local_rank[v]) ++local_rank[u];
                        return true;
                    };
                    const std::size_t begin =
                        order.size() * shard / forest_shards;
                    const std::size_t end =
                        order.size() * (shard + 1) / forest_shards;
                    auto& local = local_forests[shard];
                    local.reserve(std::min(active.size(), end - begin));
                    for (std::size_t position = begin; position < end;
                         ++position) {
                        const std::size_t index = order[position];
                        const item& edge = items[index];
                        const node_index u = dense_of[edge.u];
                        const node_index v = dense_of[edge.v];
                        assert(u != dense_npos && v != dense_npos);
                        if (local_unite(u, v)) local.push_back(index);
                    }
                });
                forest_error.rethrow();
                std::size_t candidates = 0;
                for (const auto& local : local_forests)
                    candidates += local.size();
                filtered_order.reserve(candidates);
                for (auto& local : local_forests)
                    filtered_order.insert(filtered_order.end(),
                                          local.begin(), local.end());
                forest_order = &filtered_order;
            }
        }
        std::size_t backbone_edges = 0;
        double total_weight = 0.0;
        double backbone_weight = 0.0;
        for (const item& edge : items) total_weight += edge.weight;
        for (std::size_t index : *forest_order)
            if (unite(items[index].u, items[index].v)) {
                items[index].backbone = true;
                ++backbone_edges;
                backbone_weight += items[index].weight;
            }
        const double forest_ms = std::chrono::duration<double, std::milli>(
            clock::now() - stage).count();
        stage = clock::now();
        std::vector<double> importance(items.size());
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < items.size(); ++i)
            importance[i] = std::sqrt(items[i].weight);
        auto importance_measure = [&](std::size_t index) {
            return importance[index];
        };
        const std::size_t off_tree = items.size() - backbone_edges;
        const double target = keep_probability * off_tree;
        const auto normalization = detail::normalize_residual_importance(
            std::span<const item>(items), std::span<const double>(importance),
            target);
        const double importance_scale = normalization.scale;
        auto probability = [&](std::size_t index) {
            if (items[index].backbone) return 1.0;
            return std::min(
                1.0, importance_scale * importance_measure(index));
        };
        double expected_kept = 0.0;
        double min_offtree_probability = 1.0;
        for (std::size_t i = 0; i < items.size(); ++i) {
            const item& edge = items[i];
            const double edge_probability = probability(i);
            expected_kept += edge_probability;
            if (!edge.backbone)
                min_offtree_probability =
                    std::min(min_offtree_probability, edge_probability);
        }
        auto draw = [&](node_index u, node_index v, std::uint64_t salt = 0) {
            const std::uint64_t z = detail::splitmix64_mix(seed ^ salt ^
                (static_cast<std::uint64_t>(u) * 0x9E3779B97F4A7C15ULL) ^
                (static_cast<std::uint64_t>(v) * 0xBF58476D1CE4E5B9ULL));
            return static_cast<double>(z >> 11) * 0x1.0p-53;
        };

        graph rebuilt(n_);
        rebuilt.active_ = active_;
        rebuilt.num_active_ = num_active_;
        rebuilt.excess_ = excess_;
        std::vector<node_index> physical_degree(static_cast<std::size_t>(n_), 0);
        std::size_t kept = 0;
        bool parallel_directed_rebuild = true;
        std::size_t rebuild_threads = 1;
        std::vector<node_index> dense_of;
        std::vector<node_index> thread_offsets;
        if (parallel_directed_rebuild && !active.empty() && !items.empty()) {
            std::size_t available_threads = 1;
#ifdef _OPENMP
            available_threads = static_cast<std::size_t>(
                std::max(1, omp_get_max_threads()));
#endif
            const long double edge_density =
                static_cast<long double>(items.size()) /
                static_cast<long double>(active.size());
            rebuild_threads = std::min<std::size_t>(
                available_threads,
                std::max<std::size_t>(1, static_cast<std::size_t>(
                    std::floor(std::sqrt(edge_density)))));
            if (active.size() >
                std::numeric_limits<std::size_t>::max() / rebuild_threads)
                rebuild_threads = 1;
            parallel_directed_rebuild = rebuild_threads > 1;
        } else {
            parallel_directed_rebuild = false;
        }
        if (!parallel_directed_rebuild) {
            for (std::size_t i = 0; i < items.size(); ++i) {
                item& edge = items[i];
                edge.kept = draw(edge.u, edge.v) < probability(i);
                if (edge.kept) {
                    ++physical_degree[edge.u];
                    ++physical_degree[edge.v];
                    ++kept;
                }
            }
        } else {
            // Give each worker a contiguous edge interval and a private degree
            // row over the active vertices. Prefixing those rows assigns a
            // disjoint destination interval to every (worker, vertex) pair.
            // The second scan can therefore write without endpoint atomics;
            // worker intervals are in global item order, so every adjacency
            // slab is already in the serial canonical order and needs no sort.
            const node_index dense_npos = node_index(-1);
            dense_of.assign(static_cast<std::size_t>(n_), dense_npos);
            thread_offsets.assign(rebuild_threads * active.size(), 0);
            std::vector<std::size_t> kept_by_thread(rebuild_threads, 0);
            std::size_t actual_threads = 1;
            detail::parallel_exception rebuild_error;
#pragma omp parallel num_threads(rebuild_threads)
            {
#ifdef _OPENMP
                const std::size_t tid = static_cast<std::size_t>(
                    omp_get_thread_num());
                const std::size_t team = static_cast<std::size_t>(
                    omp_get_num_threads());
#else
                const std::size_t tid = 0;
                const std::size_t team = 1;
#endif
#pragma omp single
                actual_threads = team;
#pragma omp for schedule(static)
                for (std::size_t i = 0; i < active.size(); ++i)
                    dense_of[active[i]] = static_cast<node_index>(i);

                node_index* local =
                    thread_offsets.data() + tid * active.size();
                const std::size_t begin = items.size() * tid / team;
                const std::size_t end = items.size() * (tid + 1) / team;
                std::size_t local_kept = 0;
                for (std::size_t i = begin; i < end; ++i) {
                    item& edge = items[i];
                    edge.kept = draw(edge.u, edge.v) < probability(i);
                    if (!edge.kept) continue;
                    const node_index u = dense_of[edge.u];
                    const node_index v = dense_of[edge.v];
                    assert(u != dense_npos && v != dense_npos);
                    ++local[u];
                    ++local[v];
                    ++local_kept;
                }
                kept_by_thread[tid] = local_kept;
#pragma omp barrier
#pragma omp single
                kept = std::accumulate(
                    kept_by_thread.begin(),
                    kept_by_thread.begin() +
                        static_cast<std::ptrdiff_t>(team), std::size_t{0});
#pragma omp for schedule(static)
                for (std::size_t i = 0; i < active.size(); ++i) {
                    node_index offset = 0;
                    for (std::size_t t = 0; t < team; ++t) {
                        node_index& count =
                            thread_offsets[t * active.size() + i];
                        const node_index next = offset + count;
                        count = offset;
                        offset = next;
                    }
                    physical_degree[active[i]] = offset;
                }

                rebuild_error.run([&] {
                    rebuilt.adj_.bulk_reserve_parallel(
                        active.begin(), active.end(), physical_degree);
                });
#pragma omp barrier
                if (!rebuild_error.failed()) {
                    for (std::size_t i = begin; i < end; ++i) {
                        const item& edge = items[i];
                        if (!edge.kept) continue;
                        const double inv_p = 1.0 / probability(i);
                        const detail::pool_value_t weight = static_cast<detail::pool_value_t>(
                            edge.weight * inv_p);
                        const node_index u = dense_of[edge.u];
                        const node_index v = dense_of[edge.v];
                        {
                            rebuilt.adj_.write_reserved_at(
                                edge.u, local[u]++, {edge.v, weight});
                            rebuilt.adj_.write_reserved_at(
                                edge.v, local[v]++, {edge.u, weight});
                        }
                    }
#pragma omp barrier
#pragma omp for schedule(static)
                    for (std::size_t i = 0; i < active.size(); ++i) {
                        const node_index v = active[i];
                        rebuilt.adj_.commit_reserved(v, physical_degree[v]);
                    }
                }
            }
            rebuild_error.rethrow();
            rebuild_threads = actual_threads;
        }
        {
            if (parallel_directed_rebuild) {
                if (kept > std::numeric_limits<edge_index>::max())
                    edge_index_overflow("residual sparsifier rebuild");
                rebuilt.m_ = static_cast<edge_index>(kept);
            } else {
                for (node_index v : active)
                    rebuilt.adj_.reserve_for(v, physical_degree[v]);
                for (std::size_t i = 0; i < items.size(); ++i) {
                    const item& edge = items[i];
                    if (!edge.kept) continue;
                    const double inv_p = 1.0 / probability(i);
                    const double weight = edge.weight * inv_p;
                    rebuilt.adj_.push(
                        edge.u, {edge.v, static_cast<detail::pool_value_t>(weight)});
                    rebuilt.adj_.push(
                        edge.v, {edge.u, static_cast<detail::pool_value_t>(weight)});
                    ++rebuilt.m_;
                }
            }
        }
        sparsify_stats stats;
        stats.physical_before = coalesced.multi_edges;
        stats.distinct_before = coalesced.distinct_edges;
        stats.kept_edges = kept;
        stats.backbone_edges = backbone_edges;
        stats.bytes_before = coalesced.bytes_before;
        stats.bytes_after = rebuilt.memory_bytes();
        stats.forest_candidates = forest_order->size();
        stats.forest_shards = forest_shards;
        stats.rebuild_threads = rebuild_threads;
        stats.coalesce_ms = coalesce_ms;
        stats.collect_ms = collect_ms;
        stats.order_ms = order_ms;
        stats.forest_ms = forest_ms;
        stats.rebuild_ms = std::chrono::duration<double, std::milli>(
            clock::now() - stage).count();
        stats.total_weight = total_weight;
        stats.backbone_weight = backbone_weight;
        stats.expected_kept_edges = expected_kept;
        stats.min_offtree_probability = min_offtree_probability;
        stats.max_inverse_probability = 1.0 / min_offtree_probability;
        *this = std::move(rebuilt);
        return stats;
    }

    coalesce_stats coalesce_active(std::span<const node_index> active) {
        unsigned long long before_incidence = 0;
        unsigned long long after_incidence = 0;
        const std::size_t bytes_before = memory_bytes();
        // coalesce_slab sorts the vertex's slab in place, so a vertex costs
        // d*log2(d) and the equal-count split is nowhere near equal work on a
        // hub residual: this loop scaled only 3.0x on 16 threads (as-Skitter
        // handoff, T=1 555 ms -> T=16 183 ms). The slab is contiguous, so the
        // body is compute-bound on cached data -- the regime where balancing
        // pays. Bounds are a pure function of (degrees, nt), so a fixed thread
        // count keeps one fixed partition exactly as schedule(static) did, and
        // each slab is coalesced independently with integer-only reductions,
        // so no split can change a stored byte.
        std::vector<std::uint64_t> work_prefix(active.size() + 1, 0);
        for (std::size_t k = 0; k < active.size(); ++k) {
            const std::uint64_t d = adj_[active[k]].size();
            work_prefix[k + 1] = work_prefix[k] +
                d * static_cast<std::uint64_t>(std::bit_width(d + 1));
        }
        #pragma omp parallel reduction(+:before_incidence,after_incidence)
        {
        const auto [k_lo, k_hi] = detail::work_balanced_range(
            work_prefix.data(), active.size());
        for (std::size_t k = k_lo; k < k_hi; ++k) {
            const node_index u = active[k];
            const auto [before, after] = adj_.coalesce_slab(
                u, [&](node_index v) { return is_active(v); });
            before_incidence += before;
            after_incidence += after;
        }
        }
        adj_.compact();
        coalesce_stats stats;
        stats.multi_edges = static_cast<std::size_t>(before_incidence / 2);
        stats.distinct_edges = static_cast<std::size_t>(after_incidence / 2);
        stats.bytes_before = bytes_before;
        stats.bytes_after = memory_bytes();
        return stats;
    }


private:
    using active_word = std::uint64_t;
    static constexpr size_t kActiveWordBits = 8 * sizeof(active_word);

    static constexpr size_t active_word_count(node_index n) {
        return (static_cast<size_t>(n) + kActiveWordBits - 1) /
               kActiveWordBits;
    }

    static constexpr active_word active_mask(node_index v) {
        return active_word{1} <<
               (static_cast<size_t>(v) % kActiveWordBits);
    }

    bool clear_active_bit(node_index v) {
        const active_word mask = active_mask(v);
        const active_word old = __atomic_fetch_and(
            &active_[static_cast<size_t>(v) / kActiveWordBits], ~mask,
            __ATOMIC_RELAXED);
        return (old & mask) != 0;
    }

    node_index n_ = 0;          // vertex count
    edge_index m_ = 0;          // edge count (can exceed 2^31)
    node_index num_active_ = 0; // active vertex count
    detail::directed_vec_pool_incidence adj_;
    // Active status is probed randomly in nearly every residual-edge scan, so
    // keep one bit per vertex rather than one byte.  Parallel elimination can
    // clear distinct vertices that share a word; all reads and clears therefore
    // use relaxed atomic operations.  Round barriers provide the phase ordering
    // needed by later graph traversals.
    std::vector<active_word> active_;
    std::vector<double> excess_;
};

namespace detail {

/// Internal access seam for the one-shot late-residual rebuild. Keeping these
/// operations here avoids growing graph's public API for one factorizer policy.
struct residual_coalescer {
    static auto sample(const graph& g,
                       std::span<const node_index> active,
                       std::size_t sample_vertices = 256) {
        return g.estimate_active_sparsify_sample(active, sample_vertices);
    }

    static auto sparsify(graph& g,
                         std::span<const node_index> active,
                         double keep_probability,
                         std::uint64_t seed) {
        return g.sparsify_active(active, keep_probability, seed);
    }
};

} // namespace detail

} // namespace apxchol
