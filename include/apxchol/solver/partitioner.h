#pragma once
/// Partitioner: selects the set of vertices eliminated in one round — the
/// customization seam for independent-set selection AND elimination order.
/// The global elimination order (hence the permutation P) is the order in
/// which vertices are selected across rounds; within a round the selected
/// set is eliminated in parallel, so it MUST be an independent set (debug
/// builds verify this after every round).
///
/// ── The interface (one shape for every partitioner) ──
///
///   struct my_partitioner {
///       static constexpr bool degree_prepass = true;      // optional trait
///
///       void find_partition(const partition_graph& G,
///                           std::span<const node_index> candidates,
///                           const partition_context& ctx, selection& out);
///   };
///
/// The `degree_prepass` trait (default false) decides what the orchestrator
/// does before the call:
///
///   * true — the orchestrator supplies degree counts and applies the cap
///     (ctx.options.degree_quantile / degree_multiplier): `candidates` is
///     the eligible vertex list. ctx.degrees[v] is exact after pruning or a
///     cache update, and may be conservative during audited prune skipping.
///     The scan rules (block_greedy, priority_greedy) use this.
///   * false — no O(active + edges) pre-pass runs: `candidates` is the raw
///     active list (ascending original-index order) and ctx.degrees is
///     empty. baumann_kyng works this way — it estimates degrees from its
///     hash sample and applies the cap to that estimate.
///
/// Degrees are indexed by vertex ID so rules can compare neighboring vertices.
/// Incidence-list sizes may include eliminated neighbors. The orchestrator
/// supplies the span through pruning, block_greedy's exact incremental cache,
/// or audited prune skipping, which may retain conservative raw counts.
///
/// `out` records the round's selection and answers membership:
///   out.add(v)       — select v          (thread-safe for distinct v)
///   out.remove(v)    — unselect v        (thread-safe for distinct v; exists
///                       for greedy rules that pick optimistically and drop
///                       conflicting picks in a later resolution pass — see
///                       block_greedy)
///   out.contains(u)  — is u selected?    (a plain, unsynchronized read of the
///                       shared mask: it is YOUR job to call it only where no
///                       other thread can be writing mask[u] concurrently —
///                       see the determinism contract below)
///
/// Threading contract: find_partition is called ONCE per round from the
/// orchestrating thread and is expected to parallelize internally (gate on
/// ctx.omp_threshold). This is the opposite of the eliminator seam, where
/// sample_clique is called concurrently and each call must stay
/// single-threaded. The parallelism cannot be hoisted into the orchestrator:
/// the shipped rules parallelize over different structures (contiguous
/// candidate blocks, repeated whole-set passes, or a hash sample),
/// so there is no common parallel skeleton to factor out.
///
/// ── Determinism contract (REQUIRED of every partitioner) ──
///
/// A partitioner's output — the selected set AND the order it is added in —
/// must be a pure function of (graph, candidates, ctx, team size). At a fixed
/// seed and a fixed thread count, two runs must produce the same
/// partition_result, byte for byte. It may (and does) differ ACROSS thread
/// counts: the team size decides how the work is split, and that is an input.
///
/// This is not a nicety. The round's selection is the elimination order, the
/// elimination order is the permutation, and the permutation is the factor's
/// stored structure — so a schedule-dependent selection makes nnz(L) itself
/// wobble run to run. block_greedy once violated this contract by resolving
/// cross-block conflicts against a mask other threads were concurrently
/// clearing. The guard is FactorizeDeterminism.* in tests/test_factorize.cpp.
///
/// The two patterns to avoid, concretely:
///   * reading `out.contains(u)` (or any shared flag) in a pass that also
///     writes it — decide against a snapshot, barrier, then apply;
///   * letting a `schedule(dynamic)` / `nowait` loop decide WHERE a vertex
///     lands in a per-thread buffer that is later concatenated — sort the
///     result, or partition the work statically.
///
/// `G` is a read-only view of the residual graph (see partition_graph).
///
/// Use a custom partitioner as a configured instance:
/// factorize(A, my_partitioner{...}, opts), or
/// factorize(A, my_partitioner{...}, my_eliminator{}, opts).
/// Lambdas: as_partitioner(fn) or as_prepass_partitioner(fn) — identical
/// callable signature, differing only in the trait they declare;
/// zero-overhead template wrappers, no type erasure.
///
/// Optional traits (queried via `if constexpr (requires { T::X; })`):
///   T::degree_prepass             — see above (default false).
///   T::sample_bounded             — if true, the main loop skips the
///                                   candidate-relative min_is_fraction bailout
///                                   and retries
///                                   empty rounds (bounded), for rules whose
///                                   IS size does not scale with |active|.
///   T::residual_handoff_threshold — when the main loop bails with MORE than
///                                   this many active vertices, parallel BK
///                                   rounds shrink the residual to the
///                                   threshold before the serial peel.

#include "apxchol/graph/graph.h"
#include "apxchol/solver/partitioner_helpers.h"
#include "apxchol/solver/factor_options.h"
#include "apxchol/solver/partition.h"
#include <limits>
#include <type_traits>
#include <span>
#include <utility>
#include <vector>

#include <omp.h>

namespace apxchol {

namespace detail {
struct selection_access;
struct partition_graph_access;
}

/// Read-only view of the residual graph during selection. Vertex IDs are the
/// operator's row indices; eliminated vertices stay in the ID range.
class partition_graph {
public:
    /// Vertex count, including eliminated vertices.
    node_index n() const { return g_->n(); }
    /// True while v has not been eliminated.
    bool is_active(node_index v) const { return g_->is_active(v); }
    /// Calls f(u) for each live neighbor u of v, repeating u once per
    /// parallel edge, and returns that count. Safe concurrently for distinct v.
    template<typename F>
    node_index for_each_neighbor(node_index v, F&& f) const {
        return g_->prune_and_visit(v, std::forward<F>(f));
    }
    /// Live degree of v: the count for_each_neighbor returns.
    node_index degree(node_index v) const { return g_->prune_and_degree(v); }

private:
    friend struct detail::partition_graph_access;
    explicit partition_graph(detail::graph& g) : g_(&g) {}
    // Visiting drops incidences of eliminated vertices from v's list, which
    // leaves the represented graph unchanged.
    detail::graph* g_;
};

/// Read-only services the orchestrator hands to a partitioner call.
/// Everything except `degrees` is constant for the whole factorize() run.
struct partition_context {
    /// The user's selection preferences (factor_options::partition), always
    /// populated. Who acts on the cap knobs (degree_quantile /
    /// degree_multiplier) depends on the degree_prepass trait: with the
    /// pre-pass the orchestrator has already applied them when building the
    /// candidate list; without it nothing has been applied and the rule
    /// honors them itself.
    partition_options options;
    /// Determinism source. Derive all randomness as pure functions of
    /// (seed, round, vertex) — hash priorities as the shipped rules do, or
    /// use it to seed a generator you own. A seed rather than an RNG object
    /// because draws from a shared stream would depend on thread scheduling.
    unsigned seed = 42;
    std::size_t omp_threshold = 2000;  ///< min work size before OpenMP engages
    /// Prefer the lower-degree vertex when adjacent candidates compete
    /// (factor_options::experimental.degree_tiebreak). Never applied by the
    /// orchestrator — honoring it is always the rule's job.
    bool degree_tiebreak = true;
    /// Degree counts indexed by vertex ID, valid for every active vertex.
    /// Audited prune skipping may supply conservative counts; otherwise they
    /// are exact. Non-empty when this partitioner declares degree_prepass.
    std::span<const node_index> degrees;
};

/// The round's selection: an insert/remove/membership structure over
/// vertices, owned by the orchestrator and reused across rounds.
/// add/remove are thread-safe for distinct vertices (per-thread insertion
/// lists + a byte mask); contains() is a plain read. remove() only clears
/// the membership mask — the stale insertion-list entry is dropped when the
/// orchestrator collects the round, and add() after remove() re-inserts
/// correctly.
class selection {
public:
    void add(node_index v) {
        if (!mask_[v]) {
            mask_[v] = 1;
            per_thread_[omp_get_thread_num()].push_back(v);
        }
    }
    void remove(node_index v) { mask_[v] = 0; }
    bool contains(node_index v) const { return mask_[v] != 0; }

private:
    friend struct detail::selection_access;

    void reset(node_index n) {
        if (mask_.size() < static_cast<size_t>(n)) mask_.assign(n, 0);
        const int nt = omp_get_max_threads();
        if (per_thread_.size() < static_cast<size_t>(nt)) per_thread_.resize(nt);
        for (auto& l : per_thread_) l.clear();
        result_.data.clear();
    }
    /// Concatenate the per-thread lists, drop removed entries, reset the
    /// mask. O(inserted), not O(n).
    const detail::partition_result& finalize() {
        for (auto& l : per_thread_)
            for (auto v : l)
                if (mask_[v]) {
                    result_.data.push_back(v);
                    mask_[v] = 0;
                }
        return result_;
    }

    std::vector<char> mask_;
    std::vector<std::vector<node_index>> per_thread_;
    detail::partition_result result_;
};

namespace detail {

/// Orchestrator side of `selection`: start a round and collect its result.
struct selection_access {
    static void reset(selection& s, node_index n) { s.reset(n); }
    static const partition_result& finalize(selection& s) { return s.finalize(); }
};

/// Orchestrator and built-in side of `partition_graph`.
struct partition_graph_access {
    static partition_graph view(graph& g) { return partition_graph(g); }
    static graph& graph_of(const partition_graph& G) { return *G.g_; }
};

// ── Trait queries (with defaults) ──

template<typename P>
inline constexpr bool partitioner_degree_prepass_v = [] {
    if constexpr (requires { P::degree_prepass; })
        return P::degree_prepass;
    else
        return false;
}();

template<typename P>
inline constexpr bool partitioner_sample_bounded_v = [] {
    if constexpr (requires { P::sample_bounded; })
        return P::sample_bounded;
    else
        return false;
}();

template<typename P>
inline constexpr size_t partitioner_residual_handoff_v = [] {
    if constexpr (requires { P::residual_handoff_threshold; })
        return P::residual_handoff_threshold;
    else
        return std::numeric_limits<size_t>::max();
}();

} // namespace detail

template<typename T>
concept partitioner =
    requires(T t, const partition_graph& G,
             std::span<const node_index> candidates,
             const partition_context& ctx, selection& out) {
        t.find_partition(G, candidates, ctx, out);
    };

/// Adapt a callable into a partitioner. The two adapters differ only in the
/// degree_prepass trait they declare (a callable cannot carry one itself):
/// as_partitioner — raw active list, as_prepass_partitioner — pre-pass runs
/// and ctx.degrees is filled. The callable signature is the same for both.
template<typename F, bool Prepass>
struct callable_partitioner {
    static constexpr bool degree_prepass = Prepass;

    explicit callable_partitioner(F f) : fn(std::move(f)) {}
    F fn;

    void find_partition(const partition_graph& G,
                        std::span<const node_index> candidates,
                        const partition_context& ctx, selection& out) {
        fn(G, candidates, ctx, out);
    }
};

template<typename F>
callable_partitioner<std::decay_t<F>, false> as_partitioner(F&& fn) {
    return callable_partitioner<std::decay_t<F>, false>(std::forward<F>(fn));
}

template<typename F>
callable_partitioner<std::decay_t<F>, true> as_prepass_partitioner(F&& fn) {
    return callable_partitioner<std::decay_t<F>, true>(std::forward<F>(fn));
}

namespace detail {

/// Per-round seed and vertex priority hash shared by the built-in hash-priority
/// partitioners (priority_greedy, baumann_kyng).
inline std::uint64_t partition_round_seed(std::uint64_t seed,
                                          std::uint64_t round) noexcept {
    return seed ^ (round * 6364136223846793005ULL + 1442695040888963407ULL);
}

inline std::uint64_t fibonacci_vertex_hash(node_index v,
                                           std::uint64_t round_seed) noexcept {
    return (std::uint64_t(v) ^ round_seed) * 0x9E3779B97F4A7C15ULL;
}

} // namespace detail

} // namespace apxchol
