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
///       static constexpr std::string_view name = "mine";  // dispatch string
///       static constexpr bool degree_prepass = true;      // optional trait
///
///       template<incidence_storage I>
///       void find_partition(graph<I>& G,
///                           std::span<const node_index> candidates,
///                           const partition_context& ctx, selection& out);
///   };
///
/// The `degree_prepass` trait (default false) decides what the orchestrator
/// does before the call:
///
///   * true  — the orchestrator prunes dead edges, computes fresh degrees,
///     and applies the degree cap (ctx.options.degree_quantile /
///     degree_multiplier): `candidates` is the ELIGIBLE vertex list, and
///     ctx.degrees[v] gives the current incident-edge count of every active
///     vertex. The shipped scan rules (block_greedy, luby, rootset) use this.
///   * false — no O(active + edges) pre-pass runs: `candidates` is the raw
///     active list (ascending original-index order) and ctx.degrees is
///     empty. baumann_kyng works this way — it estimates degrees from its
///     hash sample and applies the cap to that estimate.
///
/// Degrees come as a vertex-indexed span — ctx.degrees[v] is indexed by
/// VERTEX ID, not by position in `candidates` — because rules compare
/// priorities of ADJACENT vertices, i.e. they need degree-by-id lookups for
/// neighbors. They cannot be queried from G instead: a CURRENT degree is
/// not an O(1) graph query — incidence lists accumulate edges to eliminated
/// vertices, so the truth requires the prune walk the pre-pass already does
/// for the cap. (Eager degree counters in the graph were tried and lost at
/// high thread counts — the decrements become atomic contention in the
/// elimination hot loop — and stale O(1) list sizes overcount without bound.
/// A contention-free incremental scheme is an open problem, not a closed
/// door.)
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
/// candidate blocks, repeated whole-set passes, DAG levels, a hash sample),
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
/// wobble run to run. Two shipped rules violated it until 2026-08-20:
/// block_greedy resolved cross-block conflicts against a mask other threads
/// were concurrently clearing, and rootset built its peel frontier by
/// concatenating `schedule(dynamic)` per-thread buffers. Both are fixed at
/// their sites; the guard is FactorizeDeterminism.* in tests/test_factorize.cpp.
///
/// The two patterns to avoid, concretely:
///   * reading `out.contains(u)` (or any shared flag) in a pass that also
///     writes it — decide against a snapshot, barrier, then apply;
///   * letting a `schedule(dynamic)` / `nowait` loop decide WHERE a vertex
///     lands in a per-thread buffer that is later concatenated — sort the
///     result, or partition the work statically.
///
/// `G` is non-const only so the pruning helpers can run. Queries
/// (see graph/graph.h):
///   G.n(), G.m()            — vertex / edge counts
///   G.adj(v)                — incidence slots of v (may still contain edges
///                             to eliminated vertices until pruned)
///   G.edge_target(idx, v)   — the neighbor of v across incidence slot idx
///   G.is_active(u)          — u not yet eliminated?
///   G.prune_and_degree(v)   — drop v's dead edges, return its live degree
///   G.prune_and_visit(v, f) — same, calling f(u) for each live neighbor
/// Pruning removes only edges to eliminated vertices — it never changes the
/// graph being represented. It is the ONLY mutation a partitioner may
/// perform; anything else is out of contract.
///
/// Use a custom partitioner at compile time
/// (factorize<my_partitioner>(G, opts)), as a configured instance
/// (factorize(A, my_partitioner{...}, opts)), or add it to partitioner_list
/// for by-name dispatch (factor_options::is_select / CLI / bindings).
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
#include "apxchol/checkpoint.h"
#include <concepts>
#include <limits>
#include <type_traits>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace apxchol {

/// Read-only services the orchestrator hands to a partitioner call.
/// Everything except `degrees` is constant for the whole factorize() run.
struct partition_context {
    /// The user's selection preferences (factor_options::partition), always
    /// populated. Who acts on the cap knobs (degree_quantile /
    /// degree_multiplier) depends on the degree_prepass trait: with the
    /// pre-pass the orchestrator has already applied them when building the
    /// candidate list; without it nothing has been applied and the rule
    /// honors them itself. degree_tiebreak (prefer the lower-degree vertex
    /// when adjacent candidates compete) is never applied by the
    /// orchestrator — honoring it is always the rule's job.
    partition_options options;
    /// Determinism source. Derive all randomness as pure functions of
    /// (seed, round, vertex) — hash priorities as the shipped rules do, or
    /// use it to seed a generator you own. A seed rather than an RNG object
    /// because draws from a shared stream would depend on thread scheduling.
    unsigned seed = 42;
    std::size_t omp_threshold = 2000;  ///< min work size before OpenMP engages
    checkpoint* cp = nullptr;          ///< optional profiling tree (may be null)
    /// Current degrees indexed by VERTEX ID — degrees[v], not by position in
    /// the candidates span — valid for every active vertex. Non-empty
    /// exactly when this partitioner declares degree_prepass (it is the
    /// pre-pass's byproduct).
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
            per_thread_[thread_id()].push_back(v);
        }
    }
    void remove(node_index v) { mask_[v] = 0; }
    bool contains(node_index v) const { return mask_[v] != 0; }

    // ── Orchestrator side ──
    void reset(node_index n) {
        if (mask_.size() < static_cast<size_t>(n)) mask_.assign(n, 0);
        int nt = 1;
#ifdef _OPENMP
        nt = omp_get_max_threads();
#endif
        if (per_thread_.size() < static_cast<size_t>(nt)) per_thread_.resize(nt);
        for (auto& l : per_thread_) l.clear();
        result_.clear();
    }
    /// Concatenate the per-thread lists, drop removed entries, reset the
    /// mask. O(inserted), not O(n).
    const partition_result& finalize() {
        for (auto& l : per_thread_)
            for (auto v : l)
                if (mask_[v]) {
                    result_.data.push_back(v);
                    mask_[v] = 0;
                }
        return result_;
    }

private:
    static int thread_id() {
#ifdef _OPENMP
        return omp_get_thread_num();
#else
        return 0;
#endif
    }
    std::vector<char> mask_;
    std::vector<std::vector<node_index>> per_thread_;
    partition_result result_;
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

template<typename T>
concept partitioner =
    requires { { T::name } -> std::convertible_to<std::string_view>; } &&
    requires(T t, graph<vec_pool_incidence>& G,
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
    static constexpr std::string_view name = "callable";
    static constexpr bool degree_prepass = Prepass;

    explicit callable_partitioner(F f) : fn(std::move(f)) {}
    F fn;

    template<incidence_storage Incidence>
    void find_partition(graph<Incidence>& G,
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

} // namespace apxchol
