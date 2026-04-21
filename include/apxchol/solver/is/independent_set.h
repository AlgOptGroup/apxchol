#pragma once
/// Modular independent-set selection for approximate Cholesky factorization.
///
/// The factorization loop eliminates vertices in independent sets (IS).
/// The IS selection strategy determines which low-degree vertices to eliminate
/// each round.  Different strategies trade off IS size, parallelism, and
/// simplicity.
///
/// Each selector is a stateful class satisfying a common duck-typed interface:
///
///   struct SomeSelector {
///       // Mark IS vertices in chosen[v] = 1 for vertices v ∈ active.
///       // Pre: degrees[i] is the degree of active[i]; chosen[] is zero-initialized.
///       // Only vertices with degrees[i] <= degree_threshold should be considered.
///       template<incidence_storage Incidence>
///       void select(const graph<Incidence>& G,
///                   std::span<const node_index> active,
///                   std::span<const index_t> degrees,
///                   double degree_threshold,
///                   std::span<char> chosen,
///                   const factor_options& opts);
///
///       // Reset any selector-owned scratch arrays (called after collect).
///       void cleanup(std::span<const node_index> active,
///                    const factor_options& opts);
///   };
///
/// The orchestrator find_independent_set() handles the shared phases
/// (pruning dead edges, computing degrees, collecting the IS vector,
/// resetting chosen[]) and delegates the selection logic to the strategy.
///
/// Available strategies:
///   - block_greedy_is  (is_block_greedy.h)  — default, fastest in practice
///   - luby_is          (is_luby.h)          — hash-priority local minimum
///   - baumann_kyng_is  (is_baumann_kyng.h)  — random-sample isolated vertices
///   - rootset_is       (is_rootset.h)       — Blelloch rootset peeling (exact greedy)
///
/// Usage as library (compile-time choice):
///   #include <apxchol/solver/is_luby.h>
///   auto F = apxchol::factorize<apxchol::luby_is>(G, opts);
///
/// Usage as binary (runtime choice via factor_options::is_strategy):
///   opts.is_select = apxchol::is_strategy::luby;
///   auto F = apxchol::factorize(L, storage, opts);

#include "apxchol/types.h"
#include "apxchol/solver/factor_options.h"
#include "apxchol/graph/graph.h"
#include "apxchol/checkpoint.h"
#include <span>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace apxchol {

namespace detail {

struct find_is_result {
    std::vector<node_index> is;
    double avg_degree;
};

// ── Shared phase: prune dead edges and compute active degrees (parallel). ──
// Returns average degree across active vertices.
template<incidence_storage Incidence>
double prune_active_edges(graph<Incidence>& G,
                          std::span<const node_index> active,
                          std::span<index_t> degrees,
                          const factor_options& opts) {
    double total_degree = 0;
    // Dynamic schedule: prune_and_degree walks each vertex's adjacency list,
    // so per-iteration cost is proportional to (currently-stored) degree.
    // With static scheduling a single thread getting the few high-degree
    // vertices stalls the rest; dynamic with chunk=256 amortizes the
    // scheduling overhead while keeping load balanced.
    //
    // If forward_star + filter_append, set up the parallel-append arena so
    // each thread's filter() allocates survivor slots via atomic-reserve
    // instead of racing on push_back.
    if constexpr (std::is_same_v<Incidence, forward_star_incidence>) {
        if (G.adj_filter_append_enabled()
            && active.size() > opts.omp_threshold) {
            G.begin_parallel_append_adj();
        }
    }
    #pragma omp parallel for reduction(+:total_degree) schedule(dynamic, 256) if(active.size() > opts.omp_threshold)
    for (size_t i = 0; i < active.size(); ++i) {
        degrees[i] = G.prune_and_degree(active[i]);
        total_degree += degrees[i];
    }
    if constexpr (std::is_same_v<Incidence, forward_star_incidence>) {
        if (G.adj_filter_append_enabled()
            && active.size() > opts.omp_threshold) {
            G.end_parallel_append_adj();
        }
    }
    return total_degree / double(active.size());
}

// ── Shared phase: collect IS vertices from chosen[], reset chosen[]. ──
// Thread-local gather preserves ordering (active[] is sorted →
// per-thread ranges are contiguous).
inline std::vector<node_index> collect_is(
        std::span<const node_index> active,
        std::span<char> chosen,
        const factor_options& opts) {
    std::vector<node_index> is;
    {
        int nt_collect = 1;
        #ifdef _OPENMP
        nt_collect = omp_get_max_threads();
        #endif
        std::vector<std::vector<node_index>> local_is(nt_collect);
        #pragma omp parallel
        {
            int tid = 0, nthreads = 1;
            #ifdef _OPENMP
            tid = omp_get_thread_num();
            nthreads = omp_get_num_threads();
            #endif
            auto bs = active.size() * size_t(tid) / nthreads;
            auto be = active.size() * size_t(tid + 1) / nthreads;
            for (size_t i = bs; i < be; ++i)
                if (chosen[active[i]]) local_is[tid].push_back(active[i]);
        }
        for (auto& v : local_is)
            is.insert(is.end(), v.begin(), v.end());
    }

    // Reset chosen[] for next round (only touched entries, parallel).
    #pragma omp parallel for schedule(static) if(active.size() > opts.omp_threshold)
    for (size_t i = 0; i < active.size(); ++i)
        chosen[active[i]] = 0;

    return is;
}

// ── Orchestrator: prune → select → collect. ──
// ISSelector must satisfy the selector interface described above.
// Selectors with has_custom_find_is = true bypass the shared pipeline
// and provide their own O(|sample|) implementation.
template<typename ISSelector, incidence_storage Incidence>
find_is_result find_independent_set(ISSelector& selector,
                                    graph<Incidence>& G,
                                    std::span<const node_index> active,
                                    const factor_options& opts,
                                    checkpoint* cp = nullptr) {
    if (active.empty()) return {{}, 0.0};

    // Fast path: selector provides its own find_is (e.g. BK with O(|sample|) cost).
    if constexpr (requires { ISSelector::has_custom_find_is; }) {
        if constexpr (ISSelector::has_custom_find_is) {
            return selector.find_is(G, active, opts, cp);
        }
    }

    if (cp) { cp->descend("find_is"); cp->tick(); }

    // Scratch arrays shared across all selectors (static lifetime).
    static std::vector<char> chosen;
    static std::vector<index_t> degrees;

    size_t nn = G.n();
    chosen.resize(nn, 0);
    degrees.resize(nn);

    double avg_degree = prune_active_edges(G, active, degrees, opts);
    double degree_threshold = opts.degree_multiplier * avg_degree;
    if (cp) (*cp)("prune");

    selector.select(G, active, degrees, degree_threshold, chosen, opts);
    if (cp) (*cp)("select");

    auto is = collect_is(active, chosen, opts);
    selector.cleanup(active, opts);
    if (cp) { (*cp)("collect"); cp->ascend(); }

    return {std::move(is), avg_degree};
}

} // namespace detail

} // namespace apxchol
