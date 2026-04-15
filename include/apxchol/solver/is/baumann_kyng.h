#pragma once
/// Baumann-Kyng independent set via random sampling + priority IS.
///
/// Faithfully follows the O(|sample|) per-round work bound from:
///   Baumann & Kyng (2024), "A Framework for Parallelizing
///   Approximate Gaussian Elimination," SPAA '24.
///
/// ─── How it differs from the shared prune → select pipeline ───
///
/// BG's pipeline does O(|active| · d / T) per round: it prunes ALL
/// vertices, then runs greedy IS on ALL vertices.  With ~d rounds
/// to eliminate all vertices, total IS-selection work = O(n·d²/T).
///
/// BK samples O(n/(c·d)) vertices per round at probability p = 1/(c·d).
/// Only sampled vertices' adj lists are accessed or pruned.
/// Per-round heavy work: O(|sample| · d / T) = O(n / (c·T)).
/// Over ~c·d rounds, total IS-selection work = O(n·d/T) = O(m/T).
///
/// Dead edges on non-sampled vertices accumulate but each dead edge
/// is scanned at most once more (when its owner is next sampled and
/// pruned), so total dead-edge work is O(m) amortized.
///
/// ─── Priority IS (replaces isolation) ───
///
/// The paper uses isolation (degree 0 in induced subgraph).  We use
/// priority-based IS: a sampled vertex v enters IS if it has the
/// lowest hash-priority among all its sampled+active neighbors.
/// This is Luby's algorithm restricted to the sample.
///
///   Advantages over isolation:
///   - IS ≈ n·p/(p·d+1) = n/(d·(c+1)), same as greedy.
///     Isolation gives n·p·e^{-p·d} ≈ n·e^{-1/c} / (c·d), smaller.
///   - Inherently parallel (no block_of, no barriers, no fixup).
///     Each vertex's check depends only on neighbors' hashes.
///
/// ─── Eliminating O(|active|) overhead ───
///
/// The key to O(|sample|) per round: NO phase touches ALL active
/// vertices.  Specifically:
///   - No block_of assignment (priority IS doesn't need blocks)
///   - No barrier (each vertex's check is independent)
///   - No fixup pass (no cross-block conflicts)
///   - No chosen[] cleanup (we don't use a global chosen[] array)
///   - No collect_is scan (IS is built inline during the sample loop)
///   - Only cost: O(|active|/T) to scan active[] and hash-filter.
///     Each non-sampled vertex costs ~2ns (load index + hash + branch).

#include "apxchol/solver/is/independent_set.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace apxchol {

struct baumann_kyng_is {
    uint64_t round = 0;
    double est_avg_degree = 0.0;

    static constexpr bool has_custom_find_is = true;

    template<incidence_storage Incidence>
    detail::find_is_result find_is(graph<Incidence>& G,
                                   std::span<const node_index> active,
                                   const factor_options& opts,
                                   checkpoint* cp = nullptr) {
        if (active.empty()) return {{}, 0.0};
        if (cp) { cp->descend("find_is"); cp->tick(); }

        const double c = opts.bk_sampling_constant;
        // Degree estimate: previous round's sample-based estimate (unbiased),
        // or 2·m/n for round 0.  Updated after each round from sampled vertices.
        if (round == 0)
            est_avg_degree = 2.0 * G.m() / double(active.size());
        double d_est = std::max(est_avg_degree, 1.0);
        double degree_threshold = opts.degree_multiplier * d_est;
        double p = std::min(1.0 / (c * d_est), 1.0);
        uint64_t hash_thresh = (p >= 1.0) ? UINT64_MAX
                             : uint64_t(p * double(UINT64_MAX));

        uint64_t round_seed = opts.seed
            ^ (round * 6364136223846793005ULL + 1442695040888963407ULL);
        auto prio = [round_seed](node_index v) -> uint64_t {
            return (uint64_t(v) ^ round_seed) * 11400714819323198485ULL;
        };

        // Thread-local IS vectors — no shared chosen[] array needed.
        int nt = 1;
        #ifdef _OPENMP
        nt = omp_get_max_threads();
        #endif
        std::vector<std::vector<node_index>> local_is(nt);

        double sample_degree_sum = 0;
        long long sample_count = 0;

        #pragma omp parallel reduction(+:sample_degree_sum, sample_count) \
            if(active.size() > opts.omp_threshold)
        {
            int tid = 0, nthreads = 1;
            #ifdef _OPENMP
            tid = omp_get_thread_num();
            nthreads = omp_get_num_threads();
            #endif
            size_t bs = active.size() * size_t(tid) / nthreads;
            size_t be = active.size() * size_t(tid + 1) / nthreads;

            auto& my_is = local_is[tid];

            for (size_t i = bs; i < be; ++i) {
                auto v = active[i];
                uint64_t pv = prio(v);
                if (pv > hash_thresh) continue;  // not sampled — O(1), no adj access

                // Fused prune + priority IS check in a single chain walk.
                // v enters IS if no sampled active neighbor has lower prio.
                bool ok = true;
                index_t deg = G.prune_and_visit(v, [&](node_index u) {
                    if (ok) {
                        uint64_t pu = prio(u);
                        if (pu <= hash_thresh && pu < pv)
                            ok = false;
                    }
                });
                sample_degree_sum += deg;
                ++sample_count;

                if (deg > degree_threshold) continue;
                if (ok) my_is.push_back(v);
            }
        }

        if (cp) (*cp)("sample_prune_select");

        // Update degree estimate from sampled vertices.
        if (sample_count > 0)
            est_avg_degree = sample_degree_sum / double(sample_count);

        // Merge thread-local IS: O(|IS|), no O(|active|) scan.
        std::vector<node_index> is;
        for (auto& v : local_is)
            is.insert(is.end(), v.begin(), v.end());

        if (cp) { (*cp)("collect"); cp->ascend(); }
        ++round;

        return {std::move(is), est_avg_degree};
    }

    // Stubs — unused with has_custom_find_is = true.
    template<incidence_storage Incidence>
    void select(const graph<Incidence>&,
                std::span<const node_index>,
                std::span<const index_t>,
                double, std::span<char>,
                const factor_options&) {}

    void cleanup(std::span<const node_index>,
                 const factor_options&) {}
};

} // namespace apxchol
