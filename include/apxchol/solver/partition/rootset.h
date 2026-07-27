#pragma once
/// Rootset (Blelloch) independent set selection (partitioner form).
///
/// Algorithm: builds the random-priority dependency DAG over eligible active
/// candidates (each vertex counts its lower-priority eligible neighbors) and
/// peels it level by level — priority-0 candidates ("roots") join the IS, their
/// neighbors are blocked, and blocked candidates decrement their later
/// neighbors' counts to expose the next root frontier.  Yields the same IS a
/// sequential random-greedy pass would, with level-parallel execution.

#include "apxchol/solver/partitioner.h"
#include "apxchol/solver/partitioner_helpers.h"

namespace apxchol {

struct rootset_partitioner {
    static constexpr std::string_view name = "rootset";
    // Hand the bail residual to parallel BK rounds (like bg/luby) instead of
    // serial-peeling it. Without this, rootset serial-peels its dense bail-core
    // -> ~n sequential SpTRSV levels (432,943 levels observed on a 540k-vertex
    // co-authorship graph). 500 matches bg/luby.
    static constexpr size_t residual_handoff_threshold = 500;

    std::vector<int32_t> pri;
    std::vector<char>    eligible;
    std::vector<char>    dead;
    uint64_t round = 0;

    static constexpr bool degree_prepass = true;

    template<incidence_storage Incidence>
    void find_partition(graph<Incidence>& G, std::span<const node_index> candidates,
                        const partition_context& ctx, selection& out) {
        select_into_chosen(G, candidates, out, ctx);
        ++round;
        cleanup_round_state(candidates, ctx);
    }

private:
    template<incidence_storage Incidence>
    void select_into_chosen(const graph<Incidence>& G,
                            std::span<const node_index> candidates,
                            selection& out,
                            const partition_context& ctx) {
        size_t nn = G.n();
        pri.resize(nn, 0);
        eligible.resize(nn, 0);
        dead.resize(nn, 0);

        uint64_t seed = ctx.seed
            ^ (round * 6364136223846793005ULL + 1442695040888963407ULL);
        // Degree-aware priority (degree_tiebreak): degree in high bits, hash
        // in low bits, so low-degree candidates become roots first (parallel
        // min-degree); plain hash otherwise.
        const bool tiebreak = ctx.options.degree_tiebreak;
        const auto degrees  = ctx.degrees;
        auto hash_fn = [&](node_index v) -> uint64_t {
            const uint64_t h = (uint64_t(v) ^ seed) * 11400714819323198485ULL;
            if (!tiebreak) return h;
            return (uint64_t(degrees[v]) << 40) | (h >> 24);
        };

        // 1. Mark the work candidates eligible (the pre-pass already applied
        //    the cap).
        #pragma omp parallel for schedule(static) if(candidates.size() > ctx.omp_threshold)
        for (size_t i = 0; i < candidates.size(); ++i)
            eligible[candidates[i]] = 1;

        // 2. Compute initial priorities: count eligible neighbors with lower
        //    hash.
        #pragma omp parallel for schedule(static) if(candidates.size() > ctx.omp_threshold)
        for (size_t i = 0; i < candidates.size(); ++i) {
            auto v = candidates[i];
            auto hv = hash_fn(v);
            int32_t cnt = 0;
            for (auto idx : G.adj(v)) {
                auto u = G.edge_target(idx, v);
                if (G.is_active(u) && eligible[u] && hash_fn(u) < hv)
                    ++cnt;
            }
            pri[v] = cnt;
        }

        // 3. Collect initial roots (priority 0).
        std::vector<node_index> frontier;
        for (auto v : candidates) {
            if (pri[v] == 0)
                frontier.push_back(v);
        }

        // 4. Peel the priority DAG level by level.
        std::vector<node_index> removed;
        std::vector<node_index> next_frontier;

        const size_t omp_thr = ctx.omp_threshold;

        while (!frontier.empty()) {
            const size_t fsz = frontier.size();

            // Mark frontier candidates as IS.
            #pragma omp parallel for schedule(static) if(fsz > omp_thr)
            for (size_t i = 0; i < fsz; ++i) {
                auto v = frontier[i];
                out.add(v);
                dead[v] = 1;
            }

            // Find blocked candidates: eligible active neighbors of IS
            // candidates that are still undecided.
            // Use atomic CAS on dead[] to avoid duplicates across threads.
            removed.clear();
            #ifdef _OPENMP
            if (fsz > omp_thr) {
                // Thread-local buffers collected at the end.
                int nthreads = omp_get_max_threads();
                std::vector<std::vector<node_index>> thr_removed(nthreads);
                #pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    auto& local = thr_removed[tid];
                    #pragma omp for schedule(dynamic, 64)
                    for (size_t i = 0; i < fsz; ++i) {
                        auto r = frontier[i];
                        for (auto idx : G.adj(r)) {
                            auto u = G.edge_target(idx, r);
                            if (G.is_active(u) && eligible[u]) {
                                char expected = 0;
                                if (__atomic_compare_exchange_n(
                                        &dead[u], &expected, char(1),
                                        /*weak=*/false,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED))
                                    local.push_back(u);
                            }
                        }
                    }
                }
                for (auto& v : thr_removed)
                    removed.insert(removed.end(), v.begin(), v.end());
            } else
            #endif
            {
                for (auto r : frontier) {
                    for (auto idx : G.adj(r)) {
                        auto u = G.edge_target(idx, r);
                        if (G.is_active(u) && eligible[u] && !dead[u]) {
                            dead[u] = 1;
                            removed.push_back(u);
                        }
                    }
                }
            }

            // Decrement priorities: for each blocked vertex u, its later
            // eligible neighbors w (hash(u) < hash(w)) were counting u as
            // an undecided earlier neighbor.  Now u is decided -> decrement.
            const size_t rsz = removed.size();
            next_frontier.clear();
            #ifdef _OPENMP
            if (rsz > omp_thr) {
                int nthreads = omp_get_max_threads();
                std::vector<std::vector<node_index>> thr_next(nthreads);
                #pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    auto& local = thr_next[tid];
                    #pragma omp for schedule(dynamic, 64)
                    for (size_t i = 0; i < rsz; ++i) {
                        auto u = removed[i];
                        auto hu = hash_fn(u);
                        for (auto idx : G.adj(u)) {
                            auto w = G.edge_target(idx, u);
                            if (G.is_active(w) && eligible[w] && !dead[w]
                                && hu < hash_fn(w)) {
                                if (__atomic_sub_fetch(&pri[w], 1,
                                                      __ATOMIC_RELAXED) == 0)
                                    local.push_back(w);
                            }
                        }
                    }
                }
                for (auto& v : thr_next)
                    next_frontier.insert(next_frontier.end(), v.begin(), v.end());
            } else
            #endif
            {
                for (auto u : removed) {
                    auto hu = hash_fn(u);
                    for (auto idx : G.adj(u)) {
                        auto w = G.edge_target(idx, u);
                        if (G.is_active(w) && eligible[w] && !dead[w]
                            && hu < hash_fn(w)) {
                            if (--pri[w] == 0)
                                next_frontier.push_back(w);
                        }
                    }
                }
            }

            std::swap(frontier, next_frontier);
        }
    }

    void cleanup_round_state(std::span<const node_index> candidates,
                             const partition_context& ctx) {
        #pragma omp parallel for schedule(static) if(candidates.size() > ctx.omp_threshold)
        for (auto v : candidates) {
            pri[v] = 0;
            eligible[v] = 0;
            dead[v] = 0;
        }
    }
};

} // namespace apxchol
