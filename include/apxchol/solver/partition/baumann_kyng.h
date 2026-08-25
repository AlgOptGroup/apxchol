#pragma once
/// Baumann-Kyng independent set via random sampling + priority IS
/// (partitioner form).
///
/// Algorithm: each round hash-samples active vertices with probability
/// ~1/(c·d_avg) (so the expected sample is degree-bounded), then keeps the
/// sampled vertices that are priority minima among their sampled neighbors.
/// Per-round cost is O(|sample|·d), not O(|active|·d), so it bypasses the
/// shared prune pipeline; the average degree is estimated from the previous
/// round's sample instead of a full scan.

#include "apxchol/solver/partitioner.h"
#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace apxchol {

struct baumann_kyng_partitioner {
    static constexpr std::string_view name = "baumann_kyng";
    static constexpr bool sample_bounded = true;

    /// Sampling probability = 1/(c·d_avg); lower c → larger IS per round.
    /// The measured single default avoids making the factor depend on the
    /// caller's thread count.  c=0.1 wins from 2 through 72 threads while
    /// remaining neutral at one thread; c=0.05 is faster in some tails but has
    /// a weaker iteration-count margin.  See AGENTS.md for the campaign ledger.
    double sampling_constant = 0.1;

    uint64_t round = 0;
    /// Average degree the sampling probability is derived from. Left at 0 it is
    /// seeded on round 0 from 2·m/|active| — correct only on a PRISTINE graph.
    /// A caller that starts BK partway through an elimination (the residual loop
    /// in factorization_impl.h) must pre-seed this with a measured average
    /// instead: `graph::m()` is monotone — it counts every edge ever added and
    /// is never decremented when elimination prunes one — so 2·m/|active| on a
    /// half-eliminated graph is inflated by the whole eliminated prefix and
    /// under-samples round 0 by that factor.
    double   est_avg_degree = 0.0;
    // Scratch for the quantile-cap + degree-aware-priority path (two-pass round).
    std::vector<node_index>                 deg_by_vertex_;   // sampled v -> deg (this round)
    std::vector<node_index>                 deg_merge_;       // quantile nth_element scratch
    std::vector<std::vector<node_index>> local_sampled_;   // per-thread sampled lists
    std::vector<node_index>              sampled_all_;      // flattened sampled list
    size_t selected_degree_work_ = 0;

    /// Sum of the live degrees of the vertices selected by the last call.
    /// Every selected vertex's degree was already measured by the sampler, so
    /// exposing the sum lets factorization's dense-small-round gate make the
    /// same decision for BK as it does for prepass-based selectors without an
    /// extra residual-graph traversal.
    size_t selected_degree_work() const { return selected_degree_work_; }

    template<incidence_storage Incidence>
    void find_partition(graph<Incidence>& G, std::span<const node_index> active,
                        const partition_context& ctx, selection& out) {
        selected_degree_work_ = 0;
        if (active.empty()) return;

        const double c = sampling_constant;
        // Degree estimate: previous round's sample-based estimate (unbiased),
        // a caller-supplied seed, or 2·m/n for round 0 on a pristine graph (see
        // est_avg_degree).  Updated after each round from sampled vertices.
        if (round == 0 && est_avg_degree <= 0.0)
            est_avg_degree = 2.0 * G.m() / double(active.size());
        double d_est = std::max(est_avg_degree, 1.0);
        double degree_threshold = ctx.options.degree_multiplier * d_est;
        double p = std::min(1.0 / (c * d_est), 1.0);
        uint64_t hash_thresh = (p >= 1.0) ? UINT64_MAX
                             : uint64_t(p * double(UINT64_MAX));

        uint64_t round_seed = ctx.seed
            ^ (round * 6364136223846793005ULL + 1442695040888963407ULL);
        auto prio = [round_seed](node_index v) -> uint64_t {
            return (uint64_t(v) ^ round_seed) * 11400714819323198485ULL;
        };

        int nt = 1;
        #ifdef _OPENMP
        nt = omp_get_max_threads();
        #endif

        const bool use_q    = ctx.options.degree_quantile > 0.0 && ctx.options.degree_quantile < 1.0;
        const bool tiebreak = ctx.options.degree_tiebreak;
        if ((use_q || tiebreak) && deg_by_vertex_.size() < static_cast<size_t>(G.n()))
            deg_by_vertex_.resize(G.n());
        if (local_sampled_.size() < static_cast<size_t>(nt)) local_sampled_.resize(nt);
        for (auto& s : local_sampled_) s.clear();

        double sample_degree_sum = 0;
        long long sample_count = 0;
        size_t selected_degree_work = 0;

        const bool two_pass = use_q || tiebreak;
        if (!two_pass) {
            // ── Legacy single-pass: fused prune + hash-priority IS + multiplier cap.
            #pragma omp parallel reduction(+:sample_degree_sum, sample_count, selected_degree_work) \
                if(active.size() > ctx.omp_threshold)
            {
                int tid = 0, nthreads = 1;
                #ifdef _OPENMP
                tid = omp_get_thread_num(); nthreads = omp_get_num_threads();
                #endif
                size_t bs = active.size() * size_t(tid) / nthreads;
                size_t be = active.size() * size_t(tid + 1) / nthreads;
                for (size_t i = bs; i < be; ++i) {
                    auto v = active[i];
                    uint64_t pv = prio(v);
                    if (pv > hash_thresh) continue;
                    bool ok = true;
                    node_index deg = G.prune_and_visit(v, [&](node_index u) {
                        if (ok) { uint64_t pu = prio(u); if (pu <= hash_thresh && pu < pv) ok = false; }
                    });
                    sample_degree_sum += deg; ++sample_count;
                    if (deg > degree_threshold) continue;
                    if (ok) {
                        out.add(v);
                        selected_degree_work += static_cast<size_t>(deg);
                    }
                }
            }
        } else {
            // ── PASS 1: degree of every sampled vertex (one walk), stored by vertex.
            #pragma omp parallel reduction(+:sample_degree_sum, sample_count) \
                if(active.size() > ctx.omp_threshold)
            {
                int tid = 0, nthreads = 1;
                #ifdef _OPENMP
                tid = omp_get_thread_num(); nthreads = omp_get_num_threads();
                #endif
                size_t bs = active.size() * size_t(tid) / nthreads;
                size_t be = active.size() * size_t(tid + 1) / nthreads;
                auto& my_s = local_sampled_[tid];
                for (size_t i = bs; i < be; ++i) {
                    auto v = active[i];
                    if (prio(v) > hash_thresh) continue;
                    node_index deg = G.prune_and_degree(v);
                    deg_by_vertex_[v] = deg;
                    sample_degree_sum += deg; ++sample_count;
                    my_s.push_back(v);
                }
            }
            sampled_all_.clear();
            for (auto& s : local_sampled_)
                sampled_all_.insert(sampled_all_.end(), s.begin(), s.end());

            // Quantile cap from THIS round's sampled degrees (no lag).
            if (use_q && !sampled_all_.empty()) {
                deg_merge_.clear();
                for (auto v : sampled_all_) deg_merge_.push_back(deg_by_vertex_[v]);
                size_t k = std::min(deg_merge_.size() - 1,
                    static_cast<size_t>(ctx.options.degree_quantile * deg_merge_.size()));
                std::nth_element(deg_merge_.begin(), deg_merge_.begin() + k, deg_merge_.end());
                degree_threshold = static_cast<double>(deg_merge_[k]);
            }

            // Degree-aware IS priority: a sampled neighbor blocks v only if its
            // (degree,hash) is strictly lower -> ineligible high-degree neighbors
            // (higher degree) never block eligible low-degree ones, and low-degree
            // wins, giving a min-degree-like elimination order.
            auto prio2 = [&](node_index v) -> uint64_t {
                const uint64_t h = (uint64_t(v) ^ round_seed) * 11400714819323198485ULL;
                if (!tiebreak) return h;
                return (uint64_t(deg_by_vertex_[v]) << 40) | (h >> 24);
            };

            // ── PASS 2: keep eligible sampled vertices that are priority-IS minima.
            #pragma omp parallel reduction(+:selected_degree_work) \
                if(active.size() > ctx.omp_threshold)
            {
                #pragma omp for schedule(static) nowait
                for (size_t idx = 0; idx < sampled_all_.size(); ++idx) {
                    auto v = sampled_all_[idx];
                    if (static_cast<double>(deg_by_vertex_[v]) > degree_threshold) continue;
                    const uint64_t pv = prio2(v);
                    bool ok = true;
                    G.prune_and_visit(v, [&](node_index u) {
                        if (ok && prio(u) <= hash_thresh && prio2(u) < pv) ok = false;
                    });
                    if (ok) {
                        out.add(v);
                        selected_degree_work +=
                            static_cast<size_t>(deg_by_vertex_[v]);
                    }
                }
            }
        }

        // Update degree estimate from sampled vertices.
        if (sample_count > 0)
            est_avg_degree = sample_degree_sum / double(sample_count);
        selected_degree_work_ = selected_degree_work;

        // Work trace: per-round iteration cost (active = the O(|active|) full scan) and
        // edge-visit cost (sample_deg = Σ deg over sampled vertices). Sum over rounds to
        // compare total IS-finding work vs m (settles the O(m) vs O(m·d) amortization Q).
        if (std::getenv("APXCHOL_WORK_TRACE"))
            std::fprintf(stderr, "[bkwork] active=%zu sampled=%lld sample_deg=%.0f\n",
                         active.size(), sample_count, sample_degree_sum);

        ++round;
    }
};

} // namespace apxchol
