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

#include <omp.h>

namespace apxchol::detail {

struct baumann_kyng_partitioner {
    static constexpr std::string_view name = "baumann_kyng";
    static constexpr bool sample_bounded = true;

    /// Sampling probability = 1/(c·d_avg); lower c → larger IS per round.
    /// The measured single default avoids making the factor depend on the
    /// caller's thread count.  c=0.1 wins from 2 through 72 threads while
    /// remaining neutral at one thread; c=0.05 is faster in some tails but has
    /// a weaker iteration-count margin.
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

    void find_partition(graph& G, std::span<const node_index> active,
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

        const uint64_t round_seed = detail::partition_round_seed(ctx.seed, round);
        auto prio = [round_seed](node_index v) -> uint64_t {
            return detail::fibonacci_vertex_hash(v, round_seed);
        };

        const int nt = omp_get_max_threads();

        // Resolve an unset quantile here too: this partitioner can be driven
        // directly, without factorize() having normalised the options first.
        const double q_cap  = ctx.options.degree_quantile.value_or(
            detail::degree_quantile_host_default);
        const bool use_q    = q_cap > 0.0 && q_cap < 1.0;
        const bool tiebreak = ctx.degree_tiebreak;
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
                const int tid = omp_get_thread_num();
                const int nthreads = omp_get_num_threads();
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
                const int tid = omp_get_thread_num();
                const int nthreads = omp_get_num_threads();
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
                    static_cast<size_t>(q_cap * deg_merge_.size()));
                std::nth_element(deg_merge_.begin(), deg_merge_.begin() + k, deg_merge_.end());
                degree_threshold = static_cast<double>(deg_merge_[k]);
            }

            // Degree-aware IS priority: a sampled neighbor blocks v only if its
            // (degree,hash) is strictly lower -> ineligible high-degree neighbors
            // (higher degree) never block eligible low-degree ones, and low-degree
            // wins, giving a min-degree-like elimination order.
            auto prio2 = [&](node_index v) -> uint64_t {
                const uint64_t h = prio(v);
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

        ++round;
    }
};

} // namespace apxchol::detail
