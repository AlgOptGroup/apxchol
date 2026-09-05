#pragma once
// Frozen corrected K2+cycle reference from 30a8230d. Only its canonical sort
// is replaced by the cleanup branch's identical-order comparison sort.
#include "apxchol/solver/elimination/elimination.h"
#include <array>
#include <bit>
#include <cmath>
#include <limits>
namespace apxchol::research_k2 {
inline void canonical_sort_neighbors(std::span<weighted_neighbor> v) {
    std::sort(v.begin(),v.end(),[](const auto& a,const auto& b) {
        return a.weight != b.weight ? a.weight < b.weight : a.vertex < b.vertex;
    });
}
inline constexpr std::size_t kCoordinatedGksReceivers = 2;
inline constexpr std::size_t kCycleMaxDegree = 6;

// Exact second moment of the K=2 receiver-major law on a tiny star. A
// systematic phase changes its selected sources only at fractional prefix
// endpoints. Integrate those intervals for the two coordinated receivers;
// finish the independent remaining choices by their conditional moments.
// This enumerates at most d*(d-1) phase rectangles, never random samples.
// Precondition: 3<=d<=6, sorted positive weights, and finite positive degrees.
inline double tiny_k2_relative_degree_variance(
        std::span<const weighted_neighbor> neighbors,
        const std::array<double, kCycleMaxDegree>& prefix,
        const std::array<double, kCycleMaxDegree>& exact_degree,
        double pivot_degree) {
    const std::size_t d = neighbors.size();
    std::array<double, kCycleMaxDegree> mass{};
    for (std::size_t i = 0; i + 1 < d; ++i)
        mass[i] = neighbors[i].weight * (prefix[d - 1] - prefix[i]) /
                  pivot_degree;
    using partners_t = std::array<std::size_t, kCycleMaxDegree>;
    partners_t initial;
    initial.fill(d);
    double score = 0.0;
    const auto finish = [&](const partners_t& partners, double probability) {
        auto mean = mass;  // every source's outgoing mass is deterministic
        std::array<double, kCycleMaxDegree> variance{};
        for (std::size_t i = 0; i + 1 < d; ++i) {
            if (partners[i] != d) {
                mean[partners[i]] += mass[i];
                continue;
            }
            const std::size_t last = d - 3;
            for (std::size_t r = i + 1; r <= last; ++r) {
                const double p = neighbors[r].weight / (prefix[last] - prefix[i]);
                mean[r] += mass[i] * p;
                variance[r] += mass[i] * mass[i] * p * (1.0 - p);
            }
        }
        double conditional = 0.0;
        for (std::size_t r = 0; r < d; ++r) {
            const double error = mean[r] - exact_degree[r];
            conditional += (variance[r] + error * error) /
                           (exact_degree[r] * exact_degree[r]);
        }
        score += probability * conditional;
    };
    const auto integrate = [&](auto&& self, std::size_t stage,
                               partners_t partners, double probability) -> void {
        if (stage == kCoordinatedGksReceivers) {
            finish(partners, probability);
            return;
        }
        const std::size_t r = d - 1 - stage;
        std::array<std::size_t, kCycleMaxDegree> sources{};
        std::array<double, kCycleMaxDegree> starts{}, ends{};
        std::array<double, kCycleMaxDegree + 2> cuts{};
        std::size_t count = 0, cut_count = 2;
        cuts[1] = 1.0;
        double cumulative = 0.0;
        constexpr double epsilon = 32.0 * std::numeric_limits<double>::epsilon();
        for (std::size_t i = 0; i < r; ++i) {
            if (partners[i] != d) continue;
            const double p = std::clamp(neighbors[r].weight /
                                      (prefix[r] - prefix[i]), 0.0, 1.0);
            if (p <= epsilon) continue;
            if (p >= 1.0 - epsilon) {
                partners[i] = r;
                continue;
            }
            sources[count] = i;
            starts[count] = cumulative;
            cumulative += p;
            ends[count++] = cumulative;
            cuts[cut_count++] = cumulative - std::floor(cumulative);
        }
        std::sort(cuts.begin(), cuts.begin() + cut_count);
        for (std::size_t k = 1; k < cut_count; ++k) {
            const double width = cuts[k] - cuts[k - 1];
            if (!(width > 0.0)) continue;
            const double phase = (cuts[k] + cuts[k - 1]) * 0.5;
            auto next = partners;
            for (std::size_t j = 0; j < count; ++j)
                if (phase + std::ceil(starts[j] - phase) < ends[j])
                    next[sources[j]] = r;
            self(self, stage + 1, next, probability * width);
        }
    };
    integrate(integrate, 0, initial, 1.0);
    return score;
}

// A uniform Hamiltonian cycle includes each clique edge with probability
// 2/(d-1). Its Horvitz--Thompson weight therefore preserves the exact expected
// clique. Use the extra edge only when the expected sum of relative degree
// variances is strictly below coordinated K=2 GKS for this exact star, and
// its edge-budget-adjusted expected normalized Frobenius error does not increase.
// Normalization uses this clique's exact degrees; external residual degrees
// are not available here. Finite-catalogue evidence is not a spectral bound.
// This is a bounded d<=6 calculation in binary64; there is no graph/input gate.
inline bool sample_variance_gated_cycle(
        std::span<weighted_neighbor> neighbors, double pivot_degree,
        std::uint64_t seed, edge_emitter out) {
    const std::size_t d = neighbors.size();
    if (d < 3 || d > kCycleMaxDegree || !(pivot_degree > 0.0))
        return false;

    canonical_sort_neighbors(neighbors);

    std::array<double, kCycleMaxDegree> prefix{};
    std::array<double, kCycleMaxDegree> exact_degree{};
    double total_weight = 0.0;
    for (std::size_t i = 0; i < d; ++i) {
        const double weight = neighbors[i].weight;
        if (!(weight > 0.0) || !std::isfinite(weight))
            return false;
        total_weight += weight;
        prefix[i] = total_weight;
    }
    if (!std::isfinite(total_weight)) return false;

    for (std::size_t i = 0; i < d; ++i)
        exact_degree[i] = neighbors[i].weight *
            (total_weight - neighbors[i].weight) / pivot_degree;

    double cycle_score = 0.0;
    const double population = static_cast<double>(d - 1);
    const double cycle_variance_factor =
        population * (population - 2.0) /
        (2.0 * (population - 1.0));
    for (std::size_t vertex = 0; vertex < d; ++vertex) {
        const double denominator =
            exact_degree[vertex] * exact_degree[vertex];
        if (!(denominator > 0.0) || !std::isfinite(denominator))
            return false;

        // For a simple random sample of two incident edges from d-1, this
        // centered finite-population form is the exact variance of the
        // inverse-probability weighted degree. It also avoids cancellation on
        // triangles, whose cycle is the full exact clique.
        const double mean = exact_degree[vertex] / population;
        double squared_deviation = 0.0;
        for (std::size_t other = 0; other < d; ++other) {
            if (other == vertex) continue;
            const double clique_weight = neighbors[vertex].weight *
                neighbors[other].weight / pivot_degree;
            const double delta = clique_weight - mean;
            squared_deviation += delta * delta;
        }
        cycle_score += cycle_variance_factor * squared_deviation / denominator;
    }
    // For d=3 the cycle is the exact clique (both errors are zero), while a
    // positive-weight GKS tree has nonzero error. Equal-weight stars also
    // satisfy both guards: cycle Vrel=0<K2 Vrel, cycle RNF2=d(d-3)/(2(d-1)),
    // and K2 RNF2 >= its off-diagonal part 2d(d-2)/(3(d-1)). The edge-budget
    // inequality follows from d^2-3d+8>0. These exact cases need no integration.
    if (d > 3 && neighbors.front().weight != neighbors.back().weight) {
        const double k2_score = tiny_k2_relative_degree_variance(
            neighbors, prefix, exact_degree, pivot_degree);
        if (!(cycle_score < k2_score)) return false;
        // Off-diagonal variances depend only on edge marginals. K=2 preserves
        // ordinary GKS marginals, so only its diagonal moment needed integration.
        double k2_offdiag = 0.0, cycle_offdiag = 0.0;
        for (std::size_t i = 0; i + 1 < d; ++i) {
            const double suffix = total_weight - prefix[i];
            for (std::size_t j = i + 1; j < d; ++j) {
                const double c = neighbors[i].weight * neighbors[j].weight / pivot_degree;
                const double scale = 2.0 * c * c / (exact_degree[i] * exact_degree[j]);
                k2_offdiag += scale * (suffix / neighbors[j].weight - 1.0);
                cycle_offdiag += scale * (population / 2.0 - 1.0);
            }
        }
        if (!(static_cast<double>(d) * (cycle_score + cycle_offdiag) <=
              population * (k2_score + k2_offdiag))) return false;
    }

    random_stream random{seed};
    const auto bounded_random = [&](std::size_t bound) {
        const std::uint64_t n = static_cast<std::uint64_t>(bound);
        const std::uint64_t threshold = (std::uint64_t{0} - n) % n;
        std::uint64_t value;
        do value = random.next(); while (value < threshold);
        return static_cast<std::size_t>(value % n);
    };
    std::array<std::size_t, kCycleMaxDegree> order{};
    for (std::size_t i = 0; i < d; ++i) order[i] = i;
    for (std::size_t n = d; n > 1; --n)
        std::swap(order[n - 1], order[bounded_random(n)]);

    const double inverse_probability = population / 2.0;
    for (std::size_t k = 0; k < d; ++k) {
        const auto& left = neighbors[order[k]];
        const auto& right = neighbors[order[(k + 1) % d]];
        out(left.vertex, right.vertex,
            left.weight * right.weight / pivot_degree * inverse_probability);
    }
    return true;
}

// Coordinate otherwise unchanged GKS choices at exactly the two heaviest
// receivers. At receiver r, systematic PPS rounds the conditional inclusion
// probabilities of every still-open source in canonical source order. Each
// inclusion marginal is unchanged; sources rejected by both coordinated
// receivers draw independently from their remaining suffix exactly as in GKS.
//
// Every source i<d-1 emits one edge to a strictly larger index. Thus the d-1
// edges are acyclic, and repeatedly following an edge reaches d-1: the sample
// is connected for every realization, independently of the probability proof.
inline bool sample_coordinated_gks_clique(
        std::span<weighted_neighbor> neighbors, double pivot_degree,
        std::uint64_t seed, edge_emitter out) {
    const std::size_t d = neighbors.size();
    if (d < 2 || !(pivot_degree > 0.0)) return false;

    canonical_sort_neighbors(neighbors);

    struct source_state {
        double mass;
        std::size_t partner;
    };
    struct rounding_item {
        std::size_t source;
        double probability;
    };

    static thread_local std::vector<double> prefix;
    static thread_local std::vector<source_state> sources;
    static thread_local std::vector<rounding_item> rounding;
    prefix.resize(d);
    sources.resize(d - 1);
    prefix[0] = neighbors[0].weight;
    for (std::size_t i = 1; i < d; ++i)
        prefix[i] = prefix[i - 1] + neighbors[i].weight;

    std::size_t unassigned = 0;
    for (std::size_t source = 0; source + 1 < d; ++source) {
        const double suffix = prefix.back() - prefix[source];
        const double mass =
            neighbors[source].weight * suffix / pivot_degree;
        if (!std::isfinite(mass) || mass < 0.0) return false;
        // A zero-mass source still gets its structural increasing edge. Keep it
        // out of systematic rounding, where two zero masses would be ambiguous.
        sources[source] = {mass, mass == 0.0 ? source + 1 : d};
        unassigned += mass != 0.0;
    }

    random_stream random{seed};
    const std::size_t receiver_budget =
        std::min(d - 1, kCoordinatedGksReceivers);
    std::size_t processed_receivers = 0;
    std::size_t last_unprocessed_receiver = d - 1;
    for (std::size_t receiver = d;
         receiver-- > 1 && unassigned > 0 &&
         processed_receivers < receiver_budget;) {
        rounding.clear();
        const double receiver_weight = neighbors[receiver].weight;
        for (std::size_t source = 0; source < receiver; ++source) {
            if (sources[source].partner != d) continue;
            // Conditioning on rejection by all heavier receivers leaves the
            // exact support source+1,...,receiver.
            const double remaining = prefix[receiver] - prefix[source];
            const double probability = std::clamp(
                remaining > 0.0 ? receiver_weight / remaining : 1.0,
                0.0, 1.0);
            rounding.push_back({source, probability});
        }

        constexpr double epsilon =
            32.0 * std::numeric_limits<double>::epsilon();
        auto settle = [&](const rounding_item& item) {
            if (item.probability <= epsilon) return true;
            if (item.probability >= 1.0 - epsilon) {
                sources[item.source].partner = receiver;
                --unassigned;
                return true;
            }
            return false;
        };
        std::size_t fractional_count = 0;
        for (const rounding_item& item : rounding)
            if (!settle(item)) rounding[fractional_count++] = item;
        rounding.resize(fractional_count);

        // One random lattice phase gives each interval exactly its length as
        // an inclusion probability while making the receiver's total in-count
        // floor(sum p) or ceil(sum p). Source order is the accepted
        // stratification; there is no pairwise rounding or adaptive frontier.
        const double phase = random.next_unit();
        double cumulative = 0.0;
        for (const rounding_item& item : rounding) {
            const double end = cumulative + item.probability;
            const double first_point =
                phase + std::ceil(cumulative - phase);
            if (first_point < end) {
                sources[item.source].partner = receiver;
                --unassigned;
            }
            cumulative = end;
        }
        ++processed_receivers;
        last_unprocessed_receiver = receiver - 1;
    }

    for (std::size_t source = 0; source + 1 < d; ++source) {
        auto& state = sources[source];
        if (state.partner != d) continue;
        const double remaining = source < last_unprocessed_receiver
            ? prefix[last_unprocessed_receiver] - prefix[source]
            : 0.0;
        if (source >= last_unprocessed_receiver || !(remaining > 0.0)) {
            // Exact arithmetic reaches this only when the final eligible
            // receiver had conditional probability one.
            state.partner = source + 1;
            continue;
        }
        const double target =
            prefix[source] + random.next_unit() * remaining;
        auto first = prefix.begin() +
            static_cast<std::ptrdiff_t>(source) + 1;
        auto last = prefix.begin() +
            static_cast<std::ptrdiff_t>(last_unprocessed_receiver) + 1;
        const auto it = std::upper_bound(first, last, target);
        state.partner = std::min(
            last_unprocessed_receiver,
            static_cast<std::size_t>(it - prefix.begin()));
    }

    for (std::size_t source = 0; source + 1 < d; ++source) {
        const auto receiver = sources[source].partner;
        out(neighbors[source].vertex, neighbors[receiver].vertex,
            sources[source].mass);
    }
    return true;
}

} // namespace apxchol::research_k2
