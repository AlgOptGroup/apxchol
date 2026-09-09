#include <gtest/gtest.h>
#include "apxchol/solver/elimination/elimination.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

using namespace apxchol;

namespace {

std::vector<deferred_edge> reference_tree_sample(
        std::vector<weighted_neighbor> neighbors,
        double deg,
        std::uint64_t seed) {
    if (neighbors.empty()) return {};
    std::sort(neighbors.begin(), neighbors.end(),
              [](const auto& a, const auto& b) {
                  return a.weight != b.weight ? a.weight < b.weight
                                              : a.vertex < b.vertex;
              });
    std::vector<double> prefix(neighbors.size());
    prefix[0] = neighbors[0].weight;
    for (size_t i = 1; i < neighbors.size(); ++i)
        prefix[i] = prefix[i - 1] + neighbors[i].weight;

    std::vector<deferred_edge> result;
    edge_emitter out(result);
    random_stream rs{seed};
    for (size_t i = 0; i + 1 < neighbors.size(); ++i) {
        const double suffix_sum = prefix.back() - prefix[i];
        if (suffix_sum <= 0.0) continue;
        const double target = prefix[i] + rs.next_unit() * suffix_sum;
        const auto it = std::upper_bound(
            prefix.begin() + static_cast<std::ptrdiff_t>(i) + 1,
            prefix.end(), target);
        const size_t j = it == prefix.end()
            ? neighbors.size() - 1
            : static_cast<size_t>(it - prefix.begin());
        out(neighbors[i].vertex, neighbors[j].vertex,
            neighbors[i].weight * suffix_sum / deg);
    }
    return result;
}

} // namespace

TEST(EliminatorBounds, TreeReturnsDegMinusOne) {
    EXPECT_EQ(tree_elimination{}.max_clique_edges(0), 0u);
    EXPECT_EQ(tree_elimination{}.max_clique_edges(1), 0u);
    EXPECT_EQ(tree_elimination{}.max_clique_edges(5), 4u);
    // Exact-clique mode: full pair count below the threshold.
    tree_elimination xc{.exact_clique_max_degree = 8};
    EXPECT_EQ(xc.max_clique_edges(5), 10u);
    EXPECT_EQ(xc.max_clique_edges(9), 8u);   // above threshold: sampled, d-1
}

TEST(EliminatorBounds, BoundIsSafeForRandomInput) {
    // Empirical: actual clique edges emitted never exceeds max_clique_edges(deg).
    std::vector<weighted_neighbor> neighbors;
    std::vector<deferred_edge> buf;

    for (node_index deg = 2; deg <= 12; ++deg) {
        neighbors.clear();
        for (node_index i = 0; i < deg; ++i)
            neighbors.emplace_back(i, 1.0 + i);
        const double sum = double(deg) * (deg + 1) / 2.0;

        buf.clear();
        tree_elimination{}.sample_clique(neighbors, sum, /*seed=*/42 + deg,
                                         edge_emitter(buf));
        EXPECT_LE(buf.size(), tree_elimination{}.max_clique_edges(deg))
            << "tree, deg=" << deg;
    }
}

TEST(TreeSampler, MatchesIndependentExactReference) {
    // Retain coverage around the removed degree-512 directory/radix and
    // degree-2048 radix boundaries to guard the byte-identical cleanup.
    for (const node_index degree : {
            node_index{0}, node_index{1}, node_index{2}, node_index{3},
            node_index{15}, node_index{16}, node_index{17},
            node_index{511}, node_index{512},
            node_index{2047}, node_index{2048}, node_index{2049}}) {
        for (int shape = 0; shape < 3; ++shape) {
            std::vector<weighted_neighbor> input;
            input.reserve(degree);
            double sum = 0.0;
            for (node_index i = 0; i < degree; ++i) {
                double weight = 1.0;
                if (shape == 1) {
                    weight = i % 17 == 0
                        ? 1.0 + static_cast<double>(i) * 0.25
                        : 0.001 * static_cast<double>(1 + (i % 11));
                } else if (shape == 2) {
                    weight = i + 4 >= degree
                        ? 1.0 + static_cast<double>(i)
                        : 1e-12 * static_cast<double>(1 + (i % 7));
                }
                input.push_back({degree - 1 - i, weight});
                sum += weight;
            }

            for (const std::uint64_t seed : {
                    std::uint64_t{0}, std::uint64_t{42},
                    std::uint64_t{0x123456789abcdef0ULL}}) {
                const auto expected = reference_tree_sample(input, sum, seed);
                auto production_input = input;
                std::vector<deferred_edge> actual;
                tree_elimination{}.sample_clique(production_input, sum, seed,
                                                  edge_emitter(actual));

                ASSERT_EQ(actual.size(), expected.size());
                for (size_t i = 0; i < expected.size(); ++i) {
                    EXPECT_EQ(actual[i].u, expected[i].u);
                    EXPECT_EQ(actual[i].v, expected[i].v);
                    EXPECT_EQ(std::bit_cast<std::uint64_t>(actual[i].w),
                              std::bit_cast<std::uint64_t>(expected[i].w));
                }
            }
        }
    }
}

TEST(TreeSampler, SuffixUpperBoundPreservesExactBoundarySemantics) {
    const double inf = std::numeric_limits<double>::infinity();
    const double tiny = std::numeric_limits<double>::denorm_min();
    for (const std::vector<double> cdf : {
            std::vector<double>{}, {1.0}, {-0.0, 0.0, 0.0, 1.0},
            {tiny, tiny, 2*tiny, 1.0}, {1.0, 1.0, 1.0, 2.0, 3.0},
            {0x1p53, 0x1p53 + 1.0, 0x1p53 + 2.0, inf}}) {
        std::vector<double> targets{-inf, -0.0, 0.0, 0.5, 1.0, inf,
                                    std::numeric_limits<double>::quiet_NaN()};
        for (double x : cdf) {
            targets.push_back(x);
            targets.push_back(std::nextafter(x, -inf));
            targets.push_back(std::nextafter(x, inf));
        }
        for (std::size_t start=0; start<=cdf.size(); ++start) {
            const auto suffix=std::span<const double>(cdf).subspan(start);
            for (double target : targets) {
                const auto expected=std::upper_bound(suffix.begin(), suffix.end(), target);
                EXPECT_EQ(detail::suffix_upper_bound(suffix,target),
                          static_cast<std::size_t>(expected-suffix.begin()));
            }
        }
    }
}

TEST(TreeSampler, SuffixUpperBoundMatchesAllRandomizedSuffixes) {
    std::mt19937_64 rng(42);
    for (std::size_t size=0; size<=257; ++size) {
        std::vector<double> cdf(size);
        double total=0;
        for (auto& x : cdf) {
            total += static_cast<double>(rng()%5); // includes repeated CDF entries
            x=total;
        }
        for (std::size_t start=0; start<=size; ++start) {
            const auto suffix=std::span<const double>(cdf).subspan(start);
            for (unsigned trial=0; trial<8; ++trial) {
                const double target=static_cast<double>(rng()%(2*size+3))/2;
                const auto expected=std::upper_bound(suffix.begin(), suffix.end(), target);
                ASSERT_EQ(detail::suffix_upper_bound(suffix,target),
                          static_cast<std::size_t>(expected-suffix.begin()));
            }
        }
    }
}

TEST(CycleSampler, PreservesDegreeTwoAndDeclaredCapacity) {
    for (auto kind : {clique_sampler::trace_cycle, clique_sampler::heavy_core_k2}) {
        tree_elimination rule{.sampler = kind};
        for (node_index d = 0; d <= 10; ++d) {
            std::vector<weighted_neighbor> n;
            for (node_index i = 0; i < d; ++i) n.push_back({i, double(i + 1)});
            std::vector<deferred_edge> edges;
            rule.sample_clique(n, 60., 42, edge_emitter(edges));
            EXPECT_LE(edges.size(), rule.max_clique_edges(d));
            EXPECT_EQ(edges.size(), d < 2 ? 0u : d == 2 ? 1u : size_t(d));
            if (d == 2) EXPECT_DOUBLE_EQ(edges[0].w, 2. / 60.);
            for (auto edge : edges) {
                EXPECT_NE(edge.u, edge.v);
                EXPECT_GT(edge.w, 0.);
                EXPECT_TRUE(std::isfinite(edge.w));
            }
        }
        rule.exact_clique_max_degree = 4;
        EXPECT_EQ(rule.max_clique_edges(4), 6u);
    }
}

TEST(CycleSampler, SeededReferenceLawsAndCanonicalTieOrdering) {
    // Frozen research outputs: six profile/rule combinations, 16 seeds each.
    // Separate private extraction validation compared 3,584 complete outputs.
    const std::uint64_t expected[] = {
        0xdc8bb4832f53e113ULL, 0x3cacb43c9631e69eULL, 0x9f0703307804cf80ULL,
        0xdc8bb4832f53e113ULL, 0x5e91b067ce624f22ULL, 0xc36e0afa90b21ff4ULL};
    size_t test = 0;
    for (auto kind : {clique_sampler::trace_cycle, clique_sampler::heavy_core_k2}) {
        for (int profile = 0; profile < 3; ++profile) {
            std::uint64_t hash = 1469598103934665603ULL;
            for (unsigned seed = 0; seed < 16; ++seed) {
                std::vector<weighted_neighbor> n;
                double pivot = 3.; // SDDM excess must not be replaced by sum(a).
                for (int i = 0; i < 8; ++i) {
                    const double a = profile == 0 ? 1. : profile == 1 ? double(i+1)
                                                                             : (i < 5 ? .001 : 1.);
                    n.push_back({node_index(7-i), a}); pivot += a;
                }
                auto reversed = n;
                std::reverse(reversed.begin(), reversed.end());
                std::vector<deferred_edge> edges, again;
                tree_elimination rule{.sampler = kind};
                rule.sample_clique(n, pivot, seed, edge_emitter(edges));
                rule.sample_clique(reversed, pivot, seed, edge_emitter(again));
                ASSERT_EQ(edges.size(), again.size());
                for (size_t k = 0; k < edges.size(); ++k) {
                    EXPECT_EQ(edges[k].u, again[k].u);
                    EXPECT_EQ(edges[k].v, again[k].v);
                    EXPECT_DOUBLE_EQ(edges[k].w, again[k].w);
                    for (auto x : {std::uint64_t(edges[k].u), std::uint64_t(edges[k].v),
                                   std::bit_cast<std::uint64_t>(edges[k].w)}) {
                        hash ^= x; hash *= 1099511628211ULL;
                    }
                }
            }
            EXPECT_EQ(hash, expected[test++]);
        }
    }
}

namespace {
// Independent tiny-star oracle. Enumerate every core permutation and parent
// outcome, then form tr((C^+ X)^2) from edge bilinear forms. For zero-sum edge
// vectors b,f, b^T C^+ f = sum_v b_v f_v/a_v when pivot=sum(a).
long double enumerated_trace_cycle_error(const std::vector<double>& a, size_t cut) {
    const size_t d = a.size(), h = d-cut;
    const long double total = std::accumulate(a.begin(), a.end(), 0.L);
    struct edge { size_t i,j; long double w; };
    std::vector<size_t> cycle(h);
    std::iota(cycle.begin(), cycle.end(), cut);
    std::vector<edge> edges;
    long double result = 0., permutations = 0.;
    do {
        edges.clear();
        for (size_t k = 0; k < h; ++k) {
            const auto i = cycle[k], j = cycle[(k+1)%h];
            edges.push_back({i,j,a[i]*a[j]/total*(h-1)/2});
        }
        const auto parents = [&](auto&& self, size_t i, long double probability) -> void {
            if (i == cut) {
                long double second = 0.;
                for (auto e : edges) for (auto f : edges) {
                    long double inner = 0.;
                    if (e.i == f.i) inner += 1.L/a[e.i];
                    if (e.i == f.j) inner -= 1.L/a[e.i];
                    if (e.j == f.i) inner -= 1.L/a[e.j];
                    if (e.j == f.j) inner += 1.L/a[e.j];
                    second += e.w*f.w*inner*inner;
                }
                result += probability*(second-(d-1));
                return;
            }
            long double mass = 0.;
            for (size_t j = i+1; j < d; ++j) mass += a[i]+a[j];
            for (size_t j = i+1; j < d; ++j) {
                const long double q = (a[i]+a[j])/mass;
                edges.push_back({i,j,(a[i]*a[j]/total)/q});
                self(self, i+1, probability*q);
                edges.pop_back();
            }
        };
        parents(parents, 0, 1.);
        ++permutations;
    } while (std::next_permutation(cycle.begin(), cycle.end()));
    return result/permutations;
}
}

TEST(CycleSampler, TraceObjectiveMatchesEveryOutcomeAndBestSuffix) {
    for (const auto& weights : {std::vector<double>{1,1,1,1,1},
                               std::vector<double>{.01,.1,1,2,3},
                               std::vector<double>{1,1,1,1,100}}) {
        std::vector<weighted_neighbor> n;
        for (size_t i = 0; i < weights.size(); ++i) n.push_back({node_index(i),weights[i]});
        detail::trace_cycle_plan plan;
        plan.prepare(n);
        long double best = std::numeric_limits<long double>::infinity();
        double parents = 0.;
        size_t best_cut = 0;
        for (size_t cut = 0; cut+2 < n.size(); ++cut) {
            const auto exact = enumerated_trace_cycle_error(weights, cut);
            const double formula = (parents+plan.cycle_numerator(cut))/(plan.total*plan.total);
            EXPECT_NEAR(double(exact), formula, 2e-12);
            if (exact < best) { best = exact; best_cut = cut; }
            parents += plan.parent_numerator(cut);
            double qsum = 0.;
            for (size_t j = cut+1; j < n.size(); ++j) {
                const auto q = plan.probability(cut,j);
                qsum += q;
                // Arbitrary SDDM pivot scales the clique mean, not the law.
                EXPECT_NEAR(q*plan.parent_weight(n,137.,cut,j),
                            n[cut].weight*n[j].weight/137.,1e-14);
            }
            EXPECT_NEAR(qsum,1.,1e-14);
        }
        EXPECT_EQ(plan.cut,best_cut);
        EXPECT_NEAR(plan.best_score,double(best),2e-12);
    }
}

TEST(CycleSampler, PreservesRepresentableSubnormalEdges) {
    for (auto kind : {clique_sampler::trace_cycle, clique_sampler::heavy_core_k2}) {
        const double tiny = std::sqrt(double(std::numeric_limits<pool_value_t>::min())) * .1;
        std::vector<weighted_neighbor> n{{0,tiny},{1,tiny},{2,tiny}};
        std::vector<deferred_edge> edges;
        tree_elimination rule{.sampler=kind};
        ASSERT_NO_THROW(rule.sample_clique(n,1.,0,edge_emitter(edges)));
        ASSERT_EQ(edges.size(),3u);
        for (const auto& edge : edges) {
            EXPECT_DOUBLE_EQ(edge.w,tiny*tiny);
            const directed_pool_edge stored{edge.v,static_cast<pool_value_t>(edge.w)};
            EXPECT_GT(stored.w,0);
            EXPECT_EQ(std::fpclassify(stored.w),FP_SUBNORMAL);
        }
    }
}

TEST(CycleSampler, InputOnlyFallbackMatchesGksAtEverySeed) {
    const double tiny = std::sqrt(double(std::numeric_limits<pool_value_t>::denorm_min())) * .25;
    const std::vector<std::vector<weighted_neighbor>> profiles{
        {{0,tiny},{1,tiny},{2,tiny}}, // prospective cycle edges round to zero
        {{0,0.},{1,1.},{2,2.},{3,3.}}, // a rounded-zero neighbor from earlier fill
        {{0,0.},{1,0.},{2,0.}},
        {{0,std::numeric_limits<double>::denorm_min()},{1,1.},{2,1e300}} // normalized ratio underflows
    };
    for (auto kind : {clique_sampler::trace_cycle, clique_sampler::heavy_core_k2})
        for (const auto& profile : profiles) for (std::uint64_t seed=0;seed<16;++seed) {
            auto n=profile,reference=profile;
            const double D=profile.back().weight>1e200?1e300:1.;
            std::vector<deferred_edge> got{{7,8,9.}},expected=got;
            tree_elimination{}.sample_clique(reference,D,seed,edge_emitter(expected));
            ASSERT_NO_THROW((tree_elimination{.sampler=kind}.sample_clique(n,D,seed,edge_emitter(got))));
            ASSERT_EQ(got.size(),expected.size());
            for(std::size_t i=0;i<got.size();++i) {
                EXPECT_EQ(got[i].u,expected[i].u);EXPECT_EQ(got[i].v,expected[i].v);
                EXPECT_DOUBLE_EQ(got[i].w,expected[i].w);
            }
        }
}

TEST(CycleSampler, InvalidInputsNeverBecomeNumericalFallbacks) {
    for (auto kind : {clique_sampler::trace_cycle,clique_sampler::heavy_core_k2}) {
        for(double bad : {-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
            std::vector<weighted_neighbor> n{{0,0.},{1,1.},{2,bad}};
            std::vector<deferred_edge> got{{7,8,9.}};
            EXPECT_THROW((tree_elimination{.sampler=kind}.sample_clique(n,1.,0,edge_emitter(got))),std::domain_error);
            ASSERT_EQ(got.size(),1u);
        }
        std::vector<weighted_neighbor> n{{0,1e200},{1,1e200},{2,1e200}};
        std::vector<deferred_edge> got;
        EXPECT_THROW((tree_elimination{.sampler=kind}.sample_clique(n,1.,0,edge_emitter(got))),std::domain_error);
        EXPECT_TRUE(got.empty());
        // A zero neighbor must not let GKS fallback hide a later overflow.
        n={{0,0.},{1,1e200},{2,1e200}};
        EXPECT_THROW((tree_elimination{.sampler=kind}.sample_clique(n,1.,0,edge_emitter(got))),std::domain_error);
        EXPECT_TRUE(got.empty());
    }
}

TEST(CycleSampler, RejectsNonpositiveAndNonfinitePoolEdges) {
    for (double value : {0.,-1.,std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()})
        EXPECT_THROW(detail::require_pool_edge(value),std::domain_error);
    EXPECT_NO_THROW(detail::require_pool_edge(std::numeric_limits<pool_value_t>::denorm_min()));
    EXPECT_NO_THROW(detail::require_pool_edge(std::numeric_limits<pool_value_t>::max()));
    if constexpr (sizeof(pool_value_t)<sizeof(double))
        EXPECT_THROW(detail::require_pool_edge(double(std::numeric_limits<pool_value_t>::max())*2),std::domain_error);
}

TEST(CycleSampler, G3CircuitRepresentableSubnormalParent) {
    // Canonical degree-seven star captured at the original G3 trace failure.
    std::vector<weighted_neighbor> n{{639860,5.387351450834748e-36},
        {639844,7.414873919759678e-25},{642324,7.07475587106445e-15},
        {639848,1.789436806875046e-14},{640350,1.271915721190453e-05},
        {641832,.2363596986899855},{641338,.47270159004953277}};
    constexpr double D=21959.66972560798;
    detail::trace_cycle_plan plan;plan.prepare(n);ASSERT_EQ(plan.cut,4u);
    EXPECT_NO_THROW(detail::require_pool_edge(plan.parent_weight(n,D,0,1)));
    tree_elimination rule{.sampler=clique_sampler::trace_cycle};
    std::vector<deferred_edge> edges;
    ASSERT_NO_THROW(rule.sample_clique(n,D,13935637984054586637ULL,edge_emitter(edges)));
    ASSERT_EQ(edges.size(),7u);
    auto parent=std::find_if(edges.begin(),edges.end(),[](const auto& edge){return edge.u==639860;});
    ASSERT_NE(parent,edges.end());
    EXPECT_GT(static_cast<pool_value_t>(parent->w),0);
    if constexpr(sizeof(pool_value_t)==4)
        EXPECT_EQ(std::fpclassify(static_cast<pool_value_t>(parent->w)),FP_SUBNORMAL);
}

TEST(CycleSampler, G3CircuitZeroRoundingUsesOriginalGksSeed) {
    std::vector<weighted_neighbor> n{{642509,5.826756152371164e-33},
        {640531,5.998751593509741e-19},{642506,1.1017429675086099e-16},
        {642504,3.755542722575982e-13},{641515,1.8731658769788402e-10}};
    constexpr double D=21959.66929534227;
    detail::trace_cycle_plan plan;plan.prepare(n);ASSERT_EQ(plan.cut,2u);
    if constexpr(sizeof(pool_value_t)==4) {
        EXPECT_EQ(static_cast<pool_value_t>(plan.parent_weight(n,D,0,1)),0);
        EXPECT_EQ(static_cast<pool_value_t>(plan.parent_weight(n,D,0,4)),0);
        for(std::uint64_t seed : {5420014876489129564ULL,0ULL,1ULL,42ULL}) {
            auto copy=n;std::vector<deferred_edge> expected,got;
            tree_elimination{}.sample_clique(copy,D,seed,edge_emitter(expected));
            tree_elimination{.sampler=clique_sampler::trace_cycle}.sample_clique(n,D,seed,edge_emitter(got));
            ASSERT_EQ(got.size(),expected.size());
            for(std::size_t i=0;i<got.size();++i) {
                EXPECT_EQ(got[i].u,expected[i].u);EXPECT_EQ(got[i].v,expected[i].v);
                EXPECT_DOUBLE_EQ(got[i].w,expected[i].w);
            }
            EXPECT_EQ(static_cast<pool_value_t>(got.front().w),0);
        }
    } else {
        std::vector<deferred_edge> got;
        ASSERT_NO_THROW((tree_elimination{.sampler=clique_sampler::trace_cycle}.sample_clique(n,D,42,edge_emitter(got))));
        EXPECT_EQ(got.size(),5u);
    }
}
