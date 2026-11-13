// Law of the Metal trace-cycle star sampler (src/metal_setup/gks_star.inc):
// every neighbour pair (i, j) receives expected fill weight a_i a_j / D, one
// fill per neighbour, and the excess updates match GKS.
#include <gtest/gtest.h>

#include "../src/metal_setup/layout.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>
#include <vector>

namespace {
using math_t = sparse_round::compensated_math;
using wide = math_t::wide;
using row_t = sparse_round::neighbor<wide>;

using pair_map = std::map<std::pair<unsigned, unsigned>, double>;
struct Sink {
    pair_map* trial;
    unsigned* fills;
    void factor(unsigned, unsigned, float) const {}
    void fill(unsigned, unsigned a, unsigned b, float value) const {
        (*trial)[{a, b}] += value; ++*fills;
    }
    void excess(unsigned, unsigned, wide) const {}
};

// Mean fill weight per pair over `trials` seeds (second moments in `squares`).
pair_map run(pair_map& squares, const std::vector<float>& w, float excess,
                                                    unsigned trace, int trials, unsigned& fills_per_star,
                                                    unsigned& trace_rows) {
    const math_t math{};
    pair_map sums;
    squares.clear();
    fills_per_star = 0; trace_rows = 0;
    const unsigned d = unsigned(w.size());
    for (int t = 0; t < trials; ++t) {
        std::vector<row_t> rows(d);
        wide degree = math.from_float(0);
        for (unsigned i = 0; i < d; ++i) {
            rows[i] = {i + 1, i, math.from_float(w[i])};
            degree = math.add(degree, rows[i].value);
        }
        const sparse_round::star_input<wide> p{0, 0, d, d + 1, math.from_float(excess),
                                               sparse_round::seed_type(1000003u * unsigned(t) + 17u), trace};
        sparse_round::row_result r{}; r.entries = d;
        auto prepared = sparse_round::combine_star(math, p, rows.data(),
                                                   sparse_round::canonical_star<wide>{r, degree});
        EXPECT_EQ(prepared.result.status, 0u);
        std::sort(rows.begin(), rows.end(), [&](const row_t& a, const row_t& b) {
            return sparse_round::weight_before(math, a, b);
        });
        std::vector<wide> prefix(d);
        auto res = sparse_round::prepare_sample(math, rows.data(), prefix.data(), prepared.result);
        if (trace) res = sparse_round::prepare_trace(math, rows.data(), res, p.seed);
        EXPECT_EQ(res.status, 0u);
        if (res.cut < res.entries) ++trace_rows;
        unsigned fills = 0;
        pair_map trial;
        const Sink sink{&trial, &fills};
        for (unsigned i = 0; i < res.entries; ++i)
            EXPECT_EQ(sparse_round::sample_item(math, sink, p, rows.data(), prefix.data(),
                          sparse_round::canonical_star<wide>{res, prepared.degree}, i), 0u);
        fills_per_star = std::max(fills_per_star, fills);
        for (const auto& [k, v] : trial) { sums[k] += v; squares[k] += v * v; }
    }
    for (auto& [k, v] : sums) v /= trials;
    for (auto& [k, v] : squares) v /= trials;
    return sums;
}

void expect_unbiased(const std::vector<float>& w, float excess, unsigned trace) {
    double D = excess;
    for (float x : w) D += x;
    unsigned fills = 0, trace_rows = 0;
    const int trials = 200000;
    pair_map squares;
    const auto mean = run(squares, w, excess, trace, trials, fills, trace_rows);
    const unsigned d = unsigned(w.size());
    for (unsigned i = 0; i < d; ++i)
        for (unsigned j = i + 1; j < d; ++j) {
            const double expected = double(w[i]) * w[j] / D;
            const auto it = mean.find({i + 1, j + 1});
            ASSERT_NE(it, mean.end()) << "pair " << i << "," << j << " never sampled";
            const double got = it->second;
            const double sd = std::sqrt(std::max(0.0, squares.at({i + 1, j + 1}) - got * got) / trials);
            // Five standard errors, plus float storage rounding.
            EXPECT_NEAR(got, expected, 5 * sd + 1e-6 * expected) << "pair " << i << "," << j << " d=" << d;
        }
    if (trace && d >= 3) {
        EXPECT_EQ(fills, d);
        EXPECT_EQ(trace_rows, unsigned(trials));
    }
}
} // namespace

TEST(MetalTraceStar, UnbiasedSkewedStar) { expect_unbiased({1, 2, 3, 4, 5, 10, 10, 10}, 0.5f, 1); }
TEST(MetalTraceStar, UnbiasedEqualStar) { expect_unbiased({2, 2, 2, 2, 2, 2}, 0.0f, 1); }
TEST(MetalTraceStar, UnbiasedTriangle) { expect_unbiased({1, 3, 7}, 0.0f, 1); }
TEST(MetalTraceStar, UnbiasedWideRange) { expect_unbiased({1e-3f, 0.5f, 1, 1, 4, 4, 4, 9, 30, 31, 32}, 2.0f, 1); }
TEST(MetalTraceStar, GksUnchanged) { expect_unbiased({1, 2, 3, 4, 5, 10, 10, 10}, 0.5f, 0); }
