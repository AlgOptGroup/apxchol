#include <gtest/gtest.h>
#include "scoped_env.h"
#include "scoped_omp_threads.h"
#include "apxchol/graph/conversions.h"
#include "apxchol/operator_class.h"
#include "apxchol/solver/detail/gpu_diagnostics.h"
#include "apxchol/solver/elimination/gpu_round_shadow.h"
#include "apxchol/solver/gpu_block_frontend.h"
#include "apxchol/solver/solve.h"
#include "apxchol/solver/detail/solver_route.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <vector>

using csc_access = apxchol::detail::sparse_csc_access;

namespace {
using apxchol::node_index;
using matrix = Eigen::SparseMatrix<double>;
using host_graph = apxchol::graph;
using apxchol::detail::gpu_round_shadow_input;

struct weighted_edge { int u, v; double weight; };

matrix make_matrix(int n, const std::vector<weighted_edge>& edges, double shift = 0.0) {
    std::vector<Eigen::Triplet<double>> entries;
    std::vector<double> diagonal(n, shift);
    for (const auto& e : edges) {
        entries.emplace_back(e.u, e.v, -e.weight);
        entries.emplace_back(e.v, e.u, -e.weight);
        diagonal[e.u] += e.weight; diagonal[e.v] += e.weight;
    }
    for (int v = 0; v < n; ++v) entries.emplace_back(v, v, diagonal[v]);
    matrix A(n, n); A.setFromTriplets(entries.begin(), entries.end());
    return A;
}

[[maybe_unused]] matrix selector_matrix() {
    std::vector<weighted_edge> edges;
    for (int u = 0; u < 12; ++u)
        for (int v = u + 1; v < 12; ++v) edges.push_back({u, v, 1.0});
    for (int u = 12; u < 127; ++u)
        if (u != 62 && u != 63 && u != 64) edges.push_back({u, u + 1, 0.5});
    for (int u = 14; u < 120; u += 9) edges.push_back({u, u + 7, 2.0});
    edges.push_back({0, 128, 0.0}); // Stored zero remains a topological incidence.
    return make_matrix(132, edges); // Includes disconnected components and isolates.
}

[[maybe_unused]] gpu_round_shadow_input host_snapshot(const matrix& A, double reg_eps = 0.0) {
    auto graph = apxchol::make_graph(A);
    if (reg_eps > 0.0)
        for (node_index v = 0; v < graph.n(); ++v)
            graph.excess(v) = std::max(graph.excess(v), reg_eps * A.coeff(v, v));
    return apxchol::detail::make_gpu_round_shadow_input(
        graph, std::span<const node_index>{}, 0);
}

[[maybe_unused]] void expect_same_snapshot(const gpu_round_shadow_input& got,
                          const gpu_round_shadow_input& expected) {
    ASSERT_EQ(got.vertex_count, expected.vertex_count);
    EXPECT_EQ(got.owner_offsets, expected.owner_offsets);
    EXPECT_EQ(got.active, expected.active);
    EXPECT_EQ(got.pivots, expected.pivots);
    EXPECT_EQ(got.seeds, expected.seeds);
    ASSERT_EQ(got.excess.size(), expected.excess.size());
    for (std::size_t v = 0; v < got.excess.size(); ++v)
        EXPECT_EQ(std::bit_cast<std::uint64_t>(got.excess[v]),
                  std::bit_cast<std::uint64_t>(expected.excess[v])) << "excess vertex=" << v;
    ASSERT_EQ(got.incidences.size(), expected.incidences.size());
    for (std::size_t i = 0; i < got.incidences.size(); ++i) {
        EXPECT_EQ(got.incidences[i].owner, expected.incidences[i].owner) << i;
        EXPECT_EQ(got.incidences[i].neighbor, expected.incidences[i].neighbor) << i;
        EXPECT_EQ(std::bit_cast<std::uint64_t>(double(got.incidences[i].weight)),
                  std::bit_cast<std::uint64_t>(double(expected.incidences[i].weight))) << i;
    }
}

} // namespace

TEST(GpuDirectCscHost, RejectsUnsupportedStoredFormatsBeforeCuda) {
    scoped_omp_threads serial(1);
    using apxchol::detail::gpu_owned_csc_supported;
    const auto valid = make_matrix(3, {{0,1,1.0},{0,2,2.0},{1,2,0.5}});
    ASSERT_TRUE(gpu_owned_csc_supported(valid));
    auto spare_capacity = valid;
    spare_capacity.reserve(32);
    ASSERT_GT(spare_capacity.data().allocatedSize(), spare_capacity.data().size());
    EXPECT_TRUE(gpu_owned_csc_supported(spare_capacity));
    matrix resized(2, 3);
    const std::vector<Eigen::Triplet<double>> entries{{0,0,1.0},{1,1,1.0},{0,2,1.0}};
    resized.setFromTriplets(entries.begin(), entries.end());
    resized.conservativeResize(2, 2);
    ASSERT_TRUE(resized.isCompressed());
    ASSERT_EQ(resized.nonZeros(), 2);
    ASSERT_EQ(resized.data().size(), 3);
    EXPECT_TRUE(gpu_owned_csc_supported(resized));
    std::vector<matrix> unsupported;
    unsupported.emplace_back(0, 0);
    unsupported.emplace_back(2, 3);
    auto uncompressed = valid; uncompressed.uncompress();
    ASSERT_FALSE(uncompressed.isCompressed()); unsupported.push_back(std::move(uncompressed));
    for (bool lower : {false, true}) {
        matrix triangular(2, 2);
        std::vector<Eigen::Triplet<double>> entries{{0,0,1.0},{1,1,1.0}};
        entries.emplace_back(lower ? 1 : 0, lower ? 0 : 1, -1.0);
        triangular.setFromTriplets(entries.begin(), entries.end());
        unsupported.push_back(std::move(triangular));
    }
    for (int corruption = 0; corruption < 8; ++corruption) {
        auto bad = valid;
        switch (corruption) {
        case 0: bad.innerIndexPtr()[1] = bad.innerIndexPtr()[0]; break; // duplicate
        case 1: std::swap(bad.innerIndexPtr()[0], bad.innerIndexPtr()[1]); break;
        case 2: bad.innerIndexPtr()[1] = -1; break;
        case 3: bad.innerIndexPtr()[1] = 3; break;
        case 4: bad.outerIndexPtr()[1] = -1; break;
        case 5: bad.outerIndexPtr()[3] = bad.data().size() + 1; break;
        case 6: bad.outerIndexPtr()[3] = -1; break;
        case 7: bad.outerIndexPtr()[0] = 1; break;
        }
        unsupported.push_back(std::move(bad));
    }
    ASSERT_EQ(unsupported.size(), 13u);
    unsupported[2].uncompress(); // Eigen copies may recompress; fix the final stored fixture.
    ASSERT_FALSE(unsupported[2].isCompressed());
    for (std::size_t i = 0; i < unsupported.size(); ++i) {
        SCOPED_TRACE(i);
        EXPECT_FALSE(gpu_owned_csc_supported(unsupported[i]));
#if defined(APXCHOL_USE_CUDA) && defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
        // No runtime/device probe: unsupported layouts must stop on the host.
        const auto got = apxchol::detail::gpu_initial_owned_csc_for_test(unsupported[i]);
        EXPECT_FALSE(got.eligible); EXPECT_TRUE(got.initial.incidences.empty());
#endif
    }
}

TEST(GpuDirectCscHost, CertifiedEligibilityMatchesRawForValidatedInputs) {
    scoped_omp_threads serial(1);
    using apxchol::detail::gpu_owned_csc_supported;
    std::vector<matrix> fixtures;
    fixtures.push_back(make_matrix(4, {{0,1,1.0},{1,2,2.0}}, 0.5));
    fixtures.emplace_back(0, 0);
    fixtures.emplace_back(4, 4); // Isolates, no stored entries.
    fixtures.push_back(make_matrix(3, {}, 1.0));
    auto tolerated = fixtures.front();
    tolerated.coeffRef(0,1) = -std::nextafter(1.0, 2.0);
    fixtures.push_back(tolerated);
    fixtures.push_back(make_matrix(3, {{0,1,std::numeric_limits<double>::denorm_min()}}, 1.0));
    auto paired_zero = make_matrix(3, {{0,1,0.0}}, 1.0);
    paired_zero.coeffRef(0,1) = -0.0;
    fixtures.push_back(paired_zero);
    // Validation ignores zeros; structural eligibility must still reject a
    // missing zero mate, including balanced but unrelated upper/lower zeros.
    for (bool lower : {false, true}) {
        auto one_zero = make_matrix(3, {}, 1.0);
        one_zero.coeffRef(lower ? 1 : 0, lower ? 0 : 1) = 0.0;
        one_zero.makeCompressed();
        const apxchol::detail::operator_view op(one_zero);
        EXPECT_EQ(op.scan().offdiag_nnz, 0);
        EXPECT_FALSE(gpu_owned_csc_supported(op));
        fixtures.push_back(one_zero);
    }
    auto unrelated_zeros = make_matrix(3, {}, 1.0);
    unrelated_zeros.coeffRef(1,0) = 0.0;
    unrelated_zeros.coeffRef(1,2) = -0.0;
    unrelated_zeros.makeCompressed();
    EXPECT_FALSE(gpu_owned_csc_supported(apxchol::detail::operator_view(unrelated_zeros)));
    fixtures.push_back(unrelated_zeros);
    auto lumped = make_matrix(3, {{0,1,1.0},{1,2,-1e-6}}, 1.0);
    const apxchol::detail::operator_view op_lumped(lumped);
    EXPECT_EQ(op_lumped.lumped(), 2);
    fixtures.push_back(lumped);
    for (std::size_t i = 0; i < fixtures.size(); ++i) {
        SCOPED_TRACE(i);
        const apxchol::detail::operator_view op(fixtures[i]);
        EXPECT_EQ(gpu_owned_csc_supported(op), gpu_owned_csc_supported(op.matrix()));
    }
    auto uncompressed = fixtures.front();
    uncompressed.uncompress();
    const apxchol::detail::operator_view op(uncompressed);
    EXPECT_FALSE(gpu_owned_csc_supported(op));
}

TEST(GpuDirectCscHost, CertifiedEligibilityRejectsDuplicateAndUnsortedZeroColumns) {
    scoped_omp_threads serial(1);
    using apxchol::detail::gpu_owned_csc_supported;
    for (bool duplicate : {false, true}) {
        auto A = make_matrix(4, {{0,2,0.0},{0,3,0.0}}, 1.0);
        ASSERT_EQ(A.innerIndexPtr()[1], 2);
        ASSERT_EQ(A.innerIndexPtr()[2], 3);
        if (duplicate) A.innerIndexPtr()[2] = 2;
        else std::swap(A.innerIndexPtr()[1], A.innerIndexPtr()[2]);
        // The numerical scan has no edges to reject. Strict layout validation
        // must run even when an operator_view was constructed successfully.
        const apxchol::detail::operator_view op(A);
        ASSERT_EQ(op.scan().offdiag_nnz, 0);
        EXPECT_FALSE(gpu_owned_csc_supported(op));
        EXPECT_FALSE(gpu_owned_csc_supported(A));
    }
}

TEST(GpuDirectCscHost, NumericalCertificateRejectsMissingTinyNonzeroMates) {
    scoped_omp_threads serial(1);
    for (double weight : {1.0, std::numeric_limits<double>::denorm_min()}) {
        for (bool lower : {false, true}) {
            auto A = make_matrix(3, {}, 1.0);
            A.coeffRef(lower ? 1 : 0, lower ? 0 : 1) = -weight;
            A.makeCompressed();
            EXPECT_THROW((void)apxchol::detail::operator_view(A), std::invalid_argument);
        }
        auto balanced = make_matrix(3, {}, 1.0);
        balanced.coeffRef(1,0) = -weight;
        balanced.coeffRef(1,2) = -weight;
        balanced.makeCompressed();
        EXPECT_THROW((void)apxchol::detail::operator_view(balanced), std::invalid_argument);
    }
}

TEST(GpuDirectCsc, InitialArraysMatchHostBuilderIncludingRoundingAndSignedZero) {
#if !defined(APXCHOL_USE_CUDA) || !defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
    GTEST_SKIP() << "CUDA direct CSC experiment and test hooks required";
#else
    if (!apxchol::detail::gpu_round_shadow_runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    std::vector<matrix> fixtures;
    fixtures.push_back(make_matrix(9, {{0,3,2.0},{1,3,0.5},{3,4,0x1p-10},{5,6,4.0}}));
    fixtures.push_back(make_matrix(1, {}));
    fixtures.push_back(make_matrix(1, {}, 2.5));
    fixtures.emplace_back(7, 7); // Compressed, all isolates, no stored diagonal.
    fixtures.push_back(make_matrix(5, {{0,1,1.0+0x1p-24},
        {0,2,std::nextafter(1.0+0x1p-24, 2.0)}, {0,3,0x1p24+1.0}, {3,4,0x1p-140}}, 0.5));
    auto asymmetric = make_matrix(4, {{0,1,1.0},{2,3,1.0+0x1p-25}});
    for (int v = 0; v < 4; ++v) asymmetric.coeffRef(v,v) = 2.0;
    asymmetric.coeffRef(0,1) = -std::nextafter(1.0, 2.0);
    asymmetric.coeffRef(2,3) = -std::nextafter(1.0+0x1p-25, 2.0);
    const apxchol::detail::operator_view tolerated(asymmetric);
    EXPECT_EQ(tolerated.lumped(), 0);
    fixtures.push_back(asymmetric); // Tolerated asymmetry still uses LOWER weights twice.
    auto signed_zero = make_matrix(3, {{0,1,-0.0}}, 1.0);
    signed_zero.coeffRef(1,0) = +0.0; signed_zero.coeffRef(0,1) = -0.0;
    fixtures.push_back(signed_zero);
    for (int threads : {1, 3}) {
        scoped_omp_threads team(threads);
        for (std::size_t i = 0; i < fixtures.size(); ++i) {
            SCOPED_TRACE(threads);
            SCOPED_TRACE(i);
            const auto& A = fixtures[i];
            ASSERT_TRUE(apxchol::detail::gpu_owned_csc_supported(A));
            const auto expected = host_snapshot(A);
            const auto got = apxchol::detail::gpu_initial_owned_csc_for_test(A);
            ASSERT_TRUE(got.eligible);
            expect_same_snapshot(got.initial, expected);
            EXPECT_EQ(got.sddm, std::any_of(expected.excess.begin(), expected.excess.end(),
                [](double x) { return x > 0.0; }));
            EXPECT_EQ(got.initial.active, std::vector<std::uint8_t>(A.rows(), 1));
            for (const auto& edge : got.initial.incidences) {
                const double canonical = static_cast<double>(static_cast<float>(
                    -A.coeff(std::max(edge.owner,edge.neighbor),std::min(edge.owner,edge.neighbor))));
                EXPECT_EQ(std::bit_cast<std::uint64_t>(double(edge.weight)), std::bit_cast<std::uint64_t>(canonical));
            }
        }
    }
    const auto asymmetry = host_snapshot(asymmetric);
    EXPECT_NE(std::bit_cast<std::uint64_t>(asymmetry.excess[0]),
              std::bit_cast<std::uint64_t>(asymmetry.excess[1]));
    const auto zeros = host_snapshot(signed_zero);
    ASSERT_EQ(zeros.incidences.size(), 2u);
    EXPECT_TRUE(std::signbit(zeros.incidences[0].weight));
    EXPECT_TRUE(std::signbit(zeros.incidences[1].weight));
#endif
}

TEST(GpuDirectCsc, ExcessUsesOriginalOrderedDoublesAndStrictThresholdBeforeRegularization) {
#if !defined(APXCHOL_USE_CUDA) || !defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
    GTEST_SKIP() << "CUDA direct CSC experiment and test hooks required";
#else
    if (!apxchol::detail::gpu_round_shadow_runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    scoped_omp_threads serial(1);
    constexpr double diagonal = 1e12;
    const double equal = diagonal - 1.0;
    auto A = make_matrix(6, {{0,1,equal},{2,3,std::nextafter(equal,diagonal)},
                            {4,5,std::nextafter(equal,0.0)}});
    for (int v = 0; v < 6; ++v) A.coeffRef(v,v) = diagonal;
    ASSERT_EQ(diagonal - equal, diagonal * 1e-12);
    const auto expected = host_snapshot(A);
    EXPECT_EQ(expected.excess[0], 0.0); EXPECT_EQ(expected.excess[2], 0.0);
    EXPECT_GT(expected.excess[4], 1.0);
    for (double reg : {0.0, 0.5e-12, 2e-12}) {
        SCOPED_TRACE(reg);
        const auto got = apxchol::detail::gpu_initial_owned_csc_for_test(A, reg);
        ASSERT_TRUE(got.eligible);
        expect_same_snapshot(got.initial, host_snapshot(A, reg));
        EXPECT_TRUE(got.sddm);
    }
    // Deliberate non-associative degree: retaining CSC row encounter order is
    // observable in the full fp64 excess even though the edge pool is fp32.
    auto ordered = make_matrix(5, {{0,1,0x1p40},{0,2,0x1p-13},{0,3,0x1p-13},{0,4,4.0}});
    ordered.coeffRef(0,0) = 0x1p40 + 16.0;
    const auto want = host_snapshot(ordered);
    EXPECT_EQ(want.excess[0], 12.0);
    expect_same_snapshot(apxchol::detail::gpu_initial_owned_csc_for_test(ordered).initial, want);
#endif
}

TEST(GpuDirectCsc, InvalidNumericalInputsFailWithoutPublishingAnInitialSnapshot) {
#if !defined(APXCHOL_USE_CUDA) || !defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
    GTEST_SKIP() << "CUDA direct CSC experiment and test hooks required";
#else
    if (!apxchol::detail::gpu_round_shadow_runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    scoped_omp_threads serial(1);
    const auto valid = make_matrix(2, {{0,1,1.0}}, 1.0);
    for (int fault = 0; fault < 4; ++fault) {
        auto A = valid;
        if (fault == 0) A.coeffRef(0,1) = std::numeric_limits<double>::quiet_NaN();
        if (fault == 1) A.coeffRef(0,0) = std::numeric_limits<double>::infinity();
        if (fault == 2) A.coeffRef(1,0) = 1.0; // Negative canonical graph weight.
        SCOPED_TRACE(fault);
        ASSERT_TRUE(apxchol::detail::gpu_owned_csc_supported(A));
        const double reg = fault == 3 ? std::numeric_limits<double>::infinity() : 0.0;
        EXPECT_THROW((void)apxchol::detail::gpu_initial_owned_csc_for_test(A, reg), std::invalid_argument);
    }
    const double overflow = 2.0 * static_cast<double>(std::numeric_limits<float>::max());
    auto A = make_matrix(2, {{0,1,overflow}});
    EXPECT_THROW((void)apxchol::detail::gpu_initial_owned_csc_for_test(A), std::invalid_argument);
#endif
}

TEST(GpuDirectCsc, FirstSelectorHasCorrectCandidatesAndIndependentProgress) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA direct CSC required";
#else
    if (!apxchol::detail::gpu_round_shadow_runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    scoped_omp_threads serial(1);
    const auto A = selector_matrix();
    const auto graph = apxchol::make_graph(A);
    ASSERT_TRUE(apxchol::detail::gpu_owned_csc_supported(A));
    std::vector<node_index> active(graph.n()); std::iota(active.begin(),active.end(),node_index{0});
    for (bool ties : {false, true}) for (double quantile : {0.0, 0.3, 0.8}) {
        SCOPED_TRACE(ties);
        SCOPED_TRACE(quantile);
        apxchol::detail::gpu_round_shadow_session owner(true);
        std::unique_ptr<apxchol::detail::gpu_block_frontend> direct;
        owner.initialize_owned_csc(apxchol::detail::gpu_owned_csc_host_buffers(A), direct);
        apxchol::partition_options options;
        options.degree_quantile = quantile;
        const auto got = direct->prepare(active, options, 42, 0, ties);
        const auto degrees = direct->host_active_degrees();
        std::size_t degree_sum = 0;
        for (node_index v : active) degree_sum += graph.adj_count(v);
        EXPECT_EQ(got.average_degree, double(degree_sum) / active.size());
        ASSERT_EQ(degrees.size(), active.size());
        for (node_index v : active) EXPECT_EQ(degrees[v], graph.adj_count(v));
        const auto candidate_view = direct->host_candidates();
        const std::vector<node_index> candidates(candidate_view.begin(),candidate_view.end());
        double threshold = options.degree_multiplier * got.average_degree;
        if (quantile > 0.0 && quantile < 1.0) {
            std::vector<node_index> sorted(degrees.begin(), degrees.end());
            std::sort(sorted.begin(), sorted.end());
            threshold = sorted[static_cast<std::size_t>(quantile * (sorted.size()-1))];
        }
        std::vector<node_index> wanted;
        for (node_index v : active) if (degrees[v] <= threshold) wanted.push_back(v);
        EXPECT_EQ(candidates, wanted);
        EXPECT_EQ(got.candidate_count, wanted.size());
        const auto selected = direct->select_block_greedy().data;
        ASSERT_FALSE(selected.empty());
        EXPECT_EQ(direct->select_block_greedy().data, selected);
        // The owned four-pass law need not finish maximality. Its exact
        // priority/snapshot contract is checked by GpuBoundedSelection.
        std::string receipt = std::to_string(direct->selected_degree_work());
        for (const auto vertex : selected) receipt += ":" + std::to_string(vertex);
        RecordProperty(std::string("direct_selection_") + std::to_string(ties) + "_" + std::to_string(quantile), receipt);
        std::vector<bool> picked(graph.n(),false);
        std::size_t degree_work = 0;
        for (node_index v : selected) {
            ASSERT_LT(v,graph.n());
            EXPECT_TRUE(std::binary_search(candidates.begin(),candidates.end(),v));
            ASSERT_FALSE(picked[v]); picked[v]=true; degree_work+=graph.adj_count(v);
        }
        for (node_index v : candidates) {
            bool neighbor_picked = false;
            for (const auto& edge : graph.neighbors(v)) neighbor_picked |= picked[edge.to];
            if (picked[v]) {
                EXPECT_FALSE(neighbor_picked);
            }
        }
        EXPECT_EQ(direct->selected_degree_work(), degree_work);
        EXPECT_EQ(direct->transfers().owned_residual_binds, 1u);
    }
#endif
}

TEST(GpuDirectCsc, InitialOwnerOutlivesInputAndRetiresEveryBorrowedSelectorRead) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA direct CSC required";
#else
    if (!apxchol::detail::gpu_round_shadow_runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    scoped_omp_threads serial(1);
    for (int operation = 0; operation < 5; ++operation) {
        SCOPED_TRACE(operation);
        auto owner = std::make_unique<apxchol::detail::gpu_round_shadow_session>(true);
        std::unique_ptr<apxchol::detail::gpu_block_frontend> direct;
        const auto original = selector_matrix();
        const auto graph = apxchol::make_graph(original);
        {
            auto temporary = original;
            ASSERT_TRUE(apxchol::detail::gpu_owned_csc_supported(temporary));
            owner->initialize_owned_csc(apxchol::detail::gpu_owned_csc_host_buffers(temporary), direct);
            temporary.setZero(); // The owner must retain neither values nor index pointers.
        }
        std::vector<node_index> active(graph.n()); std::iota(active.begin(),active.end(),node_index{0});
        apxchol::partition_options options;
        // Independent owned-policy oracle, built from the original host graph.
        // A generic CSR selector follows a different regional selection law.
        std::vector<node_index> degrees(graph.n()), candidates, expected_selected;
        std::size_t degree_sum = 0;
        for (auto v : active) { degrees[v] = graph.adj_count(v); degree_sum += degrees[v]; }
        // The oracle must resolve an unset quantile exactly as the library
        // does, or it grades the selector against a cap nobody asked for.
        const double q_cap = options.degree_quantile.value_or(
            apxchol::detail::degree_quantile_host_default);
        double threshold = options.degree_multiplier * double(degree_sum) / graph.n();
        if (q_cap > 0.0 && q_cap < 1.0) {
            auto sorted = degrees; std::sort(sorted.begin(), sorted.end());
            threshold = sorted[static_cast<std::size_t>(q_cap * (sorted.size() - 1))];
        }
        constexpr std::uint64_t seed = 42, phase = 0;
        auto key = [](node_index v) {
            std::uint64_t z = (seed ^ (phase * 0xD6E8FEB86659FD93ULL) ^ v) + 0x9E3779B97F4A7C15ULL;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            return z ^ (z >> 31);
        };
        auto precedes = [&](node_index a, node_index b) {
            if (degrees[a] != degrees[b]) return degrees[a] < degrees[b];
            const auto ha = key(a), hb = key(b);
            return ha != hb ? ha < hb : a < b;
        };
        std::vector<unsigned char> status(graph.n(), 2);
        for (auto v : active) if (degrees[v] <= threshold) { candidates.push_back(v); status[v] = 0; }
        for (unsigned pass = 0; pass < 4; ++pass) {
            std::vector<node_index> winners;
            for (auto v : candidates) if (status[v] == 0) {
                bool wins = true;
                for (const auto& edge : graph.neighbors(v))
                    if (status[edge.to] == 0 && precedes(edge.to, v)) wins = false;
                if (wins) winners.push_back(v);
            }
            if (winners.empty()) break;
            for (auto v : winners) status[v] = 1;
            for (auto v : winners) for (const auto& edge : graph.neighbors(v))
                if (status[edge.to] == 0) status[edge.to] = 2;
        }
        std::size_t expected_work = 0;
        for (auto v : candidates) if (status[v] == 1) {
            expected_selected.push_back(v); expected_work += degrees[v];
        }
        ASSERT_FALSE(expected_selected.empty());
        direct->prepare(active, options, seed, phase);
        const auto selected = direct->select_block_greedy().data;
        EXPECT_EQ(selected, expected_selected);
        EXPECT_EQ(direct->select_block_greedy().data, selected);
        const auto got_candidates = direct->host_candidates(), got_degrees = direct->host_active_degrees();
        EXPECT_EQ(std::vector<node_index>(got_candidates.begin(), got_candidates.end()), candidates);
        EXPECT_EQ(std::vector<node_index>(got_degrees.begin(), got_degrees.end()), degrees);
        EXPECT_EQ(direct->selected_degree_work(), expected_work);
        EXPECT_EQ(direct->transfers().owned_residual_binds, 1u);
        owner.reset();
        try {
            switch (operation) {
            case 0: (void)direct->prepare(active,options); break;
            case 1: (void)direct->select_block_greedy(); break;
            case 2: (void)direct->host_candidates(); break;
            case 3: (void)direct->host_active_degrees(); break;
            case 4: (void)direct->device_selection(); break;
            }
            ADD_FAILURE() << "retired numerical owner was accepted";
        } catch (const std::exception& error) {
            EXPECT_NE(std::string(error.what()).find("residual generation is retired"),std::string::npos) << error.what();
        }
    }
#endif
}

TEST(GpuDirectCsc, ConsumingSolveRetainsDefaultFp16DropAndOriginalResidualWithoutGraphHandback) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "32-bit-node CUDA operator construction required";
#else
    if (!apxchol::detail::gpu_round_shadow_runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    scoped_omp_threads serial(1);
    constexpr int n = 257;
    std::vector<weighted_edge> edges;
    for (int u = 0; u < n; ++u) for (int offset : {1,7}) edges.push_back({u,(u+offset)%n,1.0});
    for (double shift : {0.0, 1.0}) {
        SCOPED_TRACE(shift);
        const auto A = make_matrix(n,edges,shift);
        Eigen::VectorXd exact(n);
        for (int v = 0; v < n; ++v) exact[v] = std::sin(v + 0.25);
        const Eigen::VectorXd b = A * exact;
        apxchol::solve_options setup;
        setup.backend = apxchol::solve_backend::gpu;
        setup.factor_opts.verbose = true;
        testing::internal::CaptureStderr();
        std::optional<apxchol::solver> owner;
        try { owner.emplace(A, setup); }
        catch (...) { const auto trace=testing::internal::GetCapturedStderr(); ADD_FAILURE()<<trace; throw; }
        const auto trace = testing::internal::GetCapturedStderr();
        const auto& preconditioner = std::get<apxchol::detail::cuda_route>(
            apxchol::detail::solver_access::route(*owner));
        ASSERT_NE(trace.find("[gpu-owned-csc] n=257 "),std::string::npos) << trace;
        EXPECT_NE(trace.find("host_graph_bytes=0 host_snapshot_bytes=0 selector_edge_upload_bytes=0"),std::string::npos) << trace;
        EXPECT_NE(trace.find("factor_entry_download_bytes=0"),std::string::npos) << trace;
        EXPECT_EQ(trace.find("handback_download_bytes="),std::string::npos) << trace;
        ASSERT_TRUE(preconditioner.trsv().adopted_device_factor());
        EXPECT_TRUE(preconditioner.trsv().fp16());
        EXPECT_GT(preconditioner.trsv().drop_stats().rel,0.0);
        EXPECT_TRUE(csc_access::vals(preconditioner.factor().L).empty());
        EXPECT_TRUE(csc_access::inner(preconditioner.factor().L).empty());
        EXPECT_GT(preconditioner.factor().rounds.size(),2u);
        apxchol::solve_options options; options.tol=1e-8; options.max_iter=1000;
        options.backend = apxchol::solve_backend::gpu;
        const auto solved = apxchol::solve(A,b,options);
        ASSERT_EQ(solved.x.size(),n);
        EXPECT_LE((A*solved.x-b).norm()/b.norm(),1e-8);
    }
#endif
}
