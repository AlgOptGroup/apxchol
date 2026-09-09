#include <gtest/gtest.h>
#include "apxchol/graph/conversions.h"
#include "apxchol/operator_class.h"
#include "apxchol/solver/detail/gpu_diagnostics.h"
#include "apxchol/solver/elimination/gpu_round_shadow.h"
#include "apxchol/solver/gpu_block_frontend.h"
#include "apxchol/solver/solve.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
using apxchol::node_index;
using matrix = Eigen::SparseMatrix<double>;
using host_graph = apxchol::graph<apxchol::directed_vec_pool_incidence>;
using apxchol::detail::gpu_round_shadow_input;

// Local guards follow the existing GPU round-shadow tests; no process-wide
// cached grounding knob is changed by this test file.
class scoped_environment {
public:
    scoped_environment(const char* key, const char* value) : key_(key) {
        if (const char* old = std::getenv(key)) { present_ = true; old_ = old; }
        if (value) setenv(key, value, 1); else unsetenv(key);
    }
    ~scoped_environment() {
        if (present_) setenv(key_.c_str(), old_.c_str(), 1);
        else unsetenv(key_.c_str());
    }
private:
    std::string key_, old_;
    bool present_ = false;
};

class scoped_threads {
public:
    explicit scoped_threads(int count) {
#ifdef _OPENMP
        before_ = omp_get_max_threads(); omp_set_num_threads(count);
#else
        (void)count;
#endif
    }
    ~scoped_threads() {
#ifdef _OPENMP
        omp_set_num_threads(before_);
#endif
    }
private:
    int before_ = 1;
};

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

matrix selector_matrix() {
    std::vector<weighted_edge> edges;
    for (int u = 0; u < 12; ++u)
        for (int v = u + 1; v < 12; ++v) edges.push_back({u, v, 1.0});
    for (int u = 12; u < 127; ++u)
        if (u != 62 && u != 63 && u != 64) edges.push_back({u, u + 1, 0.5});
    for (int u = 14; u < 120; u += 9) edges.push_back({u, u + 7, 2.0});
    edges.push_back({0, 128, 0.0}); // Stored zero remains a topological incidence.
    return make_matrix(132, edges); // Includes disconnected components and isolates.
}

gpu_round_shadow_input host_snapshot(const matrix& A, double reg_eps = 0.0) {
    auto graph = apxchol::make_graph<host_graph>(A);
    if (reg_eps > 0.0)
        for (node_index v = 0; v < graph.n(); ++v)
            graph.excess(v) = std::max(graph.excess(v), reg_eps * A.coeff(v, v));
    return apxchol::detail::make_gpu_round_shadow_input(
        graph, std::span<const node_index>{}, 0);
}

void expect_same_snapshot(const gpu_round_shadow_input& got,
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
        EXPECT_EQ(std::bit_cast<std::uint64_t>(got.incidences[i].weight),
                  std::bit_cast<std::uint64_t>(expected.incidences[i].weight)) << i;
    }
}

std::vector<apxchol::detail::gpu_topology_edge> topology(const host_graph& graph) {
    std::vector<apxchol::detail::gpu_topology_edge> result;
    for (node_index u = 0; u < graph.n(); ++u)
        for (const auto& edge : graph.neighbors(u))
            if (u < edge.to) result.push_back({u, edge.to});
    return result;
}
} // namespace

TEST(GpuDirectCscHost, RejectsUnsupportedStoredFormatsBeforeCuda) {
    scoped_threads serial(1);
    using apxchol::detail::gpu_owned_csc_supported;
    const auto valid = make_matrix(3, {{0,1,1.0},{0,2,2.0},{1,2,0.5}});
    ASSERT_TRUE(gpu_owned_csc_supported(valid));
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
    for (int corruption = 0; corruption < 5; ++corruption) {
        auto bad = valid;
        switch (corruption) {
        case 0: bad.innerIndexPtr()[1] = bad.innerIndexPtr()[0]; break; // duplicate
        case 1: std::swap(bad.innerIndexPtr()[0], bad.innerIndexPtr()[1]); break;
        case 2: bad.innerIndexPtr()[1] = -1; break;
        case 3: bad.innerIndexPtr()[1] = 3; break;
        case 4: bad.outerIndexPtr()[1] = -1; break;
        }
        unsupported.push_back(std::move(bad));
    }
    ASSERT_EQ(unsupported.size(), 10u);
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
    scoped_threads serial(1);
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
        const apxchol::operator_view op(one_zero);
        EXPECT_EQ(op.scan().offdiag_nnz, 0);
        EXPECT_FALSE(gpu_owned_csc_supported(op));
        fixtures.push_back(one_zero);
    }
    auto unrelated_zeros = make_matrix(3, {}, 1.0);
    unrelated_zeros.coeffRef(1,0) = 0.0;
    unrelated_zeros.coeffRef(1,2) = -0.0;
    unrelated_zeros.makeCompressed();
    EXPECT_FALSE(gpu_owned_csc_supported(apxchol::operator_view(unrelated_zeros)));
    fixtures.push_back(unrelated_zeros);
    auto lumped = make_matrix(3, {{0,1,1.0},{1,2,-1e-6}}, 1.0);
    const apxchol::operator_view op_lumped(lumped);
    if (apxchol::detail::env_knobs::get().lump) EXPECT_EQ(op_lumped.lumped(), 2);
    fixtures.push_back(lumped);
    for (std::size_t i = 0; i < fixtures.size(); ++i) {
        SCOPED_TRACE(i);
        const apxchol::operator_view op(fixtures[i]);
        EXPECT_EQ(gpu_owned_csc_supported(op), gpu_owned_csc_supported(op.matrix()));
    }
    auto uncompressed = fixtures.front();
    uncompressed.uncompress();
    const apxchol::operator_view op(uncompressed);
    EXPECT_FALSE(gpu_owned_csc_supported(op));
}

TEST(GpuDirectCscHost, CertifiedEligibilityRejectsDuplicateAndUnsortedZeroColumns) {
    scoped_threads serial(1);
    using apxchol::detail::gpu_owned_csc_supported;
    for (bool duplicate : {false, true}) {
        auto A = make_matrix(4, {{0,2,0.0},{0,3,0.0}}, 1.0);
        ASSERT_EQ(A.innerIndexPtr()[1], 2);
        ASSERT_EQ(A.innerIndexPtr()[2], 3);
        if (duplicate) A.innerIndexPtr()[2] = 2;
        else std::swap(A.innerIndexPtr()[1], A.innerIndexPtr()[2]);
        // The numerical scan has no edges to reject. Strict layout validation
        // must run even when an operator_view was constructed successfully.
        const apxchol::operator_view op(A);
        ASSERT_EQ(op.scan().offdiag_nnz, 0);
        EXPECT_FALSE(gpu_owned_csc_supported(op));
        EXPECT_FALSE(gpu_owned_csc_supported(A));
    }
}

TEST(GpuDirectCscHost, NumericalCertificateRejectsMissingTinyNonzeroMates) {
    scoped_threads serial(1);
    for (double weight : {1.0, std::numeric_limits<double>::denorm_min()}) {
        for (bool lower : {false, true}) {
            auto A = make_matrix(3, {}, 1.0);
            A.coeffRef(lower ? 1 : 0, lower ? 0 : 1) = -weight;
            A.makeCompressed();
            EXPECT_THROW((void)apxchol::operator_view(A), std::invalid_argument);
        }
        auto balanced = make_matrix(3, {}, 1.0);
        balanced.coeffRef(1,0) = -weight;
        balanced.coeffRef(1,2) = -weight;
        balanced.makeCompressed();
        EXPECT_THROW((void)apxchol::operator_view(balanced), std::invalid_argument);
    }
}

TEST(GpuDirectCsc, InitialArraysMatchHostBuilderIncludingRoundingAndSignedZero) {
#if !defined(APXCHOL_USE_CUDA) || !defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
    GTEST_SKIP() << "CUDA direct CSC experiment and test hooks required";
#else
    if (!apxchol::detail::gpu_round_shadow_runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    ASSERT_EQ(apxchol::detail::env_knobs::get().ground, apxchol::detail::grounding_kind::center_k);
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
    const apxchol::operator_view tolerated(asymmetric);
    EXPECT_EQ(tolerated.lumped(), 0);
    fixtures.push_back(asymmetric); // Tolerated asymmetry still uses LOWER weights twice.
    auto signed_zero = make_matrix(3, {{0,1,-0.0}}, 1.0);
    signed_zero.coeffRef(1,0) = +0.0; signed_zero.coeffRef(0,1) = -0.0;
    fixtures.push_back(signed_zero);
    for (int threads : {1, 3}) {
        scoped_threads team(threads);
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
                const double canonical = static_cast<double>(static_cast<apxchol::pool_value_t>(
                    -A.coeff(std::max(edge.owner,edge.neighbor),std::min(edge.owner,edge.neighbor))));
                EXPECT_EQ(std::bit_cast<std::uint64_t>(edge.weight), std::bit_cast<std::uint64_t>(canonical));
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
    ASSERT_EQ(apxchol::detail::env_knobs::get().ground, apxchol::detail::grounding_kind::center_k);
    scoped_threads serial(1);
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
    scoped_threads serial(1);
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
#ifdef APXCHOL_POOL_FP32
    const double overflow = 2.0 * static_cast<double>(std::numeric_limits<float>::max());
    auto A = make_matrix(2, {{0,1,overflow}});
    EXPECT_THROW((void)apxchol::detail::gpu_initial_owned_csc_for_test(A), std::invalid_argument);
#endif
#endif
}

TEST(GpuDirectCsc, FirstSelectorHasCorrectCandidatesAndIndependentProgress) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA direct CSC required";
#else
    if (!apxchol::detail::gpu_round_shadow_runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    scoped_threads serial(1);
    scoped_environment finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    const auto A = selector_matrix();
    const auto graph = apxchol::make_graph<host_graph>(A);
    ASSERT_TRUE(apxchol::detail::gpu_owned_csc_supported(A));
    std::vector<node_index> active(graph.n()); std::iota(active.begin(),active.end(),node_index{0});
    for (const char* regions : {"1", "7", static_cast<const char*>(nullptr)}) {
        scoped_environment blocks("APXCHOL_GPU_BLOCKS", regions);
        for (bool ties : {false, true}) for (double quantile : {0.0, 0.3, 0.8}) {
            SCOPED_TRACE(regions);
            SCOPED_TRACE(ties);
            SCOPED_TRACE(quantile);
            apxchol::detail::gpu_round_shadow_session owner(true);
            std::unique_ptr<apxchol::detail::gpu_block_frontend> direct;
            owner.initialize_owned_csc(apxchol::detail::gpu_owned_csc_host_buffers(A), direct);
            apxchol::detail::gpu_block_frontend reference(graph.n(), topology(graph));
            apxchol::partition_options options;
            options.degree_quantile = quantile; options.degree_tiebreak = ties;
            const auto got = direct->prepare(active, options);
            const auto expected = reference.prepare(active, options);
            EXPECT_EQ(got.average_degree, expected.average_degree);
            const auto degrees = direct->host_active_degrees();
            const auto want_degrees = reference.host_active_degrees();
            EXPECT_EQ(std::vector<node_index>(degrees.begin(),degrees.end()),
                      std::vector<node_index>(want_degrees.begin(),want_degrees.end()));
            ASSERT_EQ(degrees.size(), active.size());
            for (node_index v : active) EXPECT_EQ(degrees[v], graph.adj_count(v));
            const auto candidate_view = direct->host_candidates();
            const std::vector<node_index> candidates(candidate_view.begin(),candidate_view.end());
            double threshold = options.degree_multiplier * expected.average_degree;
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
            std::string receipt = std::to_string(direct->resident_region_capacity()) + ":" +
                std::to_string(direct->selected_degree_work());
            for (const auto vertex : selected) receipt += ":" + std::to_string(vertex);
            RecordProperty(std::string("direct_selection_") + (regions ? regions : "auto") +
                "_" + std::to_string(ties) + "_" + std::to_string(quantile), receipt);
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
                if (picked[v]) EXPECT_FALSE(neighbor_picked);
            }
            EXPECT_EQ(direct->selected_degree_work(), degree_work);
            EXPECT_EQ(direct->transfers().owned_residual_binds, 1u);
            EXPECT_EQ(direct->transfers().host_update_bytes, 0u);
        }
    }
#endif
}

TEST(GpuDirectCsc, InitialOwnerOutlivesInputAndRetiresEveryBorrowedSelectorRead) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA direct CSC required";
#else
    if (!apxchol::detail::gpu_round_shadow_runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    scoped_threads serial(1);
    scoped_environment finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_environment blocks("APXCHOL_GPU_BLOCKS", "1");
    for (int operation = 0; operation < 5; ++operation) {
        SCOPED_TRACE(operation);
        auto owner = std::make_unique<apxchol::detail::gpu_round_shadow_session>(true);
        std::unique_ptr<apxchol::detail::gpu_block_frontend> direct;
        const auto original = selector_matrix();
        const auto graph = apxchol::make_graph<host_graph>(original);
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
        double threshold = options.degree_multiplier * double(degree_sum) / graph.n();
        if (options.degree_quantile > 0.0 && options.degree_quantile < 1.0) {
            auto sorted = degrees; std::sort(sorted.begin(), sorted.end());
            threshold = sorted[static_cast<std::size_t>(options.degree_quantile * (sorted.size() - 1))];
        }
        constexpr std::uint64_t seed = 42, phase = 0;
        auto key = [](node_index v) {
            std::uint64_t z = (seed ^ (phase * 0xD6E8FEB86659FD93ULL) ^ v) + 0x9E3779B97F4A7C15ULL;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            return z ^ (z >> 31);
        };
        auto precedes = [&](node_index a, node_index b) {
            if (options.degree_tiebreak && degrees[a] != degrees[b]) return degrees[a] < degrees[b];
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
        EXPECT_EQ(direct->transfers().host_update_bytes, 0u);
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
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA direct CSC required";
#else
    if (!apxchol::detail::gpu_round_shadow_runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    scoped_threads serial(1);
    scoped_environment setup_trace("APXCHOL_SPTRSV_SETUP_TRACE", "1");
    scoped_environment shadow("APXCHOL_GPU_ROUND_SHADOW", "force");
    scoped_environment finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    scoped_environment frontend("APXCHOL_GPU_BLOCK_FRONTEND", "force");
    scoped_environment blocks("APXCHOL_GPU_BLOCKS", "1");
    scoped_environment fp16("APXCHOL_SPTRSV_FP16", nullptr);
    scoped_environment drop("APXCHOL_FACTOR_DROP", nullptr);
    scoped_environment backend("APXCHOL_GPU_SPTRSV", nullptr);
    scoped_environment tail("APXCHOL_RESIDUAL_SPARSIFY", "0");
    constexpr int n = 257;
    std::vector<weighted_edge> edges;
    for (int u = 0; u < n; ++u) for (int offset : {1,7}) edges.push_back({u,(u+offset)%n,1.0});
    for (double shift : {0.0, 1.0}) {
        SCOPED_TRACE(shift);
        const auto A = make_matrix(n,edges,shift);
        Eigen::VectorXd exact(n);
        for (int v = 0; v < n; ++v) exact[v] = std::sin(v + 0.25);
        const Eigen::VectorXd b = A * exact;
        apxchol::apx_cholesky preconditioner;
        testing::internal::CaptureStderr();
        try { preconditioner.compute(A); }
        catch (...) { const auto trace=testing::internal::GetCapturedStderr(); ADD_FAILURE()<<trace; throw; }
        const auto trace = testing::internal::GetCapturedStderr();
        if (apxchol::detail::gpu_setup_diagnostics()) {
            ASSERT_NE(trace.find("[gpu-owned-csc] n=257 "),std::string::npos) << trace;
            EXPECT_NE(trace.find("host_graph_bytes=0 host_snapshot_bytes=0 selector_edge_upload_bytes=0"),std::string::npos) << trace;
            EXPECT_NE(trace.find("initial_snapshots=0"),std::string::npos) << trace;
            EXPECT_NE(trace.find("residual_download_bytes=0"),std::string::npos) << trace;
            EXPECT_EQ(trace.find("handback_download_bytes="),std::string::npos) << trace;
        } else {
            EXPECT_EQ(trace.find("[gpu-owned-"), std::string::npos) << trace;
            EXPECT_EQ(trace.find("[gpu-round-shadow]"), std::string::npos) << trace;
            EXPECT_EQ(trace.find("[gpu-setup-receipt]"), std::string::npos) << trace;
        }
        ASSERT_TRUE(preconditioner.trsv().adopted_device_factor());
        EXPECT_TRUE(preconditioner.trsv().fp16());
        EXPECT_GT(preconditioner.trsv().drop_stats().rel,0.0);
        EXPECT_TRUE(preconditioner.factor().L.vals_.empty());
        EXPECT_TRUE(preconditioner.factor().L.inner_.empty());
        EXPECT_GT(preconditioner.factor().rounds.size(),2u);
        apxchol::solve_options options; options.tol=1e-8; options.max_iter=1000;
        const auto solved = apxchol::solve(A,b,options);
        ASSERT_EQ(solved.x.size(),n);
        EXPECT_LE((A*solved.x-b).norm()/b.norm(),1e-8);
    }
#endif
}
