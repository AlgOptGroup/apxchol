#include <gtest/gtest.h>

#include "apxchol/solver/elimination/gpu_round_shadow.h"
#include "apxchol/solver/gpu_block_frontend.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(APXCHOL_USE_CUDA)
namespace {
using apxchol::node_index;
using apxchol::detail::gpu_block_frontend;
using apxchol::detail::gpu_topology_edge;

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

struct cascade_fixture {
    node_index path_size;
    std::vector<std::vector<node_index>> neighbors;
    std::vector<gpu_topology_edge> edges;
    std::vector<node_index> active;
};

cascade_fixture make_fixture(node_index n) {
    // The 257-candidate case includes three isolated vertices. They start and
    // remain selected; the nontrivial path must still require many repairs.
    cascade_fixture fixture;
    fixture.path_size = n == 257 ? n - 3 : n;
    fixture.neighbors.resize(n);
    fixture.active.resize(n);
    std::iota(fixture.active.begin(), fixture.active.end(), node_index{0});
    for (node_index v = 1; v < fixture.path_size; ++v) {
        fixture.edges.push_back({v - 1, v});
        fixture.neighbors[v - 1].push_back(v);
        fixture.neighbors[v].push_back(v - 1);
    }
    return fixture;
}

struct reference_result {
    std::vector<node_index> selected;
    std::size_t initial_picks = 0;
    std::size_t dropped = 0;
    std::size_t productive_rounds = 0;
    std::size_t work = 0;
};

reference_result original_three_phase_reference(const cascade_fixture& fixture,
                                                bool degree_tiebreak) {
    // Independent CPU specification: one candidate per contiguous region makes
    // the regional greedy scan pick every candidate. All path edges cross
    // regions. Keep conflict decisions and repair commits as separate snapshots
    // so this oracle does not reproduce the candidate's early status writes.
    const auto n = fixture.active.size();
    std::vector<unsigned char> selected(n, 1), dropped(n, 0), frontier(n, 0);
    const auto precedes = [&](node_index u, node_index v) {
        const auto du = fixture.neighbors[u].size();
        const auto dv = fixture.neighbors[v].size();
        return degree_tiebreak && du != dv ? du < dv : u < v;
    };
    reference_result result;
    result.initial_picks = n;
    for (const auto v : fixture.active)
        for (const auto u : fixture.neighbors[v])
            if (selected[u] && precedes(u, v)) {
                dropped[v] = 1;
                break;
            }
    for (const auto v : fixture.active)
        if (dropped[v]) {
            ++result.dropped;
            selected[v] = 0;
            frontier[v] = 1;
            for (const auto u : fixture.neighbors[v]) frontier[u] = 1;
        }

    for (;;) {
        std::vector<unsigned char> pending(n, 0);
        std::vector<node_index> winners;
        bool any_pending = false;
        // Phase 1 reads only picks committed by an earlier repair round.
        for (const auto v : fixture.active) {
            if (!frontier[v]) continue;
            bool free = !selected[v];
            for (const auto u : fixture.neighbors[v]) free &= !selected[u];
            if (free) { pending[v] = 1; any_pending = true; }
            else frontier[v] = 0;
        }
        if (!any_pending) break;
        // Phase 2 reads a frozen pending set and makes no selected writes.
        for (const auto v : fixture.active) {
            if (!pending[v]) continue;
            bool minimum = true;
            for (const auto u : fixture.neighbors[v])
                if (pending[u] && precedes(u, v)) minimum = false;
            if (minimum) winners.push_back(v);
        }
        if (winners.empty() || result.productive_rounds >= n)
            throw std::logic_error("finite strict priority failed to make repair progress");
        // Phase 3 commits all winners simultaneously after every decision.
        for (const auto v : winners) { selected[v] = 1; frontier[v] = 0; }
        ++result.productive_rounds;
    }
    for (const auto v : fixture.active)
        if (selected[v]) {
            result.selected.push_back(v);
            result.work += fixture.neighbors[v].size();
        }
    return result;
}

void check_cascade(gpu_block_frontend& gpu, const cascade_fixture& fixture,
                   bool degree_tiebreak, const char* view_name) {
    const auto n = fixture.active.size();
    const auto path = static_cast<std::size_t>(fixture.path_size);
    const auto isolates = n - path;
    const auto expected = original_three_phase_reference(fixture, degree_tiebreak);
    ASSERT_EQ(expected.initial_picks, n);
    ASSERT_EQ(expected.dropped, path - (degree_tiebreak ? 2u : 1u));
    // ID priority leaves only the first endpoint after snapshot drops. Degree
    // priority also retains the last endpoint. The remaining increasing path
    // admits exactly one new pick per productive round: this is not vacuous
    // coverage of the termination branch or a single round of status writes.
    const auto rounds = (path - (degree_tiebreak ? 3u : 1u)) / 2;
    ASSERT_EQ(expected.productive_rounds, rounds);
    ASSERT_GT(expected.productive_rounds, 1u);
    ASSERT_EQ(expected.selected.size(), isolates + (degree_tiebreak ? 2u : 1u) + rounds);

    apxchol::partition_options options;
    options.degree_quantile = 0.0;
    options.degree_multiplier = static_cast<double>(n); // Admit every path/isolated vertex.
    options.degree_tiebreak = degree_tiebreak;
    const auto prepared = gpu.prepare(fixture.active, options);
    ASSERT_EQ(prepared.candidate_count, n);
    const auto candidates = gpu.host_candidates();
    ASSERT_EQ(std::vector<node_index>(candidates.begin(), candidates.end()), fixture.active);
    const auto degrees = gpu.host_active_degrees();
    ASSERT_EQ(degrees.size(), n);
    for (const auto v : fixture.active) EXPECT_EQ(degrees[v], fixture.neighbors[v].size());

    std::vector<node_index> first_selected;
    std::size_t first_work = 0;
    for (int repeat = 0; repeat < 2; ++repeat) {
        SCOPED_TRACE(repeat);
        const auto selected = gpu.select_block_greedy().data;
        const auto work = gpu.selected_degree_work();
        if (repeat == 0) { first_selected = selected; first_work = work; }
        ASSERT_EQ(selected, expected.selected);
        EXPECT_EQ(work, expected.work);
        std::vector<unsigned char> picked(n, 0);
        for (const auto v : selected) {
            ASSERT_LT(v, n);
            ASSERT_FALSE(picked[v]);
            picked[v] = 1;
        }
        for (const auto v : fixture.active) {
            bool adjacent_pick = false;
            for (const auto u : fixture.neighbors[v]) adjacent_pick |= picked[u] != 0;
            if (picked[v]) EXPECT_FALSE(adjacent_pick);
            else EXPECT_TRUE(adjacent_pick) << "uncovered vertex=" << v;
        }
        for (std::size_t v = path; v < n; ++v) EXPECT_TRUE(picked[v]);
    }
    // Exactly 14 keys per view (seven candidate sizes x two priorities). These
    // supplement, rather than alter, the existing 88 cross-binary properties.
    std::string receipt = "resident_capacity=" + std::to_string(gpu.resident_region_capacity()) +
        ";cpu_rounds=" + std::to_string(expected.productive_rounds) +
        ";initial_picks=" + std::to_string(expected.initial_picks) +
        ";dropped=" + std::to_string(expected.dropped) +
        ";work=" + std::to_string(first_work) + ";ids";
    for (const auto v : first_selected) receipt += ":" + std::to_string(v);
    testing::Test::RecordProperty(std::string("repair_cascade_") + view_name + "_n" +
        std::to_string(n) + "_degree" + std::to_string(degree_tiebreak), receipt);
}

Eigen::SparseMatrix<double> owned_matrix(const cascade_fixture& fixture) {
    const auto n = static_cast<int>(fixture.active.size());
    std::vector<Eigen::Triplet<double>> entries;
    for (const auto& edge : fixture.edges) {
        entries.emplace_back(edge.u, edge.v, -1.0);
        entries.emplace_back(edge.v, edge.u, -1.0);
    }
    for (int v = 0; v < n; ++v)
        entries.emplace_back(v, v, static_cast<double>(fixture.neighbors[v].size()) + 0.5);
    Eigen::SparseMatrix<double> result(n, n);
    result.setFromTriplets(entries.begin(), entries.end());
    return result;
}
} // namespace
#endif

TEST(GpuBlockFrontend, CsrRepairCascadesMatchIndependentSpec) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA required";
#else
    if (!apxchol::detail::gpu_round_shadow_runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    for (const node_index n : {31u, 32u, 33u, 255u, 256u, 257u, 1025u}) {
        SCOPED_TRACE(n);
        const auto fixture = make_fixture(n);
        const auto regions = std::to_string(n);
        scoped_environment blocks("APXCHOL_GPU_BLOCKS", regions.c_str());
        for (const bool degree_tiebreak : {false, true}) {
            SCOPED_TRACE(degree_tiebreak);
            gpu_block_frontend gpu(n, fixture.edges);
            check_cascade(gpu, fixture, degree_tiebreak, "csr");
        }
    }
#endif
}

TEST(GpuBlockFrontend, OwnedBoundedSelectionMakesIndependentProgress) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA direct CSC required";
#else
    if (!apxchol::detail::gpu_round_shadow_runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    scoped_environment finalize("APXCHOL_GPU_FACTOR_FINALIZE", "force");
    for (const node_index n : {31u, 32u, 33u, 255u, 256u, 257u, 1025u}) {
        SCOPED_TRACE(n);
        const auto fixture = make_fixture(n);
        const auto matrix = owned_matrix(fixture);
        ASSERT_TRUE(apxchol::detail::gpu_owned_csc_supported(matrix));
        const auto regions = std::to_string(n);
        scoped_environment blocks("APXCHOL_GPU_BLOCKS", regions.c_str());
        for (const bool degree_tiebreak : {false, true}) {
            SCOPED_TRACE(degree_tiebreak);
            // Destruction is ordered: the borrowing frontend dies before its
            // numerical owner; both die before the input matrix's outer scope.
            apxchol::detail::gpu_round_shadow_session owner(true);
            std::unique_ptr<gpu_block_frontend> gpu;
            owner.initialize_owned_csc(apxchol::detail::gpu_owned_csc_host_buffers(matrix), gpu);
            ASSERT_TRUE(gpu);
            apxchol::partition_options options;
            options.degree_quantile = 0.0;
            options.degree_multiplier = static_cast<double>(n);
            options.degree_tiebreak = degree_tiebreak;
            const auto prepared = gpu->prepare(fixture.active, options);
            ASSERT_EQ(prepared.candidate_count, fixture.active.size());
            const auto candidates = gpu->host_candidates();
            EXPECT_EQ(std::vector<node_index>(candidates.begin(), candidates.end()), fixture.active);
            const auto selected = gpu->select_block_greedy().data;
            ASSERT_FALSE(selected.empty());
            EXPECT_EQ(gpu->select_block_greedy().data, selected);
            const auto degrees = gpu->host_active_degrees();
            ASSERT_EQ(degrees.size(), n);
            for (node_index v : fixture.active) EXPECT_EQ(degrees[v], fixture.neighbors[v].size());
            std::vector<bool> picked(n); std::size_t work = 0;
            for (auto v : selected) {
                ASSERT_LT(v, n); ASSERT_FALSE(picked[v]); picked[v] = true;
                work += fixture.neighbors[v].size();
            }
            for (auto v : selected) for (auto u : fixture.neighbors[v]) EXPECT_FALSE(picked[u]);
            for (node_index v = fixture.path_size; v < n; ++v) EXPECT_TRUE(picked[v]);
            EXPECT_EQ(gpu->selected_degree_work(), work);
            // Owned selection deliberately stops after four snapshot passes;
            // only the generic CSR fixture above requires maximal repair.
            std::string receipt = "work=" + std::to_string(work);
            for (auto v : selected) receipt += ":" + std::to_string(v);
            RecordProperty("owned_bounded_progress_" + std::to_string(n) + "_" +
                std::to_string(degree_tiebreak), receipt);
            EXPECT_EQ(gpu->transfers().owned_residual_binds, 1u);
            EXPECT_EQ(gpu->transfers().host_update_bytes, 0u);
        }
    }
#endif
}
