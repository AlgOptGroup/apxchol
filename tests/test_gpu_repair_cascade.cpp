#include <gtest/gtest.h>
#include "apxchol/detail/eigen_boundary.h"

#include "apxchol/solver/elimination/gpu_round_shadow.h"
#include "apxchol/solver/gpu_block_frontend.h"

#include <cstddef>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#if defined(APXCHOL_USE_CUDA)
namespace {
using apxchol::node_index;
using apxchol::detail::gpu_block_frontend;
struct topology_edge { node_index u, v; };

struct cascade_fixture {
    node_index path_size;
    std::vector<std::vector<node_index>> neighbors;
    std::vector<topology_edge> edges;
    std::vector<node_index> active;
};

cascade_fixture make_fixture(node_index n) {
    // Check isolated-vertex selection alongside path independence at a
    // thread-block boundary.
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

TEST(GpuBlockFrontend, OwnedBoundedSelectionMakesIndependentProgress) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA direct CSC required";
#else
    if (!apxchol::detail::gpu_round_shadow_runtime_available()) GTEST_SKIP() << "CUDA unavailable";
    for (const node_index n : {31u, 32u, 33u, 255u, 256u, 257u, 1025u}) {
        SCOPED_TRACE(n);
        const auto fixture = make_fixture(n);
        const auto matrix = owned_matrix(fixture);
        ASSERT_TRUE(apxchol::detail::gpu_owned_csc_supported(apxchol::detail::eigen_csc(matrix)));
        for (const bool degree_tiebreak : {false, true}) {
            SCOPED_TRACE(degree_tiebreak);
            // Destruction is ordered: the borrowing frontend dies before its
            // numerical owner; both die before the input matrix's outer scope.
            apxchol::detail::gpu_round_shadow_session owner(true);
            std::unique_ptr<gpu_block_frontend> gpu;
            owner.initialize_owned_csc(apxchol::detail::gpu_owned_csc_host_buffers(apxchol::detail::eigen_csc(matrix)), gpu);
            ASSERT_TRUE(gpu);
            apxchol::partition_options options;
            options.degree_quantile = 0.0;
            options.degree_multiplier = static_cast<double>(n);
            const auto prepared = gpu->prepare(fixture.active, options, 42, 0, degree_tiebreak);
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
            std::string receipt = "work=" + std::to_string(work);
            for (auto v : selected) receipt += ":" + std::to_string(v);
            RecordProperty("owned_bounded_progress_" + std::to_string(n) + "_" +
                std::to_string(degree_tiebreak), receipt);
            EXPECT_EQ(gpu->transfers().owned_residual_binds, 1u);
        }
    }
#endif
}
