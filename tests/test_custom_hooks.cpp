/// Tests for the public customization seams: custom eliminators, custom
/// (stateful) partitioners, adopting an external factorization, keep-factor,
/// and the PCG initial guess.
#include "apxchol.h"
#include "apxchol/solver/partitioner_helpers.h"
#include <gtest/gtest.h>
#include "grid_laplacian.h"
#include <algorithm>
#include <limits>
#include <span>
#include <vector>

namespace {
// An exact FP32 factor preconditions PCG to at most three iterations.
// FP16 storage rounding may require one more iteration on this fixture.
constexpr int kExactFactorMaxIters = 3;

using apxchol::test::grid_laplacian;

// Exact Schur-complement clique: zero sampling variance, more fill.
struct exact_clique_eliminator {
    void sample_clique(std::span<apxchol::weighted_neighbor> neighbors,
                       double deg, std::uint64_t /*seed*/,
                       apxchol::edge_emitter out) const {
        for (const auto& [u, wu] : neighbors)
            for (const auto& [v, wv] : neighbors) {
                if (u == v) break;                 // each unordered pair once
                out(u, v, wu * wv / deg);
            }
    }
};
static_assert(apxchol::eliminator<exact_clique_eliminator>);

// Greedy IS scanned in a user-supplied priority order.
struct priority_partitioner {
    static constexpr std::string_view name = "priority_test";

    /// User-supplied priority per vertex (lower = eliminated earlier).
    std::vector<double> priority;

    void find_partition(apxchol::graph& G,
                        std::span<const apxchol::node_index> active,
                        const apxchol::partition_context& /*ctx*/,
                        apxchol::selection& out) {
        // Scan vertices in priority order; greedily keep an independent set
        // (the selection answers the membership queries).
        order_.assign(active.begin(), active.end());
        std::sort(order_.begin(), order_.end(),
                  [&](auto a, auto b) { return priority[a] < priority[b]; });
        for (auto v : order_) {
            bool independent = true;
            for (auto idx : G.adj(v)) {
                auto u = idx.to;
                if (G.is_active(u) && out.contains(u)) { independent = false; break; }
            }
            if (independent) out.add(v);
        }
    }

private:
    std::vector<apxchol::node_index> order_;       // sorted candidate list
};
static_assert(apxchol::partitioner<priority_partitioner>);

using vp_graph = apxchol::graph;

const apxchol::solve_options on_cpu{.backend = apxchol::solve_backend::cpu};

TEST(CustomEliminator, ExactCliqueConverges) {
    auto L = grid_laplacian(20, 20);
    auto b = apxchol::generate_test_rhs(L.rows());

    // Matrix-level convenience overload (graph built internally).
    auto F = apxchol::factorize(L, exact_clique_eliminator{});
    EXPECT_GT(F.L.nonZeros(), 0u);

    apxchol::solver slv(L, std::move(F));
    auto res = slv.solve(b, 1e-8, 500);
    EXPECT_LT(res.residual, 1e-8);
}

TEST(CustomEliminator, LambdaViaAsEliminator) {
    auto L = grid_laplacian(15, 15);
    auto b = apxchol::generate_test_rhs(L.rows());

    // Same exact-clique rule, expressed as a lambda.
    auto F = apxchol::factorize(L, apxchol::as_eliminator(
        [](std::span<apxchol::weighted_neighbor> nb, double deg,
           std::uint64_t /*seed*/, apxchol::edge_emitter out) {
            for (const auto& [u, wu] : nb)
                for (const auto& [v, wv] : nb) {
                    if (u == v) break;
                    out(u, v, wu * wv / deg);
                }
        }));

    apxchol::solver slv(L, std::move(F));
    auto res = slv.solve(b, 1e-8, 500);
    EXPECT_LE(res.iterations, kExactFactorMaxIters);   // exact factor
    EXPECT_LT(res.residual, 1e-8);
}

TEST(CustomEliminator, ExactCliqueFactorIsExact) {
    // With the exact clique rule the factorization is exact Cholesky:
    // P^T L L^T P == A up to roundoff (no sampling at all).
    auto A = grid_laplacian(8, 8);
    auto b = apxchol::generate_test_rhs(A.rows());
    auto solve_once = [&](apxchol::factor_storage_type storage) {
        auto G = apxchol::make_graph(A);
        auto F = apxchol::factorize(std::move(G), exact_clique_eliminator{});
        apxchol::solve_options options;
        options.factor_opts.factor_storage = storage;
        apxchol::solver slv(A, std::move(F), options);
        return slv.solve(b, 1e-12, 50);
    };
    for (bool fp16 : {false, true}) {
        if (fp16 && !apxchol::omp_sptrsv::fp16_supported()) continue;
        SCOPED_TRACE(fp16 ? "fp16 storage" : "fp32 storage");
        const auto storage = fp16 ? apxchol::factor_storage_type::fp16
                                  : apxchol::factor_storage_type::fp32;
        const auto res = solve_once(storage);
        EXPECT_LE(res.iterations, kExactFactorMaxIters + (fp16 ? 1 : 0));
        EXPECT_LT(res.residual, 1e-12);
    }
}

TEST(CustomPartitioner, StatefulInstanceConverges) {
    const int k = 20;
    auto L = grid_laplacian(k, k);
    auto b = apxchol::generate_test_rhs(L.rows());

    priority_partitioner part;
    part.priority.resize(size_t(k) * k);
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j)
            part.priority[size_t(i) * k + j] =
                (i == 0 || j == 0 || i == k - 1 || j == k - 1) ? 1.0 : 0.0;

    auto F = apxchol::factorize(L, std::move(part));
    EXPECT_FALSE(F.rounds.empty());

    apxchol::solver slv(L, std::move(F));
    auto res = slv.solve(b, 1e-8, 500);
    EXPECT_LT(res.residual, 1e-8);
}

TEST(CustomPartitioner, LambdaViaAsPartitioner) {
    auto L = grid_laplacian(15, 15);
    auto b = apxchol::generate_test_rhs(L.rows());

    // Trivial-but-valid rule: greedily select an independent set in the
    // natural active order (the selection answers membership queries).
    auto part = apxchol::as_partitioner(
        [](auto& G, std::span<const apxchol::node_index> active,
           const apxchol::partition_context&, apxchol::selection& out) {
            for (auto v : active) {
                bool ok = true;
                for (auto idx : G.adj(v)) {
                    auto u = idx.to;
                    if (G.is_active(u) && out.contains(u)) { ok = false; break; }
                }
                if (ok) out.add(v);
            }
        });

    auto F = apxchol::factorize(L, std::move(part));
    apxchol::solver slv(L, std::move(F));
    auto res = slv.solve(b, 1e-8, 500);
    EXPECT_LT(res.residual, 1e-8);
}

TEST(SetFactor, ExternalFactorizationInPreconditioner) {
    auto L = grid_laplacian(15, 15);
    auto b = apxchol::generate_test_rhs(L.rows());

    auto G = apxchol::make_graph(L);
    auto F = apxchol::factorize(std::move(G));

    apxchol::solver M(L, std::move(F), on_cpu);
    Eigen::VectorXd z = M.apply(b);
    EXPECT_TRUE(z.allFinite());
    EXPECT_GT(z.norm(), 0.0);
}

TEST(KeepFactor, ValuesSurviveWhenRequested) {
    auto L = grid_laplacian(10, 10);
    apxchol::solve_options opts;
    opts.backend = apxchol::solve_backend::cpu;

    apxchol::solver released(L, opts);
    EXPECT_TRUE(released.factor().L.vals_.empty());

    opts.keep_factor_values = true;
    apxchol::solver kept(L, opts);
    const auto& FL = kept.factor().L;
    EXPECT_EQ(FL.vals_.size(), size_t(FL.nonZeros()));
    EXPECT_EQ(FL.inner_.size(), size_t(FL.nonZeros()));
}

TEST(InitialGuess, ExactX0ConvergesImmediately) {
    auto L = grid_laplacian(15, 15);
    auto b = apxchol::generate_test_rhs(L.rows());

    apxchol::solver slv(L, on_cpu);
    auto ref = slv.solve(b, 1e-10, 1000);
    ASSERT_LT(ref.residual, 1e-10);

    auto warm = slv.solve(b, 1e-8, 500, &ref.x);
    EXPECT_EQ(warm.iterations, 0);
    EXPECT_LT(warm.residual, 1e-8);

    // A wrong-size x0 is rejected loudly (consistent with the adopting ctor).
    Eigen::VectorXd bad = Eigen::VectorXd::Ones(3);
    EXPECT_THROW(slv.solve(b, 1e-8, 500, &bad), std::invalid_argument);
}

TEST(OutputMemory, SolveIntoCallerBuffer) {
    auto L = grid_laplacian(15, 15);
    auto b = apxchol::generate_test_rhs(L.rows());
    apxchol::solver slv(L, on_cpu);

    auto ref = slv.solve(b);
    Eigen::VectorXd x(L.rows());
    auto res = slv.solve(b, x, 1e-8, 500);
    EXPECT_EQ(res.x.size(), 0);                 // solution lives in x, not res
    EXPECT_LT(res.residual, 1e-8);
    EXPECT_LT((x - ref.x).norm(), 1e-10 * ref.x.norm() + 1e-14);

    Eigen::VectorXd wrong(3);
    EXPECT_THROW(slv.solve(b, wrong), std::invalid_argument);
}

TEST(SolveHonesty, ZeroIterationSolveDoesNotClaimConvergence) {
    auto L = grid_laplacian(10, 10);
    auto b = apxchol::generate_test_rhs(L.rows());
    apxchol::solver slv(L, on_cpu);

    // max_iter = 0: no PCG update ran, so the reported relative residual must
    // be the honest ||b - A*0|| / ||b|| = 1, never the field's 0.0 default.
    auto res = slv.solve(b, 1e-8, 0);
    EXPECT_EQ(res.iterations, 0);
    EXPECT_GE(res.residual, 1.0);
}

TEST(SolveHonesty, ZeroRhsResetsReusedResult) {
    const auto L = grid_laplacian(3, 3);
    apxchol::solver solver(L, on_cpu);
    const Eigen::VectorXd b = Eigen::VectorXd::Zero(L.rows());
    apxchol::solve_result result;
    result.x = Eigen::VectorXd::Ones(L.rows());
    result.iterations = 7;
    result.residual = std::numeric_limits<double>::quiet_NaN();

    solver.solve(b, result);

    ASSERT_EQ(result.x.size(), L.rows());
    EXPECT_TRUE(result.x.isZero(0.0));
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.residual, 0.0);
}

TEST(SolveHonesty, ZeroRhsResetsReusedResultWithWarmStart) {
    const auto L = grid_laplacian(3, 3);
    apxchol::solver solver(L, on_cpu);
    const Eigen::VectorXd b = Eigen::VectorXd::Zero(L.rows());
    const Eigen::VectorXd x0 = Eigen::VectorXd::LinSpaced(L.rows(), -2.0, 2.0);
    apxchol::solve_result result;
    result.iterations = 7;
    result.residual = std::numeric_limits<double>::infinity();

    solver.solve(b, result, 1e-8, 200, &x0);

    ASSERT_EQ(result.x.size(), L.rows());
    EXPECT_TRUE(result.x.isZero(0.0));
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.residual, 0.0);
}

TEST(SetFactor, ReleasedFactorIsRejected) {
    auto L = grid_laplacian(10, 10);
    apxchol::solver slv(L, on_cpu);  // default: factor values released after setup

    EXPECT_THROW(apxchol::solver(L, slv.factor(), on_cpu),
                 std::invalid_argument);
}

}  // namespace
