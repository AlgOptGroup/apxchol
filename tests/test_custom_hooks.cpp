/// Tests for the public customization seams: custom eliminators, custom
/// (stateful) partitioners, adopting an external factorization, keep-factor,
/// and the PCG initial guess.
#include "apxchol.h"
#include "apxchol/solver/sptrsv/omp.h"
#include <gtest/gtest.h>
#include "csc_test_util.h"
#include "grid_laplacian.h"
#include "test_rhs.h"
#include <algorithm>
#include <span>
#include <vector>

using csc_access = apxchol::detail::sparse_csc_access;

namespace {
// An exact FP32 factor preconditions PCG to at most three iterations.
// FP16 storage rounding may require one more iteration on this fixture.
constexpr int kExactFactorMaxIters = 3;

using apxchol::test::grid_laplacian;
using apxchol::test::csc;
using apxchol::test::cspan;

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
    /// User-supplied priority per vertex (lower = eliminated earlier).
    std::vector<double> priority;

    void find_partition(const apxchol::partition_graph& G,
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
            G.for_each_neighbor(v, [&](auto u) {
                if (out.contains(u)) independent = false;
            });
            if (independent) out.add(v);
        }
    }

private:
    std::vector<apxchol::node_index> order_;       // sorted candidate list
};
static_assert(apxchol::partitioner<priority_partitioner>);

const apxchol::solve_options on_cpu{.backend = apxchol::solve_backend::cpu};

TEST(CustomEliminator, ExactCliqueConverges) {
    auto L = grid_laplacian(20, 20);
    auto b = apxchol::test::random_rhs(L.rows());

    // Matrix-level convenience overload (graph built internally).
    auto F = apxchol::factorize(csc(L), exact_clique_eliminator{});
    EXPECT_GT(F.L.nonZeros(), 0u);

    apxchol::solver slv(csc(L), std::move(F));
    auto res = slv.solve(cspan(b), 1e-8, 500);
    EXPECT_LT(res.residual, 1e-8);
}

TEST(CustomEliminator, LambdaViaAsEliminator) {
    auto L = grid_laplacian(15, 15);
    auto b = apxchol::test::random_rhs(L.rows());

    // Same exact-clique rule, expressed as a lambda.
    auto F = apxchol::factorize(csc(L), apxchol::as_eliminator(
        [](std::span<apxchol::weighted_neighbor> nb, double deg,
           std::uint64_t /*seed*/, apxchol::edge_emitter out) {
            for (const auto& [u, wu] : nb)
                for (const auto& [v, wv] : nb) {
                    if (u == v) break;
                    out(u, v, wu * wv / deg);
                }
        }));

    apxchol::solver slv(csc(L), std::move(F));
    auto res = slv.solve(cspan(b), 1e-8, 500);
    EXPECT_LE(res.iterations, kExactFactorMaxIters);   // exact factor
    EXPECT_LT(res.residual, 1e-8);
}

// The eliminator overload honours opts.partitioner: each kind selects the
// same first round as passing that built-in partitioner explicitly.
TEST(CustomEliminator, HonoursOptionsPartitioner) {
    const auto L = grid_laplacian(30, 30);
    auto first_round = [](const apxchol::factorization& F) {
        std::vector<apxchol::node_index> selected;
        const auto size = F.rounds.at(0).is_size;
        for (std::size_t v = 0; v < F.perm.size(); ++v)
            if (F.perm[v] < size) selected.push_back(apxchol::node_index(v));
        return selected;
    };
    auto by_option = [&](apxchol::partitioner_kind kind) {
        apxchol::factor_options opts;
        opts.partitioner = kind;
        return first_round(apxchol::factorize(csc(L), exact_clique_eliminator{}, opts));
    };
    auto by_instance = [&](auto partitioner) {
        return first_round(apxchol::factorize(csc(L), partitioner, exact_clique_eliminator{}));
    };
    const auto bg = by_option(apxchol::partitioner_kind::block_greedy);
    const auto pg = by_option(apxchol::partitioner_kind::priority_greedy);
    const auto bk = by_option(apxchol::partitioner_kind::baumann_kyng);
    EXPECT_EQ(bg, by_instance(apxchol::detail::block_greedy_partitioner{}));
    EXPECT_EQ(pg, by_instance(apxchol::detail::priority_greedy_partitioner{}));
    EXPECT_EQ(bk, by_instance(apxchol::detail::baumann_kyng_partitioner{}));
    EXPECT_NE(bg, bk);
}

TEST(CustomEliminator, ExactCliqueFactorIsExact) {
    // With the exact clique rule the factorization is exact Cholesky:
    // P^T L L^T P == A up to roundoff (no sampling at all).
    auto A = grid_laplacian(8, 8);
    auto b = apxchol::test::random_rhs(A.rows());
    auto solve_once = [&](apxchol::factor_storage_type storage) {
        auto F = apxchol::factorize(csc(A), exact_clique_eliminator{});
        apxchol::solve_options options;
        options.factor_opts.factor_storage = storage;
        apxchol::solver slv(csc(A), std::move(F), options);
        return slv.solve(cspan(b), 1e-12, 50);
    };
    for (bool fp16 : {false, true}) {
        if (fp16 && !apxchol::detail::omp_sptrsv::fp16_supported()) continue;
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
    auto b = apxchol::test::random_rhs(L.rows());

    priority_partitioner part;
    part.priority.resize(size_t(k) * k);
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j)
            part.priority[size_t(i) * k + j] =
                (i == 0 || j == 0 || i == k - 1 || j == k - 1) ? 1.0 : 0.0;

    auto F = apxchol::factorize(csc(L), std::move(part));
    EXPECT_FALSE(F.rounds.empty());

    apxchol::solver slv(csc(L), std::move(F));
    auto res = slv.solve(cspan(b), 1e-8, 500);
    EXPECT_LT(res.residual, 1e-8);
}

TEST(CustomPartitioner, LambdaViaAsPartitioner) {
    auto L = grid_laplacian(15, 15);
    auto b = apxchol::test::random_rhs(L.rows());

    // Trivial-but-valid rule: greedily select an independent set in the
    // natural active order (the selection answers membership queries).
    auto part = apxchol::as_partitioner(
        [](auto& G, std::span<const apxchol::node_index> active,
           const apxchol::partition_context&, apxchol::selection& out) {
            for (auto v : active) {
                bool ok = true;
                G.for_each_neighbor(v, [&](auto u) {
                    if (out.contains(u)) ok = false;
                });
                if (ok) out.add(v);
            }
        });

    auto F = apxchol::factorize(csc(L), std::move(part));
    apxchol::solver slv(csc(L), std::move(F));
    auto res = slv.solve(cspan(b), 1e-8, 500);
    EXPECT_LT(res.residual, 1e-8);
}

TEST(CustomPartitioner, GraphViewMatchesPrepassDegrees) {
    const auto L = grid_laplacian(15, 15);
    std::size_t checked = 0;
    auto part = apxchol::as_prepass_partitioner(
        [&](const apxchol::partition_graph& G,
            std::span<const apxchol::node_index> candidates,
            const apxchol::partition_context& ctx, apxchol::selection& out) {
            for (auto v : candidates) {
                std::size_t visited = 0;
                bool ok = true;
                const auto degree = G.for_each_neighbor(v, [&](auto u) {
                    ++visited;
                    EXPECT_TRUE(G.is_active(u));
                    if (out.contains(u)) ok = false;
                });
                EXPECT_EQ(degree, visited);
                EXPECT_EQ(degree, G.degree(v));
                EXPECT_EQ(degree, ctx.degrees[v]);
                ++checked;
                if (ok) out.add(v);
            }
        });
    const auto F = apxchol::factorize(csc(L), std::move(part));
    EXPECT_FALSE(F.rounds.empty());
    EXPECT_GT(checked, 0u);
}

TEST(SetFactor, ExternalFactorizationInPreconditioner) {
    auto L = grid_laplacian(15, 15);
    auto b = apxchol::test::random_rhs(L.rows());

    auto F = apxchol::factorize(csc(L));

    apxchol::solver M(csc(L), std::move(F), on_cpu);
    Eigen::VectorXd z = apxchol::test::vec(M.apply(cspan(b)));
    EXPECT_TRUE(z.allFinite());
    EXPECT_GT(z.norm(), 0.0);
}

TEST(FactorExport, InstalledFactorIsAlwaysAvailable) {
    auto L = grid_laplacian(10, 10);
    apxchol::solve_options opts;
    opts.backend = apxchol::solve_backend::cpu;

    apxchol::solver owner(csc(L), opts);
    const auto F = owner.factor();
    EXPECT_EQ(F.storage, apxchol::factor_storage_type::fp32);
    EXPECT_EQ(csc_access::vals(F.L).size(), size_t(F.L.nonZeros()));
    EXPECT_EQ(csc_access::inner(F.L).size(), size_t(F.L.nonZeros()));
    // A Laplacian's installed factor has an empty last row and column.
    const auto n = F.L.rows();
    EXPECT_EQ(csc_access::outer(F.L)[n - 1], csc_access::outer(F.L)[n]);
    for (auto row : csc_access::inner(F.L)) EXPECT_LT(row, n - 1);
}

TEST(InitialGuess, ExactX0ConvergesImmediately) {
    auto L = grid_laplacian(15, 15);
    auto b = apxchol::test::random_rhs(L.rows());

    apxchol::solver slv(csc(L), on_cpu);
    auto ref = slv.solve(cspan(b), 1e-10, 1000);
    ASSERT_LT(ref.residual, 1e-10);

    auto warm = slv.solve(cspan(b), 1e-8, 500, ref.x);
    EXPECT_EQ(warm.iterations, 0);
    EXPECT_LT(warm.residual, 1e-8);

    // A wrong-size x0 is rejected loudly (consistent with the adopting ctor).
    Eigen::VectorXd bad = Eigen::VectorXd::Ones(3);
    EXPECT_THROW(slv.solve(cspan(b), 1e-8, 500, cspan(bad)), std::invalid_argument);
}

TEST(OutputMemory, SolveIntoCallerBuffer) {
    auto L = grid_laplacian(15, 15);
    auto b = apxchol::test::random_rhs(L.rows());
    apxchol::solver slv(csc(L), on_cpu);

    auto ref = slv.solve(cspan(b));
    Eigen::VectorXd x(L.rows());
    auto res = slv.solve(cspan(b), apxchol::test::mspan(x), 1e-8, 500);
    EXPECT_EQ(res.x.size(), 0u);                 // solution lives in x, not res
    EXPECT_LT(res.residual, 1e-8);
    EXPECT_LT((x - apxchol::test::vec(ref.x)).norm(), 1e-10 * apxchol::test::vec(ref.x).norm() + 1e-14);

    Eigen::VectorXd wrong(3);
    EXPECT_THROW(slv.solve(cspan(b), apxchol::test::mspan(wrong)), std::invalid_argument);
}

TEST(SolveHonesty, ZeroIterationSolveDoesNotClaimConvergence) {
    auto L = grid_laplacian(10, 10);
    auto b = apxchol::test::random_rhs(L.rows());
    apxchol::solver slv(csc(L), on_cpu);

    // max_iter = 0: no PCG update ran, so the reported relative residual must
    // be the honest ||b - A*0|| / ||b|| = 1, never the field's 0.0 default.
    auto res = slv.solve(cspan(b), 1e-8, 0);
    EXPECT_EQ(res.iterations, 0);
    EXPECT_GE(res.residual, 1.0);
}

TEST(SolveHonesty, ZeroRhsReturnsZero) {
    const auto L = grid_laplacian(3, 3);
    apxchol::solver solver(csc(L), on_cpu);
    const Eigen::VectorXd b = Eigen::VectorXd::Zero(L.rows());

    const auto result = solver.solve(cspan(b));

    ASSERT_EQ(result.x.size(), static_cast<std::size_t>(L.rows()));
    EXPECT_TRUE(apxchol::test::vec(result.x).isZero(0.0));
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.residual, 0.0);
}

TEST(SolveHonesty, ZeroRhsReturnsZeroWithWarmStart) {
    const auto L = grid_laplacian(3, 3);
    apxchol::solver solver(csc(L), on_cpu);
    const Eigen::VectorXd b = Eigen::VectorXd::Zero(L.rows());
    const Eigen::VectorXd x0 = Eigen::VectorXd::LinSpaced(L.rows(), -2.0, 2.0);

    const auto result = solver.solve(cspan(b), 1e-8, 200, cspan(x0));

    ASSERT_EQ(result.x.size(), static_cast<std::size_t>(L.rows()));
    EXPECT_TRUE(apxchol::test::vec(result.x).isZero(0.0));
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.residual, 0.0);
}

TEST(SetFactor, ReleasedFactorIsRejected) {
    auto L = grid_laplacian(10, 10);
    apxchol::solver slv(csc(L), on_cpu);
    auto F = slv.factor();
    csc_access::release_values(F.L);

    EXPECT_THROW(apxchol::solver(csc(L), F, on_cpu),
                 std::invalid_argument);
}

}  // namespace
