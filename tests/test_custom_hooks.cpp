/// Tests for the public customization seams: custom eliminators, custom
/// (stateful) partitioners, adopting an external factorization, keep-factor,
/// and the PCG initial guess.
#include "apxchol.h"
#include "apxchol/solver/partitioner_helpers.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <span>
#include <vector>

namespace {
// RAII env override: these tests state the DEFAULT (fp32) SpTRSV storage
// contract, and the suite is also run with APXCHOL_SPTRSV_FP16=1 in the
// environment, so they pin it.
struct scoped_env {
    std::string name, saved; bool had = false;
    scoped_env(const char* var, const char* value) : name(var) {
        if (const char* e = std::getenv(var)) { had = true; saved = e; }
        if (value) setenv(var, value, 1); else unsetenv(var);
    }
    ~scoped_env() { if (had) setenv(name.c_str(), saved.c_str(), 1); else unsetenv(name.c_str()); }
};


// "Exact factor => O(1) PCG iterations" bound. An exact Cholesky factor stored
// at fp32 preconditions PCG to <= 3 iterations. (Under the fp16 STORAGE --
// APXCHOL_SPTRSV_FP16=1, see lowprec.h -- every off-diagonal carries a 2^-11
// relative rounding, so the "exact" factor is only an approximate one and PCG
// needs a few more; iteration count, not the residual floor, is what
// precision buys. These tests run at the default fp32 storage.)
constexpr int kExactFactorMaxIters = 3;

Eigen::SparseMatrix<double> grid_laplacian(int rows, int cols) {
    const int n = rows * cols;
    std::vector<Eigen::Triplet<double>> t;
    auto id = [cols](int i, int j) { return i * cols + j; };
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            int deg = 0;
            auto edge = [&](int i2, int j2) {
                if (i2 < 0 || j2 < 0 || i2 >= rows || j2 >= cols) return;
                t.emplace_back(id(i, j), id(i2, j2), -1.0);
                ++deg;
            };
            edge(i - 1, j); edge(i + 1, j); edge(i, j - 1); edge(i, j + 1);
            t.emplace_back(id(i, j), id(i, j), double(deg));
        }
    Eigen::SparseMatrix<double> L(n, n);
    L.setFromTriplets(t.begin(), t.end());
    return L;
}

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

    template<apxchol::incidence_storage Incidence>
    void find_partition(apxchol::graph<Incidence>& G,
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
                auto u = G.edge_target(idx, v);
                if (G.is_active(u) && out.contains(u)) { independent = false; break; }
            }
            if (independent) out.add(v);
        }
    }

private:
    std::vector<apxchol::node_index> order_;       // sorted candidate list
};
static_assert(apxchol::partitioner<priority_partitioner>);

using vp_graph = apxchol::graph<apxchol::vec_pool_incidence>;

TEST(CustomEliminator, ExactCliqueConverges) {
    auto L = grid_laplacian(20, 20);
    auto b = apxchol::generate_test_rhs(L.rows());

    // Matrix-level convenience overload (graph built internally).
    auto F = apxchol::factorize(L, exact_clique_eliminator{});
    EXPECT_GT(F.L.nonZeros(), 0u);

    apxchol::cpu_solver slv(L, std::move(F));
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

    apxchol::cpu_solver slv(L, std::move(F));
    auto res = slv.solve(b, 1e-8, 500);
    EXPECT_LE(res.iterations, kExactFactorMaxIters);   // exact factor
    EXPECT_LT(res.residual, 1e-8);
}

TEST(CustomEliminator, ExactCliqueFactorIsExact) {
    // These state the DEFAULT (fp32) storage contract, so pin it: the suite
    // is also run with APXCHOL_SPTRSV_FP16=1 in the environment.
    scoped_env fp32_storage("APXCHOL_SPTRSV_FP16", "0");

    // With the exact clique rule the factorization is exact Cholesky:
    // P^T L L^T P == A up to roundoff (no sampling at all).
    auto A = grid_laplacian(8, 8);
    auto b = apxchol::generate_test_rhs(A.rows());
    auto solve_once = [&] {
        auto G = apxchol::make_graph<vp_graph>(A);
        auto F = apxchol::factorize(std::move(G), exact_clique_eliminator{});
        apxchol::cpu_solver slv(A, std::move(F));
        return slv.solve(b, 1e-12, 50);
    };
#if defined(APXCHOL_USE_CUDA)
    // The GPU SpTRSV's fp16 factor storage is a RUNTIME mode, default ON
    // where a kernel backend resolves on the fp32 build -- so pin it per
    // sub-case: "exact factor => O(1) iterations" requires LOSSLESS factor
    // storage (under fp16 every stored off-diagonal carries a 2^-11 relative
    // rounding and the factor is only approximately exact: one extra
    // iteration on this grid). The residual floor is unaffected either way.
    for (bool fp16 : {false, true}) {
        SCOPED_TRACE(fp16 ? "fp16 storage" : "lossless storage");
        setenv("APXCHOL_SPTRSV_FP16", fp16 ? "1" : "0", /*overwrite=*/1);
        auto res = solve_once();
        EXPECT_LE(res.iterations, kExactFactorMaxIters + (fp16 ? 1 : 0));
        EXPECT_LT(res.residual, 1e-12);
    }
    unsetenv("APXCHOL_SPTRSV_FP16");
#else
    auto res = solve_once();
    // An exact factor preconditions PCG to convergence in O(1) iterations.
    EXPECT_LE(res.iterations, kExactFactorMaxIters);
    EXPECT_LT(res.residual, 1e-12);
#endif
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

    apxchol::cpu_solver slv(L, std::move(F));
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
                    auto u = G.edge_target(idx, v);
                    if (G.is_active(u) && out.contains(u)) { ok = false; break; }
                }
                if (ok) out.add(v);
            }
        });

    auto F = apxchol::factorize(L, std::move(part));
    apxchol::cpu_solver slv(L, std::move(F));
    auto res = slv.solve(b, 1e-8, 500);
    EXPECT_LT(res.residual, 1e-8);
}

TEST(SetFactor, ExternalFactorizationInPreconditioner) {
    auto L = grid_laplacian(15, 15);
    auto b = apxchol::generate_test_rhs(L.rows());

    auto G = apxchol::make_graph<vp_graph>(L);
    auto F = apxchol::factorize(std::move(G));

    apxchol::apx_cholesky M;
    M.set_factor(std::move(F));
    Eigen::VectorXd z = M.solve(b);
    EXPECT_TRUE(z.allFinite());
    EXPECT_GT(z.norm(), 0.0);
}

TEST(KeepFactor, ValuesSurviveWhenRequested) {
    auto L = grid_laplacian(10, 10);
    apxchol::solve_options opts;

    apxchol::cpu_solver released(L, opts);
    EXPECT_TRUE(released.preconditioner().factor().L.vals_.empty());

    opts.keep_factor_values = true;
    apxchol::cpu_solver kept(L, opts);
    const auto& FL = kept.preconditioner().factor().L;
    EXPECT_EQ(FL.vals_.size(), size_t(FL.nonZeros()));
    EXPECT_EQ(FL.inner_.size(), size_t(FL.nonZeros()));
}

TEST(InitialGuess, ExactX0ConvergesImmediately) {
    auto L = grid_laplacian(15, 15);
    auto b = apxchol::generate_test_rhs(L.rows());

    apxchol::cpu_solver slv(L);
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
    apxchol::cpu_solver slv(L);

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
    apxchol::cpu_solver slv(L);

    // max_iter = 0: no PCG update ran, so the reported relative residual must
    // be the honest ||b - A*0|| / ||b|| = 1, never the field's 0.0 default.
    auto res = slv.solve(b, 1e-8, 0);
    EXPECT_EQ(res.iterations, 0);
    EXPECT_GE(res.residual, 1.0);
}

TEST(SetFactor, ReleasedFactorIsRejected) {
    auto L = grid_laplacian(10, 10);
    apxchol::cpu_solver slv(L);  // default: factor values released after setup

    apxchol::apx_cholesky M;
    EXPECT_THROW(M.set_factor(slv.preconditioner().factor()),
                 std::invalid_argument);
}

}  // namespace
