#include <gtest/gtest.h>
#include "grid_laplacian.h"
#include "scoped_env.h"
#include "scoped_omp_threads.h"
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <tuple>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <omp.h>

#include "apxchol/solver/factorization.h"
#include "apxchol/solver/preconditioner.h"
#include "apxchol/graph/conversions.h"
#include "apxchol/graph/graph.h"
#include "apxchol/solver/partition/baumann_kyng.h"
#include "apxchol/solver/partition/priority_greedy.h"
#include "apxchol/solver/partitioner_helpers.h"
#include "apxchol/solver/solve.h"
#if defined(APXCHOL_USE_CUDA)
#include "apxchol/solver/gpu_block_frontend.h"
#include "gpu_preconditioner_fixture.h"
#include "apxchol/solver/elimination/gpu_round_shadow.h"
#endif

// ── Helpers ──────────────────────────────────────────

TEST(SetupDiagnostics, EndpointRadixSortMatchesComparisonSort) {
    std::mt19937_64 rng(1234567);
    std::vector<apxchol::node_index> values(20000);
    for (auto& value : values)
        value = static_cast<apxchol::node_index>(rng());
    values[0] = 0;
    values[1] = std::numeric_limits<apxchol::node_index>::max();
    values[2] = values[3] = 17;
    auto expected = values;
    std::sort(expected.begin(), expected.end());
    std::shuffle(values.begin(), values.end(), rng);
    std::vector<apxchol::node_index> scratch(values.size());
    std::vector<std::array<std::size_t, 256>> histograms(8);
#pragma omp parallel num_threads(8)
    apxchol::detail::parallel_radix_sort_node_indices(
        values, scratch, histograms, omp_get_thread_num(),
        omp_get_num_threads());
    EXPECT_EQ(values, expected);
}

TEST(EliminationDedup, PooledLocalHashPreservesOrderSumsAndEpochReuse) {
    apxchol::graph graph(12);
    graph.add_edge(0, 1, 2.0);
    graph.add_edge(0, 9, 4.0);  // 1 and 9 collide modulo the 8-slot table.
    graph.add_edge(0, 1, 3.0);
    graph.add_edge(0, 2, 5.0);

    apxchol::factorize_workspace::per_thread scratch;
    scratch.factor_entries =
        std::make_unique<std::pmr::monotonic_buffer_resource>();
    apxchol::detail::factor_col first;
    apxchol::detail::process_vertex(apxchol::tree_elimination{}, graph, 0, 42,
                                     first, scratch);

    ASSERT_EQ(first.entry_count, 3u);
    EXPECT_EQ(first.entries[0].neighbor, 1u);
    EXPECT_EQ(first.entries[1].neighbor, 9u);
    EXPECT_EQ(first.entries[2].neighbor, 2u);
    EXPECT_FLOAT_EQ(first.entries[0].value,
                    static_cast<float>(5.0 / std::sqrt(14.0)));
    EXPECT_FLOAT_EQ(first.entries[1].value,
                    static_cast<float>(4.0 / std::sqrt(14.0)));
    EXPECT_FLOAT_EQ(first.entries[2].value,
                    static_cast<float>(5.0 / std::sqrt(14.0)));

    apxchol::detail::factor_col second;
    apxchol::detail::process_vertex(apxchol::tree_elimination{}, graph, 0, 42,
                                     second, scratch);
    ASSERT_EQ(second.entry_count, first.entry_count);
    for (std::size_t i = 0; i < first.entry_count; ++i) {
        EXPECT_EQ(second.entries[i].neighbor, first.entries[i].neighbor);
        EXPECT_EQ(second.entries[i].value, first.entries[i].value);
    }
}

// Convert the owned sparse_csc factor to an Eigen::SparseMatrix for tests
// (the library no longer stores Eigen factors).
static Eigen::SparseMatrix<double> factor_to_eigen(const apxchol::sparse_csc& L) {
    const auto* outer = L.outerIndexPtr();
    const auto* inner = L.innerIndexPtr();
    const auto* vals  = L.valuePtr();
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<size_t>(L.nonZeros()));
    for (apxchol::node_index c = 0; c < L.cols(); ++c)
        for (apxchol::edge_index p = outer[c]; p < outer[c + 1]; ++p)
            trips.emplace_back(static_cast<int>(inner[p]), static_cast<int>(c),
                               apxchol::widen(vals[p]));
    Eigen::SparseMatrix<double> M(static_cast<int>(L.rows()), static_cast<int>(L.cols()));
    M.setFromTriplets(trips.begin(), trips.end());
    return M;
}

// Build an Eigen permutation from perm[v] = new position of original vertex v.
static Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>
perm_to_eigen(const std::vector<apxchol::node_index>& perm) {
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P(
        static_cast<Eigen::Index>(perm.size()));
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(perm.size()); ++i)
        P.indices()[i] = static_cast<int>(perm[i]);
    return P;
}

// L(i,i): scan column i of the CSC factor for the diagonal entry.
static double factor_diag(const apxchol::sparse_csc& L, int i) {
    const auto* outer = L.outerIndexPtr();
    const auto* inner = L.innerIndexPtr();
    const auto* vals  = L.valuePtr();
    for (apxchol::edge_index p = outer[i]; p < outer[i + 1]; ++p)
        if (static_cast<int>(inner[p]) == i) return apxchol::widen(vals[p]);
    return 0.0;
}

static Eigen::SparseMatrix<double> path_laplacian(int n) {
    apxchol::graph G(n);
    for (int i = 0; i + 1 < n; ++i)
        G.add_edge(i, i + 1, 1.0);
    return apxchol::laplacian(G);
}

using apxchol::test::grid_laplacian;

static Eigen::SparseMatrix<double> star_laplacian(int n) {
    apxchol::graph G(n);
    for (int i = 1; i < n; ++i)
        G.add_edge(0, i, 1.0);
    return apxchol::laplacian(G);
}

static Eigen::SparseMatrix<double> cycle_laplacian(int n) {
    apxchol::graph G(n);
    for (int i = 0; i < n; ++i)
        G.add_edge(i, (i + 1) % n, 1.0);
    return apxchol::laplacian(G);
}

static Eigen::SparseMatrix<double> weighted_grid_laplacian(int rows, int cols,
                                                           double w_horiz,
                                                           double w_vert) {
    apxchol::graph G(rows * cols);
    auto id = [cols](int r, int c) { return r * cols + c; };
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            if (r + 1 < rows) G.add_edge(id(r, c), id(r + 1, c), w_vert);
            if (c + 1 < cols) G.add_edge(id(r, c), id(r, c + 1), w_horiz);
        }
    return apxchol::laplacian(G);
}

TEST(PartitionerHelpers, ParallelDegreeQuantileMatchesNthElementExactly) {
    std::vector<std::array<size_t, 256>> histograms;
    for (size_t n : {size_t{1}, size_t{2}, size_t{257}, size_t{100000}}) {
        std::vector<apxchol::node_index> degrees(n);
        std::uint64_t state = 0x123456789abcdef0ULL;
        for (size_t i = 0; i < n; ++i) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            degrees[i] = (i % 3 == 0)
                ? static_cast<apxchol::node_index>((state >> 32) % 97)
                : static_cast<apxchol::node_index>(state >> 32);
        }
        for (double q : {0.01, 0.2, 0.5, 0.999}) {
            auto reference = degrees;
            size_t k = static_cast<size_t>(q * n);
            if (k >= n) k = n - 1;
            std::nth_element(reference.begin(), reference.begin() + k,
                             reference.end());
            EXPECT_EQ(apxchol::parallel_degree_quantile(
                          degrees, n, q, histograms),
                      reference[k])
                << "n=" << n << " q=" << q;
        }
    }
}

TEST(PartitionerHelpers, ParallelDegreeFilterPreservesCandidateOrder) {
    constexpr size_t n = 100000;
    std::vector<apxchol::node_index> active(n), degrees(n);
    for (size_t i = 0; i < n; ++i) {
        active[i] = static_cast<apxchol::node_index>(n - i - 1);
        degrees[i] = static_cast<apxchol::node_index>((i * 37 + i / 11) % 103);
    }
    std::vector<apxchol::node_index> expected, live(n), eligible;
    for (size_t i = 0; i < n; ++i)
        if (degrees[i] <= 23) expected.push_back(active[i]);
    std::vector<size_t> offsets;
    const size_t count = apxchol::parallel_ordered_degree_filter(
        active, degrees, 23, live, eligible, offsets);
    ASSERT_EQ(count, expected.size());
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), eligible.begin()));
    for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(live[active[i]], degrees[i]);
}

TEST(PartitionerHelpers, ParallelActiveFilterIsStable) {
    constexpr size_t n = 100000;
    std::vector<apxchol::node_index> active(n), expected;
    std::iota(active.begin(), active.end(), apxchol::node_index{0});
    for (auto v : active)
        if ((v * 17 + v / 13) % 11 < 4) expected.push_back(v);
    std::vector<apxchol::node_index> scratch;
    std::vector<size_t> offsets;
    apxchol::parallel_stable_active_filter(
        active, scratch, offsets,
        [](apxchol::node_index v) { return (v * 17 + v / 13) % 11 < 4; });
    EXPECT_EQ(active, expected);
}

// ── Typed test infrastructure ────────────────────────


// ── Factorization structure tests ────────────────────

class FactorizeTest : public ::testing::Test {
protected:
    apxchol::factorization factorize_with(
        const Eigen::SparseMatrix<double>& L, unsigned seed = 42) {
        return apxchol::factorize(L, {.seed = seed});
    }
};


TEST_F(FactorizeTest, PathGraphDimensions) {
    auto L = path_laplacian(10);
    auto F = this->factorize_with(L);
    EXPECT_EQ(F.L.rows(), 10);
    EXPECT_EQ(F.L.cols(), 10);
    EXPECT_EQ(F.perm.size(), 10);
}

TEST_F(FactorizeTest, GridGraphDimensions) {
    auto L = grid_laplacian(5, 5);
    auto F = this->factorize_with(L);
    EXPECT_EQ(F.L.rows(), 25);
    EXPECT_EQ(F.L.cols(), 25);
}

TEST_F(FactorizeTest, PermutationIsValid) {
    auto L = grid_laplacian(4, 4);
    auto F = this->factorize_with(L);

    const int n = F.L.rows();
    std::vector<int> seen(n, 0);
    for (int i = 0; i < n; ++i) {
        int idx = static_cast<int>(F.perm[i]);
        ASSERT_GE(idx, 0);
        ASSERT_LT(idx, n);
        seen[idx]++;
    }
    for (int i = 0; i < n; ++i)
        EXPECT_EQ(seen[i], 1) << "index " << i << " appears " << seen[i] << " times";
}

TEST_F(FactorizeTest, LowerTriangular) {
    auto L = grid_laplacian(5, 5);
    auto F = this->factorize_with(L);

    const int m = F.L.rows() - 1;
    const auto* outer = F.L.outerIndexPtr();
    const auto* inner = F.L.innerIndexPtr();
    for (int k = 0; k < static_cast<int>(F.L.outerSize()) && k < m; ++k)
        for (apxchol::edge_index p = outer[k]; p < outer[k + 1]; ++p) {
            const int row = static_cast<int>(inner[p]);
            const int col = k;
            if (row < m && col < m) {
                EXPECT_GE(row, col)
                    << "upper-triangle entry at (" << row << "," << col << ")";
            }
        }
}

TEST_F(FactorizeTest, PositiveDiagonal) {
    auto L = grid_laplacian(5, 5);
    auto F = this->factorize_with(L);

    const int m = F.L.rows() - 1;
    for (int i = 0; i < m; ++i)
        EXPECT_GT(factor_diag(F.L, i), 0.0) << "zero/negative diagonal at " << i;
}

TEST_F(FactorizeTest, Deterministic) {
    auto L = grid_laplacian(5, 5);
    auto F1 = this->factorize_with(L, 42);
    auto F2 = this->factorize_with(L, 42);

    EXPECT_EQ(F1.L.nonZeros(), F2.L.nonZeros());
    EXPECT_EQ(F1.perm, F2.perm);
    Eigen::SparseMatrix<double> diff = factor_to_eigen(F1.L) - factor_to_eigen(F2.L);
    EXPECT_LT(diff.norm(), 1e-14);
}

// ── Determinism of the PARALLEL selection path ───────
//
// FactorizeTest.Deterministic above factorizes a 5x5 grid: 25 candidates,
// far below factor_options::omp_threshold, so every round takes the SERIAL
// branch of the partitioner and the parallel one is never executed. This test
// executes it, and is the regression guard for two schedule-dependent outputs
// fixed 2026-08-20:
//
//   * block_greedy's cross-block conflict resolution read the shared chosen[]
//     mask while other threads were clearing it, so whether a boundary pick
//     dropped depended on who got there first. That changed the round's
//     independent set, hence the elimination order, hence the factor's
//     STRUCTURE and its nnz, run to run at one seed and one thread count.
//     Measured at the parent commit on a 32-core box, standalone (i.e. not
//     under ctest's OMP_NUM_THREADS=4 cap), SpTRSVSetupMemory.SetupConsuming-
//     ReleasesTheFactorAndSolvesIdentically — which factorizes a 120x120 grid
//     twice and compares — failed 19/50 at T=8, 16/50 at T=16, 19/50 at T=32,
//     and 0/50 at T=1 and T=4.
// Both are selection-side, so one storage suffices here; the storage x
// partitioner matrix was checked separately.
namespace {
// Raise the OpenMP team size for one test and put it back (gtest runs the
// whole binary in one process; ctest pins OMP_NUM_THREADS=4 for it).
struct scoped_threads {
    int saved = 1;
    int saved_dynamic = 0;
    explicit scoped_threads(int n) {
        saved = omp_get_max_threads();
        saved_dynamic = omp_get_dynamic();
        omp_set_dynamic(0);
        omp_set_num_threads(n);
    }
    ~scoped_threads() {
        omp_set_num_threads(saved);
        omp_set_dynamic(saved_dynamic);
    }
};


// Byte-for-byte equality of two factors: same column pointers, same row
// indices, same values. Structure AND values, not a norm.
void expect_same_factor(const apxchol::factorization& a,
                        const apxchol::factorization& b,
                        const std::string& what) {
    ASSERT_EQ(a.perm, b.perm) << what << ": elimination order differs";
    ASSERT_EQ(a.L.nonZeros(), b.L.nonZeros()) << what << ": factor nnz differs";
    const std::size_t nc = static_cast<std::size_t>(a.L.cols()) + 1;
    const std::size_t nz = static_cast<std::size_t>(a.L.nonZeros());
    EXPECT_EQ(std::memcmp(a.L.outerIndexPtr(), b.L.outerIndexPtr(),
                          nc * sizeof(apxchol::edge_index)), 0)
        << what << ": column pointers differ";
    EXPECT_EQ(std::memcmp(a.L.innerIndexPtr(), b.L.innerIndexPtr(),
                          nz * sizeof(apxchol::node_index)), 0)
        << what << ": row indices differ (factor STRUCTURE is not reproducible)";
    EXPECT_EQ(std::memcmp(a.L.valuePtr(), b.L.valuePtr(),
                          nz * sizeof(apxchol::factor_value_t)), 0)
        << what << ": factor values differ";
}

#if !defined(APXCHOL_POOL_FP32)
// Parallel FP64 graph updates may change later sampling, including structure.
// Check the actual rebuilt factor and solve the original system with that
// factor; comparing two independent rebuilds is not a numerical accuracy test.
// Exportable factorization uses these host updates in CUDA builds too; the
// adopting solver installs this actual factor through the normal solve path.
void expect_valid_rebuilt_laplacian_factor(
        const Eigen::SparseMatrix<double>& A,
        const apxchol::factorization& F) {
    const auto n = static_cast<apxchol::node_index>(A.rows());
    ASSERT_GT(n, 1u);
    ASSERT_FALSE(F.sddm);
    ASSERT_EQ(F.L.rows(), n);
    ASSERT_EQ(F.perm.size(), n);
    ASSERT_EQ(F.L.outer_.size(), static_cast<std::size_t>(n) + 1);
    ASSERT_EQ(F.L.outer_.front(), 0u);
    ASSERT_EQ(F.L.inner_.size(), F.L.nonZeros());
    ASSERT_EQ(F.L.vals_.size(), F.L.nonZeros());
    std::vector<bool> seen(n, false);
    for (const auto v : F.perm) {
        ASSERT_LT(v, n);
        ASSERT_FALSE(seen[v]);
        seen[v] = true;
    }
    for (apxchol::node_index col = 0; col < n; ++col) {
        const auto begin = F.L.outer_[col], end = F.L.outer_[col + 1];
        ASSERT_LE(begin, end);
        ASSERT_LE(end, F.L.nonZeros());
        if (col + 1 < n) {
            ASSERT_LT(begin, end);
            EXPECT_EQ(F.L.inner_[begin], col);
            EXPECT_GT(F.L.vals_[begin], 0.0f);
        }
        for (auto p = begin; p < end; ++p) {
            EXPECT_GE(F.L.inner_[p], col);
            EXPECT_LT(F.L.inner_[p], n);
            EXPECT_TRUE(std::isfinite(F.L.vals_[p]));
        }
    }
    if (::testing::Test::HasFailure()) return;
    const auto b = apxchol::generate_test_rhs(n);
    apxchol::cpu_solver solver(A, F, {.tol = 1e-8, .max_iter = 500});
    const auto first = solver.solve(b);
    const auto repeated = solver.solve(b);
    ASSERT_TRUE(first.x.allFinite());
    EXPECT_LE(first.iterations, 500);
    EXPECT_LT(first.residual, 1e-8);
    EXPECT_LT((A * first.x - b).norm() / b.norm(), 1e-8);
    // Same owned factor and RHS retain the exact solve-repeatability contract.
    EXPECT_EQ(first.iterations, repeated.iterations);
    EXPECT_EQ(first.residual, repeated.residual);
    EXPECT_TRUE((first.x.array() == repeated.x.array()).all());
}
#endif

template<class Partitioner>
void expect_same_graph_selection(
        const apxchol::graph& snapshot,
        const std::vector<apxchol::node_index>& candidates,
        const apxchol::partition_context& context,
        const Partitioner& initial) {
    SCOPED_TRACE(Partitioner::name);
    std::vector<apxchol::node_index> reference;
    for (int repeat = 0; repeat < 4; ++repeat) {
        // Copy graph AND selector state: pruning and BK's round/degree estimate
        // must not make the inputs of the repeated calls differ.
        auto graph = snapshot;
        auto partitioner = initial;
        apxchol::selection selected;
        selected.reset(graph.n());
        partitioner.find_partition(graph, candidates, context, selected);
        const auto result = selected.finalize().data;
        ASSERT_FALSE(result.empty());
        if (repeat == 0) reference = result;
        else EXPECT_EQ(result, reference); // Exact insertion order, never sorted.
        std::vector<bool> eligible(graph.n(), false), in_set(graph.n(), false);
        for (const auto v : candidates) eligible[v] = true;
        for (const auto v : result) {
            ASSERT_LT(v, graph.n());
            EXPECT_TRUE(eligible[v]);
            EXPECT_TRUE(graph.is_active(v));
            EXPECT_FALSE(in_set[v]);
            in_set[v] = true;
        }
        for (const auto v : result) {
            for (const auto edge : graph.adj(v)) {
                const auto u = edge.to;
                if (graph.is_active(u) && u != v) { EXPECT_FALSE(in_set[u]); }
            }
        }
    }
}
} // namespace

TEST(FactorAssembly, SerialAndParallelInvariantScansMatchByteForByte) {
    constexpr apxchol::node_index n = 32768;
    std::vector<apxchol::detail::factor_col> cols(n);
    std::vector<apxchol::detail::factor_entry> entries(n - 1);
    auto vertex_at = [=](apxchol::node_index i) {
        // n is a power of two and 5 is odd, hence this is a permutation.
        return static_cast<apxchol::node_index>((5ULL * i) % n);
    };
    for (apxchol::node_index i = 0; i < n; ++i) {
        cols[i].vertex = vertex_at(i);
        cols[i].diag = static_cast<apxchol::factor_value_t>(2.0 + (i % 7));
        if (i + 1 < n) {
            entries[i] = {vertex_at(i + 1),
                          static_cast<apxchol::factor_value_t>(0.25 + (i % 5))};
            cols[i].entries = &entries[i];
            cols[i].entry_count = 1;
        }
    }

    apxchol::factorization serial;
    {
        const scoped_threads team(1);
        apxchol::detail::build_csc(serial, cols, n, nullptr);
    }
    ASSERT_EQ(serial.perm.size(), static_cast<std::size_t>(n));
    ASSERT_EQ(serial.L.rows(), n);
    ASSERT_EQ(serial.L.cols(), n);
    ASSERT_EQ(serial.L.nonZeros(),
              static_cast<apxchol::edge_index>(2 * n - 1));
    const auto* outer = serial.L.outerIndexPtr();
    const auto* inner = serial.L.innerIndexPtr();
    const auto* values = serial.L.valuePtr();
    for (apxchol::node_index i = 0; i < n; ++i) {
        EXPECT_EQ(serial.perm[vertex_at(i)], i);
        EXPECT_EQ(outer[i], static_cast<apxchol::edge_index>(2) * i);
        const apxchol::edge_index pos = outer[i];
        EXPECT_EQ(inner[pos], i);
        EXPECT_EQ(values[pos],
                  static_cast<apxchol::factor_value_t>(2.0 + (i % 7)));
        if (i + 1 < n) {
            EXPECT_EQ(inner[pos + 1], i + 1);
            EXPECT_EQ(values[pos + 1],
                      static_cast<apxchol::factor_value_t>(
                          -(0.25 + (i % 5))));
        }
    }
    EXPECT_EQ(outer[n], static_cast<apxchol::edge_index>(2 * n - 1));

    for (const int threads : {2, 3, 8, 16}) {
        apxchol::factorization parallel;
        {
            const scoped_threads team(threads);
            int actual_threads = 0;
            #pragma omp parallel
            #pragma omp masked
            actual_threads = omp_get_num_threads();
            ASSERT_EQ(actual_threads, threads);
            apxchol::detail::build_csc(parallel, cols, n, nullptr);
        }
        expect_same_factor(
            serial, parallel,
            "serial vs invariant assembly at T=" + std::to_string(threads));
    }
}



TEST(VecPoolAos, ParallelRebuiltFactorsPreserveSolveContract) {
    const scoped_threads team(16);
    const auto L = weighted_grid_laplacian(120, 120, 7.25, 0.03125);
    apxchol::factor_options opts;
    opts.seed = 19;
    opts.omp_threshold = 256;

    const auto baseline = apxchol::factorize(
        L, opts);
    const auto repeated = apxchol::factorize(
        L, opts);
#if !defined(APXCHOL_POOL_FP32)
    ASSERT_NO_FATAL_FAILURE(expect_valid_rebuilt_laplacian_factor(L, baseline));
    ASSERT_NO_FATAL_FAILURE(expect_valid_rebuilt_laplacian_factor(L, repeated));
#else
    expect_same_factor(baseline, repeated,
                       "AoS preassigned endpoint offsets at T=16");
#endif
    // Exact production slot/scatter coverage lives in
    // GpuRoundShadowReference.ParallelProductionApplyMatchesTheCanonicalReference.
}

TEST(VecPoolAos, AutoIncrementalDegreesPreserveTheFactorByteForByte) {
    const scoped_threads team(8);
    constexpr int clique_size = 32;
    // 4096 candidates exceed the RTX 4090's 3648 resident-region handoff
    // capacity, so the CUDA build's forced block frontend owns real early
    // rounds before handing the graph back to this host-cache path.
    constexpr int clique_count = 128;
    constexpr int n = clique_size * clique_count;
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(clique_count * clique_size * clique_size);
    for (int c = 0; c < clique_count; ++c) {
        const int first = c * clique_size;
        for (int u = 0; u < clique_size; ++u) {
            triplets.emplace_back(first + u, first + u, clique_size - 1.0);
            for (int v = 0; v < clique_size; ++v)
                if (u != v) triplets.emplace_back(first + u, first + v, -1.0);
        }
    }
    Eigen::SparseMatrix<double> L(n, n);
    L.setFromTriplets(triplets.begin(), triplets.end());
    L.makeCompressed();

    apxchol::factor_options opts;
    opts.seed = 19;
    opts.omp_threshold = 16;
    opts.min_is_fraction = 0.0;
    auto run = [&](const char* mode) {
        const scoped_env gate(
            "APXCHOL_INCREMENTAL_DEGREE_SPARSE", mode);
        return apxchol::factorize(
            L, opts);
    };
    const auto baseline = run("0");
    const auto automatic = run(nullptr);
    const auto named_auto = run("auto");
    ASSERT_FALSE(baseline.rounds.empty());
    ASSERT_EQ(baseline.rounds.front().is_size,
              static_cast<std::size_t>(clique_count));
    EXPECT_TRUE(apxchol::detail::incremental_degree_worthwhile(
        n, baseline.rounds.front().avg_deg,
        clique_count * (clique_size - 1), 8));
    expect_same_factor(baseline, automatic,
                       "full prune vs default incremental degrees");
    expect_same_factor(automatic, named_auto,
                       "default vs explicitly named auto degrees");
}

TEST(VecPoolAos, IncrementalDegreeCacheDoesNotLeakIntoBkResidualLoop) {
    const scoped_threads team(8);
    constexpr int small_size = 32;
    constexpr int small_count = 72;
    constexpr int residual_size = 100;
    constexpr int n = small_size * small_count + residual_size;
    std::vector<Eigen::Triplet<double>> triplets;
    auto add_clique = [&](int first, int count) {
        for (int u = 0; u < count; ++u) {
            triplets.emplace_back(first + u, first + u, count - 1.0);
            for (int v = 0; v < count; ++v)
                if (u != v)
                    triplets.emplace_back(first + u, first + v, -1.0);
        }
    };
    for (int c = 0; c < small_count; ++c)
        add_clique(c * small_size, small_size);
    add_clique(small_count * small_size, residual_size);
    Eigen::SparseMatrix<double> L(n, n);
    L.setFromTriplets(triplets.begin(), triplets.end());
    L.makeCompressed();

    apxchol::factor_options opts;
    opts.seed = 19;
    opts.omp_threshold = 16;
    opts.parallel_residual_threshold = 5;
    auto run = [&](const char* mode) {
        const scoped_env gate(
            "APXCHOL_INCREMENTAL_DEGREE_SPARSE", mode);
        return apxchol::factorize(
            L, opts);
    };
    const auto baseline = run("0");
    const auto automatic = run(nullptr);
    ASSERT_FALSE(baseline.rounds.empty());
    EXPECT_TRUE(apxchol::detail::incremental_degree_worthwhile(
        n, baseline.rounds.front().avg_deg,
        small_count * (small_size - 1), 8))
        << "fixture did not activate incremental degree maintenance";
    std::size_t rounded = 0;
    for (const auto& round : baseline.rounds) rounded += round.is_size;
    EXPECT_GE(rounded, static_cast<std::size_t>(n - 5))
        << "fixture did not exercise the BK residual loop";
    expect_same_factor(baseline, automatic,
                       "incremental main selector followed by BK residual");
}

TEST(PruneAndDegrees, SkipAboveReportsTheRawCountAndLeavesTheVertexUnwalked) {
    // Hub 0 with eight neighbours, three of them already eliminated.
    apxchol::graph graph(10);
    for (apxchol::node_index v = 1; v <= 8; ++v) graph.add_edge(0, v, 1.0);
    graph.add_edge(8, 9, 1.0);
    for (apxchol::node_index v : {1u, 2u, 3u}) graph.deactivate(v);
    const std::vector<apxchol::node_index> active{0, 8, 9};
    std::vector<apxchol::node_index> degrees;
    constexpr std::size_t serial = std::size_t{1} << 20;

    // Raw count 8 > 4: reported as is, dead entries left in place. The others
    // are walked as usual.
    const double skipped_average = apxchol::prune_and_degrees(
        graph, std::span<const apxchol::node_index>(active), degrees, serial, 4);
    ASSERT_EQ(degrees.size(), 3u);
    EXPECT_EQ(degrees[0], 8u);
    EXPECT_EQ(graph.adj_count(0), 8u);
    EXPECT_EQ(degrees[1], 2u);
    EXPECT_EQ(degrees[2], 1u);
    EXPECT_DOUBLE_EQ(skipped_average, 11.0 / 3.0);

    // A count AT the cutoff is walked; so is everything without a cutoff.
    apxchol::prune_and_degrees(
        graph, std::span<const apxchol::node_index>(active), degrees, serial, 8);
    EXPECT_EQ(degrees[0], 5u);
    EXPECT_EQ(graph.adj_count(0), 5u);
    const double full_average = apxchol::prune_and_degrees(
        graph, std::span<const apxchol::node_index>(active), degrees, serial);
    EXPECT_EQ(degrees[0], 5u);
    EXPECT_DOUBLE_EQ(full_average, 8.0 / 3.0);
}

TEST(PruneAndDegrees, AuditCountsTheEligibleVerticesASkipWouldHide) {
    // Hub 0 has eight raw entries but only two live neighbours left: a skip
    // at cutoff 4 would keep it away from a selector whose threshold is 3.
    // Hub 9 has six raw entries, all live: above the cutoff, rightly skipped.
    apxchol::graph graph(16);
    for (apxchol::node_index v = 1; v <= 8; ++v) graph.add_edge(0, v, 1.0);
    for (apxchol::node_index v = 10; v <= 15; ++v) graph.add_edge(9, v, 1.0);
    for (apxchol::node_index v = 1; v <= 6; ++v) graph.deactivate(v);
    const std::vector<apxchol::node_index> active{0, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    std::vector<apxchol::node_index> degrees;
    constexpr std::size_t serial = std::size_t{1} << 20;
    std::size_t hidden = 99;
    apxchol::prune_and_degrees(
        graph, std::span<const apxchol::node_index>(active), degrees, serial,
        /*skip_above=*/0, /*audit_above=*/4, /*audit_eligible=*/3, &hidden);
    EXPECT_EQ(hidden, 1u);
    EXPECT_EQ(degrees[0], 2u);          // the audit is a full walk
    EXPECT_EQ(graph.adj_count(0), 2u);
    EXPECT_EQ(degrees[3], 6u);

    // Walked once, nothing is hidden any more.
    apxchol::prune_and_degrees(
        graph, std::span<const apxchol::node_index>(active), degrees, serial,
        0, 4, 3, &hidden);
    EXPECT_EQ(hidden, 0u);
}

TEST(PruneSkip, HubGridSolvesAtEveryFactorIncludingTheAggressiveOne) {
    // A grid declines the incremental-degree cache in its first round (walk
    // traffic / update work is about 2), which is what arms the prune skip;
    // the hubs are the vertices it then leaves unwalked. Factor 1 skips every
    // vertex above the previous threshold, far more than the default 4: the
    // selector and the eliminator must cope with adjacency nobody pruned.
    const scoped_threads team(8);
    constexpr int side = 48, n_grid = side * side, hubs = 6, n = n_grid + hubs;
    std::mt19937_64 rng(20260917);
    std::vector<Eigen::Triplet<double>> triplets;
    std::vector<double> diagonal(n, 1e-3);
    auto edge = [&](int u, int v, double w) {
        triplets.emplace_back(u, v, -w); triplets.emplace_back(v, u, -w);
        diagonal[u] += w; diagonal[v] += w;
    };
    for (int r = 0; r < side; ++r)
        for (int c = 0; c < side; ++c) {
            if (c + 1 < side) edge(r * side + c, r * side + c + 1, 1.0);
            if (r + 1 < side) edge(r * side + c, (r + 1) * side + c, 1.0);
        }
    for (int h = 0; h < hubs; ++h)
        for (int v = h; v < n_grid; v += 5 + h) edge(n_grid + h, v, 0.5);
    for (int v = 0; v < n; ++v) triplets.emplace_back(v, v, diagonal[v]);
    Eigen::SparseMatrix<double> A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();
    Eigen::VectorXd b(n);
    for (int v = 0; v < n; ++v) b[v] = std::sin(0.31 * v) + 0.25;

    apxchol::solve_options opts;
    opts.tol = 1e-8; opts.max_iter = 500; opts.factor_opts.seed = 7;
    opts.factor_opts.omp_threshold = 16;
    auto run = [&](const char* factor) {
        const scoped_env gate("APXCHOL_PRUNE_SKIP", factor);
        apxchol::cpu_solver solver(A, opts);
        return solver.solve(b);
    };
    const auto off = run("0");
    const auto standard = run(nullptr);
    const auto aggressive = run("1");
    for (const auto* result : {&off, &standard, &aggressive}) {
        EXPECT_LT(result->residual, 1e-8);
        EXPECT_LT((A * result->x - b).norm() / b.norm(), 1e-7);
    }
    EXPECT_LE(standard.iterations, off.iterations + 3);
    EXPECT_LE(aggressive.iterations, 2 * off.iterations);

    // The fixture must really arm the rule: an unwalked hub is reported at its
    // raw count, dead entries included, so some round's average degree is
    // larger than the exact one.
    auto round_degrees = [&](const char* factor) {
        const scoped_env gate("APXCHOL_PRUNE_SKIP", factor);
        const auto f = apxchol::factorize(
            A, opts.factor_opts);
        std::vector<double> avg;
        for (const auto& round : f.rounds) avg.push_back(round.avg_deg);
        return avg;
    };
    const auto exact = round_degrees("0");
    const auto skipped = round_degrees(nullptr);
    ASSERT_GE(exact.size(), 3u);
    bool some_round_overestimates = false;
    for (std::size_t r = 0; r < std::min(exact.size(), skipped.size()); ++r)
        some_round_overestimates |= skipped[r] > exact[r];
    EXPECT_TRUE(some_round_overestimates)
        << "fixture never skipped a walk at the default factor";
}

TEST(FactorizeDeterminism, SameGraphParallelSelectionIsExact) {
    const scoped_threads team(16);
    int actual_threads = 0;
    #pragma omp parallel
    #pragma omp single
    actual_threads = omp_get_num_threads();
    ASSERT_EQ(actual_threads, 16);
    constexpr apxchol::node_index n = 1024, block = n / 16;
    apxchol::graph initial(n);
    for (apxchol::node_index v = 0; v < n; ++v) {
        initial.add_edge(v, (v + 1) % n, 1.0);
        // Chains cross every worker's block, exercising snapshot-based
        // conflict resolution rather than only independent local picks.
        if (v + block < n) initial.add_edge(v, v + block, 1.0);
    }
    for (const bool leave_dead_neighbors : {false, true}) {
        SCOPED_TRACE(leave_dead_neighbors);
        auto snapshot = initial;
        if (leave_dead_neighbors) {
            for (apxchol::node_index v = 0; v < n; v += 17)
                snapshot.deactivate(v);
        }
        std::vector<apxchol::node_index> candidates, compact_degrees;
        for (apxchol::node_index v = 0; v < n; ++v)
            if (snapshot.is_active(v)) candidates.push_back(v);
        auto degree_graph = snapshot;
        (void)apxchol::prune_and_degrees(
            degree_graph, candidates, compact_degrees, 256);
        std::vector<apxchol::node_index> degrees(n, 0);
        for (std::size_t i = 0; i < candidates.size(); ++i)
            degrees[candidates[i]] = compact_degrees[i];
        apxchol::partition_context context{
            .options = {.degree_quantile = 0.0, .degree_multiplier = 100.0},
            .seed = 3, .omp_threshold = 256, .cp = nullptr, .degrees = degrees,
        };
        expect_same_graph_selection(snapshot, candidates, context,
                                    apxchol::block_greedy_partitioner{});
        apxchol::priority_greedy_partitioner priority;
        priority.round = 3;
        expect_same_graph_selection(snapshot, candidates, context, priority);
        apxchol::baumann_kyng_partitioner bk;
        bk.round = 3;
        bk.est_avg_degree = 4.0;
        expect_same_graph_selection(snapshot, candidates, context, bk);
    }
}

TEST(FactorizeDeterminism, ParallelRebuiltFactorsPreserveSolveContract) {
    // 14400 candidates against a 256-vertex OpenMP gate: most of the ~45
    // elimination rounds take the parallel branch. 16 threads whatever the box
    // has -- oversubscription is welcome here, it widens the window the
    // block_greedy race needed.
    const scoped_threads team(16);
    const auto L = grid_laplacian(120, 120);
    for (const char* sel :
         {"block_greedy", "priority_greedy", "baumann_kyng"}) {
        apxchol::factor_options opts;
        opts.seed = 3;
        opts.omp_threshold = 256;
        opts.is_select = sel;
        const auto ref = apxchol::factorize(L, opts);
        ASSERT_GT(ref.L.nonZeros(), 70000) << sel;   // the parallel path really ran
#if !defined(APXCHOL_POOL_FP32)
        ASSERT_NO_FATAL_FAILURE(expect_valid_rebuilt_laplacian_factor(L, ref));
#endif
        for (int rep = 1; rep <= 3; ++rep) {
            const auto F = apxchol::factorize(L, opts);
#if !defined(APXCHOL_POOL_FP32)
            SCOPED_TRACE(std::string(sel) + " rep " + std::to_string(rep));
            ASSERT_NO_FATAL_FAILURE(expect_valid_rebuilt_laplacian_factor(L, F));
#else
            expect_same_factor(ref, F, std::string(sel) + " rep " + std::to_string(rep));
#endif
            if (::testing::Test::HasFatalFailure()) return;
        }
    }
}

// The T=1 half of the contract: one thread, byte-identical factor. Cheap and
// unconditional -- it holds on the serial build too.
TEST(FactorizeDeterminism, SingleThreadedFactorizationIsByteIdentical) {
    const scoped_threads team(1);
    const auto L = grid_laplacian(60, 60);
    apxchol::factor_options opts;
    opts.seed = 11;
    const auto a = apxchol::factorize(L, opts);
    const auto b = apxchol::factorize(L, opts);
    expect_same_factor(a, b, "T=1");
}

TEST_F(FactorizeTest, DifferentSeeds) {
    auto L = grid_laplacian(8, 8);
    auto F1 = this->factorize_with(L, 1);
    auto F2 = this->factorize_with(L, 999);
    EXPECT_EQ(F1.L.rows(), F2.L.rows());
}

TEST_F(FactorizeTest, SmallPath) {
    auto L = path_laplacian(3);
    auto F = this->factorize_with(L);
    EXPECT_EQ(F.L.rows(), 3);
    EXPECT_GT(F.L.nonZeros(), 0);
}

TEST_F(FactorizeTest, StarGraph) {
    auto L = star_laplacian(10);
    auto F = this->factorize_with(L);
    EXPECT_EQ(F.L.rows(), 10);
}

TEST_F(FactorizeTest, CycleGraph) {
    auto L = cycle_laplacian(20);
    auto F = this->factorize_with(L);
    EXPECT_EQ(F.L.rows(), 20);
}

TEST_F(FactorizeTest, TwoVertices) {
    auto L = path_laplacian(2);
    auto F = this->factorize_with(L);
    EXPECT_EQ(F.L.rows(), 2);
}

TEST_F(FactorizeTest, NnzPositive) {
    auto L = grid_laplacian(10, 10);
    auto F = this->factorize_with(L);
    EXPECT_GT(F.L.nonZeros(), 0);
}

// ── Preconditioner quality tests (PCG convergence) ───

class SolveTest : public ::testing::Test {
protected:
    apxchol::solve_result solve_with(
        const Eigen::SparseMatrix<double>& L,
        const Eigen::VectorXd& b,
        double tol = 1e-6, int max_iter = 500) {
        return apxchol::solve(L, b,
            {.tol = tol, .max_iter = max_iter,
             .factor_opts = {.seed = 42}});
    }
};


TEST_F(SolveTest, PathGraphConverges) {
    auto L = path_laplacian(50);
    auto b = apxchol::generate_test_rhs(50);
    auto res = this->solve_with(L, b);
    EXPECT_LT(res.residual, 1e-5);
}

TEST_F(SolveTest, GridGraphConverges) {
    auto L = grid_laplacian(10, 10);
    auto b = apxchol::generate_test_rhs(100);
    auto res = this->solve_with(L, b);
    EXPECT_LT(res.residual, 1e-5);
}

TEST_F(SolveTest, StarGraphConverges) {
    auto L = star_laplacian(30);
    auto b = apxchol::generate_test_rhs(30);
    auto res = this->solve_with(L, b);
    EXPECT_LT(res.residual, 1e-5);
}

TEST_F(SolveTest, CycleGraphConverges) {
    auto L = cycle_laplacian(50);
    auto b = apxchol::generate_test_rhs(50);
    auto res = this->solve_with(L, b);
    EXPECT_LT(res.residual, 1e-5);
}

TEST_F(SolveTest, WeightedGridConverges) {
    auto L = weighted_grid_laplacian(10, 10, 100.0, 1.0);
    auto b = apxchol::generate_test_rhs(100);
    auto res = this->solve_with(L, b, 1e-6, 1000);
    EXPECT_LT(res.residual, 1e-4);
}

TEST_F(SolveTest, LargerGridConverges) {
    auto L = grid_laplacian(20, 20);
    auto b = apxchol::generate_test_rhs(400);
    auto res = this->solve_with(L, b, 1e-6, 1000);
    EXPECT_LT(res.residual, 1e-5);
}

TEST_F(SolveTest, SolutionSatisfiesSystem) {
    auto L = grid_laplacian(10, 10);
    auto b = apxchol::generate_test_rhs(100);
    auto res = this->solve_with(L, b, 1e-8);

    Eigen::VectorXd Lx = L * res.x;
    Lx.array() -= Lx.mean();
    Eigen::VectorXd bc = b;
    bc.array() -= bc.mean();
    double rel_err = (Lx - bc).norm() / bc.norm();
    EXPECT_LT(rel_err, 1e-5);
}

TEST_F(SolveTest, DeterministicWithSameSeed) {
    auto L = grid_laplacian(8, 8);
    auto b = apxchol::generate_test_rhs(64);

    auto r1 = this->solve_with(L, b);
    auto r2 = this->solve_with(L, b);
    EXPECT_EQ(r1.iterations, r2.iterations);
    EXPECT_DOUBLE_EQ(r1.residual, r2.residual);
}

// ── Performance / timing sanity (storage-independent) ─

TEST(Solve, TimingsReported) {
    auto L = grid_laplacian(10, 10);
    auto b = apxchol::generate_test_rhs(100);
    auto res = apxchol::solve(L, b);
    EXPECT_GT(res.timings.total("setup"), 0.0);
    // After 53e1d1d the PCG-loop ops are grouped under "pcg"; on the CPU path
    // the precond.solve() triangular-solve subtree nests as pcg.solve, while
    // the GPU-resident PCG records one pcg.gpu_pcg_loop leaf instead.
    EXPECT_GT(res.timings.total("pcg"), 0.0);
    if (res.backend == apxchol::solve_backend::gpu) {
        EXPECT_GT(res.timings.total("pcg.gpu_pcg_loop"), 0.0);
    } else {
        EXPECT_EQ(res.backend, apxchol::solve_backend::cpu);
        EXPECT_GT(res.timings.total("pcg.solve"), 0.0);
    }
}

// ── SDDM support tests ────────────────────────────────

/// Build an SDDM matrix: grid Laplacian + positive diagonal perturbation.
static Eigen::SparseMatrix<double> sddm_grid(int rows, int cols, double excess) {
    auto L = grid_laplacian(rows, cols);
    const int n = rows * cols;
    // Add excess to diagonal (makes it strictly diagonally dominant).
    for (int i = 0; i < n; ++i)
        L.coeffRef(i, i) += excess;
    return L;
}

TEST_F(SolveTest, SDDMGridConverges) {
    auto M = sddm_grid(10, 10, 2.0);
    Eigen::VectorXd b = Eigen::VectorXd::Random(100);
    auto res = this->solve_with(M, b);
    EXPECT_LT(res.residual, 1e-5);
}

TEST_F(SolveTest, SDDMLargeExcessConverges) {
    auto M = sddm_grid(10, 10, 100.0);
    Eigen::VectorXd b = Eigen::VectorXd::Random(100);
    auto res = this->solve_with(M, b);
    EXPECT_LT(res.residual, 1e-5);
}

TEST_F(FactorizeTest, SDDMFlagSet) {
    auto M = sddm_grid(5, 5, 1.0);
    auto F = this->factorize_with(M);
    EXPECT_TRUE(F.sddm);
}

TEST_F(FactorizeTest, LaplacianFlagNotSet) {
    auto L = grid_laplacian(5, 5);
    auto F = this->factorize_with(L);
    EXPECT_FALSE(F.sddm);
}

// ── PCG-loop determinism (fused, OpenMP-parallel vector kernels) ──
// The PCG outer loop and the preconditioner application stream their vectors
// in fused passes with hand-rolled per-thread partial reductions (see the
// detail:: scaffolding in preconditioner.h). Same factor + same b must give
// bit-identical iterations / residual / x run to run at the thread count the
// test process has. The fixtures are large enough to use several solve
// workers when available (a reduction() clause or any completion-order sum
// would fail this at T > 1).
static void expect_repeated_solves_bit_identical(const Eigen::SparseMatrix<double>& A,
                                                 const Eigen::VectorXd& b, bool laplacian) {
    apxchol::solve_options opts;
    opts.tol = 1e-8; opts.max_iter = 500; opts.factor_opts.seed = 42;
    apxchol::cpu_solver slv(A, opts);
    auto r1 = slv.solve(b);
    auto r2 = slv.solve(b);
    ASSERT_LT(r1.residual, 1e-8);
    EXPECT_EQ(r1.iterations, r2.iterations);
    EXPECT_EQ(r1.residual, r2.residual);                    // exact, not ULP-tolerant
    EXPECT_TRUE((r1.x.array() == r2.x.array()).all());
    // Warm start (x0 path: r = b - A x0 in one fused pass) is deterministic too.
    auto w1 = slv.solve(b, 1e-10, 500, &r1.x);
    auto w2 = slv.solve(b, 1e-10, 500, &r1.x);
    EXPECT_EQ(w1.iterations, w2.iterations);
    EXPECT_EQ(w1.residual, w2.residual);
    EXPECT_TRUE((w1.x.array() == w2.x.array()).all());
    // The Eigen-facing application (apply -> _solve_impl): same guarantee. On
    // a Laplacian it follows the center-k schedule (APXCHOL_GROUND=center-k,
    // see env_knobs.h): only every K-th application since the last
    // reset_apply_count() runs the two mean passes, so compare applications
    // at the same phase (restart the schedule before each call).
    const auto& pc = slv.preconditioner();
    pc.reset_apply_count();
    Eigen::VectorXd z1 = slv.apply(b);
    pc.reset_apply_count();
    Eigen::VectorXd z2 = slv.apply(b);
    EXPECT_TRUE((z1.array() == z2.array()).all());
    if (laplacian) {
        // The K-th application after a restart centres: mean zero to rounding.
        const int K = apxchol::detail::env_knobs::get().center_k;
        pc.reset_apply_count();
        Eigen::VectorXd zk;
        for (int i = 0; i < K; ++i) zk = slv.apply(b);
        EXPECT_NEAR(zk.mean(), 0.0, 1e-13 * zk.cwiseAbs().maxCoeff());
        // The returned SOLUTION is min-norm regardless of the schedule
        // (cpu_solver::solve centres x once at the end).
        EXPECT_NEAR(r1.x.mean(), 0.0, 1e-14 * r1.x.cwiseAbs().maxCoeff());
        EXPECT_NEAR(w1.x.mean(), 0.0, 1e-14 * w1.x.cwiseAbs().maxCoeff());
    }
}

TEST(PcgFusion, RepeatedSolvesBitIdenticalLaplacian) {
    auto L = grid_laplacian(320, 320);
    auto b = apxchol::generate_test_rhs(L.rows());
    expect_repeated_solves_bit_identical(L, b, /*laplacian=*/true);
}

TEST(PcgFusion, RepeatedSolvesBitIdenticalSDDM) {
    auto M = sddm_grid(320, 320, 0.5);
    Eigen::VectorXd b = Eigen::VectorXd::Random(M.rows());
    expect_repeated_solves_bit_identical(M, b, /*laplacian=*/false);
}

// ── Grounding: min-norm Laplacian solution ──
// cpu_solver::solve returns the MIN-NORM solution of a Laplacian system.
// Under the center-k schedule the preconditioner skips its output
// re-centring on most applications, so the PCG iterate drifts along the null
// space 1 (invisible to the residual); solve_impl subtracts mean(x) once at
// the end -- also for a warm start, whose constant component must not leak
// into the answer.
TEST(Grounding, LaplacianSolutionIsMinNorm) {
    auto L = grid_laplacian(60, 60);
    auto b = apxchol::generate_test_rhs(L.rows());
    apxchol::solve_options opts;
    opts.tol = 1e-8; opts.max_iter = 500; opts.factor_opts.seed = 42;
    apxchol::cpu_solver slv(L, opts);
    auto r = slv.solve(b);
    ASSERT_LT(r.residual, 1e-8);
    EXPECT_NEAR(r.x.mean(), 0.0, 1e-14 * r.x.cwiseAbs().maxCoeff());
    EXPECT_LT((L * r.x - b).norm() / b.norm(), 1e-6);
    // Warm start shifted by a constant, tightened tolerance -> PCG iterates
    // from x0; the answer is still mean-zero and still solves the system.
    Eigen::VectorXd x0 = r.x.array() + 5.0;
    // Subtracting the offset incurs roundoff at the warm start's scale.
    const double warm_centering_tol = 1e-14 * x0.cwiseAbs().maxCoeff();
    auto w = slv.solve(b, 1e-10, 500, &x0);
    EXPECT_GT(w.iterations, 0);
    EXPECT_NEAR(w.x.mean(), 0.0, warm_centering_tol);
    EXPECT_LT((L * w.x - b).norm() / b.norm(), 1e-8);
    // Already-converged warm start (loose tolerance -> 0 iterations): the
    // early exit centres too.
    auto e = slv.solve(b, 1e-6, 500, &x0);
    EXPECT_EQ(e.iterations, 0);
    EXPECT_NEAR(e.x.mean(), 0.0, warm_centering_tol);
    EXPECT_LT((L * e.x - b).norm() / b.norm(), 1e-6);
}

// Build a weighted grid Laplacian DIRECTLY from fp64 triplets. Deliberately
// bypasses weighted_grid_laplacian/graph: under APXCHOL_POOL_FP32 the graph
// edge pool stores weights in fp32, so routing the weights through graph
// would silently quantize them and mask the regression tested below.
static Eigen::SparseMatrix<double> fp64_weighted_grid_laplacian(
        int rows, int cols, double w_horiz, double w_vert) {
    const int n = rows * cols;
    auto id = [cols](int r, int c) { return r * cols + c; };
    std::vector<Eigen::Triplet<double>> trips;
    std::vector<double> deg(n, 0.0);
    auto edge = [&](int u, int v, double w) {
        trips.emplace_back(u, v, -w);
        trips.emplace_back(v, u, -w);
        deg[u] += w;
        deg[v] += w;
    };
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            if (r + 1 < rows) edge(id(r, c), id(r + 1, c), w_vert);
            if (c + 1 < cols) edge(id(r, c), id(r, c + 1), w_horiz);
        }
    for (int i = 0; i < n; ++i) trips.emplace_back(i, i, deg[i]);
    Eigen::SparseMatrix<double> L(n, n);
    L.setFromTriplets(trips.begin(), trips.end());
    return L;
}

// Regression (fp32-quantized wdeg vs exact diag): make_graph used to compute
// the excess test's weighted degree by re-walking the stored edge weights,
// which are fp32-quantized under APXCHOL_POOL_FP32. Exact fp64 diag minus a
// quantized wdeg left phantom excess ≈1e-8·diag — above the 1e-12 gate — so a
// pure Laplacian whose weights are not fp32-representable (0.01, 2/101) was
// misclassified as SDDM. The excess must come from the exact fp64 input.
TEST_F(FactorizeTest, NonFp32ExactWeightsLaplacianNotSDDM) {
    auto L = fp64_weighted_grid_laplacian(10, 10, 0.01, 2.0 / 101.0);
    auto F = this->factorize_with(L);
    EXPECT_FALSE(F.sddm);
}

// Positive control for the regression above: with the same non-fp32-exact
// weights, a genuinely SDDM matrix must still be detected.
TEST_F(FactorizeTest, NonFp32ExactWeightsSDDMFlagStillSet) {
    auto L = fp64_weighted_grid_laplacian(10, 10, 0.01, 2.0 / 101.0);
    L.coeffRef(0, 0) += 0.5;
    auto F = this->factorize_with(L);
    EXPECT_TRUE(F.sddm);
}

TEST_F(FactorizeTest, SDDMFullRankDiagonal) {
    auto M = sddm_grid(5, 5, 1.0);
    auto F = this->factorize_with(M);
    // All n diagonal entries should be positive (including the last).
    const int n = F.L.rows();
    for (int i = 0; i < n; ++i)
        EXPECT_GT(factor_diag(F.L, i), 0.0) << "zero diagonal at " << i;
}

// ── Comprehensive IS × elimination convergence tests ──

struct StrategyCombo {
    std::string is;
    const char* name;
};

// Stable value printer. Without it gtest appends a raw byte dump (containing
// ASLR-varying pointers) to --gtest_list_tests, which gtest_discover_tests
// folds into the registered ctest name and filter — the filter then matches
// nothing and the test "passes" in 0.00s without running.
static void PrintTo(const StrategyCombo& c, std::ostream* os) { *os << c.name; }

class StrategyConvergenceTest
    : public ::testing::TestWithParam<StrategyCombo> {};

static const StrategyCombo all_combos[] = {
    {"block_greedy", "bg_tree"},
    {"priority_greedy", "priority_greedy_tree"},
    {"baumann_kyng", "bk_tree"},
};

TEST_P(StrategyConvergenceTest, GridConverges) {
    auto [is, name] = GetParam();
    auto L = grid_laplacian(15, 15);
    auto b = apxchol::generate_test_rhs(225);
    auto res = apxchol::solve(L, b,
        {.tol = 1e-6, .max_iter = 1000,
         .factor_opts = {.seed = 42, .is_select = is}});
    EXPECT_LT(res.residual, 1e-4)
        << "strategy " << name << " failed: iters=" << res.iterations
        << " residual=" << res.residual;
}

TEST_P(StrategyConvergenceTest, SDDMConverges) {
    auto [is, name] = GetParam();
    auto M = sddm_grid(10, 10, 2.0);
    Eigen::VectorXd b = Eigen::VectorXd::Random(100);
    auto res = apxchol::solve(M, b,
        {.tol = 1e-6, .max_iter = 1000,
         .factor_opts = {.seed = 42, .is_select = is}});
    EXPECT_LT(res.residual, 1e-4)
        << "strategy " << name << " failed on SDDM: iters=" << res.iterations
        << " residual=" << res.residual;
}

INSTANTIATE_TEST_SUITE_P(AllStrategies, StrategyConvergenceTest,
    ::testing::ValuesIn(all_combos),
    [](const auto& info) { return info.param.name; });

// ─── F1: factorization quality regression tests ──────────────────────
//
// Historical bounds from forward-star: max residual observed across all
// partitioners × thread counts {1, 8, 16}, multiplied by 1.5 for headroom.
// Keep these thresholds unchanged when migrating this check to directed AoS.
//
// Update bounds only when a deliberate algorithmic change shifts the
// achievable residual.  A regression in unbiasedness shows up as
// residuals exceeding the bound on the tree strategy.

namespace f1_bounds {
    // ── grid20 (pure Laplacian, n = 400) ──
    inline constexpr double grid20_tree   = 0.311241;
    // ── sddm_grid20 (SDDM, n = 400, excess = 2 on diagonal) ──
    inline constexpr double sddm_grid20_tree   = 0.13196;
}

// Reconstruct A from F and compute ‖A − Pᵀ L Lᵀ P‖_F / ‖A‖_F.
static double factor_residual(const apxchol::factorization& F,
                              const Eigen::SparseMatrix<double>& A) {
    Eigen::SparseMatrix<double> Lmat = factor_to_eigen(F.L);
    auto P = perm_to_eigen(F.perm);
    Eigen::SparseMatrix<double> LLt = Lmat * Lmat.transpose();
    Eigen::SparseMatrix<double> reconstructed = P.transpose() * LLt * P;
    Eigen::SparseMatrix<double> diff = A - reconstructed;
    double a_norm = A.norm();
    return a_norm == 0.0 ? 0.0 : diff.norm() / a_norm;
}

static apxchol::factorization factorize_tree(
        const Eigen::SparseMatrix<double>& A, unsigned seed = 42) {
    apxchol::factor_options opts{.seed = seed};
    return apxchol::factorize(A, opts);
}

TEST(FactorQualityTest, Grid20Tree) {
    auto L = grid_laplacian(20, 20);
    auto F = factorize_tree(L);
    EXPECT_LT(factor_residual(F, L), f1_bounds::grid20_tree);
}

TEST(FactorQualityTest, SddmGrid20Tree) {
    auto M = sddm_grid(20, 20, 2.0);
    auto F = factorize_tree(M);
    EXPECT_LT(factor_residual(F, M), f1_bounds::sddm_grid20_tree);
}

// ─── Exact-clique-at-low-degree option ──────────────────────────────────────
TEST(ExactCliqueOption, Grid40_FewerItersThanSampled) {
    auto L = grid_laplacian(40, 40);
    auto b = apxchol::generate_test_rhs(L.rows());
    apxchol::factor_options opts;
    opts.seed = 42;
    opts.is_select = "block_greedy";
    auto sampled = apxchol::solve(L, b,
        {.tol = 1e-8, .max_iter = 200,
         .factor_opts = opts});
    opts.exact_clique_max_degree = 8;
    auto exact = apxchol::solve(L, b,
        {.tol = 1e-8, .max_iter = 200,
         .factor_opts = opts});
    EXPECT_LT(exact.residual, 1e-6);
    EXPECT_LE(exact.iterations, sampled.iterations)
        << "exact low-degree cliques should not need more PCG iterations: "
        << "exact=" << exact.iterations << ", sampled=" << sampled.iterations;
}

// ─── BK residual loop (parallel_residual_threshold) ─────────────────────────
//
// A complete graph is the cheapest way to reach this code from a unit test: its
// independent set is one vertex, so block_greedy trips min_is_fraction on the
// first round and hands the whole graph to the residual path. That is the same
// loop the social graphs enter at 70k-830k active.
namespace {
Eigen::SparseMatrix<double> clique_laplacian(int n) {
    Eigen::SparseMatrix<double> L(n, n);
    std::vector<Eigen::Triplet<double>> t;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            if (i != j) t.emplace_back(i, j, -1.0);
        t.emplace_back(i, i, static_cast<double>(n - 1));
    }
    L.setFromTriplets(t.begin(), t.end());
    return L;
}
}  // namespace

TEST(IsYieldHandoff, UsesCandidatesAndProtectsTheSmallTail) {
    using apxchol::detail::adaptive_is_yield_fraction;
    using apxchol::detail::is_yield_too_small;
    using apxchol::detail::selection_should_handoff;
    constexpr size_t none = std::numeric_limits<size_t>::max();

    // 9/200 is a 4.5% yield and therefore too small even though it is only
    // 0.45% of the 2000 active vertices. The candidate pool is the denominator.
    EXPECT_TRUE(is_yield_too_small(9, 200, 2000, 0.05, 500));
    EXPECT_FALSE(is_yield_too_small(10, 200, 2000, 0.05, 500));

    // Protect twice the handoff threshold. Bailing at active=524, for example,
    // lets BK remove only 24 vertices and turns the remaining 500 columns into
    // singleton peel levels. One selected vertex remains useful progress there.
    EXPECT_FALSE(is_yield_too_small(1, 5, 23, 0.05, 500));
    EXPECT_FALSE(is_yield_too_small(1, 100, 1000, 0.05, 500));
    EXPECT_TRUE(is_yield_too_small(1, 100, 1001, 0.05, 500));
    EXPECT_TRUE(is_yield_too_small(0, 5, 23, 0.05, 500));

    // A custom partitioner with no residual handoff trait keeps the ordinary
    // relative-yield rule at every active size.
    EXPECT_TRUE(is_yield_too_small(1, 100, 23, 0.05, none));

    // Adaptation needs enough runway before the 500-vertex handoff. Large
    // sparse residuals keep the public base; large dense ones triple it. Zero
    // remains an exact bailout-off switch and a larger base is not cut.
    EXPECT_DOUBLE_EQ(
        adaptive_is_yield_fraction(0.05, 2000, 10000.0, 500), 0.05);
    EXPECT_DOUBLE_EQ(
        adaptive_is_yield_fraction(0.05, 2001, 499.9, 500), 0.05);
    EXPECT_DOUBLE_EQ(
        adaptive_is_yield_fraction(0.05, 2001, 500.0, 500), 0.15);
    EXPECT_DOUBLE_EQ(
        adaptive_is_yield_fraction(0.0, 10000, 10000.0, 500), 0.0);
    EXPECT_DOUBLE_EQ(
        adaptive_is_yield_fraction(0.20, 10000, 10000.0, 500), 0.20);
    EXPECT_DOUBLE_EQ(adaptive_is_yield_fraction(
        0.05, 10000, 10000.0, std::numeric_limits<size_t>::max()), 0.05);

    // Relative yield does not discard a selection already large enough for
    // parallel elimination. Zero progress still hands off unconditionally.
    EXPECT_FALSE(selection_should_handoff(
        3700, 37375, 173224, 0.10, 500, 2000));
    EXPECT_TRUE(selection_should_handoff(
        1900, 20000, 100000, 0.10, 500, 2000));
    EXPECT_TRUE(selection_should_handoff(
        0, 20000, 100000, 0.10, 500, 2000));
}

TEST(ResidualSparsifyGate, UsesObservedTrafficAndHandoffYield) {
    using apxchol::detail::residual_sparsify_worthwhile;
    // The six naturally eligible residuals from the breadth screen all repay
    // the conservative eight-pass rebuild model.
    EXPECT_TRUE(residual_sparsify_worthwhile(3040, 500, 89, 832.773438));
    EXPECT_TRUE(residual_sparsify_worthwhile(4352, 500, 131, 380.308594));
    EXPECT_TRUE(residual_sparsify_worthwhile(21518, 500, 209, 224.363281));
    EXPECT_TRUE(residual_sparsify_worthwhile(74819, 500, 1377, 430.695312));
    EXPECT_TRUE(residual_sparsify_worthwhile(9085, 500, 264, 286.738281));
    EXPECT_TRUE(residual_sparsify_worthwhile(194925, 500, 1978, 909.019531));

    // Sparse residuals and nearly-finished handoffs do not amortize a rebuild.
    EXPECT_FALSE(residual_sparsify_worthwhile(10000, 500, 200, 2.1));
    EXPECT_FALSE(residual_sparsify_worthwhile(700, 500, 150, 400.0));
    EXPECT_FALSE(residual_sparsify_worthwhile(10000, 500, 0, 400.0));
}

TEST(ResidualSparsifyGate, DuplicateHeavyIndexedGraphConvergesWithEitherSetting) {
    const scoped_threads team(4);
    constexpr apxchol::node_index n = 96;
    apxchol::graph graph(n);
    for (apxchol::node_index u = 0; u < n; ++u)
        for (apxchol::node_index v = u + 1; v < n; ++v)
            for (int duplicate = 0; duplicate < 4; ++duplicate)
                graph.add_edge(u, v, 0.25 * (1 + (u + v + duplicate) % 7));
    const auto L = apxchol::laplacian(graph);
    const auto b = apxchol::generate_test_rhs(n);
    apxchol::factor_options options;
    options.parallel_residual_threshold = 5;
    options.min_is_fraction = 0.5;
    for (const char* setting : {"0", "1"}) {
        SCOPED_TRACE(setting);
        const scoped_env sparsify("APXCHOL_RESIDUAL_SPARSIFY", setting);
        apxchol::checkpoint timings;
        auto factor = apxchol::factorize(graph, options, &timings);
        EXPECT_EQ(timings.total("setup.sparsify_residual") > 0.0, *setting == '1')
            << "fixture must exercise the residual sparsification gate";
        apxchol::cpu_solver solver(L, std::move(factor));
        const auto result = solver.solve(b, 1e-8, 500);
        EXPECT_LT((L * result.x - b).norm() / b.norm(), 1e-8);
    }
}

TEST(EliminationTeamSizing, UsesDegreeWorkAndProtectsSerialExecution) {
    using apxchol::detail::elimination_round_team_size;
    constexpr size_t max = std::numeric_limits<size_t>::max();

    EXPECT_EQ(elimination_round_team_size(8, 17000, 2000, 16), 4u);
    EXPECT_EQ(elimination_round_team_size(8, 8191, 2000, 16), 1u);
    EXPECT_EQ(elimination_round_team_size(8, 8192, 2000, 16), 2u);
    EXPECT_EQ(elimination_round_team_size(1, max, 2000, 16), 1u);
    EXPECT_EQ(elimination_round_team_size(500, max, 2000, 1), 1u);
    EXPECT_EQ(elimination_round_team_size(3000, 1, 2000, 16), 16u);
    EXPECT_EQ(elimination_round_team_size(70, max, 2000, 72), 70u);
}

TEST(IncrementalDegreeGate, RequiresTrafficSavingsAndRadixWorkPerWorker) {
    using apxchol::detail::incremental_degree_worthwhile;
    using apxchol::detail::incremental_degree_parallel_work_worthwhile;
    using apxchol::detail::incremental_degree_traffic_can_improve;
    using apxchol::detail::incremental_degree_traffic_worthwhile;
    constexpr size_t min_work =
        apxchol::detail::kIncrementalDegreeMinWorkPerWorker;

    // A large traffic ratio is not enough when the collective radix has less
    // useful input per worker than one 256-bin histogram pass.
    EXPECT_FALSE(incremental_degree_worthwhile(
        65536, 74.953339, 6321, 36));
    EXPECT_FALSE(incremental_degree_worthwhile(
        65536, 74.953339, 6321, 72));
    EXPECT_TRUE(incremental_degree_traffic_worthwhile(
        65536, 74.953339, 6321));
    EXPECT_FALSE(incremental_degree_parallel_work_worthwhile(6321, 36));
    EXPECT_FALSE(incremental_degree_parallel_work_worthwhile(6321, 72));
    EXPECT_FALSE(incremental_degree_worthwhile(
        100000, 100.0, min_work * 8 - 1, 8));

    EXPECT_TRUE(incremental_degree_worthwhile(
        100000, 100.0, min_work * 8, 8));
    EXPECT_TRUE(incremental_degree_worthwhile(
        540486, 56.414890, 214699, 72));

    // Sufficient parallel work still does not override the traffic gate.
    EXPECT_FALSE(incremental_degree_worthwhile(
        299067, 6.538174, 128864, 72));
    EXPECT_TRUE(incremental_degree_traffic_can_improve(
        299067, 6.538174, 128864));
    EXPECT_FALSE(incremental_degree_traffic_worthwhile(
        299067, 6.538174, 128864));
    EXPECT_FALSE(incremental_degree_traffic_can_improve(
        1134890, 5.265046, 602539));
    EXPECT_FALSE(incremental_degree_traffic_can_improve(
        524288, 14.005871, 789977));
    EXPECT_FALSE(incremental_degree_worthwhile(
        100000, 100.0, min_work, 0));
}

TEST(BkResidualLoop, DrivesTheResidualToTheThresholdAndStaysDeterministic) {
    constexpr int n = 60;
    constexpr size_t thresh = 5;
    const auto L = clique_laplacian(n);
    const auto b = apxchol::generate_test_rhs(L.rows());

    // Several seeds: the loop must reach the threshold under every one of them.
    // On a clique BK's sample is empty with probability (1-p)^n ~ 3% per round
    // over ~55 rounds, so at least one seed here whiffs — and a whiff must be
    // retried, not treated as "the residual stopped shrinking". Bailing on the
    // first empty round (the behaviour until 2026-08-20) dumps the rest on the
    // serial peel and fails the peel-column bound below.
    for (unsigned seed : {7u, 11u, 42u, 1234u, 20260820u}) {
        apxchol::factor_options with_loop;          // BK rounds down to `thresh`,
        with_loop.seed = seed;                      // then the peel takes the
        with_loop.parallel_residual_threshold = thresh;   // rest.
        const auto c = apxchol::factorize(
            L, with_loop);
        ASSERT_FALSE(c.rounds.empty());
        EXPECT_EQ(c.rounds.front().active, static_cast<size_t>(n))
            << "the clique should hand the full residual to BK";

        size_t in_rounds = 0;
        for (const auto& r : c.rounds) in_rounds += r.is_size;
        EXPECT_GE(in_rounds, static_cast<size_t>(n) - thresh)
            << "seed " << seed << ": the BK residual loop handed "
            << (static_cast<size_t>(n) - in_rounds) << " columns to the serial "
            << "peel, but parallel_residual_threshold = " << thresh
            << " allows at most " << thresh;

        // Same (input, seed, thread count) => same factor, bit for bit.
        const auto c2 = apxchol::factorize(
            L, with_loop);
        expect_same_factor(c, c2, "BK residual loop, seed " +
                                  std::to_string(seed));
        if (::testing::Test::HasFatalFailure()) return;

        // And it is still a usable preconditioner.
        const auto res = apxchol::solve(L, b,
            {.tol = 1e-8, .max_iter = 200,
             .factor_opts = with_loop});
        EXPECT_LT(res.residual, 1e-8) << "seed " << seed;
    }
}

// The residual loop constructs a fresh baumann_kyng_partitioner partway through
// an elimination, where graph::m() is monotone and 2*m/|active| is inflated by
// every edge the eliminated prefix consumed. BK must honour a caller-supplied
// est_avg_degree on round 0 rather than recomputing one; this pins that seam,
// whose only visible effect is which vertices round 0 samples.
TEST(BaumannKyngSeeding, Round0UsesTheSeedInsteadOfTwoMOverActive) {
    const auto L = grid_laplacian(30, 30);          // 900 vertices, avg degree < 4
    apxchol::partition_context ctx{
        .options = {},
        .seed = 42,
        .omp_threshold = 2000,
        .cp = nullptr,
        .degrees = {},
    };
    std::vector<apxchol::node_index> active(900);
    std::iota(active.begin(), active.end(), apxchol::node_index{0});

    auto is_size = [&](double seed) {
        auto G = apxchol::make_graph(L);
        apxchol::baumann_kyng_partitioner bk;
        bk.est_avg_degree = seed;                   // 0 = "work it out yourself"
        apxchol::selection sel;
        sel.reset(G.n());
        bk.find_partition(G, std::span<const apxchol::node_index>(active),
                          ctx, sel);
        return sel.finalize().num_regions();
    };

    // Seeding a huge average degree drives p = 1/(c·d) down, so round 0 samples
    // almost nothing; the unseeded call measures the real ~4 and samples freely.
    const auto unseeded = is_size(0.0);
    const auto inflated = is_size(4000.0);
    EXPECT_GT(unseeded, 0u);
    EXPECT_LT(inflated, unseeded)
        << "round 0 ignored the caller's est_avg_degree seed (unseeded="
        << unseeded << ", seeded 4000 => " << inflated << ")";
}

TEST(BaumannKyngWorkHint, MatchesSelectedLiveDegreeSum) {
    const auto L = grid_laplacian(30, 30);
    auto G = apxchol::make_graph(L);
    std::vector<apxchol::node_index> active(900);
    std::iota(active.begin(), active.end(), apxchol::node_index{0});
    apxchol::partition_context ctx{
        .options = {},
        .seed = 42,
        .omp_threshold = 1,
        .cp = nullptr,
        .degrees = {},
    };
    apxchol::baumann_kyng_partitioner bk;
    bk.est_avg_degree = 4.0;
    apxchol::selection selected;
    selected.reset(G.n());
    bk.find_partition(G, active, ctx, selected);
    const auto& part = selected.finalize();

    size_t expected = 0;
    for (const auto v : part.data)
        expected += static_cast<size_t>(G.prune_and_degree(v));
    EXPECT_EQ(bk.selected_degree_work(), expected);

    selected.reset(G.n());
    bk.find_partition(G, std::span<const apxchol::node_index>{}, ctx, selected);
    EXPECT_EQ(bk.selected_degree_work(), 0u);
}

TEST(EliminationTeamSizing, DenseSmallRoundsUseFineSchedulingChunks) {
    using apxchol::detail::elimination_compute_chunk;
    EXPECT_EQ(elimination_compute_chunk(2000, 2000), 1);
    EXPECT_EQ(elimination_compute_chunk(2001, 2000), 64);
}

TEST(PriorityGreedy, SerialFallbackPreservesExactSelectedSetAndMaximality) {
    const scoped_threads team(4);
    constexpr apxchol::node_index n = 80;
    apxchol::graph G(n);
    for (apxchol::node_index v = 0; v < n; ++v) {
        G.add_edge(v, (v + 1) % n, 1.0);
        G.add_edge(v, (v + 7) % n, 1.0);
        if ((v % 3) == 0) G.add_edge(v, (v + 19) % n, 1.0);
    }
    std::vector<apxchol::node_index> candidates(n), degrees;
    std::iota(candidates.begin(), candidates.end(), apxchol::node_index{0});
    (void)apxchol::prune_and_degrees(G, candidates, degrees, 2000);
    apxchol::partition_context ctx{
        .options = {}, .seed = 17, .omp_threshold = 1, .cp = nullptr,
        .degrees = degrees,
    };
    auto select = [&](int passes) {
        apxchol::priority_greedy_partitioner p;
        p.parallel_passes = passes;
        apxchol::selection selected;
        selected.reset(n);
        p.find_partition(G, candidates, ctx, selected);
        return selected.finalize().data;
    };

    const auto parallel = select(16);
    const auto fallback = select(0);
    EXPECT_EQ(fallback, parallel);

    std::vector<unsigned char> in_set(n, 0);
    for (const auto v : fallback) in_set[v] = 1;
    for (const auto v : candidates) {
        bool has_selected_neighbor = false;
        for (const auto edge : G.adj(v)) {
            const auto u = edge.to;
            EXPECT_FALSE(in_set[v] && in_set[u]);
            has_selected_neighbor |= in_set[u] != 0;
        }
        EXPECT_TRUE(in_set[v] || has_selected_neighbor)
            << "uncovered vertex " << v;
    }
}

TEST(PriorityGreedy, ParallelPicksExcludeSharedNeighborsAndResetScratch) {
    const scoped_threads team(4);
    int actual_threads = 1;
#pragma omp parallel
    {
#pragma omp single
        actual_threads = omp_get_num_threads();
    }
    ASSERT_GT(actual_threads, 1) << "shared-neighbor coverage requires a parallel team";
    constexpr apxchol::node_index leaves = 128;
    constexpr apxchol::node_index n = leaves + 2;
    apxchol::graph G(n);
    for (apxchol::node_index v = 0; v < leaves; ++v) {
        G.add_edge(v, leaves, 1.0);
        G.add_edge(v, leaves + 1, 1.0);
    }
    G.add_edge(0, leaves, 1.0); // Duplicate exclusion of a shared neighbor.
    G.add_edge(0, 0, 1.0);      // A self incidence must not exclude its own pick.
    std::vector<apxchol::node_index> candidates(n), degrees;
    std::iota(candidates.begin(), candidates.end(), apxchol::node_index{0});
    (void)apxchol::prune_and_degrees(G, candidates, degrees, 2000);
    apxchol::partition_context ctx{
        .options = {}, .seed = 17, .omp_threshold = 1, .cp = nullptr,
        .degrees = degrees,
    };
    ctx.options.degree_tiebreak = true;
    std::vector<apxchol::node_index> expected(candidates.begin(),
                                             candidates.begin() + leaves);
    apxchol::priority_greedy_partitioner p;
    p.parallel_passes = 1;
    apxchol::selection selected;
    for (int repetition = 0; repetition < 8; ++repetition) {
        // Every leaf wins in the same pass; all workers exclude the same hubs.
        selected.reset(n);
        p.find_partition(G, candidates, ctx, selected);
        EXPECT_EQ(selected.finalize().data, expected);

        // Reuse the same scratch with only the previously excluded hubs eligible.
        selected.reset(n);
        const auto hubs = std::span<const apxchol::node_index>(candidates).subspan(leaves);
        p.find_partition(G, hubs, ctx, selected);
        EXPECT_EQ(selected.finalize().data,
                  (std::vector<apxchol::node_index>{leaves, leaves + 1}));
    }
}

#if defined(APXCHOL_USE_CUDA)
TEST(GpuSptrsvConfiguration, UsesDataflowWithEitherStorage) {
    for (const char* choice : {static_cast<const char*>(nullptr), "", "dataflow"}) {
        const scoped_env selected("APXCHOL_GPU_SPTRSV", choice);
        for (const char* storage : {static_cast<const char*>(nullptr), "float32", "float16"}) {
            const scoped_env fp16("APXCHOL_FACTOR_STORAGE", storage);
            EXPECT_EQ(apxchol::cuda_sptrsv::fp16_resolved(),
                      !storage || std::string_view(storage) == "float16");
            apxchol::sparse_csc factor;
            factor.n_ = 1;
            factor.outer_ = {0, 1};
            factor.inner_ = {0};
            factor.vals_ = {2.0f};
            apxchol::cuda_sptrsv trsv;
            ASSERT_NO_THROW(trsv.setup(factor, 1));
            EXPECT_STREQ(trsv.backend_name(), "dataflow");
            EXPECT_EQ(trsv.fp16(), !storage || std::string_view(storage) == "float16");
        }
    }
}

TEST(GpuSptrsvConfiguration, RejectsUnknownAndRetiredBackendsBeforeSetup) {
    for (const char* choice : {"cusparse", "auto", "levelset", "typo"}) {
        const scoped_env selected("APXCHOL_GPU_SPTRSV", choice);
        apxchol::cuda_sptrsv trsv;
        EXPECT_THROW(trsv.setup(apxchol::sparse_csc{}, 0),
                     std::invalid_argument);
    }
}

#if defined(APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
namespace {
using bounded_vertex = apxchol::node_index;
struct bounded_edge { bounded_vertex u, v; };


// Independent scalar specification: no production priority or selector helper.
static std::uint64_t bounded_fixture_key(std::uint64_t seed,
                                         std::uint64_t phase,
                                         bounded_vertex vertex) {
    std::uint64_t z = (seed ^ (phase * 0xD6E8FEB86659FD93ULL) ^ vertex)
                      + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

struct bounded_oracle_result {
    std::vector<bounded_vertex> candidates, selected, degrees;
    std::size_t work = 0, unfinished = 0;
    unsigned passes = 0;
};

static bounded_oracle_result bounded_four_pass_oracle(
        bounded_vertex n, const std::vector<bounded_edge>& edges,
        const apxchol::partition_options& options,
        std::uint64_t seed, std::uint64_t phase) {
    std::vector<std::vector<bounded_vertex>> adjacent(n);
    for (const auto e : edges) {
        adjacent[e.u].push_back(e.v); adjacent[e.v].push_back(e.u);
    }
    bounded_oracle_result result;
    std::size_t degree_sum = 0;
    for (const auto& row : adjacent) {
        result.degrees.push_back(static_cast<bounded_vertex>(row.size()));
        degree_sum += row.size();
    }
    double threshold = options.degree_multiplier * double(degree_sum) / n;
    if (options.degree_quantile > 0.0 && options.degree_quantile < 1.0) {
        auto sorted = result.degrees;
        std::sort(sorted.begin(), sorted.end());
        const auto rank = static_cast<std::size_t>(
            options.degree_quantile * static_cast<double>(n - 1));
        threshold = sorted[rank];
    }
    // 0 undecided, 1 chosen, 2 excluded/blocked. Each decision reads a snapshot.
    std::vector<unsigned char> state(n, 2);
    for (bounded_vertex v = 0; v < n; ++v)
        if (result.degrees[v] <= threshold) {
            state[v] = 0; result.candidates.push_back(v);
        }
    auto precedes = [&](bounded_vertex a, bounded_vertex b) {
        return std::tuple(options.degree_tiebreak ? result.degrees[a] : 0,
                          bounded_fixture_key(seed, phase, a), a)
             < std::tuple(options.degree_tiebreak ? result.degrees[b] : 0,
                          bounded_fixture_key(seed, phase, b), b);
    };
    for (unsigned pass = 0; pass < 4; ++pass) {
        std::vector<bounded_vertex> winners;
        bool pending = false;
        for (const auto v : result.candidates) if (state[v] == 0) {
            pending = true;
            if (std::none_of(adjacent[v].begin(), adjacent[v].end(),
                    [&](auto u) { return state[u] == 0 && precedes(u, v); }))
                winners.push_back(v);
        }
        if (!pending) break;
        ++result.passes;
        for (const auto v : winners) state[v] = 1;
        for (const auto v : winners)
            for (const auto u : adjacent[v]) if (state[u] == 0) state[u] = 2;
    }
    for (const auto v : result.candidates) {
        if (state[v] == 1) {
            result.selected.push_back(v); result.work += result.degrees[v];
        }
        result.unfinished += state[v] == 0;
    }
    return result;
}

static void check_owned_bounded_selection(
        bounded_vertex n, const std::vector<bounded_edge>& edges,
        const apxchol::partition_options& options, std::uint64_t seed,
        std::uint64_t phase, const std::string& receipt_name,
        bool require_unfinished = false) {
    apxchol::graph graph(n);
    for (const auto e : edges) graph.add_edge(e.u, e.v, 1.0);
    apxchol::detail::gpu_round_shadow_session owner(true);
    auto gpu_owner = apxchol::test::initialize_owned_selector(owner, graph);
    auto& gpu = *gpu_owner;
    std::vector<bounded_vertex> active(n);
    std::iota(active.begin(), active.end(), bounded_vertex{0});
    const auto expected = bounded_four_pass_oracle(n, edges, options, seed, phase);
    ASSERT_FALSE(expected.candidates.empty()); ASSERT_FALSE(expected.selected.empty());
    EXPECT_LE(expected.passes, 4u);
    if (require_unfinished) {
        ASSERT_EQ(expected.passes, 4u);
        ASSERT_GT(expected.unfinished, 0u) << "fixture must distinguish bounded from maximal completion";
    }
    const auto prepared = gpu.prepare(active, options, seed, phase);
    EXPECT_EQ(prepared.candidate_count, expected.candidates.size());
    const auto candidates = gpu.host_candidates();
    EXPECT_EQ(std::vector<bounded_vertex>(candidates.begin(), candidates.end()), expected.candidates);
    const auto degrees = gpu.host_active_degrees();
    EXPECT_EQ(std::vector<bounded_vertex>(degrees.begin(), degrees.end()), expected.degrees);
    const auto first = gpu.select_block_greedy().data;
    EXPECT_EQ(first, expected.selected); EXPECT_EQ(gpu.selected_degree_work(), expected.work);
    EXPECT_EQ(gpu.select_block_greedy().data, first);
    EXPECT_EQ(gpu.selected_degree_work(), expected.work);
    if (seed == 42 && phase == 0) {
        (void)gpu.prepare(active, options); // Default overload must mean42/0.
        EXPECT_EQ(gpu.select_block_greedy().data, first);
    }
    std::vector<unsigned char> picked(n, 0);
    for (const auto v : first) { ASSERT_LT(v, n); picked[v] = 1; }
    for (const auto e : edges) EXPECT_FALSE(picked[e.u] && picked[e.v]);
    std::string receipt = std::to_string(expected.passes) + ":" +
        std::to_string(expected.unfinished) + ":" + std::to_string(expected.work);
    for (const auto v : expected.candidates) receipt += ":c" + std::to_string(v);
    for (const auto v : first) receipt += ":s" + std::to_string(v);
    testing::Test::RecordProperty(receipt_name, receipt);
}
} // namespace

TEST(GpuBoundedSelection, OwnedSnapshotsMatchIndependentFourPassOracle) {
    const scoped_omp_threads serial(1);
    const std::array<std::pair<std::uint64_t, std::uint64_t>, 4> keys = {{
        {42, 0}, {0, 0}, {42, 7}, {UINT64_MAX, UINT64_MAX}}};
    for (const bounded_vertex n : {2u, 4u, 31u, 32u, 33u, 255u, 256u, 257u})
        for (unsigned profile = 0; profile < 2; ++profile) {
            std::vector<bounded_edge> edges;
            // Ring profile has exact degree ties; path profile has a cap boundary
            // (n4,q=.5 admits only endpoints), raw duplicates and mixed degrees.
            for (bounded_vertex v = 0; v + 1 < n; ++v) {
                edges.push_back({v, v + 1});
                if (profile) edges.push_back({v, v + 1});
            }
            if (!profile && n > 2) edges.push_back({0, n - 1});
            if (profile && n > 4)
                for (bounded_vertex v = 0; v + 5 < n; v += 7) edges.push_back({v, v + 5});
            apxchol::partition_options options;
            options.degree_tiebreak = true; options.degree_quantile = 0.5;
            for (std::size_t k = 0; k < keys.size(); ++k) {
                SCOPED_TRACE("n=" + std::to_string(n) + " profile=" +
                             std::to_string(profile) + " key=" + std::to_string(k));
                check_owned_bounded_selection(n, edges, options, keys[k].first,
                    keys[k].second, "bounded_selection_" + std::to_string(n) +
                    "_" + std::to_string(profile) + "_" + std::to_string(k));
                if (testing::Test::HasFatalFailure()) return;
            }
        }
}

TEST(GpuBoundedSelection, FourPassCapLeavesAnUncoveredEligibleVertex) {
    const scoped_omp_threads serial(1);
    constexpr bounded_vertex n = 65;
    std::vector<bounded_vertex> order(n);
    std::iota(order.begin(), order.end(), bounded_vertex{0});
    std::sort(order.begin(), order.end(), [](auto a, auto b) {
        return std::pair(bounded_fixture_key(42, 0, a), a) <
               std::pair(bounded_fixture_key(42, 0, b), b);
    });
    std::vector<bounded_edge> edges;
    for (bounded_vertex i = 0; i + 1 < n; ++i) edges.push_back({order[i], order[i + 1]});
    apxchol::partition_options options;
    options.degree_tiebreak = true; options.degree_quantile = 0.0;
    options.degree_multiplier = 100.0;
    check_owned_bounded_selection(n, edges, options, 42, 0, "bounded_selection_cap65", true);
}

#endif

TEST(GpuBoundedSelection, OwnedZeroDegreeCandidatesMakeProgress) {
    using bounded_vertex = apxchol::node_index;
    const scoped_omp_threads serial(1);
    for (const bounded_vertex n : {1u, 33u}) {
        SCOPED_TRACE(n);
        Eigen::SparseMatrix<double> A(n, n);
        std::vector<Eigen::Triplet<double>> diagonal;
        for (bounded_vertex v = 0; v < n; ++v) diagonal.emplace_back(v, v, 0.0);
        A.setFromTriplets(diagonal.begin(), diagonal.end());
        ASSERT_TRUE(apxchol::detail::gpu_owned_csc_supported(A));
        apxchol::detail::gpu_round_shadow_session owner(true);
        std::unique_ptr<apxchol::detail::gpu_block_frontend> gpu;
        owner.initialize_owned_csc(apxchol::detail::gpu_owned_csc_host_buffers(A), gpu);
        std::vector<bounded_vertex> active(n);
        std::iota(active.begin(), active.end(), bounded_vertex{0});
        apxchol::partition_options options;
        options.degree_quantile = 0.5;
        const auto prepared = gpu->prepare(active, options, 42, 0);
        EXPECT_EQ(prepared.candidate_count, n); EXPECT_EQ(prepared.average_degree, 0.0);
        EXPECT_EQ(gpu->select_block_greedy().data, active);
        EXPECT_EQ(gpu->selected_degree_work(), 0u);
        EXPECT_EQ(gpu->select_block_greedy().data, active);
        (void)owner.run_owned_prefix_round(gpu->device_selection(), 42);
        owner.advance_selector(*gpu);
        active.clear();
        EXPECT_EQ(gpu->prepare(active, options, 42, 1).candidate_count, 0u);
        EXPECT_TRUE(gpu->select_block_greedy().data.empty());
        EXPECT_EQ(gpu->selected_degree_work(), 0u);
        RecordProperty("bounded_zero_degree_" + std::to_string(n), "all-selected:then-empty");
    }
}

TEST(GpuBoundedSelection, ConsumingSolveChecksOriginalOperatorResidual) {
    const scoped_env setup_trace("APXCHOL_SPTRSV_SETUP_TRACE", "1");
    const scoped_env fp32("APXCHOL_FACTOR_STORAGE", "float32");
    const scoped_env drop("APXCHOL_FACTOR_DROP", "0");
    const scoped_omp_threads serial(1);
    const auto A = grid_laplacian(9, 7);
    Eigen::VectorXd exact(A.rows());
    for (Eigen::Index i = 0; i < exact.size(); ++i) exact[i] = std::sin(double(i) + 0.25);
    const Eigen::VectorXd b = A * exact;
    for (const unsigned seed : {0u, 42u}) {
        apxchol::solve_options options;
        options.backend = apxchol::solve_backend::gpu;
        options.tol = 1e-8; options.max_iter = 500; options.stagnation_window = 0;
        options.factor_opts.seed = seed;
        options.factor_opts.partition.degree_quantile = 0.5;
        Eigen::VectorXd first_solution;
        Eigen::Index first_iterations = 0;
        for (unsigned repeat = 0; repeat < 2; ++repeat) {
            const auto solved = apxchol::solve(A, b, options);
            EXPECT_GT(solved.timings.total("setup.gpu_owned_factorization"), 0.0);
            const double rr = (A * solved.x - b).norm() / b.norm();
            EXPECT_TRUE(std::isfinite(rr)); EXPECT_LE(rr, 1e-8);
            std::ostringstream receipt;
            receipt.precision(std::numeric_limits<double>::max_digits10); receipt << rr;
            RecordProperty("bounded_original_rr_" + std::to_string(seed) + "_" +
                           std::to_string(repeat), receipt.str());
            if (repeat == 0) {
                first_solution = solved.x; first_iterations = solved.iterations;
                std::string bytes = std::to_string(first_iterations);
                for (Eigen::Index i = 0; i < first_solution.size(); ++i)
                    bytes += ":" + std::to_string(std::bit_cast<std::uint64_t>(first_solution[i]));
                RecordProperty("bounded_solution_" + std::to_string(seed), bytes);
            } else {
                EXPECT_EQ(solved.iterations, first_iterations);
                ASSERT_EQ(solved.x.size(), first_solution.size());
                for (Eigen::Index i = 0; i < solved.x.size(); ++i)
                    EXPECT_EQ(std::bit_cast<std::uint64_t>(solved.x[i]),
                              std::bit_cast<std::uint64_t>(first_solution[i]));
            }
        }
        // The consuming solve exposes solution/iterations, not an exportable
        // factor or fill count. This checks within-arm solution determinism only.
    }
}

#endif
