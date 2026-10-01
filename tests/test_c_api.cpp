// The C ABI (include/apxchol/c_api.h): statuses, validation, struct
// versioning, thread scoping and agreement with the C++ cpu_solver.
#include <gtest/gtest.h>

#include "apxchol/c_api.h"
#include "apxchol/env_knobs.h"
#include "apxchol/solver/solve.h"

#include <Eigen/Sparse>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using Sparse = Eigen::SparseMatrix<double>;

struct csc64 {
    std::int64_t n = 0;
    std::vector<std::int64_t> colptr, rowval;
    std::vector<double> nzval;
};

csc64 to_csc64(const Sparse& A, std::int64_t base = 0) {
    csc64 c;
    c.n = A.rows();
    for (Eigen::Index j = 0; j <= A.cols(); ++j) c.colptr.push_back(A.outerIndexPtr()[j] + base);
    for (Eigen::Index p = 0; p < A.nonZeros(); ++p) {
        c.rowval.push_back(A.innerIndexPtr()[p] + base);
        c.nzval.push_back(A.valuePtr()[p]);
    }
    return c;
}

Sparse grid_laplacian(int rows, int cols, double shift = 0.0) {
    const int n = rows * cols;
    std::vector<Eigen::Triplet<double>> t;
    std::vector<double> deg(n, shift);
    auto id = [&](int r, int c) { return r * cols + c; };
    auto edge = [&](int a, int b, double w) {
        t.emplace_back(a, b, -w);
        t.emplace_back(b, a, -w);
        deg[a] += w;
        deg[b] += w;
    };
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const double w = 1.0 + 0.25 * ((r * 7 + c * 3) % 5);
            if (r + 1 < rows) edge(id(r, c), id(r + 1, c), w);
            if (c + 1 < cols) edge(id(r, c), id(r, c + 1), w);
        }
    for (int i = 0; i < n; ++i) t.emplace_back(i, i, deg[i]);
    Sparse L(n, n);
    L.setFromTriplets(t.begin(), t.end());
    L.makeCompressed();
    return L;
}

Eigen::VectorXd compatible_rhs(Eigen::Index n, unsigned salt = 1) {
    Eigen::VectorXd b(n);
    for (Eigen::Index i = 0; i < n; ++i)
        b[i] = std::sin(0.37 * static_cast<double>(i + 1) * salt) + 0.1 * salt;
    b.array() -= b.mean();
    return b;
}

apxchol_options defaults() {
    apxchol_options o;
    EXPECT_EQ(apxchol_options_default(&o, sizeof o), APXCHOL_STATUS_SUCCESS);
    return o;
}

struct handle {
    apxchol_solver* s = nullptr;
    ~handle() { apxchol_solver_destroy(s); }
};

apxchol_status create(const csc64& c, const apxchol_options* o, handle& h, std::int32_t base = 0,
                      std::string* message = nullptr) {
    std::array<char, 1024> err{};
    const apxchol_status s = apxchol_solver_create(c.n, c.colptr.data(), c.rowval.data(),
                                                   c.nzval.data(), base, o, &h.s, err.data(),
                                                   err.size());
    if (message != nullptr) *message = err.data();
    return s;
}

apxchol_solve_info info_struct() {
    apxchol_solve_info info{};
    info.struct_size = sizeof info;
    return info;
}

bool same_bytes(const double* a, const double* b, std::size_t n) {
    return std::memcmp(a, b, n * sizeof(double)) == 0;
}

}  // namespace

TEST(CApi, SolvesOneBasedLaplacianCsc) {
    const csc64 c{2, {1, 3, 5}, {1, 2, 1, 2}, {1.0, -1.0, -1.0, 1.0}};
    auto o = defaults();
    o.tol = 1e-10;
    o.threads = 1;
    handle h;
    std::string message;
    ASSERT_EQ(create(c, &o, h, 1, &message), APXCHOL_STATUS_SUCCESS) << message;
    const std::array<double, 2> b{1.0, -1.0};
    std::array<double, 2> x{};
    auto info = info_struct();
    ASSERT_EQ(apxchol_solver_solve(h.s, b.data(), nullptr, x.data(), -1.0, -1, &info, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_NEAR(x[0] - x[1], 1.0, 1e-12);
    EXPECT_NEAR(x[0] + x[1], 0.0, 1e-12);  // min-norm
    EXPECT_EQ(info.converged, 1);
    EXPECT_LT(info.relative_residual, 1e-10);
    EXPECT_GE(info.iterations, 1);
    EXPECT_GE(info.solve_seconds, 0.0);
}

TEST(CApi, RejectsInvalidIndexBase) {
    const csc64 c{1, {0, 1}, {0}, {1.0}};
    handle h;
    std::string message;
    EXPECT_EQ(create(c, nullptr, h, 2, &message), APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(h.s, nullptr);
    EXPECT_NE(message.find("index_base"), std::string::npos);
}

TEST(CApi, SupportsAliasedRhsGuessAndSolution) {
    const csc64 c{2, {0, 2, 4}, {0, 1, 0, 1}, {1.0, -1.0, -1.0, 1.0}};
    auto o = defaults();
    o.tol = 1e-10;
    handle h;
    ASSERT_EQ(create(c, &o, h), APXCHOL_STATUS_SUCCESS);
    std::array<double, 2> v{1.0, -1.0};
    ASSERT_EQ(apxchol_solver_solve(h.s, v.data(), v.data(), v.data(), -1.0, -1, nullptr, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_NEAR(v[0] - v[1], 1.0, 1e-12);
}

TEST(CApi, ExhaustionIsNotConvergedAndWritesTheSolution) {
    const csc64 c = to_csc64(grid_laplacian(5, 5, 0.05));
    auto o = defaults();
    o.stagnation_window = 0;
    handle h;
    ASSERT_EQ(create(c, &o, h), APXCHOL_STATUS_SUCCESS);
    const Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(c.n, 1.0, 2.0);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(c.n);
    auto info = info_struct();
    std::array<char, 256> err{};
    EXPECT_EQ(apxchol_solver_solve(h.s, b.data(), nullptr, x.data(), 1e-300, 1, &info, err.data(),
                                   err.size()),
              APXCHOL_STATUS_NOT_CONVERGED);
    EXPECT_EQ(info.converged, 0);
    EXPECT_EQ(info.iterations, 1);
    EXPECT_GT(info.relative_residual, 0.0);
    EXPECT_GT(x.norm(), 0.0);
    EXPECT_NE(std::string(err.data()).find("converge"), std::string::npos);
}

TEST(CApi, ThreadLimitIsRestoredAfterSuccessAndError) {
    const int before = apxchol_get_max_threads();
    const csc64 good = to_csc64(grid_laplacian(8, 8));
    const csc64 bad{2, {0, 2, 4}, {0, 1, 0, 1}, {0.0, 1.0, 1.0, 0.0}};  // adjacency
    auto o = defaults();
    o.threads = before > 1 ? 1 : 2;
    {
        handle h;
        ASSERT_EQ(create(good, &o, h), APXCHOL_STATUS_SUCCESS);
        EXPECT_EQ(apxchol_get_max_threads(), before);
        const Eigen::VectorXd b = compatible_rhs(good.n);
        Eigen::VectorXd x(good.n);
        EXPECT_EQ(apxchol_solver_solve(h.s, b.data(), nullptr, x.data(), -1.0, -1, nullptr, nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
        EXPECT_EQ(apxchol_get_max_threads(), before);
        apxchol_stats stats{};
        stats.struct_size = sizeof stats;
        ASSERT_EQ(apxchol_solver_stats(h.s, &stats), APXCHOL_STATUS_SUCCESS);
        EXPECT_EQ(stats.setup_max_threads, apxchol_openmp_enabled() ? o.threads : 1);
    }
    handle h;
    EXPECT_EQ(create(bad, &o, h), APXCHOL_STATUS_INVALID_OPERATOR);
    EXPECT_EQ(apxchol_get_max_threads(), before);
}

TEST(CApi, ExtremeColumnPointersDoNotOverflow) {
    const std::array<double, 1> v{1.0};
    const std::array<std::int64_t, 1> r{0};
    for (const std::array<std::int64_t, 2> colptr :
         {std::array<std::int64_t, 2>{0, std::numeric_limits<std::int64_t>::max()},
          std::array<std::int64_t, 2>{0, std::numeric_limits<std::int64_t>::min()},
          std::array<std::int64_t, 2>{std::numeric_limits<std::int64_t>::min(), 1}}) {
        int sentinel = 0;
        apxchol_solver* s = reinterpret_cast<apxchol_solver*>(&sentinel);  // must be reset
        std::array<char, 256> err{};
        EXPECT_EQ(apxchol_solver_create(1, colptr.data(), r.data(), v.data(), 0, nullptr, &s,
                                        err.data(), err.size()),
                  APXCHOL_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(s, nullptr);
        EXPECT_GT(std::strlen(err.data()), 0u);
    }
}

TEST(CApi, ExtremeRowIndicesDoNotOverflow) {
    const std::array<std::int64_t, 2> colptr{1, 2};
    const std::array<double, 1> v{1.0};
    for (const std::int64_t row : {std::numeric_limits<std::int64_t>::max(),
                                   std::numeric_limits<std::int64_t>::min(), std::int64_t{0},
                                   std::int64_t{2}}) {
        const std::array<std::int64_t, 1> r{row};
        apxchol_solver* s = nullptr;
        std::array<char, 256> err{};
        EXPECT_EQ(apxchol_solver_create(1, colptr.data(), r.data(), v.data(), 1, nullptr, &s,
                                        err.data(), err.size()),
                  APXCHOL_STATUS_INVALID_ARGUMENT);
        EXPECT_NE(std::string(err.data()).find("row index"), std::string::npos);
    }
}

TEST(CApi, TinyErrorBufferAndZeroedInfoOnError) {
    const csc64 c = to_csc64(grid_laplacian(4, 4));
    handle h;
    ASSERT_EQ(create(c, nullptr, h), APXCHOL_STATUS_SUCCESS);
    std::vector<double> b(c.n, 0.0), x(c.n, 0.0);
    b[0] = std::numeric_limits<double>::quiet_NaN();
    auto info = info_struct();
    info.iterations = 99;
    char one = 'x';
    EXPECT_EQ(apxchol_solver_solve(h.s, b.data(), nullptr, x.data(), -1.0, -1, &info, &one, 1),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(one, '\0');
    EXPECT_EQ(info.struct_size, sizeof info);
    EXPECT_EQ(info.iterations, 0);
    EXPECT_EQ(info.converged, 0);
}

TEST(CApi, RejectsNonFiniteOrZeroTolerance) {
    const csc64 c = to_csc64(grid_laplacian(4, 4));
    for (const double tol : {std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity(), 0.0, -1.0}) {
        auto o = defaults();
        o.tol = tol;
        handle h;
        EXPECT_EQ(create(c, &o, h), APXCHOL_STATUS_INVALID_ARGUMENT) << tol;
    }
    handle h;
    ASSERT_EQ(create(c, nullptr, h), APXCHOL_STATUS_SUCCESS);
    const Eigen::VectorXd b = compatible_rhs(c.n);
    Eigen::VectorXd x(c.n);
    for (const double tol : {std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::infinity(), 0.0})
        EXPECT_EQ(apxchol_solver_solve(h.s, b.data(), nullptr, x.data(), tol, -1, nullptr, nullptr, 0),
                  APXCHOL_STATUS_INVALID_ARGUMENT)
            << tol;
}

TEST(CApi, NonFiniteMatrixIsAnOperatorErrorNonFiniteRhsAnArgumentError) {
    csc64 c = to_csc64(grid_laplacian(4, 4));
    {
        csc64 bad = c;
        bad.nzval[3] = std::numeric_limits<double>::infinity();
        handle h;
        std::string message;
        EXPECT_EQ(create(bad, nullptr, h, 0, &message), APXCHOL_STATUS_INVALID_OPERATOR);
        EXPECT_NE(message.find("finite values"), std::string::npos) << message;
    }
    handle h;
    ASSERT_EQ(create(c, nullptr, h), APXCHOL_STATUS_SUCCESS);
    Eigen::VectorXd b = compatible_rhs(c.n), x(c.n), x0 = Eigen::VectorXd::Zero(c.n);
    x0[2] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(apxchol_solver_solve(h.s, b.data(), x0.data(), x.data(), -1.0, -1, nullptr, nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    b[1] = -std::numeric_limits<double>::infinity();
    std::array<char, 256> err{};
    EXPECT_EQ(apxchol_solver_solve(h.s, b.data(), nullptr, x.data(), -1.0, -1, nullptr, err.data(),
                                   err.size()),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_NE(std::string(err.data()).find("non-finite"), std::string::npos);
}

TEST(CApi, DefaultsAreTheCppDefaults) {
    const apxchol::solve_options so;
    const auto o = defaults();
    EXPECT_EQ(o.struct_size, sizeof(apxchol_options));
    EXPECT_EQ(o.backend, APXCHOL_BACKEND_CPU);
    EXPECT_EQ(o.tol, so.tol);
    EXPECT_EQ(o.max_iter, so.max_iter);
    EXPECT_EQ(o.stagnation_window, so.stagnation_window);
    EXPECT_EQ(o.seed, so.factor_opts.seed);
    EXPECT_EQ(o.threads, 0);
    EXPECT_EQ(o.sampler, APXCHOL_SAMPLER_GKS);
    EXPECT_EQ(o.partitioner, APXCHOL_PARTITIONER_BLOCK_GREEDY);
    EXPECT_EQ(o.storage, static_cast<apxchol_storage>(so.storage));
    EXPECT_EQ(o.storage, APXCHOL_STORAGE_VEC_POOL_AOS);
    EXPECT_EQ(o.keep_factor_values, so.keep_factor_values ? 1 : 0);
    EXPECT_EQ(o.degree_quantile, so.factor_opts.partition.degree_quantile);
    EXPECT_EQ(o.exact_clique_max_degree, so.factor_opts.exact_clique_max_degree);
}

TEST(CApi, StructSizeIsValidatedAndNothingPastTheStructIsWritten) {
    struct guarded_options {
        apxchol_options o;
        std::array<unsigned char, 32> guard;
    } g{};
    g.guard.fill(0xA5);
    EXPECT_EQ(apxchol_options_default(&g.o, sizeof(apxchol_options) - 8),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(apxchol_options_default(&g.o, sizeof(apxchol_options) + 8),
              APXCHOL_STATUS_UNSUPPORTED);
    EXPECT_EQ(apxchol_options_default(nullptr, sizeof(apxchol_options)),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(apxchol_options_default(&g.o, sizeof(apxchol_options)), APXCHOL_STATUS_SUCCESS);
    for (unsigned char byte : g.guard) EXPECT_EQ(byte, 0xA5);

    const csc64 c = to_csc64(grid_laplacian(4, 4));
    auto o = defaults();
    o.struct_size = sizeof(apxchol_options) + 4;
    handle h;
    EXPECT_EQ(create(c, &o, h), APXCHOL_STATUS_UNSUPPORTED);
    ASSERT_EQ(create(c, nullptr, h), APXCHOL_STATUS_SUCCESS);
    const Eigen::VectorXd b = compatible_rhs(c.n);
    Eigen::VectorXd x(c.n);
    apxchol_solve_info info{};
    info.struct_size = sizeof(apxchol_solve_info) - 4;
    EXPECT_EQ(apxchol_solver_solve(h.s, b.data(), nullptr, x.data(), -1.0, -1, &info, nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    apxchol_stats stats{};
    stats.struct_size = sizeof(apxchol_stats) + 8;
    EXPECT_EQ(apxchol_solver_stats(h.s, &stats), APXCHOL_STATUS_UNSUPPORTED);
}

TEST(CApi, SingleThreadFactorAndSolutionAreByteEqualToCppSolver) {
    const Sparse A = grid_laplacian(30, 30);
    const csc64 c = to_csc64(A, 1);
    auto o = defaults();
    o.threads = 1;
    o.keep_factor_values = 1;
    o.seed = 7;
    handle h;
    ASSERT_EQ(create(c, &o, h, 1), APXCHOL_STATUS_SUCCESS);

    apxchol::solve_options so;
    so.keep_factor_values = true;
    so.factor_opts.seed = 7;
#ifdef _OPENMP
    const int prior = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    const apxchol::cpu_solver ref(A, so);
    const Eigen::VectorXd b = compatible_rhs(c.n);
    const apxchol::solve_result r = ref.solve(b);
#ifdef _OPENMP
    omp_set_num_threads(prior);
#endif

    const apxchol::factorization& F = ref.preconditioner().factor();
    const auto nnz = static_cast<std::size_t>(F.L.nonZeros());
    std::vector<std::int64_t> colptr(c.n + 1), rowval(nnz), perm(c.n);
    std::vector<double> nzval(nnz);
    ASSERT_EQ(apxchol_solver_export_factor(h.s, 0, colptr.data(), rowval.data(), nzval.data(),
                                           perm.data(), nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    for (std::size_t j = 0; j < colptr.size(); ++j)
        ASSERT_EQ(colptr[j], static_cast<std::int64_t>(F.L.outer_[j]));
    for (std::size_t p = 0; p < nnz; ++p) {
        ASSERT_EQ(rowval[p], static_cast<std::int64_t>(F.L.inner_[p]));
        ASSERT_EQ(nzval[p], static_cast<double>(F.L.vals_[p]));
    }
    for (std::size_t v = 0; v < perm.size(); ++v)
        ASSERT_EQ(perm[v], static_cast<std::int64_t>(F.perm[v]));

    Eigen::VectorXd x(c.n);
    auto info = info_struct();
    ASSERT_EQ(apxchol_solver_solve(h.s, b.data(), nullptr, x.data(), -1.0, -1, &info, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_TRUE(same_bytes(x.data(), r.x.data(), x.size()));
    EXPECT_EQ(info.iterations, r.iterations);
    EXPECT_EQ(info.relative_residual, r.residual);
}

TEST(CApi, RepeatedSolvesAreBitIdenticalAndWarmStartAtTheSolutionIsFree) {
    const csc64 c = to_csc64(grid_laplacian(20, 20, 0.01));
    handle h;
    ASSERT_EQ(create(c, nullptr, h), APXCHOL_STATUS_SUCCESS);
    const Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(c.n, -1.0, 3.0);
    Eigen::VectorXd x1(c.n), x2(c.n), x3(c.n);
    ASSERT_EQ(apxchol_solver_solve(h.s, b.data(), nullptr, x1.data(), -1.0, -1, nullptr, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    ASSERT_EQ(apxchol_solver_solve(h.s, b.data(), nullptr, x2.data(), -1.0, -1, nullptr, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_TRUE(same_bytes(x1.data(), x2.data(), x1.size()));
    auto info = info_struct();
    ASSERT_EQ(apxchol_solver_solve(h.s, b.data(), x1.data(), x3.data(), 1e-6, -1, &info, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(info.iterations, 0);
    EXPECT_EQ(info.converged, 1);
}

TEST(CApi, BlockSolveEqualsColumnSolvesBitwise) {
    const csc64 c = to_csc64(grid_laplacian(12, 12));
    handle h;
    ASSERT_EQ(create(c, nullptr, h), APXCHOL_STATUS_SUCCESS);
    const std::int64_t k = 3;
    Eigen::MatrixXd B(c.n, k), X(c.n, k);
    B.col(0) = compatible_rhs(c.n, 1);
    B.col(1).setZero();
    B.col(2) = compatible_rhs(c.n, 5);
    std::array<std::int64_t, 3> iters{};
    std::array<double, 3> res{};
    std::array<std::int32_t, 3> conv{};
    ASSERT_EQ(apxchol_solver_solve_block(h.s, k, B.data(), nullptr, X.data(), -1.0, -1, iters.data(),
                                         res.data(), conv.data(), nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    for (std::int64_t col = 0; col < k; ++col) {
        Eigen::VectorXd x(c.n);
        const Eigen::VectorXd b = B.col(col);
        auto info = info_struct();
        ASSERT_EQ(apxchol_solver_solve(h.s, b.data(), nullptr, x.data(), -1.0, -1, &info, nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
        EXPECT_TRUE(same_bytes(x.data(), X.col(col).data(), x.size())) << col;
        EXPECT_EQ(iters[col], info.iterations);
        EXPECT_EQ(res[col], info.relative_residual);
        EXPECT_EQ(conv[col], 1);
    }
    EXPECT_EQ(iters[1], 0);
    EXPECT_EQ(X.col(1).norm(), 0.0);
    EXPECT_EQ(apxchol_solver_solve_block(h.s, 0, nullptr, nullptr, nullptr, -1.0, -1, nullptr,
                                         nullptr, nullptr, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(apxchol_solver_solve_block(h.s, -1, B.data(), nullptr, X.data(), -1.0, -1, nullptr,
                                         nullptr, nullptr, nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
}

TEST(CApi, ApplyMatchesCppApply) {
    const Sparse A = grid_laplacian(10, 10, 0.1);
    const csc64 c = to_csc64(A);
    auto o = defaults();
    o.threads = 1;
    handle h;
    ASSERT_EQ(create(c, &o, h), APXCHOL_STATUS_SUCCESS);
#ifdef _OPENMP
    const int prior = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    const apxchol::cpu_solver ref(A, apxchol::solve_options{});
    const Eigen::VectorXd r = compatible_rhs(c.n, 3);
    const Eigen::VectorXd z_ref = ref.apply(r);
#ifdef _OPENMP
    omp_set_num_threads(prior);
#endif
    Eigen::VectorXd z(c.n);
    ASSERT_EQ(apxchol_solver_apply(h.s, r.data(), z.data(), nullptr, 0), APXCHOL_STATUS_SUCCESS);
    EXPECT_TRUE(same_bytes(z.data(), z_ref.data(), z.size()));
}

TEST(CApi, StatsReportLumpingSddmAndFactorSize) {
    // SDDM with one mirrored positive off-diagonal pair: lumpable.
    Sparse A = grid_laplacian(4, 4, 0.5);
    A.coeffRef(1, 0) = -A.coeffRef(1, 0);
    A.coeffRef(0, 1) = -A.coeffRef(0, 1);
    A.makeCompressed();
    const csc64 c = to_csc64(A);
    auto o = defaults();
    o.keep_factor_values = 1;
    handle h;
    ASSERT_EQ(create(c, &o, h), APXCHOL_STATUS_SUCCESS);
    apxchol_stats stats{};
    stats.struct_size = sizeof stats;
    ASSERT_EQ(apxchol_solver_stats(h.s, &stats), APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(stats.backend, APXCHOL_BACKEND_CPU);
    EXPECT_EQ(stats.n, c.n);
    EXPECT_EQ(stats.lumped_offdiag, apxchol::detail::env_knobs::get().lump ? 2 : 0);
    EXPECT_EQ(stats.sddm, 1);
    EXPECT_GE(stats.rounds, 1);
    EXPECT_GE(stats.setup_seconds, 0.0);
    EXPECT_GE(stats.factor_nnz, c.n);

    handle lap;
    ASSERT_EQ(create(to_csc64(grid_laplacian(4, 4)), nullptr, lap), APXCHOL_STATUS_SUCCESS);
    ASSERT_EQ(apxchol_solver_stats(lap.s, &stats), APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(stats.sddm, 0);
    EXPECT_EQ(stats.lumped_offdiag, 0);
}

TEST(CApi, FactorExportStructureAndRetention) {
    const csc64 c = to_csc64(grid_laplacian(9, 7));
    handle released;
    ASSERT_EQ(create(c, nullptr, released), APXCHOL_STATUS_SUCCESS);
    std::vector<std::int64_t> perm(c.n);
    ASSERT_EQ(apxchol_solver_export_factor(released.s, 1, nullptr, nullptr, nullptr, perm.data(),
                                           nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    std::vector<std::int64_t> sorted = perm;
    std::sort(sorted.begin(), sorted.end());
    for (std::int64_t v = 0; v < c.n; ++v) EXPECT_EQ(sorted[v], v + 1);

    apxchol_stats stats{};
    stats.struct_size = sizeof stats;
    ASSERT_EQ(apxchol_solver_stats(released.s, &stats), APXCHOL_STATUS_SUCCESS);
    std::vector<std::int64_t> colptr(c.n + 1), rowval(stats.factor_nnz);
    std::vector<double> nzval(stats.factor_nnz);
    EXPECT_EQ(apxchol_solver_export_factor(released.s, 1, colptr.data(), rowval.data(), nzval.data(),
                                           nullptr, nullptr, 0),
              APXCHOL_STATUS_NO_FACTOR_VALUES);
    EXPECT_EQ(apxchol_solver_export_factor(released.s, 1, colptr.data(), nullptr, nzval.data(),
                                           nullptr, nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);

    auto o = defaults();
    o.keep_factor_values = 1;
    handle kept;
    ASSERT_EQ(create(c, &o, kept), APXCHOL_STATUS_SUCCESS);
    ASSERT_EQ(apxchol_solver_stats(kept.s, &stats), APXCHOL_STATUS_SUCCESS);
    colptr.assign(c.n + 1, 0);
    rowval.assign(stats.factor_nnz, 0);
    nzval.assign(stats.factor_nnz, 0.0);
    ASSERT_EQ(apxchol_solver_export_factor(kept.s, 1, colptr.data(), rowval.data(), nzval.data(),
                                           perm.data(), nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(colptr.front(), 1);
    EXPECT_EQ(colptr.back() - 1, stats.factor_nnz);
    for (std::int64_t j = 0; j < c.n; ++j) {
        ASSERT_LT(colptr[j], colptr[j + 1]) << "every column holds its diagonal";
        EXPECT_EQ(rowval[colptr[j] - 1], j + 1) << "diagonal first";
        for (std::int64_t p = colptr[j]; p < colptr[j + 1] - 1; ++p)
            EXPECT_GT(rowval[p], j + 1) << "lower triangular";
    }
}

TEST(CApi, AdjacencyIsAnOperatorError) {
    const csc64 c{3, {0, 1, 3, 4}, {1, 0, 2, 1}, {1.0, 1.0, 1.0, 1.0}};
    handle h;
    std::string message;
    EXPECT_EQ(create(c, nullptr, h, 0, &message), APXCHOL_STATUS_INVALID_OPERATOR);
    EXPECT_NE(message.find("positive diagonal"), std::string::npos) << message;
    EXPECT_NE(message.find("ADJACENCY"), std::string::npos) << message;
}

TEST(CApi, BackendAndEnumValidation) {
    const csc64 c = to_csc64(grid_laplacian(4, 4));
    EXPECT_EQ(apxchol_backend_available(APXCHOL_BACKEND_CPU), 1);
    EXPECT_EQ(apxchol_backend_available(42), 0);
#ifndef APXCHOL_USE_METAL
    EXPECT_EQ(apxchol_backend_available(APXCHOL_BACKEND_METAL), 0);
    {
        auto o = defaults();
        o.backend = APXCHOL_BACKEND_METAL;
        handle h;
        std::string message;
        EXPECT_EQ(create(c, &o, h, 0, &message), APXCHOL_STATUS_UNSUPPORTED);
        EXPECT_NE(message.find("Metal"), std::string::npos);
    }
#endif
    auto check = [&](auto mutate) {
        auto o = defaults();
        mutate(o);
        handle h;
        EXPECT_EQ(create(c, &o, h), APXCHOL_STATUS_INVALID_ARGUMENT);
    };
    check([](apxchol_options& o) { o.backend = 9; });
    check([](apxchol_options& o) { o.sampler = 7; });
    check([](apxchol_options& o) { o.storage = 1; });
    check([](apxchol_options& o) { o.storage = 3; });
    check([](apxchol_options& o) { o.partitioner = 9; });
    check([](apxchol_options& o) { o.keep_factor_values = 2; });
    check([](apxchol_options& o) { o.max_iter = -1; });
    check([](apxchol_options& o) { o.threads = -1; });
    check([](apxchol_options& o) { o.degree_quantile = 1.5; });

    for (const apxchol_storage storage :
         {APXCHOL_STORAGE_VEC, APXCHOL_STORAGE_BSTR, APXCHOL_STORAGE_VEC_POOL_AOS})
        for (const apxchol_partitioner partitioner :
             {APXCHOL_PARTITIONER_BLOCK_GREEDY, APXCHOL_PARTITIONER_PRIORITY_GREEDY,
              APXCHOL_PARTITIONER_BAUMANN_KYNG})
            for (const apxchol_sampler sampler :
                 {APXCHOL_SAMPLER_GKS, APXCHOL_SAMPLER_TRACE_CYCLE}) {
                auto o = defaults();
                o.storage = storage;
                o.partitioner = partitioner;
                o.sampler = sampler;
                handle h;
                std::string message;
                ASSERT_EQ(create(c, &o, h, 0, &message), APXCHOL_STATUS_SUCCESS)
                    << storage << " " << partitioner << " " << sampler << ": " << message;
                const Eigen::VectorXd b = compatible_rhs(c.n);
                Eigen::VectorXd x(c.n);
                EXPECT_EQ(apxchol_solver_solve(h.s, b.data(), nullptr, x.data(), -1.0, -1, nullptr,
                                               nullptr, 0),
                          APXCHOL_STATUS_SUCCESS);
            }
}

TEST(CApi, VersionAbiAndOpenMp) {
    EXPECT_EQ(apxchol_c_abi_version(), APXCHOL_C_ABI_VERSION);
    const std::string v = apxchol_version();
    EXPECT_NE(v.find('+'), std::string::npos) << v;
#ifdef _OPENMP
    EXPECT_EQ(apxchol_openmp_enabled(), 1);
    EXPECT_EQ(apxchol_get_max_threads(), omp_get_max_threads());
#else
    EXPECT_EQ(apxchol_openmp_enabled(), 0);
    EXPECT_EQ(apxchol_get_max_threads(), 1);
#endif
}

TEST(CApi, NullArgumentsAreErrorsAndDestroyNullIsANoOp) {
    apxchol_solver_destroy(nullptr);
    const csc64 c = to_csc64(grid_laplacian(3, 3));
    EXPECT_EQ(apxchol_solver_create(c.n, c.colptr.data(), c.rowval.data(), c.nzval.data(), 0,
                                    nullptr, nullptr, nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    apxchol_solver* s = nullptr;
    EXPECT_EQ(apxchol_solver_create(c.n, nullptr, c.rowval.data(), c.nzval.data(), 0, nullptr, &s,
                                    nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(apxchol_solver_create(0, c.colptr.data(), c.rowval.data(), c.nzval.data(), 0,
                                    nullptr, &s, nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    std::vector<double> v(c.n, 0.0);
    EXPECT_EQ(apxchol_solver_solve(nullptr, v.data(), nullptr, v.data(), -1.0, -1, nullptr, nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(apxchol_solver_apply(nullptr, v.data(), v.data(), nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(apxchol_solver_stats(nullptr, nullptr), APXCHOL_STATUS_INVALID_ARGUMENT);
    handle h;
    ASSERT_EQ(create(c, nullptr, h), APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(apxchol_solver_export_factor(h.s, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(apxchol_solver_solve(h.s, nullptr, nullptr, v.data(), -1.0, -1, nullptr, nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
}

TEST(CApi, UnsortedColumnsAndDuplicatesAreSummed) {
    const Sparse A = grid_laplacian(6, 6, 0.2);
    const csc64 sorted = to_csc64(A);
    // Reverse every column and split each diagonal into two duplicate halves.
    csc64 messy;
    messy.n = sorted.n;
    messy.colptr.push_back(0);
    for (std::int64_t j = 0; j < sorted.n; ++j) {
        for (std::int64_t p = sorted.colptr[j + 1] - 1; p >= sorted.colptr[j]; --p) {
            const double v = sorted.nzval[p];
            if (sorted.rowval[p] == j) {
                messy.rowval.insert(messy.rowval.end(), {j, j});
                messy.nzval.insert(messy.nzval.end(), {0.5 * v, v - 0.5 * v});
            } else {
                messy.rowval.push_back(sorted.rowval[p]);
                messy.nzval.push_back(v);
            }
        }
        messy.colptr.push_back(static_cast<std::int64_t>(messy.rowval.size()));
    }
    auto o = defaults();
    o.threads = 1;
    handle a, b;
    ASSERT_EQ(create(sorted, &o, a), APXCHOL_STATUS_SUCCESS);
    ASSERT_EQ(create(messy, &o, b), APXCHOL_STATUS_SUCCESS);
    const Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(sorted.n, 0.0, 1.0);
    Eigen::VectorXd xa(sorted.n), xb(sorted.n);
    ASSERT_EQ(apxchol_solver_solve(a.s, rhs.data(), nullptr, xa.data(), -1.0, -1, nullptr, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    ASSERT_EQ(apxchol_solver_solve(b.s, rhs.data(), nullptr, xb.data(), -1.0, -1, nullptr, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_TRUE(same_bytes(xa.data(), xb.data(), xa.size()));
}

TEST(CApi, DistinctHandlesSolveConcurrently) {
    const csc64 c = to_csc64(grid_laplacian(16, 16));
    std::array<int, 2> statuses{-1, -1};
    std::array<double, 2> residuals{};
    auto work = [&](int t) {
        auto o = defaults();
        o.threads = 2;
        handle h;
        if (create(c, &o, h) != APXCHOL_STATUS_SUCCESS) return;
        const Eigen::VectorXd b = compatible_rhs(c.n, static_cast<unsigned>(t + 1));
        Eigen::VectorXd x(c.n);
        auto info = info_struct();
        statuses[t] = apxchol_solver_solve(h.s, b.data(), nullptr, x.data(), -1.0, -1, &info,
                                           nullptr, 0);
        residuals[t] = info.relative_residual;
    };
    std::thread t0(work, 0), t1(work, 1);
    t0.join();
    t1.join();
    for (int t = 0; t < 2; ++t) {
        EXPECT_EQ(statuses[t], APXCHOL_STATUS_SUCCESS);
        EXPECT_LT(residuals[t], 1e-8);
    }
}
