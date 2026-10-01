// The C ABI (include/apxchol/c_api.h): statuses, validation, struct
// versioning, thread scoping and agreement with the C++ cpu_solver.
#include <gtest/gtest.h>

#include "apxchol/c_api.h"
#include "apxchol/env_knobs.h"
#include "apxchol/solver/solve.h"
#if defined(APXCHOL_USE_METAL)
#include "apxchol/solver/metal_solver.h"
#endif

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

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

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

/// `count` cliques of `size` vertices in a light ring. The first round
/// selects one vertex per clique, a yield below min_is_fraction, so the
/// factorization hands the residual over unless the selection reaches
/// omp_threshold (factor_options.h).
Sparse clique_ring(int count, int size) {
    const int n = count * size;
    std::vector<Eigen::Triplet<double>> t;
    std::vector<double> deg(n, 0.0);
    auto edge = [&](int a, int b, double w) {
        t.emplace_back(a, b, -w);
        t.emplace_back(b, a, -w);
        deg[a] += w;
        deg[b] += w;
    };
    for (int c = 0; c < count; ++c) {
        const int base = c * size;
        for (int i = 0; i < size; ++i)
            for (int j = i + 1; j < size; ++j) edge(base + i, base + j, 1.0 + 0.125 * ((i + j) % 4));
        edge(base, ((c + 1) % count) * size + 1, 0.5);
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

struct factor_handle {
    apxchol_factor* f = nullptr;
    ~factor_handle() { apxchol_factor_destroy(f); }
};

apxchol_status create_factor(const csc64& c, const apxchol_options* o, factor_handle& h,
                             std::int32_t base = 0, std::string* message = nullptr) {
    std::array<char, 1024> err{};
    const apxchol_status s = apxchol_factor_create(c.n, c.colptr.data(), c.rowval.data(),
                                                   c.nzval.data(), base, o, &h.f, err.data(),
                                                   err.size());
    if (message != nullptr) *message = err.data();
    return s;
}

apxchol_status adopt(const csc64& c, const apxchol_factor* f, const apxchol_options* o,
                     handle& h, std::int32_t base = 0, std::string* message = nullptr) {
    std::array<char, 1024> err{};
    const apxchol_status s = apxchol_solver_create_from_factor(
        c.n, c.colptr.data(), c.rowval.data(), c.nzval.data(), base, f, o, &h.s, err.data(),
        err.size());
    if (message != nullptr) *message = err.data();
    return s;
}

apxchol_stats stats_of(const apxchol_solver* s) {
    apxchol_stats st{};
    st.struct_size = sizeof st;
    EXPECT_EQ(apxchol_solver_stats(s, &st), APXCHOL_STATUS_SUCCESS);
    return st;
}

apxchol_stats stats_of(const apxchol_factor* f) {
    apxchol_stats st{};
    st.struct_size = sizeof st;
    EXPECT_EQ(apxchol_factor_stats(f, &st), APXCHOL_STATUS_SUCCESS);
    return st;
}

/// The fields of apxchol_stats that describe the factor itself.
void expect_same_factor_stats(const apxchol_stats& a, const apxchol_stats& b) {
    EXPECT_EQ(a.n, b.n);
    EXPECT_EQ(a.factor_nnz, b.factor_nnz);
    EXPECT_EQ(a.lumped_offdiag, b.lumped_offdiag);
    EXPECT_EQ(a.rounds, b.rounds);
    EXPECT_EQ(a.peak_graph_bytes, b.peak_graph_bytes);
    EXPECT_EQ(a.sddm, b.sddm);
}

struct exported {
    std::vector<std::int64_t> colptr, rowval, perm;
    std::vector<double> nzval;
    bool operator==(const exported& o) const {
        return colptr == o.colptr && rowval == o.rowval && perm == o.perm &&
               nzval.size() == o.nzval.size() &&
               same_bytes(nzval.data(), o.nzval.data(), nzval.size());
    }
};

template <class Handle, class Export>
exported export_with(const Handle* h, const apxchol_stats& st, Export fn) {
    exported e;
    e.colptr.resize(static_cast<std::size_t>(st.n + 1));
    e.rowval.resize(static_cast<std::size_t>(st.factor_nnz));
    e.nzval.resize(static_cast<std::size_t>(st.factor_nnz));
    e.perm.resize(static_cast<std::size_t>(st.n));
    EXPECT_EQ(fn(h, 0, e.colptr.data(), e.rowval.data(), e.nzval.data(), e.perm.data(), nullptr,
                 0),
              APXCHOL_STATUS_SUCCESS);
    return e;
}

exported export_of(const apxchol_solver* s) {
    return export_with(s, stats_of(s), apxchol_solver_export_factor);
}

exported export_of(const apxchol_factor* f) {
    return export_with(f, stats_of(f), apxchol_factor_export);
}

/// A copy of A with every off-diagonal weight scaled by a factor in
/// [0.6, 1.8] and edges (v, v + stride) of weight w added, the diagonal kept
/// at the new weighted degree plus A's excess: the same class, another
/// operator with a different sparsity pattern.
Sparse nearby_operator(const Sparse& A, int stride, double w) {
    const Eigen::Index n = A.rows();
    std::vector<Eigen::Triplet<double>> t;
    Eigen::VectorXd diag = Eigen::VectorXd::Zero(n);
    for (Eigen::Index k = 0; k < n; ++k)
        for (Sparse::InnerIterator it(A, k); it; ++it) {
            if (it.row() == it.col()) {
                diag[k] += it.value();
                continue;
            }
            const Eigen::Index lo = std::min(it.row(), it.col()), hi = std::max(it.row(), it.col());
            const double scale = 0.6 + 0.3 * static_cast<double>((lo * 5 + hi * 3) % 5);
            t.emplace_back(it.row(), it.col(), scale * it.value());
            diag[k] -= (scale - 1.0) * it.value();  // keeps A's row sum
        }
    for (Eigen::Index v = 0; v + stride < n; v += stride + 1) {
        t.emplace_back(v, v + stride, -w);
        t.emplace_back(v + stride, v, -w);
        diag[v] += w;
        diag[v + stride] += w;
    }
    for (Eigen::Index i = 0; i < n; ++i) t.emplace_back(i, i, diag[i]);
    Sparse B(n, n);
    B.setFromTriplets(t.begin(), t.end());
    B.makeCompressed();
    return B;
}

double true_residual(const Sparse& A, const Eigen::VectorXd& b, const Eigen::VectorXd& x) {
    return (b - A * x).norm() / b.norm();
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
    EXPECT_EQ(o.omp_threshold, so.factor_opts.omp_threshold);
    EXPECT_EQ(o.omp_threshold, apxchol::factor_options{}.omp_threshold);
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

TEST(CApi, StatsAreZeroedOnLaterErrorsAndUntouchedOnStructSizeMismatch) {
    const csc64 c = to_csc64(grid_laplacian(4, 4));
    handle h;
    ASSERT_EQ(create(c, nullptr, h), APXCHOL_STATUS_SUCCESS);
    factor_handle f;
    ASSERT_EQ(create_factor(c, nullptr, f), APXCHOL_STATUS_SUCCESS);
    apxchol_stats zeroed{};
    zeroed.struct_size = sizeof zeroed;
    apxchol_stats st;
    // A matching struct_size and a NULL handle: zeroed except struct_size.
    std::memset(&st, 0xA5, sizeof st);
    st.struct_size = sizeof st;
    EXPECT_EQ(apxchol_solver_stats(nullptr, &st), APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(std::memcmp(&st, &zeroed, sizeof st), 0);
    std::memset(&st, 0xA5, sizeof st);
    st.struct_size = sizeof st;
    EXPECT_EQ(apxchol_factor_stats(nullptr, &st), APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(std::memcmp(&st, &zeroed, sizeof st), 0);
    // A mismatched struct_size: the caller's struct is left untouched.
    for (const std::uint32_t size : {static_cast<std::uint32_t>(sizeof st + 8),
                                     static_cast<std::uint32_t>(sizeof st - 8)}) {
        const apxchol_status expected = size > sizeof st ? APXCHOL_STATUS_UNSUPPORTED
                                                         : APXCHOL_STATUS_INVALID_ARGUMENT;
        std::memset(&st, 0x5A, sizeof st);
        st.struct_size = size;
        const apxchol_stats before = st;
        EXPECT_EQ(apxchol_solver_stats(h.s, &st), expected);
        EXPECT_EQ(std::memcmp(&st, &before, sizeof st), 0);
        EXPECT_EQ(apxchol_factor_stats(f.f, &st), expected);
        EXPECT_EQ(std::memcmp(&st, &before, sizeof st), 0);
        EXPECT_EQ(apxchol_solver_stats(nullptr, &st), expected);
        EXPECT_EQ(std::memcmp(&st, &before, sizeof st), 0);
    }
}

TEST(CApi, ApplyIsAFixedMapForSddmAndFollowsTheCenterKScheduleForLaplacians) {
    const auto& knobs = apxchol::detail::env_knobs::get();
    if (knobs.ground != apxchol::detail::grounding_kind::center_k || knobs.center_k < 2)
        GTEST_SKIP() << "needs the default APXCHOL_GROUND=center-k with APXCHOL_CENTER_K >= 2";
    const int k = knobs.center_k;
    auto applications = [&](const csc64& c, const Eigen::VectorXd& r) {
        handle h;
        EXPECT_EQ(create(c, nullptr, h), APXCHOL_STATUS_SUCCESS);
        std::vector<Eigen::VectorXd> z(static_cast<std::size_t>(k), Eigen::VectorXd(c.n));
        for (Eigen::VectorXd& zi : z)
            EXPECT_EQ(apxchol_solver_apply(h.s, r.data(), zi.data(), nullptr, 0),
                      APXCHOL_STATUS_SUCCESS);
        return z;
    };
    const csc64 sddm = to_csc64(grid_laplacian(12, 12, 0.1)), lap = to_csc64(grid_laplacian(12, 12));
    const Eigen::VectorXd r = compatible_rhs(lap.n, 3);  // mean zero
    const std::vector<Eigen::VectorXd> fixed = applications(sddm, r);
    for (const Eigen::VectorXd& z : fixed)
        EXPECT_TRUE(same_bytes(z.data(), fixed[0].data(), z.size()));
    // Laplacian: applications 1..K-1 skip both centring passes; the K-th
    // centres, which moves z by a constant vector only.
    const std::vector<Eigen::VectorXd> z = applications(lap, r);
    for (int i = 1; i + 1 < k; ++i)
        EXPECT_TRUE(same_bytes(z[i].data(), z[0].data(), z[0].size())) << i;
    const Eigen::VectorXd d = z[k - 1] - z[0];
    EXPECT_GT(d.norm(), 0.0);
    EXPECT_LE((d.array() - d.mean()).matrix().norm(), 1e-10 * z[0].norm());
    EXPECT_LE(std::fabs(z[k - 1].mean()), 1e-12 * z[k - 1].norm());
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

#if defined(__unix__) || defined(__APPLE__)
TEST(CApi, OlderSmallerOptionsStructIsNotReadPastItsEnd) {
    // A 16-byte options struct from a hypothetical older header, ending at a
    // PROT_NONE page: create must reject it from struct_size alone.
    const long page = sysconf(_SC_PAGESIZE);
    void* mem = mmap(nullptr, static_cast<std::size_t>(2 * page), PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANON, -1, 0);
    ASSERT_NE(mem, MAP_FAILED);
    char* base = static_cast<char*>(mem);
    ASSERT_EQ(mprotect(base + page, static_cast<std::size_t>(page), PROT_NONE), 0);
    auto* small = reinterpret_cast<std::uint32_t*>(base + page - 16);
    std::memset(small, 0, 16);
    small[0] = 16;  // struct_size
    const csc64 c = to_csc64(grid_laplacian(4, 4));
    handle h;
    std::string message;
    const auto* older = reinterpret_cast<const apxchol_options*>(small);
    EXPECT_EQ(create(c, older, h, 0, &message), APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_NE(message.find("struct_size"), std::string::npos) << message;
    factor_handle f;
    EXPECT_EQ(create_factor(c, older, f, 0, &message), APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_NE(message.find("struct_size"), std::string::npos) << message;
    ASSERT_EQ(create_factor(c, nullptr, f), APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(adopt(c, f.f, older, h, 0, &message), APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_NE(message.find("struct_size"), std::string::npos) << message;
    munmap(mem, static_cast<std::size_t>(2 * page));
}
#endif

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
#else
    EXPECT_EQ(apxchol_backend_available(APXCHOL_BACKEND_METAL),
              apxchol::metal_solver::available() ? 1 : 0);
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

TEST(CApi, FactorHandleExportEqualsTheSolversBytewise) {
    // Single thread: the factorization is exactly repeatable (docs/precision.md).
    for (const double shift : {0.0, 0.05}) {
        const csc64 c = to_csc64(grid_laplacian(17, 13, shift), 1);
        auto o = defaults();
        o.threads = 1;
        o.seed = 11;
        factor_handle f;
        std::string message;
        ASSERT_EQ(create_factor(c, &o, f, 1, &message), APXCHOL_STATUS_SUCCESS) << message;
        o.keep_factor_values = 1;
        handle s;
        ASSERT_EQ(create(c, &o, s, 1), APXCHOL_STATUS_SUCCESS);
        EXPECT_TRUE(export_of(f.f) == export_of(s.s)) << shift;
        const apxchol_stats fs = stats_of(f.f), ss = stats_of(s.s);
        expect_same_factor_stats(fs, ss);
        EXPECT_EQ(fs.backend, APXCHOL_BACKEND_CPU);
        EXPECT_EQ(fs.sddm, shift > 0.0 ? 1 : 0);
        EXPECT_EQ(fs.setup_max_threads, 1);
        EXPECT_GE(fs.setup_seconds, 0.0);
    }
}

TEST(CApi, SolverFromFactorIsBitIdenticalOnItsOwnMatrix) {
    // Laplacian, SDDM and a lumpable SDDM operator (one positive pair): the
    // factor never refuses the matrix it was built from.
    Sparse lumpable = grid_laplacian(15, 15, 0.2);
    lumpable.coeffRef(1, 0) = -lumpable.coeffRef(1, 0);
    lumpable.coeffRef(0, 1) = -lumpable.coeffRef(0, 1);
    lumpable.makeCompressed();
    for (const Sparse& A : {grid_laplacian(24, 19), grid_laplacian(24, 19, 0.01), lumpable}) {
        const csc64 c = to_csc64(A);
        auto o = defaults();
        o.threads = 1;
        o.keep_factor_values = 1;
        handle self;
        ASSERT_EQ(create(c, &o, self), APXCHOL_STATUS_SUCCESS);
        factor_handle f;
        ASSERT_EQ(create_factor(c, &o, f), APXCHOL_STATUS_SUCCESS);
        handle adopted;
        std::string message;
        ASSERT_EQ(adopt(c, f.f, &o, adopted, 0, &message), APXCHOL_STATUS_SUCCESS) << message;

        EXPECT_TRUE(export_of(adopted.s) == export_of(self.s));
        const apxchol_stats as = stats_of(adopted.s);
        expect_same_factor_stats(as, stats_of(self.s));
        expect_same_factor_stats(as, stats_of(f.f));
        EXPECT_EQ(as.setup_max_threads, 1);

        const Eigen::VectorXd b = compatible_rhs(c.n, 3);
        Eigen::VectorXd x1(c.n), x2(c.n), x0 = 0.5 * compatible_rhs(c.n, 7);
        const Eigen::VectorXd& x0c = x0;
        for (const double* guess : {static_cast<const double*>(nullptr), x0c.data()}) {
            auto i1 = info_struct(), i2 = info_struct();
            ASSERT_EQ(apxchol_solver_solve(self.s, b.data(), guess, x1.data(), 1e-10, -1, &i1,
                                           nullptr, 0),
                      APXCHOL_STATUS_SUCCESS);
            ASSERT_EQ(apxchol_solver_solve(adopted.s, b.data(), guess, x2.data(), 1e-10, -1, &i2,
                                           nullptr, 0),
                      APXCHOL_STATUS_SUCCESS);
            EXPECT_TRUE(same_bytes(x1.data(), x2.data(), x1.size()));
            EXPECT_EQ(i1.iterations, i2.iterations);
            EXPECT_EQ(i1.relative_residual, i2.relative_residual);
        }
        Eigen::VectorXd z1(c.n), z2(c.n);
        ASSERT_EQ(apxchol_solver_apply(self.s, b.data(), z1.data(), nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
        ASSERT_EQ(apxchol_solver_apply(adopted.s, b.data(), z2.data(), nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
        EXPECT_TRUE(same_bytes(z1.data(), z2.data(), z1.size()));
    }
}

TEST(CApi, SolverFromFactorPreconditionsANearbyOperator) {
    for (const double shift : {0.0, 0.05}) {
        const Sparse A = grid_laplacian(26, 20, shift);
        const Sparse B = nearby_operator(A, 7, 0.5);
        ASSERT_GT(B.nonZeros(), A.nonZeros());
        auto o = defaults();
        o.tol = 1e-10;
        o.max_iter = 1000;
        factor_handle f;
        ASSERT_EQ(create_factor(to_csc64(A), &o, f), APXCHOL_STATUS_SUCCESS);
        const csc64 cb = to_csc64(B, 1);
        handle stale;
        std::string message;
        ASSERT_EQ(adopt(cb, f.f, &o, stale, 1, &message), APXCHOL_STATUS_SUCCESS) << message;
        expect_same_factor_stats(stats_of(stale.s), stats_of(f.f));

        const Eigen::VectorXd b = compatible_rhs(cb.n, 4);
        Eigen::VectorXd x(cb.n);
        auto info = info_struct();
        ASSERT_EQ(apxchol_solver_solve(stale.s, b.data(), nullptr, x.data(), -1.0, -1, &info,
                                       nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
        EXPECT_EQ(info.converged, 1);
        const double res = true_residual(B, b, x);
        EXPECT_LT(res, o.tol) << shift;
        if (shift == 0.0) EXPECT_LE(std::fabs(x.mean()), 1e-12 * x.norm());  // min-norm

        // The same solution as a solver that factorized B itself.
        handle fresh;
        ASSERT_EQ(create(cb, &o, fresh, 1), APXCHOL_STATUS_SUCCESS);
        Eigen::VectorXd y(cb.n);
        ASSERT_EQ(apxchol_solver_solve(fresh.s, b.data(), nullptr, y.data(), -1.0, -1, nullptr,
                                       nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
        EXPECT_LT((x - y).norm(), 1e-6 * y.norm()) << shift;
    }
}

TEST(CApi, SddmFactorMayPreconditionALaplacianButNotConversely) {
    const Sparse lap = grid_laplacian(18, 18), sddm = grid_laplacian(18, 18, 0.02);
    auto o = defaults();
    o.tol = 1e-10;
    o.max_iter = 1000;
    factor_handle lap_factor, sddm_factor;
    ASSERT_EQ(create_factor(to_csc64(lap), &o, lap_factor), APXCHOL_STATUS_SUCCESS);
    ASSERT_EQ(create_factor(to_csc64(sddm), &o, sddm_factor), APXCHOL_STATUS_SUCCESS);
    ASSERT_EQ(stats_of(lap_factor.f).sddm, 0);
    ASSERT_EQ(stats_of(sddm_factor.f).sddm, 1);

    handle refused;
    std::string message;
    EXPECT_EQ(adopt(to_csc64(sddm), lap_factor.f, &o, refused, 0, &message),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(refused.s, nullptr);
    EXPECT_NE(message.find("Laplacian factor"), std::string::npos) << message;

    // A full-rank factor on a singular operator: a solution, not centred.
    handle h;
    ASSERT_EQ(adopt(to_csc64(lap), sddm_factor.f, &o, h, 0, &message), APXCHOL_STATUS_SUCCESS)
        << message;
    EXPECT_EQ(stats_of(h.s).sddm, 1);
    const Eigen::VectorXd b = compatible_rhs(lap.rows(), 2);
    Eigen::VectorXd x(lap.rows());
    ASSERT_EQ(apxchol_solver_solve(h.s, b.data(), nullptr, x.data(), -1.0, -1, nullptr, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_LT(true_residual(lap, b, x), o.tol);
}

TEST(CApi, FactorHandleArgumentErrors) {
    const csc64 c = to_csc64(grid_laplacian(6, 5));
    std::string message;
    apxchol_factor_destroy(nullptr);

    // Creation: NULL outputs, CSC, options and the operator contract.
    EXPECT_EQ(apxchol_factor_create(c.n, c.colptr.data(), c.rowval.data(), c.nzval.data(), 0,
                                    nullptr, nullptr, nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    {
        factor_handle f;
        EXPECT_EQ(create_factor(c, nullptr, f, 2, &message), APXCHOL_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(f.f, nullptr);
        EXPECT_NE(message.find("index_base"), std::string::npos) << message;
        const csc64 adjacency{3, {0, 1, 3, 4}, {1, 0, 2, 1}, {1.0, 1.0, 1.0, 1.0}};
        EXPECT_EQ(create_factor(adjacency, nullptr, f, 0, &message),
                  APXCHOL_STATUS_INVALID_OPERATOR);
        EXPECT_NE(message.find("ADJACENCY"), std::string::npos) << message;
        auto o = defaults();
        o.sampler = 7;
        EXPECT_EQ(create_factor(c, &o, f), APXCHOL_STATUS_INVALID_ARGUMENT);
        o = defaults();
        o.struct_size = sizeof(apxchol_options) + 8;
        EXPECT_EQ(create_factor(c, &o, f), APXCHOL_STATUS_UNSUPPORTED);
        o.struct_size = sizeof(apxchol_options) - 8;
        EXPECT_EQ(create_factor(c, &o, f, 0, &message), APXCHOL_STATUS_INVALID_ARGUMENT);
        EXPECT_NE(message.find("struct_size"), std::string::npos) << message;
        EXPECT_EQ(f.f, nullptr);
    }

    factor_handle f;
    ASSERT_EQ(create_factor(c, nullptr, f), APXCHOL_STATUS_SUCCESS);  // keep_factor_values = 0
    const apxchol_stats st = stats_of(f.f);
    std::vector<std::int64_t> colptr(c.n + 1), rowval(st.factor_nnz), perm(c.n);
    std::vector<double> nzval(st.factor_nnz);
    // A factor handle always holds its values, whatever keep_factor_values said.
    EXPECT_EQ(apxchol_factor_export(f.f, 1, colptr.data(), rowval.data(), nzval.data(),
                                    perm.data(), nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(colptr.back() - 1, st.factor_nnz);
    EXPECT_EQ(apxchol_factor_export(nullptr, 0, nullptr, nullptr, nullptr, perm.data(), nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(apxchol_factor_export(f.f, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(apxchol_factor_export(f.f, 0, colptr.data(), nullptr, nzval.data(), nullptr,
                                    nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(apxchol_factor_export(f.f, 3, nullptr, nullptr, nullptr, perm.data(), nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    apxchol_stats bad{};
    bad.struct_size = sizeof bad + 8;
    EXPECT_EQ(apxchol_factor_stats(f.f, &bad), APXCHOL_STATUS_UNSUPPORTED);
    bad.struct_size = sizeof bad - 8;
    EXPECT_EQ(apxchol_factor_stats(f.f, &bad), APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(apxchol_factor_stats(nullptr, &bad), APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(apxchol_factor_stats(f.f, nullptr), APXCHOL_STATUS_INVALID_ARGUMENT);

    // Adoption: NULL factor or output, the dimension, the operator contract.
    handle h;
    EXPECT_EQ(adopt(c, nullptr, nullptr, h, 0, &message), APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_NE(message.find("factor is NULL"), std::string::npos) << message;
    EXPECT_EQ(apxchol_solver_create_from_factor(c.n, c.colptr.data(), c.rowval.data(),
                                                c.nzval.data(), 0, f.f, nullptr, nullptr,
                                                nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(adopt(to_csc64(grid_laplacian(5, 5)), f.f, nullptr, h, 0, &message),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(h.s, nullptr);
    EXPECT_NE(message.find("dimension"), std::string::npos) << message;
    csc64 asymmetric = c;
    asymmetric.nzval[1] *= 2.0;  // one off-diagonal no longer matches its partner
    EXPECT_EQ(adopt(asymmetric, f.f, nullptr, h, 0, &message), APXCHOL_STATUS_INVALID_OPERATOR);
    EXPECT_NE(message.find("symmetry"), std::string::npos) << message;
    csc64 nonfinite = c;
    nonfinite.nzval[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(adopt(nonfinite, f.f, nullptr, h), APXCHOL_STATUS_INVALID_OPERATOR);
    auto o = defaults();
    o.struct_size = sizeof(apxchol_options) + 4;
    EXPECT_EQ(adopt(c, f.f, &o, h), APXCHOL_STATUS_UNSUPPORTED);
    o = defaults();
    o.tol = -1.0;
    EXPECT_EQ(adopt(c, f.f, &o, h), APXCHOL_STATUS_INVALID_ARGUMENT);
#ifndef APXCHOL_USE_METAL
    o = defaults();
    o.backend = APXCHOL_BACKEND_METAL;
    EXPECT_EQ(adopt(c, f.f, &o, h), APXCHOL_STATUS_UNSUPPORTED);
#endif
    EXPECT_EQ(h.s, nullptr);

    // Copying a solver's factor needs retained values.
    handle released;
    ASSERT_EQ(adopt(c, f.f, nullptr, released), APXCHOL_STATUS_SUCCESS);
    int sentinel = 0;
    apxchol_factor* out = reinterpret_cast<apxchol_factor*>(&sentinel);  // must be reset
    std::array<char, 256> err{};
    EXPECT_EQ(apxchol_solver_copy_factor(released.s, &out, err.data(), err.size()),
              APXCHOL_STATUS_NO_FACTOR_VALUES);
    EXPECT_EQ(out, nullptr);
    EXPECT_NE(std::string(err.data()).find("keep_factor_values"), std::string::npos);
    EXPECT_EQ(apxchol_solver_export_factor(released.s, 0, colptr.data(), rowval.data(),
                                           nzval.data(), nullptr, nullptr, 0),
              APXCHOL_STATUS_NO_FACTOR_VALUES);
    EXPECT_EQ(apxchol_solver_copy_factor(nullptr, &out, nullptr, 0), APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(apxchol_solver_copy_factor(released.s, nullptr, nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
}

TEST(CApi, FactorHandleOutlivesAndIsUnchangedByItsSolvers) {
    const Sparse A = grid_laplacian(20, 20);
    const csc64 c = to_csc64(A);
    auto o = defaults();
    o.threads = 1;
    o.keep_factor_values = 1;
    const Eigen::VectorXd b = compatible_rhs(c.n, 6);

    factor_handle f;
    ASSERT_EQ(create_factor(c, &o, f), APXCHOL_STATUS_SUCCESS);
    const exported before = export_of(f.f);
    const apxchol_stats fs = stats_of(f.f);
    Eigen::VectorXd x_first(c.n);
    {
        handle s;
        ASSERT_EQ(adopt(c, f.f, &o, s), APXCHOL_STATUS_SUCCESS);
        ASSERT_EQ(apxchol_solver_solve(s.s, b.data(), nullptr, x_first.data(), -1.0, -1, nullptr,
                                       nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
    }  // the solver is gone; the factor is as it was
    EXPECT_TRUE(export_of(f.f) == before);
    const apxchol_stats again = stats_of(f.f);
    expect_same_factor_stats(again, fs);
    EXPECT_EQ(again.setup_seconds, fs.setup_seconds);

    // Two solvers adopt it concurrently, then outlive it.
    std::array<handle, 2> solvers;
    std::array<apxchol_status, 2> statuses{-1, -1};
    {
        std::thread t0([&] { statuses[0] = adopt(c, f.f, &o, solvers[0]); });
        std::thread t1([&] { statuses[1] = adopt(c, f.f, &o, solvers[1]); });
        t0.join();
        t1.join();
    }
    ASSERT_EQ(statuses[0], APXCHOL_STATUS_SUCCESS);
    ASSERT_EQ(statuses[1], APXCHOL_STATUS_SUCCESS);
    apxchol_factor_destroy(f.f);
    f.f = nullptr;
    for (handle& s : solvers) {
        Eigen::VectorXd x(c.n);
        ASSERT_EQ(apxchol_solver_solve(s.s, b.data(), nullptr, x.data(), -1.0, -1, nullptr,
                                       nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
        EXPECT_TRUE(same_bytes(x.data(), x_first.data(), x.size()));
        EXPECT_TRUE(export_of(s.s) == before);
    }
}

TEST(CApi, SolverFactorRoundTrip) {
    const csc64 c = to_csc64(grid_laplacian(16, 21, 0.03));
    auto o = defaults();
    o.threads = 1;
    o.keep_factor_values = 1;
    factor_handle copy;
    exported original;
    apxchol_stats solver_stats{};
    Eigen::VectorXd x_original(c.n);
    const Eigen::VectorXd b = compatible_rhs(c.n, 9);
    {
        handle s;
        ASSERT_EQ(create(c, &o, s), APXCHOL_STATUS_SUCCESS);
        std::array<char, 256> err{};
        ASSERT_EQ(apxchol_solver_copy_factor(s.s, &copy.f, err.data(), err.size()),
                  APXCHOL_STATUS_SUCCESS)
            << err.data();
        original = export_of(s.s);
        solver_stats = stats_of(s.s);
        ASSERT_EQ(apxchol_solver_solve(s.s, b.data(), nullptr, x_original.data(), -1.0, -1,
                                       nullptr, nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
    }  // the copy outlives the solver it came from
    EXPECT_TRUE(export_of(copy.f) == original);
    const apxchol_stats cs = stats_of(copy.f);
    expect_same_factor_stats(cs, solver_stats);
    EXPECT_EQ(cs.setup_seconds, solver_stats.setup_seconds);
    EXPECT_EQ(cs.setup_max_threads, solver_stats.setup_max_threads);
    EXPECT_EQ(cs.backend, APXCHOL_BACKEND_CPU);

    // ... and a solver adopting it solves as the original did.
    handle again;
    ASSERT_EQ(adopt(c, copy.f, &o, again), APXCHOL_STATUS_SUCCESS);
    Eigen::VectorXd x(c.n);
    ASSERT_EQ(apxchol_solver_solve(again.s, b.data(), nullptr, x.data(), -1.0, -1, nullptr,
                                   nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_TRUE(same_bytes(x.data(), x_original.data(), x.size()));
    factor_handle twice;
    ASSERT_EQ(apxchol_solver_copy_factor(again.s, &twice.f, nullptr, 0), APXCHOL_STATUS_SUCCESS);
    EXPECT_TRUE(export_of(twice.f) == original);
}

TEST(CApi, OmpThresholdReachesTheFactorization) {
    if (apxchol::detail::env_knobs::get().omp_threshold >= 0)
        GTEST_SKIP() << "APXCHOL_OMP_THRESHOLD overrides factor_options::omp_threshold";
    // One thread, so factors are exactly repeatable; a nondefault threshold
    // must produce the C++ factorization with that threshold, which differs
    // from the default one here: a threshold of 1 keeps the low-yield first
    // round that the default hands over.
    const Sparse A = clique_ring(40, 30);
    const csc64 c = to_csc64(A);
    auto exported_cpp = [&](std::size_t threshold) {
        apxchol::factor_options fo;
        fo.omp_threshold = threshold;
#ifdef _OPENMP
        const int prior = omp_get_max_threads();
        omp_set_num_threads(1);
#endif
        const apxchol::factorization F =
            apxchol::factorize(A, apxchol::solve_options{}.storage, fo);
#ifdef _OPENMP
        omp_set_num_threads(prior);
#endif
        exported e;
        for (auto p : F.L.outer_) e.colptr.push_back(static_cast<std::int64_t>(p));
        for (auto r : F.L.inner_) e.rowval.push_back(static_cast<std::int64_t>(r));
        for (auto v : F.L.vals_) e.nzval.push_back(static_cast<double>(v));
        for (auto v : F.perm) e.perm.push_back(static_cast<std::int64_t>(v));
        return e;
    };
    const std::uint64_t threshold = 1;
    const exported nondefault = exported_cpp(threshold);
    ASSERT_FALSE(nondefault == exported_cpp(apxchol::factor_options{}.omp_threshold))
        << "the threshold does not change this factor; pick another matrix";

    auto o = defaults();
    o.threads = 1;
    factor_handle by_default;
    ASSERT_EQ(create_factor(c, &o, by_default), APXCHOL_STATUS_SUCCESS);
    EXPECT_TRUE(export_of(by_default.f) == exported_cpp(apxchol::factor_options{}.omp_threshold));
    o.omp_threshold = threshold;
    factor_handle f;
    ASSERT_EQ(create_factor(c, &o, f), APXCHOL_STATUS_SUCCESS);
    EXPECT_TRUE(export_of(f.f) == nondefault);
    EXPECT_GT(stats_of(f.f).rounds, stats_of(by_default.f).rounds);
    o.keep_factor_values = 1;
    handle s;
    ASSERT_EQ(create(c, &o, s), APXCHOL_STATUS_SUCCESS);
    EXPECT_TRUE(export_of(s.s) == nondefault);
}

#if defined(APXCHOL_USE_METAL)
TEST(CApi, MetalBackendRoundTrip) {
    if (!apxchol::metal_solver::available()) GTEST_SKIP() << "no usable Metal device";
    EXPECT_EQ(apxchol_backend_available(APXCHOL_BACKEND_METAL), 1);
    const Sparse A = grid_laplacian(14, 13);
    const csc64 c = to_csc64(A, 1);
    auto o = defaults();
    o.backend = APXCHOL_BACKEND_METAL;
    o.keep_factor_values = 1;
    o.tol = 1e-10;
    handle h;
    std::string message;
    ASSERT_EQ(create(c, &o, h, 1, &message), APXCHOL_STATUS_SUCCESS) << message;

    apxchol_stats stats{};
    stats.struct_size = sizeof stats;
    ASSERT_EQ(apxchol_solver_stats(h.s, &stats), APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(stats.backend, APXCHOL_BACKEND_METAL);
    EXPECT_EQ(stats.n, c.n);
    EXPECT_EQ(stats.sddm, 0);

    // One right-hand side: the reported residual is the original system's.
    const Eigen::VectorXd b = compatible_rhs(c.n, 2);
    Eigen::VectorXd x(c.n);
    auto info = info_struct();
    ASSERT_EQ(apxchol_solver_solve(h.s, b.data(), nullptr, x.data(), -1.0, -1, &info, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(info.converged, 1);
    EXPECT_GT(info.iterations, 0);
    const double res = (b - A * x).norm() / b.norm();
    EXPECT_LT(res, 1e-10);
    EXPECT_NEAR(info.relative_residual, res, 1e-3 * res);
    EXPECT_LE(std::fabs(x.mean()), 1e-12 * x.norm());

    // Lockstep block: every column equals its single solve, bit for bit.
    const std::int64_t k = 3;
    Eigen::MatrixXd B(c.n, k), X(c.n, k);
    B.col(0) = compatible_rhs(c.n, 1);
    B.col(1).setZero();
    B.col(2) = compatible_rhs(c.n, 5);
    std::array<std::int64_t, 3> iters{};
    std::array<double, 3> resid{};
    std::array<std::int32_t, 3> conv{};
    ASSERT_EQ(apxchol_solver_solve_block(h.s, k, B.data(), nullptr, X.data(), -1.0, -1, iters.data(),
                                         resid.data(), conv.data(), nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    for (std::int64_t col = 0; col < k; ++col) {
        Eigen::VectorXd xc(c.n);
        const Eigen::VectorXd bc = B.col(col);
        auto ic = info_struct();
        ASSERT_EQ(apxchol_solver_solve(h.s, bc.data(), nullptr, xc.data(), -1.0, -1, &ic, nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
        EXPECT_TRUE(same_bytes(xc.data(), X.col(col).data(), xc.size())) << col;
        EXPECT_EQ(iters[col], ic.iterations);
        EXPECT_EQ(resid[col], ic.relative_residual);
        EXPECT_EQ(conv[col], 1);
    }
    EXPECT_EQ(iters[1], 0);
    EXPECT_EQ(X.col(1).norm(), 0.0);
    // In place: x aliases b.
    Eigen::MatrixXd BX = B;
    ASSERT_EQ(apxchol_solver_solve_block(h.s, k, BX.data(), nullptr, BX.data(), -1.0, -1, nullptr,
                                         nullptr, nullptr, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_TRUE(same_bytes(BX.data(), X.data(), static_cast<std::size_t>(X.size())));
    // Exhaustion is NOT_CONVERGED with the truthful residual of x = 0.
    auto none = info_struct();
    EXPECT_EQ(apxchol_solver_solve(h.s, b.data(), nullptr, x.data(), -1.0, 0, &none, nullptr, 0),
              APXCHOL_STATUS_NOT_CONVERGED);
    EXPECT_EQ(none.relative_residual, 1.0);

    // A preconditioner application and the retained factor.
    Eigen::VectorXd z(c.n);
    ASSERT_EQ(apxchol_solver_apply(h.s, b.data(), z.data(), nullptr, 0), APXCHOL_STATUS_SUCCESS);
    EXPECT_TRUE(z.allFinite());
    EXPECT_GT(z.dot(b), 0.0);
    std::vector<std::int64_t> colptr(c.n + 1), rowval(stats.factor_nnz), perm(c.n);
    std::vector<double> nzval(stats.factor_nnz);
    ASSERT_EQ(apxchol_solver_export_factor(h.s, 0, colptr.data(), rowval.data(), nzval.data(),
                                           perm.data(), nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(colptr.back(), stats.factor_nnz);
    std::vector<std::int64_t> sorted = perm;
    std::sort(sorted.begin(), sorted.end());
    for (std::int64_t v = 0; v < c.n; ++v) EXPECT_EQ(sorted[v], v);

    // keep_factor_values = 0 releases the values after setup, as on the CPU.
    o.keep_factor_values = 0;
    handle released;
    ASSERT_EQ(create(c, &o, released, 1), APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(apxchol_solver_export_factor(released.s, 0, colptr.data(), rowval.data(), nzval.data(),
                                           nullptr, nullptr, 0),
              APXCHOL_STATUS_NO_FACTOR_VALUES);
    EXPECT_EQ(apxchol_solver_export_factor(released.s, 0, nullptr, nullptr, nullptr, perm.data(),
                                           nullptr, 0),
              APXCHOL_STATUS_SUCCESS);

    // Non-finite right-hand sides are argument errors.
    Eigen::VectorXd bad = b;
    bad[2] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(apxchol_solver_solve(h.s, bad.data(), nullptr, x.data(), -1.0, -1, nullptr, nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(apxchol_solver_solve_block(h.s, 1, bad.data(), nullptr, x.data(), -1.0, -1, nullptr,
                                         nullptr, nullptr, nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
}

TEST(CApi, MetalUnrepresentableOperatorIsUnsupported) {
    if (!apxchol::metal_solver::available()) GTEST_SKIP() << "no usable Metal device";
    // A valid SDDM operator with one off-diagonal below the device's 2^-100
    // floor: the Metal backend cannot represent it, the CPU backend solves it.
    Sparse A = grid_laplacian(6, 6, 0.5);
    A.coeffRef(1, 0) = -1e-35;
    A.coeffRef(0, 1) = -1e-35;
    const csc64 c = to_csc64(A);
    auto o = defaults();
    o.backend = APXCHOL_BACKEND_METAL;
    handle h;
    std::string message;
    EXPECT_EQ(create(c, &o, h, 0, &message), APXCHOL_STATUS_UNSUPPORTED);
    EXPECT_NE(message.find("2^-100"), std::string::npos) << message;
    o.backend = APXCHOL_BACKEND_CPU;
    handle cpu;
    EXPECT_EQ(create(c, &o, cpu), APXCHOL_STATUS_SUCCESS);
}

TEST(CApi, MetalLaplacianApplyCentresEveryApplication) {
    if (!apxchol::metal_solver::available()) GTEST_SKIP() << "no usable Metal device";
    const csc64 c = to_csc64(grid_laplacian(12, 12));
    auto o = defaults();
    o.backend = APXCHOL_BACKEND_METAL;
    handle h;
    ASSERT_EQ(create(c, &o, h), APXCHOL_STATUS_SUCCESS);
    const Eigen::VectorXd r = compatible_rhs(c.n, 3).array() + 0.25;  // mean not zero
    Eigen::VectorXd z0(c.n), z(c.n);
    ASSERT_EQ(apxchol_solver_apply(h.s, r.data(), z0.data(), nullptr, 0), APXCHOL_STATUS_SUCCESS);
    EXPECT_LE(std::fabs(z0.mean()), 1e-5 * z0.cwiseAbs().maxCoeff());
    for (int i = 0; i < 12; ++i) {
        ASSERT_EQ(apxchol_solver_apply(h.s, r.data(), z.data(), nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
        EXPECT_TRUE(same_bytes(z.data(), z0.data(), z.size())) << i;
    }
}

TEST(CApi, MetalSolverFromFactorIsBitIdenticalOnItsOwnMatrix) {
    if (!apxchol::metal_solver::available()) GTEST_SKIP() << "no usable Metal device";
    for (const double shift : {0.0, 0.02}) {
        const csc64 c = to_csc64(grid_laplacian(19, 17, shift));
        auto o = defaults();
        o.backend = APXCHOL_BACKEND_METAL;
        o.threads = 1;
        o.keep_factor_values = 1;
        o.tol = 1e-10;
        handle self;
        std::string message;
        ASSERT_EQ(create(c, &o, self, 0, &message), APXCHOL_STATUS_SUCCESS) << message;
        factor_handle f;
        ASSERT_EQ(create_factor(c, &o, f), APXCHOL_STATUS_SUCCESS);
        handle adopted;
        ASSERT_EQ(adopt(c, f.f, &o, adopted, 0, &message), APXCHOL_STATUS_SUCCESS) << message;
        const apxchol_stats as = stats_of(adopted.s);
        EXPECT_EQ(as.backend, APXCHOL_BACKEND_METAL);
        expect_same_factor_stats(as, stats_of(self.s));
        EXPECT_TRUE(export_of(adopted.s) == export_of(self.s));

        const std::int64_t k = 3;
        Eigen::MatrixXd B(c.n, k), X1(c.n, k), X2(c.n, k);
        for (std::int64_t col = 0; col < k; ++col)
            B.col(col) = compatible_rhs(c.n, static_cast<unsigned>(col + 2));
        std::array<std::int64_t, 3> it1{}, it2{};
        std::array<double, 3> r1{}, r2{};
        ASSERT_EQ(apxchol_solver_solve_block(self.s, k, B.data(), nullptr, X1.data(), -1.0, -1,
                                             it1.data(), r1.data(), nullptr, nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
        ASSERT_EQ(apxchol_solver_solve_block(adopted.s, k, B.data(), nullptr, X2.data(), -1.0, -1,
                                             it2.data(), r2.data(), nullptr, nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
        EXPECT_TRUE(same_bytes(X1.data(), X2.data(), static_cast<std::size_t>(X1.size())));
        EXPECT_EQ(it1, it2);
        EXPECT_EQ(r1, r2);
        Eigen::VectorXd z1(c.n), z2(c.n);
        const Eigen::VectorXd r = B.col(0);
        ASSERT_EQ(apxchol_solver_apply(self.s, r.data(), z1.data(), nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
        ASSERT_EQ(apxchol_solver_apply(adopted.s, r.data(), z2.data(), nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
        EXPECT_TRUE(same_bytes(z1.data(), z2.data(), z1.size()));
    }
}

TEST(CApi, MetalSolverFromFactorPreconditionsANearbyOperator) {
    if (!apxchol::metal_solver::available()) GTEST_SKIP() << "no usable Metal device";
    for (const double shift : {0.0, 0.05}) {
        const Sparse A = grid_laplacian(26, 20, shift);
        const Sparse Bop = nearby_operator(A, 7, 0.5);
        auto o = defaults();
        o.tol = 1e-10;
        o.max_iter = 1000;
        factor_handle f;
        ASSERT_EQ(create_factor(to_csc64(A), &o, f), APXCHOL_STATUS_SUCCESS);
        o.backend = APXCHOL_BACKEND_METAL;
        const csc64 cb = to_csc64(Bop);
        handle stale;
        std::string message;
        ASSERT_EQ(adopt(cb, f.f, &o, stale, 0, &message), APXCHOL_STATUS_SUCCESS) << message;
        const Eigen::VectorXd b = compatible_rhs(cb.n, 4);
        Eigen::VectorXd x(cb.n);
        auto info = info_struct();
        ASSERT_EQ(apxchol_solver_solve(stale.s, b.data(), nullptr, x.data(), -1.0, -1, &info,
                                       nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
        const double res = true_residual(Bop, b, x);
        EXPECT_LT(res, o.tol) << shift;
        EXPECT_NEAR(info.relative_residual, res, 1e-3 * res);
        if (shift == 0.0) EXPECT_LE(std::fabs(x.mean()), 1e-12 * x.norm());
    }
}

TEST(CApi, MetalFactorHandlesCrossBackends) {
    if (!apxchol::metal_solver::available()) GTEST_SKIP() << "no usable Metal device";
    const csc64 c = to_csc64(grid_laplacian(12, 15));
    auto o = defaults();
    o.backend = APXCHOL_BACKEND_METAL;
    o.keep_factor_values = 1;
    handle metal;
    ASSERT_EQ(create(c, &o, metal), APXCHOL_STATUS_SUCCESS);
    factor_handle copy;
    ASSERT_EQ(apxchol_solver_copy_factor(metal.s, &copy.f, nullptr, 0), APXCHOL_STATUS_SUCCESS);
    const apxchol_stats cs = stats_of(copy.f), ms = stats_of(metal.s);
    EXPECT_EQ(cs.backend, APXCHOL_BACKEND_CPU);
    expect_same_factor_stats(cs, ms);
    EXPECT_EQ(cs.setup_seconds, ms.setup_seconds);
    EXPECT_TRUE(export_of(copy.f) == export_of(metal.s));
    // The Metal solver's factor on the CPU backend, and back again.
    o.backend = APXCHOL_BACKEND_CPU;
    handle cpu;
    ASSERT_EQ(adopt(c, copy.f, &o, cpu), APXCHOL_STATUS_SUCCESS);
    EXPECT_TRUE(export_of(cpu.s) == export_of(metal.s));
    factor_handle back;
    ASSERT_EQ(apxchol_solver_copy_factor(cpu.s, &back.f, nullptr, 0), APXCHOL_STATUS_SUCCESS);
    o.backend = APXCHOL_BACKEND_METAL;
    handle metal_again;
    ASSERT_EQ(adopt(c, back.f, &o, metal_again), APXCHOL_STATUS_SUCCESS);
    const Eigen::VectorXd b = compatible_rhs(c.n, 5);
    Eigen::VectorXd x1(c.n), x2(c.n);
    ASSERT_EQ(apxchol_solver_solve(metal.s, b.data(), nullptr, x1.data(), -1.0, -1, nullptr,
                                   nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    ASSERT_EQ(apxchol_solver_solve(metal_again.s, b.data(), nullptr, x2.data(), -1.0, -1, nullptr,
                                   nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_TRUE(same_bytes(x1.data(), x2.data(), x1.size()));

    // The adoption checks do not depend on the backend.
    std::string message;
    handle refused;
    EXPECT_EQ(adopt(to_csc64(grid_laplacian(12, 15, 0.1)), copy.f, &o, refused, 0, &message),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_NE(message.find("Laplacian factor"), std::string::npos) << message;
    EXPECT_EQ(adopt(to_csc64(grid_laplacian(12, 14)), copy.f, &o, refused, 0, &message),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_NE(message.find("dimension"), std::string::npos) << message;
    EXPECT_EQ(refused.s, nullptr);

    // An operator the device cannot represent (see
    // MetalUnrepresentableOperatorIsUnsupported) is UNSUPPORTED, as in create.
    o.keep_factor_values = 0;
    factor_handle sddm;
    ASSERT_EQ(create_factor(to_csc64(grid_laplacian(6, 6, 0.5)), &o, sddm), APXCHOL_STATUS_SUCCESS);
    Sparse tiny = grid_laplacian(6, 6, 0.5);
    tiny.coeffRef(1, 0) = -1e-35;
    tiny.coeffRef(0, 1) = -1e-35;
    EXPECT_EQ(adopt(to_csc64(tiny), sddm.f, &o, refused, 0, &message), APXCHOL_STATUS_UNSUPPORTED);
    EXPECT_NE(message.find("2^-100"), std::string::npos) << message;
    o.backend = APXCHOL_BACKEND_CPU;
    EXPECT_EQ(adopt(to_csc64(tiny), sddm.f, &o, refused), APXCHOL_STATUS_SUCCESS);
}
#endif

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
