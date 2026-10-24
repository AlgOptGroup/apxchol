// The C ABI (include/apxchol/c_api.h): statuses, validation, struct
// versioning, thread scoping and agreement with the C++ cpu_solver.
#include <gtest/gtest.h>

#include "apxchol/c_api.h"
#include "apxchol/solver/solve.h"
#if defined(APXCHOL_USE_CUDA)
#include "apxchol/solver/detail/cuda_error.h"
#endif

#include <Eigen/Sparse>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#ifdef _OPENMP
#include "scoped_omp_threads.h"
#endif

namespace {

using Sparse = Eigen::SparseMatrix<double>;

constexpr double tolerance = 1e-8;
constexpr int max_iterations = 200;

struct csc64 {
    std::int64_t n = 0;
    std::vector<std::int64_t> colptr, rowval;
    std::vector<double> nzval;

    apxchol_csc_view view(std::int32_t base = 0) const {
        return {n, colptr.data(), rowval.data(), nzval.data(),
                APXCHOL_INDEX_INT64, APXCHOL_INDEX_INT64, base};
    }
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

apxchol_options cpu_options() {
    auto o = defaults();
    o.backend = APXCHOL_BACKEND_CPU;
    return o;
}

apxchol_status solve_one(apxchol_solver* s, const double* b, const double* x0, double* x,
                         double tol, std::int32_t max_iter, apxchol_solve_info* info,
                         char* error, std::size_t capacity) {
    const apxchol_const_dense_view rhs{b, 1, 0}, guess{x0, 1, 0};
    const apxchol_dense_view out{x, 1, 0};
    return apxchol_solver_solve(s, 1, &rhs, x0 ? &guess : nullptr, &out, tol, max_iter,
                               info, sizeof(*info), error, capacity);
}

apxchol_status apply_one(apxchol_solver* s, const double* r, double* z,
                         char* error, std::size_t capacity) {
    const apxchol_const_dense_view rhs{r, 1, 0};
    const apxchol_dense_view out{z, 1, 0};
    return apxchol_solver_apply(s, 1, &rhs, &out, error, capacity);
}

std::vector<apxchol_backend> available_backends() {
    std::vector<apxchol_backend> backends{APXCHOL_BACKEND_CPU};
    if (apxchol_backend_available(APXCHOL_BACKEND_GPU)) backends.push_back(APXCHOL_BACKEND_GPU);
    return backends;
}

std::vector<apxchol_backend> adoption_backends() {
#if defined(APXCHOL_USE_METAL)
    return available_backends();
#else
    return {APXCHOL_BACKEND_CPU};
#endif
}

struct handle {
    apxchol_solver* s = nullptr;
    ~handle() { apxchol_solver_destroy(s); }
};

apxchol_status create(const csc64& c, const apxchol_options* o, handle& h, std::int32_t base = 0,
                      std::string* message = nullptr) {
    const auto fallback = cpu_options();
    if (o == nullptr) o = &fallback;
    std::array<char, 1024> err{};
    const auto matrix = c.view(base);
    const apxchol_status s = apxchol_solver_create(&matrix, nullptr, o, &h.s, err.data(), err.size());
    if (message != nullptr) *message = err.data();
    return s;
}

apxchol_solve_info info_struct() {
    apxchol_solve_info info{};
    return info;
}

template <class T>
bool same_bytes(const T* a, const T* b, std::size_t n) {
    return std::memcmp(a, b, n * sizeof(T)) == 0;
}

struct factor_handle {
    apxchol_factor* f = nullptr;
    ~factor_handle() { apxchol_factor_destroy(f); }
};

apxchol_status create_factor(const csc64& c, const apxchol_options* o, factor_handle& h,
                             std::int32_t base = 0, std::string* message = nullptr) {
    std::array<char, 1024> err{};
    const auto matrix = c.view(base);
    const apxchol_status s = apxchol_factor_create(&matrix, o, &h.f, err.data(), err.size());
    if (message != nullptr) *message = err.data();
    return s;
}

apxchol_status adopt(const csc64& c, const apxchol_factor* f, const apxchol_options* o,
                     handle& h, std::int32_t base = 0, std::string* message = nullptr) {
    const auto fallback = cpu_options();
    if (o == nullptr) o = &fallback;
    std::array<char, 1024> err{};
    const auto matrix = c.view(base);
    const apxchol_status s = apxchol_solver_create(&matrix, f, o, &h.s, err.data(), err.size());
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
    EXPECT_EQ(a.sddm, b.sddm);
}

struct exported {
    std::vector<std::int64_t> colptr, rowval, perm;
    std::vector<float> nzval;
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

exported export_of(const apxchol::factorization& f) {
    exported e;
    for (auto p : f.L.outer_) e.colptr.push_back(static_cast<std::int64_t>(p));
    for (auto r : f.L.inner_) e.rowval.push_back(static_cast<std::int64_t>(r));
    for (auto v : f.L.vals_) e.nzval.push_back(static_cast<float>(v));
    for (auto v : f.perm) e.perm.push_back(static_cast<std::int64_t>(v));
    return e;
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
    auto o = cpu_options();
    o.threads = 1;
    handle h;
    std::string message;
    ASSERT_EQ(create(c, &o, h, 1, &message), APXCHOL_STATUS_SUCCESS) << message;
    const std::array<double, 2> b{1.0, -1.0};
    std::array<double, 2> x{};
    auto info = info_struct();
    ASSERT_EQ(solve_one(h.s, b.data(), nullptr, x.data(), 1e-10, max_iterations, &info, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_NEAR(x[0] - x[1], 1.0, 1e-12);
    EXPECT_NEAR(x[0] + x[1], 0.0, 1e-12);  // min-norm
    EXPECT_EQ(info.converged, 1);
    EXPECT_LT(info.relative_residual, 1e-10);
    EXPECT_GE(info.iterations, 1);
}

TEST(CApi, RejectsInvalidIndexBase) {
    const csc64 c{1, {0, 1}, {0}, {1.0}};
    handle h;
    std::string message;
    EXPECT_EQ(create(c, nullptr, h, 2, &message), APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(h.s, nullptr);
    EXPECT_NE(message.find("index_base"), std::string::npos);
}

TEST(CApi, SolveAndApplyRejectAllInputOutputOverlap) {
    const csc64 c = to_csc64(grid_laplacian(8, 7, 0.1));
    for (const auto backend : available_backends()) {
        auto o = cpu_options(); o.backend = backend;
        handle h;
        ASSERT_EQ(create(c, &o, h), APXCHOL_STATUS_SUCCESS);
        for (const int k : {1, 5}) {
            SCOPED_TRACE(::testing::Message() << "backend=" << backend << " nrhs=" << k);
            Eigen::VectorXd storage = Eigen::VectorXd::Ones(c.n * k + 1);
            Eigen::MatrixXd other = Eigen::MatrixXd::Ones(c.n, k), out(c.n, k);
            const apxchol_const_dense_view rhs{storage.data(), 1, c.n}, other_view{other.data(), 1, c.n};
            const apxchol_dense_view out_view{out.data(), 1, c.n};
            for (const int offset : {0, 1}) {
                const apxchol_dense_view overlap{storage.data() + offset, 1, c.n};
                EXPECT_EQ(apxchol_solver_solve(h.s, k, &rhs, nullptr, &overlap, tolerance, max_iterations, nullptr, 0, nullptr, 0),
                    APXCHOL_STATUS_INVALID_ARGUMENT);
                EXPECT_EQ(apxchol_solver_solve(h.s, k, &other_view, &rhs, &overlap, tolerance, max_iterations, nullptr, 0, nullptr, 0),
                    APXCHOL_STATUS_INVALID_ARGUMENT);
                EXPECT_EQ(apxchol_solver_apply(h.s, k, &rhs, &overlap,
                    nullptr, 0), APXCHOL_STATUS_INVALID_ARGUMENT);
            }
            // Inputs may alias one another when the output is separate.
            EXPECT_EQ(apxchol_solver_solve(h.s, k, &other_view, &other_view, &out_view,
                tolerance, max_iterations, nullptr, 0, nullptr, 0), APXCHOL_STATUS_SUCCESS);
        }
    }
}

TEST(CApi, CscIndexWidthsAndBasesShareTheNativeImporter) {
    const Sparse A = grid_laplacian(4, 3, 0.25);
    auto options = cpu_options();
    options.threads = 1;
    options.keep_factor_values = 1;
    factor_handle reference;
    ASSERT_EQ(create_factor(to_csc64(A), &options, reference), APXCHOL_STATUS_SUCCESS);
    const auto expected = export_of(reference.f);
    for (const int base : {0, 1}) {
        const auto c = to_csc64(A, base);
        const std::vector<std::int32_t> ptr32(c.colptr.begin(), c.colptr.end());
        const std::vector<std::int32_t> row32(c.rowval.begin(), c.rowval.end());
        for (const auto ptr_type : {APXCHOL_INDEX_INT32, APXCHOL_INDEX_INT64}) {
            for (const auto row_type : {APXCHOL_INDEX_INT32, APXCHOL_INDEX_INT64}) {
                SCOPED_TRACE(::testing::Message() << "base=" << base << " ptr=" << ptr_type
                                                  << " row=" << row_type);
                const apxchol_csc_view matrix{c.n,
                    ptr_type == APXCHOL_INDEX_INT32 ? static_cast<const void*>(ptr32.data()) : c.colptr.data(),
                    row_type == APXCHOL_INDEX_INT32 ? static_cast<const void*>(row32.data()) : c.rowval.data(),
                    c.nzval.data(), ptr_type, row_type, base};
                factor_handle factor;
                ASSERT_EQ(apxchol_factor_create(&matrix, &options, &factor.f, nullptr, 0),
                          APXCHOL_STATUS_SUCCESS);
                EXPECT_TRUE(export_of(factor.f) == expected);
                for (const auto backend : available_backends()) {
                    auto selected = options;
                    selected.backend = backend;
                    handle h;
                    ASSERT_EQ(apxchol_solver_create(&matrix, nullptr, &selected, &h.s, nullptr, 0),
                              APXCHOL_STATUS_SUCCESS) << backend;
                    const Eigen::VectorXd b = compatible_rhs(c.n);
                    Eigen::VectorXd x(c.n);
                    ASSERT_EQ(solve_one(h.s, b.data(), nullptr, x.data(), tolerance, max_iterations,
                                        nullptr, nullptr, 0), APXCHOL_STATUS_SUCCESS);
                    EXPECT_LT(true_residual(A, b, x), tolerance);
                }
            }
        }
    }
}

TEST(CApi, InvalidCscViewsFailBeforeReadingTheirArrays) {
    const csc64 c{1, {0, 1}, {0}, {2.0}};
    const auto good = c.view();
    auto reject = [](const apxchol_csc_view* matrix) {
        handle h;
        factor_handle f;
        EXPECT_EQ(apxchol_solver_create(matrix, nullptr, nullptr, &h.s, nullptr, 0),
                  APXCHOL_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(apxchol_factor_create(matrix, nullptr, &f.f, nullptr, 0),
                  APXCHOL_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(h.s, nullptr);
        EXPECT_EQ(f.f, nullptr);
    };
    reject(nullptr);
    for (const int tag : {0, 16, -32, 65}) {
        auto bad = good;
        bad.colptr_type = tag;
        reject(&bad);
        bad = good;
        bad.rowval_type = tag;
        reject(&bad);
    }
    for (const auto n : {std::int64_t{-1}, std::int64_t{0},
                         std::int64_t{1} + std::numeric_limits<std::int32_t>::max(),
                         std::numeric_limits<std::int64_t>::max()}) {
        auto bad = good;
        bad.n = n;
        reject(&bad);
    }
    for (int field = 0; field < 3; ++field) {
        auto bad = good;
        if (field == 0) bad.colptr = nullptr;
        if (field == 1) bad.rowval = nullptr;
        if (field == 2) bad.nzval = nullptr;
        reject(&bad);
    }
    const std::int32_t ptr32[]{0, std::numeric_limits<std::int32_t>::min()};
    const apxchol_csc_view negative_nnz{1, ptr32, c.rowval.data(), c.nzval.data(),
                                      APXCHOL_INDEX_INT32, APXCHOL_INDEX_INT64, 0};
    reject(&negative_nnz);
    const std::int32_t row32[]{std::numeric_limits<std::int32_t>::max()};
    const apxchol_csc_view out_of_range{1, c.colptr.data(), row32, c.nzval.data(),
                                      APXCHOL_INDEX_INT64, APXCHOL_INDEX_INT32, 0};
    reject(&out_of_range);
}

namespace {

// Padding is NaN so accidentally reading or writing a gap is observable.
struct strided_array {
    std::int64_t rows, columns, row_stride, column_stride, origin;
    std::vector<double> storage;

    strided_array(std::int64_t n, std::int64_t k, std::int64_t rs, std::int64_t cs)
        : rows(n), columns(k), row_stride(rs), column_stride(cs) {
        const auto row_end = (n - 1) * rs, column_end = (k - 1) * cs;
        const auto first = std::min(std::int64_t{0}, row_end) + std::min(std::int64_t{0}, column_end);
        const auto last = std::max(std::int64_t{0}, row_end) + std::max(std::int64_t{0}, column_end);
        origin = 1 - first;
        storage.assign(static_cast<std::size_t>(last - first + 3),
                       std::numeric_limits<double>::quiet_NaN());
    }
    apxchol_const_dense_view input() const {
        return {storage.data() + origin, row_stride, column_stride};
    }
    apxchol_dense_view output() {
        return {storage.data() + origin, row_stride, column_stride};
    }
    void assign(const Eigen::MatrixXd& values) {
        for (std::int64_t j = 0; j < columns; ++j)
            for (std::int64_t i = 0; i < rows; ++i)
                storage[origin + i * row_stride + j * column_stride] = values(i, j);
    }
    Eigen::MatrixXd values() const {
        Eigen::MatrixXd result(rows, columns);
        for (std::int64_t j = 0; j < columns; ++j)
            for (std::int64_t i = 0; i < rows; ++i)
                result(i, j) = storage[origin + i * row_stride + j * column_stride];
        return result;
    }
    void expect_padding_unchanged() const {
        std::vector<bool> used(storage.size(), false);
        for (std::int64_t j = 0; j < columns; ++j)
            for (std::int64_t i = 0; i < rows; ++i)
                used[origin + i * row_stride + j * column_stride] = true;
        for (std::size_t i = 0; i < storage.size(); ++i) {
            if (!used[i]) {
                EXPECT_TRUE(std::isnan(storage[i])) << i;
            }
        }
    }
};

} // namespace

TEST(CApi, StridedScalarAndBatchSolveWarmStartAndApplyMatchContiguousViews) {
    const Sparse A = grid_laplacian(4, 4, 0.2);
    const auto c = to_csc64(A);
    for (const auto backend : available_backends()) {
        auto options = cpu_options();
        options.backend = backend;
        handle h;
        ASSERT_EQ(create(c, &options, h), APXCHOL_STATUS_SUCCESS);
        for (const std::int64_t k : {1, 5}) {
            Eigen::MatrixXd B(c.n, k), expected(c.n, k), applied(c.n, k);
            for (int j = 0; j < k; ++j) B.col(j) = compatible_rhs(c.n, j + 1);
            const apxchol_const_dense_view rhs{B.data(), 1, c.n};
            const apxchol_dense_view solution{expected.data(), 1, c.n}, preconditioned{applied.data(), 1, c.n};
            ASSERT_EQ(apxchol_solver_solve(h.s, k, &rhs, nullptr, &solution, tolerance,
                                          max_iterations, nullptr, 0, nullptr, 0), APXCHOL_STATUS_SUCCESS);
            ASSERT_EQ(apxchol_solver_apply(h.s, k, &rhs, &preconditioned, nullptr, 0), APXCHOL_STATUS_SUCCESS);
            const std::array<std::array<std::int64_t, 2>, 7> layouts{{
                {1, c.n}, {k, 1}, {2, 2 * c.n + 3}, {-1, c.n + 2},
                {1, -c.n - 2}, {-2, -2 * c.n - 3}, {5, 3}}};
            for (std::size_t l = 0; l < layouts.size(); ++l) {
                SCOPED_TRACE(::testing::Message() << "backend=" << backend << " columns=" << k << " layout=" << l);
                const auto bs = layouts[l], gs = layouts[(l + 1) % layouts.size()], xs = layouts[(l + 2) % layouts.size()];
                strided_array b(c.n, k, bs[0], bs[1]), guess(c.n, k, gs[0], gs[1]), x(c.n, k, xs[0], xs[1]);
                b.assign(B);
                guess.assign(0.25 * B);
                const auto input = b.input(), initial = guess.input();
                const auto output = x.output();
                std::vector<apxchol_solve_info> info(k);
                for (const bool warm : {false, true}) {
                    ASSERT_EQ(apxchol_solver_solve(h.s, k, &input, warm ? &initial : nullptr,
                        &output, tolerance, max_iterations, info.data(), sizeof(info[0]), nullptr, 0),
                        APXCHOL_STATUS_SUCCESS);
                    const auto actual = x.values();
                    EXPECT_TRUE(actual.isApprox(expected, 1e-6));
                    for (int j = 0; j < k; ++j) {
                        EXPECT_LT(true_residual(A, B.col(j), actual.col(j)), tolerance);
                        EXPECT_EQ(info[j].converged, 1);
                    }
                    x.expect_padding_unchanged();
                }
                ASSERT_EQ(apxchol_solver_apply(h.s, k, &input, &output, nullptr, 0), APXCHOL_STATUS_SUCCESS);
                EXPECT_TRUE(x.values().isApprox(applied, 1e-12));
                b.expect_padding_unchanged();
                guess.expect_padding_unchanged();
                x.expect_padding_unchanged();
            }
        }
    }
}

TEST(CApi, InvalidDenseViewsAreRejectedBeforeAccessOrOutputWrites) {
    const auto c = to_csc64(grid_laplacian(2, 2, 0.2));
    handle h;
    ASSERT_EQ(create(c, nullptr, h), APXCHOL_STATUS_SUCCESS);
    std::array<double, 12> b{}, x{};
    x.fill(7);
    const apxchol_const_dense_view rhs{b.data(), 1, c.n};
    const apxchol_dense_view out{x.data(), 1, c.n};
    constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
    constexpr auto minimum = std::numeric_limits<std::int64_t>::min();
    const auto high = reinterpret_cast<double*>(std::numeric_limits<std::uintptr_t>::max() - 7);
    const auto low = reinterpret_cast<double*>(std::uintptr_t{8});
    const std::array<apxchol_const_dense_view, 9> invalid{{
        {nullptr, 1, c.n}, {b.data(), 0, c.n}, {b.data(), 1, 0},
        {b.data(), minimum, 1}, {b.data(), maximum, 1}, {b.data(), 1, maximum},
        {high, 1, c.n}, {low, -1, -c.n}, {b.data(), maximum / 2, maximum / 2}}};
    for (const auto& bad : invalid) {
        EXPECT_EQ(apxchol_solver_solve(h.s, 3, &bad, nullptr, &out, tolerance,
            max_iterations, nullptr, 0, nullptr, 0), APXCHOL_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(apxchol_solver_solve(h.s, 3, &rhs, &bad, &out, tolerance,
            max_iterations, nullptr, 0, nullptr, 0), APXCHOL_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(apxchol_solver_apply(h.s, 3, &bad, &out, nullptr, 0), APXCHOL_STATUS_INVALID_ARGUMENT);
        const apxchol_dense_view bad_out{const_cast<double*>(bad.data), bad.row_stride, bad.column_stride};
        EXPECT_EQ(apxchol_solver_solve(h.s, 3, &rhs, nullptr, &bad_out, tolerance,
            max_iterations, nullptr, 0, nullptr, 0), APXCHOL_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(apxchol_solver_apply(h.s, 3, &rhs, &bad_out, nullptr, 0), APXCHOL_STATUS_INVALID_ARGUMENT);
        EXPECT_TRUE(std::all_of(x.begin(), x.end(), [](double v) { return v == 7; }));
    }
    for (const auto strides : {std::array<std::int64_t, 2>{1, 1}, {2, 4}, {-2, -4}}) {
        const apxchol_dense_view collision{x.data() + 8, strides[0], strides[1]};
        EXPECT_EQ(apxchol_solver_solve(h.s, 3, &rhs, nullptr, &collision, tolerance,
            max_iterations, nullptr, 0, nullptr, 0), APXCHOL_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(apxchol_solver_apply(h.s, 3, &rhs, &collision, nullptr, 0), APXCHOL_STATUS_INVALID_ARGUMENT);
    }
    EXPECT_EQ(apxchol_solver_solve(h.s, 3, nullptr, nullptr, &out, tolerance,
        max_iterations, nullptr, 0, nullptr, 0), APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(apxchol_solver_solve(h.s, 3, &rhs, nullptr, nullptr, tolerance,
        max_iterations, nullptr, 0, nullptr, 0), APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(apxchol_solver_apply(h.s, 3, nullptr, &out, nullptr, 0), APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(apxchol_solver_apply(h.s, 3, &rhs, nullptr, nullptr, 0), APXCHOL_STATUS_INVALID_ARGUMENT);
}

TEST(CApi, InactiveDenseStridesAreIgnoredAndStridedOverlapIsRejected) {
    const auto c = to_csc64(grid_laplacian(2, 2, 0.2));
    handle h;
    ASSERT_EQ(create(c, nullptr, h), APXCHOL_STATUS_SUCCESS);
    std::array<double, 4> b{1, 2, 3, 4}, x{};
    constexpr auto ignored = std::numeric_limits<std::int64_t>::min();
    const apxchol_const_dense_view rhs{b.data(), 1, ignored}, empty{nullptr, ignored, ignored};
    const apxchol_dense_view out{x.data(), 1, ignored}, empty_out{nullptr, ignored, ignored};
    EXPECT_EQ(apxchol_solver_solve(h.s, 1, &rhs, nullptr, &out, tolerance,
        max_iterations, nullptr, 0, nullptr, 0), APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(apxchol_solver_apply(h.s, 1, &rhs, &out, nullptr, 0), APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(apxchol_solver_solve(h.s, 0, &empty, &empty, &empty_out, tolerance,
        max_iterations, nullptr, 0, nullptr, 0), APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(apxchol_solver_apply(h.s, 0, &empty, &empty_out, nullptr, 0), APXCHOL_STATUS_SUCCESS);
    const csc64 scalar{1, {0, 1}, {0}, {2}};
    handle one;
    ASSERT_EQ(create(scalar, nullptr, one), APXCHOL_STATUS_SUCCESS);
    const apxchol_const_dense_view scalar_rhs{b.data(), ignored, 1};
    const apxchol_dense_view scalar_out{x.data(), ignored, 1};
    EXPECT_EQ(apxchol_solver_solve(one.s, 4, &scalar_rhs, nullptr, &scalar_out, tolerance,
        max_iterations, nullptr, 0, nullptr, 0), APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(apxchol_solver_apply(one.s, 4, &scalar_rhs, &scalar_out, nullptr, 0), APXCHOL_STATUS_SUCCESS);

    std::array<double, 16> interleaved{};
    const apxchol_const_dense_view evens{interleaved.data(), 2, 8};
    const apxchol_dense_view odds{interleaved.data() + 1, 2, 8};
    // The contract rejects overlapping bounding ranges, including disjoint gaps.
    EXPECT_EQ(apxchol_solver_solve(h.s, 2, &evens, nullptr, &odds, tolerance,
        max_iterations, nullptr, 0, nullptr, 0), APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(apxchol_solver_apply(h.s, 2, &evens, &odds, nullptr, 0), APXCHOL_STATUS_INVALID_ARGUMENT);
}

TEST(CApi, ExhaustionIsNotConvergedAndWritesTheSolution) {
    const csc64 c = to_csc64(grid_laplacian(5, 5, 0.05));
    auto o = cpu_options();
    handle h;
    ASSERT_EQ(create(c, &o, h), APXCHOL_STATUS_SUCCESS);
    const Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(c.n, 1.0, 2.0);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(c.n);
    auto info = info_struct();
    std::array<char, 256> err{};
    EXPECT_EQ(solve_one(h.s, b.data(), nullptr, x.data(), 1e-300, 1, &info, err.data(),
                                   err.size()),
              APXCHOL_STATUS_NOT_CONVERGED);
    EXPECT_EQ(info.converged, 0);
    EXPECT_EQ(info.iterations, 1);
    EXPECT_GT(info.relative_residual, 0.0);
    EXPECT_GT(x.norm(), 0.0);
    EXPECT_NE(std::string(err.data()).find("converge"), std::string::npos);
}

TEST(CApi, ThreadLimitIsRestoredAfterSuccessAndError) {
#ifdef _OPENMP
    const int before = omp_get_max_threads();
    auto o = cpu_options();
    o.threads = before > 1 ? 1 : 2;
    const csc64 c = to_csc64(grid_laplacian(8, 8));
    handle h;
    ASSERT_EQ(create(c, &o, h), APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(omp_get_max_threads(), before);
    const Eigen::VectorXd b = compatible_rhs(c.n);
    Eigen::VectorXd x(c.n);
    EXPECT_EQ(solve_one(h.s, b.data(), nullptr, x.data(), tolerance, max_iterations, nullptr, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(omp_get_max_threads(), before);
    EXPECT_EQ(apply_one(h.s, b.data(), x.data(), nullptr, 0), APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(omp_get_max_threads(), before);
    const csc64 bad{2, {0, 2, 4}, {0, 1, 0, 1}, {0.0, 1.0, 1.0, 0.0}};
    handle refused;
    EXPECT_EQ(create(bad, &o, refused), APXCHOL_STATUS_INVALID_OPERATOR);
    EXPECT_EQ(omp_get_max_threads(), before);
#endif
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
        const apxchol_csc_view matrix{1, colptr.data(), r.data(), v.data(),
                                      APXCHOL_INDEX_INT64, APXCHOL_INDEX_INT64, 0};
        EXPECT_EQ(apxchol_solver_create(&matrix, nullptr, nullptr, &s,
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
        const apxchol_csc_view matrix{1, colptr.data(), r.data(), v.data(),
                                      APXCHOL_INDEX_INT64, APXCHOL_INDEX_INT64, 1};
        EXPECT_EQ(apxchol_solver_create(&matrix, nullptr, nullptr, &s,
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
    EXPECT_EQ(solve_one(h.s, b.data(), nullptr, x.data(), tolerance, max_iterations, &info, &one, 1),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(one, '\0');
    EXPECT_EQ(info.iterations, 0);
    EXPECT_EQ(info.converged, 0);
}

TEST(CApi, SolveRequiresAnExplicitValidToleranceAndIterationBudget) {
    const csc64 c = to_csc64(grid_laplacian(4, 4));
    for (const auto backend : available_backends()) {
        SCOPED_TRACE(backend);
        auto o = cpu_options();
        o.backend = backend;
        handle h;
        ASSERT_EQ(create(c, &o, h), APXCHOL_STATUS_SUCCESS);
        const Eigen::VectorXd b = compatible_rhs(c.n);
        Eigen::VectorXd x(c.n);
        for (const double tol : {std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::infinity(),
                                -std::numeric_limits<double>::infinity(), 0.0, -1.0}) {
            EXPECT_EQ(solve_one(h.s, b.data(), nullptr, x.data(), tol, max_iterations,
                                nullptr, nullptr, 0), APXCHOL_STATUS_INVALID_ARGUMENT) << tol;
            EXPECT_EQ(apxchol_solver_solve(h.s, 0, nullptr, nullptr, nullptr, tol,
                max_iterations, nullptr, 0, nullptr, 0), APXCHOL_STATUS_INVALID_ARGUMENT);
        }
        for (const int budget : {-1, std::numeric_limits<int>::min()}) {
            EXPECT_EQ(solve_one(h.s, b.data(), nullptr, x.data(), tolerance, budget,
                                nullptr, nullptr, 0), APXCHOL_STATUS_INVALID_ARGUMENT);
            EXPECT_EQ(apxchol_solver_solve(h.s, 0, nullptr, nullptr, nullptr, tolerance,
                budget, nullptr, 0, nullptr, 0), APXCHOL_STATUS_INVALID_ARGUMENT);
        }
    }
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
    EXPECT_EQ(solve_one(h.s, b.data(), x0.data(), x.data(), tolerance, max_iterations, nullptr, nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    b[1] = -std::numeric_limits<double>::infinity();
    std::array<char, 256> err{};
    EXPECT_EQ(solve_one(h.s, b.data(), nullptr, x.data(), tolerance, max_iterations, nullptr, err.data(),
                                   err.size()),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_NE(std::string(err.data()).find("non-finite"), std::string::npos);
}

TEST(CApi, DefaultsAreTheCppDefaults) {
    const apxchol::solve_options so;
    const auto o = defaults();
    EXPECT_EQ(o.struct_size, sizeof(apxchol_options));
    EXPECT_EQ(o.backend, APXCHOL_BACKEND_AUTO);
    EXPECT_EQ(so.backend, apxchol::solve_backend::automatic);
    EXPECT_EQ(o.seed, so.factor_opts.seed);
    EXPECT_EQ(o.threads, 0);
    EXPECT_EQ(o.keep_factor_values, so.keep_factor_values ? 1 : 0);
    EXPECT_EQ(o.factor_storage, APXCHOL_FACTOR_STORAGE_AUTO);
    EXPECT_EQ(so.factor_opts.factor_storage, apxchol::factor_storage_type::automatic);
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
    auto o = cpu_options();
    o.struct_size = sizeof(apxchol_options) + 4;
    handle h;
    EXPECT_EQ(create(c, &o, h), APXCHOL_STATUS_UNSUPPORTED);
    ASSERT_EQ(create(c, nullptr, h), APXCHOL_STATUS_SUCCESS);
    const Eigen::VectorXd b = compatible_rhs(c.n);
    Eigen::VectorXd x(c.n);
    apxchol_stats stats{};
    stats.struct_size = sizeof(apxchol_stats) + 8;
    EXPECT_EQ(apxchol_solver_stats(h.s, &stats), APXCHOL_STATUS_UNSUPPORTED);
}

TEST(CApi, InfoArraySizeIsCheckedBeforeWritesAndLaterErrorsZeroEveryColumn) {
    const csc64 c = to_csc64(grid_laplacian(5, 5));
    handle h;
    ASSERT_EQ(create(c, nullptr, h), APXCHOL_STATUS_SUCCESS);
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(c.n, 5), x = Eigen::MatrixXd::Constant(c.n, 5, 7);
    const apxchol_const_dense_view rhs{b.data(), 1, c.n};
    const apxchol_dense_view out{x.data(), 1, c.n};
    std::array<apxchol_solve_info, 5> info;
    for (const std::size_t size : {sizeof(info[0]) - 1, sizeof(info[0]) + 1}) {
        std::memset(info.data(), 0xA5, sizeof info);
        const auto before = info;
        EXPECT_EQ(apxchol_solver_solve(h.s, 5, &rhs, nullptr, &out, tolerance, max_iterations,
            info.data(), size, nullptr, 0), size < sizeof(info[0])
                ? APXCHOL_STATUS_INVALID_ARGUMENT : APXCHOL_STATUS_UNSUPPORTED);
        EXPECT_TRUE(same_bytes(info.data(), before.data(), info.size()));
        EXPECT_TRUE((x.array() == 7).all());
    }
    b(c.n - 1, 4) = std::numeric_limits<double>::quiet_NaN();
    std::memset(info.data(), 0xA5, sizeof info);
    EXPECT_EQ(apxchol_solver_solve(h.s, 5, &rhs, nullptr, &out, tolerance, max_iterations,
        info.data(), sizeof(info[0]), nullptr, 0), APXCHOL_STATUS_INVALID_ARGUMENT);
    for (const auto& entry : info) {
        EXPECT_EQ(entry.iterations, 0);
        EXPECT_EQ(entry.relative_residual, 0);
        EXPECT_EQ(entry.converged, 0);
    }
    EXPECT_TRUE((x.array() == 7).all());
    b.setZero();
    EXPECT_EQ(apxchol_solver_solve(h.s, 5, &rhs, nullptr, &out, tolerance, max_iterations,
        nullptr, 999, nullptr, 0), APXCHOL_STATUS_SUCCESS);
}

TEST(CApi, FactorStorageReportsResolvedSolverStorageAndNativeFactorExport) {
    const Sparse A = grid_laplacian(12, 11, 0.25);
    const csc64 c = to_csc64(A);
    for (const auto backend : available_backends()) {
        auto o = cpu_options();
        o.backend = backend;
        o.threads = 1;
        o.keep_factor_values = 1;
        o.factor_storage = APXCHOL_FACTOR_STORAGE_FLOAT32;
        factor_handle factor;
        ASSERT_EQ(create_factor(c, &o, factor), APXCHOL_STATUS_SUCCESS);
        EXPECT_EQ(stats_of(factor.f).factor_storage, APXCHOL_FACTOR_STORAGE_FLOAT32);
        const exported native = export_of(factor.f);
        const Eigen::VectorXd b = compatible_rhs(c.n);
        for (const apxchol_factor_storage storage :
             {APXCHOL_FACTOR_STORAGE_FLOAT32, APXCHOL_FACTOR_STORAGE_FLOAT16}) {
            if (backend == APXCHOL_BACKEND_CPU && storage == APXCHOL_FACTOR_STORAGE_FLOAT16 &&
                !apxchol::omp_sptrsv::fp16_supported())
                continue;
            SCOPED_TRACE(::testing::Message() << "backend=" << backend << " storage=" << storage);
            o.factor_storage = storage;
            handle explicit_storage;
            ASSERT_EQ(create(c, &o, explicit_storage), APXCHOL_STATUS_SUCCESS);
            EXPECT_EQ(stats_of(explicit_storage.s).factor_storage, storage);
            const exported built = export_of(explicit_storage.s);
            if (backend == APXCHOL_BACKEND_CPU) {
                EXPECT_TRUE(built == native);
            }
            factor_handle copy;
            ASSERT_EQ(apxchol_solver_copy_factor(explicit_storage.s, &copy.f, nullptr, 0),
                      APXCHOL_STATUS_SUCCESS);
            EXPECT_EQ(stats_of(copy.f).factor_storage, APXCHOL_FACTOR_STORAGE_FLOAT32);
            EXPECT_TRUE(export_of(copy.f) == built);
            Eigen::VectorXd x(c.n);
            auto info = info_struct();
            ASSERT_EQ(solve_one(explicit_storage.s, b.data(), nullptr, x.data(), tolerance, max_iterations,
                               &info, nullptr, 0), APXCHOL_STATUS_SUCCESS);
            EXPECT_LT(true_residual(A, b, x), tolerance);
#if defined(APXCHOL_USE_CUDA)
            if (backend == APXCHOL_BACKEND_GPU) continue;
#endif
            handle adopted;
            ASSERT_EQ(adopt(c, factor.f, &o, adopted), APXCHOL_STATUS_SUCCESS);
            EXPECT_EQ(stats_of(adopted.s).factor_storage, storage);
            EXPECT_TRUE(export_of(adopted.s) == native);
            ASSERT_EQ(solve_one(adopted.s, b.data(), nullptr, x.data(), tolerance, max_iterations,
                                           &info, nullptr, 0),
                      APXCHOL_STATUS_SUCCESS);
            EXPECT_LT(true_residual(A, b, x), tolerance);
        }
#if defined(APXCHOL_USE_CUDA)
        if (backend == APXCHOL_BACKEND_GPU) continue;
#endif
        o.factor_storage = APXCHOL_FACTOR_STORAGE_AUTO;
        handle automatic;
        ASSERT_EQ(adopt(c, factor.f, &o, automatic), APXCHOL_STATUS_SUCCESS);
        EXPECT_EQ(stats_of(automatic.s).factor_storage, APXCHOL_FACTOR_STORAGE_FLOAT32);
        EXPECT_TRUE(export_of(automatic.s) == native);
    }
}

TEST(CApi, Fp16OnACpuWithoutItIsUnsupportedBeforeSetup) {
    const csc64 c = to_csc64(grid_laplacian(8, 7));
    if (apxchol::omp_sptrsv::fp16_supported())
        GTEST_SKIP() << "this CPU supports FP16 storage; UNSUPPORTED needs one without AVX/F16C";
    auto o = cpu_options();
    o.backend = APXCHOL_BACKEND_CPU;
    o.factor_storage = APXCHOL_FACTOR_STORAGE_FLOAT16;
    handle h;
    std::string message;
    EXPECT_EQ(create(c, &o, h, 0, &message), APXCHOL_STATUS_UNSUPPORTED);
    EXPECT_NE(message.find("FP16"), std::string::npos) << message;
}

TEST(CApi, SingleThreadFactorAndSolutionAreByteEqualToCppSolver) {
    const Sparse A = grid_laplacian(30, 30);
    const csc64 c = to_csc64(A, 1);
    auto o = cpu_options();
    o.threads = 1;
    o.keep_factor_values = 1;
    o.seed = 7;
    handle h;
    ASSERT_EQ(create(c, &o, h, 1), APXCHOL_STATUS_SUCCESS);

    apxchol::solve_options so;
    so.keep_factor_values = true;
    so.factor_opts.seed = 7;
#ifdef _OPENMP
    const scoped_omp_threads threads(1);
#endif
    const apxchol::cpu_solver ref(A, so);
    const Eigen::VectorXd b = compatible_rhs(c.n);
    const apxchol::solve_result r = ref.solve(b, tolerance, max_iterations);
    EXPECT_TRUE(export_of(h.s) == export_of(ref.preconditioner().factor()));

    Eigen::VectorXd x(c.n);
    auto info = info_struct();
    ASSERT_EQ(solve_one(h.s, b.data(), nullptr, x.data(), tolerance, max_iterations, &info, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_TRUE(same_bytes(x.data(), r.x.data(), x.size()));
    EXPECT_EQ(info.iterations, r.iterations);
    EXPECT_EQ(info.relative_residual, r.residual);
    EXPECT_LT(info.relative_residual, tolerance);
}

TEST(CApi, CpuResidualIsTheOriginalSystems) {
    for (const double shift : {0.0, 0.05}) {
        SCOPED_TRACE(::testing::Message() << "shift=" << shift);
        const Sparse A = grid_laplacian(16, 15, shift);
        const csc64 c = to_csc64(A);
        auto o = cpu_options();
        o.backend = APXCHOL_BACKEND_CPU;
        handle h;
        ASSERT_EQ(create(c, &o, h), APXCHOL_STATUS_SUCCESS);
        const Eigen::VectorXd b = compatible_rhs(c.n, 4);
        auto independent = [&](const Eigen::VectorXd& x) {
            return (b - A * x).stableNorm() / b.stableNorm();
        };
        Eigen::VectorXd x(c.n);
        auto info = info_struct();
        ASSERT_EQ(solve_one(h.s, b.data(), nullptr, x.data(), tolerance, max_iterations, &info,
                                       nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
        // Forming b - A x rounds by about eps * |A| |x| whatever the order of
        // the sums, so two correct evaluations agree to that bound.
        const double rounding = 32 * std::numeric_limits<double>::epsilon() *
                                (A.cwiseAbs() * x.cwiseAbs()).norm() / b.norm();
        EXPECT_NEAR(info.relative_residual, independent(x), rounding);
        // No iterations: x = 0 and the residual is exactly 1; b = 0 gives 0.
        auto none = info_struct();
        EXPECT_EQ(solve_one(h.s, b.data(), nullptr, x.data(), tolerance, 0, &none, nullptr,
                                       0),
                  APXCHOL_STATUS_NOT_CONVERGED);
        EXPECT_EQ(none.relative_residual, 1.0);
        const Eigen::VectorXd zero = Eigen::VectorXd::Zero(c.n);
        auto zero_info = info_struct();
        EXPECT_EQ(solve_one(h.s, zero.data(), nullptr, x.data(), tolerance, max_iterations, &zero_info,
                                       nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
        EXPECT_EQ(zero_info.relative_residual, 0.0);
        EXPECT_EQ(x.cwiseAbs().maxCoeff(), 0.0);
    }
}

TEST(CApi, RepeatedSolvesAreBitIdenticalAndWarmStartAtTheSolutionIsFree) {
    const csc64 c = to_csc64(grid_laplacian(20, 20, 0.01));
    handle h;
    ASSERT_EQ(create(c, nullptr, h), APXCHOL_STATUS_SUCCESS);
    const Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(c.n, -1.0, 3.0);
    Eigen::VectorXd x1(c.n), x2(c.n), x3(c.n);
    ASSERT_EQ(solve_one(h.s, b.data(), nullptr, x1.data(), tolerance, max_iterations, nullptr, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    ASSERT_EQ(solve_one(h.s, b.data(), nullptr, x2.data(), tolerance, max_iterations, nullptr, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_TRUE(same_bytes(x1.data(), x2.data(), x1.size()));
    auto info = info_struct();
    ASSERT_EQ(solve_one(h.s, b.data(), x1.data(), x3.data(), 1e-6, max_iterations, &info, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(info.iterations, 0);
    EXPECT_EQ(info.converged, 1);
}

TEST(CApi, CpuBatchMatchesNativeOwnerUsingTheExportedFactor) {
#ifdef _OPENMP
    const scoped_omp_threads threads(1);
#endif
    for (const double shift : {0.0, 0.05}) {
        const Sparse A = grid_laplacian(12, 11, shift);
        const csc64 c = to_csc64(A);
        auto o = cpu_options();
        o.threads = 1;
        o.keep_factor_values = 1;
        factor_handle f;
        ASSERT_EQ(create_factor(c, &o, f), APXCHOL_STATUS_SUCCESS);
        handle h;
        ASSERT_EQ(adopt(c, f.f, &o, h), APXCHOL_STATUS_SUCCESS);
        const exported bytes = export_of(f.f);
        const auto stats = stats_of(f.f);
        apxchol::factorization captured;
        captured.L.n_ = static_cast<apxchol::node_index>(c.n);
        captured.L.outer_.assign(bytes.colptr.begin(), bytes.colptr.end());
        captured.L.inner_.assign(bytes.rowval.begin(), bytes.rowval.end());
        captured.L.vals_ = bytes.nzval;
        captured.perm.assign(bytes.perm.begin(), bytes.perm.end());
        captured.sddm = stats.sddm;
        captured.lumped_offdiag = stats.lumped_offdiag;
        apxchol::solve_options native_options;
        native_options.backend = apxchol::solve_backend::cpu;
        native_options.num_threads = 1;
        native_options.keep_factor_values = true;
        const apxchol::cpu_solver native(A, std::move(captured), native_options);
        ASSERT_TRUE(export_of(native.preconditioner().factor()) == bytes);
        for (const int k : {1, 2, 4, 5, 16}) {
            Eigen::MatrixXd B(c.n, k), X(c.n, k), expected(c.n, k);
            for (int col = 0; col < k; ++col) B.col(col) = compatible_rhs(c.n, col + 1);
            if (k > 1) B.col(0).setZero();
            const Eigen::MatrixXd guess = 0.25 * B;
            const apxchol::const_matrix_view guess_view(guess);
            const apxchol_const_dense_view rhs{B.data(), 1, c.n}, initial{guess.data(), 1, c.n};
            const apxchol_dense_view out{X.data(), 1, c.n};
            for (const bool warm : {false, true}) {
                for (const int budget : {max_iterations, 0, 1}) {
                    SCOPED_TRACE(::testing::Message() << "shift=" << shift << " nrhs=" << k
                        << " warm=" << warm << " max_iter=" << budget);
                    const auto reference = native.solve(B, expected, tolerance, budget,
                                                        warm ? &guess_view : nullptr);
                    std::vector<apxchol_solve_info> info(k);
                    const bool converged = std::all_of(reference.columns.begin(),
                        reference.columns.end(), [](const auto& r) { return r.converged; });
                    ASSERT_EQ(apxchol_solver_solve(h.s, k, &rhs, warm ? &initial : nullptr, &out, tolerance, budget, info.data(), sizeof(info[0]), nullptr, 0),
                        converged ? APXCHOL_STATUS_SUCCESS : APXCHOL_STATUS_NOT_CONVERGED);
                    EXPECT_TRUE(same_bytes(X.data(), expected.data(), X.size()));
                    for (int col = 0; col < k; ++col) {
                        EXPECT_EQ(info[col].iterations, reference.columns[col].iterations);
                        EXPECT_EQ(info[col].relative_residual, reference.columns[col].residual);
                        EXPECT_EQ(info[col].converged, reference.columns[col].converged);
                    }
                }
            }
            native.apply(B, expected);
            ASSERT_EQ(apxchol_solver_apply(h.s, k, &rhs, &out, nullptr, 0),
                      APXCHOL_STATUS_SUCCESS);
            EXPECT_TRUE(same_bytes(X.data(), expected.data(), X.size()));
        }
        EXPECT_TRUE(export_of(h.s) == bytes);
        EXPECT_TRUE(export_of(f.f) == bytes);
    }
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

TEST(CApi, ApplyIsStatelessAndCentresEveryLaplacianApplication) {
    for (const auto backend : available_backends()) {
        for (const double shift : {0.0, 0.1}) {
            SCOPED_TRACE(::testing::Message() << "backend=" << backend << " shift=" << shift);
            const csc64 c = to_csc64(grid_laplacian(12, 12, shift));
            auto o = cpu_options();
            o.backend = backend;
            handle h;
            ASSERT_EQ(create(c, &o, h), APXCHOL_STATUS_SUCCESS);
            const Eigen::VectorXd r = compatible_rhs(c.n, 3).array() + 0.25;
            Eigen::VectorXd z0(c.n), z(c.n);
            ASSERT_EQ(apply_one(h.s, r.data(), z0.data(), nullptr, 0),
                      APXCHOL_STATUS_SUCCESS);
            if (shift == 0.0) {
                const double rounding = backend == APXCHOL_BACKEND_CPU ? 1e-12 : 1e-5;
                EXPECT_LE(std::fabs(z0.mean()), rounding * z0.cwiseAbs().maxCoeff());
            }
            for (int i = 0; i < 12; ++i) {
                ASSERT_EQ(apply_one(h.s, r.data(), z.data(), nullptr, 0),
                          APXCHOL_STATUS_SUCCESS);
                EXPECT_TRUE(same_bytes(z.data(), z0.data(), z.size())) << i;
            }
        }
    }
}

TEST(CApi, ApplyMatchesCppApply) {
    const Sparse A = grid_laplacian(10, 10, 0.1);
    const csc64 c = to_csc64(A);
    auto o = cpu_options();
    o.threads = 1;
    handle h;
    ASSERT_EQ(create(c, &o, h), APXCHOL_STATUS_SUCCESS);
#ifdef _OPENMP
    const scoped_omp_threads threads(1);
#endif
    const apxchol::cpu_solver ref(A, apxchol::solve_options{});
    const Eigen::VectorXd r = compatible_rhs(c.n, 3);
    const Eigen::VectorXd z_ref = ref.apply(r);
    Eigen::VectorXd z(c.n);
    ASSERT_EQ(apply_one(h.s, r.data(), z.data(), nullptr, 0), APXCHOL_STATUS_SUCCESS);
    EXPECT_TRUE(same_bytes(z.data(), z_ref.data(), z.size()));
}

TEST(CApi, BatchValidatesEveryInputColumnBeforeWritingSolutionsOrApplications) {
    const csc64 c = to_csc64(grid_laplacian(10, 10));
    for (const auto backend : available_backends()) {
        auto o = cpu_options(); o.backend = backend;
        handle h;
        ASSERT_EQ(create(c, &o, h), APXCHOL_STATUS_SUCCESS);
        Eigen::MatrixXd B(c.n, 5);
        for (int col = 0; col < 5; ++col) B.col(col) = compatible_rhs(c.n, col + 1);
        for (const bool in_guess : {false, true}) {
            Eigen::MatrixXd b = B, x0 = Eigen::MatrixXd::Zero(c.n, 5);
            (in_guess ? x0 : b)(c.n - 1, 4) = std::numeric_limits<double>::quiet_NaN();
            Eigen::MatrixXd x = Eigen::MatrixXd::Constant(c.n, 5, 7);
            const apxchol_const_dense_view rhs{b.data(), 1, c.n}, initial{x0.data(), 1, c.n};
            const apxchol_dense_view out{x.data(), 1, c.n};
            EXPECT_EQ(apxchol_solver_solve(h.s, 5, &rhs, &initial, &out, tolerance, max_iterations,
                nullptr, 0, nullptr, 0), APXCHOL_STATUS_INVALID_ARGUMENT);
            EXPECT_TRUE((x.array() == 7).all());
            if (!in_guess) {
                EXPECT_EQ(apxchol_solver_apply(h.s, 5, &rhs, &out, nullptr, 0),
                          APXCHOL_STATUS_INVALID_ARGUMENT);
                EXPECT_TRUE((x.array() == 7).all());
            }
        }
    }
}

TEST(CApi, EmptyNegativeAndOverflowingBatchDimensions) {
    const csc64 c = to_csc64(grid_laplacian(4, 4));
    handle h;
    ASSERT_EQ(create(c, nullptr, h), APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(apxchol_solver_solve(h.s, 0, nullptr, nullptr, nullptr, tolerance, max_iterations,
        nullptr, 999, nullptr, 0), APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(apxchol_solver_apply(h.s, 0, nullptr, nullptr, nullptr, 0), APXCHOL_STATUS_SUCCESS);
    const auto overflow = std::numeric_limits<std::ptrdiff_t>::max() /
        static_cast<std::int64_t>(sizeof(double)) / c.n + 1;
    double b = 1, x = 7;
    apxchol_const_dense_view rhs{&b, 1, c.n};
    apxchol_dense_view out{&x, 1, c.n};
    for (const std::int64_t k : {std::int64_t{-1}, overflow}) {
        EXPECT_EQ(apxchol_solver_solve(h.s, k, &rhs, nullptr, &out, tolerance, max_iterations, nullptr, 0, nullptr, 0),
                  APXCHOL_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(apxchol_solver_apply(h.s, k, &rhs, &out, nullptr, 0), APXCHOL_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(x, 7);
    }

    // A scalar's dense array fits while its larger info array overflows.
    const csc64 scalar{1, {0, 1}, {0}, {2}};
    handle one;
    ASSERT_EQ(create(scalar, nullptr, one), APXCHOL_STATUS_SUCCESS);
    const std::int64_t info_overflow = std::numeric_limits<std::ptrdiff_t>::max() /
        static_cast<std::int64_t>(sizeof(apxchol_solve_info)) + 1;
    ASSERT_LE(info_overflow, std::numeric_limits<std::ptrdiff_t>::max() /
        static_cast<std::int64_t>(sizeof(double)));
    rhs.column_stride = out.column_stride = 1;
    apxchol_solve_info info{17, 0.25, 1};
    const auto before = info;
    std::array<char, 256> error{};
    EXPECT_EQ(apxchol_solver_solve(one.s, info_overflow, &rhs, nullptr, &out, tolerance, max_iterations,
        &info, sizeof(info), error.data(), error.size()), APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_NE(std::string(error.data()).find("nrhs * sizeof(apxchol_solve_info)"),
              std::string::npos) << error.data();
    EXPECT_TRUE(same_bytes(&info, &before, 1));
    EXPECT_EQ(x, 7);
}

TEST(CApi, ExtremeScaleRightHandSidesAreSolvedAsTheirScaledProblem) {
    // The CPU PCG's unscaled sums of squares underflow for |b| ~ 2^-700 (it
    // used to report x = 0 as converged) and overflow for 2^700. Power-of-two
    // scaling is exact, so the solution must be the scaled reference bitwise.
    for (const double shift : {0.0, 0.05}) {
        SCOPED_TRACE(::testing::Message() << "shift=" << shift);
        const csc64 c = to_csc64(grid_laplacian(12, 12, shift));
        auto o = cpu_options();
        o.backend = APXCHOL_BACKEND_CPU;
        o.threads = 1;
        handle h;
        ASSERT_EQ(create(c, &o, h), APXCHOL_STATUS_SUCCESS);
        const Eigen::VectorXd b = compatible_rhs(c.n, 2);
        Eigen::VectorXd x_ref(c.n);
        auto ref = info_struct();
        ASSERT_EQ(solve_one(h.s, b.data(), nullptr, x_ref.data(), tolerance, max_iterations, &ref,
                                       nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
        for (const int e : {-700, 700}) {
            SCOPED_TRACE(::testing::Message() << "scale=2^" << e);
            const Eigen::VectorXd bs = b.unaryExpr([e](double v) { return std::ldexp(v, e); });
            const Eigen::VectorXd xs = x_ref.unaryExpr([e](double v) { return std::ldexp(v, e); });
            Eigen::VectorXd x(c.n);
            auto info = info_struct();
            ASSERT_EQ(solve_one(h.s, bs.data(), nullptr, x.data(), tolerance, max_iterations, &info,
                                           nullptr, 0),
                      APXCHOL_STATUS_SUCCESS);
            EXPECT_TRUE(same_bytes(x.data(), xs.data(), x.size()));
            EXPECT_EQ(info.iterations, ref.iterations);
            // Extreme scales use stable norms for the certified residual,
            // whose reduction can round differently from the ordinary path.
            EXPECT_DOUBLE_EQ(info.relative_residual, ref.relative_residual);
            // No iterations: the residual of x = 0 is exactly 1 at any scale.
            info = info_struct();
            EXPECT_EQ(solve_one(h.s, bs.data(), nullptr, x.data(), tolerance, 0, &info,
                                           nullptr, 0),
                      APXCHOL_STATUS_NOT_CONVERGED);
            EXPECT_EQ(info.relative_residual, 1.0);
        }
        // Mixed scales in one block: every column is its own scaled problem.
        Eigen::MatrixXd B(c.n, 3), X(c.n, 3);
        B.col(0) = b.unaryExpr([](double v) { return std::ldexp(v, -700); });
        B.col(1) = b;
        B.col(2) = b.unaryExpr([](double v) { return std::ldexp(v, 700); });
        const apxchol_const_dense_view rhs{B.data(), 1, B.rows()};
        const apxchol_dense_view out{X.data(), 1, X.rows()};
        ASSERT_EQ(apxchol_solver_solve(h.s, 3, &rhs, nullptr, &out, tolerance, max_iterations,
                                      nullptr, 0, nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
        for (int col = 0; col < 3; ++col) {
            const int e = (col - 1) * 700;
            EXPECT_LE((X.col(col) / std::ldexp(1.0, e) - x_ref).norm(), 1e-6 * x_ref.norm()) << col;
        }
    }
}

TEST(CApi, SubnormalRightHandSideAndWarmStartRange) {
    const Sparse A = grid_laplacian(9, 9, 0.1);
    const csc64 c = to_csc64(A);
    auto o = cpu_options();
    o.backend = APXCHOL_BACKEND_CPU;
    handle h;
    ASSERT_EQ(create(c, &o, h), APXCHOL_STATUS_SUCCESS);
    const Eigen::VectorXd b =
        compatible_rhs(c.n, 3).unaryExpr([](double v) { return std::ldexp(v, -1060); });
    ASSERT_GT(b.cwiseAbs().maxCoeff(), 0.0);
    ASSERT_LT(b.cwiseAbs().maxCoeff(), std::numeric_limits<double>::min());
    Eigen::VectorXd x(c.n);
    auto info = info_struct();
    // The solve runs at unit scale; the subnormal x it returns keeps ~14
    // significant bits, which the certified residual reports honestly.
    const apxchol_status status =
        solve_one(h.s, b.data(), nullptr, x.data(), tolerance, max_iterations, &info, nullptr, 0);
    EXPECT_EQ(status == APXCHOL_STATUS_SUCCESS, info.relative_residual < tolerance);
    EXPECT_GT(info.iterations, 0);
    EXPECT_GT(x.cwiseAbs().maxCoeff(), 0.0);
    EXPECT_LT(info.relative_residual, 1e-3);

    const Eigen::VectorXd normal = compatible_rhs(c.n, 3);
    const Eigen::VectorXd huge_guess = Eigen::VectorXd::Constant(c.n, 0x1p300);
    std::array<char, 256> err{};
    info = info_struct();
    const auto warm_status = solve_one(h.s, normal.data(), huge_guess.data(), x.data(),
                                                 tolerance, max_iterations, &info, err.data(), err.size());
    EXPECT_EQ(warm_status, info.relative_residual < tolerance ? APXCHOL_STATUS_SUCCESS
                                                         : APXCHOL_STATUS_NOT_CONVERGED);
    EXPECT_TRUE(x.allFinite());
    const double independent = true_residual(A, normal, x);
    ASSERT_TRUE(std::isfinite(independent));
    EXPECT_TRUE(std::isfinite(info.relative_residual));
    EXPECT_NEAR(info.relative_residual, independent, independent * 1e-12);

    const Eigen::VectorXd tiny = 1e-200 * normal;
    const Eigen::VectorXd unscalable_guess = 1e200 * normal;
    EXPECT_EQ(solve_one(h.s, tiny.data(), unscalable_guess.data(), x.data(), tolerance,
                                   max_iterations, nullptr, err.data(), err.size()),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_NE(std::string(err.data()).find("scaling range"), std::string::npos) << err.data();
}

TEST(CApi, UnrepresentableSolutionsAreUnsupported) {
    for (const double diagonal : {1e-30, 1e30}) {
        SCOPED_TRACE(diagonal);
        Sparse A(4, 4);
        A.setIdentity();
        A *= diagonal;
        const csc64 c = to_csc64(A);
        auto o = cpu_options();
        o.backend = APXCHOL_BACKEND_CPU;
        handle h;
        ASSERT_EQ(create(c, &o, h), APXCHOL_STATUS_SUCCESS);
        const Eigen::VectorXd b = Eigen::VectorXd::Constant(4, diagonal < 1 ? 1e300 : 1e-300);
        Eigen::VectorXd x(4);
        auto info = info_struct();
        std::array<char, 256> err{};
        EXPECT_EQ(solve_one(h.s, b.data(), nullptr, x.data(), tolerance, max_iterations,
                                       &info, err.data(), err.size()),
                  APXCHOL_STATUS_UNSUPPORTED);
        EXPECT_EQ(info.converged, 0);
        EXPECT_EQ(info.iterations, 0);
        EXPECT_GT(std::strlen(err.data()), 0u);

        Eigen::MatrixXd B(4, 2), X(4, 2);
        B.col(0).setConstant(diagonal);
        B.col(1) = b;
        const apxchol_const_dense_view rhs{B.data(), 1, B.rows()};
        const apxchol_dense_view out{X.data(), 1, X.rows()};
        EXPECT_EQ(apxchol_solver_solve(h.s, 2, &rhs, nullptr, &out, tolerance, max_iterations,
                                      nullptr, 0, err.data(), err.size()),
                  APXCHOL_STATUS_UNSUPPORTED);
        EXPECT_GT(std::strlen(err.data()), 0u);
        ASSERT_EQ(solve_one(h.s, B.col(0).data(), nullptr, x.data(), tolerance, max_iterations,
                                       &info, err.data(), err.size()),
                  APXCHOL_STATUS_SUCCESS);
        EXPECT_TRUE(x.isApprox(Eigen::VectorXd::Ones(4), tolerance));
    }
}

TEST(CApi, StatsReportLumpingSddmAndFactorSize) {
    // SDDM with one mirrored positive off-diagonal pair: lumpable.
    Sparse A = grid_laplacian(4, 4, 0.5);
    A.coeffRef(1, 0) = -A.coeffRef(1, 0);
    A.coeffRef(0, 1) = -A.coeffRef(0, 1);
    A.makeCompressed();
    const csc64 c = to_csc64(A);
    auto o = cpu_options();
    o.keep_factor_values = 1;
    handle h;
    ASSERT_EQ(create(c, &o, h), APXCHOL_STATUS_SUCCESS);
    apxchol_stats stats{};
    stats.struct_size = sizeof stats;
    ASSERT_EQ(apxchol_solver_stats(h.s, &stats), APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(stats.backend, APXCHOL_BACKEND_CPU);
    EXPECT_EQ(stats.n, c.n);
    EXPECT_EQ(stats.lumped_offdiag, 2);
    EXPECT_EQ(stats.sddm, 1);
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
    std::vector<float> nzval(stats.factor_nnz);
    EXPECT_EQ(apxchol_solver_export_factor(released.s, 1, colptr.data(), rowval.data(), nzval.data(),
                                           nullptr, nullptr, 0),
              APXCHOL_STATUS_NO_FACTOR_VALUES);
    EXPECT_EQ(apxchol_solver_export_factor(released.s, 1, colptr.data(), nullptr, nzval.data(),
                                           nullptr, nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);

    auto o = cpu_options();
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
#if defined(APXCHOL_USE_CUDA)
    using apxchol::detail::check_cuda;
    EXPECT_NO_THROW(check_cuda(cudaSuccess, "probe"));
    try {
        check_cuda(cudaErrorMemoryAllocation, "probe");
        FAIL() << "CUDA memory exhaustion did not throw";
    } catch (const std::bad_alloc& error) {
        EXPECT_EQ(error.what(), std::string("probe: ") + cudaGetErrorString(cudaErrorMemoryAllocation));
    }
    try {
        check_cuda(cudaErrorInvalidValue, "probe");
        FAIL() << "CUDA invalid argument did not throw";
    } catch (const std::runtime_error& error) {
        EXPECT_EQ(error.what(), std::string("probe: ") + cudaGetErrorString(cudaErrorInvalidValue));
    }
#endif
    const csc64 c = to_csc64(grid_laplacian(4, 4));
    EXPECT_EQ(apxchol_backend_available(APXCHOL_BACKEND_CPU), 1);
    EXPECT_EQ(apxchol_backend_available(APXCHOL_BACKEND_AUTO), 1);
    EXPECT_EQ(apxchol_backend_available(42), 0);
    if (!apxchol_backend_available(APXCHOL_BACKEND_GPU)) {
        auto o = cpu_options(); o.backend = APXCHOL_BACKEND_GPU;
        handle h;
        EXPECT_EQ(create(c, &o, h), APXCHOL_STATUS_UNSUPPORTED);
        EXPECT_EQ(h.s, nullptr);
    }
    auto check = [&](auto mutate) {
        auto o = cpu_options();
        mutate(o);
        handle h;
        EXPECT_EQ(create(c, &o, h), APXCHOL_STATUS_INVALID_ARGUMENT);
    };
    check([](apxchol_options& o) { o.backend = 9; });
    check([](apxchol_options& o) { o.backend = 3; });
    check([](apxchol_options& o) { o.factor_storage = 1; });
    check([](apxchol_options& o) { o.factor_storage = 64; });
    check([](apxchol_options& o) { o.keep_factor_values = 2; });
    check([](apxchol_options& o) { o.threads = -1; });
}

TEST(CApi, AutomaticScalarAndHostFactorAdoptionSelectCpu) {
    const csc64 scalar{1, {0, 1}, {0}, {2}};
    auto o = defaults();
    handle tiny;
    const auto matrix = scalar.view();
    ASSERT_EQ(apxchol_solver_create(&matrix, nullptr, nullptr, &tiny.s, nullptr, 0), APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(stats_of(tiny.s).backend, APXCHOL_BACKEND_CPU);
    const csc64 c = to_csc64(grid_laplacian(12, 11, 0.05));
    factor_handle f;
    ASSERT_EQ(create_factor(c, &o, f), APXCHOL_STATUS_SUCCESS);
    handle automatic;
    ASSERT_EQ(adopt(c, f.f, &o, automatic), APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(stats_of(automatic.s).backend, APXCHOL_BACKEND_CPU);
#if defined(APXCHOL_USE_CUDA)
    o.backend = APXCHOL_BACKEND_GPU;
    handle refused;
    EXPECT_EQ(adopt(c, f.f, &o, refused), APXCHOL_STATUS_UNSUPPORTED);
    EXPECT_EQ(refused.s, nullptr);
#endif
}

TEST(CApi, FactorHandleExportEqualsTheSolversBytewise) {
    // Single thread: the factorization is exactly repeatable (docs/precision.md).
    for (const double shift : {0.0, 0.05}) {
        const csc64 c = to_csc64(grid_laplacian(17, 13, shift), 1);
        auto o = cpu_options();
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
        auto o = cpu_options();
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

        const Eigen::VectorXd b = compatible_rhs(c.n, 3);
        Eigen::VectorXd x1(c.n), x2(c.n), x0 = 0.5 * compatible_rhs(c.n, 7);
        const Eigen::VectorXd& x0c = x0;
        for (const double* guess : {static_cast<const double*>(nullptr), x0c.data()}) {
            auto i1 = info_struct(), i2 = info_struct();
            ASSERT_EQ(solve_one(self.s, b.data(), guess, x1.data(), 1e-10, max_iterations, &i1,
                                           nullptr, 0),
                      APXCHOL_STATUS_SUCCESS);
            ASSERT_EQ(solve_one(adopted.s, b.data(), guess, x2.data(), 1e-10, max_iterations, &i2,
                                           nullptr, 0),
                      APXCHOL_STATUS_SUCCESS);
            EXPECT_TRUE(same_bytes(x1.data(), x2.data(), x1.size()));
            EXPECT_EQ(i1.iterations, i2.iterations);
            EXPECT_EQ(i1.relative_residual, i2.relative_residual);
        }
        Eigen::VectorXd z1(c.n), z2(c.n);
        ASSERT_EQ(apply_one(self.s, b.data(), z1.data(), nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
        ASSERT_EQ(apply_one(adopted.s, b.data(), z2.data(), nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
        EXPECT_TRUE(same_bytes(z1.data(), z2.data(), z1.size()));
    }
}

TEST(CApi, SolverFromFactorPreconditionsANearbyOperator) {
    constexpr double strict_tolerance = 1e-10;
    constexpr int large_budget = 1000;
    for (const auto backend : adoption_backends()) {
        for (const double shift : {0.0, 0.05}) {
            SCOPED_TRACE(::testing::Message() << "backend=" << backend << " shift=" << shift);
            const Sparse A = grid_laplacian(26, 20, shift);
            const Sparse B = nearby_operator(A, 7, 0.5);
            ASSERT_GT(B.nonZeros(), A.nonZeros());
            auto o = cpu_options();
            factor_handle f;
            ASSERT_EQ(create_factor(to_csc64(A), &o, f), APXCHOL_STATUS_SUCCESS);
            o.backend = backend;
            const csc64 cb = to_csc64(B, 1);
            handle stale;
            std::string message;
            ASSERT_EQ(adopt(cb, f.f, &o, stale, 1, &message), APXCHOL_STATUS_SUCCESS) << message;
            expect_same_factor_stats(stats_of(stale.s), stats_of(f.f));

            const Eigen::VectorXd b = compatible_rhs(cb.n, 4);
            Eigen::VectorXd x(cb.n);
            auto info = info_struct();
            ASSERT_EQ(solve_one(stale.s, b.data(), nullptr, x.data(), strict_tolerance, large_budget,
                                           &info, nullptr, 0),
                      APXCHOL_STATUS_SUCCESS);
            EXPECT_EQ(info.converged, 1);
            const double res = true_residual(B, b, x);
            EXPECT_LT(res, strict_tolerance);
            EXPECT_NEAR(info.relative_residual, res, 1e-3 * res);
            if (shift == 0.0) {
                EXPECT_LE(std::fabs(x.mean()), 1e-12 * x.norm());
            }

            // The same solution as a solver that factorized B itself.
            handle fresh;
            ASSERT_EQ(create(cb, &o, fresh, 1), APXCHOL_STATUS_SUCCESS);
            Eigen::VectorXd y(cb.n);
            ASSERT_EQ(solve_one(fresh.s, b.data(), nullptr, y.data(), strict_tolerance, large_budget,
                                           nullptr, nullptr, 0),
                      APXCHOL_STATUS_SUCCESS);
            EXPECT_LT((x - y).norm(), 1e-6 * y.norm());
        }
    }
}

TEST(CApi, SddmFactorMayPreconditionALaplacianButNotConversely) {
    constexpr double strict_tolerance = 1e-10;
    constexpr int large_budget = 1000;
    const Sparse lap = grid_laplacian(18, 18), sddm = grid_laplacian(18, 18, 0.02);
    auto o = cpu_options();
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
    ASSERT_EQ(solve_one(h.s, b.data(), nullptr, x.data(), strict_tolerance, large_budget, nullptr, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_LT(true_residual(lap, b, x), strict_tolerance);
}

TEST(CApi, FactorHandleArgumentErrors) {
    const csc64 c = to_csc64(grid_laplacian(6, 5));
    std::string message;
    apxchol_factor_destroy(nullptr);

    const auto matrix = c.view();
    // Creation: NULL outputs, CSC, options and the operator contract.
    EXPECT_EQ(apxchol_factor_create(&matrix,
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
        auto o = cpu_options();
        o.backend = 9;
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
    std::vector<float> nzval(st.factor_nnz);
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

    // Adoption: NULL output, the dimension, the operator contract.
    handle h;
    EXPECT_EQ(apxchol_solver_create(&matrix, f.f, nullptr, nullptr,
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
    auto o = cpu_options();
    o.struct_size = sizeof(apxchol_options) + 4;
    EXPECT_EQ(adopt(c, f.f, &o, h), APXCHOL_STATUS_UNSUPPORTED);
    o = defaults();
    o.backend = 9;
    EXPECT_EQ(adopt(c, f.f, &o, h), APXCHOL_STATUS_INVALID_ARGUMENT);
#if !defined(APXCHOL_USE_METAL) && !defined(APXCHOL_USE_CUDA)
    o = defaults();
    o.backend = APXCHOL_BACKEND_GPU;
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
    auto o = cpu_options();
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
        ASSERT_EQ(solve_one(s.s, b.data(), nullptr, x_first.data(), tolerance, max_iterations, nullptr,
                                       nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
    }  // the solver is gone; the factor is as it was
    EXPECT_TRUE(export_of(f.f) == before);
    const apxchol_stats again = stats_of(f.f);
    expect_same_factor_stats(again, fs);

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
        ASSERT_EQ(solve_one(s.s, b.data(), nullptr, x.data(), tolerance, max_iterations, nullptr,
                                       nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
        EXPECT_TRUE(same_bytes(x.data(), x_first.data(), x.size()));
        EXPECT_TRUE(export_of(s.s) == before);
    }
}

TEST(CApi, SolverFactorRoundTrip) {
    const csc64 c = to_csc64(grid_laplacian(16, 21, 0.03));
    auto o = cpu_options();
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
        ASSERT_EQ(solve_one(s.s, b.data(), nullptr, x_original.data(), tolerance, max_iterations,
                                       nullptr, nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
    }  // the copy outlives the solver it came from
    EXPECT_TRUE(export_of(copy.f) == original);
    const apxchol_stats cs = stats_of(copy.f);
    expect_same_factor_stats(cs, solver_stats);
    EXPECT_EQ(cs.backend, APXCHOL_BACKEND_CPU);

    // ... and a solver adopting it solves as the original did.
    handle again;
    ASSERT_EQ(adopt(c, copy.f, &o, again), APXCHOL_STATUS_SUCCESS);
    Eigen::VectorXd x(c.n);
    ASSERT_EQ(solve_one(again.s, b.data(), nullptr, x.data(), tolerance, max_iterations, nullptr,
                                   nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_TRUE(same_bytes(x.data(), x_original.data(), x.size()));
    factor_handle twice;
    ASSERT_EQ(apxchol_solver_copy_factor(again.s, &twice.f, nullptr, 0), APXCHOL_STATUS_SUCCESS);
    EXPECT_TRUE(export_of(twice.f) == original);
}

TEST(CApi, GpuBatchWidthsReportOriginalResidualsAndIndependentConvergence) {
    if (!apxchol_backend_available(APXCHOL_BACKEND_GPU)) GTEST_SKIP() << "no usable GPU backend";
    for (const double shift : {0.0, 0.05}) {
        const Sparse A = grid_laplacian(14, 13, shift);
        const csc64 c = to_csc64(A, 1);
        auto o = cpu_options();
        o.backend = APXCHOL_BACKEND_GPU;
        handle h;
        ASSERT_EQ(create(c, &o, h, 1), APXCHOL_STATUS_SUCCESS);
        EXPECT_EQ(stats_of(h.s).backend, APXCHOL_BACKEND_GPU);
        for (const int k : {1, 2, 4, 5, 16}) {
            SCOPED_TRACE(::testing::Message() << "shift=" << shift << " nrhs=" << k);
            Eigen::MatrixXd B(c.n, k), X(c.n, k), warm(c.n, k), Z(c.n, k), repeat(c.n, k);
            for (int col = 0; col < k; ++col) B.col(col) = compatible_rhs(c.n, col + 2);
            if (k > 1) B.col(0).setZero();
            std::vector<apxchol_solve_info> info(k);
            const apxchol_const_dense_view rhs{B.data(), 1, c.n}, initial{X.data(), 1, c.n};
            const apxchol_dense_view out{X.data(), 1, c.n}, warm_view{warm.data(), 1, c.n},
                applied{Z.data(), 1, c.n}, repeated{repeat.data(), 1, c.n};
            ASSERT_EQ(apxchol_solver_solve(h.s, k, &rhs, nullptr, &out, tolerance, max_iterations,
                info.data(), sizeof(info[0]), nullptr, 0), APXCHOL_STATUS_SUCCESS);
            for (int col = 0; col < k; ++col) {
                const double norm = B.col(col).stableNorm();
                const double measured = norm ? (B.col(col) - A * X.col(col)).stableNorm() / norm : 0;
                EXPECT_LT(measured, tolerance);
                EXPECT_EQ(info[col].converged, 1);
                const double rounding = norm ? 32 * std::numeric_limits<double>::epsilon() *
                    (A.cwiseAbs() * X.col(col).cwiseAbs()).norm() / norm : 0;
                EXPECT_NEAR(info[col].relative_residual, measured, rounding);
                if (shift == 0) {
                    EXPECT_LE(std::fabs(X.col(col).mean()), 1e-12 * X.col(col).norm());
                }
            }
            ASSERT_EQ(apxchol_solver_solve(h.s, k, &rhs, &initial, &warm_view, 1e-6, max_iterations,
                info.data(), sizeof(info[0]), nullptr, 0), APXCHOL_STATUS_SUCCESS);
            for (const auto& result : info) EXPECT_EQ(result.iterations, 0);
            ASSERT_EQ(apxchol_solver_apply(h.s, k, &rhs, &applied, nullptr, 0),
                      APXCHOL_STATUS_SUCCESS);
            ASSERT_EQ(apxchol_solver_apply(h.s, k, &rhs, &repeated, nullptr, 0),
                      APXCHOL_STATUS_SUCCESS);
            EXPECT_TRUE(Z.allFinite());
            EXPECT_TRUE(same_bytes(Z.data(), repeat.data(), Z.size()));
            ASSERT_EQ(apxchol_solver_solve(h.s, k, &rhs, nullptr, &out, tolerance, 0,
                info.data(), sizeof(info[0]), nullptr, 0), APXCHOL_STATUS_NOT_CONVERGED);
            for (int col = 0; col < k; ++col) {
                const bool zero = k > 1 && col == 0;
                EXPECT_EQ(info[col].iterations, 0);
                EXPECT_EQ(info[col].converged, zero);
                EXPECT_DOUBLE_EQ(info[col].relative_residual, zero ? 0 : 1);
            }
        }
        const Eigen::VectorXd b = compatible_rhs(c.n, 2);
        Eigen::VectorXd reference(c.n), scaled(c.n);
        ASSERT_EQ(solve_one(h.s, b.data(), nullptr, reference.data(), tolerance, max_iterations, nullptr, nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
        for (const int exponent : {-700, 700}) {
            const Eigen::VectorXd rhs = b.unaryExpr([=](double v) { return std::ldexp(v, exponent); });
            ASSERT_EQ(solve_one(h.s, rhs.data(), nullptr, scaled.data(), tolerance, max_iterations, nullptr, nullptr, 0),
                      APXCHOL_STATUS_SUCCESS);
            scaled = scaled.unaryExpr([=](double v) { return std::ldexp(v, -exponent); });
            EXPECT_LE((scaled - reference).norm(), 1e-9 * reference.norm());
        }
    }
}

TEST(CApi, GpuFactorExportAndCopyCanBeReusedOnCpu) {
    if (!apxchol_backend_available(APXCHOL_BACKEND_GPU)) GTEST_SKIP() << "no usable GPU backend";
    const Sparse A = grid_laplacian(12, 15);
    const csc64 c = to_csc64(A);
    auto o = cpu_options();
    o.backend = APXCHOL_BACKEND_GPU;
    o.keep_factor_values = 1;
    handle gpu;
    ASSERT_EQ(create(c, &o, gpu), APXCHOL_STATUS_SUCCESS);
    factor_handle copy;
    ASSERT_EQ(apxchol_solver_copy_factor(gpu.s, &copy.f, nullptr, 0), APXCHOL_STATUS_SUCCESS);
    const exported original = export_of(gpu.s);
    EXPECT_TRUE(export_of(copy.f) == original);
    expect_same_factor_stats(stats_of(copy.f), stats_of(gpu.s));
    EXPECT_EQ(stats_of(copy.f).backend, APXCHOL_BACKEND_CPU);
    o.backend = APXCHOL_BACKEND_CPU;
    handle cpu;
    ASSERT_EQ(adopt(c, copy.f, &o, cpu), APXCHOL_STATUS_SUCCESS);
    EXPECT_TRUE(export_of(cpu.s) == original);
    const Eigen::VectorXd b = compatible_rhs(c.n, 5);
    Eigen::VectorXd x(c.n);
    ASSERT_EQ(solve_one(cpu.s, b.data(), nullptr, x.data(), tolerance, max_iterations, nullptr, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_LT(true_residual(A, b, x), tolerance);
#if defined(APXCHOL_USE_METAL)
    o.backend = APXCHOL_BACKEND_GPU;
    handle gpu_again;
    ASSERT_EQ(adopt(c, copy.f, &o, gpu_again), APXCHOL_STATUS_SUCCESS);
    EXPECT_TRUE(export_of(gpu_again.s) == original);
    ASSERT_EQ(solve_one(gpu_again.s, b.data(), nullptr, x.data(), tolerance, max_iterations, nullptr, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_LT(true_residual(A, b, x), tolerance);
    handle peer;
    ASSERT_EQ(adopt(c, copy.f, &o, peer), APXCHOL_STATUS_SUCCESS);
    Eigen::MatrixXd B(c.n, 5), X(c.n, 5), Y(c.n, 5);
    for (int col = 0; col < 5; ++col) B.col(col) = compatible_rhs(c.n, col + 1);
    std::array<apxchol_solve_info, 5> first{}, second{};
    const apxchol_const_dense_view rhs{B.data(), 1, c.n};
    const apxchol_dense_view first_out{X.data(), 1, c.n}, second_out{Y.data(), 1, c.n};
    ASSERT_EQ(apxchol_solver_solve(gpu_again.s, 5, &rhs, nullptr, &first_out, tolerance, max_iterations,
        first.data(), sizeof(first[0]), nullptr, 0), APXCHOL_STATUS_SUCCESS);
    ASSERT_EQ(apxchol_solver_solve(peer.s, 5, &rhs, nullptr, &second_out, tolerance, max_iterations,
        second.data(), sizeof(second[0]), nullptr, 0), APXCHOL_STATUS_SUCCESS);
    EXPECT_TRUE(same_bytes(X.data(), Y.data(), X.size()));
    for (int col = 0; col < 5; ++col) {
        EXPECT_EQ(first[col].iterations, second[col].iterations);
        EXPECT_EQ(first[col].relative_residual, second[col].relative_residual);
        EXPECT_EQ(first[col].converged, second[col].converged);
    }
    ASSERT_EQ(apxchol_solver_apply(gpu_again.s, 5, &rhs, &first_out, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    ASSERT_EQ(apxchol_solver_apply(peer.s, 5, &rhs, &second_out, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_TRUE(same_bytes(X.data(), Y.data(), X.size()));
#endif
    o.backend = APXCHOL_BACKEND_GPU;
    o.keep_factor_values = 0;
    handle released;
    ASSERT_EQ(create(c, &o, released), APXCHOL_STATUS_SUCCESS);
    factor_handle absent;
    EXPECT_EQ(apxchol_solver_copy_factor(released.s, &absent.f, nullptr, 0),
              APXCHOL_STATUS_NO_FACTOR_VALUES);
    std::vector<std::int64_t> perm(c.n);
    EXPECT_EQ(apxchol_solver_export_factor(released.s, 0, nullptr, nullptr, nullptr, perm.data(),
                                          nullptr, 0), APXCHOL_STATUS_SUCCESS);
    std::sort(perm.begin(), perm.end());
    for (std::int64_t i = 0; i < c.n; ++i) EXPECT_EQ(perm[i], i);
}

TEST(CApi, GpuDistinguishesUnsupportedStoredLayoutFromInvalidOperator) {
    if (!apxchol_backend_available(APXCHOL_BACKEND_GPU)) GTEST_SKIP() << "no usable GPU backend";
    // Numerically diagonal and positive definite, but the stored zero has no mate.
    const csc64 unpaired_zero{2, {0, 2, 3}, {0, 1, 1}, {2.0, 0.0, 3.0}};
    for (const auto backend : {APXCHOL_BACKEND_CPU, APXCHOL_BACKEND_AUTO}) {
        auto o = defaults();
        o.backend = backend;
        handle h;
        ASSERT_EQ(create(unpaired_zero, &o, h), APXCHOL_STATUS_SUCCESS);
        EXPECT_EQ(stats_of(h.s).backend, APXCHOL_BACKEND_CPU);
        const std::array<double, 2> b{2, 6};
        std::array<double, 2> x{};
        ASSERT_EQ(solve_one(h.s, b.data(), nullptr, x.data(), tolerance, max_iterations, nullptr, nullptr, 0),
                  APXCHOL_STATUS_SUCCESS);
        EXPECT_NEAR(x[0], 1, 1e-8);
        EXPECT_NEAR(x[1], 2, 1e-8);
    }
    auto o = defaults();
    o.backend = APXCHOL_BACKEND_GPU;
    handle refused;
    EXPECT_EQ(create(unpaired_zero, &o, refused), APXCHOL_STATUS_UNSUPPORTED);
    EXPECT_EQ(refused.s, nullptr);

    const csc64 asymmetric{2, {0, 2, 4}, {0, 1, 0, 1}, {2.0, -1.0, -0.5, 2.0}};
    std::string message;
    EXPECT_EQ(create(asymmetric, &o, refused, 0, &message), APXCHOL_STATUS_INVALID_OPERATOR);
    EXPECT_EQ(refused.s, nullptr);
    EXPECT_NE(message.find("symmetry"), std::string::npos) << message;
}

#if defined(APXCHOL_USE_METAL)
TEST(CApi, MetalRejectsUnsupportedInputRangeWithoutChangingRoute) {
    if (!apxchol_backend_available(APXCHOL_BACKEND_GPU)) GTEST_SKIP() << "no usable GPU backend";
    Sparse tiny = grid_laplacian(6, 6, 0.5);
    tiny.coeffRef(1, 0) = tiny.coeffRef(0, 1) = -1e-35;
    const csc64 small_values = to_csc64(tiny);
    auto o = cpu_options();
    factor_handle f;
    ASSERT_EQ(create_factor(to_csc64(grid_laplacian(6, 6, 0.5)), &o, f), APXCHOL_STATUS_SUCCESS);
    o.backend = APXCHOL_BACKEND_GPU;
    handle refused;
    std::string message;
    EXPECT_EQ(create(small_values, &o, refused, 0, &message), APXCHOL_STATUS_UNSUPPORTED);
    EXPECT_EQ(refused.s, nullptr);
    EXPECT_NE(message.find("2^-100"), std::string::npos) << message;
    EXPECT_EQ(adopt(small_values, f.f, &o, refused), APXCHOL_STATUS_UNSUPPORTED);
    o.backend = APXCHOL_BACKEND_CPU;
    handle supported;
    EXPECT_EQ(create(small_values, &o, supported), APXCHOL_STATUS_SUCCESS);
}
#endif

TEST(CApi, VersionAndAbi) {
    EXPECT_EQ(apxchol_c_abi_version(), 3);
    EXPECT_EQ(apxchol_c_abi_version(), APXCHOL_C_ABI_VERSION);
    const std::string v = apxchol_version();
    EXPECT_NE(v.find('+'), std::string::npos) << v;
}

TEST(CApi, NullArgumentsAreErrorsAndDestroyNullIsANoOp) {
    apxchol_solver_destroy(nullptr);
    const csc64 c = to_csc64(grid_laplacian(3, 3));
    auto matrix = c.view();
    EXPECT_EQ(apxchol_solver_create(&matrix, nullptr,
                                    nullptr, nullptr, nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    apxchol_solver* s = nullptr;
    matrix.colptr = nullptr;
    EXPECT_EQ(apxchol_solver_create(&matrix, nullptr, nullptr, &s,
                                    nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    matrix = c.view();
    matrix.n = 0;
    EXPECT_EQ(apxchol_solver_create(&matrix, nullptr,
                                    nullptr, &s, nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    std::vector<double> v(c.n, 0.0), out(c.n, 0.0);
    EXPECT_EQ(solve_one(nullptr, v.data(), nullptr, out.data(), tolerance, max_iterations, nullptr, nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(apply_one(nullptr, v.data(), out.data(), nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(apxchol_solver_solve(nullptr, 0, nullptr, nullptr, nullptr, tolerance, max_iterations,
        nullptr, 0, nullptr, 0), APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(apxchol_solver_apply(nullptr, 0, nullptr, nullptr, nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(apxchol_solver_stats(nullptr, nullptr), APXCHOL_STATUS_INVALID_ARGUMENT);
    handle h;
    ASSERT_EQ(create(c, nullptr, h), APXCHOL_STATUS_SUCCESS);
    EXPECT_EQ(apxchol_solver_export_factor(h.s, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0),
              APXCHOL_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(solve_one(h.s, nullptr, nullptr, v.data(), tolerance, max_iterations, nullptr, nullptr, 0),
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
    auto o = cpu_options();
    o.threads = 1;
    handle a, b;
    ASSERT_EQ(create(sorted, &o, a), APXCHOL_STATUS_SUCCESS);
    ASSERT_EQ(create(messy, &o, b), APXCHOL_STATUS_SUCCESS);
    const Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(sorted.n, 0.0, 1.0);
    Eigen::VectorXd xa(sorted.n), xb(sorted.n);
    ASSERT_EQ(solve_one(a.s, rhs.data(), nullptr, xa.data(), tolerance, max_iterations, nullptr, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    ASSERT_EQ(solve_one(b.s, rhs.data(), nullptr, xb.data(), tolerance, max_iterations, nullptr, nullptr, 0),
              APXCHOL_STATUS_SUCCESS);
    EXPECT_TRUE(same_bytes(xa.data(), xb.data(), xa.size()));
}

TEST(CApi, DistinctHandlesSolveConcurrently) {
    const csc64 c = to_csc64(grid_laplacian(16, 16));
    std::array<int, 2> statuses{-1, -1};
    std::array<double, 2> residuals{};
    auto work = [&](int t) {
        auto o = cpu_options();
        o.threads = 2;
        handle h;
        if (create(c, &o, h) != APXCHOL_STATUS_SUCCESS) return;
        const Eigen::VectorXd b = compatible_rhs(c.n, static_cast<unsigned>(t + 1));
        Eigen::VectorXd x(c.n);
        auto info = info_struct();
        statuses[t] = solve_one(h.s, b.data(), nullptr, x.data(), tolerance, max_iterations, &info,
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
