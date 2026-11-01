#include <gtest/gtest.h>

#include "apxchol/solver/solve.h"
#include "apxchol/detail/checkpoint.h"
#include "apxchol/detail/call_guard.h"
#include "apxchol/solver/detail/solver_route.h"
#include "apxchol/solver/detail/thread_budget.h"
#include "csc_test_util.h"
#include "scoped_omp_threads.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <future>
#include <cstring>
#include <cstdlib>
#include <latch>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(APXCHOL_USE_CUDA)
#include <cuda_runtime.h>
#endif

using csc_access = apxchol::detail::sparse_csc_access;
namespace test = apxchol::test;

namespace {
using Sparse = test::sparse_matrix;
using Vector = std::vector<double>;
using Dense = test::dense_matrix;

static_assert(std::is_move_constructible_v<apxchol::solver>);
static_assert(std::is_move_assignable_v<apxchol::solver>);
static_assert(std::is_copy_constructible_v<apxchol::solver>);
static_assert(std::is_copy_assignable_v<apxchol::solver>);
static_assert(noexcept(apxchol::backend_available(apxchol::solve_backend::gpu)));

Sparse operator_matrix(int n = 64) {
    std::vector<test::triplet> entries;
    for (int i = 0; i < n; ++i) {
        entries.push_back({i, i, 5.0});
        for (const int offset : {1, 7}) {
            const int j = (i + offset) % n;
            entries.push_back({i, j, -1.0});
            entries.push_back({j, i, -1.0});
        }
    }
    return test::from_triplets(n, n, entries);
}

using apxchol::test::linspaced;
// s * v + shift.
Vector affine(double s, std::span<const double> v, double shift = 0.0) {
    Vector out(v.begin(), v.end());
    for (double& x : out) x = s * x + shift;
    return out;
}

void subtract_mean(std::span<double> v) {
    const double m = test::mean(v);
    for (double& x : v) x -= m;
}

bool is_zero(std::span<const double> v) {
    return std::all_of(v.begin(), v.end(), [](double x) { return x == 0.0; });
}

using apxchol::test::stable_norm;
// ||b - A x|| / ||b||, both norms computed stably.
double stable_residual(const Sparse& A, std::span<const double> x, std::span<const double> b) {
    return stable_norm(test::difference(b, test::multiply(A, x))) / stable_norm(b);
}

using apxchol::test::entry;
using apxchol::test::with_entry;
// value * I.
Sparse diagonal_matrix(std::int64_t n, double value = 1.0) {
    return test::scaled(test::identity(n), value);
}

// cols copies of v.
Dense replicate(std::span<const double> v, std::int64_t cols) {
    Dense D(static_cast<std::int64_t>(v.size()), cols);
    for (std::int64_t j = 0; j < cols; ++j) std::copy(v.begin(), v.end(), D.col(j).begin());
    return D;
}

void set_col(Dense& D, std::int64_t j, std::span<const double> v) {
    std::copy(v.begin(), v.end(), D.col(j).begin());
}

// Rows [first, first + count) of D as a view with D's column stride.
apxchol::matrix_view middle_rows(Dense& D, std::int64_t first, std::int64_t count) {
    return {D.data.data() + first, count, D.cols, D.rows};
}
apxchol::const_matrix_view middle_rows(const Dense& D, std::int64_t first, std::int64_t count) {
    return {D.data.data() + first, count, D.cols, D.rows};
}
apxchol::const_matrix_view strided_x_const(apxchol::matrix_view M) {
    return {M.data, M.rows, M.cols, M.col_stride};
}
std::span<const double> view_col(apxchol::const_matrix_view M, std::int64_t j) {
    return {M.data + j * M.col_stride, static_cast<std::size_t>(M.rows)};
}

// Conjugate gradients on A with an optional preconditioner z = M(r). Returns
// the iterations needed to reach ||b - A x|| <= tol ||b||, or -1.
int conjugate_gradient(const Sparse& A, std::span<const double> b, Vector& x, double tol, int max_iter,
                       const std::function<Vector(std::span<const double>)>& precondition) {
    x.assign(b.size(), 0.0);
    Vector r(b.begin(), b.end());
    Vector z = precondition ? precondition(r) : r;
    Vector p = z;
    double rz = test::dot(r, z);
    const double bound = tol * test::norm(b);
    for (int k = 0; k < max_iter; ++k) {
        if (test::norm(r) <= bound) return k;
        const Vector Ap = test::multiply(A, p);
        const double alpha = rz / test::dot(p, Ap);
        for (std::size_t i = 0; i < x.size(); ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        z = precondition ? precondition(r) : r;
        const double next = test::dot(r, z);
        for (std::size_t i = 0; i < p.size(); ++i) p[i] = z[i] + next / rz * p[i];
        rz = next;
    }
    return test::norm(r) <= bound ? max_iter : -1;
}

apxchol::solve_options cpu_options() {
    apxchol::solve_options options;
    options.backend = apxchol::solve_backend::cpu;
    options.factor_opts.factor_storage = apxchol::factor_storage_type::fp32;
    return options;
}

template <class T>
bool same_bytes(const T* left, const T* right, std::size_t count) {
    return std::memcmp(left, right, count * sizeof(T)) == 0;
}

void expect_solution(const Sparse& matrix, std::span<const double> rhs,
                     std::span<const double> x,
                     double tolerance = apxchol::default_tol) {
    ASSERT_EQ(static_cast<std::int64_t>(x.size()), matrix.rows);
    EXPECT_TRUE(test::all_finite(x));
    EXPECT_LT(test::relative_residual(matrix, x, rhs), tolerance);
}
}  // namespace

TEST(Solver, BackendAvailabilityKeepsHostQueriesOnHost) {
#if defined(APXCHOL_USE_CUDA)
    const bool started = apxchol::detail::cuda_context::state().started;
#endif
    EXPECT_TRUE(apxchol::backend_available(apxchol::solve_backend::cpu));
    EXPECT_TRUE(apxchol::backend_available(apxchol::solve_backend::automatic));
    EXPECT_FALSE(apxchol::backend_available(static_cast<apxchol::solve_backend>(-1)));
    EXPECT_FALSE(apxchol::backend_available(static_cast<apxchol::solve_backend>(99)));
#if defined(APXCHOL_USE_CUDA)
    EXPECT_EQ(apxchol::detail::cuda_context::state().started, started);
    if (const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
        visible && (std::string(visible).empty() || std::string(visible) == "-1")) {
        EXPECT_FALSE(apxchol::backend_available(apxchol::solve_backend::gpu));
    }
#elif !defined(APXCHOL_USE_METAL)
    EXPECT_FALSE(apxchol::backend_available(apxchol::solve_backend::gpu));
#endif
}

TEST(Solver, CpuInterfacesRejectEmptyOperator) {
    const Sparse empty(0, 0);
    EXPECT_THROW(apxchol::solver(test::csc(empty), cpu_options()), std::invalid_argument);
}

TEST(Solver, ExternalConjugateGradientUsesPreconditioner) {
    // An external Krylov solver calls apply() on each residual.
    const scoped_omp_threads threads(1);
    const auto options = cpu_options();
    for (const bool laplacian : {false, true}) {
        SCOPED_TRACE(laplacian ? "Laplacian" : "SDDM");
        Sparse matrix = operator_matrix();
        if (laplacian)
            for (int i = 0; i < matrix.rows; ++i) entry(matrix, i, i) -= 1.0;
        Vector rhs = linspaced(matrix.rows, 1.0, 2.0);
        if (laplacian) subtract_mean(rhs);

        apxchol::solver owner(test::csc(matrix), options);
        Vector x, plain_x;
        const int preconditioned = conjugate_gradient(
            matrix, rhs, x, apxchol::default_tol, apxchol::default_max_iter,
            [&](std::span<const double> r) { return owner.apply(r); });
        const int plain = conjugate_gradient(matrix, rhs, plain_x, apxchol::default_tol,
                                             apxchol::default_max_iter, {});
        ASSERT_GE(preconditioned, 0);
        ASSERT_GE(plain, 0);
        EXPECT_LT(preconditioned, plain);
        expect_solution(matrix, rhs, x);
    }
}

TEST(Solver, CpuBreakdownPreservesFiniteSolution) {
    const scoped_omp_threads threads(1);
    const Sparse identity = diagonal_matrix(2);
    const auto factor = apxchol::factorize(test::csc(identity));
    for (const auto& [diagonal, rhs_value] : {std::pair{1e308, 1e150}, std::pair{1e-320, 1.0}}) {
        SCOPED_TRACE(diagonal);
        const Sparse matrix = diagonal_matrix(2, diagonal);
        apxchol::solver owner(test::csc(matrix), factor, cpu_options());
        const Vector rhs = Vector(2, rhs_value);
        const auto result = owner.solve(rhs, 1e-8, 12);
        EXPECT_EQ(result.iterations, 0);
        EXPECT_FALSE(result.converged);
        EXPECT_TRUE(is_zero(result.x));
        EXPECT_EQ(result.residual, 1.0);
        EXPECT_EQ(stable_residual(matrix, result.x, rhs), result.residual);
    }
}

TEST(Solver, CpuBatchBreakdownDoesNotStopOtherColumns) {
    const scoped_omp_threads threads(1);
    const Sparse identity = diagonal_matrix(2);
    const Sparse matrix = diagonal_matrix(2, 1e308);
    apxchol::solver owner(test::csc(matrix), apxchol::factorize(test::csc(identity)), cpu_options());
    for (const int columns : {2, 4, 8}) {
        SCOPED_TRACE(columns);
        Dense rhs = Dense(2, columns), output(2, columns);
        std::ranges::fill(rhs.col(0), 1e150);
        for (int c = 2; c < columns; ++c) rhs(c % 2, c) = 1;
        const auto result = owner.solve(std::as_const(rhs).view(), output.view(), 1e-8, 12);
        EXPECT_TRUE(test::all_finite(output.data));
        EXPECT_TRUE(is_zero(output.col(0)) && is_zero(output.col(1)));
        for (int c = 0; c < columns; ++c) {
            SCOPED_TRACE(c);
            EXPECT_EQ(result.columns[c].iterations, c < 2 ? 0 : 1);
            EXPECT_EQ(result.columns[c].converged, c != 0);
            const double residual = is_zero(rhs.col(c)) ? 0
                : stable_residual(matrix, output.col(c), rhs.col(c));
            EXPECT_DOUBLE_EQ(result.columns[c].residual, residual);
        }
    }
}

TEST(Solver, CpuOwnerSurvivesInputsAndMovesAndReusesItsFactor) {
    const scoped_omp_threads threads(1);
    const Sparse matrix = operator_matrix();
    const auto options = cpu_options();
    std::unique_ptr<apxchol::solver> original;
    apxchol::detail::checkpoint setup;  // outlives the solver it is attached to
    {
        Sparse temporary = matrix;
        original = std::make_unique<apxchol::solver>(
            test::with_timings(temporary, options, setup));
        EXPECT_GT(setup.total("setup"), 0.0);
    }
    apxchol::solver moved(std::move(*original));
    original.reset();
    apxchol::solver owner(test::csc(matrix), options);
    owner = std::move(moved);
    EXPECT_EQ(owner.rows(), matrix.rows);
    EXPECT_EQ(owner.backend(), apxchol::solve_backend::cpu);
    EXPECT_EQ(owner.factor_storage(), apxchol::factor_storage_type::fp32);
    apxchol::solver reference(test::csc(matrix), options);
    for (const double start : {1.0, -3.0, 1.0}) {
        const Vector rhs = linspaced(matrix.rows, start, start + 1.0);
        const auto actual = owner.solve(rhs);
        const auto expected = reference.solve(rhs);
        EXPECT_EQ(actual.backend, apxchol::solve_backend::cpu);
        expect_solution(matrix, rhs, actual.x);
        EXPECT_EQ(actual.iterations, expected.iterations);
        EXPECT_EQ(actual.residual, expected.residual);
        EXPECT_TRUE(same_bytes(actual.x.data(), expected.x.data(), actual.x.size()));
    }
}

namespace {
constexpr const char* busy_message =
    "this solver is already in use by another thread; use a separate solver per thread, "
    "or pass all right-hand sides to solve() as one matrix";

void expect_busy(const std::function<void()>& call) {
    try {
        call();
        ADD_FAILURE() << "overlapping call was accepted";
    } catch (const std::runtime_error& error) {
        EXPECT_STREQ(error.what(), busy_message);
    }
}

// Two threads start together and call one object until one call is rejected.
// Accepted calls must match their serial results.
void expect_overlap_rejected(const std::function<void()>& call) {
    std::atomic<bool> rejected = false;
    std::latch start(2);
    const auto run = [&] {
        start.arrive_and_wait();
        for (int i = 0; i < 1000 && !rejected; ++i) {
            try {
                call();
            } catch (const std::runtime_error& error) {
                EXPECT_STREQ(error.what(), busy_message);
                rejected = true;
            }
        }
    };
    std::thread other(run);
    run();
    other.join();
    EXPECT_TRUE(rejected);
}
}  // namespace

TEST(Solver, CallGuardRejectsOtherThreadsAndAllowsReentry) {
    const apxchol::detail::call_guard guard;
    std::latch entered(1), release(1);
    std::thread holder([&] {
        const auto scope = guard.enter();
        { const auto nested = guard.enter(); }
        entered.count_down();
        release.wait();
    });
    entered.wait();
    expect_busy([&] { (void)guard.enter(); });
    const apxchol::detail::call_guard copy = guard;
    { const auto scope = copy.enter(); }
    release.count_down();
    holder.join();
    { const auto scope = guard.enter(); }
}

TEST(Solver, OverlappingCallsOnOneOwnerThrow) {
    const scoped_omp_threads threads(1);
    const Sparse matrix = operator_matrix(3000);
    const auto options = cpu_options();
    apxchol::solver owner(test::csc(matrix), options);
    const Vector rhs = linspaced(matrix.rows, 1.0, -2.0);
    const Dense panel = Dense(matrix.rows, 3, 1.0);
    const std::vector<double> solved = owner.solve(rhs).x,
                              applied = owner.apply(rhs);
    Dense expected_panel(matrix.rows, 3);
    owner.solve(std::as_const(panel).view(), expected_panel.view());
    expect_overlap_rejected([&] { EXPECT_EQ(owner.solve(rhs).x, solved); });
    expect_overlap_rejected([&] { EXPECT_EQ(owner.apply(rhs), applied); });
    expect_overlap_rejected([&] {
        Dense x(matrix.rows, 3);
        owner.solve(std::as_const(panel).view(), x.view());
        EXPECT_EQ(x.data, expected_panel.data);
    });
    // A solve and an application share the preconditioner workspace.
    std::atomic<bool> rejected = false;
    std::latch start(2);
    std::thread solver_thread([&] {
        start.arrive_and_wait();
        for (int i = 0; i < 1000 && !rejected; ++i)
            try {
                EXPECT_EQ(owner.solve(rhs).x, solved);
            } catch (const std::runtime_error& error) {
                EXPECT_STREQ(error.what(), busy_message);
                rejected = true;
            }
    });
    start.arrive_and_wait();
    for (int i = 0; i < 1000 && !rejected; ++i)
        try {
            EXPECT_EQ(owner.apply(rhs), applied);
        } catch (const std::runtime_error& error) {
            EXPECT_STREQ(error.what(), busy_message);
            rejected = true;
        }
    solver_thread.join();
    EXPECT_TRUE(rejected);
}

TEST(Solver, CopiesShareTheFactorAndSolveInParallel) {
    const scoped_omp_threads threads(2);
    const Sparse matrix = operator_matrix(20000);
    auto options = cpu_options();
    options.num_threads = 2;
    std::optional<apxchol::solver> original(std::in_place, test::csc(matrix), options);
    constexpr int callers = 4;
    std::vector<Vector> rhs;
    std::vector<std::vector<double>> solved, applied;
    for (int t = 0; t < callers; ++t) {
        rhs.push_back(linspaced(matrix.rows, t + 1.0, -2.0 * t));
        solved.push_back(original->solve(rhs.back()).x);
        applied.push_back(original->apply(rhs.back()));
    }
    const Dense panel = Dense(matrix.rows, 3, 1.0);
    Dense expected_panel(matrix.rows, 3);
    original->solve(std::as_const(panel).view(), expected_panel.view());

    std::vector<apxchol::solver> owners(callers, *original);
    const auto route = [](const apxchol::solver& owner) -> const apxchol::detail::cpu_route& {
        return std::get<apxchol::detail::cpu_route>(apxchol::detail::solver_access::route(owner));
    };
    for (int t = 0; t < callers; ++t) {
        EXPECT_EQ(&route(owners[t]).factor(), &route(*original).factor());
        EXPECT_EQ(&route(owners[t]).trsv(), &route(*original).trsv());
    }
    const auto* shared_factor = &route(*original).factor();
    original.reset();
    for (int round = 0; round < 3; ++round) {
        std::vector<std::vector<double>> x(callers), z(callers);
        std::vector<Dense> batch(callers, Dense(matrix.rows, 3));
        std::vector<std::thread> workers;
        for (int t = 0; t < callers; ++t)
            workers.emplace_back([&, t] {
                const scoped_omp_threads worker_threads(2);
                x[t] = owners[t].solve(rhs[t]).x;
                z[t] = owners[t].apply(rhs[t]);
                owners[t].solve(std::as_const(panel).view(), batch[t].view());
            });
        for (auto& worker : workers) worker.join();
        for (int t = 0; t < callers; ++t) {
            EXPECT_EQ(x[t], solved[t]);
            EXPECT_EQ(z[t], applied[t]);
            EXPECT_EQ(batch[t].data, expected_panel.data);
        }
    }
    EXPECT_EQ(&route(owners[0]).factor(), shared_factor);
    // Where the CPU triangular solve picks its hybrid schedule (as on x86
    // Linux), the parallel solves above also cover per-copy barrier state.
    RecordProperty("hybrid_schedule", route(owners[0]).trsv().uses_hybrid_schedule());

    // Assignment shares the source.
    const Sparse other_matrix = operator_matrix(500);
    apxchol::solver assigned(test::csc(other_matrix), options);
    assigned = owners[0];
    EXPECT_EQ(&route(assigned).factor(), shared_factor);
    EXPECT_EQ(assigned.solve(rhs[1]).x, solved[1]);
}

TEST(Solver, CpuOwnerAdoptsAnIndependentFactorAndAppliesIt) {
    const scoped_omp_threads threads(1);
    const Sparse matrix = operator_matrix();
    auto options = cpu_options();
    const auto expected = apxchol::factorize(test::csc(matrix), options.factor_opts);
    std::unique_ptr<apxchol::solver> owner;
    {
        Sparse temporary = matrix;
        apxchol::factorization supplied = expected;
        owner = std::make_unique<apxchol::solver>(test::csc(temporary), std::move(supplied), options);
    }
    // factor() is the installed form of the adopted construction factor.
    const auto actual = owner->factor();
    EXPECT_EQ(expected.storage, apxchol::factor_storage_type::automatic);
    EXPECT_EQ(actual.storage, apxchol::factor_storage_type::fp32);
    EXPECT_EQ(actual.perm, expected.perm);
    EXPECT_EQ(actual.L.rows(), expected.L.rows());
    EXPECT_EQ(static_cast<std::size_t>(actual.L.nonZeros()), csc_access::vals(actual.L).size());
    EXPECT_LE(actual.L.nonZeros(), expected.L.nonZeros());
    apxchol::solver reference(test::csc(matrix), expected, options);
    apxchol::solver reinstalled(test::csc(matrix), actual, options);
    const Vector rhs = linspaced(matrix.rows, -1.0, 2.0);
    const auto applied = owner->apply(rhs);
    const auto reference_applied = reference.apply(rhs);
    EXPECT_TRUE(same_bytes(applied.data(), reference_applied.data(), applied.size()));
    const auto reinstalled_applied = reinstalled.apply(rhs);
    EXPECT_TRUE(same_bytes(applied.data(), reinstalled_applied.data(), applied.size()));
    expect_solution(matrix, rhs, owner->solve(rhs).x);

    options.backend = apxchol::solve_backend::automatic;
    apxchol::solver automatic_owner(test::csc(matrix), expected, options);
    EXPECT_EQ(automatic_owner.backend(), apxchol::solve_backend::cpu);
    EXPECT_TRUE(same_bytes(automatic_owner.apply(rhs).data(),
                          reference_applied.data(), reference_applied.size()));
    expect_solution(matrix, rhs, automatic_owner.solve(rhs).x);
}

TEST(Solver, AdoptedLaplacianFactorRejectsPositiveExcessBeforeBackendSetup) {
    const scoped_omp_threads threads(1);
    const std::vector<test::triplet> entries{{0, 0, 1}, {0, 2, -1},
        {1, 1, 1}, {1, 2, -1}, {2, 0, -1}, {2, 1, -1}, {2, 2, 2}};
    const Sparse laplacian = test::from_triplets(3, 3, entries);
    const auto factor = apxchol::factorize(test::csc(laplacian));
    ASSERT_FALSE(factor.sddm);
    for (const double positive : {0.0, 0.125})
    for (const double scale : {1e-6, 1.0, 1e6}) {
        SCOPED_TRACE(::testing::Message() << positive << ' ' << scale);
        Sparse matrix = with_entry(with_entry(test::scaled(laplacian, scale), 0, 1, scale * positive),
                                   1, 0, scale * positive);
        entry(matrix, 0, 0) = entry(matrix, 1, 1) = scale * (1 - positive);
        EXPECT_NO_THROW(apxchol::solver(test::csc(matrix), factor, cpu_options()));
        entry(matrix, 0, 0) = scale * (1 - positive + 4e-12);
        for (const auto backend : {apxchol::solve_backend::cpu,
                                   apxchol::solve_backend::automatic,
                                   apxchol::solve_backend::gpu}) {
            SCOPED_TRACE(static_cast<int>(backend));
            auto options = cpu_options();
            options.backend = backend;
            try {
                apxchol::solver owner(test::csc(matrix), factor, options);
                FAIL() << "accepted a Laplacian factor for an SDDM operator";
            } catch (const std::invalid_argument& error) {
                EXPECT_NE(std::string(error.what()).find("Laplacian factor"), std::string::npos);
            }
        }
        entry(matrix, 0, 0) = scale * (1 - positive + 0.25e-12);
        EXPECT_NO_THROW(apxchol::solver(test::csc(matrix), factor, cpu_options()));
    }
}

TEST(Solver, AdoptedSddmFactorStillRequiresAValidOperator) {
    const scoped_omp_threads threads(1);
    const Sparse matrix = operator_matrix(16);
    const auto factor = apxchol::factorize(test::csc(matrix));
    ASSERT_TRUE(factor.sddm);
    Sparse asymmetric = matrix;
    entry(asymmetric, 0, 1) *= 2;
    EXPECT_THROW(apxchol::solver(test::csc(asymmetric), factor, cpu_options()), std::invalid_argument);
    Sparse nonfinite = matrix;
    entry(nonfinite, 0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(apxchol::solver(test::csc(nonfinite), factor, cpu_options()), std::invalid_argument);
    auto lower_entries = test::triplets(matrix);
    std::erase_if(lower_entries, [](const test::triplet& e) { return e.row < e.col; });
    const Sparse lower = test::from_triplets(matrix.rows, matrix.cols, lower_entries);
    EXPECT_THROW(apxchol::solver(test::csc(lower), factor, cpu_options()), std::invalid_argument);
}

TEST(Solver, AdoptedSddmFactorCanPreconditionALaplacian) {
    const scoped_omp_threads threads(1);
    const std::vector<test::triplet> entries{{0, 0, 1}, {0, 1, -1},
                                             {1, 0, -1}, {1, 1, 1}};
    const Sparse laplacian = test::from_triplets(2, 2, entries);
    Sparse sddm = laplacian;
    entry(sddm, 0, 0) += 1;
    entry(sddm, 1, 1) += 1;

    const auto factor = apxchol::factorize(test::csc(sddm));
    ASSERT_TRUE(factor.sddm);
    apxchol::solver owner(test::csc(laplacian), factor, cpu_options());
    const Vector rhs{1, -1};
    const auto result = owner.solve(rhs);
    EXPECT_TRUE(result.converged);
    expect_solution(laplacian, rhs, result.x);
}

TEST(Solver, CpuCallerOutputAndWarmStartsUseEachOwnersStorage) {
    const scoped_omp_threads threads(1);
    const Sparse matrix = operator_matrix();
    const Vector rhs = linspaced(matrix.rows, 0.5, 2.0);
    for (const auto storage : {apxchol::factor_storage_type::fp32,
                               apxchol::factor_storage_type::fp16}) {
        if (storage == apxchol::factor_storage_type::fp16 && !apxchol::detail::omp_sptrsv::fp16_supported())
            continue;
        SCOPED_TRACE(static_cast<int>(storage));
        auto options = cpu_options();
        options.factor_opts.factor_storage = storage;
        apxchol::solver owner(test::csc(matrix), options);
        EXPECT_EQ(owner.factor_storage(), storage);
        const auto expected = owner.solve(rhs);
        std::vector<double> memory(static_cast<std::size_t>(matrix.rows + 2), 1234.0);
        const std::span<double> output(memory.data() + 1, static_cast<std::size_t>(matrix.rows));

        const auto result = owner.solve(rhs, output);
        EXPECT_EQ(result.x.size(), 0u);
        EXPECT_EQ(result.iterations, expected.iterations);
        EXPECT_EQ(result.residual, expected.residual);
        EXPECT_TRUE(same_bytes(output.data(), expected.x.data(), static_cast<std::size_t>(output.size())));
        expect_solution(matrix, rhs, output);
        const auto warm = owner.solve(rhs, output, apxchol::default_tol, 0, expected.x);
        EXPECT_EQ(warm.x.size(), 0u);
        EXPECT_EQ(warm.iterations, 0);
        expect_solution(matrix, rhs, output);
        EXPECT_EQ(memory.front(), 1234.0);
        EXPECT_EQ(memory.back(), 1234.0);
    }
}

TEST(Solver, AttachedTimingTreeRecordsSetupAndEverySolve) {
    const scoped_omp_threads threads(1);
    const Sparse matrix = operator_matrix();
    apxchol::detail::checkpoint timings;
    auto owner = test::with_timings(matrix, cpu_options(), timings);
    const double setup = timings.total("setup");
    // Selecting the backend can take less than one clock tick, so check that
    // the stage was recorded rather than that it took time.
    EXPECT_NE(timings.report().find("backend_selection"), std::string::npos);
    EXPECT_EQ(timings.total("pcg"), 0.0);
    const Vector rhs = linspaced(matrix.rows, 1.0, 2.0);
    expect_solution(matrix, rhs, owner.solve(rhs).x);
    EXPECT_GT(timings.total("pcg.solve"), 0.0);
    const double first = timings.total("pcg");
    EXPECT_GT(first, 0.0);
    Dense B = replicate(rhs, 3), X(matrix.rows, 3);
    owner.solve(std::as_const(B).view(), X.view());
    EXPECT_GT(timings.total("pcg"), first);
    EXPECT_EQ(timings.total("setup"), setup);

    // A copy starts without the tree.
    apxchol::solver copy = owner;
    const double before = timings.total("pcg");
    expect_solution(matrix, rhs, copy.solve(rhs).x);
    EXPECT_EQ(timings.total("pcg"), before);
}

TEST(Solver, ReportsSetupAndSolveSeconds) {
    const scoped_omp_threads threads(1);
    const Sparse matrix = operator_matrix();
    const auto positive = [](double seconds) { return std::isfinite(seconds) && seconds > 0.0; };
    apxchol::solver owner(test::csc(matrix), cpu_options());
    const double setup = owner.setup_seconds();
    EXPECT_TRUE(positive(setup)) << setup;
    const Vector b = linspaced(matrix.rows, 1.0, 2.0);
    const auto scalar = owner.solve(b);
    EXPECT_TRUE(scalar.converged);
    EXPECT_TRUE(positive(scalar.solve_seconds)) << scalar.solve_seconds;
    Vector x(matrix.rows);
    const auto into = owner.solve(b, x);
    EXPECT_TRUE(positive(into.solve_seconds)) << into.solve_seconds;
    Dense B = replicate(b, 3), X(matrix.rows, 3);
    const auto batch = owner.solve(std::as_const(B).view(), X.view());
    EXPECT_TRUE(positive(batch.solve_seconds)) << batch.solve_seconds;
    for (const auto& column : batch.columns) {
        EXPECT_TRUE(column.converged);
        EXPECT_EQ(column.solve_seconds, 0.0);
    }
    EXPECT_TRUE(test::all_finite(owner.apply(b)));
    EXPECT_EQ(owner.setup_seconds(), setup);
    const apxchol::solver copy = owner;
    EXPECT_EQ(copy.setup_seconds(), setup);
    const apxchol::solver adopted(test::csc(matrix), apxchol::factorize(test::csc(matrix)), cpu_options());
    EXPECT_TRUE(positive(adopted.setup_seconds())) << adopted.setup_seconds();
    const auto one_shot = apxchol::solve(test::csc(matrix), b, std::nullopt, std::nullopt, {},
                                         cpu_options());
    EXPECT_TRUE(one_shot.converged);
    EXPECT_TRUE(positive(one_shot.solve_seconds)) << one_shot.solve_seconds;
}

TEST(Solver, CpuScalesExtremeRhsAndWarmStarts) {
    const scoped_omp_threads threads(1);
    const Sparse matrix = diagonal_matrix(16);
    const Vector unit = linspaced(matrix.rows, 1.0, 2.0);
    for (const auto storage : {apxchol::factor_storage_type::fp32,
                               apxchol::factor_storage_type::fp16}) {
        if (storage == apxchol::factor_storage_type::fp16 && !apxchol::detail::omp_sptrsv::fp16_supported())
            continue;
        auto options = cpu_options();
        options.factor_opts.factor_storage = storage;
        apxchol::solver owner(test::csc(matrix), options);
        for (const double magnitude : {1e-200, 1e-160, 1.0, 1e160, 1e200}) {
            SCOPED_TRACE(::testing::Message() << static_cast<int>(storage) << ' ' << magnitude);
            const Vector rhs = affine(magnitude, unit);
            const Vector x0 = affine(0.25, rhs);
            for (const Vector* guess : {static_cast<const Vector*>(nullptr), &x0}) {
                const auto result = owner.solve(rhs, apxchol::default_tol, apxchol::default_max_iter,
                                                guess ? *guess : std::span<const double>{});
                EXPECT_TRUE(test::all_finite(result.x));
                EXPECT_LT(result.residual, apxchol::default_tol);
                EXPECT_TRUE(result.converged);
                EXPECT_LT(stable_norm(test::difference(result.x, rhs)) / stable_norm(rhs), apxchol::default_tol);
            }
            Vector output(matrix.rows);
            const auto exact = owner.solve(rhs, output, apxchol::default_tol, 0,
                                           rhs);
            EXPECT_EQ(exact.iterations, 0);
            EXPECT_EQ(exact.residual, 0.0);
            EXPECT_TRUE(exact.converged);
            EXPECT_EQ(output, rhs);
            const auto stopped = owner.solve(rhs, apxchol::default_tol, 0);
            EXPECT_EQ(stopped.iterations, 0);
            EXPECT_EQ(stopped.residual, 1.0);
            EXPECT_FALSE(stopped.converged);
            EXPECT_TRUE(is_zero(stopped.x));
        }
        const Vector zero_rhs = Vector(matrix.rows, 0.0);
        const auto zero = owner.solve(zero_rhs);
        EXPECT_EQ(zero.residual, 0.0);
        EXPECT_TRUE(zero.converged);
        EXPECT_TRUE(is_zero(zero.x));
        const Vector tiny = affine(1e-200, unit), huge = affine(1e200, unit);
        EXPECT_THROW(owner.solve(tiny, apxchol::default_tol, 0, huge),
                     std::invalid_argument);
    }
}

TEST(Solver, CpuPanelsScaleExceptionalColumnsIndependently) {
    const scoped_omp_threads threads(2);
    const Sparse matrix = operator_matrix(65);
    Dense rhs(matrix.rows, 15), guess = rhs, output = rhs;
    const double magnitudes[] = {1.0, 1e-200, 1e200, 0.0, 1e-160, 1e160};
    for (int c = 0; c < rhs.cols; ++c) {
        const Vector exact = affine(magnitudes[c % 6], linspaced(matrix.rows, -1.0, 1.0));
        set_col(rhs, c, test::multiply(matrix, exact));
        set_col(guess, c, affine(0.25, exact));
    }
    const apxchol::const_matrix_view initial = std::as_const(guess).view();
    for (const auto storage : {apxchol::factor_storage_type::fp32,
                               apxchol::factor_storage_type::fp16}) {
        if (storage == apxchol::factor_storage_type::fp16 && !apxchol::detail::omp_sptrsv::fp16_supported())
            continue;
        auto options = cpu_options();
        options.factor_opts.factor_storage = storage;
        apxchol::solver owner(test::csc(matrix), options);
        for (const bool warm : {false, true})
            for (const int budget : {0, 1, 200}) {
                SCOPED_TRACE(::testing::Message() << int(storage) << ' ' << warm << ' ' << budget);
                const auto batch = owner.solve(std::as_const(rhs).view(), output.view(), 1e-8, budget,
                                               warm ? initial : apxchol::const_matrix_view{});
                for (int c = 0; c < rhs.cols; ++c) {
                    const auto b = std::as_const(rhs).col(c), x0 = std::as_const(guess).col(c);
                    const auto scalar = owner.solve(b, 1e-8, budget,
                                                    warm ? x0 : std::span<const double>{});
                    ASSERT_TRUE(test::all_finite(output.col(c)));
                    EXPECT_EQ(batch.columns[c].iterations, scalar.iterations);
                    EXPECT_NEAR(batch.columns[c].residual, scalar.residual, 1e-12);
                    EXPECT_EQ(batch.columns[c].converged, scalar.converged);
                    EXPECT_EQ(scalar.converged, scalar.residual < 1e-8);
                    if (is_zero(b)) {
                        EXPECT_TRUE(is_zero(output.col(c)));
                    } else {
                        const double residual = stable_residual(matrix, output.col(c), b);
                        EXPECT_NEAR(batch.columns[c].residual, residual, 1e-12);
                        if (budget == 200) {
                            EXPECT_LT(residual, 1e-8);
                        }
                    }
                }
            }
        Dense bad_guess = guess;
        std::ranges::fill(bad_guess.col(1), 1e200);
        EXPECT_THROW(owner.solve(std::as_const(rhs).view(), output.view(), 1e-8, 0, std::as_const(bad_guess).view()),
                     std::invalid_argument);
    }
}


TEST(Solver, CpuExitCheckPreservesSmallComponentsAndOverflowingNorms) {
    const scoped_omp_threads threads(1);
    const Sparse identity = diagonal_matrix(101);
    apxchol::solver owner(test::csc(identity), cpu_options());
    Dense rhs = Dense(101, 4), guess = rhs, output = rhs;
    rhs(0, 0) = 2;
    std::fill_n(rhs.col(0).begin() + 1, 4, std::numeric_limits<double>::denorm_min());
    rhs(0, 1) = 1e100;
    std::fill_n(rhs.col(1).begin() + 1, 100, 1e-224);
    for (int c = 0; c < rhs.cols; ++c) guess(0, c) = rhs(0, c);
    std::ranges::fill(rhs.col(2), 1e308);
    set_col(guess, 2, affine(-1.0, rhs.col(2)));
    const auto batch = owner.solve(std::as_const(rhs).view(), output.view(), 1e-8, 0, std::as_const(guess).view());
    for (int c = 0; c < rhs.cols; ++c) {
        SCOPED_TRACE(c);
        const auto b = std::as_const(rhs).col(c), x0 = std::as_const(guess).col(c);
        const auto scalar = owner.solve(b, 1e-8, 0, x0);
        const double expected = c < 2 ? stable_norm(test::difference(b, x0)) / stable_norm(b)
                                     : c == 2 ? 2.0 : 0.0;
        if (c < 2) { ASSERT_GT(expected, 0); }
        EXPECT_DOUBLE_EQ(scalar.residual, expected);
        EXPECT_DOUBLE_EQ(batch.columns[c].residual, expected);
        EXPECT_EQ(scalar.converged, expected < 1e-8);
        EXPECT_EQ(batch.columns[c].converged, expected < 1e-8);
        EXPECT_EQ(scalar.iterations, 0);
        EXPECT_EQ(batch.columns[c].iterations, 0);
        EXPECT_TRUE(std::ranges::equal(output.col(c), x0));
        EXPECT_TRUE(std::ranges::equal(scalar.x, x0));
    }
}

TEST(Solver, CpuExitCheckReducesStridedPanelsAcrossWorkerTeams) {
    // Cross both platform-specific SpMV team thresholds. Budget zero isolates
    // the exit check, including panel tails and columns requiring stable norms.
    // The right-hand sides are padded panels: column stride 2n.
    constexpr std::int64_t n = 33001;
    for (int workers : {1, 4}) {
        const scoped_omp_threads threads(workers);
        for (double diagonal : {1.0, 1.1}) {
            const Sparse matrix = diagonal_matrix(n, diagonal);
            apxchol::solver owner(test::csc(matrix), cpu_options());
            for (int columns : {1, 4, 17}) {
                Dense rhs(n, columns), guess(n, columns), output(n, columns);
                for (int c = 0; c < columns; ++c) {
                    const double scale[] = {1.0, 1e-160, 1e160, 0.0};
                    set_col(rhs, c, affine(scale[c % 4], linspaced(n, 1.0, 2.0)));
                    for (std::int64_t i = 0; i < n; ++i)
                        guess(i, c) = (c % 5 == 0 ? 1.0 : 0.25) * rhs(i, c) / diagonal;
                }
                const apxchol::const_matrix_view initial = std::as_const(guess).view();
                for (std::int64_t offset : {std::int64_t(0), n}) {
                    Dense stored = Dense(2 * n, columns, 91.0);
                    for (int c = 0; c < columns; ++c)
                        std::ranges::copy(rhs.col(c), stored.col(c).begin() + offset);
                    const auto result = owner.solve(middle_rows(std::as_const(stored), offset, n),
                                                    output.view(), 1e-8, 0, initial);
                    for (int c = 0; c < columns; ++c) {
                        SCOPED_TRACE(::testing::Message() << workers << ' ' << diagonal
                                     << ' ' << columns << ' ' << offset << ' ' << c);
                        const double expected = is_zero(rhs.col(c)) ? 0.0
                            : stable_norm(test::difference(rhs.col(c), affine(diagonal, guess.col(c))))
                                / stable_norm(rhs.col(c));
                        EXPECT_NEAR(result.columns[c].residual, expected, 1e-13);
                        EXPECT_EQ(result.columns[c].converged, expected < 1e-8);
                        EXPECT_EQ(result.columns[c].iterations, 0);
                        EXPECT_TRUE(std::ranges::equal(output.col(c), guess.col(c)));

                    }
                }
            }
        }
    }
}

TEST(Solver, InvalidDropTolerancePrecedesBackendSelection) {
    const Sparse matrix = operator_matrix();
    const auto factor = apxchol::factorize(test::csc(matrix));
    const auto reject = [](auto&& construct) {
        try {
            construct();
            FAIL() << "accepted invalid drop tolerance";
        } catch (const std::invalid_argument& error) {
            EXPECT_STREQ(error.what(), "drop_tolerance must be in [0, 1]");
        }
    };
    for (auto backend : {apxchol::solve_backend::automatic, apxchol::solve_backend::cpu,
                         apxchol::solve_backend::gpu}) {
        for (double drop : {-1.0, std::nextafter(1.0, 2.0), 2.0,
                            std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()}) {
            auto options = cpu_options();
            options.backend = backend;
            options.factor_opts.drop_tolerance = drop;
            reject([&] { apxchol::solver owner(test::csc(matrix), options); });
            reject([&] { apxchol::solver owner(test::csc(matrix), factor, options); });
        }
    }
}

TEST(Solver, RejectsInvalidOptionsAndLimits) {
    const scoped_omp_threads threads(1);
    const Sparse matrix = operator_matrix();
    const auto defaults = cpu_options();
    auto invalid = defaults;
    invalid.backend = static_cast<apxchol::solve_backend>(99);
    EXPECT_THROW(apxchol::solver(test::csc(matrix), invalid), std::invalid_argument);
    apxchol::solver owner(test::csc(matrix), defaults);
    const Vector b = Vector(matrix.rows, 1.0);
    Dense B = replicate(b, 2), X(matrix.rows, 2);
    Dense empty(matrix.rows, 0), empty_output(matrix.rows, 0);
    for (double tol : {0.0, -1.0, -1e-12, std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
        EXPECT_THROW(owner.solve(b, tol), std::invalid_argument);
        EXPECT_THROW(owner.solve(std::as_const(B).view(), X.view(), tol), std::invalid_argument);
        EXPECT_THROW(owner.solve(std::as_const(empty).view(), empty_output.view(), tol), std::invalid_argument);
        EXPECT_THROW(apxchol::solve(test::csc(matrix), b, tol, std::nullopt, {}, defaults),
                     std::invalid_argument);
    }
    for (int max_iter : {-1, -7}) {
        EXPECT_THROW(owner.solve(b, std::nullopt, max_iter), std::invalid_argument);
        EXPECT_THROW(owner.solve(std::as_const(B).view(), X.view(), std::nullopt, max_iter), std::invalid_argument);
        EXPECT_THROW(apxchol::solve(test::csc(matrix), b, std::nullopt, max_iter, {}, defaults),
                     std::invalid_argument);
    }
    EXPECT_TRUE(owner.solve(b).converged);
    EXPECT_FALSE(owner.solve(b, apxchol::default_tol, 0).converged);
}

TEST(Solver, OmittedLimitsUseLibraryDefaults) {
    const scoped_omp_threads threads(1);
    const Sparse matrix = operator_matrix();
    apxchol::solver owner(test::csc(matrix), cpu_options());
    const Vector b = linspaced(matrix.rows, 1.0, 2.0);
    Dense B(matrix.rows, 3);
    B.data = test::random_vector(matrix.rows * 3, 7);
    const auto same = [](const apxchol::solve_result& left, const apxchol::solve_result& right) {
        EXPECT_EQ(left.iterations, right.iterations);
        EXPECT_EQ(left.residual, right.residual);
        EXPECT_EQ(left.converged, right.converged);
    };
    const auto implicit = owner.solve(b);
    const auto explicit_limits = owner.solve(b, apxchol::default_tol, apxchol::default_max_iter);
    same(implicit, explicit_limits);
    EXPECT_TRUE(same_bytes(implicit.x.data(), explicit_limits.x.data(), b.size()));
    Dense X(B.rows, B.cols), Y(B.rows, B.cols);
    const auto batch = owner.solve(std::as_const(B).view(), X.view());
    const auto explicit_batch = owner.solve(std::as_const(B).view(), Y.view(), apxchol::default_tol,
                                            apxchol::default_max_iter);
    ASSERT_EQ(batch.columns.size(), explicit_batch.columns.size());
    for (std::size_t c = 0; c < batch.columns.size(); ++c)
        same(batch.columns[c], explicit_batch.columns[c]);
    EXPECT_TRUE(same_bytes(X.data.data(), Y.data.data(), X.data.size()));
    const auto one_shot = apxchol::solve(test::csc(matrix), b, std::nullopt, std::nullopt, {},
                                         cpu_options());
    same(implicit, one_shot);
}

TEST(Solver, CpuScalarViewsRejectWrongSizesAndAliasing) {
    const scoped_omp_threads threads(1);
    const Sparse matrix = operator_matrix();
    apxchol::solver owner(test::csc(matrix), cpu_options());
    Vector b = Vector(matrix.rows, 1.0), x = b;
    const Vector wrong = Vector(matrix.rows - 1, 1.0);
    EXPECT_THROW(owner.solve(wrong), std::invalid_argument);
    EXPECT_THROW(owner.solve(wrong, x), std::invalid_argument);
    EXPECT_THROW(owner.solve(b, b), std::invalid_argument);
    EXPECT_THROW(owner.solve(b, x, std::nullopt, std::nullopt, x),
                 std::invalid_argument);
}

TEST(Solver, ExitChecksRunWithoutATimingTree) {
    const scoped_omp_threads threads(1);
    const Sparse matrix = operator_matrix();
    apxchol::solver owner(test::csc(matrix), cpu_options());
    const Vector b = Vector(matrix.rows, 1.0);
    const auto scalar = owner.solve(b);
    EXPECT_TRUE(scalar.converged);
    EXPECT_NEAR(scalar.residual, test::relative_residual(matrix, scalar.x, b), 1e-15);
    Dense B = replicate(b, 3), X(matrix.rows, 3);
    const auto batch = owner.solve(std::as_const(B).view(), X.view());
    for (const auto& column : batch.columns) EXPECT_TRUE(column.converged);
}

TEST(Solver, CpuRejectsUnrepresentableSolutions) {
    const scoped_omp_threads threads(1);
    const auto options = cpu_options();
    for (const double diagonal : {1e-30, 1e30}) {
        Sparse matrix = diagonal_matrix(4, diagonal);
        apxchol::solver owner(test::csc(matrix), options);
        const Vector rhs = Vector(4, diagonal < 1 ? 1e275 : 1e-275);
        const auto finite = owner.solve(rhs);
        ASSERT_TRUE(test::all_finite(finite.x));
        EXPECT_LT(stable_residual(matrix, finite.x, rhs), apxchol::default_tol);
        const Vector outside_range = Vector(4, diagonal < 1 ? 1e300 : 1e-300);
        Dense panel(matrix.rows, 4), output(matrix.rows, 4);
        set_col(panel, 0, rhs);
        set_col(panel, 2, outside_range);
        set_col(panel, 3, rhs);
        if (diagonal < 1) {
            EXPECT_THROW(owner.solve(outside_range), std::overflow_error);
            EXPECT_THROW(owner.solve(std::as_const(panel).view(), output.view()), std::overflow_error);
        } else {
            EXPECT_THROW(owner.solve(std::as_const(panel).view(), output.view()), std::underflow_error);
            EXPECT_THROW(owner.solve(outside_range), std::underflow_error);
            entry(matrix, 0, 0) = 1;
            apxchol::solver mixed(test::csc(matrix), options);
            EXPECT_THROW(mixed.solve(outside_range), std::underflow_error);
        }
    }
}

TEST(Solver, CudaOwnerReusesOwnedInputAndAppliesFactor) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "32-bit-node CUDA owner required";
#else
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver)
        GTEST_SKIP() << "CUDA unavailable: " << cudaGetErrorString(status);
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    if (count == 0) GTEST_SKIP() << "No CUDA devices";
    ASSERT_TRUE(apxchol::backend_available(apxchol::solve_backend::gpu));
    const Sparse matrix = operator_matrix();
    auto options = cpu_options();
    options.backend = apxchol::solve_backend::gpu;
    std::unique_ptr<apxchol::solver> owner;
    {
        Sparse temporary = matrix;
        owner = std::make_unique<apxchol::solver>(test::csc(temporary), options);
    }
    EXPECT_EQ(owner->rows(), matrix.rows);
    EXPECT_EQ(owner->backend(), apxchol::solve_backend::gpu);
    EXPECT_EQ(owner->factor_storage(), apxchol::factor_storage_type::fp32);
    EXPECT_EQ(owner->factor().perm.size(), static_cast<std::size_t>(matrix.rows));
    EXPECT_GT(owner->factor().L.nonZeros(), 0);
    for (const double start : {1.0, -2.0}) {
        const Vector rhs = linspaced(matrix.rows, start, start + 1.0);
        const auto result = owner->solve(rhs);
        EXPECT_EQ(result.backend, apxchol::solve_backend::gpu);
        expect_solution(matrix, rhs, result.x);
        Vector output(matrix.rows);
        const auto into = owner->solve(rhs, output);
        EXPECT_EQ(into.backend, apxchol::solve_backend::gpu);
        EXPECT_EQ(into.x.size(), 0u);
        expect_solution(matrix, rhs, output);
        const auto applied = owner->apply(rhs);
        ASSERT_EQ(applied.size(), static_cast<std::size_t>(matrix.rows));
        EXPECT_TRUE(test::all_finite(applied));
        EXPECT_GT(test::dot(applied, rhs), 0.0);

    }
#endif
}

TEST(Solver, CudaOwnerRejectsOverlappingCalls) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "32-bit-node CUDA owner required";
#else
    if (!apxchol::backend_available(apxchol::solve_backend::gpu))
        GTEST_SKIP() << "GPU unavailable";
    const Sparse matrix = operator_matrix(3000);
    auto options = cpu_options();
    options.backend = apxchol::solve_backend::gpu;
    apxchol::solver owner(test::csc(matrix), options);
    ASSERT_EQ(owner.backend(), apxchol::solve_backend::gpu);
    const Vector rhs = linspaced(matrix.rows, 1.0, -2.0);
    const Dense panel = Dense(matrix.rows, 3, 1.0);
    const std::vector<double> solved = owner.solve(rhs).x,
                              applied = owner.apply(rhs);
    // The odd last column re-enters the scalar solve on the calling thread.
    Dense expected_panel(matrix.rows, 3);
    owner.solve(std::as_const(panel).view(), expected_panel.view());
    expect_overlap_rejected([&] { EXPECT_EQ(owner.solve(rhs).x, solved); });
    expect_overlap_rejected([&] { EXPECT_EQ(owner.apply(rhs), applied); });
    expect_overlap_rejected([&] {
        Dense x(matrix.rows, 3);
        owner.solve(std::as_const(panel).view(), x.view());
        EXPECT_EQ(x.data, expected_panel.data);
    });
    // A copy has its own guard, which rejects overlap on the copy alone.
    apxchol::solver copy(owner);
    expect_overlap_rejected([&] { EXPECT_EQ(copy.solve(rhs).x, solved); });
    EXPECT_EQ(owner.solve(rhs).x, solved);
#endif
}

TEST(Solver, GpuCopiesShareTheDeviceFactorAndSolveConcurrently) {
#if !(defined(APXCHOL_USE_CUDA) || defined(APXCHOL_USE_METAL)) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "32-bit-node GPU owner required";
#else
    if (!apxchol::backend_available(apxchol::solve_backend::gpu))
        GTEST_SKIP() << "GPU unavailable";
#if defined(APXCHOL_USE_CUDA)
    using session = apxchol::detail::cuda_route;
#else
    using session = apxchol::detail::metal_route;
#endif
    const auto route = [](const apxchol::solver& owner) -> const session& {
        return std::get<session>(apxchol::detail::solver_access::route(owner));
    };
    Sparse laplacian = operator_matrix(20000);
    for (int i = 0; i < laplacian.rows; ++i) entry(laplacian, i, i) = 4.0;
    for (const Sparse& matrix : {operator_matrix(20000), laplacian}) {
        auto options = cpu_options();
        options.backend = apxchol::solve_backend::gpu;
        std::optional<apxchol::solver> original(std::in_place, test::csc(matrix), options);
        ASSERT_EQ(original->backend(), apxchol::solve_backend::gpu);
        constexpr int callers = 4;
        std::vector<Vector> rhs;
        std::vector<std::vector<double>> solved, applied;
        for (int t = 0; t < callers; ++t) {
            rhs.push_back(linspaced(matrix.rows, t + 1.0, -2.0 * t));
            solved.push_back(original->solve(rhs.back()).x);
            applied.push_back(original->apply(rhs.back()));
        }
        // Five columns run a four-column panel, then a scalar solve. On Metal
        // the copies build the shared lane plans on first use.
        Dense panel(matrix.rows, 5);
        for (int q = 0; q < panel.cols; ++q)
            set_col(panel, q, linspaced(matrix.rows, q - 2.0, 1.0 + q));

        std::vector<apxchol::solver> owners(callers, *original);
        for (int t = 0; t < callers; ++t) {
            // On Metal the factor lives with the shared plans and operator.
            EXPECT_EQ(&route(owners[t]).factor(), &route(*original).factor());
#if defined(APXCHOL_USE_CUDA)
            EXPECT_EQ(route(owners[t]).trsv().device_factor_values(),
                      route(*original).trsv().device_factor_values());
            EXPECT_EQ(route(owners[t]).pcg().device_operator_values(),
                      route(*original).pcg().device_operator_values());
            EXPECT_NE(route(owners[t]).pcg().stream(), route(*original).pcg().stream());
#endif
        }
        const auto run = [&](std::vector<apxchol::solver>& copies) {
            std::vector<std::vector<double>> x(callers), z(callers);
            std::vector<Dense> batch(callers, Dense(matrix.rows, 5));
            std::vector<std::thread> workers;
            for (int t = 0; t < callers; ++t)
                workers.emplace_back([&, t] {
                    for (int round = 0; round < 3; ++round) {
                        copies[t].solve(std::as_const(panel).view(), batch[t].view());
                        x[t] = copies[t].solve(rhs[t]).x;
                        z[t] = copies[t].apply(rhs[t]);
                    }
                });
            for (auto& worker : workers) worker.join();
            return std::tuple(x, z, batch);
        };
        const auto [x, z, batch] = run(owners);
        Dense expected_panel(matrix.rows, 5);
        original->solve(std::as_const(panel).view(), expected_panel.view());
        for (int t = 0; t < callers; ++t) {
            EXPECT_EQ(x[t], solved[t]);
            EXPECT_EQ(z[t], applied[t]);
            EXPECT_EQ(batch[t].data, expected_panel.data);
        }
        // The copies keep the shared device arrays alive.
        original.reset();
        owners.push_back(owners[0]);
        const auto [x2, z2, batch2] = run(owners);
        for (int t = 0; t < callers; ++t) {
            EXPECT_EQ(x2[t], solved[t]);
            EXPECT_EQ(z2[t], applied[t]);
            EXPECT_EQ(batch2[t].data, expected_panel.data);
        }
        EXPECT_EQ(owners.back().solve(rhs[0]).x, solved[0]);
    }
#endif
}

TEST(Solver, CudaOwnerAdoptsHostFactorsAndRejectsInvalidWarmStarts) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "32-bit-node CUDA owner required";
#else
    const Sparse matrix = operator_matrix();
    auto options = cpu_options();
    const auto host_factor = apxchol::factorize(test::csc(matrix), options.factor_opts);
    options.backend = apxchol::solve_backend::gpu;
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver)
        GTEST_SKIP() << "CUDA unavailable: " << cudaGetErrorString(status);
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    if (count == 0) GTEST_SKIP() << "No CUDA devices";
    options.factor_opts.partitioner = apxchol::partitioner_kind::priority_greedy;
    EXPECT_THROW(apxchol::solver(test::csc(matrix), options), std::invalid_argument);
    apxchol::solver owner(test::csc(matrix), host_factor, options);
    const Vector rhs = linspaced(matrix.rows, 1.0, 2.0);
    Vector x0 = Vector(matrix.rows - 1, 0.0);
    EXPECT_THROW(owner.solve(rhs, std::nullopt, std::nullopt, x0),
                 std::invalid_argument);
    Vector output(matrix.rows);
    EXPECT_THROW(owner.solve(rhs, output, std::nullopt, std::nullopt,
                             x0), std::invalid_argument);
    x0 = Vector(matrix.rows, 0.0);
    for (const double value : {std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
        x0[0] = value;
        EXPECT_THROW(owner.solve(rhs, std::nullopt, std::nullopt, x0),
                     std::invalid_argument);
    }
    EXPECT_EQ(owner.backend(), apxchol::solve_backend::gpu);
    const auto recovered = owner.solve(rhs);
    EXPECT_EQ(recovered.backend, apxchol::solve_backend::gpu);
    expect_solution(matrix, rhs, recovered.x);
#endif
}

TEST(Solver, CudaWarmStartsPreserveOriginalResidualAndIterationBudget) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "32-bit-node CUDA owner required";
#else
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver)
        GTEST_SKIP() << "CUDA unavailable: " << cudaGetErrorString(status);
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    if (count == 0) GTEST_SKIP() << "No CUDA devices";
    const scoped_omp_threads threads(1);
    constexpr int n = 64;
    // SDDM, connected Laplacian, then two disconnected Laplacian components.
    for (const int kind : {0, 1, 2}) {
        std::vector<test::triplet> entries;
        const int component_size = kind == 2 ? n / 2 : n;
        for (int i = 0; i < n; ++i) {
            entries.push_back({i, i, kind == 0 ? 5.0 : 4.0});
            for (const int offset : {1, 7}) {
                const int j = i / component_size * component_size
                    + (i + offset) % component_size;
                entries.push_back({i, j, -1.0});
                entries.push_back({j, i, -1.0});
            }
        }
        const Sparse matrix = test::from_triplets(n, n, entries);
        const Vector exact = linspaced(n, -31.5, 31.5);
        const Vector rhs = test::multiply(matrix, exact);
        const Vector zero = Vector(n, 0.0);
        for (const auto storage : {apxchol::factor_storage_type::fp32,
                                   apxchol::factor_storage_type::fp16}) {
            SCOPED_TRACE(::testing::Message() << kind << ' ' << static_cast<int>(storage));
            auto options = cpu_options();
            options.backend = apxchol::solve_backend::gpu;
            options.factor_opts.factor_storage = storage;
            apxchol::solver owner(test::csc(matrix), options);
            const double shift = kind == 0 ? 0.0 : 8.0;
            Vector x0 = affine(1.0, exact, shift);
            auto result = owner.solve(rhs, apxchol::default_tol, apxchol::default_max_iter,
                                      x0);
            EXPECT_EQ(result.iterations, 0);
            EXPECT_DOUBLE_EQ(result.residual, 0.0);
            EXPECT_EQ(result.x, exact);
            if (kind != 0) {
                const Vector incompatible = affine(1.0, rhs, 1.0);
                result = owner.solve(incompatible, apxchol::default_tol, 0, x0);
                EXPECT_NEAR(result.residual, std::sqrt(double(n)) / test::norm(incompatible), 1e-14);
                EXPECT_GT(result.residual, apxchol::default_tol);
            }

            x0 = affine(1.0 - 0x1p-20, exact, shift);
            result = owner.solve(rhs, 0x1p-19, apxchol::default_max_iter, x0);
            EXPECT_EQ(result.iterations, 0);
            EXPECT_NEAR(result.residual, 0x1p-20, 1e-14);

            x0 = affine(0.25, exact, shift);
            result = owner.solve(rhs, apxchol::default_tol, 0, x0);
            EXPECT_EQ(result.iterations, 0);
            EXPECT_NEAR(result.residual, 0.75, 1e-14);
            EXPECT_NEAR(test::relative_residual(matrix, result.x, rhs), result.residual, 1e-14);
            EXPECT_EQ(result.x, affine(0.25, exact));
            result = owner.solve(rhs, apxchol::default_tol, 1, x0);
            EXPECT_EQ(result.iterations, 1);
            EXPECT_NEAR(test::relative_residual(matrix, result.x, rhs), result.residual, 1e-12);

            Vector output(n);
            const auto into = owner.solve(rhs, output, apxchol::default_tol,
                                          apxchol::default_max_iter, x0);
            EXPECT_GT(into.iterations, 0);
            EXPECT_EQ(into.x.size(), 0u);
            expect_solution(matrix, rhs, output);
            if (kind != 0) {
                EXPECT_NEAR(test::mean(output), 0.0, 1e-12);
            }
            result = owner.solve(rhs, apxchol::default_tol, apxchol::default_max_iter,
                                 x0);
            expect_solution(matrix, rhs, result.x);

            const auto cold = owner.solve(rhs);
            result = owner.solve(rhs, apxchol::default_tol, apxchol::default_max_iter,
                                 zero);
            EXPECT_EQ(result.iterations, cold.iterations);
            EXPECT_EQ(result.x, cold.x);
            EXPECT_EQ(result.residual, cold.residual);
            result = owner.solve(zero, apxchol::default_tol, 0, x0);
            EXPECT_EQ(result.iterations, 0);
            EXPECT_EQ(result.residual, 0.0);
            EXPECT_EQ(result.x, zero);
        }
    }
#endif
}

TEST(Solver, CudaOwnerHonorsIterationBudgetWithPoorWarmStart) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "32-bit-node CUDA owner required";
#else
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver)
        GTEST_SKIP() << "CUDA unavailable: " << cudaGetErrorString(status);
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    if (count == 0) GTEST_SKIP() << "No CUDA devices";
    const scoped_omp_threads threads(1);
    const Sparse matrix = operator_matrix();
    const Vector exact = linspaced(matrix.rows, 1.0, 2.0);
    const Vector rhs = test::multiply(matrix, exact);
    const Vector x0 = affine(-1e6, exact);

    auto options = cpu_options();
    options.backend = apxchol::solve_backend::gpu;
    apxchol::solver owner(test::csc(matrix), options);
    const auto first = owner.solve(rhs, apxchol::default_tol, 1, x0);
    ASSERT_EQ(first.iterations, 1);
    ASSERT_GT(first.residual, 0.5);
    const auto next = owner.solve(rhs, apxchol::default_tol, 2, x0);
    EXPECT_EQ(next.iterations, 2);
#endif
}

TEST(Solver, CudaScalesRhsAndWarmStartsForFloatPreconditioner) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "32-bit-node CUDA owner required";
#else
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver)
        GTEST_SKIP() << "CUDA unavailable: " << cudaGetErrorString(status);
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    if (count == 0) GTEST_SKIP() << "No CUDA devices";
    const Sparse matrix = diagonal_matrix(16);
    const Vector unit = linspaced(matrix.rows, 1.0, 2.0);
    for (const auto storage : {apxchol::factor_storage_type::fp16,
                               apxchol::factor_storage_type::fp32}) {
        auto options = cpu_options();
        options.backend = apxchol::solve_backend::gpu;
        options.factor_opts.factor_storage = storage;
        apxchol::solver owner(test::csc(matrix), options);
        for (const double magnitude : {1e-100, 1.0, 1e100}) {
            SCOPED_TRACE(::testing::Message() << static_cast<int>(storage) << ' ' << magnitude);
            const Vector rhs = affine(magnitude, unit);
            const Vector x0 = affine(0.25, rhs);
            const auto cold = owner.solve(rhs);
            const auto warm = owner.solve(rhs, apxchol::default_tol, apxchol::default_max_iter,
                                          x0);
            for (const auto* result : {&cold, &warm}) {
                EXPECT_GT(result->iterations, 0);
                EXPECT_LT(result->residual, apxchol::default_tol);
                EXPECT_LT(stable_residual(matrix, result->x, rhs), 1e-9);
            }
            const auto exact = owner.solve(rhs, apxchol::default_tol, 0, rhs);
            EXPECT_EQ(exact.iterations, 0);
            EXPECT_EQ(exact.residual, 0.0);
            EXPECT_EQ(exact.x, rhs);
            const auto applied = owner.apply(rhs);
            EXPECT_LT(stable_norm(test::difference(applied, rhs)) / stable_norm(rhs), 1e-6);
        }
        for (const double value : {std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::quiet_NaN()}) {
            const Vector invalid = Vector(matrix.rows, value);
            EXPECT_THROW(owner.solve(invalid), std::invalid_argument);
            EXPECT_THROW(owner.apply(invalid), std::invalid_argument);
        }
        const Vector tiny = affine(1e-100, unit), huge = affine(1e200, unit);
        EXPECT_THROW(owner.solve(tiny, apxchol::default_tol, 0, huge),
                     std::invalid_argument);
    }
#endif
}

#if defined(APXCHOL_USE_CUDA) || defined(APXCHOL_USE_METAL)
TEST(Solver, GpuOwnerPreservesStorageAndAdoptsItsFactor) {
#if defined(APXCHOL_USE_CUDA) && defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "32-bit-node CUDA owner required";
#else
#ifdef APXCHOL_USE_CUDA
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver)
        GTEST_SKIP() << "CUDA unavailable: " << cudaGetErrorString(status);
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    if (count == 0) GTEST_SKIP() << "No CUDA devices";
#else
    if (!apxchol::backend_available(apxchol::solve_backend::gpu)) GTEST_SKIP() << "no usable Metal device";
#endif
    const scoped_omp_threads threads(1);
    const Sparse matrix = operator_matrix();
    const Vector rhs = linspaced(matrix.rows, 1.0, 2.0);
    for (const auto storage : {apxchol::factor_storage_type::fp32,
                              apxchol::factor_storage_type::fp16}) {
        SCOPED_TRACE(static_cast<int>(storage));
        auto options = cpu_options();
        options.backend = apxchol::solve_backend::gpu;
        options.factor_opts.factor_storage = storage;
        std::unique_ptr<apxchol::solver> owner;
        {
            Sparse temporary = matrix;
            owner = std::make_unique<apxchol::solver>(test::csc(temporary), options);
        }
        EXPECT_EQ(owner->backend(), apxchol::solve_backend::gpu);
        EXPECT_EQ(owner->rows(), matrix.rows);
        EXPECT_EQ(owner->factor_storage(), storage);
        const auto exported = owner->factor();
        EXPECT_EQ(exported.storage, storage);
        options.factor_opts.partitioner = apxchol::partitioner_kind::priority_greedy;
        EXPECT_THROW(apxchol::solver(test::csc(matrix), options), std::invalid_argument);
        apxchol::solver adopted(test::csc(matrix), exported, options);
        EXPECT_EQ(adopted.backend(), apxchol::solve_backend::gpu);
        EXPECT_EQ(adopted.factor_storage(), storage);
        const auto retained = adopted.factor();
        EXPECT_EQ(retained.scale, exported.scale);
        EXPECT_EQ(retained.sddm, exported.sddm);
        EXPECT_EQ(retained.L.rows(), exported.L.rows());
        EXPECT_EQ(retained.perm, exported.perm);
        EXPECT_EQ(csc_access::outer(retained.L), csc_access::outer(exported.L));
        EXPECT_EQ(csc_access::inner(retained.L), csc_access::inner(exported.L));
        ASSERT_EQ(csc_access::vals(retained.L).size(), csc_access::vals(exported.L).size());
        EXPECT_TRUE(same_bytes(csc_access::vals(retained.L).data(), csc_access::vals(exported.L).data(),
                               csc_access::vals(retained.L).size()));
        const auto result = owner->solve(rhs);
        const auto expected = adopted.solve(rhs);
        EXPECT_EQ(result.backend, apxchol::solve_backend::gpu);
        expect_solution(matrix, rhs, result.x);
        expect_solution(matrix, rhs, expected.x);
        auto reference_options = options;
        reference_options.backend = apxchol::solve_backend::cpu;
        apxchol::solver reference(test::csc(matrix), exported, reference_options);
        const auto reference_applied = reference.apply(rhs);
        // Resident and host preparation can accumulate entries in different orders.
        for (auto* candidate : {owner.get(), &adopted}) {
            const auto applied = candidate->apply(rhs);
            ASSERT_EQ(applied.size(), reference_applied.size());
            EXPECT_LT(test::norm(test::difference(applied, reference_applied))
                          / test::norm(reference_applied), 2e-5);
        }
        Vector output(matrix.rows);
        const auto warm = adopted.solve(rhs, output, apxchol::default_tol, 0, result.x);
        EXPECT_EQ(warm.iterations, 0);
        EXPECT_EQ(warm.x.size(), 0u);
        expect_solution(matrix, rhs, output);
        const Vector negated = affine(-1.0, rhs);
        expect_solution(matrix, negated, owner->solve(negated).x);
    }
#endif
}
#endif

#if defined(APXCHOL_USE_CUDA) || defined(APXCHOL_USE_METAL)
TEST(Solver, GpuScalarAndBatchedResidualsDescribeReturnedSolutions) {
#if defined(APXCHOL_USE_CUDA) && defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "32-bit-node CUDA owner required";
#else
#ifdef APXCHOL_USE_CUDA
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver)
        GTEST_SKIP() << cudaGetErrorString(status);
    ASSERT_EQ(status, cudaSuccess);
    if (!devices) GTEST_SKIP() << "No CUDA devices";
#else
    if (!apxchol::backend_available(apxchol::solve_backend::gpu)) GTEST_SKIP() << "No usable Metal device";
#endif
    const scoped_omp_threads threads(2);
    for (const bool laplacian : {false, true}) {
        Sparse matrix = operator_matrix(65);
        if (laplacian)
            for (int i = 0; i < matrix.rows; ++i) entry(matrix, i, i) -= 1.0;
        matrix = test::scaled(matrix, 1.1);
        Dense exact(matrix.rows, 5);
        const double magnitudes[] = {1.0, 0.0, 1e-200, 1e200, -1.0};
        for (int q = 0; q < exact.cols; ++q)
            set_col(exact, q, affine(magnitudes[q], linspaced(matrix.rows, -1.0, 1.0)));
        Dense rhs_columns(matrix.rows, exact.cols), guess(matrix.rows, exact.cols), output(matrix.rows, exact.cols);
        for (int q = 0; q < exact.cols; ++q) {
            set_col(rhs_columns, q, test::multiply(matrix, exact.col(q)));
            set_col(guess, q, affine(0.25, exact.col(q)));
        }
        const Dense rhs = rhs_columns;
        set_col(guess, 0, exact.col(0));
        std::ranges::fill(guess.col(1), 1.0);
        const apxchol::const_matrix_view initial = std::as_const(guess).view();
        for (const auto storage : {apxchol::factor_storage_type::fp16,
                                   apxchol::factor_storage_type::fp32}) {
            auto options = cpu_options();
            options.backend = apxchol::solve_backend::gpu;
            options.factor_opts.factor_storage = storage;
            apxchol::solver owner(test::csc(matrix), options);
            for (const bool warm : {false, true}) {
                for (const int budget : {0, 1, 200}) {
                    SCOPED_TRACE(::testing::Message() << laplacian << ' '
                        << int(storage) << ' ' << warm << ' ' << budget);
                    const auto batch = owner.solve(std::as_const(rhs).view(), output.view(), 1e-8, budget,
                                                   warm ? initial : apxchol::const_matrix_view{});
                    ASSERT_EQ(batch.columns.size(), 5u);
                    for (int q = 0; q < rhs.cols; ++q) {
                        const auto b = rhs.col(q), x0 = std::as_const(guess).col(q);
                        const auto scalar = owner.solve(b, 1e-8, budget,
                                                        warm ? x0 : std::span<const double>{});
                        const auto check = [&](const apxchol::solve_result& result,
                                               std::span<const double> x) {
                            ASSERT_TRUE(test::all_finite(x));
                            const double norm = stable_norm(b);
                            const double measured = norm ? stable_residual(matrix, x, b) : 0;
                            if (!norm) {
                                EXPECT_TRUE(is_zero(x));
                            }
                            EXPECT_LE(result.iterations, budget);
                            EXPECT_NEAR(result.residual, measured, 1e-12 * (1 + measured));
                            EXPECT_EQ(result.converged, result.residual < 1e-8);
                            if (budget == 0 && !warm && norm > 0) {
                                EXPECT_FALSE(result.converged);
                            }
                        };
                        check(batch.columns[q], output.col(q));
                        check(scalar, scalar.x);
                    }
                }
            }
        }
    }
    // Small components still contribute to the reported residual beside a
    // much larger component that the initial guess already solves exactly.
    const Sparse identity = diagonal_matrix(101);
    Dense rhs = Dense(101, 2), guess = rhs, output = rhs;
    rhs(0, 0) = 2;
    std::fill_n(rhs.col(0).begin() + 1, 4, std::numeric_limits<double>::denorm_min());
    rhs(0, 1) = 1e100;
    std::fill_n(rhs.col(1).begin() + 1, 100, 1e-224);
    for (int c = 0; c < rhs.cols; ++c) guess(0, c) = rhs(0, c);
    auto options = cpu_options();
    options.backend = apxchol::solve_backend::gpu;
    apxchol::solver owner(test::csc(identity), options);
    const auto batch = owner.solve(std::as_const(rhs).view(), output.view(), 1e-8, 0, std::as_const(guess).view());
    ASSERT_EQ(batch.columns.size(), 2u);
    for (int q = 0; q < rhs.cols; ++q) {
        const auto b = std::as_const(rhs).col(q), x0 = std::as_const(guess).col(q);
        const auto scalar = owner.solve(b, 1e-8, 0, x0);
        const double expected = stable_norm(test::difference(b, x0)) / stable_norm(b);
        ASSERT_GT(expected, 0);
        EXPECT_EQ(batch.columns[q].residual, expected);
        EXPECT_EQ(scalar.residual, expected);
        EXPECT_EQ(batch.columns[q].converged, expected < 1e-8);
        EXPECT_EQ(scalar.converged, expected < 1e-8);
        EXPECT_TRUE(std::ranges::equal(output.col(q), x0));
        EXPECT_TRUE(std::ranges::equal(scalar.x, x0));
    }
#endif
}
#endif

TEST(Solver, CpuMatrixRhsUsesIndependentStopsAndOneTimingTree) {
    const scoped_omp_threads threads(2);
    constexpr int side = 96, n = side * side;
    std::vector<test::triplet> entries;
    std::vector<double> diagonal(n);
    auto edge = [&](int a, int b) {
        entries.push_back({a, b, -1.0});
        entries.push_back({b, a, -1.0});
        ++diagonal[a]; ++diagonal[b];
    };
    for (int i = 0; i < n; ++i) {
        if (i % side) edge(i - 1, i);
        if (i >= side) edge(i - side, i);
    }
    for (int i = 0; i < n; ++i) entries.push_back({i, i, diagonal[i]});
    const Sparse matrix = test::from_triplets(n, n, entries);
    auto options = cpu_options();
    apxchol::solver owner(test::csc(matrix), options);
    Dense rhs(n, 10), x0 = Dense(n, 10), x(n, 10);
    for (int c = 0; c < rhs.cols; ++c) {
        for (int i = 0; i < n; ++i) rhs(i, c) = std::sin(double(i + 1) * (c + 1));
        subtract_mean(rhs.col(c));
    }
    std::ranges::fill(rhs.col(0), 0.0);
    std::ranges::fill(x0.col(0), 1.0);
    set_col(x0, 1, linspaced(n, -1.0, 1.0));
    set_col(rhs, 1, test::multiply(matrix, x0.col(1)));
    const auto batch = owner.solve(std::as_const(rhs).view(), x.view(), 1e-10, 200, std::as_const(x0).view());
    ASSERT_EQ(batch.columns.size(), 10u);
    EXPECT_EQ(batch.columns[0].iterations, 0);
    EXPECT_EQ(batch.columns[1].iterations, 0);
    EXPECT_TRUE(is_zero(x.col(0)));
    EXPECT_GT(batch.solve_seconds, 0.0);
    for (int c = 0; c < rhs.cols; ++c) {
        EXPECT_EQ(batch.columns[c].x.size(), 0u);
        EXPECT_EQ(batch.columns[c].solve_seconds, 0.0);
        EXPECT_EQ(batch.columns[c].backend, apxchol::solve_backend::cpu);
        if (c > 0) expect_solution(matrix, rhs.col(c), x.col(c), 1e-9);
        const auto b = std::as_const(rhs).col(c), guess = std::as_const(x0).col(c);
        const auto scalar = owner.solve(b, 1e-10, 200, guess);
        EXPECT_LE(test::norm(test::difference(x.col(c), scalar.x)), 1e-7 * std::max(1.0, test::norm(scalar.x)));
    }
    const auto exhausted = owner.solve(std::as_const(rhs).view(), x.view(), 1e-10, 0);
    for (int c = 0; c < rhs.cols; ++c) {
        EXPECT_EQ(exhausted.columns[c].iterations, 0);
        EXPECT_DOUBLE_EQ(exhausted.columns[c].residual, c ? 1.0 : 0.0);
    }
}

TEST(Solver, CpuMatrixRhsRejectsAliasingAndAcceptsEmptyBatches) {
    const scoped_omp_threads threads(1);
    const Sparse matrix = operator_matrix();
    apxchol::solver owner(test::csc(matrix), cpu_options());
    Dense rhs = Dense(owner.rows(), 3, 1.0);
    EXPECT_THROW(owner.solve(std::as_const(rhs).view(), rhs.view()), std::invalid_argument);
    EXPECT_THROW(owner.apply(std::as_const(rhs).view(), rhs.view()), std::invalid_argument);
    Dense wrong(owner.rows() - 1, 3);
    EXPECT_THROW(owner.solve(std::as_const(rhs).view(), wrong.view()), std::invalid_argument);
    Dense empty(owner.rows(), 0), output(owner.rows(), 0);
    const auto result = owner.solve(std::as_const(empty).view(), output.view());
    EXPECT_TRUE(result.columns.empty());
    EXPECT_NO_THROW(owner.apply(std::as_const(empty).view(), output.view()));
}

namespace {
template<class Owner>
void check_nonfinite_apply(Owner&& owner) {
    const Vector finite = Vector(owner.rows(), 1.0);
    for (const double value : {std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity()}) {
        SCOPED_TRACE(value);
        Vector invalid = finite;
        invalid[owner.rows() / 2] = value;
        EXPECT_THROW(owner.apply(invalid), std::invalid_argument);
        Dense input = replicate(finite, 4), output(owner.rows(), 4);
        set_col(input, 3, invalid);

        EXPECT_THROW(owner.apply(std::as_const(input).view(), output.view()), std::invalid_argument);
    }
    EXPECT_TRUE(test::all_finite(owner.apply(finite)));
}
} // namespace

TEST(Solver, CpuApplyRejectsNonFiniteInputs) {
    const scoped_omp_threads threads(1);
    const auto matrix = operator_matrix();
    check_nonfinite_apply(apxchol::solver(test::csc(matrix), cpu_options()));
}

TEST(Solver, GpuApplyRejectsNonFiniteInputs) {
#if defined(APXCHOL_USE_CUDA) && defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "32-bit-node CUDA owner required";
#else
    if (!apxchol::backend_available(apxchol::solve_backend::gpu))
        GTEST_SKIP() << "GPU unavailable";
    const scoped_omp_threads threads(1);
    auto options = cpu_options();
    options.backend = apxchol::solve_backend::gpu;
    const auto matrix = operator_matrix();
    check_nonfinite_apply(apxchol::solver(test::csc(matrix), options));
#endif
}

namespace {
void check_slow_progress(apxchol::solve_backend backend) {
    const scoped_omp_threads threads(1);
    constexpr int n = 256;
    const Sparse identity = diagonal_matrix(n);
    std::vector<test::triplet> entries;
    for (int i = 0; i < n; ++i) {
        entries.push_back({i, i, 2.0});
        if (i) {
            entries.push_back({i, i - 1, -1.0});
            entries.push_back({i - 1, i, -1.0});
        }
    }
    const Sparse matrix = test::from_triplets(n, n, entries);
    auto options = cpu_options();
    options.backend = backend;
    apxchol::solver owner(test::csc(matrix), apxchol::factorize(test::csc(identity)), options);
    const Vector rhs = Vector(n, 1.0);
    // With an identity preconditioner this SPD problem initially grows the
    // residual, so the former 50-iteration progress test stopped it early.
    const auto limited = owner.solve(rhs, 1e-8, 50);
    ASSERT_EQ(limited.iterations, 50);
    EXPECT_GT(limited.residual, 0.5);
    EXPECT_FALSE(limited.converged);
    const auto continued = owner.solve(rhs, 1e-8, 51);
    EXPECT_EQ(continued.iterations, 51);
    const auto solved = owner.solve(rhs, 1e-8, 512);
    EXPECT_GT(solved.iterations, 50);
    EXPECT_LE(solved.iterations, 512);
    EXPECT_TRUE(solved.converged);
    expect_solution(matrix, rhs, solved.x, 1e-8);

    Dense B(n, 4), X(n, 4);
    set_col(B, 0, rhs);
    set_col(B, 2, affine(-2, rhs));
    set_col(B, 3, affine(0.5, rhs));
    for (const int budget : {50, 51, 512}) {
        SCOPED_TRACE(budget);
        const auto batch = owner.solve(std::as_const(B).view(), X.view(), 1e-8, budget);
        ASSERT_EQ(batch.columns.size(), 4u);
        EXPECT_EQ(batch.columns[1].iterations, 0);
        EXPECT_TRUE(is_zero(X.col(1)));
        for (const int c : {0, 2, 3}) {
            const auto& result = batch.columns[c];
            if (budget < 512) {
                EXPECT_EQ(result.iterations, budget);
                EXPECT_GT(result.residual, 0.5);
                EXPECT_FALSE(result.converged);
            } else {
                EXPECT_GT(result.iterations, 50);
                EXPECT_LE(result.iterations, budget);
                EXPECT_TRUE(result.converged);
                expect_solution(matrix, B.col(c), X.col(c), 1e-8);
            }
        }
    }
}
}

TEST(Solver, CpuScalarAndBatchContinueAfterSlowProgress) {
    check_slow_progress(apxchol::solve_backend::cpu);
}

TEST(Solver, MetalScalarAndBatchContinueAfterSlowProgress) {
#ifdef APXCHOL_USE_METAL
    if (!apxchol::backend_available(apxchol::solve_backend::gpu)) GTEST_SKIP() << "Metal unavailable";
    check_slow_progress(apxchol::solve_backend::gpu);
#else
    GTEST_SKIP() << "Metal build required";
#endif
}

TEST(Solver, CpuMatrixRhsKeepsConvergedColumnsWhileOtherColumnsContinue) {
    const scoped_omp_threads threads(1);
    constexpr int n = 17;
    const Sparse identity = diagonal_matrix(n);
    Sparse matrix = identity;
    matrix.values = linspaced(n, 1.0, double(n));
    apxchol::solver owner(test::csc(matrix), apxchol::factorize(test::csc(identity)), cpu_options());
    Dense rhs = Dense(n, 4), guess = rhs, x(n, 4);
    guess(n - 1, 0) = 1;
    set_col(rhs, 0, test::multiply(matrix, guess.col(0)));
    rhs(0, 2) = 1;
    std::ranges::fill(rhs.col(3), 1.0);
    const auto result = owner.solve(std::as_const(rhs).view(), x.view(), 1e-10, 200, std::as_const(guess).view());
    ASSERT_EQ(result.columns.size(), 4u);
    EXPECT_EQ(result.columns[0].iterations, 0);
    EXPECT_EQ(result.columns[1].iterations, 0);
    EXPECT_EQ(result.columns[2].iterations, 1);
    EXPECT_GT(result.columns[3].iterations, 1);
    EXPECT_TRUE(std::ranges::equal(x.col(0), guess.col(0)));
    EXPECT_TRUE(is_zero(x.col(1)));
    EXPECT_TRUE(std::ranges::equal(x.col(2), rhs.col(2)));
    for (int c : {0, 2, 3})
        expect_solution(matrix, rhs.col(c), x.col(c));
}

TEST(Solver, CpuSingletonPanelsRespectStridesWarmStartsAndBudgets) {
    const scoped_omp_threads threads(2);
    const Sparse matrix = operator_matrix();
    apxchol::solver owner(test::csc(matrix), cpu_options());
    const std::int64_t n = owner.rows();
    for (int columns : {1, 9}) {
        Dense rhs(n, columns), guess = Dense(n, columns, 1.0);
        for (int c = 0; c < columns; ++c)
            for (std::int64_t i = 0; i < n; ++i)
                rhs(i, c) = std::sin(double(i + 1) * (c + 1));
        set_col(rhs, columns - 1, test::multiply(matrix, guess.col(columns - 1)));
        if (columns > 1) std::ranges::fill(rhs.col(0), 0.0);
        const apxchol::const_matrix_view initial = std::as_const(guess).view();
        // Padded panels: n rows at a row offset into 2n-row storage, so the
        // column stride is 2n.
        for (std::int64_t offset : {std::int64_t(0), n}) {
            Dense stored_rhs = Dense(2 * n, columns, 91.0);
            Dense stored_guess = stored_rhs, stored_x = stored_rhs;
            for (int c = 0; c < columns; ++c) {
                std::ranges::copy(rhs.col(c), stored_rhs.col(c).begin() + offset);
                std::ranges::copy(guess.col(c), stored_guess.col(c).begin() + offset);
            }
            const apxchol::const_matrix_view strided_rhs = middle_rows(std::as_const(stored_rhs), offset, n);
            const apxchol::matrix_view strided_x = middle_rows(stored_x, offset, n);
            const apxchol::const_matrix_view strided_initial = middle_rows(std::as_const(stored_guess), offset, n);
            for (bool warm : {false, true})
                for (int budget : {0, 1, 200}) {
                    Dense x(n, columns);
                    const auto direct = owner.solve(std::as_const(rhs).view(), x.view(), 1e-10, budget,
                                                    warm ? initial : apxchol::const_matrix_view{});
                    const auto packed = owner.solve(strided_rhs, strided_x, 1e-10, budget,
                                                    warm ? strided_initial : apxchol::const_matrix_view{});
                    for (int c = 0; c < columns; ++c) {
                        EXPECT_EQ(direct.columns[c].iterations, packed.columns[c].iterations);
                        EXPECT_DOUBLE_EQ(direct.columns[c].residual, packed.columns[c].residual);
                        EXPECT_LE(test::norm(test::difference(x.col(c), view_col(strided_x_const(strided_x), c))),
                                  1e-12 * std::max(1.0, test::norm(x.col(c))));
                        if (budget == 0 || (warm && c == columns - 1)) {
                            EXPECT_EQ(direct.columns[c].iterations, 0);
                        }
                        if (budget == 200 && !is_zero(rhs.col(c)))
                            expect_solution(matrix, rhs.col(c), x.col(c));
                    }
                }
            Dense applied(n, columns);
            owner.apply(std::as_const(rhs).view(), applied.view());
            owner.apply(strided_rhs, strided_x);
            double difference = 0;
            for (int c = 0; c < columns; ++c) {
                const auto d = test::norm(test::difference(applied.col(c), view_col(strided_x_const(strided_x), c)));
                difference += d * d;
            }
            EXPECT_LE(std::sqrt(difference), 1e-12 * std::max(1.0, test::norm(applied.data)));
            const std::int64_t padding = offset == 0 ? n : 0;
            for (int c = 0; c < columns; ++c) {
                const auto column = stored_x.col(c).subspan(padding, n);
                EXPECT_TRUE(std::ranges::all_of(column, [](double v) { return v == 91.0; }));
            }

        }
        EXPECT_THROW(owner.solve(std::as_const(rhs).view(), rhs.view()), std::invalid_argument);
        EXPECT_THROW(owner.solve(std::as_const(rhs).view(), guess.view(), 1e-10, 200, initial), std::invalid_argument);
    }
}

namespace {
template<class Owner>
void check_thread_budget(const Sparse& matrix, int ambient, int budget,
                         apxchol::solve_backend backend = apxchol::solve_backend::cpu) {
    const scoped_omp_threads caller(ambient);
    auto options = cpu_options();
    options.backend = backend;
    options.num_threads = budget;
    Owner owner(test::csc(matrix), options);
    EXPECT_EQ(omp_get_max_threads(), ambient);
    Owner adopted(test::csc(matrix), owner.factor(), options);
    const Vector rhs = linspaced(matrix.rows, 1.0, 2.0);
    for (int repetition = 0; repetition < 2; ++repetition) {
        expect_solution(matrix, rhs, owner.solve(rhs).x);
        expect_solution(matrix, rhs, adopted.solve(rhs).x);
        EXPECT_EQ(omp_get_max_threads(), ambient);
        EXPECT_TRUE(test::all_finite(owner.apply(rhs)));
        EXPECT_EQ(omp_get_max_threads(), ambient);
        Vector output(matrix.rows);
        if constexpr (requires { owner.solve(rhs, output); }) {
            owner.solve(rhs, output);
            expect_solution(matrix, rhs, output);
        }
        EXPECT_EQ(omp_get_max_threads(), ambient);
    }
    Dense rhs_panel = replicate(rhs, 5), output(matrix.rows, 5);
    if constexpr (requires {
                      owner.solve(std::as_const(rhs_panel).view(), output.view());
                      owner.apply(std::as_const(rhs_panel).view(), output.view());
                  }) {
        const auto batch = owner.solve(std::as_const(rhs_panel).view(), output.view());
        EXPECT_EQ(omp_get_max_threads(), ambient);
        EXPECT_EQ(batch.columns.size(), 5u);
        for (int c = 0; c < output.cols; ++c) expect_solution(matrix, rhs, output.col(c));
        owner.apply(std::as_const(rhs_panel).view(), output.view());
        EXPECT_TRUE(test::all_finite(output.data));
        EXPECT_EQ(omp_get_max_threads(), ambient);
        EXPECT_THROW(owner.solve(std::as_const(rhs_panel).view(), rhs_panel.view()), std::invalid_argument);
        EXPECT_EQ(omp_get_max_threads(), ambient);
    }
    const Vector wrong_guess = Vector(1, 0.0);
    EXPECT_THROW(owner.solve(rhs, std::nullopt, std::nullopt, wrong_guess),
                 std::invalid_argument);
    EXPECT_EQ(omp_get_max_threads(), ambient);
    auto invalid = options;
    invalid.factor_opts.partitioner = static_cast<apxchol::partitioner_kind>(99);
    EXPECT_THROW(Owner(test::csc(matrix), invalid), std::invalid_argument);
    EXPECT_EQ(omp_get_max_threads(), ambient);
}
}

TEST(Solver, ThreadBudgetRestoresCallerAfterSettingChanges) {
    const scoped_omp_threads caller(3);
    for (const int budget : {2, 3, 4}) {
        for (const bool fail : {false, true}) {
            const auto operation = [&] {
                const apxchol::detail::thread_budget threads(budget);
                EXPECT_EQ(omp_get_max_threads(), budget);
                omp_set_num_threads(1);
                if (fail) throw std::runtime_error("operation failed");
            };
            if (fail) EXPECT_THROW(operation(), std::runtime_error);
            else operation();
            EXPECT_EQ(omp_get_max_threads(), 3);
        }
    }
}

TEST(Solver, ThreadBudgetRestoresCaller) {
    const Sparse matrix = operator_matrix();
    for (const int budget : {0, 1, 2, 3, 4})
        check_thread_budget<apxchol::solver>(matrix, 3, budget);
}

TEST(Solver, ThreadBudgetsAreLocalToIndependentHostThreads) {
    const Sparse matrix = operator_matrix();
    auto first = std::async(std::launch::async, [&] {
        check_thread_budget<apxchol::solver>(matrix, 3, 1);
    });
    auto second = std::async(std::launch::async, [&] {
        check_thread_budget<apxchol::solver>(matrix, 4, 2);
    });
    first.get();
    second.get();
}

#ifdef APXCHOL_USE_METAL
TEST(Solver, ThreadBudgetCoversMetalSolvers) {
    if (!apxchol::backend_available(apxchol::solve_backend::gpu)) GTEST_SKIP() << "no usable Metal device";
    check_thread_budget<apxchol::solver>(operator_matrix(), 3, 2, apxchol::solve_backend::gpu);
}
#endif

TEST(Solver, ThreadBudgetPreservesNestedTeamFallback) {
    const int previous = omp_get_max_active_levels();
    omp_set_max_active_levels(1);
    const Sparse matrix = operator_matrix(8192);
    #pragma omp parallel num_threads(2)
    {
        check_thread_budget<apxchol::solver>(matrix, 3, 2);
    }
    omp_set_max_active_levels(previous);
}

TEST(Solver, RejectsNegativeThreadBudget) {
    const Sparse matrix = operator_matrix();
    auto options = cpu_options();
    options.num_threads = -1;
    EXPECT_THROW(apxchol::solver(test::csc(matrix), options), std::invalid_argument);
}

namespace {
void check_verified_tolerance(apxchol::solve_backend backend) {
    const scoped_omp_threads threads(1);
    constexpr int n = 64;
    constexpr double tolerance = 1e-10;
    const Sparse identity = diagonal_matrix(n);
    std::vector<test::triplet> entries;
    for (int i = 0; i < n; ++i) {
        entries.push_back({i, i, (i == 0 || i == n - 1 ? 1.0 : 2.0) + 1e-7});
        if (i) {
            entries.push_back({i, i - 1, -1.0});
            entries.push_back({i - 1, i, -1.0});
        }
    }
    const Sparse matrix = test::from_triplets(n, n, entries);
    auto options = cpu_options();
    options.backend = backend;
    options.factor_opts.seed = 42;
    apxchol::solver owner(test::csc(matrix), apxchol::factorize(test::csc(identity), options.factor_opts),
                          options);
    Vector rhs(n);
    for (int i = 0; i < n; ++i) rhs[i] = (17 * i + 13) % 31 - 15 + 0.125;
    // The recursive residual can cross tolerance before the original residual.
    // The remaining budget must be available to correct that gap.
    for (int budget : {0, 1, 68, 512}) {
        SCOPED_TRACE(budget);
        const auto result = owner.solve(rhs, tolerance, budget);
        EXPECT_LE(result.iterations, budget);
        EXPECT_EQ(result.converged, result.residual < tolerance);
        if (budget == 512) {
            EXPECT_TRUE(result.converged);
            expect_solution(matrix, rhs, result.x, tolerance);
        }
    }
    Dense B(n, 4), X(n, 4), guess = Dense(n, 4);
    set_col(B, 0, rhs);
    std::ranges::fill(guess.col(1), 1.0);
    std::ranges::fill(guess.col(2), 1.0);
    set_col(B, 2, test::multiply(matrix, guess.col(2)));
    set_col(B, 3, affine(-2, rhs));
    for (int budget : {0, 1, 68, 512}) {
        SCOPED_TRACE(budget);
        const auto result = owner.solve(std::as_const(B).view(), X.view(), tolerance, budget, std::as_const(guess).view());
        for (int c = 0; c < 4; ++c) {
            const auto& column = result.columns[c];
            EXPECT_LE(column.iterations, budget);
            EXPECT_EQ(column.converged, column.residual < tolerance);
            if (budget == 512) {
                EXPECT_TRUE(column.converged);
                if (c != 1) expect_solution(matrix, B.col(c), X.col(c), tolerance);
            }
        }
        EXPECT_EQ(result.columns[1].iterations, 0);
        EXPECT_TRUE(is_zero(X.col(1)));
        EXPECT_EQ(result.columns[2].iterations, 0);
        EXPECT_TRUE(std::ranges::equal(X.col(2), guess.col(2)));

    }
}
}

TEST(Solver, CpuContinuesAfterFailedOriginalResidualVerification) {
    check_verified_tolerance(apxchol::solve_backend::cpu);
}

TEST(Solver, GpuContinuesAfterFailedOriginalResidualVerification) {
#if defined(APXCHOL_USE_CUDA) && defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "32-bit-node CUDA owner required";
#else
    if (!apxchol::backend_available(apxchol::solve_backend::gpu))
        GTEST_SKIP() << "GPU unavailable";
    check_verified_tolerance(apxchol::solve_backend::gpu);
#endif
}
