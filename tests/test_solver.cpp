#include <gtest/gtest.h>

#include "apxchol/solver/solve.h"
#include "apxchol/solver/detail/thread_budget.h"
#include "scoped_env.h"
#include "scoped_omp_threads.h"

#include <Eigen/Sparse>
#include <cmath>
#include <future>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(APXCHOL_USE_CUDA)
#include <cuda_runtime.h>
#elif defined(APXCHOL_USE_METAL)
#include "apxchol/solver/metal_solver.h"
#endif

namespace {
using Sparse = Eigen::SparseMatrix<double>;

static_assert(std::is_move_constructible_v<apxchol::solver>);
static_assert(std::is_move_assignable_v<apxchol::solver>);
static_assert(!std::is_copy_constructible_v<apxchol::solver>);
static_assert(!std::is_copy_assignable_v<apxchol::solver>);

Sparse operator_matrix(int n = 64) {
    std::vector<Eigen::Triplet<double>> entries;
    for (int i = 0; i < n; ++i) {
        entries.emplace_back(i, i, 5.0);
        for (const int offset : {1, 7}) {
            const int j = (i + offset) % n;
            entries.emplace_back(i, j, -1.0);
            entries.emplace_back(j, i, -1.0);
        }
    }
    Sparse matrix(n, n);
    matrix.setFromTriplets(entries.begin(), entries.end());
    matrix.makeCompressed();
    return matrix;
}

apxchol::solve_options cpu_options() {
    apxchol::solve_options options;
    options.backend = apxchol::solve_backend::cpu;
    options.factor_opts.factor_storage = apxchol::factor_storage_type::fp32;
    options.tol = 1e-10;
    options.max_iter = 400;
    return options;
}

template <class T>
bool same_bytes(const T* left, const T* right, std::size_t count) {
    return std::memcmp(left, right, count * sizeof(T)) == 0;
}

void expect_solution(const Sparse& matrix, const Eigen::VectorXd& rhs,
                     const Eigen::Ref<const Eigen::VectorXd>& x, double tolerance = 1e-10) {
    ASSERT_EQ(x.size(), matrix.rows());
    EXPECT_TRUE(x.allFinite());
    EXPECT_LT((matrix * x - rhs).norm() / rhs.norm(), tolerance);
}
}  // namespace

TEST(Solver, CpuOwnerSurvivesInputsAndMovesAndReusesItsFactor) {
    const scoped_omp_threads threads(1);
    const scoped_env checkpoints("APXCHOL_NO_CHECKPOINT", nullptr);
    const Sparse matrix = operator_matrix();
    const auto options = cpu_options();
    std::unique_ptr<apxchol::solver> original;
    {
        Sparse temporary = matrix;
        apxchol::checkpoint setup;
        original = std::make_unique<apxchol::solver>(temporary, options, &setup);
        EXPECT_GT(setup.total("setup"), 0.0);
    }
    apxchol::solver moved(std::move(*original));
    original.reset();
    apxchol::solver owner(matrix, options);
    owner = std::move(moved);
    EXPECT_EQ(owner.rows(), matrix.rows());
    EXPECT_EQ(owner.backend(), apxchol::solve_backend::cpu);
    EXPECT_EQ(owner.factor_storage(), apxchol::factor_storage_type::fp32);
    const apxchol::cpu_solver reference(matrix, options);
    for (const double start : {1.0, -3.0, 1.0}) {
        const Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(matrix.rows(), start, start + 1.0);
        const auto actual = owner.solve(rhs);
        const auto expected = reference.solve(rhs);
        EXPECT_EQ(actual.backend, apxchol::solve_backend::cpu);
        expect_solution(matrix, rhs, actual.x);
        EXPECT_EQ(actual.iterations, expected.iterations);
        EXPECT_EQ(actual.residual, expected.residual);
        EXPECT_TRUE(same_bytes(actual.x.data(), expected.x.data(), actual.x.size()));
    }
}

TEST(Solver, CpuOwnerAdoptsAnIndependentFactorAndAppliesIt) {
    const scoped_omp_threads threads(1);
    const Sparse matrix = operator_matrix();
    auto options = cpu_options();
    options.keep_factor_values = true;
    const auto expected = apxchol::factorize(matrix, options.factor_opts);
    std::unique_ptr<apxchol::solver> owner;
    {
        Sparse temporary = matrix;
        apxchol::factorization supplied = expected;
        owner = std::make_unique<apxchol::solver>(temporary, std::move(supplied), options);
    }
    const auto& actual = owner->factor();
    ASSERT_EQ(actual.L.nonZeros(), expected.L.nonZeros());
    ASSERT_EQ(actual.L.outer_.size(), expected.L.outer_.size());
    ASSERT_EQ(actual.L.inner_.size(), expected.L.inner_.size());
    ASSERT_EQ(actual.L.vals_.size(), expected.L.vals_.size());
    EXPECT_EQ(actual.perm, expected.perm);
    EXPECT_TRUE(same_bytes(actual.L.outer_.data(), expected.L.outer_.data(), actual.L.outer_.size()));
    EXPECT_TRUE(same_bytes(actual.L.inner_.data(), expected.L.inner_.data(), actual.L.inner_.size()));
    EXPECT_TRUE(same_bytes(actual.L.vals_.data(), expected.L.vals_.data(), actual.L.vals_.size()));
    const apxchol::cpu_solver reference(matrix, expected, options);
    const Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(matrix.rows(), -1.0, 2.0);
    const auto applied = owner->apply(rhs);
    const auto reference_applied = reference.apply(rhs);
    EXPECT_TRUE(same_bytes(applied.data(), reference_applied.data(), applied.size()));
    expect_solution(matrix, rhs, owner->solve(rhs).x);

    options.backend = apxchol::solve_backend::automatic;
    options.keep_factor_values = false;
    apxchol::solver automatic_owner(matrix, expected, options);
    EXPECT_EQ(automatic_owner.backend(), apxchol::solve_backend::cpu);
    EXPECT_TRUE(same_bytes(automatic_owner.apply(rhs).data(),
                          reference_applied.data(), reference_applied.size()));
    expect_solution(matrix, rhs, automatic_owner.solve(rhs).x);
}

TEST(Solver, CpuCallerOutputAndWarmStartsUseEachOwnersStorage) {
    const scoped_omp_threads threads(1);
    const Sparse matrix = operator_matrix();
    const Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(matrix.rows(), 0.5, 2.0);
    for (const auto storage : {apxchol::factor_storage_type::fp32,
                               apxchol::factor_storage_type::fp16}) {
        if (storage == apxchol::factor_storage_type::fp16 && !apxchol::omp_sptrsv::fp16_supported())
            continue;
        SCOPED_TRACE(static_cast<int>(storage));
        const scoped_env conflicting("APXCHOL_FACTOR_STORAGE",
            storage == apxchol::factor_storage_type::fp16 ? "float32" : "float16");
        auto options = cpu_options();
        options.factor_opts.factor_storage = storage;
        apxchol::solver owner(matrix, options);
        EXPECT_EQ(owner.factor_storage(), storage);
        const auto expected = owner.solve(rhs);
        std::vector<double> memory(static_cast<std::size_t>(matrix.rows() + 2), 1234.0);
        Eigen::Map<Eigen::VectorXd> output(memory.data() + 1, matrix.rows());
        const auto result = owner.solve(rhs, output);
        EXPECT_EQ(result.x.size(), 0);
        EXPECT_EQ(result.iterations, expected.iterations);
        EXPECT_EQ(result.residual, expected.residual);
        EXPECT_TRUE(same_bytes(output.data(), expected.x.data(), output.size()));
        expect_solution(matrix, rhs, output);
        const auto warm = owner.solve(rhs, output, options.tol, 0, &expected.x);
        EXPECT_EQ(warm.x.size(), 0);
        EXPECT_EQ(warm.iterations, 0);
        expect_solution(matrix, rhs, output);
        EXPECT_EQ(memory.front(), 1234.0);
        EXPECT_EQ(memory.back(), 1234.0);
    }
}

TEST(Solver, CpuSolveTimingsBelongToTheCurrentResult) {
    const scoped_omp_threads threads(1);
    const scoped_env checkpoints("APXCHOL_NO_CHECKPOINT", nullptr);
    const Sparse matrix = operator_matrix();
    apxchol::checkpoint setup;
    apxchol::solver owner(matrix, cpu_options(), &setup);
    const std::string setup_before = setup.report();
    const Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(matrix.rows(), 1.0, 2.0);
    auto first = std::make_unique<apxchol::solve_result>();
    owner.solve(rhs, *first);
    EXPECT_GT(first->timings.total("pcg"), 0.0);
    EXPECT_GT(first->timings.total("pcg.solve"), 0.0);
    EXPECT_EQ(first->timings.total("setup"), 0.0);
    const std::string first_before = first->timings.report();
    apxchol::solve_result second;
    owner.solve(-rhs, second);
    EXPECT_GT(second.timings.total("pcg"), 0.0);
    EXPECT_GT(second.timings.total("pcg.solve"), 0.0);
    EXPECT_EQ(first->timings.report(), first_before);
    EXPECT_EQ(setup.report(), setup_before);
    auto moved = std::move(*first);
    first.reset();
    auto copied = moved;
    owner.solve(rhs, moved);
    owner.solve(-rhs, copied);
    expect_solution(matrix, rhs, moved.x);
    expect_solution(matrix, -rhs, copied.x);
    const double second_before = second.timings.total("pcg");
    {
        apxchol::solve_result temporary;
        owner.solve(rhs, temporary);
        expect_solution(matrix, rhs, temporary.x);
    }
    EXPECT_TRUE(owner.apply(rhs).allFinite());
    owner.solve(rhs, second);
    EXPECT_GT(second.timings.total("pcg"), second_before);
    EXPECT_EQ(setup.report(), setup_before);
    expect_solution(matrix, rhs, second.x);
}

TEST(Solver, CpuScalesExtremeRhsAndWarmStarts) {
    const scoped_omp_threads threads(1);
    Sparse matrix(16, 16);
    matrix.setIdentity();
    const Eigen::VectorXd unit = Eigen::VectorXd::LinSpaced(matrix.rows(), 1.0, 2.0);
    for (const auto storage : {apxchol::factor_storage_type::fp32,
                               apxchol::factor_storage_type::fp16}) {
        if (storage == apxchol::factor_storage_type::fp16 && !apxchol::omp_sptrsv::fp16_supported())
            continue;
        auto options = cpu_options();
        options.factor_opts.factor_storage = storage;
        apxchol::solver owner(matrix, options);
        for (const double magnitude : {1e-200, 1e-160, 1.0, 1e160, 1e200}) {
            SCOPED_TRACE(::testing::Message() << static_cast<int>(storage) << ' ' << magnitude);
            const Eigen::VectorXd rhs = magnitude * unit;
            const Eigen::VectorXd x0 = 0.25 * rhs;
            for (const Eigen::VectorXd* guess : {static_cast<const Eigen::VectorXd*>(nullptr), &x0}) {
                const auto result = owner.solve(rhs, options.tol, options.max_iter, guess);
                EXPECT_TRUE(result.x.allFinite());
                EXPECT_LT(result.residual, options.tol);
                EXPECT_LT((result.x - rhs).stableNorm() / rhs.stableNorm(), options.tol);
            }
            Eigen::VectorXd output(matrix.rows());
            const auto exact = owner.solve(rhs, output, options.tol, 0, &rhs);
            EXPECT_EQ(exact.iterations, 0);
            EXPECT_EQ(exact.residual, 0.0);
            EXPECT_EQ(output, rhs);
            const auto stopped = owner.solve(rhs, options.tol, 0);
            EXPECT_EQ(stopped.iterations, 0);
            EXPECT_EQ(stopped.residual, 1.0);
            EXPECT_TRUE(stopped.x.isZero(0.0));
        }
        const auto zero = owner.solve(Eigen::VectorXd::Zero(matrix.rows()));
        EXPECT_EQ(zero.residual, 0.0);
        EXPECT_TRUE(zero.x.isZero(0.0));
        const Eigen::VectorXd tiny = 1e-200 * unit, huge = 1e200 * unit;
        EXPECT_THROW(owner.solve(tiny, options.tol, 0, &huge), std::invalid_argument);
    }
}

TEST(Solver, CpuPanelsScaleExceptionalColumnsIndependently) {
    const scoped_omp_threads threads(2);
    const Sparse matrix = operator_matrix(65);
    Eigen::MatrixXd rhs(matrix.rows(), 15), guess = rhs, output = rhs;
    const double magnitudes[] = {1.0, 1e-200, 1e200, 0.0, 1e-160, 1e160};
    for (int c = 0; c < rhs.cols(); ++c) {
        const Eigen::VectorXd exact = magnitudes[c % 6] *
            Eigen::VectorXd::LinSpaced(matrix.rows(), -1.0, 1.0);
        rhs.col(c) = matrix * exact;
        guess.col(c) = 0.25 * exact;
    }
    const apxchol::const_matrix_view initial(guess);
    for (const auto storage : {apxchol::factor_storage_type::fp32,
                               apxchol::factor_storage_type::fp16}) {
        if (storage == apxchol::factor_storage_type::fp16 && !apxchol::omp_sptrsv::fp16_supported())
            continue;
        auto options = cpu_options();
        options.factor_opts.factor_storage = storage;
        const apxchol::solver owner(matrix, options);
        for (const bool warm : {false, true})
            for (const int budget : {0, 1, 200}) {
                SCOPED_TRACE(::testing::Message() << int(storage) << ' ' << warm << ' ' << budget);
                const auto batch = owner.solve(rhs, output, 1e-8, budget, warm ? &initial : nullptr);
                for (int c = 0; c < rhs.cols(); ++c) {
                    const Eigen::VectorXd b = rhs.col(c), x0 = guess.col(c);
                    const auto scalar = owner.solve(b, 1e-8, budget, warm ? &x0 : nullptr);
                    ASSERT_TRUE(output.col(c).allFinite());
                    EXPECT_EQ(batch.columns[c].iterations, scalar.iterations);
                    EXPECT_NEAR(batch.columns[c].residual, scalar.residual, 1e-12);
                    if (b.isZero(0.0)) {
                        EXPECT_TRUE(output.col(c).isZero(0.0));
                    } else {
                        const double residual = (b - matrix * output.col(c)).stableNorm() / b.stableNorm();
                        EXPECT_NEAR(batch.columns[c].residual, residual, 1e-12);
                        if (budget == 200) {
                            EXPECT_LT(residual, 1e-8);
                        }
                    }
                }
            }
        Eigen::MatrixXd bad_guess = guess;
        bad_guess.col(1).setConstant(1e200);
        const apxchol::const_matrix_view incompatible(bad_guess);
        EXPECT_THROW(owner.solve(rhs, output, 1e-8, 0, &incompatible), std::invalid_argument);
    }
}

TEST(Solver, CpuRejectsUnrepresentableSolutions) {
    const scoped_omp_threads threads(1);
    const auto options = cpu_options();
    for (const double diagonal : {1e-30, 1e30}) {
        Sparse matrix(4, 4);
        matrix.setIdentity();
        matrix *= diagonal;
        apxchol::solver owner(matrix, options);
        const Eigen::VectorXd rhs = Eigen::VectorXd::Constant(4, diagonal < 1 ? 1e275 : 1e-275);
        const auto finite = owner.solve(rhs);
        ASSERT_TRUE(finite.x.allFinite());
        EXPECT_LT((matrix * finite.x - rhs).stableNorm() / rhs.stableNorm(), options.tol);
        const Eigen::VectorXd outside_range = Eigen::VectorXd::Constant(4, diagonal < 1 ? 1e300 : 1e-300);
        Eigen::MatrixXd panel(matrix.rows(), 4), output(matrix.rows(), 4);
        panel.col(0) = rhs;
        panel.col(1).setZero();
        panel.col(2) = outside_range;
        panel.col(3) = rhs;
        if (diagonal < 1) {
            EXPECT_THROW(owner.solve(outside_range), std::overflow_error);
            EXPECT_THROW(owner.solve(panel, output), std::overflow_error);
        } else {
            EXPECT_THROW(owner.solve(panel, output), std::underflow_error);
            EXPECT_THROW(owner.solve(outside_range), std::underflow_error);
            matrix.coeffRef(0, 0) = 1;
            apxchol::solver mixed(matrix, options);
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
    const Sparse matrix = operator_matrix();
    auto options = cpu_options();
    options.backend = apxchol::solve_backend::gpu;
    std::unique_ptr<apxchol::solver> owner;
    {
        Sparse temporary = matrix;
        apxchol::checkpoint setup;
        owner = std::make_unique<apxchol::solver>(temporary, options, &setup);
    }
    EXPECT_EQ(owner->rows(), matrix.rows());
    EXPECT_EQ(owner->backend(), apxchol::solve_backend::gpu);
    EXPECT_EQ(owner->factor_storage(), apxchol::factor_storage_type::fp32);
    EXPECT_EQ(owner->factor().perm.size(), static_cast<std::size_t>(matrix.rows()));
    EXPECT_GT(owner->factor().L.nonZeros(), 0);
    for (const double start : {1.0, -2.0}) {
        const Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(matrix.rows(), start, start + 1.0);
        const auto result = owner->solve(rhs);
        EXPECT_EQ(result.backend, apxchol::solve_backend::gpu);
        expect_solution(matrix, rhs, result.x);
        Eigen::VectorXd output(matrix.rows());
        const auto into = owner->solve(rhs, output);
        EXPECT_EQ(into.backend, apxchol::solve_backend::gpu);
        EXPECT_EQ(into.x.size(), 0);
        expect_solution(matrix, rhs, output);
        const auto applied = owner->apply(rhs);
        ASSERT_EQ(applied.size(), matrix.rows());
        EXPECT_TRUE(applied.allFinite());
        EXPECT_GT(applied.dot(rhs), 0.0);
    }
#endif
}

TEST(Solver, CudaOwnerRejectsHostFactorsAndInvalidWarmStarts) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "32-bit-node CUDA owner required";
#else
    const Sparse matrix = operator_matrix();
    auto options = cpu_options();
    const auto host_factor = apxchol::factorize(matrix, options.factor_opts);
    options.backend = apxchol::solve_backend::gpu;
    EXPECT_THROW(apxchol::solver(matrix, host_factor, options), std::invalid_argument);
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver)
        GTEST_SKIP() << "CUDA unavailable: " << cudaGetErrorString(status);
    ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
    if (count == 0) GTEST_SKIP() << "No CUDA devices";
    apxchol::solver owner(matrix, options);
    const Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(matrix.rows(), 1.0, 2.0);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(matrix.rows() - 1);
    EXPECT_THROW(owner.solve(rhs, -1.0, -1, &x0), std::invalid_argument);
    Eigen::VectorXd output(matrix.rows());
    EXPECT_THROW(owner.solve(rhs, output, -1.0, -1, &x0), std::invalid_argument);
    apxchol::solve_result result;
    EXPECT_THROW(owner.solve(rhs, result, -1.0, -1, &x0), std::invalid_argument);
    x0 = Eigen::VectorXd::Zero(matrix.rows());
    for (const double value : {std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
        x0[0] = value;
        EXPECT_THROW(owner.solve(rhs, result, -1.0, -1, &x0), std::invalid_argument);
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
    const scoped_env checkpoints("APXCHOL_NO_CHECKPOINT", nullptr);
    constexpr int n = 64;
    // SDDM, connected Laplacian, then two disconnected Laplacian components.
    for (const int kind : {0, 1, 2}) {
        std::vector<Eigen::Triplet<double>> entries;
        const int component_size = kind == 2 ? n / 2 : n;
        for (int i = 0; i < n; ++i) {
            entries.emplace_back(i, i, kind == 0 ? 5.0 : 4.0);
            for (const int offset : {1, 7}) {
                const int j = i / component_size * component_size
                    + (i + offset) % component_size;
                entries.emplace_back(i, j, -1.0);
                entries.emplace_back(j, i, -1.0);
            }
        }
        Sparse matrix(n, n);
        matrix.setFromTriplets(entries.begin(), entries.end());
        const Eigen::VectorXd exact = Eigen::VectorXd::LinSpaced(n, -31.5, 31.5);
        const Eigen::VectorXd rhs = matrix * exact;
        const Eigen::VectorXd zero = Eigen::VectorXd::Zero(n);
        for (const auto storage : {apxchol::factor_storage_type::fp32,
                                   apxchol::factor_storage_type::fp16}) {
            SCOPED_TRACE(::testing::Message() << kind << ' ' << static_cast<int>(storage));
            auto options = cpu_options();
            options.backend = apxchol::solve_backend::gpu;
            options.factor_opts.factor_storage = storage;
            apxchol::solver owner(matrix, options);
            const double shift = kind == 0 ? 0.0 : 8.0;
            Eigen::VectorXd x0 = exact.array() + shift;
            auto result = owner.solve(rhs, options.tol, options.max_iter, &x0);
            EXPECT_EQ(result.iterations, 0);
            EXPECT_DOUBLE_EQ(result.residual, 0.0);
            EXPECT_EQ(result.x, exact);
            if (kind != 0) {
                const Eigen::VectorXd incompatible = rhs.array() + 1.0;
                result = owner.solve(incompatible, options.tol, 0, &x0);
                EXPECT_NEAR(result.residual, std::sqrt(double(n)) / incompatible.norm(), 1e-14);
                EXPECT_GT(result.residual, options.tol);
            }

            x0 = ((1.0 - 0x1p-20) * exact).array() + shift;
            result = owner.solve(rhs, 0x1p-19, options.max_iter, &x0);
            EXPECT_EQ(result.iterations, 0);
            EXPECT_NEAR(result.residual, 0x1p-20, 1e-14);

            x0 = (0.25 * exact).array() + shift;
            result = owner.solve(rhs, options.tol, 0, &x0);
            EXPECT_EQ(result.iterations, 0);
            EXPECT_NEAR(result.residual, 0.75, 1e-14);
            EXPECT_NEAR((matrix * result.x - rhs).norm() / rhs.norm(), result.residual, 1e-14);
            EXPECT_EQ(result.x, 0.25 * exact);
            result = owner.solve(rhs, options.tol, 1, &x0);
            EXPECT_EQ(result.iterations, 1);
            EXPECT_NEAR((matrix * result.x - rhs).norm() / rhs.norm(), result.residual, 1e-12);

            Eigen::VectorXd output(n);
            const auto into = owner.solve(rhs, output, options.tol, options.max_iter, &x0);
            EXPECT_GT(into.iterations, 0);
            EXPECT_EQ(into.x.size(), 0);
            expect_solution(matrix, rhs, output);
            if (kind != 0) EXPECT_NEAR(output.mean(), 0.0, 1e-12);
            const double before = result.timings.total("pcg");
            owner.solve(rhs, result, options.tol, options.max_iter, &x0);
            EXPECT_GT(result.timings.total("pcg"), before);
            expect_solution(matrix, rhs, result.x);

            const auto cold = owner.solve(rhs);
            result = owner.solve(rhs, options.tol, options.max_iter, &zero);
            EXPECT_EQ(result.iterations, cold.iterations);
            EXPECT_EQ(result.x, cold.x);
            EXPECT_EQ(result.residual, cold.residual);
            result = owner.solve(zero, options.tol, 0, &x0);
            EXPECT_EQ(result.iterations, 0);
            EXPECT_EQ(result.residual, 0.0);
            EXPECT_EQ(result.x, zero);
        }
    }
#endif
}

TEST(Solver, CudaOwnerHonorsStagnationWindow) {
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
    const Eigen::VectorXd exact = Eigen::VectorXd::LinSpaced(matrix.rows(), 1.0, 2.0);
    const Eigen::VectorXd rhs = matrix * exact;
    const Eigen::VectorXd x0 = -1e6 * exact;
    for (const int window : {1, 0, -1}) {
        SCOPED_TRACE(window);
        auto options = cpu_options();
        options.backend = apxchol::solve_backend::gpu;
        options.stagnation_window = window;
        apxchol::solver owner(matrix, options);
        const auto first = owner.solve(rhs, options.tol, 1, &x0);
        ASSERT_EQ(first.iterations, 1);
        ASSERT_GT(first.residual, 0.5);
        const auto next = owner.solve(rhs, options.tol, 2, &x0);
        EXPECT_EQ(next.iterations, window > 0 ? 1 : 2);
        if (window > 0) {
            EXPECT_EQ(next.x, first.x);
            EXPECT_EQ(next.residual, first.residual);
        }
    }
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
    Sparse matrix(16, 16);
    matrix.setIdentity();
    const Eigen::VectorXd unit = Eigen::VectorXd::LinSpaced(matrix.rows(), 1.0, 2.0);
    for (const auto storage : {apxchol::factor_storage_type::fp16,
                               apxchol::factor_storage_type::fp32}) {
        auto options = cpu_options();
        options.backend = apxchol::solve_backend::gpu;
        options.factor_opts.factor_storage = storage;
        apxchol::solver owner(matrix, options);
        for (const double magnitude : {1e-100, 1.0, 1e100}) {
            SCOPED_TRACE(::testing::Message() << static_cast<int>(storage) << ' ' << magnitude);
            const Eigen::VectorXd rhs = magnitude * unit;
            const Eigen::VectorXd x0 = 0.25 * rhs;
            const auto cold = owner.solve(rhs);
            const auto warm = owner.solve(rhs, options.tol, options.max_iter, &x0);
            for (const auto* result : {&cold, &warm}) {
                EXPECT_GT(result->iterations, 0);
                EXPECT_LT(result->residual, options.tol);
                EXPECT_LT((matrix * result->x - rhs).stableNorm() / rhs.stableNorm(), 1e-9);
            }
            const auto exact = owner.solve(rhs, options.tol, 0, &rhs);
            EXPECT_EQ(exact.iterations, 0);
            EXPECT_EQ(exact.residual, 0.0);
            EXPECT_EQ(exact.x, rhs);
            const auto applied = owner.apply(rhs);
            EXPECT_LT((applied - rhs).stableNorm() / rhs.stableNorm(), 1e-6);
        }
        for (const double value : {std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::quiet_NaN()}) {
            const Eigen::VectorXd invalid = Eigen::VectorXd::Constant(matrix.rows(), value);
            EXPECT_THROW(owner.solve(invalid), std::invalid_argument);
            EXPECT_THROW(owner.apply(invalid), std::invalid_argument);
        }
        const Eigen::VectorXd tiny = 1e-100 * unit, huge = 1e200 * unit;
        EXPECT_THROW(owner.solve(tiny, options.tol, 0, &huge), std::invalid_argument);
    }
#endif
}

#if defined(APXCHOL_USE_METAL)
TEST(Solver, MetalOwnerPreservesStorageAndAdoptsItsFactor) {
    if (!apxchol::metal_solver::available()) GTEST_SKIP() << "no usable Metal device";
    const scoped_omp_threads threads(1);
    const Sparse matrix = operator_matrix();
    const Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(matrix.rows(), 1.0, 2.0);
    for (const auto storage : {apxchol::factor_storage_type::fp32,
                              apxchol::factor_storage_type::fp16}) {
        SCOPED_TRACE(static_cast<int>(storage));
        auto options = cpu_options();
        options.backend = apxchol::solve_backend::gpu;
        options.factor_opts.factor_storage = storage;
        options.keep_factor_values = true;
        std::unique_ptr<apxchol::solver> owner;
        {
            Sparse temporary = matrix;
            apxchol::checkpoint setup;
            owner = std::make_unique<apxchol::solver>(temporary, options, &setup);
        }
        EXPECT_EQ(owner->backend(), apxchol::solve_backend::gpu);
        EXPECT_EQ(owner->rows(), matrix.rows());
        EXPECT_EQ(owner->factor_storage(), storage);
        apxchol::solver adopted(matrix, owner->factor(), options);
        EXPECT_EQ(adopted.backend(), apxchol::solve_backend::gpu);
        EXPECT_EQ(adopted.factor_storage(), storage);
        const auto result = owner->solve(rhs);
        const auto expected = adopted.solve(rhs);
        EXPECT_EQ(result.backend, apxchol::solve_backend::gpu);
        expect_solution(matrix, rhs, result.x);
        ASSERT_EQ(result.x.size(), expected.x.size());
        EXPECT_TRUE(same_bytes(result.x.data(), expected.x.data(), result.x.size()));
        const auto applied = owner->apply(rhs);
        const auto reference = adopted.apply(rhs);
        ASSERT_EQ(applied.size(), reference.size());
        EXPECT_TRUE(same_bytes(applied.data(), reference.data(), applied.size()));
        Eigen::VectorXd output(matrix.rows());
        const auto warm = adopted.solve(rhs, output, options.tol, 0, &result.x);
        EXPECT_EQ(warm.iterations, 0);
        EXPECT_EQ(warm.x.size(), 0);
        expect_solution(matrix, rhs, output);
        expect_solution(matrix, -rhs, owner->solve(-rhs).x);
    }
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
    if (!apxchol::metal_solver::available()) GTEST_SKIP() << "No usable Metal device";
#endif
    const scoped_omp_threads threads(2);
    for (const bool laplacian : {false, true}) {
        Sparse matrix = operator_matrix(65);
        if (laplacian)
            for (int i = 0; i < matrix.rows(); ++i) matrix.coeffRef(i, i) -= 1.0;
        matrix *= 1.1;
        Eigen::MatrixXd exact(matrix.rows(), 5);
        const double magnitudes[] = {1.0, 0.0, 1e-200, 1e200, -1.0};
        for (int q = 0; q < exact.cols(); ++q)
            exact.col(q) = magnitudes[q] *
                Eigen::VectorXd::LinSpaced(matrix.rows(), -1.0, 1.0);
        const Eigen::MatrixXd rhs = matrix * exact;
        Eigen::MatrixXd guess = 0.25 * exact, output(matrix.rows(), exact.cols());
        guess.col(0) = exact.col(0);
        guess.col(1).setOnes();
        const apxchol::const_matrix_view initial(guess);
        for (const auto storage : {apxchol::factor_storage_type::fp16,
                                   apxchol::factor_storage_type::fp32}) {
            auto options = cpu_options();
            options.backend = apxchol::solve_backend::gpu;
            options.factor_opts.factor_storage = storage;
            const apxchol::solver owner(matrix, options);
            for (const bool warm : {false, true}) {
                for (const int budget : {0, 1, 200}) {
                    SCOPED_TRACE(::testing::Message() << laplacian << ' '
                        << int(storage) << ' ' << warm << ' ' << budget);
                    const auto batch = owner.solve(rhs, output, 1e-8, budget,
                                                   warm ? &initial : nullptr);
                    ASSERT_EQ(batch.columns.size(), 5u);
                    for (int q = 0; q < rhs.cols(); ++q) {
                        const Eigen::VectorXd b = rhs.col(q), x0 = guess.col(q);
                        const auto scalar = owner.solve(b, 1e-8, budget, warm ? &x0 : nullptr);
                        const auto check = [&](const apxchol::solve_result& result,
                                               const Eigen::VectorXd& x) {
                            ASSERT_TRUE(x.allFinite());
                            const double norm = b.stableNorm();
                            const double measured = norm ? (b - matrix * x).stableNorm() / norm : 0;
                            if (!norm) EXPECT_TRUE(x.isZero(0.0));
                            EXPECT_LE(result.iterations, budget);
                            EXPECT_NEAR(result.residual, measured, 1e-12 * (1 + measured));
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
    Sparse identity(101, 101);
    identity.setIdentity();
    Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(101, 2), guess = rhs, output = rhs;
    rhs(0, 0) = 2;
    rhs.col(0).segment(1, 4).setConstant(std::numeric_limits<double>::denorm_min());
    rhs(0, 1) = 1e100;
    rhs.col(1).tail(100).setConstant(1e-224);
    guess.row(0) = rhs.row(0);
    const apxchol::const_matrix_view initial(guess);
    auto options = cpu_options();
    options.backend = apxchol::solve_backend::gpu;
    const apxchol::solver owner(identity, options);
    const auto batch = owner.solve(rhs, output, 1e-8, 0, &initial);
    ASSERT_EQ(batch.columns.size(), 2u);
    for (int q = 0; q < rhs.cols(); ++q) {
        const Eigen::VectorXd b = rhs.col(q), x0 = guess.col(q);
        const auto scalar = owner.solve(b, 1e-8, 0, &x0);
        const double expected = (b - x0).stableNorm() / b.stableNorm();
        ASSERT_GT(expected, 0);
        EXPECT_EQ(batch.columns[q].residual, expected);
        EXPECT_EQ(scalar.residual, expected);
        EXPECT_TRUE(output.col(q).isApprox(x0, 0));
        EXPECT_TRUE(scalar.x.isApprox(x0, 0));
    }
#endif
}
#endif

TEST(Solver, CpuMatrixRhsUsesIndependentStopsAndOneTimingTree) {
    const scoped_omp_threads threads(2);
    const scoped_env checkpoints("APXCHOL_NO_CHECKPOINT", nullptr);
    constexpr int side = 96, n = side * side;
    std::vector<Eigen::Triplet<double>> entries;
    std::vector<double> diagonal(n);
    auto edge = [&](int a, int b) {
        entries.emplace_back(a, b, -1.0);
        entries.emplace_back(b, a, -1.0);
        ++diagonal[a]; ++diagonal[b];
    };
    for (int i = 0; i < n; ++i) {
        if (i % side) edge(i - 1, i);
        if (i >= side) edge(i - side, i);
    }
    for (int i = 0; i < n; ++i) entries.emplace_back(i, i, diagonal[i]);
    Sparse matrix(n, n);
    matrix.setFromTriplets(entries.begin(), entries.end());
    auto options = cpu_options();
    apxchol::solver owner(matrix, options);
    Eigen::MatrixXd rhs(n, 10), x0 = Eigen::MatrixXd::Zero(n, 10), x(n, 10);
    for (int c = 0; c < rhs.cols(); ++c) {
        for (int i = 0; i < n; ++i) rhs(i, c) = std::sin(double(i + 1) * (c + 1));
        rhs.col(c).array() -= rhs.col(c).mean();
    }
    rhs.col(0).setZero();
    x0.col(0).setOnes();
    x0.col(1) = Eigen::VectorXd::LinSpaced(n, -1.0, 1.0);
    rhs.col(1) = matrix * x0.col(1);
    const apxchol::const_matrix_view initial(x0);
    const auto batch = owner.solve(rhs, x, 1e-10, 200, &initial);
    ASSERT_EQ(batch.columns.size(), 10u);
    EXPECT_EQ(batch.columns[0].iterations, 0);
    EXPECT_EQ(batch.columns[1].iterations, 0);
    EXPECT_TRUE(x.col(0).isZero(0.0));
    EXPECT_GT(batch.timings.total("pcg"), 0.0);
    for (int c = 0; c < rhs.cols(); ++c) {
        EXPECT_EQ(batch.columns[c].x.size(), 0);
        EXPECT_DOUBLE_EQ(batch.columns[c].timings.total("pcg"), 0.0);
        EXPECT_EQ(batch.columns[c].backend, apxchol::solve_backend::cpu);
        if (c > 0) expect_solution(matrix, rhs.col(c), x.col(c), 1e-9);
        const Eigen::VectorXd b = rhs.col(c), guess = x0.col(c);
        const auto scalar = owner.solve(b, 1e-10, 200, &guess);
        EXPECT_LE((x.col(c) - scalar.x).norm(), 1e-7 * std::max(1.0, scalar.x.norm()));
    }
    const auto exhausted = owner.solve(rhs, x, 1e-10, 0);
    for (int c = 0; c < rhs.cols(); ++c) {
        EXPECT_EQ(exhausted.columns[c].iterations, 0);
        EXPECT_DOUBLE_EQ(exhausted.columns[c].residual, c ? 1.0 : 0.0);
    }
}

TEST(Solver, CpuMatrixRhsRejectsAliasingAndAcceptsEmptyBatches) {
    const scoped_omp_threads threads(1);
    apxchol::solver owner(operator_matrix(), cpu_options());
    Eigen::MatrixXd rhs = Eigen::MatrixXd::Ones(owner.rows(), 3);
    EXPECT_THROW(owner.solve(rhs, rhs), std::invalid_argument);
    EXPECT_THROW(owner.apply(rhs, rhs), std::invalid_argument);
    Eigen::MatrixXd wrong(owner.rows() - 1, 3);
    EXPECT_THROW(owner.solve(rhs, wrong), std::invalid_argument);
    Eigen::MatrixXd empty(owner.rows(), 0), output(owner.rows(), 0);
    const auto result = owner.solve(empty, output);
    EXPECT_TRUE(result.columns.empty());
    EXPECT_DOUBLE_EQ(result.timings.total("pcg"), 0.0);
    EXPECT_NO_THROW(owner.apply(empty, output));
}

TEST(Solver, CpuMatrixRhsKeepsConvergedColumnsWhileOtherColumnsContinue) {
    const scoped_omp_threads threads(1);
    constexpr int n = 17;
    Sparse identity(n, n);
    identity.setIdentity();
    Sparse matrix = identity;
    matrix.diagonal() = Eigen::VectorXd::LinSpaced(n, 1.0, double(n));
    apxchol::cpu_solver owner(matrix, apxchol::factorize(identity), cpu_options());
    Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(n, 4), guess = rhs, x(n, 4);
    guess(n - 1, 0) = 1;
    rhs.col(0) = matrix * guess.col(0);
    rhs(0, 2) = 1;
    rhs.col(3).setOnes();
    const apxchol::const_matrix_view initial(guess);
    const auto result = owner.solve(rhs, x, 1e-10, 200, &initial);
    ASSERT_EQ(result.columns.size(), 4u);
    EXPECT_EQ(result.columns[0].iterations, 0);
    EXPECT_EQ(result.columns[1].iterations, 0);
    EXPECT_EQ(result.columns[2].iterations, 1);
    EXPECT_GT(result.columns[3].iterations, 1);
    EXPECT_TRUE(x.col(0).isApprox(guess.col(0), 0.0));
    EXPECT_TRUE(x.col(1).isZero(0.0));
    EXPECT_TRUE(x.col(2).isApprox(rhs.col(2), 0.0));
    for (int c : {0, 2, 3})
        expect_solution(matrix, rhs.col(c), x.col(c));
}

TEST(Solver, CpuSingletonPanelsRespectStridesWarmStartsAndBudgets) {
    const scoped_omp_threads threads(2);
    const Sparse matrix = operator_matrix();
    apxchol::solver owner(matrix, cpu_options());
    const Eigen::Index n = owner.rows();
    using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    using View = Eigen::Map<Eigen::MatrixXd, 0, Stride>;
    for (int columns : {1, 9}) {
        Eigen::MatrixXd rhs(n, columns), guess = Eigen::MatrixXd::Ones(n, columns);
        for (int c = 0; c < columns; ++c)
            for (Eigen::Index i = 0; i < n; ++i)
                rhs(i, c) = std::sin(double(i + 1) * (c + 1));
        rhs.col(columns - 1) = matrix * guess.col(columns - 1);
        if (columns > 1) rhs.col(0).setZero();
        const apxchol::const_matrix_view initial(guess);
        for (int row_stride : {2, -2}) {
            Eigen::MatrixXd stored_rhs = Eigen::MatrixXd::Constant(2 * n, columns, 91.0);
            Eigen::MatrixXd stored_guess = stored_rhs, stored_x = stored_rhs;
            const Eigen::Index offset = row_stride < 0 ? 2 * n - 2 : 0;
            const Stride stride(2 * n, row_stride);
            View strided_rhs(stored_rhs.data() + offset, n, columns, stride);
            View strided_guess(stored_guess.data() + offset, n, columns, stride);
            View strided_x(stored_x.data() + offset, n, columns, stride);
            strided_rhs = rhs;
            strided_guess = guess;
            const apxchol::const_matrix_view strided_initial(strided_guess);
            for (bool warm : {false, true})
                for (int budget : {0, 1, 200}) {
                    Eigen::MatrixXd x(n, columns);
                    const auto direct = owner.solve(rhs, x, 1e-10, budget,
                                                     warm ? &initial : nullptr);
                    const auto packed = owner.solve(strided_rhs, strided_x, 1e-10, budget,
                                                     warm ? &strided_initial : nullptr);
                    for (int c = 0; c < columns; ++c) {
                        EXPECT_EQ(direct.columns[c].iterations, packed.columns[c].iterations);
                        EXPECT_DOUBLE_EQ(direct.columns[c].residual, packed.columns[c].residual);
                        EXPECT_LE((x.col(c) - strided_x.col(c)).norm(),
                                  1e-12 * std::max(1.0, x.col(c).norm()));
                        if (budget == 0 || (warm && c == columns - 1)) {
                            EXPECT_EQ(direct.columns[c].iterations, 0);
                        }
                        if (budget == 200 && !rhs.col(c).isZero(0))
                            expect_solution(matrix, rhs.col(c), x.col(c));
                    }
                }
            Eigen::MatrixXd applied(n, columns);
            owner.apply(rhs, applied);
            owner.apply(strided_rhs, strided_x);
            EXPECT_LE((applied - strided_x).norm(), 1e-12 * std::max(1.0, applied.norm()));
            for (Eigen::Index i = 1; i < 2 * n; i += 2)
                EXPECT_TRUE(stored_x.row(i).isConstant(91.0));
        }
        EXPECT_THROW(owner.solve(rhs, rhs), std::invalid_argument);
        EXPECT_THROW(owner.solve(rhs, guess, 1e-10, 200, &initial), std::invalid_argument);
    }
}

#ifdef _OPENMP
namespace {
template<class Owner>
void check_thread_budget(const Sparse& matrix, int ambient, int budget,
                         apxchol::solve_backend backend = apxchol::solve_backend::cpu) {
    const scoped_omp_threads caller(ambient);
    auto options = cpu_options();
    options.backend = backend;
    options.num_threads = budget;
    options.keep_factor_values = true;
    const Owner owner(matrix, options);
    EXPECT_EQ(omp_get_max_threads(), ambient);
    const Owner adopted(matrix, owner.factor(), options);
    const Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(matrix.rows(), 1.0, 2.0);
    for (int repetition = 0; repetition < 2; ++repetition) {
        expect_solution(matrix, rhs, owner.solve(rhs).x);
        expect_solution(matrix, rhs, adopted.solve(rhs).x);
        EXPECT_EQ(omp_get_max_threads(), ambient);
        EXPECT_TRUE(owner.apply(rhs).allFinite());
        EXPECT_EQ(omp_get_max_threads(), ambient);
        Eigen::VectorXd output(matrix.rows());
        if constexpr (requires { owner.solve(rhs, Eigen::Ref<Eigen::VectorXd>(output)); }) {
            owner.solve(rhs, Eigen::Ref<Eigen::VectorXd>(output));
            expect_solution(matrix, rhs, output);
        }
        EXPECT_EQ(omp_get_max_threads(), ambient);
    }
    Eigen::MatrixXd rhs_panel = rhs.replicate(1, 5), output(matrix.rows(), 5);
    if constexpr (requires { owner.solve(rhs_panel, output); owner.apply(rhs_panel, output); }) {
        const auto batch = owner.solve(rhs_panel, output);
        EXPECT_EQ(omp_get_max_threads(), ambient);
        EXPECT_EQ(batch.columns.size(), 5u);
        for (int c = 0; c < output.cols(); ++c) expect_solution(matrix, rhs, output.col(c));
        owner.apply(rhs_panel, output);
        EXPECT_TRUE(output.allFinite());
        EXPECT_EQ(omp_get_max_threads(), ambient);
        EXPECT_THROW(owner.solve(rhs_panel, rhs_panel), std::invalid_argument);
        EXPECT_EQ(omp_get_max_threads(), ambient);
    }
    const Eigen::VectorXd wrong_guess = Eigen::VectorXd::Zero(1);
    EXPECT_THROW(owner.solve(rhs, -1, -1, &wrong_guess), std::invalid_argument);
    EXPECT_EQ(omp_get_max_threads(), ambient);
    auto invalid = options;
    invalid.factor_opts.is_select = "not_a_partitioner";
    EXPECT_THROW(Owner(matrix, invalid), std::invalid_argument);
    EXPECT_EQ(omp_get_max_threads(), ambient);
}

// Direct CPU owners expose their factor through the preconditioner.
struct budget_cpu_solver : apxchol::cpu_solver {
    using cpu_solver::cpu_solver;
    const apxchol::factorization& factor() const { return preconditioner().factor(); }
};
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

TEST(Solver, ThreadBudgetRestoresCallerForBothOwners) {
    const Sparse matrix = operator_matrix();
    for (const int budget : {0, 1, 2, 3, 4}) {
        check_thread_budget<apxchol::solver>(matrix, 3, budget);
        check_thread_budget<budget_cpu_solver>(matrix, 3, budget);
    }
}

TEST(Solver, ThreadBudgetsAreLocalToIndependentHostThreads) {
    const Sparse matrix = operator_matrix();
    auto first = std::async(std::launch::async, [&] {
        check_thread_budget<apxchol::solver>(matrix, 3, 1);
    });
    auto second = std::async(std::launch::async, [&] {
        check_thread_budget<budget_cpu_solver>(matrix, 4, 2);
    });
    first.get();
    second.get();
}

#ifdef APXCHOL_USE_METAL
TEST(Solver, ThreadBudgetCoversDirectMetalOwners) {
    if (!apxchol::metal_solver::available()) GTEST_SKIP() << "no usable Metal device";
    check_thread_budget<apxchol::metal_solver>(operator_matrix(), 3, 2,
                                              apxchol::solve_backend::gpu);
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
#endif

TEST(Solver, RejectsNegativeThreadBudget) {
    const Sparse matrix = operator_matrix();
    auto options = cpu_options();
    options.num_threads = -1;
    EXPECT_THROW(apxchol::solver(matrix, options), std::invalid_argument);
    EXPECT_THROW(apxchol::cpu_solver(matrix, options), std::invalid_argument);
}
