#include <gtest/gtest.h>
#include "scoped_env.h"
#include "apxchol/solver/solve.h"
#include "apxchol/solver/detail/solve_backend.h"
#if defined(APXCHOL_USE_METAL)
#include "apxchol/solver/metal_solver.h"
#endif
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

Eigen::SparseMatrix<double> operator_matrix() {
    constexpr int n = 64;
    std::vector<Eigen::Triplet<double>> entries;
    for (int i = 0; i < n; ++i) {
        entries.emplace_back(i, i, 5.0);
        for (int offset : {1, 7}) {
            const int j = (i + offset) % n;
            entries.emplace_back(i, j, -1.0);
            entries.emplace_back(j, i, -1.0);
        }
    }
    Eigen::SparseMatrix<double> result(n, n);
    result.setFromTriplets(entries.begin(), entries.end());
    return result;
}

void expect_solution(const Eigen::SparseMatrix<double>& matrix,
                     const Eigen::VectorXd& rhs, const apxchol::solve_result& result) {
    ASSERT_EQ(result.x.size(), matrix.rows());
    EXPECT_TRUE(result.x.allFinite());
    EXPECT_LT((matrix * result.x - rhs).norm() / rhs.norm(), 1e-8);
}

std::vector<apxchol::solve_options> host_only_options() {
    std::vector<apxchol::solve_options> options(5);
    options[0].storage = apxchol::graph_storage::vec;
    options[1].factor_opts.is_select = "priority_greedy";
    options[2].factor_opts.exact_clique_max_degree = 3;
    options[3].factor_opts.sampler = apxchol::clique_sampler::trace_cycle;
    options[3].factor_opts.exact_core_max_h = 3;
    options[4].factor_opts.sampler = apxchol::clique_sampler::trace_cycle;
    options[4].factor_opts.double_cycle_min_h = 4;
    return options;
}
} // namespace

TEST(SolveRoutes, CpuPreconditionerTypeIsIndependentOfCudaBuild) {
    static_assert(std::is_same_v<decltype(std::declval<apxchol::apx_cholesky&>().trsv()),
                                 apxchol::omp_sptrsv&>);
    SUCCEED();
}

TEST(SolveRoutes, ExplicitCpuDoesNotPrewarm) {
#if defined(APXCHOL_USE_CUDA)
    const bool started = apxchol::cuda_ctx::detail::state().started;
    if (const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
        visible && std::string(visible) == "-1") {
        // The dedicated CTest runs this before any device test, in a fresh process.
        EXPECT_FALSE(started);
    }
#endif
    const auto matrix = operator_matrix();
    const Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(matrix.rows(), 1.0, 2.0);
    apxchol::solve_options options;
    options.backend = apxchol::solve_backend::cpu;
    options.tol = 1e-10;
    const auto result = apxchol::solve(matrix, rhs, options);
    EXPECT_EQ(result.backend, apxchol::solve_backend::cpu);
    EXPECT_EQ(result.solve_vram_mb, -1.0);
    expect_solution(matrix, rhs, result);
    // Both named CPU owners stay on CPU even with their default options.
    apxchol::cpu_solver reusable(matrix);
    const auto reused = reusable.solve(rhs, 1e-10);
    EXPECT_EQ(reused.backend, apxchol::solve_backend::cpu);
    expect_solution(matrix, rhs, reused);
    const auto factor = apxchol::factorize(matrix, apxchol::graph_storage::vec_pool_aos);
    EXPECT_EQ(factor.L.vals_.size(), factor.L.nonZeros());
    EXPECT_EQ(factor.L.inner_.size(), factor.L.nonZeros());
    EXPECT_EQ(factor.perm.size(), static_cast<std::size_t>(matrix.rows()));
#if defined(APXCHOL_USE_CUDA)
    EXPECT_FALSE(factor.research_device_factor);
#endif
    apxchol::apx_cholesky eigen_preconditioner;
    eigen_preconditioner.compute(matrix);
    const Eigen::VectorXd applied = eigen_preconditioner.solve(rhs);
    EXPECT_TRUE(applied.allFinite());
#if defined(APXCHOL_USE_CUDA)
    EXPECT_EQ(apxchol::cuda_ctx::detail::state().started, started);
#endif
}

TEST(SolveRoutes, AutomaticSelectionRespectsConfigurationAndAvailability) {
    const auto matrix = operator_matrix();
    const Eigen::VectorXd rhs = Eigen::VectorXd::Ones(matrix.rows());
    const auto cases = host_only_options();
    ASSERT_EQ(cases.size(), 5u);
#if defined(APXCHOL_USE_METAL)
    const auto expected = apxchol::metal_solver::available()
        ? apxchol::solve_backend::gpu : apxchol::solve_backend::cpu;
#else
    const auto expected = apxchol::solve_backend::cpu;
#endif
    for (std::size_t i = 0; i < cases.size(); ++i) {
        SCOPED_TRACE(i);
        EXPECT_EQ(apxchol::detail::select_solve_backend(cases[i]), expected);
        const auto result = apxchol::solve(matrix, rhs, cases[i]);
        EXPECT_EQ(result.backend, expected);
        expect_solution(matrix, rhs, result);
    }
}

#if defined(APXCHOL_USE_METAL)
TEST(SolveRoutes, MetalGpuSupportsFp32AndFp16Storage) {
    if (!apxchol::metal_solver::available()) GTEST_SKIP() << "no usable Metal device";
    const auto matrix = operator_matrix();
    const Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(matrix.rows(), 1.0, 2.0);
    for (auto storage : {apxchol::factor_storage_type::fp32, apxchol::factor_storage_type::fp16}) {
        SCOPED_TRACE(storage == apxchol::factor_storage_type::fp16 ? "float16" : "float32");
        apxchol::solve_options options;
        options.backend = apxchol::solve_backend::gpu;
        options.factor_opts.factor_storage = storage;
        options.tol = 1e-10;
        const auto result = apxchol::solve(matrix, rhs, options);
        EXPECT_EQ(result.backend, apxchol::solve_backend::gpu);
        expect_solution(matrix, rhs, result);
    }
}
#else
TEST(SolveRoutes, ExplicitGpuRejectsUnsupportedConfigurationsBeforeSetup) {
#if defined(APXCHOL_USE_CUDA)
    const bool started = apxchol::cuda_ctx::detail::state().started;
#endif
    const auto matrix = operator_matrix();
    const Eigen::VectorXd rhs = Eigen::VectorXd::Ones(matrix.rows());
    auto cases = host_only_options();
    ASSERT_EQ(cases.size(), 5u);
    for (auto& options : cases) {
        options.backend = apxchol::solve_backend::gpu;
        EXPECT_THROW(apxchol::detail::select_solve_backend(options), std::invalid_argument);
        EXPECT_THROW(apxchol::solve(matrix, rhs, options), std::invalid_argument);
    }
#if defined(APXCHOL_USE_CUDA)
    EXPECT_EQ(apxchol::cuda_ctx::detail::state().started, started);
#endif
}
#endif

TEST(SolveRoutes, CpuSolverRejectsExplicitGpuRequest) {
    apxchol::solve_options options;
    options.backend = apxchol::solve_backend::gpu;
    EXPECT_THROW(apxchol::cpu_solver(operator_matrix(), options), std::invalid_argument);
}

TEST(SolveRoutes, CommonOwnerRejectsEmptyInput) {
    const Eigen::SparseMatrix<double> empty(0, 0);
    for (const auto backend : {apxchol::solve_backend::automatic,
                              apxchol::solve_backend::cpu, apxchol::solve_backend::gpu}) {
        apxchol::solve_options options;
        options.backend = backend;
        EXPECT_THROW(apxchol::solver(empty, options), std::invalid_argument);
    }
}

TEST(SolveRoutes, CpuFactorReuseAndExportRemainAvailable) {
    const auto matrix = operator_matrix();
    apxchol::solve_options options;
    options.backend = apxchol::solve_backend::cpu;
    options.keep_factor_values = true;
    apxchol::cpu_solver solver(matrix, options);
    ASSERT_FALSE(solver.preconditioner().factor().L.vals_.empty());
    auto factor = solver.preconditioner().factor();
    apxchol::cpu_solver adopted(matrix, std::move(factor), options);
    for (double offset : {0.5, 2.0}) {
        const Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(matrix.rows(), offset, offset + 1);
        const auto result = adopted.solve(rhs);
        EXPECT_EQ(result.backend, apxchol::solve_backend::cpu);
        expect_solution(matrix, rhs, result);
    }
}

TEST(SolveRoutes, FactorStorageOverridesEnvironmentForNewAndAdoptedFactors) {
    const auto matrix = operator_matrix();
    const Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(matrix.rows(), 1.0, 2.0);
    for (bool fp16 : {false, true}) {
        scoped_env env("APXCHOL_FACTOR_STORAGE", fp16 ? "float32" : "float16");
        apxchol::solve_options options;
        options.backend = apxchol::solve_backend::cpu;
        options.keep_factor_values = true;
        options.factor_opts.factor_storage = fp16
            ? apxchol::factor_storage_type::fp16 : apxchol::factor_storage_type::fp32;
#if defined(APXCHOL_USE_CUDA)
        EXPECT_EQ(apxchol::cuda_sptrsv::fp16_resolved(options.factor_opts.factor_storage), fp16);
#endif
        if (fp16 && !apxchol::omp_sptrsv::fp16_supported()) {
            EXPECT_THROW(apxchol::cpu_solver(matrix, options), std::runtime_error);
            continue;
        }
        apxchol::cpu_solver original(matrix, options);
        auto factor = original.preconditioner().factor();
        options.keep_factor_values = false;
        apxchol::cpu_solver adopted(matrix, std::move(factor), options);
        EXPECT_EQ(original.preconditioner().trsv().fp16(), fp16);
        EXPECT_EQ(adopted.preconditioner().trsv().fp16(), fp16);
        expect_solution(matrix, rhs, original.solve(rhs, 1e-10));
        expect_solution(matrix, rhs, adopted.solve(rhs, 1e-10));
    }
}

TEST(SolveRoutes, RequiredGpuPreconditionerRejectsHostFactorBeforeCuda) {
#if !defined(APXCHOL_USE_CUDA)
    GTEST_SKIP() << "CUDA preconditioner type required";
#else
    const bool started = apxchol::cuda_ctx::detail::state().started;
    const auto matrix = operator_matrix();
    apxchol::solve_options options;
    options.keep_factor_values = true;
    apxchol::cpu_solver cpu(matrix, options);
    ASSERT_FALSE(cpu.preconditioner().factor().research_device_factor);
    apxchol::detail::gpu_preconditioner gpu;
    EXPECT_THROW(gpu.set_factor(cpu.preconditioner().factor()), std::invalid_argument);
    EXPECT_EQ(apxchol::cuda_ctx::detail::state().started, started);
#endif
}

TEST(SolveRoutes, AutomaticAndExplicitGpuUseOwnedSetup) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "32-bit-node CUDA solve required";
#else
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP() << "CUDA unavailable";
    const auto matrix = operator_matrix();
    const Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(matrix.rows(), 1.0, 2.0);
    for (const auto backend : {apxchol::solve_backend::automatic, apxchol::solve_backend::gpu}) {
        for (const auto sampler : {apxchol::clique_sampler::gks, apxchol::clique_sampler::trace_cycle}) {
            SCOPED_TRACE(static_cast<int>(backend));
            SCOPED_TRACE(static_cast<int>(sampler));
            apxchol::solve_options options;
            options.backend = backend;
            options.factor_opts.sampler = sampler;
            options.tol = 1e-10;
            const auto result = apxchol::solve(matrix, rhs, options);
            EXPECT_EQ(result.backend, apxchol::solve_backend::gpu);
            EXPECT_GT(result.timings.total("setup.gpu_owned_factorization"), 0.0);
            EXPECT_GT(result.timings.total("pcg.gpu_pcg_loop"), 0.0);
            expect_solution(matrix, rhs, result);
        }
    }
#endif
}

TEST(SolveRoutes, UnsupportedStoredLayoutSelectsCpuBeforeSetup) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "32-bit-node CUDA route required";
#else
    for (bool stored_zero : {false, true}) {
        SCOPED_TRACE(stored_zero);
        auto matrix = operator_matrix();
        if (stored_zero) {
            // Numerically symmetric, but the device builder needs stored mates.
            matrix.coeffRef(0, 2) = 0.0;
            matrix.makeCompressed();
        } else matrix.uncompress();
        const Eigen::VectorXd rhs = Eigen::VectorXd::Ones(matrix.rows());
        apxchol::solve_options options;
        ASSERT_EQ(apxchol::detail::select_solve_backend(options, &matrix), apxchol::solve_backend::cpu);
        const auto result = apxchol::solve(matrix, rhs, options);
        EXPECT_EQ(result.backend, apxchol::solve_backend::cpu);
        EXPECT_GT(result.timings.total("setup.backend_selection"), 0.0);
        expect_solution(matrix, rhs, result);
        options.backend = apxchol::solve_backend::gpu;
        EXPECT_THROW(apxchol::solve(matrix, rhs, options), std::invalid_argument);
    }
#endif
}

#if defined(APXCHOL_USE_METAL)
TEST(SolveRoutes, OperatorOutsideMetalRangeSelectsCpuBeforeSetup) {
    Eigen::SparseMatrix<double> matrix(2, 2);
    matrix.setIdentity();
    matrix *= 0x1p104;
    const Eigen::VectorXd rhs = Eigen::VectorXd::Ones(matrix.rows());
    apxchol::solve_options options;
    EXPECT_EQ(apxchol::detail::select_solve_backend(options, &matrix), apxchol::solve_backend::cpu);
    const auto result = apxchol::solve(matrix, rhs, options);
    EXPECT_EQ(result.backend, apxchol::solve_backend::cpu);
    EXPECT_GT(result.timings.total("setup.backend_selection"), 0.0);
    expect_solution(matrix, rhs, result);
    options.backend = apxchol::solve_backend::gpu;
    EXPECT_THROW(apxchol::solve(matrix, rhs, options), std::exception);

    options.backend = apxchol::solve_backend::automatic;
    matrix.coeffRef(0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(apxchol::solve(matrix, rhs, options), std::invalid_argument);
}
#endif

TEST(SolveRoutes, HiddenDeviceUsesCpuAutomaticallyAndRejectsExplicitGpu) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "32-bit-node CUDA route required";
#else
    const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
    if (!visible || std::string(visible) != "-1")
        GTEST_SKIP() << "exercised by solve_routes_no_cuda_device in a fresh hidden-device process";
    const auto matrix = operator_matrix();
    const Eigen::VectorXd rhs = Eigen::VectorXd::Ones(matrix.rows());
    apxchol::solve_options options;
    EXPECT_EQ(apxchol::detail::select_solve_backend(options), apxchol::solve_backend::cpu);
    const auto automatic = apxchol::solve(matrix, rhs, options);
    EXPECT_EQ(automatic.backend, apxchol::solve_backend::cpu);
    expect_solution(matrix, rhs, automatic);
    options.backend = apxchol::solve_backend::gpu;
    EXPECT_THROW(apxchol::solve(matrix, rhs, options), std::exception);
    apxchol::solve_options cpu;
    cpu.backend = apxchol::solve_backend::cpu;
    expect_solution(matrix, rhs, apxchol::solve(matrix, rhs, cpu));
#endif
}

TEST(SolveRoutes, IncompatibleBuildRejectsExplicitGpuAndAutoSelectsCpu) {
#if defined(APXCHOL_USE_METAL) || (defined(APXCHOL_USE_CUDA) && !defined(APXCHOL_64BIT_NODE_INDICES))
    GTEST_SKIP() << "exercised by the CPU-only and wide-node configurations";
#else
    apxchol::solve_options options;
    EXPECT_EQ(apxchol::detail::select_solve_backend(options), apxchol::solve_backend::cpu);
#if defined(APXCHOL_USE_CUDA)
    const bool started = apxchol::cuda_ctx::detail::state().started;
#endif
    const auto matrix = operator_matrix();
    const Eigen::VectorXd rhs = Eigen::VectorXd::Ones(matrix.rows());
    const auto result = apxchol::solve(matrix, rhs, options);
    EXPECT_EQ(result.backend, apxchol::solve_backend::cpu);
    expect_solution(matrix, rhs, result);
    options.backend = apxchol::solve_backend::gpu;
    EXPECT_THROW(apxchol::solve(matrix, rhs, options), std::invalid_argument);
#if defined(APXCHOL_USE_CUDA)
    EXPECT_EQ(apxchol::cuda_ctx::detail::state().started, started);
#endif
#endif
}

TEST(SolveRoutes, GpuSingletonSddmAndUnsupportedEmptyFactorAreExplicit) {
#if !defined(APXCHOL_USE_CUDA) || defined(APXCHOL_64BIT_NODE_INDICES)
    GTEST_SKIP() << "32-bit-node CUDA solve required";
#else
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP() << "CUDA unavailable";
    apxchol::solve_options options;
    options.backend = apxchol::solve_backend::gpu;
    options.tol = 1e-10;
    Eigen::SparseMatrix<double> matrix(1, 1);
    matrix.insert(0, 0) = 2.0;
    matrix.makeCompressed();
    apxchol::detail::gpu_preconditioner preconditioner;
    preconditioner.compute(matrix);
    ASSERT_TRUE(preconditioner.trsv().adopted_device_factor());
    const Eigen::VectorXd rhs = Eigen::VectorXd::Constant(1, 2.0);
    const auto result = apxchol::solve(matrix, rhs, options);
    EXPECT_EQ(result.backend, apxchol::solve_backend::gpu);
    expect_solution(matrix, rhs, result);

    // The current device finalizer does not support a zero-dimensional L11.
    // A singleton Laplacian is a valid CPU problem; automatic selection keeps
    // this unsupported factor on CPU before setup.
    const Eigen::SparseMatrix<double> singleton_laplacian(1, 1);
    const Eigen::VectorXd zero = Eigen::VectorXd::Zero(1);
    apxchol::solve_options cpu;
    cpu.backend = apxchol::solve_backend::cpu;
    const auto trivial = apxchol::solve(singleton_laplacian, zero, cpu);
    EXPECT_EQ(trivial.backend, apxchol::solve_backend::cpu);
    EXPECT_EQ(trivial.iterations, 0);
    EXPECT_EQ(trivial.residual, 0.0);
    EXPECT_TRUE(trivial.x.isZero(0.0));
    EXPECT_THROW(apxchol::solve(singleton_laplacian, zero, options), std::invalid_argument);
    options.backend = apxchol::solve_backend::automatic;
    const auto automatic = apxchol::solve(singleton_laplacian, zero, options);
    EXPECT_EQ(automatic.backend, apxchol::solve_backend::cpu);
    EXPECT_EQ(automatic.residual, 0.0);
    EXPECT_TRUE(automatic.x.isZero(0.0));
#endif
}
