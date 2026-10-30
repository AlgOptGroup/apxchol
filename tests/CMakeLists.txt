include(FetchContent)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
target_include_directories(gtest SYSTEM PRIVATE
        "${googletest_SOURCE_DIR}/googletest/include")

add_executable(unit_tests
        test_core.cpp
        test_checkpoint.cpp
        test_residual_normalization.cpp
        test_factorize.cpp
        test_elimination.cpp
        test_custom_hooks.cpp
        test_sptrsv_transpose.cpp
        test_sptrsv_drop.cpp
        test_gpu_round_shadow.cpp
        test_gpu_factor_finalize.cpp
        test_gpu_owned_sparsify.cpp
        test_gpu_direct_csc.cpp
        test_gpu_repair_cascade.cpp
        test_gpu_operator_csr.cpp
        test_gpu_pcg.cpp
        test_sptrsv_memory.cpp
        test_sptrsv_levelset.cpp
        test_sptrsv_kernels.cpp
        test_lowprec.cpp
        test_mtx_input.cpp
        test_operator_class.cpp
        test_solve_routes.cpp
        test_solver.cpp
        test_solver_archive.cpp
        test_cuda_device_scope.cpp
        test_permuted_operator.cpp
        test_metal_schedule.cpp
        test_metal_host.cpp
        test_metal_finalize.cpp
)
if (APXCHOL_DEVICE STREQUAL "METAL")
    target_sources(unit_tests PRIVATE test_metal.cpp)
endif()

if (APXCHOL_DEVICE STREQUAL "CUDA")
    target_link_libraries(unit_tests PRIVATE apxchol_test_core)
    # GCC 15 and later (through trunk as of 2026-10) propagate wrong known bits
    # through std::max(max_iter, 0) into gpu_pcg::solve for this file's test
    # preconditioner and delete its iteration loop; scalar GpuSolveHonesty
    # solves then stop at iteration zero. GCC 14 and Clang are correct.
    # https://gcc.gnu.org/bugzilla/show_bug.cgi?id=128092
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
        CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
        set_source_files_properties(test_gpu_operator_csr.cpp PROPERTIES
            COMPILE_OPTIONS -fno-ipa-bit-cp)
    endif()
else()
    target_link_libraries(unit_tests PRIVATE apxchol_core)
endif()
target_link_libraries(unit_tests PRIVATE
        apxchol_mtx_input
        GTest::gtest_main
        ZLIB::ZLIB
)

include(GoogleTest)
# Default per-test wall-clock timeout: 120s. Individual tests are <250ms
# when run alone; the long tail is when ctest -j<nproc> starves them of
# CPU. 120s gives plenty of headroom for the worst-case starved test.
#
# OMP_NUM_THREADS=4 caps each test's OpenMP thread pool so `ctest -j<N>`
# doesn't oversubscribe. Without the cap, each test grabs all hardware
# threads — N tests × all-threads on N cores → N× slowdown each, easily
# blowing the 120s timeout on the bigger grids (e.g. LargerGridConverges).
# Four threads is enough to exercise the parallel paths while leaving
# headroom for ctest -j8 on a 32-core box. Override per-test with set_tests_
# properties if a specific test needs more.
gtest_discover_tests(unit_tests PROPERTIES TIMEOUT 120
        ENVIRONMENT "OMP_NUM_THREADS=4")
# This test requests eight workers. A three-worker runtime limit exercises
# actual-team slicing and rejects stale requested-worker radix histograms.
add_test(NAME factorize_reduced_team_incremental_degrees
    COMMAND unit_tests
    --gtest_filter=VecPoolAos.AutoIncrementalDegreesPreserveTheFactorByteForByte)
set_tests_properties(factorize_reduced_team_incremental_degrees PROPERTIES
    TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=8;OMP_THREAD_LIMIT=3;OMP_DYNAMIC=FALSE")
if (APXCHOL_DEVICE STREQUAL "CUDA")
    # The CPU route must run in a fresh process with no visible CUDA devices;
    # automatic selection uses CPU; explicit GPU requests still fail.
    add_test(NAME solve_routes_no_cuda_device COMMAND unit_tests
        --gtest_filter=SolveRoutes.ExplicitCpuDoesNotPrewarm:SolveRoutes.ScalarAutomaticUsesCpuWithoutPrewarming:SolveRoutes.AutomaticSelectionRespectsConfigurationAndAvailability:SolveRoutes.ExplicitGpuRejectsUnsupportedConfigurationsBeforeSetup:SolveRoutes.HiddenDeviceUsesCpuAutomaticallyAndRejectsExplicitGpu)
    set_tests_properties(solve_routes_no_cuda_device PROPERTIES TIMEOUT 120
        ENVIRONMENT "OMP_NUM_THREADS=4;CUDA_VISIBLE_DEVICES=-1")
    # Device-wide free-memory accounting also sees other GPU applications.
    # Check this process's repeated finalizer allocations with the CUDA tool.
    find_program(APXCHOL_COMPUTE_SANITIZER NAMES compute-sanitizer
            HINTS "${CUDAToolkit_BIN_DIR}")
    mark_as_advanced(APXCHOL_COMPUTE_SANITIZER)
    if (APXCHOL_COMPUTE_SANITIZER)
        add_test(NAME gpu_factor_finalize_leak_check
                COMMAND "${CMAKE_COMMAND}"
                "-DSANITIZER=${APXCHOL_COMPUTE_SANITIZER}"
                "-DUNIT_TESTS=$<TARGET_FILE:unit_tests>"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/run_gpu_factor_finalize_leak_check.cmake")
        set_tests_properties(gpu_factor_finalize_leak_check PROPERTIES
                TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=4"
                SKIP_REGULAR_EXPRESSION "APXCHOL_GPU_FINALIZER_LEAK_CHECK_SKIPPED")
    else()
        message(STATUS "Compute Sanitizer not found: GPU finalizer leak check is not registered")
    endif()
endif()

if (APXCHOL_BUILD_CLI)
    # CLI assertions check BOTH exit status and diagnostic text. CTest's
    # PASS_REGULAR_EXPRESSION alone overrides a failing solver exit status.
    function(apxchol_cli_test name fixture status output)
        add_test(NAME ${name} COMMAND "${CMAKE_COMMAND}"
            "-DCLI=$<TARGET_FILE:apxchol>"
            "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/${fixture}.mtx"
            "-DEXPECT_STATUS=${status}" "-DEXPECT_OUTPUT=${output}"
            "-DCLI_ARGS=${ARGN}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/check_cli.cmake")
        set_tests_properties(${name} PROPERTIES TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=4")
    endfunction()

    # The input is the operator unless --input-kind adjacency says otherwise.
    apxchol_cli_test(cli_adjacency_default_is_rejected_with_hint path5_pattern 2
        "graph ADJACENCY matrix.*--input-kind adjacency")
    apxchol_cli_test(cli_explicit_adjacency_converges path5_pattern 0 "ADJACENCY graph"
        --input-kind adjacency)
    apxchol_cli_test(cli_disconnected_adjacency_converges disconnected_paths_pattern 0
        "projected random RHS independently over 2 connected components"
        --input-kind adjacency)
    apxchol_cli_test(cli_laplacian_converges path5_laplacian 0 "LAPLACIAN/SDDM operator")
    apxchol_cli_test(cli_sddm_converges path5_sddm 0 "LAPLACIAN/SDDM operator")
    apxchol_cli_test(cli_nearly_sddm_converges path5_nearly_sddm 0 "LAPLACIAN/SDDM operator")
    apxchol_cli_test(cli_explicit_adjacency_on_laplacian path5_laplacian 0 "ADJACENCY graph"
        --input-kind adjacency)
    # auto reads a graph only when the operator reading fails and the matrix is
    # an adjacency matrix or a pattern file; anything else is used as given.
    apxchol_cli_test(cli_auto_keeps_laplacian path5_laplacian 0
        "LAPLACIAN/SDDM operator.*auto: used as given" --input-kind auto)
    apxchol_cli_test(cli_auto_keeps_sddm path5_sddm 0
        "LAPLACIAN/SDDM operator.*auto: used as given" --input-kind auto)
    apxchol_cli_test(cli_auto_converts_adjacency path5_adjacency 0
        "ADJACENCY graph.*auto: not a valid operator" --input-kind auto)
    apxchol_cli_test(cli_auto_converts_pattern path5_pattern 0
        "ADJACENCY graph.*auto: not a valid operator" --input-kind auto)
    apxchol_cli_test(cli_auto_uses_ambiguous_input_as_given ambiguous 2
        "auto: used as given.*Condition FAILED" --input-kind auto)
    apxchol_cli_test(cli_explicit_operator_on_adjacency path5_adjacency 2
        "--input-kind adjacency. or .--input-kind auto" --input-kind operator)
    apxchol_cli_test(cli_laplacian_input_kind_is_rejected path5_laplacian 105 "not in"
        --input-kind laplacian)
    apxchol_cli_test(cli_forced_wrong_kind_is_refused path5_pattern 2 "Condition FAILED: positive diagonal"
        --input-kind operator)
    # Enum transformations must not widen the CLI to accept raw numeric values.
    apxchol_cli_test(cli_numeric_input_kind_is_rejected path5_pattern 105 "not in"
        --input-kind 0)

    # Complete solve-route CLI contract: the CPU choice is usable in every build,
    # and CLI enum transformations must not admit numeric or unknown selectors.
    apxchol_cli_test(cli_backend_cpu_converges path5_sddm 0 "backend: +cpu" --backend cpu)
    # Auto keeps scalar systems on CPU in every build. CUDA builds with a visible
    # device route larger inputs, including path5, to the GPU.
    apxchol_cli_test(cli_backend_auto_scalar_uses_cpu scalar_sddm 0 "backend: +cpu")
    if (NOT APXCHOL_DEVICE STREQUAL "CUDA")
        apxchol_cli_test(cli_backend_auto_tiny_uses_cpu path5_sddm 0 "backend: +cpu")
    endif()
    apxchol_cli_test(cli_numeric_backend_is_rejected path5_sddm 105 "not in" --backend 0)
    apxchol_cli_test(cli_invalid_backend_is_rejected path5_sddm 105 "not in" --backend hybrid)
    if (APXCHOL_DEVICE STREQUAL "CPU")
        apxchol_cli_test(cli_unavailable_gpu_backend_is_rejected path5_sddm 2
            "GPU route requested but GPU support is not built" --backend gpu)
    endif()

    apxchol_cli_test(cli_thread_budget path5_sddm 0 "backend: +cpu" --backend cpu --threads 2)
    apxchol_cli_test(cli_negative_thread_budget path5_sddm 105 "not in range" --threads -1)
    apxchol_cli_test(cli_zero_iteration_budget path5_sddm 1
        "did not reach tol .* in 0 iterations" --backend cpu --maxiter 0)
    apxchol_cli_test(cli_output_failure path5_sddm 2
        "cannot write solution to" --backend cpu -o "${CMAKE_CURRENT_BINARY_DIR}")
    apxchol_cli_test(cli_output_success path5_sddm 0
        "solution written to" --backend cpu -o "${CMAKE_CURRENT_BINARY_DIR}/cli-solution.mtx")

    # An n x k RHS file is solved as one batch and written back as n x k; an
    # n x 1 file keeps the scalar path and its log.
    function(apxchol_cli_rhs_test name rhs status output solution)
        set(path "${CMAKE_CURRENT_BINARY_DIR}/${name}.mtx")
        add_test(NAME ${name} COMMAND "${CMAKE_COMMAND}"
            "-DCLI=$<TARGET_FILE:apxchol>"
            "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/path5_sddm.mtx"
            "-DRHS=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/${rhs}.mtx"
            "-DEXPECT_STATUS=${status}" "-DEXPECT_OUTPUT=${output}"
            "-DSOLUTION=${path}" "-DEXPECT_SOLUTION=${solution}"
            "-DCLI_ARGS=-o;${path};${ARGN}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/check_cli.cmake")
        set_tests_properties(${name} PROPERTIES TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=4")
    endfunction()
    apxchol_cli_rhs_test(cli_batch_rhs_round_trip path5_rhs3 0
        "columns: +3.*column 0: iterations [0-9]+, residual.*column 2: iterations"
        "array real general\n5 3\n")
    apxchol_cli_rhs_test(cli_single_column_rhs_file path5_rhs1 0
        "iterations: [0-9]+" "array real general\n5 1\n")
    if (APXCHOL_DEVICE STREQUAL "CPU")
        set(cli_backend_label omp)
    else()
        string(TOLOWER "${APXCHOL_DEVICE}" cli_backend_label)
    endif()
    apxchol_cli_test(cli_build_backend path5_sddm 0
        "SpTRSV backend: ${cli_backend_label}" --version)
endif()
