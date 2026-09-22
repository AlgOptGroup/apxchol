include(FetchContent)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(unit_tests
        test_core.cpp
        test_residual_normalization.cpp
        test_factorize.cpp
        test_elimination.cpp
        test_custom_hooks.cpp
        test_sptrsv_transpose.cpp
        test_sptrsv_drop.cpp
        test_gpu_round_shadow.cpp
        test_gpu_owned_sparsify.cpp
        test_gpu_direct_csc.cpp
        test_gpu_repair_cascade.cpp
        test_gpu_operator_csr.cpp
        test_sptrsv_memory.cpp
        test_sptrsv_levelset.cpp
        test_sptrsv_kernels.cpp
        test_lowprec.cpp
        test_mtx_input.cpp
        test_operator_class.cpp
        test_pcg_cuda_host.cpp
        test_cuda_device_scope.cpp
)

target_link_libraries(unit_tests PRIVATE
        apxchol_core
        apxchol_mtx_input
        GTest::gtest_main
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
if (OpenMP_CXX_FOUND)
    # This test requests eight workers. A three-worker runtime limit exercises
    # actual-team slicing and rejects stale requested-worker radix histograms.
    add_test(NAME factorize_reduced_team_incremental_degrees
        COMMAND unit_tests
        --gtest_filter=VecPoolAos.AutoIncrementalDegreesPreserveTheFactorByteForByte)
    set_tests_properties(factorize_reduced_team_incremental_degrees PROPERTIES
        TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=8;OMP_THREAD_LIMIT=3;OMP_DYNAMIC=FALSE")
endif()
if (APXCHOL_USE_CUDA)
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

apxchol_cli_test(cli_pattern_adjacency_converges path5_pattern 0 "ADJACENCY graph")
apxchol_cli_test(cli_disconnected_adjacency_converges disconnected_paths_pattern 0
    "projected random RHS independently over 2 connected components")
apxchol_cli_test(cli_laplacian_converges path5_laplacian 0 "LAPLACIAN/SDDM operator")
apxchol_cli_test(cli_sddm_converges path5_sddm 0 "LAPLACIAN/SDDM operator")
apxchol_cli_test(cli_nearly_sddm_converges path5_nearly_sddm 0 "LAPLACIAN/SDDM operator")
apxchol_cli_test(cli_lumping_disabled_warns path5_nearly_sddm 0 "M-matrix lumping is disabled")
set_tests_properties(cli_lumping_disabled_warns PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=4;APXCHOL_LUMP=0")
apxchol_cli_test(cli_explicit_adjacency_on_laplacian path5_laplacian 0 "ADJACENCY graph"
    --input-kind adjacency)
apxchol_cli_test(cli_ambiguous_input_is_rejected ambiguous 2 "cannot tell whether this matrix is")
apxchol_cli_test(cli_forced_wrong_kind_is_refused path5_pattern 2 "Condition FAILED: positive diagonal"
    --input-kind laplacian)
# Enum transformations must not widen the CLI to accept raw numeric values.
apxchol_cli_test(cli_numeric_input_kind_is_rejected path5_pattern 105 "not in"
    --input-kind 0)
apxchol_cli_test(cli_numeric_storage_is_rejected path5_pattern 105 "not in"
    --graph-storage 0)
