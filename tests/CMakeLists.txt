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
        test_factorize.cpp
        test_elimination.cpp
        test_custom_hooks.cpp
        test_sptrsv_transpose.cpp
        test_bf16.cpp
)

target_link_libraries(unit_tests PRIVATE
        apxchol_core
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

# Developer analysis tools (not tests, not registered with ctest).
# APXCHOL_BUILD_TOOLS is declared in the root CMakeLists.txt.
if (APXCHOL_BUILD_TOOLS)
    add_executable(bench_setup bench_setup.cpp)
    target_link_libraries(bench_setup PRIVATE apxchol_core)

    add_executable(analyze_factor analyze_factor.cpp)
    # fast_matrix_market: this tool reads .mtx files itself (the core doesn't).
    target_link_libraries(analyze_factor PRIVATE apxchol_core fast_matrix_market::fast_matrix_market)
endif()
