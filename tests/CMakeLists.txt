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
        test_sptrsv_drop.cpp
        test_sptrsv_memory.cpp
        test_sptrsv_levelset.cpp
        test_sptrsv_kernels.cpp
        test_lowprec.cpp
        test_mtx_input.cpp
        test_operator_class.cpp
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

# ── CLI-level tests ─────────────────────────────────
# The Laplacian-vs-adjacency bug was in the CLI, not the core, so it gets
# tests that actually run the binary on files of each input kind. The fixtures
# are tiny path graphs, one file per kind. They live in fixtures/, not data/:
# the root .gitignore's `data/` rule is unanchored and would swallow them.
set(APXCHOL_CLI_DATA "${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

# These four must CONVERGE, so they are checked by exit code (0 = reached
# --tol) plus a FAIL_REGULAR_EXPRESSION that trips if the file was read as the
# wrong kind. Do not add PASS_REGULAR_EXPRESSION here: setting it makes CTest
# ignore the exit code, which is the convergence signal.

# An adjacency/pattern file is what SuiteSparse ships and what the README
# tells users to download: it must be recognised and solved, not silently
# turned into a garbage system (the regression: "iterations 0 / residual 1").
add_test(NAME cli_pattern_adjacency_converges
         COMMAND apxchol "${APXCHOL_CLI_DATA}/path5_pattern.mtx" --random-rhs --tol 1e-8)
set_tests_properties(cli_pattern_adjacency_converges PROPERTIES
        FAIL_REGULAR_EXPRESSION "LAPLACIAN/SDDM operator")

# A globally centred random RHS is generally inconsistent on a disconnected
# Laplacian. The CLI must project each connected component before solving.
add_test(NAME cli_disconnected_adjacency_converges
         COMMAND apxchol "${APXCHOL_CLI_DATA}/disconnected_paths_pattern.mtx"
                 --random-rhs --tol 1e-8)
set_tests_properties(cli_disconnected_adjacency_converges PROPERTIES
        PASS_REGULAR_EXPRESSION "projected random RHS independently over 2 connected components")

# An assembled Laplacian keeps being solved as given.
add_test(NAME cli_laplacian_converges
         COMMAND apxchol "${APXCHOL_CLI_DATA}/path5_laplacian.mtx" --random-rhs --tol 1e-8)
set_tests_properties(cli_laplacian_converges PROPERTIES
        FAIL_REGULAR_EXPRESSION "ADJACENCY graph")

# The research sampler uses the same CLI/C++ selector value and must complete
# an actual SDDM solve, exercising its W/degree extension. GKS remains covered
# by every CLI test that omits the option.
add_test(NAME cli_bkz26_sampler_converges
         COMMAND apxchol "${APXCHOL_CLI_DATA}/path5_sddm.mtx"
                 --random-rhs --seed 42 --clique-sampler bkz26 --tol 1e-8)
set_tests_properties(cli_bkz26_sampler_converges PROPERTIES
        FAIL_REGULAR_EXPRESSION "unknown clique sampler")

# So does an SDDM system (positive diagonal excess), which must NOT be rebuilt
# as a Laplacian -- that would discard the excess.
add_test(NAME cli_sddm_converges
         COMMAND apxchol "${APXCHOL_CLI_DATA}/path5_sddm.mtx" --random-rhs --tol 1e-8)
set_tests_properties(cli_sddm_converges PROPERTIES
        FAIL_REGULAR_EXPRESSION "ADJACENCY graph")

# A nearly-SDDM SPD operator (one off-diagonal positive) is repaired by
# M-matrix lumping when the preconditioner is built and must converge. Without
# it the positive entry becomes a negative edge weight, sample_clique's prefix
# sampler breaks, and the run ends in the stagnation guard.
add_test(NAME cli_nearly_sddm_converges
         COMMAND apxchol "${APXCHOL_CLI_DATA}/path5_nearly_sddm.mtx" --random-rhs --tol 1e-8)
set_tests_properties(cli_nearly_sddm_converges PROPERTIES
        FAIL_REGULAR_EXPRESSION "ADJACENCY graph")

# The A/B knob. It must RUN with lumping off -- both arms of an A/B have to
# produce a number -- but say loudly that it is handing the factorization
# negative edge weights, which has no other symptom than a bad residual.
add_test(NAME cli_lumping_disabled_warns
         COMMAND apxchol "${APXCHOL_CLI_DATA}/path5_nearly_sddm.mtx" --random-rhs --tol 1e-8)
set_tests_properties(cli_lumping_disabled_warns PROPERTIES
        ENVIRONMENT "APXCHOL_LUMP=0"
        PASS_REGULAR_EXPRESSION "M-matrix lumping is disabled")

# --input-kind adjacency on an already-assembled Laplacian is a no-op
# (|-w| = w and the rebuilt diagonal is the degree), so it still converges.
add_test(NAME cli_explicit_adjacency_on_laplacian
         COMMAND apxchol "${APXCHOL_CLI_DATA}/path5_laplacian.mtx" --random-rhs
                 --tol 1e-8 --input-kind adjacency)
set_tests_properties(cli_explicit_adjacency_on_laplacian PROPERTIES
        FAIL_REGULAR_EXPRESSION "LAPLACIAN/SDDM operator")

# An input that is neither kind must fail loudly (exit 2), naming what was
# detected -- never solve one of the two readings silently. PASS_REGULAR_
# EXPRESSION decides the verdict from the output, so the deliberate non-zero
# exit code does not mark the test failed.
add_test(NAME cli_ambiguous_input_is_rejected
         COMMAND apxchol "${APXCHOL_CLI_DATA}/ambiguous.mtx" --random-rhs --tol 1e-8)
set_tests_properties(cli_ambiguous_input_is_rejected PROPERTIES
        PASS_REGULAR_EXPRESSION "cannot tell whether this matrix is")

# Forcing the wrong kind must still be loud. The CLI honours --input-kind (it
# is a human tool and says what it did), but the LIBRARY then asserts the
# operator class and refuses: an adjacency matrix carries no positive diagonal,
# which no positive semidefinite operator can do. It used to reach PCG and
# report "iterations 0 / residual 1"; now it never gets that far.
add_test(NAME cli_forced_wrong_kind_is_refused
         COMMAND apxchol "${APXCHOL_CLI_DATA}/path5_pattern.mtx" --random-rhs
                 --tol 1e-8 --input-kind laplacian)
set_tests_properties(cli_forced_wrong_kind_is_refused PROPERTIES
        PASS_REGULAR_EXPRESSION "Condition FAILED: positive diagonal")

# Developer analysis tools (not tests, not registered with ctest).
# APXCHOL_BUILD_TOOLS is declared in the root CMakeLists.txt.
if (APXCHOL_BUILD_TOOLS)
    add_executable(bench_setup bench_setup.cpp)
    target_link_libraries(bench_setup PRIVATE apxchol_core)

    add_executable(analyze_factor analyze_factor.cpp)
    # fast_matrix_market: this tool reads .mtx files itself (the core doesn't),
    # so it needs the CLI's Laplacian-vs-adjacency interpretation too.
    target_link_libraries(analyze_factor PRIVATE apxchol_core apxchol_mtx_input
                          fast_matrix_market::fast_matrix_market)

    add_executable(bkz26_quality_probe
        ../experiments/2026-08-31-bkz26-prufer/bkz26_quality_probe.cpp)
    target_link_libraries(bkz26_quality_probe PRIVATE
        apxchol_core apxchol_mtx_input fast_matrix_market::fast_matrix_market)
endif()
