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
)

target_link_libraries(unit_tests PRIVATE
        laplacian_core
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(unit_tests)

add_executable(bench_factorize bench_factorize.cpp)
target_link_libraries(bench_factorize PRIVATE laplacian_core)

add_executable(bench_backends bench_backends.cpp)
target_link_libraries(bench_backends PRIVATE laplacian_core)

add_executable(bench_setup bench_setup.cpp)
target_link_libraries(bench_setup PRIVATE laplacian_core)
