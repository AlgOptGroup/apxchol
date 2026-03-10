cmake_minimum_required(VERSION 3.16)
project(apxchol_v0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT MSVC)
    add_compile_options(-Wall -Wextra -Wpedantic)
    if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        add_compile_options(-march=native)
    endif()
endif()

# Eigen
find_package(Eigen3 3.3 CONFIG QUIET)
if (NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(eigen)
endif()

# Core library
add_library(apxchol_v0_core
    graphs.cpp
    simple_solver.cpp
    mmio.cpp
)
target_link_libraries(apxchol_v0_core PUBLIC Eigen3::Eigen)
target_include_directories(apxchol_v0_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Demo executable
add_executable(apxchol_v0_demo main.cpp)
target_link_libraries(apxchol_v0_demo PRIVATE apxchol_v0_core)

# Tests
include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(v0_tests test_sanity.cpp test_solver.cpp)
target_link_libraries(v0_tests PRIVATE apxchol_v0_core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(v0_tests)
