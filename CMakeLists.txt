cmake_minimum_required(VERSION 3.20)
project(apxchol VERSION 0.2.2 LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE not set; defaulting to Release")
endif()

include(cmake/apxchol_core.cmake)
add_compile_options(-Wall -Wextra "$<$<COMPILE_LANGUAGE:CXX,OBJCXX>:-Wpedantic>" ${APXCHOL_ARCH_FLAG})
include(FetchContent)

# ── CLI11 ────────────────────────────────────────────
FetchContent_Declare(cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.4.2
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(cli11)

# ── spdlog ───────────────────────────────────────────
find_package(spdlog 1.14 CONFIG QUIET)
if (NOT spdlog_FOUND)
    set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG        v1.15.2
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(spdlog)
endif()

# ── fast_matrix_market ───────────────────────────────
FetchContent_Declare(fast_matrix_market
    GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market.git
    GIT_TAG        v1.7.6
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(fast_matrix_market)

option(APXCHOL_BUILD_EXAMPLES "Build the examples/ programs" ON)
option(APXCHOL_BUILD_TESTS "Build the unit tests (fetches GoogleTest)" ON)

# ── Git SHA (for --version) ──────────────────────────
execute_process(
    COMMAND git -C "${CMAKE_CURRENT_SOURCE_DIR}" rev-parse --short HEAD
    OUTPUT_VARIABLE APXCHOL_GIT_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if (NOT APXCHOL_GIT_SHA)
    set(APXCHOL_GIT_SHA "unknown")
endif()
set(APXCHOL_VERSION "${PROJECT_VERSION}")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/apxchol/version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/apxchol/version.h"
    @ONLY
)

# ── Core library: ApxChol preconditioner ─────────────
add_library(apxchol_core)
apxchol_configure_core(apxchol_core)
if (APXCHOL_BUILD_TESTS AND APXCHOL_DEVICE STREQUAL "CUDA")
    # Test-only declarations must agree across the whole core and its consumers.
    add_library(apxchol_test_core STATIC)
    apxchol_configure_core(apxchol_test_core)
    target_compile_definitions(apxchol_test_core PUBLIC APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
endif()

# ── CLI input interpretation ─────────────────────────
# Laplacian-vs-adjacency detection for the CLI's .mtx input. Its own target so
# the unit tests can link it without pulling in CLI11/spdlog; Eigen-only, no
# file I/O (the CLI reads the file and hands over the matrix).
add_library(apxchol_mtx_input STATIC src/mtx_input.cpp)
target_link_libraries(apxchol_mtx_input PUBLIC Eigen3::Eigen)
target_include_directories(apxchol_mtx_input PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_features(apxchol_mtx_input PUBLIC cxx_std_23)
if (APXCHOL_ARCH_FLAG)
    target_compile_options(apxchol_mtx_input INTERFACE "${APXCHOL_ARCH_FLAG}")
endif()

# ── CLI solver ───────────────────────────────────────
add_executable(apxchol src/main.cpp src/config.cpp)
target_link_libraries(apxchol PRIVATE apxchol_core apxchol_mtx_input CLI11::CLI11
                      spdlog::spdlog fast_matrix_market::fast_matrix_market)

# ── Examples ─────────────────────────────────────────
if (APXCHOL_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ── Tests ────────────────────────────────────────────
if (APXCHOL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
