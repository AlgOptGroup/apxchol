cmake_minimum_required(VERSION 3.16)
project(laplacian_solver VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT MSVC)
    add_compile_options(-Wall -Wextra -Wpedantic)
    if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        add_compile_options(-march=native)
    endif()
endif()

include(FetchContent)
# ── Eigen ────────────────────────────────────────────
find_package(Eigen3 3.3 CONFIG QUIET)
if (NOT Eigen3_FOUND)
    FetchContent_Declare(eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(eigen)
endif()

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

# ── OpenMP (optional) ────────────────────────────────
find_package(OpenMP)

# ── SpTRSV backend (optional) ───────────────────────
option(APXCHOL_USE_CUDA "Use cuSPARSE (GPU) for sparse triangular solve" OFF)
option(APXCHOL_64BIT_INDICES "Use 64-bit signed indices for large graphs" OFF)

if (APXCHOL_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# ── Git SHA (for --version) ──────────────────────────
execute_process(
    COMMAND git -C "${CMAKE_SOURCE_DIR}" rev-parse --short HEAD
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
add_library(laplacian_core
    src/factorization.cpp
    src/solve.cpp
)
target_link_libraries(laplacian_core PUBLIC Eigen3::Eigen fast_matrix_market::fast_matrix_market)
if (OpenMP_CXX_FOUND)
    target_link_libraries(laplacian_core PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(laplacian_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
)

if (APXCHOL_USE_CUDA)
    target_compile_definitions(laplacian_core PUBLIC APXCHOL_USE_CUDA)
    target_link_libraries(laplacian_core PUBLIC CUDA::cudart CUDA::cusparse)
endif()

if (APXCHOL_64BIT_INDICES)
    target_compile_definitions(laplacian_core PUBLIC APXCHOL_64BIT_INDICES)
endif()

# ── CLI solver ───────────────────────────────────────
add_executable(laplacian_solver src/main.cpp src/config.cpp)
target_link_libraries(laplacian_solver PRIVATE laplacian_core CLI11::CLI11 spdlog::spdlog)

# ── Tests ────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)
