cmake_minimum_required(VERSION 3.16)
project(laplacian_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

# ── Eigen ────────────────────────────────────────────
# Prefer system Eigen; fall back to FetchContent.
find_package(Eigen3 3.3 CONFIG QUIET HINTS /usr/share/eigen3/cmake)
if (NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(eigen)
endif()

# ── Core library ─────────────────────────────────────
add_library(laplacian_core
        src/graphs.cpp
        src/simple_solver.cpp
        src/mmio.cpp
)
target_link_libraries(laplacian_core PUBLIC Eigen3::Eigen)
target_include_directories(laplacian_core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ── Main executable ──────────────────────────────────
add_executable(laplacian_solver
        src/main.cpp
)
target_link_libraries(laplacian_solver PRIVATE laplacian_core)

# ── Benchmark executable ─────────────────────────────
add_executable(benchmark
        src/benchmark.cpp
)
target_link_libraries(benchmark PRIVATE laplacian_core)

enable_testing()
add_subdirectory(tests)
