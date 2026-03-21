cmake_minimum_required(VERSION 3.16)
project(laplacian_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT MSVC)
    add_compile_options(-Wall -Wextra -Wpedantic)
    if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        add_compile_options(-march=native)
    endif()
endif()

# ── Eigen ────────────────────────────────────────────
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

# ── Core library: ApxChol preconditioner + graph I/O ──
add_library(laplacian_core
    src/graph.cpp
    src/adj_list_graph.cpp
    src/factorization.cpp
    src/preconditioner.cpp
    src/solver.cpp
)
target_link_libraries(laplacian_core PUBLIC Eigen3::Eigen)
target_include_directories(laplacian_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ── CLI solver ───────────────────────────────────────
add_executable(laplacian_solver src/main.cpp)
target_link_libraries(laplacian_solver PRIVATE laplacian_core)

# ── Tests ────────────────────────────────────────────
enable_testing()
add_subdirectory(tests)
