cmake_minimum_required(VERSION 3.20)
project(apxchol VERSION 0.2.2 LANGUAGES C CXX)
include(cmake/apxchol_device.cmake)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE not set; defaulting to Release")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Native tuning is the default for local builds; turn OFF for portable
# binaries (distributed wheels/packages would SIGILL on older CPUs).
# The flag itself is ISA-dependent (-march=native on x86, -mcpu=native on ARM)
# and is probed in cmake/apxchol_native_arch.cmake. Untuned x86-64 builds
# use the x86-64-v2 baseline, including Debug builds.
option(APXCHOL_NATIVE_ARCH "Compile tuned for the build machine (Release/RelWithDebInfo)" ON)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/apxchol_arch.cmake)
add_compile_options(-Wall -Wextra -Wpedantic ${APXCHOL_ARCH_FLAG})

include(FetchContent)
include(cmake/apxchol_eigen.cmake)

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

# ── OpenMP ────────────────────────────────
include(cmake/apxchol_openmp.cmake)

# ── Build options ───────────────────────────────────
option(APXCHOL_64BIT_EDGE_INDICES "64-bit edge/offset indices, for factors or pools exceeding uint32 capacity (2^32-1 entries)" OFF)
option(APXCHOL_64BIT_NODE_INDICES "64-bit vertex indices (implies 64-bit edge indices)" OFF)
option(APXCHOL_64BIT_INDICES "Deprecated alias for APXCHOL_64BIT_EDGE_INDICES" OFF)
option(APXCHOL_POOL_FP32 "Store residual-pool weights in FP32; reads promote to FP64. OFF selects the FP64 reference." ON)
option(APXCHOL_BUILD_EXAMPLES "Build the examples/ programs" ON)
option(APXCHOL_BUILD_TESTS "Build the unit tests (fetches GoogleTest)" ON)

if (APXCHOL_64BIT_INDICES)
    message(DEPRECATION "APXCHOL_64BIT_INDICES is deprecated; use APXCHOL_64BIT_EDGE_INDICES")
    set(APXCHOL_64BIT_EDGE_INDICES ON)
endif()

if (APXCHOL_DEVICE STREQUAL "METAL")
    # The Metal implementation uses Objective-C++ for its host and device owner.
    enable_language(OBJCXX)
endif()

if (APXCHOL_DEVICE STREQUAL "CUDA")
    enable_language(CUDA)
    find_package(CUDAToolkit 13.3 REQUIRED)
    # Must be set before the target is created — CMAKE_CUDA_ARCHITECTURES is
    # read at add_library() time.
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures")
    endif()
endif()

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
function(apxchol_add_core target)
    add_library(${target} ${ARGN}
        src/factorization.cpp
        src/operator_class.cpp
        src/solve.cpp
    )
    target_link_libraries(${target} PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)

    target_include_directories(${target} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include
    )

    if (APXCHOL_DEVICE STREQUAL "CUDA")
        target_sources(${target} PRIVATE src/cuda_dataflow.cu
            src/cuda_block_frontend.cu src/cuda_round_shadow.cu src/cuda_pcg_kernels.cu)
        target_compile_definitions(${target} PUBLIC APXCHOL_USE_CUDA)
        # cudart only: the SpTRSV kernels, the GPU-resident PCG's SpMV / vector
        # kernels and its reductions are all ours; no cuSPARSE or cuBLAS.
        target_link_libraries(${target} PUBLIC CUDA::cudart)
        set_target_properties(${target} PROPERTIES CUDA_STANDARD 23 CUDA_STANDARD_REQUIRED ON)
    endif()

    if (APXCHOL_DEVICE STREQUAL "METAL")
        target_sources(${target} PRIVATE src/metal_solver.mm)
        target_compile_definitions(${target} PUBLIC APXCHOL_USE_METAL)
        set_source_files_properties(src/metal_solver.mm PROPERTIES COMPILE_OPTIONS -fobjc-arc)
        target_compile_options(${target} PRIVATE
            "$<$<COMPILE_LANGUAGE:OBJCXX>:SHELL:${OpenMP_CXX_FLAGS}>")
        set_target_properties(${target} PROPERTIES OBJCXX_STANDARD 23 OBJCXX_STANDARD_REQUIRED ON)
        # PUBLIC: the static library hands the framework dependencies to its
        # consumers. CoreGraphics lets MTLCreateSystemDefaultDevice find the GPU
        # from command-line tools.
        target_link_libraries(${target} PUBLIC "-framework Metal" "-framework Foundation"
                              "-framework CoreGraphics")
    endif()

    if (APXCHOL_64BIT_EDGE_INDICES)
        target_compile_definitions(${target} PUBLIC APXCHOL_64BIT_EDGE_INDICES)
    endif()
    if (APXCHOL_64BIT_NODE_INDICES)
        target_compile_definitions(${target} PUBLIC APXCHOL_64BIT_NODE_INDICES)
    endif()
    if (APXCHOL_POOL_FP32)
        target_compile_definitions(${target} PUBLIC APXCHOL_POOL_FP32)
    endif()
endfunction()

apxchol_add_core(apxchol_core)
if (APXCHOL_BUILD_TESTS AND APXCHOL_DEVICE STREQUAL "CUDA")
    # Test-only declarations must agree across the whole core and its consumers.
    apxchol_add_core(apxchol_test_core STATIC)
    target_compile_definitions(apxchol_test_core PUBLIC APXCHOL_GPU_ROUND_SHADOW_TEST_FAULTS)
endif()

# ── CLI input interpretation ─────────────────────────
# Laplacian-vs-adjacency detection for the CLI's .mtx input. Its own target so
# the unit tests can link it without pulling in CLI11/spdlog; Eigen-only, no
# file I/O (the CLI reads the file and hands over the matrix).
add_library(apxchol_mtx_input STATIC src/mtx_input.cpp)
target_link_libraries(apxchol_mtx_input PUBLIC Eigen3::Eigen)
target_include_directories(apxchol_mtx_input PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Directory settings do not reach targets in a parent add_subdirectory caller.
# Public Eigen objects must use the same alignment/ISA configuration on both
# sides of the library boundary. Export the architecture flag used by this build.
foreach (_apxchol_target IN ITEMS apxchol_core apxchol_test_core apxchol_mtx_input)
    if (NOT TARGET ${_apxchol_target})
        continue()
    endif()
    target_compile_features(${_apxchol_target} PUBLIC cxx_std_23)
    if (APXCHOL_ARCH_FLAG)
        target_compile_options(${_apxchol_target} INTERFACE
            "$<$<COMPILE_LANGUAGE:CXX>:${APXCHOL_ARCH_FLAG}>")
    endif()
endforeach()
unset(_apxchol_target)
unset(APXCHOL_ARCH_FLAG)

# Exception-safe C interface for non-C++ consumers.
add_library(apxchol_c src/c_api.cpp)
target_link_libraries(apxchol_c PUBLIC apxchol_core)

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
