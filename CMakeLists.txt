cmake_minimum_required(VERSION 3.16)
project(apxchol VERSION 0.2.2 LANGUAGES CXX)

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
# Fetched projects must not register their own test suites in apxchol's CTest
# tree. In particular, Eigen defaults BUILD_TESTING to ON and otherwise adds
# hundreds of unbuilt tests. Temporarily override both its visible value and
# cache entry while dependencies are added, then restore a parent project's
# original state exactly below.
if (DEFINED BUILD_TESTING)
    set(_APXCHOL_SAVED_BUILD_TESTING_VISIBLE TRUE)
    set(_APXCHOL_SAVED_BUILD_TESTING_VALUE "${BUILD_TESTING}")
else()
    set(_APXCHOL_SAVED_BUILD_TESTING_VISIBLE FALSE)
endif()
get_property(_APXCHOL_SAVED_BUILD_TESTING_TYPE
             CACHE BUILD_TESTING PROPERTY TYPE)
if (_APXCHOL_SAVED_BUILD_TESTING_TYPE)
    get_property(_APXCHOL_SAVED_BUILD_TESTING_CACHE
                 CACHE BUILD_TESTING PROPERTY VALUE)
    get_property(_APXCHOL_SAVED_BUILD_TESTING_HELP
                 CACHE BUILD_TESTING PROPERTY HELPSTRING)
endif()
# Eigen 3.4 resets CMP0077 to its legacy behavior, so a normal variable alone
# is insufficient: temporarily pin the cache too, then restore it exactly.
set(BUILD_TESTING OFF CACHE BOOL "Disable tests in apxchol dependencies" FORCE)
set(BUILD_TESTING OFF)

# ── Eigen ────────────────────────────────────────────
find_package(Eigen3 3.3 CONFIG QUIET)
if (NOT Eigen3_FOUND)
    FetchContent_Declare(eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(eigen)
    set_property(TARGET eigen APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
        "$<TARGET_PROPERTY:eigen,INTERFACE_INCLUDE_DIRECTORIES>")
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

if (_APXCHOL_SAVED_BUILD_TESTING_TYPE)
    set(BUILD_TESTING "${_APXCHOL_SAVED_BUILD_TESTING_CACHE}"
        CACHE ${_APXCHOL_SAVED_BUILD_TESTING_TYPE}
        "${_APXCHOL_SAVED_BUILD_TESTING_HELP}" FORCE)
else()
    unset(BUILD_TESTING CACHE)
endif()
if (_APXCHOL_SAVED_BUILD_TESTING_VISIBLE)
    set(BUILD_TESTING "${_APXCHOL_SAVED_BUILD_TESTING_VALUE}")
else()
    unset(BUILD_TESTING)
endif()
unset(_APXCHOL_SAVED_BUILD_TESTING_CACHE)
unset(_APXCHOL_SAVED_BUILD_TESTING_HELP)
unset(_APXCHOL_SAVED_BUILD_TESTING_TYPE)
unset(_APXCHOL_SAVED_BUILD_TESTING_VALUE)
unset(_APXCHOL_SAVED_BUILD_TESTING_VISIBLE)

# ── OpenMP ────────────────────────────────
include(cmake/apxchol_openmp.cmake)

# ── Build options ───────────────────────────────────
option(APXCHOL_USE_CUDA "GPU build: our sync-free dataflow CUDA SpTRSV kernel + the GPU-resident PCG with our own SpMV / vector kernels; links cudart only" OFF)
option(APXCHOL_64BIT_EDGE_INDICES "64-bit edge/offset indices, for factors or pools exceeding uint32 capacity (2^32-1 entries)" OFF)
option(APXCHOL_64BIT_NODE_INDICES "64-bit vertex indices (implies 64-bit edge indices)" OFF)
option(APXCHOL_64BIT_INDICES "Deprecated alias for APXCHOL_64BIT_EDGE_INDICES" OFF)
option(APXCHOL_POOL_FP32 "Store residual-pool weights in FP32; reads promote to FP64. OFF selects the FP64 reference." ON)
option(APXCHOL_BUILD_EXAMPLES "Build the examples/ programs" ON)
option(APXCHOL_BUILD_TESTS "Build the unit tests (fetches GoogleTest)" ON)
option(APXCHOL_BUILD_TOOLS "Build the developer analysis tools (bench_setup, analyze_factor)" OFF)
option(APXCHOL_USE_METAL "Apple GPU block PCG (explicit apxchol::metal_solver; defaults unchanged)" OFF)

if (APXCHOL_64BIT_INDICES)
    message(DEPRECATION "APXCHOL_64BIT_INDICES is deprecated; use APXCHOL_64BIT_EDGE_INDICES")
    set(APXCHOL_64BIT_EDGE_INDICES ON)
endif()

if (APXCHOL_USE_METAL)
    if (NOT APPLE)
        message(FATAL_ERROR "APXCHOL_USE_METAL requires an Apple platform")
    endif()
    if (APXCHOL_USE_CUDA)
        message(FATAL_ERROR "APXCHOL_USE_METAL and APXCHOL_USE_CUDA are mutually exclusive")
    endif()
    # Only src/metal_device.mm is Objective-C++; OpenMP host work stays in the
    # C++ sources (FindOpenMP's flags apply to CXX only).
    enable_language(OBJCXX)
endif()

if (APXCHOL_USE_CUDA)
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
add_library(apxchol_core
    src/factorization.cpp
    src/operator_class.cpp
    src/solve.cpp
    # Device precision-cast kernels for the mixed-precision GPU SpTRSV. Compiled
    # by nvcc only on the CUDA build (the .cpp TUs are host-compiled and can't
    # host __global__).
    $<$<BOOL:${APXCHOL_USE_CUDA}>:src/cuda_cast.cu>
    # Sync-free dataflow SpTRSV kernel (APXCHOL_GPU_SPTRSV=dataflow), same reason.
    $<$<BOOL:${APXCHOL_USE_CUDA}>:src/cuda_dataflow.cu>
    # GPU-resident residual topology for the opt-in block-region frontend.
    $<$<BOOL:${APXCHOL_USE_CUDA}>:src/cuda_block_frontend.cu>
    $<$<BOOL:${APXCHOL_USE_CUDA}>:src/cuda_round_shadow.cu>
    # The GPU-resident PCG's own CSR SpMV + fused vector kernels with
    # deterministic reductions (pcg_cuda.h; no cuSPARSE / cuBLAS).
    $<$<BOOL:${APXCHOL_USE_CUDA}>:src/cuda_pcg_kernels.cu>
    # Apple GPU block PCG (metal_solver.h): C++ host side and the Objective-C++
    # device side that compiles src/metal_kernels.inc at run time.
    $<$<BOOL:${APXCHOL_USE_METAL}>:src/metal_solver.cpp>
    $<$<BOOL:${APXCHOL_USE_METAL}>:src/metal_device.mm>
)
# The core uses Eigen and OpenMP. fast_matrix_market provides file I/O
# solely for the CLI and analyze_factor, and is linked on those targets.
target_link_libraries(apxchol_core PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)

target_include_directories(apxchol_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
)

if (APXCHOL_USE_CUDA)
    target_compile_definitions(apxchol_core PUBLIC APXCHOL_USE_CUDA)
    # cudart only: the SpTRSV kernels, the GPU-resident PCG's SpMV / vector
    # kernels and its reductions are all ours; no cuSPARSE or cuBLAS.
    target_link_libraries(apxchol_core PUBLIC CUDA::cudart)
    set_target_properties(apxchol_core PROPERTIES CUDA_STANDARD 23 CUDA_STANDARD_REQUIRED ON)
endif()

if (APXCHOL_USE_METAL)
    target_compile_definitions(apxchol_core PUBLIC APXCHOL_USE_METAL)
    set_source_files_properties(src/metal_device.mm PROPERTIES COMPILE_OPTIONS -fobjc-arc)
    set_target_properties(apxchol_core PROPERTIES OBJCXX_STANDARD 23 OBJCXX_STANDARD_REQUIRED ON)
    # PUBLIC: the static library hands the framework dependencies to its
    # consumers. CoreGraphics lets MTLCreateSystemDefaultDevice find the GPU
    # from command-line tools.
    target_link_libraries(apxchol_core PUBLIC "-framework Metal" "-framework Foundation"
                          "-framework CoreGraphics")
endif()

if (APXCHOL_64BIT_EDGE_INDICES)
    target_compile_definitions(apxchol_core PUBLIC APXCHOL_64BIT_EDGE_INDICES)
endif()
if (APXCHOL_64BIT_NODE_INDICES)
    target_compile_definitions(apxchol_core PUBLIC APXCHOL_64BIT_NODE_INDICES)
endif()
if (APXCHOL_POOL_FP32)
    target_compile_definitions(apxchol_core PUBLIC APXCHOL_POOL_FP32)
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
foreach (_apxchol_target IN ITEMS apxchol_core apxchol_mtx_input)
    target_compile_features(${_apxchol_target} PUBLIC cxx_std_23)
    if (APXCHOL_ARCH_FLAG)
        target_compile_options(${_apxchol_target} INTERFACE
            "$<$<COMPILE_LANGUAGE:CXX>:${APXCHOL_ARCH_FLAG}>")
    endif()
endforeach()
unset(_apxchol_target)
unset(APXCHOL_ARCH_FLAG)

# ── C ABI ────────────────────────────────────────────
# Exception-safe C interface over cpu_solver (include/apxchol/c_api.h) for
# non-C++ consumers. Not built with CUDA: there cpu_solver's triangular solves
# run on the device, so its CPU backend would be mislabelled.
if (NOT APXCHOL_USE_CUDA)
    add_library(apxchol_c src/c_api.cpp)
    target_link_libraries(apxchol_c PUBLIC apxchol_core)
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
    # Only the C consumer of the C ABI needs a C compiler.
    enable_language(C)
    enable_testing()
    add_subdirectory(tests)
endif()

# Developer tools are independent of GoogleTest and APXCHOL_BUILD_TESTS.
# Preserve their existing build/tests/ paths for scripts and users.
if (APXCHOL_BUILD_TOOLS)
    add_executable(bench_setup tests/bench_setup.cpp)
    target_link_libraries(bench_setup PRIVATE apxchol_core)
    add_executable(analyze_factor tests/analyze_factor.cpp)
    target_link_libraries(analyze_factor PRIVATE apxchol_core apxchol_mtx_input
                          fast_matrix_market::fast_matrix_market)
    set_target_properties(bench_setup analyze_factor PROPERTIES
                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tests")
endif()
