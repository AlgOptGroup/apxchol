cmake_minimum_required(VERSION 3.16)
project(apxchol VERSION 0.2.2 LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE not set; defaulting to Release")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -march=native is the default for local builds; turn OFF for portable
# binaries (distributed wheels/packages would SIGILL on older CPUs).
option(APXCHOL_NATIVE_ARCH "Compile with -march=native (Release/RelWithDebInfo)" ON)

if (NOT MSVC)
    add_compile_options(-Wall -Wextra -Wpedantic)
    if (APXCHOL_NATIVE_ARCH AND
        (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
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

# ── Build options ───────────────────────────────────
option(APXCHOL_USE_CUDA "Use cuSPARSE (GPU) for sparse triangular solve" OFF)
option(APXCHOL_64BIT_EDGE_INDICES "64-bit edge/offset indices, for factors with more than 2^31 nonzeros" OFF)
option(APXCHOL_64BIT_NODE_INDICES "64-bit vertex indices (implies 64-bit edge indices)" OFF)
option(APXCHOL_64BIT_INDICES "Deprecated alias for APXCHOL_64BIT_EDGE_INDICES" OFF)
option(APXCHOL_SPTRSV_FP32 "fp32 SpTRSV factor values, diagonal included (the PCG recurrence stays fp64; still converges to 1e-8); OFF gives an fp64 baseline" ON)
option(APXCHOL_POOL_FP32 "fp32 residual-pool edge weights (reads promote to fp64; convergence unchanged); OFF gives an fp64 baseline" ON)
set(APXCHOL_SPTRSV_LOWPREC "OFF" CACHE STRING "Low-precision SpTRSV factor-value STORAGE variant for the OFF-diagonals (diagonal kept exact fp32 in a separate array; reads widen to fp64 in registers, compute unchanged -- a preconditioner-quality knob: PCG still converges, maybe in more iters): OFF | FP16_SCALED (IEEE binary16 of L_ij / s_j, per-column scale s_j = max |off-diagonal|; entries < 2^-25 of their column max flush to zero -- counted, printed under APXCHOL_VERBOSE). See include/apxchol/lowprec.h. CPU/omp backend only -- with APXCHOL_USE_CUDA=ON the value is treated as OFF for that build (fp32 CPU storage, STATUS message), never an error; takes precedence over APXCHOL_SPTRSV_FP32. Under FP16_SCALED stored fp16 subnormals are flushed to zero by default (env APXCHOL_FP16_KEEP_SUBNORMAL=1 keeps them) and the per-column storage-rounding residual can be folded into the fp32 diagonal (env APXCHOL_LOWPREC_DIAG_COMP=1). Every build honours the runtime env APXCHOL_FACTOR_DROP=<rel> (compacting drop: off-diagonals below rel * column max are REMOVED from the SpTRSV CSR/CSC at setup), APXCHOL_FTZ=1 (MXCSR FTZ+DAZ on every thread at PCG entry) and APXCHOL_FP16_GATHER=simd|scalar (fat-level kernel flavour of the fp16 storage); FP16_SCALED also APXCHOL_FP16_DIAG=1 (divide by the fp16 diagonal slot; refused unless every slot is a normal fp16). The scale is folded into the vectors (forward_solve returns D y, transpose_solve takes it); see include/apxchol/solver/sptrsv/omp.h")
set_property(CACHE APXCHOL_SPTRSV_LOWPREC PROPERTY STRINGS OFF FP16_SCALED)
option(APXCHOL_BUILD_EXAMPLES "Build the examples/ programs" ON)
option(APXCHOL_BUILD_TESTS "Build the unit tests (fetches GoogleTest)" ON)
option(APXCHOL_BUILD_TOOLS "Build the developer analysis tools (bench_setup, analyze_factor)" OFF)

if (APXCHOL_64BIT_INDICES)
    message(DEPRECATION "APXCHOL_64BIT_INDICES is deprecated; use APXCHOL_64BIT_EDGE_INDICES")
    set(APXCHOL_64BIT_EDGE_INDICES ON)
endif()

# Normalise the low-precision variant: case, validity.
string(TOUPPER "${APXCHOL_SPTRSV_LOWPREC}" APXCHOL_SPTRSV_LOWPREC)
if (NOT APXCHOL_SPTRSV_LOWPREC MATCHES "^(OFF|FP16_SCALED)$")
    message(FATAL_ERROR "APXCHOL_SPTRSV_LOWPREC must be OFF or FP16_SCALED (got '${APXCHOL_SPTRSV_LOWPREC}'; the BF16 / BF16_SCALED / FP24 variants were removed 2026-08-18).")
endif()
# CUDA builds have no COMPILE-TIME low-precision SpTRSV path (the cuSPARSE
# backend is fp32/fp64 only; the GPU's fp16 storage is the RUNTIME opt-in
# APXCHOL_GPU_SPTRSV_FP16=1 on our dataflow / level-set kernels, see
# include/apxchol/solver/sptrsv/cuda.h): with APXCHOL_USE_CUDA=ON any APXCHOL_SPTRSV_LOWPREC value is
# TREATED AS OFF for this build -- CPU SpTRSV storage forced to fp32, no
# APXCHOL_SPTRSV_LOWPREC_* macro defined -- with a STATUS line, instead of
# failing the configure. So a LOWPREC default other than OFF (or a shared
# cache / preset) can never break a CUDA build; the requested value stays in
# the cache untouched (it is what the CPU builds of the same tree get). Same
# rule in benchmarks/CMakeLists.txt.
if (NOT APXCHOL_SPTRSV_LOWPREC STREQUAL "OFF" AND APXCHOL_USE_CUDA)
    message(STATUS "APXCHOL_SPTRSV_LOWPREC=${APXCHOL_SPTRSV_LOWPREC} requested together with APXCHOL_USE_CUDA=ON: the CUDA backend has no low-precision SpTRSV path -- CPU SpTRSV storage forced to fp32 (APXCHOL_SPTRSV_LOWPREC treated as OFF) for this build")
    set(APXCHOL_SPTRSV_LOWPREC "OFF")
endif()

if (APXCHOL_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    # Must be set before the target is created — CMAKE_CUDA_ARCHITECTURES is
    # read at add_library() time. The "native" value requires CMake >= 3.24;
    # pass -DCMAKE_CUDA_ARCHITECTURES=<sm> explicitly on older CMake.
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
    src/solve.cpp
    # Device precision-cast kernels for the mixed-precision GPU SpTRSV. Compiled
    # by nvcc only on the CUDA build (the .cpp TUs are host-compiled and can't
    # host __global__). Harmless/unused when APXCHOL_SPTRSV_FP32 is off.
    $<$<BOOL:${APXCHOL_USE_CUDA}>:src/cuda_cast.cu>
    # Level-set SpTRSV kernels: sptrsv/cuda.h calls levelset_solve
    # unconditionally, so the CUDA build needs this TU to link.
    $<$<BOOL:${APXCHOL_USE_CUDA}>:src/cuda_levelset.cu>
    # Sync-free dataflow SpTRSV kernel (APXCHOL_GPU_SPTRSV=dataflow), same reason.
    $<$<BOOL:${APXCHOL_USE_CUDA}>:src/cuda_dataflow.cu>
)
# Eigen is the only dependency of the core TUs/headers.  fast_matrix_market
# is file I/O used solely by the CLI and analyze_factor — linked on those
# targets, not here.
target_link_libraries(apxchol_core PUBLIC Eigen3::Eigen)
if (OpenMP_CXX_FOUND)
    target_link_libraries(apxchol_core PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(apxchol_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
)

if (APXCHOL_USE_CUDA)
    target_compile_definitions(apxchol_core PUBLIC APXCHOL_USE_CUDA)
    target_link_libraries(apxchol_core PUBLIC CUDA::cudart CUDA::cusparse CUDA::cublas)
    # nvcc (13.x) + recent CMake reject a "CUDA23" dialect, so the .cu device TUs
    # compile as C++20 while the host .cpp TUs stay C++23 — same fix as
    # benchmarks/apxchol_v1. Without this the CUDA configure fails outright.
    set_target_properties(apxchol_core PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
endif()

if (APXCHOL_64BIT_EDGE_INDICES)
    target_compile_definitions(apxchol_core PUBLIC APXCHOL_64BIT_EDGE_INDICES)
endif()
if (APXCHOL_64BIT_NODE_INDICES)
    target_compile_definitions(apxchol_core PUBLIC APXCHOL_64BIT_NODE_INDICES)
endif()
if (APXCHOL_SPTRSV_FP32)
    target_compile_definitions(apxchol_core PUBLIC APXCHOL_SPTRSV_FP32)
endif()
if (NOT APXCHOL_SPTRSV_LOWPREC STREQUAL "OFF")
    target_compile_definitions(apxchol_core PUBLIC APXCHOL_SPTRSV_LOWPREC_${APXCHOL_SPTRSV_LOWPREC})
endif()
if (APXCHOL_POOL_FP32)
    target_compile_definitions(apxchol_core PUBLIC APXCHOL_POOL_FP32)
endif()

# ── CLI solver ───────────────────────────────────────
add_executable(apxchol src/main.cpp src/config.cpp)
target_link_libraries(apxchol PRIVATE apxchol_core CLI11::CLI11 spdlog::spdlog
                      fast_matrix_market::fast_matrix_market)

# ── Examples ─────────────────────────────────────────
if (APXCHOL_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ── Tests ────────────────────────────────────────────
if (APXCHOL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
