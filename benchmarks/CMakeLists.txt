cmake_minimum_required(VERSION 3.20)
project(laplacian_benchmarks LANGUAGES C CXX)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/apxchol_device.cmake)
if (APXCHOL_DEVICE STREQUAL "METAL")
    message(FATAL_ERROR "The competitor benchmark supports CPU or CUDA; use the root project for Metal")
endif()

# The shipped core configuration: C++23, Eigen, OpenMP, the architecture flag,
# index/pool options and, for CUDA, static cudart. apxchol_v1 below is built by
# the same _apxchol_configure_core() as the root library and Python module.
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/apxchol_core.cmake)
set(CMAKE_CXX_STANDARD 17) # The apxchol targets request C++23 explicitly.

# Competitors get the same machine-tuning flag as the core. It is recorded into
# the binary as BUILD_META's arch_flags (see the benchmark target), so a stored
# cell records the native tuning or portable baseline used by that build.
add_compile_options(-Wall -Wextra -Wpedantic ${_APXCHOL_ARCH_FLAG})
set(APXCHOL_BENCH_ARCH_FLAGS "${_APXCHOL_ARCH_FLAG}")

include(FetchContent)

# ── Path to benchmark sources ────────────────────────
set(BENCH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ── Matrix input and graph generators shared by the harness ──
add_library(bench_input STATIC ${BENCH_SRC}/graphs.cpp ${BENCH_SRC}/mmio.cpp)
target_link_libraries(bench_input PUBLIC Eigen3::Eigen)
target_include_directories(bench_input PUBLIC ${BENCH_SRC})

# ══════════════════════════════════════════════════════
# External solver dependencies (all fetched at configure time)
# ══════════════════════════════════════════════════════

# ── Optional: CHOLMOD (SuiteSparse, system) ──────────
find_library(CHOLMOD_LIB cholmod)
find_path(CHOLMOD_INCLUDE cholmod.h PATH_SUFFIXES suitesparse)
if (CHOLMOD_LIB AND CHOLMOD_INCLUDE)
    set(HAVE_CHOLMOD TRUE)
    message(STATUS "Found CHOLMOD: ${CHOLMOD_LIB}")
else()
    set(HAVE_CHOLMOD FALSE)
    message(STATUS "CHOLMOD not found — CHOLMOD solver disabled")
endif()

# ── Optional: MKL ────────────────────────────────────
# MKL backs RCHOL's own PCG (their util/pcg.cpp, compiled into rchol_lib below) and the
# ParAC CPU driver. Without it RCHOL uses the explicitly labelled portable PCG below;
# the ParAC CPU driver is skipped. Point at an install with
# -DMKL_ROOT=/path/to/oneapi/mkl/<version>, via the MKLROOT environment variable (oneAPI's setvars.sh exports it), or from a
# gitignored benchmarks/paths_local.cmake holding `set(MKL_ROOT ... CACHE PATH ...)`.
if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/paths_local.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/paths_local.cmake")
endif()
set(MKL_ROOT "$ENV{MKLROOT}" CACHE PATH "MKL root directory (empty = MKL-dependent solvers skipped)")
find_path(MKL_INCLUDE mkl.h HINTS ${MKL_ROOT}/include)
# ILP64, not LP64: RCHOL's util/pcg.hpp does `#define MKL_INT size_t` BEFORE including
# mkl_types.h, whose `#ifndef MKL_INT` guards let that definition win, so every MKL call
# in their shipped PCG passes 64-bit integers. Their own Makefile links -lmkl_intel_ilp64
# for exactly this reason. The LP64 layer is still resolved for the out-of-tree
# gpu_rchol / ParAC CPU driver, which uses MKL's default 32-bit MKL_INT.
find_library(MKL_ILP64 mkl_intel_ilp64 HINTS ${MKL_ROOT}/lib)
find_library(MKL_LP64 mkl_intel_lp64 HINTS ${MKL_ROOT}/lib)
find_library(MKL_SEQ mkl_sequential HINTS ${MKL_ROOT}/lib)  # sequential layer: gnu_thread thrashed (per-iter MKL sparse oversubscription, ~8x slower); rchol PCG is bandwidth-bound so threaded MKL gives no win anyway. Parallel-rchol = rchol_par (METIS factor) instead.
find_library(MKL_CORE mkl_core HINTS ${MKL_ROOT}/lib)
if (MKL_INCLUDE AND MKL_ILP64 AND MKL_SEQ AND MKL_CORE)
    set(HAVE_MKL TRUE)
    set(MKL_LIBRARIES ${MKL_ILP64} ${MKL_SEQ} ${MKL_CORE} pthread m dl)
    set(MKL_LIBRARIES_LP64 ${MKL_LP64} ${MKL_SEQ} ${MKL_CORE} pthread m dl)
    message(STATUS "Found MKL: ${MKL_ROOT} (ILP64 interface for RCHOL's util/pcg.cpp)")
else()
    set(HAVE_MKL FALSE)
    message(STATUS "MKL not found (MKL_ROOT='${MKL_ROOT}') — MKL-accelerated "
                   "solvers disabled; set -DMKL_ROOT=... or MKLROOT to enable")
endif()

# ── Optional: METIS (system, then portable source fallback) ───────────────────
option(APXCHOL_BENCH_FETCH_METIS
       "Fetch pinned GKlib/METIS sources when no system METIS is available" ON)
find_library(METIS_LIB metis)
find_path(METIS_INCLUDE metis.h)
if (METIS_LIB AND METIS_INCLUDE)
    set(HAVE_METIS TRUE)
    message(STATUS "Found METIS: ${METIS_LIB}")
elseif (APXCHOL_BENCH_FETCH_METIS)
    # METIS itself is portable; treating a missing system package as an x86-only
    # pRCHOL limitation left Grace/GH200 with a factor-only row.  Build only the
    # library sources (not METIS' command-line programs) against pinned GKlib.
    FetchContent_Declare(gklib
        GIT_REPOSITORY https://github.com/KarypisLab/GKlib.git
        GIT_TAG        3b7d61b9f885063c89901f3901fb4426f9cfb58f
        GIT_SHALLOW    TRUE
    )
    set(GKLIB_BUILD_APPS OFF CACHE BOOL "" FORCE)
    if (NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86"))
        set(NO_X86 ON CACHE BOOL "" FORCE)
    endif()
    FetchContent_MakeAvailable(gklib)
    FetchContent_Declare(metis_source
        GIT_REPOSITORY https://github.com/KarypisLab/METIS.git
        GIT_TAG        272d4a91c5f66c92327493339a476c553ebf1f5d
        GIT_SHALLOW    TRUE
    )
    FetchContent_GetProperties(metis_source)
    if (NOT metis_source_POPULATED)
        FetchContent_Populate(metis_source)
    endif()
    file(GLOB APXCHOL_BENCH_METIS_SOURCES
         CONFIGURE_DEPENDS "${metis_source_SOURCE_DIR}/libmetis/*.c")
    add_library(apxchol_bench_metis STATIC ${APXCHOL_BENCH_METIS_SOURCES})
    target_include_directories(apxchol_bench_metis PUBLIC
        ${metis_source_SOURCE_DIR}/include
        ${metis_source_SOURCE_DIR}/libmetis)
    target_link_libraries(apxchol_bench_metis PUBLIC GKlib::GKlib)
    # METIS' standalone build configures these two values into build/xinclude/
    # metis.h.  We compile only libmetis, so provide its documented/default ABI
    # explicitly (32-bit graph indices and float weights).
    target_compile_definitions(apxchol_bench_metis PUBLIC
        IDXTYPEWIDTH=32 REALTYPEWIDTH=32)
    set_target_properties(apxchol_bench_metis PROPERTIES
        C_STANDARD 99 POSITION_INDEPENDENT_CODE ON)
    set(METIS_LIB apxchol_bench_metis)
    set(METIS_INCLUDE ${metis_source_SOURCE_DIR}/include)
    set(HAVE_METIS TRUE)
    message(STATUS "System METIS not found — using pinned portable GKlib/METIS sources")
else()
    set(HAVE_METIS FALSE)
    message(STATUS "METIS not found — parallel RCHOL disabled")
endif()

# ── RCHOL (FetchContent — replaces former git submodule) ─
FetchContent_Declare(rchol
    GIT_REPOSITORY https://github.com/ut-padas/rchol.git
    GIT_TAG        cff22c967562c43047a19ee68f3f7e6bbb1779f1  # master as of 2026-10
)
FetchContent_GetProperties(rchol)
if (NOT rchol_POPULATED)
    FetchContent_Populate(rchol)
endif()

# rchol_lap.cpp carries a vestigial `#include<immintrin.h>` (no x86 intrinsic is
# actually used anywhere in the file) that breaks every non-x86 build (aarch64,
# e.g. Grace / GH200). Strip it from the fetched tree; upstream is unmaintained.
if (NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86"))
    file(READ ${rchol_SOURCE_DIR}/c++/rchol_lap/rchol_lap.cpp _rchol_lap_src)
    string(REPLACE "#include<immintrin.h>"
                   "// immintrin.h: x86-only, unused -- stripped for ${CMAKE_SYSTEM_PROCESSOR}"
                   _rchol_lap_src "${_rchol_lap_src}")
    file(WRITE ${rchol_SOURCE_DIR}/c++/rchol_lap/rchol_lap.cpp "${_rchol_lap_src}")
endif()

# spcol.cpp declares the two scratch arrays of gsl_spmatrix_add() as VLAs with an
# initialiser — a GCC extension that clang rejects outright, so the fetched tree
# does not build under clang on any ISA. Swap them for heap vectors of the same
# length, zero-initialised the same way; they live to the end of the function, so
# they still outlive every use of c->i / c->data (including the sp_copy at the
# bottom). Upstream is unmaintained.
#
# Applied unconditionally, NOT only under clang. gsl_spmatrix_add has no callers
# in the fetched tree (grep: one declaration in spcol.hpp, one definition here,
# zero calls), so the heap allocation can never execute and no rchol timing can
# move. Patching it under only one compiler would leave a competitor's sources
# differing between our gcc and clang builds — which would quietly bias the
# clang-vs-gcc comparison against RCHOL.
file(READ ${rchol_SOURCE_DIR}/c++/rchol_lap/spcol.cpp _rchol_spcol_src)
string(REPLACE "#include <stdexcept>\n\n#include \"spcol.hpp\""
               "#include <stdexcept>\n#include <vector>  // for the VLA replacement below\n\n#include \"spcol.hpp\""
               _rchol_spcol_src "${_rchol_spcol_src}")
string(REPLACE "    size_t i_array[ a->nz + 4 + b->nz] = {0};\n    double data_array[ a->nz + 4 + b->nz] = {0};"
               "    std::vector<size_t> i_array(a->nz + 4 + b->nz, 0);       // was a VLA (GCC ext)\n    std::vector<double> data_array(a->nz + 4 + b->nz, 0.0);  // was a VLA (GCC ext)"
               _rchol_spcol_src "${_rchol_spcol_src}")
string(REPLACE "c->i = i_array;" "c->i = i_array.data();"
               _rchol_spcol_src "${_rchol_spcol_src}")
string(REPLACE "c->data = data_array;" "c->data = data_array.data();"
               _rchol_spcol_src "${_rchol_spcol_src}")
file(WRITE ${rchol_SOURCE_DIR}/c++/rchol_lap/spcol.cpp "${_rchol_spcol_src}")

set(RCHOL_DIR ${rchol_SOURCE_DIR}/c++)
set(RCHOL_SOURCES
    ${RCHOL_DIR}/sparse.cpp
    ${RCHOL_DIR}/rchol/rchol.cpp
    ${RCHOL_DIR}/rchol_lap/rchol_lap.cpp
    ${RCHOL_DIR}/rchol_lap/spcol.cpp
    ${RCHOL_DIR}/util/util.cpp
)
option(APXCHOL_RCHOL_PORTABLE_PCG
       "Use the portable Eigen RCHOL PCG port even when upstream's MKL loop is available" OFF)
if (HAVE_MKL AND NOT APXCHOL_RCHOL_PORTABLE_PCG)
    # Their SHIPPED PCG. Every rchol/rchol_par solve number now comes from this file
    # (the same loop their ex_laplace.cpp drives), not from a re-implementation of it
    # in benchmark.cpp. It hard-includes mkl_spblas.h/mkl.h, so it compiles only where
    # MKL does (x86); without it benchmark.cpp supplies the explicitly labelled
    # portable PCG path below.
    list(APPEND RCHOL_SOURCES ${RCHOL_DIR}/util/pcg.cpp)
endif()
if (HAVE_METIS)
    list(APPEND RCHOL_SOURCES
        ${RCHOL_DIR}/rchol/find_separator.cpp
        ${RCHOL_DIR}/rchol/rchol_parallel.cpp
    )
endif()

add_library(rchol_lib STATIC ${RCHOL_SOURCES})
target_include_directories(rchol_lib PUBLIC
    ${RCHOL_DIR}
    ${RCHOL_DIR}/rchol
    ${RCHOL_DIR}/rchol_lap
    ${RCHOL_DIR}/util
)
set_target_properties(rchol_lib PROPERTIES
    CXX_STANDARD 14
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(rchol_lib PRIVATE -w -O3)
if (HAVE_METIS)
    target_compile_definitions(rchol_lib PUBLIC HAVE_METIS)
    target_include_directories(rchol_lib PUBLIC ${METIS_INCLUDE})
    target_link_libraries(rchol_lib PUBLIC ${METIS_LIB})
    target_link_libraries(rchol_lib PUBLIC OpenMP::OpenMP_CXX)
endif()
if (HAVE_MKL AND NOT APXCHOL_RCHOL_PORTABLE_PCG)
    target_compile_definitions(rchol_lib PUBLIC HAVE_MKL HAVE_RCHOL_PCG)
    target_include_directories(rchol_lib PUBLIC ${MKL_INCLUDE})
    target_link_libraries(rchol_lib PUBLIC ${MKL_LIBRARIES})
else()
    # The factorization, permutation, recurrence and stopping test remain RCHOL's.
    # Only MKL's sparse/vector calls are replaced by the explicit portable path in
    # benchmark.cpp, and BUILD_META labels the distinction in every stored cell.
    target_compile_definitions(rchol_lib PUBLIC APXCHOL_RCHOL_PORTABLE_PCG HAVE_RCHOL_PCG)
endif()

# ── AMGCL (header-only, FetchContent) ────────────────
FetchContent_Declare(amgcl
    GIT_REPOSITORY https://github.com/ddemidov/amgcl.git
    GIT_TAG        1.4.4
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(amgcl)
if (NOT amgcl_POPULATED)
    FetchContent_Populate(amgcl)
endif()
set(AMGCL_INCLUDE ${amgcl_SOURCE_DIR})

# ── Hypre (FetchContent — production multigrid library) ──
# Pinned to commit after PR #1547 (merged 2026-05-19) which adds the missing
# <thrust/tuple.h> include required by CUDA 13.x Thrust / CCCL headers.
# v3.1.0 had thrust::get not found errors against CUDA 13.2 + GCC 16.
FetchContent_Declare(hypre
    GIT_REPOSITORY https://github.com/hypre-space/hypre.git
    GIT_TAG 41cac9a4ca9655f9b2b88ee605158cc4bde55210  # post PR #1547; was v3.1.0
    SOURCE_SUBDIR src
)
set(HYPRE_ENABLE_MPI OFF CACHE BOOL "" FORCE)         # avoid MPI plumbing
set(HYPRE_ENABLE_OPENMP ON CACHE BOOL "" FORCE)       # threading via OMP
set(HYPRE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(HYPRE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HYPRE_ENABLE_UMPIRE OFF CACHE BOOL "" FORCE)      # opt out: no system Umpire
option(BENCH_HYPRE_USE_CUDA "Enable Hypre CUDA support (requires CUDA 13.x; needs PR #1547+)" OFF)
if (APXCHOL_DEVICE STREQUAL "CUDA" AND BENCH_HYPRE_USE_CUDA)
    set(HYPRE_ENABLE_CUDA ON CACHE BOOL "" FORCE)
endif()
FetchContent_MakeAvailable(hypre)

# ── GPU RCHOL — Liang et al. (2025) ──────────────────
# Builds both the CPU driver (experiment/) and GPU driver (gpu_implementation/).
# Invoked as standalone binaries from run.sh; parses stdout for timing.
option(BUILD_GPU_RCHOL "Build Liang et al. RCHOL drivers" OFF)
if (BUILD_GPU_RCHOL)
    FetchContent_Declare(gpu_rchol
        GIT_REPOSITORY https://github.com/Tianyu-Liang/Parallel-Randomized-Cholesky.git
        GIT_TAG        44ef39d2f5c2c52aa577f58f005d62f2675cefbc
        PATCH_COMMAND
            git apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/parac/0001-configurable-tolerance-and-any-thread-count.patch
            COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/parac/0002-report-complete-setup-boundaries.patch
            # Patch 0003 is generated with zero context so upstream's
            # whitespace-only blank lines cannot make the patch ambiguous.
            COMMAND git apply --unidiff-zero
                    ${CMAKE_CURRENT_SOURCE_DIR}/patches/parac/0003-report-complete-gpu-boundaries.patch
            COMMAND git apply --unidiff-zero
                    ${CMAKE_CURRENT_SOURCE_DIR}/patches/parac/0004-report-cuda-init-separately.patch
            COMMAND git apply
                    ${CMAKE_CURRENT_SOURCE_DIR}/patches/parac/0005-compensate-physics-global-sum.patch
            COMMAND git apply
                    ${CMAKE_CURRENT_SOURCE_DIR}/patches/parac/0006-separate-graph-algorithm-preparation.patch
            COMMAND git apply
                    ${CMAKE_CURRENT_SOURCE_DIR}/patches/parac/0007-original-residual-stopping.patch
            COMMAND git apply
                    ${CMAKE_CURRENT_SOURCE_DIR}/patches/parac/0008-benchmark-io-portable-solve-and-by-value-kernels.patch
    )
    FetchContent_Declare(fast_matrix_market
        GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market.git
        GIT_TAG        v1.7.6  # same release as the root project
        GIT_SHALLOW    TRUE
    )
    FetchContent_GetProperties(gpu_rchol)
    if (NOT gpu_rchol_POPULATED)
        FetchContent_Populate(gpu_rchol)
    endif()
    FetchContent_GetProperties(fast_matrix_market)
    if (NOT fast_matrix_market_POPULATED)
        FetchContent_Populate(fast_matrix_market)
    endif()

    # A SOURCE_DIR override skips FetchContent's PATCH_COMMAND. Fail closed rather
    # than build a driver whose partial timers our parser would silently accept.
    file(READ "${gpu_rchol_SOURCE_DIR}/experiment/driver_local.cpp" _parac_cpu_driver)
    file(READ "${gpu_rchol_SOURCE_DIR}/experiment/custom_cg.hpp" _parac_cpu_solver)
    file(READ "${gpu_rchol_SOURCE_DIR}/experiment/pre_process.hpp" _parac_cpu_prep)
    file(READ "${gpu_rchol_SOURCE_DIR}/gpu_implementation/driver.cu" _parac_gpu_driver)
    file(READ "${gpu_rchol_SOURCE_DIR}/gpu_implementation/solver.hpp" _parac_gpu_solver)
    function(_parac_require_marker _content_var _needle _patch_name)
        string(FIND "${${_content_var}}" "${_needle}" _found)
        if (_found EQUAL -1)
            message(FATAL_ERROR "gpu_rchol source is missing ${_patch_name}: ${_needle}")
        endif()
    endfunction()
    _parac_require_marker(_parac_cpu_driver "PARAC_REL_TOL" "patch 0001")
    _parac_require_marker(_parac_cpu_solver "rhs norm:" "patch 0001")
    _parac_require_marker(_parac_cpu_driver "APX factor setup time:" "patch 0002")
    _parac_require_marker(_parac_cpu_prep "APX adapter preprocessing time:" "patch 0002")
    _parac_require_marker(_parac_gpu_driver "APX GPU factor setup time:" "patch 0003")
    _parac_require_marker(_parac_gpu_solver "APX GPU solve phase time:" "patch 0003")
    _parac_require_marker(_parac_cpu_solver "APX stop contract: original-v1" "patch 0007")
    _parac_require_marker(_parac_gpu_solver "APX stop contract: original-v1" "patch 0007")
    _parac_require_marker(_parac_gpu_driver "APX CUDA init time:" "patch 0004")
    _parac_require_marker(_parac_cpu_solver "APXCHOL_PARAC_PORTABLE_CPP" "patch 0008")
    _parac_require_marker(_parac_gpu_solver "apx_io::write_xb" "patch 0008")
    file(READ "${gpu_rchol_SOURCE_DIR}/gpu_implementation/driver_physics.cu" _parac_gpu_physics)
    _parac_require_marker(_parac_gpu_physics
        "input_check_last(sparse_matrix_device<type_int, type_data> spmat" "patch 0008")

    # CPU solve backend: AUTO keeps upstream's oneMKL RCI loop when MKL LP64 is
    # found and otherwise builds patch 0008's serial portable translation.
    set(APXCHOL_PARAC_CPU_BACKEND "auto" CACHE STRING
        "ParAC CPU solve backend: auto, mkl-rci or portable-cpp")
    set_property(CACHE APXCHOL_PARAC_CPU_BACKEND PROPERTY STRINGS auto mkl-rci portable-cpp)
    set(_parac_cpu_backend "${APXCHOL_PARAC_CPU_BACKEND}")
    if (_parac_cpu_backend STREQUAL "auto")
        if (HAVE_MKL AND MKL_LP64)
            set(_parac_cpu_backend "mkl-rci")
        else()
            set(_parac_cpu_backend "portable-cpp")
        endif()
    endif()
    if (NOT _parac_cpu_backend MATCHES "^(mkl-rci|portable-cpp)$")
        message(FATAL_ERROR "APXCHOL_PARAC_CPU_BACKEND must be auto, mkl-rci or portable-cpp")
    endif()
    if (_parac_cpu_backend STREQUAL "mkl-rci" AND NOT (HAVE_MKL AND MKL_LP64))
        message(FATAL_ERROR "ParAC mkl-rci backend requested, but oneMKL LP64 was not found")
    endif()

    # CPU driver (experiment/ — includes MKL PCG solve). LP64: unlike RCHOL's pcg.hpp
    # this driver uses MKL's default 32-bit MKL_INT, so it must NOT get the ILP64 layer.
    if (OpenMP_CXX_FOUND)
        add_executable(gpu_rchol_cpu_driver
            ${gpu_rchol_SOURCE_DIR}/experiment/driver_local.cpp
        )
        target_compile_features(gpu_rchol_cpu_driver PRIVATE cxx_std_20)
        target_include_directories(gpu_rchol_cpu_driver PRIVATE
            ${gpu_rchol_SOURCE_DIR}/experiment
            ${gpu_rchol_SOURCE_DIR}/cpu_implementation
            ${fast_matrix_market_SOURCE_DIR}/include
        )
        target_link_libraries(gpu_rchol_cpu_driver PRIVATE OpenMP::OpenMP_CXX)
        if (_parac_cpu_backend STREQUAL "mkl-rci")
            target_compile_definitions(gpu_rchol_cpu_driver PRIVATE APXCHOL_PARAC_MKL_RCI)
            target_include_directories(gpu_rchol_cpu_driver PRIVATE ${MKL_INCLUDE})
            target_link_libraries(gpu_rchol_cpu_driver PRIVATE ${MKL_LIBRARIES_LP64})
        else()
            target_compile_definitions(gpu_rchol_cpu_driver PRIVATE APXCHOL_PARAC_PORTABLE_CPP)
        endif()
        # Upstream relies on transitive standard headers for INT_MAX/int64_t.
        # Supply them as build flags, without rewriting its source tree.
        # Keep each -include pair in one SHELL group: CMake otherwise de-duplicates
        # the repeated `-include` token and leaves `cstdint` as a bogus input file.
        target_compile_options(gpu_rchol_cpu_driver PRIVATE
            -O3 -w "SHELL:-include climits" "SHELL:-include cstdint")
        message(STATUS "GPU RCHOL CPU driver enabled: ParAC backend=${_parac_cpu_backend}")
    else()
        message(STATUS "GPU RCHOL CPU driver needs OpenMP — skipped")
    endif()

    # GPU driver (needs CUDA). The convergence test remains upstream: the runner
    # calibrates its existing tolerance knob from the true residual it prints.

    include(CheckLanguage)
    # Remember the requested CUDA host compiler BEFORE check_language runs. check_language()
    # executes a sub-CMake and writes `set(CMAKE_CUDA_HOST_COMPILER "<what the sub-build
    # resolved>")` into its result file (CheckLanguage.cmake:111), which SHADOWS the cache
    # entry in this scope. So -DCMAKE_CUDA_HOST_COMPILER was silently discarded and nvcc fell
    # back to its own default -- every benchmarks/build*-cuda configured this way was hosted
    # by whatever nvcc picked (g++-15 here) no matter what was requested, with no diagnostic.
    set(_apxchol_requested_cuda_host "${CMAKE_CUDA_HOST_COMPILER}")
    check_language(CUDA)
    if (_apxchol_requested_cuda_host AND
        NOT "${CMAKE_CUDA_HOST_COMPILER}" STREQUAL "${_apxchol_requested_cuda_host}")
        message(STATUS "Restoring requested CUDA host compiler: ${_apxchol_requested_cuda_host}")
        set(CMAKE_CUDA_HOST_COMPILER "${_apxchol_requested_cuda_host}"
            CACHE FILEPATH "Host compiler nvcc drives" FORCE)
    endif()
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)

        # Auto-detect GPU architecture if not specified
        if (NOT DEFINED GPU_RCHOL_CUDA_ARCH)
            # Try to detect via nvidia-smi
            execute_process(
                COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
                OUTPUT_VARIABLE _gpu_caps OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET RESULT_VARIABLE _smi_ok
            )
            if (_smi_ok EQUAL 0 AND _gpu_caps)
                # Take first GPU, convert "8.9" → "89"
                string(REGEX MATCH "[0-9]+\\.[0-9]+" _cap "${_gpu_caps}")
                string(REPLACE "." "" GPU_RCHOL_CUDA_ARCH "${_cap}")
                message(STATUS "Auto-detected GPU compute capability: ${_cap} → sm_${GPU_RCHOL_CUDA_ARCH}")
            else()
                set(GPU_RCHOL_CUDA_ARCH "80")
                message(STATUS "Could not detect GPU — defaulting to sm_80 (Ampere)")
            endif()
        endif()

        add_executable(gpu_rchol_gpu_driver
            ${gpu_rchol_SOURCE_DIR}/gpu_implementation/driver.cu
        )
        set_target_properties(gpu_rchol_gpu_driver PROPERTIES
            CUDA_STANDARD 20
            CUDA_ARCHITECTURES "${GPU_RCHOL_CUDA_ARCH}"
        )
        target_include_directories(gpu_rchol_gpu_driver PRIVATE
            ${gpu_rchol_SOURCE_DIR}/gpu_implementation
            ${gpu_rchol_SOURCE_DIR}/cpu_implementation
            ${fast_matrix_market_SOURCE_DIR}/include
        )
        target_link_libraries(gpu_rchol_gpu_driver PRIVATE
            cusparse cublas
        )
        target_link_libraries(gpu_rchol_gpu_driver PRIVATE OpenMP::OpenMP_CXX)
        target_compile_options(gpu_rchol_gpu_driver PRIVATE -O3 -w)
        message(STATUS "GPU RCHOL GPU driver enabled (gpu_implementation/driver.cu)")

        # Physics/SDDM variant (driver_physics.cu) — needed for IPM normal-equation
        # matrices (it augments+trims the last row/col). Same interface as driver.cu.
        add_executable(gpu_rchol_gpu_driver_physics
            ${gpu_rchol_SOURCE_DIR}/gpu_implementation/driver_physics.cu
        )
        set_target_properties(gpu_rchol_gpu_driver_physics PROPERTIES
            CUDA_STANDARD 20
            CUDA_ARCHITECTURES "${GPU_RCHOL_CUDA_ARCH}"
        )
        target_include_directories(gpu_rchol_gpu_driver_physics PRIVATE
            ${gpu_rchol_SOURCE_DIR}/gpu_implementation
            ${gpu_rchol_SOURCE_DIR}/cpu_implementation
            ${fast_matrix_market_SOURCE_DIR}/include
        )
        target_link_libraries(gpu_rchol_gpu_driver_physics PRIVATE cusparse cublas)
        target_link_libraries(gpu_rchol_gpu_driver_physics PRIVATE OpenMP::OpenMP_CXX)
        target_compile_options(gpu_rchol_gpu_driver_physics PRIVATE -O3 -w)
        message(STATUS "GPU RCHOL GPU physics driver enabled (gpu_implementation/driver_physics.cu)")
    else()
        message(STATUS "CUDA not found — GPU RCHOL GPU driver disabled")
    endif()
endif()

# ══════════════════════════════════════════════════════
# ApxChol v1 (C++23 solver — multiple IS × elimination combos)
# ══════════════════════════════════════════════════════
add_library(apxchol_v1 STATIC)
_apxchol_configure_core(apxchol_v1)

# Boost headers are needed only by AMGCL (HAVE_AMGCL gate below); the apxchol
# core itself is Boost-free.
find_package(Boost CONFIG QUIET)

# ══════════════════════════════════════════════════════
# Benchmark executable
# ══════════════════════════════════════════════════════
add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark PRIVATE bench_input rchol_lib apxchol_v1)
target_include_directories(benchmark PRIVATE ${AMGCL_INCLUDE})
target_compile_definitions(benchmark PRIVATE HAVE_RCHOL HAVE_APXCHOL_V1)
target_compile_features(benchmark PRIVATE cxx_std_23)

# ── BUILD_META inputs (emit_build_meta in src/benchmark.cpp) ───────────────────
# The compiler names itself through its own predefined macros and probes its
# OpenMP runtime at run time; these two facts it cannot know about itself. They
# are COMPILED IN rather than handed to the runner, so they travel with the
# binary: a runner reading them off a build-directory name would keep describing
# a stale binary by the directory it happens to sit in.
target_compile_definitions(benchmark PRIVATE
    APXCHOL_BUILD_ARCH_FLAGS="${APXCHOL_BENCH_ARCH_FLAGS}")
# dlsym for the OpenMP-runtime probe. Empty on glibc >= 2.34 (libdl merged into
# libc); CMAKE_DL_LIBS keeps it linking on the systems where it is not.
target_link_libraries(benchmark PRIVATE ${CMAKE_DL_LIBS})
if (APXCHOL_DEVICE STREQUAL "CUDA")
    # BUILD_META's cuda_host_compiler: the host compiler nvcc was pointed at with
    # -ccbin. Empty means CMake passed no -ccbin and nvcc used its own default,
    # which the binary reports as `unset(nvcc-default)` rather than as a name we
    # would be guessing.
    target_compile_definitions(benchmark PRIVATE
        APXCHOL_BUILD_CUDA_HOST_COMPILER="${CMAKE_CUDA_HOST_COMPILER}")
endif()

# AMGCL CUDA backend adapter (uses thrust + cuSPARSE, compiled with nvcc).
if (APXCHOL_DEVICE STREQUAL "CUDA" AND Boost_FOUND)
    target_sources(benchmark PRIVATE src/amgcl_cuda.cu)
    # -Xcompiler=-fopenmp is REQUIRED: AMGCL builds its AMG hierarchy host-side
    # with `#pragma omp parallel for` (smoothed_aggregation coarsening, SpMV,
    # transpose). Without -fopenmp in THIS nvcc TU those pragmas are no-ops and
    # the setup runs single-threaded — which made amgcl_cuda's setup 2-10x slower
    # than BoomerAMG-GPU (e.g. G3_circuit 2.1s vs 0.5s). The OpenMP runtime is
    # already linked transitively via apxchol_v1.
    set_source_files_properties(src/amgcl_cuda.cu PROPERTIES
        LANGUAGE CUDA
        COMPILE_OPTIONS "--extended-lambda;--expt-relaxed-constexpr;-Xcompiler=-fopenmp")
    set_target_properties(benchmark PROPERTIES
        CUDA_STANDARD 23
        CUDA_STANDARD_REQUIRED ON)
    # AMGCL uses cuSPARSE. This dependency belongs to the competitor driver;
    # our apxchol library links static cudart only, which also serves AMGCL.
    target_link_libraries(benchmark PRIVATE CUDA::cusparse)
endif()

if (Boost_FOUND)
    target_compile_definitions(benchmark PRIVATE HAVE_AMGCL)
    if (TARGET Boost::headers)
        target_link_libraries(benchmark PRIVATE Boost::headers)
    endif()
    message(STATUS "Boost found — AMGCL solver enabled")
else()
    message(STATUS "Boost not found — AMGCL solver disabled (needs Boost headers)")
endif()
target_link_libraries(benchmark PRIVATE HYPRE)
target_compile_definitions(benchmark PRIVATE HAVE_HYPRE)
if (APXCHOL_DEVICE STREQUAL "CUDA" AND BENCH_HYPRE_USE_CUDA)
    # The benchmark must not expose a CPU-only Hypre build under the GPU label.
    # This definition describes how the linked HYPRE target was configured; the
    # generic APXCHOL_USE_CUDA definition says only that our own CUDA path exists.
    target_compile_definitions(benchmark PRIVATE APXCHOL_BENCH_HYPRE_CUDA)
endif()
if (HAVE_CHOLMOD)
    target_compile_definitions(benchmark PRIVATE HAVE_CHOLMOD)
    target_include_directories(benchmark PRIVATE ${CHOLMOD_INCLUDE})
    target_link_libraries(benchmark PRIVATE ${CHOLMOD_LIB})
endif()
if (HAVE_METIS)
    target_compile_definitions(benchmark PRIVATE HAVE_METIS)
endif()
if (HAVE_MKL)
    target_compile_definitions(benchmark PRIVATE HAVE_MKL)
    target_include_directories(benchmark PRIVATE ${MKL_INCLUDE})
    target_link_libraries(benchmark PRIVATE ${MKL_LIBRARIES})
endif()


# ── Guard-rail tests ────────────────────────────────────────────────────────────
# The matrix contract is DECLARED, never sniffed: --kind says how to read the file,
# --class says whether the system it defines is singular. Both are required with
# --mtx, and --class is cross-checked against the structural scan so a WRONG
# declaration is a hard error rather than a quietly different benchmark.
#
# Those three guards are the whole defence, and they had no automated coverage --
# every negative case was only ever run by hand. A benchmark that silently solves
# the wrong system is exactly the failure this project has paid for repeatedly, so
# the guards get tests even though nothing else here does.
enable_testing()

function(test_solver_rejection name solver)
    add_test(NAME ${name}
        COMMAND ${CMAKE_COMMAND}
            -DBENCHMARK=$<TARGET_FILE:benchmark> -DSOLVER=${solver}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_solver_rejection.cmake)
endfunction()
test_solver_rejection(contract_rejects_unknown_solver unknown_solver)
test_solver_rejection(contract_rejects_retired_solver apxchol)
test_solver_rejection(contract_rejects_no_solver none)
test_solver_rejection(contract_all_rejects_unknown_solver "all,unknown_solver")
add_test(NAME contract_rejects_retired_storage
    COMMAND ${CMAKE_COMMAND} -DBENCHMARK=$<TARGET_FILE:benchmark>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_storage_rejection.cmake)
add_test(NAME contract_v1_complete_cpu_route
    COMMAND ${CMAKE_COMMAND} -DBENCHMARK=$<TARGET_FILE:benchmark>
        -DUSE_CUDA=$<STREQUAL:${APXCHOL_DEVICE},CUDA> -DWIDE_NODES=$<STREQUAL:${APXCHOL_INDEX_WIDTH},64>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_v1_complete_route.cmake)
add_test(NAME contract_none_allows_input_exports
    COMMAND ${CMAKE_COMMAND} -DBENCHMARK=$<TARGET_FILE:benchmark>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_input_exports.cmake)
if (NOT Boost_FOUND)
    test_solver_rejection(contract_rejects_unavailable_amgcl amgcl)
endif()
if (NOT APXCHOL_DEVICE STREQUAL "CUDA")
    test_solver_rejection(contract_rejects_unavailable_cuda amgcl_cuda)
endif()

foreach(case requires_class rejects_wrong_class rejects_class_on_graph)
    add_test(NAME contract_${case}
        COMMAND ${CMAKE_COMMAND}
            -DBENCHMARK=$<TARGET_FILE:benchmark> -DCASE=${case}
            -DMATRIX=${CMAKE_CURRENT_SOURCE_DIR}/../tests/fixtures/path5_laplacian.mtx
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_class_rejection.cmake)
endforeach()

add_executable(original_stop_tests tests/test_original_stop.cpp)
add_test(NAME contract_original_stopping COMMAND original_stop_tests)

add_executable(split_components_tests tests/test_split_components.cpp)
target_compile_features(split_components_tests PRIVATE cxx_std_23)
target_link_libraries(split_components_tests PRIVATE bench_input apxchol_v1 ${CMAKE_DL_LIBS})
add_test(NAME contract_split_components COMMAND split_components_tests)

add_executable(v1_config_tests tests/test_v1_configs.cpp)
target_compile_features(v1_config_tests PRIVATE cxx_std_23)
target_compile_definitions(v1_config_tests PRIVATE HAVE_APXCHOL_V1)
target_link_libraries(v1_config_tests PRIVATE bench_input apxchol_v1 ${CMAKE_DL_LIBS})
add_test(NAME contract_v1_configs COMMAND v1_config_tests)

# RCHOL's stdout silencing must tolerate the parallel variant's worker threads.
find_package(Threads REQUIRED)
add_executable(discard_streambuf_tests tests/test_discard_streambuf.cpp)
target_include_directories(discard_streambuf_tests PRIVATE src)
target_link_libraries(discard_streambuf_tests PRIVATE Threads::Threads)
add_test(NAME contract_discard_streambuf COMMAND discard_streambuf_tests)
