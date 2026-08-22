cmake_minimum_required(VERSION 3.16)
project(laplacian_benchmarks LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The machine-tuning flags actually put on the command line below. Recorded into
# the binary as BUILD_META's arch_flags (see the benchmark target), so a stored
# cell says whether its numbers came from a tuned build. Empty = untuned.
set(APXCHOL_BENCH_ARCH_FLAGS "")

if (NOT MSVC)
    add_compile_options(-Wall -Wextra -Wpedantic)
    if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        # This is a SEPARATE cmake project (not an add_subdirectory of the root),
        # so the shared native-tuning module is included by relative path.
        include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/apxchol_native_arch.cmake)
        if (APXCHOL_NATIVE_ARCH_FLAG)
            add_compile_options(${APXCHOL_NATIVE_ARCH_FLAG})
            set(APXCHOL_BENCH_ARCH_FLAGS "${APXCHOL_NATIVE_ARCH_FLAG}")
        endif()
    endif()
endif()

include(FetchContent)

# ── Path to benchmark sources ────────────────────────
set(BENCH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ── Shared sources from parent project ───────────────
set(PARENT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../src)
set(PARENT_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# ── Legacy v0 baseline sources (graphs, simple_solver, mmio) ──
set(ARCHIVE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/v0)
set(ARCHIVE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/src/v0)

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

# ── Core library (legacy v0 solver) ───────────────────
add_library(laplacian_core STATIC
    ${ARCHIVE_SRC}/graphs.cpp
    ${ARCHIVE_SRC}/simple_solver.cpp
    ${ARCHIVE_SRC}/mmio.cpp
)
target_link_libraries(laplacian_core PUBLIC Eigen3::Eigen)
target_include_directories(laplacian_core PUBLIC ${ARCHIVE_INCLUDE})

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
# ParAC CPU driver. Without it the rchol/rchol_par cells become FACTOR-ONLY rows and the
# ParAC CPU driver is skipped entirely. Point at an install with
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

# ── Optional: METIS (system) ─────────────────────────
find_library(METIS_LIB metis)
find_path(METIS_INCLUDE metis.h)
if (METIS_LIB AND METIS_INCLUDE)
    set(HAVE_METIS TRUE)
    message(STATUS "Found METIS: ${METIS_LIB}")
else()
    set(HAVE_METIS FALSE)
    message(STATUS "METIS not found — parallel RCHOL disabled")
endif()

# ── RCHOL (FetchContent — replaces former git submodule) ─
FetchContent_Declare(rchol
    GIT_REPOSITORY https://github.com/ut-padas/rchol.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
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
if (HAVE_MKL)
    # Their SHIPPED PCG. Every rchol/rchol_par solve number now comes from this file
    # (the same loop their ex_laplace.cpp drives), not from a re-implementation of it
    # in benchmark.cpp. It hard-includes mkl_spblas.h/mkl.h, so it compiles only where
    # MKL does (x86); without it the RCHOL rows are factor-only — see benchmark.cpp.
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
    find_package(OpenMP QUIET)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(rchol_lib PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()
if (HAVE_MKL)
    target_compile_definitions(rchol_lib PUBLIC HAVE_MKL)
    target_include_directories(rchol_lib PUBLIC ${MKL_INCLUDE})
    target_link_libraries(rchol_lib PUBLIC ${MKL_LIBRARIES})
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
if (APXCHOL_USE_CUDA AND BENCH_HYPRE_USE_CUDA)
    set(HYPRE_ENABLE_CUDA ON CACHE BOOL "" FORCE)
endif()
# Pre-enable CUDA language before FetchContent_MakeAvailable(hypre) so that
# Hypre's internal enable_language(CUDA) call is a no-op (avoids re-running
# the compiler ID test, which fails with GCC 16 + CUDA 13.2 stdlib headers).
# Also set CMAKE_CUDA_ARCHITECTURES so CMake's internal try_compile tests
# don't fail with "CUDA_ARCHITECTURES is empty".
if (APXCHOL_USE_CUDA OR BENCH_HYPRE_USE_CUDA)
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
        if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            # Auto-detect via nvidia-smi, fallback to native (CMake 3.18+ keyword)
            execute_process(
                COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
                OUTPUT_VARIABLE _gpu_caps OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET RESULT_VARIABLE _smi_ok
            )
            if (_smi_ok EQUAL 0 AND _gpu_caps)
                string(REGEX MATCH "[0-9]+\\.[0-9]+" _cap "${_gpu_caps}")
                string(REPLACE "." "" _arch "${_cap}")
                set(CMAKE_CUDA_ARCHITECTURES "${_arch}" CACHE STRING "CUDA architectures")
                message(STATUS "Auto-detected CUDA architecture: sm_${_arch}")
            else()
                set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures")
            endif()
        endif()
        enable_language(CUDA)
    endif()
endif()
FetchContent_MakeAvailable(hypre)

# ── GPU RCHOL — Liang et al. (2025) ──────────────────
# Builds both the CPU driver (experiment/) and GPU driver (gpu_implementation/).
# Invoked as standalone binaries from run.sh; parses stdout for timing.
option(BUILD_GPU_RCHOL "Build Liang et al. RCHOL drivers" OFF)
if (BUILD_GPU_RCHOL)
    FetchContent_Declare(gpu_rchol
        GIT_REPOSITORY https://github.com/Tianyu-Liang/Parallel-Randomized-Cholesky.git
        GIT_TAG        main
        GIT_SHALLOW    TRUE
    )
    FetchContent_Declare(fast_matrix_market
        GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market.git
        GIT_TAG        main
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

    find_package(OpenMP QUIET)

    # experiment/auxilliary.hpp uses INT_MAX (line ~121) without including
    # <climits>; it only compiles where libstdc++ happens to leak the macro in.
    # It is the copy the CPU driver actually sees (experiment/custom_cg.hpp
    # includes it by quoted name from the same directory), and clang does not
    # leak it, so add the include the source is missing.
    set(_exp_aux_hpp "${gpu_rchol_SOURCE_DIR}/experiment/auxilliary.hpp")
    file(READ "${_exp_aux_hpp}" _exp_aux_content)
    string(FIND "${_exp_aux_content}" "#include <climits>" _exp_aux_pos)
    if (_exp_aux_pos EQUAL -1)
        string(REPLACE
            "#include \"pre_process.hpp\""
            "#include \"pre_process.hpp\"\n#include <climits>  // INT_MAX"
            _exp_aux_content "${_exp_aux_content}")
        file(WRITE "${_exp_aux_hpp}" "${_exp_aux_content}")
        message(STATUS "Patched gpu_rchol experiment/auxilliary.hpp with the missing <climits>")
    endif()

    # CPU driver (experiment/ — includes MKL PCG solve). LP64: unlike RCHOL's pcg.hpp
    # this driver uses MKL's default 32-bit MKL_INT, so it must NOT get the ILP64 layer.
    if (OpenMP_CXX_FOUND AND HAVE_MKL AND MKL_LP64)
        add_executable(gpu_rchol_cpu_driver
            ${gpu_rchol_SOURCE_DIR}/experiment/driver_local.cpp
        )
        target_compile_features(gpu_rchol_cpu_driver PRIVATE cxx_std_20)
        target_include_directories(gpu_rchol_cpu_driver PRIVATE
            ${gpu_rchol_SOURCE_DIR}/experiment
            ${gpu_rchol_SOURCE_DIR}/cpu_implementation
            ${fast_matrix_market_SOURCE_DIR}/include
            ${MKL_INCLUDE}
        )
        target_link_libraries(gpu_rchol_cpu_driver PRIVATE
            OpenMP::OpenMP_CXX ${MKL_LIBRARIES_LP64}
        )
        target_compile_options(gpu_rchol_cpu_driver PRIVATE -O3 -w)
        message(STATUS "GPU RCHOL CPU driver enabled (experiment/driver_local.cpp)")
    else()
        message(STATUS "GPU RCHOL CPU driver needs OpenMP + MKL (LP64) — skipped")
    endif()

    # GPU driver (needs CUDA)
    # Patch const-correctness for CUB 2.x (CUDA 13+)
    set(_aux_hpp "${gpu_rchol_SOURCE_DIR}/gpu_implementation/auxilliary.hpp")
    file(READ "${_aux_hpp}" _aux_content)
    string(FIND "${_aux_content}" "compare_row_reference(Output<int, double> &a" _pos)
    if (NOT _pos EQUAL -1)
        string(REPLACE
            "compare_row_reference(Output<int, double> &a, Output<int, double> &b)"
            "compare_row_reference(const Output<int, double> &a, const Output<int, double> &b)"
            _aux_content "${_aux_content}")
        string(REPLACE
            "compare_value_reference(Output<int, double> &a, Output<int, double> &b)"
            "compare_value_reference(const Output<int, double> &a, const Output<int, double> &b)"
            _aux_content "${_aux_content}")
        file(WRITE "${_aux_hpp}" "${_aux_content}")
        message(STATUS "Patched gpu_rchol auxilliary.hpp for CUDA 13+ const-correctness")
    endif()

    # Patch the GPU PCG stopping criterion to the TRUE Euclidean relative
    # residual ||r||_2 / ||b||_2 (x0=0 => ||r0||_2 = ||b||_2) — the same metric
    # Eigen CG / apxchol and the already-patched CPU ParAC driver use. The
    # pristine solver.hpp stops on the PRECONDITIONED residual ratio
    # sqrt(<r,M^-1 r>)/sqrt(<r0,M^-1 r0>), which is optimistic and halts at a true
    # residual up to ~10x above tol on IPM matrices, making the iteration count
    # an unfair apples-to-oranges vs every other solver (which all reach true
    # <=1e-8). r1=<r,z> is still used for the CG alpha/beta recurrence; only the
    # convergence TEST changes, so the iteration math is untouched.
    set(_solver_hpp "${gpu_rchol_SOURCE_DIR}/gpu_implementation/solver.hpp")
    file(READ "${_solver_hpp}" _solver_content)
    string(FIND "${_solver_content}" "initial_exit_tol" _spos)
    if (_spos EQUAL -1)
        string(REPLACE
            "bool tol_condition = r1 > TOL;"
            "double initial_exit_tol = exit_tol;\n    bool tol_condition = (exit_tol / initial_exit_tol) > TOL;"
            _solver_content "${_solver_content}")
        string(REPLACE
            "tol_condition = (sqrt(r1) / sqrt(initial_r1)) > TOL;"
            "tol_condition = (exit_tol / initial_exit_tol) > TOL;"
            _solver_content "${_solver_content}")
        file(WRITE "${_solver_hpp}" "${_solver_content}")
        message(STATUS "Patched gpu_rchol solver.hpp PCG to stop on true relative residual (fairness)")
    endif()

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
        if (OpenMP_CXX_FOUND)
            target_link_libraries(gpu_rchol_gpu_driver PRIVATE OpenMP::OpenMP_CXX)
        endif()
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
        if (OpenMP_CXX_FOUND)
            target_link_libraries(gpu_rchol_gpu_driver_physics PRIVATE OpenMP::OpenMP_CXX)
        endif()
        target_compile_options(gpu_rchol_gpu_driver_physics PRIVATE -O3 -w)
        message(STATUS "GPU RCHOL GPU physics driver enabled (gpu_implementation/driver_physics.cu)")
    else()
        message(STATUS "CUDA not found — GPU RCHOL GPU driver disabled")
    endif()
endif()

# ══════════════════════════════════════════════════════
# ApxChol v1 (C++23 solver — multiple IS × elimination combos)
# ══════════════════════════════════════════════════════
find_package(OpenMP QUIET)

add_library(apxchol_v1 STATIC
    ${PARENT_SRC}/factorization.cpp
    # The operator class the factorization asserts its input against
    # (include/apxchol/operator_class.h). factorization.cpp constructs an
    # operator_view, and benchmark.cpp calls require_operator/describe_operator
    # directly for --kind operator, so both need this translation unit.
    ${PARENT_SRC}/operator_class.cpp
    ${PARENT_SRC}/solve.cpp
)
target_compile_features(apxchol_v1 PUBLIC cxx_std_23)
target_link_libraries(apxchol_v1 PUBLIC Eigen3::Eigen)
target_include_directories(apxchol_v1 PUBLIC ${PARENT_INCLUDE})
if (OpenMP_CXX_FOUND)
    target_link_libraries(apxchol_v1 PUBLIC OpenMP::OpenMP_CXX)
endif()
# Match the root project's two unsigned index-width knobs. The default uint32
# edge_index already handles com-Orkut's roughly 2.15e9-entry raw factor; widen
# only when a factor or residual pool can exceed 2^32-1 entries. Definitions are
# PUBLIC so benchmark.cpp and the linked library always agree on their ABI.
option(APXCHOL_64BIT_EDGE_INDICES "64-bit apxchol edge/offset indices, for factors or pools exceeding uint32 capacity (2^32-1 entries)" OFF)
option(APXCHOL_64BIT_NODE_INDICES "64-bit apxchol vertex indices (implies 64-bit edge indices)" OFF)
option(APXCHOL_64BIT_INDICES "Deprecated alias for APXCHOL_64BIT_EDGE_INDICES" OFF)
if (APXCHOL_64BIT_INDICES)
    message(DEPRECATION "APXCHOL_64BIT_INDICES is deprecated; use APXCHOL_64BIT_EDGE_INDICES")
    set(APXCHOL_64BIT_EDGE_INDICES ON)
endif()
if (APXCHOL_64BIT_EDGE_INDICES)
    target_compile_definitions(apxchol_v1 PUBLIC APXCHOL_64BIT_EDGE_INDICES)
endif()
if (APXCHOL_64BIT_NODE_INDICES)
    target_compile_definitions(apxchol_v1 PUBLIC APXCHOL_64BIT_NODE_INDICES)
endif()
option(APXCHOL_USE_CUDA "GPU build of apxchol_v1: our sync-free dataflow CUDA SpTRSV kernel + the GPU-resident PCG (own SpMV / vector kernels); links cudart only unless APXCHOL_CUDA_WITH_CUSPARSE" OFF)
option(APXCHOL_CUDA_WITH_CUSPARSE "CUDA build only: also compile + link the cuSPARSE SpSV backend of the GPU SpTRSV (env APXCHOL_GPU_SPTRSV=cusparse) and the benchmark's legacy cuBLAS/cuSPARSE `apxchol_gpu` solver. OFF = apxchol_v1 links cudart alone (see the root CMakeLists.txt)" OFF)
option(APXCHOL_POOL_FP32 "Store residual-pool edge weights in fp32 (halves edges_; reads promote to fp64; multigraph terminal writes keep it safe). DEFAULT ON (measured -23% setup IPM / -6.5% social, neutral grids, convergence unchanged); -DAPXCHOL_POOL_FP32=OFF for fp64 baseline" ON)
if (APXCHOL_POOL_FP32)
    target_compile_definitions(apxchol_v1 PUBLIC APXCHOL_POOL_FP32)
endif()
if (APXCHOL_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(apxchol_v1 PUBLIC APXCHOL_USE_CUDA)
    # Device precision-cast kernels for the mixed-precision GPU SpTRSV
    # (nvcc-compiled; the .cpp TUs are host-compiled and can't host __global__).
    target_sources(apxchol_v1 PRIVATE ${PARENT_SRC}/cuda_cast.cu)
    # Sync-free dataflow SpTRSV kernel (APXCHOL_GPU_SPTRSV=dataflow): one
    # persistent launch per sweep, 8 B/row of epoch-tagged state.
    target_sources(apxchol_v1 PRIVATE ${PARENT_SRC}/cuda_dataflow.cu)
    # Runtime-guarded GPU residual-topology + exact Luby setup front-end.
    target_sources(apxchol_v1 PRIVATE ${PARENT_SRC}/cuda_luby_frontend.cu)
    # The GPU-resident PCG's own CSR SpMV + fused vector kernels (pcg_cuda.h).
    target_sources(apxchol_v1 PRIVATE ${PARENT_SRC}/cuda_pcg_kernels.cu)
    # cudart only; cuSPARSE (the opt-in SpSV backend) under APXCHOL_CUDA_WITH_CUSPARSE.
    target_link_libraries(apxchol_v1 PUBLIC CUDA::cudart)
    if (APXCHOL_CUDA_WITH_CUSPARSE)
        target_compile_definitions(apxchol_v1 PUBLIC APXCHOL_CUDA_WITH_CUSPARSE)
        target_link_libraries(apxchol_v1 PUBLIC CUDA::cusparse)
    endif()
    # Pin the CUDA dialect to 20: the target's host features are cxx_std_23, but nvcc
    # (13.x) + recent CMake reject a "CUDA23" dialect, so the cuda_cast.cu sources must
    # compile as C++20 (same reason amgcl_cuda.cu/benchmark pin CUDA_STANDARD 20 below).
    # Host .cpp TUs still build at C++23; only the .cu device sources drop to 20.
    set_target_properties(apxchol_v1 PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
endif()
# Boost headers are needed only by AMGCL (HAVE_AMGCL gate below); the apxchol
# core itself is Boost-free.
find_package(Boost CONFIG QUIET)

# ══════════════════════════════════════════════════════
# Benchmark executable
# ══════════════════════════════════════════════════════
add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark PRIVATE laplacian_core rchol_lib apxchol_v1)
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
if (APXCHOL_USE_CUDA)
    # BUILD_META's cuda_host_compiler: the host compiler nvcc was pointed at with
    # -ccbin. Empty means CMake passed no -ccbin and nvcc used its own default,
    # which the binary reports as `unset(nvcc-default)` rather than as a name we
    # would be guessing.
    target_compile_definitions(benchmark PRIVATE
        APXCHOL_BUILD_CUDA_HOST_COMPILER="${CMAKE_CUDA_HOST_COMPILER}")
endif()

# AMGCL CUDA backend adapter (uses thrust + cuSPARSE, compiled with nvcc).
# CUDA standard pinned to 20 (nvcc may not fully support C++23 with our toolchain).
if (APXCHOL_USE_CUDA AND Boost_FOUND)
    target_sources(benchmark PRIVATE src/amgcl_cuda.cu)
    # -Xcompiler=-fopenmp is REQUIRED: AMGCL builds its AMG hierarchy host-side
    # with `#pragma omp parallel for` (smoothed_aggregation coarsening, SpMV,
    # transpose). Without -fopenmp in THIS nvcc TU those pragmas are no-ops and
    # the setup runs single-threaded — which made amgcl_cuda's setup 2-10x slower
    # than BoomerAMG-GPU (e.g. G3_circuit 2.1s vs 0.5s). libgomp is already linked
    # transitively via apxchol_v1.
    set_source_files_properties(src/amgcl_cuda.cu PROPERTIES
        LANGUAGE CUDA
        COMPILE_OPTIONS "--extended-lambda;--expt-relaxed-constexpr;-Xcompiler=-fopenmp"
        CUDA_STANDARD 20)
    set_target_properties(benchmark PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED OFF)
    # AMGCL's CUDA backend is cuSPARSE-based (thrust + cusparseSpMV): the one
    # closed-source dependency the benchmark DRIVER keeps whenever AMGCL-CUDA
    # is in (the apxchol library itself links cudart only).
    target_link_libraries(benchmark PRIVATE CUDA::cudart CUDA::cusparse)
endif()
# The benchmark's legacy `apxchol_gpu` solver (run_apxchol_gpu_pcg: a cuBLAS
# axpy/dot/nrm2 + cuSPARSE SpMV PCG around the host-pointer preconditioner
# API; not used by any sweep) is the only cuBLAS consumer left -- compiled
# and linked only with the cuSPARSE opt-in.
if (APXCHOL_USE_CUDA AND APXCHOL_CUDA_WITH_CUSPARSE)
    target_link_libraries(benchmark PRIVATE CUDA::cudart CUDA::cusparse CUDA::cublas)
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

add_test(NAME contract_requires_class
         COMMAND benchmark --mtx ${CMAKE_CURRENT_SOURCE_DIR}/../data/matrices/ecology1.mtx
                           --kind operator --solver apxchol_v1 --maxiter 1)
set_tests_properties(contract_requires_class PROPERTIES WILL_FAIL TRUE)

add_test(NAME contract_rejects_wrong_class
         COMMAND benchmark --mtx ${CMAKE_CURRENT_SOURCE_DIR}/../data/matrices/ecology1.mtx
                           --kind operator --class sddm --solver apxchol_v1 --maxiter 1)
set_tests_properties(contract_rejects_wrong_class PROPERTIES WILL_FAIL TRUE)

add_test(NAME contract_rejects_class_on_graph
         COMMAND benchmark --graph grid --n 32 --class sddm --solver apxchol_v1 --maxiter 1)
set_tests_properties(contract_rejects_class_on_graph PROPERTIES WILL_FAIL TRUE)
