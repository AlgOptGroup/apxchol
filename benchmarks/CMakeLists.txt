cmake_minimum_required(VERSION 3.16)
project(laplacian_benchmarks LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT MSVC)
    add_compile_options(-Wall -Wextra -Wpedantic)
    if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        add_compile_options(-march=native)
    endif()
endif()

include(FetchContent)

# ── Path to benchmark sources ────────────────────────
set(BENCH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ── Shared sources from parent project ───────────────
set(PARENT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../src)
set(PARENT_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# ── Legacy v0 sources (graphs, simple_solver, mmio) ──
set(ARCHIVE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../archive/v0)
set(ARCHIVE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../archive/v0)

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
set(MKL_ROOT "/opt/intel/oneapi/mkl/2025.0" CACHE PATH "MKL root directory")
find_path(MKL_INCLUDE mkl.h HINTS ${MKL_ROOT}/include)
find_library(MKL_LP64 mkl_intel_lp64 HINTS ${MKL_ROOT}/lib)
find_library(MKL_SEQ mkl_sequential HINTS ${MKL_ROOT}/lib)
find_library(MKL_CORE mkl_core HINTS ${MKL_ROOT}/lib)
if (MKL_INCLUDE AND MKL_LP64 AND MKL_SEQ AND MKL_CORE)
    set(HAVE_MKL TRUE)
    set(MKL_LIBRARIES ${MKL_LP64} ${MKL_SEQ} ${MKL_CORE} pthread m dl)
    message(STATUS "Found MKL: ${MKL_ROOT}")
else()
    set(HAVE_MKL FALSE)
    message(STATUS "MKL not found — MKL-accelerated solvers disabled")
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

set(RCHOL_DIR ${rchol_SOURCE_DIR}/c++)
set(RCHOL_SOURCES
    ${RCHOL_DIR}/sparse.cpp
    ${RCHOL_DIR}/rchol/rchol.cpp
    ${RCHOL_DIR}/rchol_lap/rchol_lap.cpp
    ${RCHOL_DIR}/rchol_lap/spcol.cpp
    ${RCHOL_DIR}/util/util.cpp
)
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

    # CPU driver (experiment/ — includes MKL PCG solve)
    if (OpenMP_CXX_FOUND AND HAVE_MKL)
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
            OpenMP::OpenMP_CXX ${MKL_LIBRARIES}
        )
        target_compile_options(gpu_rchol_cpu_driver PRIVATE -O3 -w)
        message(STATUS "GPU RCHOL CPU driver enabled (experiment/driver_local.cpp)")
    else()
        message(STATUS "GPU RCHOL CPU driver needs OpenMP + MKL — skipped")
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

    include(CheckLanguage)
    check_language(CUDA)
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
    ${PARENT_SRC}/solve.cpp
)
target_compile_features(apxchol_v1 PUBLIC cxx_std_23)
target_link_libraries(apxchol_v1 PUBLIC Eigen3::Eigen)
target_include_directories(apxchol_v1 PUBLIC ${PARENT_INCLUDE})
if (OpenMP_CXX_FOUND)
    target_link_libraries(apxchol_v1 PUBLIC OpenMP::OpenMP_CXX)
endif()
option(APXCHOL_USE_CUDA "Use cuSPARSE (GPU) for sparse triangular solve" OFF)
if (APXCHOL_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(apxchol_v1 PUBLIC APXCHOL_USE_CUDA)
    target_link_libraries(apxchol_v1 PUBLIC CUDA::cudart CUDA::cusparse)
endif()
# Boost needed for small_vector incidence backend
find_package(Boost QUIET)
if (Boost_FOUND)
    target_include_directories(apxchol_v1 PUBLIC ${Boost_INCLUDE_DIRS})
endif()

# ══════════════════════════════════════════════════════
# Benchmark executable
# ══════════════════════════════════════════════════════
add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark PRIVATE laplacian_core rchol_lib apxchol_v1)
target_include_directories(benchmark PRIVATE ${AMGCL_INCLUDE})
target_compile_definitions(benchmark PRIVATE HAVE_RCHOL HAVE_APXCHOL_V1)
target_compile_features(benchmark PRIVATE cxx_std_23)

if (Boost_FOUND)
    target_compile_definitions(benchmark PRIVATE HAVE_AMGCL)
    message(STATUS "Boost found — AMGCL solver enabled")
else()
    message(STATUS "Boost not found — AMGCL solver disabled (needs Boost headers)")
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
