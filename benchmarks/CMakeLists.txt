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
# MKL backs the RCHOL PCG solve and the ParAC CPU driver; both are skipped when it
# is absent. Point at an install with -DMKL_ROOT=/path/to/oneapi/mkl/<version>, via
# the MKLROOT environment variable (oneAPI's setvars.sh exports it), or from a
# gitignored benchmarks/paths_local.cmake holding `set(MKL_ROOT ... CACHE PATH ...)`.
if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/paths_local.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/paths_local.cmake")
endif()
set(MKL_ROOT "$ENV{MKLROOT}" CACHE PATH "MKL root directory (empty = MKL-dependent solvers skipped)")
find_path(MKL_INCLUDE mkl.h HINTS ${MKL_ROOT}/include)
find_library(MKL_LP64 mkl_intel_lp64 HINTS ${MKL_ROOT}/lib)
find_library(MKL_SEQ mkl_sequential HINTS ${MKL_ROOT}/lib)  # sequential layer: gnu_thread thrashed (per-iter MKL sparse oversubscription, ~8x slower); rchol PCG is bandwidth-bound so threaded MKL gives no win anyway. Parallel-rchol = rchol_par (METIS factor) instead.
find_library(MKL_CORE mkl_core HINTS ${MKL_ROOT}/lib)
if (MKL_INCLUDE AND MKL_LP64 AND MKL_SEQ AND MKL_CORE)
    set(HAVE_MKL TRUE)
    set(MKL_LIBRARIES ${MKL_LP64} ${MKL_SEQ} ${MKL_CORE} pthread m dl)
    message(STATUS "Found MKL: ${MKL_ROOT}")
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
    check_language(CUDA)
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
    ${PARENT_SRC}/solve.cpp
)
target_compile_features(apxchol_v1 PUBLIC cxx_std_23)
target_link_libraries(apxchol_v1 PUBLIC Eigen3::Eigen)
target_include_directories(apxchol_v1 PUBLIC ${PARENT_INCLUDE})
if (OpenMP_CXX_FOUND)
    target_link_libraries(apxchol_v1 PUBLIC OpenMP::OpenMP_CXX)
endif()
# 64-bit indices for graphs whose apxchol FACTOR exceeds 2^31 nnz (e.g. com-Orkut:
# 237M-nnz scale-free graph -> ~2-3e9-nnz factor -> int32 offset overflow -> SIGSEGV).
# PUBLIC so benchmark.cpp's apxchol::index_t matches the linked library (else ABI UB).
option(APXCHOL_64BIT_INDICES "64-bit apxchol indices for >2^31 nnz factors" OFF)
if (APXCHOL_64BIT_INDICES)
    target_compile_definitions(apxchol_v1 PUBLIC APXCHOL_64BIT_INDICES)
endif()
option(APXCHOL_SPTRSV_FP32 "fp32 SpTRSV factor values (diag + PCG stay fp64). DEFAULT ON (validated 1e-8 incl. marginal IPM); -DAPXCHOL_SPTRSV_FP32=OFF for fp64 baseline" ON)
if (APXCHOL_SPTRSV_FP32)
    target_compile_definitions(apxchol_v1 PUBLIC APXCHOL_SPTRSV_FP32)
endif()
option(APXCHOL_USE_CUDA "Use cuSPARSE (GPU) for sparse triangular solve" OFF)
set(APXCHOL_SPTRSV_LOWPREC "OFF" CACHE STRING "Low-precision SpTRSV factor-value STORAGE for the off-diagonals, diagonal fp32 (reads widen to fp64; compute unchanged; the per-column scale is folded into the vectors and the fp16 fat-level kernels are SIMD, env APXCHOL_FP16_GATHER=simd|scalar; env APXCHOL_FP16_DIAG=1 = fp16 diagonal): OFF | FP16_SCALED -- see the root CMakeLists.txt / include/apxchol/lowprec.h. CPU/omp only -- treated as OFF (fp32, STATUS message) when APXCHOL_USE_CUDA=ON; overrides APXCHOL_SPTRSV_FP32. DEFAULT OFF")
set_property(CACHE APXCHOL_SPTRSV_LOWPREC PROPERTY STRINGS OFF FP16_SCALED)
string(TOUPPER "${APXCHOL_SPTRSV_LOWPREC}" APXCHOL_SPTRSV_LOWPREC)
if (NOT APXCHOL_SPTRSV_LOWPREC MATCHES "^(OFF|FP16_SCALED)$")
    message(FATAL_ERROR "APXCHOL_SPTRSV_LOWPREC must be OFF or FP16_SCALED (got '${APXCHOL_SPTRSV_LOWPREC}'; the BF16 / BF16_SCALED / FP24 variants were removed 2026-08-18).")
endif()
# CUDA builds have no COMPILE-TIME low-precision SpTRSV path (the GPU's fp16
# storage is the runtime opt-in APXCHOL_GPU_SPTRSV_FP16=1 on our dataflow /
# level-set kernels):
# with APXCHOL_USE_CUDA=ON the
# variant is TREATED AS OFF for this build (fp32 CPU storage, no macro, STATUS
# line) instead of failing the configure -- same rule as the root
# CMakeLists.txt; the cache value is left as requested.
if (APXCHOL_USE_CUDA AND NOT APXCHOL_SPTRSV_LOWPREC STREQUAL "OFF")
    message(STATUS "APXCHOL_SPTRSV_LOWPREC=${APXCHOL_SPTRSV_LOWPREC} requested together with APXCHOL_USE_CUDA=ON: the CUDA backend has no low-precision SpTRSV path -- CPU SpTRSV storage forced to fp32 (APXCHOL_SPTRSV_LOWPREC treated as OFF) for this build")
    set(APXCHOL_SPTRSV_LOWPREC "OFF")
endif()
if (NOT APXCHOL_SPTRSV_LOWPREC STREQUAL "OFF")
    target_compile_definitions(apxchol_v1 PUBLIC APXCHOL_SPTRSV_LOWPREC_${APXCHOL_SPTRSV_LOWPREC})
endif()
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
    # Custom level-set SpTRSV kernels (APXCHOL_GPU_SPTRSV=levelset): O(n) schedule
    # instead of cuSPARSE's O(nnz) analysis buffer.
    target_sources(apxchol_v1 PRIVATE ${PARENT_SRC}/cuda_levelset.cu)
    # Sync-free dataflow SpTRSV kernel (APXCHOL_GPU_SPTRSV=dataflow): one
    # persistent launch per sweep, 8 B/row of epoch-tagged state.
    target_sources(apxchol_v1 PRIVATE ${PARENT_SRC}/cuda_dataflow.cu)
    target_link_libraries(apxchol_v1 PUBLIC CUDA::cudart CUDA::cusparse CUDA::cublas)
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

