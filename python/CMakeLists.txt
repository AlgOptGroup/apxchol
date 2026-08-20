cmake_minimum_required(VERSION 3.16)
project(apxchol_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Repo root (this CMakeLists lives in python/).
set(APX_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

# ── Eigen (system or fetched, mirrors root) ──
find_package(Eigen3 3.3 CONFIG QUIET)
if (NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(eigen)
endif()

find_package(OpenMP)
find_package(pybind11 CONFIG REQUIRED)
# The core TUs depend only on Eigen + OpenMP (no fast_matrix_market, no
# generated version.h, no CLI11/spdlog), so we compile them straight into the
# extension. The benchmark-critical apxchol::solve in solve.cpp is reused as-is;
# operator_class.cpp supplies the operator-contract assertion and the M-matrix
# lumping, shared verbatim with the CLI and the Octave binding. (src/mtx_input.cpp
# is NOT compiled here: it is the CLI's own file-interpretation layer.)
pybind11_add_module(_apxchol
    src/apxchol_ext.cpp
    ${APX_ROOT}/src/factorization.cpp
    ${APX_ROOT}/src/operator_class.cpp
    ${APX_ROOT}/src/solve.cpp
)
target_include_directories(_apxchol PRIVATE ${APX_ROOT}/include ${APX_ROOT}/src)
target_link_libraries(_apxchol PRIVATE Eigen3::Eigen)
if (OpenMP_CXX_FOUND)
    target_link_libraries(_apxchol PRIVATE OpenMP::OpenMP_CXX)
endif()

# Match the root library default (ON there). The SpTRSV's fp32 factor storage
# needs no macro any more, and its fp16 variant is a runtime env knob
# (APXCHOL_SPTRSV_FP16) that the portable wheel build below cannot offer:
# without -march=native there is no __F16C__, so omp_sptrsv compiles the fp32
# storage only and the env falls back with a note. See include/apxchol/lowprec.h.
target_compile_definitions(_apxchol PRIVATE APXCHOL_POOL_FP32)
if (NOT MSVC)
    # Distributable wheels (cibuildwheel sets APXCHOL_PORTABLE_BUILD=1) must not
    # use -march=native — it would SIGILL on older CPUs. Local source builds keep it.
    if (DEFINED ENV{APXCHOL_PORTABLE_BUILD})
        target_compile_options(_apxchol PRIVATE -O3)
    else()
        target_compile_options(_apxchol PRIVATE -O3 -march=native)
    endif()
endif()

install(TARGETS _apxchol LIBRARY DESTINATION apxchol)
