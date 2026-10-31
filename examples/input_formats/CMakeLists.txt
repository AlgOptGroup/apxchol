# The C++ programs of docs/input-formats.md, as a separate project: Eigen,
# CHOLMOD and fast_matrix_market are needed here only, never by apxchol.
#   cmake -S examples/input_formats -B build-input-formats
cmake_minimum_required(VERSION 3.20)
project(apxchol_input_formats LANGUAGES CXX)
include(../../cmake/apxchol_core.cmake)
apxchol_add_core()

find_package(Eigen3 3.3 CONFIG REQUIRED)
find_path(CHOLMOD_INCLUDE_DIR cholmod.h PATH_SUFFIXES suitesparse REQUIRED)
find_library(CHOLMOD_LIBRARY cholmod REQUIRED)
include(FetchContent)
FetchContent_Declare(fast_matrix_market
    GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market.git
    GIT_TAG        v1.7.6
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(fast_matrix_market)

enable_testing()
add_compile_options(-Wall -Wextra -Wpedantic)
foreach(example raw_csc eigen cholmod matrix_market)
    add_executable(input_${example} ${example}.cpp)
    target_link_libraries(input_${example} PRIVATE apxchol::core)
    add_test(NAME input_${example} COMMAND input_${example})
    set_tests_properties(input_${example} PROPERTIES TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=2")
endforeach()
target_link_libraries(input_eigen PRIVATE Eigen3::Eigen)
target_include_directories(input_cholmod PRIVATE ${CHOLMOD_INCLUDE_DIR})
target_link_libraries(input_cholmod PRIVATE ${CHOLMOD_LIBRARY})
target_link_libraries(input_matrix_market PRIVATE fast_matrix_market::fast_matrix_market)
