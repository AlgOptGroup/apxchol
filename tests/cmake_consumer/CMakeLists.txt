cmake_minimum_required(VERSION 3.20)
project(apxchol_consumer_regression LANGUAGES CXX)

# Deliberately older than the public headers: the linked target must upgrade it.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
include(../../cmake/apxchol_core.cmake)
apxchol_add_core()
if (NOT CMAKE_CXX_STANDARD EQUAL 14 OR NOT CMAKE_CXX_EXTENSIONS
    OR NOT CMAKE_CUDA_RUNTIME_LIBRARY STREQUAL "Shared")
    message(FATAL_ERROR "apxchol changed the consuming project's compiler defaults")
endif()
enable_testing()
# This fixture drives the full CI suite; request its non-library targets explicitly.
option(APXCHOL_BUILD_CLI "Build the command-line solver" ON)
option(APXCHOL_BUILD_TESTS "Build the unit tests (fetches GoogleTest)" ON)
option(APXCHOL_BUILD_EXAMPLES "Build the examples/ programs" ON)
add_subdirectory(../.. apxchol)

# apxchol's architecture flag must not reach consumers.
get_target_property(_options apxchol::core INTERFACE_COMPILE_OPTIONS)
if (_options MATCHES "-m(arch|cpu)=")
    message(FATAL_ERROR "apxchol::core exports an architecture flag: ${_options}")
endif()
add_executable(apxchol_consumer consumer.cpp)
target_link_libraries(apxchol_consumer PRIVATE apxchol_core)
add_test(NAME cmake_parent_consumer COMMAND apxchol_consumer)
set_tests_properties(cmake_parent_consumer PROPERTIES
    TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=2")
