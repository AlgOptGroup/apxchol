cmake_minimum_required(VERSION 3.20)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/apxchol_macos_compiler.cmake)
project(apxchol_consumer_regression LANGUAGES CXX)

# Deliberately older than the public headers: the linked target must upgrade it.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()
add_subdirectory(../.. apxchol)

# Compile this probe as part of the core to observe its actual Eigen alignment.
target_sources(apxchol_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/core_alignment.cpp")
add_executable(apxchol_consumer consumer.cpp)
target_link_libraries(apxchol_consumer PRIVATE apxchol_core)
add_test(NAME cmake_parent_consumer COMMAND apxchol_consumer)
set_tests_properties(cmake_parent_consumer PROPERTIES
    TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=2")
