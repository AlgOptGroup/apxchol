cmake_minimum_required(VERSION 3.20)
project(apxchol_installed_consumer LANGUAGES CXX)

find_package(apxchol CONFIG REQUIRED)
if (COMMAND apxchol_openmp_runtime)
    message(FATAL_ERROR "The package config exposed an internal helper")
endif()
add_executable(apxchol_installed_consumer consumer.cpp)
target_link_libraries(apxchol_installed_consumer PRIVATE apxchol::core)

enable_testing()
add_test(NAME cmake_installed_consumer COMMAND apxchol_installed_consumer)
set_tests_properties(cmake_installed_consumer PROPERTIES
    TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=2")
