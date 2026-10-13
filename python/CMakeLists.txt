cmake_minimum_required(VERSION 3.20)
project(apxchol_ext LANGUAGES CXX)

get_filename_component(APX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
include(${APX_ROOT}/cmake/apxchol_core.cmake)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_apxchol src/apxchol_ext.cpp)
apxchol_configure_core(_apxchol)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND APXCHOL_ARCH_FLAG STREQUAL "-march=x86-64-v2")
    # Keep import runnable on older CPUs so it can report the core's requirement.
    set_source_files_properties(src/apxchol_ext.cpp PROPERTIES
        COMPILE_OPTIONS "-march=x86-64"
        COMPILE_DEFINITIONS APXCHOL_REQUIRE_X86_64_V2)
endif()

install(TARGETS _apxchol LIBRARY DESTINATION apxchol)
