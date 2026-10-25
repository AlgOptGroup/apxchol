cmake_minimum_required(VERSION 3.20)
include(${CMAKE_CURRENT_LIST_DIR}/../apxchol_version.cmake)
project(apxchol_core VERSION ${APXCHOL_VERSION} LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# GPU languages must be enabled in the parent that links this target too.
if (NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    if ((APXCHOL_DEVICE STREQUAL "CUDA" AND NOT CMAKE_CUDA_COMPILER_LOADED) OR
        (APXCHOL_DEVICE STREQUAL "METAL" AND NOT CMAKE_OBJCXX_COMPILER_LOADED))
        message(FATAL_ERROR "Include cmake/apxchol_core.cmake and call apxchol_add_core() in the consuming project's scope, so its GPU language is enabled there.")
    endif()
endif()
option(APXCHOL_NATIVE_ARCH "Compile tuned for the build machine (Release/RelWithDebInfo)" OFF)
include(${CMAKE_CURRENT_LIST_DIR}/../apxchol_core.cmake)
apxchol_add_core()
