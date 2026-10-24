cmake_minimum_required(VERSION 3.20)
include(${CMAKE_CURRENT_LIST_DIR}/../apxchol_version.cmake)
project(apxchol_core VERSION ${APXCHOL_VERSION} LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
include(${CMAKE_CURRENT_LIST_DIR}/../apxchol_core.cmake)
apxchol_add_core()
