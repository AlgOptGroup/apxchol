cmake_minimum_required(VERSION 3.20)
project(apxchol_mex LANGUAGES C CXX)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# MEX wrappers currently expose CPU execution with 32-bit operator indices.
if ((DEFINED APXCHOL_DEVICE AND NOT APXCHOL_DEVICE STREQUAL "CPU") OR
    (DEFINED APXCHOL_INDEX_WIDTH AND NOT APXCHOL_INDEX_WIDTH STREQUAL "32"))
    message(FATAL_ERROR "The MATLAB/Octave binding requires APXCHOL_DEVICE=CPU and APXCHOL_INDEX_WIDTH=32")
endif()
set(BUILD_SHARED_LIBS OFF)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/apxchol_core.cmake)
apxchol_add_core()

if (Matlab_ROOT_DIR)
    find_package(Matlab REQUIRED)
    matlab_add_mex(NAME apxchol_mex SRC apxchol_mex.cpp R2018a LINK_TO apxchol::core)
else()
    find_program(MKOCTFILE mkoctfile REQUIRED)
    execute_process(COMMAND "${MKOCTFILE}" -p OCTINCLUDEDIR
        OUTPUT_VARIABLE octave_include OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    execute_process(COMMAND "${MKOCTFILE}" -p LDFLAGS
        OUTPUT_VARIABLE octave_ldflags OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    get_filename_component(octave_include_parent "${octave_include}" DIRECTORY)
    add_library(apxchol_mex MODULE apxchol_mex.cpp)
    target_include_directories(apxchol_mex SYSTEM PRIVATE "${octave_include}" "${octave_include_parent}")
    target_compile_definitions(apxchol_mex PRIVATE MEX_DEBUG)
    target_link_libraries(apxchol_mex PRIVATE apxchol::core)
    set_target_properties(apxchol_mex PROPERTIES PREFIX "" SUFFIX ".mex")
    # mkoctfile supplies Octave's ABI-version stub. Its argument parser rejects
    # full shared-library paths, so pass CMake's link dependencies via LDFLAGS.
    set(CMAKE_CXX_CREATE_SHARED_MODULE
        "\"${CMAKE_COMMAND}\" -E env \"CC=${CMAKE_C_COMPILER}\" \"CXX=${CMAKE_CXX_COMPILER}\" \"CXXFLAGS=\" 'LDFLAGS=${octave_ldflags} <LINK_LIBRARIES>' \"${MKOCTFILE}\" --mex <LINK_FLAGS> <OBJECTS> -o <TARGET>")
endif()
set_target_properties(apxchol_mex PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
