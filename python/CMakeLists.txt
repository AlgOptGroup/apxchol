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

if (SKBUILD)
    install(FILES "${APX_ROOT}/LICENSE" DESTINATION "${SKBUILD_METADATA_DIR}/licenses")

    # Preserve the notice from the zlib headers actually selected by this build.
    find_package(ZLIB REQUIRED)
    file(READ "${ZLIB_INCLUDE_DIR}/zlib.h" zlib_header)
    string(FIND "${zlib_header}" "/* zlib.h" notice_start)
    string(FIND "${zlib_header}" "*/" notice_end)
    if (notice_start LESS 0 OR notice_end LESS notice_start)
        message(FATAL_ERROR "Cannot find the selected zlib header's license notice")
    endif()
    math(EXPR notice_length "${notice_end} + 2 - ${notice_start}")
    string(SUBSTRING "${zlib_header}" ${notice_start} ${notice_length} zlib_notice)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/zlib-LICENSE.txt" "${zlib_notice}\n")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/zlib-LICENSE.txt"
        DESTINATION "${SKBUILD_METADATA_DIR}/licenses/zlib" RENAME LICENSE)

    execute_process(COMMAND "${Python_EXECUTABLE}" -c [=[
from importlib.metadata import distribution
import sys
package = distribution("pybind11")
if package.version != sys.argv[1]:
    raise RuntimeError("Python and CMake selected different pybind11 versions")
license, = [package.locate_file(p) for p in package.files or []
            if p.name == "LICENSE" and any(s.endswith(".dist-info") for s in p.parts)]
print(license)
]=] "${pybind11_VERSION}"
        OUTPUT_VARIABLE pybind11_license OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    install(FILES "${pybind11_license}"
        DESTINATION "${SKBUILD_METADATA_DIR}/licenses/pybind11" RENAME LICENSE)

    configure_file(Eigen-NOTICE.txt.in Eigen-NOTICE.txt @ONLY)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/Eigen-NOTICE.txt"
        DESTINATION "${SKBUILD_METADATA_DIR}/licenses/eigen" RENAME NOTICE)
    if (TARGET apxchol_eigen)
        file(GLOB eigen_licenses "${eigen_SOURCE_DIR}/COPYING.*")
        install(FILES ${eigen_licenses} DESTINATION "${SKBUILD_METADATA_DIR}/licenses/eigen")
    endif()
endif()
