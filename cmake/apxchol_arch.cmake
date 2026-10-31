# Shared architecture choice for the library, Python bindings and benchmarks.
# Include after apxchol_eigen.cmake: the alignment probe compiles against Eigen.
if (MSVC)
    message(FATAL_ERROR "apxchol does not support MSVC or clang-cl; use GCC or Clang on Linux or macOS")
endif()

set(_apxchol_user_flags "${CMAKE_CXX_FLAGS}")
if (CMAKE_BUILD_TYPE)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _apxchol_config)
    string(APPEND _apxchol_user_flags " ${CMAKE_CXX_FLAGS_${_apxchol_config}}")
endif()

set(APXCHOL_ARCH_FLAG "")
if (_apxchol_user_flags MATCHES "(^| )-m(arch|cpu)=")
    message(STATUS "apxchol: using the -march/-mcpu flag from CMAKE_CXX_FLAGS")
else()
    if (APXCHOL_NATIVE_ARCH AND
        (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
        include(${CMAKE_CURRENT_LIST_DIR}/apxchol_native_arch.cmake)
        set(APXCHOL_ARCH_FLAG "${APXCHOL_NATIVE_ARCH_FLAG}")
    endif()
    if (NOT APXCHOL_ARCH_FLAG AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        set(APXCHOL_ARCH_FLAG "-march=x86-64-v2")
    endif()
endif()

# The arch flag stays private, but Eigen derives its heap alignment from the
# instruction set: AVX selects 32 bytes and AVX-512 64, where plain malloc
# gives 16. A vector allocated in the core and freed by a consumer compiled
# without the flag would then use mismatched allocators. Export the core's
# alignment when it exceeds 16, so consumers keep their own flags.
get_target_property(_apxchol_eigen_includes Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
string(REGEX REPLACE "\\$<BUILD_INTERFACE:([^>]*)>" "\\1" _apxchol_eigen_includes "${_apxchol_eigen_includes}")
string(GENEX_STRIP "${_apxchol_eigen_includes}" _apxchol_eigen_includes)
set(APXCHOL_EIGEN_ALIGN_BYTES "")
foreach (_apxchol_align 64 32)
    string(MAKE_C_IDENTIFIER "APXCHOL_EIGEN_ALIGN_${_apxchol_align}_${_apxchol_user_flags}_${APXCHOL_ARCH_FLAG}"
           _apxchol_probe)
    set(CMAKE_REQUIRED_FLAGS "${_apxchol_user_flags} ${APXCHOL_ARCH_FLAG}")
    set(CMAKE_REQUIRED_INCLUDES "${_apxchol_eigen_includes}")
    set(CMAKE_REQUIRED_QUIET ON)
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles("#include <Eigen/Core>
static_assert(EIGEN_MAX_ALIGN_BYTES == ${_apxchol_align}, \"\");
int main() { return 0; }" ${_apxchol_probe})
    unset(CMAKE_REQUIRED_FLAGS)
    unset(CMAKE_REQUIRED_INCLUDES)
    unset(CMAKE_REQUIRED_QUIET)
    if (${_apxchol_probe})
        set(APXCHOL_EIGEN_ALIGN_BYTES ${_apxchol_align})
        message(STATUS "apxchol: exporting EIGEN_MAX_ALIGN_BYTES=${_apxchol_align} to match the core")
        break()
    endif()
endforeach()
unset(_apxchol_align)
unset(_apxchol_probe)
unset(_apxchol_eigen_includes)
unset(_apxchol_user_flags)
unset(_apxchol_config)
