# Shared architecture choice for the library, Python bindings and benchmarks.
# Include after apxchol_eigen.cmake: the alignment probe compiles against Eigen.
if (MSVC)
    message(FATAL_ERROR "apxchol does not support MSVC or clang-cl; use GCC or Clang on Linux or macOS")
endif()

# User flags for the configuration being built: CMAKE_BUILD_TYPE, or Release
# under a multi-config generator, whose per-configuration flags would
# otherwise be missed.
set(_apxchol_user_flags "${CMAKE_CXX_FLAGS}")
if (CMAKE_BUILD_TYPE)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _apxchol_config)
    string(APPEND _apxchol_user_flags " ${CMAKE_CXX_FLAGS_${_apxchol_config}}")
elseif (CMAKE_CONFIGURATION_TYPES)
    string(APPEND _apxchol_user_flags " ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

set(_APXCHOL_ARCH_FLAG "")
if (_apxchol_user_flags MATCHES "(^| )-m(arch|cpu)=")
    message(STATUS "apxchol: using the -march/-mcpu flag from CMAKE_CXX_FLAGS")
else()
    if (APXCHOL_NATIVE_ARCH AND
        (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
        include(${CMAKE_CURRENT_LIST_DIR}/apxchol_native_arch.cmake)
        set(_APXCHOL_ARCH_FLAG "${APXCHOL_NATIVE_ARCH_FLAG}")
    endif()
    if (NOT _APXCHOL_ARCH_FLAG AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        set(_APXCHOL_ARCH_FLAG "-march=x86-64-v2")
    endif()
endif()

# The arch flag stays private, but Eigen objects cross the library boundary.
# Eigen derives its alignment from the instruction set (16 bytes by default,
# 32 with AVX, 64 with AVX-512) and above 16 allocates through its own
# aligned allocator instead of malloc. Consumers get the core's
# EIGEN_MAX_ALIGN_BYTES, so neither side assumes more alignment than the
# other provides, and EIGEN_MALLOC_ALREADY_ALIGNED=0, so both sides allocate
# and free through the same allocator whichever instruction set each uses.
get_target_property(_apxchol_eigen_includes Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
string(REGEX REPLACE "\\$<BUILD_INTERFACE:([^>]*)>" "\\1" _apxchol_eigen_includes "${_apxchol_eigen_includes}")
string(GENEX_STRIP "${_apxchol_eigen_includes}" _apxchol_eigen_includes)
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
set(_APXCHOL_EIGEN_ALIGN_BYTES "")
foreach (_apxchol_align 64 32 16)
    string(MAKE_C_IDENTIFIER "APXCHOL_EIGEN_ALIGN_${_apxchol_align}_${_apxchol_user_flags}_${_APXCHOL_ARCH_FLAG}"
           _apxchol_probe)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_FLAGS "${_apxchol_user_flags} ${_APXCHOL_ARCH_FLAG}")
    set(CMAKE_REQUIRED_INCLUDES "${_apxchol_eigen_includes}")
    set(CMAKE_REQUIRED_QUIET ON)
    check_cxx_source_compiles("#include <Eigen/Core>
static_assert(EIGEN_MAX_ALIGN_BYTES == ${_apxchol_align}, \"\");
int main() { return 0; }" ${_apxchol_probe})
    cmake_pop_check_state()
    if (${_apxchol_probe})
        set(_APXCHOL_EIGEN_ALIGN_BYTES ${_apxchol_align})
        break()
    endif()
endforeach()
if (NOT _APXCHOL_EIGEN_ALIGN_BYTES)
    message(FATAL_ERROR "apxchol: could not determine Eigen's alignment for the core's flags")
endif()
message(STATUS "apxchol: exporting EIGEN_MAX_ALIGN_BYTES=${_APXCHOL_EIGEN_ALIGN_BYTES} to match the core")
unset(_apxchol_align)
unset(_apxchol_probe)
unset(_apxchol_eigen_includes)
unset(_apxchol_user_flags)
unset(_apxchol_config)
