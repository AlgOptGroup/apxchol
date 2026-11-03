# Shared architecture choice for the library, Python bindings and benchmarks.
# The flag is applied PRIVATE: no public header depends on the instruction set.
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

unset(_apxchol_user_flags)
unset(_apxchol_config)
