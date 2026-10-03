# Shared architecture choice for the library, Python bindings and benchmarks.
set(APXCHOL_ARCH_FLAG "")
if (NOT MSVC)
    if (APXCHOL_NATIVE_ARCH AND
        (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"))
        include(${CMAKE_CURRENT_LIST_DIR}/apxchol_native_arch.cmake)
        set(APXCHOL_ARCH_FLAG "${APXCHOL_NATIVE_ARCH_FLAG}")
    endif()
    if (NOT APXCHOL_ARCH_FLAG AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        set(APXCHOL_ARCH_FLAG "-march=x86-64-v2")
    endif()
endif()
