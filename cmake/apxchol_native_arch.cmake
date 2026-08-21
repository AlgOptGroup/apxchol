# Pick the compiler flag that means "tune for the machine we are building on".
#
# `-march=native` is the x86 spelling. On aarch64 the ISA spelling is
# `-mcpu=native`: GCC/aarch64 rejects `-march=native` outright, and clang/aarch64
# ACCEPTS it while silently falling back to `-target-cpu generic` — so an ARM
# clang build that passes `-march=native` is untuned, and any number taken from
# it is measuring the wrong binary. Ask for `-mcpu=native` first on ARM.
#
# The flags are probed, not assumed: a toolchain that accepts neither (a cross
# build, an exotic target) gets no tuning flag rather than a hard failure.
#
# Sets APXCHOL_NATIVE_ARCH_FLAG to the flag to use, or to "" if there is none.
# Included by both the root project and the separate benchmarks/ project.

include_guard(GLOBAL)
include(CheckCXXCompilerFlag)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64|armv[78]")
    set(_apxchol_native_candidates "-mcpu=native" "-march=native")
else()
    set(_apxchol_native_candidates "-march=native" "-mcpu=native")
endif()

set(APXCHOL_NATIVE_ARCH_FLAG "")
foreach (_apxchol_flag IN LISTS _apxchol_native_candidates)
    string(MAKE_C_IDENTIFIER "APXCHOL_HAVE_FLAG_${_apxchol_flag}" _apxchol_probe)
    check_cxx_compiler_flag("${_apxchol_flag}" ${_apxchol_probe})
    if (${_apxchol_probe})
        set(APXCHOL_NATIVE_ARCH_FLAG "${_apxchol_flag}")
        break()
    endif()
endforeach()

if (APXCHOL_NATIVE_ARCH_FLAG)
    message(STATUS "Native tuning flag for ${CMAKE_SYSTEM_PROCESSOR}: ${APXCHOL_NATIVE_ARCH_FLAG}")
else()
    message(STATUS "No native tuning flag accepted by ${CMAKE_CXX_COMPILER_ID} on ${CMAKE_SYSTEM_PROCESSOR} — building untuned")
endif()

unset(_apxchol_native_candidates)
unset(_apxchol_flag)
unset(_apxchol_probe)
