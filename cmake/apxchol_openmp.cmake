# Add Homebrew's keg-only libomp after other search prefixes on Apple Clang.
# Explicit FindOpenMP inputs take precedence; compiler selection is unchanged.
set(_apxchol_saved_system_prefix_path "${CMAKE_SYSTEM_PREFIX_PATH}")
if (APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"
    AND NOT DEFINED OpenMP_ROOT AND NOT DEFINED ENV{OpenMP_ROOT}
    AND NOT DEFINED OPENMP_ROOT AND NOT DEFINED ENV{OPENMP_ROOT}
    AND NOT DEFINED OpenMP_CXX_FLAGS)
    find_program(_apxchol_brew brew)
    if (_apxchol_brew)
        execute_process(COMMAND "${_apxchol_brew}" --prefix libomp
            RESULT_VARIABLE _apxchol_brew_result
            OUTPUT_VARIABLE _apxchol_libomp OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if (_apxchol_brew_result EQUAL 0 AND EXISTS "${_apxchol_libomp}/include/omp.h")
            list(APPEND CMAKE_SYSTEM_PREFIX_PATH "${_apxchol_libomp}")
        endif()
    endif()
    unset(_apxchol_brew CACHE)
    unset(_apxchol_brew)
    unset(_apxchol_brew_result)
    unset(_apxchol_libomp)
endif()
find_package(OpenMP REQUIRED COMPONENTS CXX)
set(CMAKE_SYSTEM_PREFIX_PATH "${_apxchol_saved_system_prefix_path}")
unset(_apxchol_saved_system_prefix_path)
