# Find Homebrew libomp for Apple Clang without overriding explicit runtime inputs.
# Append it after other search prefixes; keep normal FindOpenMP behavior elsewhere.
set(_apxchol_saved_system_prefix_path "${CMAKE_SYSTEM_PREFIX_PATH}")
if (APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"
    AND NOT DEFINED OpenMP_ROOT AND NOT DEFINED ENV{OpenMP_ROOT}
    AND NOT DEFINED OPENMP_ROOT AND NOT DEFINED ENV{OPENMP_ROOT}
    AND NOT DEFINED OpenMP_CXX_FLAGS)
    foreach (_apxchol_keg IN ITEMS /opt/homebrew/opt/libomp /usr/local/opt/libomp)
        if (EXISTS "${_apxchol_keg}/include/omp.h")
            list(APPEND CMAKE_SYSTEM_PREFIX_PATH "${_apxchol_keg}")
        endif()
    endforeach()
    unset(_apxchol_keg)
endif()
find_package(OpenMP)
set(CMAKE_SYSTEM_PREFIX_PATH "${_apxchol_saved_system_prefix_path}")
unset(_apxchol_saved_system_prefix_path)

if (NOT OpenMP_CXX_FOUND AND NOT CMAKE_DISABLE_FIND_PACKAGE_OpenMP)
    message(WARNING
        "apxchol: no OpenMP runtime found; building a SERIAL library. "
        "Apple Clang: `brew install libomp` or pass -DOpenMP_ROOT=<prefix>. "
        "Pass -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON to request a serial "
        "build explicitly.")
endif()
