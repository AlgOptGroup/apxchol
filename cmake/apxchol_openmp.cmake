# Locate OpenMP for the root project and the Python extension.
#
# Only the standard FindOpenMP inputs select the runtime; apxchol adds no
# OpenMP option of its own:
#   -DOpenMP_ROOT=<prefix>                    runtime prefix (Homebrew keg,
#                                             Julia artifact, a wheel's libomp)
#   -DOpenMP_CXX_FLAGS=... -DOpenMP_CXX_LIB_NAMES=...
#   -DOpenMP_<lib>_LIBRARY=<path>             an exact runtime, any compiler
#   -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON    an explicitly serial build
#
# Apple Clang ships no OpenMP runtime and Homebrew's libomp is keg-only, so
# FindOpenMP misses it unless told where to look. Append the keg as the
# lowest-priority prefix: OpenMP_ROOT and caller prefixes still win. Other
# compilers see the plain find_package(OpenMP) call unchanged.
#
# Sets the usual OpenMP_CXX_FOUND / OpenMP::OpenMP_CXX in the caller's scope.

set(_apxchol_saved_prefix_path "${CMAKE_PREFIX_PATH}")
if (APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    foreach (_apxchol_keg IN ITEMS /opt/homebrew/opt/libomp /usr/local/opt/libomp)
        if (EXISTS "${_apxchol_keg}/include/omp.h")
            list(APPEND CMAKE_PREFIX_PATH "${_apxchol_keg}")
        endif()
    endforeach()
    unset(_apxchol_keg)
endif()
find_package(OpenMP)
set(CMAKE_PREFIX_PATH "${_apxchol_saved_prefix_path}")
unset(_apxchol_saved_prefix_path)

if (NOT OpenMP_CXX_FOUND AND NOT CMAKE_DISABLE_FIND_PACKAGE_OpenMP)
    message(WARNING
        "apxchol: no OpenMP runtime found; building a SERIAL library. "
        "Apple Clang: `brew install libomp` or pass -DOpenMP_ROOT=<prefix>. "
        "Pass -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON to request a serial "
        "build explicitly.")
endif()
