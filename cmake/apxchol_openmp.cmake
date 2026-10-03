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

# Root and Python targets need the same compiler-dependent runtime linkage.
function(_apxchol_link_openmp target visibility)
    target_link_libraries(${target} ${visibility} OpenMP::OpenMP_CXX)
    # Some compiler/target combinations lower long-double OpenMP reductions
    # to libatomic calls. Detect the runtime rather than making
    # consumers supply a platform-specific linker flag themselves.
    include(CheckCXXSourceCompiles)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_LIBRARIES OpenMP::OpenMP_CXX)
    if (MSVC)
        set(CMAKE_REQUIRED_FLAGS "/Od")
    else()
        set(CMAKE_REQUIRED_FLAGS "-O0")
    endif()
    set(_apxchol_atomic_probe "
        int main(int argc, char**) {
            long double sum = 0.0L;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < argc; ++i) sum += i;
            return sum < 0.0L;
        }")
    check_cxx_source_compiles("${_apxchol_atomic_probe}" APXCHOL_OPENMP_REDUCTION_LINKS)
    if (NOT APXCHOL_OPENMP_REDUCTION_LINKS)
        list(APPEND CMAKE_REQUIRED_LIBRARIES atomic)
        check_cxx_source_compiles("${_apxchol_atomic_probe}" APXCHOL_OPENMP_REDUCTION_WITH_ATOMIC)
        if (NOT APXCHOL_OPENMP_REDUCTION_WITH_ATOMIC)
            message(FATAL_ERROR "The OpenMP long-double reduction needs an unavailable atomic runtime")
        endif()
        target_link_libraries(${target} ${visibility} atomic)
    endif()
    unset(_apxchol_atomic_probe)
    cmake_pop_check_state()
endfunction()
