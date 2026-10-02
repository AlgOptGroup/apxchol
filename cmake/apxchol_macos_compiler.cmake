# Select Homebrew LLVM before project() on macOS. Explicit toolchains,
# compilers and parent-project compiler choices always take precedence.
if (NOT CMAKE_HOST_APPLE OR CMAKE_CXX_COMPILER_LOADED
    OR DEFINED CMAKE_TOOLCHAIN_FILE OR DEFINED ENV{CMAKE_TOOLCHAIN_FILE}
    OR DEFINED CMAKE_CXX_COMPILER OR DEFINED ENV{CXX}
    OR DEFINED CMAKE_C_COMPILER OR DEFINED ENV{CC})
    return()
endif()

find_program(_apxchol_brew brew HINTS /opt/homebrew/bin /usr/local/bin)
if (_apxchol_brew)
    execute_process(COMMAND "${_apxchol_brew}" --prefix llvm
        RESULT_VARIABLE _apxchol_brew_result
        OUTPUT_VARIABLE _apxchol_llvm OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if (_apxchol_brew_result EQUAL 0 AND EXISTS "${_apxchol_llvm}/bin/clang++")
        set(CMAKE_C_COMPILER "${_apxchol_llvm}/bin/clang")
        set(CMAKE_CXX_COMPILER "${_apxchol_llvm}/bin/clang++")
        list(APPEND CMAKE_PREFIX_PATH "${_apxchol_llvm}")
    endif()
endif()
unset(_apxchol_brew CACHE)
unset(_apxchol_brew)
unset(_apxchol_brew_result)
unset(_apxchol_llvm)
