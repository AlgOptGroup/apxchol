include_guard(GLOBAL)
include(GNUInstallDirs)
include(${CMAKE_CURRENT_LIST_DIR}/apxchol_version.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/apxchol_device.cmake)

option(APXCHOL_NATIVE_ARCH "Compile tuned for the build machine (Release/RelWithDebInfo)" ON)
option(APXCHOL_64BIT_EDGE_INDICES "64-bit edge/offset indices, for factors or pools exceeding uint32 capacity (2^32-1 entries)" OFF)
option(APXCHOL_64BIT_NODE_INDICES "64-bit vertex indices (implies 64-bit edge indices)" OFF)
option(APXCHOL_POOL_FP32 "Store residual-pool weights in FP32; reads promote to FP64. OFF selects the FP64 reference." ON)

include(${CMAKE_CURRENT_LIST_DIR}/apxchol_arch.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/apxchol_eigen.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/apxchol_openmp.cmake)

if (APXCHOL_DEVICE STREQUAL "METAL")
    enable_language(OBJCXX)
elseif (APXCHOL_DEVICE STREQUAL "CUDA")
    if (CMAKE_VERSION VERSION_LESS 4.4)
        message(FATAL_ERROR "CUDA C++23 builds require CMake 4.4 or newer")
    endif()
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit 13.3 REQUIRED)
endif()

function(apxchol_configure_core target)
    get_filename_component(root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    target_sources(${target} PRIVATE
        ${root}/src/factorization.cpp
        ${root}/src/operator_class.cpp
        ${root}/src/solve.cpp
    )
    target_link_libraries(${target} PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)
    target_compile_features(${target} PUBLIC cxx_std_23)
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
    target_compile_options(${target} PRIVATE
        -Wall -Wextra "$<$<COMPILE_LANGUAGE:C,CXX,OBJCXX>:-Wpedantic>")
    if (APXCHOL_ARCH_FLAG)
        target_compile_options(${target} PUBLIC "${APXCHOL_ARCH_FLAG}")
    endif()

    set(APXCHOL_GIT_SHA "unknown")
    execute_process(COMMAND git -C "${root}" rev-parse --show-toplevel
        OUTPUT_VARIABLE git_root OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if (git_root)
        file(REAL_PATH "${root}" native_root)
        file(REAL_PATH "${git_root}" git_root)
        if (git_root STREQUAL native_root)
            execute_process(COMMAND git -C "${root}" rev-parse --short HEAD
                OUTPUT_VARIABLE APXCHOL_GIT_SHA OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            if (NOT APXCHOL_GIT_SHA)
                set(APXCHOL_GIT_SHA "unknown")
            endif()
        endif()
    endif()
    configure_file("${root}/include/apxchol/version.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/include/apxchol/version.h" @ONLY)
    target_include_directories(${target} PUBLIC
        "$<BUILD_INTERFACE:${root}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    )

    if (APXCHOL_DEVICE STREQUAL "CUDA")
        # nvcc's serial host pass also parses OpenMP pragmas in shared headers.
        target_compile_options(${target} PRIVATE
            "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=-Wno-unknown-pragmas>")
        target_sources(${target} PRIVATE ${root}/src/cuda_dataflow.cu
            ${root}/src/cuda_block_frontend.cu ${root}/src/cuda_round_shadow.cu ${root}/src/cuda_pcg_kernels.cu)
        target_compile_definitions(${target} PUBLIC APXCHOL_USE_CUDA)
        # Static cudart keeps installed libraries independent of the Toolkit.
        target_link_libraries(${target} PUBLIC CUDA::cudart_static)
        set_target_properties(${target} PROPERTIES CUDA_STANDARD 23 CUDA_STANDARD_REQUIRED ON
            CUDA_RUNTIME_LIBRARY Static)
    endif()

    if (APXCHOL_DEVICE STREQUAL "METAL")
        target_sources(${target} PRIVATE ${root}/src/metal_solver.mm ${root}/src/metal_setup.mm)
        target_compile_definitions(${target} PUBLIC APXCHOL_USE_METAL)
        set_source_files_properties(${root}/src/metal_solver.mm ${root}/src/metal_setup.mm PROPERTIES COMPILE_OPTIONS -fobjc-arc)
        target_compile_options(${target} PRIVATE
            "$<$<COMPILE_LANGUAGE:OBJCXX>:SHELL:${OpenMP_CXX_FLAGS}>")
        set_target_properties(${target} PROPERTIES OBJCXX_STANDARD 23 OBJCXX_STANDARD_REQUIRED ON)
        # PUBLIC: the static library hands the framework dependencies to its
        # consumers. CoreGraphics lets MTLCreateSystemDefaultDevice find the GPU
        # from command-line tools.
        target_link_libraries(${target} PUBLIC "-framework Metal" "-framework Foundation"
                              "-framework CoreGraphics")
    endif()

    if (APXCHOL_64BIT_EDGE_INDICES)
        target_compile_definitions(${target} PUBLIC APXCHOL_64BIT_EDGE_INDICES)
    endif()
    if (APXCHOL_64BIT_NODE_INDICES)
        target_compile_definitions(${target} PUBLIC APXCHOL_64BIT_NODE_INDICES)
    endif()
    if (APXCHOL_POOL_FP32)
        target_compile_definitions(${target} PUBLIC APXCHOL_POOL_FP32)
    endif()
endfunction()

# Include this module in the consuming project's scope before calling this
# helper, so GPU languages are enabled for both the core and its consumers.
function(apxchol_add_core)
    if (TARGET apxchol::core)
        return()
    endif()
    add_library(apxchol_core)
    add_library(apxchol::core ALIAS apxchol_core)
    set_target_properties(apxchol_core PROPERTIES EXPORT_NAME core POSITION_INDEPENDENT_CODE ON)
    apxchol_configure_core(apxchol_core)
    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/apxchol_install.cmake)
    apxchol_install_core()
endfunction()
