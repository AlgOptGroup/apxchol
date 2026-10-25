if (TARGET Eigen3::Eigen)
    return()
endif()

find_package(Eigen3 3.3 CONFIG QUIET)
if (NOT Eigen3_FOUND)
    set(Eigen3_VERSION 3.4.0)
    include(FetchContent)
    if (POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)
    endif()
    FetchContent_Declare(eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/${Eigen3_VERSION}/eigen-${Eigen3_VERSION}.tar.gz
        URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
        SOURCE_SUBDIR Eigen) # Headers only; do not configure Eigen's tests or install rules.
    FetchContent_MakeAvailable(eigen)
    add_library(apxchol_eigen INTERFACE)
    add_library(Eigen3::Eigen ALIAS apxchol_eigen)
    set_target_properties(apxchol_eigen PROPERTIES EXPORT_NAME eigen)
    target_include_directories(apxchol_eigen SYSTEM INTERFACE
        "$<BUILD_INTERFACE:${eigen_SOURCE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/apxchol/eigen>")
endif()
