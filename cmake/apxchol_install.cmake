include(CMakePackageConfigHelpers)

function(apxchol_install_core)
    get_filename_component(root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    apxchol_openmp_runtime(APXCHOL_OPENMP_RUNTIME)
    get_target_property(APXCHOL_CORE_TYPE apxchol_core TYPE)
    set(package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/apxchol")
    set(APXCHOL_BUNDLED_EIGEN OFF)
    if (TARGET apxchol_eigen)
        set(APXCHOL_BUNDLED_EIGEN ON)
        FetchContent_GetProperties(eigen SOURCE_DIR eigen_source)
        install(TARGETS apxchol_eigen EXPORT apxcholTargets)
        install(DIRECTORY "${eigen_source}/Eigen"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/apxchol/eigen")
        file(GLOB eigen_licenses "${eigen_source}/COPYING.*")
        install(FILES ${eigen_licenses}
            DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/apxchol/eigen")
    endif()
    configure_package_config_file("${root}/cmake/apxcholConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/apxcholConfig.cmake"
        INSTALL_DESTINATION "${package_dir}")
    write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/apxcholConfigVersion.cmake"
        VERSION "${APXCHOL_VERSION}" COMPATIBILITY SameMinorVersion)
    install(TARGETS apxchol_core EXPORT apxcholTargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
    install(DIRECTORY "${root}/include/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        FILES_MATCHING PATTERN "*.h" PATTERN "*.inc")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/apxchol/version.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/apxchol")
    install(FILES "${root}/LICENSE" DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/apxchol")
    install(EXPORT apxcholTargets NAMESPACE apxchol:: DESTINATION "${package_dir}")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/apxcholConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/apxcholConfigVersion.cmake"
        "${root}/cmake/apxchol_openmp.cmake"
        DESTINATION "${package_dir}")
endfunction()
