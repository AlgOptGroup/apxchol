include("${CMAKE_CURRENT_LIST_DIR}/apxchol_version.cmake")

if (NOT "${RELEASE_TAG}" STREQUAL "v${APXCHOL_VERSION}")
    message(FATAL_ERROR "Release tag '${RELEASE_TAG}' must be v${APXCHOL_VERSION}")
endif()
