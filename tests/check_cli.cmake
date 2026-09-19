execute_process(
    COMMAND "${CLI}" "${INPUT}" --random-rhs --tol 1e-8 ${CLI_ARGS}
    RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
if (NOT status STREQUAL "${EXPECT_STATUS}" OR
    NOT "${stdout}${stderr}" MATCHES "${EXPECT_OUTPUT}")
    message(FATAL_ERROR
        "Expected exit ${EXPECT_STATUS} and '${EXPECT_OUTPUT}', got ${status}:\n${stdout}${stderr}")
endif()
