if (DEFINED RHS)
    set(rhs_args --rhs "${RHS}")
else()
    set(rhs_args --random-rhs)
endif()
execute_process(
    COMMAND "${CLI}" "${INPUT}" ${rhs_args} --tol 1e-8 ${CLI_ARGS}
    RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
if (NOT status STREQUAL "${EXPECT_STATUS}" OR
    NOT "${stdout}${stderr}" MATCHES "${EXPECT_OUTPUT}")
    message(FATAL_ERROR
        "Expected exit ${EXPECT_STATUS} and '${EXPECT_OUTPUT}', got ${status}:\n${stdout}${stderr}")
endif()
if (DEFINED SOLUTION)
    file(READ "${SOLUTION}" solution)
    if (NOT solution MATCHES "${EXPECT_SOLUTION}")
        message(FATAL_ERROR "Expected '${EXPECT_SOLUTION}' in ${SOLUTION}, got:\n${solution}")
    endif()
endif()
