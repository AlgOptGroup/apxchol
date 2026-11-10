execute_process(
        COMMAND "${SANITIZER}" --tool initcheck --error-exitcode 99
        "${UNIT_TESTS}"
        --gtest_filter=Solver.GpuScalarAndBatchedResidualsDescribeReturnedSolutions
        RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
message("${stdout}${stderr}")
if (NOT status STREQUAL "0")
    message(FATAL_ERROR "GPU factor setup initcheck failed: ${status}")
endif()
# CTest's skip regex overrides a failing exit status. Emit our marker only
# after the sanitizer and child have both exited successfully.
if ("${stdout}${stderr}" MATCHES "\\[  SKIPPED \\]")
    message("APXCHOL_GPU_FACTOR_SETUP_INITCHECK_SKIPPED")
endif()
