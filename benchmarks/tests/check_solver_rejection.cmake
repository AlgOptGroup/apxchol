# A missing matrix makes the ordering observable: reject the solver before IO.
execute_process(
    COMMAND "${BENCHMARK}" --solver "${SOLVER}" --mtx missing-solver-test.mtx --csv
    RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
if (NOT status STREQUAL "2" OR NOT stderr MATCHES "is unknown or unavailable in this build")
    message(FATAL_ERROR "Expected solver rejection, got ${status}: ${stderr}")
endif()
if (stdout MATCHES "solver,graph,n,nnz")
    message(FATAL_ERROR "Unavailable solver emitted a misleading CSV header")
endif()
