# Reject retired indexed labels, including a mixed request, before IO.
foreach(config IN ITEMS "bg+tree[vec_pool]" "bk+tree[vec_pool]"
                        "greedy+tree[vec_pool]"
                        "bg+tree[vec_pool_aos],bg+tree[vec_pool]")
    execute_process(
        COMMAND "${BENCHMARK}" --solver apxchol_v1 --v1-configs "${config}"
            --mtx missing-storage-test.mtx --csv
        RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    if (NOT status STREQUAL "2" OR NOT stderr MATCHES "Storage 'vec_pool' is retired")
        message(FATAL_ERROR "Expected indexed storage rejection for ${config}, got ${status}: ${stderr}")
    endif()
    if (stdout MATCHES "solver,graph,n,nnz")
        message(FATAL_ERROR "Retired storage emitted a misleading CSV header")
    endif()
endforeach()

# Surviving tags must pass selection and reach the expected input error.
foreach(config IN ITEMS "bg+tree[vec_pool_aos]" "bg+tree[bstr]")
    execute_process(
        COMMAND "${BENCHMARK}" --solver apxchol_v1 --v1-configs "${config}"
            --mtx missing-storage-test.mtx --kind graph --threads 1 --csv
        RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    if (NOT status STREQUAL "1" OR NOT stderr MATCHES "cannot read missing-storage-test.mtx")
        message(FATAL_ERROR "Surviving ${config} did not reach input IO: ${status}: ${stderr}")
    endif()
endforeach()
