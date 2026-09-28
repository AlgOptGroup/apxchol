# Reject retired storage labels, including a mixed request, before IO.
foreach(storage IN ITEMS vec_pool fwd_star forward_star)
    foreach(config IN ITEMS "${storage}" "bg+tree[${storage}]" "bk+tree[${storage}]"
                            "greedy+tree[${storage}]"
                            "bg+tree[vec_pool_aos],bg+tree[${storage}]")
        execute_process(
            COMMAND "${BENCHMARK}" --solver apxchol_v1 --v1-configs "${config}"
                --mtx missing-storage-test.mtx --csv
            RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
        if (NOT status STREQUAL "2" OR NOT stderr MATCHES "Storage '${storage}' is retired")
            message(FATAL_ERROR "Expected storage rejection for ${config}, got ${status}: ${stderr}")
        endif()
        if (stdout MATCHES "solver,graph,n,nnz")
            message(FATAL_ERROR "Retired storage emitted a misleading CSV header")
        endif()
    endforeach()
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
