# CPU routing must work in both CPU and CUDA builds, without a visible GPU.
# Auto must also avoid CUDA for a host-only configuration, and for default
# options when this build cannot select the GPU route (CPU or wide nodes).
set(cases explicit_cpu explicit_options auto_host_only)
if (NOT USE_CUDA OR WIDE_NODES)
    list(APPEND cases auto_default)
endif()
foreach(case IN LISTS cases)
    set(backend auto)
    set(config "bg+tree[vec_pool_aos]")
    set(options)
    if (case STREQUAL "explicit_cpu")
        set(backend cpu)
    elseif (case STREQUAL "explicit_options")
        set(backend cpu)
        set(options --factor-storage float32 --cpu-schedule levels --factor-drop 0
            --regularization 0 --lump-max-mass 0 --degree-quantile 0.3 --profile)
    elseif (case STREQUAL "auto_host_only")
        set(config "greedy+tree[vec_pool_aos]")
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1
            "${BENCHMARK}" --graph grid --n 4 --report-fill
            --solver apxchol_v1 --v1-configs "${config}"
            --v1-backend ${backend} --threads 1 --repeat 2 --tol 1e-8 --maxiter 100 --csv ${options}
        RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    if (NOT status STREQUAL "0")
        message(FATAL_ERROR "Complete CPU benchmark route failed (${case}): ${status}: ${stderr}")
    endif()
    set(label "v1 ${config} [vec_pool_aos] [1t]")
    string(FIND "${stdout}" "${label}," label_position)
    if (label_position LESS 0)
        message(FATAL_ERROR "CSV changed the existing configuration label (${case}): ${stdout}")
    endif()
    if (NOT stdout MATCHES "stop_check_s,execution_route" OR
        NOT stdout MATCHES "original-v1,[^,]+,[^,]+,cpu")
        message(FATAL_ERROR "CSV lacks the complete CPU route receipt (${case}): ${stdout}")
    endif()
    string(REGEX MATCHALL "BENCH_REPEAT[^\n]*execution_route=cpu" repeats "${stderr}")
    list(LENGTH repeats repeat_count)
    string(REGEX MATCHALL "FILL[^\n]*execution_route=cpu source=measured_owner" fills "${stderr}")
    list(LENGTH fills fill_count)
    if (NOT repeat_count EQUAL 2 OR NOT fill_count EQUAL 2)
        message(FATAL_ERROR "Expected two measured owners/receipts (${case}), got ${repeat_count}/${fill_count}: ${stderr}")
    endif()
    if (stderr MATCHES "\\[bench\\] cuda_init")
        message(FATAL_ERROR "CPU route attempted CUDA warmup (${case}): ${stderr}")
    endif()
    if (case STREQUAL "explicit_options" AND
        (NOT stderr MATCHES "drop_rel=0" OR NOT stderr MATCHES "--- profile:"))
        message(FATAL_ERROR "Explicit factor/profile options were not honored: ${stderr}")
    endif()
endforeach()
execute_process(
    COMMAND "${BENCHMARK}" --v1-backend hybrid --mtx missing-route-test.mtx
    RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
if (status STREQUAL "0" OR NOT stderr MATCHES "--v1-backend must be auto\\|cpu\\|gpu")
    message(FATAL_ERROR "Invalid complete route was not rejected before IO: ${status}: ${stderr}")
endif()

foreach(case IN ITEMS "factor-storage;float64" "cpu-schedule;invalid" "factor-drop;-1"
        "regularization;nan" "lump-max-mass;2" "degree-quantile;2")
    list(GET case 0 option)
    list(GET case 1 value)
    execute_process(
        COMMAND "${BENCHMARK}" --${option} "${value}" --mtx missing-options-test.mtx
            --kind graph --threads 1
        RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    if (NOT status STREQUAL "2" OR stderr MATCHES "cannot read missing-options-test.mtx")
        message(FATAL_ERROR "Invalid --${option} was not rejected before IO: ${status}: ${stderr}")
    endif()
endforeach()
