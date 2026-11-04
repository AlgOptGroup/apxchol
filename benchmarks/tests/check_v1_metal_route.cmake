# One tiny explicit Metal cell: the route reports itself as metal in the CSV,
# every repetition and the FILL line, the device warm-up is reported once
# outside timing, and the factor-drop statistics come from the Metal setup.
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env OMP_NUM_THREADS=2
        "${BENCHMARK}" --graph grid --n 64 --report-fill
        --solver apxchol_v1 --v1-configs "bg+tree[vec_pool_aos]" --v1-backend gpu
        --factor-drop 0.01 --threads 2 --repeat 2 --tol 1e-8 --maxiter 1000 --csv
    RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
if (NOT status STREQUAL "0" AND stderr MATCHES "Metal backend is unavailable")
    message("SKIP: no usable Metal device: ${stderr}")
    return()
endif()
if (NOT status STREQUAL "0")
    message(FATAL_ERROR "Metal benchmark route failed: ${status}: ${stderr}")
endif()
if (NOT stdout MATCHES "original-v1,[^,]+,[^,]+,metal\n")
    message(FATAL_ERROR "CSV lacks the Metal route receipt: ${stdout}")
endif()
string(REGEX MATCHALL "BENCH_REPEAT[^\n]*execution_route=metal" repeats "${stderr}")
list(LENGTH repeats repeat_count)
string(REGEX MATCHALL "FILL[^\n]*execution_route=metal source=measured_owner" fills "${stderr}")
list(LENGTH fills fill_count)
string(REGEX MATCHALL "\\[bench\\] metal_init" inits "${stderr}")
list(LENGTH inits init_count)
if (NOT repeat_count EQUAL 2 OR NOT fill_count EQUAL 2 OR NOT init_count EQUAL 1)
    message(FATAL_ERROR "Expected two Metal receipts and one warm-up, got "
                        "${repeat_count}/${fill_count}/${init_count}: ${stderr}")
endif()
if (NOT stderr MATCHES "stored_nnz=[1-9][0-9]* \\(L11_nnz=[1-9][0-9]* dropped=[0-9]+ drop_rel=0\\.010*\\)")
    message(FATAL_ERROR "FILL lacks the Metal factor-drop statistics: ${stderr}")
endif()
if (stderr MATCHES "\\[bench\\] cuda_init")
    message(FATAL_ERROR "Metal route attempted CUDA warmup: ${stderr}")
endif()
message("${stdout}${stderr}")
