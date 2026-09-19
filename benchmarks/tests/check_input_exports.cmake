string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef unique)
foreach(option dump-mtx dump-rhs)
    set(output "${CMAKE_CURRENT_BINARY_DIR}/input-export-${unique}-${option}.mtx")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env OMP_NUM_THREADS=1 KMP_AFFINITY=norespect
            "${BENCHMARK}" --graph grid --n 4 --solver none --${option} "${output}"
        RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    if (NOT status STREQUAL "0" OR NOT EXISTS "${output}")
        message(FATAL_ERROR "Input export rejected (${option}): ${status}: ${stderr}")
    endif()
    file(READ "${output}" content LIMIT 200)
    file(REMOVE "${output}")
    if (NOT content MATCHES "^%%MatrixMarket")
        message(FATAL_ERROR "Input export produced no Matrix Market data")
    endif()
endforeach()
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env OMP_NUM_THREADS=1 KMP_AFFINITY=norespect
        "${BENCHMARK}" --graph grid --n 4 --solver none --component-info
    RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
if (NOT status STREQUAL "0")
    message(FATAL_ERROR "Component inspection rejected: ${status}: ${stderr}")
endif()
