# Example programs — each is a self-verifying demonstration of one public seam
# and exits nonzero when its solve misses the residual it checks.
foreach(ex basic_solve custom_eliminator custom_order)
    add_executable(example_${ex} ${ex}.cpp)
    target_link_libraries(example_${ex} PRIVATE apxchol_core)
    add_test(NAME example_${ex} COMMAND example_${ex})
    set_tests_properties(example_${ex} PROPERTIES
        TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=4")
endforeach()
