# Example programs — each is a self-verifying demonstration of one public seam.
foreach(ex basic_solve custom_eliminator custom_order)
    add_executable(example_${ex} ${ex}.cpp)
    target_link_libraries(example_${ex} PRIVATE apxchol_core)
endforeach()
