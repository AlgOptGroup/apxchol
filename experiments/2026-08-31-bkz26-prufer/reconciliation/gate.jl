using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)
Threads.nthreads()==1 || error("one Julia thread required")
VERSION == v"1.12.7" || error("pinned Julia required")
include("test_adapter.jl")
println("INTERVENTION_NATIVE_GATE_PASS")
