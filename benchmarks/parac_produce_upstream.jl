# Dispatch to the pinned, patched ParAC producer; no graph/order arithmetic here.
# julia --project=benchmarks/julia parac_produce_upstream.jl \
#       <write_graph.jl> <prefix> <physics|graph> [amd|nnz-sort|random|nd]
#
# Graph/AMD uses patch0006's in-memory entry: load common input before timing,
# charge the graph transform, ordering/permutation, explicit final GC and remaining
# producer overhead, and report final MatrixMarket serialization separately.
# No original-system capsule is made by this maintained entry; its audit and
# separate operator/RHS-permutation intervals are explicitly zero. Graph G[p,p]
# remains charged inside producer_ordering_s. The private original-A,b campaign
# additionally charges its actual operator/RHS permutation; see patches/parac/.
# Physics and other ordering methods retain their complete producer interval.
# Reusable Julia/package/JIT startup is excluded using a distinct tiny warmup;
# the RNG state is restored before the real input. Output stays in our cache.

length(ARGS) >= 3 || error("usage: parac_produce_upstream.jl <write_graph.jl> <prefix> <physics|graph> [method]")
write_graph_jl = ARGS[1]
prefix         = ARGS[2]
mode           = ARGS[3]
method         = length(ARGS) >= 4 ? ARGS[4] : "amd"

isfile(write_graph_jl) || error("ParAC write_graph.jl not found at $write_graph_jl")
isfile(prefix * ".mtx") || error("input not found at $(prefix).mtx")

using SparseArrays, MatrixMarket, Random

# Pre-flight ONLY for the physics path: report their `@assert false` as a message.
if mode == "physics"
    local G = SparseMatrixCSC{Float64,Int64}(MatrixMarket.mmread(prefix * ".mtx"))
    local check_sum = sum(G)
    if check_sum < 0 && abs(check_sum) > 1e-9
        error("ParAC physics_produce would assert: 'not diagonally dominant' " *
              "(sum(G) = $check_sum < -1e-9)")
    end
    G = nothing
    GC.gc()
end

include(write_graph_jl)   # Pinned upstream plus the recorded benchmark patches.

# ParAC ships preprocessing as Julia functions. Charging the first invocation
# would charge several seconds of reusable JIT work to every independently
# launched matrix, while all C++ competitors are ahead-of-time compiled and the
# equally reusable CUDA primary-context startup is reported separately. Compile
# this exact mode/method on a different tiny matrix before touching the real one.
let saved_rng = copy(Random.default_rng())
    try
        mktempdir() do warm_dir
            warm_prefix = joinpath(warm_dir, "warm")
            warm = spdiagm(-1 => [-1.0, -1.0, -1.0],
                            0 => [2.25, 2.25, 2.25, 2.25],
                            1 => [-1.0, -1.0, -1.0])
            if mode == "graph" && method == "amd"
                applicable(graph_produce, warm_prefix, warm, method) ||
                    error("ParAC Graph/AMD requires patch0006 in-memory producer")
                graph_produce(warm_prefix, warm, method)
            elseif mode == "physics"
                MatrixMarket.mmwrite(warm_prefix * ".mtx", warm)
                physics_produce(warm_prefix, method)
            else
                MatrixMarket.mmwrite(warm_prefix * ".mtx", warm)
                graph_produce(warm_prefix, method)
            end
        end
    finally
        copy!(Random.default_rng(), saved_rng)
    end
end

if mode == "graph" && method == "amd"
    input_start = time()
    G = SparseMatrixCSC{Float64,Int64}(MatrixMarket.mmread(prefix * ".mtx"))
    input_read_s = time() - input_start
    prep_start = time()
    producer_times = graph_produce(prefix, G, method)
    complete_s = time() - prep_start
    serialization_s = producer_times.serialization_s
    algorithm_s = complete_s - serialization_s
    algorithm_s >= 0 || error("overlapping/invalid preparation intervals")
    println("APX preprocessing accounting schema: graph-algorithm-preprocessing-v1")
    println("APX complete preprocessing time: ", complete_s)
    println("APX algorithm preprocessing time: ", algorithm_s)
    println("APX preprocessing audit time: ", 0.0)
    println("APX preprocessing serialization time: ", serialization_s)
    println("APX preprocessing input read time: ", input_read_s)
    println("APX preprocessing operator permutation time: ", 0.0)
    println("APX preprocessing producer transform time: ", producer_times.transform_s)
    println("APX preprocessing producer ordering time: ", producer_times.ordering_s)
    println("APX preprocessing producer cleanup time: ", producer_times.cleanup_s)
    # Explicit final GC is charged; automatic GC follows the interval it occurs in.
else
    prep_start = time()
    if mode == "physics"
        physics_produce(prefix, method)
    elseif mode == "graph"
        graph_produce(prefix, method)
    else
        error("mode must be 'physics' or 'graph', got $mode")
    end
    println("APX complete preprocessing time: ", time() - prep_start)
end

println("upstream ", mode, "_produce(", prefix, ", \"", method, "\") done")
