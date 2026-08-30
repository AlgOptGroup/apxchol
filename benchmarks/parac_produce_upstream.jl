# Run ParAC's OWN preprocessing on a matrix we dumped — THEIR code, unmodified.
#
# ParAC prepares every input it benchmarks with cpu_implementation/write_graph.jl:
#   physics_produce(path, method)  permute, then APPEND the ground row/column that
#                                  turns an SDDM operator into a Laplacian (the node
#                                  driver_physics' remove_last_row_and_column trims).
#   graph_produce(path, method)    strip the diagonal, force the off-diagonals
#                                  negative, permute, then REBUILD the diagonal as
#                                  -colsum, i.e. re-derive the pure Laplacian.
# Our runner charges this preprocessing to ParAC's setup time, so it has to be
# their implementation doing their work — not our reimplementation of it.
#
#   julia --project=benchmarks/julia benchmarks/parac_produce_upstream.jl \
#         <write_graph.jl> <prefix> <physics|graph> [amd|nnz-sort|random|nd]
#
# `prefix` follows write_graph.jl's OWN convention: it reads `prefix.mtx` and
# writes `prefix-amd.mtx` (or `-nnz-sorted.mtx`). The caller puts `prefix` inside
# OUR cache directory and symlinks `prefix.mtx` at the dump, so nothing is ever
# written into the ParAC checkout.
#
# This file only dispatches: every line of preprocessing runs inside their
# write_graph.jl. Their "amd time:"/"sort time:" covers only the ordering kernel,
# before permutation materialization, augmentation and Matrix Market output. The
# complete interval below therefore brackets their whole producer call, after
# Julia/package loading and first-call compilation, and is the setup time the
# runner charges. The warm-up uses a distinct tiny matrix, so the real input and
# output remain cold. nnz-sort's global RNG state is restored before the real
# call, so warming cannot change the sampled permutation.
#
# Their `physics_produce` asserts on a globally diagonally-DEFICIENT input
# (sum(G) < -1e-9). Julia's @assert aborts with no usable message, so translate
# it into an explicit, greppable failure first — the check is theirs, verbatim;
# only the way it is reported is ours. Matrices that fail it keep our own
# parac_reorder_amd.jl (see benchmarks/patches/parac/README.md).

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

include(write_graph_jl)   # THEIR script, from THEIR checkout, byte-for-byte

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
            MatrixMarket.mmwrite(warm_prefix * ".mtx", warm)
            if mode == "physics"
                physics_produce(warm_prefix, method)
            else
                graph_produce(warm_prefix, method)
            end
        end
    finally
        copy!(Random.default_rng(), saved_rng)
    end
end

prep_start = time()
if mode == "physics"
    physics_produce(prefix, method)
elseif mode == "graph"
    graph_produce(prefix, method)
else
    error("mode must be 'physics' or 'graph', got $mode")
end
println("APX complete preprocessing time: ", time() - prep_start)

println("upstream ", mode, "_produce(", prefix, ", \"", method, "\") done")
