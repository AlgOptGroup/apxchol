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
# write_graph.jl. Its "amd time:"/"sort time:" print is what the runner parses,
# unchanged, so the reorder seconds we charge are the ones their code measured.
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

using SparseArrays, MatrixMarket

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

if mode == "physics"
    physics_produce(prefix, method)
elseif mode == "graph"
    graph_produce(prefix, method)
else
    error("mode must be 'physics' or 'graph', got $mode")
end

println("upstream ", mode, "_produce(", prefix, ", \"", method, "\") done")
