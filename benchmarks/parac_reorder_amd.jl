# AMD reorder for ParAC — OUR preprocessing step, not a modification of ParAC.
#
# ParAC requires a fill-reducing ordering: fed the raw input its elimination tree
# is pathological (the CPU factor does not finish in reasonable time), and its own
# benchmark scripts only ever run `*-amd.mtx` files. Its own producer for those
# files is cpu_implementation/write_graph.jl, which we deliberately do NOT use:
# `graph_produce` REBUILDS the diagonal as a pure Laplacian, which silently
# destroys the diagonal of an SDDM operator. This script applies only the
# permutation, preserving every value including the diagonal.
#
#   julia parac_reorder_amd.jl <in.mtx> <out.mtx> [--augment] [--perm <file>]
#
# --augment reproduces ParAC's own physics input construction (write_graph.jl
# `physics_produce`): permute first, then append the ground row/column that turns
# an SDDM operator into a Laplacian, so the appended node is LAST. That is the
# node ParAC's physics mode (`driver <mtx> <threads> "" 1`) trims, which is why
# physics mode then solves the published operator itself. Without this step the
# trim would delete a REAL degree of freedom of the published operator.
#
# --perm writes the permutation p (1-based, one index per line) with G_out =
# G_in[p, p], so a solution can be mapped back to the input's index space and
# scored against the published operator.
using SparseArrays, LinearAlgebra, MatrixMarket, AMD

inpath  = ARGS[1]
outpath = ARGS[2]
augment  = "--augment" in ARGS
permpath = let i = findfirst(==("--perm"), ARGS)
    i === nothing ? nothing : ARGS[i + 1]
end

G = MatrixMarket.mmread(inpath)
G = SparseMatrixCSC{Float64, Int64}(G)
if !issymmetric(G)
    println("warning: input not symmetric, symmetrizing")
    G = (G + G') / 2
end
t1 = time()
p = amd(G)
println("amd time: ", time() - t1, " s")
Gnew = G[p, p]

if augment
    # ParAC's write_graph.jl physics_produce, verbatim in effect: the appended
    # column holds the per-row diagonal excess (clamped at 0), and the appended
    # diagonal is their sum, so the result is an exact Laplacian.
    check_sum = sum(Gnew)
    if check_sum < 0 && abs(check_sum) > 1e-9
        error("not diagonally dominant: sum(G) = $check_sum")
    end
    if check_sum > 1e-9
        println("not directly laplacian, append to make laplacian")
        col_append = -sum(Gnew, dims = 1)
        col_append[findall(x -> x > 0, col_append)] .= 0
        last_sum = -sum(col_append)
        Gnew = vcat(Gnew, col_append)
        Gnew = hcat(Gnew, [col_append'; last_sum])
    else
        println("already a laplacian, nothing appended")
    end
end

MatrixMarket.mmwrite(outpath, Gnew)
if permpath !== nothing
    open(permpath, "w") do io
        for i in p
            println(io, i)
        end
    end
    println("wrote permutation to ", permpath)
end
println("wrote ", outpath, "  n=", size(Gnew, 1), "  nnz=", nnz(Gnew))
