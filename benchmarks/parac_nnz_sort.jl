# Minimal nnz-sort reorder for ParAC GPU — the FALLBACK.
#
# THE RUNNER DOES NOT NORMALLY CALL THIS: the GPU input is produced by ParAC's own
# cpu_implementation/write_graph.jl with method "nnz-sort", run from their checkout
# through benchmarks/parac_produce_upstream.jl (see parac_runner._nnz_sort). This
# script is the stand-in for an input their producer refuses, and a cell prepared
# with it records that in matrix_meta.parac_prep.
#
# It reproduces their ordering from graph_share "nnz-sort": a RANDOM permutation
# first, THEN a sort by per-column nnz. Like parac_reorder_amd.jl it PRESERVES all
# values (incl. the diagonal) so it is correct for SDDM too — it only applies
# ParAC's depth-reducing permutation. It prints the SORT COMPUTE time (excluding
# the reindex + mmwrite I/O), matching how their producer reports sort time, so
# GPU and CPU ParAC count reorder time the SAME way.
#
#   julia parac_nnz_sort.jl <in.mtx> <out.mtx> [--augment]
#
# --augment appends ParAC's ground row/column AFTER the permutation, the same step
# write_graph.jl's `physics_share` does. driver_physics.cu trims that node back off
# (trim_input_laplacian_device), so with it the GPU physics driver solves the
# published operator; without it the trim deletes a real degree of freedom.
using SparseArrays, LinearAlgebra, MatrixMarket, Random

inpath  = ARGS[1]
outpath = ARGS[2]
augment = "--augment" in ARGS
Random.seed!(0)                      # reproducible

G = MatrixMarket.mmread(inpath)
G = SparseMatrixCSC{Float64, Int64}(G)
if !issymmetric(G)
    G = (G + G') / 2
end
n = size(G, 1)
deg = diff(G.colptr)                 # per-column nnz of G

t1 = time()                          # time ONLY the permutation compute
p = randperm(n)                      # break structure first (essential)
q = sortperm(deg[p])                 # then sort by nnz ascending (stable-ish)
order = p[q]                         # composed permutation
println("sort time: ", time() - t1, " s")

Gnew = G[order, order]

if augment
    # write_graph.jl physics_share: the appended column holds the per-row diagonal
    # excess (clamped at 0) and the appended diagonal is their sum.
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
println("wrote ", outpath, "  n=", size(Gnew, 1), "  nnz=", nnz(Gnew))
