# Minimal nnz-sort reorder for ParAC GPU — ParAC's own ordering from
# cpu_implementation/write_graph.jl (graph_share "nnz-sort"): a RANDOM permutation
# first, THEN a sort by per-column nnz. Like reorder_amd.jl it PRESERVES all values
# (incl. the diagonal) so it is correct for SDDM too — it only applies ParAC's
# depth-reducing permutation. It prints the SORT COMPUTE time (excluding the
# reindex + mmwrite I/O), matching how reorder_amd.jl reports amd time, so GPU and
# CPU ParAC count reorder time the SAME way.
using SparseArrays, LinearAlgebra, MatrixMarket, Random

inpath  = ARGS[1]
outpath = ARGS[2]
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
MatrixMarket.mmwrite(outpath, Gnew)
println("wrote ", outpath, "  n=", size(Gnew, 1), "  nnz=", nnz(Gnew))
