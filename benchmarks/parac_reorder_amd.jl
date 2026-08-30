# AMD reorder for ParAC — the FALLBACK preprocessing step.
#
# THE RUNNER DOES NOT NORMALLY CALL THIS. ParAC's inputs are prepared by ParAC's
# OWN cpu_implementation/write_graph.jl (`graph_produce` / `physics_produce`),
# invoked from their checkout through benchmarks/parac_produce_upstream.jl: the
# runner charges that preprocessing to ParAC's setup time, so it has to be their
# code. This script is what parac_runner falls back to when their producer refuses
# an input — `physics_produce` asserts on a globally diagonally-deficient matrix —
# and a cell prepared this way says so in matrix_meta.parac_prep.
#
# It is a faithful stand-in, not an approximation: its output is byte-identical to
# theirs on every matrix compared (apache2, G3_circuit via physics_produce;
# com-Amazon comp0, grid_1000 comp0 via graph_produce — see
# benchmarks/patches/parac/README.md). It differs only in applying ONLY the
# permutation, so it is also correct for an input `graph_produce` would rebuild
# the diagonal of, and in offering --perm, which their producer does not write.
#
# ParAC requires a fill-reducing ordering: fed the raw input its elimination tree
# is pathological (the CPU factor does not finish in reasonable time), and its own
# benchmark scripts only ever run `*-amd.mtx` files.
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
augment = "--augment" in ARGS
permpath = let i = findfirst(==("--perm"), ARGS)
    i === nothing ? nothing : ARGS[i + 1]
end

function produce(inpath, outpath, augment, permpath; report=false)
    G = SparseMatrixCSC{Float64, Int64}(MatrixMarket.mmread(inpath))
    if !issymmetric(G)
        report && println("warning: input not symmetric, symmetrizing")
        G = (G + G') / 2
    end
    t1 = time()
    p = amd(G)
    report && println("amd time: ", time() - t1, " s")
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
            report && println("not directly laplacian, append to make laplacian")
            col_append = -sum(Gnew, dims = 1)
            col_append[findall(x -> x > 0, col_append)] .= 0
            last_sum = -sum(col_append)
            Gnew = vcat(Gnew, col_append)
            Gnew = hcat(Gnew, [col_append'; last_sum])
        else
            report && println("already a laplacian, nothing appended")
        end
    end

    MatrixMarket.mmwrite(outpath, Gnew)
    if permpath !== nothing
        open(permpath, "w") do io
            for i in p
                println(io, i)
            end
        end
        report && println("wrote permutation to ", permpath)
    end
    report && println("wrote ", outpath, "  n=", size(Gnew, 1), "  nnz=", nnz(Gnew))
end

# Compile the complete fallback path on a distinct tiny matrix. The real input
# and output stay cold, so the charged interval retains all mandatory I/O and
# transformation work but not reusable Julia first-call compilation.
mktempdir() do warm_dir
    warm_in = joinpath(warm_dir, "warm.mtx")
    warm_out = joinpath(warm_dir, "warm-out.mtx")
    warm = spdiagm(-1 => [-1.0, -1.0, -1.0],
                    0 => [2.25, 2.25, 2.25, 2.25],
                    1 => [-1.0, -1.0, -1.0])
    MatrixMarket.mmwrite(warm_in, warm)
    produce(warm_in, warm_out, augment, nothing)
end

prep_start = time()
produce(inpath, outpath, augment, permpath; report=true)
println("APX complete preprocessing time: ", time() - prep_start)
