#!/usr/bin/env julia
# Fill-only measurement for Laplacians.jl ApproxChol (AC / AC2). Loads the
# adjacency from an .mtx (off-diagonal magnitudes, mirroring
# bench_laplacians.load_mtx_adj), runs ONLY the low-level factorization (no PCG
# solve, so it is fast even on the power-law graphs where AC's solve is
# iteration-heavy), and prints the fill on the same definition every solver uses:
#     fill = 2 * offdiag(L) / offdiag(A) = 2 * length(ldli.fval) / nnz(adj)
# (this is exactly Laplacians' own "ratio of operator edges to original edges",
# approxChol.jl:1620, and matches apxchol's 2*offdiag(L)/adj_nnz.)
#
# AC  = ApproxCholParams(:deg)            -> LLmatp(a),          approxChol(llmat)
# AC2 = ApproxCholParams(:deg, 5, 2, 2)   -> LLmatp(a, split=2), approxChol(llmat, split=2, merge=2)
# AC2 oversamples (splits each edge into `split` multiedges in the LLmatp build),
# so its factor is denser than AC's — must be replicated exactly (the split has to
# be passed to BOTH LLmatp and approxChol, per approxchol_lapGreedy at :1600-1605).
#   julia --project=benchmarks/julia benchmarks/julia/ac_fill.jl <matrix.mtx> [ac|ac2]
using Laplacians, SparseArrays

function load_mtx_adj(path)
    open(path) do io
        header = lowercase(readline(io))
        is_symmetric = occursin("symmetric", header)
        is_pattern   = occursin("pattern", header)
        occursin("array", header) && error("dense format unsupported: $path")
        line = ""
        while !eof(io); line = readline(io); startswith(line, '%') || break; end
        parts = split(line); rows = parse(Int, parts[1]); nnz_ = parse(Int, parts[3])
        I = Int[]; J = Int[]; V = Float64[]
        for _ in 1:nnz_
            p = split(readline(io)); i = parse(Int, p[1]); j = parse(Int, p[2])
            i == j && continue
            w = is_pattern ? 1.0 : abs(parse(Float64, p[3]))
            push!(I, i); push!(J, j); push!(V, w)
            is_symmetric && (push!(I, j); push!(J, i); push!(V, w))
        end
        return sparse(I, J, V, rows, rows)
    end
end

# Full signed loader (keeps diagonal + sign) for SDDM matrices (IPM normal eqns).
function load_mtx_full(path)
    open(path) do io
        header = lowercase(readline(io))
        is_symmetric = occursin("symmetric", header); is_pattern = occursin("pattern", header)
        line = ""
        while !eof(io); line = readline(io); startswith(line, '%') || break; end
        parts = split(line); rows = parse(Int, parts[1]); nnz_ = parse(Int, parts[3])
        I = Int[]; J = Int[]; V = Float64[]
        for _ in 1:nnz_
            p = split(readline(io)); i = parse(Int, p[1]); j = parse(Int, p[2])
            v = is_pattern ? 1.0 : parse(Float64, p[3])     # keep sign
            push!(I, i); push!(J, j); push!(V, v)
            is_symmetric && i != j && (push!(I, j); push!(J, i); push!(V, v))
        end
        return sparse(I, J, V, rows, rows)
    end
end

variant = length(ARGS) >= 2 ? ARGS[2] : "ac"
sddm    = length(ARGS) >= 3 && ARGS[3] == "sddm"
if sddm
    # IPM matrices are SDDM. Mirror approxchol_sddm = sddmWrapLap(approxchol_lap):
    # extract adjacency + diagonal excess, augment to a Laplacian (extra ground
    # vertex), factor that. fill is taken against the ORIGINAL SDDM off-diagonal
    # (nnz(a)) so the denominator matches apxchol's adj_nnz; the ground vertex adds
    # only one extra row of fill, negligible vs the n-vertex factor.
    M = load_mtx_full(ARGS[1])
    a, dVal, dExcess = Laplacians.adjValAndExcess(M)
    a1 = Laplacians.extendMatrix(a, dVal, dExcess)
    llm = variant == "ac2" ? Laplacians.LLmatp(a1, 2) : Laplacians.LLmatp(a1)
    ldli = variant == "ac2" ? Laplacians.approxChol(llm, 2, 2) : Laplacians.approxChol(llm)
    offL = length(ldli.fval); adjnz = nnz(a); nn = size(M, 1)
else
    # Use the HIGH-LEVEL approxchol_lap (exactly what the bench uses) -- it
    # preprocesses the graph robustly, so it handles dense scale-free graphs like
    # kron where the low-level approxChol(LLmatp) BoundsErrors. verbose=true prints
    # "Ratio of operator edges to original edges:" = 2*offdiag(L)/nnz(adj) = our
    # fill (verified identical to the low-level path on grids: 2.203 vs 2.2025).
    adj  = load_mtx_adj(ARGS[1])
    params = variant == "ac2" ? Laplacians.ApproxCholParams(:deg, 5, 2, 2) :
                                 Laplacians.ApproxCholParams(:deg)
    # capture the verbose "Ratio of operator edges..." via a temp file (redirect to
    # an IOBuffer fails -- redirect_stdout needs a real fd).
    tmp = tempname()
    open(tmp, "w") do io
        redirect_stdout(io) do
            approxchol_lap(adj; params=params, verbose=true)
        end
    end
    out = read(tmp, String); rm(tmp, force=true)
    mr = match(r"Ratio of operator edges to original edges:\s*([0-9.eE+-]+)", out)
    fillv = parse(Float64, mr.captures[1])
    adjnz = nnz(adj); nn = size(adj, 1); offL = round(Int, fillv * adjnz / 2)
end
fill  = 2.0 * offL / adjnz
println("AC_FILL variant=$variant n=$nn offdiagL=$offL offdiagA=$adjnz fill=$(round(fill, digits=4))")
