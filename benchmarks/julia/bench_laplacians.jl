#!/usr/bin/env julia
# bench_laplacians.jl — Benchmark Laplacians.jl ApproxChol solver
#
# Follows the SDDM2023 protocol (arXiv:2303.00709):
#   tolerance 1e-8, measures build_time, solve_time, iterations, ‖Ax-b‖/‖b‖
#
# Usage:
#   julia --project=benchmarks/julia benchmarks/julia/bench_laplacians.jl [--csv] [--operator PATH] [options]
#
# Two ways in, and only one of them is used by the sweep:
#
#   --operator PATH  (what sweep_fair.py passes)  PATH holds the operator the
#       C++ benchmark assembled and wrote with --dump-mtx: L = D - A for a
#       kind=graph matrix, the published matrix for a kind=operator one. It is
#       read AS AN OPERATOR — diagonal included, signs untouched — and solved as
#       it stands, so AC/AC2 demonstrably solve the same system as every other
#       solver in the cell instead of re-deriving one of their own.
#
#   --mtx PATH  (manual use)  PATH is read as an ADJACENCY file the way the
#       old benchmark did (drop the diagonal, take |value|) and the Laplacian is
#       rebuilt here. Kept for ad-hoc comparisons; it is NOT how the suite runs,
#       precisely because "both sides happen to build the same matrix" is an
#       assumption rather than something the run shows.
#
# Generated graphs (--graph grid|checkerboard|erdos) match our C++ generators.

using Laplacians
using SparseArrays
using LinearAlgebra
using Statistics
using Printf
using Random

# ── Argument parsing ──────────────────────────────────

function parse_args(args)
    opts = Dict{String,Any}(
        "graph"   => "checkerboard",
        "n"       => 500,
        "kappa"   => 1000.0,
        "tile"    => 4,
        "er_p"    => 0.01,
        "mtx"     => "",
        "operator" => "",
        "solver"  => "all",  # ac, ac2, chol, cg, all
        "tol"     => 1e-8,
        "maxiter" => 500,
        "csv"     => false,
        "seed"    => 42,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--csv"
            opts["csv"] = true
        elseif a == "--graph" && i < length(args)
            i += 1; opts["graph"] = args[i]
        elseif a == "--n" && i < length(args)
            i += 1; opts["n"] = parse(Int, args[i])
        elseif a == "--kappa" && i < length(args)
            i += 1; opts["kappa"] = parse(Float64, args[i])
        elseif a == "--tile" && i < length(args)
            i += 1; opts["tile"] = parse(Int, args[i])
        elseif a == "--er-p" && i < length(args)
            i += 1; opts["er_p"] = parse(Float64, args[i])
        elseif a == "--mtx" && i < length(args)
            i += 1; opts["mtx"] = args[i]; opts["graph"] = "mtx"
        elseif a == "--operator" && i < length(args)
            i += 1; opts["operator"] = args[i]; opts["graph"] = "operator"
        elseif a == "--solver" && i < length(args)
            i += 1; opts["solver"] = args[i]
        elseif a == "--tol" && i < length(args)
            i += 1; opts["tol"] = parse(Float64, args[i])
        elseif a == "--maxiter" && i < length(args)
            i += 1; opts["maxiter"] = parse(Int, args[i])
        elseif a == "--seed" && i < length(args)
            i += 1; opts["seed"] = parse(Int, args[i])
        end
        i += 1
    end
    return opts
end

# ── Graph construction (matching our C++ generators) ──

function grid_graph_checkerboard_adj(n, m, a_high, a_low, tile)
    # Build a weighted grid graph as sparse adjacency matrix
    N = n * m
    I = Int[]; J = Int[]; V = Float64[]

    id(r, c) = (r - 1) * m + c  # 1-indexed

    function coeff(r, c)
        rr = div(r - 1, tile)
        cc = div(c - 1, tile)
        return ((rr + cc) & 1 == 1) ? a_low : a_high
    end

    function hmean(a, b)
        return 2.0 * a * b / (a + b)
    end

    for r in 1:n, c in 1:m
        u = id(r, c)
        au = coeff(r, c)
        for (dr, dc) in [(-1,0),(1,0),(0,-1),(0,1)]
            nr, nc = r + dr, c + dc
            if 1 <= nr <= n && 1 <= nc <= m
                v = id(nr, nc)
                w = hmean(au, coeff(nr, nc))
                push!(I, u); push!(J, v); push!(V, w)
            end
        end
    end

    return sparse(I, J, V, N, N)
end

function grid_graph_adj(n, m)
    return grid_graph_checkerboard_adj(n, m, 1.0, 1.0, 1)
end

function erdos_renyi_adj(n, p, seed)
    rng = MersenneTwister(seed)
    I = Int[]; J = Int[]; V = Float64[]
    for i in 1:n, j in (i+1):n
        if rand(rng) < p
            push!(I, i); push!(J, j); push!(V, 1.0)
            push!(I, j); push!(J, i); push!(V, 1.0)
        end
    end
    return sparse(I, J, V, n, n)
end

function load_mtx_adj(path)
    # Simple MTX loader for coordinate format
    open(path) do io
        # Header
        header = readline(io)
        header_lower = lowercase(header)
        is_symmetric = occursin("symmetric", header_lower)
        is_pattern = occursin("pattern", header_lower)
        if occursin("array", header_lower)
            error("Array (dense) format not supported: $path")
        end

        # Skip comments
        line = ""
        while !eof(io)
            line = readline(io)
            startswith(line, '%') || break
        end

        # Dimensions
        parts = split(line)
        rows, cols, nnz = parse(Int, parts[1]), parse(Int, parts[2]), parse(Int, parts[3])
        rows == cols || error("Non-square matrix: $path")

        I = Int[]; J = Int[]; V = Float64[]
        for _ in 1:nnz
            line = readline(io)
            parts = split(line)
            i, j = parse(Int, parts[1]), parse(Int, parts[2])
            w = is_pattern ? 1.0 : abs(parse(Float64, parts[3]))
            i == j && continue  # skip diagonal
            push!(I, i); push!(J, j); push!(V, w)
            if is_symmetric && i != j
                push!(I, j); push!(J, i); push!(V, w)
            end
        end
        return sparse(I, J, V, rows, cols)
    end
end

function load_mtx_operator(path)
    # Read a coordinate MTX file AS AN OPERATOR: every stored entry keeps its
    # published value, the diagonal included, and a `symmetric` header is
    # expanded into both triangles. The mirror image of the C++
    # load_mtx_as_operator, and the reason AC/AC2 can be shown to solve the same
    # matrix as everything else rather than assumed to.
    open(path) do io
        header = readline(io)
        header_lower = lowercase(header)
        is_symmetric = occursin("symmetric", header_lower)
        if occursin("pattern", header_lower)
            error("$path is a `pattern` file: it carries no values, so it is not an operator")
        end
        if occursin("array", header_lower)
            error("Array (dense) format not supported: $path")
        end

        line = ""
        while !eof(io)
            line = readline(io)
            startswith(line, '%') || break
        end

        parts = split(line)
        rows, cols, nnz = parse(Int, parts[1]), parse(Int, parts[2]), parse(Int, parts[3])
        rows == cols || error("Non-square matrix: $path")

        I = Int[]; J = Int[]; V = Float64[]
        sizehint!(I, is_symmetric ? 2nnz : nnz)
        sizehint!(J, is_symmetric ? 2nnz : nnz)
        sizehint!(V, is_symmetric ? 2nnz : nnz)
        for _ in 1:nnz
            parts = split(readline(io))
            i, j = parse(Int, parts[1]), parse(Int, parts[2])
            v = parse(Float64, parts[3])
            push!(I, i); push!(J, j); push!(V, v)
            if is_symmetric && i != j
                push!(I, j); push!(J, i); push!(V, v)
            end
        end
        return sparse(I, J, V, rows, cols)   # duplicates are summed, as MatrixMarket specifies
    end
end

# Structural facts about an operator, matching include/apxchol/operator_class.h:
# whether it is a singular Laplacian, and how much of its off-diagonal mass is
# positive (the quantity apxchol's lumping ceiling is expressed in).
function operator_facts(A)
    n = size(A, 1)
    d = diag(A)
    rs = A * ones(n)
    is_lap = maximum(abs.(rs)) / max(maximum(abs.(d)), 1e-30) < 1e-10
    rowsA = rowvals(A); valsA = nonzeros(A)
    pos_cnt = 0; pos_mass = 0.0; abs_mass = 0.0
    for j in 1:n, k in nzrange(A, j)
        i = rowsA[k]
        i == j && continue
        v = valsA[k]
        abs_mass += abs(v)
        if v > 0
            pos_cnt += 1; pos_mass += v
        end
    end
    return is_lap, pos_cnt, (abs_mass > 0 ? pos_mass / abs_mass : 0.0)
end

# apxchol's default ceiling on the positive off-diagonal MASS fraction
# (kDefaultLumpMassCeiling in include/apxchol/operator_class.h).
const LUMP_MASS_CEILING = 0.25

# The n/a sentinel: iterations and rel_residual = -1, which the runner's
# classify() turns into an "n/a" cell — distinct from a crash (no CSV row at
# all) and from not_converged (ran, missed tol). An unsupported combination has
# to be visible as unsupported, not as a silent failure or a missing row.
function na_result(sname, graph_name, n, nnz_A, reason)
    @printf(stderr, "[n/a] %s on %s: %s\n", sname, graph_name, reason)
    return BenchResult(sname * " [n/a]", graph_name, n, nnz_A,
                       0.0, 0.0, 0.0, -1, -1.0, 0.0, 0.0)
end

function run_approxchol_operator(A, b, graph_name, tol, maxiter; variant=:ac)
    # Solve A x = b with the AC preconditioner, on the operator EXACTLY as it
    # was handed to us, and score the residual against that same A.
    n = size(A, 1)
    nnz_A = nnz(A)
    sname = variant == :ac2 ? "AC2 [Kyng16;Jl]" : "AC [Kyng16;Jl]"
    params = variant == :ac2 ? ApproxCholParams(:deg, 5, 2, 2) : ApproxCholParams(:deg)

    is_lap, pos_cnt, pos_mass = operator_facts(A)

    # Laplacians.jl reaches the operator through its adjacency: approxchol_lap
    # takes A_adj directly, and approxchol_sddm = sddmWrapLap(approxchol_lap)
    # forms Diagonal(diag(M)) - M before augmenting with a ground node. Either
    # way a POSITIVE off-diagonal becomes a NEGATIVE edge weight, which its
    # sampler cannot draw from — the same defect our own sample_clique has.
    # apxchol repairs it by lumping (preconditioner only; PCG keeps applying the
    # true operator), but AC bundles factorization and PCG together, so lumping
    # here would make it CONVERGE ON A DIFFERENT MATRIX and then be scored
    # against this one. There is no honest way to run it: report n/a.
    if pos_cnt > 0
        # (@sprintf needs a literal format string, so the sentence is assembled
        # from one formatted head plus a fixed tail.)
        head = @sprintf("%d positive off-diagonal entries carrying %.3g%% of the off-diagonal mass",
                        pos_cnt, 100 * pos_mass)
        where = pos_mass <= LUMP_MASS_CEILING ? "under" : "over"
        reason = head * ". Laplacians.jl needs non-negative edge weights and solves the " *
                 "matrix it factorizes, so it cannot take this operator (" * where *
                 @sprintf(" apxchol's %.2f lumping ceiling", LUMP_MASS_CEILING) *
                 ", but apxchol lumps for the PRECONDITIONER only and still applies the " *
                 "true operator in PCG)."
        return na_result(sname, graph_name, n, nnz_A, reason)
    end

    pcg_its = [0]
    t_setup = 0.0
    solver = nothing
    if is_lap
        # Singular Laplacian: recover the adjacency it is the Laplacian OF
        # (A_adj = Diag(diag(A)) - A, exact, no reinterpretation) and use
        # approxchol_lap, which is Laplacian-native.
        adj = sparse(Diagonal(diag(A))) - A
        t_setup = @elapsed begin
            solver = approxchol_lap(adj; tol=tol, maxits=maxiter, params=params, pcgIts=pcg_its)
        end
    else
        # Full-rank SDDM: approxchol_sddm is Laplacians.jl's OWN published-
        # operator entry point (augment with a ground node carrying the row
        # excess, solve, project back), so the operator is solved as it stands.
        #
        # Its stopping test is relative to the AUGMENTED right-hand side
        # [b; -sum(b)] (sddmWrapLap in solverInterface.jl), not to b, so "tol on
        # the augmented system" is a weaker demand than the ‖b − Ax‖/‖b‖ every
        # other solver in the cell is held to. Rescale the request onto our basis
        # so AC is asked for the same accuracy; ‖r‖ ≤ ‖r_aug‖, so this is
        # conservative. Measured 2026-08-20: a no-op on apache2 (sum(b) ≈ 0,
        # because b = A·g and A's row sums nearly vanish) and a 1.63x tightening
        # on G3_circuit. Tolerance basis is worth being pedantic about — it once
        # produced a bogus "ParAC is 1.6x faster".
        #
        # It is NOT what limits AC on these two: both still floor above 1e-8
        # (apache2 6.5e-6, G3_circuit 1.4e-7) and are recorded as not_converged.
        # The augmentation gives the ground vertex an edge of weight dExcess[i]
        # (row i's sum) to each vertex, and neither matrix is diagonally
        # dominant — apache2 has 2 rows with a negative row sum, worst -756;
        # G3_circuit has 665473, worst -0.0031 — so those ground edges have
        # NEGATIVE weight and the augmented matrix is not the Laplacian of a
        # nonnegatively-weighted graph that approxChol assumes.
        bn = norm(b)
        baug = sqrt(bn^2 + sum(b)^2)
        tol_eff = baug > 0 ? tol * bn / baug : tol
        @printf(stderr, "[ac] SDDM wrapper: |b|=%.3g, |[b;-sum b]|=%.3g -> requesting tol %.3g so the TRUE residual meets %.3g\n", bn, baug, tol_eff, tol)
        t_setup = @elapsed begin
            solver = approxchol_sddm(A; tol=tol_eff, maxits=maxiter, params=params, pcgIts=pcg_its)
        end
    end

    t_solve = @elapsed begin
        x = solver(b)
    end

    iters = isempty(pcg_its) ? 0 : pcg_its[end]
    # Only a singular Laplacian's solution and residual are defined modulo the
    # constant vector; a full-rank operator's are not, and centring them would
    # corrupt a unique solution.
    if is_lap
        x .-= mean(x)
    end
    r = b - A * x
    if is_lap
        r .-= mean(r)
    end
    rel_res = norm(r) / max(norm(b), 1e-16)

    return BenchResult(sname, graph_name, n, nnz_A,
        t_setup, t_solve, t_setup + t_solve,
        iters, rel_res, 0.0, (t_setup + t_solve) / nnz_A * 1e6)
end

# ── Benchmark result ──────────────────────────────────

struct BenchResult
    solver_name::String
    graph_name::String
    n::Int
    nnz::Int
    setup_time::Float64
    solve_time::Float64
    total_time::Float64
    iterations::Int
    rel_residual::Float64
    fillin::Float64
    us_per_nnz::Float64
end

function print_header_csv()
    println("solver,graph,n,nnz,setup_s,solve_s,total_s,iters,rel_res,fillin,us_per_nnz")
end

function print_result_csv(r::BenchResult)
    @printf("%s,%s,%d,%d,%.6e,%.6e,%.6e,%d,%.6e,%.4f,%.4f\n",
        r.solver_name, r.graph_name, r.n, r.nnz,
        r.setup_time, r.solve_time, r.total_time,
        r.iterations, r.rel_residual, r.fillin, r.us_per_nnz)
end

function print_header_pretty()
    @printf("%-16s %-22s %-10s %-12s %-12s %-12s %-12s %-8s %-14s %-10s %-12s\n",
        "Solver", "Graph", "n", "nnz", "Setup(s)", "Solve(s)", "Total(s)",
        "Iters", "RelRes", "Fill-in", "µs/nnz")
    println(repeat('-', 136))
end

function print_result_pretty(r::BenchResult)
    @printf("%-16s %-22s %-10d %-12d %-12.4f %-12.4f %-12.4f %-8d %-14.3e %-10.2f %-12.2f\n",
        r.solver_name, r.graph_name, r.n, r.nnz,
        r.setup_time, r.solve_time, r.total_time,
        r.iterations, r.rel_residual, r.fillin, r.us_per_nnz)
end

# ── Solver runners ────────────────────────────────────

function run_approxchol(adj, L, b, graph_name, tol, maxiter; variant=:ac)
    n = size(L, 1)
    nnz_L = nnz(L)

    # Build preconditioner
    if variant == :ac2
        params = ApproxCholParams(:deg, 5, 2, 2)
        sname = "AC2 [Kyng16;Jl]"
    else
        params = ApproxCholParams(:deg)
        sname = "AC [Kyng16;Jl]"
    end

    # Laplacians.jl records the PCG iteration count via pcgIts[1] = its, which it
    # SKIPS when the array is empty -- so pre-size to [0] (not Int[]) or iters stays 0.
    pcg_its = [0]

    t_setup = @elapsed begin
        solver = approxchol_lap(adj; tol=tol, maxits=maxiter, params=params, pcgIts=pcg_its)
    end

    t_solve = @elapsed begin
        x = solver(b)
    end

    iters = isempty(pcg_its) ? 0 : pcg_its[end]
    x .-= mean(x)

    r = b - L * x
    r .-= mean(r)
    rel_res = norm(r) / max(norm(b), 1e-16)

    return BenchResult(sname, graph_name, n, nnz_L,
        t_setup, t_solve, t_setup + t_solve,
        iters, rel_res, 0.0, (t_setup + t_solve) / nnz_L * 1e6)
end

function run_cg_julia(L, b, graph_name, tol, maxiter)
    n = size(L, 1)
    nnz_L = nnz(L)

    # Pin one vertex by removing last row/col
    m = n - 1
    Lsub = L[1:m, 1:m]
    bsub = b[1:m]
    bsub = bsub .- mean(bsub)

    pcg_its = Int[]

    t = @elapsed begin
        x_sub = Laplacians.cg(Lsub, bsub; tol=tol, maxits=maxiter, pcgIts=pcg_its)
    end

    x = vcat(x_sub, 0.0)
    x .-= mean(x)
    iters = isempty(pcg_its) ? 0 : pcg_its[end]

    r = b - L * x
    r .-= mean(r)
    rel_res = norm(r) / max(norm(b), 1e-16)

    return BenchResult("CG [Julia]", graph_name, n, nnz_L,
        0.0, t, t, iters, rel_res, 0.0, t / nnz_L * 1e6)
end

function run_cholesky_julia(L, b, graph_name)
    n = size(L, 1)
    nnz_L = nnz(L)

    m = n - 1
    Lsub = L[1:m, 1:m]
    bsub = b[1:m]
    bsub = bsub .- mean(bsub)

    t_setup = @elapsed F = cholesky(Lsub)
    t_solve = @elapsed x_sub = F \ bsub

    x = vcat(x_sub, 0.0)
    x .-= mean(x)

    r = b - L * x
    r .-= mean(r)
    rel_res = norm(r) / max(norm(b), 1e-16)

    return BenchResult("Chol [Julia]", graph_name, n, nnz_L,
        t_setup, t_solve, t_setup + t_solve,
        1, rel_res, 0.0, (t_setup + t_solve) / nnz_L * 1e6)
end

# ── Main ──────────────────────────────────────────────

function main_operator(opts)
    path = opts["operator"]
    A = load_mtx_operator(path)
    graph_name = basename(path)
    n = size(A, 1)
    is_lap, pos_cnt, pos_mass = operator_facts(A)
    @printf(stderr, "Operator: %s, n=%d, nnz=%d, %s, positive_offdiag=%d (%.3g%% of mass)\n",
            graph_name, n, nnz(A),
            is_lap ? "singular Laplacian" : "full-rank (SDDM)", pos_cnt, 100 * pos_mass)

    # RHS by the same recipe as the C++ make_rhs: b = A*g for a standard normal
    # g, projected onto range(A) only when A is singular, then normalised. (The
    # RNG differs between Julia and C++, so this is the same recipe, not the
    # same vector — as it has always been for this reference.)
    rng = MersenneTwister(opts["seed"])
    b = randn(rng, n)
    b = A * b
    if is_lap
        b .-= mean(b)
    end
    bn = norm(b)
    if bn > 0
        b ./= bn
    end

    solvers_to_run = opts["solver"] == "all" ? ["ac", "ac2"] : split(opts["solver"], ",")

    opts["csv"] ? print_header_csv() : print_header_pretty()
    pf = opts["csv"] ? print_result_csv : print_result_pretty

    # JIT warmup on a tiny operator, so compilation stays out of the timings.
    warmup_L = lap(grid_graph_adj(5, 5))
    warmup_b = warmup_L * randn(MersenneTwister(0), size(warmup_L, 1))
    warmup_b .-= mean(warmup_b)
    for s in solvers_to_run
        try
            (s == "ac" || s == "ac2") &&
                run_approxchol_operator(warmup_L, warmup_b, "warmup", 1e-6, 100;
                                        variant = (s == "ac2" ? :ac2 : :ac))
        catch; end
    end

    for s in solvers_to_run
        try
            r = if s == "ac" || s == "ac2"
                run_approxchol_operator(A, b, graph_name, opts["tol"], opts["maxiter"];
                                        variant = (s == "ac2" ? :ac2 : :ac))
            else
                # The plain-CG / dense-Cholesky references ground the system by
                # dropping the last row and column, which is only valid for a
                # singular Laplacian. Rather than quietly solve a different
                # matrix, say so.
                na_result(s, graph_name, n, nnz(A),
                          "this reference grounds by dropping the last row/column, " *
                          "which is only valid for a singular Laplacian; run it via --mtx")
            end
            pf(r)
        catch e
            @printf(stderr, "[error] %s on %s: %s\n", s, graph_name, e)
        end
    end
end

function main()
    opts = parse_args(ARGS)

    # ── operator mode: solve the matrix we were handed, as it stands ──────────
    if opts["graph"] == "operator"
        return main_operator(opts)
    end

    # Build graph
    local adj, graph_name
    if opts["graph"] == "grid"
        n = opts["n"]
        adj = grid_graph_adj(n, n)
        graph_name = "grid_$(n)"
    elseif opts["graph"] == "checkerboard"
        n = opts["n"]; k = opts["kappa"]; t = opts["tile"]
        adj = grid_graph_checkerboard_adj(n, n, k, 1.0, t)
        graph_name = "checker_$(n)_k$(Int(k))_t$(t)"
    elseif opts["graph"] == "erdos"
        n = opts["n"]; p = opts["er_p"]
        adj = erdos_renyi_adj(n, p, opts["seed"])
        graph_name = "erdos_$(n)_p$(p)"
    elseif opts["graph"] == "mtx"
        adj = load_mtx_adj(opts["mtx"])
        fname = basename(opts["mtx"])
        graph_name = fname
    else
        error("Unknown graph: $(opts["graph"])")
    end

    n = size(adj, 1)
    L = lap(adj)  # Laplacians.jl: builds Laplacian from adjacency
    nnz_L = nnz(L)
    @printf(stderr, "Graph: %s, n=%d, nnz=%d\n", graph_name, n, nnz_L)

    # RHS
    rng = MersenneTwister(opts["seed"])
    b = randn(rng, n)
    b = L * b
    b .-= mean(b)
    bn = norm(b)
    if bn > 0
        b ./= bn
    end

    solvers_to_run = if opts["solver"] == "all"
        ["ac", "ac2", "cg", "chol"]
    else
        split(opts["solver"], ",")
    end

    # Print header
    if opts["csv"]
        print_header_csv()
    else
        print_header_pretty()
    end

    pf = opts["csv"] ? print_result_csv : print_result_pretty

    # JIT warmup: run each solver once on a tiny graph so the timed run
    # doesn't include compilation overhead
    warmup_adj = grid_graph_adj(5, 5)
    warmup_L = lap(warmup_adj)
    warmup_b = randn(MersenneTwister(0), size(warmup_L, 1))
    warmup_b = warmup_L * warmup_b
    warmup_b .-= mean(warmup_b)
    for s in solvers_to_run
        try
            if s == "ac"
                run_approxchol(warmup_adj, warmup_L, warmup_b, "warmup", 1e-6, 100; variant=:ac)
            elseif s == "ac2"
                run_approxchol(warmup_adj, warmup_L, warmup_b, "warmup", 1e-6, 100; variant=:ac2)
            elseif s == "cg"
                run_cg_julia(warmup_L, warmup_b, "warmup", 1e-6, 100)
            elseif s == "chol"
                run_cholesky_julia(warmup_L, warmup_b, "warmup")
            end
        catch; end
    end

    for s in solvers_to_run
        try
            r = if s == "ac"
                run_approxchol(adj, L, b, graph_name, opts["tol"], opts["maxiter"]; variant=:ac)
            elseif s == "ac2"
                run_approxchol(adj, L, b, graph_name, opts["tol"], opts["maxiter"]; variant=:ac2)
            elseif s == "cg"
                run_cg_julia(L, b, graph_name, opts["tol"], opts["maxiter"])
            elseif s == "chol"
                run_cholesky_julia(L, b, graph_name)
            else
                @warn "Unknown solver: $s"
                continue
            end
            pf(r)
        catch e
            @printf(stderr, "[error] %s on %s: %s\n", s, graph_name, e)
        end
    end
end

main()
