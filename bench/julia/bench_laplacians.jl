#!/usr/bin/env julia
# bench_laplacians.jl — Benchmark Laplacians.jl ApproxChol solver
#
# Follows the SDDM2023 protocol (arXiv:2303.00709):
#   tolerance 1e-8, measures build_time, solve_time, iterations, ‖Ax-b‖/‖b‖
#
# Usage:
#   julia --project=bench/julia bench/julia/bench_laplacians.jl [--csv] [--mtx PATH] [options]
#
# Generates the same test instances as our C++ benchmark for direct comparison.

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
        sname = "AC2(Julia)"
    else
        params = ApproxCholParams(:deg)
        sname = "AC(Julia)"
    end

    pcg_its = Int[]

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

    return BenchResult("CG(Julia)", graph_name, n, nnz_L,
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

    return BenchResult("Chol(Julia)", graph_name, n, nnz_L,
        t_setup, t_solve, t_setup + t_solve,
        1, rel_res, 0.0, (t_setup + t_solve) / nnz_L * 1e6)
end

# ── Main ──────────────────────────────────────────────

function main()
    opts = parse_args(ARGS)

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
