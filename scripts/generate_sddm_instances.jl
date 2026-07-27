#!/usr/bin/env julia
# generate_sddm_instances.jl — Generate SDDM2023 benchmark instances as MTX files
#
# Uses the same graph generators from Laplacians.jl that the SDDM2023 paper
# (Gao, Kyng, Spielman; arXiv:2303.00709) uses for their experiments.
#
# Usage:
#   julia --project=benchmarks/julia scripts/generate_sddm_instances.jl [--outdir DIR] [--scale small|medium|large]
#
# Scale presets:
#   small:  up to ~100K nnz  (seconds to generate)
#   medium: up to ~2M nnz    (default, minutes)
#   large:  up to ~20M nnz   (may take a while)
#   full:   up to ~200M nnz  (matches paper exactly, needs ~32GB RAM)

using Laplacians
using SparseArrays
using LinearAlgebra
using MatrixMarket
using Printf
using Random

# ── Argument parsing ──────────────────────────────────

function parse_args(args)
    opts = Dict{String,String}(
        "outdir" => joinpath(dirname(dirname(@__FILE__)), "data", "sddm2023"),
        "scale"  => "medium",
    )
    i = 1
    while i <= length(args)
        if args[i] == "--outdir" && i < length(args)
            i += 1; opts["outdir"] = args[i]
        elseif args[i] == "--scale" && i < length(args)
            i += 1; opts["scale"] = args[i]
        end
        i += 1
    end
    return opts
end

# ── File writing ──────────────────────────────────────

function save_mtx(path::String, M::SparseMatrixCSC)
    mkpath(dirname(path))
    if isfile(path)
        n_existing = 0
        open(path) do io
            for line in eachline(io)
                startswith(line, '%') && continue
                parts = split(line)
                if length(parts) >= 2
                    n_existing = parse(Int, parts[1])
                    break
                end
            end
        end
        if n_existing == size(M, 1)
            @printf("  [skip] %s (already exists, n=%d)\n", basename(path), n_existing)
            return
        end
    end
    MatrixMarket.mmwrite(path, M)
    @printf("  [ok]   %s  n=%d  nnz=%d\n", basename(path), size(M, 1), nnz(M))
end

function save_adj_as_laplacian(path::String, a::SparseMatrixCSC)
    L = lap(a)
    save_mtx(path, L)
end

# ── Instance definitions ──────────────────────────────

function generate_uniform_grid(outdir, nnz_targets)
    println("\n=== Uniform Grid (Poisson) ===")
    for nz in nnz_targets
        M = uniform_grid_sddm(Int(nz))
        name = @sprintf("uniform_grid_nnz%s.mtx", format_nnz(nz))
        save_mtx(joinpath(outdir, name), M)
    end
end

function generate_checkered(outdir, nnz_target, blocks, weight)
    println("\n=== Checkered Grid (high contrast, w=$(weight)) ===")
    for b in blocks
        try
            M = checkered_grid_sddm(Int(nnz_target), b, b, b, weight)
            name = @sprintf("checkered_nnz%s_b%d_w%.0e.mtx", format_nnz(nnz_target), b, weight)
            save_mtx(joinpath(outdir, name), M)
        catch e
            println("  [skip] block=$b: grid too small ($e)")
        end
    end
end

function generate_aniso(outdir, nnz_target, stretches)
    println("\n=== Anisotropic Grid ===")
    for s in stretches
        try
            M = aniso_grid_sddm(Int(nnz_target), s)
            name = @sprintf("aniso_nnz%s_s%.0e.mtx", format_nnz(nnz_target), s)
            save_mtx(joinpath(outdir, name), M)
        catch e
            println("  [skip] stretch=$s: $e")
        end
    end
end

function generate_wgrid(outdir, nnz_target, weights)
    println("\n=== Weighted Grid ===")
    for w in weights
        try
            M = wgrid_sddm(Int(nnz_target), w)
            name = @sprintf("wgrid_nnz%s_w%.0e.mtx", format_nnz(nnz_target), w)
            save_mtx(joinpath(outdir, name), M)
        catch e
            println("  [skip] weight=$w: $e")
        end
    end
end

function generate_chimera(outdir, sizes, num_instances)
    println("\n=== Chimera Graphs (unweighted) ===")
    for n in sizes
        for idx in 1:num_instances
            a = chimera(Int(n), idx)
            name = @sprintf("chimera_%d_i%d.mtx", Int(n), idx)
            save_adj_as_laplacian(joinpath(outdir, name), a)
        end
    end
end

function generate_wted_chimera(outdir, sizes, num_instances)
    println("\n=== Chimera Graphs (weighted) ===")
    for n in sizes
        for idx in 1:num_instances
            a = wtedChimera(Int(n), idx)
            name = @sprintf("wchimera_%d_i%d.mtx", Int(n), idx)
            save_adj_as_laplacian(joinpath(outdir, name), a)
        end
    end
end

function generate_star(outdir, k_values)
    println("\n=== Sachdeva Star ===")
    for k in k_values
        a = star_join(complete_graph(k), div(k, 2))
        name = @sprintf("star_k%d.mtx", k)
        save_adj_as_laplacian(joinpath(outdir, name), a)
    end
end

function format_nnz(n)
    if n >= 1e8
        return @sprintf("%.0eM", n / 1e6)
    elseif n >= 1e6
        return @sprintf("%.0fM", n / 1e6)
    elseif n >= 1e3
        return @sprintf("%.0fK", n / 1e3)
    else
        return @sprintf("%d", Int(n))
    end
end

# ── Scale presets ─────────────────────────────────────

function get_config(scale)
    if scale == "small"
        # Small instances but SAME parameter granularity as the paper.
        # Each sweep has 7 values to produce readable plots.
        return Dict(
            "uniform_nnz"      => [1e4, 3e4, 1e5, 3e5],
            "checkered_nnz"    => 1e5,
            "checkered_blocks" => [2, 4, 8, 16, 32, 64, 128],
            "checkered_weight" => 1e7,
            "aniso_nnz"        => 1e5,
            "aniso_stretches"  => [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "wgrid_nnz"        => 1e5,
            "wgrid_weights"    => [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "chimera_sizes"    => [1e3, 3e3, 1e4, 3e4],
            "chimera_instances"=> 2,
            "star_k"           => [50, 100, 150, 200, 300, 400],
        )
    elseif scale == "medium"
        return Dict(
            "uniform_nnz"      => [1e5, 1e6, 2e6],
            "checkered_nnz"    => 2e6,
            "checkered_blocks" => [2, 4, 8, 16, 32],
            "checkered_weight" => 1e7,
            "aniso_nnz"        => 2e6,
            "aniso_stretches"  => [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "wgrid_nnz"        => 2e6,
            "wgrid_weights"    => [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "chimera_sizes"    => [1e4, 1e5],
            "chimera_instances"=> 4,
            "star_k"           => [100, 200, 400, 600, 800],
        )
    elseif scale == "large"
        return Dict(
            "uniform_nnz"      => [1e5, 1e6, 2e6, 2e7],
            "checkered_nnz"    => 2e7,
            "checkered_blocks" => [2, 4, 8, 16, 32, 64, 128],
            "checkered_weight" => 1e7,
            "aniso_nnz"        => 2e7,
            "aniso_stretches"  => [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "wgrid_nnz"        => 2e7,
            "wgrid_weights"    => [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "chimera_sizes"    => [1e4, 1e5, 1e6],
            "chimera_instances"=> 8,
            "star_k"           => [100, 200, 300, 400, 500, 600, 700, 800],
        )
    elseif scale == "full"
        return Dict(
            "uniform_nnz"      => [2e6, 2e7, 2e8],
            "checkered_nnz"    => 2e8,
            "checkered_blocks" => [2, 4, 8, 16, 32, 64, 128],
            "checkered_weight" => 1e7,
            "aniso_nnz"        => 2e8,
            "aniso_stretches"  => [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "wgrid_nnz"        => 2e8,
            "wgrid_weights"    => [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "chimera_sizes"    => [1e4, 1e5, 1e6, 1e7],
            "chimera_instances"=> 8,
            "star_k"           => [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800],
        )
    else
        error("Unknown scale: $scale (use small, medium, large, full)")
    end
end

# ── Main ──────────────────────────────────────────────

function main()
    opts = parse_args(ARGS)
    outdir = opts["outdir"]
    scale = opts["scale"]

    println("SDDM2023 Instance Generator")
    println("  Output: $outdir")
    println("  Scale:  $scale")
    mkpath(outdir)

    cfg = get_config(scale)

    t0 = time()

    generate_uniform_grid(outdir, cfg["uniform_nnz"])
    generate_checkered(outdir, cfg["checkered_nnz"], cfg["checkered_blocks"], cfg["checkered_weight"])
    generate_aniso(outdir, cfg["aniso_nnz"], cfg["aniso_stretches"])
    generate_wgrid(outdir, cfg["wgrid_nnz"], cfg["wgrid_weights"])
    generate_chimera(outdir, cfg["chimera_sizes"], cfg["chimera_instances"])
    generate_wted_chimera(outdir, cfg["chimera_sizes"], cfg["chimera_instances"])
    generate_star(outdir, cfg["star_k"])

    elapsed = time() - t0
    nfiles = length(filter(f -> endswith(f, ".mtx"), readdir(outdir)))
    @printf("\nDone. Generated %d MTX files in %.1fs\n", nfiles, elapsed)
    @printf("Directory: %s\n", outdir)

    # Print summary
    total_bytes = sum(filesize(joinpath(outdir, f)) for f in readdir(outdir) if endswith(f, ".mtx"))
    @printf("Total size: %.1f MB\n", total_bytes / 1e6)
end

main()
