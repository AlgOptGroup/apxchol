using Test
include("bench_laplacians.jl")
@testset "original stopping" begin
    A = sparse([2.0 0; 0 3.0]); b = [1.0, 2.0]
    budgets = Int[]
    r = checked_solve(A,b,1e-8,10,Vector{Vector{Int}}()) do tol, remaining, warm, x
        push!(budgets,remaining)
        # Two small plateaus are not evidence of irrecoverable failure.
        error = length(budgets) < 3 ? 1e-6 : 0.0
        A\b .+ error, 3
    end
    @test budgets == [10,7,4]
    @test r.iterations == 9 && r.passes == 3
    @test norm(b-A*r.x)/norm(b) <= 1e-8
    capped = checked_solve(A,b,1e-8,1,Vector{Vector{Int}}()) do tol, remaining, warm, x
        zeros(2), remaining
    end
    @test capped.iterations == 1 && capped.residual > 1e-8
    L = blockdiag(sparse([1.0 -1; -1 1]),sparse([2.0 -2; -2 2]),spzeros(1,1))
    c = operator_components(L)
    v = [10.0,12.0,-3.0,1.0,8.0]
    project_components!(v,c)
    @test v == [-1,1,-2,2,0]
    for is_lap in (true,false), variant in (:ac,:ac2)
        op = is_lap ? L : L + 0.1I
        rhs = op * collect(1.0:5.0)
        original = copy(rhs)
        Random.seed!(42)
        result = run_approxchol_operator(op,rhs,"fixture",1e-8,100,is_lap;variant=variant)
        @test result.rel_residual <= 1e-8
        @test rhs == original
        @test 0 <= result.stop_check_seconds <= result.solve_time
    end
    grid = lap(grid_graph_adj(8,8))
    rhs = grid * randn(MersenneTwister(42),size(grid,1)); rhs /= norm(rhs)
    cg = run_cg_julia(grid,rhs,"cg",1e-8,500)
    @test cg.rel_residual <= 1e-8 && cg.iterations <= 500
    for direct in (false,true)
        result = run_grounded_julia(L,L*collect(1.0:5.0),"disconnected",1e-8,100,direct)
        if direct
            @test result.rel_residual <= 1e-8 && result.iterations <= 100
        else
            # Upstream cg returns its previous bestx when an exactly solved
            # iterate trips nr < eps before updating bestx (pcg.jl). Bounded
            # corrections must report failure honestly, not claim convergence.
            @test isfinite(result.rel_residual) && result.rel_residual > 1e-8 &&
                  result.solve_passes == 8 && result.iterations <= 100
        end
    end
    zero = run_approxchol_operator(spzeros(5,5),zeros(5),"zero",1e-8,100,true)
    @test zero.iterations == 0 && zero.rel_residual == 0
end
