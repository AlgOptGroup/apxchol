using Laplacians, SparseArrays, LinearAlgebra, Random, Statistics, Test
include("verify_source.jl")
include("adapter.jl")
const CJ = CastIntervention
same(a,b) = all(getfield(a,k) == getfield(b,k) for k in (:col,:colptr,:rowval,:fval,:d))
@testset "Exact public GKS and replay" begin
    for n in (3, 7, 20), seed in (1, 17, 42)
        r = MersenneTwister(11+n)
        a = sparse([i == j ? 0.0 : 0.1+rand(r) for i in 1:n, j in 1:n])
        a = (a+a')/2
        Random.seed!(seed); expected = Laplacians.approxChol(Laplacians.LLmatp(a))
        Random.seed!(seed); actual, _ = CJ.factor(Laplacians.LLmatp(a))
        @test same(expected,actual)
        Random.seed!(seed); replay, _ = CJ.factor(Laplacians.LLmatp(a); fixed_order=actual.col)
        @test same(actual,replay)
        Random.seed!(seed); cast, _ = CJ.factor(Laplacians.LLmatp(a); sampler=:cast1)
        Random.seed!(seed); replay, _ = CJ.factor(Laplacians.LLmatp(a); sampler=:cast1, fixed_order=cast.col)
        @test same(cast,replay)
        for factor in (actual,cast)
            direct = CJ.direct_factor(factor)
            b = randn(r,n); b .-= mean(b)
            x = Laplacians.LDLsolver(factor,b)
            @test isapprox(CJ.apply_direct(direct,b), x; rtol=2e-12, atol=2e-12)
            # Independent dense inverse of the triangular factor on zero-sum
            # vectors, with the single null direction fixed at the last pivot.
            p = [factor.col; findall(==(0),factor.d)]
            T = Matrix{Float64}(I,n,n)
            for ii in eachindex(direct.col)
                i = direct.col[ii]
                for e in direct.colptr[ii]:direct.colptr[ii+1]-1
                    T[direct.rowval[e],i] -= direct.alpha[e]
                end
            end
            D = copy(direct.diag); D[p[end]] = 1
            M = T*Diagonal(D)*T'
            xd = M\b; xd .-= mean(xd)
            @test isapprox(x,xd; rtol=2e-11,atol=2e-11)
            its = [0]
            y = Laplacians.pcg(Laplacians.lap(a),b,z->Laplacians.LDLsolver(factor,z); tol=1e-10,maxits=1000,pcgIts=its)
            @test norm(Laplacians.lap(a)*y-b)/norm(b) <= 1e-8
        end
    end
end
@testset "Exact weighted Prufer law" begin
    for w in ([1//1,1//1,1//1], [1//1,2//1,5//1], [1//1,2//1,3//1,7//1])
        n=length(w); total=sum(w); expectation=zeros(Rational{Int},n,n)
        for code in Iterators.product(ntuple(_->1:n,n-2)...)
            p = CJ.code_parents(collect(code),n)
            probability = prod(w[k]/total for k in code)
            @test p[n] == n
            @test all(1 .<= p .<= n)
            for j in 1:n-1
                k=p[j]; z = probability*w[j]*w[k]/(w[j]+w[k])
                expectation[j,j]+=z; expectation[k,k]+=z
                expectation[j,k]-=z; expectation[k,j]-=z
            end
        end
        @test expectation == Diagonal(w)-w*w'/total
    end
end
# Actual alias-path marginal smoke; exact law above does not validate alias tables.
@testset "Alias draw marginals" begin
    w=[0.1,0.7,3.0,8.0]; target=Diagonal(w)-w*w'/sum(w); E=zeros(4,4)
    Random.seed!(883)
    for _ in 1:20000
        p=CJ.cast_parents(w)
        for j in 1:3
            k=p[j]; z=w[j]*w[k]/(w[j]+w[k])
            E[j,j]+=z;E[k,k]+=z;E[j,k]-=z;E[k,j]-=z
        end
    end
    @test norm(E/20000-target)/norm(target) < 0.025
end

@testset "Exact-tail and degree-two controls" begin
    for n in (3, 5, 9), seed in (1, 17, 42)
        r=MersenneTwister(seed); a=sparse([i==j ? 0.0 : 0.1+rand(r) for i in 1:n,j in 1:n]);a=(a+a')/2
        Random.seed!(seed); order,_=CJ.factor(Laplacians.LLmatp(a))
        Random.seed!(seed); f,ds=CJ.factor(Laplacians.LLmatp(a);sampler=:gks,exact_tail=true,common_d2=true,fixed_order=order.col)
        b=randn(r,n);b.-=mean(b); x=Laplacians.LDLsolver(f,b)
        @test norm(Laplacians.lap(a)*x-b)/norm(b)<1e-10
        @test maximum(ds)==n-1
        # A weighted cycle only has degree-one/two pivots. Matching recurrence
        # plus stream burn must recover all original factor bits and RNG state.
        a=spdiagm(1=>collect(1.0:n-1),-1=>collect(1.0:n-1));a[1,n]=0.75;a[n,1]=0.75
        Random.seed!(seed); g,_=CJ.factor(Laplacians.LLmatp(a)); words=rand(UInt64,32)
        Random.seed!(seed); c,_=CJ.factor(Laplacians.LLmatp(a);sampler=:cast1,common_d2=true,burn_d2=true,fixed_order=g.col)
        @test same(g,c)
        @test words==rand(UInt64,32)
    end
end
