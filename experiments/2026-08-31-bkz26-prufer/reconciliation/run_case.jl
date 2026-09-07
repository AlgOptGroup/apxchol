using Laplacians, MatrixMarket, SparseArrays, LinearAlgebra, Random, Statistics, SHA, TOML, Sockets
include("verify_source.jl")
include("adapter.jl")
include("reference_adapter.jl")
const CJ=CastIntervention
LinearAlgebra.BLAS.set_num_threads(1)
VERSION==v"1.12.7" && Threads.nthreads()==1 || error("Pinned Julia/one thread required")
length(ARGS)==3 || error("run_case.jl MATRIX EXPECTED_SHA OUTPUT")
path,expected_sha,out=ARGS;mkpath(out)
bytes2hex(open(SHA.sha256,path))==expected_sha || error("Matrix hash mismatch")
A=SparseMatrixCSC{Float64,Int}(MatrixMarket.mmread(path));issymmetric(A)||error("Not symmetric")
adj,dv,de=Laplacians.adjValAndExcess(A);a=Laplacians.extendMatrix(adj,dv,de);dropzeros!(a)
La=Laplacians.lap(a);n=size(A,1)
n==338402 && size(a,1)==n || error("Expected unaugmented Spielman operator")
norm(A-La)/norm(A)<1e-12 || error("Operator reconstruction differs")
hasharray(a)=bytes2hex(sha256(reinterpret(UInt8,vec(a))))
function factor_hash(f)
    io=IOBuffer();for k in (:col,:colptr,:rowval,:fval,:d);write(io,reinterpret(UInt8,getfield(f,k)));end
    bytes2hex(sha256(take!(io)))
end
same(a,b)=all(getfield(a,k)==getfield(b,k) for k in (:col,:colptr,:rowval,:fval,:d))
function save(name,data)
    file=joinpath(out,name);ispath(file)&&error("Existing receipt: $file")
    open(file,"w") do io;TOML.print(io,data);end
end
rng=MersenneTwister(20260901);bs=Vector{Vector{Float64}}()
for j in 1:250
    b=randn(rng,n);b.-=mean(b);b./=norm(b);push!(bs,b)
end
rh=hasharray.(bs)
Random.seed!(42);public=Laplacians.approxChol(Laplacians.LLmatp(a))
Random.seed!(42);reference,_=CastJulia.factor(Laplacians.LLmatp(a))
same(public,reference)||error("Unchanged source/public GKS identity failed")
order=copy(reference.col)
Random.seed!(42);new,_=CJ.factor(Laplacians.LLmatp(a);fixed_order=order)
same(reference,new)||error("Default GKS changed")
Random.seed!(42);cast,_=CastJulia.factor(Laplacians.LLmatp(a);sampler=:cast1,fixed_order=order)
Random.seed!(42);newcast,_=CJ.factor(Laplacians.LLmatp(a);sampler=:cast1,fixed_order=order)
same(cast,newcast)||error("Default CAST changed")
save("preflight.toml",Dict("public_gks_identity"=>true,"default_gks_identity"=>true,"default_cast_identity"=>true,
 "matrix_sha256"=>expected_sha,"pivot_sha256"=>hasharray(order),"preflight_factor_builds"=>5,
 "operator_relative_difference"=>norm(A-La)/norm(A),"rhs_count"=>250,"rhs_hashes"=>rh))
public=nothing;reference=nothing;new=nothing;cast=nothing;newcast=nothing;GC.gc()
arms=[
 ("gks_native",(sampler=:gks,)),
 ("cast_native",(sampler=:cast1,)),
 ("cast_common_d2",(sampler=:cast1,common_d2=true)),
 ("cast_common_d2_burn",(sampler=:cast1,common_d2=true,burn_d2=true)),
 ("gks_pivot_rng",(sampler=:gks,pivot_rng=true)),
 ("cast_common_d2_pivot_rng",(sampler=:cast1,common_d2=true,pivot_rng=true)),
 ("exact_tail",(sampler=:gks,common_d2=true,exact_tail=true)),
 ("gks_native_repeat",(sampler=:gks,))]
for (position,(name,options)) in enumerate(arms)
    stars=Dict{String,Any}[]
    function observer(ordinal,total,vertex,colspace,vals,len,pivot)
        if len>=3
            push!(stars,Dict("ordinal"=>ordinal,"vertex"=>vertex,"degree"=>len,"pivot"=>pivot,
                "rows"=>[colspace[k].row for k in 1:len],"weight_bits"=>[bitstring(vals[k]) for k in 1:len]))
        end
    end
    GC.gc();Random.seed!(42);started=time_ns()
    f,ds=CJ.factor(Laplacians.LLmatp(a);fixed_order=order,observer=observer,options...)
    setup=(time_ns()-started)*1e-9;fh=factor_hash(f)
    f.col==order||error("Fixed pivot ordering changed")
    if name in ("gks_native","cast_native","gks_native_repeat")
        h=TOML.parsefile(joinpath(@__DIR__,name=="cast_native" ? "historical-cast.toml" : "historical-gks.toml"))
        fh==h["factor_sha256"]||error("Historical factor mismatch: $name")
        hasharray(order)==h["pivot_sha256"] && rh==h["rhs_hashes"]||error("Historical protocol binding mismatch")
        count(==(3),ds)==49 && maximum(ds)==3 || error("Historical degree census mismatch")
    end
    pre=b->Laplacians.LDLsolver(f,b);warmit=[0]
    wx=Laplacians.pcg(La,bs[1],pre;tol=1e-8,maxits=1000,stag_test=5,pcgIts=warmit)
    norm(A*wx-bs[1])/norm(bs[1])<=1e-8 || error("Warmup did not converge")
    save(lpad(string(position),2,'0')*"-"*name*"-factor.toml",Dict("name"=>name,"position"=>position,"factor_sha256"=>fh,
      "pivot_sha256"=>hasharray(f.col),"factor_entries"=>length(f.fval),"setup_s"=>setup,"options"=>Dict(string(k)=>string(v) for (k,v) in pairs(options)),
      "degree_counts"=>Dict(string(d)=>count(==(d),ds) for d in sort(unique(ds))),"nontrivial_stars"=>stars,
      "excluded_warmup_calls"=>1,"warmup_iterations"=>warmit[1],"warmup_original_rr"=>norm(A*wx-bs[1])/norm(bs[1])))
    iterations=Int[];residuals=Float64[];solver_residuals=Float64[];solutions=String[]
    file=joinpath(out,lpad(string(position),2,'0')*"-"*name*"-solves.toml")
    ispath(file)&&error("Existing solve file")
    open(file,"w") do io
        for j in eachindex(bs)
            its=[0];x=Laplacians.pcg(La,bs[j],pre;tol=1e-8,maxits=1000,stag_test=5,pcgIts=its)
            rr=norm(A*x-bs[j])/norm(bs[j]);sr=norm(La*x-bs[j])/norm(bs[j]);sh=hasharray(x)
            push!(iterations,its[1]);push!(residuals,rr);push!(solver_residuals,sr);push!(solutions,sh)
            TOML.print(io,Dict("rows"=>[Dict("rhs_index"=>j,"rhs_sha256"=>rh[j],"solution_sha256"=>sh,
             "iterations"=>its[1],"original_rr"=>rr,"solver_rr"=>sr)]));println(io);flush(io)
        end
    end
    if name in ("gks_native","cast_native","gks_native_repeat")
        h=TOML.parsefile(joinpath(@__DIR__,name=="cast_native" ? "historical-cast.toml" : "historical-gks.toml"))
        iterations==h["iterations"] && solutions==h["solution_sha256"] || error("Historical public solves differ")
    end
    all(isfinite,residuals)&&maximum(residuals)<=1e-8&&maximum(solver_residuals)<=1e-8||error("Nonconverged arm: $name")
    factor_hash(f)==fh||error("Factor mutated")
    save(lpad(string(position),2,'0')*"-"*name*"-summary.toml",Dict("name"=>name,"position"=>position,"solves"=>250,
      "mean_iterations"=>mean(iterations),"max_original_rr"=>maximum(residuals),"max_solver_rr"=>maximum(solver_residuals),
      "factor_sha256"=>fh,"factor_entries"=>length(f.fval),"nontrivial_pivots"=>length(stars),"max_degree"=>maximum(ds),
      "all_converged"=>true,"factor_unchanged"=>true))
    println(name," mean_iterations=",mean(iterations)," fill=",length(f.fval));flush(stdout)
end
save("COMPLETE.toml",Dict("retained_factors"=>8,"retained_solves"=>2000,"warmup_solves"=>8,"preflight_factor_builds"=>5,
 "matrix_sha256"=>expected_sha,"status"=>"complete"))
