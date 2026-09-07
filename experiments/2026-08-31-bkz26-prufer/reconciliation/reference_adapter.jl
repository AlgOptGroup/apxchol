# Experiment-only adaptation of Laplacians.jl 1.4.1 approxChol(LLmatp).
# The GKS branch preserves its arithmetic/RNG/update ordering. CAST changes
# only the sampled residual tree; pivot column factors retain the same LDLinv
# construction. See MODEL.md for the factor identity and rooted edge reuse.
module CastJulia
using Laplacians, Random, LinearAlgebra, SparseArrays, Statistics
import Laplacians: LLmatp, LLp, LDLinv, ApproxCholPQ, ApproxCholPQElem,
    approxCholPQPop!, approxCholPQInc!, approxCholPQDec!, get_ll_col,
    compressCol!, keyMap

# Removing a prescribed active vertex leaves the same degree bookkeeping
# intact. Replay tests must recover the exact original factor and order.
function pop_given!(pq, i)
    e = pq.elems[i]
    list = keyMap(e.key, pq.split * pq.n, 2 * pq.split * pq.n + 1)
    if e.prev == 0
        pq.lists[list] == i || error("Fixed-order vertex is not active")
        pq.lists[list] = e.next
    else
        p = pq.elems[e.prev]
        pq.elems[e.prev] = ApproxCholPQElem(p.prev, e.next, p.key)
    end
    if e.next != 0
        n = pq.elems[e.next]
        pq.elems[e.next] = ApproxCholPQElem(e.prev, n.next, n.key)
    end
    pq.nitems -= 1
    return i
end

# Decode a specified Prufer code with the linear smallest-leaf algorithm.
# Rooting at terminal d permits reusing exactly the d-1 pivot-edge pairs.
function code_parents(code, d)
    degrees = ones(Int, d)
    for v in code; degrees[v] += 1; end
    heads = zeros(Int, d)
    to = zeros(Int, 2max(0, d-1)); nxt = similar(to); count = 0
    function edge(u, v)
        count += 1; to[count] = v; nxt[count] = heads[u]; heads[u] = count
        count += 1; to[count] = u; nxt[count] = heads[v]; heads[v] = count
    end
    if d > 1
        ptr = findfirst(==(1), degrees)
        leaf = ptr
        for v in code
            edge(leaf, v)
            degrees[leaf] -= 1; degrees[v] -= 1
            if degrees[v] == 1 && v < ptr
                leaf = v
            else
                ptr += 1
                while ptr <= d && degrees[ptr] != 1; ptr += 1; end
                leaf = ptr
            end
        end
        leaves = findall(==(1), degrees)
        length(leaves) == 2 || error("Malformed Prufer code")
        edge(leaves[1], leaves[2])
    end
    parent = zeros(Int, d); stack = [d]; parent[d] = d
    while !isempty(stack)
        u = pop!(stack); e = heads[u]
        while e != 0
            v = to[e]
            if parent[v] == 0
                parent[v] = u; push!(stack, v)
            end
            e = nxt[e]
        end
    end
    all(>(0), parent) || error("Disconnected sampled tree")
    return parent
end

function cast_parents(weights)
    d = length(weights)
    d <= 2 && return fill(d, d)
    mass = sum(weights)
    prob = weights .* (d / mass)
    alias = collect(1:d); small = findall(<(1), prob); large = findall(>=(1), prob)
    while !isempty(small) && !isempty(large)
        s = pop!(small); l = pop!(large)
        alias[s] = l
        prob[l] -= 1 - prob[s]
        push!(prob[l] < 1 ? small : large, l)
    end
    for i in small; prob[i] = 1; end
    for i in large; prob[i] = 1; end
    code = Vector{Int}(undef, d-2)
    for i in eachindex(code)
        bucket = rand(1:d)
        code[i] = rand() < prob[bucket] ? bucket : alias[bucket]
    end
    return code_parents(code, d)
end

function factor(a::LLmatp{Tind,Tval}; sampler=:gks, fixed_order=nothing, observer=nothing) where {Tind,Tval}
    n = a.n

    ldli = LDLinv(a)
    ldli_row_ptr = one(Tind)

    d = zeros(n)

    pq = ApproxCholPQ(a.degs)

    it = 1
    degrees = Int[]

    colspace = Array{LLp{Tind,Tval}}(undef, n)
    cumspace = Array{Tval}(undef, n)
    vals = Array{Tval}(undef, n) # will be able to delete this

    o = Base.Order.ord(isless, identity, false, Base.Order.Forward)

    @inbounds while it < n

        i = fixed_order === nothing ? approxCholPQPop!(pq) : pop_given!(pq, fixed_order[it])

        ldli.col[it] = i # conversion!
        ldli.colptr[it] = ldli_row_ptr

        it = it + 1

        len = get_ll_col(a, i, colspace)

        len = compressCol!(a, colspace, len, pq)
        len > 0 || error("Disconnected residual at pivot $i")
        push!(degrees, len)

        csum = zero(Tval)
        for ii in 1:len
            vals[ii] = colspace[ii].val
            csum = csum + colspace[ii].val
            cumspace[ii] = csum
        end
        wdeg = csum
        observer === nothing || observer(it-1, n-1, i, colspace, vals, len, csum) # BRIDGE OBSERVER

        original_rows = sampler === :cast1 ? [colspace[k].row for k in 1:len] : Tind[]
        original_weights = sampler === :cast1 ? copy(view(vals, 1:len)) : Tval[]
        parents = sampler === :cast1 ? cast_parents(original_weights) : Int[]

        colScale = one(Tval)

        for joffset in 1:(len-1)

            ll = colspace[joffset]
            w = vals[joffset] * colScale
            j = ll.row
            revj = ll.reverse

            f = w/(wdeg)

            vals[joffset] = zero(Tval)

            # kind = Laplacians.blockSample(vals,k=1)[1]
            koff = if sampler === :gks
                r = rand() * (csum - cumspace[joffset]) + cumspace[joffset]
                searchsortedfirst(cumspace,r,one(len),len,o)
            else
                parents[joffset]
            end

            k = sampler === :gks ? colspace[koff].row : original_rows[koff]

            approxCholPQInc!(pq, k)

            newEdgeVal = sampler === :gks ? f*(one(Tval)-f)*wdeg :
                original_weights[joffset] * original_weights[koff] /
                (original_weights[joffset] + original_weights[koff])

            # fix row k in col j
            revj.row = k   # dense time hog: presumably becaus of cache
            revj.val = newEdgeVal
            revj.reverse = ll

            # fix row j in col k
            khead = a.cols[k]
            a.cols[k] = ll
            ll.next = khead
            ll.reverse = revj
            ll.val = newEdgeVal
            ll.row = j


            colScale = colScale*(one(Tval)-f)
            wdeg = wdeg*(one(Tval)-f)^2

            push!(ldli.rowval,j)
            push!(ldli.fval, f)
            ldli_row_ptr = ldli_row_ptr + one(Tind)

            # push!(ops, IJop(i,j,1-f,f))  # another time suck


        end # for


        ll = colspace[len]
        w = vals[len] * colScale
        j = ll.row
        revj = ll.reverse

        if it < n
            approxCholPQDec!(pq, j)
        end

        revj.val = zero(Tval)

        push!(ldli.rowval,j)
        push!(ldli.fval, one(Tval))
        ldli_row_ptr = ldli_row_ptr + one(Tind)

        d[i] = w

    end

    ldli.colptr[it] = ldli_row_ptr

    ldli.d = d

    return ldli, degrees
end

# Same mathematical factor, direct (unscaled) triangular recurrences in fp64.
# This isolates the LDLinv representation from sampler-induced factor changes.
struct DirectFactor
    col::Vector{Int}
    colptr::Vector{Int}
    rowval::Vector{Int}
    alpha::Vector{Float64}
    diag::Vector{Float64}
end
function direct_factor(ldli)
    alpha = similar(ldli.fval); diag = copy(ldli.d)
    for ii in eachindex(ldli.col)
        i = ldli.col[ii]; lo = ldli.colptr[ii]; hi = ldli.colptr[ii+1]-1
        s = 1.0
        for e in lo:hi-1
            alpha[e] = s * ldli.fval[e]
            s *= 1-ldli.fval[e]
        end
        alpha[hi] = s
        diag[i] /= s*s
    end
    return DirectFactor(ldli.col, ldli.colptr, ldli.rowval, alpha, diag)
end
function apply_direct(f::DirectFactor, b)
    y = copy(b)
    @inbounds for ii in eachindex(f.col)
        i = f.col[ii]; yi = y[i]
        for e in f.colptr[ii]:f.colptr[ii+1]-1
            y[f.rowval[e]] += f.alpha[e] * yi
        end
    end
    @inbounds for i in eachindex(y)
        if f.diag[i] != 0; y[i] /= f.diag[i]; end
    end
    @inbounds for ii in length(f.col):-1:1
        i = f.col[ii]; yi = y[i]
        for e in f.colptr[ii+1]-1:-1:f.colptr[ii]
            yi += f.alpha[e] * y[f.rowval[e]]
        end
        y[i] = yi
    end
    y .-= mean(y)
    return y
end
end # module
