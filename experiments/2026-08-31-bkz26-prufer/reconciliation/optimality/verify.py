#!/usr/bin/env python3
"""Independent stdlib-only exact verifier; never imports the optimizer."""
from fractions import Fraction as F
import argparse
import hashlib
import itertools
import json
from pathlib import Path


def verify(path):
    record=json.loads(path.read_text())
    kind=record["kind"]
    relative = kind == "global_relative_frobenius_rational_certificate_v3"
    assert kind in {"global_frobenius_rational_certificate_v1", "global_frobenius_rational_certificate_v2", "global_relative_frobenius_rational_certificate_v3"}
    quantum=F(record["upper_rounding_quantum"]) if kind.endswith("v2") or relative else None
    d,m=record["d"],record["edges"]
    a=list(map(F,record["weights"]))
    D=F(record["pivot_degree"])
    s=list(map(F,record["normalization"]))
    assert len(a)==len(s)==d and D>0 and all(x>0 for x in a+s)
    edges=list(itertools.combinations(range(d),2))
    all_topologies=[]
    for T in itertools.combinations(range(len(edges)),m):
        adjacency=[[] for _ in range(d)]
        for e in T:
            u,v=edges[e]
            adjacency[u].append(v);adjacency[v].append(u)
        seen={0};todo=[0]
        while todo:
            v=todo.pop()
            for u in adjacency[v]:
                if u not in seen:
                    seen.add(u);todo.append(u)
        if len(seen)==d:
            all_topologies.append(list(T))
    assert record["topologies"]==all_topologies
    if quantum is not None:
        assert 0 < quantum*len(all_topologies) <= F(1,10**10)
        assert F(record["upper_rounding_slack_bound"])==quantum*len(all_topologies)
        assert D==sum(a)
        assert s == (a if relative else [v*(D-v)/D for v in a])
    c=[a[i]*a[j]/D for i,j in edges]
    # Rebuild Gram entries from signed endpoint intersections, independently.
    H=[]
    for u,v in edges:
        row=[]
        for x,y in edges:
            product=F(0)
            for i,sign in ((u,1),(v,-1)):
                if i==x:product+=F(sign)/s[i]
                if i==y:product-=F(sign)/s[i]
            row.append(product*product)
        H.append(row)
    norm=sum(c[i]*H[i][j]*c[j] for i in range(len(c)) for j in range(len(c)))
    if relative:
        assert norm == d-1
    p=list(map(F,record["probabilities"]))
    z=[list(map(F,row)) for row in record["edge_masses"]]
    y=list(map(F,record["dual_prices"]))
    witnesses=[list(map(F,row)) for row in record["dual_minimizers"]]
    assert len(p)==len(z)==len(witnesses)==len(all_topologies)
    assert len(y)==len(c) and sum(p)==1 and all(x>=0 for x in p)
    total=[F(0)]*len(c);upper=F(0);minima=[];active=0
    for T,pt,zt,w in zip(all_topologies,p,z,witnesses):
        assert len(zt)==len(w)==m
        if pt==0:
            assert all(x==0 for x in zt)
        else:
            assert all(x>0 for x in zt), "active sampled topology is not strictly positive"
            active+=1
            term=sum(zt[i]*H[e][f]*zt[j] for i,e in enumerate(T) for j,f in enumerate(T))/pt
            if quantum is None:
                upper+=term
            else:
                scaled=term/quantum
                upper+=((-scaled.numerator)//scaled.denominator)*(-quantum)
        for e,mass in zip(T,zt):total[e]+=mass
        gradient=[2*sum(H[e][f]*wj for f,wj in zip(T,w))-y[e] for e in T]
        assert all(wi>=0 for wi in w) and all(g>=0 for g in gradient)
        assert all(wi*g==0 for wi,g in zip(w,gradient)), "dual inner QP failed exact KKT"
        minima.append(sum(wi*H[e][f]*wj for e,wi in zip(T,w) for f,wj in zip(T,w))-sum(y[e]*wi for e,wi in zip(T,w)))
    assert total==c, "primal edge mean is not exactly unbiased"
    upper-=norm
    lam=min(minima)
    raw=lam+sum(yi*ci for yi,ci in zip(y,c))-norm
    lower=max(F(0),raw)
    assert F(record["lambda"])==lam and F(record["raw_lower_bound"])==raw
    assert F(record["upper_bound"])==upper and F(record["lower_bound"])==lower
    assert F(record["gap"])==upper-lower and 0<=lower<=upper
    return {"file":path.name,"topologies_checked":len(all_topologies),"active_positive_topologies":active,
            "lower_bound":float(lower),"upper_bound":float(upper),"gap":float(upper-lower),
            "all_checks_exact_rational":True,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("certificate",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    result=verify(args.certificate)
    result["status"]="PASS"
    with args.output.open("x") as f:
        json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
