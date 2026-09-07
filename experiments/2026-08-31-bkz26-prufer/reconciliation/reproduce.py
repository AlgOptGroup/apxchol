#!/usr/bin/env python3
"""Reproduce the saved Spielman tables with standard-library exact arithmetic.
This validates recorded solver evidence; it does not run PCG or recompute
residuals from solution vectors (which are not included).
"""
import collections,csv,hashlib,json,math,statistics,struct,tomllib
from fractions import Fraction as F
from pathlib import Path
P=Path(__file__).resolve().parent

def expected_errors(weights):
    a=sorted(weights);A=sum(a);target={(i,j):a[i]*a[j]/A for i in range(3) for j in range(i+1,3)}
    laws=[[(a[j]/(a[1]+a[2]),{(0,j):a[0]*(a[1]+a[2])/A,(1,2):a[1]*a[2]/A}) for j in (1,2)],
          [(a[k]/A,{tuple(sorted((k,j))):a[k]*a[j]/(a[k]+a[j]) for j in range(3) if j!=k}) for k in range(3)]]
    answers=[]
    for law in laws:
        means={e:F(0) for e in target};answer=F(0);assert sum(p for p,_ in law)==1
        for p,edges in law:
            assert p>0 and len(edges)==2 and all(w>0 for w in edges.values())
            for e,w in edges.items():means[e]+=p*w
            error={e:edges.get(e,F(0))-c for e,c in target.items()}
            diagonal=[sum(w for (u,v),w in error.items() if i in (u,v)) for i in range(3)]
            answer+=p*(sum(diagonal[i]**2/a[i]**2 for i in range(3))+2*sum(w*w/(a[i]*a[j]) for (i,j),w in error.items()))
        assert means==target
        answers.append(answer)
    x,y,z=a
    assert answers==[2*x/A+(x/A)**2*(z-y)**2/(y*z),4*x*y*z/((x+y)*(x+z)*(y+z))]
    return answers

def main():
    for line in (P/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,name
    data=json.loads((P/'spielman.json').read_text());assert len(data['arms'])==8
    summary=[]
    print('Spielman: arm, mean iterations, factor entries, summed CAST/GKS local J')
    for arm in data['arms']:
        rows=arm['solves'];stars=arm['stars'];assert len(rows)==250 and len(stars)==49
        assert [r['rhs_index'] for r in rows]==list(range(1,251))
        assert arm['degree_counts']=={'1':1,'2':338351,'3':49} and arm['factor_entries']==676850
        assert arm['pivot_sha256']==data['protocol']['fixed_pivot_sha256']
        assert all(math.isfinite(r[k]) and 0<=r[k]<=1e-8 for r in rows for k in ('original_rr','solver_rr'))
        scores=[expected_errors([F(struct.unpack('>d',int(b,2).to_bytes(8,'big'))[0]) for b in s['weight_bits']]) for s in stars]
        gj=sum(s[0] for s in scores);cj=sum(s[1] for s in scores)
        result=dict(arm=arm['name'],mean_iterations=statistics.mean(r['iterations'] for r in rows),factor_entries=arm['factor_entries'],cast_over_gks_J=float(cj/gj),cast_wins=sum(c<g for g,c in scores),first_three_gks_share=float(sum(s[0] for s in scores[:3])/gj),last_39_gks_share=float(sum(s[0] for s in scores[10:])/gj))
        summary.append(result)
        print(result['arm'],result['mean_iterations'],result['factor_entries'],f"{result['cast_over_gks_J']:.9f}")
    for index,filename in [(0,'historical-gks.toml'),(1,'historical-cast.toml'),(7,'historical-gks.toml')]:
        old=tomllib.loads((P/filename).read_text());arm=data['arms'][index]
        assert arm['factor_sha256']==old['factor_sha256'] and arm['pivot_sha256']==old['pivot_sha256']
        assert [r['rhs_sha256'] for r in arm['solves']]==old['rhs_hashes']
        assert [r['iterations'] for r in arm['solves']]==old['iterations']
        assert [r['solution_sha256'] for r in arm['solves']]==old['solution_sha256']
    assert data['arms'][0]['solves']==data['arms'][7]['solves']
    assert all(r['iterations']==1 for r in data['arms'][6]['solves'])
    print('\nFour-input crossed-order table: input GG CC CG GC')
    inputs=json.loads((P/'four-inputs.json').read_text())
    assert 0<=inputs['max_original_residual']<=1e-8
    for case in inputs['inputs']:
        assert len(case['methods'])==8
        methods={m['arm']:m for m in case['methods'][:4]}
        for m in case['methods'][4:]:
            assert m['factor_sha256']==methods[m['arm']]['factor_sha256']
            assert m['mean_iterations']==methods[m['arm']]['mean_iterations']
        print(case['matrix'],*[methods[name]['mean_iterations'] for name in ('GG','CC','CG','GC')])
    star=list(csv.DictReader((P/'star-k50.tsv').open(),delimiter='\t'));assert len(star)==9
    assert all(int(r['rhs_count'])==250 and 0<=float(r['max_true_residual'])<1e-8 for r in star)
    print('\nStar k50: median of three per-factor mean iteration counts')
    for name in ('gks','cast1','cast2'):
        print(name,statistics.median(float(r['mean_iters']) for r in star if r['arm']==name))
    print('\nChecked 8 factors, 2000 saved solves, 392 triangle records, 784 exact laws, 1960 outcomes, 750 historical control rows.')
    print('First native-GKS three-triangle share:',summary[0]['first_three_gks_share'])
    print('Native-GKS last39 share:',summary[0]['last_39_gks_share'])
if __name__=='__main__':main()
