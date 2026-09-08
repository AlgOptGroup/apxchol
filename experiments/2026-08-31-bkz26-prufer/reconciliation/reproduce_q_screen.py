#!/usr/bin/env python3
"""Reaggregate all 30 saved observations; no native benchmark or matvec."""
import json,math,statistics,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
for line in (P/'SHA256SUMS').read_text().splitlines():
 digest,name=line.split('  ',1);assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,name
x=json.loads((P/'q-screen.json').read_text());rows=x['rows']
assert len(rows)==30 and [r['index'] for r in rows]==list(range(30))
assert {r['matrix'] for r in rows}=={'grid_2000','iter0040','as-Skitter'}
assert all(r['status']=='complete' and r['returncode']==0 and r['threads']==72 and r['rhs_count']==1 for r in rows)
for r in rows:
 assert r['solve']['pass'] and r['summary']['all_pass']
 assert math.isfinite(r['solve']['true_residual']) and 0<=r['solve']['true_residual']<=1e-8
 assert r['solve']['seconds']==r['summary']['solve_total_s']
 r['metrics']={'setup':r['summary']['setup_s'],'solve':r['solve']['seconds'],'total':r['summary']['setup_s']+r['solve']['seconds'],'iterations':r['solve']['iterations'],'raw_fill':r['summary']['raw_factor_nnz'],'stored_fill':r['summary']['stored_factor_nnz'],'rss':r['summary']['max_rss_kib']}
comparisons={'tree_q_over_gks':('tree_q','gks'),'cycle_gks_over_gks':('cycle_gks','gks'),'cycle_q_over_gks':('cycle_q','gks'),'cycle_q_over_cycle_gks':('cycle_q','cycle_gks')}
metrics=x['protocol']['control_metrics'];ratios={c:{m:[] for m in metrics} for c in comparisons};controls=0
print('Matrix seed GKS-before GKS-after tree-q cycle-GKS cycle-q')
for matrix in ('grid_2000','iter0040','as-Skitter'):
 for seed in (42,314159):
  block=[r for r in rows if r['matrix']==matrix and r['seed']==seed];assert len(block)==5
  b={r['role']:r for r in block};assert set(b)=={'gks_before','gks_after','tree_q','cycle_gks','cycle_q'}
  print(matrix,seed,*[b[k]['solve']['iterations'] for k in ('gks_before','gks_after','tree_q','cycle_gks','cycle_q')])
  for m in metrics:
   before=b['gks_before']['metrics'][m];after=b['gks_after']['metrics'][m]
   assert min(before,after)>0 and max(before,after)/min(before,after)<=1.15;controls+=1
   baseline=math.sqrt(before*after)
   for c,(num,den) in comparisons.items():ratios[c][m].append(b[num]['metrics'][m]/(baseline if den=='gks' else b[den]['metrics'][m]))
assert controls==42
print('\nComparison iterations stored-fill setup solve total')
for name,values in ratios.items():
 actual={m:math.exp(statistics.mean(math.log(v) for v in vs)) for m,vs in values.items()}
 for m,v in actual.items():assert math.isclose(v,x['expected_ratios'][name][m],rel_tol=1e-12,abs_tol=1e-12)
 print(name,*[f'{actual[m]:.6f}' for m in ('iterations','stored_fill','setup','solve','total')])
print('Checked 30/30 solves, 6/6 blocks, 42/42 controls, 28/28 aggregate ratios.')
