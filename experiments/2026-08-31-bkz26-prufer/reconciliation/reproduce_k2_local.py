#!/usr/bin/env python3
"""Exact saved-law moments and weighted saved eigenvalue summaries; no eigensolver."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,math
P=Path(__file__).resolve().parent
rows=json.loads((P/'k2-local/index.json').read_text())['rows'];assert len(rows)==12
count=0
for r in rows:
 path=P/'k2-local'/r['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
 outcomes=[json.loads(l) for l in path.read_text().splitlines()];assert len(outcomes)==r['outcomes']
 a=list(map(F,r['weights']));A=sum(a);d=len(a);means={(i,j):F(0) for i in range(d) for j in range(i+1,d)};second=F(0);prob=F(0);rho=0
 for o in outcomes:
  p=F(o['probability']);prob+=p;deg=[F(0)]*d;off=F(0);assert len(o['edges'])==d
  for i,j,w in o['edges']:
   w=F(w);assert i<j and w>0;means[i,j]+=p*w;deg[i]+=w;deg[j]+=w;off+=2*w*w/(a[i]*a[j])
  second+=p*(sum(x*x/(b*b) for x,b in zip(deg,a))+off)
  saved=max(abs(v) for v in o['relative_error_eigenvalues']);assert math.isclose(saved,o['rho'],rel_tol=1e-12);rho+=float(p)*saved;count+=1
 assert prob==1 and all(means[i,j]==a[i]*a[j]/A for i,j in means)
 J=second-(d-1);assert J==F(r['J']) and math.isclose(rho,r['mean_rho'],rel_tol=1e-12)
 cert=json.loads((P/'optimality'/r['case']/'certificate.json').read_text());assert r['lower_bound']==cert['lower_bound'] and r['upper_bound']==cert['upper_bound']
 low=100*(J/F(r['upper_bound'])-1);high=100*(J/F(r['lower_bound'])-1)
 print(r['case'],r['law'],'J',float(J),'mean rho',rho,'percent gap interval',float(low),float(high))
assert count==108
print('Checked12laws,108outcomes, exact edge means/J, and weighted saved spectral values; no new spectral computation.')
