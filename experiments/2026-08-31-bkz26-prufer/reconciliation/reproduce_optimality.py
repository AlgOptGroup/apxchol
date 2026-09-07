#!/usr/bin/env python3
"""Verify all eight exact certificates and independently enumerate named laws.
Standard-library rational arithmetic only; no optimizer or eigensolver.
"""
from fractions import Fraction as F
from pathlib import Path
import itertools as it,json,hashlib,runpy
P=Path(__file__).resolve().parent
verify=runpy.run_path(str(P/'optimality/verify.py'))['verify']
published=json.loads((P/'optimality/comparison.json').read_text())
def score(a,law):
 d=len(a); A=sum(a);pairs=list(it.combinations(range(d),2));means={e:F(0) for e in pairs};total=F(0);pr=F(0);entries=0
 for p,edges in law:
  assert p>0 and len(edges)==len({(i,j) for i,j,w in edges})
  deg=[F(0)]*d; off=F(0);adj=[set() for _ in a]
  for i,j,w in edges:
   assert i<j and w>0
   deg[i]+=w;deg[j]+=w;off+=2*w*w/(a[i]*a[j]);means[i,j]+=p*w;entries+=1;adj[i].add(j);adj[j].add(i)
  seen={0};todo=[0]
  while todo:
   for j in adj[todo.pop()]:
    if j not in seen:seen.add(j);todo.append(j)
  assert len(seen)==d
  total+=p*(sum(x*x/(ai*ai) for x,ai in zip(deg,a))+off);pr+=p
 assert pr==1 and all(means[i,j]==a[i]*a[j]/A for i,j in pairs)
 return total-(d-1),{'outcomes':len(law),'selected_edges':entries,'edge_means':len(pairs)}

def parents(a,k,trace):
 d=len(a);A=sum(a)
 for js in it.product(*(range(i+1,d) for i in range(k))):
  p=F(1);es=[]
  for i,j in enumerate(js):
   S=sum(a[i+1:]);m=d-i-1;q=(a[i]+a[j])/(m*a[i]+S) if trace else a[j]/S
   p*=q;es.append((i,j,a[i]*a[j]/(A*q)))
  yield p,es

def prufer(a):
 d=len(a);A=sum(a);law=[]
 for code in it.product(range(d),repeat=d-2):
  p=F(1);degrees=[1]*d
  for i in code:p*=a[i]/A;degrees[i]+=1
  edges=[]
  for j in code:
   i=next(i for i in range(d) if degrees[i]==1);degrees[i]-=1;degrees[j]-=1
   i,j=sorted((i,j));edges.append((i,j,a[i]*a[j]/(a[i]+a[j])))
  i,j=[i for i in range(d) if degrees[i]==1];edges.append((i,j,a[i]*a[j]/(a[i]+a[j])))
  law.append((p,edges))
 return law

def trace_cycles(a,h):
 d=len(a);k=d-h;A=sum(a);cycles=[]
 for tail in it.permutations(range(k+1,d)):
  if tail[0]>tail[-1]:continue
  order=(k,)+tail;es=[]
  for x,y in zip(order,order[1:]+order[:1]):
   i,j=sorted((x,y));es.append((i,j,F(h-1,2)*a[i]*a[j]/A))
  cycles.append(es)
 return [(p/len(cycles),es+cy) for p,es in parents(a,k,True) for cy in cycles]

def fracrow(value,L,U):
 return {'J_exact':str(value),'J':float(value),'ratio_lower_exact':str(value/U),'ratio_upper_exact':str(value/L),'ratio_lower':float(value/U),'ratio_upper':float(value/L),'excess_lower':float(value-U),'excess_upper':float(value-L)}


counts={'certificates':0,'supports':0,'gks_outcomes':0,'cast1_outcomes':0,'trace_cutoff_laws':0,'trace_outcomes':0}
for info in published['certificates']:
 f=P/'optimality'/info['case']/'certificate.json';checked=verify(f);c=json.loads(f.read_text())
 assert checked['sha256']==info['sha256']
 a=list(map(F,c['weights']));d=len(a);A=sum(a);L=F(c['lower_bound']);U=F(c['upper_bound'])
 assert L==F(info['lower_exact']) and U==F(info['upper_exact'])
 counts['certificates']+=1;counts['supports']+=checked['topologies_checked']
 profile=published['profiles'][info['case'].rsplit('-m',1)[0]]
 if c['edges']==d-1:
  g,gc=score(a,list(parents(a,d-1,False)));cast,cc=score(a,prufer(a))
  counts['gks_outcomes']+=gc['outcomes'];counts['cast1_outcomes']+=cc['outcomes']
  assert g==F(profile['tree']['GKS']['J_exact'])
  assert cast==F(profile['tree']['weighted_Prufer_CAST1']['J_exact'])
  print(info['case'],'GKS',float(g),'CAST1',float(cast),'global interval',float(L),float(U))
 else:
  cuts=[]
  for h in range(3,d+1):
   value,cc=score(a,trace_cycles(a,h));counts['trace_cutoff_laws']+=1;counts['trace_outcomes']+=cc['outcomes']
   cuts.append((value,-h))
  value,negative_h=min(cuts)
  expected=profile['tree_plus_one']['relative_trace_cycle']
  assert value==F(expected['J_exact']) and -negative_h==expected['core_size']
  print(info['case'],'relative trace cycle',float(value),'global interval',float(L),float(U))
assert counts=={k:published['counts'][k] for k in counts}
print('Checked',json.dumps(counts,sort_keys=True))

# A separate exact counterexample within the weighted-Prufer family.
a=list(map(F,[1,1,8]));A=sum(a);counter=[]
for theta in ([F(1,10),F(1,10),F(4,5)],[F(2,21),F(2,21),F(17,21)]):
 law=[]
 for k in range(3):
  es=[(min(k,j),max(k,j),a[k]*a[j]/(A*(theta[k]+theta[j]))) for j in range(3) if j!=k]
  law.append((theta[k],es))
 value,_=score(a,law);counter.append(value)
assert counter==[F(16,81),F(7117,36100)]
assert counter[0]-counter[1]==F(1123,2924100)
print('Separate two-law/six-outcome Prufer counterexample:',*[str(x) for x in counter])
