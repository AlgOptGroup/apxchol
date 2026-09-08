# Weighted Prüfer versus GKS: findings and remaining gaps

**Weighted Prüfer is not a generally better replacement for GKS in apxchol.**
The original eight-matrix campaign increased mean PCG iterations from **44.93
to 59.90**, with only **1.3%** more stored factor entries. A controlled
Spielman example nevertheless favors Prüfer. Its complete elimination-star
census explains why these observations can coexist: the important local
weight distributions differ. Neither result establishes a universal PCG
predictor.

## What the methods do

Eliminating a vertex with incident weights $a_i$, total $A=\sum_i a_i$ and pivot
$D$ creates clique conductances $c_{ij}=a_i a_j/D$. The sampled graph must
preserve these conductances in expectation.

- **GKS:** sort weights increasingly. Each vertex except the last independently
  chooses a later parent with probability $a_j/S_i$, where $S_i=\sum_{j\gt i}a_j$,
  emitting weight $a_i S_i/D$. This gives a connected $d-1$-edge tree.
- **Weighted Prüfer / CAST-1:** draw $d-2$ independent Prüfer symbols with
  probabilities $a_i/A$. Selected edge $ij$ has inclusion probability
  $(a_i+a_j)/A$ and receives weight $c_{ij}$ divided by that probability.
  This is the local rule studied in [VAC](https://rasmuskyng.com/papers/BKZ26.pdf),
  embedded here in apxchol—not a complete alternative VAC implementation.
- **CAST-2:** split each neighbor into two half-weight copies, sample **one
  tree on all $2d$ copies**, then contract copies. This differs from both
  averaging two trees and AC2's persistent split/merge mechanism. Its larger
  edge budget requires reporting resulting fill, not just iterations: the
  [historical C++ star-k50 check](reconciliation/star-k50.tsv) had mean stored
  fill **31930 versus GKS's 31901** (about **0.09%** more), not doubled fill.
  This is not a matched Julia CAST-2/AC2 comparison.
- **Current cycle-q candidate:** put a random Hamiltonian cycle on a heavy
  suffix and independently attach lighter vertices using
  $q(i,j)=(a_i+a_j)/\sum_{l\gt i}(a_l+a_i)$. Choose the suffix minimizing this
  construction's relative-Frobenius variance. It uses $d$ edges for $d\ge 3$
  (one edge at $d=2$). **Heavy-K2** instead uses GKS parent marginals and
  coordinates choices at the two heaviest receivers; these remain research rules.

## What the measurements establish

The original campaign checked **8 matrices × 5 seeds**, with GKS before/after
controls: **120/120 converged solves**. Prüfer required more iterations in
all five seeds on six matrices. Its unsynchronized timings do not support
close speed comparisons. The [historical report](appendix/historical-benchmarks.md)
preserves every table and qualification.

On **one specific Spielman `k100, step1` trajectory**, only 49 of 338401
pivots have nontrivial tree choices; all 49 have three neighbors. This is
**not a description of the whole CAST corpus**. With matched pivot order and
250 common RHSs, GKS/CAST average **7.892/7.000 iterations**. Matching
otherwise-exact degree-two arithmetic and controlling random streams retains
the CAST advantage. Exact clique updates at those 49 triangles give **one
iteration on all 250 RHSs**, at unchanged **676850 stored factor entries**.

The first three triangles carry **92.8%** of summed GKS local variance.
CAST improves those nearly uniform stars, reducing the complete trajectory's
summed variance **14.1%**. An earlier four-star sample missed them. This
explains that apparent discrepancy without claiming summed local variance
determines PCG. Crossed-order tests on two Chimera and two Spielman inputs
also show that sampler and adaptive ordering interact.

Two separate **30-solve** screens (three matrices, two seeds, 72 threads)
separated these effects. Pure q trees needed **2.0% more iterations** than
GKS; cycle-q improved **3.3%** over cycle-GKS. A later common-cut comparison
found heavy-cycle GKS improved **35.6%** over GKS, with K2 adding **4.4%**.
A full cycle only when weight ratio is at most two, otherwise GKS, needed
**37.3% more iterations** than heavy-cycle GKS. The q screen uses each
rule's own relative-variance cutoff; the K2 screen uses one older
degree-normalized cutoff. Their ratios must not be pooled. These are quick
quality screens, not statistically significant timing or causal theorems.

## Distance from a specified optimum

Here $J=\mathbb E\lVert R(X-C)R\rVert_F^2$, with $R=[(A/D)\mathrm{diag}(a)]^{-1/2}$, measures whole-clique
relative variance, including degree and off-diagonal errors. The table gives
**percent above the global infimum**, allowing arbitrary connected supports,
probabilities and positive outcome-dependent weights. Each column uses its
**own matching edge budget**; tree and cycle percentages compare different
optima. Eight exact certificates cover all 1072 supports, with bound gaps
below $5.1\times10^{-9}$.

| Weights | GKS, $d-1$ edges | Prüfer, $d-1$ edges | Heavy-K2, $d$ edges | Cycle-q, $d$ edges |
|---|---:|---:|---:|---:|
| $1,2,3,4$ |+17.1%|+34.5%|+39.4%|+37.0%|
| $1,2,3,4,5$ |+25.4%|+41.2%|+23.6%|+22.2%|
| $1,1,1,1,8$ |+91.2%|+44.4%|+196.0%|+143.9%|
| $1,1,1,8,8$ |+43.8%|+59.9%|+111.8%|+82.4%|

K2 and cycle-q use different cutoff criteria, but their core sizes coincide
on these four synthetic profiles; this is not representative matrix evidence.

The near-optimal one-hub $d$-edge law mostly uses a hub-star plus a chord,
with some hub-as-leaf outcomes. The two-band law often puts only one heavy
vertex on its cycle. Both use support-dependent weights: fixing one heavy
core and independently attaching everything else misses useful possibilities.
Exact uniform weights are a special case where a cycle is globally
Frobenius-optimal under an at-most-$d$-edge budget. None of these statements
establishes spectral or PCG optimality.

[Detailed comparisons and limits](appendix/reconciliation-details.md) ·
[Math appendix](appendix/sampling-model.md) ·
[Evidence and reproduction](reconciliation/README.md)
