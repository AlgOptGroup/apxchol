# Weighted Prüfer versus GKS: findings and remaining gaps

**Weighted Prüfer is not a generally better replacement for GKS in apxchol.**
The original eight-matrix campaign increased mean PCG iterations from **44.93
to 59.90**, with only **1.3%** more stored factor entries. A controlled
Spielman example nevertheless favors Prüfer. Its complete elimination-star
census explains why these observations can coexist: the important local
weight distributions differ. Neither result establishes a universal PCG
predictor.

## What the methods do

Eliminating a vertex with incident weights `a_i`, total `A=Σa_i` and pivot
`D` creates clique conductances `c_ij=a_i a_j/D`. The sampled graph must
preserve these conductances in expectation.

- **GKS:** sort weights increasingly. Each vertex except the last independently
  chooses a later parent with probability `a_j/S_i`, where `S_i=Σ_{j>i}a_j`,
  emitting weight `a_i S_i/D`. This gives a connected `d−1`-edge tree.
- **Weighted Prüfer / CAST-1:** draw `d−2` independent Prüfer symbols with
  probabilities `a_i/A`. Selected edge `ij` has inclusion probability
  `(a_i+a_j)/A` and receives weight `c_ij` divided by that probability.
  This is the local rule studied in [VAC](https://rasmuskyng.com/papers/BKZ26.pdf),
  embedded here in apxchol—not a complete alternative VAC implementation.
- **CAST-2:** split each neighbor into two half-weight copies, sample **one
  tree on all `2d` copies**, then contract copies. This differs from both
  averaging two trees and AC2's persistent split/merge mechanism. Its larger
  edge budget requires reporting resulting fill, not just iterations.
- **Current cycle-q candidate:** put a random Hamiltonian cycle on a heavy
  suffix and independently attach lighter vertices using
  `q(i,j)=(a_i+a_j)/Σ_{l>i}(a_l+a_i)`. Choose the suffix minimizing this
  construction's relative-Frobenius variance. It uses `d` edges for `d≥3`
  (one edge at `d=2`), with no K2 coordination; it remains research code.

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

A separate **30-solve** screen on three matrices, two seeds and 72 threads
compared q directly with GKS. Pure q trees needed **2.0% more iterations**
and are set aside. Cycle-q used **3.3% fewer iterations** than cycle-GKS,
with **0.15% more stored fill**. Each cycle selected its own cutoff; this is
not a fixed-core probability-only test. The small timing differences are
not significance claims.

## Distance from a specified optimum

Here `J=E‖R(X−C)R‖²_F`, with `R=[(A/D)diag(a)]⁻¹ᐟ²`, measures whole-clique
relative variance, including degree and off-diagonal errors. The table gives
**percent above the global infimum**, allowing arbitrary connected supports,
probabilities and positive outcome-dependent weights. Each column uses its
**own matching edge budget**; tree and cycle percentages compare different
optima. Eight exact certificates cover all 1072 supports, with bound gaps
below `5.1e−9`.

| Weights | GKS, `d−1` edges | Prüfer, `d−1` edges | Cycle-q, `d` edges |
|---|---:|---:|---:|
| `1,2,3,4` |+17.1%|+34.5%|+37.0%|
| `1,2,3,4,5` |+25.4%|+41.2%|+22.2%|
| `1,1,1,1,8` |+91.2%|+44.4%|+143.9%|
| `1,1,1,8,8` |+43.8%|+59.9%|+82.4%|

The near-optimal one-hub `d`-edge law mostly uses a hub-star plus a chord,
with some hub-as-leaf outcomes. The two-band law often puts only one heavy
vertex on its cycle. Both use support-dependent weights: fixing one heavy
core and independently attaching everything else misses useful possibilities.
Exact uniform weights are a special case where a cycle is globally
Frobenius-optimal under an at-most-`d`-edge budget. None of these statements
establishes spectral or PCG optimality.

[Detailed comparisons and limits](appendix/reconciliation-details.md) ·
[Math appendix](appendix/sampling-model.md) ·
[Evidence and reproduction](reconciliation/README.md)
