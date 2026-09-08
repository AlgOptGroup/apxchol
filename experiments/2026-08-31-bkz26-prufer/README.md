# Weighted Prüfer, cycles, and GKS: current findings

**Weighted Prüfer did not improve apxchol's tested default.** Across eight
matrices and five seeds, mean PCG iterations rose from 44.93 to 59.90 with
only 1.3% more stored factor entries. All 120 bracketed solves converged;
Prüfer lost every seed on six matrices. Unsynchronized timings support no
close speed ranking. [Matrix-by-matrix evidence](appendix/historical-benchmarks.md).

For positive incident weights $a_i$ and eliminated diagonal $D$ (their sum
for a Laplacian), exact elimination creates clique edges $a_i a_j/D$.
The compared rules preserve those weights in expectation:

- **GKS:** each increasingly ordered vertex independently chooses a later
  parent proportional to its weight; the result is a tree.
- **Weighted Prüfer / CAST-1:** iid weight-proportional Prüfer symbols define
  a dependent tree; edges receive inverse-inclusion weights.
- **CAST-2:** sample one tree on two half-weight copies per neighbor, then
  contract. The result may contain cycles; it differs from averaging trees
  and from AC2. Stored fill must be measured, not assumed doubled.
- **Cycle-q:** cycle on an optimized heavy suffix, independent light-parent
  probabilities proportional to $a_i+a_j$. **Heavy-K2** instead coordinates
  GKS choices at two heaviest receivers. For $d\ge3$, the cycle constructions
  use one extra edge; degree two uses its single exact edge. These remain research candidates.

Larger real pivot neighborhoods support an input-dependent explanation.
Among 106 retained original-IPM (Yves iter0010) stars (degrees 3–68), CAST improved local
relative-Frobenius variance on only three. On 170 Chimera stars (degrees
3–350), it improved 78. These stratified samples are not whole-factor error
budgets. Crossed-order full solves also show sampling and ordering interact.
Spielman supplies one narrow positive diagnostic, not a universal explanation.
[Controlled comparisons and provenance](appendix/reconciliation-details.md).

Two separate T72 screens each checked 30 solves and 42 controls:

| Comparison | Iteration change |
|---|---:|
| q tree versus GKS |+2.0%|
| cycle-q versus cycle-GKS, own relative cutoffs |−3.3%|
| heavy-GKS versus GKS, older cutoff |−35.6%|
| heavy-K2 versus heavy-GKS, same older cutoff |−4.4%|
| full cycle only at weight ratio ≤2, versus heavy-GKS |+37.3%|

The large cycle gain already exists without q or K2; it includes extra edges
and changed later stars. These quick screens do not establish timing
significance. Pure q trees are set aside; cycle-q remains promising.

[Absolute tiny-star variance and spectral comparisons](appendix/sampling-model.md)
show substantial room below all named rules, using separately certified tree
and tree-plus-one budgets. Four synthetic profiles cannot predict matrix PCG.
The mathematical model explains q's local variance guarantee and why it does
not guarantee fewer iterations.

[Evidence and reproduction](reconciliation/README.md) ·
[Original negative campaign](daint-broad/README.md)
