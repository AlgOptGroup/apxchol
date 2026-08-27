# Connectivity-safe late-residual sparsification

Status: **production candidate at `p = 0.25`; local quality and interleaved
timing gates passed, Daint 36/72-core validation remains before merge**.

## Estimator

The feature performs at most one rebuild, immediately before the late BK
residual loop:

1. combine parallel edges exactly;
2. retain a deterministic coarse maximum-weight spanning forest, so every
   residual component remains connected in every realization;
3. retain each off-forest edge independently with a probability proportional
   to the square root of its weight, normalized to mean `p` over off-tree
   edges;
4. multiply a retained off-tree weight by the inverse of its own probability.

Thus every off-tree numerical weight is a Horvitz--Thompson estimator with the
correct expectation, while forest edges are exact. The indexed pool also
reweights the parallel-edge multiplicity by the same inverse probability, so
its multiplicity-based degree is unbiased. The default AoS pool deliberately
uses its already-shipped distinct-neighbour degree semantics.

The forest uses one 12-high-bit fp32 weight bucket pass followed by union-find.
This preserves the quality benefit of a heavy forest without paying a complete
four-pass radix sort. Retaining the forest is not optional: first-found forests
previously doubled Skitter's iterations, and unconstrained thinning can
disconnect a component.

## Residual-observed gate

There is no matrix-name or input-size rule. At a candidate BK handoff, 256
evenly spaced active adjacency lists estimate the distinct degree. Let `A` be
the active count, `E = A * average_distinct_degree / 2`, `F <= A - 1` the
forest-edge upper bound, and `p = 0.25`. The optimistic removable-edge count is

```
D = (1 - p) * (E - F).
```

The gate estimates at least `(A - handoff) / selected_vertices` remaining BK
rounds and
requires `D * rounds >= 8 * E`. This is intentionally conservative: it omits
solve-side factor savings, but the BK selector scans a sample rather than all
`E` edges per round, so the expression is a structural-work proxy rather than
a literal traffic prediction. `APXCHOL_RESIDUAL_SPARSIFY=0` is the rollback.

## Laptop quality and performance screen, T=16

Five eligible matrices, eight seeds each: **checked 40/40 pairs**. These are
ordered baseline/candidate screens rather than the final publication timing;
ratios are pairwise geomeans. At `p = 0.25`:

| metric | candidate / baseline |
|---|---:|
| stored factor nnz | **0.8148x** |
| setup | **0.9280x** |
| PCG | **0.7777x** |
| single-RHS total | **0.8854x** |

The aggregate iteration delta is exactly zero; 32/40 pairs are unchanged and
the range is -2 to +2. Per-matrix sums over eight seeds are kron -4, YouTube 0,
coPapersDBLP 0, LiveJournal +1, and Skitter +3. The more aggressive `p = 0.20`
reduces total to 0.8458x but adds six iterations in aggregate, so it is rejected
as the default. Logs:

- `/tmp/apxchol-residual-quality-r2-20260827`
- `/tmp/apxchol-residual-p25-more-seeds-r2-20260827`
- `/tmp/apxchol-residual-p20-more-seeds-r2-20260827`

A separate interleaved rollback/auto/rollback confirmation used three seeds
and all five eligible non-Orkut matrices: **checked 15/15 cells**. Each auto run
is divided by the geometric mean of its adjacent rollback runs:

| matrix | setup | PCG | total | iteration sum |
|---|---:|---:|---:|---:|
| kron | 0.8395x | 0.6206x | **0.8091x** | 0 |
| YouTube | 0.9009x | 0.8740x | **0.8919x** | 0 |
| coPapersDBLP | 0.9399x | 0.7977x | **0.8932x** | 0 |
| LiveJournal | 0.8937x | 0.7206x | **0.8323x** | 0 |
| Skitter | 0.9865x | 0.8830x | **0.9541x** | +2 |
| geomean | **0.9108x** | **0.7726x** | **0.8746x** | +2 |

Stored factor nnz is 0.8146x. Eleven of 15 iteration counts are unchanged;
the range is -2 to +2. Logs:
`/tmp/apxchol-residual-auto-bracket-r2-20260827`.

The auto gate reproduced the manually configured `p = 0.25` factors,
iterations and residuals exactly in 15/15 cells. The resulting factors also
became much shallower in seed 42: kron 1904 -> 980 levels, YouTube 1741 -> 895,
coPapersDBLP 2475 -> 1376, LiveJournal 7051 -> 3513, and Skitter 2805 -> 1023.

The smallest observed off-tree inclusion probabilities were 0.0229--0.0370,
or maximum inverse weights of 27.1--43.8. This is far below the 374--659x local
weight outliers measured in the rejected weighted-Pruefer experiment.

Orkut was checked separately for three seeds. Stored nnz, setup, PCG and total
geomeans were 0.6405x, 0.8924x, 0.6684x and 0.8368x; iteration deltas were
+1, 0 and 0. Seeds 43--44 are current-session runs; seed 42 is a recovered log
and its timings were visibly noisier. Logs:
`/tmp/apxchol-orkut-auto-gate-r2-20260827`.

## Breadth and validation

The local matrix directory contains 30 inputs. Checked 23/30 supported square
Laplacian/adjacency inputs; 7/30 were not solver-compatible (two non-SDDM and
five rectangular fixtures). The gate selected six matrices: kron, YouTube,
coPapersDBLP, LiveJournal, Skitter and Orkut. It was an exact no-op on the other
17/23. Explicit rollback controls on com-Amazon, iter0040 and grid inputs had
identical iterations and residuals.

- Release build and **306/306 tests pass**.
- Typed graph-level tests cover both indexed and AoS pools, deterministic
  connectivity, retention of heavy forest edges, and numerical off-tree
  expectation over 1024 seeds.
- A separate indexed-pool test covers multiplicity expectation.
- Gate tests cover representative positive and negative observations.

The production patch has one shared implementation for indexed and AoS pooled
storage; the experimental trigger/probability/forest/sample controls and the
traffic probes have been removed. Before integration, bracket the auto gate
against rollback on Grace at 36 and 72 cores. Keep only the default auto rule
and its rollback.
