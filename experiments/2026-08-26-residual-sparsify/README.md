# Connectivity-safe late-residual sparsification

Status: **production candidate at `p = 0.25`; local quality, interleaved timing,
and Daint 36/72-core gates passed**.

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

The indexed-pool production implementation was rerun after adding unbiased
stochastic rounding of the multiplicity sidecar: **checked 15/15 pairs** over
the same five matrices and three seeds. Every solve converged, all 15 auto arms
triggered, stored nnz was 0.8167x, and the iteration delta sum was +2 (10/15
unchanged, range -1 to +2). Logs:
`/tmp/apxchol-residual-indexed-clean-r2-20260827`.

Orkut was checked separately for three seeds. Stored nnz, setup, PCG and total
geomeans were 0.6405x, 0.8924x, 0.6684x and 0.8368x; iteration deltas were
+1, 0 and 0. Seeds 43--44 are current-session runs; seed 42 is a recovered log
and its timings were visibly noisier. Logs:
`/tmp/apxchol-orkut-auto-gate-r2-20260827`.

## Daint 36/72-core decision gate

Daint job 4545868 checked **18/18 full-solve cells**: nine matrices at T=36/72,
with every auto run bracketed by two byte-identical rollback controls. All 18
residuals were below `1e-8`. Auto / geometric rollback:

| scope | setup | elimination | PCG | one-RHS total | stored nnz |
|---|---:|---:|---:|---:|---:|
| all 18 cells | **0.8950** | **0.6281** | **0.7138** | **0.8466** | **0.8487** |
| six eligible graphs, 12 cells | **0.8504** | **0.4984** | **0.6046** | **0.7821** | **0.7819** |
| eligible T=36 | 0.8589 | 0.5001 | 0.6255 | 0.7954 | 0.7826 |
| eligible T=72 | 0.8421 | 0.4967 | 0.5844 | 0.7690 | 0.7812 |

The quality cost is bounded but real: iteration deltas sum to +7, with 11/18
unchanged and every changed cell exactly +1. Per-matrix sums over T=36/72 are
Skitter +1, coPapers +2, LiveJournal +2, YouTube +1, kron +1 and Orkut 0.
Despite that, every eligible matrix wins one-RHS total: kron 0.710, LiveJournal
0.744, coPapers 0.776, YouTube 0.816, Orkut 0.823 and Skitter 0.831. The three
no-op controls are near one (Amazon 0.988, grid 0.990, iter0040 0.997).

The 36-to-72-thread setup speedup changes only 1.099x -> 1.109x. This is a
work-removal win, not a fundamental scaling fix. Downloaded evidence:
`/tmp/apxchol-residual-segmented-daint-r2` (432/432 result checksums verified
for the combined campaign).

An exact maximum-weight forest was also tested as a possible way to recover the
quality margin at `p = 0.20`. After correcting disconnected-graph RHSs, 15/15
valid brackets showed 14 unchanged iterations and one +1, setup 1.0555x and
total 1.0413x. It captured only 0.0126 percentage points more residual weight;
keep the one-pass coarse forest. Evidence:
`/tmp/apxchol-residual-backbone-compatible-r3` and research commit `5cb107d`.

## Exact stage parallelization follow-up

The estimator above is unchanged. Three formerly serial implementation stages
have exact parallel forms:

1. Split the already weight-ordered edge stream into
   `K = min(P, floor(sqrt(edges / active)))` contiguous shards, build a local
   Kruskal forest in each shard, concatenate those forests, then run the
   original global Kruskal scan. A locally rejected edge has an earlier
   same-shard path, so it cannot affect the global ordered scan. The filter is
   exact, not another sampled forest.
2. Compute every `sqrt(weight)` once in parallel and retain the original serial
   summation order. The cache replaces the obsolete radix scratch allocation,
   so it adds no second O(edges) live buffer.
3. Rebuild the directed AoS residual with the same derived `K`: each worker
   owns a contiguous edge interval and a private degree row over active
   vertices. Prefixing those rows gives disjoint slab offsets. This removes
   endpoint atomics and slab sorting while preserving the serial global edge
   order byte for byte.

The first Daint implementation used endpoint atomics plus slab sorting for step
3. Job 4547191 completed in 10:20 and checked **108/108 full solves**: six
eligible matrices, T=36/72, three repetitions, with every candidate bracketed
by exact baseline factors, iterations and residuals. Candidate / geometric
baseline geomeans over all 36 cells were setup **0.9034x**, residual-sparsify
**0.6112x**, PCG 0.9891x and single-RHS total **0.9219x**. Orkut setup was
0.8060x at T=36 and 0.7760x at T=72. The one defect was Skitter at T=72:
rebuild contention made setup 1.0660x despite a 0.2883x forest stage.
Downloaded evidence: `/tmp/apxchol-residual-parallel-daint-r1-results`.

The atomic-free prefix rebuild closes that mechanism locally. A 54-record,
three-repetition T=16 gate kept factor nnz exact in every arm and measured
setup **0.9170x**, residual-sparsify **0.6104x**, forest **0.2417x**, rebuild
**0.4964x**, and peak RSS 0.9996x. Every matrix's setup geomean improved;
Skitter was 0.9860x and Orkut 0.8723x. Evidence:
`/tmp/apxchol-residual-prefix-rebuild-confirm-20260827`.

Corrected Daint job 4547244 then completed in 9:58 and checked **108/108 full
solves** with exact factors, iterations and residuals in every bracket. Across
all 36 matrix/thread/repetition cells, candidate / geometric baseline was:

| scope | setup | residual sparsify | forest | rebuild | PCG | one-RHS total |
|---|---:|---:|---:|---:|---:|---:|
| T=36 | **0.9022x** | **0.6187x** | **0.2403x** | **0.4648x** | 1.0035x | **0.9241x** |
| T=72 | **0.8873x** | **0.5489x** | **0.2439x** | **0.3815x** | 1.0002x | **0.9104x** |
| overall | **0.8947x** | **0.5828x** | **0.2421x** | **0.4211x** | 1.0019x | **0.9172x** |

The Skitter defect is closed: setup is 0.9533x at T=36 and 0.9233x at T=72;
its rebuild is 0.4261x and 0.3688x. The residual stage wins 36/36 cells; setup
and total each win 32/36. Peak-RSS ratios are not decision-grade on these lazy
segmented pools: unchanged baseline runs jump among approximately 256, 750 and
1280 MiB page-residency tiers on YouTube. The graph's own logical-byte census
and the T=16 process RSS are neutral, while the new largest explicit temporary
is only `K * active * sizeof(node_index)` (56 MiB for Orkut even at K=72; the
derived K is 20 there). Downloaded evidence:
`/tmp/apxchol-residual-parallel-daint-r2-results`.

Finally, a T=1 bracket on LiveJournal, coPapersDBLP and Orkut isolates the
square-root cache while both parallel paths collapse to serial. Its setup and
residual-stage geomeans are **0.9750x** and **0.9082x**; Orkut setup is neutral
at 1.0002x while its residual stage is 0.9472x. Evidence:
`/tmp/apxchol-residual-cache-t1-20260827`. Full tests pass 309/309. **Decision:
promote the exact stages without separate runtime knobs; the existing whole
feature rollback remains sufficient.**

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
traffic probes have been removed. **Decision: ship the default auto rule and
its rollback.** The single-RHS total win is broad and much larger than the
0-or-1 iteration cost; callers reusing one factor for many solves can disable
the feature with `APXCHOL_RESIDUAL_SPARSIFY=0`.
