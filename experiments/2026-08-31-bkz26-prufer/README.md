# BKZ26 Algorithm 1 clique sampler embedded in apxchol

This research branch exposes an opt-in implementation of Algorithm 1 from:

> Yves Baumann, Rasmus Kyng, and Gernot Zöcklein, “VAC: A
> Volume-sampling-based Elimination Rule for Approximate Cholesky
> Factorization,” manuscript, July 28, 2026.
> [PDF](https://rasmuskyng.com/papers/BKZ26.pdf)

The implementation is deliberately labeled **“BKZ26 Algorithm 1 clique
sampler embedded in apxchol.”** It is not an implementation of the paper's
Algorithm 3 (Volume Approximate Cholesky): apxchol retains its own pivot
selection, parallel independent-set rounds, residual handoff, graph storage,
factor assembly, factor drop, and PCG solve. The production GKS sampler remains
the default.

Source-material provenance: the sampler patch was adapted from `463960d` and
the historical note/log from `ba72922`, both from
`codex/vac-prufer-bkz26`, onto current main without cherry-picking their older
base or their Python changes.

## What the paper's Algorithm 3 is

Algorithm 3, **Volume Approximate Cholesky (VAC)**, is the complete sequential
factorization wrapper around Algorithm 1:

1. draw a uniformly random permutation of all vertices;
2. eliminate the vertices one at a time in that order;
3. form the exact Cholesky column of the current pivot;
4. remove the pivot star and replace its exact Schur clique by one weighted
   spanning tree from Algorithm 1; and
5. return the resulting approximate factor and permutation.

The paper proves that Algorithm 1 is a positive-semidefinite unbiased clique
estimator and invokes the resulting expectation identity
`E[L_tilde L_tilde^T] = L` for Algorithm 3. That identity is not a
high-probability spectral approximation or a PCG-convergence guarantee.
Algorithm 1 also has a linear-work sequential realization and an
`O(n log n)`-work, `O(log n)`-depth parallel realization, but Algorithm 3's
outer pivot loop is still sequential as written.

This branch intentionally isolates the new scientific question: it substitutes
Algorithm 1 for apxchol's GKS clique sampler while holding apxchol's parallel
independent-set rounds, adaptive order, residual graph, factor assembly, drop,
and solver fixed. That makes it a cleaner A/B of the sampler than implementing
all of Algorithm 3 at once. A literal Algorithm 3 implementation could still be
useful as a small reference implementation or GPU research baseline, but the
experiments below show that its defining clique sampler is not currently a
competitive replacement for GKS in apxchol.

## Algorithm 1 and the apxchol embedding

Let a pivot have distinct active-neighbor conductances
`a[0],...,a[d-1]`, let `W = sum(a)`, and let `D` be the pivot diagonal passed
to the clique sampler. The exact Schur clique has edge conductance

```text
c(i,j) = a[i] * a[j] / D.
```

The sampler draws a Prüfer code of length `d-2`, independently choosing each
symbol with

```text
Pr[P[t] = i] = a[i] / W.
```

The decoded weighted-uniform spanning tree includes `{i,j}` with probability
`(a[i]+a[j])/W`. apxchol therefore emits a sampled edge with conductance

```text
(W / D) * a[i] * a[j] / (a[i] + a[j]).
```

Its expected conductance is `a[i]*a[j]/D`, so the sampled clique is unbiased.

### Pure graph Laplacian

For a pure Laplacian pivot, `D = W`. The emitted conductance reduces to

```text
a[i] * a[j] / (a[i] + a[j]),
```

which is BKZ26 Algorithm 1 after representing the Schur clique as a product
clique.

### SDDM extension in apxchol

For an SDDM pivot, diagonal excess acts as an implicit edge to ground, so
`D > W`. It is not a neighbor and is not a Prüfer symbol. The `W/D` factor is
the apxchol embedding's extension that preserves the correct Schur-clique
expectation `a[i]*a[j]/D`. This extension does not turn the surrounding
factorization into BKZ26 Algorithm 3.

The implementation is
[`volume_tree_elimination`](../../include/apxchol/solver/elimination/volume_tree.h).
It canonicalizes neighbors by `(weight, vertex)`, samples the exact CDF with
binary search, and decodes in linear work. Thus this implementation has
`O(d log d)` sampling work even though the paper also describes a linear-work
realization. Non-positive or non-finite weights are outside Algorithm 1's
premise and retain apxchol's established GKS behavior.

## Relation to the earlier implementations

There have been three packaging stages, but not three different probability
laws:

- `a9ee7a5` (2026-08-21) contained the first standalone experiment,
  `ust_elim.cpp`. It already drew iid weighted Prüfer symbols and emitted
  `(W/D) a_i a_j/(a_i+a_j)`. Mathematically, this is the same BKZ26 Algorithm 1
  estimator used here.
- `463960d` put that sampler behind apxchol's eliminator seam on the first
  `codex/vac-prufer-bkz26` branch.
- this branch rebases the feature onto current main, gives it the unambiguous
  `bkz26` name, moves it into a dedicated header, and adds configuration and
  independent tests.

The 2026-08-21 prototype used the residual's incoming neighbor order directly.
The current implementation canonicalizes `(weight, vertex)` before consuming
random numbers, so a fixed seed is independent of adjacency arrival order. It
also validates the positive finite-weight premise, falls back safely to GKS
outside it, handles apxchol's SDDM extension explicitly, and tests the
exact-clique override. Those are reproducibility and integration improvements;
there is no evidence that the old prototype sampled a better distribution or
produced a better preconditioner.

## One selector across C++, CLI, and benchmark

The accepted values are exactly `gks` and `bkz26`; there is no `vac` alias.

C++:

```cpp
apxchol::factor_options opts;
opts.seed = 42;
opts.clique_sampler = "bkz26";
auto F = apxchol::factorize(A, opts);
```

CLI:

```bash
./build/apxchol A.mtx --random-rhs --seed 42 \
  --clique-sampler bkz26 --tol 1e-8
```

The standalone benchmark maps `APXCHOL_CLIQUE_SAMPLER=gks|bkz26` to that same
`factor_options::clique_sampler` field. It adds no permanent result series and
does not alter `results/cells`.

## Why local `d-1` output does not determine final fill

Above the optional exact-clique threshold, both GKS and BKZ26 emit `d-1`
clique edges for one pivot with `d` active neighbors. This does **not** imply
equal raw factor fill or equal stored fill:

- the two samplers choose different local topologies and weights;
- emitted edges merge into the residual graph and change later pivot
  neighborhoods and elimination decisions;
- raw `nnz(L)` is accumulated over that complete evolving factorization; and
- `stored_nnz` is measured only after the independent SpTRSV factor-drop
  policy is applied.

Local connectivity likewise does not establish global preconditioner quality.
Both samplers are unbiased, but their variance and downstream elimination
trajectories differ.

## Tests

`tests/test_elimination.cpp` checks the production inverse CDF against an
independent linear scan and the production linear Prüfer decoder against an
independent min-heap decoder. It also checks hard-coded seed-42 ordered output,
input-order invariance, connectivity, exact pure-Laplacian and SDDM weights,
the exact-clique override, and exhaustive four-vertex unbiasedness.

`tests/test_factorize.cpp` checks that the configured `bkz26` path is exactly
the explicit eliminator path, that omitted configuration remains GKS, and that
an unknown selector is rejected. CTest includes an actual CLI solve with
`--clique-sampler bkz26`.

## Factor-drop on/off protocol

Factor drop is downstream of clique sampling. Test it as a separate axis and
never compare samplers under different drop policies. From a benchmark build:

```bash
APXCHOL_CLIQUE_SAMPLER=gks APXCHOL_FACTOR_DROP=0 \
APXCHOL_REPORT_FILL=1 APXCHOL_DUMP_NNZ=1 \
./benchmarks/build-bkz26/benchmark --graph checkerboard --n 120 --kappa 1000000 \
  --solver apxchol_v1 --v1-configs 'bg+tree[vec_pool_aos]' \
  --threads 1 --repeat 1 --tol 1e-8

APXCHOL_CLIQUE_SAMPLER=gks APXCHOL_FACTOR_DROP=1e-4 \
APXCHOL_REPORT_FILL=1 APXCHOL_DUMP_NNZ=1 \
./benchmarks/build-bkz26/benchmark --graph checkerboard --n 120 --kappa 1000000 \
  --solver apxchol_v1 --v1-configs 'bg+tree[vec_pool_aos]' \
  --threads 1 --repeat 1 --tol 1e-8

APXCHOL_CLIQUE_SAMPLER=bkz26 APXCHOL_FACTOR_DROP=0 \
APXCHOL_REPORT_FILL=1 APXCHOL_DUMP_NNZ=1 \
./benchmarks/build-bkz26/benchmark --graph checkerboard --n 120 --kappa 1000000 \
  --solver apxchol_v1 --v1-configs 'bg+tree[vec_pool_aos]' \
  --threads 1 --repeat 1 --tol 1e-8

APXCHOL_CLIQUE_SAMPLER=bkz26 APXCHOL_FACTOR_DROP=1e-4 \
APXCHOL_REPORT_FILL=1 APXCHOL_DUMP_NNZ=1 \
./benchmarks/build-bkz26/benchmark --graph checkerboard --n 120 --kappa 1000000 \
  --solver apxchol_v1 --v1-configs 'bg+tree[vec_pool_aos]' \
  --threads 1 --repeat 1 --tol 1e-8
```

For each sampler, check that raw `nnz(L)` is unchanged between drop arms;
record `stored_nnz`, iterations, and the harness-recomputed true residual
separately. `stored_nnz` with drop enabled must not exceed its drop-off arm.
Use multiple factor seeds for any quality conclusion; this four-arm smoke only
isolates the drop policy.

## What failed in the experiments

[`historical-seed42.log`](historical-seed42.log) preserves the row payload from
the named source commit `ba72922`. That commit attributes the rows to
`a9ee7a5:experiments/2026-08-21-residual-density/ab.log` and the standalone
prototype to `759719c`. The log records a single factor seed (`42`). Its setup
times were collected under load average 12--28 and are not performance
evidence. Two invalid as-Skitter rows with component-incompatible residuals
were removed rather than presented as sampler evidence.

Even that weak first screen showed the important signal: on IPM `iter0010`, GKS
needed 41 PCG iterations and the weighted Prüfer tree needed 90; on `iter0040`
the counts were 59 and 76. Grids and com-Amazon were much closer. Both samplers
emit `d-1` edges per sampled clique, so this was not explained by a different
nominal per-pivot edge budget.

A later study on `codex/sampling-study` (`0c693db`, `86bc4d2`, `2c47ab6`)
repeated `iter0010` for factor seeds `{1,17,42,73,97}` with a fixed RHS:

| sampler | PCG iterations | approximate `nnz(L)` |
|---|---:|---:|
| GKS | 40, 42, 42, 44, 42 | 8.91M |
| BKZ26 weighted UST (`alpha=1`) | 93, 86, 92, 98, 83 | 8.85M |
| best tested product-tree tilt (`alpha=1.75`) | 56, 60, 60, 56, 60 | 8.77M |
| uniform Prüfer tree (`alpha=0`) | 3000, true residual about 4.4 | 9.62M |

The study swept exponents
`alpha = 0,.5,1,1.25,1.5,1.6,1.7,1.75,1.8,1.9,2,2.5,3` in
`q_i proportional to a_i^alpha`, plus mixtures intended to protect leverage
marginals. None matched GKS.

A current-branch CLI smoke on 2026-09-01 independently checked that rebasing and
hardening the implementation did not remove the loss. On the locally available
`iter0010` fixture, seeds `{1,17,42,73,97}` gave GKS
`41,42,43,45,42` iterations and BKZ26 `96,86,93,100,83`; every reported true
residual was below `1e-8`. This smoke used `--random-rhs`, so the generated RHS
varied with the factor seed and it does not replace the fixed-RHS historical
study. The fixture is not part of this branch, either, so these numbers remain a
local confirmation rather than a self-contained benchmark artifact.

### Diagnosed mechanism

Unbiased edge marginals, exact expected trace, and connectivity are not enough
to control the error seen by later eliminations. On seed 42, the aggregate
per-clique error in each neighbor's Schur-diagonal contribution was:

| sampler | relative L1 | relative L2 | observed trace ratio |
|---|---:|---:|---:|
| GKS | 0.0836 | 0.1031 | 1.0000 |
| BKZ26 weighted UST | 0.1299 | 0.2204 | 1.0004 |
| `alpha=2` product tree | 0.1038 | 0.1314 | 0.9999 |
| two half-weight GKS trees | 0.0544 | 0.0692 | 1.0000 |

GKS sorts by weight and gives every source one edge into its heavier suffix.
That construction controls how much Schur mass each individual neighbor
receives. A Prüfer tree instead permits that mass to concentrate randomly on a
few endpoints. The `alpha=2` tilt improved the average error but produced
observed local clique-degree errors of 374--659 times the exact contribution on
rare Horvitz--Thompson-reweighted edges. These local errors compound as the
sampled graph becomes the input to subsequent elimination rounds. This explains
why a one-clique expectation or trace check looked healthy while PCG quality
degraded.

Removing only the iid Prüfer symbol-count variance did not fix the endpoint
pairing. An exact-marginal balanced-count variant was slower and had means
91.0, 61.2, and 73.8 iterations for `alpha=1,1.75,2`, respectively, versus
90.4, 58.4, and 71.8 for the iid variants.

### Reproducibility limit

The detailed study code and memo are retained in repository history, but its
IPM fixture and raw logs were not committed. Therefore the table above is
attributed historical evidence, not a result reproducible from this branch
alone. The broad campaign below supersedes it for current cross-matrix claims;
its aggregate is committed and its scripts reproduce the run when the named
matrices are supplied independently. The opt-in implementation remains a
research branch rather than the default sampler.

## Reproducible broad-sweep protocol

The branch now includes `bkz26_quality_probe` and a checksum-guarded Daint
campaign under [`daint-broad`](daint-broad). It parses each matrix once,
generates one fixed component-compatible RHS independent of factor seed, and
runs `GKS / BKZ26 / GKS` for seeds `{1,17,42,73,97}` with identical
`APXCHOL_FACTOR_DROP=1e-4`. The denominator is eight valid inputs: the four IPM
iterates plus grid_500, G3_circuit, thermal2 and com-Amazon. Every row records
raw/stored factor size, independently recomputed true residual, iterations and
wall intervals. The two GKS arms must match exactly for all 40 matrix/seed
brackets before the 120-record campaign is accepted.

Two-tree oversampling, thinning and even-cycle cancellation are intentionally
absent from this BKZ report. They are separate clique-sampling experiments and
do not test BKZ26 Algorithm 1.

## Broad Daint result

Daint job `4571740` completed in 0:53 (one exclusive GH200 node, four isolated
72-thread Grace ranks, Clang 22 native tuning). Checked **8/8 matrices, 5/5
seeds, 120/120 records, and 40/40 exact repeated-GKS brackets**; all true
residuals were at most `1e-8`. The downloaded result manifest verifies 48/48
files under `/tmp/bkz26-broad-results-20260901`. The committed aggregate is
[`result-aggregate.tsv`](daint-broad/result-aggregate.tsv).

BKZ26 divided by the geometric mean of its two GKS brackets:

| scope | mean iterations GKS -> BKZ26 | iteration delta | raw factor | stored factor | setup | solve | one-RHS total |
|---|---:|---:|---:|---:|---:|---:|---:|
| four IPM iterates (20 pairs) | 46.35 -> 71.75 | +508 | 1.0062x | 1.0088x | 0.9404x | 1.4950x | **1.0952x** |
| four controls (20 pairs) | 43.50 -> 48.05 | +91 | 1.0175x | 1.0175x | 1.1016x | 1.1387x | **1.1129x** |
| all 40 pairs | 44.93 -> 59.90 | +599 | 1.0119x | 1.0132x | 1.0178x | 1.3047x | **1.1040x** |

The loss is broad, not an iter0010 accident: every IPM iterate has a positive
iteration delta in all five seeds; G3_circuit and thermal2 also lose every
seed. `grid_500` is near neutral. `iter0040` is the lone total-time win
(0.9254x) because its BKZ26 setup is 0.7885x despite iterations increasing
56.2 -> 73.2; it does not reverse the aggregate single-RHS verdict.

**Decision:** publish/retain BKZ26 Algorithm 1 as the requested research
branch and reference implementation, but do not replace GKS. A linear-work or
GPU realization can reduce sampler setup cost and can be embedded in a
ParAC-style schedule, yet neither changes the observed estimator-quality gap.
