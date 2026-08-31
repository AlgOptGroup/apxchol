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

## Historical seed-42 log and limits

[`historical-seed42.log`](historical-seed42.log) preserves the row payload from
the named source commit `ba72922`. That commit attributes the rows to
`a9ee7a5:experiments/2026-08-21-residual-density/ab.log` and the standalone
prototype to `759719c`. The log records a single factor seed (`42`). Its setup
times were collected under load average 12--28 and are not performance
evidence; the as-Skitter solves are invalid and remain only as raw provenance.

The historical IPM fixture under `data/yves_ipm/` was untracked. A later
five-seed memo also had no committed raw logs. **Neither the memo nor its
five-seed numbers are reproducible from this branch**, and this branch does not
present them as validation. Current claims must come from the tests and commands
run against the committed branch.
