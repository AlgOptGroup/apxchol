# BKZ26 weighted-Prüfer clique sampler

This branch provides an opt-in implementation of Algorithm 1 from:

> Yves Baumann, Rasmus Kyng, and Gernot Zöcklein, “VAC: A
> Volume-sampling-based Elimination Rule for Approximate Cholesky
> Factorization,” manuscript, July 28, 2026.
> [PDF](https://rasmuskyng.com/papers/BKZ26.pdf)

It replaces only apxchol's GKS clique sampler. Pivot selection, parallel
independent-set rounds, residual storage, factor assembly/drop, and PCG remain
apxchol's. This is therefore a controlled sampler comparison, not the paper's
complete VAC algorithm. GKS remains the default.

## Sampler

For active-neighbor conductances `a[0],...,a[d-1]`, let `W=sum(a)` and let `D`
be the pivot diagonal. The exact Schur clique contains

```text
c(i,j) = a[i] * a[j] / D.
```

The sampler draws a Prüfer code of length `d-2` with iid symbols
`Pr[P[t]=i]=a[i]/W`. The decoded tree contains `{i,j}` with probability
`(a[i]+a[j])/W`, so apxchol emits

```text
(W / D) * a[i] * a[j] / (a[i] + a[j]).
```

Its expected edge conductance is `c(i,j)`. For a pure Laplacian `D=W`, this is
exactly BKZ26 Algorithm 1. The `W/D` factor is apxchol's unbiased extension to
SDDM pivots with diagonal excess.

The implementation is
[`volume_tree_elimination`](../../include/apxchol/solver/elimination/volume_tree.h)
and is selected with `factor_options::clique_sampler="bkz26"`, CLI option
`--clique-sampler bkz26`, or benchmark environment variable
`APXCHOL_CLIQUE_SAMPLER=bkz26`.

### Linear-work implementation remains open

The present reference builds one prefix CDF and performs `d-2` binary searches:
`O(d log d)` sampling work, followed by linear Prüfer decoding. The paper's
sequential `O(d)` bound can instead be realized by building one alias table in
`O(d)` and reusing it for all iid symbols. Neither that version nor a Huffman
decision tree has been implemented or timed here.

This is a legitimate implementation follow-up, especially for GPU use, but it
does not explain the measured quality loss below: BKZ setup is only 1.018x GKS
overall, whereas its solve is 1.305x. A linear sampler can remove setup overhead;
it cannot change the sampled-tree distribution or its PCG iterations.

## Broad comparison with GKS

The reproducible Daint campaign under [`daint-broad`](daint-broad) parses each
matrix once, uses one fixed component-compatible RHS, and runs
`GKS / BKZ26 / GKS` for seeds `{1,17,42,73,97}` with the same factor-drop
policy. Job `4571740` checked **8/8 matrices, 5/5 seeds, 120/120 runs, and 40/40
exact repeated-GKS brackets**; every independently recomputed true residual was
at most `1e-8`. The aggregate is
[`result-aggregate.tsv`](daint-broad/result-aggregate.tsv).

BKZ26 divided by the geometric mean of its two GKS brackets:

| scope | mean iterations GKS → BKZ26 | raw factor | setup | solve | one-RHS total |
|---|---:|---:|---:|---:|---:|
| four IPM iterates | 46.35 → 71.75 | 1.006x | 0.940x | 1.495x | **1.095x** |
| grid_500, G3_circuit, thermal2, com-Amazon | 43.50 → 48.05 | 1.018x | 1.102x | 1.139x | **1.113x** |
| all 40 pairs | 44.93 → 59.90 | 1.012x | 1.018x | 1.305x | **1.104x** |

The loss is not confined to one IPM matrix: every IPM iterate, G3_circuit, and
thermal2 had more iterations in all five seeds. `grid_500` was near neutral.
`iter0040` was the sole one-RHS time win (0.925x), caused by setup falling to
0.789x despite iterations rising from 56.2 to 73.2.

An earlier exponent sweep sampled Prüfer symbols with
`q_i proportional to a_i^alpha` and reweighted each selected edge by its exact
inclusion probability. On `iter0010` with five seeds:

| sampler | PCG iterations | approximate `nnz(L)` |
|---|---:|---:|
| GKS | 40, 42, 42, 44, 42 | 8.91M |
| BKZ26 (`alpha=1`) | 93, 86, 92, 98, 83 | 8.85M |
| best tested tilt (`alpha=1.75`) | 56, 60, 60, 56, 60 | 8.77M |

The sweep covered `alpha=0,.5,1,1.25,1.5,1.6,1.7,1.75,1.8,1.9,2,2.5,3`
and leverage-protecting mixtures. The surprising `1.75` result is real: tilting
toward heavy vertices reduces local degree error on skewed stars, but none of
the tested product-tree laws matched GKS.

## Local-error evidence

The original relative-error table was one realized seed-42 `iter0010`
factorization, not an expected-error theorem. At every encountered pivot it
compared each neighbor's sampled clique degree `d_hat[i]` with
`d*[i]=a[i](W-a[i])/D`, then aggregated

```text
relative L1 = sum |d_hat[i]-d*[i]| / sum d*[i]
relative L2 = sqrt(sum (d_hat[i]-d*[i])^2 / sum d*[i]^2).
```

The observed GKS/BKZ values were `0.0836/0.1031` and `0.1299/0.2204`.
Different samplers encounter different later stars, so these values diagnose a
full-factorization trajectory; they are not repeated samples of identical stars.

[`small_star_error.py`](small_star_error.py) supplies the missing controlled
comparison. It exactly enumerates all outcomes for small stars and checks the
L2 result against closed-form second moments:

| six-neighbor weights | GKS L1 / L2 | BKZ L1 / L2 | `alpha=1.75` L1 / L2 |
|---|---:|---:|---:|
| all 1 | .3867 / .5164 | .3858 / **.4472** | .3858 / .4472 |
| 1, 1.5, 2, 3, 5, 8 | **.2325 / .3159** | .3187 / .3960 | .2949 / .3520 |
| 1, 2, 4, 8, 16, 32 | **.1810 / .2411** | .2669 / .3501 | .2520 / .2883 |
| 1, 1, 1, 1, 100, 100 | **.0146 / .0195** | .0392 / .1360 | .0158 / .0333 |

There is no universal ordering: BKZ is better on uniform stars. Across 2,000
random eight-neighbor stars, BKZ had lower exact expected L2 on every uniform
case and 1,035/2,000 half-decade cases; with one decade of weight spread, GKS
won 1,941/2,000, and with two decades it won 1,991/2,000. This supports the
observed mechanism: GKS's sorted one-outgoing-edge construction controls local
Schur-degree error much better once neighbor weights spread, while the Prüfer
tree can concentrate mass on a few endpoints. It does not prove a global PCG
ordering.

The closed-form derivation and `q` optimization are in
[`local-degree-moments.md`](local-degree-moments.md). BKZ's `q_i proportional
to w_i` is not variance-optimal on skewed stars: the best power lies between
1.44 and 1.88 in the examples. However, even unrestricted numerical optimization
of every `q_i` remains worse than GKS on all four skewed profiles and often
creates enormous rare-edge multipliers. This points to a limitation of the
additive marginals `q_i+q_j`, not merely a poorly chosen fixed exponent.

## Verdict

Retain this branch as the requested BKZ26 Algorithm 1 reference and experiment,
but do not replace GKS. The broad convergence loss is an estimator-quality
effect, not merely an `iter0010` anomaly or the current binary-search overhead.
An alias-table implementation remains worthwhile for faithfulness and setup
performance, but it should be evaluated as such—not as a likely repair of solve
quality.
