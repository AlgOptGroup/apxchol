# Original weighted-Prüfer result

This branch implements the local weighted-Prüfer rule from
[VAC](https://rasmuskyng.com/papers/BKZ26.pdf) inside apxchol. It does not
reimplement the complete VAC solver. GKS remains the baseline; local formulas
are defined once in the [sampling model](sampling-model.md).

## Broad matrix evidence: BKZ26 versus GKS

Historical Daint job 4571740 used source
`6be1d2dda7ba70313e66f1a88da19804242b4dc3`, a component-compatible RHS per
matrix, and seeds 1/17/42/73/97. GKS-before/BKZ26/GKS-after brackets covered
8/8 matrices, 120/120 arm records, and 120/120 converged true residuals.
The repeated GKS counts, iterations, and residuals agreed in all 40 brackets;
this alone is not factor-identity proof.

| Matrix | Mean GKS iterations | Mean Prüfer iterations | Stored-fill ratio |
|---|---:|---:|---:|
| iter0010 |41.0|90.0|0.992|
| iter0020 |37.2|61.2|1.010|
| iter0030 |51.0|62.6|1.014|
| iter0040 |56.2|73.2|1.020|
| grid_500 |47.2|48.6|1.006|
| G3_circuit |46.2|52.6|1.012|
| thermal2 |44.6|50.6|1.009|
| com-Amazon |36.0|40.4|1.044|
| All forty pairs |44.93|59.90|1.013|

Prüfer lost every seed on six matrices. The small fill difference does not
explain away its quality loss. Four ranks advanced without synchronized
timing phases: null-arm drift exceeded 5% in 7/40 setup, 15/40 solve, and
5/40 total pairs. Timings remain exploratory; no close speed ranking is
accepted. Historical as-Skitter data were excluded because their RHS was
incompatible with its 756 components.

[Per-seed evidence](../daint-broad/result-pairs.tsv),
[aggregates](../daint-broad/result-aggregate.tsv), and
[campaign instructions](../daint-broad/README.md) preserve the denominator.

## Exponent and tiny-star studies

Changing symbol probabilities to $q_i\propto a_i^\alpha$ remains unbiased
with the corresponding inverse-inclusion weights. On iter0010, five-seed
mean iterations were GKS 42.0, ordinary Prüfer 90.4, best tested
$\alpha=1.75$ 58.4, and $\alpha=2$ 71.8. The best exponent still lost to GKS.
The [recovered table](../alpha-sweep-recovered.tsv) is summary-level quality
evidence; missing raw alpha-0/0.5 records and unstable workstation timing
prevent stronger claims.

[Degree moments](../local-degree-moments.md) explain why uniform and strongly
skewed stars differ. The older [spectral study](../tiny-star-spectral.tsv)
enumerated four-vertex outcomes but used an uncertified multistart optimizer
for its all-tree comparator. Do not confuse that numerical search with the
later exact certificates. Neither local metric is a general PCG predictor.

From the experiment root, reproduce the retained figures with:

```sh
python3 small_star_error.py > small-star-error.tsv
python3 tiny_star_spectral.py --tsv tiny-star-spectral.tsv --plot tiny-star-spectral.svg
python3 plot_alpha_sweep.py --input alpha-sweep-recovered.tsv --output alpha-sweep-iter0010.svg
```

These scripts use NumPy/SciPy/Matplotlib; the historical artifact versions
were 2.5.2/1.18.1/3.11.1. The figures and raw summaries remain tracked.
The conclusion is limited to this configuration: ordinary weighted Prüfer
was not a useful default replacement. Alias sampling could reduce reference
setup overhead but would not change its ideal distribution or iteration law.
