# Broad GKS/Prüfer campaign

Historical job 4571740 used source
`6be1d2dda7ba70313e66f1a88da19804242b4dc3`, Clang 22.1.8, one exclusive
node, and four 72-core ranks. CPU directed-AoS factorization used drop
$10^{-4}$, tolerance $10^{-8}$, and at most 3000 PCG iterations.

The denominator was eight matrices (iter0010–0040, grid_500, G3_circuit,
thermal2, com-Amazon), five seeds (1/17/42/73/97), and three arms per pair:
GKS-before/Prüfer/GKS-after. The report checks 120/120 converged records and
40/40 matching repeated-GKS counts, iterations, and residuals. The
[probe](../bkz26_quality_probe.cpp) uses one common RHS per matrix, projected
per Laplacian component.

[Pair rows](result-pairs.tsv) divide Prüfer measurements by the geometric
mean of its GKS brackets; [aggregates](result-aggregate.tsv) use geometric
means of ratios and arithmetic mean iterations. `raw_nnz` precedes storage
preparation; `stored_nnz` counts actual solve storage. Setup includes factor
and solver construction.

Ranks did not synchronize measured phases. Null-arm drift exceeded 5% in
7/40 setup, 15/40 solve, and 5/40 total pairs, reaching 1.624× for solve.
Consequently these are quality results with exploratory timings.

Reproduction requires the detached source bundle, `SOURCE_COMMIT`, `SEEDS`,
input/package hashes, and external matrix paths expected by [job.sbatch](job.sbatch).
Validate the exact package and allocation with `sbatch --test-only` before
submission. The script builds/tests the source and writes `CAMPAIGN_OK` only
after its complete denominator passes. Reanalyze retained raw files with:

```sh
python3 summarize.py /path/to/campaign
```

The [historical findings](../appendix/historical-benchmarks.md) explain the
negative default decision and exclusions. Public TSVs preserve the accepted
quality summaries; they do not bundle every original native input/output.
