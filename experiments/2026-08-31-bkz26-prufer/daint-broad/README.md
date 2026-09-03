# Daint broad GKS/BKZ26 campaign

This is the broad quality evidence used by the parent report. It also records
exploratory timings, but its four ranks did not synchronize measured phases,
so those timings are not acceptance-quality comparisons.

## Provenance and denominator

- Daint job `4571740`, node `nid005956`, completed 2026-09-01 in 53 seconds.
- Source commit `6be1d2dda7ba70313e66f1a88da19804242b4dc3`.
- Clang 22.1.8; build sentinel reports `331/331` tests passed.
- One exclusive node, four ranks, 72 physical cores per rank; each rank handled
  two matrices serially with `OMP_NUM_THREADS=72`.
- CPU `vec_pool_aos` factorization with `APXCHOL_FACTOR_DROP=1e-4` and a
  maximum of 3,000 PCG iterations at tolerance `1e-8`.
- Matrices: `iter0010`, `iter0020`, `iter0030`, `iter0040`, `grid_500`,
  `G3_circuit`, `thermal2`, and `com-Amazon`.
- Seeds: `1, 17, 42, 73, 97`.
- Checked: **8/8 matrices, 5/5 seeds, 120/120 arm records, 40/40 exact
  repeated-GKS brackets, 120/120 converged true residuals**.

For each matrix and seed, [`bkz26_quality_probe.cpp`](../bkz26_quality_probe.cpp)
runs `GKS-before / BKZ26 / GKS-after` against one RHS generated independently
of the factor seed. Laplacian right-hand sides are projected separately on
every connected component. The two GKS records had identical raw factor count,
stored factor count, iteration count, and true residual in all 40 brackets.

`raw_nnz` is `factor.L.nonZeros()` before SpTRSV storage preparation;
`stored_nnz` is the preconditioner's actual triangular-solve storage. `setup_s`
runs from factorization start through solver construction; `solve_s` covers the
subsequent PCG solve. Every ratio in [`result-pairs.tsv`](result-pairs.tsv) is
BKZ26 divided by the geometric mean of the two surrounding GKS measurements.
[`result-aggregate.tsv`](result-aggregate.tsv) takes geometric means of those
pair ratios and arithmetic means of iteration counts.

A post-run null audit compared the two unchanged GKS arms in all 40 brackets.
The median symmetric before/after ratios were `1.008` for setup, `1.031` for
solve, and `1.017` for total. Drift exceeded `1.05x` in 7/40, 15/40, and 5/40
respectively; maximum solve drift was `1.624x`. This does not affect exact
factor counts or iterations, but it prevents precise timing claims from this
campaign. A timing-quality rerun would synchronize arms across ranks or use an
isolated rank with its own stability gate.

The complete package and raw records remain at
`daint:/capstor/scratch/cscs/okulkov/apxchol-codex/bkz26-broad-20260901-r1`.
In the verified package, `PACKAGE.sha256` passed for every source/input file and
`RESULTS.sha256` passed for all **48/48** result files. The two committed TSVs
are LF-normalized copies of the campaign outputs.

## Reproduction

The campaign directory must contain `source.bundle`, `SOURCE_COMMIT`, `SEEDS`,
`PACKAGE.sha256`, and the scripts in this directory. On Daint, after adapting
the account and matrix paths if needed:

```bash
export APXCHOL_CAMPAIGN="$PWD"
sbatch --export=ALL,APXCHOL_CAMPAIGN="$APXCHOL_CAMPAIGN" job.sbatch
```

The batch job verifies the package, builds and tests the detached source
commit, runs the complete bracketed campaign, calls `summarize.py`, and writes
`CAMPAIGN_OK` only after the denominator above is complete. To regenerate the
summary from retained raw records:

```bash
python3 summarize.py "$APXCHOL_CAMPAIGN"
```
