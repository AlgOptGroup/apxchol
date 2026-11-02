# Daint: ARM64 Grace and CUDA

CSCS Daint GH200: 72-core NVIDIA Grace CPU and Hopper GPU. These are the
ARM CPU results. All 540 planned results (27 matrices × 20 solver
configurations) were run, and 507 of them met the 1e-8 tolerance.

```{note}
These results were recorded in September 2026 under older benchmark rules.
Competitors had no time cap relative to apxchol, so speedups above 20 appear
as numbers. ParAC used its own stopping rule, residual and right-hand side.
The GPU apxchol reference is GKS at degree quantile 0.8; the current default
is 0.5. The pRCHOL failures predate a heap-corruption fix in the benchmark
adapter.
```

Labels: `fail` (failed), `nc` (did not converge), `t/o` (timed out) and `n/a`
(input not supported).

## Speedup over each solver family

![Daint: speedup of apxchol over each solver family](performance/daint-families.svg)

![Daint: apxchol against the fastest other solver on each matrix](performance/daint-best-other.svg)

## ARM CPU

72 threads.

![Daint CPU: speedup of apxchol over each family on every matrix](performance/daint-cpu-families.svg)

![Daint CPU: every variant on every matrix](performance/daint-cpu-variants.svg)

```{include} performance/daint-cpu-agreement.md
```

```{include} performance/daint-cpu-times.md
```

## CUDA

![Daint CUDA: speedup of apxchol over each family on every matrix](performance/daint-gpu-families.svg)

![Daint CUDA: every variant on every matrix](performance/daint-gpu-variants.svg)

```{include} performance/daint-gpu-agreement.md
```

```{include} performance/daint-gpu-times.md
```

## Provenance

[Every result](performance/daint-20260910.csv) and
[what all results share](performance/daint-20260910.json), converted from the
published [results table](https://github.com/AlgOptGroup/apxchol/blob/390467c0dcb665a531ef5ab6bf53e2e895646900/benchmarks/daint/results.csv),
which also records each cell's source revision and toolchain.
