# Euler: x86_64 and CUDA

ETH Euler `eu-g6` node: 2 × AMD EPYC 9554 (64 cores each) and 8 × NVIDIA
RTX 4090 (24 GB). CPU series use 64 threads on one socket; CMG is serial.
The campaign planned 783 results (27 matrices × 29 solver configurations), and
all of them ran: 541 met the 1e-8 tolerance on the original system.

Competitors are capped at 20 times apxchol's fastest setup + solve time on the
same matrix and device, and at least 30 s. A run over the cap is shown as
`≥20×`, not as a time. Other labels: `fail` (failed), `oom` (out of memory),
`nc` (did not converge) and `n/a` (input not supported).

The [Performance page](performance.md) groups solvers into families and
compares apxchol with each family's fastest variant. This page shows every
variant, how closely the variants of a family agree, and setup and solve
separately.

```{note}
ParAC runs in two ways. "Own RHS" uses ParAC's own zero-sum right-hand side,
as on Daint; "shared RHS" uses the campaign's common right-hand side. Most
`n/a` cells are ParAC: its Physics mode grounds a singular Laplacian by
dropping a row, and its Graph mode cannot solve a disconnected graph with a
single zero-sum right-hand side. CMG runs as the canonical MATLAB code and as
a serial native port of it. BoomerAMG "cut" forces very high-degree rows to
fine points (`CoarsenCutFactor` 10).
```

## CPU

Each cell is a variant's setup + solve time divided by that of apxchol's
default configuration (GKS) on the same matrix.

![Euler CPU: every variant on every matrix](performance/euler-cpu-variants.svg)

How far apart the variants of each family are, on the matrices where both
finished: the median and largest ratio of their setup + solve times.

```{include} performance/euler-cpu-agreement.md
```

Setup and solve in seconds, for apxchol and each family's fastest variant on
that matrix:

```{include} performance/euler-cpu-times.md
```

## CUDA

The GPU apxchol series use the default degree quantile 0.5, plus 0.8 and 0.2
for comparison with Daint.

![Euler CUDA: every variant on every matrix](performance/euler-gpu-variants.svg)

```{include} performance/euler-gpu-agreement.md
```

```{include} performance/euler-gpu-times.md
```

## Thread scaling

Setup + solve time from 1 to 128 threads on three matrices; 128 threads use
both sockets. Competitors are capped at 20 times apxchol's time at the same
thread count, and a cross marks a run over the cap.

![Euler: setup + solve time against thread count](performance/euler-scaling.svg)

## Provenance

[Download all results](performance/euler-20261031.json): apxchol revision
`1c4abb35`, toolchains, binary and matrix checksums, the reason for every
cell without a result, and the thread-scaling series. GPU memory is not
published: these runs recorded device-wide usage, not per-process usage.
