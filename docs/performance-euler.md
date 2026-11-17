# Euler: x86_64 and CUDA

ETH Euler `eu-g6` node: 2 × AMD EPYC 9554 (64 cores each) and 8 × NVIDIA
RTX 4090 (24 GB). CPU series use 64 threads on one socket. CMG is serial.
The campaign planned 810 results (27 matrices × 30 solver configurations).
149 are inputs a solver does not accept, and 563 met the 1e-8 tolerance on the
original system.

Competitors are capped at 20 times apxchol's fastest setup + solve time on the
same matrix and device, and at least 30 s. A run over the cap is shown as
`cap`, not as a time. Other labels: `fail` (failed), `oom` (out of memory),
`nc` (did not converge) and `n/a` (input not supported).

An asterisk (`*`) marks a time whose setup includes input adaptation by the
benchmark. The benchmark finds connected components, pins one vertex per
component or slices components, so that the solver accepts a singular or
disconnected Laplacian. The `adapt_s` column of the data files gives that part of setup.
The historical headline and scaling results have empty `adapt_s` values
(not measured). The November follow-ups below report measured adaptation and
mark it with an asterisk.

The [Performance page](performance.md) groups solvers into families and
compares apxchol with each family's fastest variant. This page shows every
variant, how closely the variants of a family agree, and setup and solve
separately.

```{note}
ParAC runs in two ways. "Own RHS" uses ParAC's own zero-sum right-hand side.
"Shared RHS" uses the campaign's common right-hand side. Most
`n/a` cells are ParAC: its Physics mode grounds a singular Laplacian by
dropping a row, and its Graph mode cannot solve a disconnected graph with a
single zero-sum right-hand side. CMG runs as the canonical MATLAB code and as
a serial native port of it. BoomerAMG "cut" forces very high-degree rows to
fine points (`CoarsenCutFactor` 10).

AMGX 2.5.0 runs its shipped `PCG_CLASSICAL_V_JACOBI` configuration on every
matrix. It runs out of the RTX 4090's 24 GB during setup on iter0010 and on
the four largest social graphs; every other shipped AMGX configuration fails
the same way on iter0010.

BoomerAMG on thermal2 solves the whole operator. The harness used to solve
each connected component of this disconnected SDDM matrix separately, which
built 959 BoomerAMG hierarchies per setup. On coAuthorsDBLP, the default
BoomerAMG GPU setup time is a slow outlier: a recheck measured 0.16 s, the
same as the "cut" variant, instead of 0.32 s.
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

Setup + solve time from 1 to 128 threads on three matrices. 128 threads use
both sockets. Competitors are capped at 20 times apxchol's time at the same
thread count, and a cross marks a run over the cap.

![Euler: setup + solve time against thread count](performance/euler-scaling.svg)

## November scaling follow-ups

These two profiles use separate exclusive EPYC 9654 allocations and frozen source `7dee3a3a`, with clang 19.1.3 and x86-64-v3. Each contains 40 measurements at the original-system tolerance of 1e-8. At 64 threads, work stays on one socket. 128 threads use 96 cores on one socket and 32 on the other. Keep these profiles separate from the historical headline results above.

C++ series use one warmup and three retained runs. ParAC uses three measured runs without a warmup. The 1,800-second limit covers the whole logical cell, including preparation and repetitions. Setup includes measured input adaptation. BoomerAMG also includes its once-charged initialization.

### Grid 2000 × 2000

![Grid follow-up: setup plus solve by thread count on eu-g9-052-3](performance/euler-21174947-descriptive-scaling-scaling.svg)

```{include} performance/euler-21174947-descriptive-scaling-table.md
```

[Grid measurements](performance/euler-21174947-descriptive-scaling.csv) · [Grid provenance](performance/euler-21174947-descriptive-scaling.json)

### IPM iter0040

ParAC's 32-thread observation includes a Julia producer time of 23.26 seconds, compared with about 5 seconds at adjacent thread counts. Its cause is unknown. The diamond retains that observation while excluding it from the trend line.

![IPM follow-up: setup plus solve by thread count on eu-g9-052-4, with the preparation outlier marked separately](performance/euler-21174948-descriptive-scaling-scaling.svg)

```{include} performance/euler-21174948-descriptive-scaling-table.md
```

[IPM measurements](performance/euler-21174948-descriptive-scaling.csv) · [IPM provenance](performance/euler-21174948-descriptive-scaling.json)

## Shared-node follow-ups

These additional profiles retain 185 outcomes separately from the exclusive runs and historical rankings. The shared allocations used boost and source `7dee3a3a`. ParAC used a gcc+clang 12.2.0+19.1.3 driver with unknown ISA flags. The C++ scaling series used clang 19.1.3 with native tuning. Saved-vector original-system grades were reviewed, but the remote vector bytes were not replayed locally.

### Shared-node thread scaling

The grid and IPM profiles each contain 35 completed measurements from job 21003377 on eu-g6-038. C++ uses one warmup and three retained runs. ParAC uses three measured runs without a warmup. ParAC preparation and solving are largely serial, so nominal thread counts do not establish parallel efficiency. These allocations are distinct from the exclusive follow-ups above.

![Shared grid: setup plus solve by thread count](performance/euler-21003377-grid-2000-scaling.svg)

```{include} performance/euler-21003377-grid-2000-table.md
```

[Shared grid measurements](performance/euler-21003377-grid-2000.csv) · [Provenance](performance/euler-21003377-grid-2000.json)

![Shared IPM: setup plus solve by thread count](performance/euler-21003377-iter0040-scaling.svg)

```{include} performance/euler-21003377-iter0040-table.md
```

[Shared IPM measurements](performance/euler-21003377-iter0040.csv) · [Provenance](performance/euler-21003377-iter0040.json)

### Shared-RHS ParAC records

The two headline follow-ups retain 108 outcomes: 35 cost-eligible completions, 11 over-cap results, 58 unsupported inputs and four timeouts. Portable and MKL ran in separate shared 64-thread allocations, on eu-g6-038 and eu-g6-032 respectively. These are absolute descriptive costs. No matched apxchol ratio or portable-versus-MKL speedup is inferred. Setup here excludes the separately retained inclusive graph-serialization cost.

Caps come from the predeclared historical `ed63b8bc` table. Portable grid3d_250 exceeds its 128.44994-second cap by 0.56%. A shared-only reference would admit it. Its original over-cap classification remains visible. Over-cap timings stay blank even when the solution met tolerance.

#### Portable ParAC

```{include} performance/euler-21013867-parac-followup-table.md
```

[Portable ParAC records](performance/euler-21013867-parac-followup.csv) · [Provenance](performance/euler-21013867-parac-followup.json)

#### MKL ParAC

```{include} performance/euler-21013868-parac-followup-table.md
```

[MKL ParAC records](performance/euler-21013868-parac-followup.csv) · [Provenance](performance/euler-21013868-parac-followup.json)

The earlier Skitter profile contains seven 1,800-second logical-cell timeouts. No native-solve evidence was collected for those cells. Missing dimensions, residuals and costs remain missing.

#### Earlier Skitter timeouts

```{include} performance/euler-21003376-skitter-timeouts-table.md
```

[Earlier Skitter timeouts records](performance/euler-21003376-skitter-timeouts.csv) · [Provenance](performance/euler-21003376-skitter-timeouts.json)

## Skitter follow-up on EPYC 7742

This exclusive allocation on eu-g5-041-1 used frozen source `7dee3a3a`, clang 19.1.3 and x86-64-v3. It produced 23 completed C++ measurements at the original-system tolerance of 1e-8. Up to 64 threads stay on one socket. At 128 threads, work uses 64 cores on each socket. Keep this allocation separate from the EPYC 9654 and 9554 profiles above.

![Skitter: measured setup plus solve on EPYC 7742](performance/euler-21174263-cpp-scaling-scaling.svg)

```{include} performance/euler-21174263-cpp-scaling-table.md
```

[Measured costs](performance/euler-21174263-cpp-scaling.csv) · [Measurement provenance](performance/euler-21174263-cpp-scaling.json)

The complete scope was 40 slots. The other 17 are retained below: five ParAC preparation-stage timeouts, eleven slots not run for budget, and one BoomerAMG interruption when the allocation budget ran out. None of the five ParAC attempts reached its native solver. These outcomes establish no ParAC solve-speed comparison. An em dash means there is no solver cell receipt, with the controller's reason shown beside it.

```{include} performance/euler-21174263-status-coverage-table.md
```

[Remaining outcomes](performance/euler-21174263-status-coverage.csv) · [Outcome provenance](performance/euler-21174263-status-coverage.json)

## Additional Skitter CPU measurements

Two allocations on the EPYC 7742 filled nine previously unmeasured solver and thread-count combinations. Job 21205904 used one thread. Job 21205906 used two threads and also measured four-thread BoomerAMG. Each job built its own binary. The tables report costs for each allocation separately.

### Single-thread costs (job 21205904)

```{include} performance/euler-20261117-21205904-table.md
```

[Download costs and outcomes](performance/euler-20261117-21205904.csv) · [Measurement provenance](performance/euler-20261117-21205904.json)

### Two- and four-thread costs (job 21205906)

```{include} performance/euler-20261117-21205906-table.md
```

[Download costs and outcomes](performance/euler-20261117-21205906.csv) · [Measurement provenance](performance/euler-20261117-21205906.json)


## Provenance

[Every result](performance/euler-20261031.csv), the
[thread-scaling series](performance/euler-20261031-scaling.csv) and
[what all results share](performance/euler-20261031.json): apxchol revision
`1c4abb35`, hardware, toolchains, tolerance and the cap rule. The header points
to the full extract, with per-repeat values, checksums and the reason for every
cell without a result. GPU memory is not published: these runs recorded
device-wide usage, not per-process usage.
