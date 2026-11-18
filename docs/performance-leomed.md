# LeoMed: x86_64 and CUDA

ETH LeoMed nodes: AMD EPYC Milan with 28 virtual cores per node, and one
NVIDIA RTX 4090 (24 GB) for the CUDA series. Headline CPU series use 28 threads on a
whole node.

The headline measurements were taken on 14 November 2026 at apxchol revision
`7dee3a3a`, before 1.0.0. Later changes added one-call sampler selection
([#166](https://github.com/AlgOptGroup/apxchol/pull/166)) and lowered the CUDA
trace-cycle threshold to 16 right-hand sides while changing dense-tail setup
([#171](https://github.com/AlgOptGroup/apxchol/pull/171)).

This page shows a partial set. A dash (`—`) marks an unmeasured cell.
Unmeasured cells are excluded from comparisons and do not count as apxchol
wins.

Competitors are capped at 20 times apxchol's fastest setup + solve time on the
same matrix and device, or 30 s if that is longer. A run over the cap is shown
as `cap`, not as a time. Other labels: `fail` (failed), `oom` (out of memory),
`nc` (did not converge) and `n/a` (input not supported).

An asterisk (`*`) marks a time whose setup includes input adaptation by the
benchmark. The benchmark finds connected components, pins one vertex per
component or slices components, so that the solver accepts a singular or
disconnected Laplacian. The `adapt_s` column of the data files gives that part
of setup.

```{note}
Serial RCHOL and pRCHOL come from a separate build with LLVM 22.1.8. With the
LLVM 20.1.8 toolchain used for the other headline series, serial RCHOL stops on
every matrix with a zero factor diagonal.
```

## CPU

![LeoMed CPU: speedup of apxchol over each family on every matrix](performance/leomed-cpu-families.svg)

![LeoMed CPU: every variant on every matrix](performance/leomed-cpu-variants.svg)

```{include} performance/leomed-cpu-agreement.md
```

```{include} performance/leomed-cpu-times.md
```

## CUDA

![LeoMed CUDA: speedup of apxchol over each family on every matrix](performance/leomed-gpu-families.svg)

![LeoMed CUDA: every variant on every matrix](performance/leomed-gpu-variants.svg)

```{include} performance/leomed-gpu-agreement.md
```

```{include} performance/leomed-gpu-times.md
```

## November RCHOL follow-up

This separate run on compute-biomed-20 used source `7dee3a3a`, clang 22.1.8 and native tuning. The host exposes a virtualized EPYC-Milan topology with 28 vCPUs. Physical-core and NUMA placement are not established. pRCHOL reports 16 effective workers in its 21 outcomes with repetition receipts.

All 54 outcomes are retained: 30 completed measurements, 21 over-cap results and three failures. Completed measurements meet the original-system tolerance of 1e-8 in the warmup and all three retained runs. These are descriptive costs from one allocation and do not enter the historical headline rankings.

The frozen cap rule counts the warmup as well as retained runs. The table flags warmup-only cap exceedances and processes stopped before complete repetition receipts. The pRCHOL YouTube result also misses tolerance after 500 iterations, even though its raw status is over cap. It is not an accurate solution. The failure notes report the observed errors without assigning an unverified native cause.

```{include} performance/leomed-20261117-rchol-10019793-table.md
```

[RCHOL outcomes](performance/leomed-20261117-rchol-10019793.csv) · [RCHOL provenance](performance/leomed-20261117-rchol-10019793.json)

## November CPU follow-ups

Five allocations on compute-biomed-20 covered 234 planned results. The two ParAC builds produced 31 eligible timings on the shared RHS. Another 15 completed above the cap, 58 used unsupported routes and four timed out. The C++ scaling runs produced 72 timings. Julia qualification stopped before any of its 54 measurements. The following tables keep each allocation separate from the headline scores.

### Portable ParAC (job 10019826)

```{include} performance/leomed-20261117-10019826-table.md
```

[Download costs and outcomes](performance/leomed-20261117-10019826.csv) · [Measurement provenance](performance/leomed-20261117-10019826.json)

### MKL ParAC (job 10019827)

```{include} performance/leomed-20261117-10019827-table.md
```

[Download costs and outcomes](performance/leomed-20261117-10019827.csv) · [Measurement provenance](performance/leomed-20261117-10019827.json)

### Skitter thread scaling (job 10020051)

```{include} performance/leomed-20261117-10020051-table.md
```

[Download costs and outcomes](performance/leomed-20261117-10020051.csv) · [Measurement provenance](performance/leomed-20261117-10020051.json)

### Grid and IPM thread scaling (job 10020052)

```{include} performance/leomed-20261117-10020052-table.md
```

[Download costs and outcomes](performance/leomed-20261117-10020052.csv) · [Measurement provenance](performance/leomed-20261117-10020052.json)

### Julia AC and AC2 (job 10020053)

```{include} performance/leomed-20261117-10020053-table.md
```

[Download costs and outcomes](performance/leomed-20261117-10020053.csv) · [Measurement provenance](performance/leomed-20261117-10020053.json)


### Julia AC and AC2 follow-up (job 10030941)

On 18 November, both Julia solvers passed qualification. Across 27 matrices, 30 of the 54 outcomes yielded eligible timings, 18 exceeded the time cap, 4 were unsupported and 2 missed the residual tolerance. The capped rows on G3_circuit also missed tolerance. For AC2 on com-Orkut, only one repeat finished.

```{include} performance/leomed-20261118-10030941-table.md
```

[Download costs and outcomes](performance/leomed-20261118-10030941.csv) · [Measurement provenance](performance/leomed-20261118-10030941.json)

## Provenance

[Every result](performance/leomed-20261114.csv) and
[what all results share](performance/leomed-20261114.json): apxchol revision
`7dee3a3a`, hardware, toolchains, tolerance and the cap rule.
