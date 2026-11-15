# LeoMed: x86_64 and CUDA

ETH LeoMed nodes: AMD EPYC Milan with 28 virtual cores per node, and one
NVIDIA RTX 4090 (24 GB) for the CUDA series. CPU series use 28 threads on a
whole node.

This page shows a partial set. RCHOL, pRCHOL, AC, AC2, CMG and ParAC are
still running on LeoMed, so their cells are empty for now.

Competitors are capped at 20 times apxchol's fastest setup + solve time on the
same matrix and device, or 30 s if that is longer. A run over the cap is shown
as `≥20×`, not as a time. Other labels: `fail` (failed), `oom` (out of memory),
`nc` (did not converge) and `n/a` (input not supported).

An asterisk (`*`) marks a time whose setup includes input adaptation by the
benchmark. The benchmark finds connected components, pins one vertex per
component or slices components, so that the solver accepts a singular or
disconnected Laplacian. The `adapt_s` column of the data files gives that part
of setup.

```{note}
Serial RCHOL and pRCHOL come from a separate build with LLVM 22.1.8. With the
LLVM 20.1.8 toolchain used for every other series here, serial RCHOL stops on
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

## Provenance

[Every result](performance/leomed-20261114.csv) and
[what all results share](performance/leomed-20261114.json): apxchol revision
`7dee3a3a`, hardware, toolchains, tolerance and the cap rule.
