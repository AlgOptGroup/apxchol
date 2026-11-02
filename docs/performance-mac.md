# Apple M1 Max: CPU and Metal

Apple M1 Max: 8 performance and 2 efficiency cores, a 32-core GPU and 32 GB
unified memory, on macOS 26.0.1. The CPU runs use the 8 performance cores;
macOS can't pin threads, so runs were unpinned, and processes ran at lower
priority on a shared machine.

No other solver in the comparison has a Metal backend, so the Metal section
shows apxchol's own times. The CPU section compares apxchol with the other
solvers, including Apple's Accelerate sparse Cholesky.

## CPU

Competitors ran on the 21 matrices that fit comfortably in 32 GB; on the other
six, `n/a` means they were not run. apxchol ran on all 27. Competitors are
capped at 20 times apxchol's setup + solve time on the same matrix, and at
least 30 s.

![Apple M1 Max CPU: speedup of apxchol over each solver family](performance/mac-families.svg)

On this CPU, AMGCL is faster than apxchol on 11 of 21 matrices, including
every grid, and the fastest other solver ties apxchol on linear programs.
apxchol is ahead on most social graphs.

![Apple M1 Max CPU: apxchol against the fastest other solver on each matrix](performance/mac-best-other.svg)

![Apple M1 Max CPU: speedup of apxchol over each family on every matrix](performance/mac-cpu-families.svg)

```{note}
Accelerate is a direct sparse Cholesky factorization (`SparseFactor`) with
refinement on the original system; it reports one iteration per solve. Its
`oom` cells were stopped before factorization, when the predicted factor did
not fit in half of the memory. RCHOL and pRCHOL use a portable PCG and run
without thread pinning. ParAC uses its portable CPU solve and its own
right-hand side. CMG is the serial native port; MATLAB is not available here.
```

![Apple M1 Max CPU: every variant on every matrix](performance/mac-cpu-variants.svg)

```{include} performance/mac-cpu-agreement.md
```

```{include} performance/mac-cpu-times.md
```

## Metal

![Apple M1 Max Metal: apxchol setup + solve time per matrix](performance/mac-metal.svg)

All 27 matrices met the 1e-8 tolerance on the original system, including
com-Orkut (237 million stored nonzeros) within the 32 GB of unified memory.
Times come from the repeat with the median setup + solve time, out of three
after one warm-up. Peak memory is the whole process.

```{include} performance/mac-metal-table.md
```

## Provenance

[Every result](performance/mac-20261101.csv) and
[what all results share](performance/mac-20261101.json): hardware, toolchains,
revisions and the cap rule. The CPU runs used revision `45f42824`; the Metal
runs used `0009d670`, from the pull request that added the benchmark's Metal
route. The header points to the full extract with binary and matrix checksums.
