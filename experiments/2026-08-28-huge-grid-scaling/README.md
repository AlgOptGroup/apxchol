# Large-grid scaling: size helps, but does not identify the cause

This historical Daint study used 2D grids with 16 million and 64 million
vertices. Both achieved roughly 11× CPU setup speedup from one to 72 threads,
with little further improvement from 36 to 72. These results do not establish
whether size or grid structure explains the advantage over smaller,
irregular matrices.

The campaign contained 72 records: four calibration, four profiling,
60 setup-sweep records (two sizes, two builds, five thread counts, three
seeds), and four converged full solves. The setup sweep used one PCG
iteration; it is not evidence of converged-solve scaling.

| Grid | CPU setup T=1 | CPU setup T=72 | Speedup |
|---|---:|---:|---:|
| 16M vertices | 16.578 s | 1.501 s | 11.05× |
| 64M vertices | 67.386 s | 6.089 s | 11.07× |

On the 64M/T72 profile, elimination and triangular-solver setup each accounted
for about 26%, graph construction and partitioning about 14% each, and factor
assembly about 11%. Selector scaling was about 26×; candidate collection was
only 3.8% of total setup. Optimizing that collection alone cannot close the
whole setup gap.

The four full solves reported CPU/GPU totals of 2.709/2.030 seconds for 16M
and 10.757/8.309 seconds for 64M, with 53–54 iterations and residuals below
$10^{-8}$. CPU and CUDA builds used different compilers, so this was not a
compiler-controlled backend comparison.

Use the current benchmark runner and thread-scaling workflow for a new study;
this directory contains no executable campaign package. The [full historical
record](https://github.com/AlgOptGroup/apxchol/blob/1a526f25aec8829e8a9217b558ac2290a3840ae0/experiments/2026-08-28-huge-grid-scaling/README.md) identifies the source, builds, and collected records. Reusing its
setup-only results as a current converged scaling plot would change their
meaning.
