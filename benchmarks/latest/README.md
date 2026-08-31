# Laptop benchmark snapshot

This directory is a committed presentation snapshot for an AMD Ryzen 9 7945HX
(16 physical cores), 128 GB RAM, and an NVIDIA RTX 4090 Laptop GPU (16 GB).
Absolute times apply only to that machine and run environment; use the
[Daint snapshot](../daint/) only as a separate machine study.

The campaign uses pinned T=16 runs and a true relative-residual target of
`1e-8`. The common operator, grading, timing, timeout, and series rules are in
the [benchmark protocol](../README.md). Per-cell provenance in the result store
is authoritative: this directory is historical, and missing or differently
grounded cells must not be silently combined into a new denominator.

## Useful views

| question | direct views |
|---|---|
| total time | [grids](figures/combined_overview_grids.png), [LP-IPM](figures/combined_overview_ipm.png), [SuiteSparse](figures/combined_overview_suitesparse.png) |
| CPU totals | [grids](figures/combined_overview_cpu_grids.png), [LP-IPM](figures/combined_overview_cpu_ipm.png), [SuiteSparse](figures/combined_overview_cpu_suitesparse.png) |
| GPU totals | [grids](figures/combined_overview_gpu_grids.png), [LP-IPM](figures/combined_overview_gpu_ipm.png), [SuiteSparse](figures/combined_overview_gpu_suitesparse.png) |
| setup versus solve | [CPU grids](figures/combined_breakdown_cpu_grids_2d.png), [GPU grids](figures/combined_breakdown_gpu_grids_2d.png), [CPU SuiteSparse](figures/combined_breakdown_cpu_suitesparse_small.png), [GPU SuiteSparse](figures/combined_breakdown_gpu_suitesparse_small.png) |
| apxchol choices | [selector overview](figures/poster_selectors_cpu.png), [SuiteSparse ablation](figures/ablation_suitesparse.png) |
| scaling | [setup](figures/threads_setup_speedup.png), [solve](figures/threads_solve_speedup.png), [GPU grids](figures/scaling_gpu_grids.png) |

The heatmaps normalize each matrix column to the fastest completed displayed
cell and annotate absolute values. Timeout cells show lower bounds where a cap
is known; failures, non-convergence, OOM, `n/a`, and missing cells remain
distinct. Linear breakdown plots use solid setup and hatched solve segments.

For exact values and outcomes, use the [generated summary](summary.md) and
[CSV extract](results.csv). The `figures/` directory also contains memory,
iteration, accuracy, fill, and family-split views.

## Interpretation boundaries

- The plotted apxchol row is a declared configuration, not a per-matrix best of
  several selectors. Selector and storage variants belong in ablation views.
- CMG uses the canonical MATLAB MEX path. Its iteration count is the useful
  algorithmic signal; MATLAB wall time is not a C++ speed baseline. Labelled
  regularized CMG cells are not part of original-operator comparisons.
- AC/AC2 are serial Julia reference implementations; their iteration counts are
  more meaningful than cross-language wall time.
- RCHOL/pRCHOL and ParAC include required reordering and conversion in setup.
- Grids are weighted anisotropic problems, not uniform Poisson matrices.

Regenerate this snapshot through the commands in the
[benchmark README](../README.md#build-and-run); the renderers filter stale cells
and refuse ambiguous series.
