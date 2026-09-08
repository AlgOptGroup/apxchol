# Daint benchmarks

GH200: 72-core Grace CPU + Hopper GPU. [Protocol](../README.md) ·
[Headline CSV](results.csv) · [Coverage](coverage.json) ·
[Source selection](selection_provenance.json) · [Platform exceptions](PLATFORM.md)

**Headline:** 459/486 identities: 426 complete, 14 failed, 7 nonconverged,
6 timeout, 6 recorded n/a. The 27 absent numerical cells are canonical MATLAB
CMG platform exceptions; packed serial CMG is a separate series. Required
preparation is charged; CUDA initialization is separate. Every retained solve
must satisfy original-system true residual ≤1e-8. Unknown memory remains unknown.

Totals: [grids](figures/combined_overview_grids.png) ·
[IPM](figures/combined_overview_ipm.png) ·
[SuiteSparse](figures/combined_overview_suitesparse.png) · [Tables](summary.md)

## CPU setup scaling

Three representative matrices, T=1/2/4/8/16/36/72: **84 identities, 78 complete,
2 whole-cell timeouts, 4 unattempted**. Sources: apxchol `1a782c8d` (4619429),
AMGCL/Hypre `ea01e2ff` (4619591), ParAC adapter `8827024f` (4619591/4622020).
Each curve uses its own campaign’s T1; comparisons between solvers are not
interleaved A/B tests. Log axes emphasize ratios; dashed lines show ideal scaling.

![Setup speedup](figures/threads_cpu_representative_setup_speedup.png)

## CPU converged-solve scaling

![Solve speedup](figures/threads_cpu_representative_solve_speedup.png)

T72 seconds, **setup / solve**:

| Matrix | apxchol | AMGCL | Hypre | ParAC Graph |
|---|---:|---:|---:|---:|
| grid_2000 | 0.144 / 0.182 | 0.305 / 0.103 | 1.596 / 0.410 | 28.613 / 3.905 |
| iter0040 | 0.211 / 0.120 | 7.826 / 0.213 | 0.425 / 0.208 | 12.014 / 1.257 |
| as-Skitter | 0.514 / 0.247 | 1.842 / 4.907 | unattempted | 62.893 / 1.580 |

**ParAC comparisons are provisional pending adapter/accounting repair.** Charged
preparation is 24.837/10.452/58.097s; logged AMD is only 0.721/0.356/32.363s.
The remainder mixes necessary transformations with avoidable ASCII interchange
and audit work; it is not an isolated I/O measurement. Published values remain
unchanged, not retrospectively discounted. The portable-cpp solve is serial;
requested factorization threads do not establish measured worker utilization.

Skitter Hypre T2 completed (218.208s setup/4.712s solve), but T1/T4 exhausted
whole-cell budgets; no T1 speedup is fabricated. Higher T remain unattempted.
Original 4619591 FAILED85 remains recorded; 4622020 supplies only seven previously
unattempted ParAC points. Grid Hypre’s solve downturn T36→T72 (0.1495→0.4099s)
is retained; separate-campaign endpoint variation is not a source-regression proof.

[84-row extract](thread_scaling_cpu_representative.csv) ·
[Campaign provenance](representative_scaling_provenance.json) · Absolute
[setup](figures/threads_cpu_representative_setup_seconds.png)/[solve](figures/threads_cpu_representative_solve_seconds.png)

Full 12-matrix CPU: [setup](figures/threads_setup_speedup.png),
[solve](figures/threads_solve_speedup.png), [total](figures/threads_total_speedup.png),
[84 cells](thread_scaling.csv), [controls](cpu_scaling_provenance.json),
[iteration diagnostics](cpu_scaling_diagnostics.csv). Orkut uses same-node T1
references. Iterations vary with T; factor fill was not measured.

[Historical studies](HISTORICAL.md), including unchanged source-ea01 GPU scaling,
remain separate. These scaling refreshes did not rerun the headline table.
