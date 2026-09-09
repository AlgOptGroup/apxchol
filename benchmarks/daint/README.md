# Daint benchmarks

GH200: 72-core Grace CPU + Hopper GPU. [Protocol](../README.md) ·
[Values](results.csv) · [Coverage](coverage.json) ·
[Source selection](selection_provenance.json) · [Platform exceptions](PLATFORM.md)

**513/540 identities:** 479 complete, 15 failed, 7 nonconverged, 6 timeout and
6 recorded n/a. The 27 absent cells are canonical MATLAB CMG platform exceptions.
All 108 APX identities were refreshed; the 405 competitor cells are unchanged.
CPU trace-cycle's failed `G3_circuit` warmup remains a failure, with its three
retained repetitions unattempted.

The four APX rows compare CPU GKS/trace-cycle at degree quantile 0.2 and GPU-owned
GKS at quantiles 0.8/0.2. Here the quantile controls vertex selection, not the
sampler's edge probabilities. Required host preparation is included in GPU setup;
these labels do not claim that every setup stage resides on the GPU.

Each refreshed cell has one warmup and three retained attempts, T72, seed 42,
and original-system residual tolerance 1e-8. One coherent median-total repetition
supplies setup/solve times; timing spread warnings remain in the CSV. CUDA
initialization, common input/output and independent grading are outside API timing.
CPU uses Clang/libomp with library-default waiting; GPU uses GCC/PASSIVE.
The fixed-profile campaigns are separate, not interleaved A/B controls.

## Total time

![Grid totals](figures/combined_overview_grids.png)

![IPM totals](figures/combined_overview_ipm.png)

![SuiteSparse totals](figures/combined_overview_suitesparse.png)

Colours compare solvers within each matrix. Missing measurements, failures,
nonconvergence and timeouts stay distinct. Unknown memory is not plotted as zero.

| Detail | Grids | IPM | SuiteSparse |
|---|---|---|---|
| Setup | [Figure](figures/combined_setup_grids.png) | [Figure](figures/combined_setup_ipm.png) | [Figure](figures/combined_setup_suitesparse.png) |
| Solve | [Figure](figures/combined_solve_grids.png) | [Figure](figures/combined_solve_ipm.png) | [Figure](figures/combined_solve_suitesparse.png) |
| Iterations | [Figure](figures/combined_iters_grids.png) | [Figure](figures/combined_iters_ipm.png) | [Figure](figures/combined_iters_suitesparse.png) |
| Peak host memory | [Figure](figures/combined_rss_peak_grids.png) | [Figure](figures/combined_rss_peak_ipm.png) | [Figure](figures/combined_rss_peak_suitesparse.png) |

[Tables](summary.md) · [Six-matrix sampler tradeoffs](SAMPLERS.md)

## Thread scaling

These are earlier, separately labelled campaigns, not scaling measurements of
all four refreshed profiles. The representative CPU study has **84 identities:
78 complete, 2 whole-cell timeouts and 4 unattempted**. Each curve uses its own T1.
Sources: APX `1a782c8d`, AMGCL/Hypre `ea01e2ff`, and the corrected private ParAC
adapter. ParAC preparation is included; its repeated preparation variability is
unknown. Skitter T2 and IPM T36 ParAC solve-control gaps remain visible.

![CPU setup speedup](figures/threads_cpu_representative_setup_speedup.png)

![CPU solve speedup](figures/threads_cpu_representative_solve_speedup.png)

[Representative values](thread_scaling_cpu_representative.csv) ·
[Sources and exceptions](representative_scaling_provenance.json) ·
Absolute [setup](figures/threads_cpu_representative_setup_seconds.png)/[solve](figures/threads_cpu_representative_solve_seconds.png)

Full CPU [setup](figures/threads_setup_speedup.png)/[solve](figures/threads_solve_speedup.png)/[total](figures/threads_total_speedup.png),
[values](thread_scaling.csv), [controls](cpu_scaling_provenance.json).
Earlier GPU [setup](figures/threads_gpu_setup_speedup.png)/[solve](figures/threads_gpu_solve_speedup.png),
[values](thread_scaling_gpu.csv). [Historical studies](HISTORICAL.md) retain their own revisions.
