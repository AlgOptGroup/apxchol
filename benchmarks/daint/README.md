# Daint benchmarks

GH200: 72-core Grace CPU + Hopper GPU. [Protocol](../README.md) ·
[Values](results.csv) · [Coverage](coverage.json) ·
[Source selection](selection_provenance.json) · [Platform exceptions](PLATFORM.md)

**513/513 identities:** 480 complete, 14 failed, 7 nonconverged, 6 timeout and
6 recorded n/a. Packed serial CMG replaces MATLAB CMG in the current profile.
All 108 APX identities were refreshed; the 405 competitor cells are unchanged.
Both CPU samplers now pass `G3_circuit`; its earlier trace-cycle failure and
the separate successful repair run remain in [source selection](selection_provenance.json).

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

## Total and solve time

Each pair shows one-RHS total, then solve time. Trace-cycle improves CPU solve
time on 26/27 matrices but improves total on only 8/27. GPU q=0.8 wins total on
25/27 matrices; q=0.2 wins on LiveJournal and `kron_g500-logn16`. Both alternatives
remain visible rather than selecting a different winner for each column.

![Grid totals](figures/combined_overview_grids.png)

![Grid solve times](figures/combined_solve_grids.png)

![IPM totals](figures/combined_overview_ipm.png)

![IPM solve times](figures/combined_solve_ipm.png)

![SuiteSparse totals](figures/combined_overview_suitesparse.png)

![SuiteSparse solve times](figures/combined_solve_suitesparse.png)

Colours compare solvers within each matrix. Missing measurements, failures,
nonconvergence and timeouts stay distinct. Unknown memory is not plotted as zero.

| Time | Devices | Grids | IPM | SuiteSparse |
|---|---|---|---|---|
| Setup | CPU | [Figure](figures/combined_setup_cpu_grids.png) | [Figure](figures/combined_setup_cpu_ipm.png) | [Figure](figures/combined_setup_cpu_suitesparse.png) |
| Setup | GPU | [Figure](figures/combined_setup_gpu_grids.png) | [Figure](figures/combined_setup_gpu_ipm.png) | [Figure](figures/combined_setup_gpu_suitesparse.png) |
| Setup | CPU + GPU | [Figure](figures/combined_setup_grids.png) | [Figure](figures/combined_setup_ipm.png) | [Figure](figures/combined_setup_suitesparse.png) |
| Solve | CPU | [Figure](figures/combined_solve_cpu_grids.png) | [Figure](figures/combined_solve_cpu_ipm.png) | [Figure](figures/combined_solve_cpu_suitesparse.png) |
| Solve | GPU | [Figure](figures/combined_solve_gpu_grids.png) | [Figure](figures/combined_solve_gpu_ipm.png) | [Figure](figures/combined_solve_gpu_suitesparse.png) |
| Solve | CPU + GPU | [Figure](figures/combined_solve_grids.png) | [Figure](figures/combined_solve_ipm.png) | [Figure](figures/combined_solve_suitesparse.png) |
| Total | CPU | [Figure](figures/combined_overview_cpu_grids.png) | [Figure](figures/combined_overview_cpu_ipm.png) | [Figure](figures/combined_overview_cpu_suitesparse.png) |
| Total | GPU | [Figure](figures/combined_overview_gpu_grids.png) | [Figure](figures/combined_overview_gpu_ipm.png) | [Figure](figures/combined_overview_gpu_suitesparse.png) |
| Total | CPU + GPU | [Figure](figures/combined_overview_grids.png) | [Figure](figures/combined_overview_ipm.png) | [Figure](figures/combined_overview_suitesparse.png) |

| Detail | Grids | IPM | SuiteSparse |
|---|---|---|---|
| Iterations | [Figure](figures/combined_iters_grids.png) | [Figure](figures/combined_iters_ipm.png) | [Figure](figures/combined_iters_suitesparse.png) |
| Peak host memory | [Figure](figures/combined_rss_peak_grids.png) | [Figure](figures/combined_rss_peak_ipm.png) | [Figure](figures/combined_rss_peak_suitesparse.png) |

[Tables](summary.md) · [Six-matrix sampler tradeoffs](SAMPLERS.md)

## Thread scaling

The representative CPU study has **105/105 complete identities**: GKS,
trace-cycle, AMGCL, BoomerAMG and ParAC on three matrices at 1–72 threads.
The 42 APX points use the current CPU implementation and complete API timing;
competitors retain their documented sources. Each curve uses its own T1.
These are separate campaigns, and factors/iterations can change with thread count.
One APX point (IPM trace-cycle T36) has 21.5% solve-time spread and remains
visible with its warning in the source metadata. ParAC's Skitter T2/IPM T36
solve-control gaps remain visible; preparation variability is unmeasured.

**ParAC uses parallel factorization and serial portable PCG on ARM.** Thread
counts reach its factorizer; solve scaling is therefore not expected here.
Required preparation limits setup scaling. At T72, AMD accounts for 75% of
Skitter setup but only 8–9% on grid/IPM. The complete factor-setup interval scales
1.87× on IPM and 1.72× on Skitter; the grid interval is too variable to quote a
speedup. The upstream thread-0 timer excludes barrier waiting and is not that
complete interval. Historical laptop scaling requested multithreaded MKL; this ARM port uses
serial PCG. Laptop setup charged the narrower timer plus AMD, so those setup
speedups are not directly comparable.

Hypre's Skitter curve is complete: five new points repair insufficient campaign
deadlines, while successful T2/T72 points reuse the same binary with explicit
mixed-campaign provenance. The earlier timeout/unattempted outcomes remain
recorded. T72 gives 3.59× setup and 3.91× solve speedup over T1.

![CPU setup speedup](figures/threads_cpu_representative_setup_speedup.png)

![CPU solve speedup](figures/threads_cpu_representative_solve_speedup.png)

[Representative values](thread_scaling_cpu_representative.csv) ·
[Sources and exceptions](representative_scaling_provenance.json) ·
Absolute [setup](figures/threads_cpu_representative_setup_seconds.png)/[solve](figures/threads_cpu_representative_solve_seconds.png)

Full CPU [setup](figures/threads_setup_speedup.png)/[solve](figures/threads_solve_speedup.png)/[total](figures/threads_total_speedup.png),
[values](thread_scaling.csv), [controls](cpu_scaling_provenance.json).
Earlier GPU [setup](figures/threads_gpu_setup_speedup.png)/[solve](figures/threads_gpu_solve_speedup.png),
[values](thread_scaling_gpu.csv). [Historical studies](HISTORICAL.md) retain their own revisions.
