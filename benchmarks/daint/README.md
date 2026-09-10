# Daint benchmarks

GH200: 72-core Grace CPU + Hopper GPU. [Protocol](../README.md) ·
[Values](results.csv) · [Coverage](coverage.json) ·
[Source selection](selection_provenance.json) · [Platform exceptions](PLATFORM.md)

**540/540 identities:** 507 complete, 12 failed, 9 nonconverged, 6 timeout and
6 recorded n/a. Packed serial CMG replaces MATLAB CMG in the current profile.
All 135 APX identities were measured, including GPU trace-cycle on all 27 matrices.
ParAC Graph CPU preparation was reconciled for all 27 matrices;
377 competitor cells retain their earlier measurements.
Both CPU samplers now pass `G3_circuit`; its earlier trace-cycle failure and
the separate successful repair run remain in [source selection](selection_provenance.json).

The five APX rows compare CPU GKS/trace-cycle at degree quantile 0.2, GPU-owned
GKS at quantiles 0.8/0.2 and GPU trace-cycle at 0.8. The quantile controls vertex
selection, not the sampler's edge probabilities. Required host preparation is included in GPU setup;
these labels do not claim that every setup stage resides on the GPU.

Each refreshed cell has one warmup and three retained attempts, T72, seed 42,
and original-system residual tolerance 1e-8. One coherent median-total repetition
supplies setup/solve times; timing spread warnings remain in the CSV. CUDA
initialization, common input/output and independent grading are outside API timing.
APX CPU rows use Clang/libomp with library-default waiting. APX GPU GKS uses GCC;
GPU trace uses a Clang/libomp driver with GCC as the CUDA host compiler. All
APX GPU rows use PASSIVE waiting. The CSV records toolchains per series; these are
separate campaigns/builds, not interleaved A/B controls.
GPU trace retains two current spread warnings: `parabolic_fem` solve (1.195×)
and `G3_circuit` setup/total (1.164×/1.151×), using the prescribed median-total
repetitions. The [previous source641 snapshot](historical/gpu-trace-source641-20260910/)
retains its own values and warnings. The current 27-matrix run follows a separate
96-call optimization acceptance study; these are distinct measurement campaigns.

ParAC's 26 successful Graph CPU cells combine corrected preparation with their
original native timing repetitions after input/output identity checks. Required
transformation, ordering and cleanup remain charged; serialization is excluded.
This is an accounting correction, with unmeasured preparation variability and
unknown corrected whole-pipeline peak memory. G3 Graph CPU/GPU hit calibration
iteration limits and are reported as nonconverged without retained timings.

## Total and solve time

The tables separate setup, solve and one-RHS total. Trace-cycle improves CPU solve
time on 26/27 matrices but improves total on only 8/27. GPU GKS q=0.8 wins total on
25/27 matrices; q=0.2 wins on LiveJournal and `kron_g500-logn16`. Both alternatives
remain visible rather than selecting a different winner for each column.

On GPU, trace-cycle has lower solve time on 26/27 matrices and lower one-RHS total
on only 1/27 versus GKS q=0.8. Across all 27, its geometric-mean ratios are
0.743× solve, 0.639× iterations, 1.221× setup and 1.143× total. These describe
separate fixed-profile campaigns/builds. GKS remains the default; the solve view
shows trace-cycle's benefit alongside its setup cost.

Colours use an uncapped logarithmic scale within each matrix. CPU-only and
GPU-only views compare against their own fastest solver; combined views use
the fastest result across both devices. Missing measurements, failures,
nonconvergence and timeouts stay distinct. Unknown memory is not plotted as zero.

| Time | Devices | Grids | IPM | SuiteSparse |
|---|---|---|---|---|
| Setup | CPU | ![Figure](figures/combined_setup_cpu_grids.png) | ![Figure](figures/combined_setup_cpu_ipm.png) | ![Figure](figures/combined_setup_cpu_suitesparse.png) |
| Setup | GPU | ![Figure](figures/combined_setup_gpu_grids.png) | ![Figure](figures/combined_setup_gpu_ipm.png) | ![Figure](figures/combined_setup_gpu_suitesparse.png) |
| Setup | CPU + GPU | ![Figure](figures/combined_setup_grids.png) | ![Figure](figures/combined_setup_ipm.png) | ![Figure](figures/combined_setup_suitesparse.png) |
| Solve | CPU | ![Figure](figures/combined_solve_cpu_grids.png) | ![Figure](figures/combined_solve_cpu_ipm.png) | ![Figure](figures/combined_solve_cpu_suitesparse.png) |
| Solve | GPU | ![Figure](figures/combined_solve_gpu_grids.png) | ![Figure](figures/combined_solve_gpu_ipm.png) | ![Figure](figures/combined_solve_gpu_suitesparse.png) |
| Solve | CPU + GPU | ![Figure](figures/combined_solve_grids.png) | ![Figure](figures/combined_solve_ipm.png) | ![Figure](figures/combined_solve_suitesparse.png) |
| Total | CPU | ![Figure](figures/combined_overview_cpu_grids.png) | ![Figure](figures/combined_overview_cpu_ipm.png) | ![Figure](figures/combined_overview_cpu_suitesparse.png) |
| Total | GPU | ![Figure](figures/combined_overview_gpu_grids.png) | ![Figure](figures/combined_overview_gpu_ipm.png) | ![Figure](figures/combined_overview_gpu_suitesparse.png) |
| Total | CPU + GPU | ![Figure](figures/combined_overview_grids.png) | ![Figure](figures/combined_overview_ipm.png) | ![Figure](figures/combined_overview_suitesparse.png) |

| Detail | Grids | IPM | SuiteSparse |
|---|---|---|---|
| Iterations | ![Figure](figures/combined_iters_grids.png) | ![Figure](figures/combined_iters_ipm.png) | ![Figure](figures/combined_iters_suitesparse.png) |
| Peak host memory | ![Figure](figures/combined_rss_peak_grids.png) | ![Figure](figures/combined_rss_peak_ipm.png) | ![Figure](figures/combined_rss_peak_suitesparse.png) |
| Factor fill | ![Fill](figures/fill_heatmap_grids.png) | ![Fill](figures/fill_heatmap_ipm.png) | ![Fill](figures/fill_heatmap_suitesparse.png) |
| Sampled GPU memory | ![Memory](figures/gpu_sampled_vram_grids.png) | ![Memory](figures/gpu_sampled_vram_ipm.png) | ![Memory](figures/gpu_sampled_vram_suitesparse.png) |

Fill is $2\,\mathrm{offdiag}(L)/\mathrm{offdiag}(A)$: 275/351 AC-family
observations are available; missing entries remain blank. RCHOL ratios carry
reported rounding uncertainty; solve status is retained alongside structural fill.
[Fill values](fill.csv) · [Sources](fill_provenance.json).

A separate memory run of the refreshed GPU source covers **54/216 identities**:
GKS and trace-cycle q=0.8 on all 27 matrices. Sampled process maxima are lower bounds on whole-run
peaks; the other 162 identities remain unknown. Monitored timings are excluded.
[Memory details and sampling limits](GPU-MEMORY.md) · [Values](sampled_gpu_vram.csv).

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

| CPU scaling | Setup | Solve |
|---|---|---|
| Time | ![Setup time](figures/threads_cpu_representative_setup_seconds.png) | ![Solve time](figures/threads_cpu_representative_solve_seconds.png) |
| Speedup | ![Setup speedup](figures/threads_cpu_representative_setup_speedup.png) | ![Solve speedup](figures/threads_cpu_representative_solve_speedup.png) |

[Representative values](thread_scaling_cpu_representative.csv) ·
[Sources and exceptions](representative_scaling_provenance.json)

| Earlier studies | Setup | Solve | Total | Data |
|---|---|---|---|---|
| Full CPU | [Figure](figures/threads_setup_speedup.png) | [Figure](figures/threads_solve_speedup.png) | [Figure](figures/threads_total_speedup.png) | [Values](thread_scaling.csv), [controls](cpu_scaling_provenance.json) |
| GPU | [Figure](figures/threads_gpu_setup_speedup.png) | [Figure](figures/threads_gpu_solve_speedup.png) | [Figure](figures/threads_gpu_total_speedup.png) | [Values](thread_scaling_gpu.csv) |

[Historical studies](HISTORICAL.md) retain their own revisions.
