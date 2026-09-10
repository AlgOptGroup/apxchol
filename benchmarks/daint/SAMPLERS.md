# Earlier six-matrix sampler comparison

The [current headline](README.md) uses refreshed 27-matrix GPU measurements. This earlier screen retains its original sources and setup costs.

Six matrices: grid_2000, grid3d_100, iter0010, iter0040, as-Skitter and com-LiveJournal. Each cell uses one warmup and three retained solves at T=72, seed 42 and original-system tolerance 1e-8. All 232 comparison calls and six smoke calls returned accepted solutions. The [58-cell extract](sampler_detail.csv) preserves setup route, compiler, source, fill and timing variability.

Ratios below are geometric means against GKS within each matched six-matrix route; below 1 is better. They describe this screen, not a universal ranking.

| Setup → solve | Sampler | Setup | Solve | One-RHS total | Iterations | Stored entries |
|---|---|---:|---:|---:|---:|---:|
| CPU → CPU | Trace-cycle | 1.203 | 0.729 | 1.041 | 0.647 | 1.122 |
| CPU → CPU | Heavy-core K2 | 1.221 | 0.710 | 1.045 | 0.631 | 1.125 |
| GPU-owned → GPU | Trace-cycle | 1.755 | 0.631 | 1.599 | 0.561 | 1.123 |
| GPU-owned → GPU | Heavy-core K2 | 1.801 | 0.613 | 1.635 | 0.552 | 1.126 |
| CPU → GPU | Trace-cycle | 1.168 | 0.735 | 1.104 | 0.637 | 1.122 |
| CPU → GPU | Heavy-core K2 | 1.193 | 0.726 | 1.123 | 0.626 | 1.125 |

Trace-cycle and heavy-core K2 reduce iterations but store more entries. On CPU their one-RHS totals are close to GKS; the older GPU-owned port had a large setup penalty. In the CPU-setup/GPU-solve route, trace-cycle has a slower one-RHS total on all six matrices despite faster solves.

A separate instrumented diagnostic (job 4633827; four large calls and two smokes, all residuals accepted) attributes 95.1% of the extra trace-cycle setup on LiveJournal and 90.9% on Skitter to the sampling stage. Its LiveJournal sampling time rises from 0.315 to 7.963 seconds; 7.004 seconds are in rounds containing only oversized rows. This stage includes sorting, emission, status transfer, compaction and excess propagation, so it does not isolate one kernel. This motivated the later cooperative sampling optimization; current measurements are in the headline. These single-call diagnostics do not replace the repeated timings above.

CPU setup uses quantile 0.2; GPU-owned primary cells use 0.8. The extract also retains four owned trace/K2 quantile 0.2 social-graph cells; all four had slower totals than their corresponding 0.8 cells. CPU and hybrid runs use Clang/libomp with library-default waiting; owned runs use GCC with explicit PASSIVE waiting. These compiler/runtime differences are recorded, not treated as sampler effects across routes.

Retained solve spread exceeds 1.15 for CPU trace-cycle on LiveJournal. Hybrid K2 on grid_2000 has setup and total spread warnings (1.375 and 1.324); its solve timing is stable. Raw observations remain in the extract. Reuse estimates of setup + R×first-solve time are extrapolations, not measurements on multiple right-hand sides.
