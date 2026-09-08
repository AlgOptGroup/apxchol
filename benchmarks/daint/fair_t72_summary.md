# Historical Daint T72 campaign: 2b755997

750/750 planned cells: 670 complete, 7 nonconverged, 35 timeout, 7 failed, 31 n/a.
Maximum completed true relative residual 9.936114e-9. Three full repetitions per
cell; table times are medians. Historical scope omitted ParAC CPU and CMG;
[current coverage](coverage.json) differs.

CPU/GPU ratios are CPU divided by GPU. Competitor ratios are competitor divided
by apxchol-default. Both use paired completed cells; excluded failures/timeouts
remain in the outcome table. Ratios above 1 favor GPU/apxchol respectively.

## CPU/GPU crossover on paired completed cells

| solver | pairs | setup CPU/GPU | solve CPU/GPU | total CPU/GPU |
|---|---:|---:|---:|---:|
| apxchol/bg | 27 | 0.388× | 0.853× | 0.461× |
| AMGCL | 26 | 0.325× | 2.809× | 0.421× |
| BoomerAMG | 23 | 0.560× | 5.434× | 0.667× |
| BoomerAMG/cut | 25 | 0.522× | 7.558× | 0.750× |

## Headline total-time comparison

| device | competitor | pairs | competitor/apxchol | apxchol wins | competitor wins |
|---|---|---:|---:|---:|---:|
| CPU | AMGCL | 26 | 1.735× | 10 | 16 |
| CPU | BoomerAMG | 23 | 3.267× | 23 | 0 |
| CPU | BoomerAMG/cut | 25 | 3.555× | 25 | 0 |
| CPU | RCHOL (portable PCG) | 14 | 44.150× | 14 | 0 |
| CPU | pRCHOL (portable PCG) | 13 | 33.127× | 13 | 0 |
| CPU | AC | 21 | 18.474× | 21 | 0 |
| CPU | AC2 | 21 | 27.382× | 21 | 0 |
| GPU | AMGCL | 26 | 1.835× | 22 | 4 |
| GPU | BoomerAMG | 26 | 2.028× | 25 | 1 |
| GPU | BoomerAMG/cut | 26 | 1.984× | 24 | 2 |
| GPU | ParAC Graph | 16 | 5.900× | 16 | 0 |
| GPU | ParAC Physics | 7 | 4.393× | 7 | 0 |

## Non-complete outcomes

| matrix | device | solver/config | status | evidence |
|---|---|---|---|---|
| G3_circuit | CPU | `ac` | not_converged | true residual 1.4e-07 > 1e-8 |
| G3_circuit | CPU | `ac2` | not_converged | true residual 1.4e-07 > 1e-8 |
| apache2 | CPU | `ac` | not_converged | true residual 6.5e-06 > 1e-8 |
| apache2 | CPU | `ac2` | not_converged | true residual 6.5e-06 > 1e-8 |
| as-Skitter | CPU | `hypre_boomeramg` | timeout | wall-clock cap 60s |
| as-Skitter | CPU | `rchol` | timeout | wall-clock cap 60s |
| as-Skitter | CPU | `rchol_par` | timeout | wall-clock cap 60s |
| as-Skitter | GPU | `parac_graph` | not_converged | true residual 2.32e+04 > 1e-8 |
| coPapersDBLP | CPU | `rchol` | timeout | wall-clock cap 60s |
| coPapersDBLP | CPU | `rchol_par` | timeout | wall-clock cap 60s |
| com-LiveJournal | CPU | `amgcl` | timeout | wall-clock cap 60s |
| com-LiveJournal | CPU | `hypre_boomeramg` | timeout | wall-clock cap 60s |
| com-LiveJournal | CPU | `hypre_boomeramg/cut` | timeout | wall-clock cap 60s |
| com-LiveJournal | CPU | `rchol` | timeout | wall-clock cap 60s |
| com-LiveJournal | CPU | `rchol_par` | timeout | wall-clock cap 60s |
| com-LiveJournal | GPU | `amgcl_cuda` | timeout | wall-clock cap 60s |
| com-Orkut | CPU | `ac` | timeout | wall-clock cap 151s |
| com-Orkut | CPU | `ac2` | timeout | wall-clock cap 151s |
| com-Orkut | CPU | `hypre_boomeramg` | timeout | wall-clock cap 151s |
| com-Orkut | CPU | `hypre_boomeramg/cut` | timeout | wall-clock cap 151s |
| com-Orkut | CPU | `rchol` | timeout | wall-clock cap 151s |
| com-Orkut | CPU | `rchol_par` | timeout | wall-clock cap 151s |
| com-Orkut | GPU | `hypre_boomeramg_gpu` | failed | cuSPARSE insufficient resources during SpGEMM |
| com-Orkut | GPU | `hypre_boomeramg_gpu/cut` | failed | cuSPARSE insufficient resources during SpGEMM |
| com-Orkut | GPU | `parac_graph` | timeout | wall-clock cap 600s |
| com-Youtube | CPU | `hypre_boomeramg` | timeout | wall-clock cap 60s |
| com-Youtube | CPU | `rchol` | timeout | wall-clock cap 60s |
| com-Youtube | CPU | `rchol_par` | timeout | wall-clock cap 60s |
| ecology1 | CPU | `rchol_par` | failed | process terminated with SIGSEGV |
| grid3d_150 | CPU | `rchol` | timeout | wall-clock cap 60s |
| grid3d_150 | CPU | `rchol_par` | timeout | wall-clock cap 60s |
| grid3d_200 | CPU | `rchol` | timeout | wall-clock cap 60s |
| grid3d_200 | CPU | `rchol_par` | timeout | wall-clock cap 60s |
| grid3d_250 | CPU | `rchol` | timeout | wall-clock cap 60s |
| grid3d_250 | CPU | `rchol_par` | timeout | wall-clock cap 60s |
| grid_2000 | CPU | `rchol` | timeout | wall-clock cap 60s |
| grid_2000 | CPU | `rchol_par` | timeout | wall-clock cap 60s |
| grid_3000 | CPU | `rchol` | timeout | wall-clock cap 60s |
| grid_3000 | CPU | `rchol_par` | timeout | wall-clock cap 60s |
| grid_4000 | CPU | `rchol` | timeout | wall-clock cap 60s |
| grid_4000 | CPU | `rchol_par` | timeout | wall-clock cap 60s |
| grid_5000 | CPU | `ac` | failed | Julia driver failed |
| grid_5000 | CPU | `ac2` | failed | Julia driver failed |
| grid_5000 | CPU | `rchol` | timeout | wall-clock cap 60s |
| grid_5000 | CPU | `rchol_par` | timeout | wall-clock cap 60s |
| kron_g500-logn16 | GPU | `parac_graph` | not_converged | true residual 9.39e+09 > 1e-8 |
| thermal2 | CPU | `rchol` | failed | portable PCG rejected an invalid factor diagonal |
| thermal2 | CPU | `rchol_par` | failed | portable PCG rejected an invalid factor diagonal |
| thermal2 | GPU | `parac_physics` | not_converged | true residual 0.747 > 1e-8 |

## Coverage boundary
