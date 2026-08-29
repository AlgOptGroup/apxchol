# Daint T=72 fair-solver campaign

The combined campaign contains **750/750 planned cells** over 27 matrices: **670 complete**, **7 not converged**, **35 timeout**, **7 failed**, **31 n/a**. Every completed cell has true relative residual at most `9.936114e-09`.

Times are medians of three full setup+solve repetitions at T=72. The CPU/GPU ratio below is CPU time divided by GPU time, so values above one favor the GPU.

## CPU/GPU crossover on paired completed cells

| solver | pairs | setup CPU/GPU | solve CPU/GPU | total CPU/GPU |
|---|---:|---:|---:|---:|
| apxchol/bg | 27 | 0.388× | 0.853× | 0.461× |
| AMGCL | 26 | 0.325× | 2.809× | 0.421× |
| BoomerAMG | 23 | 0.560× | 5.434× | 0.667× |
| BoomerAMG/cut | 25 | 0.522× | 7.558× | 0.750× |

The GPU frequently accelerates the iterative solve but pays a larger setup interval (device preparation, factor upload, and GPU triangular-solve setup). For the apxchol default it wins total time on the largest 2D grid and the two largest social graphs, while the 72-core CPU path remains faster on most smaller inputs.

## Headline total-time comparison

Ratios are competitor/apxchol-default on paired completed cells; above one favors apxchol. Timeouts and failures are excluded from the geometric mean, but remain visible in the heatmaps and outcome table.

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

Daint's ARM64 environment supports the 13 CPU apxchol configurations, AMGCL, BoomerAMG default/cut, portable-PCG RCHOL/pRCHOL, AC/AC2 under native ARM64 Julia, three GPU apxchol configurations, AMGCL-CUDA, genuine Hypre-CUDA default/cut, and ParAC's CUDA graph/physics drivers. The existing main sweep intentionally omits six fwd_star/bstr Orkut ablations. ParAC-CPU remains excluded because its upstream driver requires oneMKL; CMG remains excluded because native MATLAB for Linux is x86-64 and Octave timing is not the same series.
