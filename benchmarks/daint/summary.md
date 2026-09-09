# Benchmark summary — t72, tol 1e-8, original operator

Blank = not run; `X` = ran but did not reach 1e-8; `T` = timed out without a recoverable cap; `T≥seconds` = timed out at the exact cap persisted by the runner; `—` = solver doesn't support that de-singularization cell.

**Series rule (uniform).** Every column is exactly ONE (solver, configuration); no column is a per-cell minimum over configurations. Headline tables and charts use the explicitly declared rows (`apxchol/bg`, `apxchol/trace-cycle`). The chart thread count is selected a priori (t72, with a t1 fallback); duplicate cells are rejected, so neither status nor time can select the representative.

## Total solve time (s)


### grids

| matrix | apxchol/bg | apxchol/trace-cycle | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (packed, serial) | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| grid_500 | 0.03 | 0.04 | 0.62 | 0.83 | 0.14 | 0.14 | 0.02 | 0.49 | 1.21 | 2.23 | 0.61 | 0.58 |
| grid_1000 | 0.09 | 0.10 | 3.19 | 3.60 | 0.45 | 0.46 | 0.09 | 1.68 | 3.55 | 8.12 | 2.63 | 3.30 |
| grid3d_100 | 0.17 | 0.24 | 4.70 | 4.96 | 0.57 | 0.55 | 0.16 | 3.01 | 5.38 | 10.90 | 4.26 | 8.28 |
| grid_2000 | 0.35 | 0.35 | 17.29 | 18.90 | 1.95 | 2.26 | 0.41 | 8.44 | 12.11 | 31.41 | 10.86 | 14.36 |
| grid3d_150 | 0.52 | 0.68 | 19.16 | 16.90 | 2.10 | 2.15 | 0.64 | 8.75 | 17.45 | 38.14 | 17.01 | 32.70 |
| grid_3000 | 0.79 | 0.78 | 45.97 | 51.98 | 4.66 | 4.22 | 0.94 | 16.04 | 28.02 | 70.43 | 28.34 | 38.15 |
| grid3d_200 | 1.29 | 1.55 | 52.05 | 41.47 | 5.18 | 4.87 | 1.56 | 25.60 | 42.01 | 92.03 | 42.31 | 82.55 |
| grid_4000 | 1.70 | 1.64 | 96.39 | X | 7.21 | 6.93 | 1.73 | 30.45 | 51.36 | 128.58 | 53.14 | 65.76 |
| grid3d_250 | 2.63 | 3.23 | 119.50 | 84.91 | 9.63 | 9.83 | 2.99 | 41.26 | 83.77 | 182.61 | 102.71 | 154.63 |
| grid_5000 | 2.98 | 3.22 | 175.80 | 202.81 | 11.68 | 10.62 | 2.84 | 56.32 | 81.20 | 203.42 | 91.31 | 114.53 |

### ipm

| matrix | apxchol/bg | apxchol/trace-cycle | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (packed, serial) | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iter0010 | 0.30 | 0.29 | 3.68 | X | 0.47 | 0.55 | 2.73 | 4.14 | 4.52 | 12.95 | 5.18 | 6.25 |
| iter0020 | 0.32 | 0.32 | 3.82 | 3.30 | 0.61 | 0.66 | 9.92 | 3.45 | 5.09 | 12.55 | 5.41 | 6.81 |
| iter0030 | 0.33 | 0.31 | 4.32 | X | 0.61 | 0.60 | 7.56 | 3.91 | 5.35 | 12.61 | 5.95 | 7.35 |
| iter0040 | 0.33 | 0.31 | 4.76 | 4.32 | 0.65 | 0.70 | 7.98 | 3.97 | 5.03 | 13.22 | 6.21 | 7.06 |

### suitesparse

| matrix | apxchol/bg | apxchol/trace-cycle | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (packed, serial) | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| com-Amazon | 0.09 | 0.12 | 3.01 | 2.75 | 0.49 | 0.64 | 0.40 | 2.04 | 2.39 | 4.30 | 1.23 | 1.84 |
| coAuthorsDBLP | 0.12 | 0.14 | 3.01 | X | 0.81 | 0.62 | 0.59 | 1.90 | 2.26 | 4.20 | 0.78 | 1.62 |
| parabolic_fem | 0.07 | 0.09 | 2.62 | X | 0.32 | 0.32 | 0.04 | — | 2.89 | 6.27 | — | — |
| apache2 | 0.09 | 0.11 | 2.82 | 2.89 | 0.40 | 0.37 | 0.09 | 2.10 | 3.36 | 8.46 | X | X |
| kron_g500-logn16 | 0.18 | 0.19 | 3.13 | X | 0.58 | 0.45 | 0.08 | 3.22 | 4.14 | 9.91 | 2.51 | 3.39 |
| ecology1 | 0.09 | 0.10 | 3.02 | 3.44 | 0.48 | 0.47 | 0.09 | 1.73 | 3.52 | 8.20 | 2.65 | 3.25 |
| com-Youtube | 0.32 | 0.33 | 59.44 | X | 39.18 | 4.07 | 4.71 | 13.53 | 12.60 | 18.32 | 3.86 | 5.68 |
| G3_circuit | 0.15 | 0.17 | 5.82 | 6.00 | 0.81 | 0.77 | 0.26 | 9.98 | X | 13.87 | X | X |
| thermal2 | 0.16 | 0.17 | X | X | 0.82 | 0.86 | 0.19 | — | 10.47 | 15.52 | — | — |
| as-Skitter | 0.76 | 0.76 | X | X | 69.75 | 12.13 | 6.78 | 47.10 | 44.30 | 63.60 | 10.01 | 22.20 |
| coPapersDBLP | 0.83 | 0.80 | 17.63 | 14.56 | 9.49 | 8.67 | 1.91 | 31.30 | 15.96 | 38.08 | 14.64 | 25.13 |
| com-LiveJournal | 4.68 | 4.74 | T | T | 149.04 | 68.36 | 83.74 | 261.27 | 524.54 | 580.88 | 104.18 | 175.87 |
| com-Orkut | 15.74 | 16.78 | X | T | X | 214.60 | 11.45 | T | 2309.14 | 2500.74 | T | T |

## PCG iterations (preconditioner quality, threads-independent)


### grids

| matrix | apxchol/bg | apxchol/trace-cycle | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (packed, serial) | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| grid_500 | 39 | 25 | 34 | 41 | 8 | 8 | 14 | 16 | 48 | 44 | 38 | 24 |
| grid_1000 | 44 | 27 | 43 | 51 | 8 | 8 | 12 | 16 | 49 | 52 | 41 | 26 |
| grid3d_100 | 30 | 21 | 22 | 25 | 9 | 9 | 13 | 15 | 34 | 34 | 28 | 20 |
| grid_2000 | 47 | 28 | 59 | 77 | 8 | 8 | 13 | 14 | 55 | 54 | 43 | 26 |
| grid3d_150 | 31 | 21 | 22 | 26 | 9 | 9 | 13 | 16 | 37 | 36 | 27 | 20 |
| grid_3000 | 49 | 29 | 68 | 101 | 8 | 8 | 13 | 15 | 62 | 59 | 45 | 27 |
| grid3d_200 | 32 | 21 | 22 | 26 | 9 | 9 | 13 | 15 | 36 | 37 | 27 | 20 |
| grid_4000 | 53 | 29 | 83 |  | 8 | 8 | 14 | 16 | 62 | 62 | 48 | 26 |
| grid3d_250 | 32 | 21 | 22 | 27 | 9 | 9 | 15 | 16 | 38 | 38 | 28 | 20 |
| grid_5000 | 51 | 31 | 98 | 154 | 7 | 7 | 14 | 14 | 64 | 65 | 50 | 27 |

### ipm

| matrix | apxchol/bg | apxchol/trace-cycle | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (packed, serial) | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iter0010 | 30 | 21 | 36 |  | 10 | 22 | 20 | 58 | 29 | 29 | 30 | 19 |
| iter0020 | 29 | 22 | 37 | 36 | 12 | 39 | 19 | 27 | 48 | 31 | 29 | 19 |
| iter0030 | 41 | 27 | 47 |  | 12 | 33 | 20 | 32 | 53 | 38 | 41 | 22 |
| iter0040 | 46 | 27 | 55 | 62 | 12 | 42 | 19 | 33 | 47 | 45 | 47 | 24 |

### suitesparse

| matrix | apxchol/bg | apxchol/trace-cycle | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (packed, serial) | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| com-Amazon | 33 | 19 | 95 | 78 | 14 | 50 | 74 | 27 | 30 | 30 | 29 | 24 |
| coAuthorsDBLP | 26 | 16 | 98 |  | 16 | 33 | 102 | 37 | 25 | 24 | 20 | 17 |
| parabolic_fem | 40 | 25 | 48 |  | 8 | 8 | 20 |  | 62 | 46 |  |  |
| apache2 | 26 | 19 | 26 | 26 | 8 | 8 | 17 | 17 | 45 | 31 | 25 | 17 |
| kron_g500-logn16 | 14 | 13 | 38 |  | 6 | 15 | 16 | 12 | 13 | 13 | 10 | 8 |
| ecology1 | 43 | 28 | 36 | 46 | 8 | 8 | 12 | 17 | 50 | 49 | 42 | 25 |
| com-Youtube | 19 | 13 | 471 | 500 | 17 | 110 | 69 | 66 | 20 | 19 | 16 | 12 |
| G3_circuit | 34 | 21 | 36 | 46 | 9 | 9 | 17 | 26 |  | 50 | 35 | 20 |
| thermal2 | 38 | 28 |  |  | 10 | 10 | 20 |  | 166 | 40 |  |  |
| as-Skitter | 21 | 15 | 500 | 500 | 29 | 227 | 270 | 86 | 20 | 19 | 16 | 15 |
| coPapersDBLP | 37 | 20 | 51 | 53 | 21 | 35 | 174 | 56 | 24 | 22 | 19 | 15 |
| com-LiveJournal | 30 | 18 |  |  | 19 | 155 | 431 | 118 | 24 | 25 | 22 | 17 |
| com-Orkut | 18 | 13 |  |  |  | 30 | 48 |  | 15 | 15 |  |  |
