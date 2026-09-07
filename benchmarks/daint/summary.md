# Benchmark summary — t72, tol 1e-8, original singular L (per-solver grounding; ParAC per-component-consistent RHS scored vs original L, CMG reg-rel)

`† CMG (MATLAB)` = canonical Koutis CMG (MEX, matlab-deps container). MATLAB-pcg wall-time isn't cross-language-comparable, so its **iteration count** is the comparable signal — see below.
Blank = not run; `X` = ran but did not reach 1e-8; `T` = timed out without a recoverable cap; `T≥seconds` = timed out at the exact cap persisted by the runner; `—` = solver doesn't support that de-singularization cell.

**Series rule (uniform).** Every column is exactly ONE (solver, configuration); no column is a per-cell minimum over configurations. Headline tables and charts use apxchol's declared default, `apxchol/bg`; the selector spread (`apxchol/bg`, `apxchol/greedy`, `apxchol/bk`) is confined to dedicated compact ablation figures. The chart thread count is selected a priori (t72, with a t1 fallback); duplicate cells are rejected, so neither status nor time can select the representative.

## Total solve time (s)


### grids

| matrix | apxchol/bg | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (MATLAB)† | CMG (packed, serial) | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| grid_500 | 0.03 | 0.62 | 0.83 | 0.14 | 0.14 | 0.02 |  | 0.49 | 2.61 | 2.23 | 0.61 | 0.58 |
| grid_1000 | 0.08 | 3.19 | 3.60 | 0.45 | 0.46 | 0.09 |  | 1.68 | 8.21 | 8.12 | 2.63 | 3.30 |
| grid3d_100 | 0.16 | 4.70 | 4.96 | 0.57 | 0.55 | 0.16 |  | 3.01 | 10.98 | 10.90 | 4.26 | 8.28 |
| grid_2000 | 0.33 | 17.29 | 18.90 | 1.95 | 2.26 | 0.41 |  | 8.44 | 30.87 | 31.41 | 10.86 | 14.36 |
| grid3d_150 | 0.50 | 19.16 | 16.90 | 2.10 | 2.15 | 0.64 |  | 8.75 | 37.56 | 38.14 | 17.01 | 32.70 |
| grid_3000 | 0.79 | 45.97 | 51.98 | 4.66 | 4.22 | 0.94 |  | 16.04 | 71.47 | 70.43 | 28.34 | 38.15 |
| grid3d_200 | 1.38 | 52.05 | 41.47 | 5.18 | 4.87 | 1.56 |  | 25.60 | 92.20 | 92.03 | 42.31 | 82.55 |
| grid_4000 | 1.69 | 96.39 | X | 7.21 | 6.93 | 1.73 |  | 30.45 | 129.24 | 128.58 | 53.14 | 65.76 |
| grid3d_250 | 2.64 | 119.50 | 84.91 | 9.63 | 9.83 | 2.99 |  | 41.26 | 182.47 | 182.61 | 102.71 | 154.63 |
| grid_5000 | 3.18 | 175.80 | 202.81 | 11.68 | 10.62 | 2.84 |  | 56.32 | 207.62 | 203.42 | 91.31 | 114.53 |

### ipm

| matrix | apxchol/bg | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (MATLAB)† | CMG (packed, serial) | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iter0010 | 0.29 | 3.68 | X | 0.47 | 0.55 | 2.73 |  | 4.14 | 12.70 | 12.95 | 5.18 | 6.25 |
| iter0020 | 0.30 | 3.82 | 3.30 | 0.61 | 0.66 | 9.92 |  | 3.45 | 12.88 | 12.55 | 5.41 | 6.81 |
| iter0030 | 0.33 | 4.32 | X | 0.61 | 0.60 | 7.56 |  | 3.91 | 12.88 | 12.61 | 5.95 | 7.35 |
| iter0040 | 0.35 | 4.76 | 4.32 | 0.65 | 0.70 | 7.98 |  | 3.97 | 12.88 | 13.22 | 6.21 | 7.06 |

### suitesparse

| matrix | apxchol/bg | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (MATLAB)† | CMG (packed, serial) | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| com-Amazon | 0.09 | 3.01 | 2.75 | 0.49 | 0.64 | 0.40 |  | 2.04 | 4.63 | 4.30 | 1.23 | 1.84 |
| coAuthorsDBLP | 0.11 | 3.01 | X | 0.81 | 0.62 | 0.59 |  | 1.90 | 4.50 | 4.20 | 0.78 | 1.62 |
| parabolic_fem | 0.06 | 2.62 | X | 0.32 | 0.32 | 0.04 |  | — | 6.23 | 6.27 | — | — |
| apache2 | 0.08 | 2.82 | 2.89 | 0.40 | 0.37 | 0.09 |  | 2.10 | 8.06 | 8.46 | X | X |
| kron_g500-logn16 | 0.17 | 3.13 | X | 0.58 | 0.45 | 0.08 |  | 3.22 | 8.72 | 9.91 | 2.51 | 3.39 |
| ecology1 | 0.08 | 3.02 | 3.44 | 0.48 | 0.47 | 0.09 |  | 1.73 | 8.30 | 8.20 | 2.65 | 3.25 |
| com-Youtube | 0.31 | 59.44 | X | 39.18 | 4.07 | 4.71 |  | 13.53 | 18.78 | 18.32 | 3.86 | 5.68 |
| G3_circuit | 0.16 | 5.82 | 6.00 | 0.81 | 0.77 | 0.26 |  | 9.98 | X | 13.87 | X | X |
| thermal2 | 0.16 | X | X | 0.82 | 0.86 | 0.19 |  | — | 19.19 | 15.52 | — | — |
| as-Skitter | 0.75 | X | X | 69.75 | 12.13 | 6.78 |  | 47.10 | 64.25 | 63.60 | 10.01 | 22.20 |
| coPapersDBLP | 0.78 | 17.63 | 14.56 | 9.49 | 8.67 | 1.91 |  | 31.30 | 40.22 | 38.08 | 14.64 | 25.13 |
| com-LiveJournal | 5.01 | T | T | 149.04 | 68.36 | 83.74 |  | 261.27 | 578.82 | 580.88 | 104.18 | 175.87 |
| com-Orkut | 16.30 | X | T | X | 214.60 | 11.45 |  | T | 2514.99 | 2500.74 | T | T |

## PCG iterations (preconditioner quality, threads-independent)


### grids

| matrix | apxchol/bg | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (MATLAB)† | CMG (packed, serial) | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| grid_500 | 39 | 34 | 41 | 8 | 8 | 14 |  | 16 | 48 | 44 | 38 | 24 |
| grid_1000 | 44 | 43 | 51 | 8 | 8 | 12 |  | 16 | 49 | 52 | 41 | 26 |
| grid3d_100 | 30 | 22 | 25 | 9 | 9 | 13 |  | 15 | 34 | 34 | 28 | 20 |
| grid_2000 | 47 | 59 | 77 | 8 | 8 | 13 |  | 14 | 55 | 54 | 43 | 26 |
| grid3d_150 | 31 | 22 | 26 | 9 | 9 | 13 |  | 16 | 37 | 36 | 27 | 20 |
| grid_3000 | 49 | 68 | 101 | 8 | 8 | 13 |  | 15 | 62 | 59 | 45 | 27 |
| grid3d_200 | 32 | 22 | 26 | 9 | 9 | 13 |  | 15 | 36 | 37 | 27 | 20 |
| grid_4000 | 53 | 83 |  | 8 | 8 | 14 |  | 16 | 62 | 62 | 48 | 26 |
| grid3d_250 | 32 | 22 | 27 | 9 | 9 | 15 |  | 16 | 38 | 38 | 28 | 20 |
| grid_5000 | 51 | 98 | 154 | 7 | 7 | 14 |  | 14 | 64 | 65 | 50 | 27 |

### ipm

| matrix | apxchol/bg | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (MATLAB)† | CMG (packed, serial) | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iter0010 | 30 | 36 |  | 10 | 22 | 20 |  | 58 | 29 | 29 | 30 | 19 |
| iter0020 | 29 | 37 | 36 | 12 | 39 | 19 |  | 27 | 48 | 31 | 29 | 19 |
| iter0030 | 41 | 47 |  | 12 | 33 | 20 |  | 32 | 53 | 38 | 41 | 22 |
| iter0040 | 46 | 55 | 62 | 12 | 42 | 19 |  | 33 | 47 | 45 | 47 | 24 |

### suitesparse

| matrix | apxchol/bg | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (MATLAB)† | CMG (packed, serial) | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| com-Amazon | 33 | 95 | 78 | 14 | 50 | 74 |  | 27 | 30 | 30 | 29 | 24 |
| coAuthorsDBLP | 26 | 98 |  | 16 | 33 | 102 |  | 37 | 25 | 24 | 20 | 17 |
| parabolic_fem | 40 | 48 |  | 8 | 8 | 20 |  |  | 62 | 46 |  |  |
| apache2 | 26 | 26 | 26 | 8 | 8 | 17 |  | 17 | 45 | 31 | 25 | 17 |
| kron_g500-logn16 | 14 | 38 |  | 6 | 15 | 16 |  | 12 | 13 | 13 | 10 | 8 |
| ecology1 | 43 | 36 | 46 | 8 | 8 | 12 |  | 17 | 50 | 49 | 42 | 25 |
| com-Youtube | 19 | 471 | 500 | 17 | 110 | 69 |  | 66 | 20 | 19 | 16 | 12 |
| G3_circuit | 34 | 36 | 46 | 9 | 9 | 17 |  | 26 |  | 50 | 35 | 20 |
| thermal2 | 38 |  |  | 10 | 10 | 20 |  |  | 166 | 40 |  |  |
| as-Skitter | 21 | 500 | 500 | 29 | 227 | 270 |  | 86 | 20 | 19 | 16 | 15 |
| coPapersDBLP | 37 | 51 | 53 | 21 | 35 | 174 |  | 56 | 24 | 22 | 19 | 15 |
| com-LiveJournal | 30 |  |  | 19 | 155 | 431 |  | 118 | 24 | 25 | 22 | 17 |
| com-Orkut | 18 |  |  |  | 30 | 48 |  |  | 15 | 15 |  |  |
