# Latest benchmark summary — t16, tol 1e-8, original singular L (per-solver grounding; ParAC per-component-consistent RHS scored vs original L, CMG reg-rel)

`† CMG (MATLAB)` = canonical Koutis CMG (MEX, matlab-deps container). MATLAB-pcg wall-time isn't cross-language-comparable, so its **iteration count** is the comparable signal — see below.
Blank = not run; `X` = ran but did not reach 1e-8; `T` = timed out (> 10× apxchol's wall time on that matrix); `—` = solver doesn't support that de-singularization cell.

## Total solve time (s)


### grids

| matrix | apxchol | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (MATLAB)† | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|
| grid_500 | 0.10 | 4.13 | X | 0.24 | 0.29 | 0.06 |  | 0.52 | 0.47 |  |  |
| grid_1000 | 0.59 | 7.52 | X | 1.28 | 0.98 | 0.41 |  | 2.04 | 1.99 |  |  |
| grid3d_100 | 1.38 | 7.63 | 4.44 | 1.61 | 1.51 | 0.62 |  | 3.04 | 2.96 |  |  |
| grid_2000 | 3.09 | T | T | 4.40 | 4.79 | 1.81 |  | 8.31 | 8.47 |  |  |
| grid3d_150 | 3.73 | T | 17.90 | 6.00 | 6.43 | 2.48 |  | 12.15 | 11.64 |  |  |
| grid_3000 | 7.24 | T | T | 8.63 | 8.34 | 3.94 |  | 19.90 | 20.48 |  |  |
| grid3d_200 | 9.87 | T | T | 17.72 | 15.01 | 5.77 |  | 34.30 | 34.60 |  |  |
| grid_4000 | 14.80 | T | T | 20.77 | 14.88 | 7.58 |  | 38.19 | 41.72 |  |  |
| grid3d_250 | 20.78 | T | T | 28.68 | 32.45 | 13.19 |  | 63.48 | 64.76 |  |  |
| grid_5000 | 25.76 | T | T | 25.21 | 24.89 | 13.80 |  | 60.38 | 59.79 |  |  |

### ipm

| matrix | apxchol | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (MATLAB)† | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|
| iter0020 | 1.08 | 7.46 | 2.85 | 1.52 | 2.00 | 4.16 |  | 2.58 | 2.58 |  |  |
| iter0030 | 1.34 | 9.01 | 3.32 | 1.58 | 1.86 | 4.36 |  | 4.65 | 2.90 |  |  |
| iter0010 | 1.26 | 7.34 | 2.64 | 1.77 | 1.30 | 0.96 |  | 2.47 | 2.44 |  |  |
| iter0040 | 1.16 | 9.68 | 3.37 | 1.49 | 1.92 | 4.78 |  | 2.95 | 2.96 |  |  |

### suitesparse

| matrix | apxchol | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (MATLAB)† | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|
| com-Amazon | 0.37 | 11.49 | 2.09 | 1.32 | 1.24 | 1.15 |  | 1.10 | 1.08 |  |  |
| coAuthorsDBLP | 0.60 | 12.90 | 2.63 | 3.13 | 2.19 | 2.26 |  | 0.91 | 0.93 |  |  |
| parabolic_fem | 0.47 | 7.46 | 1.45 | 0.74 | 0.82 | 0.27 |  | 1.80 | 1.35 |  |  |
| apache2 | 0.50 | 5.35 | 2.29 | 0.73 | 0.77 | 0.60 |  | 2.01 | 1.83 |  |  |
| kron_g500-logn16 | 1.00 | 6.36 | 2.21 | 1.84 | 1.43 | 0.14 |  | 1.63 | 1.53 |  |  |
| ecology1 | 0.59 | 6.29 | 2.77 | 0.86 | 1.16 | 0.37 |  | 2.02 | 1.93 |  |  |
| com-Youtube | 1.54 | T | T | T | 7.27 | 12.55 |  | 7.50 | 7.50 |  |  |
| G3_circuit | 1.25 | 10.60 | 4.63 | 2.51 | 2.25 | 1.24 |  | 4.47 | 4.40 |  |  |
| thermal2 | 1.11 | 9.93 | 4.13 | 2.59 | 2.63 | 1.13 |  | 3.49 | 3.49 |  |  |
| as-Skitter | 3.34 | T | T | T | T | T |  | 27.68 | 27.31 |  |  |
| coPapersDBLP | 5.17 | T | 12.41 | T | T | 15.61 |  | 6.30 | 6.07 |  |  |
| com-LiveJournal | 25.99 | T | T | T | T | T |  | 787.74 | 871.00 |  |  |
| com-Orkut | 102.66 | X | T | T | T | 31.38 |  | T | T |  |  |

## PCG iterations (preconditioner quality, threads-independent)


### grids

| matrix | apxchol | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (MATLAB)† | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|
| grid_500 | 40 | 34 |  | 8 | 8 | 14 |  | 49 | 50 |  |  |
| grid_1000 | 41 | 44 |  | 7 | 7 | 12 |  | 54 | 55 |  |  |
| grid3d_100 | 28 | 22 | 25 | 8 | 8 | 13 |  | 34 | 34 |  |  |
| grid_2000 | 46 |  |  | 8 | 8 | 13 |  | 57 | 61 |  |  |
| grid3d_150 | 31 |  | 26 | 8 | 8 | 13 |  | 36 | 36 |  |  |
| grid_3000 | 50 |  |  | 7 | 7 | 13 |  | 64 | 66 |  |  |
| grid3d_200 | 31 |  |  | 11 | 11 | 13 |  | 37 | 37 |  |  |
| grid_4000 | 52 |  |  | 7 | 7 | 14 |  | 68 | 69 |  |  |
| grid3d_250 | 30 |  |  | 8 | 8 | 15 |  | 39 | 39 |  |  |
| grid_5000 | 51 |  |  | 7 | 7 | 14 |  | 73 | 72 |  |  |

### ipm

| matrix | apxchol | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (MATLAB)† | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|
| iter0020 | 29 | 36 | 35 | 10 | 38 | 19 |  | 36 | 36 |  |  |
| iter0030 | 42 | 45 | 45 | 10 | 38 | 20 |  | 91 | 46 |  |  |
| iter0010 | 32 | 34 | 33 | 10 | 22 | 24 |  | 38 | 38 |  |  |
| iter0040 | 44 | 54 | 53 | 10 | 47 | 19 |  | 53 | 52 |  |  |

### suitesparse

| matrix | apxchol | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (MATLAB)† | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|
| com-Amazon | 32 | 89 | 73 | 14 | 49 | 74 |  | 32 | 32 |  |  |
| coAuthorsDBLP | 26 | 104 | 118 | 16 | 33 | 102 |  | 24 | 26 |  |  |
| parabolic_fem | 41 | 53 | 48 | 8 | 8 | 20 |  | 61 | 47 |  |  |
| apache2 | 26 | 26 | 26 | 8 | 8 | 17 |  | 47 | 38 |  |  |
| kron_g500-logn16 | 14 | 38 | 30 | 6 | 14 | 16 |  | 13 | 12 |  |  |
| ecology1 | 43 | 36 | 46 | 8 | 8 | 12 |  | 57 | 53 |  |  |
| com-Youtube | 20 |  |  |  | 110 | 69 |  | 20 | 21 |  |  |
| G3_circuit | 45 | 42 | 39 | 9 | 9 | 19 |  | 62 | 61 |  |  |
| thermal2 | 39 | 45 | 44 | 9 | 9 | 20 |  | 46 | 42 |  |  |
| as-Skitter | 19 |  |  |  |  |  |  | 29 | 20 |  |  |
| coPapersDBLP | 36 |  | 52 |  |  | 174 |  | 26 | 24 |  |  |
| com-LiveJournal | 29 |  |  |  |  |  |  | 29 | 26 |  |  |
| com-Orkut | 12 |  |  |  |  | 48 |  |  |  |  |  |
