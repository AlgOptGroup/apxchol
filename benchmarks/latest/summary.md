# Latest benchmark summary — t16, tol 1e-8, original singular L (per-solver grounding; ParAC per-component-consistent RHS scored vs original L, CMG reg-rel)

`† CMG (MATLAB)` = canonical Koutis CMG (MEX, matlab-deps container). MATLAB-pcg wall-time isn't cross-language-comparable, so its **iteration count** is the comparable signal — see below.
Blank = not run; `X` = ran but did not reach 1e-8; `T` = timed out (> 10× apxchol's wall time on that matrix); `—` = solver doesn't support that de-singularization cell.

## Total solve time (s)


### grids

| matrix | apxchol | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (MATLAB)† | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|
| grid_500 | 0.16 | 4.04 | 0.69 | 0.16 | 0.17 | 0.05 | 0.65 | 0.52 | 0.45 | 0.83 | 1.12 |
| grid_1000 | 0.92 | 7.16 | 3.43 | 0.81 | 0.96 | 0.33 | 2.31 | 2.00 | 1.93 | 3.04 | 3.64 |
| grid3d_100 | 1.47 | 7.14 | 4.36 | 1.23 | 1.29 | 0.56 | 2.78 | 2.89 | 2.95 | 4.51 | 8.36 |
| grid_2000 | 4.21 | T | T | 3.86 | 3.86 | 1.83 | 9.49 | 8.28 | 8.56 | 14.28 | 17.95 |
| grid3d_150 | 4.81 | T | T | 5.23 | 4.75 | 2.46 | 10.05 | 11.02 | 11.18 | 18.68 | 31.47 |
| grid_3000 | 10.75 | T | T | 9.32 | 9.12 | 4.32 | 23.37 | 19.51 | 20.03 | 34.82 | 45.50 |
| grid3d_200 | 12.33 | T | T | 17.35 | 15.06 | 5.91 | 24.34 | 28.41 | 29.56 | 46.60 | 83.88 |
| grid_4000 | 19.45 | T | X | 17.84 | 16.48 | 8.45 | 39.77 | 39.81 | 40.80 | 62.91 | 76.96 |
| grid3d_250 | 26.45 | T | T | 30.81 | 31.77 | 12.86 | 46.47 | 55.91 | 58.84 | 111.15 | 166.03 |
| grid_5000 | 29.60 | T | T | 25.36 | 25.13 | 12.58 | 62.17 | 57.95 | 65.65 | 109.48 | 129.91 |

### ipm

| matrix | apxchol | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (MATLAB)† | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|
| iter0030 | 1.43 | 8.98 | 3.37 | 1.23 | 1.50 | 3.59 | 2.45 | 4.35 | 2.86 | 5.12 | 6.85 |
| iter0040 | 1.48 | 10.48 | 3.77 | 1.20 | 1.70 | 3.71 | 2.48 | 2.91 | 3.00 | 5.16 | 6.81 |
| iter0010 | 1.39 | 7.44 | 3.12 | 1.55 | 1.23 | 0.88 | 3.29 | 2.26 | 2.42 | 5.10 | 7.23 |
| iter0020 | 1.33 | 7.49 | 3.04 | 1.17 | 1.58 | 3.94 | 2.43 | 2.35 | 2.53 | 4.96 | 6.92 |

### suitesparse

| matrix | apxchol | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (MATLAB)† | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|
| com-Amazon | 0.40 | 11.41 | 2.29 | 1.12 | 1.19 | 0.94 | 1.62 | 1.07 | 1.08 | 1.41 | 1.73 |
| coAuthorsDBLP | 0.40 | 13.56 | 2.80 | 2.49 | 1.65 | 1.99 | 1.43 | 0.91 | 0.94 | 1.19 | 1.81 |
| parabolic_fem | 0.62 | 7.24 | 1.71 | 0.67 | 0.71 | 0.41 | 1.78 | 1.81 | 1.37 | 2.64 | 3.22 |
| apache2 | 0.75 | 5.21 | 2.58 | 0.74 | 0.78 | 0.51 | 1.97 | 2.08 | 1.85 | 2.77 | 4.16 |
| kron_g500-logn16 | 0.90 | 6.52 | 2.47 | 2.13 | 1.31 | 0.17 | 1.40 | 1.69 | 1.61 | 2.92 | 3.85 |
| ecology1 | 0.93 | 6.36 | 3.35 | 0.93 | 0.90 | 0.37 | 2.28 | 2.04 | 1.96 | 2.99 | 3.91 |
| com-Youtube | 1.56 | T | T | 211.98 | 7.31 | 8.48 | 6.79 | 7.61 | 7.61 | 3.84 | 5.47 |
| G3_circuit | 1.65 | 11.48 | 4.99 | 1.75 | 1.73 | 1.12 | 4.95 | 4.52 | 4.45 | 4.76 | 6.28 |
| thermal2 | 1.37 | 10.37 | 5.06 | 2.69 | 2.65 | 0.92 | 4.89 | 3.53 | 3.52 | 5.78 | 6.96 |
| as-Skitter | 4.43 | T | T | 275.01 | 24.81 | 22.42 | 19.71 | 27.89 | 27.63 | 13.67 | 18.98 |
| coPapersDBLP | 6.46 | T | 15.49 | 40.67 | 33.86 | 10.60 | 15.25 | 6.98 | 6.61 | 14.35 | 23.49 |
| com-LiveJournal | 32.04 | T | T | 439.53 | 251.05 | 287.72 | 142.74 | 789.15 | 879.70 | 77.09 | 150.47 |
| com-Orkut | 120.82 | X | T | X | 1181.65 | 26.65 | 151.22 | T | T | 262.12 | 566.55 |

## PCG iterations (preconditioner quality, threads-independent)


### grids

| matrix | apxchol | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (MATLAB)† | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|
| grid_500 | 41 | 34 | 41 | 8 | 8 | 14 | 25 | 50 | 50 | 36 | 24 |
| grid_1000 | 43 | 40 | 57 | 7 | 7 | 12 | 26 | 56 | 54 | 41 | 25 |
| grid3d_100 | 28 | 22 | 25 | 8 | 8 | 13 | 26 | 34 | 34 | 27 | 20 |
| grid_2000 | 48 |  |  | 8 | 8 | 13 | 25 | 61 | 58 | 45 | 25 |
| grid3d_150 | 29 |  |  | 8 | 8 | 13 | 28 | 36 | 36 | 27 | 20 |
| grid_3000 | 50 |  |  | 7 | 7 | 13 | 26 | 65 | 64 | 45 | 27 |
| grid3d_200 | 29 |  |  | 11 | 11 | 13 | 26 | 37 | 37 | 29 | 21 |
| grid_4000 | 50 |  |  | 7 | 7 | 14 | 25 | 69 | 72 | 49 | 27 |
| grid3d_250 | 32 |  |  | 8 | 8 | 15 | 28 | 38 | 38 | 28 | 21 |
| grid_5000 | 51 |  |  | 7 | 7 | 14 | 26 | 72 | 74 | 50 | 27 |

### ipm

| matrix | apxchol | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (MATLAB)† | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|
| iter0030 | 40 | 45 | 48 | 10 | 38 | 20 | 32 | 93 | 49 | 42 | 23 |
| iter0040 | 45 | 52 | 59 | 10 | 47 | 19 | 34 | 55 | 53 | 42 | 23 |
| iter0010 | 29 | 36 | 36 | 10 | 22 | 24 | 64 | 37 | 37 | 33 | 18 |
| iter0020 | 30 | 34 | 33 | 10 | 38 | 19 | 30 | 38 | 39 | 30 | 19 |

### suitesparse

| matrix | apxchol | RCHOL | pRCHOL | BoomerAMG | BoomerAMG/cut | AMGCL | CMG (MATLAB)† | ParAC Graph | ParAC Physics | AC (Jl ref)† | AC2 (Jl ref)† |
|---|---|---|---|---|---|---|---|---|---|---|---|
| com-Amazon | 33 | 90 | 75 | 14 | 49 | 74 | 44 | 31 | 31 | 30 | 23 |
| coAuthorsDBLP | 24 | 104 | 116 | 16 | 33 | 102 | 48 | 24 | 25 | 20 | 17 |
| parabolic_fem | 41 | 49 | 49 | 8 | 8 | 20 | 26 | 60 | 48 | 41 | 24 |
| apache2 | 26 | 25 | 25 | 8 | 8 | 17 | 22 | 49 | 40 | 24 | 17 |
| kron_g500-logn16 | 14 | 37 | 28 | 6 | 14 | 16 | 12 | 12 | 12 | 9 | 9 |
| ecology1 | 44 | 35 | 48 | 8 | 8 | 12 | 26 | 57 | 54 | 42 | 25 |
| com-Youtube | 19 |  |  | 17 | 110 | 69 | 68 | 20 | 20 | 16 | 12 |
| G3_circuit | 41 | 40 | 38 | 9 | 9 | 19 | 39 | 62 | 60 | 38 | 23 |
| thermal2 | 41 | 45 | 45 | 9 | 9 | 20 | 26 | 47 | 43 | 38 | 24 |
| as-Skitter | 20 |  |  | 29 | 219 | 270 | 85 | 29 | 21 | 18 | 14 |
| coPapersDBLP | 36 |  | 51 | 21 | 34 | 174 | 55 | 26 | 24 | 21 | 16 |
| com-LiveJournal | 31 |  |  | 19 | 151 | 431 | 111 | 26 | 26 | 23 | 18 |
| com-Orkut | 12 |  |  |  | 32 | 48 | 28 |  |  | 12 | 10 |
