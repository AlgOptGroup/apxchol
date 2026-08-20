# Benchmarks

Comparative benchmarks for Laplacian / SDDM linear-system solvers, comparing our
approximate-Cholesky preconditioner against third-party direct, randomized, and
multigrid solvers.

The benchmark's defining feature is a **fair comparison**: every solver attacks
the *same* operator and is judged by the *same* true residual (see
[Protocol](#protocol)). The latest results live in [`latest/`](latest/) and are
committed so the charts render on GitHub — see [`latest/README.md`](latest/README.md)
for methodology and [`latest/summary.md`](latest/summary.md) for the tables.

## Solvers

| Solver | Type | Threads | Source |
|---|---|---|---|
| **apxchol** | Approximate Cholesky + PCG; headline = best of 4 IS-selectors (bg/luby/root/bk) per matrix/device | 16 (C++) | this repo (`src/`, `include/apxchol/`) |
| ↳ IS selectors | bg=block-greedy, luby, root=Blelloch rootset, bk=Baumann-Kyng; luby/root give shallow factors that the GPU SpTRSV prefers (per-selector spread in the ablation chart) | 16 (C++) | this repo |
| **RCHOL** | Randomized Cholesky + PCG (serial factorization) | serial factor | [ut-padas/rchol](https://github.com/ut-padas/rchol) |
| **BoomerAMG** | Classical *algebraic* multigrid + PCG | 16 (C, OpenMP) | [Hypre](https://github.com/hypre-space/hypre) |
| **AMGCL** | Smoothed-aggregation *algebraic* multigrid + PCG; Dirichlet-pin de-singularization, default config | 16 (C++) | [ddemidov/amgcl](https://github.com/ddemidov/amgcl) |
| **ParAC** | Parallel randomized Cholesky + PCG (AMD-reordered, MKL) | 16 (C++) | [Tianyu-Liang/Parallel-Randomized-Cholesky](https://github.com/Tianyu-Liang/Parallel-Randomized-Cholesky) |
| **CMG (MATLAB)** † | *Combinatorial* multigrid + PCG | serial (MATLAB) | [ikoutis/cmg-solver](https://github.com/ikoutis/cmg-solver) |
| **AC / AC2** | ApproxChol — Julia *reference* of our method | serial (Julia) | [Laplacians.jl](https://github.com/danspielman/Laplacians.jl) |

BoomerAMG (algebraic multigrid) and CMG (combinatorial multigrid, the closest
cousin to our approximate-Cholesky — both from the Spielman–Teng / support-theory
line) are the two multigrids kept.

**Status notes:**
- **AMGCL** (smoothed-aggregation AMG) is de-singularized with the same symmetric
  **Dirichlet pin** as the other algebraic multigrids (→ SPD → AMGCL defaults, *direct*
  coarse solve) and run with its default config.
- **†** `CMG` runs as the **canonical MATLAB CMG** (Koutis `cmg-solver`, MEX
  recompiled for R2026a, in the `matlab-deps` container). Its MATLAB-`pcg`
  wall-time isn't cross-language-comparable to the C++ solvers, so its
  **iteration count** is the comparable signal (use it, not the wall-time).
- **CHOLMOD** (direct supernodal Cholesky) is out of scope — the suite compares
  preconditioned iterative methods. `CG/LDLT [Eigen]` and the apxchol IS/storage
  ablations (`bk/luby/root`, `vec`) exist in the binary but are not in the headline set.

## Protocol

- **Matrix interpretation (what system each file defines).** A `.mtx` file can hold
  either of two different things, and they define two different linear systems. Every
  entry in the registry (`benchmarks/runner_common.py`) **declares** which, and the
  benchmark binary **requires** the declaration on the command line (`--kind`). It is
  never auto-detected: a value-carrying file can perfectly well be a graph
  (`kron_g500-logn16` stores integer *edge weights*), so any heuristic would be a
  silent decision about which problem the suite reports on. An undeclared matrix is a
  hard error, not a default.

  | kind | what the file holds | the system we solve |
  |---|---|---|
  | `graph` | adjacency / `pattern` matrix (no operator diagonal) | `L = D − A`, assembled from `|value|` (unit weights for a `pattern` file; a self-loop on the file's diagonal is dropped). That **is** the system a graph file defines, so it is not a deviation. |
  | `operator` | already-assembled Laplacian / SDDM operator | the published matrix itself — **solved as it stands**, diagonal included, signs untouched. |

  | matrix | kind | | matrix | kind |
  |---|---|---|---|---|
  | `grid_*`, `grid3d_*` (generated) | `graph` | | `com-Amazon` | `graph` |
  | `parabolic_fem` | `operator` | | `coAuthorsDBLP` | `graph` |
  | `apache2` | `operator` | | `kron_g500-logn16` | `graph` |
  | `ecology1` | `operator` | | `com-Youtube` | `graph` |
  | `G3_circuit` | `operator` | | `coPapersDBLP` | `graph` |
  | `thermal2` | `operator` | | `as-Skitter` | `graph` |
  | `iter0010…0040` (IPM) | `operator` | | `com-LiveJournal` | `graph` |
  | | | | `com-Orkut` | `graph` |

  The five SuiteSparse `operator` files carry a stored positive diagonal on every row;
  the `graph` files carry no usable diagonal at all (`kron_g500` has one on 327 of
  65536 rows — self-loops, not an operator). ParAC's own benchmark splits its matrices
  the same way ("physics matrices: used as-is" vs "graph matrices: build Laplacian from
  adjacency"), and four of our five operator files are on its physics list.

  Every solver in a cell receives the **same** matrix: the in-process ones directly,
  the external ones (ParAC, AC/AC2, CMG) through `--dump-mtx`, which writes exactly the
  operator the binary assembled. Each run prints one `[matrix] …` line saying how the
  file was read, and each stored cell repeats it under `matrix_meta`. Solvers that
  cannot take a given matrix produce an **`n/a`** cell (`iters = rel_res = −1`), never a
  silent failure or a missing row — for example ParAC's *graph* mode on an `operator`
  matrix (graph mode grounds a Laplacian; the matrix goes through ParAC's *physics*
  path instead), or AC/AC2 on an operator carrying positive off-diagonals (Laplacians.jl
  needs non-negative edge weights, and it solves the matrix it factorizes, so lumping
  them would make it converge on a different matrix than it is scored against). `n/a` is
  reserved for *cannot take this matrix*: a solver that runs and misses the tolerance is
  `not_converged` — AC/AC2 do reach `apache2` and `G3_circuit` through
  `approxchol_sddm`, but floor at 6.5e-6 / 1.4e-7 because the ground-vertex
  augmentation gives those two negative-weight edges where their row sums go negative.

  Whether the assembled operator is **singular** is a separate, downstream fact, read off
  the matrix rather than declared: a graph always yields a singular Laplacian, and an
  `operator` file may be one (`ecology1` is published as an exact Laplacian) or full-rank
  SDDM (`apache2`, `G3_circuit`, `thermal2`, IPM). Only a singular operator is grounded
  and only its solution and residual are mean-centred.

- **De-singularization (the fairness model).** Grid / SuiteSparse Laplacians are
  **singular** (a constant null vector per connected component). Under the **current**
  protocol they are benchmarked on the **original singular `L`** — each solver removes
  that null space the way its own machinery prefers and is scored on the *true*
  residual `‖b − L x‖` (so the multigrid solvers no longer floor at ~1e-4 from a
  pin-vs-score mismatch):
  **apxchol** native rank-aware solve (mean-centring); **BoomerAMG** a
  provably-safe (DFS-tree-leaf) Dirichlet pin, one node per connected component → SPD;
  **AMGCL** the same Dirichlet pin, run with its default config.
  **ParAC grounds itself**, and is given the input its own scripts build (see
  [`patches/parac/README.md`](patches/parac/README.md) for the measurements behind
  this). A `kind=graph` matrix goes to its **graph** driver as the **pure singular
  `L`**, one connected component at a time: graph mode generates its own zero-sum
  RHS, which is consistent for a connected Laplacian, so it solves the very `L` we
  report on and its printed `‖b − L·x‖ / ‖b‖` is against that `L`. We do **not**
  hand it a Dirichlet-pinned matrix — its RHS generator knows nothing about the
  pin, and the residual against the original `L` then floors at ~1e-3 (com-Amazon
  1.4e-3, coAuthorsDBLP 2.0e-3) no matter how tight the tolerance. A
  `kind=operator` matrix goes to its **physics** driver as the **published
  operator**, AMD-reordered and then augmented with the ground row/column exactly
  as ParAC's own `write_graph.jl physics_produce` does; physics mode's trim removes
  that appended node, so what it solves is the published operator itself. Each
  matrix runs **one** mode and the other cell is `n/a` with the reason recorded.
  ParAC's stopping test is absolute (`‖r‖` vs `sqrt(rel_tol)`) and on the
  recurrence residual, so the tolerance is **calibrated from one probe run** —
  configuration, not a patched convergence test — until its own printed residual
  lands under 1e-8. The same competitor wall-clock cap applies to ParAC:
  on **com-Orkut** it exceeds the shared ~20 min competitor cap during setup and is
  recorded as `timeout`, like rchol_par/rchol on the same matrix. **CMG** still keeps the `ε·I`
  regularization (`A = L + ε·I`, `ε = 1e-6·mean|diag|`, via `--reg-rel`), scored on `L + εI`.
- **Which protocol each committed cell was measured under (mixed).** The store holds
  **1048** cells recorded over two protocols; every cell carries its own provenance note.
  The pin protocol above is the **current** one and covers **essentially the whole GPU
  axis** (237 of the 249 GPU cells) plus part of the CPU axis — 216 cells under the plain
  pin note (183 GPU + 33 CPU) plus the 40 re-run ParAC cells scored against the original
  `L`, and the 54 ParAC-GPU cells run by the same drivers. The **majority of the cells —
  697, and almost all of them CPU (260 grids, 104 IPM, 321 SuiteSparse; 12 GPU) — predate
  it** and come from the earlier
  **unified regularization**: *every* solver on the same `L + ε·I` operator
  (`ε = 1e-6·mean|diag|`), scored on that operator. That protocol is *internally consistent*
  (one matrix, one score, all solvers) — it answers "same regularized matrix" where the
  current one answers "same singular matrix". They are not interchangeable in principle, but
  re-running solvers under both moved **iteration counts by only ~10%** at identical setup
  cost, which is inside the session-variance envelope (see [Machine](#machine)) — so read
  cross-protocol cells as comparable to within that band, not as exact. The remaining cells
  are CMG (27, `ε·I`, scored on `L + εI`) and an older ParAC set (14).
- **All third-party backends read `.mtx`.** Synthetic grids are dumped once via
  `--dump-mtx` and fed to the Julia / MATLAB / external solvers, so they consume the
  *exact* same matrix the C++ solvers build (no per-backend grid generators — e.g.
  3D grids work without a Julia `grid3d` generator).
- **Tolerance**: true relative residual `‖b − A x‖ / ‖b‖ ≤ 1e-8`. The runner's
  acceptance band is one order looser than that — a cell is recorded `complete` at
  `rel_res ≤ 10·tol` (1e-7) and `not_converged` above it — but in the committed store
  **every** `complete` cell is in fact at or below 1e-8 (largest is 9.996e-9), so the
  headline "converged" set means true 1e-8.
- **Reps**: 3, median — except the **CMG** cells, which are single-shot (`repeat=1`,
  one MATLAB run per matrix). **Threads**: 16 physical cores, pinned, run without contention.
- **Metrics**: setup_s, solve_s, total_s, PCG iterations, rel_res, µs/nnz.
- **Disconnected matrices (multi-component, e.g. the social giants `as-Skitter`,
  `kron_g500`, `thermal2`).** Solvers handle the per-component null space two ways.
  **apxchol** (native mean-centring) and **AMGCL** (+pin, its default) solve the **whole**
  disconnected operator in one shot — faster than paying per-component build overhead.
  **BoomerAMG** (and coarse-grounded AMGCL) instead **split**: decompose into connected
  components, solve each independently (per-component Dirichlet pin → SPD), and recombine
  the cell as **setup_s, solve_s = SUM** over components (plus the union-find decomposition
  + sub-matrix extraction + recombine overhead, charged to **setup**); **iters = MAX** over
  components (the bottleneck block — summing would let hundreds of trivial singleton specks
  swamp the count); **rel_res = √(Σ‖res_c‖²)/‖b‖** (valid because `L` is block-diagonal
  across components, so per-component residuals are independent); **peak RSS / VRAM = MAX**
  over components (peak resident during the largest sub-solve). A connected matrix is a
  single component, so the split path is a no-op.
- **CSV schema**: `solver,graph,n,nnz,setup_s,solve_s,total_s,iters,rel_res,fillin,us_per_nnz`.

### Machine

Every committed number was measured on one machine: **AMD Ryzen 9 7945HX** (16 cores /
32 threads, Zen 4), **128 GB** RAM, **NVIDIA RTX 4090 Laptop GPU (16 GB)**, Linux, GCC.
Runs are **boost-on** (no frequency lock), which on this laptop gives a **±20–30%
session-to-session spread** on the same commit and config — so treat only ratios well
above that band as signal, not individual cells or small deltas. The locked-frequency
harness for paired A/B work is in `benchmarks/dev/` (see [Timing harness](#timing-harness)).

### Workloads (current run)

- **Grids** — 2D square (`grid_500…grid_5000`, κ=100 *weighted*, not uniform Poisson)
  and 3D 7-point (`grid3d_100…grid3d_250`).
- **IPM** — interior-point normal-equation matrices (`iter0010…0040`).
- **SuiteSparse** — planar/PDE (`parabolic_fem`, `apache2`, `ecology1`, `G3_circuit`,
  `thermal2`) + smaller scale-free/social (`com-Amazon`, `coAuthorsDBLP`, `kron_g500-logn16`).
  The charts split SuiteSparse into **`_small`** (these) and **`_giants`** (the large
  social graphs) so the giants don't flatten the FEM matrices.

Both **CPU and GPU** axes are run. The 5 social **giants** (`as-Skitter`,
`coPapersDBLP`, `com-LiveJournal`, `com-Orkut`, `com-Youtube`) are covered on **both**
axes — 9 GPU cells each, essentially all under the current pin protocol. What is still
outstanding for them is the **CPU** side: 117 of their 139 CPU cells carry the earlier
unified-`reg_rel` provenance and have not been re-run under the pin protocol, so the
CPU `_giants` panels mix protocols (see the provenance bullet above).

## Running

**Input matrices.** Grid instances are generated in-process. The SuiteSparse
matrices come from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu);
`./scripts/download_graphs.sh` (repo root) fetches every one used by the sweep
into `data/matrices/`. The `ipm_*` cells use a ladder of LP interior-point
Schur-complement matrices (`data/ipm/iterNNNN/matrix.mtx`) that is not
redistributed with the repo — without the files those cells error out (the
sweep records the failure and continues); contact the authors for the dataset
or point `data/ipm/` at your own IPM sequence (Matrix Market, one directory
per iterate).

**Prerequisites.** Only Eigen is mandatory; every competitor is behind a CMake gate and
is silently skipped when its dependency is missing (configure prints which):

| Solver / axis | Needs |
| --- | --- |
| `rchol_par` (pRCHOL) | system **METIS** (`libmetis` + `metis.h`) |
| `cholmod` | system **CHOLMOD** (SuiteSparse) |
| `amgcl`, `amgcl_cuda` | **Boost** headers (AMGCL only; the apxchol core is Boost-free) |
| `rchol` PCG, ParAC CPU driver | **MKL** (`-DMKL_ROOT=…`, `MKLROOT`, or `paths_local.cmake`) |
| GPU axis | **CUDA** + `-DAPXCHOL_USE_CUDA=ON`; Hypre-GPU also needs `-DBENCH_HYPRE_USE_CUDA=ON`, ParAC needs `-DBUILD_GPU_RCHOL=ON` |
| `cmg`, `parac`, `ac`/`ac2` | out-of-tree tools — see [Local configuration](#local-configuration-out-of-tree-tools) |

BoomerAMG (Hypre) and AMGCL are pulled in by FetchContent, so they need network
access on the first configure.

```bash
# Build (root library + benchmarks are separate CMake projects)
cmake -B benchmarks/build -S benchmarks -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/build -j$(nproc) benchmark

# One solver on one matrix (original singular L; BoomerAMG/AMGCL self-ground via the pin)
benchmarks/build/benchmark --graph grid --n 2000 \
    --solver hypre_boomeramg --threads 16 --tol 1e-8 --repeat 3 --csv

# Dump a matrix to .mtx (for external solvers / ParAC)
benchmarks/build/benchmark --graph grid3d --n 100 --dump-mtx /tmp/g.mtx --solver none

# The single fair CPU sweep — ONE runner for grids + SuiteSparse + IPM.
# Original singular L; each solver self-grounds (--only <ids> refreshes a subset).
# ParAC (graph+physics) runs IN-PROCESS as part of the sweep (parac_runner.py);
# --no-parac skips it, --parac-only runs only it. Shared harness: runner_common.py.
python3 benchmarks/sweep_fair.py
# ParAC alone (AMD-reorder + driver), standalone:
python3 benchmarks/parac_runner.py --device cpu  # [--only id1,id2]
# CMG (canonical MATLAB CMG in the matlab-deps container) — its own runner:
python3 benchmarks/cmg_matlab_runner.py
# CPU charts + tables, then the thread-scaling charts:
PYTHONPATH=benchmarks python3 benchmarks/fair_charts.py --out benchmarks/latest
python3 benchmarks/thread_scaling.py
# Cholesky-family fill (one-time, factor-only — measures factor density, no solve):
python3 benchmarks/fill_pass.py
python3 benchmarks/fill_chart.py --out benchmarks/latest/figures
# Per-(selector, matrix) SpTRSV back-solve level counts + the selector_levels figure:
python3 benchmarks/selector_levels.py
# SpTRSV level structure per matrix (level counts / work concentration):
python3 benchmarks/level_stats.py
```

For the **GPU axis** (same runner, same per-cell store — see
[Results](#results)) build the benchmark with CUDA, then run the fair
sweep with `--device gpu` (apxchol uses the GPU-resident PCG) and re-render:

```bash
cmake -B benchmarks/build-cuda -S benchmarks -DCMAKE_BUILD_TYPE=Release \
      -DAPXCHOL_USE_CUDA=ON -DBENCH_HYPRE_USE_CUDA=ON -DBUILD_GPU_RCHOL=ON
cmake --build benchmarks/build-cuda -j$(nproc) benchmark
# BENCH_HYPRE_USE_CUDA (Hypre GPU backend) and BUILD_GPU_RCHOL (FetchContents the
# ParAC repo and builds its CPU+GPU drivers) both default OFF; without them the
# hypre_*_gpu and ParAC-GPU cells can't run.
# GPU cells written into the SAME results/cells store (device=gpu), resume-safe.
# ParAC-GPU (both CUDA drivers; needs the ParAC repo + drivers built) runs
# in-process as part of the sweep, or alone via parac_runner.py --device gpu:
python3 benchmarks/sweep_fair.py --device gpu
# render GPU-only + combined CPU-vs-GPU figures from the store:
python3 benchmarks/gpu_charts.py --root results/cells --out benchmarks/latest/figures
PYTHONPATH=benchmarks python3 benchmarks/combined_charts.py --out benchmarks/latest/figures
```

ParAC is charted as **two drivers on both devices** — graph (Laplacian) and
physics (SDDM). The same self-contained binary runs both: on GPU via driver.cu /
driver_physics.cu, on CPU via one extra argv (`is_graph=0`). **Each matrix runs
exactly one of them**, the one ParAC documents for it: `kind=graph` → graph
(pure `L`, its own zero-sum RHS), `kind=operator` → physics (published operator
plus ParAC's ground-node augmentation, which the mode's trim removes). The other
cell is `n/a` and carries the reason, so a chart shows a deliberate gap rather
than a solver that silently vanished. Running the *other* mode is not a second
data point but a wrong answer: physics on an un-augmented operator deletes a real
degree of freedom (apache2 scores 3.1e-3 against the published matrix while ParAC
prints 8.8e-9; G3_circuit does not converge at all). Note: ParAC's GPU SpTRSV is extremely
sensitive to the elimination ordering — its required nnz-sort is a **random
permutation then a sort by column-nnz** (`parac_nnz_sort.jl`, matching their
`write_graph.jl`). A deterministic degree-sort instead builds a very deep
elimination tree and makes the cuSPARSE level-set SpTRSV ~1000× slower
(1.5 s/iter vs ~1 ms/iter on a 250k-node grid). Setup is counted as
`factorization + reorder-compute` (no disk I/O, no CSR-conversion), the same way
the CPU ParAC counts its AMD reorder, so all families are measured consistently.

**Binary solver names**: `apxchol_v1` (with `--v1-configs bg+tree[vec_pool]`),
`rchol`, `rchol_par`, `hypre_boomeramg`, `amgcl`, `cholmod`, `cg`,
`ldlt`. GPU variants (CUDA build): `hypre_boomeramg_gpu`,
`amgcl_cuda`, plus apxchol's GPU-resident PCG (the default on CUDA builds).
External: `ac`/`ac2` (Julia), `cmg` (canonical MATLAB CMG via `cmg_matlab_runner.py` + the matlab-deps container), `parac` (driver via `parac_runner.py`).

**Key flags**: `--reg-rel <c>` (SDDM regularization), `--dump-mtx <path>`,
`--threads`, `--tol`, `--repeat`, `--mtx <path>`, `--graph {grid,grid3d,checkerboard,erdos}`.

### Local configuration (out-of-tree tools)

Some competitors live outside this repository, so their paths are **not** hardcoded:

| What | Environment variable | `paths_local.py` name |
| --- | --- | --- |
| ParAC CPU driver (`experiment/driver`) | `APXCHOL_PARAC_DRIVER` | `PARAC_CPU_DRIVER` |
| ParAC AMD-reorder cache directory | `APXCHOL_PARAC_REORDER_DIR` | `PARAC_REORD` |
| ParAC `reorder_amd.jl` | `APXCHOL_PARAC_REORDER_JL` | `REORDER_JL` |
| Runtime libs the ParAC driver needs | `APXCHOL_PARAC_LDLIB` | `PARAC_LDLIB` |
| MATLAB install tree (CMG) | `APXCHOL_MATLAB_ROOT` | `MATLAB` |
| `cmg-solver` checkout (CMG) | `APXCHOL_CMG_SOLVER` | `CMG_SOLVER` |

Set the environment variables, or drop the same names into a **gitignored**
`benchmarks/paths_local.py` (assignments there win over the environment). Leaving
them unset is fine until a ParAC/CMG cell actually runs; the runner then fails with
a message naming the variable to set. Everything else — the repo root, the benchmark
binaries, the cell store — is derived from the source tree, so no configuration is
needed for the in-tree solvers.

**MKL** (optional; it backs the RCHOL PCG solve and the ParAC CPU driver) is found
via `-DMKL_ROOT=/path/to/oneapi/mkl/<version>`, the `MKLROOT` environment variable,
or a gitignored `benchmarks/paths_local.cmake` containing
`set(MKL_ROOT "..." CACHE PATH "")`. Without it, configure prints
`MKL not found ...` and the MKL-dependent targets are skipped.

### Timing harness

`benchmarks/dev/` holds the thermal-stable timing harness used for perf A/B work:
`bench_stable_setup.sh` (disable boost / pin the governor; needs root),
`bench_stable.sh` (warmup + N-rep median with cooldown gating), `bench_stable_all.sh`
(the standard workload set, CPU + CUDA), `bench_reference_table.sh` (the cross-solver
reference table) and `bench_stable_teardown.sh` (restore the CPU state).

## Results

`benchmarks/latest/` holds the committed charts/tables. The **comparison** figures all
read the one committed per-cell store (`results/cells`): CPU figures from
`fair_charts.py`, GPU figures from `gpu_charts.py` (device=gpu cells), and the
CPU-vs-GPU overlays from `combined_charts.py`. The remaining figures read their own
stores, which are **gitignored but regenerable** by the commands in [Running](#running):
fill from `results/fill_cells` (`fill_pass.py` → `fill_chart.py`), thread scaling from
`results/scaling_cells` (`thread_scaling.py`), and the SpTRSV level-count chart from
`results/selector_levels.csv` (`selector_levels.py`). Raw data:
[`latest/results.csv`](latest/results.csv). Note the `[svec]` rows there (and the matching
cells) are **historical**: that storage backend has since been removed from the library and
can no longer be re-measured.

Each headline section below is shown three ways: a **combined** CPU-vs-GPU overlay
(solid = CPU t16, `///` = GPU; one colour per method), then the **CPU-only** and
**GPU-only** views with the full solver set for that device. CPU and GPU wall times
are different hardware, so the combined view is for per-method "what does the GPU
buy", not a single ranking. Every solver attacks the original singular `L`
(de-singularized per the [Protocol](#protocol); every solver — the in-house ones *and*
ParAC, which now uses a per-component-consistent pin-zeroed RHS — scores `‖b−Lx‖` against
the original L, only CMG uses the `εI`-regularized variant) to true 1e-8. On apxchol, the GPU bar is the **GPU-resident PCG** (
all PCG vectors stay on device; since 2026-08-18 our own SpMV / vector kernels, no cuBLAS / cuSPARSE), which is 2-4.5× faster per iter
than the host-PCG hybrid (measured on an RTX 4090 Laptop at locked clocks).

GitHub markdown has no tabs, so each metric uses collapsible `<details>` blocks —
click to switch between the **Combined**, **CPU**, and **GPU** views. Bars are
**sorted fastest→slowest within each matrix group**; matrices appear in the same
nnz-ascending order in every figure (for grids only the largest 2D and largest 3D
grid are shown per-matrix; the full ladder is in the scaling chart). Non-converged
runs are **excluded** from the time/iteration charts (only in the accuracy chart).

### Setup + solve time (t16)

<details open><summary><b>Combined — CPU vs GPU per method</b></summary>

Per method, the **GPU bar is on the left (outlined in black)** and the **CPU bar on
the right (plain)**; each bar stacks setup (solid) + solve (`///`). Total time is just
the bar height. (Grids reduce to the largest 2D + largest 3D here — grouped bars get
cramped past a handful; the full grid ladder is in the heatmaps below.)

![Grids 2D combined breakdown](latest/figures/combined_breakdown_grids_2d.png)
![Grids 3D combined breakdown](latest/figures/combined_breakdown_grids_3d.png)
![IPM combined breakdown](latest/figures/combined_breakdown_ipm.png)
![SuiteSparse small combined breakdown](latest/figures/combined_breakdown_suitesparse_small.png)
![SuiteSparse giants combined breakdown](latest/figures/combined_breakdown_suitesparse_giants.png)
![SuiteSparse XL giants combined breakdown](latest/figures/combined_breakdown_suitesparse_giants_xl.png)

(Grids show the largest **two** 2D + **two** 3D sizes — the very largest, grid_5000 /
grid3d_250, OOM Hypre-GPU, so the next size down is included to keep a Hypre-GPU bar.
The social giants split into mid — com-Youtube / as-Skitter / coPapersDBLP — and **XL**
— com-LiveJournal / com-Orkut — since the two heaviest dwarf the rest on a shared axis.)

The setup-only and solve-only cuts are **heatmaps** (matrix × method, every measured
grid): colour = time relative to the best solver in that matrix (per column, green =
fastest, **log** scale), each cell annotated with the absolute time over the ratio.

| setup only | solve only |
|---|---|
| ![grids setup](latest/figures/combined_setup_grids.png) | ![grids solve](latest/figures/combined_solve_grids.png) |
| ![ipm setup](latest/figures/combined_setup_ipm.png) | ![ipm solve](latest/figures/combined_solve_ipm.png) |
| ![ss setup](latest/figures/combined_setup_suitesparse.png) | ![ss solve](latest/figures/combined_solve_suitesparse.png) |

Heatmap cell legend: a value/×best is shown for solvers that completed; **`Timeout`** =
hit the *total* wall-clock cap, so there's no setup/solve breakdown — the **total**
heatmap fairly clamps it to `≥cap`, the setup/solve cuts just mark it `Timeout`, and in
the bar charts it appears **last** as a hatched red-edged `≥cap` bar; **`OOM`** = ran out
of memory (RAM = host, GPU = device); a **blank/white** cell = never run.
</details>

**Solver × matrix overview** — one grid per family, colour = total time relative to
the best solver in that matrix (per column; green = winner, **log** scale so the slow
tier doesn't saturate to one red wall), each cell annotated with absolute total (s)
over the ratio. The at-a-glance "who wins where" companion to the per-matrix grouped
bars. The headline **combined** grid puts every (solver, device) on its own row, so a
GPU variant beating its CPU twin is visible and the per-column winner is taken across
both devices; CPU-only and GPU-only variants are in the dropdown.

![Grids overview](latest/figures/combined_overview_grids.png)
![IPM overview](latest/figures/combined_overview_ipm.png)
![SuiteSparse overview](latest/figures/combined_overview_suitesparse.png)

<details><summary><b>CPU-only / GPU-only overview</b></summary>

![Grids overview CPU](latest/figures/combined_overview_cpu_grids.png)
![Grids overview GPU](latest/figures/combined_overview_gpu_grids.png)
![IPM overview CPU](latest/figures/combined_overview_cpu_ipm.png)
![IPM overview GPU](latest/figures/combined_overview_gpu_ipm.png)
![SuiteSparse overview CPU](latest/figures/combined_overview_cpu_suitesparse.png)
![SuiteSparse overview GPU](latest/figures/combined_overview_gpu_suitesparse.png)
</details>

<details><summary><b>CPU only</b> (the combined chart, CPU rows only)</summary>

![Grids 2D CPU breakdown](latest/figures/combined_breakdown_cpu_grids_2d.png)
![Grids 3D CPU breakdown](latest/figures/combined_breakdown_cpu_grids_3d.png)
![IPM CPU breakdown](latest/figures/combined_breakdown_cpu_ipm.png)
![SuiteSparse small CPU breakdown](latest/figures/combined_breakdown_cpu_suitesparse_small.png)
![SuiteSparse giants CPU breakdown](latest/figures/combined_breakdown_cpu_suitesparse_giants.png)
![SuiteSparse XL giants CPU breakdown](latest/figures/combined_breakdown_cpu_suitesparse_giants_xl.png)
</details>

<details><summary><b>GPU only</b> (the combined chart, GPU rows only)</summary>

![Grids 2D GPU breakdown](latest/figures/combined_breakdown_gpu_grids_2d.png)
![Grids 3D GPU breakdown](latest/figures/combined_breakdown_gpu_grids_3d.png)
![IPM GPU breakdown](latest/figures/combined_breakdown_gpu_ipm.png)
![SuiteSparse small GPU breakdown](latest/figures/combined_breakdown_gpu_suitesparse_small.png)
![SuiteSparse giants GPU breakdown](latest/figures/combined_breakdown_gpu_suitesparse_giants.png)
![SuiteSparse XL giants GPU breakdown](latest/figures/combined_breakdown_gpu_suitesparse_giants_xl.png)
</details>

### Memory — peak and solve-held RSS (t16)

Two **heatmaps** per family (matrix × method): colour = resident memory relative to the
**least** in that matrix (per column, green = lightest, **log** scale), each cell
annotated in GB over the ratio. **Peak RSS** is the high-water mark over the whole run
(setup-dominated — the factorization/coarsening peak); **solve-held RSS** is the memory
still resident during the PCG solve (the operator + factor that has to stay live). `OOM`
marks a solver that exceeded the host memory cap (e.g. serial RCHOL's natural-order
fill on com-Orkut hits ~110 GB); a blank cell = never run.

These are **host** RSS and show **CPU solvers only** — a GPU solver's host RSS is not its
real footprint (that lives in **VRAM**, charted separately just below), so including GPU
rows here would misleadingly paint a GPU competitor as memory-light right up until it hits
a GPU-OOM. GPU device-memory pressure shows both in the VRAM heatmap and as `OOM GPU` in
the time heatmaps.

| peak RSS (setup-dominated) | solve-held RSS |
|---|---|
| ![grids peak rss](latest/figures/combined_rss_peak_grids.png) | ![grids solve rss](latest/figures/combined_rss_solve_grids.png) |
| ![ipm peak rss](latest/figures/combined_rss_peak_ipm.png) | ![ipm solve rss](latest/figures/combined_rss_solve_ipm.png) |
| ![ss peak rss](latest/figures/combined_rss_peak_suitesparse.png) | ![ss solve rss](latest/figures/combined_rss_solve_suitesparse.png) |

### GPU memory — peak VRAM (t16)

The device-memory analog of peak RSS, for **GPU solvers only**: one **heatmap** per family
(matrix × method), colour = peak device VRAM relative to the **least** in that matrix
(per column, green = lightest, **log** scale), annotated in GB over the ratio. Peak VRAM is
the high-water mark of the solver's *own* device allocations over the whole run, sampled
per-process (`nvidia-smi --query-compute-apps`), so it excludes the desktop/compositor and
is comparable across solvers. `OOM GPU` marks a solver that exceeded the 16 GB device (e.g.
BoomerAMG on as-Skitter/com-LiveJournal), `FAIL` an error (BoomerAMG/ParAC on com-Orkut).
Note apxchol is the only solver that fits **every** giant on the device — com-Orkut at
~14.3 GB — where the AMG/ParAC competitors OOM or fail. (Per-process sampling can't isolate
the solve *phase* alone, so unlike host RSS there is no separate solve-held VRAM panel.)

![grids peak vram](latest/figures/combined_vram_peak_grids.png)

![ipm peak vram](latest/figures/combined_vram_peak_ipm.png)

![ss peak vram](latest/figures/combined_vram_peak_suitesparse.png)

### PCG iterations (preconditioner quality, threads/device-independent)

**Heatmaps** (matrix × method, every measured grid): colour = iterations relative to
the fewest in that matrix (per column, green = fewest, log scale), each cell annotated
with the absolute iteration count over the ratio. BoomerAMG is the green floor
(near-constant ~7–8); the Cholesky-type solvers sit several × above.

<details open><summary><b>Combined — CPU vs GPU</b></summary>

![Grids combined iters](latest/figures/combined_iters_grids.png)
![IPM combined iters](latest/figures/combined_iters_ipm.png)
![SuiteSparse combined iters](latest/figures/combined_iters_suitesparse.png)
</details>

<details><summary><b>CPU only</b></summary>

![Grids CPU iters](latest/figures/combined_iters_cpu_grids.png)
![IPM CPU iters](latest/figures/combined_iters_cpu_ipm.png)
![SuiteSparse CPU iters](latest/figures/combined_iters_cpu_suitesparse.png)
</details>

<details><summary><b>GPU only</b></summary>

![Grids GPU iters](latest/figures/combined_iters_gpu_grids.png)
![IPM GPU iters](latest/figures/combined_iters_gpu_ipm.png)
![SuiteSparse GPU iters](latest/figures/combined_iters_gpu_suitesparse.png)
</details>

### Solution accuracy (final ‖b−Ax‖/‖b‖; bars above the 1e-8 line did not converge)

<details open><summary><b>CPU</b></summary>

![Grids 2D accuracy](latest/figures/accuracy_grids_2d.png)
![Grids 3D accuracy](latest/figures/accuracy_grids_3d.png)
![IPM accuracy](latest/figures/accuracy_ipm.png)
![SuiteSparse small accuracy](latest/figures/accuracy_suitesparse_small.png)
![SuiteSparse giants accuracy](latest/figures/accuracy_suitesparse_giants.png)
</details>

<details><summary><b>GPU</b></summary>

![Grids GPU accuracy](latest/figures/accuracy_gpu_grids.png)
![IPM GPU accuracy](latest/figures/accuracy_gpu_ipm.png)
![SuiteSparse GPU accuracy](latest/figures/accuracy_gpu_suitesparse.png)
</details>

### Thread scaling (1/2/4/8/16 threads) — setup and solve reported separately

Setup and solve scale very differently (setup parallelizes; the SpTRSV-bound solve
barely does), so they are charted apart. Speedup = t1/tN; efficiency = speedup/N.

**Caveat:** these two charts predate the pin protocol — they were measured on the
`ε·I`-regularized operator (`thread_scaling.py` now runs the singular `L` unshifted, so a
re-run replaces them). Scaling ratios are unaffected to first order, absolute times are not.

![Setup speedup](latest/figures/threads_setup_speedup.png)
![Solve speedup](latest/figures/threads_solve_speedup.png)

### Scaling (total time vs nnz, log-log — grid size ladder only)

| CPU | GPU |
|---|---|
| ![Grids scaling CPU](latest/figures/scaling_grids.png) | ![Grids scaling GPU](latest/figures/scaling_gpu_grids.png) |

### Fill (factor density — Cholesky family only)

Fill = `2·offdiag(L) / offdiag(A)` (factor off-diagonals per input off-diagonal).
This is the **fill-vs-iterations tradeoff** axis: a denser factor is a stronger
preconditioner (fewer PCG iterations) but costs more memory and SpTRSV work. One
consistent definition across the Cholesky-type solvers — AMG (BoomerAMG/AMGCL) has
no triangular factor, so no comparable metric and they are absent here. Series:
**apxchol per IS selector** (bg/luby/root/bk — the order changes the factor density),
the **AC/AC2** Kyng16 reference, **RCHOL/pRCHOL**, and **ParAC** split into
**graph/physics × CPU/GPU** (the CPU AMD-reordered and GPU random-nnz-sort
implementations factor differently, so their fill genuinely differs). apxchol and AC
produce the **sparsest** factors (~1.5–2.2×), ParAC a bit denser (~2.5×), RCHOL/pRCHOL
the densest (up to ~17× on 3D grids). AC2's `(split=2, merge=2)` oversampling makes its
factor visibly denser than AC. (AC/AC2 cells missing from these heatmaps were not swept for fill.)

Each family is a **matrix × method heatmap** (every measured grid; colour = fill
relative to the sparsest factor in that matrix, green = sparsest, log scale; cell =
absolute fill over the ×sparsest ratio).

![Grids fill heatmap](latest/figures/fill_heatmap_grids.png)
![IPM fill heatmap](latest/figures/fill_heatmap_ipm.png)
![SuiteSparse fill heatmap](latest/figures/fill_heatmap_suitesparse.png)

### apxchol internal ablation: IS selector × storage backend

apxchol has two orthogonal design axes — the independent-set **selector** (bg =
block-greedy, luby, root = Blelloch rootset, bk = Baumann-Kyng) and the incidence
**storage backend** (`fwd_star` → `vec` → `bstr`
(bit-string) → `vec_pool`). The full selector×storage sweep is shown as
small-multiple heatmaps (total / setup / solve / iterations), each cell medianed
over the family's matrices (a matrix set common to every config, for fairness),
colour = ×best CPU cell (green = fastest). A trailing `vec_pool (GPU)` column shows
the default backend's CPU→GPU shift (the GPU axis sweeps vec_pool only).

**Takeaway (read across the families):** vec_pool is the default because it is
*robust*, not because it dominates everywhere. It is best-or-tied-best CPU on every
family, and the only backend that doesn't collapse on the **high-degree IPM**
matrices — there `fwd_star`'s per-edge linked-list pointer chase blows setup up to
~3× (median total ≈3.1–3.7 s across the four selectors vs vec_pool's ≈1.4–1.9 s). On the
**low-degree grids** `fwd_star` is competitive (and marginally faster for `luby` and `bk`,
though not for `bg`/`root`), confirming the pointer chase only bites at high degree.
The `vec` (dense array) and `bstr`
(bit-string) backends are middling — never the fastest. Among selectors `bg` / `luby`
/ `root` are competitive and `bk` is consistently slowest. The GPU vec_pool column is
fastest overall on grids (its solve is ≈4× the CPU's) and on most SuiteSparse, mixed
on IPM.

![apxchol ablation grids](latest/figures/ablation_grids.png)
![apxchol ablation IPM](latest/figures/ablation_ipm.png)
![apxchol ablation SuiteSparse](latest/figures/ablation_suitesparse.png)

### apxchol IS selector × graph type

A cross-family view of the same selector question: which IS selector wins on which
**graph type**, at the default `vec_pool` storage (t16). Rows = the four selectors,
columns span the structured→irregular axis (2D/3D grids → FEM/planar → IPM →
social/scale-free); colour normalises *per column* so green = the best selector for
that graph and **bold** = the per-graph winner.

**Takeaway:** on structured grids and FEM, `bg` is best and the spread is small;
on the **dense social/citation graphs the selector matters a lot** — `root+tree`'s
deep elimination tree makes its SpTRSV explode (coPapersDBLP ≈45 s, com-LiveJournal
≈69 s) while `luby` stays robust and fastest there. `luby+tree` is the most uniformly
good choice across graph types; `bk` is never the winner. (The iteration-count
variant is more uniform — the dramatic differences are in the SpTRSV solve, not the
PCG iteration count.)

![apxchol selector × graph, total time](latest/figures/selector_graph_total.png)
![apxchol selector × graph, PCG iterations](latest/figures/selector_graph_iters.png)

Per-family selector × matrix panels (all matrices of each family, total / setup /
solve / iterations small-multiples; per-column normalised), on both axes:

![selector × matrix, grids CPU](latest/figures/selector_grids_cpu.png)
![selector × matrix, grids GPU](latest/figures/selector_grids_gpu.png)
![selector × matrix, IPM CPU](latest/figures/selector_ipm_cpu.png)
![selector × matrix, IPM GPU](latest/figures/selector_ipm_gpu.png)
![selector × matrix, SuiteSparse CPU](latest/figures/selector_suitesparse_cpu.png)
![selector × matrix, SuiteSparse GPU](latest/figures/selector_suitesparse_gpu.png)

**Why the selector changes the solve so much — SpTRSV level count.** The triangular
solve is scheduled in level sets (each level is a batch of columns with no remaining
dependencies), so the back-solve *level count* is the length of the critical path —
fewer levels = a shallower factor = a faster SpTRSV. Dumping the per-`(selector,
matrix)` level count (`APXCHOL_LEVEL_DUMP`, vec_pool, t16) shows exactly the structure
behind the timing heatmap: `root+tree` builds a near-degenerate, very deep tree on the
dense social/citation graphs (coPapersDBLP **432,943** levels vs `bg`'s 4,956 — an 87×
deeper critical path; as-Skitter 21,530 vs 7,214; grid3d 14,664 vs 1,368), which is why
its SpTRSV explodes there. Conversely `bk+tree` deepens the structured grids and IPM
ladders (iter0010 1,834 vs `bg`'s 156, a 12× blow-up). `bg`/`luby` stay shallow on every
family — the level count is the single best predictor of the selector's solve cost.

![apxchol selector × matrix, SpTRSV back-solve level count](latest/figures/selector_levels.png)

**De-singularization mode (per solver):** see the [Protocol](#protocol). The original
singular `L` is benchmarked; apxchol / BoomerAMG / AMGCL / RCHOL ground its null space
without changing the *scored* operator (mean-centring / Dirichlet pin / Dirichlet pin /
εI-preconditioner-only) and are scored on the true `‖b − L x‖`. ParAC is pin-grounded
(no longer `εI`) and — via the patched driver's per-component-consistent, pin-zeroed RHS —
is now **scored on the true `‖b − L x‖` against the original `L`** too (Protocol); only CMG
still keeps `L + εI`, `ε = 1e-6·mean|diag|`.

## Poster figures (CPU)

Two side-by-side heatmaps over **one shared, representative column set** — `grid 2D
(25M)`, `grid 3D (15.6M)` (the largest grids tested), IPM (geomean over the iter
ladder), and a popular SuiteSparse spread ordered structured→irregular that includes
both where apxchol **loses** (`G3_circuit`, `thermal2`, the `kron_g500` scale-free
graph, `coPapersDBLP`) and where it **wins** (`coAuthorsDBLP` + the social giants).
CPU, t16, **total time**, tol 1e-8. Colour normalises *per column* on a **log scale**
(green = fastest for that graph; `1×…16×` colorbar), each cell shows the absolute
seconds over its `×-best` ratio, **bold** = per-column winner. Timed-out cells (real
time unknown) are clamped to **`≥` 10× apxchol** and drawn deep-red with a black border
(as in the house overview heatmap).

The first answers *which IS selector* to use; the second answers *apxchol's best
selector vs the multigrid / Cholesky field*. Same columns, so they read together.

![apxchol IS-selector × graph, CPU total time](latest/figures/poster_selectors_cpu.png)
![apxchol vs multigrid/Cholesky, CPU total time](latest/figures/poster_comparison_cpu.png)

**Headline:** the algebraic multigrids (AMGCL, BoomerAMG) win the structured grids, FEM
and IPM ladders, while **apxchol wins the irregular social/scale-free giants**, where
classical-AMG hub coarsening is expensive (BoomerAMG 212 s on com-Youtube, 440 s on
com-LiveJournal vs apxchol's 1.6 s / 32 s; see the tables). On CPU the IS-selector choice is a near-wash
(robust); the dramatic selector spread is on the GPU SpTRSV and in the level-count chart
above. `RCHOL` times out (`≥`) on the grids and the giants.

**One row carries a wall-clock caveat** (read its *iteration count*, not its bar):
**CMG (†)** is **serial + MATLAB-runtime** (no GPU/threads — see the methods table), so
its wall-clock isn't comparable to the 16-thread C++ solvers even though its iteration
quality is excellent (25-28 on grids vs apxchol's ~45). **ParAC** is now scored against the
original `L` like every in-house solver (per-component-consistent RHS, Protocol), so its
bars *are* comparable — its physics driver is the per-component split on the pure Laplacian.

## Headline findings

- **On grids, multigrid wins.** Fastest on **total time** is **AMGCL with the Dirichlet
  pin** (e.g. grid_3000 ≈4.3 s vs BoomerAMG ≈9.3 s, apxchol ≈10.8 s); **BoomerAMG** has
  the **fewest iterations** (7–11, near-constant in size) but a heavier setup. The
  Cholesky-type solvers cluster behind — **apxchol (best IS-selector) strongest of them**.
  PCG iters on grids: BoomerAMG 7–11 ≪ AMGCL 12–15 / CMG 25–28 ≪ RCHOL 22–40 (only the
  three smallest grids completed) ≈ apxchol 28–51 < ParAC 34–74.
- **On IPM (SDDM), BoomerAMG leads** (10 iterations on every rung, to true 1e-8);
  apxchol tracks it within ~25% on total time (and wins `iter0010`); the other
  Cholesky-type solvers trail it by 1.5–9× (ParAC 1.5–3.5×, pRCHOL 2–3×, RCHOL 5–9×).
  Among apxchol configs the IPM lead is
  **`bg`/`luby`** (median total 1.42 s each vs `root` 1.47 s, `bk` 1.86 s); on the GPU
  axis `luby` takes IPM (1.16 s vs `bg` 1.31). No selector generalizes: on the CPU
  medians over the matrices every selector completed, `bg` leads grids (7.78 s) and
  SuiteSparse (1.15 s) with `root` third there (8.50 s / 1.25 s) and `bk` last
  everywhere. Pick per graph type — see the selector heatmaps above.
- **On the scale-free / social graphs (com-Amazon, coAuthorsDBLP), apxchol is fastest** —
  the regime where the Cholesky-type structure pays off (com-Amazon: apxchol ≈0.40 s vs
  AMGCL ≈0.94, BoomerAMG ≈1.12; coAuthorsDBLP: 0.40 vs 1.99 / 2.49).
- **Nobody scales well with threads** (see the thread charts): at 16 cores the best
  is BoomerAMG/apxchol at ~2–3×; RCHOL/ParAC are flat and pRCHOL can go *negative*.
  The work is bandwidth-bound with serial bookends (AMD reorder, deep-factor SpTRSV).
- **pRCHOL** (parallel RCHOL) beats serial RCHOL on **every matrix where both
  completed** — 1.6–5.8× on total time (grid_500 5.8×, com-Amazon 5.0×,
  coAuthorsDBLP 4.8×, the IPM ladder 2.4–2.8×, grid3d_100 1.6×). The win is in the
  **solve**, not the factorization: its ND ordering usually costs a few extra PCG
  iterations (e.g. grid_1000 57 vs 40) but gives a parallel triangular solve
  (grid_1000 2.0 s vs 5.7 s). It is not a free win at scale — pRCHOL still times out
  wherever serial RCHOL does (the large grids and most giants), and it is the only
  one of the two that finishes `coPapersDBLP`.
- **ParAC** factorizes fast (parallel dependency-DAG over the AMD order) but its
  triangular solve dominates; AMD reordering (author-recommended) is required for
  competitive performance and is counted in its setup time.
- **The GPU-resident PCG is apxchol's fastest solve.** Keeping all PCG vectors on
  device (measured then with cuBLAS + cuSPARSE; the loop is our own kernels since 2026-08-18) is **2-4.5× faster per iteration**
  than the host-PCG hybrid — ~13 ms/it vs ~59 ms/it on `grid_2000`, ~6.5 vs ~23 on
  `G3_circuit` (RTX 4090 Laptop, locked clocks). The host path does its SpMV on the
  contended CPU cores, so it is both slower and far noisier; GPU-PCG is now the only
  charted `apxchol (GPU)` config.

The breakdown charts make the cost split explicit: ParAC's and RCHOL's *solve*
dominates (many iterations × deep triangular solve), while multigrid has a tiny solve.

## Open / not yet covered

- The rectangular grid; boost-off (locked-frequency) "published" tier.
- VRAM heatmaps (`vram_peak`/`vram_solve`, GPU axis) need a re-sweep with the
  instrumented CUDA build to populate; CPU cells stay unmeasured by design.
