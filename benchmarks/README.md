Current campaign repetition contract: the C++ driver accepts `--warmup N`
(default 0) separately from `--repeat R` retained measurements. It emits every
warmup and retained timing as `BENCH_REPEAT`, selects one coherent median-total
retained result, and exports the maximum retained true residual. The common
parser grades every retained repetition through that maximum. CUDA context
initialization remains outside these solver intervals and is stored separately.
`thread_scaling.py` accepts `--thread-counts`, `--matrices`, `--series`, `--device`,
`--warmup`, and `--timeout`; declared scope is validated before rendering. This
allows laptop and Daint to use the same converged-solve runner and figures.

# apxchol benchmarks

This standalone CMake project compares `apxchol` with randomized-Cholesky and
multigrid solvers on generated grids, SuiteSparse matrices, and an LP-IPM
sequence.

Results are machine-specific snapshots, not a live leaderboard:

| snapshot | useful views | data |
|---|---|---|
| Ryzen 9 7945HX + RTX 4090 Laptop, T=16 | [laptop index](latest/) and [generated tables](latest/summary.md) | [CSV](latest/results.csv) |
| CSCS Daint GH200, 72-core Grace + Hopper, T=72 | [Daint index](daint/) and [fair-solver summary](daint/fair_t72_summary.md) | [CSV](daint/fair_t72.csv) |

Do not compare absolute times across these machines. The auditable source is the
[cell store](../results/cells/); committed CSVs, summaries, and figures are
presentation extracts.

## Fairness contract

### Inputs and grading

Every registry entry declares whether the Matrix Market file is a `graph` or an
assembled `operator`, and whether the resulting system is `laplacian` or
full-rank `sddm`. A graph is assembled as `L = D - A`; the benchmark binary
checks these declarations instead of guessing from stored values.

Singular Laplacians are solved on the original operator. Each solver removes the
component-wise constant null space through its supported route: native
rank-aware solving, safe Dirichlet pins, or component-wise decomposition. For
the original-operator series, the stored grade is independently recomputed as

```text
||b - A x||_2 / ||b||_2
```

against the original assembled operator. `complete` means this value is at most
`1e-8`; a solver's recurrence residual is diagnostic only. CMG cells using the
older lightly regularized operator are labelled and must not be merged with the
original-operator series. ParAC's tolerance translation and audited driver
changes are documented in [patches/parac/README.md](patches/parac/README.md).

### Timing and accounting

Common file parsing and operator assembly happen before solver timers.

- `setup_s` includes every solver-required grounding, conversion, reordering,
  upload, hierarchy or factor construction, and triangular-solve analysis.
- `solve_s` includes per-RHS preparation, iteration, device transfers, and
  returning the solution in the caller's ordering.
- `total_s = setup_s + solve_s`. Independent residual grading and cleanup are
  excluded.
- Process-wide CUDA primary-context creation is prewarmed once and reported as
  `cuda_init`, outside solver ranking. Solver-specific module loading, handles,
  allocations, preparation, and transfers remain charged to that solver.
- Hypre's process-wide initialization is charged once per reported Hypre row.
  ParAC reports one real median-total repetition after reusable Julia load/JIT
  warm-up; fields are not medianed independently.
- A complete logical-cell deadline can include calibration and all requested
  repetitions. Persist that scope: time spent on earlier repetitions is not a
  lower bound on one setup+solve. Native CMG tags such deadlines `logical_cell`,
  so charts show `T` without a numerical single-solve bound. Never fabricate a
  completed time from a timeout.

### Series and status

One plotted row is one declared `(solver, configuration, device)` tuple. No
headline series takes a per-matrix minimum over selectors, seeds, or toolchains;
apxchol's default and its ablations are shown separately.

`complete`, `not_converged`, `timeout`, `failed`, `oom`, and `n/a` remain
distinct. CPU RSS and GPU VRAM are separate metrics. Each cell records source,
compiler/runtime, thread count, repetitions, affinity, and solver-specific
provenance. Renderers reject ambiguous series and stale cells invalidated by
changes to the operator, RHS, timing, convergence, or solver semantics.

Published CPU timing uses explicit rank-local OpenMP pinning. Because the driver
may contain both LLVM `libomp` and GNU `libgomp`, use `sweep_fair.py`; the binary
rejects a bound multi-thread run that enters `main()` with too few physical
cores.

Run the series audit with:

```bash
PYTHONPATH=benchmarks python3 benchmarks/dev/audit_series_rule.py
```

## Compared solvers

| series | implementation |
|---|---|
| `apxchol/bg` | this repository's declared default; other selectors/storage are ablations |
| BoomerAMG | Hypre PCG with BoomerAMG, CPU and CUDA |
| AMGCL | smoothed-aggregation AMG with CG, CPU and CUDA |
| RCHOL / pRCHOL | upstream factors; MKL PCG on x86 or explicitly labelled portable PCG |
| ParAC graph / physics | upstream CPU/CUDA drivers plus the audited patch stack |
| AC / AC2 | Laplacians.jl reference implementation and oversampled variant |
| CMG | canonical MATLAB `cmg-solver`; cross-language wall time is caveated |
| `cmg_packed/original-operator` | private generated CMG packed port; serial native core, unshifted operator |

ParAC CPU requires MKL. On ARM64, RCHOL/pRCHOL use the labelled portable path,
AC/AC2 use the official Julia build. Canonical native Linux MATLAB is x86-64;
the separately labelled CMG packed port runs on ARM without a MATLAB runtime.
The snapshot pages state the exact historical coverage boundary.

To opt into native CMG, build `benchmarks/cmg/native` with
`CMG_GENERATED_DIR` pointing to the private recovered generated source,
`FMM_SOURCE_DIR` to an existing fast_matrix_market checkout, and optionally
`EIGEN_INCLUDE_DIR` to bundled Eigen headers. Set `APXCHOL_CMG_NATIVE_BIN` to
the resulting absolute executable path. The ordinary CMG sweep step then
writes `cmg_packed/original-operator` cells through `cmg_native_runner.py`.
Generated source is deliberately not tracked or distributed with this adapter.

The native series uses compatible `b=A*g` with NumPy PCG64 seed 42, normalized
to unit norm, and records exact input/operator/RHS hashes. It applies no shift
or pin. This shares the original-operator grading contract, but is not a
bit-identical RHS to another language's RNG and is not canonical MATLAB CMG:
the packed port has a documented valid terminal hierarchy where canonical
shallow_water1 setup errors at `H{0}`. Its generated core is serial; requested
CPU affinity and effective thread count 1 are recorded separately. Every
returned solution is independently checked with SciPy before acceptance.
Setup includes the one-based CSC adapter and hierarchy construction. The
packed PCG call includes hierarchy destruction in solve time; it cannot be
separated without changing generated implementation. Calibration and all
retained runs share one persisted per-cell timeout. All retained repetitions
must pass, and timing fields come from one median-total repetition.

## Build and run

The root library and benchmark suite use separate build trees:

```bash
cmake -S benchmarks -B benchmarks/build -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/build -j"$(nproc)" --target benchmark

benchmarks/build/benchmark --graph grid --n 2000 \
  --solver apxchol_v1,hypre_boomeramg,amgcl \
  --threads 16 --tol 1e-8 --repeat 3 --csv
```

The direct command is suitable for functional checks. Use the resume-safe,
affinity-controlled runner for timing:

```bash
julia --project=benchmarks/julia -e 'using Pkg; Pkg.instantiate()'
python3 benchmarks/sweep_fair.py
PYTHONPATH=benchmarks python3 benchmarks/fair_charts.py --out benchmarks/latest
PYTHONPATH=benchmarks python3 benchmarks/combined_charts.py --out benchmarks/latest/figures
```

CUDA build and sweep:

```bash
cmake -S benchmarks -B benchmarks/build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DAPXCHOL_USE_CUDA=ON -DBENCH_HYPRE_USE_CUDA=ON -DBUILD_GPU_RCHOL=ON
cmake --build benchmarks/build-cuda -j"$(nproc)" --target benchmark
python3 benchmarks/sweep_fair.py --device gpu
PYTHONPATH=benchmarks python3 benchmarks/gpu_charts.py --out benchmarks/latest/figures
PYTHONPATH=benchmarks python3 benchmarks/combined_charts.py --out benchmarks/latest/figures
```

Useful runner controls are `--only`, `--threads`, `--repeat`, and `--store`.
External build and cache paths can be supplied through environment variables or
the gitignored `paths_local.py` / `paths_local.cmake`; authoritative names are in
[runner_common.py](runner_common.py) and [parac_runner.py](parac_runner.py).
CMake fetches Hypre, AMGCL, RCHOL, Eigen, and test helpers when they are not
provided; optional solvers remain disabled when their required runtime is
unavailable.

To inspect semantic invalidation without deleting anything:

```bash
python3 benchmarks/stale_cells.py
```

`sweep_fair.py` uses the same stale predicate: reusable terminal cells skip, and
invalidated terminal cells rerun while their old JSON remains in place until a
replacement is ready. `stale_cells.py --delete` is optional cleanup, recoverable
from Git. Add a stale-cell rule whenever a change alters the operator, RHS,
timing boundary, convergence semantics, or solver result.


### ParAC input and calibration eligibility

Both CPU and GPU physics routes reject any strictly positive stored
MatrixMarket off-diagonal before preprocessing or cache lookup: ParAC's
`-abs(weight)` conversion would change that original operator. Such cells are
`n/a` with a count and reason; malformed inputs are `failed`. Graph mode keeps
its intended adjacency-to-Laplacian normalization.

A CPU probe at its iteration cap, with an explicit failed stopping flag, a
nonzero exit, or missing/nonfinite calibration statistics produces a failed
cell and no retained runs for that probe. The GPU probe also rejects its
upstream 300-iteration cap and invalid statistics; upstream provides no
trustworthy recurrence-stop flag, so below-cap iteration count is not a proof
of recurrence convergence. Every retained CPU, component-aggregated CPU and
GPU true residual must pass the requested tolerance. Timing fields still
come from one real median-total repetition; `repeat_rel_res` preserves the
other residuals. Calibration is an estimate, not a convergence guarantee.
