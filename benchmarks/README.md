# Benchmarks

[Daint results](daint/) are the current performance reference. [Laptop results](archive/)
are retired. Raw campaign stores are private; committed CSVs and provenance identify
selected measurements. Local `results/cells/` is not automatically the published source.

## Measurement contract

- Inputs declare graph adjacency (`L=D-A`) or an assembled operator, and Laplacian
  or SDDM class. Preserve the original operator/RHS and component-wise nullspace.
- Accept only when **every retained true residual** `||b-Ax||/||b|| ≤ 1e-8`.
  A recurrence residual is insufficient. Keep shifted CMG series separately labelled.
- Select one coherent median-total repetition after explicit warmups; never select
  timing fields independently or minimize across configurations.
- Setup includes required conversion, grounding, ordering, uploads, factor/hierarchy
  construction and triangular analysis. Solve includes RHS work, iterations,
  transfers and returning the solution. Total is setup + solve.
- Common parsing/assembly and independent grading are excluded. Process-wide CUDA
  initialization is separately reported; solver-specific allocations/module loading
  remain charged. Hypre initialization is charged once per row.
- Whole-cell deadlines may cover calibration and all repetitions: a timeout is **not**
  a lower bound on one solve. Preserve failure, nonconvergence, timeout, unsupported,
  unattempted and missing statuses. Unknown memory is not zero.
- Pin CPU affinity; record requested/effective threads, source/binary hashes,
  compiler/runtime, warmups and retained receipts. Never poll `nvidia-smi` inside
  timed C++ or ParAC calls.

## Solvers

Hypre/BoomerAMG and AMGCL have CPU/CUDA series; AC/AC2 are Julia references.
RCHOL/pRCHOL retain upstream factors with labelled MKL or portable PCG.
[ParAC Graph/Physics](patches/parac/) separates common input reading and final
interchange from CPU Graph/AMD algorithm preparation, while retaining required
transformations, ordering/permutation and native factor/workspace setup. Complete
preparation stays diagnostic; source/schema-bound caches reject old accounting.
Physics/GPU preparation keeps its explicit complete timing contract. The corrected
private Daint study and this maintained source port have separate provenance;
do not subtract old overhead retrospectively or relabel historical cells.
The bundled CPU build needs MKL; Daint's separately labelled portable implementation
has a serial solve. Positive stored Physics off-diagonals are unsupported.
Failed/capped calibration must not launch fallback-tolerance retained runs.

Canonical MATLAB CMG is unavailable on ARM64. The separate `cmg_packed` port is
serial, uses private generated source, and is not canonical MATLAB CMG: even
terminal hierarchy behavior can differ. Its default normalized `b=A*g` uses NumPy
PCG64 seed 42; an explicit `rhs_path` avoids cross-language RNG differences.
Original-operator grading is independent. Setup charges CSC conversion; solve
includes generated hierarchy destruction. Build `benchmarks/cmg/native` with
`CMG_GENERATED_DIR`, `FMM_SOURCE_DIR`, optionally `EIGEN_INCLUDE_DIR`; set
`APXCHOL_CMG_NATIVE_BIN`. Do not distribute generated proprietary sources.

## Build and run

```sh
cmake -S benchmarks -B benchmarks/build -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/build -j6 --target benchmark
python3 benchmarks/sweep_fair.py --threads 72 --store results/cells
```

For CUDA, use a separate build directory and add `-DAPXCHOL_USE_CUDA=ON
-DBENCH_HYPRE_USE_CUDA=ON -DBUILD_GPU_RCHOL=ON`; sweep with `--device gpu`.
Optional solvers require their runtimes. Julia dependencies:

```sh
julia --project=benchmarks/julia -e 'using Pkg; Pkg.instantiate()'
```

Runner controls include `--only`, `--threads`, `--repeat`, `--store`.
Machine-local paths belong in ignored `paths_local.py`/`paths_local.cmake`;
[runner_common.py](runner_common.py) defines defaults. `--dump-mtx` and
`--dump-rhs` export the shared operator/RHS in separate driver invocations.

## Render selected results

Rendering does not run solvers. Defaults write ignored `results/plots` previews.
Publication requires an explicitly selected, audited store:

```sh
PYTHONPATH=benchmarks python3 benchmarks/render_snapshot.py \
  --cells STORE --out OUTPUT --threads 72 --platform Daint
PYTHONPATH=benchmarks python3 benchmarks/thread_scaling.py --render-only --compact \
  --store SCALING_STORE --matrices grid_2000,iter0040,as-Skitter \
  --series 'apxchol bg+tree,AMGCL,BoomerAMG,ParAC' \
  --thread-counts 1,2,4,8,16,36,72 --out results/plots
```

Compact plots show absolute times and log-log speedups; no complete T1 means no
speedup. Full-study extracts remain separate. `stale_cells.py` checks semantic
invalidation without deleting evidence; renderers reject stale/ambiguous cells.
Use `benchmarks/dev/audit_series_rule.py` for the declared-series audit.

[Earlier protocol detail](https://github.com/AlgOptGroup/apxchol/blob/1a526f25aec8829e8a9217b558ac2290a3840ae0/benchmarks/README.md)
