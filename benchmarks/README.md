# Benchmarks

[Daint results](daint/) are the current performance reference. [Laptop results](archive/)
are retired. Raw campaign stores are private; committed CSVs and provenance identify
selected measurements. Local `results/cells/` is not automatically the published source.

## Measurement contract

- Inputs declare graph adjacency (`L=D-A`) or an assembled operator, and Laplacian
  or SDDM class. Preserve the original operator/RHS and component-wise nullspace.
- Accept only when **every retained true residual** `||b-Ax||/||b|| ≤ 1e-8`.
  A recurrence residual is insufficient. Keep shifted CMG series separately labelled.
- `original-v1` stopping checks that residual at native solver exits and continues
  when needed, within one total iteration budget and at most eight attempts.
  Disconnected split solves apply the budget per component. Direct solvers count
  a solve/refinement as one work unit. Exhaustion remains `not_converged`.
- Solve includes required stopping checks and retries. `stop_check_s` reports
  residual-evaluation time **within** Solve; do not subtract it for comparisons.
  `solve_passes` (or ParAC's `stop_checks`) explains additional work. No new
  host check/transfer is inserted into every GPU iteration. Independent final
  grading remains outside timing. Factors/hierarchies are retained; the packed
  CMG API is the exception and reports/charges every repeated setup explicitly.
- CPU apxchol, Eigen CG, AMGCL and Hypre warm-start native retries. GPU apxchol
  and RCHOL reuse setup with cold native restarts. Julia AC/AC2 reuse their native
  solver closures; Julia CG, regularized ICC and direct methods can use residual
  corrections. These are adapter stopping policies, not vendor kernel rewrites.
- New rows must carry `stop_contract=original-v1`. Old timings remain historical
  evidence, but cannot resume or enter current comparisons as if their cheaper
  stopping rule were the same measurement. Re-run affected comparisons; never
  retroactively add/subtract a modeled checking cost.
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
  compiler/runtime, warmups and retained receipts. Report the placement with the
  thread count: on an EPYC 7742 socket the same 16 threads take 1.00 on one NUMA
  domain and 0.86 on four ([threads and placement](../README.md#threads-and-placement)). Never poll `nvidia-smi` inside
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
ParAC patch 0007 checks original accuracy inside each measured solve. There is
no untimed, matrix-specific tolerance calibration before retained runs. Julia
uses fixed tiny connected/disconnected/SDDM fixtures to warm its native closure
types before timing; these fixtures do not depend on the measured matrix.

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

## Weighted and additional inputs

Generate four matched reweightings of an input's undirected off-diagonal
support (requires NumPy and SciPy):

```sh
python3 benchmarks/weighted_inputs.py data/matrices/com-Amazon.mtx \
  data/weighted/amazon-s42 --name amazon --seed 42
python3 benchmarks/sweep_fair.py \
  --matrix-manifest data/weighted/amazon-s42/manifest.json --families weighted \
  --threads 16 --store results/weighted-cells
```

The output directory must be new. The variants are unit weights, log-uniform
weights in `[0.1,10]`, log-uniform weights in `[0.001,1000]`, and a backbone
variant with the narrow range on a deterministic BFS spanning forest and the
wide range elsewhere. Every connected component gets a tree; isolated vertices
are preserved. The narrow and wide models share random variates from NumPy
PCG64. These are explicitly **log-uniform**, not uniform in value. The input's
original weights and diagonal are discarded; generated files declare graph
adjacency so the existing harness constructs their Laplacians.

The manifest binds each file's SHA-256, model, seed, support source hash, component
count and forest size to its matrix ID. The sweep verifies hashes before
registration and retains that identity in every cell. Changed input or manifest
identities prevent reuse of old cells. Existing matrix IDs cannot be replaced.
The standard sweep's solver set and measurement contract still apply; generating
inputs is not a performance result.

The same manifest format accepts additional supplied matrices, including PageRank
inputs, without altering their values. Use `schema_version: 1` and a `matrices`
list with `id`, `family`, `path` (relative to the manifest), positive integer `n`,
`sha256`, and `kind`. An assembled operator requires `kind: "operator"` and an
explicit `class: "laplacian"` or `"sddm"`; graph adjacency uses `kind: "graph"`
without a class. Select its family with `--families`. The supplier must establish
that a proposed PageRank formulation meets the solver's symmetric operator
contract; an arbitrary directed PageRank matrix must not be relabelled as SDDM.

## Render selected results

Rendering does not run solvers. Defaults write ignored `results/plots` previews.
Publication requires an explicitly selected, audited store:

```sh
PYTHONPATH=benchmarks python3 benchmarks/render_snapshot.py \
  --cells STORE --out OUTPUT --threads 72 --platform Daint --sampler-comparison
PYTHONPATH=benchmarks python3 benchmarks/thread_scaling.py --render-only --compact \
  --store SCALING_STORE --matrices grid_2000,iter0040,as-Skitter \
  --series 'apxchol bg+tree,apxchol trace-cycle,AMGCL,BoomerAMG,ParAC' \
  --thread-counts 1,2,4,8,16,36,72 --out results/plots
```

Add `--fill-cells FILL_STORE` to include the explicit factor-fill observations.
The timing heatmaps use an uncapped logarithmic colour scale.

Compact plots show absolute times and log-log speedups; no complete T1 means no
speedup. Full-study extracts remain separate. `stale_cells.py` checks semantic
invalidation without deleting evidence; renderers reject stale/ambiguous cells.
Use `benchmarks/dev/audit_series_rule.py` for the declared-series audit.

[Earlier protocol detail](https://github.com/AlgOptGroup/apxchol/blob/1a526f25aec8829e8a9217b558ac2290a3840ae0/benchmarks/README.md)
