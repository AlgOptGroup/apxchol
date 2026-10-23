# Benchmarks

This separate project compares apxchol with other sparse solvers using common inputs, stopping criteria and timing boundaries. The maintained runners write results and plots outside tracked source. A published comparison must name the source revisions and protocol it measured; its numbers do not describe later checkouts.

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
  grading remains outside timing. Factors/hierarchies are retained through retries.
- CPU apxchol, Eigen CG, AMGCL and Hypre warm-start native retries. GPU apxchol
  and RCHOL reuse setup with cold native restarts. Julia AC/AC2 reuse their native
  solver closures; Julia CG, regularized ICC and direct methods can use residual
  corrections. These are adapter stopping policies, not vendor kernel rewrites.
- apxchol v1 uses a complete `--v1-backend cpu|gpu|auto` route. CPU includes
  host setup, OpenMP triangular solves and host PCG even in a CUDA build. GPU
  includes device-owned setup and GPU PCG, and rejects unsupported configurations.
  Route selection is fixed before setup and retained through original-v1 retries;
  no failure silently falls back to another route. The fair/scaling runners force
  their device and require matching `execution_route` CSV/repetition receipts.
  Old mixed-route rows remain historical and cannot resume a current comparison.
- `--report-fill` reads the actual held factor and its SpTRSV drop statistics.
  It never constructs another factor or uploads a host diagnostic factor. Raw
  factor off-diagonals and stored L11 entries are distinct counts; CSR and CSC each
  hold the latter. The input denominator counts stored off-diagonal entries after
  timing, rather than subtracting the row count: zero isolates may omit diagonal
  storage. A zero off-diagonal input denominator makes the ratio unavailable.
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
- v1 host `solve_rss_mb` is sampled while the measured owner is still alive; the
  `/proc` sampling cost is included in Solve. Older v1 rows sampled after owner
  destruction and are not comparable as solve-held RSS. GNU child peak RSS keeps
  its separate whole-process definition; do not infer a memory win from that
  boundary correction.
- Whole-cell deadlines may cover calibration and all repetitions: a timeout is **not**
  a lower bound on one solve. Preserve failure, nonconvergence, timeout, unsupported,
  unattempted and missing statuses. Unknown memory is not zero.
- Pin CPU affinity; record requested/effective threads, placement, source/binary hashes, compiler/runtime, warmups and retained receipts. See [threads and placement](../README.md#threads-and-placement).
- Measure GPU memory in separate runs: polling `nvidia-smi` during a timed C++ or ParAC call perturbs its setup.

## Solvers

Hypre/BoomerAMG and AMGCL have CPU/CUDA series; AC/AC2 are Julia references.
RCHOL/pRCHOL retain upstream factors with labelled MKL or portable PCG.
[ParAC Graph/Physics](patches/parac/) separates common input reading and final
interchange from CPU Graph/AMD algorithm preparation, while retaining required
transformations, ordering/permutation and native factor/workspace setup. Complete
preparation stays diagnostic; source/schema-bound caches reject old accounting.
Physics/GPU preparation keeps its explicit complete timing contract. Do not subtract overhead retrospectively or relabel historical cells.
The bundled CPU build needs MKL; separately labelled portable implementations
may have different threading and must record it. Positive stored Physics off-diagonals are unsupported.
ParAC patch 0007 checks original accuracy inside each measured solve. There is
no untimed, matrix-specific tolerance calibration before retained runs. Julia
uses fixed tiny connected/disconnected/SDDM fixtures to warm its native closure
types before timing; these fixtures do not depend on the measured matrix.

The MATLAB CMG adapter is excluded from comparisons until it uses the shared
operator and RHS: it currently generates its own RHS and shifts Laplacians.
The Linux container recipe remains in [cmg/Dockerfile.matlab](cmg/Dockerfile.matlab):

```sh
docker build -t apxchol-matlab-cmg:r2026a -f benchmarks/cmg/Dockerfile.matlab benchmarks/cmg
```

Set `APXCHOL_MATLAB_ROOT` to a licensed MATLAB R2026a installation and
`APXCHOL_CMG_SOLVER` to a CMG checkout whose MEX files were built with `makeCMG`
for that MATLAB version.

## Build and run

```sh
cmake -S benchmarks -B benchmarks/build -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/build -j6 --target benchmark
python3 benchmarks/sweep_fair.py --threads N --store results/cells
```

For CUDA, use a separate build directory and add `-DAPXCHOL_DEVICE=CUDA
-DBENCH_HYPRE_USE_CUDA=ON -DBUILD_GPU_RCHOL=ON`; sweep with `--device gpu`.
Optional solvers require their runtimes. Julia dependencies:

```sh
julia --project=benchmarks/julia -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmarks/julia benchmarks/julia/test_original_stop.jl
```

Runner controls include `--only`, `--threads`, `--repeat`, `--store`.
Machine-local paths belong in ignored `paths_local.py`/`paths_local.cmake`;
[runner_common.py](runner_common.py) defines defaults. `--dump-mtx` and
`--dump-rhs` export the shared operator/RHS in separate driver invocations.

The C++ benchmark accepts `--factor-storage auto|float16|float32`, `--cpu-schedule auto|levels`, `--factor-drop`, `--regularization`, `--lump-max-mass` and `--degree-quantile` for apxchol v1. These pass directly to the library's options; named configurations retain their sampler/partitioner choices. `--profile` prints its timing breakdown. Timing collection remains enabled so Setup and Solve retain their measured boundaries; ambient `APXCHOL_*` solver overrides are not read.

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

## Thread-scaling plots

Rendering does not run solvers and writes ignored `results/plots` previews by
default. Every selected cell must meet the measurement contract and match the
selected source revision, which defaults to the current checkout. To render an
archived package's results, explicitly select its full revision with
`--source-commit SHA`; that output describes the archived source, not the current
checkout. Mixed revisions are rejected.

```sh
PYTHONPATH=benchmarks python3 benchmarks/thread_scaling.py --render-only --compact \
  --store SCALING_STORE --matrices grid_2000,iter0040,as-Skitter \
  --series 'apxchol bg+tree,apxchol trace-cycle,AMGCL,BoomerAMG,ParAC' \
  --thread-counts 1,2,4,8,16 --out results/plots
```

Compact plots show absolute times and log-log speedups; no complete T1 means no
speedup. The fair-sweep renderers live in the research archive.
