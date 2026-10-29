# Benchmarks

This separate project compares apxchol with other sparse solvers using common inputs, stopping criteria and timing boundaries. The maintained runners write results and plots outside tracked source. A published comparison must name the source revisions and protocol it measured; its numbers do not describe later checkouts.

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

## Measurement contract

### Hosted checks

The `PR performance` workflow installs the exact PR base and head in separate
Python environments on each runner. Both use the same downloaded dependencies,
build flags, compiler cache and frozen harness. Linux measures CPU with two
workers; macOS 15 requires the explicit Metal route with two host workers.
The initial smoke fixtures are the 192-by-192 and 512-by-512 Dirichlet grids from
`docs_benchmark.py`; this scope does not establish performance on irregular graphs.

Each fixture has three alternating ABBA/BAAB rounds, with the same seed and a tiny
untimed warmup in every fresh process. All 24 owners must pass the independent
original-system residual check. Setup, solve and their sum are reported separately.
A phase signals a slowdown only if all three head/base ratios are at least 1.20
and both revisions' repeat controls stay within 10% in every round. Incomplete or
invalid coverage fails first. Any qualified phase slowdown fails the overall
check, even when another phase is noisy; each phase retains its own status.
If no phase signals a slowdown but any phase is noisy, the overall check is
**inconclusive**, with a job-summary warning and successful exit to avoid a noisy
merge blocker. This is a conservative screening rule,
not a statistical significance test. JSON, raw logs, input/binary/harness hashes
and dependency/build records remain in artifacts, including after failures.

The manual `Manual competitor benchmark` workflow uses the existing native build
and `thread_scaling.py`: `grid_500`, `grid3d_100` and the public weighted irregular
`kron_g500-logn16` graph, each with apxchol, AMGCL and BoomerAMG at two CPU workers.
The downloaded Matrix Market file must match the pinned SHA-256; download or
identity failure stops the run. Nine cells each receive one warmup and three
retained repetitions. Timeout and nonconvergence statuses remain visible.
Artifacts include setup/solve plots, residuals and the existing repetition receipts.

PR timing has a four-minute campaign limit and a 30-second owner limit; the manual
run has a 60-second whole-cell limit (nine minutes for nine cells). Outer job caps
are 30 and 45 minutes including builds. These are limits, not measured runtimes;
actual hosted cost depends on compiler-cache hits. Neither workflow publishes
comments, Pages or README measurements.

### Accuracy and stopping

- Inputs declare graph adjacency (`L=D-A`) or an assembled operator, and Laplacian
  or SDDM class. Preserve the original operator/RHS and component-wise nullspace.
- Accept only when **every retained true residual** `||b-Ax||/||b|| ≤ 1e-8`.
  A recurrence residual is insufficient. Keep shifted CMG series separately labelled.
- `original-v1` stopping checks that residual at native solver exits and continues
  when needed, within one total iteration budget and at most eight attempts.
  Disconnected split solves apply the budget per component. Direct solvers count
  a solve/refinement as one work unit. Exhaustion remains `not_converged`.
- CPU apxchol, Eigen CG, AMGCL and Hypre warm-start native retries. GPU apxchol
  and RCHOL reuse setup with cold native restarts. Julia AC/AC2 reuse their native
  solver closures; Julia CG, regularized ICC and direct methods can use residual
  corrections. These are adapter stopping policies, not vendor kernel rewrites.

### Timing and memory

- Solve includes required stopping checks and retries. `stop_check_s` reports
  residual-evaluation time **within** Solve; do not subtract it for comparisons.
  `solve_passes` (or ParAC's `stop_checks`) explains additional work. No new
  host check/transfer is inserted into every GPU iteration. Independent final
  grading remains outside timing. Factors/hierarchies are retained through retries.
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
- Measure GPU memory in separate runs: polling `nvidia-smi` during a timed C++ or ParAC call perturbs its setup.

### Backend and factor accounting

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

### Retained results

- New rows must carry `stop_contract=original-v1`. Old timings remain historical
  evidence, but cannot resume or enter current comparisons as if their cheaper
  stopping rule were the same measurement. Re-run affected comparisons; never
  retroactively add/subtract a modeled checking cost.
- Select one coherent median-total repetition after explicit warmups; never select
  timing fields independently or minimize across configurations.
- Whole-cell deadlines may cover calibration and all repetitions: a timeout is **not**
  a lower bound on one solve. Preserve failure, nonconvergence, timeout, unsupported,
  unattempted and missing statuses. Unknown memory is not zero.
- Pin CPU affinity; record requested/effective threads, placement, source/binary hashes, compiler/runtime, warmups and retained receipts. See [thread placement](#thread-placement) below.
- `campaign_extract.py STORE --matrices ... --source-commit ...` turns a cell store
  into a public JSON summary. Its denominator is every listed matrix times every
  series in the store; missing cells are listed. It keeps each non-complete cell's
  last error line and process cap, and binds hardware, toolchains, binary and
  matrix hashes.

## Thread placement

Thread count controls how many CPU workers an operation may use. Placement
controls which cores those workers run on. These are separate choices.

For a placement experiment, compare the system default with
`OMP_PLACES=cores OMP_PROC_BIND=spread`: the first defines individual cores as
places, and the second spreads the OpenMP workers across those places. Measure
both setup and solve; spreading workers is not always faster.

On a multi-socket machine, constrain the available CPUs when you want to test
one socket. Otherwise a thread-count comparison can also change memory
locality. Record that CPU set and the OpenMP settings with the results.

## Euler RTX 4090 nodes

These settings let the CPU and CUDA benchmark builds and every series run on
ETH Euler's `eu-g6` nodes (2 × EPYC 9554, 8 × RTX 4090) in October 2026.

- Compile C, C++ and the nvcc host side with clang 19.1.3 from the 2025-06
  stack; keep `stack/2024-05 gcc/13.2.0` loaded for its runtime libraries.
  nvcc 13.4 with a gcc 13.2 host silently compiles `-std=c++23` as C++17
  (`__cplusplus` 201703), so `std::span` and `std::bit_cast` disappear.
- Pass `-DCMAKE_CUDA_FLAGS=--ftz=false`, link LLVM `libomp`
  (`-DOpenMP_omp_LIBRARY=…/libomp.so`) and set `CMAKE_BUILD_RPATH` to the LLVM,
  gcc 13.2 and CUDA library directories.
- Give each build tree its own `FETCHCONTENT_BASE_DIR`. A shared one builds
  Hypre in one place, and a CUDA configure leaks a CUDA `libHYPRE.a` into the
  CPU link.
- AMGCL needs Boost headers; CMake does not find them unless given
  `-DBoost_DIR=<2024-05 stack>/boost-1.83.0-…/lib/cmake/Boost-1.83.0`. Check for
  "Boost found — AMGCL solver enabled" in the configure output.
- With libomp, do not set `KMP_AFFINITY=norespect` in a shared (non-exclusive)
  allocation: libomp then binds outside the job's cgroup and aborts with
  `pthread_setaffinity_np` EINVAL. These builds link libomp only.
- Request GPUs as `--gpus=rtx_4090:N --constraint=cuda13` with the group's
  account; `--mem=0` is rejected, and `--mem-per-cpu` above about 2.9 GB does
  not fit the 384 GB nodes at 128 cores. Each socket is split into eight
  8-core NUMA domains, so choose cores from the job's allowed set, not from
  NUMA node 0.
- The login nodes are Intel Broadwell; `-march=native` binaries built on the
  compute nodes do not run there.

## Solvers

Hypre/BoomerAMG and AMGCL have CPU/CUDA series; AC/AC2 are Julia references.
RCHOL/pRCHOL retain upstream factors with labelled MKL or portable PCG.

RCHOL prints progress to `std::cout`, and pRCHOL does so from its worker
threads. The adapters silence it with `discard_cout`, a scoped stateless
buffer ([src/discard_streambuf.h](src/discard_streambuf.h)) that also restores
`std::cout` if RCHOL throws. The earlier shared `std::ostringstream` let those
threads grow one buffer concurrently and corrupt the heap. In October 2026
Euler diagnostics rchol_par crashed in 4 of 20 runs, and the archived laptop
cells record the same "double free or corruption" and signal 11.
`contract_discard_streambuf` runs 64 writers through the guard.

When Hypre holds device memory, IJ `SetValues`/`GetValues` read and write
their index and value arrays as device memory. The adapters stage those
arrays through Hypre's allocator in that case and pass host arrays otherwise,
so CPU-only Hypre builds are unchanged. Host arrays only worked where the GPU
can address pageable host memory (HMM drivers, GH200). On Euler's RTX 4090
nodes in October 2026 they failed in Hypre's device-to-device `cudaMemcpy`
("invalid argument"), followed by a CUB `exclusive_scan` "invalid device
ordinal". Input staging and the solution buffer are charged to Hypre GPU
setup; each solve attempt's copy of the solution back to the host is charged
to Solve.

[ParAC Graph/Physics](patches/parac/) separates common input reading and final
interchange from CPU Graph/AMD algorithm preparation, while retaining required
transformations, ordering/permutation and native factor/workspace setup. Complete
preparation stays diagnostic; source/schema-bound caches reject old accounting.
Physics/GPU preparation keeps its explicit complete timing contract. Do not subtract overhead retrospectively or relabel historical cells.

The bundled CPU build needs MKL; separately labelled portable implementations
may have different threading and must record it. Positive stored Physics off-diagonals are unsupported.

ParAC patch 0007 checks original accuracy inside each measured solve. There is
no untimed, matrix-specific tolerance calibration before retained runs.

Julia uses fixed tiny connected/disconnected/SDDM fixtures to warm its native closure
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
PCG64.

These are explicitly **log-uniform**, not uniform in value. The input's
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
without a class. Select its family with `--families`.

The supplier must establish
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
