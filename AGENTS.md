# AGENTS.md

## Scope and evidence

Keep this file current when builds, APIs, defaults or architecture change.
Deliver the requested scope; prefer removing redundancy to adding modes or knobs.
Verify claims with current-session evidence; label historical results and hypotheses.
Factor identity requires structure/value digests, not equal fill or residuals.
Audits report the complete denominator, checked/total and exclusions.
Consult [implementation history](docs/implementation-history.md) before changing
an established algorithm/default; historical examples are not current verification.

## Build and tests

The root project builds the library, CLI, and unit tests. The separate
`benchmarks/` project fetches competitor implementations.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j6
ctest --test-dir build --output-on-failure
./build/tests/unit_tests --gtest_filter='FactorizeTest/*.PermutationIsValid'
./build/apxchol path/to/matrix.mtx --random-rhs --tol 1e-8
```

CLI input requires `--rhs file.mtx` or `--random-rhs`. Input kind is detected
and reported; `--input-kind` overrides it. Random RHS is projected separately
on each Laplacian component. An explicitly supplied RHS is never altered.

Relevant build options:

- `APXCHOL_USE_CUDA=ON`: our dataflow SpTRSV and GPU-resident PCG. The library
  links `cudart` only. There is no cuSPARSE backend or build option. Benchmark
  competitors independently require cuSPARSE/cuBLAS; distinguish their driver
  linkage from our library linkage.
- `APXCHOL_POOL_FP32=OFF`: fp64 residual-pool baseline. Factor values remain
  fp32; fp16 storage is a runtime choice, not another build configuration.
- `APXCHOL_64BIT_EDGE_INDICES=ON`: wide cumulative edge offsets.
  `APXCHOL_64BIT_NODE_INDICES=ON` additionally widens vertices and implies
  wide edges. Do not use the deprecated combined index option in new code.

Run focused regressions, then the relevant full suite; repeat only after changes
or unresolved failures. CUDA correctness requires the appropriate build and device.
Do not time alongside builds or other laptop workloads.

When `compute-sanitizer` is available, CUDA builds register
`gpu_factor_finalize_leak_check`; run it with:
`ctest --test-dir build-cuda -R '^gpu_factor_finalize_leak_check$' --output-on-failure`.
Without it, ordinary tests do not establish leak freedom. Device-wide
`cudaMemGetInfo` equality is not a process-leak assertion.

## Architecture and contracts

- Public headers are under `include/apxchol/`; `include/apxchol.h` is the
  convenience entry point. `src/factorization.cpp`, `src/operator_class.cpp` and
  `src/solve.cpp` provide the CPU compiled core. `benchmarks/src/v0/` is a frozen competitor baseline.
- `operator_class.h` and `src/operator_class.cpp` own operator validation and
  M-matrix lumping. `src/mtx_input.h` owns CLI-only interpretation of graph
  adjacency versus an assembled operator. Bindings must use the operator
  contract, not CLI guesses.
- `graph/` provides adjacency layouts. Directed `vec_pool_aos` is the
  high-level default; indexed `vec_pool` remains supported. Changing storage
  must not silently choose a different selector.
- `solver/elimination/` owns clique sampling and the public eliminator seam.
  Canonical neighbors use comparison sort by weight and vertex; sampling uses
  exact cumulative weights and `upper_bound`. Preserve estimator semantics,
  tie ordering, and per-elimination random streams.
- `solver/partition/` contains block-greedy, priority-greedy, and Baumann-Kyng.
  `factor_options.h` is authoritative for defaults and their rationale.
  Preserve deterministic conflict resolution and thread-team fallback rules.
- CPU `solver/sptrsv/omp.h` supports `APXCHOL_CPU_SPTRSV=auto|levels`.
  AUTO uses the structural critical-tail schedule when metadata permits;
  `levels` is the reference. Share row arithmetic across schedules and storage.
  Research schedules on other branches are not production modes.
- CPU SpTRSV's nnz-sized CSR/CSC index and value output buffers use
  `big_alloc<T,32,false,false>`: the transpose/copy fully overwrites them, so
  writer threads perform the first touch. Pointer, diagonal, scale, and other
  allocations retain the populated, value-initialized default. Require a full
  overwrite before first read when using the output-buffer specialization.
- The CPU Laplacian L11 temporary index/value arrays likewise use uninitialized
  owning arrays: the existing column copy writes every retained entry before
  any read. Preserve their early release after compaction or their last use.
- GPU SpTRSV is dataflow-only. The old `APXCHOL_GPU_SPTRSV=dataflow` spelling
  is accepted; other nonempty values are errors. `APXCHOL_SPTRSV_FP16` controls
  factor storage (GPU default on, CPU default off); the old GPU-only alias is
  retired. GPU block setup is explicit opt-in
  through `APXCHOL_GPU_BLOCK_FRONTEND=on|force|1`, independent of host threads.
- Resident factor finalization is research-only. With
  `APXCHOL_GPU_ROUND_SHADOW=force`, `APXCHOL_GPU_FACTOR_FINALIZE=force`,
  `APXCHOL_SPTRSV_FP16=0`, `APXCHOL_FACTOR_DROP=0` and `vec_pool_aos`, the
  device append log becomes fp32 CSR L/LT and is adopted by dataflow SpTRSV.
  The audited prefix stays on device; the CPU tail and permutation upload.
  The trusted finalizer downloads only O(n) row pointers for host plan packing;
  external adoption capsules retain full structural validation. Unique internal
  consuming solves omit host CSC row/value arrays; public factorization,
  custom factors and `keep_factor=true` retain exportable arrays. Copied public
  capsules use ordinary host setup. CPU authoritative elimination, CPU tail,
  metadata and host plans remain; do not call this fully resident setup or a
  measured speedup. Validate `GpuFactorFinalize.*`, `GpuRoundShadow*.*`,
  `GpuSptrsvAdoption*.*`, `GpuDataflow.*` and `GpuHostPrep.*` on a CUDA device.
- With the round shadow and GPU block selector both explicitly enabled, an
  accepted, CPU-certified resident residual projects its paired incidences
  directly into the same selector's device COO/CSR and active mask. The handoff
  preserves multigraph multiplicity and binds the live producer, CUDA device
  and consumed selection/topology generations before mutation. It replaces
  CPU-produced endpoint/id uploads for that handoff only; independent CPU
  snapshots, elimination, certification, order/excess refreshes, occupancy
  handoff and tail remain. No new public option or performance claim is implied.
- The internal consuming finalizer also omits duplicate CPU prefix factor
  entries: workers stream the same fp32 entry hashes for the mandatory shadow
  comparison. Vertex/diagonal/count metadata and all CPU graph/RNG operations
  remain. CPU tail entries are still allocated for finalizer upload; public,
  custom and `keep_factor=true` paths retain their payloads. The assembly trace
  reports requested/written/omitted factor-entry bytes, excluding monotonic
  allocator chunk slack and metadata; these are not peak-RSS measurements.
- An internal consuming solve with all three existing GPU block, round-shadow
  and factor-finalizer flags forced may execute an at-most-two-round GPU-owned
  prefix, then download the full ordered residual and factor-column headers
  once. It performs no CPU numerical replay or full snapshots between those
  rounds. Direct slab handback preserves per-owner order, duplicate weights,
  active/excess state and ever-added edge accounting. CPU continuation starts
  with fresh caches and retained factor entries; device prefix entries remain
  resident. Owned acceptance is separate from CPU certification. Public/custom/
  keep-factor paths retain the audited route; ordinary defaults are unchanged.
  This bounded ownership milestone is not full GPU residency or a speedup claim.
- GPU PCG's host operator builder uses column ownership for compressed,
  strictly sorted, unique, fully paired symmetric CSC input. Preserve canonical
  lower values, lower-only fp32 exactness and sorting by permuted column ids.
  Unsorted, duplicate, unpaired and uncompressed inputs retain the general
  atomic builder. The CUDA-free helper is internal; no new runtime knob exists.
- Keep substantial mechanisms: compensated factor dropping, GPU long-row
  segmentation, critical-tail solving, incremental degrees, and connectivity
  preserving residual sparsification. Standalone residual coalescing policy
  is retired; coalescing and multiplicity remain sparsification internals.
- Residual importance normalization uses fixed
  16,384-item ordered partial sums followed by an ordered block fold, with one
  persistent OpenMP team for the initial sum and up to six updates. The same
  ordered input gives the same normalization at every thread count; arithmetic
  intentionally differs from the legacy global serial fold. Final statistics,
  conditional Bernoulli/HT law and connectivity backbone remain unchanged.
  `ResidualBlockedNormalization.*` and `ResidualBlockedGraph/*.*` check the new
  contract. No runtime knob is added. The implementation was validated on
  Daint before integration; see [CPU-RESIDUAL-BLOCKED.md](CPU-RESIDUAL-BLOCKED.md)
  for the bounded setup improvement and the separate limits of the timing study.
- Full symmetric CSC inputs with unique sorted indices construct directed pool
  incidences by column ownership; upper incidences use canonical lower weights.
  Duplicate, uncompressed, one-triangle or unpaired stored patterns retain the
  general graph builder. No extra public builder or runtime knob is exposed.
- Lazy segmented adjacency reservations request THPs only when the kernel's
  reported PMD granularity is at most 2 MiB (cached once); larger or unknown
  granularities use `MADV_NOHUGEPAGE`. This preserves the measured laptop
  benefit while preventing Daint's 512 MiB first-touch inflation. Intermediate
  geometries are not claimed performance-optimal. Fully sized factor/output
  buffers retain their separate `big_alloc` policies; there is no runtime knob.
- Pooled compaction must remain inside the factorizer's collective `omp single`:
  moving its decision to independently arriving workers can diverge barriers.
  Preserve factor-buffer lifetime through assembly and release transients at
  their last use. `clear()` does not release vector capacity.

## Benchmarks and experiments

Daint is primary (`benchmarks/daint`); laptop data are historical. Preserve the
shared 27-matrix/18-series denominator, unavailable canonical MATLAB CMG cells,
effective versus requested threads, and source/binary/cell hashes.
[benchmarks/README.md](benchmarks/README.md) defines runner/solver contracts;
use the existing harness. Rendering runs no benchmarks.

Grade **every retained original-system residual** against the common tolerance.
Calibration failures/caps must not trigger fallback retained runs. A median
repetition selects timing fields only. Whole-cell/prerequisite deadlines are
not lower bounds on individual solves. Report setup, solve, memory and reuse
separately; preserve complete timing boundaries and separate CUDA initialization.
Do not wrap timed calls in `VramSampler`: nvidia-smi polling perturbs GH200 setup.
Measure VRAM separately; missing peaks stay unknown.

ParAC: patch0005 uses Neumaier compensation for the Physics producer global
sum; preserve ordering, per-column sums and both ±1e-9 thresholds. External
checkouts require patching before rebuilding. Physics inputs with positive
stored off-diagonals are unsupported for original-operator comparison,
**before** preparation cache hits or fallbacks.

Native CMG is opt-in via `APXCHOL_CMG_NATIVE_BIN` and `benchmarks/cmg/native`.
Keep private generated sources untracked. Its generated core is serial;
record affinity/effective threads and the `cmg_packed/original-operator` label.
Preserve independent returned-solution checks, explicit RHS support and the
canonical exception in the benchmark README.

Use `--warmup N`, `--repeat R` and existing thread-scaling scopes as needed.
`--dump-rhs` exports the common RHS without solving. `render_snapshot.py`
provides the common presentation path.

Timing campaigns record immutable source/binary/input identities, affinity,
repetitions, raw outputs and the entire planned denominator. Balance arm order
and use null controls. Concurrent ranks on one node are not independent machines.
Separate correctness, timing validity and the performance decision. Preserve
original verdicts and valid observations when a subset fails; label later
subset analysis retrospective. A faster solve with slower one-RHS total is a
tradeoff. Never multiply isolated ratios into a cumulative speedup claim.

Before a Daint submission:

1. State the hypothesis, planned records, CPU/GPU/memory/walltime ceiling,
   cancellation command, and whether the laptop may be shut down.
2. Perform bounded login-node checks of the exact package, target interpreters,
   paths, compiler/link configuration, allocation arguments, and analyzer on
   representative success/failure fixtures. Use `sbatch --test-only`. Keep
   heavy builds and solver runs off login nodes.
3. Use a compute smoke when new compiler/runtime/resource behavior needs it
   or debug materially shortens turnaround. Reuse valid evidence for an
   unchanged binary; a parser-only repair does not require another numerical
   campaign. Respect the user's explicit smoke and resource instructions.
4. Use absolute external output/build/run paths and a fresh result namespace.
   A linked worktree has a `.git` file: use `git rev-parse`, not a directory
   assertion. Keep source immutable after submission. Redirect `srun` stdin
   when it runs inside a loop reading a phase plan.
5. Size watchdogs and walltime from measured or censored record bounds with
   margin. Include implicitly allocated GPUs in exclusive-node budgets and
   keep nested steps within the allocation. Record actual affinity. Verify
   the returned job ID; distinguish submitted, pending, running, and complete.
6. Collect the result and diagnose the first cause of failure before retrying.
   Preserve raw evidence. Make recoverable harness failures local regressions;
   use one independent review and review only its repair delta if necessary.
   A scientific rejection must not abort unrelated ranks or erase their data.

## Workspace and handoff

Check the active branch and dirty state before editing. The current checkout
may be an older research branch even when `origin/main` has newer work. Give
each editing agent explicit file/worktree ownership and keep laptop timing
in one lane.

Keep work that must survive shutdown in persistent paths, not `/tmp` clones
or worktrees. Before handoff record each active branch/commit, dirty patch,
result path, exact remote job ID/status, and next action in one durable local
ledger. Confirm the checkpoint exists. An agent's assignment is not evidence
that implementation or a cluster job started.

`data/`, `results/` and `benchmarks/results/` are ignored. Daint is the current
performance source: committed extracts, coverage and source/protocol metadata
live in `benchmarks/daint/`; audited campaign cell stores remain private.
`results/cells/` is a generated local store, not automatically the source of
published Daint data. Renderers default to ignored `results/plots/` previews;
publication requires an explicitly selected store and output.
`thread_scaling.py --compact` renders up to three scoped matrices as absolute
setup/solve times and log-log speedups, with separate extract names; full-study
figures and data remain available. The retired
laptop snapshot and its 612 legacy T16 cells are preserved in
`benchmarks/archive/laptop-20260908/`; do not blend them into current results.
Keep internal reports and private comparison sources outside tracked files. Do not publish them or contact collaborators
without authorization. Preserve source and evidence before any worktree
retirement; obey the user's destructive-operation approval requirements.
