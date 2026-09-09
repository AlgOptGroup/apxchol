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
- GPU-owned numerical setup requires all three existing flags:
  `APXCHOL_GPU_BLOCK_FRONTEND=on|force|1`, `APXCHOL_GPU_ROUND_SHADOW=force`
  and `APXCHOL_GPU_FACTOR_FINALIZE=force`. It applies to an internal consuming
  block-greedy/tree solve on directed AoS. It can eliminate supported rounds on
  device and install the append log through dataflow SpTRSV. Public factorization,
  custom strategies, exported factors and `keep_factor=true` retain their audited
  or ordinary host path; copied capsules keep independent ownership validation.
- An eligible compressed, sorted, unique, fully paired symmetric CSC operator
  initializes the owned graph directly after operator validation. Its fresh,
  unchanged operator view supplies the pairing proof only after strict layout
  checks and when all stored off-diagonals are nonzero. Stored zeros, lumped
  inputs and raw/test entry points retain full structural mate checks. Other
  stored formats retain the host import fallback before device mutation. GPU PCG
  constructs the permuted operator CSR on device only with all three existing
  owned-setup flags enabled and its format checks satisfied. Ordinary calls and
  unsupported formats keep host construction. Canonical lower values and
  lossless-fp32 selection are unchanged.
  Preserve original-system residual grading and full fallback validation.
- `factor_options.sampler` / `--sampler` selects `gks` (default), `trace_cycle`
  or `heavy_core_k2`. Both cycle families emit at most d edges from a degree-d
  star and retain GKS for degree below three. CPU setup and full GPU-owned
  setup support them; alternative samplers reject forced CPU-shadow/export
  combinations instead of silently substituting a backend. The owned device
  sampler covers normal and oversized rows and retains per-session moment scratch.
  Both cycle families retain positive subnormals; finite zero neighbors or
  unrepresentable numerical plans use input-only GKS fallback with the original
  seed before sampling. Invalid inputs and overflow remain errors. Device traces
  report these numerical fallbacks.
- The owned selector uses degree/hash/id priority, all ties at the degree
  quantile, and at most four immutable decision/commit passes with stable
  unresolved-candidate compaction. It guarantees independence and progress,
  not maximality or the CPU regional selected set. Generic CSR selection keeps
  its regional law. Seed and actual round index reach the owned priority rule.
- Owned elimination packs rows of at most 128 physical slots into local batches;
  oversized rows keep the global stable-sort fallback. Counts publish batch
  descriptors and compact offsets; one warp emits each batch into its original
  disjoint factor/fill slots. Preserve ordered duplicate sums, raw degree,
  canonical weight/neighbor order, prefix/upper_bound sampling and random streams.
  Compact-owned row counts and dead-fill excess scratch reuse retain provenance
  and lifetime checks. Requested and retained scratch enter memory preflight.
- The owned loop makes one CPU-style sparsification decision at its first
  low-yield selection boundary. The existing residual switch/traffic gate
  controls directed coalescing, one bucket/ordinal forest and conditional
  Bernoulli/HT sampling; normalization keeps full-stream ordered 16384-item
  chunks. Revoke and reprepare the preview without advancing the numerical
  round. Rebuild degrees/fingerprints and the handback edge-accounting basis;
  preserve active/excess and factor append state. Generic pre-import inputs
  retain the CPU coalescer fallback. Validate `GpuOwnedSparsify.*` on device;
  source integration requires fresh native and performance validation.
- Device finalization supports the existing fp16/drop contracts and omits host
  factor values only for a unique internal consuming owner. Private finalized
  factors build dataflow plans on device; generic capsules retain host planning
  and validation. Host factor metadata remains. Optional GPU forest-tail thinning
  is not included; CPU residual sparsification remains unchanged. Adopted factors
  and device-built operators complete their queued setup work before returning.
  Host-built setup keeps its existing synchronization boundaries; the common
  benchmark harness synchronizes every measured API boundary. Optional receipts/events use
  existing verbose or setup/frontend trace flags. Memory, nnz and SpTRSV statistics
  keep their own controls. No private experiment compile definition is required.
- Validate the owned path with `GpuBoundedSelection.*`, `GpuDirectCsc.*`,
  `GpuOperatorCsr.*`, `GpuOwnedPrefix.*`, `GpuFactorFinalize.*`, the audited round
  and adoption fixtures, and sanitizer/leak checks. Source integration is not
  performance acceptance: compare the default route against current main and
  the owned route against its frozen research reference, with original-system
  quality, setup, solve, one-RHS total, RSS and owner memory.
- CUDA PCG reuses the host RHS buffer for the solution download and unpermutation
  only after its upload has completed and no further host RHS reads remain.
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
27-matrix denominator: the historical profile has 18 series, while the explicit
current comparison has 20 (CPU GKS/trace and GPU GKS at two degree cutoffs).
Preserve unavailable canonical MATLAB CMG cells,
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
Patch0006 shares the Graph producer with an in-memory entry. CPU Graph/AMD charges
required transform, ordering/permutation and explicit final GC; common input read
and measured final serialization are separate. Require finite reconciled intervals
and source/schema-bound caches; keep inclusive diagnostics. Physics/GPU complete
timing and native adapter/factor/workspace charges remain unchanged.

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
