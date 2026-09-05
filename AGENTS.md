# AGENTS.md

## Scope and evidence

Keep this file current when build steps, public APIs, defaults, or architecture
change. Deliver the requested scope. Prefer removing a redundant path over
adding a new policy, mode, framework, or tuning knob.

State a claim as verified only when this session produced supporting evidence.
Label historical results and untested hypotheses. Equal fill, iteration counts,
or residuals do not prove equal factors: use a digest over structure and values
when identity matters. Enumerate the complete denominator before an audit and
report checked/total plus anything not checked.

Historical rationale, measurements, and rejected experiments are preserved in
[implementation-history.md](docs/implementation-history.md). Consult the
relevant entry before changing an established algorithm or default; historical
numbers and retired configuration examples are not current verification.

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

Use focused regressions for affected behavior, then the relevant full suite.
Test CUDA changes with the required backend build options. A CPU-only test
cannot establish GPU correctness. Do not run timing campaigns concurrently
with builds or other laptop workloads. Stop repeating checks once they pass
unless another change or unresolved failure justifies it.

## Architecture and contracts

- Public headers are under `include/apxchol/`; `include/apxchol.h` is the
  convenience entry point. `src/factorization.cpp` and `src/solve.cpp` provide
  the compiled core. `benchmarks/src/v0/` is a frozen competitor baseline.
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
- GPU SpTRSV is dataflow-only. The old `APXCHOL_GPU_SPTRSV=dataflow` spelling
  is accepted; other nonempty values are errors. `APXCHOL_SPTRSV_FP16` controls
  factor storage (GPU default on, CPU default off); the old GPU-only alias is
  retired. GPU block setup is explicit opt-in
  through `APXCHOL_GPU_BLOCK_FRONTEND=on|force|1`, independent of host threads.
- Keep substantial mechanisms: compensated factor dropping, GPU long-row
  segmentation, critical-tail solving, incremental degrees, and connectivity
  preserving residual sparsification. Standalone residual coalescing policy
  is retired; coalescing and multiplicity remain sparsification internals.
- Pooled compaction must remain inside the factorizer's collective `omp single`:
  moving its decision to independently arriving workers can diverge barriers.
  Preserve factor-buffer lifetime through assembly and release transients at
  their last use. `clear()` does not release vector capacity.

## Benchmarks and experiments

Read [benchmarks/README.md](benchmarks/README.md) for the actual runner and
solver contracts. Use the existing runner/parser where it fits; do not make
adopting a new general harness a prerequisite for a small experiment.

`runner_common.classify()` accepts a solve only when the independently
recomputed true relative residual is at most `tol`, for every solver. Preserve
solver-complete setup/solve boundaries and report CUDA initialization
separately. Portable competitor implementations must retain their own
provenance labels and stopping semantics.

Native CMG is an opt-in external competitor: build `benchmarks/cmg/native`
against the private generated source and set `APXCHOL_CMG_NATIVE_BIN`.
The existing sweep writes separately labelled `cmg_packed/original-operator`
cells. Its generated core is serial; record actual affinity and effective
threads separately. Preserve original-operator grading, independent returned
solution checks, and the packed-port canonical exception documented in the
benchmark README. Do not commit generated proprietary comparison sources.
Tag logical-cell timeouts explicitly; a budget shared by calibration and
repetitions is not a numerical lower bound on one setup+solve invocation.

Report setup, solve, memory, and factor reuse separately. Evaluate the user's
actual objective; a faster solve with slightly slower one-RHS total is a
tradeoff, not automatically a rejected improvement. Do not multiply isolated
optimization ratios and call the product a measured cumulative improvement.

For timing, record source/binary/input identity, affinity, repetitions, raw
outputs, and the full planned denominator. Balance arm order and use suitable
null controls. Concurrent ranks on one node are not independent machines.
Separate correctness, timing validity, and the performance decision. If one
metric or cell fails a control, identify that exact scope, preserve valid
observations, and withhold affected conclusions; do not silently discard data
or manufacture a whole-campaign performance aggregate from a passing subset.
Retain the original preregistered verdict. Label later subset analysis as
retrospective, with its selection rule and narrower denominator.

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

`data/` and `results/` are ignored except tracked `results/cells/`, which feeds
the published benchmark charts. Keep internal reports and private comparison
sources outside tracked files. Do not publish them or contact collaborators
without authorization. Preserve source and evidence before any worktree
retirement; obey the user's destructive-operation approval requirements.
