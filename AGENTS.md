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

## Benchmark original-system stopping

The `original-v1` benchmark contract checks the defining operator/RHS at native
solve exits, reuses setup for bounded retries, and keeps one total iteration
budget (per component for split solves). Exhaustion is nonconvergence, never a
relaxed acceptance mark. Required stopping checks and retries are included in
Solve; `stop_check_s` is an included diagnostic, not a subtraction. At most eight
attempts/checks prevent zero-iteration loops; no short-plateau rejection rule.
Do not add per-iteration host checks or GPU transfers. Keep native kernels and
first-pass requests intact except documented tolerance-basis conversions.
The internal `detail/gpu_solve_session.h` shares the one-shot GPU setup lifetime
with benchmark retries; it is not a new public solver API. Packed CMG exposes no
reusable hierarchy and must charge/report each setup. ParAC patch 0007 replaces
untimed calibration with measured original checks. Current runners reject old
stopping receipts/caches; preserve historical results without relabelling them.

## Daint / Slurm campaign gate

Do not use a production Slurm allocation as the first integration test. Every
new or materially changed campaign must pass these gates, in order, against
the exact immutable source/package bytes that the production job will use:

1. State the complete planned-record denominator, resource ceiling,
   cancellation command, and whether a nonzero exit means infrastructure
   failure or a deliberate scientific rejection.
2. Run a package-owned login-node preflight. It must call the same prologue
   implementation as the batch job (not a second look-alike validator) and
   check the exact source commit/bundle and package hashes, clean source,
   matrix/dependency/toolchain identities, all required environment variables,
   and a fresh writable result namespace. Exercise missing, empty, and zero
   forms of optional Slurm variables used by the prologue. Validate a compiler
   by compiling and linking a tiny program with the campaign's exact compiler
   and flags; do not infer that from the presence of a support toolchain's
   `gcc`/`g++` driver binaries (for example, Clang may use that installation
   only for GNU headers and runtime libraries). Check resolved CMake cache
   values without assuming whether an explicitly supplied entry is typed as
   `STRING` or `FILEPATH`. Execute package tests with the same absolute Python
   interpreter used in batch; parsing the files with a newer Python is not a
   compatibility test. Disable or redirect bytecode so validation cannot dirty
   the immutable source checkout. For CUDA campaigns, record and validate
   `CMAKE_CUDA_COMPILER` and `CMAKE_CUDA_HOST_COMPILER` separately; neither is
   evidence for the C or C++ compiler identity.
   The known-working Daint CUDA baseline is LLVM 22.1.8 `clang`/`clang++` for
   C, C++, and nvcc's host compiler, with
   `CMAKE_CUDA_FLAGS=--allow-unsupported-compiler` recorded explicitly. It
   configured, built, and executed the GPU/dataflow backend on a GH200 on
   2026-08-31. Do not silently replace it with GNU merely because GNU appears
   in NVIDIA's supported-version table; any compiler change needs its own
   compute-node smoke and numerical receipt before production.
   Dependency Git identities must be canonical full object IDs (40 hexadecimal
   characters for SHA-1 repositories), resolved and compared rather than
   copied by eye. Reject ignored as well as untracked Python bytecode/cache
   before importing package-local modules; a late scan cannot make an earlier
   stale import trustworthy. Execute every package entry point with the exact
   target Python before submission; a syntax-only parse under another version
   is not runtime compatibility evidence.
3. Keep `SLURM_SUBMIT_DIR`, stdout, stderr, builds, and results outside the
   source checkout. Submit through a package-owned wrapper with explicit
   absolute `--chdir`, `--output`, and `--error` paths. The wrapper must run
   `sbatch --test-only` using the final account/resource arguments.
4. Run a small compute-node smoke allocation that builds the exact target(s)
   and exercises the relevant CPU/GPU runtime, affinity, and one tiny solver
   path. Persist a receipt binding source/package hashes, compiler/runtime,
   build outputs, and smoke results. Receipt verification must require the
   exact named artifact set (no missing, duplicate, or extra entries), and the
   production build must either reuse those immutable tested artifacts or
   prove that its rebuilt binaries, caches, compile commands, build IDs, and
   toolchain contract match the receipt. Every timing-producing entry point,
   not only the advertised wrapper, must require the validated receipt. The
   production job must verify it before starting; any source/package change
   invalidates it. Receipt fields such as test counts, completion state, job
   identity, and allocated resources must be derived from hashed smoke output
   and scheduler evidence, never emitted as unconditional PASS constants.
5. Only then submit the full campaign. Size wall time and per-process guards
   from observed upper bounds with margin; a censored timeout is a lower bound,
   not a runtime estimate. Preserve failed-run evidence in a separate result
   namespace and never reuse a partial run root. Timing jobs must persist the
   process mask and observed OpenMP-worker affinity; do not combine Slurm
   binding with `KMP_AFFINITY=norespect` unless the resulting confinement is
   explicitly verified. Concurrent ranks in one route-homogeneous phase are a
   single blocked timing observation, not independent seed replicates; either
   synchronize the measured stage and balance arms within each wave, or model
   the whole phase as the experimental unit. Bootstrap or other resampling
   must keep every measurement belonging to one declared independent cluster
   together. Resource ceilings must include CPUs, memory, and every GRES that
   Slurm grants implicitly (notably the GPUs granted by bare `--exclusive` on
   Daint). If a claim requires byte-identical factors, persist and compare a
   digest over factor structure and values; equal fill, iterations, and
   residuals do not establish factor identity.

Login nodes are for bounded validation, hashing, parsing, configuration probes,
and `sbatch --test-only`, not compilation-heavy or solver timing work. A
complete denominator that intentionally exits nonzero on a failed scientific
quality gate is evidence; a launch/build/parser/provenance failure is not.

### Submit rules learned from 92 non-completed jobs (2026-08-30..09-02)

- Batch scripts execute from `/var/spool/slurmd/`: never derive paths from `BASH_SOURCE`/`$0`/cwd; require `--export=APXCHOL_SOURCE=<abs>` and verify `git -C "$APXCHOL_SOURCE" rev-parse --is-inside-work-tree` before anything else (linked worktrees have a `.git` file). <!-- e.g. job 4560009, 4559543, 4558607 -->
- Pin every interpreter and tool by absolute path (`/usr/bin/python3.11` or `/user-environment/env/default/bin/python3`, `$LLVM/bin/clang++`); bare `python3` changes with the uenv/view boundary and is never a contract. Never pass `env PATH=...` expanded by the batch shell into an `srun --view` step. <!-- e.g. job 4559960, 4559443, 4560221 -->
- `srun` launches only a shell driver (`daint_build.sh`, `run_rank.sh`), never `cmake`/`python3` directly; every CMake line passes absolute `CMAKE_C_COMPILER`/`CMAKE_CXX_COMPILER` (+`CMAKE_CUDA_HOST_COMPILER`, `--allow-unsupported-compiler`) and asserts the identification lines. <!-- e.g. job 4559617, 4559632, 4560294, 4560272 -->
- Clang-built binaries carry `-DCMAKE_BUILD_RPATH=$LLVM/lib/aarch64-unknown-linux-gnu` (and `-Wl,--build-id` if provenance needs it); the driver fails on any `ldd | grep 'not found'` before the first rank starts. <!-- e.g. job 4560227, 4560228, 4576826 -->
- Every guard prints WHY to stderr before exiting (no bare `test ...` under `set -e`), and the batch stdout/stderr must carry each step's tail; a 0-byte log is a package defect. <!-- e.g. job 4568522, 4560313, 4582447 -->
- Run every audit/summary/verify script on the login node against a fixture that mirrors the submitted topology (zeros in timing fields, 4 ranks, real traces) before sbatch; after a complete denominator the audit writes its verdict and exits 0 - it never fails the batch. <!-- e.g. job 4571850, 4568596, 4559544, 4559443 -->
- `ctest` runs only tests whose executables the driver built (`--target apxchol analyze_factor unit_tests`, or `-R`); verify with `ctest -N` on the login node. <!-- e.g. job 4569010, 4576963, 4558309 -->
- Slurm `--output/--error` and build/run roots live outside any tree the script asserts clean; use `git diff --quiet HEAD` not `status --short`. <!-- e.g. job 4569006, 4584187 -->
- Never edit a script or lower `--time` after `sbatch`; `scancel` and resubmit, recording `sha256sum` of the script in the submit log. <!-- e.g. job 4559617, 4559638, 4559857 -->
- Budget before submit: `records_per_rank * measured_s + startup <= 0.67 * walltime`, per-record watchdog >= 3x a measured single record, per-rank memory from a measured smoke; a TIMEOUT is not a measurement. <!-- e.g. job 4560318, 4568082, 4567831, 4558610 -->
- An `--exclusive` Daint `normal` job grants the whole node (cpu=288, gpu=4): assert `SLURM_CPUS_ON_NODE=288`, never describe that allocation as a sub-node shape. <!-- e.g. job 4568522, 4576312 -->
- A per-rank gate writes its verdict and exits 0; do not combine a nonzero gate with `--kill-on-bad-exit=1` (it destroys the other ranks' records). <!-- e.g. job 4571918, 4582001, 4577006 -->
- CMake cache assertions use the type the script itself passes (`-DX:BOOL=ON` ⇔ `grep 'X:BOOL=ON'`) and are executed once against a login-node configure; `bash -n` + readonly-collision lint on every shell driver. <!-- e.g. job 4567920, 4575779, 4567992 -->
- One package, one login preflight, one debug-partition smoke, then production; never fix one defect per allocation. <!-- e.g. jobs 4559617..4560243 (6 tries), 4575916..4582372 (8 tries) -->
- A content-addressed capability is the trust root, not package-owned Python:
  the submit wrapper and spooled shell must authenticate the package manifest
  and every helper needed to interpret capability fields *before* importing or
  executing that helper.  A helper cannot establish its own identity, and a
  mutually consistent replacement of helper plus manifest must be a negative
  package test.  Hash and decode one captured byte string rather than hashing a
  path and reopening it. <!-- caught before allocation by R2b review, 2026-09-04 -->
- Exercise each path at the stage that consumes it: login-owned paths outside
  the view, and `/user-environment` paths inside the exact `uenv` view.  A
  login-only existence check cannot validate a later batch-view dereference.
  Re-hash long-lived matrix/dependency inputs at campaign end before accepting
  timing evidence. <!-- jobs 4600418, 4600521, 4600726 -->
- Smoke fixtures must exercise zero/absent child metrics and finalization after
  child exit.  Do not copy an expected matrix/RSS value into a record without
  measuring the object the child actually used.  A late infrastructure
  downgrade must scrub performance summaries and reseal a terminal failure;
  changing only the Slurm exit code is insufficient. <!-- jobs 4600229, 4600606 -->
- Independent review is one finite gate: implementer validation, one package
  audit, and—only if blocked—one delta audit of the repair.  Never submit while
  that audit is running, and do not restart a full review after it is clean.
  A failed smoke must first become an allocation-free regression test before a
  second smoke is submitted. <!-- job 4601725 -->

Taxonomy of those 92 jobs (157 total since 2026-08-30): harness/parser bug 21, unknown (empty or redirected logs) 21, walltime underestimate 11, bad path/missing file 10, user cancellation 8, CMake configure 5, Python-version syntax 4, compile/link 4, OOM 2, scientific gate rejection 2, Slurm resource shape 2, dependency identity 1, gcc-vs-clang 1. Only 2 of 92 were deliberate scientific rejections; 8.65 of 14.85 node-hours (58%) were infrastructure failures, 3.66 h of them TIMEOUTs. The recurring pattern was one defect fixed per allocation (block-sptrsv 6 tries, saturation-cut 6, r1b-correctness 8). Evidence: the sacct/log classification of 2026-09-02 (session scratch `final/daint_taxonomy.json`).

### Login-node preflight checklist (run in this order; no allocation)

- Identity: `git -C $APXCHOL_SOURCE rev-parse HEAD` == pinned commit; `git -C $APXCHOL_SOURCE diff --quiet HEAD --`; both `git -C $APXCHOL_SOURCE ls-files --others --exclude-standard` and the package's ignored-generated-state scan are empty; `git -C "$APXCHOL_SOURCE" rev-parse --is-inside-work-tree`; write `sha256sum *.sbatch *.sh *.py` to a fresh external preflight/submit directory, never back into the asserted-clean source (spool-copy rule, 4560009).
- Static lint: `grep -nE 'BASH_SOURCE|\$\{?0\}?' *.sbatch` empty; `grep -nE '^\s*srun .*\b(cmake|python3?)\b' *.sh *.sbatch` empty; `grep -nE 'env PATH=' *.sh` reviewed; `bash -n` on every shell file; `grep -n readonly` vs later assignments (4559543, 4559617, 4567992).
- Interpreter: every Python reference is an absolute path chosen for that stage. Parse every entry file with that exact interpreter, then execute every entry point with `-I -B` and its package-owned `--self-test`/`--help`; keep any generated state in external preflight scratch and verify the source remains clean. Repeat inside `uenv run ... --view=default` for view-owned stages (4559443, 4559960).
- Tools on the step PATH: `uenv run prgenv-gnu/26.3:v1 --view=default -- sh -c 'command -v clang++ nvcc cmake ninja git; c++ --version'` and assert clang++ is $LLVM/bin/clang++, not /usr/bin/c++ (4560221, 4559632).
- Configure only (no build): `uenv run ... --view=default -- cmake -S $SRC -B $S/pf-build <exact -D flags>` under `timeout 120`; require '-- Configuring done' and the compiler identification lines; then run every `grep ... CMakeCache.txt` assertion from the driver verbatim; `cmake --build $S/pf-build -- -n` plus `ctest --test-dir $S/pf-build -N` to confirm target/test consistency (4559617, 4575779, 4569010).
- Runtime linkage of any pre-existing binary: `ldd <bin> | grep -c 'not found'` == 0 and `readelf -d <bin> | grep -E 'RPATH|RUNPATH'` contains the LLVM runtime dir (4560227).
- Dependencies/inputs: `test -d` every offline dependency dir and matrix; FetchContent source set equals the anchor; dependency SHAs are 40 hex (4568503, 4576779).
- Audit/summary scripts against a fixture: `python summarize.py $S/fixture` with zero timing fields and 4-rank status records; `python audit.py --run-root $S/fixture` must print VALID; `verify_source.py` over the real `_deps` tree (4571850, 4568596, 4573575).
- Trust-bootstrap negatives: replace the capability, package helper, and
  manifest independently and as a mutually consistent set; every case must be
  rejected before helper import.  Require an exact package-test denominator,
  not merely an `OK` substring (zero tests also print `OK`).
- Lifecycle negatives: mutate a matrix after its initial manifest, tamper with
  a manifest-excluded receipt reference, make terminal-manifest verification
  fail, and force a late infrastructure downgrade after a ratio-bearing
  summary; every case must end as sealed infrastructure failure with no public
  performance aggregate.
- Output namespace: `mkdir -p` the parent of every --output/--error/run root and `test -w`; confirm they are outside $APXCHOL_SOURCE; `test ! -e <run root>` (4582964, 4569006).
- Slurm shape: `#SBATCH --time` matches the budget line in README; `--ntasks-per-node=4 --cpus-per-task=72` (full node) and script assertions use 288 CPUs / 4 GPUs; the agent runs `sbatch --test-only <script>` after the immutable package passes login preflight (4568522, 4576312).
- Budget arithmetic printed: planned records, measured seconds per record (from previous run logs), startup, walltime, watchdog, per-rank memory; abort if `plan > 0.67 * walltime` (4560318, 4568082).
- Only then: 5-minute `-p debug` smoke of build driver + one tiny record per rank; then production.


## Build options

Current options are declared in the root `CMakeLists.txt`. Historical backend
choices, retired knobs, and measurements belong in
[implementation history](docs/implementation-history.md), not this contract.

- `APXCHOL_BUILD_EXAMPLES` / `APXCHOL_BUILD_TESTS`: ON by default. Only tests
  require GoogleTest. `APXCHOL_BUILD_TOOLS`: OFF by default, independently builds
  `build/tests/bench_setup` and `build/tests/analyze_factor`, including when tests
  are disabled. `analyze_factor MATRIX --solve [--seed N]` reports setup, solve,
  iterations and the original-system residual for a component-compatible RHS.
- `APXCHOL_NATIVE_ARCH`: ON for local builds. Root, benchmark, and local Python
  builds share the architecture-specific compiler probe; portable Python wheels
  omit native tuning. `scripts/rebuild.sh [all|core|bench]` uses CMake dependency
  tracking without touching source files; both build helpers stop on failures.


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
- `operator_scan::triangles_bit_identical` is the proof that lets consumers of
  the caller's operator skip the per-entry transpose-partner search (about
  log2(column length) cache misses per entry, in the hub columns of power-law
  and IPM operators). The scan sets it only for strictly index-sorted columns
  with no explicit-zero or non-finite off-diagonal and equal per-triangle entry
  fingerprints (`detail::symmetric_entry_hash`, `csc_work.h`); it then also
  skips its own partner search, which otherwise runs unchanged as a second pass
  with the same count and witness. `detail::make_graph_from_operator` takes the
  proof only from `factorize_for_solver` and only when nothing was lumped;
  public `make_graph` never assumes it. The owned SpMV copy proves it for itself
  while copying (fingerprint includes the position in a duplicate run) and falls
  back to the pairing loop on any difference. Results are bit-identical with and
  without the proof; keep it that way.
- Loops over the rows or columns of the caller's operator (SpMV, owned copy,
  graph builder) split work with `detail::work_balanced_range`, not by index
  count: equal-count chunks carry 2x (IPM) to 4x (as-Skitter) the mean stored
  entries in the heaviest chunk. The bounds depend only on the pointer array and
  team size, so thread-ordered reductions stay bit-identical run to run. The
  level-scheduled SpTRSV deliberately does not (tried and removed 2026-08-18).
- Prune-walk skip (block-greedy on directed vec_pool): once AUTO has declined
  the exact incremental-degree cache, skipping rounds report a vertex whose raw
  adjacency count exceeds 4x the previous round's eligibility threshold at that
  count instead of walking it (`prune_and_degrees(..., skip_above)`). Its degree
  is then an upper bound and its dead entries wait for a later walk; that is
  sound only because this path's selector and eliminator filter dead entries
  themselves. A raw count only grows until the vertex is walked, so skipping
  every round hides hubs whose neighbours died (com-Youtube under GKS: +14 %
  iterations). Full walks therefore alternate with the skipping rounds and audit
  the skip they replace; a clean audit doubles the skipping rounds before the
  next one (1..8), a hidden eligible vertex resets them to 1. Do not remove the
  audit. Never active while the cache is on or undecided, nor under
  `APXCHOL_INCREMENTAL_DEGREE_SPARSE=0|1` (exact references).
  `APXCHOL_PRUNE_SKIP=<factor>` overrides the factor, `0` disables the rule,
  `APXCHOL_PRUNE_SKIP_TRACE=1` prints the audits.
- `factor_options.sampler` / `--sampler` selects `gks` (default) or
  `trace_cycle`. Trace-cycle emits at most d edges from a degree-d star and
  retains GKS for degree below three. CPU setup and full GPU-owned setup support
  it; it rejects forced CPU-shadow/export combinations instead of silently
  substituting a backend. The owned device sampler covers normal and oversized
  rows and retains per-session moment scratch. Trace-cycle retains positive
  subnormals; finite zero neighbors or unrepresentable numerical plans use
  input-only GKS fallback with the original seed before sampling. Invalid inputs
  and overflow remain errors. Device traces report these numerical fallbacks.
  (`heavy_core_k2` was removed 2026-09-17: same fill and iterations within one
  of trace-cycle, setup +4.6%, one-RHS total +2.1%, 16-RHS total -1.8% on 22
  matrices; `benchmarks/daint/SAMPLERS.md` keeps its published measurements.)
- Trace-cycle rows with more than 128 canonical neighbors use cooperative warp
  moment/prefix scans, cutoff reduction, parent searches and edge emission.
  Suffix maxima repair floating CDF monotonicity. Lane zero retains the core
  shuffle; light parents use indexed direct-CDF draws after its actual rejection
  consumption. Intended laws are unchanged; floating folds/cutoffs and seeded
  factors can differ from CPU. Input-only fallback and range guards remain.
  Owned oversized-only rounds use one warp per block; mixed rounds retain larger blocks.
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
current comparison has 20 (CPU GKS/trace, GPU GKS at two degree cutoffs and
GPU trace-cycle at q=0.8).
Canonical MATLAB CMG is excluded from current charts; packed serial CMG remains.
Preserve historical outcomes, effective versus requested threads, and source/binary/cell hashes.
[benchmarks/README.md](benchmarks/README.md) defines runner/solver contracts;
use the existing harness. Rendering runs no benchmarks.
An explicit `--solver` must name a solver compiled into that benchmark binary;
unknown or unavailable solvers exit 2 before matrix loading, without a CSV row.
`--solver none` is accepted only for input exports and component inspection.
Julia/CMG operator exports share the harness's staged cache writer: only a
successful export is published; failed or interrupted writes leave no cache.
ParAC retains its separate timing-accounted adapter cache.
`benchmarks/weighted_inputs.py` generates four explicit weighted graph variants;
`sweep_fair.py --matrix-manifest` loads additional hash-bound input records.
Each graph component receives a backbone tree. Weight distributions and seeds
are recorded in cell metadata, and changed input identities invalidate resume.

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
provides the common presentation path. Its optional `--fill-cells` consumes an
explicit derived store with common `2*offdiag(L)/offdiag(A)` fill; preserve missing
counts, source solve status and reported rounding uncertainty.

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
