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

- Batch scripts execute from `/var/spool/slurmd/`: never derive paths from `BASH_SOURCE`/`$0`/cwd; require `--export=APXCHOL_SOURCE=<abs>` and assert `test -d "$APXCHOL_SOURCE/.git"` before anything else. <!-- e.g. job 4560009, 4559543, 4558607 -->
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

- Identity: `git -C $APXCHOL_SOURCE rev-parse HEAD` == pinned commit; `git -C $APXCHOL_SOURCE diff --quiet HEAD --`; both `git -C $APXCHOL_SOURCE ls-files --others --exclude-standard` and the package's ignored-generated-state scan are empty; `test -d $APXCHOL_SOURCE/.git`; write `sha256sum *.sbatch *.sh *.py` to a fresh external preflight/submit directory, never back into the asserted-clean source (spool-copy rule, 4560009).
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


Build options (root `CMakeLists.txt`):
- `-DAPXCHOL_USE_CUDA=ON` — switch the SpTRSV backend from OpenMP level-sets to the GPU (`include/apxchol/solver/sptrsv/cuda.h`: **our sync-free dataflow kernel, the ONLY GPU SpTRSV backend of the library** — the AUTO choice since 2026-08-18, `cuda_dataflow.h` / `src/cuda_dataflow.cu`, see the paragraph after this one; plus cuSPARSE SpSV as an opt-in COMPARISON backend with env `APXCHOL_GPU_SPTRSV=cusparse` — ONLY on a build that opted in with `-DAPXCHOL_CUDA_WITH_CUSPARSE=ON` (default OFF; see the "No closed-source dependencies" bullet below)) + GPU-resident PCG (`pcg_cuda.h`, our own SpMV / vector kernels since 2026-08-18, `pcg_cuda_kernels.h` / `src/cuda_pcg_kernels.cu`). Defines `APXCHOL_USE_CUDA`, requires CUDAToolkit; **the CUDA library links `cudart` only** by default (no cuSPARSE, no cuBLAS). Runtime env knobs of the GPU SpTRSV (all read at every setup): `APXCHOL_GPU_SPTRSV=dataflow|cusparse` — an explicit override, exactly that backend, no fallback (`=cusparse` is NOT protected by any fitting check); **unset = AUTO = the dataflow backend, unconditionally**: it has no analysis buffer — its device state is the two CSRs the level-set backend needs anyway plus 8 B/row and O(n) batch tables — so there is nothing to fit and nothing to decide, and it beats cuSPARSE on every measured workload (`gpu_pcg_loop` ms/iter, RTX 4090 Laptop, T=16, bg+tree[vec_pool], default drop: **iter0040 cuSPARSE 1.08 → dataflow 0.98, grid_2000 3.54 → 3.20**) with bit-deterministic results. The pre-dataflow OOM-aware AUTO rule — cuSPARSE unless its two SpSV analysis buffers (`cusparseSpSV_bufferSize`, O(nnz): 129 MB iter0040-after-drop, 766 MB grid_2000, several GB on the giant social factors) fit into free device memory minus a 10%-of-total margin minus a caller-declared reserve (`cuda_sptrsv::set_reserve_bytes`) — was **removed 2026-08-20 with the fp64 build that was its only reachable caller**, along with `set_reserve_bytes` / `auto_prefers_dataflow` / `dataflow_supported`; an explicit `=cusparse` gets cuSPARSE with no fitting check. The decision is `cuda_sptrsv::backend_reason()`, printed under `APXCHOL_VERBOSE` as `[apxchol] GPU SpTRSV backend: ...` (`backend_forced()` false = AUTO). **The O(n)-schedule LEVEL-SET GPU backend (`cuda_levelset.h` / `src/cuda_levelset.cu`, `APXCHOL_GPU_SPTRSV=levelset`, `cuda_host::compute_levels`, `cuda_sptrsv::levelset()`) was REMOVED 2026-08-20** — the dataflow kernel beat it on every workload measured in the same session (`gpu_pcg_loop` ms/iter iter0040 0.98 vs 1.54, grid_2000 3.20 vs 6.26; s/iteration com-Orkut 0.77 vs 0.87, com-LiveJournal 0.21 vs 0.25), needs strictly less device state (no O(n) level schedule) and is deterministic in the same way; the value is now an unknown one, so it lands on AUTO with a stderr note, and `benchmarks/sweep_fair.py`'s retry-with-`levelset`-after-OOM went with it (AUTO has been O(n)-state since 2026-08-18, so there is nothing to fall back FROM). Its independent-implementation cross-check survives, retargeted to dataflow-vs-CPU (`GpuDataflow.*`). The dataflow backend is deterministic run to run, cuSPARSE SpSV is not (same factor, ±1 PCG iteration / different RelRes — the CUDA build's `SolveTest.DeterministicWithSameSeed` flaked under cuSPARSE and passes under the dataflow AUTO); measured on the RTX 4090 Laptop (T=16, bg+tree[vec_pool], `gpu_pcg_loop` ms/iter, 3 reps): cuSPARSE ≈ 3.7 ms/iter on grid_2000 vs level-set ≈ 6.5 (both far below GDDR bandwidth — launch/latency-bound, one launch per level; bg's IPM level count varies 79–138 run to run at T=16 and the level-set time scales with it), on iter0040 (5 interleaved reps, medians) the drop takes cuSPARSE from 6.6 (6.1–13.1) to 1.08 ms/iter and the level-set from 2.7 (2.6–6.6, 95–141 levels) to 1.58 (81–96 levels — the drop also removes dependency edges, so the DAG gets shallower), at unchanged iteration counts (T=1: iter0040 46/46, grid_2000 50/50, grid_500 42/42); the **compacting factor drop runs on this backend too** (same `APXCHOL_FACTOR_DROP` knob and default as the CPU, applied on the host L11 arrays before the upload — one implementation, `factor_drop.h`, see "Compacting factor drop"; `cuda_sptrsv::drop_stats()` / `stored_nnz()`, and the benchmark's FILL line prints `stored_nnz=` plus `gpu=<backend>/<storage> factor_dev_MB= dev_delta_MB= cusparse_buf_MB=`); `APXCHOL_GPU_DF_SPLIT=<off-diagonals>` — the dataflow kernel's ROW-SEGMENTATION threshold (see the dataflow bullet below): **unset = the derived per-direction default**, a positive value pins it, **`=0` turns segmentation off** and is the rollback switch (the plan, and the output, become bit-identical to the unsplit kernel); `APXCHOL_GPU_SPTRSV_STATS=1` prints each direction's row-length histogram and plan sizes at setup (`[df-stats]`), which is the evidence the threshold rule was derived from; `APXCHOL_GPU_SPTRSV_FP16=1` (default off) — **fp16 factor storage on our dataflow kernel** (AUTO = the fp16 dataflow kernel; `=cusparse` is impossible — a stderr note, then the AUTO choice: cuSPARSE 12.8 SpSV rejects `CUDA_R_16F` matrix values with `CUSPARSE_STATUS_NOT_SUPPORTED` for every A/vector/compute combination — probed): the CPU FP16_SCALED contract (`cuda_host.h` file header: binary16 of the column-scaled `L~ = L D^-1`, separate fp32 `diag[j] = fp32(L_jj)/s_j` and `inv_scale[j]`, forward returns `D y`, back reads its input times `inv_scale^2`, compute in `cuda_value_t` = fp32 after the half→float widen), host-side narrowing through `lowprec.h`'s `fp16_t` (same RNE / subnormal flush as the CPU build; `APXCHOL_FP16_KEEP_SUBNORMAL=1` honoured), and the column rounding residual folded into `diag[j]` BY DEFAULT (`APXCHOL_LOWPREC_DIAG_COMP=0` turns it off — the CPU reads the same variable with the opposite default; without it the Laplacian path pays iter0040 fp16 45 → 64 iterations, with it fp16 matches fp32's counts). Device factor bytes per stored entry: dataflow fp32 2×(4+4) B, fp16 2×(4+2) B + 8 B/row; cuSPARSE 4+4 B + its O(nnz) analysis buffers (iter0040 device factor 82.5 → 68.2 MB with fp16, grid_2000 414 → 358 MB). `APXCHOL_GPU_MEM_DEBUG=1` prints the device-memory lines. Host-side prep (`cuda_host.h`) is CUDA-free and unit-tested on every build (`GpuHostPrep.*` in `tests/test_sptrsv_drop.cpp`: the arrays the GPU uploads are the arrays `omp_sptrsv` stores; the fp16 storage contract; the dataflow batch packing).
  - **Retired fixed-priority GPU setup path (2026-08-23):** the CPU `priority_greedy` selector remains available, but its GPU residual-topology frontend and `APXCHOL_GPU_PRIORITY_FRONTEND` / `APXCHOL_GPU_PRIORITY_TRACE` knobs were removed rather than kept as indefinite research surface. The direct forced-GPU campaign checked 90/90 T=16 records over ten matrices and produced a 1.102x setup geomean versus the bracketed CPU path. A separate exact asynchronous persistent kernel, shaped after ECL-MIS but independently implemented with the repository's full priority order, preserved factors and iterations in 54/54 bracketed runs; it still made selector time 1.057x and `find_partition` 1.069x overall (including 1.580x selector time on com-Amazon), while setup was noise-level 0.989x. It therefore did not rescue the mode. The GPU block-region frontend below is independent and remains.
  - **GPU block front-end transfer path (2026-08-23):** active vertex ids, active degrees, candidate filtering and the selected-degree work sum remain device-resident across rounds; production does not upload the active list or download all active degrees. Parallel elimination exposes non-owning views of its existing `deferred_edge {u,v,w}` buffers. A bounded 1M-edge (16 MiB maximum) staging buffer copies those records contiguously and a GPU kernel extracts `{u,v}`, avoiding both an endpoint-only CPU write and a second host copy. The retained benchmark history records a 0.973x setup geomean against the previous transfer path over grid_500, iter0040, com-Amazon, as-Skitter and coPapersDBLP; a strided `cudaMemcpy2D` gather was rejected after making `advance` 8-27x slower. This machinery is now owned solely by the block-region frontend.
  - **GPU block/region setup front-end (2026-08-23; `gpu_block_frontend.h` / `cuda_block_frontend.cu`):** for `bg+tree` on either pooled adjacency representation (`vec_pool` or `vec_pool_aos`), `APXCHOL_GPU_BLOCK_FRONTEND` is **AUTO when unset**: use the GPU at **at most 8 host OpenMP threads**, otherwise stay on the CPU. Supporting both layouts is semantic, not just performance: changing graph storage must not silently change the selector, and `VecPoolAos.SerialFactorMatchesIndexedVecPoolByteForByte` guards the resulting factor equality. `=0|off` is the rollback; `=1|on|force` bypasses the thread governor and fails loudly if cooperative launch or memory fitting is unavailable; `=auto` names the default. The boundary follows the measured CPU-selector scaling crossover, not graph size: a strict GH200 single-RHS campaign checked 270/270 records over nine matrices, bracketed CPU/GPU/CPU; total-time geomeans were **0.711× / 0.762× / 0.826× / 0.927× at T=1/2/4/8**, then **1.050× at T=16** (three per-seed geomeans 1.034/1.055/1.062) and **1.201× at T=72**. Peak host RSS is about **1.21×** overall; the GPU path still loses on com-Amazon (1.21× total at T=8) and grid_2000 (1.41×), so this is a throughput default, not a per-matrix dominance claim. The selector reuses the device-resident residual topology while preserving the independent-region model: one warp serial-greedy-scans each contiguous candidate region, a snapshot pass removes cross-region conflicts under `(degree,index)`, and a cooperative frontier-restricted repair restores maximality. Region count defaults to the number of region-scan warps that can be resident on the GPU (`3648` on the RTX 4090 Laptop); once the candidate count fits within that resident-warp capacity, setup permanently hands the remaining rounds back to the already-current CPU graph instead of paying fixed cooperative-kernel and round-trip latency for under-occupied GPU scans. A warm RTX 4090 Laptop campaign checked 405/405 factorization records over nine matrices, five host thread counts and three seeds (360 converged single-RHS totals; the 45 as-Skitter RHS runs were component-incompatible): handoff/full-GPU total geomeans were **0.984× / 0.973× / 0.953× / 0.947× / 0.938× at T=1/2/4/8/16**, and handoff/CPU was **0.787× / 0.869× / 0.939× / 0.991× / 0.992×**. The handoff changes later CPU-vs-GPU selector choices, so factor structure and iteration count need not match either pure arm; stored-nnz ratios versus full GPU stayed within 0.06% at every thread count. `APXCHOL_GPU_BLOCKS=<positive>` is A/B-only. `APXCHOL_GPU_BLOCK_TRACE=1` prints per-round timings. `GpuBlockFrontend.*` compares the result with an independent CPU specification and checks determinism, independence and maximality. Exact-clique mode, unsupported cooperative launch and insufficient free VRAM fall back under AUTO and remain errors under `force`. The earlier `n >= 500000` governor was rejected: initial size does not predict the result (grid_2000 loses while smaller iter0040 wins).
  - **Dataflow backend (the DEFAULT GPU SpTRSV — AUTO — since 2026-08-18, and since 2026-08-20 the only one the library implements; `APXCHOL_GPU_SPTRSV=dataflow` names it explicitly, `=cusparse` overrides; `include/apxchol/solver/sptrsv/cuda_dataflow.h`, the file header is the design note):** a sync-free triangular solve with O(n) STATE — ONE persistent kernel launch per sweep (vs one launch per level on the removed level-set backend, 70–100 per sweep, and cuSPARSE's O(nnz) analysis buffers), on CSR of L / CSR of L^T, no level schedules at all. Every row owns an 8-byte epoch-tagged word `{value, epoch}`; warps claim batches of consecutive rows in natural (forward) / reverse (back) row order — a topological order of the factor, coalesced, and for an elimination factor roughly level order — with one atomic ticket; the host packs rows into 32-lane batches with lane groups G = 1, 2, …, 32 by row length (`cuda_host::dataflow_batches`, `dataflow_lane_group`, per-lane prefetch depth `dataflow_prefetch_depth()` = 8), so every row's structure is prefetched into registers before the wait; lanes poll the tagged words of their missing neighbours (pending-mask, only laggards re-read; clock spin 100→800 ns between polls — `__nanosleep`'s granularity is ~1 µs on this GPU), a row publishes value + flag in one 64-bit store: no fence, counter or atomic on the data path. Deadlock-free by construction (a lane only waits on smaller tickets, all held by running warps; the grid is the exact resident block count, `dataflow_grid_size()`, env `APXCHOL_GPU_DATAFLOW_BLOCKS=<n>` overrides for A/B). Bit-deterministic run to run and across grid sizes (cuSPARSE is not). fp32 values (the tagged word packs a 4-byte value next to a 4-byte epoch), fp16 storage supported. **ROW SEGMENTATION (2026-08-20, the fix for the hub-factor gap):** a row is walked by its lane group in chunks of `G*kPre` entries with G capped at 32, so an unsplit row is `ceil(len / 256)` load→poll→accumulate round trips deep ON THE CRITICAL PATH — chunk k+1's loads cannot issue before chunk k has been polled to completion — and a hub factor's elimination tail is a CHAIN of such rows (`APXCHOL_GPU_SPTRSV_STATS=1`: com-Orkut's forward CSR holds **84% of its 544M-nnz factor in the 10% of rows at least 256 entries long**, longest 134368; com-LiveJournal 63% of 122M, kron_g500-logn16 90% of 4.0M; com-Amazon only 11% of 2.0M, iter0040 0.01%, grid_500 none). The host therefore CUTS such a row into S **segments** of one chunk each plus one **finalizer** (`cuda_host::dataflow_build_plan`): a segment is a full 32-lane item over one CSR slice that accumulates exactly as an unsplit G=32 row would and publishes its partial into an extra epoch-tagged SLOT word `tag[m + slot]` — the partials live in the SAME array as the rows, so a stale slot is rejected by its epoch and no per-sweep memset is needed; the finalizer polls the S slots IN FIXED INDEX ORDER (no CSR, no value loads, no atomicAdd anywhere), divides by the diagonal and publishes the row. The S+1 items are ALONE in their batches (the segments' zero-width in sweep positions, the finalizer's owning the split row's), so `batch_start` stays monotone, the common path is byte-for-byte unchanged behind ONE broadcast `batch_spec[b]` load per BATCH, ticket order stays sweep order with a row's segments before its finalizer (every wait is still on a strictly smaller ticket — the deadlock argument carries over verbatim), and the kernel's IMPLICIT "a multi-chunk item never shares a warp with a possible producer" property survives (asserted host-side in `cuda_host::dataflow_plan_check`, which setup runs on both plans and throws on). Determinism is untouched: the split is a pure function of row lengths, so the output is still bit-identical run to run AND across grid sizes (it does differ from the unsplit kernel's, by association — a forced S=1 split is bit-identical, which is the protocol-isolation test). Device state grows by `4 B/batch` + `16 B/spec item` + `8 B/slot`: com-Orkut, the biggest, pays 59 MB of tables and 1.95M slots against a 544M-nnz factor. **THE THRESHOLD IS DERIVED, NOT A CONSTANT** (`cuda_host::dataflow_split_threshold`, per direction, from that direction's histogram): where rows ≥ one chunk hold ≥ 25% of the nnz the aggressive threshold C = 256 wins, elsewhere only rows past the depth break-even 3C = 768 are cut — the same threshold that is **2× on kron_g500 is +31% on com-Amazon**, whose long rows are 11% of the work and whose extra batches are pure overhead. Measured (RTX 4090 Laptop, T=16, bg+tree[vec_pool], default drop + fp16 storage, `gpu_pcg_loop` **ms per PCG iteration**, interleaved medians, against the PRE-CHANGE binary in the same session): **kron_g500-logn16 17.28 → 8.97 (0.52×; total solve 251 → 129 ms), com-LiveJournal 153.1 → 50.6 (0.33×), com-Orkut 726 → 136 (0.19×)**; neutral where nothing splits — iter0040 0.947 → 0.925, grid_500 0.393 → 0.394, com-Amazon 2.114 → 2.117 (the null control, pre-change vs new-with-segmentation-off, spans 0.95–1.02× on those three, i.e. the differences are noise). Registers went DOWN, not up: `dataflow_special` is `__noinline__` so its live state does not merge with the common path's ~112-register cliff — `cuobjdump --dump-resource-usage` sm_89 `dataflow_kernel<false>` 98 → 97, `<true>` 104 → 102, STACK 0 (no ABI spill), so the occupancy-derived grid is unchanged. Measured DEAD END: LONGER SEGMENTS. With the threshold pinned at 256, segments of 256 / 512 / 1024 entries cost kron 8.87 / 10.34 / 11.62 and com-LiveJournal 50.5 / 60.6 / 113.0 ms/iteration — the depth model's `sqrt(len)` optimum is swamped by the extra round trip each additional chunk of a segment puts on the critical path, so the segment length is fixed at one chunk and is not a knob. Measured (RTX 4090 Laptop, T=16, bg+tree[vec_pool], default drop, `cudaEvent` ms per forward+back pair on the device-resident path): grid_500 cuSPARSE 0.24 / level-set 0.52 / dataflow 0.26; iter0040 0.44–0.53 / 0.87–1.02 / 0.34–0.42; grid_2000 1.80 / 4.2–4.7 / 1.33–1.46 — i.e. 1.2–1.3× faster than cuSPARSE on the two big ones, 2.5–3× faster than the level-set, at 8 B/row of mutable state; end to end (`gpu_pcg_loop` ms/iter, medians of 5 interleaved reps) iter0040 cuSPARSE 1.08 → dataflow 0.98, grid_2000 3.54 → 3.20, at equal iteration counts, with O(n) memory (device factor bytes iter0040: cuSPARSE 39 MB + 129 MB SpSV buffers vs dataflow 82.8 MB; grid_2000 191 + 766 vs 415) and bit-deterministic results — which is why it became the AUTO choice (see the commit for the tuning history: per-row in-degree counters + dependents lists were tried first and lost to the tagged word). Re-measured on the AUTO commit (3 interleaved reps, same protocol, `gpu_pcg_loop` ms/iter medians, AUTO=dataflow vs `=cusparse`): grid_500 0.52 vs 0.51, grid_2000 3.23 vs 3.58, iter0040 1.00 vs 1.02, com-Amazon 2.68 vs 2.66, coAuthorsDBLP 2.88 vs 2.91 — i.e. parity on the small grid and on the skewed-degree social pair (long thin level tails: both backends are latency-bound there), the win on the two big ones, never a loss; fp16 storage on the dataflow kernel (`APXCHOL_GPU_SPTRSV_FP16=1`, AUTO): REWORKED 2026-08-19 after a 1.8–2x back-sweep regression on hub-heavy factors (kron_g500-logn16, com-LiveJournal — a hub back sweep is a SEQUENTIAL publish chain of multi-chunk rows, so per-entry fp16 taxes are paid once per chain link and the widen-at-load cvts had serialized the 8-deep load pipeline past a ~112-register scheduling cliff, see the REGISTER-BUDGET WARNING in `src/cuda_dataflow.cu`). Final fp16 path: raw 2-byte loads, diagonal-slot bits zeroed at load (that slot can be fp16 Inf — `diag_bad` — and the accumulate is branchless; also fixes a NaN), one packed `__half22float2` widen per entry pair inside a branchless two-chain ILP accumulate, in-kernel double `in_scale` at leader prefetch (latency-hidden even on chains; a separate prescale launch was tried and costs +9–115 µs/iteration). Interleaved warm per-sweep vs fp32 dataflow: iter0040 pair 0.92x, grid_2000 0.97x, kron fwd 0.94x / bck 1.06x, LiveJournal fwd 0.99x / bck 1.03–1.08x — i.e. faster-or-parity everywhere except a small hub-back-sweep residual, sub-noise at total-solve level; vs the pre-rework fp16: kron bck 24 → 13 ms, LJ solve 9.6 → 5.4 s, iteration counts unchanged. Measured DEAD ENDS (documented at the kernel; do not re-attempt without new structure): fp16 HFMA2 compute (−4 ms kron bck but relres 1e-3..4e-1, no convergence), mixed-width fp32-for-hub-rows storage (116 regs → past the cliff, bck 2x slower, AND wide rows hold 70–91% of hub-factor nnz so memory exceeds pure fp32), pair 32-bit loads + `__maxnreg__(98)` (won only against the old branchy accumulate; a pure ALU tax against the lean one, iter0040 sweeps +13–17%). **Setup caveat:** `sptrsv_setup` of both kernel backends pays the HOST transpose to CSR of L (`cuda_host::transpose_csr`) that cuSPARSE does not. Since 2026-08-18 that transpose is THE shared blocked counting-sort parallel transpose of `sptrsv/transpose.h` (the very code `omp_sptrsv::setup` runs for its CSC→CSR; one implementation, byte-identical to the serial scatter at any thread count, `SpTRSVTranspose.GpuHostTranspose*` guard both callers against one serial reference) instead of a serial O(nnz) scatter: measured `sptrsv_setup` (T=16, dataflow AUTO, medians of 3, first setup of the process — which also pays the ~100 ms CUDA context creation inside it, before and after alike) grid_2000 331 → 248 ms (transpose alone 155 → 77 ms per `APXCHOL_SPTRSV_SETUP_TRACE=1`, the CPU backend's per-stage knob, now honoured by the GPU setup too, `[sptrsv-setup gpu]` lines), iter0040 148 → 131 ms, grid_500 93 → 96 (noise; a few ms of transpose there); with a warm context (`--repeat 2`, second setup) grid_2000 155 ms / iter0040 55 ms. At one thread both backends fall back to the serial scatter (`use_parallel_transpose()`: enabled, m > 50000, `omp_get_max_threads() > 1` — the blocked sort's third pass loses at T=1). Tests: `GpuHostPrep.DataflowBatchesPackRowsIntoAlignedLaneGroups` (every build), `GpuDataflow.*` (CUDA build: each sweep against a SERIAL DOUBLE-precision host reference over the same arrays, and the `L L^T` pair through `cuda_sptrsv` against the CPU `omp_sptrsv` pair — tolerance-level, the references accumulate in fp64 — plus bit-identical across launches and grids 1 / 7 / resident, fp32 + fp16. Before 2026-08-20 the reference was the level-set kernel.)
  - **No closed-source dependencies (2026-08-18; branch `gpu/no-closed-deps`):** the CUDA library links **`CUDA::cudart` only**. The GPU-resident PCG (`pcg_cuda.h`) runs OUR OWN kernels (`include/apxchol/solver/pcg_cuda_kernels.h`, `src/cuda_pcg_kernels.cu`, namespace `apxchol::pcg_cuda`) instead of cuSPARSE SpMV + cuBLAS dot/nrm2/axpy/scal: a CSR SpMV with p·Ap folded into the row loop (`spmv_pAp`; LANES ∈ {1,…,32} consecutive threads per row — the CUSP vector-CSR family, `spmv_lanes_for(avg nnz/row)` = the power of two nearest to the average in log2, so grids (~5/row) get 4 and the IPM factors (~15/row) 16; env `APXCHOL_GPU_SPMV_LANES=1|2|4|8|16|32` overrides for A/B, `cuda_pcg::spmv_lanes()` reports; fp64 operator values or the fp32-exact operator's fp32 values promoted per product, fp64 accumulate — the fp32-exact rule mirrors the CPU's `op_fp32_`, `APXCHOL_GPU_FP32_OPERATOR` overrides as before), the fused vector passes of the CPU loop (`update_xr` = x += αp, r −= αAp, r·r in one pass; `dot` for r·z; `update_p` = z + βp) and DETERMINISTIC reductions: a fixed grid (`pcg_blocks(units)` = min(ceil(units/256), 2048), a pure function of n), each thread a fixed strided subset in a fixed order, a fixed-order warp-shuffle + shared-memory tree per block, one partial per block, a fixed-order single-block final reduce (`reduce_partials`) → 8 bytes back into pinned host memory — no floating-point atomics anywhere, three host syncs per iteration (p·Ap, r·r, r·z), so together with the dataflow / level-set SpTRSV the whole GPU solve is bit-identical run to run (`SolveTest.DeterministicWithSameSeed` on the CUDA build, `--gtest_repeat=25` clean). The old `spmv_f32A_f64` warp-per-row kernel (fp32-operator path) is gone. Measured vs the cuBLAS/cuSPARSE loop (RTX 4090 Laptop, T=16, bg+tree[vec_pool], `gpu_pcg_loop` ms/iter, medians of 3): grid_500 0.71 → 0.38, grid_2000 5.98 → 2.91 (the old warp-per-row fp32-operator kernel was the bottleneck there: 32 lanes on 5-nnz rows), iter0040 0.92 → 0.87; with the fp64 operator forced on both (`APXCHOL_GPU_FP32_OPERATOR=0`, i.e. cuSPARSE SpMV vs ours, T=1) grid_2000 3.21 → 2.95, grid_500 0.49 → 0.35 — our SpMV is never slower than cuSPARSE's. Iteration counts / RelRes at T=1 (deterministic factor) vs the cuBLAS loop: grid_500 40 / 9.911e-09 → 40 / 9.911e-09, iter0040 46 / 7.389e-09 → 46 / 7.389e-09, grid_2000 47 / 9.734e-09 → 47 / 9.716e-09 (grid_2000 sits ON the tolerance at iteration 47 — the residual there is 9.7e-9…1.06e-8 across the six lane mappings of the SAME kernel, so it flips 47/48 with any change of summation order; the shipped 4-lane rule lands on 47). **cuSPARSE is an OPT-IN**: `-DAPXCHOL_CUDA_WITH_CUSPARSE=ON` (root and `benchmarks/` CMake; default OFF) compiles + links the cuSPARSE SpSV backend of `cuda_sptrsv` (env `APXCHOL_GPU_SPTRSV=cusparse`, the fp64 build's OOM-aware AUTO, `cuda_sptrsv::cusparse_available()`, kept as a comparison baseline; the code is `#if defined(APXCHOL_CUDA_WITH_CUSPARSE)` in `cuda.h`, `cusparse_buffer_bytes()` stays and reads 0 without it) and, in the benchmark, the legacy `apxchol_gpu` bench solver (`run_apxchol_gpu_pcg`: a cuBLAS+cuSPARSE PCG around the host-pointer preconditioner API, not used by any sweep — the last cuBLAS consumer, so cuBLAS is linked to the DRIVER only under that option); without it `APXCHOL_GPU_SPTRSV=cusparse` is a stderr note + the AUTO backend. cuBLAS is gone from the library entirely. **The benchmark DRIVER still links cuSPARSE whenever Boost is found**: AMGCL's CUDA backend (`benchmarks/src/amgcl_cuda.cu`, competitor) is cuSPARSE-based — `ldd benchmarks/build-cuda/benchmark` shows `libcusparse` (from AMGCL) but no `libcublas` on the default build; `ldd build-cuda/apxchol` / `unit_tests` show `libcudart` only, and `libapxchol_v1.a` has no undefined cuSPARSE / cuBLAS symbols.
  - **CUDA device init: prewarmed, and accounted under its OWN label `cuda_init` (2026-08-20; `include/apxchol/solver/cuda_context.h`).** Creating the per-process CUDA primary context is a fixed PLATFORM cost that the runtime pays LAZILY inside whichever CUDA call comes first — which was the GPU SpTRSV's first allocation, i.e. in the middle of the timed setup, and it was charged to `sptrsv_setup` (and, in the trace, to `build_L11`). It is constant in n, in T and in the factor size: **~80–135 ms on the RTX 4090 Laptop, ~715 ms on a GH200** (Grace: 55% of the whole reported setup; `/home/adamant/.cache/apxchol_prof/daint_gpu_setup_breakdown.md`). GPU persistence mode is NOT a substitute (laptop 80–97 → 76–86 ms measured). Two independent changes:
    1. **Library.** `apx_cholesky::factorize` calls `cuda_ctx::prewarm()`, which starts ONE helper thread per process (mutex-guarded, started at most once, joined by `ensure_context()` or by the state's destructor — so a process that never touches the GPU still exits cleanly) doing `cudaFree(0)`, so the context is created WHILE make_graph / find_partition / eliminate / assembly run. `apx_cholesky::install_factor` then calls `cuda_ctx::ensure_context()` immediately BEFORE `trsv_.setup` and records the REMAINING wait under its own checkpoint label **`cuda_init`** (plus a `[sptrsv-setup gpu] cuda_init … (context creation … ms)` line under `APXCHOL_SPTRSV_SETUP_TRACE`). Correctness never depends on the prewarm: the primary context is reference-counted and its creation is serialized inside the driver, so the helper racing the main thread is well defined — the prewarm only decides WHEN the cost is paid, `ensure_context()` alone makes it VISIBLE and keeps it out of `sptrsv_setup`. Without `APXCHOL_USE_CUDA` the whole header is empty inline functions, so call sites carry no `#if`.
    2. **Benchmark driver.** `benchmarks/src/benchmark.cpp` does one `cudaFree(0)` before ANY in-process solver is timed and prints `[bench] cuda_init (once, before any timed solver): … ms`. This is a **fairness** measure, not an optimization: apxchol, AMGCL-CUDA and Hypre-CUDA share that boundary; the standalone ParAC CUDA drivers perform and print the same pre-timed initialization through benchmark patch 0004. `sweep_fair.py` / `parac_runner.py` persist the separately reported interval as `cuda_init_s`, never inside `setup_s` or `total_s`. Hiding only ours would tilt the tables.
    **Consequence: published GPU setup numbers change meaning.** Every historical GPU setup cell is inflated by one context creation — including `--repeat 2` ones, because `median_run` with 2 repeats returns the SLOWER of the two runs, which is the cold one 80% of the time (measured, 30 pairs). **The benchmark cells need regenerating.** Measured on the laptop (RTX 4090 Laptop, T=16, `bg+tree[vec_pool]`, 5-arm interleaved A/B with an A-vs-A null control, 12 reps/arm/cell; context creation there is 78–353 ms, median 87): the cold-minus-warm gap of the FIRST setup in a process collapses — `sptrsv_setup` **grid_500 +90 → +3 ms, grid_2000 +94 → +16 ms, iter0040 +99 → +12 ms**, and with the harness warmup as well **grid_500 setup 207 → 69 ms (−67%)**, grid_2000 1310 → 1138, iter0040 844 → 777. Warm setups are unchanged; iterations and residual are bit-identical (T=1: grid_2000 50 / 8.383146e-09, iter0040 46 / 7.393451e-09 on both). Caveat, and it is in the header: the overlap is worth at most min(host setup, context) because context creation is CPU work — on grid_500, whose host setup is shorter than the context, the library prewarm alone is a wash (154 → 156 ms) and only the harness warmup helps. **Refuted in the same pass:** parallelizing `cuda_host::csr_row_lengths`, the only part of `setup_kernel_backend` that is not `dataflow_batches`, costs `kernel_backend_tables` **12.3 → 20.3 ms** on grid_2000 (its consumer reads `len` back serially); see the comment on that function. `dataflow_batches` is ~85% of the stage and is the only lever left there.
- `-DAPXCHOL_64BIT_EDGE_INDICES=ON` — 64-bit `edge_index` (cumulative offsets/edge ids) for factors or residual pools exceeding the default unsigned-32-bit capacity (2³²−1 entries). com-Orkut's roughly 2.15B-entry raw factor fits the default build; this was the purpose of moving offsets from signed to unsigned 32-bit. `-DAPXCHOL_64BIT_NODE_INDICES=ON` additionally widens vertex ids (implies 64-bit edges). The old `APXCHOL_64BIT_INDICES` is a deprecated alias for the EDGE knob. The standalone `benchmarks/` project exposes the same two options. See `include/apxchol/types.h`.
- `APXCHOL_POOL_FP32` — **ON by default** (fp32 residual-pool edge weights); pass `=OFF` for an fp64 baseline. The SpTRSV's own factor values are fp32 **unconditionally** — the `APXCHOL_SPTRSV_FP32` option was removed 2026-08-20 together with the fp64 storage path it gated (`factor_value_t` = `sptrsv_value_t` = `float` in `sparse_csc.h`, `cuda_value_t` = `float` on the GPU). Nothing shipped or was measured against the fp64 storage, and the GPU's dataflow backend never supported it (its tagged word packs a 4-byte value next to a 4-byte epoch), so the only thing the option bought was a second, untested compile of every SpTRSV path.
- **fp16 SpTRSV factor storage is a RUNTIME switch, not a build option** — `APXCHOL_SPTRSV_FP16=0|1`, read by BOTH backends at every setup (`apxchol::sptrsv_fp16_env_tristate()` in `include/apxchol/lowprec.h` is the one reader; `omp_sptrsv::fp16_from_env()` / `cuda_sptrsv::fp16_resolved()` apply the per-device default). **Unset resolves per device: ON on the GPU, OFF on the CPU** — the two have opposite measured verdicts (GPU: −12% `gpu_pcg_loop` on iter0040, −4..8% grids, −15..20% device factor bytes, iteration-neutral on all 9 suite matrices; CPU: a memory lever whose iteration cost has to be paid back, so opt-in). The CPU kernel is INSTANTIATED for both storage types (`omp_sptrsv::setup_impl<V>` / `solve_levelset<Dir, V>`, V = `float` | `fp16_t`) and `setup` branches ONCE on the env — never per row — so one binary does either. `APXCHOL_GPU_SPTRSV_FP16` (the GPU-only name until 2026-08-20) is still read as a DEPRECATED alias with a one-shot stderr note; since it aliases the unified variable it now governs the CPU too. **Distribution guard:** the fp16 storage is only compiled and only offered where the target has F16C (`omp_sptrsv::fp16_supported()`, `__F16C__`); without it the fp16→fp32 widen becomes a libgcc `__extendhfdf2` call in the inner loop (measured 3× slower solve), so a portable baseline-x86-64 build — the PyPI wheels — compiles fp32 ONLY and the env falls back with a note. The storage itself is unchanged: IEEE binary16 of `L_ij / s_j`, per-column `s_j` = max |off-diagonal| (`omp_sptrsv::column_scale`), fp32 diagonal in a separate `diag_[]` (`omp_sptrsv::stored_diag` + the column's rounding residual), fp16 subnormals flushed to signed zero, the scale FOLDED INTO THE VECTORS (`forward_solve` returns `y' = D y`, `transpose_solve` takes `y'` and scales its input by `inv_scale_[j]^2` — `preconditioner.h` only ever calls the pair), fat-level (`omp for`) kernels SIMD on AVX2+F16C+FMA (`omp_sptrsv::simd_fp16_kernel()`: 8 values per `_mm256_cvtph_ps`, stack buffer, 4-way scalar FMA chain), arithmetic always fp64. **Degenerate scales** (both backends, same contract): a column whose fp32 `1/s_j` or `fp32(L_jj)/s_j` would overflow falls back to `s_j = 1` before the drop (`lowprec_stats().scale_fallback`); setup then verifies every diagonal is finite and nonzero and every `r_j^2` finite, and THROWS otherwise — a factor the storage cannot represent fails loudly instead of dissolving into NaN. Any new consumer of stored factor values must `widen()` them (`fp16_t`'s float/double conversions are explicit on purpose). **Removed 2026-08-18:** the `BF16` / `BF16_SCALED` (bfloat16, 7-bit mantissa; RNE or stochastic rounding) and `FP24` storage variants and their types/tests — bf16 cost 3–6× the PCG iterations on IPM, fp24 was marginal. **Removed 2026-08-20:** the `APXCHOL_SPTRSV_LOWPREC` CMake cache variable itself (root + `benchmarks/`), which had made the CPU storage a compile-time choice and forced CUDA builds to treat it as OFF.
- Runtime env knobs of the OpenMP SpTRSV / CPU PCG (both storages; read per setup). After the 2026-08-20 surface reduction this list is ONE entry — everything else that used to sit here is in "Retired knobs" below:
  - `APXCHOL_FACTOR_DROP=<rel>` — the COMPACTING drop with column-sum compensation (the compensation is unconditional; the threshold is the only knob), **ON by default at 1e-4**: see "Compacting factor drop" below (that section is authoritative). What the storage format adds to it: the pure predicate `omp_sptrsv::keep_offdiag(v, s_j, rel, fp16_flush_subnormal)` also drops what the storage format would store as zero anyway (`omp_sptrsv::format_flushes`: exact zeros on fp32/fp64; on FP16_SCALED also everything fp16 flushes, subnormals included by default — at the default rel = 1e-4 > 2^-14 this adds nothing), the drop and the compensation run on the fp32 factor BEFORE `narrow_value` (dropped entries never reach the storage format; `s_j` is the pre-drop column max), and `omp_sptrsv::lowprec_stats()` (== `drop_stats()`, one record) splits `dropped` into `dropped_threshold` + `dropped_flush` next to `rel` / `compensate` / `nnz_factor` / `nnz_stored`. Supersedes the removed `APXCHOL_LOWPREC_DROP` diagnostic (same threshold and numerics — a stored zero and an absent entry solve identically — but that one only zeroed in place).
- `APXCHOL_BUILD_EXAMPLES` / `APXCHOL_BUILD_TESTS` — both ON by default; `APXCHOL_BUILD_TESTS` gates the GoogleTest fetch. `APXCHOL_BUILD_TOOLS` — OFF by default; builds the `bench_setup` / `analyze_factor` dev tools. `analyze_factor MATRIX --solve [--seed N]` is the lightweight single-RHS diagnostic: CUDA builds prewarm the process context before the timed setup, then print one parseable `solve_result` line with setup, PCG, total, iterations, residual and solver VRAM. Adjacency inputs are exact graph Laplacians, so the diagnostic projects its random RHS separately in every connected component; a merely global projection is incompatible on disconnected graphs such as as-Skitter.

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
