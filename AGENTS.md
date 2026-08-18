# AGENTS.md

Guidance for coding agents working in this repository. **Keep this file
accurate**: when a change alters
build steps, options, defaults, the public API surface, or the architecture
described below, update this file in the same commit.

## Build & test

The core library is two CMake projects: the root (library + CLI + unit tests) and `benchmarks/` (standalone, FetchContent-pulls external solvers).

```bash
# Root project
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# All tests (gtest_discover_tests registers each TYPED_TEST instance)
ctest --test-dir build --output-on-failure

# Single test or filter
./build/tests/unit_tests --gtest_filter='FactorizeTest/*.PermutationIsValid'

# CLI solver (a RHS is mandatory: --rhs file.mtx or --random-rhs)
./build/apxchol path/to/matrix.mtx --random-rhs --tol 1e-8
```

Build options (root `CMakeLists.txt`):
- `-DAPXCHOL_USE_CUDA=ON` — switch the SpTRSV backend from OpenMP level-sets to cuSPARSE (+ GPU-resident PCG). Defines `APXCHOL_USE_CUDA`, requires CUDAToolkit.
- `-DAPXCHOL_64BIT_EDGE_INDICES=ON` — 64-bit `edge_index` (cumulative offsets/edge ids) for factors with > 2³¹ nnz, e.g. com-Orkut. `-DAPXCHOL_64BIT_NODE_INDICES=ON` additionally widens vertex ids (implies 64-bit edges). The old `APXCHOL_64BIT_INDICES` is a deprecated alias for the EDGE knob. See `include/apxchol/types.h`.
- `APXCHOL_SPTRSV_FP32` / `APXCHOL_POOL_FP32` — both **ON by default** (fp32 factor values in the SpTRSV / fp32 residual-pool weights); pass `=OFF` for an fp64 baseline.
- `APXCHOL_SPTRSV_LOWPREC` — string cache variable, **`OFF` by default**, one of `OFF | BF16 | BF16_SCALED | FP16_SCALED | FP24` (the old boolean `APXCHOL_SPTRSV_BF16=ON` is an alias for `BF16`). Selects a low-precision STORAGE format for the SpTRSV's CSR/CSC OFF-diagonal values (`sptrsv_value_t` = `bf16_t` / `fp16_t` / `fp24_t`, `include/apxchol/bf16.h` + `include/apxchol/lowprec.h`), narrowed once in `omp_sptrsv::setup` from the fp32 factor through `omp_sptrsv::narrow_value` (`sparse_csc::vals_` is `factor_value_t` = fp32 under every variant, same as the fp32 build); the DIAGONAL is kept fp32 in a separate `omp_sptrsv::diag_` array (rounding it to 8 bits was the dominant iteration-count damage): exactly `L_jj` under `BF16`/`FP24`, the scaled `fp32(L_jj / s_j)` under the `*_SCALED` variants (`omp_sptrsv::stored_diag` is the contract). The kernels are ONE source for every storage type — per entry (value load, `widen()`, index load, y gather, fma), per row/column the diagonal division (`fwd_diag()`/`bck_diag()`) — the arithmetic stays fp64. Variants: `BF16` = bfloat16, RNE (env `APXCHOL_BF16_STOCHASTIC=1`, read at every setup, switches to unbiased stochastic rounding with a deterministic per-entry CSC-index-hashed threshold; run-to-run deterministic, CSR/CSC-consistent); `BF16_SCALED` = bf16 of `L_ij / s_j` with a per-column scale `s_j` = max |off-diagonal| of column j (fp32 `omp_sptrsv::scale_`, `omp_sptrsv::column_scale`); `FP16_SCALED` = IEEE binary16 of `L_ij / s_j` (2^-11 relative; entries below 2^-25 of their column max flush to zero, and entries below 2^-14 — fp16 SUBNORMALS, 1–10 significant bits — are ALSO flushed to (signed) zero at storage time by default, env `APXCHOL_FP16_KEEP_SUBNORMAL=1` (read at every setup) restores IEEE subnormal storage; `setup` counts flushed / subnormal, `omp_sptrsv::lowprec_stats()`, printed under `APXCHOL_VERBOSE`); `FP24` = top 24 bits of the fp32 pattern (2^-16 relative, 3 B/entry, no scaling). **`*_SCALED` pair contract (the scale is FOLDED INTO THE VECTORS, `omp.h` file header):** what is stored is the column-scaled factor `L~ = L D^-1`, `D = diag(s_j)`, and the kernels never multiply a scale back: `forward_solve` runs on `L~` and returns `y' = D y` (NOT `y`), `transpose_solve` takes `y'` and solves `L~^T z = D^-2 y'` (its input read is scaled once per column by `inv_scale_[j]^2`, `inv_scale_[j] = fp32(1/s_j)`, `omp_sptrsv::inv_scale`), so the pair applies `(L L^T)^-1` (up to the 2^-23 of `inv_scale_`) — `preconditioner.h` only ever calls the pair; on the non-scaled builds `D = I` and both calls mean what they always did. Fat-level (`omp for`) kernels of the 16-bit variants (`BF16`, `BF16_SCALED`, `FP16_SCALED`) are SIMD on AVX2+F16C+FMA targets (`omp_sptrsv::simd_dot()`): 8 values per vector widen (`_mm256_cvtph_ps` / bf16 shift), two 4-double FMA lanes, y gather either an 8-double stack buffer + scalar gathers (`APXCHOL_FP16_GATHER=scalar`, the default — measured faster on grids) or `_mm256_i32gather_pd` (`=simd`); env read at every setup on every build; the fp32/fp64/FP24 fat-level loop is unchanged (instruction-identical, `kSimdDot`). Takes precedence over `APXCHOL_SPTRSV_FP32`; CPU/omp backend only — with `APXCHOL_USE_CUDA=ON` any value is TREATED AS OFF for that build (fp32 CPU storage, no `APXCHOL_SPTRSV_LOWPREC_*` macro, a `STATUS` line at configure; the cache value is left as requested), never a configure error, so a non-OFF default cannot break a CUDA build. Any new consumer of stored factor values must `widen()` them (the storage types' float/double conversions are explicit on purpose). Same variable exists in `benchmarks/CMakeLists.txt` for `apxchol_v1`.
- Runtime env knobs of the OpenMP SpTRSV / CPU PCG (every build incl. the default fp32 one and all LOWPREC variants; all read per setup / per solve; all default OFF except the factor drop):
  - `APXCHOL_FACTOR_DROP=<rel>` / `APXCHOL_FACTOR_DROP_COMPENSATE=0` — the COMPACTING drop with column-sum compensation, **ON by default at 1e-4**: see "Compacting factor drop" below (that section is authoritative). What the storage variants add to it: the pure predicate `omp_sptrsv::keep_offdiag(v, s_j, rel, fp16_flush_subnormal)` also drops what the storage format would store as zero anyway (`omp_sptrsv::format_flushes`: exact zeros on fp32/fp64/bf16/fp24; on FP16_SCALED also everything fp16 flushes, subnormals included by default — at the default rel = 1e-4 > 2^-14 this adds nothing), the drop and the compensation run on the fp32 factor BEFORE `narrow_value` (dropped entries never reach the storage format; `s_j` is the pre-drop column max), and `omp_sptrsv::lowprec_stats()` (== `drop_stats()`, one record) splits `dropped` into `dropped_threshold` + `dropped_flush` next to `rel` / `compensate` / `nnz_factor` / `nnz_stored`. Supersedes the removed `APXCHOL_LOWPREC_DROP` diagnostic (same threshold and numerics — a stored zero and an absent entry solve identically — but that one only zeroed in place).
  - `APXCHOL_FTZ=1` — `cpu_solver::solve_impl` sets the x86 MXCSR FTZ+DAZ bits on the master and on every OpenMP team thread at PCG entry (per-thread state; sticky). The subnormal census (`lowprec_stats().factor_subnormal` = fp32 factor entries that are fp32 subnormals, diagonal included; on the fp32 build these are the stored values) is printed under `APXCHOL_VERBOSE` on the storage line.
  - `APXCHOL_FP16_GATHER=scalar|simd` — fat-level (`omp for`) kernel flavour of the 16-bit LOWPREC variants (`omp_sptrsv::simd_dot()`; no effect on the fp32/fp64/FP24 builds): `scalar` (default) = vector widen into an 8-double stack buffer + scalar gathers + 4-way scalar FMA chain, `simd` = `_mm256_i32gather_pd` y-gather + vector FMA (loses ~10-15% on grid_2000 T=1, unlocked A/B). Read at every setup; `omp_sptrsv::fat_gather_simd()` reports the resolved value.
  - `APXCHOL_FP16_DIAG=1` — FP16_SCALED only: the kernels divide by the fp16 diagonal slot `fp16(L_jj / s_j)` already stored in the CSR/CSC (11 significant bits) instead of the fp32 `diag_[]` = `fp32(L_jj / s_j)`. Sound because `L_jj >= s_j` for apxchol's factors (setup counts violations, `lowprec_stats().diag_below_scale`), so the slot is a normal fp16 unless `L_jj / s_j >= 65520` overflows to inf; setup counts non-normal slots (`lowprec_stats().diag_fp16_bad`) and REFUSES the mode (stderr warning, fp32 diagonal everywhere) if any exist. `omp_sptrsv::fp16_diag()` reports it; printed under `APXCHOL_VERBOSE` (`fp16 diag:` line). Purpose: the T=1 iteration-count test of an 11-bit diagonal (evidence for dropping `diag_`, 4 B/row) — measured (tol 1e-8, with/without the drop): grid_500 40→40, grid_2000 47→47, iter0040 44→50 (+14%), so NOT neutral on IPM; `diag_` stays fp32, the env is a diagnostic.
- `APXCHOL_BUILD_EXAMPLES` / `APXCHOL_BUILD_TESTS` — both ON by default; `APXCHOL_BUILD_TESTS` gates the GoogleTest fetch. `APXCHOL_BUILD_TOOLS` — OFF by default; builds the `bench_setup` / `analyze_factor` dev tools.

The project targets **C++23** (`CMAKE_CXX_STANDARD 23`) and adds `-march=native` in Release/RelWithDebInfo.

The Python package under `python/` is its own scikit-build-core project (`pip install -e python`); it compiles the two core TUs directly and never touches the root build.

Benchmarks live in a separate CMake project; see `benchmarks/README.md` for the runner pipeline. The single fair sweep is `benchmarks/sweep_fair.py` (grids + SuiteSparse + IPM; `--device cpu|gpu` covers both axes and runs ParAC in-process via `parac_runner.py`; `--parac-only`/`--no-parac` scope it). Shared harness (hardened `sh`, matrix registry, cell store, VRAM sidecar) is `runner_common.py`. `fair_charts.py` + `combined_charts.py` + `gpu_charts.py` render the committed `benchmarks/latest/` charts. Build the driver with `cmake -B benchmarks/build -S benchmarks -DCMAKE_BUILD_TYPE=Release && cmake --build benchmarks/build -j$(nproc) benchmark` (CUDA axis: `-B benchmarks/build-cuda` with the CUDA flags).

## Architecture

### Library layout

The public surface is the header tree under `include/apxchol/` plus two compiled TUs (`src/factorization.cpp`, `src/solve.cpp`) that form `apxchol_core`. The convenience header is `include/apxchol.h`. Subdirectories:

- `apxchol/graph/` — graph data structure templated on an `incidence_storage` concept. Implementations: `vec` (std::vector), `forward_star` (linked-list pool), `bstr` (bit-string), `vec_pool` (slab pool — the most robust backend in the suite, the only one that does not collapse on high-degree IPM matrices, and the `apxchol::solve` default). `graph_storage` enum (in `types.h`) picks one at runtime via `make_graph<...>`.
- `apxchol/solver/` — factorization and PCG glue.
  - `elimination/` — the tree-based clique-sampling kernel plus the public `eliminator` seam (`weighted_neighbor`, `deferred_edge`, `edge_emitter`, `random_stream`, `as_eliminator`; per-elimination seeds — no library-owned RNG stream); the `clique`/`iid` variants no longer exist.
  - `partition/` — independent-set partitioners (`block_greedy` default, plus `luby`, `baumann_kyng`, `rootset`); the list lives in `partitioner_list.h` (`hybrid` and the separator partitioners no longer exist).
  - `sptrsv/` — triangular-solve backend (`omp.h` by default, `cuda.h` when `APXCHOL_USE_CUDA`).

### Dispatch and customization seams

`apxchol::factorize` is templated on the IS selector (partitioner), the incidence storage, and — via overloads taking an instance — the eliminator. Custom eliminators/partitioners are passed as instances (matrix- or graph-level overloads; see `docs/extending.md` and `examples/`); partitioners implement one uniform `find_partition(G, span<const node_index> candidates, ctx, selection&)` shape — the `degree_prepass` trait decides whether the orchestrator runs the prune/degree/cap pre-pass (candidates = eligible list, vertex-indexed `ctx.degrees` filled) or passes the raw active list (`ctx.degrees` empty); selection knobs are grouped as `factor_options::partition`. `graph<>` and the matrix overloads default to `vec_pool`. The call most users hit is the runtime-dispatch overload:

```cpp
factorization factorize(const Eigen::SparseMatrix<double>& L,
                        graph_storage storage,
                        const factor_options& opts, ...);
```

This dispatches storage from the `graph_storage` enum and the partitioner by name from `factor_options::is_select` (via `dispatch_partitioner` over `partitioner_list` in `partitioner_list.h`). When adding a new partitioner, add the type and append it to `partitioner_list`; when adding a storage backend, add the type, extend the `graph_storage` enum, AND extend the runtime dispatch switch in `src/factorization.cpp` — otherwise the template path works but the CLI/preconditioner path silently can't reach it.

### Eigen integration

`apx_cholesky` (`include/apxchol/solver/preconditioner.h`) inherits `Eigen::SparseSolverBase` and is meant to be plugged into `Eigen::ConjugateGradient<SpMat, UpLo, apxchol::apx_cholesky>`. `analyzePattern` is a no-op; `factorize` calls into `apxchol::factorize` and then sets up the SpTRSV backend. Both `_solve_impl` (the Eigen entry) and `apply_fused` (the PCG-loop entry used by `cpu_solver`: takes Σr, returns r·z) share one body, `apply_core`.

### PCG loop: fused, deterministic vector kernels

`cpu_solver::solve_impl` (`src/solve.cpp`) is not a chain of Eigen expressions: each iteration is four fused, OpenMP-parallel passes — `spmv+pAp` (p·Ap folded into the SpMV row loop), `update_xr_norm` (x += αp, r −= αAp, r·r, Σr in one pass), the preconditioner application (`pcg.solve.{permute,forward,back,unpermute+rz}`: input centering fused into the permute scatter, output re-centering + r·z fused into the unpermute gather — both centrings on centring applications only, see below), and `update_p`; after the loop, `center_x` (Laplacian only: x −= mean(x) once). Those are also the checkpoint labels under `pcg.*`. Every reduction uses the scheme in `preconditioner.h` `detail::` (fixed `static_chunk` partition, per-thread partials summed in thread order — never a `reduction()` clause), so results are bit-identical run to run for a fixed thread count (they differ across thread counts at fp-rounding level, like every other parallel pass). `PcgFusion.*` in `tests/test_factorize.cpp` guards this; keep new vector passes on the same scheme.

### Laplacian vs SDDM rank

`factorization::sddm` distinguishes the two cases and changes the preconditioner application path:
- Laplacian (positive semidefinite, rank `n−1`): factor dimension is `n−1`; the application (`apply_core`) subtracts the mean of `b` before solving and re-centers the output afterwards — but only on every K-th application (the center-k schedule, K = 10 by default; see the env knobs below and `take_center_()`); the other applications skip both mean passes. `cpu_solver::solve_impl` restarts the schedule per solve (`reset_apply_count`) and subtracts `mean(x)` once from the returned solution (`center_x`), so the solution is min-norm regardless of K or of a warm start's constant.
- SDDM (positive definite, full rank): full `n×n` factor, no centering.

When touching `apply_core` or anything that constructs a `factorization`, check both branches.

### Experiment env knobs (`include/apxchol/env_knobs.h`)

Process-wide, read once. `APXCHOL_GROUND=center-k|reg` (+ `APXCHOL_CENTER_K`, `APXCHOL_REG_EPS`) selects how a pure Laplacian is grounded: `center-k` (the default, K = `APXCHOL_CENTER_K` = 10) centres only every K-th preconditioner application of a solve; K = 1 centres every application (`APXCHOL_GROUND=center` is accepted as an alias for K = 1, canonical spelling `center-k` + `APXCHOL_CENTER_K=1`); `reg` adds an explicit `eps·diag` self-loop at `make_graph` time so the matrix classifies as SDDM (full-rank factor, no centring). `APXCHOL_OMP_THRESHOLD` overrides `factor_options::omp_threshold` (unset = unchanged); `APXCHOL_TAIL_THREADS` runs sub-threshold ("tail") elimination rounds on the fused parallel path with a small pinned team instead of the serial path (unset = serial tail). Unit tests assume the knobs are unset, i.e. the center-k default (`reg` deliberately flips the SDDM flag; a parallel tail/partitioner is only reproducible up to fp merge-order ulps and the racy block_greedy conflict resolution).

### Compacting factor drop (`include/apxchol/solver/sptrsv/omp.h`, CPU/omp backend)

`omp_sptrsv::setup` REMOVES factor off-diagonals below `rel * (column max |off-diagonal|)` before it builds its CSR/CSC (diagonal always kept; O(nnz) parallel, no atomics, deterministic; everything downstream — transpose, level sets, sync-free counters — sees only the compacted factor) and folds each column's dropped mass back into its kept off-diagonals (column sums preserved — the factor of a Laplacian has zero column sums, which is what keeps `L L^T` a Laplacian and the grounded `L11 L11^T` a reduced Laplacian with the right grounding mass; plain removal turns every dropped tiny edge into extra grounding at both endpoints and cost iter0040 45 → 67 PCG iterations on the Laplacian path, see the header comment). **ON by default** at `kFactorDropRelDefault = 1e-4`; env `APXCHOL_FACTOR_DROP=<rel>` (read at every setup) overrides: `0` (or anything <= 0) disables it, any other value replaces the threshold; `APXCHOL_FACTOR_DROP_COMPENSATE=0` gives plain removal (A/B only). The default is measured, not guessed — see the comment on `kFactorDropRelDefault` (0 PCG iterations change on grid_500 / grid_2000 / iter0040 / com-Amazon; grids and social graphs drop nothing, iter0040 drops ~52% of its off-diagonals). `omp_sptrsv::drop_stats()` / `stored_nnz()` report nnz before/after; `APXCHOL_VERBOSE` prints a `[apxchol] sptrsv storage` line per setup and the benchmark's `APXCHOL_REPORT_FILL` FILL line prints `stored_nnz=`. Unit tests: `tests/test_sptrsv_drop.cpp` (every build) plus the storage-variant clause in `tests/test_lowprec.cpp`. When you need the un-dropped factor in the SpTRSV (an exactness argument, a byte-comparison against an external factor), run with `APXCHOL_FACTOR_DROP=0`.

### Setup memory lifetime (peak RSS)

Peak RSS is set during setup, not the solve, and it is an overlap of transients — keep the lifetimes below in mind when touching setup. `factorize_impl` frees the elimination state (working graph, per-thread `factorize_workspace` — T vertex-indexed dedup buckets — round scratch, `selection`) **before** `build_csc`; assembly reads only `factor_cols` (+ n), and this overlap used to be the process peak. `detail::factor_col::entries` holds its values as `factor_value_t` (the factor's own precision — fp32 under `APXCHOL_SPTRSV_FP32` and every `APXCHOL_SPTRSV_LOWPREC` variant, NOT the SpTRSV's storage type; narrowing there is bit-identical to narrowing at assembly) — it is the largest setup-transient array (one entry per factor nnz, held until assembly). On the CPU path `apx_cholesky::install_factor` hands the factor to `omp_sptrsv::setup_consuming`, which calls `sparse_csc::release_values` on it the moment setup stops reading it (Laplacian path: right after its L11 copy) unless `set_keep_factor(true)`; `omp_sptrsv::setup(const&)` never releases; setup's own transients (L11 copy, compacted copy, transpose bucket, scratch) are released at their last use and `omp_sptrsv::memory_bytes()` is everything it keeps — `tests/test_sptrsv_memory.cpp` guards both (setup-only VmHWM vs the intended transient overlap; RSS after setup vs `memory_bytes()`). Freeing a `std::vector` means swap-with-empty — `v = {}` / `clear()` keep the capacity (same for `Eigen::SparseMatrix`: assign-from-empty keeps the buffers, use `.swap`). Verified peaks (T=16, `/usr/bin/time %M`, benchmark-style driver): iter0040 785 → 666 MiB, grid_2000 2337 → 2160 MiB, bit-identical factor/iterations at T=1.

### factor_options tuning knobs

`include/apxchol/solver/factor_options.h` is the authoritative source for tuning parameters. Several knobs have non-obvious empirical defaults documented in long comments there:
- `fs_compact_threshold = 0.0` (compaction off — empirically hurt grid and several IPM matrices when on).
- `parallel_residual_threshold = SIZE_MAX` (serial residual peel — fork-join overhead dominates on dense residuals).
- `residual_peel` defaults to `natural`; `min_degree` / `bk_serial` are opt-in for low-degree residuals.

If you change a default, re-read the rationale comment in `factor_options.h` before flipping it.

## Source-of-truth caveats

- `benchmarks/src/v0/` holds the frozen v0 prototype sources (`simple_solver`, `graphs`, `mmio`) that the benchmark suite links as a baseline. The current library namespace is `apxchol::` under `include/apxchol/` — don't take API patterns from the v0 files.
- The `data/`, `results/`, and `benchmarks/results/` directories are gitignored, **except `results/cells/`** — the per-cell benchmark store is tracked (it is the source-of-truth the committed `benchmarks/latest/` charts are rendered from; ~1MB of JSON). Sweeps will produce cell diffs. `scripts/download_graphs.sh` fetches SuiteSparse matrices into `data/matrices/`.
- `cmake-build-*` directories are CLion artifacts; the canonical build dir is `build/`.
