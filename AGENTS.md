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
- `APXCHOL_SPTRSV_LOWPREC` — string cache variable, **`OFF` by default**, one of `OFF | BF16 | BF16_SCALED | FP16_SCALED | FP24` (the old boolean `APXCHOL_SPTRSV_BF16=ON` is an alias for `BF16`). Selects a low-precision STORAGE format for the SpTRSV's CSR/CSC OFF-diagonal values (`sptrsv_value_t` = `bf16_t` / `fp16_t` / `fp24_t`, `include/apxchol/bf16.h` + `include/apxchol/lowprec.h`), narrowed once in `omp_sptrsv::setup` from the fp32 factor through `omp_sptrsv::narrow_value` (`sparse_csc::vals_` is `factor_value_t` = fp32 under every variant, same as the fp32 build); the DIAGONAL is kept fp32 in a separate `omp_sptrsv::diag_` array (rounding it to 8 bits was the dominant iteration-count damage): exactly `L_jj` under `BF16`/`FP24`, the scaled `fp32(L_jj / s_j)` under the `*_SCALED` variants (`omp_sptrsv::stored_diag` is the contract). The kernels are ONE source for every storage type — per entry (value load, `widen()`, index load, y gather, fma), per row/column the diagonal division (`fwd_diag()`/`bck_diag()`) — the arithmetic stays fp64. Variants: `BF16` = bfloat16, RNE (env `APXCHOL_BF16_STOCHASTIC=1`, read at every setup, switches to unbiased stochastic rounding with a deterministic per-entry CSC-index-hashed threshold; run-to-run deterministic, CSR/CSC-consistent); `BF16_SCALED` = bf16 of `L_ij / s_j` with a per-column scale `s_j` = max |off-diagonal| of column j (fp32 `omp_sptrsv::scale_`, `omp_sptrsv::column_scale`); `FP16_SCALED` = IEEE binary16 of `L_ij / s_j` (2^-11 relative; entries below 2^-25 of their column max flush to zero, and entries below 2^-14 — fp16 SUBNORMALS, 1–10 significant bits — are ALSO flushed to (signed) zero at storage time by default, env `APXCHOL_FP16_KEEP_SUBNORMAL=1` (read at every setup) restores IEEE subnormal storage; `setup` counts flushed / subnormal, `omp_sptrsv::lowprec_stats()`, printed under `APXCHOL_VERBOSE`); `FP24` = top 24 bits of the fp32 pattern (2^-16 relative, 3 B/entry, no scaling). **`*_SCALED` pair contract (the scale is FOLDED INTO THE VECTORS, `omp.h` file header):** what is stored is the column-scaled factor `L~ = L D^-1`, `D = diag(s_j)`, and the kernels never multiply a scale back: `forward_solve` runs on `L~` and returns `y' = D y` (NOT `y`), `transpose_solve` takes `y'` and solves `L~^T z = D^-2 y'` (its input read is scaled once per column by `inv_scale_[j]^2`, `inv_scale_[j] = fp32(1/s_j)`, `omp_sptrsv::inv_scale`), so the pair applies `(L L^T)^-1` (up to the 2^-23 of `inv_scale_`) — `preconditioner.h` only ever calls the pair; on the non-scaled builds `D = I` and both calls mean what they always did. Fat-level (`omp for`) kernels of the 16-bit variants (`BF16`, `BF16_SCALED`, `FP16_SCALED`) are SIMD on AVX2+F16C+FMA targets (`omp_sptrsv::simd_dot()`): 8 values per vector widen (`_mm256_cvtph_ps` / bf16 shift), two 4-double FMA lanes, y gather either an 8-double stack buffer + scalar gathers (`APXCHOL_FP16_GATHER=scalar`, the default — measured faster on grids) or `_mm256_i32gather_pd` (`=simd`); env read at every setup on every build; the fp32/fp64/FP24 fat-level loop is unchanged (instruction-identical, `kSimdDot`). Takes precedence over `APXCHOL_SPTRSV_FP32`; CPU/omp backend only (configure errors out with `APXCHOL_USE_CUDA`). Any new consumer of stored factor values must `widen()` them (the storage types' float/double conversions are explicit on purpose). Same variable exists in `benchmarks/CMakeLists.txt` for `apxchol_v1`.
- Runtime env knobs of the OpenMP SpTRSV / CPU PCG (every build incl. the default fp32 one and all LOWPREC variants; all read per setup / per solve, all default OFF):
  - `APXCHOL_FACTOR_DROP=<rel>` — COMPACTING drop in `omp_sptrsv::setup`, before the CSR/CSC (and the transpose) are built: off-diagonal (i,j) is kept iff |L_ij| >= rel * (column j's max |off-diagonal|, `omp_sptrsv::column_scale`, computed BEFORE the drop) AND the storage format does not store it as zero anyway (`omp_sptrsv::format_flushes`: exact zeros on fp32/fp64/bf16/fp24; on FP16_SCALED also everything fp16 flushes, subnormals included by default) — `omp_sptrsv::keep_offdiag` is the pure predicate; the diagonal is always kept. O(nnz) parallel (per-column count → prefix → compacted copy); everything downstream (transpose, CSC copy, level sets, round-as-level) sees only the compacted factor, so nnz(L stored) and the CSR/CSC bytes shrink. Measured (fp32 build, T=1, tol 1e-8): rel=1e-4 costs 0 PCG iterations on grid_500 / grid_2000 / iter0040 and removes ~52% of iter0040's off-diagonals (0% on grids). `omp_sptrsv::lowprec_stats()` reports `nnz_factor` / `nnz_stored` (`stored_nnz()`) / `dropped` (= `dropped_threshold` + `dropped_flush`); `APXCHOL_VERBOSE` prints them (`stored_nnz=` on the storage line, plus a `factor drop` line), and the benchmark's `APXCHOL_REPORT_FILL` FILL line prints `stored_nnz=`. Supersedes the removed `APXCHOL_LOWPREC_DROP` diagnostic (same threshold and numerics — a stored zero and an absent entry solve identically — but that one only zeroed in place).
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

`apx_cholesky` (`include/apxchol/solver/preconditioner.h`) inherits `Eigen::SparseSolverBase` and is meant to be plugged into `Eigen::ConjugateGradient<SpMat, UpLo, apxchol::apx_cholesky>`. `analyzePattern` is a no-op; `factorize` calls into `apxchol::factorize` and then sets up the SpTRSV backend.

### Laplacian vs SDDM rank

`factorization::sddm` distinguishes the two cases and changes the preconditioner application path:
- Laplacian (positive semidefinite, rank `n−1`): factor dimension is `n−1`; `_solve_impl` subtracts the mean of `b` before solving and re-centers `x` afterwards.
- SDDM (positive definite, full rank): full `n×n` factor, no centering.

When touching `_solve_impl` or anything that constructs a `factorization`, check both branches.

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
