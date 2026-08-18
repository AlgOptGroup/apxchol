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

`cpu_solver::solve_impl` (`src/solve.cpp`) is not a chain of Eigen expressions: each iteration is four fused, OpenMP-parallel passes — `spmv+pAp` (p·Ap folded into the SpMV row loop), `update_xr_norm` (x += αp, r −= αAp, r·r, Σr in one pass), the preconditioner application (`pcg.solve.{permute,forward,back,unpermute+rz}`: input centering fused into the permute scatter, output re-centering + r·z fused into the unpermute gather), and `update_p`. Those are also the checkpoint labels under `pcg.*`. Every reduction uses the scheme in `preconditioner.h` `detail::` (fixed `static_chunk` partition, per-thread partials summed in thread order — never a `reduction()` clause), so results are bit-identical run to run for a fixed thread count (they differ across thread counts at fp-rounding level, like every other parallel pass). `PcgFusion.*` in `tests/test_factorize.cpp` guards this; keep new vector passes on the same scheme.

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
