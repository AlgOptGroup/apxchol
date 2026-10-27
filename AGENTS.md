# AGENTS.md

## Scope and evidence

- Deliver the requested scope. Prefer removing redundancy to adding modes, options or knobs. Keep this file current when builds, APIs, defaults or architecture change.
- State a result as fact only with evidence from the current session; label historical results and hypotheses. Audits report the complete denominator, checked/total and exclusions.
- Factor identity requires structure and value digests; equal fill, iterations, residuals or seeds are not identity.
- Check the worktree, branch and dirty state before editing; give concurrent agents explicit file ownership. Keep private notes, runbooks and campaign records out of tracked source.

## Build and validation

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j6
ctest --test-dir build --output-on-failure        # unit tests, CLI checks, examples
./build/tests/unit_tests --gtest_filter='FactorizeTest.PermutationIsValid'
```

Run focused regressions first, then the relevant suite. CUDA and Metal correctness needs a matching build and device; CI only compiles CUDA. Check affected Python (`pytest python/tests`) and Octave (`octave/build.sh`) interfaces. Keep timing separate from builds and other workloads.

## Where things live

- Algorithm defaults and their rationale: `include/apxchol/solver/factor_options.h`. Build options: root `CMakeLists.txt` and `cmake/apxchol_core.cmake`, which the root, Python and benchmark projects share.
- `APXCHOL_INDEX_WIDTH=32|64_EDGES|64` selects vertex/offset widths of 32/32, 32/64 or 64/64 bits; the default is `32`.
- `cmake/apxchol_version.cmake` is the version source for native builds and Python packages. Release tags must be `v` followed by that version.
- `cmake/core` builds just the native library and defaults to portable code (`APXCHOL_NATIVE_ARCH=OFF`); standalone root and local Python builds default to native tuning. Root `add_subdirectory` embedding defaults to portable core only; CLI/tests/examples are opt-in there, while standalone root defaults remain enabled. Source consumers include `cmake/apxchol_core.cmake` and call `apxchol_add_core()` at their top level; installed consumers use `find_package(apxchol CONFIG)`. Both link `apxchol::core`; installed consumers must use the same OpenMP runtime family.
- `.github/scripts/build-sdk.sh` packages the shared native install with portable tuning and bundled Eigen; `tests/check_sdk_archive.py` extracts it and builds the `tests/cmake_install` save/load consumer.
- Native SDK installs include Eigen headers when apxchol fetched them; builds using an existing Eigen package retain that dependency. Python wheels do not install the SDK.
- Portable save/load: `solver/archive.h` and `docs/serialization.md`; the versioned binary format stores the original CSC operator and retained canonical factor. Typed arrays carry explicit widths, byte order and zlib CRC32 checksums; loading validates and prepares the backend without memory mapping.
- MATLAB/Octave builds use `octave/CMakeLists.txt` and the shared `apxchol::core` target. Archive methods reuse native save/load and require `keep_factor=true` when saving.
- Public API scope and compatibility: `docs/api.md`. Precision and storage: `docs/precision.md`. Extension seams: `docs/extending.md`.
- Operator validation and M-matrix lumping: `operator_class`. Adjacency detection belongs to the CLI (`src/mtx_input.h`); bindings accept assembled operators only.
- Measurement contracts: `benchmarks/README.md`.
- CLI archives: `.github/workflows/cli.yml` uses the `cli` install component, bundles OpenMP/notices, and runs `tests/check_cli_archive.py` after relocation.
- Documentation: `python docs/build.py` reuses the guides, public C++ comments, Python docstrings and CLI help. Build instructions are in `CONTRIBUTING.md`; generated HTML goes to `build/docs`.

## Route invariants

- Solver controls are typed per-instance options, not environment variables. Validate them before setup; `drop_tolerance` is in `[0, 1]`. Bindings share compiled defaults. Libraries are quiet by default, and `collect_timings` controls optional checkpoints. Standard OpenMP/CUDA runtime and build controls remain external; reference-only tuning stays internal to tests.
- `solve_backend` (`automatic`, `cpu`, `gpu`) fixes one complete setup/solve route before setup. Explicit GPU requests fail for unsupported options, input or devices. Once selected, errors never retry on another route. Report the selected route in results and CLI output.
- `cpu_solver`, public `factorize()` and Eigen preconditioning stay on CPU in GPU builds and must not initialize CUDA. Disabled or pristine GPU lifetimes must not initialize CUDA either.
- `backend_available()` checks build/device availability, not compatibility with every solver option. CPU and automatic queries never initialize a GPU.
- CUDA and Metal construct new factors on device or upload an explicitly supplied host factor. Operator preparation and PCG remain on device. They require compressed, sorted, unique, fully paired CSC input; automatic selection uses CPU otherwise. They report a recomputed FP64 residual against the original operator, with verification inside Solve timing.

## Determinism and numerical invariants

- Public solver preconditioner application rejects non-finite inputs and centers Laplacian input/output on every call; only internal PCG uses periodic centering. All backends derive `converged` from the original-system exit residual and effective tolerance, with verification included in Solve timing.
- Solves stop on tolerance, the iteration budget or numerical breakdown. Residual growth or a temporary plateau is not a stopping condition.
- Canonical neighbors sort by weight, then vertex. Preserve ordered duplicate sums, exact cumulative weights, `upper_bound` sampling and per-elimination random streams.
- OpenMP work uses the actual team size, never the requested one; every logical range must execute under reduced or nested teams. Caller-operator loops split work with `detail::work_balanced_range`; reductions stay thread-ordered.
- Partitioner conflict resolution is deterministic for a given graph, context and team size. Symmetric-input pairing shortcuts must preserve results bit for bit. Prune skipping must retain its full-walk audit for hidden eligible vertices; it need not reproduce the same factor as full pruning.
- Residual importance normalization uses fixed ordered 16,384-item partials, invariant to thread count. Same-graph selection and same-factor solves are deterministic; independently rebuilt parallel factors need not be identical (see `docs/precision.md`).
- GPU selection guarantees independence and progress, not the CPU's selected set; GPU floating folds may differ from CPU while preserving the sampling law.

## Memory invariants

- Uninitialized output buffers (`big_alloc<T,32,false,false>`, L11 temporaries) must be fully overwritten before first read; writer threads perform first touch.
- Release transients at their last use and preserve factor-buffer lifetime through assembly; `clear()` does not release capacity. Requested and retained GPU scratch enter memory preflight.

## Test map

| Change | Tests |
|---|---|
| Graph, partitioners, elimination, sampling | `FactorizeTest.*`, `FactorizeDeterminism.*`, `AllStrategies/*`, `VecPoolAos.*`, `DirectedCscGraph.*`, `PruneSkip.*`, `IncrementalDegreeGate.*`, `TreeSampler.*`, `CycleSampler.*`, `Custom*` |
| Residual sparsification | `ResidualBlocked*`, `ResidualSparsify*` |
| CPU triangular solves, storage | `SpTRSV*`, `CriticalSchedule.*`, `BigAlloc.*`, `FP16.*`, `LowPrec*`, `FactorStorage.*` |
| Root embedding | `tests/cmake_subdirectory`, with default options and with tests ON / CLI OFF; GPU parents enable CUDA or OBJCXX before adding the root |
| Operators, input, routes, owners | `Operator*`, `MMatrixLumping.*`, `PermutedOperator.*`, `MtxInput*`, `AdjacencyInputRegression.*`, `SolveRoutes.*`, `Solve*`, `Solver.*`, `Checkpoint.*` |
| CUDA | `Gpu*`, `CudaDeviceScope.*`, `gpu_factor_finalize_leak_check` (with compute-sanitizer) |
| Metal | `Metal*`, `GpuDataflowPlan.*`; also under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` |
| Benchmark harness | `benchmarks/tests` (unittest) and the benchmark project's CTest contracts |

Integration correctness is not performance acceptance: compare original-system quality, setup, solve, one-RHS total, reuse and memory against a frozen baseline, following `benchmarks/README.md`.
