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
- Precision and storage contract: `docs/precision.md`. Extension seams: `docs/extending.md`.
- Operator validation and M-matrix lumping: `operator_class`. Adjacency detection belongs to the CLI (`src/mtx_input.h`); bindings accept assembled operators only.
- Measurement contracts: `benchmarks/README.md`.

## Route invariants

- Solver controls are typed per-instance options, not environment variables. Validate them before setup; bindings share compiled defaults. Libraries are quiet by default, and `collect_timings` controls optional checkpoints. Standard OpenMP/CUDA runtime and build controls remain external; reference-only tuning stays internal to tests.
- `solve_backend` (`automatic`, `cpu`, `gpu`) fixes one complete setup/solve route before setup. Explicit GPU requests fail for unsupported options, input or devices. Once selected, errors never retry on another route. Report the selected route in results and CLI output.
- `cpu_solver`, public `factorize()` and Eigen preconditioning stay on CPU in GPU builds and must not initialize CUDA. Disabled or pristine GPU lifetimes must not initialize CUDA either.
- CUDA and Metal own factorization, finalization, operator preparation and PCG on device. They require compressed, sorted, unique, fully paired CSC input; automatic selection uses CPU otherwise. They report a recomputed FP64 residual against the original operator, with verification inside Solve timing.

## Determinism and numerical invariants

- Public preconditioner application centers Laplacian input/output on every call; only internal PCG uses periodic centering. All backends derive `converged` from the original-system exit residual and effective tolerance, with verification included in Solve timing.
- Canonical neighbors sort by weight, then vertex. Preserve ordered duplicate sums, exact cumulative weights, `upper_bound` sampling and per-elimination random streams.
- OpenMP work uses the actual team size, never the requested one; every logical range must execute under reduced or nested teams. Caller-operator loops split work with `detail::work_balanced_range`; reductions stay thread-ordered.
- Partitioner conflict resolution is deterministic for a given graph, context and team size. Symmetric-input pairing shortcuts must preserve results bit for bit. Prune skipping must retain its full-walk audit for hidden eligible vertices; it need not reproduce the same factor as full pruning.
- Residual importance normalization uses fixed ordered 16,384-item partials, invariant to thread count. Same-graph selection and same-factor solves are deterministic; independently rebuilt parallel factors need not be identical (see `docs/precision.md`).
- GPU selection guarantees independence and progress, not the CPU's selected set; GPU floating folds may differ from CPU while preserving the sampling law.

## Memory invariants

- Uninitialized output buffers (`big_alloc<T,32,false,false>`, L11 temporaries) must be fully overwritten before first read; writer threads perform first touch.
- Release transients at their last use and preserve factor-buffer lifetime through assembly; `clear()` does not release capacity. Requested and retained GPU scratch enter memory preflight.

## C interface

`include/apxchol/c_api.h` and `src/c_api.cpp` define the exception-safe `apxchol_c` target. Scalar and batched calls delegate to the shared CPU/CUDA/Metal owner; dense outputs must not overlap inputs. Keep defaults aligned with `solve_options` and ABI layouts aligned with `tests/test_c_header.c`. ABI 2 replaces the unreleased prototype; incompatible signatures or layouts require a version bump. `NOT_CONVERGED` writes outputs without relaxing acceptance. Factor handles own immutable FP32 values; adoption copies them and is unsupported on CUDA. Thread ceilings restore the caller's setting.

## Test map

| Change | Tests |
|---|---|
| Graph, partitioners, elimination, sampling | `FactorizeTest.*`, `FactorizeDeterminism.*`, `AllStrategies/*`, `VecPoolAos.*`, `DirectedCscGraph.*`, `PruneSkip.*`, `IncrementalDegreeGate.*`, `TreeSampler.*`, `CycleSampler.*`, `Custom*` |
| Residual sparsification | `ResidualBlocked*`, `ResidualSparsify*` |
| CPU triangular solves, storage | `SpTRSV*`, `CriticalSchedule.*`, `BigAlloc.*`, `FP16.*`, `LowPrec*`, `FactorStorage.*` |
| Operators, input, routes, owners | `Operator*`, `MMatrixLumping.*`, `PermutedOperator.*`, `MtxInput*`, `AdjacencyInputRegression.*`, `SolveRoutes.*`, `Solve*`, `Solver.*`, `Checkpoint.*` |
| CUDA | `Gpu*`, `CudaDeviceScope.*`, `gpu_factor_finalize_leak_check` (with compute-sanitizer) |
| Metal | `Metal*`, `GpuDataflowPlan.*`; also under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` |
| C ABI | `CApi.*`, `c_api_c_consumer` |
| Benchmark harness | `benchmarks/tests` (unittest) and the benchmark project's CTest contracts |

Integration correctness is not performance acceptance: compare original-system quality, setup, solve, one-RHS total, reuse and memory against a frozen baseline, following `benchmarks/README.md`.
