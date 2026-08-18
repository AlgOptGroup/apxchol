# Contributing

Issues and pull requests are welcome — bug reports, portability fixes, new
matrices for the benchmark set, and solver improvements alike.

## Build and test

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

A single test or filter:

```bash
./build/tests/unit_tests --gtest_filter='FactorizeTest/*.PermutationIsValid'
```

Tests live in `tests/` (GoogleTest, typed across the storage backends, so one
`TYPED_TEST` registers several ctest cases). New behaviour needs a test there;
CI (`.github/workflows/ci.yml`) builds and runs the library, Python and Octave
suites — its path filters cover the library, tests, bindings, and examples.

The Python package (`pip install -e python`, `pytest python/tests`) and the
Octave MEX (`cd octave && ./build.sh`) are separate build systems that compile
the two core TUs directly — they do not use the root CMake build, so a change to
`src/` or `include/` should be checked against all three. (Deliberate: the
wheel build runs in isolated manylinux containers where nothing is installed,
and the core is two small TUs — compiling them directly is simpler and more
robust than exporting/consuming a CMake package. Revisit if the TU count
grows.)

## Code style

- **C++23** (`CMAKE_CXX_STANDARD 23`; the core uses "deducing this", so GCC ≥ 14
  or an equivalent Clang is required).
- `snake_case` for types, functions, and variables; trailing `_` on private
  members. Namespace is `apxchol::`.
- The core is header-only under `include/apxchol/` plus two compiled TUs
  (`src/factorization.cpp`, `src/solve.cpp`; CUDA builds add `src/cuda_cast.cu`,
  `src/cuda_levelset.cu` and `src/cuda_dataflow.cu`). Templates and inline logic belong
  in the headers; the TUs exist to instantiate the runtime-dispatch entry points.
- Keep the two-axis dispatch consistent: a new independent-set partitioner must
  be appended to `partitioner_list` in `partitioner_list.h`, and a new storage
  backend must be added to the `graph_storage` enum *and* the dispatch switch in
  `src/factorization.cpp` — otherwise the CLI and preconditioner paths silently
  cannot reach it.
- Tuning defaults in `include/apxchol/solver/factor_options.h` carry rationale
  comments explaining why they are what they are. Read the comment before
  flipping a default, and update it if you do.

## Benchmarks

The benchmark suite is a **separate CMake project** under `benchmarks/` that
FetchContent-pulls external solvers (Hypre, AMGCL, RCHOL, ...). It is not built
by the root project and is not part of CI:

```bash
cmake -B benchmarks/build -S benchmarks -DCMAKE_BUILD_TYPE=Release
cmake --build benchmarks/build -j$(nproc) benchmark
python3 benchmarks/sweep_fair.py --device cpu
```

See [benchmarks/README.md](benchmarks/README.md) for the protocol, the solver
list, and how the per-cell result store feeds the committed charts.

## Performance claims

Timing on a thermally constrained machine drifts by tens of percent between
sessions, so a "this is X% faster" claim needs more than one run of each side:

- Use the stable-bench harness in `benchmarks/dev/`: run
  `sudo bash benchmarks/dev/bench_stable_setup.sh` first (disables boost, pins
  the governor), then `bench_stable.sh` / `bench_stable_all.sh`, and restore
  with `bench_stable_teardown.sh`.
- Report **paired A/B** numbers measured in the same session with the same
  binary flags, medians vs medians (never a median against a single-shot run),
  together with the interquartile range (IQR, the spread between the 25th and
  75th percentile of the repeats — a robust noise estimate). A delta smaller
  than the IQR is not a result.
- State the thread count, the matrix, and whether boost was on or off. Numbers
  taken with boost on are not comparable to locked-clock numbers.
