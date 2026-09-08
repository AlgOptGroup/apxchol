# Contributing

Bug reports, portability fixes, benchmark matrices and solver improvements
are welcome.

## Build and test

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j6
ctest --test-dir build --output-on-failure
./build/tests/unit_tests --gtest_filter='FactorizeTest/*.PermutationIsValid'
```

Use focused regressions for changed behavior, then the relevant suite. CUDA
changes require a CUDA build and device. Python and Octave compile the core
sources independently; check affected binding interfaces too:

```bash
pip install -e python
pytest python/tests
./octave/build.sh
```

## Code conventions

Use C++23, `snake_case`, namespace `apxchol`, and trailing underscores for
private members. Public headers are under `include/apxchol/`; the CPU compiled
core is `factorization.cpp`, `operator_class.cpp` and `solve.cpp` in `src/`.

Register new partitioners in `partitioner_list.h`; register storage backends
in both `graph_storage` and factorization dispatch. Preserve deterministic
neighbor ordering, sampling streams and thread-team behavior. Consult the
rationale in `factor_options.h` and [implementation history](docs/implementation-history.md)
before changing established defaults. Prefer a shared implementation over
another mode or public tuning option.

## Performance

The [benchmark suite](benchmarks/README.md) is a separate CMake project.
Daint is the primary performance platform; laptop measurements are diagnostic.
Use matched, interleaved repetitions with pinned source, binaries, inputs,
affinity and timing boundaries. Report setup and solve separately, together
with residuals, fill, memory and control spread. Do not combine isolated
speedups into an unmeasured cumulative claim.
