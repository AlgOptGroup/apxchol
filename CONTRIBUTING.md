# Contributing

Bug reports, portability fixes, benchmark matrices and solver improvements
are welcome.

## Build and test

Build and run the suite as in the [README](README.md#build-and-run); `ctest`
also runs the examples. Run one test while iterating with
`./build/tests/unit_tests --gtest_filter='FactorizeTest.PermutationIsValid'`.

CI builds `tests/cmake_consumer` to check the
[CMake consumer](README.md#use-from-cmake) requirements with warnings as errors.
The [workflow](.github/workflows/ci.yml) defines compiler and backend coverage;
the CUDA job compiles without running device tests.

Use focused regressions for changed behavior, then the relevant suite. CUDA
changes require a local CUDA build and device run. Metal changes require an
`-DAPXCHOL_DEVICE=METAL` build on an Apple-silicon Mac; also run the device
tests under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`. Python and Octave
compile the core sources independently; check affected binding interfaces too:

```bash
pip install -e python
pytest python/tests
./octave/build.sh
```

The release version is set in `cmake/apxchol_version.cmake`; native builds and
Python packages share it. PyPI publication requires a matching `v<version>`
GitHub release tag. Manually running the publish workflow only builds the wheels.

## Code conventions

Use C++23, `snake_case`, namespace `apxchol`, and trailing underscores for
private members. C++ headers are under `include/`; [API scope](docs/api.md)
defines the public interface. The CPU compiled core is `factorization.cpp`,
`operator_class.cpp` and `solve.cpp` in `src/`.

Register new partitioners in `partitioner_list.h`. The working graph uses
directed AoS adjacency. Preserve deterministic neighbor ordering, sampling
streams and thread-team behavior. Consult the
rationale in `factor_options.h` and the [precision contract](docs/precision.md)
before changing established defaults. Prefer a shared implementation over
another mode or public tuning option.

## Performance

The [benchmark suite](benchmarks/README.md) is a separate CMake project.
Use a controlled, recorded machine and keep campaign outputs outside tracked
source. Interactive laptop measurements are diagnostic.
Use matched, interleaved repetitions with pinned source, binaries, inputs,
affinity and timing boundaries. Report setup and solve separately, together
with residuals, fill, memory and control spread. Do not combine isolated
speedups into an unmeasured cumulative claim.
