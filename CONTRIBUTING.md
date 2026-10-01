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

CMake consumers should link `apxchol_core` rather than copy its compiler flags:

```cmake
add_subdirectory(path/to/apxchol)
target_link_libraries(my_program PRIVATE apxchol_core)
```

The target supplies C++23 and the native architecture flag used by the library.
Matching architecture settings matter because Eigen objects cross the library
boundary. Use `-DAPXCHOL_NATIVE_ARCH=OFF` for portable builds. The CI library
build uses `tests/cmake_consumer` as an outer project to test these requirements
alongside the ordinary suite.

On macOS, `brew install libomp` first; CI builds the same consumer with Apple
Clang on macos-15.

Use focused regressions for changed behavior, then the relevant suite. CUDA
changes require a CUDA build and device. Metal changes require an
`-DAPXCHOL_USE_METAL=ON` build on an Apple-silicon Mac; also run the device
tests under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`. Python and Octave compile the core
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
Use a controlled, recorded machine; committed platform results are historical
snapshots. Interactive laptop measurements are diagnostic.
Use matched, interleaved repetitions with pinned source, binaries, inputs,
affinity and timing boundaries. Report setup and solve separately, together
with residuals, fill, memory and control spread. Do not combine isolated
speedups into an unmeasured cumulative claim.
