# Contributing

Bug reports, portability fixes, benchmark matrices and solver improvements
are welcome.

## Build and test

Build and run the suite as in the [README](README.md#build-and-run); `ctest`
also runs the examples. Run one test while iterating with
`./build/tests/unit_tests --gtest_filter='FactorizeTest.PermutationIsValid'`.

CI builds `tests/cmake_consumer` to check the
[CMake consumer](docs/cpp.md#use-from-cmake) requirements with warnings as errors. The root-embedding
consumer reuses the same alignment and original-residual checks:

```sh
cmake -S tests/cmake_subdirectory -B build-subdirectory
cmake --build build-subdirectory -j2
ctest --test-dir build-subdirectory --output-on-failure
```

To check native tests without CLI dependencies, repeat in a fresh build directory
with `-DAPXCHOL_BUILD_TESTS=ON -DAPXCHOL_BUILD_CLI=OFF`. The same consumer fixture
supports `-DAPXCHOL_DEVICE=CUDA` or `METAL`, enabling the GPU language in its
parent scope before embedding.
The [workflow](.github/workflows/ci.yml) defines compiler and backend coverage;
the CUDA job compiles without running device tests.

Use focused regressions for changed behavior, then the relevant suite. CUDA
changes require a local CUDA build and device run. Metal changes require an
`-DAPXCHOL_DEVICE=METAL` build on an Apple-silicon Mac; also run the device
tests under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`. Python and Octave
compile the core sources independently; check affected binding interfaces too:

```bash
pip install -e "python[test,scipy]"
pytest python/tests
./octave/build.sh
```

The release version is set in `cmake/apxchol_version.cmake`; native builds and
Python packages share it. PyPI publication requires a matching `v<version>`
GitHub release tag. Manually running the publish workflow builds artifacts without publishing them.
A GitHub prerelease with that tag also builds everything and attaches the CLI and
SDK archives, but skips PyPI. Turning a prerelease into a release does not start
the workflow again; delete the prerelease and its tag, then publish the release.

The [CLI workflow](.github/workflows/cli.yml) sets up each platform and calls
`.github/scripts/build-cli.sh`: normal CMake install, OpenMP/notices, tar and checksum.
`tests/check_cli_archive.py` checks relocation and the existing SDDM fixture on CPU
and Metal. The version comes from the same CMake source used by wheels.

## Documentation

Install Doxygen, build the CLI as above, then run:

```bash
python -m pip install -r docs/requirements.txt -e ./python
python docs/build.py
python -m http.server 8000 --directory build/docs
```

The site reuses the repository guides, selected C++ comments, Python docstrings
and the CLI's `--help`. Generated files stay in `build/docs`. To use another
CLI or output directory, pass them as positional arguments to `docs/build.py`.
To add a page, list it in `GUIDES` in `docs/build.py` and in a `docs/index.md`
toctree. Development builds show a banner naming the release from the
changelog's `Unreleased (since vX.Y.Z)` heading.
Documentation CI uploads an HTML preview for review. Successful main builds
publish development documentation at `/dev/`. Each stable GitHub release saves
its rendered HTML as `documentation.tar.gz`; reruns retain the existing snapshot.
The site root serves the most recently published stable release with a snapshot,
or development documentation until the first snapshot exists. Each snapshot is
also served at `/vX.Y.Z/`. The sidebar version selector lists these snapshots and
development documentation; releases without a documentation asset are omitted.
Assembly updates only the marked selector in deployed HTML copies; the release
archives remain unchanged. Older HTML without the marker remains reachable from
the current site's selector. Publishing a snapshot triggers a main documentation
build to refresh the site.

## Changelog

A pull request with a user-visible change adds a line to the `Unreleased`
section of `CHANGELOG.md`. Mark breaking changes. Benchmark-internal and
CI-only changes usually need no entry.

## Code conventions

Use C++23, `snake_case`, namespace `apxchol`, and trailing underscores for
private members. C++ headers are under `include/`; [API scope](docs/api.md)
defines the public interface. The CPU compiled core is `factorization.cpp`,
`operator_class.cpp` and `solve.cpp` in `src/`.

A new built-in partitioner needs a `partitioner_kind` value and name in
`factor_options.h` and a case in `detail::factorize_with_named_partitioner`.
The working graph uses directed AoS adjacency. Preserve deterministic neighbor ordering, sampling
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
