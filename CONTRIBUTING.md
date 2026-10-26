# Contributing

Bug reports, portability fixes, benchmark matrices and solver improvements
are welcome.

## Build and test

Build and run the suite as in the [README](README.md#build-and-run); `ctest`
also runs the examples. Run one test while iterating with
`./build/tests/unit_tests --gtest_filter='FactorizeTest.PermutationIsValid'`.

CI builds `tests/cmake_consumer` to check the
[CMake consumer](docs/building.md#use-from-cmake) requirements with warnings as errors.
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
GitHub release tag. Manually running the publish workflow builds artifacts without publishing them.

## CLI archives

`.github/workflows/cli.yml` builds Linux CPU and macOS CPU+Metal archives for CI
and release publishing. Artifact names and version checks use
`cmake/apxchol_version.cmake`. The reusable jobs upload `cli-*` artifacts; release
publishing attaches their uniquely named `.tar.gz` and `.tar.gz.sha256` files only
after both CLI and wheel builds pass. Manual publish runs only build artifacts.

The workflow contains the complete portable build commands. Given such a static
build with fetched dependencies and the relative install RPATH, package it with:

```bash
python packaging/cli.py build-cli dist
python packaging/check_cli_archive.py dist/*.tar.gz "$PWD" "$PWD/build-cli"
```

The Linux build uses the pinned manylinux image and copies the installed OpenMP
RPM's license. macOS uses its installed Homebrew OpenMP version and license (or the
matching upstream license when the bottle omits it). Both include the fetched CLI
dependency notices and source provenance. Packaging fails on unexpected dynamic
dependencies. The relocation check verifies help/version, bundled runtime loading
and independently recomputed residuals; macOS also requires an explicit Metal
solve with validation layers and denies reads from source/build/runtime paths.
Linux checks run in a fresh offline container without source or build mounts.

The macOS job runs on macOS 15, builds arm64 and checks ad-hoc signatures. It does
not use signing credentials or notarization services. Linux CUDA archives are not
part of this distribution matrix; CUDA remains available through source builds.

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
Documentation CI uploads an HTML preview for review. Successful main builds
publish the same site to GitHub Pages.

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
