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

## Writing user-facing text

These rules apply to text a user reads: `README.md`, the guides in `docs/`, `python/README.md`, the CLI's
`--help`, and the messages of exceptions that reach a user. They don't apply to code comments, debug output or
commit messages.

1. One idea per sentence. No semicolons. Split the sentence or use a list.
2. Keep sentences short: about 20 words on average, and none over 30.
3. Put the outcome first. The first words say what happened or what the reader must do.
4. An error says what happened, why, and what to do next, in that order. Put codes and IDs after the sentence. Don't
   write "sorry", "oops" or "something went wrong" on its own.
5. Use the reader's words. Defined technical terms such as preconditioner, PCG or CSR are fine. Define an internal
   name (a route code, a struct name) the first time, or leave it out.
6. Use active voice and name the actor: you, the solver, the CLI.
7. Write numbers as digits with units, such as "1 hour" or "5 GB". Don't write "may", "soon" or "a while" when the
   code knows the answer.
8. Use one term per thing and one verb per action. The list is in `scripts/text-rules.json`. A default in the guides
   matches `--help` and `factor_options.h`.
9. Don't write "we" unless the text says who "we" is.
10. Every number, limit, default or guarantee in the text matches the code that sets it. A change to the value
    changes the text in the same commit.
11. A `--help` line fits on 1 line and ends with the default.

Check: `python3 scripts/check_text_rules.py`. It reads `scripts/text-rules.json` and compares each file with
`scripts/text-rules-baseline.json`. Existing text is allowed. A file that gains a hit fails. A file that loses one
fails too, so lower the baseline in the same commit with `--update-baseline`. Mark a deliberate exception with a
comment `text-rules-allow: R1 <reason>` on that line or the line before.
A review of a change to user-facing text ends with a line "Text rules: pass" or the list of breaks.

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
