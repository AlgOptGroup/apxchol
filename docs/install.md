# Installation

| You want | Install |
|---|---|
| Python | `python -m pip install "apxchol[scipy]"` |
| A command-line solver without Python | A [CLI archive](#cli-archives) |
| A C++ library for your CMake project | An [SDK archive](sdk.md) or the [source tree](cpp.md) |
| MATLAB or Octave | A [source build](../octave/README.md#build-for-octave) |
| The development version | A [source build](building.md) |

## Python wheels

```bash
python -m pip install "apxchol[scipy]"
```

The `scipy` extra adds SciPy sparse-matrix input. Without SciPy, use
[NumPy-only input](../python/README.md#numpy-only-input). Wheels run on CPU
without a GPU and include CUDA (Linux) or Metal (macOS); the
[Python guide](../python/README.md#install-and-solve) lists supported platforms.

The latest release on PyPI is 0.2.2, with wheels for CPython 3.10–3.14 on
Linux x86-64. It predates the changes in the [changelog](../CHANGELOG.md),
such as `as_operator` and the macOS and PyPy wheels. To use them before the
next release, [build from source](../python/README.md#build-from-source).

## CLI archives

New releases attach standalone CLI archives for Linux (CPU) and Apple silicon
(CPU and Metal); successful [CI runs](https://github.com/AlgOptGroup/apxchol/actions/workflows/ci.yml)
upload the same archives as `cli-*` artifacts. Python is not required.

Linux archives require x86-64-v2, glibc 2.28+ and system C++/GCC runtimes. macOS
archives require Apple silicon and macOS 15+.

1. Verify the matching `.sha256` file with `sha256sum -c` (Linux) or
   `shasum -a 256 -c` (macOS).
2. Extract the archive and run `bin/apxchol --help`.

Keep `bin/` and `lib/` together; OpenMP and redistribution notices are included.
macOS binaries are ad-hoc signed, not notarized, so Gatekeeper may ask for
manual approval.

From a source build, `cmake --install build --component cli --prefix /path/to/prefix`
installs the executable using your system runtime libraries.

## From source

[Building from source](building.md) lists the compilers and dependencies, and
the build options for CUDA and Metal.
