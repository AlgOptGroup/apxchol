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

## CLI archives

New releases attach standalone CLI archives for Linux (CPU) and Apple silicon
(CPU and Metal); successful [CI runs](https://github.com/AlgOptGroup/apxchol/actions/workflows/ci.yml)
upload the same archives as `cli-*` artifacts. Python is not required.

Linux archives require x86-64-v2, glibc 2.28+ and system C++/GCC runtimes. macOS
archives require Apple silicon and macOS 15+.

1. Verify the archive with its checksum file:
   `sha256sum -c apxchol-*.tar.gz.sha256` (Linux) or
   `shasum -a 256 -c apxchol-*.tar.gz.sha256` (macOS).
2. Extract the archive and run `bin/apxchol --help`.

Keep `bin/` and `lib/` together; OpenMP and redistribution notices are included.
macOS binaries are ad-hoc signed, not notarized. If Gatekeeper blocks the first
run, approve the binary in System Settings, Privacy & Security.

From a source build, `cmake --install build --component cli --prefix /path/to/prefix`
installs the executable using your system runtime libraries.

## From source

[Building from source](building.md) lists the compilers and dependencies, and
the build options for CUDA and Metal.
