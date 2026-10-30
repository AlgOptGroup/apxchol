# Building and installation

## Dependencies

CPU builds require CMake 3.20+, GCC or Clang with C++23 and OpenMP, and zlib headers.
CMake uses an installed Eigen package when available and fetches it otherwise.
Windows builds with MSVC or `clang-cl` are not supported.
On x86-64, the minimum target is x86-64-v2.

CUDA builds require CUDA Toolkit 13.3+ and CMake 4.4+ for C++23 support.
Metal builds require macOS and Apple silicon. On macOS, install Xcode Command
Line Tools and `brew install cmake libomp`.

These are source-build dependencies. Using a Python wheel needs neither the
CUDA Toolkit nor Homebrew, and every wheel can run on CPU without a GPU.

## Build the CLI and tests

```bash
git clone https://github.com/AlgOptGroup/apxchol
cd apxchol
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j6
ctest --test-dir build --output-on-failure
```

The default build is CPU-only. Add `-DAPXCHOL_DEVICE=CUDA` or
`-DAPXCHOL_DEVICE=METAL` at configuration to include that GPU backend as well.
The same executable still supports CPU solves.

## CLI archives

Download `cli-*` artifacts from a successful [CI run](https://github.com/AlgOptGroup/apxchol/actions/workflows/ci.yml);
new [releases](https://github.com/AlgOptGroup/apxchol/releases) attach the archives directly.
Linux archives require x86-64-v2, glibc 2.28+ and system C++/GCC runtimes. macOS
archives require Apple silicon and macOS 15+, and include CPU and Metal routes.
Verify the matching `.sha256` file with `sha256sum -c` (Linux) or `shasum -a 256 -c`
(macOS), extract the archive, then run its `bin/apxchol --help`. Keep `bin/` and
`lib/` together; OpenMP and redistribution notices are included. macOS binaries
are ad-hoc signed, not notarized; Gatekeeper may require manual approval.

For a source build, `cmake --install build --component cli --prefix /path/to/prefix`
installs the executable using your system runtime libraries.

## Native C++ SDK archives

The [SDK guide](sdk.md) covers extracted C++ SDKs, compiler and runtime
requirements, and building archives with the native CMake install rules.

## Python source installation

The [Python guide](../python/README.md#build-from-source) gives the editable
installation command and tests. Run it from a repository checkout with the
native build dependencies above.

## Command-line use

```bash
./build/apxchol matrix.mtx --rhs rhs.mtx -o solution.mtx
./build/apxchol graph.mtx --input-kind adjacency --random-rhs --tol 1e-8
./build/apxchol --help
```

Supply `--rhs` for your actual problem or `--random-rhs` for an experiment.
An `--rhs` file with `k` columns is solved as one batch, and `-o` writes an
`n` by `k` solution. The input is read as `A`; pass `--input-kind adjacency`
to solve the Laplacian of a graph adjacency matrix. The website's command-line reference is
generated from `--help`, so it lists every option for that revision.

## Use from CMake

For a source checkout, call the helper in your consuming project's top-level
`CMakeLists.txt`, after its `project(...)` declaration:

```cmake
include(path/to/apxchol/cmake/apxchol_core.cmake)
apxchol_add_core()
target_link_libraries(my_program PRIVATE apxchol::core)
```

This builds just the library and enables any GPU language in the consuming
project's scope. Set `APXCHOL_DEVICE` when configuring the consuming project.
For distribution, set `APXCHOL_NATIVE_ARCH=OFF`; CUDA consumers must also
choose `CMAKE_CUDA_ARCHITECTURES` instead of its `native` default.

Alternatively, embed the repository root directly:

```cmake
add_subdirectory(path/to/apxchol)
target_link_libraries(my_program PRIVATE apxchol::core)
```

Root embedding defaults to the portable library only. Set `APXCHOL_BUILD_CLI`,
`APXCHOL_BUILD_TESTS` or `APXCHOL_BUILD_EXAMPLES` before `add_subdirectory` to opt
into those targets. CLI dependencies are fetched only when the CLI is enabled;
tests with the CLI disabled still include the operator-input unit tests.

For CUDA or Metal, enable `CUDA` or `OBJCXX` in the consuming parent before
`add_subdirectory`, for example with `project(my_project LANGUAGES CXX CUDA)`.
The top-level helper above enables the selected language for you.

For a separately installed SDK, use:

```cmake
find_package(apxchol CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE apxchol::core)
```

Build and install that SDK with:

```bash
cmake -S cmake/core -B build-sdk -DCMAKE_INSTALL_PREFIX=/path/to/prefix
cmake --build build-sdk -j6
cmake --install build-sdk
```

Configure the consuming project with `-DCMAKE_PREFIX_PATH=/path/to/prefix`.
If apxchol fetched Eigen, the install includes its headers. Otherwise, the
consumer must also provide Eigen. Static SDK consumers also need zlib; CUDA
SDK consumers need the CUDA Toolkit.
Use compatible compiler, standard-library and OpenMP runtime settings.

`ExternalProject_Add` builds in a separate project. Use it to build/install
`cmake/core`, then configure the consuming project against that install with
`find_package`. It does not create an immediately linkable `apxchol::core`
target in the parent configuration.

## Build options

These are all apxchol-owned CMake controls. Solver options such as tolerance,
backend selection and factor storage are runtime settings; see the
[C++ reference](reference/cpp.md) and language guides.

| Option | Default | Meaning |
|---|---|---|
| `APXCHOL_DEVICE` | `CPU` | `CPU` only, or CPU plus `CUDA` / `METAL` |
| `APXCHOL_NATIVE_ARCH` | ON for standalone root/helper/local Python; OFF for root embedding/SDK/wheels | Tune for the build machine; turn OFF for distribution |
| `APXCHOL_POOL_FP32` | ON | Store mutable graph weights in FP32; OFF uses FP64 |
| `APXCHOL_INDEX_WIDTH` | `32` | `32`, `64_EDGES` or `64`; see layouts below |
| `APXCHOL_BUILD_CLI` | ON standalone; OFF when embedded | Build the CLI and its input/logging dependencies |
| `APXCHOL_BUILD_TESTS` | ON standalone; OFF when embedded | Build the test suite |
| `APXCHOL_BUILD_EXAMPLES` | ON standalone; OFF when embedded | Build the examples |

The index setting selects one of three layouts:

| Vertex indices | Offsets | Configuration |
|---|---|---|
| 32-bit | 32-bit | `APXCHOL_INDEX_WIDTH=32` (default) |
| 32-bit | 64-bit | `APXCHOL_INDEX_WIDTH=64_EDGES` |
| 64-bit | 64-bit | `APXCHOL_INDEX_WIDTH=64` |

`64_EDGES` widens total edge counts and offsets while keeping vertex indices
at 32 bits. Offsets count stored entries, so they can need 64 bits before
vertex indices do. GPU setup requires 32-bit vertices.

Standard CMake controls also apply: `CMAKE_BUILD_TYPE`, `BUILD_SHARED_LIBS`,
`CMAKE_INSTALL_PREFIX`, and `CMAKE_CUDA_ARCHITECTURES` for CUDA builds.
`cmake -LH build` lists configured cache options.

Portable x86 builds use x86-64-v2 and dispatch supported FP16/SIMD kernels at
runtime. Native tuning can require a newer CPU than this baseline.
`apxchol::core` propagates architecture and C++23 requirements to consumers;
these matter because Eigen objects cross the library boundary.
