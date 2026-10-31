# Building from source

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
The same executable still supports CPU solves. To run it, see the
[command-line guide](cli.md); to link the library from your own project, see
[C++ and CMake](cpp.md#use-from-cmake).

## Python source installation

The [Python guide](../python/README.md#build-from-source) gives the editable
installation command and tests. Run it from a repository checkout with the
native build dependencies above.

## Build options

These are all apxchol-owned CMake controls. Solver options such as tolerance,
backend selection and factor storage are runtime settings; see the
[C++ reference](reference/cpp.md) and language guides.

| Option | Default | Meaning |
|---|---|---|
| `APXCHOL_DEVICE` | `CPU` | `CPU` only, or CPU plus `CUDA` / `METAL` |
| `APXCHOL_NATIVE_ARCH` | ON for standalone root, local Python and benchmarks; OFF for embedding, the SDK and wheels | Tune for the build machine; turn OFF for distribution. Ignored when `CMAKE_CXX_FLAGS` sets `-march`/`-mcpu` |
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
`apxchol::core` propagates C++23 and two Eigen settings to consumers, because
Eigen objects cross the library boundary: `EIGEN_MAX_ALIGN_BYTES`, the
alignment the core's flags give Eigen, and `EIGEN_MALLOC_ALREADY_ALIGNED=0`,
so both sides allocate and free through Eigen's own aligned allocator. A
consumer can then target a narrower or wider instruction set than the core.
