# Native C++ SDK archives

Download `sdk-*` artifacts from a successful [CI run](https://github.com/AlgOptGroup/apxchol/actions/workflows/ci.yml);
[releases](https://github.com/AlgOptGroup/apxchol/releases) attach the archives and
checksums. Manual publishing-workflow runs produce CI artifacts without
uploading to PyPI or a release.

Extract an SDK archive and point CMake at its directory:

```cmake
find_package(apxchol CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE apxchol::core)
```

```sh
cmake -S my-program -B build -DCMAKE_PREFIX_PATH=/path/to/extracted-sdk
cmake --build build
```

The archive contains the shared native library, public headers, Eigen headers,
CMake package files and bundled-code licenses. Use the imported CMake target to retain the required
C++23 setting, index layout and backend definitions. SDKs use 32-bit indices and
portable tuning: x86-64-v2 on Linux, arm64 on macOS.

## Compiler and runtime requirements

Use a compatible C++23 compiler and standard library. The compiler's OpenMP
runtime family must match the SDK; CMake rejects a different family. Linux archives target glibc 2.28+ and use GCC 14 with GNU OpenMP (`libgomp`).
macOS builds use Clang, libc++ and arm64 LLVM OpenMP (`libomp`). Install the matching OpenMP development package when
it is not supplied by the compiler.

A CPU SDK does not need a GPU toolkit. A CUDA SDK includes CPU and CUDA routes
and currently requires CUDA Toolkit 13.3 or newer when compiling consumers,
including CPU-route consumers: its public headers and exported CMake target
use CUDA. Use a compatible NVIDIA driver to execute GPU routes. A Metal SDK
includes CPU and Metal routes and needs Apple's system frameworks, plus libomp.
The macOS deployment target is 15.0. The release set has Linux CPU, Linux
CPU+CUDA and macOS CPU+Metal archives; each uses one shared-library and
index-layout configuration. Use the headers and library from the same archive.
The library's runtime name includes the major and minor version: the Linux
soname (for example, `libapxchol_core.so.0.3`) and the macOS install name (for
example, `libapxchol_core.0.3.dylib`). A program therefore never loads a
library from a different minor release.

Keep the shared library accessible at runtime when installing your program.
CMake handles the build-tree library path; set your application's installation
RPATH or install the SDK library into its normal library search path.

## Building an archive

From the repository root, use the existing native install rules:

```sh
bash .github/scripts/build-sdk.sh CPU
```

Use `CUDA` on Linux or `METAL` on macOS for the GPU-enabled variants. CUDA uses
`all-major`, matching the wheel build; additional arguments are ordinary CMake
arguments. The CUDA archive producer uses the same CUDA 13.4 RPM packages and
license paths as the Linux wheel build. Each archive has a SHA256 checksum.
The script creates a new staging prefix and refuses to overwrite an existing one.
