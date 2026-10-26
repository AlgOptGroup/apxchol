# Native C++ SDK archives

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
CMake package files and bundled-code licenses. The native build embeds minimal
static libzip support. Use the imported CMake target to retain the required
C++23 setting, index layout and backend definitions. SDKs use 32-bit indices and
portable tuning: x86-64-v2 on Linux, arm64 on macOS.

## Compiler and runtime requirements

Use a compatible C++23 compiler and standard library. The compiler's OpenMP
runtime family must match the SDK; CMake rejects a different family. The Linux
pilot uses GCC 14 and GNU OpenMP (`libgomp`); macOS builds use Clang, libc++ and
LLVM OpenMP (`libomp`). Install the matching OpenMP development package when
it is not supplied by the compiler. The zlib runtime is also required.

A CPU SDK does not need a GPU toolkit. A CUDA SDK includes CPU and CUDA routes
and currently requires CUDA Toolkit 13.3 or newer when compiling consumers,
including CPU-route consumers: its public headers and exported CMake target
use CUDA. Use a compatible NVIDIA driver to execute GPU routes. A Metal SDK
includes CPU and Metal routes and needs Apple's system frameworks, plus libomp.
The macOS deployment target is 15.0.

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
arguments. Each archive has a SHA256 checksum. The script creates a new staging
prefix and refuses to overwrite an existing one.
