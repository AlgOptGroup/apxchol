# C++ and CMake

Include `apxchol.h` and link the `apxchol::core` CMake target:

```cpp
#include "apxchol.h"

apxchol::solver solver(A);            // A: Eigen::SparseMatrix<double>
auto result = solver.solve(b, 1e-8);  // check result.converged
```

The [basic example](examples.md#basic-solves) shows one-shot, reusable and
batched solves. [Using the solver](usage.md) covers the solver itself, and the
[C++ reference](reference/cpp.md) lists every option.

Add the library to a CMake project in one of three ways, described below:
build it from a source checkout, embed the repository root, or use an
installed SDK such as a prebuilt [SDK archive](sdk.md).

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

