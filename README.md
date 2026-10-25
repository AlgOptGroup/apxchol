# apxchol

Parallel approximate-Cholesky preconditioning and PCG for sparse Laplacian
and SDDM systems, with OpenMP CPU, CUDA and Apple Metal backends.

- Factor once and reuse it for individual or batched right-hand sides.
- Use the built-in solver or apply the preconditioner through Eigen or SciPy.
- Check convergence against the original operator and right-hand side.

## Interfaces

```bash
pip install apxchol                 # or: pip install -e python
```

```python
import apxchol
import numpy as np
from scipy.sparse import diags

n = 1000
A = diags([-np.ones(n-1), 3*np.ones(n), -np.ones(n-1)],
          [-1, 0, 1], format="csc")   # assembled SDDM operator
solver = apxchol.factorize(A)
result = solver.solve(np.ones(n), rtol=1e-8)
assert result.converged

B = np.column_stack((np.ones(n), np.arange(n)))
batch = solver.solve(B)               # reuse the factor; solutions in batch.x
assert batch.converged.all()
```

Wheels support CPython 3.10–3.14 on Linux x86-64-v2 (CUDA) and macOS 15+
arm64 (Metal). Both retain CPU support. GPU use requires a compatible device
and driver, but no CUDA Toolkit or Homebrew installation.
`solver.backend` reports the selected backend.

Bindings expect assembled operators. Use `apxchol.laplacian(Adj)` in Python
or `apxchol_laplacian(Adj)` in Octave for adjacency input. A Laplacian RHS
must sum to zero on each connected component. See the [Python](python/README.md)
and [Octave/MATLAB](octave/README.md) guides for options and factor export.

The C++ interface uses the same reusable owner:

```cpp
#include "apxchol.h"
apxchol::solver solver(A);
auto result = solver.solve(b, 1e-8);
Eigen::MatrixXd X(B.rows(), B.cols());
auto batch = solver.solve(B, X);
Eigen::VectorXd z = solver.apply(b);
```

One-shot `apxchol::solve(A, b)` selects the same backend. `cpu_solver` and
Eigen's `apxchol::apx_cholesky` always use CPU. The [examples](examples/)
cover reuse and custom algorithms; [factor handoff](docs/extending.md#reusing-a-custom-factor)
reuses a host factor on CPU, CUDA or Metal.

## Build and run

Requires CMake 3.20+, GCC or Clang with C++23 and OpenMP, and Eigen (fetched if absent).
MSVC and clang-cl are not supported. The x86-64 baseline is x86-64-v2.
CUDA builds require CUDA Toolkit 13.3+ and CMake 4.4+ for C++23 support.
On macOS, install Xcode Command Line Tools and `brew install cmake libomp`.

```bash
git clone https://github.com/AlgOptGroup/apxchol
cd apxchol
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j6
ctest --test-dir build --output-on-failure
./build/apxchol matrix.mtx --random-rhs --tol 1e-8
./build/apxchol matrix.mtx --rhs rhs.mtx -o solution.mtx
```

The CLI requires an explicit RHS or `--random-rhs`. It detects assembled
operators versus adjacency matrices, forming `L = D - A` for the latter.
`--input-kind` overrides detection; `--help` lists options.

### Use from CMake

Build the core in your project and link its target:

```cmake
include(path/to/apxchol/cmake/apxchol_core.cmake)
apxchol_add_core()
target_link_libraries(my_program PRIVATE apxchol::core)
```

For an installed library, replace the first two lines with
`find_package(apxchol CONFIG REQUIRED)`. A standalone library build uses
`cmake -S cmake/core -B build`, followed by `cmake --build build` and
`cmake --install build --prefix /path/to/prefix`.
If Eigen was fetched during configuration, its headers are installed with the
library; otherwise consumers need the Eigen package used for the build.

The standalone SDK defaults to portable code; root and local Python builds
default to native tuning. Set `APXCHOL_NATIVE_ARCH=OFF` for distribution.
The target supplies C++23 and matching architecture flags, which matter because
Eigen objects cross the library boundary.

## Configuration

| CMake option | Purpose |
|---|---|
| `APXCHOL_DEVICE=CPU` | Build for `CPU` (default), `CUDA`, or `METAL` (macOS). GPU builds also support CPU solves |
| `APXCHOL_POOL_FP32=OFF` | fp64 residual-pool weights instead of default fp32 |
| `APXCHOL_64BIT_EDGE_INDICES=ON` | Wide factor/pool offsets |
| `APXCHOL_64BIT_NODE_INDICES=ON` | Wide vertices and offsets |

`cmake -LH build` lists build options; [factor_options.h](include/apxchol/solver/factor_options.h)
defines algorithm defaults. `factor_storage` selects triangular-solve storage:
`automatic` uses CUDA FP16 and CPU/Metal FP32; explicit `fp16` and `fp32` are
available. FP16 narrows scaled off-diagonals while retaining FP32 diagonals;
outer PCG does not use FP16. See [precision and storage](docs/precision.md).
`--sampler gks|trace_cycle` selects the sampler; GKS is the default.

### Backend capabilities

GPU setup requires compressed, sorted, unique, fully paired CSC input and 32-bit vertices.
Both GPU backends support batched solves, warm starts and optional FP32 factor export.

| Backend | Factor construction |
|---|---|
| CPU | OpenMP; built-in and custom partitioners/eliminators |
| CUDA | Device-owned, `block_greedy`, natural residual peeling, GKS or trace-cycle; exact-clique/core and double-cycle overrides require CPU setup |
| Metal | Device-owned, default `block_greedy` GKS options, FP32 graph weights; [precision limits](docs/precision.md#apple-metal-pcg) |

Select a route with `solve_options.backend`, Python `backend`, or CLI
`--backend auto|cpu|gpu`. `auto` uses CPU for scalar systems and for Metal inputs
of at most 32,768 rows when CPU supports the requested storage, then checks
device availability and compatibility. Explicit GPU requests fail if unsupported;
execution errors never trigger a CPU retry. Results report the selected backend.

### Threads and placement

Set `solve_options.num_threads`, Python `num_threads`, or CLI `--threads` to
bound CPU parallelism. A positive value overrides the caller's OpenMP setting;
zero inherits it, normally from `OMP_NUM_THREADS`. Each operation restores the
setting. Small workloads may use fewer threads. More threads need not be faster:
measure both setup and solve.

The library does not change affinity. Compare `OMP_PLACES=cores` and
`OMP_PROC_BIND=spread` with your system default; restrict the CPU set when
testing one socket or NUMA node.

## Further reading

- [Benchmark protocol and reproduction](benchmarks/README.md).
- [Examples](examples/), [extending the algorithm](docs/extending.md), and [contributing](CONTRIBUTING.md).
- Kyng–Sachdeva ([2016](https://arxiv.org/abs/1605.02353)),
  Gao–Kyng–Spielman ([2023](https://arxiv.org/abs/2303.00709)),
  Baumann–Kyng ([2024](https://dl.acm.org/doi/10.1145/3626183.3659987)).

Developed at ETH Zürich. Contact: <apxchol@inf.ethz.ch>.
[License](LICENSE) · [Citation](CITATION.cff).
