# apxchol

Parallel approximate-Cholesky preconditioning and PCG for sparse Laplacian
and SDDM systems, with OpenMP CPU, CUDA and Apple Metal backends.
C++, C, Python and Octave/MATLAB interfaces are included.

## Build and run

Requires CMake 3.20+, GCC or Clang with C++23 and OpenMP, and Eigen (fetched if absent).
MSVC and clang-cl are not supported.
On x86-64, the minimum CPU requirement is x86-64-v2.
Optional CUDA builds require CUDA Toolkit 13.3+ and CMake 4.4+ for C++23 support.

```bash
git clone https://github.com/AlgOptGroup/apxchol
cd apxchol
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j6
ctest --test-dir build --output-on-failure
./build/apxchol matrix.mtx --random-rhs --tol 1e-8
./build/apxchol matrix.mtx --rhs rhs.mtx -o solution.mtx
```

On macOS, install Xcode Command Line Tools and `brew install cmake libomp`,
then use the same build commands above.

The CLI requires an explicit RHS or `--random-rhs`. It reports whether input
is an assembled operator or adjacency matrix, forming `L = D - A` for the
latter. `--input-kind` overrides detection; `--help` lists options.

## Interfaces

```cpp
#include "apxchol.h"
auto result = apxchol::solve(L, b, {.tol = 1e-8});

apxchol::solver solver(L);           // factor once, solve many
auto r1 = solver.solve(b1);
auto r2 = solver.solve(b2, 1e-10, 1000);
Eigen::VectorXd z = solver.apply(r);
```

`solver` selects the same backend as one-shot `solve()` and owns its factor,
operator and workspace. `cpu_solver` and Eigen's `apxchol::apx_cholesky` always
use CPU setup and solves, including in CUDA builds. Singular
Laplacians use their compatible subspace; SDDM operators retain a full factor.

The `apxchol_c` target provides a [C API](include/apxchol/c_api.h) with reusable
solver and factor handles, FP32 factor export, and status codes instead of exceptions.
It supports CPU and Metal; CUDA builds expose CPU execution through this API.
The block call solves RHS columns sequentially with the same factor and workspace.
See the header for options, ownership and error handling.

```bash
pip install apxchol                 # or: pip install -e python
```

```python
solver = apxchol.factorize(A, backend="auto")
result = solver.solve(b, rtol=1e-8)
```

Linux wheels include CUDA; macOS arm64 wheels include Metal. Both retain CPU
support. GPU use requires a compatible device and driver, but no CUDA Toolkit
installation. `solver.backend` reports the selected backend.

Bindings expect assembled operators. Use `apxchol.laplacian(Adj)` in Python
or `apxchol_laplacian(Adj)` in Octave for adjacency input. See the
[Python](python/README.md) and [Octave/MATLAB](octave/README.md) guides.

### Use from CMake

Link `apxchol_core` rather than copying its compiler flags:

```cmake
add_subdirectory(path/to/apxchol)
target_link_libraries(my_program PRIVATE apxchol_core)
```

The target supplies C++23 and the architecture flag used by the library.
Matching architecture settings matter because Eigen objects cross the library
boundary. Use `-DAPXCHOL_NATIVE_ARCH=OFF` for portable builds.

## Configuration

| CMake option | Purpose |
|---|---|
| `APXCHOL_DEVICE=CPU` | Build for `CPU` (default), `CUDA`, or `METAL` (macOS). GPU builds also support CPU solves |
| `APXCHOL_POOL_FP32=OFF` | fp64 residual-pool weights instead of default fp32 |
| `APXCHOL_64BIT_EDGE_INDICES=ON` | Wide factor/pool offsets |
| `APXCHOL_64BIT_NODE_INDICES=ON` | Wide vertices and offsets |

`cmake -LH build` lists build options. Algorithm defaults live in
[factor_options.h](include/apxchol/solver/factor_options.h).
`factor_options::factor_storage` selects triangular-solve storage (`automatic`, `fp16`, or `fp32`); the default is CUDA FP16 and CPU/Metal FP32.
FP16 narrows scaled off-diagonals, retaining FP32 diagonals; it
does not change the outer PCG to FP16. See [precision and storage](docs/precision.md).
`--sampler gks|trace_cycle` selects the clique sampler; GKS remains
its default. Trace-cycle supports CPU and CUDA setup.
`solve_options.backend` (`solve_backend::automatic`, `cpu`, or `gpu`) and CLI
`--backend auto|cpu|gpu` select a complete setup/solve route. The default `auto`
uses CPU for scalar systems and for Metal inputs of at most 32,768 rows when CPU supports the requested storage. Otherwise selection follows device availability, input and option compatibility.
`solve_result.backend` and the CLI report the actual route.

### Backend capabilities

CUDA uses device-owned factorization with 32-bit vertices, `block_greedy` selection and natural residual peeling. It requires compressed, sorted, unique, fully paired CSC input and supports both GKS and trace-cycle, warm starts and FP32 factor export. Exact-clique, exact-core and double-cycle overrides require CPU setup.

Metal uses device-owned factorization for default `block_greedy` GKS options with FP32 graph weights and 32-bit vertices. It requires the same CSC layout as CUDA and supports FP32/FP16 solve storage, warm starts and optional FP32 factor export. Explicit host-factor adoption remains available through `metal_solver`; see its [precision limits](docs/precision.md#apple-metal-pcg).

Automatic selection uses CPU when a configuration is incompatible with the available GPU backend. Explicit GPU requests fail if unsupported. Once selected, GPU setup or execution errors are reported without retrying on CPU. Public `factorize()` and custom CPU strategies remain available independently; see the [extension guide](docs/extending.md) for explicit factor handoff.

### Threads and placement

Set `solve_options.num_threads`, Python `num_threads`, or CLI `--threads` to bound CPU parallelism for one solver. A positive value overrides the calling OpenMP task's setting; zero inherits it, normally from `OMP_NUM_THREADS`. Small workloads may use fewer threads; each operation restores the caller's setting. More threads are not always faster: memory bandwidth, NUMA placement and matrix structure affect scaling. Benchmark the intended workload on the target machine, including both setup and solve.

The library does not change affinity. `OMP_PLACES=cores` and `OMP_PROC_BIND=spread` provide an explicit placement to compare with your system default; use a restricted CPU set when testing one socket or NUMA node.

## Further reading

- [Benchmark protocol and reproduction](benchmarks/README.md).
- [Examples](examples/), [extending the algorithm](docs/extending.md), and [contributing](CONTRIBUTING.md).
- Kyng–Sachdeva ([2016](https://arxiv.org/abs/1605.02353)),
  Gao–Kyng–Spielman ([2023](https://arxiv.org/abs/2303.00709)),
  Baumann–Kyng ([2024](https://dl.acm.org/doi/10.1145/3626183.3659987)).

Developed at ETH Zürich. Contact: <apxchol@inf.ethz.ch>.
[License](LICENSE) · [Citation](CITATION.cff).
