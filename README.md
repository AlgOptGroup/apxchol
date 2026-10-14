# apxchol

Parallel approximate-Cholesky preconditioning and PCG for sparse Laplacian
and SDDM systems, with OpenMP CPU, CUDA and Apple Metal backends.
C++, Python and Octave/MATLAB interfaces are included.

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

## Configuration

| CMake option | Purpose |
|---|---|
| `APXCHOL_DEVICE=CPU` | Build for `CPU` (default), `CUDA`, or `METAL` (macOS). GPU builds also support CPU solves |
| `APXCHOL_POOL_FP32=OFF` | fp64 residual-pool weights instead of default fp32 |
| `APXCHOL_64BIT_EDGE_INDICES=ON` | Wide factor/pool offsets |
| `APXCHOL_64BIT_NODE_INDICES=ON` | Wide vertices and offsets |

`cmake -LH build` lists build options. Algorithm defaults live in
[factor_options.h](include/apxchol/solver/factor_options.h).
`APXCHOL_FACTOR_STORAGE=auto|float16|float32` controls triangular-solve factor storage (`auto`: CUDA FP16, CPU/Metal FP32).
FP16 narrows scaled off-diagonals, retaining FP32 diagonals; it
does not change the outer PCG to FP16. See [precision and storage](docs/precision.md).
`--sampler gks|trace_cycle` selects the clique sampler; GKS remains
its default. Trace-cycle supports CPU setup and full GPU-owned setup.
`solve_options.backend` (`solve_backend::automatic`, `cpu`, or `gpu`) and CLI
`--backend auto|cpu|gpu` select a complete setup/solve route. The default `auto`
uses CPU for systems of at most 64 rows when CPU supports the requested storage;
otherwise it selects an available, compatible GPU or CPU. Larger systems are
selected by compatibility, not predicted performance.
`solve_result.backend` and the CLI report the actual route.

Explicit GPU requests fail if unsupported. Once selected, GPU setup or execution
errors are reported without retrying on CPU. CUDA requires compressed, sorted,
unique and fully paired CSC input and uses device-owned factorization. It supports
warm starts and optionally downloads an FP32 factor for export. Metal uses host
factorization and supports FP32/FP16 storage; see its [precision limits](docs/precision.md#apple-metal-pcg).
Public `factorize()` and custom CPU strategies remain available independently;
see the [extension guide](docs/extending.md) for explicit factor handoff.

### Threads and placement

The CPU solver is limited by memory traffic rather than core count, so where the
threads sit matters more than how many there are. One 64-core EPYC 7742 socket
(four NUMA domains, two memory channels each), 8 matrices, both samplers,
geometric mean of the one-RHS total (Euler, 2026-09-18):

| Threads × NUMA domains | 16 × 1 | 16 × 4 | 32 × 4 | 64 × 4 |
|---|---:|---:|---:|---:|
| Relative time | 1.00 | 0.86 | **0.77** | 0.83 |

Spread the threads over the memory domains of one socket and stop near eight per
domain: all 64 cores were slower than 32, and slower than the compact 16 on
social graphs; spreading 16 threads over both sockets was slower than one
domain on three of four matrices. The library never changes affinity. Pin
explicitly, one place per chosen core (`lscpu -p=CPU,CORE,NODE` lists them):
`OMP_PROC_BIND=close OMP_PLACES="{0},{1},{16},{17},..."`.

## Further reading

- [Historical Daint results](benchmarks/daint/) and [benchmark protocol](benchmarks/README.md).
  Laptop measurements are [historical](benchmarks/archive/).
- [Examples](examples/), [extending the algorithm](docs/extending.md),
  [contributing](CONTRIBUTING.md), [implementation history](docs/implementation-history.md).
- Kyng–Sachdeva ([2016](https://arxiv.org/abs/1605.02353)),
  Gao–Kyng–Spielman ([2023](https://arxiv.org/abs/2303.00709)),
  Baumann–Kyng ([2024](https://dl.acm.org/doi/10.1145/3626183.3659987)).

Developed at ETH Zürich. Contact: <apxchol@inf.ethz.ch>.
[License](LICENSE) · [Citation](CITATION.cff).
