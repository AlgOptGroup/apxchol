# apxchol

Parallel approximate-Cholesky preconditioning and PCG for sparse Laplacian
and SDDM systems. CPU setup and solve use OpenMP; CUDA provides an optional
device-owned setup and GPU-resident solve. C++, Python and Octave/MATLAB interfaces are included.

## Build and run

Requires CMake, GCC or Clang with C++23 and OpenMP, and Eigen (fetched if absent).
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

apxchol::cpu_solver solver(L);       // factor once, solve many
auto r1 = solver.solve(b1);
auto r2 = solver.solve(b2, 1e-10, 1000);
Eigen::VectorXd z = solver.apply(r);
```

`cpu_solver` and Eigen's `apxchol::apx_cholesky` always use CPU setup and
CPU solves, including in CUDA builds. Singular
Laplacians use their compatible subspace; SDDM operators retain a full factor.

```bash
pip install apxchol                 # or: pip install -e python
```

```python
solver = apxchol.factorize(A)
result = solver.solve(b, rtol=1e-8)
```

Bindings expect assembled operators. Use `apxchol.laplacian(Adj)` in Python
or `apxchol_laplacian(Adj)` in Octave for adjacency input. See the
[Python](python/README.md) and [Octave/MATLAB](octave/README.md) guides.

## Configuration

| CMake option | Purpose |
|---|---|
| `APXCHOL_USE_CUDA=ON` | Device-owned setup, dataflow triangular solves and GPU PCG; core links only `cudart` |
| `APXCHOL_POOL_FP32=OFF` | fp64 residual-pool weights instead of default fp32 |
| `APXCHOL_64BIT_EDGE_INDICES=ON` | Wide factor/pool offsets |
| `APXCHOL_64BIT_NODE_INDICES=ON` | Wide vertices and offsets |

`cmake -LH build` lists build options. Algorithm defaults live in
[factor_options.h](include/apxchol/solver/factor_options.h).
`APXCHOL_SPTRSV_STORAGE=auto|float16|float32` controls triangular-solve factor storage (`auto`: GPU FP16, CPU FP32). The legacy `APXCHOL_SPTRSV_FP16=0|1` is accepted when the named setting is unset.
FP16 narrows scaled off-diagonals, retaining FP32 diagonals; it
does not change the outer PCG to FP16. See [precision and storage](docs/precision.md).
`--sampler gks|trace_cycle` selects the clique sampler; GKS remains
its default. Trace-cycle supports CPU setup and full GPU-owned setup.
`solve_options.backend` (`solve_backend::automatic`, `cpu`, or `gpu`) and CLI
`--backend auto|cpu|gpu` select a complete setup/solve route. The default `auto`
selects GPU in a CUDA build for supported block-greedy/AoS options with 32-bit
nodes and no factor export; other configurations select CPU. This is an option
compatibility policy, not a speed predictor or device-availability probe.
`solve_result.backend` and the CLI report the actual route.

The GPU route requires device-owned factor and operator construction. Unsupported
stored formats, unavailable devices, allocation failures and execution errors are
reported without retrying on CPU. Use `--backend cpu` (or `solve_backend::cpu`)
for CPU execution in a CUDA build, including lower-only or otherwise unsupported
CSC layouts. GPU operator input must be compressed, sorted, unique and fully
paired. Ordinary GPU solves no longer need the three per-stage environment flags;
those flags remain low-level diagnostic controls. Public factorization, custom
strategies and factor export stay available on CPU. See the
[extension guide](docs/extending.md) for explicit factor handoff.

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
