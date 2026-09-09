# apxchol

Parallel approximate-Cholesky preconditioning and PCG for sparse Laplacian
and SDDM systems. CPU setup and solve use OpenMP; CUDA provides an optional
GPU-resident solve. C++, Python and Octave/MATLAB interfaces are included.

## Build and run

Requires CMake, a C++23 compiler, and Eigen (fetched if absent).

```bash
git clone https://github.com/AlgOptGroup/apxchol
cd apxchol
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j6
ctest --test-dir build --output-on-failure
./build/apxchol matrix.mtx --random-rhs --tol 1e-8
./build/apxchol matrix.mtx --rhs rhs.mtx -o solution.mtx
```

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

`apxchol::apx_cholesky` provides Eigen's preconditioner interface. Singular
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
| `APXCHOL_USE_CUDA=ON` | Dataflow triangular solve and GPU PCG; core links only `cudart` |
| `APXCHOL_POOL_FP32=OFF` | fp64 residual-pool weights instead of default fp32 |
| `APXCHOL_64BIT_EDGE_INDICES=ON` | Wide factor/pool offsets |
| `APXCHOL_64BIT_NODE_INDICES=ON` | Wide vertices and offsets |

`cmake -LH build` lists build options. Algorithm defaults live in
[factor_options.h](include/apxchol/solver/factor_options.h).
`APXCHOL_SPTRSV_FP16=0|1` controls factor storage (GPU default on, CPU off).
`--sampler gks|trace_cycle|heavy_core_k2` selects the clique sampler; GKS remains
its default. The cycle alternatives support CPU setup and full GPU-owned setup.
Factor construction defaults to CPU. `APXCHOL_GPU_BLOCK_FRONTEND=on` enables
GPU selection; experimental GPU-owned numerical setup additionally requires
`APXCHOL_GPU_ROUND_SHADOW=force` and `APXCHOL_GPU_FACTOR_FINALIZE=force`.
This applies to one-shot block-greedy/tree solves with directed AoS storage;
public/exported factors and unsupported stored formats retain validated fallback
paths. See [the setup contracts](AGENTS.md#architecture-and-contracts).

## Further reading

- [Daint results](benchmarks/daint/) and [benchmark protocol](benchmarks/README.md).
  Laptop measurements are [historical](benchmarks/archive/).
- [Examples](examples/), [extending the algorithm](docs/extending.md),
  [contributing](CONTRIBUTING.md), [implementation history](docs/implementation-history.md).
- Kyng–Sachdeva ([2016](https://arxiv.org/abs/1605.02353)),
  Gao–Kyng–Spielman ([2023](https://arxiv.org/abs/2303.00709)),
  Baumann–Kyng ([2024](https://dl.acm.org/doi/10.1145/3626183.3659987)).

Developed at ETH Zürich. Contact: <apxchol@inf.ethz.ch>.
[License](LICENSE) · [Citation](CITATION.cff).
