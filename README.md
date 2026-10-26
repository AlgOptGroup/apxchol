# apxchol

[![CI](https://github.com/AlgOptGroup/apxchol/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AlgOptGroup/apxchol/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/apxchol)](https://pypi.org/project/apxchol/)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://algoptgroup.github.io/apxchol/)

apxchol solves sparse graph-Laplacian and SDDM linear systems, `A x = b`.
It builds an approximate-Cholesky preconditioner and uses preconditioned
conjugate gradients (PCG) to compute the solution.

- Reuse one factorization for single or batched right-hand sides.
- Run on multicore CPUs, NVIDIA GPUs or Apple GPUs.
- Get convergence results verified against your original matrix and RHS.

## Quick start

This page follows the development branch. Use a
[source installation](docs/building.md#python-source-installation) for a matching build.
For the latest published package and its API, see
[PyPI](https://pypi.org/project/apxchol/).

```python
import apxchol
import numpy as np
from scipy.sparse import diags

n = 1000
# Tridiagonal matrix: 3 on the diagonal, -1 directly above and below it.
A = diags([-1.0, 3.0, -1.0], [-1, 0, 1], shape=(n, n), format="csc")
solver = apxchol.factorize(A)
result = solver.solve(np.ones(n), rtol=1e-8)
assert result.converged

# Solve several RHS columns together, reusing the same factorization.
B = np.column_stack((np.ones(n), np.arange(n)))
batch = solver.solve(B)
assert batch.converged.all()           # solutions are in batch.x
```

**No GPU is required.** Wheels built by this project include the CPU solver plus
CUDA on Linux or Metal on macOS.
GPU execution needs a compatible device and driver, but using the wheels
requires neither the CUDA Toolkit nor Homebrew.

| Wheel builds | Minimum platform | Included backends |
|---|---|---|
| CPython 3.10–3.14, Linux | x86-64-v2 CPU, glibc 2.28 | CPU + CUDA |
| CPython 3.10–3.14, macOS | macOS 15, Apple silicon | CPU + Metal |

`solver.backend` reports which backend was selected. Use `backend="cpu"`
or `backend="gpu"` in `factorize` to choose explicitly.

Pass the matrix `A` from `A x = b`. For a graph adjacency matrix, first form
its Laplacian with `apxchol.laplacian(Adj)`. See the
[solver guide](docs/usage.md) for supported matrices and convergence checks.

## Interfaces

Start with the built-in solver for factor reuse and batched solves.
The [Python](python/README.md), [C++](docs/reference/cpp.md) and
[MATLAB/Octave](octave/README.md) guides cover each interface.
Eigen and SciPy adapters let you use the preconditioner with another iterative
solver; see [integration and factor reuse](docs/usage.md#using-another-iterative-solver).

## Build and run

A source build needs CMake, GCC or Clang with C++23 and OpenMP, and Eigen.
See [building and installation](docs/building.md) for dependencies, GPU builds
and CMake integration.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j6
ctest --test-dir build --output-on-failure
./build/apxchol matrix.mtx --rhs rhs.mtx -o solution.mtx
```

The CLI also accepts adjacency matrices. `--random-rhs` generates a test RHS;
[`--help`](docs/building.md#command-line-use) lists the available controls.

## Learn more

- [Choosing a backend, reusing factors and checking results](docs/usage.md)
- [Build options](docs/building.md#build-options), [precision](docs/precision.md) and [API compatibility](docs/api.md)
- [Custom algorithms](docs/extending.md), [examples](examples/) and [contributing](CONTRIBUTING.md)
- [Benchmarking and reproducibility](benchmarks/README.md)

Based on Kyng–Sachdeva ([2016](https://arxiv.org/abs/1605.02353)),
Gao–Kyng–Spielman ([2023](https://arxiv.org/abs/2303.00709)) and
Baumann–Kyng ([2024](https://dl.acm.org/doi/10.1145/3626183.3659987)).

Developed at ETH Zürich. Contact: <apxchol@inf.ethz.ch>.
[License](LICENSE) · [Citation](CITATION.cff).
