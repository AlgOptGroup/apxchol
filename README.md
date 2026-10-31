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

Until the next release, `pip` installs 0.2.2, which solves one right-hand
side per call and has no `as_operator`. To run this example, install the
development version [from source](python/README.md#build-from-source).

<!-- quickstart -->
```bash
python -m pip install "apxchol[scipy]"
```

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
<!-- /quickstart -->

Wheels run on CPU without a GPU and include CUDA (Linux) or Metal (macOS).
See [installation](docs/install.md) for the latest release, CLI archives and
source builds.

`solver.backend` reports which backend was selected. Use `backend="cpu"`
or `backend="gpu"` in `factorize` to choose explicitly.

Pass the matrix `A` from `A x = b`. For a graph adjacency matrix, first form
its Laplacian with `apxchol.as_operator(Adj)`. See the
[solver guide](docs/usage.md) for supported matrices and convergence checks.

## Interfaces

Start with the built-in solver for factor reuse and batched solves.
The [Python](python/README.md), [C++](docs/reference/cpp.md) and
[MATLAB/Octave](octave/README.md) guides cover each interface.
`solver.apply` (C++) and `aslinearoperator()` (Python) let you use the
preconditioner with another iterative solver; see [integration and factor reuse](docs/usage.md#using-another-iterative-solver).

## Build and run

Standalone [CLI archives](docs/install.md#cli-archives) for Linux CPU and
Apple-silicon CPU/Metal need no Python.

A source build needs CMake, GCC or Clang with C++23 and OpenMP, and zlib.
See [building from source](docs/building.md) for dependencies, GPU builds
and CMake integration.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j6
ctest --test-dir build --output-on-failure
./build/apxchol matrix.mtx --rhs rhs.mtx -o solution.mtx
```

The CLI reads the matrix as `A`; `--convert-input` forms the Laplacian of an
adjacency matrix. An `--rhs` file with several columns is solved as one batch.
`--random-rhs` generates a test RHS;
the [command-line guide](docs/cli.md) lists the available controls.

## Performance

Cross-solver results are in preparation for 1.0; see [performance](docs/performance.md)
for the methodology and the measured [batching case study](docs/batching.md).

## Learn more

- [Installation](docs/install.md) and [building from source](docs/building.md)
- [Choosing a backend, reusing factors and checking results](docs/usage.md)
- [Precision](docs/precision.md) and [API compatibility](docs/api.md)
- [Examples](docs/examples.md), [custom algorithms](docs/extending.md) and [contributing](CONTRIBUTING.md)
- [Benchmarking and reproducibility](benchmarks/README.md)

Based on Kyng–Sachdeva ([2016](https://arxiv.org/abs/1605.02353)),
Gao–Kyng–Spielman ([2023](https://arxiv.org/abs/2303.00709)) and
Baumann–Kyng ([2024](https://dl.acm.org/doi/10.1145/3626183.3659987)).

Developed at ETH Zürich. Contact: <apxchol@inf.ethz.ch>.
[License](LICENSE) · [Citing apxchol](docs/citing.md) · [Changelog](CHANGELOG.md).

## Acknowledgements

Samuel Talkington contributed the Apple Metal backend and macOS support.
