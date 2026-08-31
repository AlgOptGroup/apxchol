# apxchol

`apxchol` solves sparse Laplacian and SDDM systems with a randomized
approximate-Cholesky preconditioner and PCG. It provides parallel CPU setup and
solve with OpenMP, plus an optional CUDA-resident solve.

```cpp
#include "apxchol.h"

auto result = apxchol::solve(L, b, {.tol = 1e-8});
```

Python and Octave/MATLAB bindings are included. Developed at ETH Zürich;
questions and use cases are welcome at <apxchol@inf.ethz.ch>.

## Quick start

The core library needs CMake, a C++23 compiler, and Eigen. CMake fetches Eigen
when it is not installed.

```bash
git clone https://github.com/AlgOptGroup/apxchol
cd apxchol
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

The CLI requires either an explicit right-hand side or `--random-rhs`:

```bash
./build/apxchol matrix.mtx --random-rhs --tol 1e-8
./build/apxchol matrix.mtx --rhs rhs.mtx -o solution.mtx
```

It detects an assembled Laplacian/SDDM operator versus a graph adjacency matrix
from which it forms `L = D - A`, and reports the decision. Use `--input-kind` to
override detection and `./build/apxchol --help` for all options.

## Library interfaces

For repeated C++ solves, factor once and reuse the workspace:

```cpp
apxchol::cpu_solver solver(L);
auto r1 = solver.solve(b1);
auto r2 = solver.solve(b2, 1e-10, 1000);
Eigen::VectorXd z = solver.apply(r);  // one preconditioner application
```

`apxchol::apx_cholesky` also implements Eigen's preconditioner interface.
Singular Laplacians are solved in their compatible subspace; full-rank SDDM
operators retain the full factor. Public headers live under
[include/apxchol](include/apxchol/), with options documented in
[factor_options.h](include/apxchol/solver/factor_options.h).

Python:

```bash
pip install apxchol
# or, from a checkout:
pip install -e python
```

```python
import apxchol

solver = apxchol.factorize(A)
result = solver.solve(b, rtol=1e-8)
```

Python expects an assembled operator; use `apxchol.laplacian(A)` for an
adjacency matrix. See [python/README.md](python/README.md).

Octave and MATLAB:

```bash
./octave/build.sh
```

```matlab
s = apxchol_solver(A);
result = s.solve(b);
```

Use `apxchol_laplacian(Adj)` for adjacency input. The MATLAB MEX build and usage
are documented in [octave/README.md](octave/README.md).

## Configuration

Common CMake options:

| option | purpose |
|---|---|
| `APXCHOL_USE_CUDA=ON` | CUDA triangular solve and GPU-resident PCG |
| `APXCHOL_POOL_FP32=ON` | fp32 residual-pool weights; default on |
| `APXCHOL_64BIT_EDGE_INDICES=ON` | widen factor and pool offsets |
| `APXCHOL_64BIT_NODE_INDICES=ON` | widen vertex ids |
| `APXCHOL_BUILD_TESTS=OFF` | skip unit tests |
| `APXCHOL_BUILD_EXAMPLES=OFF` | skip examples |

`cmake -LH build` lists every option. The default CUDA triangular solve is the
project's persistent dataflow kernel; the core CUDA library links only
`cudart`. cuSPARSE SpSV is an opt-in comparison backend through
`APXCHOL_CUDA_WITH_CUSPARSE=ON`.

Runtime controls and their numerical contracts are documented beside their
implementations. The most common are `APXCHOL_SPTRSV_FP16`,
`APXCHOL_FACTOR_DROP`, and `APXCHOL_CPU_SPTRSV`.

This research branch additionally exposes `--clique-sampler bkz26` (default:
`gks`). It is the [BKZ26 Algorithm 1 clique sampler embedded in
apxchol](experiments/2026-08-31-bkz26-prufer/README.md), not the paper's full
Algorithm 3.

## Documentation and benchmarks

- [Extending the algorithm](docs/extending.md): custom elimination rules,
  partitioners, and orderings.
- [Examples](examples/): small integration examples.
- [Benchmark protocol](benchmarks/README.md): fairness and timing definitions.
- [Laptop snapshot](benchmarks/latest/) and [CSCS Daint snapshot](benchmarks/daint/):
  machine-specific results and direct figure links.
- [Contributing](CONTRIBUTING.md), [license](LICENSE), and
  [software citation metadata](CITATION.cff). The BKZ26 manuscript entry is in
  [CITATION.bib](CITATION.bib).

## References

- Kyng and Sachdeva, *Approximate Gaussian Elimination for Laplacians*, 2016
  ([arXiv](https://arxiv.org/abs/1605.02353)).
- Gao, Kyng, and Spielman, *Robust and Practical Solution of Laplacian
  Equations by Approximate Elimination*, 2023
  ([arXiv](https://arxiv.org/abs/2303.00709)).
- Baumann and Kyng, *A Framework for Parallelizing Approximate Gaussian
  Elimination*, SPAA 2024
  ([DOI](https://dl.acm.org/doi/10.1145/3626183.3659987)).
- Baumann, Kyng, and Zöcklein, *VAC: A Volume-sampling-based Elimination Rule
  for Approximate Cholesky Factorization*, manuscript, 2026
  ([PDF](https://rasmuskyng.com/papers/BKZ26.pdf)).
