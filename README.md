# apxchol

`apxchol` solves sparse Laplacian and SDDM systems with a randomized
approximate-Cholesky preconditioner and PCG. It supports parallel CPU setup and
solve with OpenMP, plus an optional CUDA-resident solve. A fixed seed gives a
deterministic factor at one thread.

```cpp
#include "apxchol.h"

auto result = apxchol::solve(L, b, {.tol = 1e-8});
```

Python and Octave/MATLAB bindings are included. The standalone
[benchmark suite](benchmarks/README.md) compares against BoomerAMG, AMGCL,
RCHOL/pRCHOL, ParAC, CMG, and Laplacians.jl.

Developed at ETH Zürich. Questions and use cases are welcome at
<apxchol@inf.ethz.ch>.

## Build and run

The core library needs CMake, a C++23 compiler, and Eigen. Eigen is fetched when
it is not installed.

```bash
git clone https://github.com/AlgOptGroup/apxchol
cd apxchol
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

Solve a Matrix Market operator or graph adjacency matrix:

```bash
./build/apxchol matrix.mtx --random-rhs --tol 1e-8
./build/apxchol matrix.mtx --rhs rhs.mtx -o solution.mtx
```

The CLI detects whether the input is an assembled Laplacian/SDDM operator or a
graph from which it should form `L = D - A`, and reports the decision. Use
`--input-kind` to override it. Run `./build/apxchol --help` for all options.

## C++ API

One-shot solve:

```cpp
#include "apxchol.h"

Eigen::SparseMatrix<double> L = /* Laplacian or SDDM */;
Eigen::VectorXd b = /* right-hand side */;

auto result = apxchol::solve(L, b, {.tol = 1e-8, .max_iter = 500});
// result.x, result.iterations, result.residual
```

Factor once for repeated solves:

```cpp
apxchol::cpu_solver solver(L);
auto r1 = solver.solve(b1);
auto r2 = solver.solve(b2, 1e-10, 1000);

Eigen::VectorXd x(L.rows());
auto r3 = solver.solve(b3, x);  // caller-owned output; workspace is reused
Eigen::VectorXd z = solver.apply(r);  // one M^-1 application
```

The preconditioner also implements Eigen's interface:

```cpp
Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower,
                         apxchol::apx_cholesky> cg;
cg.compute(L);
Eigen::VectorXd x = cg.solve(b);
```

For allocation-free repeated solves, prefer `cpu_solver`; Eigen's CG allocates
its own temporaries.

Singular Laplacians are solved in their rank-`n-1` subspace; full-rank SDDM
operators retain the full factor. The default setup uses block-greedy
independent-set selection and the pooled AoS adjacency backend. Factorization
and solve options are in
[factor_options.h](include/apxchol/solver/factor_options.h) and the public
headers under [include/apxchol](include/apxchol/).

## Python

```bash
pip install apxchol
# From a checkout:
pip install -e python
```

```python
import apxchol
import scipy.sparse as sp
from scipy.io import mmread

A = sp.csc_matrix(mmread("operator.mtx"))
solver = apxchol.factorize(A)
result = solver.solve(b, rtol=1e-8)
z = solver.apply(b)
P, L, D = solver.P, solver.L, solver.D
```

Python expects an assembled operator. Convert an adjacency matrix explicitly
with `apxchol.laplacian(A)`. See [python/README.md](python/README.md).

## Octave and MATLAB

```bash
./octave/build.sh
```

```matlab
addpath('octave');
s = apxchol_solver(A);
result = s.solve(b);
z = s.apply(b);
```

The same source builds with MATLAB `mex`; see
[octave/README.md](octave/README.md). Inputs are assembled operators. Use
`apxchol_laplacian(Adj)` for an adjacency matrix.

## Configuration

Common CMake options are:

| option | purpose |
|---|---|
| `APXCHOL_USE_CUDA=ON` | CUDA triangular solve and GPU-resident PCG |
| `APXCHOL_POOL_FP32=ON` | fp32 residual-pool weights; default on |
| `APXCHOL_64BIT_EDGE_INDICES=ON` | factors or pools exceeding 32-bit edge offsets |
| `APXCHOL_64BIT_NODE_INDICES=ON` | graphs exceeding 32-bit vertex ids |
| `APXCHOL_BUILD_TESTS=OFF` | skip unit tests |
| `APXCHOL_BUILD_EXAMPLES=OFF` | skip examples |

`cmake -LH build` shows the complete list and help strings. Release builds use
native architecture tuning.

The default CUDA triangular solve is the project's persistent dataflow kernel;
the core CUDA library links only `cudart`. cuSPARSE SpSV is an opt-in comparison
backend through `APXCHOL_CUDA_WITH_CUSPARSE=ON`.

Runtime storage controls include:

- `APXCHOL_SPTRSV_FP16=0|1`: scaled fp16 off-diagonal factor storage (default on
  for GPU, off for CPU);
- `APXCHOL_FACTOR_DROP=0|<relative threshold>`: compensated column-relative
  factor dropping;
- `APXCHOL_CPU_SPTRSV=auto|levels`: hybrid CPU DAG schedule or pure level sets.

The implementation headers document their numerical contracts and rollback
switches.

## Extending the algorithm

Custom elimination rules, partitioners, and orderings plug into the same factor
and PCG machinery. The contracts and worked examples are in
[docs/extending.md](docs/extending.md) and [examples](examples/).

## Benchmarks

The benchmark project has its own build, dependencies, result store, and
fairness checks:

- [benchmark protocol and headline figures](benchmarks/README.md)
- [current laptop results](benchmarks/latest/)
- [CSCS Daint results](benchmarks/daint/)

## References

- Kyng and Sachdeva, *Approximate Gaussian Elimination for Laplacians*, 2016
  ([arXiv](https://arxiv.org/abs/1605.02353)).
- Gao, Kyng, and Spielman, *Robust and Practical Solution of Laplacian
  Equations by Approximate Elimination*, 2023
  ([arXiv](https://arxiv.org/abs/2303.00709)).
- Baumann and Kyng, *A Framework for Parallelizing Approximate Gaussian
  Elimination*, SPAA 2024
  ([DOI](https://dl.acm.org/doi/10.1145/3626183.3659987)).

Additional competitor references and versions are recorded in the benchmark
result metadata.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md). The project uses the
[BSD 4-Clause license](LICENSE). Cite releases through
[CITATION.cff](CITATION.cff).
