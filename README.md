# apxchol — Scalable Approximate Cholesky for Laplacian/SDDM Systems

C++23 library implementing a parallel approximate-Cholesky preconditioner
(Kyng–Sachdeva-style elimination with tree-based clique sampling) for graph
Laplacian and SDDM linear systems, with an Eigen-compatible interface, an
OpenMP or CUDA triangular-solve backend, a CLI solver, Python and
Octave/MATLAB bindings, and a standalone benchmark suite comparing against
Hypre BoomerAMG, AMGCL, RCHOL/pRCHOL, ParAC, CMG, and Laplacians.jl.

The library is the header tree under `include/apxchol/` (namespace
`apxchol::`) plus two compiled translation units in `src/` (CUDA builds add
three device TUs); `python/` and
`octave/` are self-contained bindings, `examples/` demonstrates the public
customization seams, and `benchmarks/` is a standalone comparison suite with
its own [README](benchmarks/README.md) and committed
[charts](benchmarks/latest/).

## Quick start

```bash
git clone https://github.com/AlgOptGroup/apxchol
cd apxchol

# Core library + CLI + tests (needs GCC >= 14 or a comparable C++23 compiler;
# Eigen is fetched automatically if not installed)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure

# Fetch the SuiteSparse test matrices into data/matrices/ (gitignored).
# Downloads ~3.3 GB total (com-Orkut alone is 1.7 GB and additionally needs
# -DAPXCHOL_64BIT_EDGE_INDICES=ON); only ecology1 is needed for the smoke
# test below.
./scripts/download_graphs.sh

# Solve a Matrix Market Laplacian/SDDM system against a generated random RHS
./build/apxchol data/matrices/ecology1.mtx --random-rhs --tol 1e-8
# ... or bring your own right-hand side:
./build/apxchol data/matrices/ecology1.mtx --rhs your_rhs.mtx  # MatrixMarket vector of length n
```

Useful CLI knobs: `--tol`, `--maxiter`, `--is {block_greedy|luby|baumann_kyng|rootset}`,
`--graph-storage {vec|forward_star|bstr|vec_pool}`,
`-o solution.mtx`, `--seed`. See `--help` for the full list.

## Library API

The supported C++ consumption path is `add_subdirectory()` / FetchContent
on this repository, linking the `apxchol_core` target (there are no
`install()` rules yet).

One-shot solve (PCG with the approximate-Cholesky preconditioner):

```cpp
#include "apxchol.h"

Eigen::SparseMatrix<double> L = ...;   // Laplacian (singular) or SDDM (SPD)
Eigen::VectorXd b = ...;
auto res = apxchol::solve(L, b, {.tol = 1e-8, .max_iter = 500});
// res.x, res.iterations, res.residual
```

Repeated solves against one matrix — factor once with `cpu_solver`, then
solve any number of right-hand sides (the recommended path; per-solve
allocations can be avoided entirely by handing the solver your output
memory):

```cpp
apxchol::cpu_solver slv(L);              // factorize + build the PCG operator once
auto r1 = slv.solve(b1);                 // tol/max_iter from solve_options
auto r2 = slv.solve(b2, 1e-10, 1000);    // per-call override; optional x0 warm start
Eigen::VectorXd z = slv.apply(r);        // one preconditioner application M^{-1} r

Eigen::VectorXd x(n);                    // caller-owned output memory:
auto r3 = slv.solve(b3, x);              // solution written into x, no per-solve
                                         // allocation (internal workspace reused)
```

Or plug the preconditioner into Eigen's CG (template parameters: the matrix
type, which triangular part of it CG reads — `Eigen::Lower` here — and the
preconditioner type):

```cpp
Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower,
                         apxchol::apx_cholesky> cg;
cg.compute(L);
Eigen::VectorXd x = cg.solve(b);
```

Note Eigen's CG allocates its own temporaries per solve and offers no
user-provided-memory hook — for allocation-free repeated solves use
`cpu_solver` above, which also runs a faster (parallel-SpMV) PCG loop.

Laplacian vs SDDM is auto-detected: singular Laplacians get a rank-(n−1)
factor with null-space centering; SDDM systems get the full-rank factor.
Both `apxchol::solve` and `apx_cholesky` default to the `vec_pool` storage
backend and the `block_greedy` independent-set partitioner — the best
default across the benchmark suite (the headline series picks the best of
four selectors per matrix; `luby`/`rootset` win some GPU cells); reconfigure
via `solve_options` / `set_storage` / `set_options`.

The factorization is randomized (`--seed` / `factor_options::seed`). The
random draws are schedule-independent: the factor is bit-identical for a
fixed seed at one thread; at T>1 merged edge weights can differ in their
final ulps (accumulation order).

With `-DAPXCHOL_USE_CUDA=ON` the triangular solves run on the GPU — by
default through our sync-free dataflow kernel (one persistent launch per
sweep, O(n) state, bit-deterministic; `APXCHOL_GPU_SPTRSV=levelset` selects
our level-set kernels instead) — and the one-shot `apxchol::solve` uses a
fully GPU-resident PCG (our own CSR SpMV and fused vector kernels with
deterministic reductions; nothing but three scalars per iteration crosses
the bus). The CUDA build links `cudart` only — no cuSPARSE, no cuBLAS;
`-DAPXCHOL_CUDA_WITH_CUSPARSE=ON` opts back into the cuSPARSE SpSV
triangular-solve backend (`APXCHOL_GPU_SPTRSV=cusparse`) as a comparison
baseline. The GPU-resident loop applies to the one-shot `apxchol::solve`
only; `cpu_solver` on a CUDA build runs the host PCG with GPU triangular
solves.

### Customizing the solver

Three research seams are public, each with a worked example under
[examples/](examples/):

- **Star-vertex elimination rule** — how the Schur-complement clique is
  sparsified when a vertex is eliminated. Implement the `eliminator` concept
  ([elimination.h](include/apxchol/solver/elimination/elimination.h)) and pass
  an instance to `factorize`:
  [examples/custom_eliminator.cpp](examples/custom_eliminator.cpp).
- **Independent-set selection and elimination order** — which vertices are
  eliminated each round, and in what order (the permutation *is* the selection
  order). Implement the `partitioner` concept
  ([partitioner.h](include/apxchol/solver/partitioner.h)); stateful instances
  can be passed directly:
  [examples/custom_order.cpp](examples/custom_order.cpp). Built-in
  partitioners are runtime-selectable by name
  (`factor_options::is_select`); custom ones can be added to
  `partitioner_list` for name dispatch.
- **External factorizations in the solver** — a factorization produced by any
  of the above plugs back into the full PCG machinery via
  `apx_cholesky::set_factor(F)` or `cpu_solver(L, F)`.

A step-by-step guide to writing your own eliminator / partitioner /
ordering — contracts, invariants, and how the pieces plug back into the
solver — is in [docs/extending.md](docs/extending.md). The knobs of the
default pipeline (degree quantile, exact-clique
threshold, residual-peel order, …) live in
[factor_options.h](include/apxchol/solver/factor_options.h), each with its
empirical rationale.

## Python package (CPU)

```bash
pip install apxchol        # wheels from PyPI (Linux x86_64, Python 3.10-3.14)
# or build from a checkout: compiles the C++ extension from ../src, installs
# it editable so python/ changes are picked up without reinstalling:
pip install -e python
```

```python
import numpy as np
import scipy.sparse as sp
from scipy.io import mmread
import apxchol

A = sp.csc_matrix(mmread("laplacian.mtx"))   # any scipy sparse Laplacian/SDDM
rng = np.random.default_rng(0)
b = rng.standard_normal(A.shape[0]); b -= b.mean()  # zero-mean: orthogonal to the Laplacian null space

solver = apxchol.factorize(A)        # factor built once, reusable
res = solver.solve(b, rtol=1e-8)     # tuned PCG; res.x, res.iters, res.residual
x = np.empty_like(b)
res = solver.solve(b, out=x)         # ... writing into caller-owned memory x
z = solver.apply(b)                  # one preconditioner application M^{-1} b
P, L, D = solver.P, solver.L, solver.D   # factor export (P[v] = elimination position)
p = np.argsort(P)                        # A[p][:, p] ≈ L @ scipy.sparse.diags(D) @ L.T

# Interop: use apxchol as the M= preconditioner of scipy's Krylov solvers
# (slower than solver.solve — scipy's Python-level CG loop — but composes
# with scipy's callbacks and other methods like MINRES):
M = solver.aspreconditioner()
```

See [python/README.md](python/README.md) for the full API.

## Octave / MATLAB package (CPU)

```bash
./octave/build.sh    # mkoctfile MEX; no MATLAB license needed
```

```matlab
addpath('octave');
s = apxchol_solver(A);            % sparse A; factor built once
res = s.solve(b);                 % res.x, res.iters, res.residual, res.converged
x = pcg(A, b, 1e-8, 500, @(r) s.apply(r));   % or as pcg's preconditioner
```

The same source builds under MATLAB with `mex` (recipe in
[octave/README.md](octave/README.md)); all heavy work happens inside the
compiled MEX, so Octave-vs-MATLAB interpreter speed barely matters here.
Prefer `s.solve(b)` (the C++ PCG) over wrapping `s.apply` in `pcg` unless
you need `pcg`'s interface.

## Build options

Compile-time options (`APXCHOL_USE_CUDA`, `APXCHOL_SPTRSV_FP32`,
`APXCHOL_SPTRSV_LOWPREC`, `APXCHOL_POOL_FP32`, `APXCHOL_64BIT_EDGE_INDICES` /
`APXCHOL_64BIT_NODE_INDICES`, `APXCHOL_BUILD_EXAMPLES` /
`APXCHOL_BUILD_TESTS` / `APXCHOL_BUILD_TOOLS`) are declared and documented
where they live, in [CMakeLists.txt](CMakeLists.txt) — `cmake -LH build`
lists them with their help strings. `APXCHOL_SPTRSV_FP32` and
`APXCHOL_POOL_FP32` default ON (fp32 factor values / fp32 residual-pool
weights; the PCG recurrence stays fp64; pass `=OFF` for an fp64 baseline);
`APXCHOL_SPTRSV_LOWPREC=FP16_SCALED` (default `OFF`, CPU backend only)
narrows the SpTRSV off-diagonal factor STORAGE further to per-column-scaled
IEEE fp16 (the diagonal stays exact fp32) — reads widen to fp64 in
registers, so it only changes the preconditioner quality (iteration count),
not the attainable residual; see
[include/apxchol/lowprec.h](include/apxchol/lowprec.h) (with
`APXCHOL_LOWPREC_DIAG_COMP=1` in the environment the per-column rounding
residual is folded into the fp32 diagonal, which is what keeps the iteration
count at the fp32 build's on Laplacians); everything else defaults OFF except the
`APXCHOL_BUILD_EXAMPLES` / `APXCHOL_BUILD_TESTS` toggles. Release builds
add `-march=native`.

One runtime knob (CPU/OpenMP backend): the SpTRSV setup drops factor
off-diagonals below `1e-4 ×` their column's max |off-diagonal| before it
builds its CSR/CSC, folding the dropped mass back into the kept entries of
the column (a compacting, column-sum-preserving drop — fewer stored entries,
same preconditioner within the measured 0-iteration tolerance; see
`kFactorDropRelDefault` in
[include/apxchol/solver/sptrsv/omp.h](include/apxchol/solver/sptrsv/omp.h)).
`APXCHOL_FACTOR_DROP=0` disables it, `APXCHOL_FACTOR_DROP=<rel>` overrides
the threshold.

## Benchmarks

The benchmark suite is a separate CMake project under `benchmarks/`
(FetchContent pulls Hypre, AMGCL, and RCHOL — no submodules; ParAC, CMG
(MATLAB), and Laplacians.jl are out-of-tree, wired via env vars — see
[benchmarks/README.md](benchmarks/README.md)). It
compares apxchol against BoomerAMG (Hypre, CPU+GPU), AMGCL (CPU+CUDA),
RCHOL/pRCHOL, ParAC (CPU+GPU), CMG, and Laplacians.jl AC/AC2 on grid,
SuiteSparse, and LP-IPM matrices (the LP-IPM ladder is not redistributed —
see [benchmarks/README.md](benchmarks/README.md)) — on the **original
singular Laplacians**, with each solver grounding the null space its own
native way (symmetric Dirichlet pin per connected component for the
multigrids; mean-centering for the Cholesky-types).
Protocol, charts, and the result store are documented in
[benchmarks/README.md](benchmarks/README.md); rendered charts live in
[benchmarks/latest/](benchmarks/latest/).

## Dependencies

Core library: a C++23 compiler, CMake ≥ 3.16, Eigen3 (system or FetchContent),
OpenMP (optional but strongly recommended). The root project also fetches
CLI11, spdlog, and fast_matrix_market (CLI) and GoogleTest (tests;
`-DAPXCHOL_BUILD_TESTS=OFF` to skip) at configure time. The benchmark and
Python projects manage their own dependencies.

## References

- Kyng & Sachdeva (2016). Approximate Gaussian Elimination for Laplacians.
  [arXiv:1605.02353](https://arxiv.org/abs/1605.02353)
- Kyng. Approximate Gaussian elimination for Laplacians (dissertation, Ch. 3).
  [PDF](https://rasmuskyng.com/rjkyng-dissertation.pdf)
- Gao, Kyng & Spielman (2023). Robust and Practical Solution of Laplacian
  Equations by Approximate Elimination.
  [arXiv:2303.00709](https://arxiv.org/abs/2303.00709) — reference benchmark
  protocol; code at [SDDM2023](https://github.com/rjkyng/SDDM2023)
- Baumann & Kyng (2024). A Framework for Parallelizing Approximate Gaussian
  Elimination. SPAA 2024.
  [doi:10.1145/3626183.3659987](https://dl.acm.org/doi/10.1145/3626183.3659987)
- Chen, Liang, Biros (2020). RCHOL: Randomized Cholesky Factorization.
  [arXiv:2011.07769](https://arxiv.org/abs/2011.07769) — code at
  [ut-padas/rchol](https://github.com/ut-padas/rchol)
- Liang et al. (2025). Parallel GPU-Accelerated Randomized Cholesky (ParAC).
  [arXiv:2505.02977](https://arxiv.org/abs/2505.02977)
- Spielman. [Laplacians.jl](https://github.com/danspielman/Laplacians.jl)
- Demidov. [AMGCL](https://github.com/ddemidov/amgcl)
- Falgout et al. [Hypre](https://github.com/hypre-space/hypre)

## Contributing

Build, test, and style conventions are in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[BSD 4-Clause](LICENSE) (the original "BSD with advertising clause"
license). Copyright (c) 2026 ETH Zürich and the apxchol contributors.

## Citation

If you use this software, cite it via [CITATION.cff](CITATION.cff) (GitHub's
"Cite this repository" renders it as BibTeX/APA), together with the algorithm
references above.
