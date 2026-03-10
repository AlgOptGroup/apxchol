# Scalable Approximate Cholesky for Laplacian Systems

C++17 library providing a CRTP preconditioner template for Laplacian/SDDM
linear systems, compatible with Eigen's PCG solver. Includes graph generators,
Matrix Market I/O, and a standalone benchmark suite comparing against RCHOL,
GPU-RCHOL, CHOLMOD, and Laplacians.jl.

## Repository Structure

```
├── include/
│   ├── laplacian_preconditioner.h  # CRTP template — plug into Eigen's PCG
│   ├── graphs.h                    # Graph generator interfaces
│   └── mmio.h                      # Matrix Market loader
├── src/
│   ├── graphs.cpp        # Graph generators (grid, checkerboard, Erdős-Rényi)
│   └── mmio.cpp          # Matrix Market loader
├── tests/                # GoogleTest suite (9 tests)
├── archive/v0/           # Frozen v0 snapshot of ApxChol (self-contained, buildable)
├── benchmarks/           # Benchmark suite (links against core library)
│   ├── CMakeLists.txt    # Own build system — FetchContent for solver deps
│   ├── src/              # Benchmark driver (links core library for graphs/solver)
│   ├── julia/            # Laplacians.jl benchmark driver (4 Julia solvers)
│   ├── run.sh            # Run benchmarks (--bench, --solver, --quick, --append)
│   ├── run_gpu_rchol.py  # GPU RCHOL benchmark wrapper
│   ├── plot.py           # Generate PDF+PNG charts from CSV
│   ├── update_readme.py  # Auto-update results table
│   └── README.md         # Benchmark docs & auto-updated results
├── scripts/
│   ├── setup_all.sh      # One-shot build + install everything
│   └── download_graphs.sh# Fetch SuiteSparse test matrices
└── data/matrices/        # Downloaded test matrices (gitignored)
```

## Quick Start

```bash
git clone <repo-url>
cd laplacian_solver

# Build core library + tests (only needs Eigen)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure
cd ..

# Benchmarks are a separate standalone project (FetchContent pulls all deps)
cd benchmarks
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) benchmark
cd ../..
bash benchmarks/run.sh          # full suite (CSV + PDF charts)
bash benchmarks/run.sh --quick  # quick smoke test
```

## Preconditioner Interface

The core library provides a CRTP base template `laplacian_preconditioner<Derived>`
that integrates directly with Eigen's `ConjugateGradient` solver. Implement
`apply(const VectorXd&)` in your derived class:

```cpp
#include "laplacian_preconditioner.h"

class my_preconditioner
    : public laplacian_preconditioner<my_preconditioner> {
public:
    using laplacian_preconditioner::laplacian_preconditioner;
    Eigen::VectorXd apply(const Eigen::VectorXd& rhs) const {
        // Your M^{-1} * rhs implementation
    }
};

// Use with Eigen's PCG:
Eigen::ConjugateGradient<SpMat, Eigen::Lower|Eigen::Upper, my_preconditioner> cg;
cg.compute(L);
Eigen::VectorXd x = cg.solve(b);
```

The base class automatically handles zero-mean centering for Laplacian solves.

## Solvers Compared

### Our Implementation (C++)
- **ApxChol+PCG**: Approximate Cholesky via greedy independent-set elimination
  with random clique sampling, used as PCG preconditioner.

### External: RCHOL (UT Austin / NC State, C++)
- **RCHOL+PCG [Chen20]**: Randomized incomplete Cholesky (Chen, Liang, Biros 2020).
  Fetched via FetchContent in benchmarks.
- **RCHOL+MKL [Chen20]**: Same factorization, MKL sparse BLAS for tri-solves.
- **pRCHOL+PCG [Chen20;par]**: Parallel RCHOL with METIS graph partitioning.

### External: CHOLMOD (SuiteSparse)
- **CHOLMOD**: Supernodal sparse Cholesky direct solver (regularization + iterative refinement → 1e-16 residuals on Laplacians).

### External: AMGCL (header-only C++)
- ~~**AMG+CG**~~: Excluded from default benchmarks — smoothed_aggregation + spai0
  does not converge on graph Laplacians. Available via `--solver amgcl`.

### External: Laplacians.jl (Julia)
- **AC [Kyng16;Jl]**: ApproxChol — degree-ordered approximate Cholesky (Kyng & Sachdeva 2016)
- **AC2 [Kyng16;Jl]**: Robust variant with split/merge parameters
- **CG [Julia]**: Conjugate gradients baseline
- **Chol [Julia]**: Direct Cholesky via SuiteSparse

### Baseline (Eigen)
- **CG [Eigen]**: CG with diagonal preconditioner
- **LDLT [Eigen]**: Eigen sparse direct solver (regularization + iterative refinement for machine precision on Laplacians)

### GPU RCHOL
- **GPU-RCHOL+PCG [Liang25]**: GPU-parallel RCHOL (Liang et al. 2025,
  arXiv:2505.02977) via CUDA cusparse/cublas. Benchmarked via Python wrapper.
  Source: https://github.com/Tianyu-Liang/Parallel-Randomized-Cholesky
  Built via `-DBUILD_GPU_RCHOL=ON` (FetchContent, requires CUDA 12.4+).

## Benchmark Protocol

Following the SDDM2023 framework (arXiv:2303.00709):

- **Tolerance**: 1e-8 relative residual
- **Repetitions**: C++ solvers use `--repeat 3` (median of 3 runs); Julia uses JIT warmup on a tiny graph before the timed run
- **Metrics**: setup time, solve time, total time, PCG iterations, ‖Ax−b‖/‖b‖, fill-in
- **Test instances**: Grid, checkerboard, Erdős-Rényi, SDDM2023 families, SuiteSparse matrices
- **CSV schema**: `solver,graph,n,nnz,setup_s,solve_s,total_s,iters,rel_res,fillin,us_per_nnz`

See [benchmarks/README.md](benchmarks/README.md) for detailed results and charts.

## Dependencies

### Core library (no submodules, no heavy deps)
- C++17 compiler (GCC ≥ 9 or Clang ≥ 10)
- CMake ≥ 3.16
- Eigen3 (auto-detected; fetched via FetchContent if missing)

### Benchmarks (standalone, all deps via FetchContent)
The benchmark suite has its own CMakeLists.txt and fetches all dependencies
(RCHOL, AMGCL) at configure time — no git submodules needed.

- **Boost** (headers only, for AMGCL)
- **CHOLMOD/SuiteSparse** (optional, system library)
- **MKL** (optional, for MKL-accelerated RCHOL)
- **METIS** (optional, for parallel RCHOL)
- **Julia ≥ 1.6** (optional, for Laplacians.jl benchmarks)
- **Python 3 + matplotlib + pandas** (optional, for plot generation)

## Tests

```bash
cd build && ctest --output-on-failure   # 9 tests
```

## References

- Kyng & Sachdeva (2016). Approximate Gaussian Elimination for Laplacians.
  [arXiv:1605.02353](https://arxiv.org/abs/1605.02353)
- Chen, Liang, Biros (2020). RCHOL: Randomized Cholesky Factorization.
  [arXiv:2011.07769](https://arxiv.org/abs/2011.07769)
- Kyng et al. (2023). SDDM Solver Benchmarks.
  [arXiv:2303.00709](https://arxiv.org/abs/2303.00709)
- Liang et al. (2025). Parallel GPU-Accelerated Randomized Cholesky.
  [arXiv:2505.02977](https://arxiv.org/abs/2505.02977)
- Spielman. [Laplacians.jl](https://github.com/danspielman/Laplacians.jl)
- Demidov. [AMGCL](https://github.com/ddemidov/amgcl)
