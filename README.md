# Scalable Approximate Cholesky for Laplacian Systems

Benchmark suite comparing approximate Cholesky preconditioners for solving
Laplacian/SDDM linear systems. Includes our C++17 implementation alongside
reference implementations from UC Berkeley (RCHOL) and Spielman's group
(Laplacians.jl).

## Repository Structure

```
├── src/                  # Our C++ implementation
│   ├── simple_solver.cpp # Approximate Cholesky (greedy IS + random clique sampling)
│   ├── benchmark.cpp     # C++ benchmark harness (PCG with various preconditioners)
│   ├── graphs.cpp        # Graph generators (grid, checkerboard, Erdős-Rényi)
│   ├── mmio.cpp          # Matrix Market loader
│   └── main.cpp          # Quick demo driver
├── include/              # Headers
├── tests/                # GoogleTest suite
│   ├── test_sanity.cpp
│   └── test_solver.cpp   # Solver correctness & convergence tests
├── bench/                # External solver benchmarks
│   ├── julia/            # Laplacians.jl benchmark driver
│   └── rchol/            # RCHOL benchmark driver & build integration
├── extern/               # Git submodules (external code)
│   ├── Laplacians.jl/    # Spielman et al. — Julia approxChol
│   ├── rchol/            # UC Berkeley — C++ randomized Cholesky
│   └── SDDM2023/        # Kyng et al. — benchmark framework reference
├── scripts/
│   ├── setup_all.sh      # One-shot build + install everything
│   ├── run_benchmarks.sh # Run all benchmarks, collect CSV results
│   └── download_graphs.sh# Fetch SuiteSparse test matrices
└── data/matrices/        # Downloaded test matrices (gitignored)
```

## Quick Start

```bash
# Clone with submodules
git clone --recursive https://github.com/AlgOptGroup/Scalable-Approximate-Cholesky.git
cd Scalable-Approximate-Cholesky

# Full setup (builds everything, installs Julia deps, downloads matrices)
bash scripts/setup_all.sh

# Or manually:
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure
```

## Solvers Compared

### Our Implementation (C++)
- **ApxChol+PCG**: Approximate Cholesky via greedy independent-set elimination
  with random clique sampling, used as PCG preconditioner.

### External: Laplacians.jl (Julia)
- **AC** (approxChol): Degree-ordered approximate Cholesky (Kyng & Sachdeva 2016)
- **AC2**: Robust variant with split/merge parameters
- **CG**: Conjugate gradients baseline
- **Cholesky**: Direct exact factorization via SuiteSparse

### External: RCHOL (UC Berkeley, C++)
- **RCHOL**: Randomized incomplete Cholesky (Chen, Liang, Biros 2020)
  - Paper: https://arxiv.org/abs/2011.07769
  - Sequential C++ with optional OpenMP parallelism (needs METIS)

### Note on GPU RCHOL
The GPU-parallel version (Liang et al. 2025, arXiv:2505.02977) has no public
repository. The ut-padas/rchol README states "We do not support GPUs."

## Benchmark Protocol

Following the SDDM2023 framework (arXiv:2303.00709):

- **Tolerance**: 1e-8 relative residual
- **Metrics**: setup time, solve time, total time, PCG iterations, ‖Ax−b‖/‖b‖, fill-in ratio
- **Test instances**:
  - Grid Laplacians (uniform weights)
  - Checkerboard grids (varying condition number κ and tile size)
  - Erdős-Rényi random graphs
  - SuiteSparse Matrix Market files (real-world)

All solvers output CSV with a common schema:
```
solver,graph,n,nnz,setup_s,solve_s,total_s,iters,rel_res,fillin,us_per_nnz
```

### Running Benchmarks

```bash
# C++ benchmark (our solver + Eigen baselines)
bash scripts/run_benchmarks.sh

# Julia benchmark (Laplacians.jl)
julia --project=bench/julia bench/julia/bench_laplacians.jl --csv --graph checkerboard --n 500

# RCHOL benchmark (if built successfully)
build/bench_rchol --csv --n 100 --kappa 1000 --tile 4
```

## Dependencies

### Required
- C++17 compiler (GCC ≥ 9 or Clang ≥ 10)
- CMake ≥ 3.16
- Eigen3

### Optional
- **Julia ≥ 1.6** — for Laplacians.jl benchmarks
- **METIS** — for parallel RCHOL (`pacman -S metis` / `apt install libmetis-dev`)
- **CHOLMOD/SuiteSparse** — for direct solver comparison

## Tests

```bash
cd build && ctest --output-on-failure
```

Tests verify:
- Graph generators produce valid symmetric adjacency structures
- Laplacian matrices have correct properties (row-sums zero, symmetric, positive diagonal)
- ApxChol preconditioner produces finite solutions
- ApxChol+PCG converges to tolerance on grid, checkerboard, and ER graphs
- Solution verification: check ‖Lx−b‖/‖b‖

## References

- Kyng & Sachdeva (2016). Approximate Gaussian Elimination for Laplacians.
  [arXiv:1605.02353](https://arxiv.org/abs/1605.02353)
- Chen, Liang, Biros (2020). RCHOL: Randomized Cholesky Factorization.
  [arXiv:2011.07769](https://arxiv.org/abs/2011.07769)
- Kyng et al. (2023). SDDM Solver Benchmarks.
  [arXiv:2303.00709](https://arxiv.org/abs/2303.00709)
- Liang et al. (2025). Parallel GPU Randomized Cholesky.
  [arXiv:2505.02977](https://arxiv.org/abs/2505.02977)
- Spielman. [Laplacians.jl](https://github.com/danspielman/Laplacians.jl)
