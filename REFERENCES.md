# Approximate Cholesky Laplacian Solvers – References & Implementations

## Papers

### Core Algorithm
- **Kyng's thesis** (Chapter 3): Approximate Gaussian elimination for Laplacians
  - PDF: https://rasmuskyng.com/rjkyng-dissertation.pdf

- **Algorithm-only paper (2016)**: Approximate Gaussian elimination for Laplacians — an exact characterization
  - arXiv: https://arxiv.org/abs/1605.02353

### Practical Versions
- **GKS25**: The practical approximate Cholesky algorithm
  - PDF: https://rasmuskyng.com/papers/GKS25.pdf

- **Published version (2025)**:
  - arXiv: https://arxiv.org/abs/2505.02977

### Parallelization
- **SPAA 2024 paper**: Framework for parallelizing approximate Cholesky
  - ACM: https://dl.acm.org/doi/10.1145/3626183.3659987

### UC Berkeley – RCHOL
- **Sequential C++**: https://arxiv.org/abs/2011.07769
- **Repo**: https://github.com/ut-padas/rchol
- **Parallel/GPU implementation**: https://arxiv.org/abs/2505.02977 (no public repo)

## Existing Implementations

### 1. Julia – Laplacians.jl (Spielman et al.)
- **Repo**: https://github.com/danspielman/Laplacians.jl
- **Notes**: The foundational implementation. Contains CMG, LAMG, and approximate Cholesky.
- **Install**: `julia -e 'using Pkg; Pkg.add("Laplacians")'`

### 2. Julia – GKS25 code (Kyng group, ETH)
- **Source**: Available via the paper authors (link in email from Yves).
- **Notes**: This is the Julia implementation of the practical version.

### 3. UC Berkeley – RCHOL (Chen, Liang, Biros)
- **Paper**: https://arxiv.org/abs/2011.07769
- **Repo**: https://github.com/ut-padas/rchol
- **Notes**: Sequential + parallel (OpenMP+METIS) C++ randomized incomplete Cholesky.
  GPU version (arXiv:2505.02977) not publicly available.

### 4. SDDM2023 benchmark framework (Kyng et al.)
- **Paper**: https://arxiv.org/abs/2303.00709
- **Repo**: https://github.com/rjkyng/SDDM2023
- **Notes**: Reference benchmark protocol. We follow their tolerance (1e-8) and metrics.

### 5. This project (laplacian_solver)
- **What**: C++17 implementation of approximate Cholesky with greedy independent set
  elimination and random clique sampling.
- **Status**: Working prototype. PCG with the apxChol preconditioner converges
  on checkerboard grids.

## Comparison Solvers (for benchmarking)

### Direct
- **Eigen SimplicialLDLT**: Built-in sparse Cholesky in Eigen.
- **CHOLMOD** (SuiteSparse): State-of-the-art direct sparse Cholesky.
  System-installed at `/usr/lib/libcholmod.so`.

### Iterative
- **CG (no preconditioner)**: Baseline conjugate gradient.
- **CG + Incomplete Cholesky**: Eigen's `IncompleteCholesky` preconditioner.
- **CG + Diagonal**: Jacobi preconditioner.

### Multigrid
- **CMG** (Combinatorial Multigrid): MATLAB-based. Not readily available in C++.
- **LAMG** (Lean Algebraic Multigrid): MATLAB/C++.

## Benchmark Graphs

### Synthetic
- Grid graphs (uniform, jump interface, checkerboard)
- Erdős-Rényi random graphs
- Path graph, star graph

### Real-World (SuiteSparse Matrix Collection)
- `delaunay_n*`: Delaunay triangulation 2D (Laplacian-like)
- `AG-Monien/grid*`: Grid graphs
- Road networks from DIMACS10

### Tips for Downloading
```bash
# Download test matrices:
./scripts/download_graphs.sh

# Or manually from: https://sparse.tamu.edu/
```

## Key Metrics
| Metric | Description |
|--------|-------------|
| Setup time | Time to build the preconditioner / factorize |
| Solve time | Time for PCG to converge |
| Total time | Setup + Solve |
| Iterations | Number of PCG iterations |
| Rel. residual | ‖b - Lx‖ / ‖b‖ |
| Fill-in | nnz(factor) / nnz(L) |
| µs/nnz | Total time per nonzero (quality metric from ac(k)) |
