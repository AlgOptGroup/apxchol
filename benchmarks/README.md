# Benchmarks

Comparative benchmarks for Laplacian/SDDM linear system solvers.

## Solvers

| Solver | Type | Implementation | Source |
|--------|------|----------------|--------|
| **ApxChol+PCG [Kyng16]** | Approximate Cholesky preconditioner + PCG | Ours (C++) | `benchmarks/src/simple_solver.cpp` |
| **CG [Eigen]** | Conjugate gradient (diagonal precond.) | Eigen | — |
| **LDLT [Eigen]** | Sparse direct LDL^T | Eigen | — |
| **RCHOL+PCG [Chen20]** | Randomized Cholesky + PCG (Eigen tri-solves) | [ut-padas/rchol](https://github.com/ut-padas/rchol) | FetchContent |
| **RCHOL+MKL [Chen20]** | Randomized Cholesky + PCG (MKL sparse BLAS) | [ut-padas/rchol](https://github.com/ut-padas/rchol) + MKL | FetchContent |
| **pRCHOL+PCG [Chen20;par]** | Parallel RCHOL (METIS partitioning) + PCG | [ut-padas/rchol](https://github.com/ut-padas/rchol) + METIS | FetchContent |
| **CHOLMOD [SuiteSparse]** | Supernodal sparse Cholesky (direct) | [SuiteSparse](https://people.engr.tamu.edu/davis/suitesparse.html) | System library |
| **GPU-RCHOL+PCG [Liang25]** | GPU parallel RCHOL + PCG | [Tianyu-Liang/Parallel-Randomized-Cholesky](https://github.com/Tianyu-Liang/Parallel-Randomized-Cholesky) | FetchContent (CUDA) |
| ~~AMG+CG [AMGCL]~~ | ~~Algebraic multigrid + CG~~ | ~~[AMGCL](https://github.com/ddemidov/amgcl)~~ | *Excluded: broken on Laplacians* |
| ~~CG+ICC [Eigen]~~ | ~~CG + Incomplete Cholesky~~ | ~~Eigen~~ | *Excluded: hits iteration cap* |
| **AC [Kyng16;Jl]** | ApproxChol (degree-ordered) | [Laplacians.jl](https://github.com/danspielman/Laplacians.jl) | `benchmarks/julia/` |
| **AC2 [Kyng16;Jl]** | ApproxChol (augmented params) | [Laplacians.jl](https://github.com/danspielman/Laplacians.jl) | `benchmarks/julia/` |
| **CG [Julia]** | Conjugate gradient | Julia stdlib | — |
| **Chol [Julia]** | Sparse Cholesky (direct) | Julia SparseArrays | — |

## Benchmark Protocol

Following [SDDM2023](https://arxiv.org/abs/2303.00709):

- **Tolerance**: 1e-8 relative residual (‖Ax−b‖/‖b‖)
- **Test instances**:
  - Grid Laplacians (uniform weights)
  - Checkerboard grids (condition number κ, tile size)
  - Erdős-Rényi random graphs
  - SuiteSparse Matrix Market files
- **Metrics**: setup time, solve time, total time, PCG iterations, relative residual, fill-in ratio, µs/nnz
- **CSV schema**: `solver,graph,n,nnz,setup_s,solve_s,total_s,iters,rel_res,fillin,us_per_nnz`

## Running

```bash
# Build benchmarks (standalone, from benchmarks/ directory)
cd benchmarks
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) benchmark

# Full benchmark suite (all graph families, all solvers)
bash benchmarks/run.sh

# Quick smoke test (smaller sizes, n up to 300)
bash benchmarks/run.sh --quick

# Select specific bench sets
bash benchmarks/run.sh --quick --bench grid,checker

# Select specific solvers
bash benchmarks/run.sh --quick --solver rchol,apxchol

# Append to existing CSV
bash benchmarks/run.sh --quick --bench erdos --append benchmarks/results/latest.csv

# Individual solver via the binary directly
benchmarks/build/benchmark --graph checkerboard --n 500 --kappa 1000 --solver rchol,apxchol --csv
```

**Bench sets**: `grid`, `checker`, `erdos`, `tile`, `julia`, `gpu`, `mtx`
**Solvers**: `apxchol`, `cg`, `ldlt`, `rchol`, `rchol_mkl`, `rchol_par`, `cholmod`, `amgcl`, `icc`

## Results

Results are saved to `benchmarks/results/` as timestamped CSV files (gitignored).
`benchmarks/results/latest.csv` always points to the most recent run.

The `benchmarks/latest/` directory contains the most recent results CSV and
PNG plots, and **is committed to the repo** so results are visible on GitHub:

- `latest/results.csv` — full CSV data
- `latest/scaling.png`, `latest/efficiency.png`, etc. — charts

### Plots

Generated automatically by `benchmarks/plot.py` after each run:

**Per-family scaling** (SDDM2023 methodology):
- `scaling_checkerboard.png` — Checkerboard (κ=1000, tile=4)
- `scaling_grid.png` — Uniform grid Laplacian
- `scaling_erdos.png` — Erdős-Rényi (varying p, ~1M edges)

**Cross-cutting analyses**:
- `iterations_vs_kappa.png` — PCG iterations vs condition number (preconditioned solvers only)
- `bar_comparison.png` — Setup/solve time breakdown (stacked)
- `residual.png` — Solution accuracy with tol=1e-8 reference line
- `efficiency.png` — µs per nonzero (median per nnz bucket)

**Time per edge** (SDDM2023 core metric):
- `tve_checkerboard.png`, `tve_grid.png`, `tve_erdos_renyi.png`

```bash
# Regenerate plots manually
python3 benchmarks/plot.py benchmarks/results/latest.csv
```

## Comparison to Published Results

### vs SDDM2023 (Kyng et al., arXiv:2303.00709)

The SDDM2023 paper tests at much larger scale (n=28.7M, nnz=200M) on
a high-end workstation. We test at n≤90K on a laptop (GCC 15.2.1,
RTX 4090 Laptop, `-O3 -march=native`). Key comparisons on a
checkerboard grid (κ=1000, tile=4, n=90000):

| Metric | Our ApxChol [Kyng16] | Our RCHOL [Chen20] | RCHOL+MKL [Chen20] | SDDM2023 AC (Julia) | SDDM2023 HyPre |
|--------|---------------------|--------------------|--------------------|---------------------|-----------------|
| Iterations | 37 | 27 | 27 | ~24 | ~8 |
| Total time | 0.27s | 0.14s | 0.13s | — | — |
| Per-iteration | 2.20ms | 2.34ms | 2.43ms | — | — |
| Setup time | 0.18s | 0.07s | 0.06s | — | — |
| µs/nnz | ~0.50 | ~0.26 | ~0.24 | ~0.77 | ~0.49 |
| Fill-in | ~2.2x | ~3.2x | ~3.1x | — | — |

**Per-iteration time**: Our RCHOL (~2.3ms) and ApxChol (~2.2ms) are
comparable per iteration, showing that the preconditioner quality
(fewer iterations) is the main differentiator, not the tri-solve cost.

**Our ApxChol** uses more iterations (37 vs 24) than Laplacians.jl's AC
at similar size. This is expected — the Julia version uses a more
sophisticated degree-ordered elimination with PCG tuning. Our C++
implementation uses greedy IS elimination which is simpler but produces
a weaker preconditioner.

**RCHOL** achieves the best iteration count (27) among our
implementations on this problem, with a fast setup time (0.06–0.07s).

**Direct solvers**: CHOLMOD (0.12s) and LDLT (0.14s) are competitive
in total time and now achieve machine-precision residuals (~1e-16)
via regularization (L + εI, ε = 1e-12·mean|diag|) followed by
iterative refinement on the original Laplacian.

### Key takeaways

- **Fastest overall**: RCHOL+MKL and CHOLMOD (tied at ~0.12s for n=90K).
- **Best accuracy**: CHOLMOD and LDLT achieve machine precision (1e-16).
- **Best scaling**: Approximate Cholesky methods (RCHOL, ApxChol)
  show near-linear scaling with log factors.
- **Direct solvers** (CHOLMOD, LDLT): Competitive speed with
  machine-precision accuracy via regularization + refinement.
- **GPU-RCHOL**: Currently slower than CPU methods at all tested sizes
  (n ≤ 90K) due to deep elimination trees on grid Laplacians and
  sequential sparse triangular solves. See GPU section below.

## GPU Implementation

A GPU-accelerated parallel RCHOL is described in Liang et al. (2025),
"Parallel GPU-Accelerated Randomized Construction of Approximate Cholesky
Preconditioners" (arXiv:2505.02977).

- **Code**: https://github.com/Tianyu-Liang/Parallel-Randomized-Cholesky
- **Requirements**: CUDA 12.4+, CUDA architecture auto-detected via nvidia-smi
- **Authors**: Tianyu Liang (UCB), Chao Chen (NC State, original RCHOL
  co-author), + 6 collaborators from LBNL/Rice/UCB

### GPU Performance Notes

At our benchmark sizes (n ≤ 90K), **GPU-RCHOL is significantly slower
than CPU methods** for grid-structured Laplacians:

| n (side) | N (vertices) | GPU-RCHOL total | CPU RCHOL total | Ratio |
|---------|-------------|----------------|----------------|------:|
| 50 | 2,500 | 161ms | 2.7ms | 60x |
| 100 | 10,000 | 527ms | 14ms | 38x |
| 200 | 40,000 | 2.3s | 36ms | 64x |
| 300 | 90,000 | 4.8s | 131ms | 37x |

**Root causes** (confirmed by profiling):
1. **Deep elimination trees**: Grid Laplacians with natural ordering
   produce elimination trees of depth ≈ N, killing GPU parallelism.
2. **Sequential triangular solves**: The RCHOL preconditioner
   M = LDL^T requires sparse triangular solves (cusparse SpSV) each
   PCG iteration. These are inherently sequential on GPU.
3. **Launch overhead**: CUDA kernel setup (~27–57ms) and cusparse
   analysis (~50–300ms) overhead dominates at small sizes.
4. **n_blocks has minimal effect**: Tested 32–1024 blocks,
   <15% variation in total time.

## Latest Results

<!-- BENCHMARK_RESULTS_START -->
*Last updated: 2026-03-09 23:49:19*

No checkerboard (κ=1000) results found.

<!-- BENCHMARK_RESULTS_END -->
