# RCHOL Integration Notes

## About RCHOL

RCHOL (https://github.com/ut-padas/rchol) is a C++ implementation of
randomized incomplete Cholesky factorization for SDD linear systems.

**Paper**: "RCHOL: Randomized Cholesky Factorization for Solving SDD Linear Systems"
(Chen, Liang, Biros, 2020) — https://arxiv.org/abs/2011.07769

## Dependencies

- **METIS** — required for parallel mode (graph partitioning via nested dissection)
  - Arch: `pacman -S metis`
  - Ubuntu: `apt install libmetis-dev`
- **MKL or OpenBLAS** — the bundled PCG solver uses MKL sparse BLAS
- **Compiler**: Original Makefile uses Intel `icc`/`icpc`. Needs adaptation for `g++`.

## Building with g++

The original Makefile hardcodes Intel compilers and MKL. To build with g++:

```bash
cd extern/rchol/c++

# Edit Makefile:
#   1. Replace icc/icpc with gcc/g++
#   2. Replace MKL flags with: -lopenblas -llapack
#   3. Set METIS_INC and METIS_LIB paths
#   4. Remove -mkl flag

# Or use our adapted Makefile:
make -f Makefile.gcc
```

## Integration Approach

RCHOL operates on CSR sparse matrices (`SparseCSR`), while our project uses
Eigen `SparseMatrix` (CSC format). Integration requires:

1. Convert Eigen SparseMatrix → RCHOL SparseCSR (transpose + copy)
2. Call `rchol(A, G)` to get the approximate factor
3. Use their PCG solver, or convert G back and use Eigen's PCG

For benchmarking, we call the RCHOL binary directly with generated matrices.

## GPU Version

The GPU extension (arXiv:2505.02977, Liang et al.) is not publicly released.
The ut-padas/rchol README explicitly states "We do not support GPUs."
If/when it becomes available, it should be added as a separate submodule.
