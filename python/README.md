# apxchol (Python, CPU)

Approximate-Cholesky preconditioner for graph-Laplacian / SDDM linear systems.

```bash
pip install apxchol
```

Prebuilt wheels: Linux x86_64 (manylinux), CPython 3.10–3.14. CPU only.
The wheels accept input matrices with at most signed-int32-sized dimensions and
nonzeros because their SciPy-to-Eigen import path uses 32-bit Eigen indices.
The factor itself uses unsigned 32-bit edge offsets and may exceed 2.1e9 entries;
com-Orkut therefore fits the default build. The Python import limit is separate
from the core library's `APXCHOL_64BIT_EDGE_INDICES` factor/pool-width option.

```python
import apxchol

solver = apxchol.factorize(A)           # scipy sparse Laplacian or SDDM; factor once
res = solver.solve(b, rtol=1e-8, maxiter=500)
res.x, res.iters, res.residual, res.converged

z = solver.apply(r)                     # M^{-1} r
M = solver.aspreconditioner()           # use as M= in scipy.sparse.linalg.cg

res = apxchol.solve(A, b)               # one-shot convenience
```

`A` must be the **assembled operator** (Laplacian or SDDM), not the adjacency
matrix of a graph. The operator contract — symmetric, positive diagonal,
non-positive off-diagonals — is asserted, and a violation raises a `ValueError`
naming which condition failed. An adjacency matrix carries no positive diagonal
anywhere, so it is caught there, never converted silently. `apxchol.laplacian`
does the conversion explicitly:

```python
A = scipy.io.mmread("com-Amazon.mtx").tocsc()   # adjacency matrix
L = apxchol.laplacian(A)                        # L = D - A, self-loops dropped
res = apxchol.solve(L, b)
```

A symmetric operator that is SPD but carries a few POSITIVE off-diagonals (so
not an M-matrix) is repaired rather than refused: each positive pair is lumped
onto the diagonal, `a_ii += a_ij; a_jj += a_ij; a_ij = 0`, when the
preconditioner is built — row sums preserved, and the PCG keeps applying `A`
itself, so `res.residual` is for the system you passed. `Solver.lumped` counts
what was moved (0 for a Laplacian/SDDM operator).

Laplacian vs SDDM is auto-detected: singular Laplacians get a rank-(n−1)
factor with native null-space handling; SDDM systems get the full-rank factor.
The factor is built once per `factorize(A)` (alias: `apxchol.solver(A)`) and
reused across right-hand sides; `solve` runs the library's OpenMP-parallel PCG
(threads via `OMP_NUM_THREADS`).

## Solving

```python
res = solver.solve(b, rtol=1e-8, maxiter=500, x0=guess)
```

`rtol` is the relative-residual target (SciPy's name; `tol` is kept as an
alias and `rtol` wins if both are given, default `1e-8`). `x0` is an optional
initial guess — an already-converged `x0` returns `iters == 0`.

`out` is an optional writable C-contiguous float64 array of length `n`: the
solution is written into it in place (no per-solve allocation) and returned as
`SolveResult.x`.

## Options

```python
solver = apxchol.factorize(A, seed=42, partitioner="block_greedy",
                           storage="vec_pool", keep_factor=True)
```

- `seed` — RNG seed for the randomized clique sampling.
- `partitioner` — independent-set selector: `block_greedy` (default),
  `priority_greedy`, or `baumann_kyng`.
- `storage` — graph backend: `vec_pool` (default), `forward_star`, `vec`, `bstr`.
- `keep_factor` — keep the factor arrays alive for export (default `True`).
  Costs one extra factor-sized copy in memory (~8 bytes per factor nonzero in
  the default fp32 wheels); with `keep_factor=False` the `chol()`/`L`/`D`
  export raises, while `P`, `factor_nnz` and `fill_ratio` stay available.
  Pass `keep_factor=False` for the leanest factor-once / solve-many footprint
  (the 0.1.x behavior); `apxchol.solve()` (one-shot) uses `False`.

Advanced core knobs are passed through as extra keywords:
`degree_quantile`, `degree_multiplier`, `degree_tiebreak`,
`exact_clique_max_degree`, `residual_peel`
(`natural` | `min_degree` | `bk_serial`), `stagnation_window`. Unknown
keywords raise `ValueError`. Note: `degree_multiplier` only takes effect when
`degree_quantile=0` (the quantile cap, default 0.2, replaces it).

## Factor export

```python
P = solver.P            # int64: P[original_vertex] = position in elimination order
G = solver.chol()       # scipy.sparse.csc_matrix, lower-triangular incl. sqrt-diagonal
L, D = solver.L, solver.D          # unit-lower CSC and the diagonal of L·D·L^T
solver.factor_nnz, solver.fill_ratio
```

The factor lives in **permuted** space, ordered by elimination:

```python
import numpy as np
import scipy.sparse as sp

p = np.argsort(solver.P)     # original index of the k-th eliminated vertex
A_perm = A[p][:, p]
# A_perm ~= G @ G.T ~= L @ sp.diags(D) @ L.T
```

`G` is an *approximate*, randomly sampled factor, so that identity is
approximate by construction; `L @ diags(D) @ L.T == G @ G.T` is exact. For a
pure Laplacian (`solver.sddm == False`) it holds on the rank-(n−k) subspace
only, where k is the number of connected components — the last eliminated
column of each component carries a placeholder diagonal.

`fill_ratio` is `(2 * factor_nnz - n) / nnz(A)`: the factor `G` reflected to a
full symmetric pattern (diagonal counted once) against the nonzeros of the full
symmetric `A`.

Exported values are float64 numpy arrays, but default builds store factor
values in fp32, so they carry fp32 precision (~7 digits).

## Thread safety

A `Solver` is not safe for concurrent use: `solve()` and `apply()` write shared
internal scratch buffers. Use one `Solver` per thread, or serialize calls
(each call is itself OpenMP-parallel).

## License

BSD 4-Clause (the original "BSD with advertising clause" license) —
see `LICENSE`. Copyright (c) 2026 ETH Zürich and the apxchol contributors.

## From source

The wheel build compiles the library's two core translation units directly;
building from a repository checkout works the same way:

```bash
pip install -e python          # from the repository root
pytest python/tests -v
```

Source builds use `-O3 -march=native` (the distributed wheels are built
portable). If your environment requires `--no-build-isolation`, first
`pip install pybind11 scikit-build-core`.
